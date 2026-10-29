# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Genre detection must follow the audio, over the whole track, and say so when it cannot decide.

Bug (owner report): "Enter - Nigdy Wiecej Wojny" was labelled brazilian_funk. The detector scored ONE 10 s
excerpt from the middle of the track, multiplied the CLAP score by a BPM-range bonus (x1.8 for funk) and
penalty, and always returned a winner with a "confidence" that was just the multiplied number. A wrong label then
flowed into the sound palette, the eval floors (funk's 0.85 floor), the chance control and the calibration
memory's neighbours.

The CLAP model is stubbed (a 'laion_clap'-style object whose text embedding is a basis vector per genre and whose
audio embedding is a function of the waveform level), so these tests are about the DECISION LOGIC only. They drive
the detector through its CLI entry point so the same test runs against the old and the new implementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))

import detect_genre_dl as dg  # noqa: E402

SR = 48000
GENRES = list(dg.GENRE_DEFINITIONS)
DIM = len(GENRES) + 1
LOW, HIGH = 0.1, 0.4          # two waveform levels the stub reads as "class A" / "class B" music


def _basis(genre):
    v = np.zeros(DIM)
    v[GENRES.index(genre)] = 1.0
    return v


class StubClap:
    """laion_clap-shaped: text -> the genre's basis vector; audio -> a mix of two genres set by the level."""

    def __init__(self, low_genre, high_genre, high_weight=0.6):
        self.low, self.high, self.w = low_genre, high_genre, high_weight
        self.text2genre = {d: g for g, i in dg.GENRE_DEFINITIONS.items() for d in i["descriptions"]}

    def get_text_embedding(self, texts, use_tensor=False):
        # prompts that are no genre description (the old code's 'negative' prompts) map to an unrelated axis
        return np.stack([_basis(self.text2genre[t]) if t in self.text2genre else np.eye(DIM)[-1] for t in texts])

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        out = []
        for y in x:
            hi = float(np.mean(y) > (LOW + HIGH) / 2)           # the level of this window
            g_a, g_b = (self.high, self.low) if hi else (self.low, self.high)
            out.append(0.8 * _basis(g_a) + self.w * _basis(g_b))   # 'g_a' is what this window sounds like
        return np.stack(out)


def run_detector(monkeypatch, tmp_path, stub, y, bpm=None):
    """Run detect_genre_dl.main() on a synthetic waveform with the stubbed model; return the written JSON."""
    monkeypatch.setattr(dg, "load_clap_model", lambda: (stub, "laion_clap"))
    monkeypatch.setattr(dg.librosa, "load", lambda *a, **k: (y.astype(np.float32), SR))
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"x")            # only has to exist
    out = tmp_path / "genre.json"
    argv = ["detect_genre_dl.py", str(audio), "-o", str(out)] + (["--bpm", str(bpm)] if bpm else [])
    monkeypatch.setattr(sys, "argv", argv)
    dg.main()
    return json.loads(out.read_text())


def _track(levels):
    """One 10 s block per level."""
    return np.concatenate([np.full(10 * SR, lv) for lv in levels])


def test_bpm_range_does_not_overrule_the_audio(monkeypatch, tmp_path):
    # every window sounds like lofi (cos 0.8) with a weaker funk component (0.6). Old code: funk x1.8 for a BPM of
    # 137 (inside its range) beats lofi x0.8 (outside its range) -> funk. The audio says lofi.
    stub = StubClap(low_genre="lofi", high_genre="brazilian_funk")
    res = run_detector(monkeypatch, tmp_path, stub, _track([LOW] * 6), bpm=137)
    assert res["detected_genre"] == "lofi", res["rankings"][:2]


def test_score_covers_the_whole_track_not_one_excerpt(monkeypatch, tmp_path):
    # 60 s: the 10 s the old code looked at (25-35 s) sounds like lofi, 4 of the 6 windows sound like funk.
    stub = StubClap(low_genre="lofi", high_genre="brazilian_funk")
    y = np.concatenate([np.full(20 * SR, HIGH), np.full(20 * SR, LOW), np.full(20 * SR, HIGH)])
    res = run_detector(monkeypatch, tmp_path, stub, y)
    assert res["detected_genre"] == "brazilian_funk"


def test_split_windows_spans_the_track_and_drops_silence():
    y = np.concatenate([np.full(20 * SR, 0.3), np.zeros(10 * SR), np.full(30 * SR, 0.3)])
    wins = dg.split_windows(y, SR)
    assert len(wins) == 5 and all(len(w) == 10 * SR for w in wins)           # the silent 10 s window is gone
    long = np.full(600 * SR, 0.3)
    assert len(dg.split_windows(long, SR)) == dg.MAX_WINDOWS                  # cost cap, still evenly spread
    assert len(dg.split_windows(np.full(4 * SR, 0.3), SR)) == 1               # shorter than a window


def test_a_toss_up_between_two_genres_is_flagged_uncertain_not_picked_silently(monkeypatch, tmp_path):
    # windows alternate between 'sounds like funk' and 'sounds like lofi': the top-2 margin is pure noise.
    stub = StubClap(low_genre="lofi", high_genre="brazilian_funk")
    res = run_detector(monkeypatch, tmp_path, stub, _track([LOW, HIGH] * 4))
    assert res.get("uncertain") is True
    assert set(res["candidates"]) >= {"lofi", "brazilian_funk"}
    assert res["uncertain_reasons"]


def test_a_consistent_track_is_certain(monkeypatch, tmp_path):
    stub = StubClap(low_genre="lofi", high_genre="brazilian_funk")
    res = run_detector(monkeypatch, tmp_path, stub, _track([HIGH] * 6))
    assert res["detected_genre"] == "brazilian_funk"
    assert res.get("uncertain") is False and res["candidates"] == ["brazilian_funk"]
    assert res["vote_share"] == 1.0 and res["n_windows"] == 6


def test_margin_rule_uses_the_measured_noise_of_the_windows():
    # same mean margin (0.3), different window-to-window noise: only the noisy one is uncertain.
    P = {g: np.stack([_basis(g)]) for g in GENRES}
    rng = np.random.default_rng(0)

    def windows(noise):
        out = []
        for _ in range(12):
            a = 0.8 + rng.normal(0, noise)
            out.append(a * _basis("house") + 0.5 * _basis("jazz"))
        return np.stack(out)

    calm = dg.rank_genres(windows(0.01), P)
    noisy = dg.rank_genres(windows(0.6), P)
    assert calm["uncertain"] is False and calm["margin_z"] > dg.Z_MIN
    assert noisy["margin_z"] < calm["margin_z"]


def test_too_few_windows_cannot_support_a_confident_label():
    P = {g: np.stack([_basis(g)]) for g in GENRES}
    res = dg.rank_genres(np.stack([0.9 * _basis("house") + 0.1 * _basis("jazz")] * 2), P)
    assert res["uncertain"] is True and "no estimate" in res["uncertain_reasons"][0]


# ----------------------------------------------------------------------------------- downstream consumers
def _meta_track(tmp_path):
    track = tmp_path / "T"
    (track / "v001").mkdir(parents=True)
    (track / "metadata.json").write_text(json.dumps({"title": "T"}))
    (track / "v001" / "metadata.json").write_text(json.dumps({"bpm": 130.0, "key": "C minor", "style": "trance", "version": 1}))
    return track


def _stamp(tmp_path, det, **kw):
    from pipeline_helpers import stamp_track_metadata
    track = _meta_track(tmp_path)
    f = tmp_path / "genre.json"
    f.write_text(json.dumps(det))
    return stamp_track_metadata(track, detector_json=f, **kw)


def test_uncertain_detection_keeps_the_guess_for_the_palette_but_not_for_reference_tables(tmp_path):
    m = _stamp(tmp_path, {"detected_genre": "brazilian_funk", "confidence": 0.5, "uncertain": True,
                          "candidates": ["brazilian_funk", "retro_wave"], "uncertain_reasons": ["margin 1.1 s.e."]})
    assert m["genre"] == "brazilian_funk" and m["genre_uncertain"] is True
    assert m["genre_ref"] == "unreferenced"                       # funk's floors / neighbours must not apply
    assert m["genre_candidates"] == ["brazilian_funk", "retro_wave"]


def test_certain_detection_and_explicit_override_are_referenced(tmp_path):
    m = _stamp(tmp_path, {"detected_genre": "house", "confidence": 0.9, "uncertain": False, "candidates": ["house"]})
    assert m["genre_ref"] == "house" and m["genre_basis"] == "detector"
    m2 = _stamp(tmp_path / "b", {"detected_genre": "house", "uncertain": True, "candidates": ["house", "trance"]},
                genre_override="retro_wave") if (tmp_path / "b").mkdir() is None else None
    assert m2["genre"] == "retro_wave" and m2["genre_ref"] == "retro_wave" and m2["genre_basis"] == "override"


def test_detector_json_without_a_verdict_is_not_trusted(tmp_path):
    m = _stamp(tmp_path, {"detected_genre": "house", "confidence": 0.31, "rankings": []})
    assert m["genre"] == "house" and m["genre_ref"] == "unreferenced"


def test_no_floor_is_ever_recorded_for_an_unreferenced_genre(tmp_path):
    from pipeline_helpers import record_mode_floor
    import yaml
    th = tmp_path / "thresholds.yaml"
    th.write_text(yaml.safe_dump({"modes": {"sample_instrument": {"genres": {"brazilian_funk": 0.85}}}}))
    cmp_ = tmp_path / "c.json"
    cmp_.write_text(json.dumps({"editability": "pass", "comparison": {"overall_similarity": 0.8, "section_aware_similarity": 0.79}}))
    before = th.read_text()
    assert record_mode_floor(th, mode="sample-instrument", genre="unreferenced", comparison_path=cmp_, run="x") is None
    assert th.read_text() == before


def test_gate_gives_an_unreferenced_genre_the_default_floor_and_says_it_is_unmeasured():
    from eval.gate import evaluate_comparison_data, load_thresholds, resolve_floor
    th = load_thresholds()
    funk_floor, src = resolve_floor("brazilian_funk", th, "sample-instrument")
    assert src == "modes.sample_instrument"
    floor, src = resolve_floor("unreferenced", th, "sample-instrument")
    assert floor == th["default"] and src == "default" and floor != funk_floor
    comp = {"generation_mode": "sample-instrument",
            "comparison": {"overall_similarity": 0.80, "static_similarity": 0.80, "worst_band_diff": 5.0}}
    res = evaluate_comparison_data(comp, "unreferenced", th, "sample-instrument")
    assert res.passed and res.reference_floor is False and "NOTE" in res.message
    # a genre that HAS a measured floor is gated by it, and is a reference verdict
    res_funk = evaluate_comparison_data(comp, "brazilian_funk", th, "sample-instrument")
    assert not res_funk.passed and res_funk.reference_floor is True


def test_a_genre_with_only_a_genre_wide_floor_is_reported_as_unmeasured_for_the_mode():
    from eval.gate import evaluate_comparison_data, load_thresholds
    th = load_thresholds()
    comp = {"comparison": {"overall_similarity": 0.9, "static_similarity": 0.9, "worst_band_diff": 5.0}}
    res = evaluate_comparison_data(comp, "retro_wave", th, "sample-instrument")
    assert res.floor_source == "genres" and res.floor == 0.55          # not funk's 0.85
    assert res.reference_floor is False and "no measured floor for retro_wave" in res.message


def test_detector_genre_names_resolve_to_a_real_palette():
    from sound_selector import GENRE_PALETTES, palette_key, palette_sounds
    for g in GENRES:                                   # every label the detector can emit
        key = palette_key(g)
        if g in ("brazilian_funk", "trance", "house", "lofi", "jazz"):
            assert key == g
        else:
            assert key != "default", f"detector label {g!r} silently gets the default palette"
    assert palette_sounds("retro_wave", "bass") == GENRE_PALETTES["synthwave"]["bass"]
    assert palette_sounds("brazilian_phonk", "drums") == GENRE_PALETTES["phonk"]["drums"]
    assert palette_key("electro") == "synthwave"       # electro / electropop is the synthwave sound world
    assert palette_key("unreferenced") == "default" and palette_key(None) == "default"
