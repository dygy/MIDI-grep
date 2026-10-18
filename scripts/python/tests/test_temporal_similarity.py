# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Similarity must see musical content, not just time-averaged statistics.

Divergence from spec 003 §2.4: every component of the old overall_similarity (mean MFCC, mean
chroma, average band shares, mean centroid, RMS) is a time-average, so a DIFFERENT song of the same
genre scored within a few points of the right one (measured 0.932 vs 0.876 on real tracks).

Synthetic proof (numpy only, ~20 s @ 22050 Hz): an "original" of 8 bars, each with its own chord and
click pattern; a "render" with the same events but different timbre/gain; a "different song" made of
the SAME bars in shuffled order (identical global statistics). The time-aligned score must separate
them by >= 0.1; the old time-average formula cannot.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_temporal_similarity.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import compare_audio  # noqa: E402

SR = 22050
BPM = 96.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT                       # 2.5 s -> 8 bars = 20 s
ROOTS = [57, 62, 59, 64, 55, 60, 66, 53]   # a distinct chord root per bar (MIDI)
ORDER = list(range(8))
SHUFFLED = [3, 7, 1, 5, 0, 4, 2, 6]        # no bar stays in place, no cyclic shift of ORDER


def _bar(idx: int, harmonics: int, click_seed: int, gain: float) -> np.ndarray:
    t = np.arange(int(BAR * SR)) / SR
    y = np.zeros_like(t)
    for semis in (0, 4, 7):                                   # major triad on this bar's root
        f = 440.0 * 2 ** ((ROOTS[idx] + semis - 69) / 12)
        for h in range(1, harmonics + 1):
            y += np.sin(2 * np.pi * f * h * t) / h
    y *= 0.2 * np.exp(-1.5 * (t % BEAT))                      # per-beat pluck decay
    pattern = np.random.default_rng(1000 + idx).random(16) < 0.5   # this bar's own click grid
    pattern[0] = True
    noise = np.random.default_rng(click_seed)
    for step in np.flatnonzero(pattern):
        s = int(step * BAR / 16 * SR)
        n = int(0.02 * SR)
        y[s:s + n] += noise.standard_normal(n) * np.exp(-np.arange(n) / (0.004 * SR)) * 0.6
    return gain * y


def _song(order, harmonics, click_seed, gain) -> np.ndarray:
    return np.concatenate([_bar(i, harmonics, click_seed, gain) for i in order]).astype(np.float32)


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    d = tmp_path_factory.mktemp("temporal")
    paths = {
        "orig": (ORDER, 6, 1, 1.0),
        "render": (ORDER, 3, 2, 0.7),          # same events, different timbre / gain / click noise
        "shuffled": (SHUFFLED, 3, 2, 0.7),     # same bars, different order: identical global stats
    }
    out = {}
    for name, (order, harm, seed, gain) in paths.items():
        p = d / f"{name}.wav"
        soundfile.write(p, _song(order, harm, seed, gain), SR)
        out[name] = p
    good = compare_audio.compare_audio(str(out["orig"]), str(out["render"]), duration=30)["comparison"]
    bad = compare_audio.compare_audio(str(out["orig"]), str(out["shuffled"]), duration=30)["comparison"]
    return good, bad


def test_overall_separates_right_song_from_shuffled_song(scores):
    good, bad = scores
    assert good["overall_similarity"] - bad["overall_similarity"] >= 0.1, (
        good["overall_similarity"], bad["overall_similarity"])


def test_chroma_in_time_sees_the_chord_sequence(scores):
    good, bad = scores
    assert good["temporal"]["chroma_corr"] > bad["temporal"]["chroma_corr"]


def test_temporal_block_shape_and_blend(scores):
    good, _ = scores
    t = good["temporal"]
    assert set(t) == {"lag_seconds", "chroma_corr", "mel_corr", "onset_f1", "envelope_corr", "temporal_similarity"}
    comps = [t[k] for k in ("chroma_corr", "mel_corr", "onset_f1", "envelope_corr")]
    assert all(0.0 <= c <= 1.0 for c in comps)
    assert t["temporal_similarity"] == pytest.approx(np.mean(comps))
    assert good["overall_similarity"] == pytest.approx(0.5 * good["static_similarity"] + 0.5 * t["temporal_similarity"])
    assert abs(t["lag_seconds"]) <= BAR + 0.05


def test_section_windows_carry_temporal_fields(scores):
    good, _ = scores
    wins = good["section_windows"]
    assert wins
    for w in wins:
        for k in ("chroma_corr", "mel_corr", "onset_f1", "envelope_corr", "temporal", "static", "orig_bands",
                  "rend_bands", "band_diff", "orig_rms", "rend_rms"):
            assert k in w
        assert w["win"] == pytest.approx(0.5 * w["static"] + 0.5 * w["temporal"])


def test_lag_is_recovered(tmp_path):
    shift = int(0.3 * SR)
    base = _song(ORDER, 3, 2, 1.0)
    orig = tmp_path / "o.wav"
    soundfile.write(orig, base, SR)
    late = tmp_path / "late.wav"
    soundfile.write(late, np.concatenate([np.zeros(shift, np.float32), base]), SR)
    t = compare_audio.compare_audio(str(orig), str(late), duration=30)["comparison"]["temporal"]
    assert t["lag_seconds"] == pytest.approx(-0.3, abs=0.03)   # render is 0.3 s LATE -> negative
    assert t["onset_f1"] > 0.7


def test_onset_f_measure_is_one_to_one():
    assert compare_audio.onset_f_measure([1.0, 2.0], [1.0, 2.0]) == 1.0
    # two references cannot both claim the single estimate
    assert compare_audio.onset_f_measure([1.0, 1.02], [1.01]) == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert compare_audio.onset_f_measure([1.0], [1.2]) == 0.0


def test_control_flag_writes_chance_and_margin(tmp_path):
    base = tmp_path / "o.wav"; soundfile.write(base, _song(ORDER, 6, 1, 1.0), SR)
    rend = tmp_path / "r.wav"; soundfile.write(rend, _song(ORDER, 3, 2, 0.7), SR)
    ctrl = tmp_path / "c.wav"; soundfile.write(ctrl, _song(SHUFFLED, 3, 2, 0.7), SR)
    with_ctrl = compare_audio.compare_audio(str(base), str(rend), duration=30, control_path=str(ctrl))["comparison"]
    assert with_ctrl["chance"]["path"] == str(ctrl)
    assert with_ctrl["margin_over_chance"] == pytest.approx(
        with_ctrl["overall_similarity"] - with_ctrl["chance"]["overall"])
    assert with_ctrl["margin_over_chance"] > 0
    without = compare_audio.compare_audio(str(base), str(rend), duration=30)["comparison"]
    assert "chance" not in without and "margin_over_chance" not in without


# ---------------------------------------------------------------------------
# Gate: margin over chance
# ---------------------------------------------------------------------------
from eval.gate import evaluate_comparison, load_thresholds  # noqa: E402

THRESHOLDS = load_thresholds()


def _write(tmp_path: Path, **extra) -> Path:
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"comparison": {"overall_similarity": 0.95, "worst_band_diff": 5.0, **extra}}))
    return p


def test_gate_fails_when_margin_over_chance_below_min(tmp_path):
    min_margin = THRESHOLDS["min_margin_over_chance"]
    res = evaluate_comparison(_write(tmp_path, margin_over_chance=min_margin - 0.01),
                              genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed and res.chance_passed is False
    assert "margin over chance" in res.message


def test_gate_passes_with_margin_and_reports_it(tmp_path):
    margin = THRESHOLDS["min_margin_over_chance"] + 0.05
    res = evaluate_comparison(_write(tmp_path, margin_over_chance=margin),
                              genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.chance_passed is True and res.margin_over_chance == pytest.approx(margin)
    assert "margin over chance" in res.message


def test_gate_skips_silently_without_margin(tmp_path):
    res = evaluate_comparison(_write(tmp_path), genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed and res.margin_over_chance is None and res.chance_passed is None


# ---------------------------------------------------------------------------
# Gate: genre-wide floors are STATIC-scale, mode floors are overall-scale
# ---------------------------------------------------------------------------
def _scaled(tmp_path: Path, overall, static, sa, window_static=None, mode=None) -> Path:
    comp = {"overall_similarity": overall, "static_similarity": static, "section_aware_similarity": sa,
            "worst_band_diff": 5.0,
            "section_windows": [{"static": v} for v in (window_static or [])]}
    p = tmp_path / "comparison.json"
    doc = {"comparison": comp}
    if mode:
        doc["generation_mode"] = mode
    p.write_text(json.dumps(doc))
    return p


def test_genre_wide_floor_compares_static_scale(tmp_path):
    # low overall (temporal half drags it) but static clears the static-scale genre floor and window statics clear the sa floor
    res = evaluate_comparison(_scaled(tmp_path, 0.40, 0.90, 0.40, [0.9, 0.8]), genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.similarity == pytest.approx(0.90) and res.section_aware_similarity == pytest.approx(0.85)
    assert "static similarity" in res.message and "static section_aware" in res.message


def test_genre_wide_floor_fails_on_static_and_says_static(tmp_path):
    res = evaluate_comparison(_scaled(tmp_path, 0.95, 0.30, 0.95, [0.1]), genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed
    assert "static similarity 0.300 < floor" in res.message and "static section_aware" in res.message


def test_genre_wide_section_check_skipped_without_window_statics(tmp_path):
    res = evaluate_comparison(_scaled(tmp_path, 0.40, 0.90, 0.40), genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed and res.section_aware_passed is None


def test_mode_floor_uses_overall_scale_not_static(tmp_path):
    floor = THRESHOLDS["modes"]["sample_instrument"]["genres"]["brazilian_funk"]
    res = evaluate_comparison(_scaled(tmp_path, floor - 0.1, 0.99, 0.99, [0.99], mode="sample-instrument"),
                              genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed and res.floor_source == "modes.sample_instrument"
    assert "static" not in res.message.split(";")[0]


def test_gate_margin_uses_same_scale_selection(tmp_path):
    import gate_margin
    p = _scaled(tmp_path, 0.40, 0.90, 0.40, [0.9, 0.8])
    p.write_text(json.dumps({"comparison": json.loads(p.read_text())["comparison"]}))
    m = gate_margin.margins(p, "brazilian_funk", None)
    assert m["overall"] == pytest.approx(0.90) and m["sa"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Review fixes: strict margin (#6), temporal failure (#7), aligned windows (#3),
# control lookup exit codes (#5), --iterate uses the gate's own logic (#1/#2)
# ---------------------------------------------------------------------------
def test_gate_margin_must_be_strictly_above_the_minimum(tmp_path):
    min_margin = THRESHOLDS["min_margin_over_chance"]
    res = evaluate_comparison(_write(tmp_path, margin_over_chance=min_margin),
                              genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed and res.chance_passed is False          # "more than", not "at least"
    import gate_margin
    p = _write(tmp_path, margin_over_chance=min_margin)
    assert gate_margin.margins(p, "brazilian_funk", None)["passed"] is False
    p2 = _write(tmp_path, margin_over_chance=min_margin + 0.01)
    assert gate_margin.margins(p2, "brazilian_funk", None)["passed"] is True


def test_temporal_failure_is_flagged_not_silently_scored(tmp_path, monkeypatch):
    o = tmp_path / "o.wav"; soundfile.write(o, _song(ORDER, 6, 1, 1.0), SR)
    r = tmp_path / "r.wav"; soundfile.write(r, _song(ORDER, 3, 2, 0.7), SR)

    def boom(*a, **k):
        raise RuntimeError("cqt exploded")
    monkeypatch.setattr(compare_audio, "compute_temporal_frames", boom)
    doc = compare_audio.compare_audio(str(o), str(r), duration=30)
    c = doc["comparison"]
    assert c["temporal_failed"] is True and "cqt exploded" in c["temporal"]["error"]
    assert c["overall_similarity"] == pytest.approx(c["static_similarity"])      # not silently halved
    assert all(w["win"] == pytest.approx(w["static"]) for w in c["section_windows"])
    res = evaluate_comparison_data(doc, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed and res.temporal_failed is True and "temporal scoring failed" in res.message
    import gate_margin
    p = tmp_path / "c.json"; p.write_text(json.dumps(doc))
    assert gate_margin.margins(p, "brazilian_funk", None)["passed"] is False


def test_failed_chance_control_fails_the_gate(tmp_path):
    p = _write(tmp_path, chance={"path": "x.wav", "error": "ValueError: unreadable"}, margin_over_chance=0.5)
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed and "chance control failed" in res.message
    ctrl = tmp_path / "missing.wav"
    base = tmp_path / "o.wav"; soundfile.write(base, _song(ORDER, 6, 1, 1.0), SR)
    rend = tmp_path / "r.wav"; soundfile.write(rend, _song(ORDER, 3, 2, 0.7), SR)
    c = compare_audio.compare_audio(str(base), str(rend), duration=30, control_path=str(ctrl))["comparison"]
    assert "error" in c["chance"] and "margin_over_chance" not in c


def test_window_halves_cover_the_same_aligned_audio(tmp_path, monkeypatch):
    order = ORDER + ORDER[:4]                                  # 12 bars = 30 s
    base = _song(order, 3, 2, 1.0)
    shift = 13 * compare_audio.TEMPORAL_HOP                    # a whole number of onset-envelope frames (~0.30 s)
    late = np.concatenate([np.zeros(shift, np.float32), base])
    o = tmp_path / "o.wav"; soundfile.write(o, base, SR)
    r = tmp_path / "r.wav"; soundfile.write(r, late, SR)
    seen = []
    real = compare_audio.compute_mfcc_features
    monkeypatch.setattr(compare_audio, "compute_mfcc_features", lambda y, *a, **k: (seen.append(np.array(y)), real(y, *a, **k))[1])
    c = compare_audio.compare_audio(str(o), str(r), duration=40)["comparison"]
    assert c["temporal"]["lag_seconds"] == pytest.approx(-shift / SR, abs=1e-6)
    win_calls = seen[2:]                                       # first two calls are the whole-file MFCCs
    assert len(win_calls) == 2 * len(c["section_windows"]) > 0
    for ow, rw in zip(win_calls[0::2], win_calls[1::2]):       # the static half's (orig, render) windows
        assert np.corrcoef(ow, rw)[0, 1] > 0.99                # same audio, sample-aligned
    aligned_len = len(base) - shift                            # the lag-trimmed head is gone; old code windowed all 30 s (3)
    assert len(c["section_windows"]) == aligned_len // int(10.0 * SR)


import pipeline_helpers  # noqa: E402


def test_control_lookup_distinguishes_none_from_failure(tmp_path, monkeypatch, capsys):
    root = tmp_path
    (root / "eval" / "datasets").mkdir(parents=True)
    ds = root / "eval" / "datasets" / "reference_tracks.yaml"
    ds.write_text("tracks:\n  - {genre: g, cache_key: A}\n  - {genre: g, cache_key: B}\n")
    assert pipeline_helpers.control_track_path("g", "A", repo_root=root) is None      # no cached original.wav
    wav = root / ".cache" / "stems" / "B" / "original.wav"
    wav.parent.mkdir(parents=True); wav.write_bytes(b"x")
    assert pipeline_helpers.control_track_path("g", "A", repo_root=root) == wav
    assert pipeline_helpers.control_track_path("g", "B", repo_root=root) is None      # never the track itself
    ds.write_text("tracks: [unclosed\n")
    with pytest.raises(Exception):
        pipeline_helpers.control_track_path("g", "A", repo_root=root)

    argv = ["control", "--genre", "g", "--exclude-key", "A"]
    monkeypatch.setattr(pipeline_helpers, "control_track_path", lambda g, k: wav)
    assert pipeline_helpers.main(argv) == 0 and capsys.readouterr().out.strip() == str(wav)
    monkeypatch.setattr(pipeline_helpers, "control_track_path", lambda g, k: None)
    assert pipeline_helpers.main(argv) == pipeline_helpers.CONTROL_NONE_EXIT == 3
    def broken(g, k):
        raise ValueError("bad yaml")
    monkeypatch.setattr(pipeline_helpers, "control_track_path", broken)
    assert pipeline_helpers.main(argv) == 1


def test_iterate_gate_is_the_eval_gates_own_logic(tmp_path, monkeypatch):
    """ai_improver's per-iteration / final verdicts come from eval.gate.evaluate_comparison_data."""
    src = (HERE.parent / "ai_improver.py").read_text()
    assert "evaluate_comparison_data(comparison, genre, _GATE_THRESHOLDS, mode=generation_mode)" in src
    assert "current_similarity >= gate_floor" not in src and "best_similarity >= gate_floor" not in src
    # the in-memory dict compare_audio returns: a v027-like render clears every floor but fails the chance margin
    floor = THRESHOLDS["modes"]["synth"]["genres"]["brazilian_funk"]
    sa_floor = THRESHOLDS["modes"]["synth"]["section_aware"]["brazilian_funk"]
    doc = {"comparison": {"overall_similarity": floor + 0.1, "section_aware_similarity": sa_floor + 0.1,
                          "worst_band_diff": 5.0, "margin_over_chance": -0.018}}
    res = evaluate_comparison_data(doc, genre="brazilian_funk", thresholds=THRESHOLDS, mode="synth")
    assert not res.passed and "margin over chance" in res.message
    # genre-wide floors (no mode) are compared on the static scale, as in eval/gate.py
    doc2 = {"comparison": {"overall_similarity": 0.30, "static_similarity": 0.90, "section_aware_similarity": 0.30,
                           "worst_band_diff": 5.0, "section_windows": [{"static": 0.9}]}}
    assert evaluate_comparison_data(doc2, genre="brazilian_funk", thresholds=THRESHOLDS).passed

    import ai_improver
    monkeypatch.setattr(pipeline_helpers, "control_track_path", lambda g, k: tmp_path / "c.wav")
    assert ai_improver._control_args("brazilian_funk", str(tmp_path / "x" / "original.wav")) == ["--control", str(tmp_path / "c.wav")]
    monkeypatch.setattr(pipeline_helpers, "control_track_path", lambda g, k: None)
    assert ai_improver._control_args("brazilian_funk", "o.wav") == []
    def broken(g, k):
        raise ValueError("bad yaml")
    monkeypatch.setattr(pipeline_helpers, "control_track_path", broken)
    with pytest.raises(ValueError):
        ai_improver._control_args("brazilian_funk", "o.wav")


from eval.gate import evaluate_comparison_data  # noqa: E402
