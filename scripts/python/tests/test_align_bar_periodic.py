# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Alignment must not flip by a bar when the onset envelope is bar-periodic.

Bug: `_align_by_onset_envelope` took the argmax of the onset-envelope cross-correlation within +/-1 bar.
Looped drums make lag 0 and +/-1 bar nearly tie, so the winner was a coin flip (Regime v032 chose
-1.765 s, i.e. a whole bar, and read temporal 0.46 instead of 0.61 at the true lag 0).

Synthetic proof: 8 bars, one distinct chord per bar, one identical drum pattern per bar. The ORIGINAL has a
drum fill in bar 1, the RENDER has it in bar 0 (a misplaced accent), chords/drums otherwise aligned at lag 0.
Under onset-only alignment the +1 bar lag wins (0.84 vs 0.76); the harmony trajectory rejects it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import compare_audio  # noqa: E402

SR = 22050
BEAT = 60.0 / 96.0
BAR = 4 * BEAT
ROOTS = [57, 62, 59, 64, 55, 60, 66, 53]
# the real max_lag is one bar at the ESTIMATED tempo; a bar-lag peak sits at or just beyond that edge
MAX_LAG_S = 1.1 * BAR


def _chord(i):
    t = np.arange(int(BAR * SR)) / SR
    y = np.zeros_like(t)
    for semis in (0, 4, 7):
        f = 440.0 * 2 ** ((ROOTS[i] + semis - 69) / 12)
        for h in (1, 2, 3):
            y += np.sin(2 * np.pi * f * h * t) / h
    return 0.12 * y


def _drums(fill):
    y = np.zeros(int(BAR * SR))
    steps = [(0, 1.0), (7, 0.6), (10, 0.5)] + ([(i, 0.7) for i in range(1, 16, 2)] if fill else [])
    for st, g in steps:
        s, m = int(st * BAR / 16 * SR), int(0.03 * SR)
        y[s:s + m] += np.random.default_rng(st).standard_normal(m) * np.exp(-np.arange(m) / (0.006 * SR)) * g * 0.5
    return y


def _song(fill_bar):
    return np.concatenate([_chord(i) + _drums(i == fill_bar) for i in range(8)]).astype(np.float32)


def test_bar_periodic_onsets_do_not_pull_the_lag_a_bar_away():
    orig, rend = _song(fill_bar=1), _song(fill_bar=0)
    _, _, lag = compare_audio._align_by_onset_envelope(orig, rend, SR, MAX_LAG_S)
    assert abs(lag) < 0.05, f"aligned {lag:+.3f} s (a bar is {BAR:.3f} s)"


def test_candidates_expose_the_runner_up_and_the_onset_only_choice():
    orig, rend = _song(fill_bar=1), _song(fill_bar=0)
    _, _, lag, cands = compare_audio._align_by_onset_envelope(orig, rend, SR, MAX_LAG_S, return_info=True)
    onset_only = max(cands, key=lambda c: c["onset_score"])
    # construction check: the old rule (argmax of the onset curve) would have chosen a whole bar
    assert abs(abs(onset_only["lag_seconds"]) - BAR) < 0.05, cands
    assert cands[0]["chosen"] and not any(c["chosen"] for c in cands[1:])
    assert cands[0]["lag_seconds"] == pytest.approx(lag)
    assert cands[0]["harmony_score"] > cands[1]["harmony_score"]


def test_genuine_offset_is_still_found():
    shift = int(0.3 * SR)
    base = _song(fill_bar=1)
    late = np.concatenate([np.zeros(shift, np.float32), base])
    _, _, lag, _ = compare_audio._align_by_onset_envelope(base, late, SR, MAX_LAG_S, return_info=True)
    assert lag == pytest.approx(-0.3, abs=0.03)          # render 0.3 s late -> negative
    early = base[shift:]                                    # original head trimmed -> render events come earlier
    _, _, lag, _ = compare_audio._align_by_onset_envelope(base, early, SR, MAX_LAG_S, return_info=True)
    assert lag == pytest.approx(0.3, abs=0.03)


def test_default_return_is_unchanged_three_tuple():
    base = _song(fill_bar=1)
    out = compare_audio._align_by_onset_envelope(base, base, SR, MAX_LAG_S)
    assert len(out) == 3 and out[2] == 0.0


# --- review findings (code-review of f1eb032) -------------------------------------------------------

def _kill_harmony(monkeypatch):
    """Make the harmony score unscorable (all -inf / None), under both the old and the new function names."""
    monkeypatch.setattr(compare_audio, "_harmony_trajectory_score", lambda *a, **k: -np.inf, raising=False)
    monkeypatch.setattr(compare_audio, "_harmony_terms", lambda *a, **k: (None, []), raising=False)


def test_unscorable_harmony_falls_back_to_the_onset_best(monkeypatch):
    """Review 1: with no harmony evidence the onset best stays, it must not be the leftmost peak."""
    orig, rend = _song(fill_bar=1), _song(fill_bar=0)
    _kill_harmony(monkeypatch)
    _, _, lag = compare_audio._align_by_onset_envelope(orig, rend, SR, MAX_LAG_S)
    assert lag == pytest.approx(BAR, abs=0.05), lag     # the onset curve's best (+1 bar), not -1 bar / cand[0]


def test_weak_harmony_cannot_overrule_the_onset_best():
    """Review 2: the render's chords are unrelated to the original's (a static chord: harmony ~ noise) while the onset
    candidates are bar-aliased: the harmony differences are within their own noise, so the onset best (+1 bar
    here) stays. (Harmony alone used to decide, i.e. on noise.)"""
    orig = _song(fill_bar=1)
    drums = np.concatenate([_drums(i == 0) for i in range(8)])
    other = np.concatenate([_chord(0) for _ in range(8)])      # no progression: nothing for harmony to latch on
    rend = (drums + other).astype(np.float32)
    _, _, lag, cands = compare_audio._align_by_onset_envelope(orig, rend, SR, MAX_LAG_S, return_info=True)
    onset_best = max(cands, key=lambda c: c["onset_score"])["lag_seconds"]
    assert lag == pytest.approx(onset_best, abs=0.03), (lag, cands)


def test_rising_window_edge_is_not_a_peak():
    """Review 3: the curve continues past the window and is still rising at its edge -> not a candidate."""
    curve = np.array([0.1, 0.2, 0.5, 0.3, 0.4, 0.6, 0.8])
    inside = np.array([True] * 6 + [False])
    assert compare_audio._curve_peaks(curve, inside, 1) == [2]
    inside_all = np.array([True] * 7)
    assert compare_audio._curve_peaks(curve, inside_all, 1) == [2, 6] or compare_audio._curve_peaks(curve, inside_all, 1) == [2]


def test_unscorable_values_are_null_in_json(monkeypatch, tmp_path):
    """Review 6: no -Infinity/NaN in comparison.json (Go / JS parsers reject them)."""
    import json
    sf = pytest.importorskip("soundfile")
    o, r = tmp_path / "o.wav", tmp_path / "r.wav"
    sf.write(o, _song(1), SR)
    sf.write(r, _song(0), SR)
    _kill_harmony(monkeypatch)
    res = compare_audio.compare_audio(str(o), str(r), duration=30)
    text = json.dumps(res, allow_nan=False)             # raises on inf / nan
    assert "Infinity" not in text and "NaN" not in text
    cands = res["comparison"]["temporal"]["lag_candidates"]
    assert cands and all(c["harmony_score"] is None for c in cands)


def test_fallback_temporal_blocks_have_the_same_keys(monkeypatch, tmp_path):
    """Review 7: the empty-render and alignment-failure fallbacks carry the full normal key set."""
    sf = pytest.importorskip("soundfile")
    o, r, tiny = tmp_path / "o.wav", tmp_path / "r.wav", tmp_path / "t.wav"
    sf.write(o, _song(1), SR)
    sf.write(r, _song(0), SR)
    sf.write(tiny, np.zeros(100, np.float32), SR)
    normal = set(compare_audio.compare_audio(str(o), str(r), duration=30)["comparison"]["temporal"])
    empty = compare_audio.compare_audio(str(o), str(tiny), duration=30)["comparison"]["temporal"]
    assert set(empty) == normal and empty["lag_candidates"] == []

    def boom(*a, **k):
        raise RuntimeError("no alignment")
    monkeypatch.setattr(compare_audio, "_align_by_onset_envelope", boom)
    failed = compare_audio.compare_audio(str(o), str(r), duration=30)["comparison"]["temporal"]
    assert set(failed) == normal | {"error"} and failed["lag_candidates"] == []


def _click_train(bpb, bars=24):
    """Kick on beat 1, soft hats on the other beats (a spectral accent the onset envelope can see)."""
    beat = int(0.5 * SR)
    y = np.zeros(beat * bpb * bars, np.float32)
    n, m = int(0.15 * SR), int(0.02 * SR)
    kick = np.sin(2 * np.pi * 55 * np.arange(n) / SR) * np.exp(-np.arange(n) / SR * 25)
    hat = np.random.default_rng(0).standard_normal(m) * np.hanning(m) * 0.3
    for k in range(bpb * bars):
        if k % bpb == 0:
            y[k * beat:k * beat + n] += kick
        else:
            y[k * beat:k * beat + m] += hat
    return y


@pytest.mark.parametrize("bpb", [3, 4])
def test_beats_per_bar_is_measured(bpb):
    """Review 10: the bar is not assumed to be 4 beats."""
    import librosa
    env = librosa.onset.onset_strength(y=_click_train(bpb), sr=SR, hop_length=512)
    assert compare_audio._beats_per_bar(env, 0.5 * SR / 512) == bpb


def test_reused_chroma_matches_recomputed():
    """Review 10: slicing the full-signal chroma equals chroma of the aligned audio (up to edge padding)."""
    import librosa
    y = _song(1)
    shifted = y[int(0.3 * SR) // 512 * 512:]
    full = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    lag = int(0.3 * SR) // 512
    sl, _ = compare_audio._slice_chroma((full, full), lag, len(shifted), 512)
    ref = librosa.feature.chroma_cqt(y=shifted, sr=SR, hop_length=512)
    assert sl.shape == ref.shape
    assert np.abs(sl - ref).mean() < 0.02


def test_report_tolerates_null_alignment_scores():
    """Review 6: null onset/harmony scores and a null chance lag must not break the HTML report."""
    import generate_report
    comp = {"comparison": {
        "overall_similarity": 0.7, "mfcc_similarity": 0.7, "chroma_similarity": 0.7, "brightness_similarity": 0.7,
        "tempo_similarity": 1.0, "frequency_balance_similarity": 0.9, "energy_similarity": 0.9,
        "static_similarity": 0.8, "margin_over_chance": 0.1,
        "temporal": {"lag_seconds": 0.0, "lag_candidates": [
            {"lag_seconds": 0.0, "onset_score": 0.4, "harmony_score": None, "onset_tied": True, "chosen": True},
            {"lag_seconds": 1.7, "onset_score": None, "harmony_score": None, "onset_tied": False, "chosen": False}],
            "chroma_corr": 0.5, "mel_corr": 0.6, "onset_f1": 0.4, "envelope_corr": 0.7, "temporal_similarity": 0.55},
        "chance": {"path": "x", "overall": 0.4, "section_aware": 0.4, "temporal": 0.1,
                   "lag_seconds": None, "lag_candidates": None}}}
    html = generate_report.generate_charts_html(comp)
    assert "Temporal (time-aligned)" in html
