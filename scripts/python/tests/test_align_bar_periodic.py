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
