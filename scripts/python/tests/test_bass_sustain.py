# @layer: unit
# @regression
"""Held bass notes keep sounding where the stem does (Enter, a constant-amplitude sine bass).

1. ``StemEvidence.end_after``: a note ended on a 0.3 dB wobble of a constant tone (``fall`` 0.966 sits inside the
   tone's own ripple) - holes where the stem sounds. A dip ends a note only if the level stays down until the next
   attack.
2. ``loop_points``: the loop window landed in the grain's fade-out (any seam-smooth window down to -6 dB under the
   peak) so held notes dropped 4-6 dB. The loop sits in the steady body, whose extent comes from the grain's own
   body-level spread.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_bass_sustain.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bass_shape as BS  # noqa: E402

HOP_S = BS.StemEvidence.HOP / BS.StemEvidence.SR


def _evidence(rms: np.ndarray, attacks: list[float], fall: float) -> BS.StemEvidence:
    """A StemEvidence over a hand-made RMS track (no audio): the fields end_after reads."""
    ev = object.__new__(BS.StemEvidence)
    ev.rms = rms
    ev.hop_s = HOP_S
    ev.level = BS.sounding_level(rms)
    ev.fall = fall
    ev.peak_attacks = np.asarray(attacks, dtype=float)
    ev.attacks = ev.peak_attacks - 0.03                 # onsets are backtracked ahead of the peak
    return ev


def _sine_bass_rms(seconds: float, stop_s: float, ripple: float = 0.04, seed: int = 3) -> np.ndarray:
    """Constant tone with a +-0.35 dB ripple (Enter's fall of 0.966 is a 0.3 dB dip), then digital silence from ``stop_s``."""
    rng = np.random.default_rng(seed)
    n = int(seconds / HOP_S)
    rms = 0.2 * (1.0 + ripple * rng.uniform(-1.0, 1.0, n))
    rms[int(stop_s / HOP_S):] = 1e-5
    return rms


def test_a_wobble_of_a_constant_tone_does_not_end_the_note() -> None:
    rms = _sine_bass_rms(4.0, stop_s=3.0)
    ev = _evidence(rms, attacks=[0.0, 1.0, 2.0], fall=0.966)       # fall as measured on Enter
    end = ev.end_after(0.0, 4.0)
    assert 0.97 - 0.15 <= end <= 0.97             # rang until the next attack (+-ripple at its edge); the old code ended at the first wobble (0.05 s)


def test_a_constant_tone_that_really_stops_ends_at_the_stop() -> None:
    rms = _sine_bass_rms(4.0, stop_s=0.6)
    ev = _evidence(rms, attacks=[0.0, 2.0], fall=0.966)
    assert abs(ev.end_after(0.0, 4.0) - 0.6) < 3 * HOP_S


def test_a_decaying_note_still_ends_where_it_crossed_its_floor() -> None:
    n = int(3.0 / HOP_S)
    rms = 0.4 * np.exp(-np.arange(n) * HOP_S / 0.15)
    rms[int(1.5 / HOP_S):] = 0.4 * 0.3                             # the next note re-attacks at 1.5 s
    ev = _evidence(np.maximum(rms, 1e-4), attacks=[0.0, 1.5], fall=0.2)
    end = ev.end_after(0.0, 3.0)
    assert 0.15 < end < 0.5                                        # ~ 0.15 * ln(5): the decay to 0.2 x peak


def _grain(seconds: float = 0.6, sr: int = 22050, hz: float = 41.2, noise: float = 0.15, seed: int = 5) -> np.ndarray:
    """A bass grain as the trainer cuts it: one-period fade-in, steady (slightly noisy) body, quarter-grain release."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = np.sin(2 * np.pi * hz * t) + noise * rng.standard_normal(len(t))
    n, a, r = len(y), int(sr / hz), len(y) // 4
    y[:a] *= np.linspace(0, 1, a)
    u = np.linspace(0, 1, r)
    y[-r:] *= 1.0 - (u * u * (3 - 2 * u))
    return y.astype(np.float32)


def _level_db(y: np.ndarray, b: int, e: int) -> float:
    return 20 * np.log10(np.sqrt(np.mean(y[b:e].astype(float) ** 2)) + 1e-12)


def test_the_loop_sits_in_the_steady_body_not_the_fade_out() -> None:
    sr, hz = 22050, 41.2
    y = _grain(sr=sr, hz=hz)
    lp = BS.loop_points(y, sr, hz)
    assert lp is not None
    b, e = int(lp[0] * len(y)), int(lp[1] * len(y))
    body = _level_db(y, int(0.1 * len(y)), int(0.7 * len(y)))
    assert _level_db(y, b, e) > body - 0.5                          # old code: ~ -6 dB or more under the body
    assert e <= int(0.75 * len(y)) + int(sr / hz)                   # the release is the last quarter
