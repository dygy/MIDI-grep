# @layer: unit
# @regression
"""A pitch's grains are compared at one level window, not by whole-grain RMS.

A ringed grain includes its decay, so its whole-grain RMS is lower than a short grain struck at the same level;
``select_per_pitch`` then read the long grain as "quieter than the pitch's usual" and rejected it.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_pitched_grain_level_window.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pitched_grains as pg  # noqa: E402

SR = 22050
TAU = 0.3    # decay time constant of the synthetic 808


def _grain(dur: float, quality: float) -> pg.Grain:
    t = np.arange(int(dur * SR)) / SR
    y = (np.exp(-t / TAU) * np.sin(2 * np.pi * 55 * t)).astype(np.float32)   # same peak (1.0) and decay for all
    g = pg.Grain(y, 28, 28, "salience", quality, 0.0, dur, dur > 0.3, pg._rms(y))
    g.raw, g.sr = y, SR                    # the grain as the stem plays it
    return g


def test_a_long_decaying_grain_is_not_rejected_for_its_decay() -> None:
    long_, mid = _grain(1.0, 5.0), _grain(0.5, 1.0)
    shorts = [_grain(0.2, 1.0) for _ in range(3)]
    assert long_.rms < mid.rms < shorts[0].rms                 # the whole-grain RMS is the artefact
    chosen = pg.select_per_pitch([long_, mid, *shorts], target_len_s=0.4)
    assert chosen[28] is long_                                  # same strike level -> the cleanest long grain stands
