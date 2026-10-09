"""T1 multi-dimensional stem match: tests for the resample/lag math and the per-dimension wiring
(pitch/rhythm/timbre), using synthetic signals so no audio fixtures are needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_match import (  # noqa: E402
    SR, _resample_matrix, _best_lag_colsim, _shape_corr,
)


def test_resample_matrix_changes_time_axis_only():
    m = np.random.RandomState(0).rand(4, 100)
    out = _resample_matrix(m, 30)
    assert out.shape == (4, 30)


def test_best_lag_colsim_identical_is_one():
    m = np.random.RandomState(1).rand(12, 240)
    assert _best_lag_colsim(m, m) > 0.999


def test_best_lag_colsim_recovers_shifted():
    rng = np.random.RandomState(2)
    base = rng.rand(12, 240)
    shifted = np.roll(base, 5, axis=1)  # constant time offset within the lag budget
    # zero-lag would be low; best-lag should recover near-identity
    assert _best_lag_colsim(base, shifted) > 0.9


def _tone(freq: float, secs: float = 3.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shape_corr_reports_all_dimensions_for_pitched_stem():
    y = _tone(220.0)
    res = _shape_corr(y, y.copy(), stem="bass")
    for k in ("corr", "pitch", "rhythm", "timbre", "sil_orig", "sil_rend", "pass"):
        assert k in res
    assert res["pitch"] is not None  # bass is pitched


def test_shape_corr_skips_pitch_for_drums():
    y = _tone(220.0)
    res = _shape_corr(y, y.copy(), stem="drums")
    assert res["pitch"] is None          # drums have no meaningful chroma
    assert res["rhythm"] is not None      # rhythm/timbre still computed
    assert res["timbre"] is not None


def test_shape_corr_short_signal_is_graceful():
    res = _shape_corr(np.zeros(10), np.zeros(10), stem="bass")
    assert res["pass"] is False
    assert res["pitch"] is None and res["rhythm"] is None and res["timbre"] is None
