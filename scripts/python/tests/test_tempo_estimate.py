# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""compare_audio.py tempo estimate: robust to dense off-grid content, never biased to agree.

Synthetic 20 s kick/click signals (soundfile WAVs, no Demucs / render) through the real
`compare_audio.compare_audio()`:

  (a) 136 render vs 136 original         -> tempo_similarity >= 0.98
  (b) 123 render vs 136 original         -> rendered tempo ~123 (the 136 prior must NOT pull it)
  (c) 136 kick grid + dense off-grid
      high-band noise bursts             -> still ~136 (the low-band candidate wins)

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_tempo_estimate.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import compare_audio as ca  # noqa: E402

SR = 22050
DUR = 20.0


def _kick(n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / SR
    f = 40.0 + 110.0 * np.exp(-t * 40.0)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14.0)


def _kick_grid(bpm: float, rng: np.random.Generator, beat_hat: bool = True) -> np.ndarray:
    """Kick on every beat, hat on the off-beat; light noise floor."""
    y = np.zeros(int(DUR * SR))
    beat = 60.0 / bpm
    k = _kick(int(0.25 * SR))
    h = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.006 * SR)) * 0.25
    t = 0.2
    while t < DUR - 0.3:
        i = int(t * SR)
        y[i:i + len(k)] += k
        if beat_hat:
            j = int((t + beat / 2) * SR)
            if j + len(h) < len(y):
                y[j:j + len(h)] += h
        t += beat
    return y + 0.002 * rng.standard_normal(len(y))


def _off_grid_bursts(rng: np.random.Generator, n: int = 150) -> np.ndarray:
    """Dense high-band (>3 kHz) noise bursts at random, tempo-unrelated times."""
    from scipy.signal import butter, sosfilt
    y = np.zeros(int(DUR * SR))
    sos = butter(4, 3000, btype='highpass', fs=SR, output='sos')
    for t in rng.uniform(0.2, DUR - 0.3, n):
        i = int(t * SR)
        L = int(0.04 * SR)
        y[i:i + L] += sosfilt(sos, rng.standard_normal(L)) * np.exp(-np.arange(L) / (0.01 * SR)) * 0.5
    return y


def _write(path: Path, y: np.ndarray) -> str:
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    soundfile.write(str(path), y.astype(np.float32), SR)
    return str(path)


def _compare(tmp_path: Path, orig: np.ndarray, rend: np.ndarray) -> dict:
    o = _write(tmp_path / "orig.wav", orig)
    r = _write(tmp_path / "rend.wav", rend)
    res = ca.compare_audio(o, r, duration=int(DUR))
    assert res is not None and 'tempo_similarity' in res['comparison']
    return res


def test_same_tempo_scores_high(tmp_path):
    rng = np.random.default_rng(1)
    res = _compare(tmp_path, _kick_grid(136, rng), _kick_grid(136, np.random.default_rng(2)))
    c = res['comparison']
    assert c['tempo_similarity'] >= 0.98, (c['tempo_similarity'], res['rendered']['rhythm']['tempo'])


def test_true_123_render_is_not_pulled_to_136_by_the_prior(tmp_path):
    res = _compare(tmp_path, _kick_grid(136, np.random.default_rng(1)),
                   _kick_grid(123, np.random.default_rng(2)))
    rend = res['rendered']['rhythm']['tempo']
    assert abs(rend - 123.0) <= 4.0, f"rendered read {rend}, expected ~123 (prior must not bias)"
    assert res['comparison']['tempo_similarity'] < 0.7
    # both candidates + chosen method are recorded for transparency
    assert res['comparison']['tempo_method'] in {'full_band_beat_track', 'low_band_tempo'}
    assert len(res['comparison']['tempo_candidates']) == 2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_dense_off_grid_high_band_bursts_keep_the_kick_tempo(tmp_path, seed):
    rng = np.random.default_rng(seed)
    noisy = _kick_grid(136, rng, beat_hat=False) + 2.0 * _off_grid_bursts(rng, n=300)
    res = _compare(tmp_path, _kick_grid(136, np.random.default_rng(10)), noisy)
    rend = res['rendered']['rhythm']['tempo']
    cands = {c['method']: c['bpm'] for c in res['comparison']['tempo_candidates']}
    # The scenario is only meaningful if the legacy full-band estimator is fooled by the bursts.
    assert abs(cands['full_band_beat_track'] - 136.0) > 4.0, cands
    assert abs(rend - 136.0) <= 2.0, (rend, cands)
    assert res['comparison']['tempo_method'] == 'low_band_tempo'
    assert res['comparison']['tempo_similarity'] >= 0.9


def test_estimate_tempo_records_candidates():
    y = _kick_grid(136, np.random.default_rng(5))
    bpm, info = ca.estimate_tempo(y, sr=SR, prior_bpm=136.0)
    assert info['tempo_method'] in {'full_band_beat_track', 'low_band_tempo'}
    assert {c['method'] for c in info['tempo_candidates']} >= {'full_band_beat_track'}
    assert abs(bpm - 136.0) <= 4.0
