#!/usr/bin/env python3
"""Sub-bin tempo refinement by phase coherence.

``librosa.beat.beat_track`` returns a value on the tempogram's lag grid: BPM = 60*sr/hop/k for an
integer k. At sr 22050 / hop 512 the bins are ~6-7 BPM apart near 130 BPM, so the reported tempo can be
0.6-0.7 % off the real pulse (about 1 s, two beats, of drift over a 135 s comparison). This module takes
that coarse estimate and finds the period that maximises the *phase coherence* of the onset-strength
envelope -- the resultant length of the envelope energy placed on the beat circle,

    r(P) = | sum_t e(t) * exp(2*pi*i*t/P) | / sum_t e(t)

which is 1 when every onset falls at the same beat phase and ~0 when onsets are unrelated to P -- within
one tempogram bin of the coarse value. Nothing here is a per-track constant: the search window is the
tempogram's own bin width, and the period step is a fraction of the coherence peak's own width
(~P^2 / T for an excerpt of length T seconds).
"""
from __future__ import annotations

import numpy as np

REFINE_HOP = 128            # onset-envelope hop (samples); small so frame quantisation << beat period
PEAK_WIDTH_OVERSAMPLE = 20  # period step = (peak width) / this; the peak is also parabola-interpolated
TEMPOGRAM_HOP = 512         # librosa.beat.beat_track default hop_length (defines the coarse grid)


def tempogram_bin_bpm(bpm: float, sr: int, hop: int = TEMPOGRAM_HOP) -> float:
    """BPM distance between adjacent tempogram lag bins around ``bpm`` (bpm = 60*sr/hop/k)."""
    return bpm * bpm * hop / (60.0 * sr)


def refine_bpm(y: np.ndarray, sr: int, coarse_bpm: float, *, hop: int = REFINE_HOP,
               window_bpm: float | None = None) -> dict:
    """Refine ``coarse_bpm`` to the phase-coherence maximum within +-one tempogram bin.

    Returns ``{"bpm", "bpm_coarse", "confidence", "method"}``; ``confidence`` is the resultant length
    ``r`` at the peak (0..1). When the signal has no usable onset energy the coarse value is returned
    with confidence 0 and ``method`` ``"coarse"``.
    """
    import librosa

    coarse = float(coarse_bpm)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float64)
    env = env - env.min()
    total = float(env.sum())
    if total <= 0 or len(env) < 8:
        return {"bpm": coarse, "bpm_coarse": coarse, "confidence": 0.0, "method": "coarse"}
    fps = sr / hop
    t = np.arange(len(env)) / fps
    duration = float(t[-1]) or 1.0
    half = float(window_bpm) if window_bpm else tempogram_bin_bpm(coarse, sr)
    p_hi, p_lo = 60.0 / max(coarse - half, 1.0), 60.0 / (coarse + half)
    p0 = 60.0 / coarse
    step = p0 * p0 / (PEAK_WIDTH_OVERSAMPLE * duration)
    periods = np.arange(p_lo, p_hi + step, step)

    def coherence(ps: np.ndarray) -> np.ndarray:
        out = np.empty(len(ps))
        for i0 in range(0, len(ps), 256):           # chunked: bounded memory on long stems
            chunk = ps[i0:i0 + 256]
            ph = np.exp(2j * np.pi * t[None, :] / chunk[:, None])
            out[i0:i0 + 256] = np.abs(ph @ env) / total
        return out

    r = coherence(periods)
    k = int(np.argmax(r))
    best = float(periods[k])
    if 0 < k < len(periods) - 1:                    # parabolic interpolation of the peak
        a, b, c = r[k - 1], r[k], r[k + 1]
        denom = a - 2 * b + c
        if denom < 0:
            best += 0.5 * (a - c) / denom * step
    return {"bpm": 60.0 / best, "bpm_coarse": coarse, "confidence": float(r[k]), "method": "phase-coherence"}


def refine_bpm_file(path: str, coarse_bpm: float, sr: int = 22050) -> dict:
    """:func:`refine_bpm` on an audio file (mono, resampled to ``sr``)."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return refine_bpm(y, sr, coarse_bpm)
