#!/usr/bin/env python3
"""
Build-time parametric (peaking/bell) EQ for sample files.

Strudel only offers gain/lpf/hpf — no bell cut — so when a pitched-sample
instrument carries a spectral excess in one band (e.g. the lead's high-mid),
the fix is to bake a peaking cut into the samples BEFORE hosting them.

Note: a fixed-Hz notch shifts with playback rate when Strudel pitch-shifts the
sample, so keep Q moderate (~1) and target the middle of the offending band.

Usage:
  eq_samples.py FILE [FILE ...] --freq 3000 --q 1.0 --gain-db -8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def peaking_biquad(fs: float, f0: float, q: float, gain_db: float):
    """RBJ cookbook peaking-EQ biquad coefficients (b, a)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_eq(path: Path, f0: float, q: float, gain_db: float) -> None:
    y, sr = sf.read(str(path), always_2d=False)
    b, a = peaking_biquad(sr, f0, q, gain_db)
    if y.ndim == 1:
        out = lfilter(b, a, y)
    else:
        out = np.stack([lfilter(b, a, y[:, c]) for c in range(y.shape[1])], axis=1)
    peak = float(np.max(np.abs(out)))
    if peak > 0.999:  # guard against the boost case / numerical overshoot
        out = out / peak * 0.98
    sf.write(str(path), out.astype(np.float32), sr, subtype="PCM_16")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake a peaking-EQ cut/boost into sample files.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--freq", type=float, required=True, help="center frequency (Hz)")
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--gain-db", type=float, required=True, help="negative = cut")
    args = ap.parse_args()

    done = 0
    for f in args.files:
        if not f.exists():
            print(f"skip (missing): {f}", file=sys.stderr)
            continue
        apply_eq(f, args.freq, args.q, args.gain_db)
        done += 1
    print(f"EQ applied to {done} file(s): {args.gain_db:+.1f}dB @ {args.freq:.0f}Hz Q{args.q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
