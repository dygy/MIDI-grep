#!/usr/bin/env python3
"""Level of the BASS voice on its own, against the original bass stem.

Why: the calibrator used to set the bass layers from the WHOLE-mix band shares (sub_bass / bass). Those bands
are shared with the kick, so a kick that is too hot in the sub hides a bass voice that is far too quiet: on
VAGABUNDO the mix shares matched (sub_bass 34.7 % vs 34.1 %) while the bass voice rendered alone sat 8.5 dB under
the bass stem (20-45 Hz: -14.4 dB), and the loop LOWERED the bass multiplier (0.351 -> 0.331) because the mix sub
read "over". The voice cannot be told from the kick inside the mix, but it can be rendered alone (the offline
render of one channel takes seconds).

``solo``     strudel -> strudel with only the BASS channel sounding (others silenced, not removed: the kick's duck stays)
``measure``  per project band, the power of the original bass stem (scaled by the master gain every voice is played
             at) over the power of the bass voice rendered alone: ratio > 1 = the voice is too quiet.
             An ABSOLUTE level, not a share of the mix: a share saturates where the bass dominates the mix (measured on
             Regime: bass-mult 0.86 -> 1.24 left the share ratio at 1.66 -> 1.65), so it cannot see a gain change.
             Power ratio -> amplitude is the square root, which is exactly what ``calibrate_dynamic`` applies.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SR = 22050
EPS = 1e-12
BAND_MIN_FRAC = 1e-4   # a band holding less than this share of the original bass stem has no evidence (ratio = None)
_BLOCK_RE = re.compile(r"^\$:", re.M)   # melodic_solo's block scan


def solo_strudel(text: str) -> str | None:
    """The generated strudel with only its BASS channel sounding. None when the file has no separable BASS channel
    (a single ``stack()`` of channels, or no bass voice): the caller then keeps the mix-share levers.

    The other channels stay in the file, silenced (``solo_stems.solo_code``): the DRUMS channel's kick carries the
    ``duckorbit`` that dips the bass orbit, so dropping it would render the bass without the dip the stem has."""
    from solo_stems import StemSplitError, solo_code
    try:
        return solo_code(text, "bass")
    except StemSplitError:
        return None


def band_power(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Total STFT power per ``compare_audio.FREQ_BANDS`` band (same STFT as the score's band shares)."""
    import librosa
    from compare_audio import FREQ_BANDS
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    return {name: float(S[(freqs >= lo) & (freqs < hi), :].sum()) for name, (lo, hi) in FREQ_BANDS.items()}


def _load(path: Path, seconds: float | None) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=seconds)
    return y


def measure(orig_bass: Path, solo_render: Path, master_gain: float = 1.0, seconds: float | None = None) -> dict:
    """{"level_orig": {band: P(stem) * master_gain^2}, "level_render": {band: P(solo)}, "ratio": {band: orig/render}}.
    A band where the original bass has no energy gives ratio None (no evidence; nothing may be driven by it)."""
    po = band_power(_load(orig_bass, seconds))
    ps = band_power(_load(solo_render, seconds))
    total_o = max(sum(po.values()), EPS)
    out: dict = {"master_gain": master_gain, "level_orig": {}, "level_render": {}, "ratio": {}}
    for band in po:
        lo, lr = po[band] * master_gain ** 2, ps[band]
        out["level_orig"][band], out["level_render"][band] = lo, lr
        usable = po[band] > BAND_MIN_FRAC * total_o     # a band the stem itself barely occupies carries no evidence
        out["ratio"][band] = (lo / max(lr, EPS)) if usable else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solo")
    s.add_argument("src", type=Path)
    s.add_argument("dst", type=Path)
    m = sub.add_parser("measure")
    m.add_argument("--orig-bass", type=Path, required=True)
    m.add_argument("--solo", type=Path, required=True)
    m.add_argument("--master-gain", type=float, default=1.0,
                   help="the generator's master gain (every voice is played at it): the stem is compared at that level")
    m.add_argument("--seconds", type=float, default=None)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "solo":
        text = solo_strudel(a.src.read_text())
        if text is None:
            print(f"bass_solo: {a.src} has no separable BASS channel", file=sys.stderr)
            return 3
        a.dst.write_text(text)
        return 0
    res = measure(a.orig_bass, a.solo, a.master_gain, a.seconds)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    r = res["ratio"]
    print("bass voice level ratio (orig/render, power): " + "  ".join(
        f"{b} {('n/a' if v is None else f'{v:.2f}')}" for b, v in r.items() if b in ("sub_bass", "bass")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
