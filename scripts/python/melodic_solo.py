#!/usr/bin/env python3
"""Level and brightness of the MELODIC voice group (LEAD + CHORDS) on its own, against the original melodic stem.

Why: the calibrator set cal-lead and lead-lpf from whole-mix shares and the whole-mix centroid, which the bass, the
kick and the hats also fill. VAGABUNDO's melodic voices rendered alone sat 7-9 dB under melodic.wav at 200 Hz-1.3 kHz
and 22 dB under below 200 Hz while the mix shares looked acceptable; matching that long-term spectrum alone lifts the
per-frame log-mel correlation of the voice from 0.639 to 0.772. The voice cannot be told from the other voices inside
the mix, but it can be rendered alone (seconds), exactly like the bass (``bass_solo.py``, whose helpers this reuses).

``solo``     strudel -> strudel with only the LEAD and CHORDS channels (header and shared definitions kept)
``measure``  per project band, the power of the original melodic stem (scaled by the master gain every voice is played
             at) over the power of the voice group rendered alone: ratio > 1 = the voices are too quiet. Plus the
             spectral centroid of both (level-independent brightness: > 1 = the render is too dark).
             Per-band powers are kept in the JSON so the calibrator picks the bands that define the voice's body.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bass_solo import BAND_MIN_FRAC, EPS, SR, _BLOCK_RE, _load, band_power  # noqa: E402

_MELODIC_TAG_RE = re.compile(r"//[^\n]*\b(?:LEAD|CHORDS)\s+—")


def solo_strudel(text: str) -> str | None:
    """The generated strudel with only its LEAD and CHORDS channels. None when the file has no separable melodic
    channel (a single ``stack()`` of channels, or no lead/chord voice): the caller then keeps the mix-share levers."""
    starts = [m.start() for m in _BLOCK_RE.finditer(text)]
    if not starts:
        return None
    ends = starts[1:] + [len(text)]
    keep = [b for b in (text[a:e] for a, e in zip(starts, ends)) if _MELODIC_TAG_RE.search(b)]
    if not keep:
        return None
    return text[:starts[0]] + "\n\n".join(b.rstrip() for b in keep) + "\n"


def centroid(y: np.ndarray, sr: int = SR) -> float:
    """Mean spectral centroid (Hz), the statistic ``compare_audio`` scores brightness with."""
    import librosa
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))


def measure(orig_melodic: Path, solo_render: Path, master_gain: float = 1.0, seconds: float | None = None) -> dict:
    """{"level_orig": {band: P(stem) * master_gain^2}, "level_render": {band: P(solo)}, "ratio": {band: orig/render},
    "centroid": {"orig": Hz, "render": Hz}}. A band the original stem barely occupies gives ratio None (no evidence)."""
    yo, ys = _load(orig_melodic, seconds), _load(solo_render, seconds)
    po, ps = band_power(yo), band_power(ys)
    total_o = max(sum(po.values()), EPS)
    out: dict = {"master_gain": master_gain, "level_orig": {}, "level_render": {}, "ratio": {},
                 "centroid": {"orig": centroid(yo), "render": centroid(ys) if np.any(ys) else None}}
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
    m.add_argument("--orig-melodic", type=Path, required=True)
    m.add_argument("--solo", type=Path, required=True)
    m.add_argument("--master-gain", type=float, default=1.0,
                   help="the generator's master gain (every voice is played at it): the stem is compared at that level")
    m.add_argument("--seconds", type=float, default=None)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "solo":
        text = solo_strudel(a.src.read_text())
        if text is None:
            print(f"melodic_solo: {a.src} has no separable LEAD/CHORDS channel", file=sys.stderr)
            return 3
        a.dst.write_text(text)
        return 0
    res = measure(a.orig_melodic, a.solo, a.master_gain, a.seconds)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    r, c = res["ratio"], res["centroid"]
    print("melodic voice level ratio (orig/render, power): " + "  ".join(
        f"{b} {('n/a' if v is None else f'{v:.2f}')}" for b, v in r.items() if b in ("low_mid", "mid"))
        + f"  centroid orig/render {c['orig']:.0f}/{(c['render'] or 0):.0f} Hz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
