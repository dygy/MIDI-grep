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
             spectral centroid of both (level-independent brightness: > 1 = the render is too dark), and the
             ARTICULATION of both (``articulation``): how much of each note window the voices sound.
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


def articulation(yo: np.ndarray, ys: np.ndarray, sr: int = SR) -> dict:
    """How long the notes sound, stem vs render: per note window (consecutive onsets OF THE STEM, so both signals are cut
    at the same times and no written note is needed), the energy-equivalent duration of the band envelope over the
    window, as a fraction of it: sum(((e - trough) / (peak - trough))^2) / frames. A note held at its peak until the next
    one reads near 1, one that falls evenly across the window 1/3, a stab that dies at once near 0; no threshold. The band is the voices' body (from the bottom of low_mid up):
    the melodic stem's bass bleed and pads below it would hold every window up.

    BibiBabydoll "Melhor Sobrar" synth (2026-11-15, 150 BPM): stem 0.185, the lead at the generator's clip 1.3 0.33
    (clip 0.5: 0.27, 0.2: 0.19); Regime CLT stem 0.313, render 0.307. Shorter notes on the first: 0.787 -> 0.804; on
    the references the same clip 0.5 cost 0.007-0.026 - so it is measured per track, never a constant."""
    import librosa
    from compare_audio import FREQ_BANDS
    hop, n_fft = 128, 512
    n = min(len(yo), len(ys))
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = f >= FREQ_BANDS["low_mid"][0]
    eo, er = (np.sqrt((np.abs(librosa.stft(y[:n], n_fft=n_fft, hop_length=hop)) ** 2)[band].sum(0)) for y in (yo, ys))
    on = librosa.onset.onset_detect(onset_envelope=librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(eo[None, :] + EPS), sr=sr, hop_length=hop), sr=sr, hop_length=hop)

    def ed(e: np.ndarray) -> float | None:
        v = []
        for a, b in zip(on[:-1], on[1:]):
            if b - a < 3:
                continue
            seg = e[a:b]
            pk, tr = float(seg.max()), float(seg.min())
            if pk > tr:
                v.append(float(np.sum(((seg - tr) / (pk - tr)) ** 2)) / (b - a))
        return float(np.median(v)) if v else None

    return {"orig": ed(eo), "render": ed(er) if np.any(ys) else None, "windows": int(max(len(on) - 1, 0))}


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
    out["articulation"] = articulation(yo, ys)
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
