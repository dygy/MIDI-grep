#!/usr/bin/env python3
"""
Band-split drum onset detector — a more reliable groove extractor than detect_drums.py.

detect_drums.py classifies each onset by a brittle energy-ratio rule and on this material
mislabels ~55% of hits as open-hats (oh=159 vs bd=45), producing a hat-dominated groove
that is both rhythmically wrong (onset_corr ~0.1) and far too bright (centroid 6.7k vs the
original drum stem's 2.3k).

Instead, this detects onsets INDEPENDENTLY in three bands of the drum stem and emits one
hit stream per drum voice, so each voice's timing comes from the band that actually carries
that instrument:
  kick (bd)  : low band   20-160 Hz
  snare (sd) : mid band   180-1200 Hz (skin/body) — gated to avoid kick bleed
  hat (hh)   : high band  6-11 kHz

A loud bar never ends with no hits (2026-11-08): Creagina bars 4-7 / 34-35 hold a pitched riff (93% of the energy at
500-2000 Hz, harmonic/percussive 10-25, ~9 attacks per bar) that Demucs filed under drums; no band's global threshold
fired there, the generator wrote empty drum bars and the render was digitally silent where the original played
(-15 dB vs the original, the track's worst envelope error). In a bar the generator's own gate calls present
(``bar_env_pattern``: RMS >= ``silence`` x the median active bar) with no hit from any band, the full-band onsets of
that bar are kept, each as the band whose peak is loudest relative to that band's own median hit. On the 18 other
tracks (2026-11-08) no bar qualifies, so nothing changes there.

Output is drums.json-compatible: {"hits": [{"time": s, "type": "bd|sd|hh", "strength": peak}], "bpm": ...}.
``strength`` is the linear peak of the hit's own band around its onset; the generator turns strength / class
median into the per-hit velocity it writes into the drum patterns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import scipy.signal as sps

SR = 22050
VERSION = 2   # stage 3 rebuilds drums_bands.json when this changes (editable-pipeline.sh, $PW/drums_bands.bpm)
# window around a detected onset in which the band peak is read: the attack lands within the detector's
# frame resolution before it (hop 256 @ 22.05 kHz = 11.6 ms), the hit's body within the next 60 ms
STRENGTH_PRE_S = 0.012
STRENGTH_POST_S = 0.060


def _bandpass(y, lo, hi):
    nyq = SR / 2
    lo_n = max(1e-4, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    b, a = sps.butter(2, [lo_n, hi_n], btype="band")
    return sps.lfilter(b, a, y)


def _onsets(env, sr_env, *, delta_frac=0.30, min_gap_s=0.06):
    """Pick peaks on a band onset-strength envelope. delta_frac sets the prominence relative
    to the band's own dynamic range; min_gap_s prevents double-triggering one hit."""
    if env.max() < 1e-6:
        return np.array([], dtype=int)
    thr = env.mean() + delta_frac * (env.max() - env.mean())
    min_gap = int(round(min_gap_s * sr_env))
    peaks, _ = sps.find_peaks(env, height=thr, distance=max(1, min_gap))
    return peaks


def detect(stem_path: Path, bpm: float | None) -> dict:
    y, _ = librosa.load(str(stem_path), sr=SR, mono=True)
    hop = 256
    sr_env = SR / hop
    bands = {
        "bd": (20, 160, 0.28),
        "sd": (180, 1200, 0.34),
        "hh": (6000, 11000, 0.40),
    }
    hits = []
    counts = {}
    for t, (lo, hi, df) in bands.items():
        yb = _bandpass(y, lo, hi)
        env = librosa.onset.onset_strength(y=yb, sr=SR, hop_length=hop)
        pk = _onsets(env, sr_env, delta_frac=df)
        times = librosa.frames_to_time(pk, sr=SR, hop_length=hop)
        for tm in times:
            # measured hit strength: peak amplitude of this band's signal around the onset (linear, so
            # a ratio of two strengths is the gain ratio the original played them at)
            i0 = max(0, int((float(tm) - STRENGTH_PRE_S) * SR))
            i1 = min(len(yb), int((float(tm) + STRENGTH_POST_S) * SR) + 1)
            strength = float(np.max(np.abs(yb[i0:i1]))) if i1 > i0 else 0.0
            hits.append({"time": float(tm), "type": t, "strength": strength})
        counts[t] = len(times)
    # snare often also fires the kick band; drop sd hits that coincide with a bd hit (±40ms)
    bd_times = np.array([h["time"] for h in hits if h["type"] == "bd"])
    cleaned = []
    for h in hits:
        if h["type"] == "sd" and bd_times.size:
            if np.min(np.abs(bd_times - h["time"])) < 0.04:
                continue
        cleaned.append(h)
    if bpm:
        cleaned += _uncovered_bar_hits(y, bpm, cleaned, {t: (lo, hi) for t, (lo, hi, _) in bands.items()})
    cleaned.sort(key=lambda h: h["time"])
    counts = {t: sum(1 for h in cleaned if h["type"] == t) for t in bands}
    counts["uncovered_bar_hits"] = sum(1 for h in cleaned if h.get("uncovered_bar"))
    return {"hits": cleaned, "bpm": bpm, "counts": counts, "version": VERSION}


def _uncovered_bar_hits(y: np.ndarray, bpm: float, hits: list[dict], bands: dict[str, tuple[int, int]]) -> list[dict]:
    """Hits for the bars the generator's gate calls loud but no band fired in (see the module docstring)."""
    import inspect
    sys.path.insert(0, str(Path(__file__).parent))
    import generate_dynamic_strudel as G
    silence = inspect.signature(G.bar_env_pattern).parameters["silence"].default
    bar = 240.0 / bpm
    hb = int(round(bar * SR))
    n = len(y) // hb
    if n == 0:
        return []
    rms = np.array([np.sqrt(np.mean(y[i * hb:(i + 1) * hb] ** 2)) for i in range(n)])
    active = rms[rms > rms.max() * 0.01]            # bar_env_pattern's reference: the median ACTIVE unit
    ref = float(np.median(active)) if active.size else 0.0
    covered = {int(h["time"] // bar) for h in hits}
    filt = {t: _bandpass(y, lo, hi) for t, (lo, hi) in bands.items()}
    med = {t: float(np.median([h["strength"] for h in hits if h["type"] == t]))
           for t in bands if any(h["type"] == t for h in hits)}
    out = []
    for i in range(n):
        if i in covered or rms[i] < silence * ref or not med:
            continue
        seg = y[i * hb:(i + 1) * hb]
        for tm in librosa.onset.onset_detect(y=seg, sr=SR, units="time"):
            t_abs = i * bar + float(tm)
            i0 = max(0, int((t_abs - STRENGTH_PRE_S) * SR))
            i1 = min(len(y), int((t_abs + STRENGTH_POST_S) * SR) + 1)
            peak = {t: float(np.max(np.abs(filt[t][i0:i1]))) for t in med}
            band = max(peak, key=lambda t: peak[t] / med[t])
            out.append({"time": t_abs, "type": band, "strength": peak[band], "uncovered_bar": True})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Band-split drum onset detector.")
    if "--version" in sys.argv[1:]:
        print(VERSION)
        return 0
    ap.add_argument("stem", type=Path, help="drum stem wav")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.stem.exists():
        print(f"missing {args.stem}", file=sys.stderr)
        return 1
    res = detect(args.stem, args.bpm)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"out": str(args.out), "counts": res["counts"],
                      "total": len(res["hits"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
