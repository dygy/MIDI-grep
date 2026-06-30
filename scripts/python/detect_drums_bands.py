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

Output is drums.json-compatible: {"hits": [{"time": s, "type": "bd|sd|hh"}], "bpm": ...}.
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
            hits.append({"time": float(tm), "type": t})
        counts[t] = len(times)
    # snare often also fires the kick band; drop sd hits that coincide with a bd hit (±40ms)
    bd_times = np.array([h["time"] for h in hits if h["type"] == "bd"])
    cleaned = []
    for h in hits:
        if h["type"] == "sd" and bd_times.size:
            if np.min(np.abs(bd_times - h["time"])) < 0.04:
                continue
        cleaned.append(h)
    cleaned.sort(key=lambda h: h["time"])
    counts["sd"] = sum(1 for h in cleaned if h["type"] == "sd")
    return {"hits": cleaned, "bpm": bpm, "counts": counts}


def main() -> int:
    ap = argparse.ArgumentParser(description="Band-split drum onset detector.")
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
