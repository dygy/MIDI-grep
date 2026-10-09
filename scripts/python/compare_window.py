#!/usr/bin/env python3
"""Ground-truth self-test: compare a TIME WINDOW of rendered stems vs original stems on real
acoustic features — not aggregate similarity scores.

The aggregate gate can read "85%" while the arrangement is wrong (e.g. bass playing through an
intro where the original is silent). This tool surfaces that by reporting, per stem, over a window:
  - presence  : is the voice actually active, or (near) silent?  ← catches voice-presence mismatches
  - level     : RMS (loudness)
  - brightness: spectral centroid (Hz)
  - density   : onset count + first onset times

Usage:
  python compare_window.py <track_cache_dir> <version> [--start S] [--dur D]
  python compare_window.py ".cache/stems/<track>" v013 --start 0 --dur 5
  # multiple windows at once:
  python compare_window.py ".cache/stems/<track>" v013 --windows 0:5,30:5,60:5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("librosa required", file=sys.stderr)
    sys.exit(2)

SR = 22050
STEMS = ("bass", "drums", "melodic")
# A stem is "active" in a window if its loudest frame is well above the noise floor.
ACTIVE_RMS = 0.005


def _features(path: Path, start: float, dur: float) -> dict | None:
    if not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=SR, offset=start, duration=dur, mono=True)
    if len(y) == 0:
        return None
    rms = float(np.sqrt(np.mean(y ** 2)))
    env = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peak = float(env.max()) if len(env) else 0.0
    silent_frac = float(np.mean(env < (peak * 0.1 + 1e-9))) if len(env) else 1.0
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time")
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR))) if rms > 1e-6 else 0.0
    return {
        "rms": rms,
        "active": peak >= ACTIVE_RMS,
        "silent_frac": silent_frac,
        "centroid": cent,
        "onsets": len(onsets),
        "onset_t": [round(float(t), 2) for t in onsets[:6]],
    }


def _rendered_stem(version_dir: Path, stem: str) -> Path:
    for ext in (".wav", ".mp3"):
        p = version_dir / f"render_{stem}{ext}"
        if p.exists():
            return p
    return version_dir / f"render_{stem}.wav"


def compare_window(track_dir: Path, version_dir: Path, start: float, dur: float):
    print(f"\n=== Window {start:.0f}–{start+dur:.0f}s  (orig vs {version_dir.name}) ===")
    issues = []
    for stem in STEMS:
        fo = _features(track_dir / f"{stem}.wav", start, dur)
        fr = _features(_rendered_stem(version_dir, stem), start, dur)
        if fo is None or fr is None:
            print(f"  {stem:8}: missing stem (orig={fo is not None}, rend={fr is not None})")
            continue
        print(f"  {stem:8}: "
              f"orig[{'ON ' if fo['active'] else 'off'} rms {fo['rms']:.4f} {fo['centroid']:>5.0f}Hz "
              f"on {fo['onsets']:>2} sil {fo['silent_frac']:.0%}]  "
              f"rend[{'ON ' if fr['active'] else 'off'} rms {fr['rms']:.4f} {fr['centroid']:>5.0f}Hz "
              f"on {fr['onsets']:>2} sil {fr['silent_frac']:.0%}]")
        # The key check: voice-presence mismatch.
        if fo["active"] != fr["active"]:
            who = "rendered plays but original silent" if fr["active"] else "original plays but rendered silent"
            issues.append(f"{stem} @ {start:.0f}s: PRESENCE MISMATCH ({who})")
    if issues:
        print("  ⚠ " + "\n  ⚠ ".join(issues))
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare rendered vs original stems over time windows")
    ap.add_argument("track_dir", help="track cache dir holding bass/drums/melodic.wav")
    ap.add_argument("version", help="version dir name (e.g. v013) under track_dir")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--windows", default=None, help="comma list 'start:dur,start:dur' (overrides --start/--dur)")
    args = ap.parse_args(argv)

    track_dir = Path(args.track_dir)
    version_dir = track_dir / args.version
    if not version_dir.exists():
        print(f"version dir not found: {version_dir}", file=sys.stderr)
        return 2

    windows = ([(float(a), float(b)) for a, b in (w.split(":") for w in args.windows.split(","))]
               if args.windows else [(args.start, args.dur)])
    all_issues = []
    for start, dur in windows:
        all_issues += compare_window(track_dir, version_dir, start, dur)

    print(f"\n=== {len(all_issues)} presence mismatch(es) across {len(windows)} window(s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
