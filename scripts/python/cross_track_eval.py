#!/usr/bin/env python3
"""T4 — cross-track regression + calibration harness for the multi-dimensional stem self-test.

Two jobs:

1. CALIBRATE the loose pitch/rhythm/timbre bars. A score is only meaningful relative to CHANCE, so
   for every dimension we measure both:
     - MATCHED  pairs: a track's render vs its OWN original stems  → signal
     - MISMATCHED pairs: a track's render vs ANOTHER track's stems → chance baseline
   A good bar sits between the mismatched mean and the matched mean. We recommend
       bar = clip(mismatched_mean + 0.5*(matched_mean - mismatched_mean))
   i.e. the midpoint, only when matched actually beats chance (else the dimension is non-discriminating
   and we flag it). This is what answers "is chroma 0.82 real signal or just chroma's high baseline?".

2. REGRESSION guard. Saves a baseline JSON of matched scores per (track, stem, dimension); --check
   re-runs and fails if any score drops more than --tol below baseline, so an algorithm change can't
   silently regress another genre.

Usage:
  python cross_track_eval.py                       # calibrate + write baseline
  python cross_track_eval.py --tracks 8            # cap fixture size (runtime)
  python cross_track_eval.py --check               # regression check vs saved baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stem_match as sm  # reuse the exact dimension math

CACHE = Path(__file__).resolve().parent.parent.parent / ".cache" / "stems"
BASELINE = Path(__file__).resolve().parent.parent.parent / "eval" / "stem_match_baseline.json"
DIMS = ("corr", "pitch", "rhythm", "timbre")
DIM_LABEL = {"corr": "shape", "pitch": "pitch", "rhythm": "rhythm", "timbre": "timbre"}


def discover(limit: int) -> list[tuple[str, str]]:
    """Tracks (sorted) that have original stems AND a render version; returns (track, version)."""
    out = []
    if not CACHE.exists():
        return out
    for d in sorted(CACHE.iterdir()):
        if not (d / "bass.wav").exists():
            continue
        for v in sorted(d.glob("v0*"), reverse=True):  # newest version first
            if (v / "render_bass.wav").exists() or (v / "render_bass.mp3").exists():
                out.append((d.name, v.name))
                break
        if len(out) >= limit:
            break
    return out


def eval_pair(orig_dir: Path, rend_dir: Path, stem: str) -> dict:
    """Dimension scores for one (original stem, rendered stem) pair — reuses stem_match._shape_corr."""
    o = sm._load(orig_dir / f"{stem}.wav")
    solo = sm._solo_drums_path(rend_dir) if stem == "drums" else None
    r = sm._load(solo) if solo is not None else sm._load(sm._rendered_path(rend_dir, stem))
    return sm._shape_corr(o, r, stem)


def _collect(matched_pairs: list, mismatched_pairs: list):
    """Returns matched[dim] -> [vals], mismatched[dim] -> [vals] aggregated over all stems/pairs."""
    matched = {d: [] for d in DIMS}
    mismatched = {d: [] for d in DIMS}
    for label, bucket, pairs in (("matched", matched, matched_pairs),
                                 ("mismatched", mismatched, mismatched_pairs)):
        for res, stem in pairs:
            for d in DIMS:
                v = res.get(d)
                if v is not None:
                    bucket[d].append(v)
    return matched, mismatched


def calibrate(tracks: list[tuple[str, str]]) -> dict:
    matched_pairs, mismatched_pairs = [], []
    per_track = {}  # track -> stem -> dims  (matched, for the regression baseline)

    n = len(tracks)
    for i, (track, ver) in enumerate(tracks):
        td = CACHE / track
        vd = td / ver
        per_track[track] = {}
        for stem in sm.STEMS:
            res = eval_pair(td, vd, stem)
            matched_pairs.append((res, stem))
            per_track[track][stem] = {d: res.get(d) for d in DIMS}
        # mismatched: this track's render vs the NEXT two tracks' originals (deterministic sample)
        for j in (i + 1, i + 2):
            other_td = CACHE / tracks[j % n][0]
            if other_td == td:
                continue
            for stem in sm.STEMS:
                mismatched_pairs.append((eval_pair(other_td, vd, stem), stem))
        print(f"  [{i + 1}/{n}] {track[:40]:40} / {ver}", file=sys.stderr)

    matched, mismatched = _collect(matched_pairs, mismatched_pairs)
    return {"per_track": per_track, "calibration": recommend_bars(matched, mismatched), "n_tracks": n}


def recommend_bars(matched: dict, mismatched: dict) -> dict:
    """Per-dimension threshold recommendation from matched (signal) vs mismatched (chance) value lists.
    Primary bar = chance_mean + 2σ ("beats random at ~95%"), robust even when matched renders are
    poor since it depends only on the chance distribution. A dimension is 'discriminating' only when
    matched clearly exceeds chance (sep > 0.02)."""
    recommend = {}
    for d in DIMS:
        mvals, xvals = matched.get(d, []), mismatched.get(d, [])
        if not mvals or not xvals:
            recommend[d] = {"note": "insufficient data"}
            continue
        mm, xm = float(np.mean(mvals)), float(np.mean(xvals))
        xs = float(np.std(xvals))
        sep = mm - xm
        chance_bar = round(xm + 2 * xs, 3)
        midpoint_bar = round(xm + 0.5 * sep, 3) if sep > 0.02 else None
        recommend[d] = {
            "matched_mean": round(mm, 3), "matched_std": round(float(np.std(mvals)), 3),
            "mismatched_mean": round(xm, 3), "mismatched_std": round(xs, 3),
            "separation": round(sep, 3),
            "chance_plus_2std_bar": chance_bar,
            "midpoint_bar": midpoint_bar,
            "recommended_bar": chance_bar,
            "discriminating": sep > 0.02,
        }
    return recommend


def _print_calibration(cal: dict):
    print("\n=== DIMENSION CALIBRATION (matched=signal vs mismatched=chance) ===")
    print(f"  {'dim':8} {'matched':>9} {'chance':>9} {'sep':>8} {'bar(2σ)':>9} {'midpt':>7}  verdict")
    for d in DIMS:
        c = cal["calibration"][d]
        if "note" in c:
            print(f"  {DIM_LABEL[d]:8} {c['note']}")
            continue
        mid = c["midpoint_bar"]
        verdict = "discriminates" if c["discriminating"] else "NON-discriminating (chance≈signal)"
        print(f"  {DIM_LABEL[d]:8} {c['matched_mean']:>9.3f} {c['mismatched_mean']:>9.3f} "
              f"{c['separation']:>+8.3f} {c['chance_plus_2std_bar']:>9.3f} "
              f"{('%.3f' % mid) if mid else '  --':>7}  {verdict}")
    print("  (recommended bar = chance + 2σ = 'beats random'; midpt needs good renders)")


def regression_check(cal: dict, tol: float) -> int:
    if not BASELINE.exists():
        print(f"No baseline at {BASELINE} — run without --check first.", file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text())["per_track"]
    cur = cal["per_track"]
    regressions = []
    for track, stems in base.items():
        if track not in cur:
            continue
        for stem, dims in stems.items():
            for d, bv in dims.items():
                cv = cur.get(track, {}).get(stem, {}).get(d)
                if bv is None or cv is None:
                    continue
                if cv < bv - tol:
                    regressions.append((track, stem, d, bv, cv))
    print(f"\n=== REGRESSION CHECK (tol {tol}) vs {BASELINE.name} ===")
    if not regressions:
        print("  ✓ no regressions")
        return 0
    for track, stem, d, bv, cv in regressions:
        print(f"  ✗ {track[:32]:32} {stem:8} {DIM_LABEL[d]:7} {bv:+.3f} → {cv:+.3f}")
    return 1


def main(argv=None) -> int:
    global BASELINE
    ap = argparse.ArgumentParser(description="Cross-track calibration + regression for stem_match")
    ap.add_argument("--tracks", type=int, default=8, help="max tracks in the fixture (runtime cap)")
    ap.add_argument("--check", action="store_true", help="regression check vs saved baseline")
    ap.add_argument("--tol", type=float, default=0.08, help="allowed score drop before flagging")
    ap.add_argument("--baseline", default=str(BASELINE))
    args = ap.parse_args(argv)
    BASELINE = Path(args.baseline)

    tracks = discover(args.tracks)
    if len(tracks) < 2:
        print("Need ≥2 tracks with stems+render for cross-track calibration.", file=sys.stderr)
        return 2
    print(f"Evaluating {len(tracks)} tracks (matched + mismatched)…", file=sys.stderr)
    cal = calibrate(tracks)
    _print_calibration(cal)

    if args.check:
        return regression_check(cal, args.tol)

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(cal, indent=2))
    print(f"\nBaseline written: {BASELINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
