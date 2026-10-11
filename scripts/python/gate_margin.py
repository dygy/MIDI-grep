#!/usr/bin/env python3
"""Smallest margin of one comparison.json to the eval-gate floors (overall and section-aware).

Used by scripts/auto-calibrate.sh to pick the BEST iteration: a render that passes both floors beats one
that does not, and among failures the one closest to passing (largest minimum margin) wins - not simply
the highest overall score. Floors come from eval/thresholds.yaml through eval/gate.py's own loader
(per-mode block selected by --mode), never retyped here.

stdout: ``<min_margin> <PASS|FAIL> <overall> <section_aware|-> <floor> <sa_floor|->``; exit 2 on any
problem (missing yaml, unreadable comparison) so the caller can fall back and say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def margins(comparison: Path, genre: str, mode: str | None) -> dict:
    from eval import gate

    thr = gate.load_thresholds()
    comp = json.loads(comparison.read_text())["comparison"]
    overall = float(comp["overall_similarity"])
    sa = comp.get("section_aware_similarity")
    floor = gate.floor_for_genre(genre, thr, mode)
    sa_floor = gate.section_aware_floor_for_genre(genre, thr, mode)
    ms = [overall - floor]
    if sa is not None and sa_floor is not None:
        ms.append(float(sa) - sa_floor)
    return {"margin": min(ms), "overall": overall, "sa": sa, "floor": floor, "sa_floor": sa_floor}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--genre", required=True)
    ap.add_argument("--mode", default=None)
    a = ap.parse_args()
    try:
        m = margins(a.comparison, a.genre, a.mode)
    except Exception as exc:  # noqa: BLE001
        print(f"gate_margin: {exc}", file=sys.stderr)
        return 2
    sa = "-" if m["sa"] is None else f"{m['sa']:.4f}"
    saf = "-" if m["sa_floor"] is None else f"{m['sa_floor']:.4f}"
    print(f"{m['margin']:.4f} {'PASS' if m['margin'] >= 0 else 'FAIL'} {m['overall']:.4f} {sa} {m['floor']:.4f} {saf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
