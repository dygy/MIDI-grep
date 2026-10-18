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
    floor, floor_source = gate.resolve_floor(genre, thr, mode)
    sa_floor, sa_source = gate.resolve_section_aware_floor(genre, thr, mode)
    sa = comp.get("section_aware_similarity")
    # same scale selection as gate.evaluate_comparison (genre-wide floors are static-scale)
    overall, sa, _, _ = gate.select_scores(
        comp, float(comp["overall_similarity"]), None if sa is None else float(sa), floor_source, sa_source)
    ms = [overall - floor]
    if sa is not None and sa_floor is not None:
        ms.append(float(sa) - sa_floor)
    chance = comp.get("margin_over_chance")
    min_chance = thr.get("min_margin_over_chance")
    chance_margin = None
    if chance is not None and min_chance is not None:
        chance_margin = float(chance) - float(min_chance)
        ms.append(chance_margin)
    # A measurement failure (temporal scoring or the chance control) blocks the gate (eval/gate.py FAILs it);
    # it sorts below every real margin (real margins live in [-1, 1]) so BEST selection never prefers it.
    chance_block = comp.get("chance") or {}
    blocked = bool(comp.get("temporal_failed") or chance_block.get("error") or chance_block.get("temporal_failed"))
    if blocked:
        ms.append(-2.0)
    # the chance margin must be STRICTLY above min_margin_over_chance (spec: "more than"), so exactly 0 fails it
    passed = min(ms) >= 0 and not blocked and (chance_margin is None or chance_margin > 0)
    return {"margin": min(ms), "overall": overall, "sa": sa, "floor": floor, "sa_floor": sa_floor, "passed": passed}


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
    print(f"{m['margin']:.4f} {'PASS' if m['passed'] else 'FAIL'} {m['overall']:.4f} {sa} {m['floor']:.4f} {saf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
