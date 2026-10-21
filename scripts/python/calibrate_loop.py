#!/usr/bin/env python3
"""Control logic of the calibration loop (scripts/auto-calibrate.sh): which iteration is best, whether to stop,
where the next step starts from and how long it is. Kept out of bash so it is testable.

Rules (all comparisons use ``NOISE`` = eval/thresholds.yaml ``calibration_noise``, MEASURED: see there):
  * ranking     passing the gate beats failing; among equals the larger minimum margin wins UNLESS the margins
                differ by <= NOISE (a tie), in which case higher overall wins, then higher section-aware.
  * regression  a step whose result is worse than the best so far beyond NOISE (same rule). The next step does
                not continue from it: it starts again from the BEST knobs along the same proposed direction with
                a shorter step, alpha = alpha_trial * sqrt(NOISE / drop). Derivation: the loss along the step is
                taken as quadratic (zero slope at the start, the conservative case), so the step whose loss
                equals NOISE is sqrt(NOISE/drop) of the one that lost ``drop``. Derived from the observed
                response, not a constant.
  * stopping    PATIENCE consecutive steps that did not beat the best-so-far beyond NOISE, or MAX_ITERS iterations
                (the caller's ``--iters N`` replaces MAX_ITERS as the hard cap).

Subcommands (stdout is machine-read by auto-calibrate.sh):
  max-iters | patience | cap <track|neighbour|seed|cold>   (cap: the budget a warm start earns, WARM_ITERS)
  select <iters.tsv>                      -> best tag
  next   <iters.tsv> --cap N              -> "STOP <reason>" | "CONTINUE <parent-tag> <alpha>"
  blend  --parent "<9 knobs>" --prop "<9 knobs>" --eff "<bass_lpf lead_hpf>" --alpha A -> 9 knobs
         (BM SG CL LLPF HG MG BL LH SU; SU = sub-sine octave up, 0/1, not interpolated)
  blend-env --used <json|-> --new <json> --alpha A --out <json>
  shape  <iters.tsv> <best-tag> <shape.tsv> -> "<winner-shape> <winner-tag>": the bass-shape trial (the BEST knobs
         rendered once with the other --bass-shape). The non-default shape wins only if it improves on the default
         one beyond NOISE (same rule as a step); a tie keeps the default.
iters.tsv columns: tag overall margin gate_out params envused sa parent alpha shape
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# PATIENCE = 2: the smallest K that has seen both a full step and the reduced-step retry that follows a
# regression fail to beat the best (K=1 would stop right after the revert, before the shorter step is tried).
PATIENCE = 2
# MAX_ITERS = 8: the repo's VAGABUNDO/Regime runs reached their best by iteration 3-5 at ~1.5 min per
# generate+render+compare; 8 leaves room for a revert-and-retry cycle or two and bounds a mode to ~12 min.
MAX_ITERS = 8
# WARM_ITERS = 3: a warm start (this track's own recorded best, or the nearest recorded track's) renders that best
# as cal1 and needs only a verifying step plus one correction; the observed warm-vs-cold cost was 5-8 renders.
# 3 = cal1 + the two steps PATIENCE needs to see (a full step and the shortened retry after a regression).
WARM_ITERS = 3
WARM_SOURCES = ("track", "neighbour")      # sources that earn the short budget; "seed" / "cold" run the full MAX_ITERS
FIELDS = ("tag", "overall", "margin", "gate_out", "params", "envused", "sa", "parent", "alpha", "shape")
SHAPES = ("stem", "off")        # generate_dynamic_strudel.py --bass-shape; the first is the generator default (wins ties)


def iteration_cap(source: str) -> int:
    """Iteration budget implied by where cal1's knobs came from (see WARM_ITERS)."""
    return min(WARM_ITERS, MAX_ITERS) if source in WARM_SOURCES else MAX_ITERS


def noise() -> float:
    if os.environ.get("AC_NOISE"):          # tests
        return float(os.environ["AC_NOISE"])
    import yaml
    thr = yaml.safe_load((ROOT / "eval" / "thresholds.yaml").read_text())
    if "calibration_noise" not in thr:
        raise SystemExit("eval/thresholds.yaml has no calibration_noise")
    return float(thr["calibration_noise"])


class Row:
    def __init__(self, line: str):
        f = line.rstrip("\n").split("\t")
        f += [""] * (len(FIELDS) - len(f))
        d = dict(zip(FIELDS, f))
        self.tag = d["tag"]
        self.overall = float(d["overall"])
        self.margin = None if d["margin"] == "n/a" else float(d["margin"])
        g = d["gate_out"].split()
        self.passed = None if self.margin is None or len(g) < 2 else g[1] == "PASS"
        self.sa = float(d["sa"]) if d["sa"] not in ("", "n/a", "-") else 0.0
        self.alpha = float(d["alpha"]) if d["alpha"] else 1.0
        self.shape = d["shape"] if d["shape"] in SHAPES else SHAPES[0]


def rows_of(path: str) -> list[Row]:
    return [Row(ln) for ln in open(path) if ln.strip()]


def improves(a: Row, b: Row, n: float) -> bool:
    """a is better than b beyond noise."""
    if a.margin is not None and b.margin is not None:
        if a.passed != b.passed:
            return bool(a.passed)
        if abs(a.margin - b.margin) > n:
            return a.margin > b.margin
    return a.overall - b.overall > n


def drop(best: Row, t: Row, n: float) -> float:
    d = best.overall - t.overall
    if best.margin is not None and t.margin is not None:
        d = max(d, best.margin - t.margin)
    return max(d, n)


def select(rows: list[Row], n: float) -> Row:
    if all(r.margin is not None for r in rows):
        top = max(rows, key=lambda r: (bool(r.passed), r.margin))
        group = [r for r in rows if bool(r.passed) == bool(top.passed) and r.margin >= top.margin - n]
    else:
        group = rows        # gate unreadable somewhere: margins are not comparable, rank everything by overall
    return max(group, key=lambda r: (r.overall, r.sa))


def next_step(rows: list[Row], cap: int, patience: int, n: float) -> tuple:
    best = select(rows, n)
    stall = 0
    for i in range(len(rows)):
        if i == 0 or improves(rows[i], select(rows[:i], n), n):
            stall = 0
        else:
            stall += 1
    if len(rows) >= cap:
        return ("STOP", f"iteration cap {cap} reached")
    if stall >= patience:
        return ("STOP", f"no improvement beyond noise {n:g} for {stall} consecutive steps")
    last = rows[-1]
    if last is not best and improves(best, last, n):
        alpha = last.alpha * math.sqrt(n / drop(best, last, n))
        return ("CONTINUE", best.tag, f"{alpha:.4f}")
    return ("CONTINUE", last.tag, "1")


def choose_shape(best: Row, trial: Row, n: float) -> Row:
    """Of the loop's BEST render and the other-bass-shape trial, the one to keep: the default shape unless the
    other improves on it beyond noise."""
    default, other = (best, trial) if best.shape == SHAPES[0] else (trial, best)
    return other if improves(other, default, n) else default


def blend_knobs(parent: list[str], prop: list[str], eff: list[float], alpha: float) -> list[str]:
    out = []
    for i, (p, q) in enumerate(zip(parent, prop)):
        pf, qf = float(p), float(q)
        if i == 8:                              # SU, the sub-sine octave (0/1): a hypothesis, not a magnitude - taken whole
            out.append(q if alpha >= 1.0 else p)   # at a full step, or kept at the parent's value on a shortened retry
            continue
        if i >= 6 and qf == pf:                 # BL / LH override untouched by the calibrator: keep (0 = default)
            out.append(p)
            continue
        base = pf if (i < 6 or pf > 0) else float(eff[i - 6])
        v = base + alpha * (qf - base)
        out.append(str(int(round(v))) if i in (3, 6, 7) else f"{v:.4g}")
    return out


def blend_env(used: dict | None, new: dict, alpha: float) -> dict:
    out = json.loads(json.dumps(new))
    lo, hi = new.get("clamp", [0.5, 2.0])
    for v, curve in new["voices"].items():
        u = (used or {}).get("voices", {}).get(v) or [1.0] * len(curve)
        if len(u) != len(curve):
            u = [1.0] * len(curve)
        out["voices"][v] = [round(max(lo, min(hi, (a ** (1 - alpha)) * (b ** alpha))), 4) for a, b in zip(u, curve)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("max-iters")
    sp.add_parser("patience")
    c = sp.add_parser("cap"); c.add_argument("source", help="warm-start source: track | neighbour | seed | cold")
    sp.add_parser("noise")
    s = sp.add_parser("select"); s.add_argument("tsv")
    x = sp.add_parser("next"); x.add_argument("tsv"); x.add_argument("--cap", type=int, required=True)
    x.add_argument("--patience", type=int, default=None)
    b = sp.add_parser("blend"); b.add_argument("--parent", required=True); b.add_argument("--prop", required=True)
    b.add_argument("--eff", default="0 0"); b.add_argument("--alpha", type=float, required=True)
    h = sp.add_parser("shape"); h.add_argument("tsv"); h.add_argument("best_tag"); h.add_argument("shape_tsv")
    e = sp.add_parser("blend-env"); e.add_argument("--used", required=True); e.add_argument("--new", required=True)
    e.add_argument("--alpha", type=float, required=True); e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "max-iters":
        print(MAX_ITERS)
    elif a.cmd == "patience":
        print(int(os.environ.get("AC_PATIENCE", PATIENCE)))
    elif a.cmd == "cap":
        print(iteration_cap(a.source))
    elif a.cmd == "select":
        print(select(rows_of(a.tsv), noise()).tag)
    elif a.cmd == "noise":
        print(noise())
    elif a.cmd == "next":
        pat = a.patience if a.patience is not None else int(os.environ.get("AC_PATIENCE", PATIENCE))
        print(*next_step(rows_of(a.tsv), a.cap, pat, noise()))
    elif a.cmd == "blend":
        print(*blend_knobs(a.parent.split(), a.prop.split(), [float(v) for v in a.eff.split()], a.alpha))
    elif a.cmd == "shape":
        best = next(r for r in rows_of(a.tsv) if r.tag == a.best_tag)
        win = choose_shape(best, rows_of(a.shape_tsv)[0], noise())
        print(win.shape, win.tag)
    elif a.cmd == "blend-env":
        new = json.loads(Path(a.new).read_text())
        used = json.loads(Path(a.used).read_text()) if a.used != "-" and Path(a.used).is_file() else None
        Path(a.out).write_text(json.dumps(blend_env(used, new, a.alpha), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
