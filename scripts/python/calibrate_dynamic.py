#!/usr/bin/env python3
"""Data-driven calibrator for ``generate_dynamic_strudel.py``.

Closes the hand-tuning loop: read a render's measured ``comparison.json`` (the honest
MAE band balance + energy from ``compare_audio.py``) and emit the NEXT set of generator
tuning knobs so the following render's per-band balance moves toward the original's.

This is the system's "AI learns from analysis, never hardcodes" principle applied to the
mix: every knob delta is a damped proportional correction of an OBSERVED band ratio
(original/rendered), not a value tuned by hand for one track. The same math generalizes to
any track/genre because it only ever reacts to that render's own measured gap.

Levers (all multiply the *current* value, sqrt-damped + clamped to avoid oscillation):
  sub-gain   <- sub_bass band   (raise to fill 20-60 Hz, lower if booming)
  bass-mult  <- bass band       (60-250 Hz body of the pitched bass sample)
  cal-lead   <- low_mid+mid     (lead/melodic body; the usual deficit)
  lead-lpf   <- high+high_mid   (excess brightness -> lower the cutoff)
  master-gain<- overall RMS      (gentle; the recorder limiter caps absolute loudness)

Usage:
  python calibrate_dynamic.py --comparison <comparison.json> \
      [--bass-mult 0.62 --sub-gain 0.5 --lead-lpf 6500 --master-gain 0.6 --cal-lead 1.0] \
      [--state <prev_params.json>] [--out <next_params.json>]

Prints a human-readable diff and a ready-to-paste CLI fragment; writes the next param set
as JSON (consumed by the wrapper to drive the next ``generate_dynamic_strudel`` run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def damp(ratio: float, *, strength: float = 0.5, lo: float = 0.7, hi: float = 1.4) -> float:
    """Damped, clamped multiplicative correction for an observed orig/rend band ratio.

    ``strength`` is the exponent (0.5 = sqrt = half-step toward target each pass), which
    converges without overshoot. ``lo``/``hi`` cap a single step so one noisy band can't
    swing a knob wildly.
    """
    if ratio <= 0:
        return 1.0
    corr = ratio ** strength
    return max(lo, min(hi, corr))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison", required=True, type=Path,
                    help="comparison.json from compare_audio.py for the LAST render")
    # current generator knob values (what produced the last render). Defaults match the
    # generator's own defaults so the calibrator is usable from a cold start.
    ap.add_argument("--bass-mult", type=float, default=0.40)
    ap.add_argument("--sub-gain", type=float, default=0.7)
    ap.add_argument("--cal-lead", type=float, default=1.0)
    ap.add_argument("--lead-lpf", type=int, default=5000)
    ap.add_argument("--hat-gain", type=float, default=0.0)
    ap.add_argument("--master-gain", type=float, default=0.6)
    ap.add_argument("--state", type=Path, default=None,
                    help="prev_params.json to read current knob values from (overrides the "
                         "per-knob flags above for any key it contains)")
    ap.add_argument("--out", type=Path, default=None, help="write next params as JSON here")
    args = ap.parse_args()

    cur = {
        "bass_mult": args.bass_mult,
        "sub_gain": args.sub_gain,
        "cal_lead": args.cal_lead,
        "lead_lpf": float(args.lead_lpf),
        "hat_gain": args.hat_gain,
        "master_gain": args.master_gain,
    }
    if args.state and args.state.exists():
        prev = json.loads(args.state.read_text())
        for k in cur:
            if k in prev:
                cur[k] = prev[k]

    data = json.loads(args.comparison.read_text())
    ob = data["original"]["bands"]
    rb = data["rendered"]["bands"]
    comp = data["comparison"]

    eps = 1e-4

    def ratio(band_keys: list[str]) -> float:
        o = sum(ob.get(k, 0.0) for k in band_keys)
        r = sum(rb.get(k, 0.0) for k in band_keys)
        return (o + eps) / (r + eps)

    r_sub = ratio(["sub_bass"])
    r_bass = ratio(["bass"])
    r_body = ratio(["low_mid", "mid"])       # lead/melodic body
    # Brightness lever uses the spectral CENTROID, not the high bands: in bass-heavy genres
    # the high/high_mid bands sit at ~1% magnitude where demucs bleed + noise swamp the
    # signal, so their ratio is unreliable (it kept lead-lpf pinned while the centroid said
    # the mix was clearly too dull). The centroid is the stable, perceptual brightness metric.
    cent_o = data["original"]["spectral"]["centroid_mean"]
    cent_r = data["rendered"]["spectral"]["centroid_mean"]
    r_cent = (cent_o + eps) / (cent_r + eps)   # >1 => rendered too dark, open the cutoff

    nxt = dict(cur)
    # --- sub_bass: fill or tame the 20-60 Hz sine layer -----------------------------------
    nxt["sub_gain"] = round(clamp(cur["sub_gain"] * damp(r_sub), 0.2, 1.6), 3)
    # --- bass (60-250 Hz): body of the pitched bass sample --------------------------------
    nxt["bass_mult"] = round(clamp(cur["bass_mult"] * damp(r_bass), 0.2, 1.3), 3)
    # --- lead/melodic body: raise gain when low_mid+mid is deficient -----------------------
    nxt["cal_lead"] = round(clamp(cur["cal_lead"] * damp(r_body, hi=1.6), 0.4, 3.0), 3)
    # --- brightness: move the lead low-pass toward centroid parity ------------------------
    # GENTLE + dead-banded: the lead is only ONE source of the highs (hats/drums live there
    # too), and over-darkening tanks the brightness metric hard (a 28% lpf cut dropped it
    # 97%->67% in testing). Drive off the centroid ratio, react only outside +-8%, small step.
    if r_cent > 1.08:        # rendered too DARK -> open the cutoff a little
        lpf_corr = clamp(r_cent ** 0.4, 1.0, 1.25)
        nxt["lead_lpf"] = int(clamp(cur["lead_lpf"] * lpf_corr, 3500, 9000))
    elif r_cent < 0.92:      # rendered too BRIGHT -> shrink the cutoff a little
        lpf_corr = clamp(r_cent ** 0.4, 0.8, 1.0)
        nxt["lead_lpf"] = int(clamp(cur["lead_lpf"] * lpf_corr, 3500, 9000))
    else:
        nxt["lead_lpf"] = int(cur["lead_lpf"])
    # --- hats: the OTHER brightness source -------------------------------------------------
    # When the mix is still too dark (centroid low) but the lead low-pass is already wide open,
    # the missing brightness lives in the DRUMS — the original's hats/cymbals carry most of the
    # high band, and the extracted drum kit under-represents them. A steady TR808 hh*8 layer
    # injects that high-frequency air, lifting BOTH the drum stem's high bands and the overall
    # centroid. Only engage once lead-lpf can't help (>=7000) so the two brightness levers don't
    # fight. Step gently — hats read loud perceptually.
    LEAD_LPF_WIDE = 7000
    if r_cent > 1.15 and nxt["lead_lpf"] >= LEAD_LPF_WIDE:
        bump = clamp((r_cent - 1.0) * 0.35, 0.04, 0.18)   # darker mix -> bigger hat bump
        nxt["hat_gain"] = round(clamp(cur["hat_gain"] + bump, 0.0, 0.7), 3)
    elif r_cent < 0.9:                                     # overshot bright -> back hats off
        nxt["hat_gain"] = round(clamp(cur["hat_gain"] - 0.06, 0.0, 0.7), 3)
    else:
        nxt["hat_gain"] = round(cur["hat_gain"], 3)

    # --- master gain: gentle nudge toward energy parity, capped to keep the recorder's -----
    # 0 dBFS limiter out of it (clipping destroys the band balance we just fixed). -----------
    rms_ratio = comp.get("raw_rms_ratio", 1.0)
    if rms_ratio > 0:
        m_corr = clamp((1.0 / rms_ratio) ** 0.2, 0.92, 1.12)
        nxt["master_gain"] = round(clamp(cur["master_gain"] * m_corr, 0.4, 0.78), 3)

    # -------- report ----------------------------------------------------------------------
    def pct(x):
        return f"{x*100:5.1f}%"

    print("== measured band balance (orig vs rend) ==", file=sys.stderr)
    for k in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
        o, r = ob.get(k, 0), rb.get(k, 0)
        flag = "  <-- low" if r < o * 0.8 else ("  <-- high" if r > o * 1.25 else "")
        print(f"   {k:9s} {pct(o)} vs {pct(r)}{flag}", file=sys.stderr)
    print(f"   overall {comp.get('overall_similarity',0)*100:.1f}%  "
          f"energy {comp.get('energy_similarity',0)*100:.0f}%  "
          f"rms_ratio {rms_ratio:.2f}", file=sys.stderr)
    print(f"   centroid orig {cent_o:.0f} vs rend {cent_r:.0f} "
          f"(bright {comp.get('brightness_similarity',0)*100:.0f}%)", file=sys.stderr)
    print("== knob deltas ==", file=sys.stderr)
    for k in ["sub_gain", "bass_mult", "cal_lead", "lead_lpf", "hat_gain", "master_gain"]:
        a, b = cur[k], nxt[k]
        arrow = "->" if a != b else "=="
        print(f"   {k:12s} {a} {arrow} {b}", file=sys.stderr)

    cli = (f"--bass-mult {nxt['bass_mult']} --sub-gain {nxt['sub_gain']} "
           f"--cal-lead {nxt['cal_lead']} --lead-lpf {nxt['lead_lpf']} "
           f"--hat-gain {nxt['hat_gain']} --master-gain {nxt['master_gain']}")
    print(cli)   # stdout: the CLI fragment, easy to capture in a wrapper

    if args.out:
        args.out.write_text(json.dumps(nxt, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
