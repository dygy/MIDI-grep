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
  cal-vocal  <- vocal stem RMS  (spec 003 Slice 3: the editable vocal voice is balanced by
                                 measurement. Primary source: ``stems.vocals`` in the sibling
                                 ``stem_comparison.json`` (original vs rendered rms_mean).
                                 compare_audio.py does not pair a vocals stem yet, so until it
                                 does the lever falls back to the full-mix ``high_mid`` band
                                 ratio — the band the vocal's presence region dominates once
                                 lead/hats are driven by their own levers — with a wide
                                 dead-band and a gentler step because it is a proxy.)
  master-gain<- overall RMS      (gentle; the recorder limiter caps absolute loudness)

Usage:
  python calibrate_dynamic.py --comparison <comparison.json> \
      [--bass-mult 0.62 --sub-gain 0.5 --lead-lpf 6500 --master-gain 0.6 --cal-lead 1.0 \
       --cal-vocal 1.0] [--stem-comparison <stem_comparison.json>] \
      [--state <prev_params.json>] [--out <next_params.json>]

Prints a human-readable diff and a ready-to-paste CLI fragment; writes the next param set
as JSON (consumed by the wrapper to drive the next ``generate_dynamic_strudel`` run).
"""
from __future__ import annotations

import argparse
import json
import math
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


ENV_VOICES = {          # voice -> band keys whose orig/rend share ratio drives it
    "bass": ["sub_bass", "bass"],
    "lead": ["mid", "high_mid"],
}
ENV_EPS = 1e-4
# Bound on the COMPOSED per-bar multiplier: corrections multiply across iterations, so the product
# needs a cap; a single step is already limited by damp(). Emitted in the JSON as "clamp" so the
# generator reads the bound from the data instead of duplicating it.
ENV_LO, ENV_HI = 0.5, 2.0


def window_corrections(windows: list[dict]) -> dict[str, list[float]]:
    """Per-window damped multipliers {bass, lead, master} from compare_audio section_windows."""
    out: dict[str, list[float]] = {"bass": [], "lead": [], "master": []}
    for w in windows:
        for voice, keys in ENV_VOICES.items():
            o = sum(w["orig_bands"].get(k, 0.0) for k in keys)
            r = sum(w["rend_bands"].get(k, 0.0) for k in keys)
            out[voice].append(damp((o + ENV_EPS) / (r + ENV_EPS)))
        out["master"].append(damp((w["orig_rms"] + ENV_EPS) / (w["rend_rms"] + ENV_EPS)))
    return out


def per_bar(centres: list[float], vals: list[float], bars: int, bar_s: float) -> list[float]:
    """Linear interpolation across window centres, held flat beyond the first/last centre."""
    res = []
    for b in range(bars):
        t = (b + 0.5) * bar_s
        if t <= centres[0]:
            res.append(vals[0])
        elif t >= centres[-1]:
            res.append(vals[-1])
        else:
            for i in range(1, len(centres)):
                if t <= centres[i]:
                    f = (t - centres[i - 1]) / (centres[i] - centres[i - 1])
                    res.append(vals[i - 1] + f * (vals[i] - vals[i - 1]))
                    break
    return res


def build_env_correction(data: dict, comparison_path: Path, bpm: float | None, bars: int | None,
                         prev: dict | None) -> dict | None:
    windows = (data.get("comparison") or {}).get("section_windows") or []
    if not windows:
        return None
    if bpm is None:
        bpm = float(data["original"]["rhythm"]["tempo"])   # measured fallback
    bar_s = 240.0 / bpm
    if bars is None:
        bars = int(math.ceil(max(w["t1"] for w in windows) / bar_s))
    centres = [(w["t0"] + w["t1"]) / 2.0 for w in windows]
    wc = window_corrections(windows)
    voices = {}
    for v, vals in wc.items():
        curve = per_bar(centres, vals, bars, bar_s)
        pv = (prev or {}).get("voices", {}).get(v, [])
        if pv and len(pv) != bars:
            # previous curve is on another bar grid: resample it (linear over its bar centres)
            # instead of silently discarding the accumulated correction.
            prev_bar_s = 240.0 / float(prev["bpm"]) if prev.get("bpm") else bar_s
            pv = per_bar([(i + 0.5) * prev_bar_s for i in range(len(pv))], pv, bars, bar_s)
            print(f"env-correction: previous curve has a different bar count "
                  f"({len(prev['voices'][v])} vs {bars}) for '{v}'; resampled onto the new grid",
                  file=sys.stderr)
        if pv:
            curve = [c * p for c, p in zip(curve, pv)]
        voices[v] = [round(clamp(c, ENV_LO, ENV_HI), 4) for c in curve]
    return {"window_s": float(windows[0]["t1"] - windows[0]["t0"]), "bpm": bpm, "bars": bars,
            "clamp": [ENV_LO, ENV_HI], "source": str(comparison_path), "voices": voices}


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
    ap.add_argument("--cal-vocal", type=float, default=1.0)
    ap.add_argument("--stem-comparison", type=Path, default=None,
                    help="per-stem comparison JSON (compare_audio.py --stems). Default: a "
                         "stem_comparison.json next to --comparison when present")
    ap.add_argument("--state", type=Path, default=None,
                    help="prev_params.json to read current knob values from (overrides the "
                         "per-knob flags above for any key it contains)")
    ap.add_argument("--out", type=Path, default=None, help="write next params as JSON here")
    ap.add_argument("--env-correction-out", type=Path, default=None,
                    help="write per-bar bass/lead/master multiplier curves (from the comparison's "
                         "section_windows) as JSON here")
    ap.add_argument("--env-correction-in", type=Path, default=None,
                    help="previous env-correction JSON; new curves are multiplied onto it")
    ap.add_argument("--bpm", type=float, default=None,
                    help="generator BPM for bar mapping (default: measured original tempo)")
    ap.add_argument("--bars", type=int, default=None,
                    help="bar count of the generated arrangement (default: ceil(analysed span / bar))")
    args = ap.parse_args()

    cur = {
        "bass_mult": args.bass_mult,
        "sub_gain": args.sub_gain,
        "cal_lead": args.cal_lead,
        "lead_lpf": float(args.lead_lpf),
        "hat_gain": args.hat_gain,
        "master_gain": args.master_gain,
        "cal_vocal": args.cal_vocal,
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

    # --- vocal: balance the editable vocal voice by measurement ----------------------------
    # Primary: the vocal stem's own RMS ratio from stem_comparison.json (demucs re-separation
    # of the render vs the original vocal stem). Fallback: full-mix high_mid band ratio —
    # a proxy, so dead-banded (+-15%) and damped harder (0.35) with a narrower step clamp.
    sc_path = args.stem_comparison or args.comparison.with_name("stem_comparison.json")
    voc_stem = None
    if sc_path.exists():
        try:
            voc_stem = (json.loads(sc_path.read_text()).get("stems") or {}).get("vocals")
        except (OSError, ValueError):
            voc_stem = None
    try:
        o_v = voc_stem["original"]["spectral"]["rms_mean"]
        r_v = voc_stem["rendered"]["spectral"]["rms_mean"]
        r_vocal = (o_v + eps) / (r_v + eps)
        vocal_src = f"vocal stem rms ({sc_path.name})"
        nxt["cal_vocal"] = round(clamp(cur["cal_vocal"] * damp(r_vocal, hi=1.6), 0.3, 3.0), 3)
    except (TypeError, KeyError):
        r_vocal = ratio(["high_mid"])
        vocal_src = "high_mid band proxy (no stems.vocals in stem_comparison.json)"
        if r_vocal > 1.15 or r_vocal < 0.85:
            nxt["cal_vocal"] = round(clamp(cur["cal_vocal"] * damp(r_vocal, strength=0.35, lo=0.85, hi=1.2),
                                           0.3, 3.0), 3)
        else:
            nxt["cal_vocal"] = round(cur["cal_vocal"], 3)

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
    print(f"   vocal ratio {r_vocal:.2f} from {vocal_src}", file=sys.stderr)
    print("== knob deltas ==", file=sys.stderr)
    for k in ["sub_gain", "bass_mult", "cal_lead", "lead_lpf", "hat_gain", "master_gain", "cal_vocal"]:
        a, b = cur[k], nxt[k]
        arrow = "->" if a != b else "=="
        print(f"   {k:12s} {a} {arrow} {b}", file=sys.stderr)

    cli = (f"--bass-mult {nxt['bass_mult']} --sub-gain {nxt['sub_gain']} "
           f"--cal-lead {nxt['cal_lead']} --lead-lpf {nxt['lead_lpf']} "
           f"--hat-gain {nxt['hat_gain']} --master-gain {nxt['master_gain']} "
           f"--cal-vocal {nxt['cal_vocal']}")
    print(cli)   # stdout: the CLI fragment, easy to capture in a wrapper

    if args.out:
        args.out.write_text(json.dumps(nxt, indent=2) + "\n")

    if args.env_correction_out:
        prev = None
        if args.env_correction_in and args.env_correction_in.exists():
            prev = json.loads(args.env_correction_in.read_text())
        env = build_env_correction(data, args.comparison, args.bpm, args.bars, prev)
        if env is None:
            print("env-correction: comparison has no section_windows (re-run compare_audio.py); "
                  "nothing written", file=sys.stderr)
        else:
            args.env_correction_out.write_text(json.dumps(env, indent=2) + "\n")
            print(f"env-correction: wrote {env['bars']} bars x {len(env['voices'])} voices "
                  f"-> {args.env_correction_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
