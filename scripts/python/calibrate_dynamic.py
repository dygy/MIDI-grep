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
  sub-gain   <- sub_bass band   (raise to fill 20-60 Hz, lower if booming; no absolute cap - each step is the
                                 damped correction that closes the observed band ratio)
  sub-up f   <- sub_bass:bass split (fraction 0..1 of the sine layer's energy played an octave above the bass note:
                                 moves share between sub_bass and bass without changing the layer's total; set from the
                                 measured split imbalance by sub_split_step(), its own step: gains hold while it moves)
  bass-mult  <- bass band       (60-250 Hz body of the pitched bass sample)
  cal-lead   <- low_mid+mid     (lead/melodic body; the usual deficit)
  lead-lpf   <- high+high_mid   (excess brightness -> lower the cutoff)
  bass-lpf   <- low_mid/bass share (synth mode: a sawtooth bass puts harmonics 2-5 in
                                 250-500 Hz = low_mid while the bass band drains. Rendered
                                 low_mid-per-bass too high -> lower the bass cutoff, too low ->
                                 raise it. Bounded by the material: floor = 2x the highest bass
                                 fundamental, ceiling = the analysis-derived cutoff)
  master-gain<- overall RMS      (gentle, capped for the loop; the FINAL level is set by mix_dynamics.py after BEST)

Usage:
  python calibrate_dynamic.py --comparison <comparison.json> \
      [--bass-mult 0.62 --sub-gain 0.5 --lead-lpf 6500 --master-gain 0.6 --cal-lead 1.0 \
       --bass-lpf <effective Hz> --bass-lpf-max <analysis Hz> \
       --bass-note-max-hz <Hz>] [--stem-comparison <stem_comparison.json>] \
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_audio import FREQ_BANDS  # noqa: E402  (single source for the band edges)


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


# One step of a bass layer measured ALONE (no kick in the number): the amplitude correction is the exact sqrt of the
# power ratio; the bounds only guard against a bad estimate (a 20 dB gap closes in two steps, not one).
VOICE_STEP_LO, VOICE_STEP_HI = 0.5, 2.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sub_split_step(ob: dict, rb: dict, f: float, f_max: float = 1.0, parent: dict | None = None,
                   eps: float = 1e-4) -> float | None:
    """Next octave-up fraction ``f`` of the sine layer from the measured sub_bass:bass SPLIT, or None (no move).

    The layer is two sines from the same notes, energy (1-f) at the note (feeds sub_bass) and f an octave up (feeds
    bass), so f moves energy between the two bands and leaves the layer's total alone. With E = the layer's total
    share of the mix::

        render sub_bass = S_other + E (1-f)        render bass = B_other + E f

    Target: the render splits the sub_bass+bass pair the way the original does (the pair's total is the gain levers'
    business): sub_bass should hold ``T So/(So+Bo)`` with T = Sr+Br. The share to move from sub_bass to bass is
    ``D = Sr - T So/(So+Bo)`` (D < 0: back), and the fraction of the layer that carries it is ``D / E``.

    E is not measured per layer (one render, mix shares only), so:
      * with a parent render whose only difference was f (same sub-gain and bass-mult: ``parent`` = its band shares
        and knobs), E is MEASURED: (Sr_parent - Sr)/(f - f_parent) and (Br - Br_parent)/(f - f_parent) both equal E, and
        the mean is used (the secant step);
      * otherwise E is bounded above by the donor band's own share: the layer's energy in the donor band cannot
        exceed that band's share, E (1-f) <= Sr (moving up) or E f <= Br (moving back). An upper bound on E gives a step
        that undershoots the target and never passes it; the next step then has a measurement. The measured E is also
        capped by that bound.
    No tuned constant. A split inside ``SHARE_DEAD_BAND`` (the share-ratio wobble demucs bleed alone causes) does not
    move f, and f stays within [0, f_max] (f_max = 0 when the generator refused the up sine).
    """
    so, bo = ob.get("sub_bass", 0.0), ob.get("bass", 0.0)
    sr, br = rb.get("sub_bass", 0.0), rb.get("bass", 0.0)
    split = ((so + eps) / (sr + eps)) / ((bo + eps) / (br + eps))      # orig/rend sub per bass; 1 = the same split
    if 1 / SHARE_DEAD_BAND <= split <= SHARE_DEAD_BAND:
        return None
    pair, o_pair = sr + br, so + bo
    if pair <= 0 or o_pair <= 0:
        return None
    move = sr - pair * so / o_pair                                      # share to carry sub_bass -> bass
    if move > 0 and f < 1.0:
        e_max = sr / (1.0 - f)
    elif move < 0 and f > 0.0:
        e_max = br / f
    else:
        return None                                                     # nothing left to move that way
    e = e_max
    if parent is not None and abs(f - parent["f"]) > 1e-6:
        df = f - parent["f"]
        e_meas = 0.5 * ((parent["rb"].get("sub_bass", 0.0) - sr) / df + (br - parent["rb"].get("bass", 0.0)) / df)
        if math.isfinite(e_meas) and e_meas > 0:
            e = min(e_meas, e_max)
    if e <= 0:
        return None
    return clamp(f + move / e, 0.0, max(0.0, min(1.0, f_max)))


def fmt_fraction(f: float) -> str:
    """0 / 1 as the integers the binary lever printed (old --start strings and tsv rows stay byte-identical)."""
    f = round(float(f), 3)
    return str(int(f)) if f in (0.0, 1.0) else f"{f:.3f}".rstrip("0")


def measured_exponent(r_now: float, r_parent: float | None, f: float, sign: float) -> float:
    """Elasticity of a share ratio to a cutoff move, MEASURED from the step that produced this render.

    ``r_*`` = orig/rend share ratio now / at the render this one was derived from, ``f`` = cutoff_now /
    cutoff_parent, ``sign`` = +1 when raising the cutoff raises the ratio (lead high-pass), -1 when lowering
    it does (bass low-pass). The step rule below assumes exponent 1 (a cutoff move ``f`` moves the ratio by
    ``f**1``); VAGABUNDO synth cal1->cal2 measured 2.7 (hpf 300->428 Hz moved low_mid-per-mid 0.49 -> 1.27 of the
    original's), so the next step was 2.7x too long and overshot. The result is floored at 1 so a measured
    exponent can only SHORTEN a step relative to the unmeasured rule, never lengthen it.
    """
    if r_parent is None or r_parent <= 0 or r_now <= 0 or f <= 0 or abs(math.log(f)) < 0.05:
        return 1.0
    e = sign * math.log(r_now / r_parent) / math.log(f)
    return e if math.isfinite(e) and e > 1.0 else 1.0


HIGH_MID_TOP_HZ = FREQ_BANDS["high_mid"][1]   # lead cutoff above this adds nothing to high_mid
LOW_MID_TOP_HZ = FREQ_BANDS["low_mid"][1]     # lead high-pass ceiling: above it we would eat the mid band
# DEAD BANDS (not tuned gains). SHARE_DEAD_BAND: a share ratio inside [1/1.33, 1.33] (+-33%) is left
# alone because the bass/low_mid/mid share ratios of a SOUND match still wobble ~+-25% from demucs
# bleed alone; checked against the stored sample-instrument comparisons (VAGABUNDO v002 1.23 / v004
# 1.10, Regime v026 1.09 for low_mid-per-bass; v002 1.36 / v004 1.31 for low_mid-per-mid) where the
# synth-only levers must stay silent anyway. HAT_HI_DEAD_BAND: the drum-hat path needs the render's
# high+high_mid share to be under HALF the original's (ratio > 2.0): VAGABUNDO synth v003 measured
# 4.5, the sample-instrument fixtures 0.6-0.9, so 2.0 separates "missing source" from "level wobble".
SHARE_DEAD_BAND = 1.33
HAT_HI_DEAD_BAND = 2.0


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
                         prev: dict | None, lead_no_raise: bool = False) -> dict | None:
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
    if lead_no_raise:
        # the lead-hpf lever moved this step: the lead is being re-PLACED in the spectrum, so its
        # gain must not also climb (a louder lead re-inflates low_mid). Cut-only for this step.
        wc["lead"] = [min(v, 1.0) for v in wc["lead"]]
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
    ap.add_argument("--sub-up", type=float, default=0.0,
                    help="EFFECTIVE fraction (0..1) of the sub-sine layer's energy an octave above the written bass note, "
                         "of the last render (generator summary 'sub_up'; the old 0/1 values read as f = 0 / 1)")
    ap.add_argument("--sub-up-max", type=float, default=1.0,
                    help="the most the generator can play up (summary 'sub_up_max'; 0 = it refused the up sine)")
    ap.add_argument("--parent-knobs", default="",
                    help="the knobs 'BM SG CL LLPF HG MG BL LH SU' of the render this one was stepped from (with "
                         "--parent-comparison): when only the octave fraction differs, the split step is a measured secant")
    ap.add_argument("--cal-lead", type=float, default=1.0)
    ap.add_argument("--lead-lpf", type=int, default=5000)
    ap.add_argument("--hat-gain", type=float, default=0.0)
    ap.add_argument("--master-gain", type=float, default=0.6)
    ap.add_argument("--bass-lpf", type=int, default=0,
                    help="EFFECTIVE bass low-pass (Hz) of the last render (generator summary "
                         "'bass_lpf'); 0 = unknown -> the bass-lpf lever stays off")
    ap.add_argument("--bass-lpf-max", type=int, default=0,
                    help="ceiling for the bass-lpf lever: the analysis-derived cutoff (summary "
                         "'bass_lpf_analysis'); 0 = never raise above the current value")
    ap.add_argument("--mode", choices=["sample-instrument", "synth"], default="sample-instrument",
                    help="generation mode of the last render. The spectral-placement levers "
                         "(bass-lpf, lead-hpf) are synth-only: a sampled instrument already has the "
                         "original's spectrum, so moving its filters would be a silent behaviour change")
    ap.add_argument("--lead-hpf", type=int, default=0,
                    help="EFFECTIVE lead high-pass (Hz) of the last render (generator summary "
                         "'lead_hpf'); 0 = unknown -> the lead-hpf lever stays off")
    ap.add_argument("--lead-note-min-hz", type=float, default=0.0,
                    help="lowest lead fundamental (summary 'lead_note_min_hz'); floor of the lead-hpf "
                         "lever. 0 = unknown -> lever stays off")
    ap.add_argument("--parent-comparison", type=Path, default=None,
                    help="comparison.json of the render these knobs were derived from; with --parent-lead-hpf / "
                         "--parent-bass-lpf it lets the placement levers measure their own response")
    ap.add_argument("--parent-lead-hpf", type=int, default=0)
    ap.add_argument("--parent-bass-lpf", type=int, default=0)
    ap.add_argument("--bass-note-max-hz", type=float, default=0.0,
                    help="highest bass fundamental (summary 'bass_note_max_hz'); sets the lever's "
                         "floor. 0 = unknown -> lever stays off")
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
    ap.add_argument("--bass-solo", type=Path, default=None,
                    help="bass_solo.py measure JSON for the LAST render: the bass voice rendered alone vs the original bass "
                         "stem, as a share of its mix, per band. When present the bass layers (sub-gain / bass-mult) follow "
                         "THAT ratio instead of the whole-mix band shares, which the kick also fills")
    ap.add_argument("--melodic-solo", type=Path, default=None,
                    help="melodic_solo.py measure JSON for the LAST render: the LEAD+CHORDS voices rendered alone vs the "
                         "original melodic stem. When present cal-lead follows THAT absolute body level and lead-lpf THAT "
                         "centroid, instead of the whole-mix low_mid+mid share and centroid the bass/kick/hats also fill")
    ap.add_argument("--responses", type=Path, default=None,
                    help="JSON {knob: damping exponent} LEARNED from recorded (knob change -> band share change) pairs "
                         "(calibration_memory.py responses): replaces the fixed 0.5 (sqrt) exponent of sub_gain / "
                         "bass_mult / cal_lead where present. Absent file or knob = the sqrt damping, unchanged")
    ap.add_argument("--bpm", type=float, default=None,
                    help="generator BPM for bar mapping (default: measured original tempo)")
    ap.add_argument("--bars", type=int, default=None,
                    help="bar count of the generated arrangement (default: ceil(analysed span / bar))")
    args = ap.parse_args()

    cur = {
        "bass_mult": args.bass_mult,
        "sub_gain": args.sub_gain,
        "sub_up": min(1.0, max(0.0, float(args.sub_up))),
        "cal_lead": args.cal_lead,
        "lead_lpf": float(args.lead_lpf),
        "hat_gain": args.hat_gain,
        "master_gain": args.master_gain,
        "bass_lpf": float(args.bass_lpf),
        "lead_hpf": float(args.lead_hpf),
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

    learned: dict[str, float] = {}
    if args.responses is not None:
        try:
            learned = {k: float(v) for k, v in json.loads(args.responses.read_text()).items()}
        except (OSError, ValueError, AttributeError) as e:
            print(f"WARNING: --responses {args.responses} unreadable ({e}); using the fixed sqrt damping", file=sys.stderr)

    def strength(knob: str) -> float:
        return learned.get(knob, 0.5)

    def ratio(band_keys: list[str]) -> float:
        o = sum(ob.get(k, 0.0) for k in band_keys)
        r = sum(rb.get(k, 0.0) for k in band_keys)
        return (o + eps) / (r + eps)

    r_sub = ratio(["sub_bass"])
    r_bass = ratio(["bass"])
    # The bass voice measured ALONE (bass_solo.py). The kick fills sub_bass/bass too, so the mix shares cannot say
    # whether the BASS is right: VAGABUNDO's mix sub read 34.7 % vs 34.1 % while the bass voice sat 8.5 dB under its
    # stem, and the loop lowered it. It sets the SAMPLE layer (bass-mult), the one layer that carries the whole voice,
    # and being one linear voice (power ratio -> amplitude = sqrt) it takes the full measured step, not the damped one.
    # The sine layer (sub-gain) stays on the mix shares: both layers feed the 60-250 Hz band, so driving both from the
    # same band overshoots (measured on VAGABUNDO: bass-band ratio 3.15 -> 0.90 in one step with both raised).
    voice: dict[str, float] = {}
    if args.bass_solo is not None:
        try:
            solo_ratio = json.loads(args.bass_solo.read_text()).get("ratio") or {}
            voice = {b: float(v) for b, v in solo_ratio.items() if b in ("sub_bass", "bass") and v}
        except (OSError, ValueError, AttributeError, TypeError) as e:
            print(f"WARNING: --bass-solo {args.bass_solo} unreadable ({e}); bass layers follow the mix band shares",
                  file=sys.stderr)

    def sample_layer_step(band: str, knob: str) -> float:
        """Multiplicative correction of the sample layer, driven by the voice's own level in `band` when measured."""
        if band in voice:
            return damp(voice[band], strength=0.5, lo=VOICE_STEP_LO, hi=VOICE_STEP_HI)
        return damp(r_sub if band == "sub_bass" else r_bass, strength=strength(knob))
    r_body = ratio(["low_mid", "mid"])       # lead/melodic body
    # The melodic voices measured ALONE (melodic_solo.py): the same body bands, but as an absolute level of the voices
    # against the melodic stem, not a share of a mix the bass, kick and hats also fill. One linear voice group, so the
    # full measured step (power ratio -> amplitude = sqrt) with the bass-solo bounds; brightness is the voices' own
    # centroid ratio (> 1 = too dark), which replaces the mix centroid for the lead low-pass only.
    r_body_voice, r_cent_voice = None, None
    if args.melodic_solo is not None:
        try:
            msolo = json.loads(args.melodic_solo.read_text())
            lo_o, lo_r = (sum(float(msolo[k][b]) for b in ("low_mid", "mid")) for k in ("level_orig", "level_render"))
            if all(msolo["ratio"].get(b) for b in ("low_mid", "mid")) and lo_r > 0:
                r_body_voice = lo_o / lo_r
            c_o, c_r = msolo["centroid"]["orig"], msolo["centroid"]["render"]
            if c_o and c_r:
                r_cent_voice = float(c_o) / float(c_r)
        except (OSError, ValueError, AttributeError, TypeError, KeyError) as e:
            print(f"WARNING: --melodic-solo {args.melodic_solo} unreadable ({e}); lead level/brightness follow the mix",
                  file=sys.stderr)
    # Brightness lever uses the spectral CENTROID, not the high bands: in bass-heavy genres
    # the high/high_mid bands sit at ~1% magnitude where demucs bleed + noise swamp the
    # signal, so their ratio is unreliable (it kept lead-lpf pinned while the centroid said
    # the mix was clearly too dull). The centroid is the stable, perceptual brightness metric.
    cent_o = data["original"]["spectral"]["centroid_mean"]
    cent_r = data["rendered"]["spectral"]["centroid_mean"]
    r_cent = (cent_o + eps) / (cent_r + eps)   # >1 => rendered too dark, open the cutoff

    def share_ratio(num: str, den: str) -> float:
        """orig/rend of (band `num` per band `den`): >1 => render has too LITTLE `num` for its `den`."""
        return (((ob.get(num, 0.0) + eps) / (ob.get(den, 0.0) + eps)) /
                ((rb.get(num, 0.0) + eps) / (rb.get(den, 0.0) + eps)))

    nxt = dict(cur)
    # --- sub_bass: fill or tame the 20-60 Hz sine layer -----------------------------------
    # No absolute cap: a fixed 0.2-1.6 range saturated at 1.6 on VAGABUNDO (sub_bass still short) and left the
    # lever blind. The step is the damped correction that closes the OBSERVED ratio (sqrt = amplitude from an
    # energy share ratio, further damped by damp() and limited to damp()'s per-step bounds); positive by construction.
    #
    # Octave: when the sub_bass:bass SPLIT is off, more gain cannot help: moving sine energy an octave up shifts it
    # from sub_bass into bass (or back). The dead band is the split ratio's (SHARE_DEAD_BAND): Regime CLT sample (render
    # sub 41.1% vs the original's 34.4%, bass 31.1% vs 36.4%, split ratio 0.72) gained +0.042 overall (0.7510 -> 0.7933)
    # from the former full flip, and it is outside the band; the stored VAGABUNDO renders (0.81-0.88) are inside it.
    #
    # f is CONTINUOUS (sub_split_step: the sine layer is two sines, energy 1-f at the note and f an octave up). It is its
    # own step - the gains hold while it moves, as the binary flip held them - and is scored by the loop like any other
    # (revert / tie rules). The binary lever moved ~30 points of share in one go and overshot (Sem Aliança: sub 45.2 ->
    # 64.4 -> reverted, locked; Tatuagem / BibiBabydoll kept the flip and sat 15 / 9 points under), so there is no lock
    # any more: a split inside the dead band simply does not move f, and a step is the share the split is off by
    # divided by the share the layer can carry (sub_split_step).
    parent_bands = None
    if args.parent_comparison and args.parent_comparison.exists():
        try:
            pd = json.loads(args.parent_comparison.read_text())
            parent_bands = (pd["original"]["bands"], pd["rendered"]["bands"])
        except (OSError, ValueError, KeyError):
            parent_bands = None

    f_cur = float(cur["sub_up"])
    parent_split = None
    if parent_bands is not None and args.parent_knobs.split():
        try:
            pk = [float(x) for x in args.parent_knobs.split()]
            # a measurement only when the parent differed in f alone: the level levers were held on that step
            if len(pk) >= 9 and abs(pk[0] - cur["bass_mult"]) < 5e-4 and abs(pk[1] - cur["sub_gain"]) < 5e-4:
                parent_split = {"f": pk[8], "rb": parent_bands[1]}
        except ValueError:
            parent_split = None
    f_next = sub_split_step(ob, rb, f_cur, args.sub_up_max, parent_split)
    splitting = f_next is not None and round(f_next, 3) != round(f_cur, 3)
    nxt["sub_up"] = round(f_next, 3) if splitting else f_cur
    # WHICH BAND EACH LAYER FEEDS depends on where the sine's energy sits. Mostly at the note (f < 0.5): it is inside
    # sub_bass (20-60 Hz) and the main layer carries the bass band. Mostly an octave up: the sine (note +12, low-pass x2)
    # lands in the bass band and the sub_bass share is left to the main layer. So the two gains swap their driving
    # ratios: sub-gain follows `bass` and bass-mult follows `sub_bass`. (At f = 0 / 1 this is exactly the binary
    # lever's rule.) Driving sub-gain by sub_bass with the octave up raises a layer that no longer feeds that band:
    # Regime CLT synth cal2->cal3 raised it 0.70 -> 0.95 while render bass was already 55% vs the original's 41% ->
    # bass 79%, overall 0.7446 -> 0.6566.
    sine_up = f_cur >= 0.5
    if splitting:
        nxt["sub_gain"] = round(cur["sub_gain"], 3)
    else:
        nxt["sub_gain"] = round(cur["sub_gain"] * damp(r_bass if sine_up else r_sub, strength=strength("sub_gain")), 3)
    synth = args.mode == "synth"
    r_lm_bass = float("nan")
    cur_blpf, cur_lhpf = int(cur["bass_lpf"]), int(cur["lead_hpf"])
    sc_path = args.stem_comparison or args.comparison.with_name("stem_comparison.json")
    sc_stems = {}
    if sc_path.exists():
        try:
            sc_stems = json.loads(sc_path.read_text()).get("stems") or {}
        except (OSError, ValueError):
            sc_stems = {}
    nxt["bass_lpf"], nxt["lead_hpf"] = cur_blpf, cur_lhpf

    def parent_share_ratio(num: str, den: str) -> float | None:
        if parent_bands is None:
            return None
        po, pr = parent_bands
        return (((po.get(num, 0.0) + eps) / (po.get(den, 0.0) + eps)) /
                ((pr.get(num, 0.0) + eps) / (pr.get(den, 0.0) + eps)))

    # low_mid-per-mid placement ratio, needed by the lead-hpf lever
    r_lm_mid = share_ratio("low_mid", "mid") if synth else float("nan")

    # --- lead high-pass: place the lead's energy in mid instead of low_mid (synth only) -----
    # Measured on VAGABUNDO synth cal1-3: lowering the bass cutoff 603->295 Hz moved low_mid only
    # 36->33% while it stayed 4x the original's (7.5%) and mid sat at half; the bass share is low, yet render low_mid is 51-63% => the saw LEAD's 130-500 Hz
    # fundamentals (median 330 Hz) carry it. A gain lever cannot move spectral placement, a high-pass
    # can. Driver: low_mid-per-mid share ratio (orig/rend); render too heavy in low_mid -> raise hpf.
    # Bounds from the material/bands: floor = lowest lead fundamental (a cutoff below it filters
    # nothing); ceiling = top of the low_mid band (above it the hpf starts eating the mid band we
    # are trying to fill) - and never below an already-higher current value.
    lead_hpf_moved = False
    if synth and cur_lhpf > 0 and args.lead_note_min_hz > 0:
        floor_h = int(math.ceil(args.lead_note_min_hz))
        ceil_h = max(cur_lhpf, LOW_MID_TOP_HZ)
        if floor_h <= ceil_h and not (1 / SHARE_DEAD_BAND <= r_lm_mid <= SHARE_DEAD_BAND):
            e_h = measured_exponent(r_lm_mid, parent_share_ratio("low_mid", "mid"),
                                    cur_lhpf / args.parent_lead_hpf if args.parent_lead_hpf > 0 else 1.0, +1.0)
            nxt["lead_hpf"] = int(clamp(cur_lhpf / damp(r_lm_mid, lo=0.7, hi=1.3) ** (1.0 / e_h), floor_h, ceil_h))
            lead_hpf_moved = nxt["lead_hpf"] != cur_lhpf

    # --- bass low-pass: tame the saw's upper harmonics (synth only) -------------------------
    # low_mid RELATIVE TO bass in the original vs the render: a sawtooth on a 40-130 Hz fundamental
    # has strong harmonics 2-5 at 250-500 Hz that inflate low_mid and (via the shares) shrink bass;
    # bass-mult cannot fix that (it raises the harmonics too). Same symptom as the lead lever, one
    # lever per step: the lead lever has priority because on VAGABUNDO its sibling saturated fast
    # (603->422 Hz: low_mid -4 points; 422->295: +0.6).
    bass_lpf_moved = False
    if synth and cur_blpf > 0 and args.bass_note_max_hz > 0:
        r_lm_bass = share_ratio("low_mid", "bass")
        floor = int(math.ceil(2.0 * args.bass_note_max_hz))        # keep the top note's fundamental + 2nd harmonic
        ceil_ = max(cur_blpf, args.bass_lpf_max)                   # never open past the analysis value
        if (not lead_hpf_moved and floor <= ceil_
                and not (1 / SHARE_DEAD_BAND <= r_lm_bass <= SHARE_DEAD_BAND)):
            e_b = measured_exponent(r_lm_bass, parent_share_ratio("low_mid", "bass"),
                                    cur_blpf / args.parent_bass_lpf if args.parent_bass_lpf > 0 else 1.0, -1.0)
            nxt["bass_lpf"] = int(clamp(cur_blpf * damp(r_lm_bass, lo=0.7, hi=1.3) ** (1.0 / e_b), floor, ceil_))
            bass_lpf_moved = nxt["bass_lpf"] != cur_blpf

    # --- bass (60-250 Hz): body of the pitched bass sample --------------------------------
    # One lever per step: while the cutoff moves, bass-mult may fall but not rise (a rise would
    # re-inflate the very harmonics the cutoff is removing; cal2->cal3 did exactly that: +3%).
    bass_corr = sample_layer_step("sub_bass" if sine_up else "bass", "bass_mult")     # see the layer/band assignment at the sub-gain step
    if splitting:
        bass_corr = 1.0       # the split move IS this step's bass/sub rebalance: a level move too would count the same share twice
    if bass_lpf_moved:
        bass_corr = min(bass_corr, 1.0)
    # No absolute ceiling (a fixed 0.2-1.3 was pegged at 1.3 on every synth track, the references included, so the
    # bass band had no lever up - the same blindness sub-gain lost its 0.2-1.6 range for). The step is bounded like
    # sub-gain's: the voice's own measured step (VOICE_STEP_LO..HI, power ratio -> amplitude) or damp()'s per-step
    # bounds, and positive by construction.
    nxt["bass_mult"] = round(cur["bass_mult"] * bass_corr, 3)
    # --- lead/melodic body: raise gain when low_mid+mid is deficient -----------------------
    lead_corr = (damp(r_body_voice, strength=0.5, lo=VOICE_STEP_LO, hi=VOICE_STEP_HI) if r_body_voice is not None
                 else damp(r_body, strength=strength("cal_lead"), hi=1.6))
    nxt["cal_lead"] = round(clamp(cur["cal_lead"] * lead_corr, 0.4, 3.0), 3)
    # --- brightness: move the lead low-pass toward centroid parity ------------------------
    # GENTLE + dead-banded: the lead is only ONE source of the highs (hats/drums live there
    # too), and over-darkening tanks the brightness metric hard (a 28% lpf cut dropped it
    # 97%->67% in testing). Drive off the centroid ratio, react only outside +-8%, small step.
    # The drums-as-source hat path (below) owns the brightness lever on a step where it engages:
    # one brightness lever per step, as the >=7000 rule intended.
    r_hi = ratio(["high_mid", "high"])
    # The TR808 hh layer adds energy ABOVE 4 kHz (the `high` band) only - it cannot fill high_mid.
    # So its engagement/back-off is governed by the measured `high` share: the drums stem's own
    # (stem_comparison.json, VAGABUNDO v006: 21% vs 0.7%) when present, else the full mix's.
    # Render `high` more than HAT_HI_DEAD_BAND x the original's = overshoot -> shrink the layer
    # multiplicatively; render `high` already above SHARE_DEAD_BAND x the original's = hats
    # cannot be the missing source, so the drums path stays shut.
    r_high = ratio(["high"])
    # Two evidence levels. (1) The drums STEM's own `high` (stem_comparison.json): isolates the hats, so it
    # may act freely. The auto-calibrate loop does not produce that file, so (2) there the only evidence is
    # the FULL-MIX `high` (which also contains a lead opened toward 9 kHz ): a proxy, so it may
    # shrink the hat layer only on a step where lead-lpf did NOT move (one brightness lever per step -
    # otherwise the two would fight over the same band).
    try:
        d_o, d_r = (sc_stems["drums"][k]["bands"]["high"] for k in ("original", "rendered"))
        r_dhigh, from_stem = (d_o + eps) / (d_r + eps), True
    except (KeyError, TypeError):
        r_dhigh, from_stem = r_high, False
    over_raw = r_dhigh < 1 / HAT_HI_DEAD_BAND
    hat_via_drums = (r_cent > 1.15 and cur["lead_lpf"] >= HIGH_MID_TOP_HZ and r_hi > HAT_HI_DEAD_BAND
                     and r_high >= 1 / SHARE_DEAD_BAND and not over_raw)
    r_lead_cent = r_cent if r_cent_voice is None else r_cent_voice   # the lead cutoff follows the lead's own brightness
    if hat_via_drums:
        nxt["lead_lpf"] = int(cur["lead_lpf"])
    elif r_lead_cent > 1.08:        # rendered too DARK -> open the cutoff a little
        lpf_corr = clamp(r_lead_cent ** 0.4, 1.0, 1.25)
        nxt["lead_lpf"] = int(clamp(cur["lead_lpf"] * lpf_corr, 3500, 9000))
    elif r_lead_cent < 0.92:      # rendered too BRIGHT -> shrink the cutoff a little
        lpf_corr = clamp(r_lead_cent ** 0.4, 0.8, 1.0)
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
    # Data-driven second engagement path (synth mode, bug #11): the lead can't be the high_mid
    # source once its cutoff already sits above the high_mid band's top edge (compare_audio:
    # 2-4 kHz) - opening it further adds nothing there. If the render still holds under half the
    # original's high+high_mid share, the drums are the only source left, so engage the hats
    # without waiting for lead-lpf to crawl up to 7000 (it moves ~14%/step: 3+ wasted renders).
    LEAD_LPF_WIDE = 7000
    # the full-mix proxy may only act on a step where lead-lpf stood still (see the two evidence levels above)
    lead_lpf_moved = int(nxt["lead_lpf"]) != int(cur["lead_lpf"])
    hat_overshoot = over_raw and (from_stem or not lead_lpf_moved)
    if hat_overshoot and cur["hat_gain"] > 0:
        nxt["hat_gain"] = round(cur["hat_gain"] * damp(r_dhigh, lo=0.5, hi=1.0), 3)
    elif r_cent > 1.15 and (nxt["lead_lpf"] >= LEAD_LPF_WIDE or hat_via_drums) and not hat_overshoot:
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
    print(f"   low_mid/bass share ratio {r_lm_bass:.2f}, low_mid/mid {r_lm_mid:.2f} "
          f"(orig/rend; nan = lever off)", file=sys.stderr)
    for k in ["sub_gain", "sub_up", "bass_mult", "cal_lead", "lead_lpf", "hat_gain", "master_gain",
              "bass_lpf", "lead_hpf"]:
        a, b = cur[k], nxt[k]
        arrow = "->" if a != b else "=="
        print(f"   {k:12s} {a} {arrow} {b}", file=sys.stderr)

    cli = (f"--bass-mult {nxt['bass_mult']} --sub-gain {nxt['sub_gain']} --sub-up {fmt_fraction(nxt['sub_up'])} "
           f"--cal-lead {nxt['cal_lead']} --lead-lpf {nxt['lead_lpf']} "
           f"--hat-gain {nxt['hat_gain']} --master-gain {nxt['master_gain']}")
    if cur_blpf > 0 and nxt["bass_lpf"] != cur_blpf:
        cli += f" --bass-lpf {nxt['bass_lpf']}"
    if cur_lhpf > 0 and nxt["lead_hpf"] != cur_lhpf:
        cli += f" --lead-hpf {nxt['lead_hpf']}"
    print(cli)   # stdout: the CLI fragment, easy to capture in a wrapper

    if args.out:
        args.out.write_text(json.dumps(nxt, indent=2) + "\n")

    if args.env_correction_out:
        prev = None
        if args.env_correction_in and args.env_correction_in.exists():
            prev = json.loads(args.env_correction_in.read_text())
        env = build_env_correction(data, args.comparison, args.bpm, args.bars, prev, lead_hpf_moved)
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
