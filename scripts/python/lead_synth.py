#!/usr/bin/env python3
"""Synth-mode lead as FITTED synthesis parameters instead of a fixed palette preset.

The palette lead (``gm_lead_2_sawtooth``) is a soundfont sample: one timbre, no controls. Here the lead is a
plain Strudel oscillator (``supersaw`` / ``sawtooth`` / ``square`` / ``triangle``) whose controls are

    unison, detune          line width of the partials (supersaw only)
    attack, decay, sustain  amplitude ADSR
    lpf, lpenv, lpdecay     low-pass base cutoff, filter-envelope depth (octaves) and decay time (s)
    gain                    loudness multiplier on the lead's calibrated gain

``measure_lead`` derives every initial value and bound from the melodic stem (no typed sound constants);
``fit`` runs a Nelder-Mead search whose objective is supplied by the caller (the real metric on the rendered
MIX). The result is plain, editable Strudel controls on the lead line (``lead_chain``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Oscillators a measurement can select. Harmonic amplitude of partial k, relative to partial 1.
FAMILIES = {
    "sawtooth": lambda k: 1.0 / k,
    "square": lambda k: (1.0 / k) if k % 2 else 0.0,
    "triangle": lambda k: (1.0 / k ** 2) if k % 2 else 0.0,
}
N_HARM = 8
PARAM_NAMES = ("lpf", "lpenv", "lpdecay", "attack", "decay", "sustain", "detune", "gain")


# --------------------------------------------------------------------------------------------- measurement
def harmonic_profile(mag: np.ndarray, sr: int, n_fft: int, f0s: np.ndarray) -> np.ndarray:
    """Median amplitude of partials 1..N_HARM (relative to partial 1) over frames with a valid f0."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    rows = []
    for t, f0 in enumerate(f0s):
        if not np.isfinite(f0) or f0 <= 0 or f0 * N_HARM >= sr / 2 * 0.9:
            continue
        amps = []
        for k in range(1, N_HARM + 1):
            sel = np.abs(freqs - k * f0) <= f0 * 0.03
            amps.append(mag[sel, t].max() if sel.any() else 0.0)
        if amps[0] > 0:
            rows.append(np.asarray(amps) / amps[0])
    return np.median(np.asarray(rows), axis=0) if rows else np.zeros(N_HARM)


def pick_family(profile: np.ndarray) -> tuple[str, dict]:
    """Oscillator whose partial series is nearest (log-amplitude distance, partials 2..N) to the measured one."""
    eps = 1e-3
    dists = {}
    for name, fn in FAMILIES.items():
        ref = np.asarray([fn(k) for k in range(1, N_HARM + 1)])
        dists[name] = float(np.mean(np.abs(np.log(profile[1:] + eps) - np.log(ref[1:] + eps))))
    return min(dists, key=dists.get), dists


def measure_lead(melodic: Path, bpm: float, *, max_seconds: float = 60.0) -> dict:
    """Everything the parameter search starts from, measured on the melodic stem."""
    import librosa
    y, sr = librosa.load(str(melodic), sr=None, mono=True, duration=max_seconds)
    n_fft, hop = 4096, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    active = rms > np.median(rms[rms > 0]) if (rms > 0).any() else rms > 0
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    f0 = librosa.yin(y, fmin=80, fmax=1500, sr=sr, frame_length=n_fft, hop_length=hop)
    n = min(len(f0), S.shape[1])
    f0 = np.where(active[:n], f0[:n], np.nan)
    profile = harmonic_profile(S[:, :n], sr, n_fft, f0)
    family, fam_dist = pick_family(profile)

    # line width of partial 3 (cents, -6 dB) vs the analysis resolution: detune and whether to stack voices
    width_cents = _line_width_cents(y, sr, f0, hop)

    # per-onset envelope shape: attack, decay and sustain ratio; centroid at attack vs sustain -> filter envelope
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, backtrack=True)
    fr = hop / sr
    atk, dec, sus, c_ratio, c_dec = [], [], [], [], []
    for a, b in zip(onsets[:-1], onsets[1:]):
        if b - a < 4:
            continue
        seg = rms[a:b]
        pk = int(np.argmax(seg))
        if seg[pk] <= 0:
            continue
        e = seg / seg[pk]
        atk.append(max(pk, 1) * fr * 0.9)
        s_level = float(np.median(e[pk:])) if pk < len(e) else 1.0
        sus.append(min(max(s_level, 0.05), 1.0))
        below = np.where(e[pk:] <= s_level + (1 - s_level) / math.e)[0]
        dec.append((below[0] if len(below) else len(e) - pk) * fr)
        c = cent[a:b]
        if len(c) > pk + 2 and c[-1] > 0:
            c_ratio.append(math.log2(max(c[pk], 1.0) / max(np.median(c[pk:]), 1.0)))
            cb = np.where(c[pk:] <= np.median(c[pk:]) + (c[pk] - np.median(c[pk:])) / math.e)[0]
            c_dec.append((cb[0] if len(cb) else len(c) - pk) * fr)
    med = lambda v, d: float(np.median(v)) if len(v) else d
    step = 60.0 / bpm / 4  # a sixteenth: the longest sensible time constant for the shortest note
    return {
        "family": family, "family_dist": fam_dist,
        "harmonics": [float(h) for h in profile],
        "centroid_hz": float(np.mean(cent[active])), "rolloff85_hz": float(np.mean(roll[active])),
        "width_cents": float(width_cents),
        "attack_s": med(atk, fr), "decay_s": med(dec, step), "sustain": med(sus, 0.5),
        "lpenv_oct": max(med(c_ratio, 0.0), 0.0), "lpdecay_s": med(c_dec, step),
        "step_s": step, "sr": int(sr), "onsets": int(len(onsets)),
    }


def _line_width_cents(y, sr, f0, hop) -> float:
    """-6 dB width (cents) of the 3rd partial in the long-window spectrum of the steadiest frames."""
    import librosa
    n_fft = 16384
    ok = np.where(np.isfinite(f0))[0]
    if len(ok) < 8:
        return 0.0
    widths = []
    for t in ok[:: max(1, len(ok) // 24)]:
        s = int(t * hop)
        seg = y[s:s + n_fft]
        if len(seg) < n_fft:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        c = 3 * f0[t]
        sel = np.where(np.abs(freqs - c) <= c * 0.03)[0]
        if len(sel) < 5:
            continue
        pk = sel[np.argmax(spec[sel])]
        half = spec[pk] / 2
        lo, hi = pk, pk
        while lo > sel[0] and spec[lo] > half:
            lo -= 1
        while hi < sel[-1] and spec[hi] > half:
            hi += 1
        fw = (freqs[hi] - freqs[lo]) / c
        # subtract the analysis window's own main-lobe width (Hann: 2 bins at -6 dB)
        res = 2 * (sr / n_fft) / c
        widths.append(1200 * math.log2(1 + max(fw - res, 0.0)))
    return float(np.median(widths)) if widths else 0.0


# ------------------------------------------------------------------------------------- parameter vector
def initial_params(m: dict) -> dict:
    """Starting values, all from the measurement. Cutoff = the stem's 85% rolloff (the filter has to pass it)."""
    return {
        "lpf": float(np.clip(m["rolloff85_hz"], 300.0, 12000.0)),
        "lpenv": float(m["lpenv_oct"]),
        "lpdecay": float(m["lpdecay_s"]),
        "attack": float(max(m["attack_s"], 0.002)),
        "decay": float(max(m["decay_s"], 0.01)),
        "sustain": float(m["sustain"]),
        "detune": float(m["width_cents"] / 100.0),
        "gain": 1.0,
    }


def bounds(m: dict, p0: dict) -> dict:
    """Search box, derived from the measurement: cutoff spans the stem's own centroid..rolloff spread, times
    scale 1/4..4 around the measured values, sustain 0..1."""
    lo_f = max(m["centroid_hz"] / 2, 200.0)
    hi_f = min(m["rolloff85_hz"] * 4, 18000.0)
    return {
        "lpf": (lo_f, hi_f),
        # envelope depth: at most the octaves the cutoff box itself spans
        "lpenv": (0.0, max(math.log2(hi_f / lo_f), p0["lpenv"])),
        "lpdecay": (0.01, max(m["step_s"] * 4, 0.1)),
        "attack": (0.001, max(m["step_s"], 0.02)),
        "decay": (0.01, max(m["step_s"] * 4, 0.1)),
        "sustain": (0.0, 1.0),
        "detune": (0.0, max(m["width_cents"] / 100.0 * 4, 0.3)),
        "gain": (0.4, 2.5),
    }


_LOG = {"lpf", "lpdecay", "attack", "decay", "gain"}


def to_vec(p: dict, b: dict) -> np.ndarray:
    """Unit-cube coordinates (log axes for frequency/time/gain) so one simplex step means the same everywhere."""
    out = []
    for k in PARAM_NAMES:
        lo, hi = b[k]
        v = min(max(p[k], lo), hi)
        if k in _LOG:
            out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else:
            out.append((v - lo) / (hi - lo) if hi > lo else 0.0)
    return np.asarray(out)


def from_vec(x: np.ndarray, b: dict) -> dict:
    p = {}
    for xi, k in zip(x, PARAM_NAMES):
        lo, hi = b[k]
        xi = float(min(max(xi, 0.0), 1.0))
        p[k] = math.exp(math.log(lo) + xi * (math.log(hi) - math.log(lo))) if k in _LOG else lo + xi * (hi - lo)
    return p


def spec_from(m: dict, p: dict) -> dict:
    """Serializable lead spec: the oscillator + the controls. Unison is a count: stacked only when the
    measured line is wider than a single voice (detune > 0)."""
    unison = 1 if p["detune"] < 0.02 else 5
    family = m["family"]
    sound = "supersaw" if (family == "sawtooth" and unison > 1) else family
    return {"sound": sound, "unison": unison, **{k: round(float(v), 4) for k, v in p.items()}}


def lead_chain(spec: dict) -> str:
    """The lead's sound + filter + envelope as plain Strudel controls (placed where .s(...).lpf(...) was)."""
    s = f'.s("{spec["sound"]}")'
    if spec["sound"] == "supersaw":
        s += f'.unison({spec["unison"]}).detune({round(spec["detune"], 3)})'
    s += (f'.attack({round(spec["attack"], 4)}).decay({round(spec["decay"], 3)}).sustain({round(spec["sustain"], 2)})'
          f'.lpf({int(round(spec["lpf"]))})')
    if spec["lpenv"] > 0.05:
        s += (f'.fanchor(0).lpenv({round(spec["lpenv"], 2)}).lpattack({round(spec["attack"], 4)})'
              f'.lpdecay({round(spec["lpdecay"], 3)}).lpsustain(0)')
    return s


# ------------------------------------------------------------------------------------------- the search
def nelder_mead_steps(x0: np.ndarray, *, max_evals: int, step: float = 0.15):
    """Bounded (unit cube) Nelder-Mead as a generator: it yields each point to evaluate and receives the value to
    MINIMISE (``gen.send``). Deterministic, so a run can be resumed by replaying the recorded values (``ask``)."""
    n = len(x0)
    clip = lambda x: np.clip(x, 0.0, 1.0)
    pts = [clip(x0)]
    for i in range(n):
        x = x0.copy()
        x[i] = x[i] + step if x[i] + step <= 1 else x[i] - step
        pts.append(clip(x))
    evals = 0

    def ev(x):
        nonlocal evals
        evals += 1
        return (yield x)

    vals = []
    for x in pts:
        if evals >= max_evals:
            return
        vals.append((yield from ev(x)))
    while evals < max_evals:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        cen = np.mean(pts[:-1], axis=0)
        xr = clip(cen + (cen - pts[-1]))
        fr = yield from ev(xr)
        if fr < vals[0]:
            if evals >= max_evals:
                pts[-1], vals[-1] = xr, fr
                return
            xe = clip(cen + 2 * (cen - pts[-1]))
            fe = yield from ev(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            if evals >= max_evals:
                return
            xc = clip(cen + 0.5 * ((xr if fr < vals[-1] else pts[-1]) - cen))
            fc = yield from ev(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for k in range(1, len(pts)):
                    if evals >= max_evals:
                        return
                    pts[k] = clip(pts[0] + 0.5 * (pts[k] - pts[0]))
                    vals[k] = yield from ev(pts[k])


def nelder_mead(f, x0: np.ndarray, *, max_evals: int, step: float = 0.15, log=print):
    """Minimise ``f`` (unit cube -> value). Returns (best x, best value, value history)."""
    gen = nelder_mead_steps(x0, max_evals=max_evals, step=step)
    seen, history = [], []
    try:
        x = next(gen)
        while True:
            v = float(f(x))
            seen.append(x)
            history.append(v)
            log(f"  eval {len(history):3d} value={v:.5f}")
            x = gen.send(v)
    except StopIteration:
        pass
    k = int(np.argmin(history))
    return seen[k], history[k], history


# ------------------------------------------------------------------ resumable search for auto-calibrate.sh
FREE_DEFAULT = ("lpf", "lpenv", "detune", "gain")   # the four that moved the score; the envelope timings stay measured


def _search_setup(hist: dict, melodic: Path | None, bpm: float, free: list[str], warm: dict | None):
    if "measure" not in hist:
        hist["measure"] = measure_lead(melodic, bpm)
    m = hist["measure"]
    p_meas = initial_params(m)
    b = bounds(m, p_meas)
    p0 = {k: float(warm[k]) for k in PARAM_NAMES} if warm and all(k in warm for k in PARAM_NAMES) else p_meas
    idx = [PARAM_NAMES.index(k) for k in free]
    xfull = to_vec(p0, b)
    return m, b, idx, xfull


def ask(hist: dict, *, melodic: Path | None, bpm: float, cap: int, free: list[str], warm: dict | None):
    """Next candidate to render: ('EVAL', spec) or ('DONE', best spec). ``hist['evals']`` = [{x, score, spec}] of the
    candidates already scored (score = similarity, higher is better); the search is replayed from them."""
    m, b, idx, xfull = _search_setup(hist, melodic, bpm, free, warm)
    evals = hist.setdefault("evals", [])
    scored = [e for e in evals if e.get("score") is not None]
    gen = nelder_mead_steps(xfull[idx], max_evals=cap)
    try:
        x = next(gen)
        for e in scored:
            if not np.allclose(x, np.asarray(e["xs"]), atol=1e-9):
                raise ValueError("history does not match the deterministic search - was the cap/free set changed?")
            x = gen.send(-float(e["score"]))
    except StopIteration:
        x = None
    if x is None or len(scored) >= cap:
        best = max(scored, key=lambda e: e["score"])
        return "DONE", best["spec"]
    full = xfull.copy()
    full[idx] = x
    spec = spec_from(m, from_vec(full, b))
    evals[:] = scored + [{"xs": [float(v) for v in x], "score": None, "spec": spec}]
    return "EVAL", spec


def tell(hist: dict, score: float) -> None:
    pending = [e for e in hist.get("evals", []) if e.get("score") is None]
    if not pending:
        raise ValueError("no pending candidate")
    pending[-1]["score"] = float(score)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="synth lead fit: measure / resumable ask-tell search")
    sp = ap.add_subparsers(dest="cmd", required=True)
    m_ = sp.add_parser("measure", help="print the measured starting spec of a melodic stem")
    m_.add_argument("melodic", type=Path); m_.add_argument("--bpm", type=float, required=True)
    a_ = sp.add_parser("ask", help="print 'EVAL <spec.json>' (render it) or 'DONE <spec.json>' (best so far)")
    a_.add_argument("--history", type=Path, required=True); a_.add_argument("--melodic", type=Path, required=True)
    a_.add_argument("--bpm", type=float, required=True); a_.add_argument("--cap", type=int, required=True)
    a_.add_argument("--free", default=",".join(FREE_DEFAULT)); a_.add_argument("--warm", type=Path)
    a_.add_argument("--out", type=Path, required=True, help="where the candidate / best spec JSON is written")
    t_ = sp.add_parser("tell", help="record the similarity of the pending candidate")
    t_.add_argument("--history", type=Path, required=True); t_.add_argument("--score", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "measure":
        m = measure_lead(a.melodic, a.bpm)
        p0 = initial_params(m)
        print(json.dumps({"measure": m, "initial": spec_from(m, p0), "bounds": bounds(m, p0)}, indent=1))
        return 0
    hist = json.loads(a.history.read_text()) if a.history.is_file() else {}
    if a.cmd == "tell":
        tell(hist, a.score)
        a.history.write_text(json.dumps(hist))
        return 0
    free = [k for k in PARAM_NAMES if k in a.free.split(",")]
    warm = json.loads(a.warm.read_text()) if a.warm and a.warm.is_file() else None
    state, spec = ask(hist, melodic=a.melodic, bpm=a.bpm, cap=a.cap, free=free, warm=warm)
    a.history.write_text(json.dumps(hist))
    a.out.write_text(json.dumps(spec, indent=1))
    print(state, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
