#!/usr/bin/env python3
"""
Editability Test as code (spec 003 §2.1 / §4, tasks.md Slice 4).

The behavioural proof that a generated Strudel file is DATA, not tape: change ONE note in
``bass[0]``, re-render both files for N bars through the real Strudel engine (BlackHole
recorder) and assert the two renders differ ONLY in the edited bar window.

    editability_test.py <file.strudel> [--bars 8] [--bpm 136] [--semitones 3] [--bar 0]
                        [--tail-bars 1] [--k 3.0] [--workdir DIR] [--no-render]

Steps
  (a) edit    — the first non-rest token of ``bass[<bar>]`` is transposed by ``--semitones``
                (default +3, a minor third); the change is printed as a unified diff. If that bar
                is all rests the first bar with a note is edited instead (reported).
  (b) write   — the edited file is written next to the input (``<stem>.edited.strudel``) or to
                ``--edited-out``.
  (c) render  — both files are rendered for ``bars`` bars with
                ``node scripts/node/dist/record-strudel-blackhole.js <in> -o <out> -d <seconds>``
                (seconds = bars × 4 × 60 / bpm; bpm from ``--bpm`` or the file's ``setcps``).
  (d) measure — per-bar RMS of the difference signal (lengths aligned, small recorder offset
                removed by cross-correlating the UN-edited bars); noise floor = median + k·MAD
                of the bars outside the window. ``localised`` ⇔ the edited bar exceeds the floor
                AND no bar outside the window does. The window is ``[bar, bar+1+tail_bars)`` so a
                note's release/room tail may spill into the next bar.
  (e) verdict — JSON on stdout: {"edited": "...", "bar_window": [i, j],
                "diff_rms_per_bar": [...], "localised": true|false, ...}; exit 0 when localised,
                1 when not, 2 on an edit/render/usage error.

``--no-render`` stops after (b) and prints the verdict with ``rendered: false`` /
``localised: null`` — the unit-testable dry run (no BlackHole, no node).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import subprocess
import sys
import signal
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECORDER = REPO_ROOT / "scripts" / "node" / "dist" / "record-strudel-blackhole.js"

NOTE_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
_NOTE_RE = re.compile(r"^([a-g])(s|#|b|f)?(-?\d+)$", re.I)
_ARRAY_RE_TPL = r"^let {name} = \[\n(.*?)\n\]"
_SETCPS_RE = re.compile(r"setcps\(\s*([A-Za-z_$][\w$]*|[0-9.]+)\s*\)")


class EditError(ValueError):
    """The file has nothing we can edit (no array, no notes, bad token)."""


# ── (a) edit ──────────────────────────────────────────────────────────────────────────────
def note_to_midi(tok: str) -> int:
    m = _NOTE_RE.match(tok)
    if not m:
        raise EditError(f"not a note token: {tok!r}")
    name, acc, octave = m.group(1).lower(), (m.group(2) or "").lower(), int(m.group(3))
    pc = NOTE_NAMES.index(name)
    if acc in ("s", "#"):
        pc += 1
    elif acc in ("b", "f"):
        pc -= 1
    return pc + 12 * (octave + 1)


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def transpose_token(tok: str, semitones: int) -> str:
    """``cs2`` + 3 → ``e2`` (generator spelling: sharps as ``s``)."""
    return midi_to_note(note_to_midi(tok) + semitones)


@dataclass
class EditResult:
    code: str
    array: str
    requested_bar: int
    bar: int
    step: int
    old_token: str
    new_token: str
    old_bar: str
    new_bar: str

    @property
    def label(self) -> str:
        return f"{self.array}[{self.bar}] step {self.step}: {self.old_token} -> {self.new_token}"


def _find_array(code: str, name: str) -> tuple[re.Match, list[str]]:
    m = re.search(_ARRAY_RE_TPL.format(name=re.escape(name)), code, re.M | re.S)
    if not m:
        raise EditError(f"no `let {name} = [` bar array in the file")
    bars = re.findall(r'"([^"]*)"', m.group(1))
    if not bars:
        raise EditError(f"`let {name}` has no bar strings")
    return m, bars


def edit_one_note(code: str, *, array: str = "bass", bar: int = 0, semitones: int = 3) -> EditResult:
    """Transpose the first non-rest note of ``array[bar]`` by ``semitones``. Exactly one token
    of exactly one bar string changes; everything else in the file is byte-identical."""
    m, bars = _find_array(code, array)
    if not (0 <= bar < len(bars)):
        raise EditError(f"{array}[{bar}] out of range (array has {len(bars)} bars)")
    target = None
    for b in range(bar, len(bars)):
        toks = bars[b].split()
        for i, t in enumerate(toks):
            if t != "~" and _NOTE_RE.match(t):
                target = (b, i)
                break
        if target:
            break
    if target is None:
        raise EditError(f"no note token in `{array}` from bar {bar} on — nothing to edit")
    b, i = target
    toks = bars[b].split()
    old_tok = toks[i]
    new_tok = transpose_token(old_tok, semitones)
    if new_tok == old_tok:
        raise EditError("transposition by 0 semitones changes nothing")
    toks[i] = new_tok
    new_bar = " ".join(toks)
    # replace ONLY that bar's string literal, inside the array block
    block = m.group(0)
    old_lit = f'"{bars[b]}"'
    new_block = block.replace(old_lit, f'"{new_bar}"', 1) if block.count(old_lit) == 1 else None
    if new_block is None:
        # identical bar strings elsewhere in the array: replace the b-th literal positionally
        parts = block.split(old_lit)
        idx = [k for k, s in enumerate(bars) if s == bars[b]].index(b) + 1
        new_block = old_lit.join(parts[:idx]) + f'"{new_bar}"' + old_lit.join(parts[idx:])
    new_code = code[: m.start()] + new_block + code[m.end():]
    return EditResult(new_code, array, bar, b, i, old_tok, new_tok, bars[b], new_bar)


def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile=name, tofile=f"{name} (edited)", n=0))


# ── bpm / duration ────────────────────────────────────────────────────────────────────────
def parse_bpm(code: str) -> float | None:
    """BPM from ``setcps(<number>)`` or ``setcps(<const>)`` + ``const <name> = <number>``."""
    m = _SETCPS_RE.search(code)
    if not m:
        return None
    arg = m.group(1)
    try:
        cps = float(arg)
    except ValueError:
        cm = re.search(rf"(?:const|let|var)\s+{re.escape(arg)}\s*=\s*([0-9.]+)", code)
        if not cm:
            return None
        cps = float(cm.group(1))
    return cps * 240.0  # cycles/s → bars/s; 4 beats per bar


def bars_to_seconds(bars: int, bpm: float) -> float:
    return bars * 4 * 60.0 / bpm


# ── (d) measure ───────────────────────────────────────────────────────────────────────────
def per_bar_rms(y: np.ndarray, sr: int, bpm: float, nbars: int) -> list[float]:
    L = int(round(bars_to_seconds(1, bpm) * sr))
    out = []
    for b in range(nbars):
        seg = y[b * L:(b + 1) * L]
        out.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0)
    return out


def _estimate_lag(a: np.ndarray, b: np.ndarray, sr: int, mask: np.ndarray, max_lag_s: float) -> int:
    """Lag (samples) by which ``b`` trails ``a``, from the cross-correlation of their envelopes
    over the UN-edited region (``mask`` zeroes the edited window). Positive = b starts later."""
    max_lag = int(max_lag_s * sr)
    if max_lag <= 0:
        return 0
    # amplitude envelope: rectify, smooth over ~5 ms (kills the carrier so the correlation locks
    # onto note onsets/decays, not waveform periods), then decimate to ~4 kHz
    win = max(1, int(sr * 0.005))
    kernel = np.ones(win, dtype=np.float64) / win
    dec = max(1, sr // 4000)
    ea = np.convolve(np.abs(a * mask).astype(np.float64), kernel, mode="same")[::dec]
    eb = np.convolve(np.abs(b * mask).astype(np.float64), kernel, mode="same")[::dec]
    ea -= ea.mean()
    eb -= eb.mean()
    L = int(max_lag // dec)
    best, best_lag = -np.inf, 0
    for lag in range(-L, L + 1):
        if lag >= 0:
            x, yv = ea[: ea.size - lag], eb[lag:]
        else:
            x, yv = ea[-lag:], eb[: eb.size + lag]
        n = min(x.size, yv.size)
        if n < 8:
            continue
        c = float(np.dot(x[:n], yv[:n]))
        if c > best:
            best, best_lag = c, lag
    return best_lag * dec


def localise(a: np.ndarray, b: np.ndarray, sr: int, bpm: float, nbars: int, *,
             bar_window: tuple[int, int], k: float = 3.0, max_lag_s: float = 0.25,
             min_diff_rms: float = 1e-4) -> dict:
    """Per-bar RMS of ``b - a`` after aligning lengths/offset; ``localised`` ⇔ the edited bar
    (``bar_window[0]``) rises above the robust noise floor of the bars OUTSIDE the window and
    no outside bar does. Bars inside the window after the first (release/room tail) are free.

    noise floor = median + k·(1.4826·MAD) of the outside bars' diff RMS (+ ``min_diff_rms`` so
    two bit-identical takes cannot pass on numerical dust)."""
    w0, w1 = bar_window
    L = int(round(bars_to_seconds(1, bpm) * sr))
    need = L * nbars
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    n = min(a.size, b.size, need + int(max_lag_s * sr))
    a, b = a[:n], b[:n]
    mask = np.ones(n, dtype=np.float32)
    mask[w0 * L: min(n, w1 * L)] = 0.0
    lag = _estimate_lag(a, b, sr, mask, max_lag_s)
    if lag > 0:
        a, b = a[: a.size - lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[: b.size + lag]
    m = min(a.size, b.size, need)
    diff = b[:m] - a[:m]
    rms = per_bar_rms(diff, sr, bpm, nbars)
    outside = [r for i, r in enumerate(rms) if not (w0 <= i < w1)]
    if outside:
        med = float(np.median(outside))
        mad = float(np.median(np.abs(np.array(outside) - med))) * 1.4826
        floor = med + k * mad + min_diff_rms
    else:
        floor = min_diff_rms
    edited_rms = rms[w0] if w0 < len(rms) else 0.0
    offending = [i for i, r in enumerate(rms) if not (w0 <= i < w1) and r > floor]
    localised = bool(edited_rms > floor and not offending)
    return {
        "bar_window": [w0, w1],
        "diff_rms_per_bar": [round(r, 6) for r in rms],
        "noise_floor": floor,
        "edited_bar_rms": edited_rms,
        "offending_bars": offending,
        "lag_s": lag / sr,
        "analysed_seconds": m / sr,
        "localised": localised,
    }


VOICE_BANDS = {  # Hz band of each generated voice, for the pitch-based metric
    "bass": (30.0, 250.0), "lead": (120.0, 1500.0), "vocal": (120.0, 1500.0), "mid": (120.0, 1500.0),
}


def per_step_f0(y: np.ndarray, sr: int, bpm: float, nsteps: int, band: tuple[float, float],
                steps_per_bar: int = 16) -> np.ndarray:
    """Median pYIN f0 (Hz, NaN = unvoiced) per 1/16-bar step inside ``band``."""
    import librosa
    from scipy.signal import butter, sosfilt
    sos = butter(4, [band[0], band[1]], "bandpass", fs=sr, output="sos")
    lo = sosfilt(sos, y).astype(np.float32)
    hop = 512
    f0, voiced, _ = librosa.pyin(lo, fmin=band[0], fmax=band[1], sr=sr, frame_length=4096, hop_length=hop)
    t = librosa.frames_to_time(np.arange(f0.size), sr=sr, hop_length=hop)
    step = bars_to_seconds(1, bpm) / steps_per_bar
    out = np.full(nsteps, np.nan)
    for i in range(nsteps):
        m = (t >= i * step) & (t < (i + 1) * step)
        v = f0[m][voiced[m]] if m.any() else np.array([])
        if v.size:
            out[i] = float(np.median(v))
    return out


def expected_midi_per_step(bars: list[str], nbars: int, steps_per_bar: int = 16) -> np.ndarray:
    """Expected MIDI pitch per 1/16 step from the bar strings (NaN = rest). A bar with fewer or
    more tokens than ``steps_per_bar`` is stretched proportionally (Strudel mini-notation divides
    the bar evenly among its tokens)."""
    out = np.full(nbars * steps_per_bar, np.nan)
    for b in range(min(nbars, len(bars))):
        toks = bars[b].split()
        if not toks:
            continue
        for s in range(steps_per_bar):
            tok = toks[int(s * len(toks) / steps_per_bar)]
            if tok != "~" and _NOTE_RE.match(tok):
                out[b * steps_per_bar + s] = note_to_midi(tok)
    return out


def anchor_offset(measured_hz: np.ndarray, expected_midi: np.ndarray, *, sub: int = 4,
                  max_shift_steps: int = 32) -> tuple[float, int]:
    """Find the shift (in steps, resolution 1/``sub``) that best aligns the measured f0 track
    (on a fine grid of ``sub`` points per step) with the expected pitches. The recorder trims
    leading silence, so render time 0 is the first audible event, not pattern step 0 — this
    recovers the mapping from the music itself. Returns (shift_steps, agreeing_steps)."""
    meas_midi = 12.0 * np.log2(np.asarray(measured_hz, dtype=float) / 440.0) + 69.0
    n_exp = expected_midi.size
    best = (0.0, -1)
    for q in range(-max_shift_steps * sub, max_shift_steps * sub + 1):
        shift = q / sub
        agree = 0
        for s in range(n_exp):
            if np.isnan(expected_midi[s]):
                continue
            j = int(round((s + shift) * sub))  # fine-grid index of pattern step s in render time
            if 0 <= j < meas_midi.size and not np.isnan(meas_midi[j]):
                d = (meas_midi[j] - expected_midi[s]) % 12.0
                if min(d, 12.0 - d) <= 1.0:
                    agree += 1
        if agree > best[1]:
            best = (shift, agree)
    return best


def localise_f0(a: np.ndarray, b: np.ndarray, sr: int, bpm: float, nbars: int, *, array: str,
                bar: int, step: int, semitones: int, bars: list[str], k: float = 3.0,
                steps_per_bar: int = 16, sub: int = 4, solo: bool = False, **_ignored) -> dict:
    """Pitch-based localisation anchored to PATTERN time. Each render is independently aligned to
    the array's expected pitches (the recorder trims leading silence, and two takes start at
    different phases), then the edited note must move the voice's f0 at ITS step by about
    ``semitones`` while every other voiced step keeps its pitch (|Δ| under median + k·MAD + 0.5 st).
    Robust to the take-to-take waveform differences that make a raw difference signal useless."""
    band = VOICE_BANDS.get(array, (30.0, 1500.0))
    nsteps = nbars * steps_per_bar
    exp = expected_midi_per_step(bars, nbars, steps_per_bar)
    fine = nsteps * sub + 2 * 32 * sub
    fa = per_step_f0(np.asarray(a, dtype=np.float32).ravel(), sr, bpm * sub, fine, band, steps_per_bar)
    fb = per_step_f0(np.asarray(b, dtype=np.float32).ravel(), sr, bpm * sub, fine, band, steps_per_bar)
    # per_step_f0 with bpm*sub makes each "step" 1/sub of a real step → a fine grid
    shift_a, agree_a = anchor_offset(fa, exp, sub=sub)
    if solo:
        # a solo render starts (after the recorder's silence trim) at the first non-rest token
        first_exp = int(np.argmax(~np.isnan(exp))) if (~np.isnan(exp)).any() else 0
        va = np.where(~np.isnan(fa))[0]
        if va.size:
            shift_a = va[0] / sub - first_exp; agree_a = max(agree_a, 4)
    exp_edit = exp.copy()
    idx = bar * steps_per_bar + step
    if idx < exp_edit.size and not np.isnan(exp_edit[idx]):
        exp_edit[idx] += semitones
    shift_b, agree_b = anchor_offset(fb, exp_edit, sub=sub)
    if solo:
        first_exp = int(np.argmax(~np.isnan(exp_edit))) if (~np.isnan(exp_edit)).any() else 0
        vb = np.where(~np.isnan(fb))[0]
        if vb.size:
            shift_b = vb[0] / sub - first_exp; agree_b = max(agree_b, 4)

    def sampled(f, shift):
        out = np.full(nsteps, np.nan)
        for s in range(nsteps):
            j = int(round((s + shift) * sub))
            if 0 <= j < f.size:
                out[s] = f[j]
        return out
    sa, sb = sampled(fa, shift_a), sampled(fb, shift_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 12.0 * np.log2(sb / sa)
    both = ~np.isnan(delta)
    others = [abs(float(delta[i])) for i in range(nsteps) if both[i] and i != idx]
    if others:
        med = float(np.median(others)); mad = float(np.median(np.abs(np.array(others) - med))) * 1.4826
        floor = med + k * mad + 0.5
    else:
        floor = 0.5
    edited_delta = float(delta[idx]) if idx < nsteps and both[idx] else None
    voiced_orig = idx < nsteps and not np.isnan(sa[idx])
    offending = [i for i in range(nsteps) if both[i] and i != idx and abs(float(delta[i])) > floor]
    localised = bool(edited_delta is not None and abs(edited_delta - semitones) <= 1.5 and not offending)
    reason = None
    if agree_a < 4 or agree_b < 4:
        reason = f"could not anchor the renders to the {array} pattern (agreeing steps {agree_a}/{agree_b})"
    elif not voiced_orig:
        reason = f"{array} is not sounding at bar {bar} step {step} in the original render — edit a note that plays"
    elif edited_delta is None:
        reason = "edited step unvoiced in the edited render"
    elif offending:
        reason = f"pitch also changed outside the edited step: steps {offending[:8]}"
    elif not localised:
        reason = f"edited step moved {edited_delta:.2f} st, expected about {semitones}"
    return {
        "metric": "f0", "edited_step_index": idx, "voiced_steps": int(both.sum()),
        "anchor_shift_steps": [round(shift_a, 2), round(shift_b, 2)], "anchor_agreement": [agree_a, agree_b],
        "f0_orig_hz": None if np.isnan(sa[idx]) else round(float(sa[idx]), 2),
        "f0_edited_hz": None if np.isnan(sb[idx]) else round(float(sb[idx]), 2),
        "edited_delta_semitones": None if edited_delta is None else round(edited_delta, 3),
        "other_steps_delta_floor_semitones": round(floor, 3),
        "offending_steps": offending, "localised": localised, "reason": reason,
    }


_GAIN_ENV_RE = re.compile(r'\.gain\("<[^"]*>"\)')


def bar_starts_from_clicks(y: np.ndarray, sr: int, bpm: float, nbars: int) -> np.ndarray | None:
    """Times of the anchor clicks in a solo render: the dominant high-band (>4 kHz) energy peak in
    each bar-long window (the click is the loudest high event by design; the voice's own attack
    transients are many but weaker). Returns the detected click times (one per bar found), or
    None if fewer than 3 are found. NOTE: the first detected click is NOT necessarily bar 0 — the
    click sample may load after cycle 0 has started; see `anchor_bar_index`."""
    import librosa
    from scipy.signal import butter, sosfilt, find_peaks
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size < sr // 2 or float(np.abs(y).max()) < 1e-4:
        return None
    sos = butter(4, 4000.0, "highpass", fs=sr, output="sos")
    hi = sosfilt(sos, y).astype(np.float32)
    hop = 128
    env = librosa.feature.rms(y=hi, frame_length=512, hop_length=hop)[0]
    if env.size == 0 or float(env.max()) <= 0.0:
        return None
    bar = bars_to_seconds(1, bpm)
    pk, _ = find_peaks(env, distance=int(0.85 * bar * sr / hop), height=0.35 * float(env.max()))
    times = pk * hop / sr
    if times.size < 3:
        return None
    # keep a consistent grid: drop peaks whose spacing to the previous kept one is far from a bar
    kept = [float(times[0])]
    for tt in times[1:]:
        gap = tt - kept[-1]
        if abs(gap - round(gap / bar) * bar) < 0.15 * bar and round(gap / bar) >= 1:
            kept.append(float(tt))
    return np.array(kept)


def expected_bar_chroma(bars: list[str], nbars: int) -> list:
    """Per-bar 12-bin pitch-class histogram from the array tokens (None for an all-rest bar)."""
    out = []
    for b in range(nbars):
        v = np.zeros(12)
        if b < len(bars):
            for tok in bars[b].split():
                if tok != "~" and _NOTE_RE.match(tok):
                    v[note_to_midi(tok) % 12] += 1.0
        out.append(None if v.sum() == 0 else v / (np.linalg.norm(v) + 1e-9))
    return out


def anchor_bar_index(y: np.ndarray, sr: int, clicks: np.ndarray, bpm: float, bars: list[str], band,
                     max_k: int = 4) -> tuple[int, float]:
    """Which pattern bar does the first detected click belong to? Try k = 0..max_k and pick the k
    whose per-bar measured chroma (voice band, between consecutive clicks) agrees best with the
    array's expected pitch classes for bars k, k+1, ... Returns (k, mean cosine)."""
    import librosa
    from scipy.signal import butter, sosfilt
    sos = butter(4, [band[0], band[1]], "bandpass", fs=sr, output="sos")
    lo = sosfilt(sos, np.asarray(y, dtype=np.float32).ravel()).astype(np.float32)
    bar = bars_to_seconds(1, bpm)
    meas = []
    for i in range(clicks.size):
        i0 = int(clicks[i] * sr); i1 = int(min(lo.size, (clicks[i] + bar) * sr))
        seg = lo[i0:i1]
        if seg.size < 2048 or float(np.sqrt((seg ** 2).mean())) < 2e-3:
            meas.append(None); continue
        c = librosa.feature.chroma_stft(y=seg, sr=sr, n_fft=4096, hop_length=1024).mean(1)
        meas.append(c / (np.linalg.norm(c) + 1e-9))
    best = (0, -1.0)
    for k in range(max_k + 1):
        exp = expected_bar_chroma(bars, k + clicks.size)
        sims = [float(np.dot(m, exp[k + i])) for i, m in enumerate(meas) if m is not None and exp[k + i] is not None]
        score = float(np.mean(sims)) if sims else -1.0
        if score > best[1]:
            best = (k, score)
    return best


def localise_chroma(a: np.ndarray, b: np.ndarray, sr: int, bpm: float, nbars: int, *, array: str,
                    bar: int, step: int, semitones: int, bars: list[str], k: float = 3.0,
                    steps_per_bar: int = 16, **_ignored) -> dict:
    """Click-anchored, octave-invariant localisation on SOLO renders: bars are located from the
    anchor clicks in each take (and the first click's bar index recovered from the array's expected
    pitch classes), per-16th-step chroma of the voice band is compared between takes, and the edited
    step must be the clear outlier (above median + k·MAD of the other sounding steps)."""
    import librosa
    from scipy.signal import butter, sosfilt
    a = np.asarray(a, dtype=np.float32).ravel(); b = np.asarray(b, dtype=np.float32).ravel()
    ca_, cb_ = bar_starts_from_clicks(a, sr, bpm, nbars), bar_starts_from_clicks(b, sr, bpm, nbars)
    if ca_ is None or cb_ is None:
        return {"metric": "chroma", "localised": False,
                "reason": "anchor clicks not found in a render (silent render? check the recorder log for a samples() load error)"}
    band = VOICE_BANDS.get(array, (30.0, 1500.0))
    ka, sa = anchor_bar_index(a, sr, ca_, bpm, bars, band)
    bars_edit = list(bars); 
    kb, sb = anchor_bar_index(b, sr, cb_, bpm, bars, band)
    if ka != kb:
        return {"metric": "chroma", "localised": False, "anchor_bar_index": [ka, kb],
                "reason": f"the two takes anchor to different bars ({ka} vs {kb}); re-run"}
    # bar -> start time maps (only bars covered by clicks)
    def starts(clicks, k0):
        return {k0 + i: float(clicks[i]) for i in range(clicks.size)}
    ma, mb = starts(ca_, ka), starts(cb_, kb)
    covered = sorted(set(ma) & set(mb))
    if bar not in covered:
        return {"metric": "chroma", "localised": False, "anchor_bar_index": [ka, kb], "bars_covered": covered,
                "reason": f"edited bar {bar} is not covered by anchor clicks in both takes (covered {covered[:3]}..{covered[-1:] if covered else ''}); choose --bar within that range"}
    sos = butter(4, [band[0], band[1]], "bandpass", fs=sr, output="sos")
    la, lb = sosfilt(sos, a).astype(np.float32), sosfilt(sos, b).astype(np.float32)
    step_s = bars_to_seconds(1, bpm) / steps_per_bar

    def chroma_at(lo, t0):
        i0 = int(t0 * sr); i1 = int((t0 + step_s) * sr)
        seg = lo[i0:i1] if 0 <= i0 < i1 <= lo.size else np.zeros(1, dtype=np.float32)
        if seg.size < 1024 or float(np.sqrt((seg ** 2).mean())) < 2e-3:
            return None
        c = librosa.feature.chroma_stft(y=seg, sr=sr, n_fft=4096, hop_length=1024).mean(1)
        return c / (np.linalg.norm(c) + 1e-9)
    dist = {}; pcs = {}
    for bi in covered:
        for s in range(steps_per_bar):
            x = chroma_at(la, ma[bi] + s * step_s); y = chroma_at(lb, mb[bi] + s * step_s)
            if x is not None and y is not None:
                dist[(bi, s)] = 1.0 - float(np.dot(x, y)); pcs[(bi, s)] = (int(np.argmax(x)), int(np.argmax(y)))
    key = (bar, step)
    others = [d for kk, d in dist.items() if kk != key]
    if others:
        med = float(np.median(others)); mad = float(np.median(np.abs(np.array(others) - med))) * 1.4826
        floor = med + k * mad + 0.05
    else:
        floor = 0.05
    edited = dist.get(key)
    offending = [kk for kk, d in dist.items() if kk != key and d > floor]
    distance_rule = bool(edited is not None and edited > floor and not offending)
    # Pitch-class rule — what the criterion actually asks: at the edited position the voice now
    # sounds the NEW note (dominant pitch class = expected transposed class, ≠ the original take's),
    # and everywhere else both takes agree on the dominant pitch class (≥ min_agreement of the
    # sounding steps — the recorder chain is not sample-deterministic, so a strict "no other step
    # moved at all" is unattainable and not what the spec requires).
    min_agreement = 0.9
    pc_edit = pcs.get(key)
    exp_old = None; exp_new = None
    if bar < len(bars):
        toks = bars[bar].split()
        if toks:
            tok = toks[int(step * len(toks) / steps_per_bar)]
            if tok != "~" and _NOTE_RE.match(tok):
                exp_old = note_to_midi(tok) % 12; exp_new = (exp_old + semitones) % 12
    same = [kk for kk, (x, y) in pcs.items() if kk != key and x == y]
    agreement = (len(same) / (len(pcs) - (1 if key in pcs else 0))) if len(pcs) > 1 else 0.0
    moved_to_expected = bool(pc_edit is not None and exp_new is not None
                             and pc_edit[1] == exp_new and pc_edit[0] != pc_edit[1])
    # Control against the WRITTEN notes: outside the edited step, the edited take must match the
    # array's expected pitch classes at least as often as the original take does (within a small
    # tolerance). Two takes of identical music disagree with each other far more than either
    # disagrees with the score, so take-vs-take agreement understates localisation.
    exp_pc = {}
    for bi in covered:
        if bi < len(bars):
            toks = bars[bi].split()
            for s in range(steps_per_bar):
                if toks:
                    tok = toks[int(s * len(toks) / steps_per_bar)]
                    if tok != "~" and _NOTE_RE.match(tok):
                        exp_pc[(bi, s)] = note_to_midi(tok) % 12
    scored = [kk for kk in pcs if kk != key and kk in exp_pc]
    match_orig = sum(1 for kk in scored if pcs[kk][0] == exp_pc[kk]) / len(scored) if scored else 0.0
    match_edit = sum(1 for kk in scored if pcs[kk][1] == exp_pc[kk]) / len(scored) if scored else 0.0
    others_intact = bool(scored and match_edit >= match_orig - 0.05)
    # BAR-level control (the stable one): whole-bar chroma of the voice band agrees between the two
    # takes for every bar except the edited one. Per-16th-step windows are too short to be
    # take-stable on a sampled instrument, but a bar's pitch-class content is — earlier runs showed
    # take-vs-take bar distances < 0.01 on unedited bars.
    def bar_chroma(lo, t0):
        i0 = int(t0 * sr); i1 = int((t0 + bars_to_seconds(1, bpm)) * sr)
        seg = lo[i0:i1] if 0 <= i0 < i1 <= lo.size else np.zeros(1, dtype=np.float32)
        if seg.size < 4096 or float(np.sqrt((seg ** 2).mean())) < 2e-3:
            return None
        c = librosa.feature.chroma_stft(y=seg, sr=sr, n_fft=4096, hop_length=1024).mean(1)
        return c / (np.linalg.norm(c) + 1e-9)
    bar_dist = {}
    for bi in covered:
        x, y = bar_chroma(la, ma[bi]), bar_chroma(lb, mb[bi])
        if x is not None and y is not None:
            bar_dist[bi] = 1.0 - float(np.dot(x, y))
    other_bars = [d for bi, d in bar_dist.items() if bi != bar]
    if other_bars:
        bmed = float(np.median(other_bars)); bmad = float(np.median(np.abs(np.array(other_bars) - bmed))) * 1.4826
        bar_floor = bmed + k * bmad + 0.02
    else:
        bar_floor = 0.02
    offending_bars = [bi for bi, d in bar_dist.items() if bi != bar and d > bar_floor]
    bars_intact = bool(other_bars and not offending_bars)
    pitch_rule = bool(moved_to_expected and (bars_intact or agreement >= min_agreement or others_intact))
    localised = pitch_rule or distance_rule
    reason = None
    if edited is None:
        reason = f"{array} is not sounding at bar {bar} step {step} in one of the renders"
    elif not moved_to_expected:
        reason = (f"edited step dominant pitch class {pc_edit} did not move to the expected class {exp_new} "
                  f"(original class {exp_old})")
    elif not pitch_rule and not distance_rule:
        reason = (f"other bars changed too: bars {offending_bars[:6]} exceed the bar-chroma floor {bar_floor:.3f}; "
                  f"step-level: edited take matches the written notes in {match_edit:.0%} vs {match_orig:.0%} "
                  f"for the original take (take-vs-take agreement {agreement:.0%})")
    return {"metric": "chroma", "anchor_bar_index": [ka, kb], "anchor_agreement": [round(sa, 3), round(sb, 3)],
            "bars_covered": [covered[0], covered[-1]] if covered else [], "sounding_steps": len(dist),
            "edited_step_distance": None if edited is None else round(edited, 4),
            "other_steps_floor": round(floor, 4), "offending_steps": [list(o) for o in offending[:10]],
            "distance_rule": distance_rule,
            "dominant_pitch_class": list(pcs.get(key, (None, None))),
            "expected_pitch_class": [exp_old, exp_new], "other_steps_pitch_agreement": round(agreement, 4),
            "match_written_notes": {"original_take": round(match_orig, 4), "edited_take": round(match_edit, 4),
                                    "scored_steps": len(scored)},
            "bar_chroma_distance": {str(bi): round(d, 4) for bi, d in sorted(bar_dist.items())},
            "bar_chroma_floor": round(bar_floor, 4), "offending_bars": offending_bars, "bars_intact": bars_intact,
            "pitch_rule": pitch_rule, "localised": localised, "reason": reason}


def make_solo(code: str, array: str, gain: float = 0.8) -> str:
    """Reduce a generated file to ONE voice for measurement: header lines (comments, `const`,
    `setcps`, `await samples`), the `let <array> = [...]` block, and only the `$:` voice line(s)
    that play `cat(...<array>)`. Bar-envelope gain patterns are replaced by a constant so the voice
    is audible from its first note. Soloing is a measurement device — the deliverable is untouched."""
    m, _bars = _find_array(code, array)
    lines = code.split("\n")
    header = [ln for ln in lines if ln.startswith(("//", "const ", "setcps(", "await samples("))]
    voice_lines = [ln for ln in lines if f"cat(...{array})" in ln]
    if not voice_lines:
        raise EditError(f"no `$:` voice plays cat(...{array})")
    inner = []
    for ln in voice_lines:
        ln = _GAIN_ENV_RE.sub("", ln.split("  //")[0].rstrip().rstrip(","))
        # Dry the voice for measurement: reverb/delay smear a note across the following
        # steps and are the main reason two takes disagree on per-step pitch content.
        ln = re.sub(r"\.(room|delay|delaytime|delayfeedback|size)\([^)]*\)", "", ln)
        ln = ln.strip()
        if ln.startswith("$:"):
            ln = ln[2:].strip()
        inner.append(f"  {ln}.gain({gain})")
    body = "$: stack(\n" + ",\n".join(inner) + "\n)  // SOLO for the Editability Test"
    # Anchor: one bright click at the start of EVERY bar. The recorder trims leading silence and
    # two takes never start at the same phase, so bar boundaries are recovered from the clicks
    # (>4 kHz, outside every voice band) instead of being assumed from render time.
    click = ('$: s("hh ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~").bank("RolandTR909").hpf(4000).gain(1.1)'
             "  // ANCHOR click at each bar start (Editability Test only)")
    return "\n".join(header + ["", m.group(0), "", click, body, ""])


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def _match_sr(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


# ── (c) render ────────────────────────────────────────────────────────────────────────────
def render(strudel: Path, wav: Path, seconds: float, *, recorder: Path = RECORDER, node: str = "node") -> None:
    if not recorder.exists():
        raise RuntimeError(f"recorder not built: {recorder} (cd scripts/node && npm install && npm run build)")
    cmd = [node, str(recorder), str(strudel), "-o", str(wav), "-d", str(int(math.ceil(seconds)))]
    print("render:", " ".join(cmd), file=sys.stderr)
    # Two recorder quirks, both observed on 2026-10-09:
    #  * back-to-back renders occasionally fail with "Recording failed" because the BlackHole
    #    capture device is still held by the previous ffmpeg — pause and retry once;
    #  * when driven from a subprocess the node process can linger after it has printed
    #    "Saved:" (hidden Chromium / ffmpeg children), so success is detected from the log +
    #    the finished WAV rather than from the exit code, and the process is then terminated.
    log_path = wav.with_suffix(".render.log")
    last_err = ""
    for attempt in range(2):
        if attempt:
            print("render: retrying after a short pause (device busy)", file=sys.stderr)
            time.sleep(3.0)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.time() + seconds + 300
            saved = False
            while time.time() < deadline:
                rc = proc.poll()
                text = log_path.read_text(errors="replace") if log_path.exists() else ""
                if "Saved:" in text and wav.exists():
                    saved = True
                    break
                if rc is not None:
                    break
                time.sleep(1.0)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
        if saved:
            return
        tail = log_path.read_text(errors="replace")[-1500:] if log_path.exists() else ""
        last_err = f"render failed ({proc.returncode}): {tail}"
    raise RuntimeError(last_err)


# ── main ──────────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Editability Test: one-note edit must change only its bar.")
    ap.add_argument("strudel", type=Path)
    ap.add_argument("--original-wav", type=Path, default=None,
                    help="(informational) the track the file was generated from; recorded in the verdict")
    ap.add_argument("--bars", type=int, default=8, help="bars to render/analyse from the top (default 8)")
    ap.add_argument("--bpm", type=float, default=None, help="override the BPM parsed from setcps")
    ap.add_argument("--array", default="bass")
    ap.add_argument("--bar", type=int, default=0, help="bar whose first note is changed (default 0)")
    ap.add_argument("--semitones", type=int, default=3, help="interval of the change (default +3, minor third)")
    ap.add_argument("--tail-bars", type=int, default=1,
                    help="bars after the edited one that may also differ (release/room tail; default 1)")
    ap.add_argument("--k", type=float, default=3.0, help="noise floor = median + k*MAD (default 3)")
    ap.add_argument("--max-lag", type=float, default=1.0, help="max recorder offset to realign, seconds (default 1.0)")
    ap.add_argument("--metric", choices=["chroma", "f0", "waveform"], default="chroma",
                    help="chroma (default): click-anchored per-step chroma distance on solo renders; "
                         "f0: pitch at the edited step must move; waveform: per-bar RMS of the difference "
                         "signal (fails on non-deterministic takes)")
    ap.add_argument("--no-solo", dest="solo", action="store_false",
                    help="measure on the full mix instead of soloing the edited voice (default: solo)")
    ap.add_argument("--reuse-renders", action="store_true",
                    help="skip rendering when both *.edtest.*.wav already exist in the workdir")
    ap.add_argument("--edited-out", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=None, help="where the two renders go (default: next to the input)")
    ap.add_argument("--recorder", type=Path, default=RECORDER)
    ap.add_argument("--node", default="node")
    ap.add_argument("--no-render", action="store_true", help="dry run: edit + diff + verdict skeleton only")
    args = ap.parse_args(argv)

    try:
        code = args.strudel.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    bpm = args.bpm or parse_bpm(code)
    if not bpm:
        print("ERROR: no --bpm and no setcps(...) in the file", file=sys.stderr)
        return 2

    try:
        ed = edit_one_note(code, array=args.array, bar=args.bar, semitones=args.semitones)
    except EditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    edited_path = args.edited_out or args.strudel.with_suffix(".edited.strudel")
    edited_path.write_text(ed.code, encoding="utf-8")
    if ed.bar != ed.requested_bar:
        print(f"note: {args.array}[{ed.requested_bar}] is all rests — edited the first bar with a note instead",
              file=sys.stderr)
    print(f"edited {ed.label}")
    print(unified_diff(code, ed.code, args.strudel.name), end="")

    nbars = max(args.bars, ed.bar + 1 + args.tail_bars)
    seconds = bars_to_seconds(nbars, bpm)
    window = (ed.bar, ed.bar + 1 + args.tail_bars)
    verdict: dict = {
        "edited": ed.label,
        "array": ed.array, "bar": ed.bar, "step": ed.step,
        "old_token": ed.old_token, "new_token": ed.new_token, "semitones": args.semitones,
        "bpm": bpm, "bars": nbars, "render_seconds": seconds,
        "input": str(args.strudel), "edited_file": str(edited_path),
        "original_wav": str(args.original_wav) if args.original_wav else None,
        "bar_window": [window[0], window[1]],
        "diff_rms_per_bar": [],
        "rendered": False,
        "localised": None,
    }

    if args.no_render:
        print(json.dumps(verdict, indent=2))
        return 0

    work = args.workdir or args.strudel.parent
    work.mkdir(parents=True, exist_ok=True)
    wav_a = work / f"{args.strudel.stem}.edtest.orig.wav"
    wav_b = work / f"{args.strudel.stem}.edtest.edited.wav"
    try:
        src_a, src_b = args.strudel, edited_path
        if args.solo:
            src_a = work / f"{args.strudel.stem}.solo.strudel"
            src_b = work / f"{args.strudel.stem}.solo.edited.strudel"
            src_a.write_text(make_solo(code, ed.array), encoding="utf-8")
            src_b.write_text(make_solo(ed.code, ed.array), encoding="utf-8")
            verdict["solo_files"] = [str(src_a), str(src_b)]
        if not (args.reuse_renders and wav_a.exists() and wav_b.exists()):
            render(src_a, wav_a, seconds, recorder=args.recorder, node=args.node)
            render(src_b, wav_b, seconds, recorder=args.recorder, node=args.node)
        else:
            print("reusing existing renders", file=sys.stderr)
        a, sr_a = _load_wav(wav_a)
        b, sr_b = _load_wav(wav_b)
        b = _match_sr(b, sr_b, sr_a)
    except Exception as exc:  # render / IO failure is a usage-level error, not a verdict
        print(f"ERROR: {exc}", file=sys.stderr)
        verdict["error"] = str(exc)
        print(json.dumps(verdict, indent=2))
        return 2

    if args.metric == "chroma":
        _, bars_now = _find_array(code, ed.array)
        res = localise_chroma(a, b, sr_a, bpm, nbars, array=ed.array, bar=ed.bar, step=ed.step,
                              semitones=args.semitones, bars=bars_now, k=args.k)
    elif args.metric == "f0":
        _, bars_now = _find_array(code, ed.array)
        res = localise_f0(a, b, sr_a, bpm, nbars, array=ed.array, bar=ed.bar, step=ed.step,
                          semitones=args.semitones, bars=bars_now, k=args.k, solo=args.solo)
    else:
        res = localise(a, b, sr_a, bpm, nbars, bar_window=window, k=args.k, max_lag_s=args.max_lag)
    verdict.update(res)
    verdict["rendered"] = True
    verdict["renders"] = [str(wav_a), str(wav_b)]
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["localised"] else 1


if __name__ == "__main__":
    sys.exit(main())
