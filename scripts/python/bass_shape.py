#!/usr/bin/env python3
"""The bass NOTE SHAPE, measured from the bass stem (not from bass.mid fragments).

The original bass is one continuous 808 tone; the grid line written from the transcription is a row of short
separate hits (each re-triggering a <= 0.4 s grain). Four things are measured from ``bass.wav`` and written as
editable Strudel data; none is a per-track constant:

1. HOLD     a written note lasts until the stem's next re-attack, the stem going quiet, or the bar end
            (plan_notes, written ``c2@3``); un-re-attacked same-pitch repeats merge into the held note.
2. GLIDE    a pitch change that the stem's f0 shows as a continuous slide (no re-attack, voiced frames with
            intermediate pitches between the two notes) gets ``glide(seconds)`` = the measured duration of that
            transition; every other note gets 0 (:func:`measure_glides`).
3. LOOP     sample mode: the grain's own steady part is looped so a held note sustains. The loop is a whole number
            of the grain's measured periods between zero crossings, placed where the seam is most seamless
            (:func:`loop_points`).
4. DUCK     the kick pumps the bass: the dip of the stem's level after kick onsets, in excess of the dip at
            non-kick times, gives ``duckdepth/duckonset/duckattack``; no significant excess -> no ducking
            (:func:`measure_duck`).

Pure functions take plain arrays so they are unit-testable without audio.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NOTE_RE = re.compile(r"^([a-g]s?)(-?\d+)(?:@(\d+))?$")
_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
GLIDE_END_TOL_ST = 0.5      # a frame is "on" a note when its f0 is within half a semitone of it
DUCK_MIN_T = 0.01          # superdough schedules the duck at t - 0.01 s: a kick earlier than that fails the render
DUCK_MIN_DB = 1.0          # ~ the just-noticeable level difference: a smaller dip is statistically real but inaudible
DUCK_SIGMA = 1.645         # one-sided 95%: only a dip is hypothesised, so the one-sided normal quantile


def otsu(values: np.ndarray) -> float | None:
    """Threshold maximising between-class variance of a 1-D sample; None if it cannot split (<3 distinct)."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size < 3 or v[0] == v[-1]:
        return None
    best, thr = -1.0, None
    csum = np.cumsum(v)
    n = v.size
    total = csum[-1]
    for k in range(1, n):
        if v[k] == v[k - 1]:
            continue
        w0, w1 = k / n, (n - k) / n
        m0, m1 = csum[k - 1] / k, (total - csum[k - 1]) / (n - k)
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best:
            best, thr = var, 0.5 * (v[k - 1] + v[k])
    return thr


def reattack_threshold(strengths) -> float:
    """Onset strength at/above which an onset is a re-attack (Otsu over the log strengths of the stem's own
    onsets; the weak class are ripples of a sounding note). 0.0 = every onset counts."""
    s = np.asarray([x for x in strengths if x > 0], dtype=float)
    thr = otsu(np.log(s)) if s.size else None
    return 0.0 if thr is None else float(np.exp(thr))


def sounding_level(rms: np.ndarray) -> float:
    """RMS above which the stem is 'sounding' (Otsu over the log RMS of the stem's own frames)."""
    thr = otsu(np.log(np.asarray(rms, dtype=float) + 1e-9))
    return 0.0 if thr is None else float(np.exp(thr))


@dataclass
class Trigger:
    step: int            # global 16th-step index
    t: float             # onset time, s
    pitch: int           # MIDI
    end: float           # time the stem stops sounding after this onset (s)
    attack: bool = True  # the stem has an audible re-attack here


class StemEvidence:
    """What the stem itself says about its attacks and how long its notes ring (measured once per stem)."""

    SR, HOP = 22050, 256

    def __init__(self, y: np.ndarray):
        import librosa
        self._y = y
        self._chroma = None
        self.hop_s = self.HOP / self.SR
        self.rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=self.HOP)[0]
        self.ostr = librosa.onset.onset_strength(y=y, sr=self.SR, hop_length=self.HOP)
        self.level = sounding_level(self.rms)
        onset_times = np.asarray(librosa.onset.onset_detect(y=y, sr=self.SR, hop_length=self.HOP, units="time",
                                                            backtrack=True, onset_envelope=self.ostr), dtype=float)
        self.onsets = onset_times                      # every detected onset (the weak class included)
        strengths = np.array([self.strength_at(float(t)) for t in onset_times])
        self.thr = reattack_threshold(strengths)
        self.attacks = onset_times[strengths >= self.thr] if onset_times.size else onset_times
        # ATTACK-time onsets. ``backtrack=True`` puts each onset at the previous energy minimum (25-60 ms before the
        # attack, a different distance per stem and note) and the onset-strength peak lags it by a window (~30 ms
        # after): neither can be compared with a written step. The audible attack is the steepest rise of the
        # fine-resolution log level between the two.
        peak_frames = librosa.onset.onset_detect(y=y, sr=self.SR, hop_length=self.HOP, units="frames",
                                                 backtrack=False, onset_envelope=self.ostr)
        back_frames = librosa.onset.onset_backtrack(peak_frames, self.ostr) if len(peak_frames) else peak_frames
        fine_hop = max(self.HOP // 4, 1)
        rise = np.gradient(np.log(librosa.feature.rms(y=y, frame_length=2 * self.HOP, hop_length=fine_hop)[0] + 1e-6))
        k = self.HOP // fine_hop
        times, pk_str = [], []
        for pf, bf in zip(peak_frames, back_frames):
            i0, i1 = int(bf) * k, min(len(rise), int(pf) * k + k)
            if i1 <= i0:
                i0, i1 = max(0, int(pf) * k - k), min(len(rise), int(pf) * k + k)
            times.append((i0 + int(np.argmax(rise[i0:i1]))) * fine_hop / self.SR)
            pk_str.append(float(self.ostr[max(0, int(pf) - 2):int(pf) + 3].max()))
        self.peak_onsets = np.array(times, dtype=float)
        self.peak_attacks = self.peak_onsets[np.array(pk_str) >= self.thr] if len(times) else self.peak_onsets
        self.fall = self._typical_fall(onset_times)

    @classmethod
    def load(cls, stem_path: Path) -> "StemEvidence":
        import librosa
        y, _ = librosa.load(str(stem_path), sr=cls.SR, mono=True)
        return cls(y)

    def claim_attacks(self, times: list[float], tol_s: float, peak: bool = False) -> list[bool]:
        """Which triggers carry an audible attack: every strong stem onset is claimed by the ONE trigger nearest
        to it (within ``tol_s``); the rest are transcription echoes. No strong onsets -> every trigger counts.
        ``peak``: the onsets are the envelope peaks (``peak_attacks``), not the backtracked ``attacks``."""
        attacks = self.peak_attacks if peak else self.attacks
        flags = [False] * len(times)
        if attacks.size == 0:
            return [True] * len(times)
        ts = np.asarray(times, dtype=float)
        for a in attacks:
            if ts.size == 0:
                break
            j = int(np.argmin(np.abs(ts - a)))
            if abs(ts[j] - a) <= tol_s:
                flags[j] = True
        return flags

    def chroma_over(self, t0: float, t1: float, lo: int, hi: int) -> np.ndarray | None:
        """Mean 12-class chroma of the stem between ``t0`` and ``t1`` (CQT spanning the register [lo, hi]); None when
        the span holds no frame."""
        import librosa
        if self._chroma is None:
            base = 12 * (lo // 12)
            self._chroma = librosa.feature.chroma_cqt(
                y=self._y, sr=self.SR, hop_length=512, fmin=float(librosa.midi_to_hz(base)), bins_per_octave=36,
                n_octaves=max(2, (hi - base) // 12 + 1))
        fr = np.arange(self._chroma.shape[1]) * 512 / self.SR
        sel = (fr >= t0) & (fr <= t1)
        return self._chroma[:, sel].mean(axis=1) if sel.any() else None

    def unclaimed_attacks(self, times: list[float], tol_s: float) -> list[float]:
        """Peak-time strong attacks that NO trigger is within ``tol_s`` of (the transcription missed them)."""
        ts = np.asarray(times, dtype=float)
        if ts.size == 0:
            return [float(a) for a in self.peak_attacks]
        return [float(a) for a in self.peak_attacks if float(np.min(np.abs(ts - a))) > tol_s]

    def strength_at(self, t: float) -> float:
        f = int(round(t / self.hop_s))
        a, b = max(0, f - 2), min(len(self.ostr), f + 7)
        return float(self.ostr[a:b].max()) if b > a else 0.0

    def _typical_fall(self, onsets: np.ndarray) -> float:
        """Median trough/peak of the stem's envelope between consecutive onsets = how far a note must decay to
        count as ended."""
        ratios = []
        for a, b in zip(onsets[:-1], onsets[1:]):
            i0, i1 = int(a / self.hop_s), int(b / self.hop_s)
            if i1 - i0 < 3:
                continue
            seg = self.rms[i0:i1]
            pk = int(np.argmax(seg[: max(2, len(seg) // 2)]))
            if seg[pk] > 0:
                ratios.append(float(seg[pk:].min() / seg[pk]))
        return float(np.median(ratios)) if ratios else 0.0

    def end_after(self, t: float, horizon_s: float) -> float:
        """When the note struck at ``t`` stops sounding: its envelope decayed to the stem's typical between-note
        trough (``fall`` x its peak) or the stem is below its sounding level. Capped at the horizon."""
        f0 = int(t / self.hop_s)
        f1 = min(len(self.rms), f0 + int(horizon_s / self.hop_s) + 1)
        seg = self.rms[f0:f1]
        if seg.size < 3:
            return t
        pk = int(np.argmax(seg[:7]))
        floor = max(self.level, float(seg[pk]) * self.fall)
        below = np.nonzero(seg[pk + 1:] < floor)[0]
        return (f0 + pk + 1 + int(below[0])) * self.hop_s if below.size else (f1 * self.hop_s)


@dataclass
class Note:
    step: int
    pitch: int
    length: int                 # steps
    glide: float = 0.0          # seconds (0 = no glide)
    t: float = 0.0
    attack: bool = True
    end_t: float = 0.0          # when the stem stops sounding (latest of the merged triggers)
    merged_to: int = 0          # last step of the triggers merged into this note
    proven: bool = False        # a legato pitch change the stem's f0 shows (step or slide)


def plan_notes(trigs: list[Trigger], step_dur: float, quantize: int, total_steps: int) -> list[Note]:
    """Merge continuations; a note is sized by size_notes.

    A trigger WITHOUT an audible stem attack while the previous note of the same bar is still sounding: same pitch -> it extends that note; different pitch -> a LEGATO note (attack False), kept: the
    transcription's pitch changes are real even without a new stem attack (dropping them cost 0.02 overall /
    0.03 section-aware on VAGABUNDO sample mode); the stem's f0 decides only whether it glides."""
    trigs = sorted(trigs, key=lambda x: x.step)
    groups: list[list[Trigger]] = []
    for tr in trigs:
        g = groups[-1] if groups else None
        sounding = (g is not None and g[-1].step // quantize == tr.step // quantize
                    and max(x.end for x in g) >= tr.t)
        if sounding and not tr.attack and g[-1].pitch == tr.pitch:
            g.append(tr)
            continue
        groups.append([tr])
    out = [Note(g[0].step, g[0].pitch, 1, 0.0, g[0].t, g[0].attack, max(x.end for x in g), g[-1].step) for g in groups]
    return size_notes(out, step_dur, quantize, total_steps)


def size_notes(notes: list[Note], step_dur: float, quantize: int, total_steps: int) -> list[Note]:
    """A hold ends at the earliest of: the stem going quiet, the next written note, the bar end; never shorter
    than the span of triggers merged into it."""
    for i, n in enumerate(notes):
        bar_end = (n.step // quantize + 1) * quantize
        nxt = notes[i + 1].step if i + 1 < len(notes) else total_steps
        length = max(int(round((n.end_t - n.t) / step_dur)), n.merged_to - n.step + 1)
        n.length = max(1, min(length, nxt - n.step, bar_end - n.step))
    return notes


def measure_glides(notes: list[Note], midi: np.ndarray, times: np.ndarray, active: np.ndarray,
                   step_dur: float) -> int:
    """Examine every legato candidate (no stem attack, pitch differs from the previous, still-sounding note) in
    the stem's f0.

    Around the new onset the f0 leaves the previous pitch ``a`` (last frame within GLIDE_END_TOL_ST of it) and
    arrives at the new ``b`` (first frame within GLIDE_END_TOL_ST of it): that PROVES the pitch change
    (``Note.proven``, the note stays). If voiced frames with intermediate pitches lie between departure and
    arrival it is a SLIDE and ``Note.glide`` = arrival - departure seconds (>= one f0 hop); a jump within one
    hop is a step (glide 0). Returns the number of slides set."""
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.0
    n_set = 0
    for prev, cur in zip(notes[:-1], notes[1:]):
        if cur.attack or cur.pitch == prev.pitch:
            continue
        if cur.step > prev.step + prev.length:          # a rest between them: a new phrase, not a slide
            continue
        a, b = float(prev.pitch), float(cur.pitch)
        lo_t, hi_t = cur.t - 2 * step_dur, cur.t + 2 * step_dur
        sel = np.nonzero((times >= lo_t) & (times <= hi_t))[0]
        if sel.size < 3:
            continue
        m = np.where(active[sel], midi[sel], np.nan)
        on_a = np.nonzero(np.abs(m - a) <= GLIDE_END_TOL_ST)[0]
        on_b = np.nonzero(np.abs(m - b) <= GLIDE_END_TOL_ST)[0]
        if on_a.size == 0 or on_b.size == 0 or not (on_b > on_a[0]).any():
            continue
        arr = int(on_b[on_b > on_a[0]][0])
        dep = int(on_a[on_a < arr][-1])
        cur.proven = True
        between = m[dep + 1:arr]
        if between.size == 0 or np.isnan(between).any():
            continue
        lo, hi = min(a, b), max(a, b)
        if not ((between > lo + GLIDE_END_TOL_ST) & (between < hi - GLIDE_END_TOL_ST)).any():
            continue
        cur.glide = round(max(float(times[sel[arr]] - times[sel[dep]]), hop), 3)
        n_set += 1
    return n_set


def format_bars(notes: list[Note], nbars: int, quantize: int, name, glide: bool = False) -> list[str]:
    """One mini-notation string per bar; a hold of n>1 steps is ``c2@n`` and absorbs its n-1 slots. With
    ``glide`` the PARALLEL glide-seconds string (same structure: ``0.06@2 ~ 0 ...``; rests stay ``~``)."""
    by_step = {n.step: n for n in notes}
    bars = []
    for b in range(nbars):
        toks, s = [], 0
        while s < quantize:
            n = by_step.get(b * quantize + s)
            if n is None:
                toks.append("~")
                s += 1
            else:
                ln = max(1, min(n.length, quantize - s))
                tok = f"{n.glide:g}" if glide else name(n.pitch)
                toks.append(tok if ln == 1 else f"{tok}@{ln}")
                s += ln
        bars.append(" ".join(toks))
    return bars


def parse_bar(bar: str) -> list[tuple[int, int, int]]:
    """[(slot, midi, length_steps)] of one written bar (rests advance one slot; ``@n`` advances n)."""
    out, slot = [], 0
    for tok in bar.split():
        if tok == "~":
            slot += 1
            continue
        m = NOTE_RE.match(tok)
        if not m or m.group(1) not in _NAMES:
            slot += 1
            continue
        n = int(m.group(3) or 1)
        out.append((slot, (int(m.group(2)) + 1) * 12 + _NAMES.index(m.group(1)), n))
        slot += n
    return out


def stem_pitch_after(ev: StemEvidence, register, t: float, step_dur: float, lo: int, hi: int,
                     near: int | None) -> int | None:
    """MIDI pitch of the stem's note struck at ``t`` where the transcription has none.

    Pitch CLASS: the f0 track's, when the stem's own chroma over the next two steps supports it (that class lies in
    the strong class of the 12 chroma values, split by Otsu) - pyin locks onto a harmonic on some stems, the chroma
    does not name the octave; otherwise the chroma's strongest class. OCTAVE: the one nearest the f0 (else nearest
    ``near``, the previous written pitch), inside the stem's register window [lo, hi]."""
    v = ev.chroma_over(t + 0.03, t + 2 * step_dur, lo, hi)
    f = register.local(t, t + step_dur)
    if v is None:
        return None if f is None else _fold(int(round(f)), lo, hi)
    pc = int(np.argmax(v))
    thr = otsu(v)
    if f is not None and thr is not None and v[int(round(f)) % 12] >= thr:
        pc = int(round(f)) % 12
    ref = f if f is not None else near
    cands = [m for m in range(lo, hi + 1) if m % 12 == pc]
    if not cands:
        return None
    return min(cands, key=lambda m: abs(m - ref)) if ref is not None else cands[len(cands) // 2]


def _fold(p: int, lo: int, hi: int) -> int | None:
    while p > hi:
        p -= 12
    while p < lo:
        p += 12
    return p if lo <= p <= hi else None


def add_missed_attacks(trigs: list[Trigger], ev: StemEvidence, register, step_dur: float, total_steps: int) -> int:
    """The transcription misses some of the stem's strong attacks (no Basic Pitch note within a step). Where the
    stem re-attacks and no trigger is within half a step, write an attack on the step nearest to it, its pitch
    from the stem's f0 track; a step already holding a trigger is never doubled. Returns the number added."""
    try:
        lo, hi = register.window()
    except ValueError:
        return 0
    taken = {tr.step for tr in trigs}
    n = 0
    for a in ev.unclaimed_attacks([tr.t for tr in trigs], step_dur / 2):
        step = int(round(a / step_dur))
        if step in taken or not 0 <= step < total_steps:
            continue
        prev = [tr.pitch for tr in sorted(trigs, key=lambda x: x.step) if tr.step < step]
        pitch = stem_pitch_after(ev, register, a, step_dur, lo, hi, prev[-1] if prev else None)
        if pitch is None:
            continue
        t = step * step_dur
        trigs.append(Trigger(step=step, t=t, pitch=pitch, end=ev.end_after(t, step_dur * 16), attack=True))
        taken.add(step)
        n += 1
    return n


def shape_bars(bars: list[str], bpm: float, ev: StemEvidence, register, quantize: int = 16
               ) -> tuple[list[Note], dict]:
    """Written 1-step bars -> held notes with glides. ``register`` is a bass_register.Register (f0 track)."""
    bar_dur = 60.0 / bpm * 4
    step_dur = bar_dur / quantize
    steps, pitches = [], []
    for b, bar in enumerate(bars):
        for slot, p, _n in parse_bar(bar):
            steps.append(b * quantize + slot)
            pitches.append(p)
    times = [s * step_dur for s in steps]
    # a written onset sits on the 16th grid, the stem's attack (at its envelope PEAK, not the backtracked minimum)
    # within +-half a step of it
    flags = ev.claim_attacks(times, step_dur / 2, peak=True)
    trigs = [Trigger(step=s, t=t, pitch=p, end=ev.end_after(t, bar_dur), attack=a)
             for s, t, p, a in zip(steps, times, pitches, flags)]
    total = len(bars) * quantize
    added = add_missed_attacks(trigs, ev, register, step_dur, total) if register is not None else 0
    notes = plan_notes(trigs, step_dur, quantize, total)
    n_gl = measure_glides(notes, register.midi, register.times, register.active, step_dur) if register is not None else 0
    lens = [n.length * step_dur for n in notes]
    info = {"triggers_in": len(trigs), "notes": len(notes), "median_len_s": float(np.median(lens)) if lens else 0.0,
            "median_len_steps": float(np.median([n.length for n in notes])) if notes else 0.0,
            "merged": len(trigs) - len(notes), "attacks_added": added, "glides": n_gl, "glide_s_median": float(np.median([n.glide for n in notes if n.glide])) if n_gl else 0.0}
    return notes, info


# ---- sample loop points ------------------------------------------------------------------------------------

def loop_points(y: np.ndarray, sr: int, f0_hz: float) -> tuple[float, float] | None:
    """(loopBegin, loopEnd) as FRACTIONS of the grain for a sustaining loop, or None when the grain has no steady
    part long enough (two periods).

    Steady part = from one period after the envelope peak to the last frame still within -6 dB (half amplitude,
    the usual 'sustain' convention) of that peak. Inside it the loop is ``k`` whole periods (the most that fit),
    both ends at upward zero crossings, positioned where the seam is smoothest: the cost of a begin position is the
    normalised squared difference between the period before the loop end and the period before its begin."""
    y = np.asarray(y, dtype=float)
    P = max(int(round(sr / f0_hz)), 2)
    hop = max(P // 4, 1)
    nfr = (len(y) - 2 * P) // hop
    if nfr < 3:
        return None
    env = np.array([np.sqrt(np.mean(y[i * hop:i * hop + 2 * P] ** 2)) for i in range(nfr)])
    ip = int(np.argmax(env))
    if env[ip] <= 0:
        return None
    # the attack transient can out-shout the sustain: take the sustain level from the loudest frame after it has
    # passed (two analysis windows later), so "-6 dB" is measured against the tone, not the click
    settle = int(np.ceil(2 * P / hop))
    if ip + settle < nfr:
        ip = ip + settle + int(np.argmax(env[ip + settle:]))
    last = ip
    while last + 1 < nfr and env[last + 1] >= 0.5 * env[ip]:
        last += 1
    start = ip * hop + P
    stop = last * hop + 2 * P
    if (stop - start) // P < 2:
        return None
    zc = [i for i in range(max(start, 1), min(stop, len(y) - 1)) if y[i - 1] < 0 <= y[i]]
    best = None
    for b in zc:
        k = (stop - b) // P                       # the most whole periods that fit from this start
        if k < 2:
            continue
        e0 = b + k * P
        e = min((z for z in zc if z >= e0 - P // 2), key=lambda z: abs(z - e0), default=None)
        if e is None or e - b < 2 * P or e > stop + P // 2 or b - P < 0:
            continue
        a, c = y[b - P:b], y[e - P:e]
        cost = float(np.sum((a - c) ** 2) / (np.sum(a ** 2) + np.sum(c ** 2) + 1e-12))
        if best is None or cost < best[0] or (cost == best[0] and e - b > best[2] - best[1]):
            best = (cost, b, e)
    if best is None:
        return None
    return (round(best[1] / len(y), 5), round(best[2] / len(y), 5))


# ---- sidechain ----------------------------------------------------------------------------------------------

@dataclass
class Duck:
    depth_db: float            # excess dip of the stem level after kicks (dB, positive = ducks)
    onset_s: float             # kick -> deepest point
    attack_s: float            # deepest point -> recovered (linear-in-dB ramp through the half-depth point)
    depth: float               # Strudel duckdepth (gain = 1 - sqrt(depth))
    sigma: float               # excess dip in standard errors
    kicks: int = 0
    curve: list = field(default_factory=list)


def _curve_stats(env_db: np.ndarray, fs: float, times: np.ndarray, pre: int, post: int, level_db: float):
    rows = []
    for t in times:
        i = int(t * fs)
        if i - pre < 0 or i + post >= len(env_db):
            continue
        seg = env_db[i - pre:i + post]
        base = float(np.mean(seg[:pre]))
        if base < level_db:                 # only where the bass was sounding before the event
            continue
        rows.append(seg - base)
    return np.array(rows)


def measure_duck(rms: np.ndarray, hop_s: float, kick_times, *, window_s: float | None = None,
                 seed: int = 0, diag: dict | None = None) -> Duck | None:
    """Kick-triggered dip of the bass stem's level.

    ``rms``/``hop_s``: stem RMS envelope. For every kick where the bass was sounding (pre-kick level above the
    stem's own Otsu sounding level) the dB curve relative to its pre-kick mean is taken; the SAME is taken at
    times away from any kick (the bass' own decay/phrasing). The excess = median kick curve - median control
    curve; it is ducking only when its minimum is below zero by DUCK_SIGMA standard errors (of the medians).
    The window is one kick inter-onset interval (median)."""
    kick_times = np.sort(np.asarray(kick_times, dtype=float))
    if kick_times.size < 8:
        return None
    ioi = float(np.median(np.diff(kick_times))) if kick_times.size > 1 else 0.5
    window_s = window_s or min(ioi, 0.5)
    fs = 1.0 / hop_s
    db = 20 * np.log10(np.asarray(rms, dtype=float) + 1e-9)
    level_db = 20 * np.log10(sounding_level(rms) + 1e-9)
    pre, post = max(int(0.06 * fs), 2), max(int(window_s * fs), 4)
    k = _curve_stats(db, fs, kick_times, pre, post, level_db)
    rng = np.random.default_rng(seed)
    dur = len(db) * hop_s
    cand = rng.uniform(window_s, dur - window_s, 8000)
    cand = np.array([t for t in cand if np.min(np.abs(kick_times - t)) > window_s])
    c = _curve_stats(db, fs, cand, pre, post, level_db)
    if len(k) < 8 or len(c) < 8:
        return None

    def se_med(a):
        return 1.2533 * np.std(a, axis=0) / np.sqrt(len(a))

    ex = np.median(k, axis=0) - np.median(c, axis=0)
    se = np.sqrt(se_med(k) ** 2 + se_med(c) ** 2)
    post_ex = ex[pre:]
    post_se = se[pre:]
    i = int(np.argmin(post_ex))
    depth_db = float(-post_ex[i])
    sigma = float(depth_db / post_se[i]) if post_se[i] > 0 else 0.0
    if diag is not None:
        diag.update(depth_db=round(depth_db, 2), sigma=round(sigma, 2), kicks=len(k), controls=len(c),
                    at_s=round(i * hop_s, 3))
    if depth_db < DUCK_MIN_DB or sigma < DUCK_SIGMA:
        return None
    half = -depth_db / 2
    rec = np.nonzero(post_ex[i:] >= half)[0]
    t_half = (rec[0] if rec.size else len(post_ex) - i) * hop_s
    onset = i * hop_s
    attack = 2.0 * t_half
    g = 10 ** (-depth_db / 20)
    return Duck(depth_db=round(depth_db, 2), onset_s=round(onset, 3), attack_s=round(attack, 3),
                depth=round((1 - g) ** 2, 3), sigma=round(sigma, 1), kicks=len(k),
                curve=[round(float(x), 2) for x in post_ex])


# ---- sample-instrument loop table ---------------------------------------------------------------------------

def _label_midi(label: str) -> int | None:
    m = re.fullmatch(r"([a-g])(#?)(-?\d+)", label.strip().lower())
    if not m:
        return None
    base = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[m.group(1)] + (1 if m.group(2) else 0)
    return (int(m.group(3)) + 1) * 12 + base


def sample_loops(samples_json: Path, instrument: str, pitches) -> tuple[dict[int, tuple[float, float]], dict]:
    """{midi: (loopBegin, loopEnd)} for every written ``pitches`` entry whose playing grain is loopable.

    Strudel plays a note with the instrument key NEAREST to it (first key wins a tie, in manifest order), so the
    table is keyed by the written pitch and looked up from the same nearest key. Keys whose grain has no steady
    part (:func:`loop_points`) leave their pitches out: those notes play the grain once, as before."""
    import soundfile as sf
    data = __import__("json").loads(Path(samples_json).read_text())
    inst = data.get(instrument) or {}
    loops, durs = {}, {}
    for key, files in inst.items():
        if str(key).startswith("_"):
            continue
        midi = _label_midi(str(key))
        f = files[0] if isinstance(files, list) else files
        path = Path(samples_json).parent / f
        if midi is None or not path.exists():
            continue
        y, sr = sf.read(str(path))
        y = y if y.ndim == 1 else y.mean(axis=1)
        durs[key] = len(y) / sr
        lp = loop_points(y, sr, 440.0 * 2 ** ((midi - 69) / 12))
        if lp:
            loops[key] = lp
    keys = [k for k in inst if not str(k).startswith("_") and _label_midi(str(k)) is not None]
    table: dict[int, tuple[float, float]] = {}
    for p in sorted(set(pitches)):
        best = None
        for k in keys:
            if best is None or abs(_label_midi(str(k)) - p) < abs(_label_midi(str(best)) - p):
                best = k
        if best in loops:
            table[p] = loops[best]
    stats = {"samples": len(durs), "looped": len(loops),
             "loop_len_s": [round((e - b) * durs[k], 3) for k, (b, e) in loops.items()],
             "loop_start_s": [round(b * durs[k], 3) for k, (b, e) in loops.items()],
             "pitches": len(set(pitches)), "pitches_looped": len(table),
             "pitches_unlooped": sorted(set(pitches) - set(table))}      # these play their (nearest key's) grain once
    return table, stats


def stem_rms_fine(stem: Path, sr: int = 22050, hop: int = 110, frame: int = 441) -> tuple[np.ndarray, float]:
    """(RMS envelope, hop seconds) at ~5 ms resolution, the scale of a kick's dip."""
    import librosa
    y, _ = librosa.load(str(stem), sr=sr, mono=True)
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0], hop / sr
