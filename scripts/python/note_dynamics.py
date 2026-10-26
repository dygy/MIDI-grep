#!/usr/bin/env python3
"""note_dynamics.py - the level each written lead/chord note must play at, measured from the melodic stem.

Why: the sampled lead and chords play every written note at the voice's one gain (times a per-beat envelope), but
the stem's notes differ in level, and a pitch's grain differs in level from its neighbours' (the instrument is cut
from the stem, one grain per transcribed note). Measured on the real renders, a per-note amplitude oracle (our own
melodic audio rescaled per written note to the stem's level over that note) lifted VAGABUNDO overall 0.867 -> 0.873.

So per onset (every voice's note that starts together is one group - the stem is one signal) the stem's RMS over the
note's own window is divided by the RMS the written notes would play at there:

    played = sqrt( sum_voice ( voice gain * grain_rms(pitch, window) * envelope(window) )^2 )
    ratio  = stem_rms(window) / played

and written as a parallel ``.velocity()`` pattern (velocity MULTIPLIES gain in superdough: the voice keeps ONE
``.gain()``, the calibrators' levers and the per-beat envelope compose with it untouched). Ratios are divided by their
median (a typical note stays 1, so cal-lead / melodic_solo still own the voice's overall level) and clamped to the
median +- 3 robust sigmas in the log domain (one bleed spike or a near-silent window cannot write an absurd
velocity). A group whose stem or played level is unreadable (silent) is written 1.

Pure functions (numpy only): the generator passes the stem samples and the grain / envelope lookups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

NOTE_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
_NOTE = re.compile(r"([a-g]s?)(-?\d+)")


@dataclass
class Slot:
    bar: int
    index: int          # token index inside the bar
    start: float        # fraction of the bar
    dur: float          # fraction of the bar
    pitches: list[int]
    weight: str         # "" or "@n" (kept so the velocity pattern has the note pattern's exact shape)


def _midi(name: str, octv: str) -> int:
    return (int(octv) + 1) * 12 + NOTE_NAMES.index(name)


def parse_slots(bars: list[str]) -> Optional[list[list[Optional[Slot]]]]:
    """Per bar, per token: a Slot for a note/chord token, None for a rest. None overall when any token is not one of
    ``~``, ``c4``, ``c4@2``, ``[c3,e3]`` (the shapes the generator writes) - the velocity pattern could then not mirror it."""
    out: list[list[Optional[Slot]]] = []
    for b, bar in enumerate(bars):
        ws, parsed = [], []
        for tok in bar.split():
            m = re.fullmatch(r"(.+?)(@\d+(?:\.\d+)?)?", tok)
            body, w = m.group(1), m.group(2) or ""
            ws.append(float(w[1:]) if w else 1.0)
            if body == "~":
                parsed.append((None, w))
                continue
            inner = body[1:-1] if body.startswith("[") and body.endswith("]") else body
            names = inner.split(",")
            if not all(_NOTE.fullmatch(n) for n in names):
                return None
            parsed.append(([_midi(*_NOTE.fullmatch(n).groups()) for n in names], w))
        tot = sum(ws) or 1.0
        pos, row = 0.0, []
        for i, ((ps, w), wt) in enumerate(zip(parsed, ws)):
            row.append(Slot(b, i, pos / tot, wt / tot, ps, w) if ps else None)
            pos += wt
        out.append(row)
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


class GrainLevels:
    """RMS of the sample Strudel plays for a written pitch: the instrument key NEAREST to it (first key wins a tie, the
    rule of ``bass_shape.sample_loops``), played from its start at speed 2^((pitch-key)/12)."""

    def __init__(self, samples_json: Path, instrument: str):
        import json
        import soundfile as sf
        from bass_shape import _label_midi
        data = json.loads(Path(samples_json).read_text())
        self.keys: list[tuple[int, np.ndarray, int]] = []
        for key, files in (data.get(instrument) or {}).items():
            midi = _label_midi(str(key))
            f = files[0] if isinstance(files, list) else files
            path = Path(samples_json).parent / f
            if midi is None or not path.exists():
                continue
            y, sr = sf.read(str(path))
            self.keys.append((midi, y if y.ndim == 1 else y.mean(axis=1), sr))

    def _nearest(self, pitch: int):
        best = None
        for k in self.keys:
            if best is None or abs(k[0] - pitch) < abs(best[0] - pitch):
                best = k
        return best

    def length_s(self, pitch: int) -> float:
        k = self._nearest(pitch)
        return len(k[1]) / k[2] / 2 ** ((pitch - k[0]) / 12) if k else 0.0

    def rms(self, pitch: int, window_s: float) -> float:
        """RMS over the first ``window_s`` of OUTPUT time of the note (the sample's own window is ``speed`` times longer)."""
        k = self._nearest(pitch)
        if not k:
            return 0.0
        midi, y, sr = k
        speed = 2 ** ((pitch - midi) / 12)
        return _rms(y[: max(1, int(window_s * speed * sr))])


def robust_clamp(v: np.ndarray, k: float = 3.0) -> np.ndarray:
    """v with its log clamped to median +- k robust sigmas (1.4826 MAD) of the log values."""
    lv = np.log(v)
    med = float(np.median(lv))
    sig = 1.4826 * float(np.median(np.abs(lv - med)))
    if sig <= 0.0:      # more than half the values equal (MAD 0): the mean absolute deviation still sees the rest
        sig = 1.4826 * float(np.mean(np.abs(lv - med)))
    return np.exp(np.clip(lv, med - k * sig, med + k * sig))


def note_velocities(stem: np.ndarray, sr: int, bpm: float, voices: dict[str, tuple[list[str], float]],
                    grain: Callable[[int, float], float], grain_len: Callable[[int], float],
                    env_mean: Callable[[float, float], float]) -> Optional[dict[str, list[str]]]:
    """{voice: per-bar velocity strings mirroring that voice's bars} or None when no voice could be parsed/measured.

    ``voices`` maps a name to (bars, played gain of that voice); ``grain(pitch, window_s)`` is the RMS the instrument
    plays for the pitch over the window, ``grain_len(pitch)`` the length (s) of that sample, ``env_mean(t0, t1)`` the
    mean per-beat envelope (incl. its correction) over the span. Bars are one cycle of ``60/bpm*4`` s."""
    bar_s = 60.0 / bpm * 4
    parsed = {n: parse_slots(b) for n, (b, _) in voices.items() if b}
    parsed = {n: p for n, p in parsed.items() if p is not None}
    if not parsed:
        return None
    groups: dict[float, list[tuple[str, Slot]]] = {}
    for n, rows in parsed.items():
        for row in rows:
            for s in row:
                if s is not None:
                    groups.setdefault(round((s.bar + s.start) * bar_s, 4), []).append((n, s))
    ratio: dict[float, float] = {}
    for t0, members in groups.items():
        w = min(s.dur * bar_s for _, s in members)
        w = min([w] + [grain_len(p) for _, s in members for p in s.pitches if grain_len(p) > 0])
        a = int(t0 * sr)
        seg = stem[a: a + max(1, int(w * sr))]
        e = env_mean(t0, t0 + w)
        played = np.sqrt(sum((voices[n][1] * grain(p, w) * e) ** 2 for n, s in members for p in s.pitches))
        sl = _rms(seg)
        if sl > 1e-6 and played > 1e-9:
            ratio[t0] = sl / played
    if len(ratio) < 4:
        return None
    ts = sorted(ratio)
    v = np.array([ratio[t] for t in ts])
    v = robust_clamp(v / np.median(v))
    vel = {t: float(x) for t, x in zip(ts, v)}
    out: dict[str, list[str]] = {}
    for n, rows in parsed.items():
        bars = []
        for row in rows:
            toks = []
            for s in row:
                if s is None:
                    toks.append("~")
                else:
                    toks.append(f"{vel.get(round((s.bar + s.start) * bar_s, 4), 1.0):.3g}{s.weight}")
            bars.append(" ".join(toks))
        out[n] = bars
    return out
