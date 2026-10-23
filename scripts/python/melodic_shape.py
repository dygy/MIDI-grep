#!/usr/bin/env python3
"""The melodic voices' (lead + chords) NOTE SHAPE, measured from the melodic stem.

The transcription is quantised to a 16th grid and ``sustain`` fills every step a note sounds on, so the written lead
is a retrigger stream: on Regime CLT 1062 of 1126 consecutive written notes are one step apart, against 226 real
stem onsets (80 % of written onsets have no stem onset; VAGABUNDO 61 %). The stem plays a note and lets it ring.
Here the stem decides, per written trigger:

1. ATTACK  a trigger is kept only if a strong onset of the stem lands on it (``StemEvidence.claim_attacks``: the
           re-attack threshold is Otsu over the stem's own onset strengths, the claim tolerance is half a grid step).
2. HOLD    a kept note lasts until the next attack, the stem going quiet (the envelope decayed to the stem's own
           typical between-note trough) or the bar end; written ``c4@3`` (mini-notation elongation).
3. CHORDS  follow the LEAD's attacks (the stem has one onset stream; the harmony sounds with the note that attacks):
           a chord is kept on a step the lead attacks, and ends at the next lead attack.

No constant here is per-track: thresholds come from :class:`bass_shape.StemEvidence`, lengths from the grid.
Tokens are opaque (``c4``, ``[c3,e3]``), so the same planner shapes both voices. Pure functions, unit-testable
without audio.
"""
from __future__ import annotations

from dataclasses import dataclass

from bass_shape import StemEvidence


@dataclass
class Event:
    step: int          # global 16th-step index
    token: str         # ``c4`` or ``[c3,e3]``


def load_evidence(stem_path, *, strong_only: bool = False) -> StemEvidence:
    """The melodic stem's evidence. A polyphonic stem's soft notes are real notes (the weak onset class of the
    bass' Otsu split is the bass' ripple, not the melody's), so by default EVERY onset the detector finds counts as
    an attack; ``strong_only`` restores the bass rule (Otsu over the stem's own onset strengths)."""
    ev = StemEvidence.load(stem_path)
    if not strong_only:
        ev.attacks = ev.onsets
    return ev


def silent_after(ev: StemEvidence, t: float, horizon_s: float) -> float:
    """When the stem next drops below its own sounding level (Otsu over its frames' log RMS) after the attack at
    ``t``, capped at the horizon. A polyphonic stem's notes overlap and ring into each other, so unlike the bass
    (``end_after``: the envelope decaying to the typical between-note trough) a melodic note lasts until the stem
    itself is quiet or the next attack."""
    f0 = int(t / ev.hop_s)
    f1 = min(len(ev.rms), f0 + int(horizon_s / ev.hop_s) + 1)
    seg = ev.rms[f0:f1]
    if seg.size < 3:
        return t
    below = __import__("numpy").nonzero(seg[2:] < ev.level)[0]
    return (f0 + 2 + int(below[0])) * ev.hop_s if below.size else f1 * ev.hop_s


def parse_events(bars: list[str], quantize: int = 16) -> list[Event]:
    """Every non-rest token of the 1-step-per-slot bars as an Event at its global step (an ``@n`` hold advances n)."""
    out: list[Event] = []
    for b, bar in enumerate(bars):
        slot = 0
        for tok in bar.split():
            n = 1
            if "@" in tok:
                tok, _, k = tok.partition("@")
                n = max(1, int(k))
            if tok != "~":
                out.append(Event(b * quantize + slot, tok))
            slot += n
    return out


def attack_steps(events: list[Event], ev: StemEvidence, step_dur: float) -> set[int]:
    """The steps whose written trigger carries an audible stem attack (each strong stem onset is claimed by the ONE
    trigger nearest to it within half a step). A stem with no strong onsets claims every trigger."""
    flags = ev.claim_attacks([e.step * step_dur for e in events], step_dur / 2)
    return {e.step for e, f in zip(events, flags) if f}


def shape_voice(events: list[Event], keep: set[int], bounds: set[int], ev: StemEvidence, step_dur: float,
                nbars: int, quantize: int = 16) -> tuple[list[str], dict]:
    """Written bars of the events at ``keep`` steps, each held until the earliest of: the next step in ``bounds``,
    the stem going quiet, the bar end. Returns (bars, info)."""
    total = nbars * quantize
    bar_dur = step_dur * quantize
    kept = sorted((e for e in events if e.step in keep), key=lambda e: e.step)
    bset = sorted(bounds | {e.step for e in kept})
    held: dict[int, tuple[str, int]] = {}
    for e in kept:
        nxt = next((s for s in bset if s > e.step), total)
        bar_end = (e.step // quantize + 1) * quantize
        quiet = int(round((silent_after(ev, e.step * step_dur, bar_dur) - e.step * step_dur) / step_dur))
        held[e.step] = (e.token, max(1, min(quiet, nxt - e.step, bar_end - e.step)))
    bars = []
    for b in range(nbars):
        toks, s = [], 0
        while s < quantize:
            h = held.get(b * quantize + s)
            if h is None:
                toks.append("~")
                s += 1
            else:
                tok, ln = h
                toks.append(tok if ln == 1 else f"{tok}@{ln}")
                s += ln
        bars.append(" ".join(toks))
    lens = [ln for _, ln in held.values()]
    info = {"triggers_in": len(events), "notes": len(held),
            "median_len_steps": float(sorted(lens)[len(lens) // 2]) if lens else 0.0}
    return bars, info


def shape_melodic(lead_bars: list[str], chord_bars: list[str] | None, bpm: float, ev: StemEvidence,
                  quantize: int = 16) -> tuple[list[str], list[str] | None, dict]:
    """Lead (and chord) 1-step bars -> bars that retrigger only on stem attacks and hold between them."""
    nbars = len(lead_bars)
    step_dur = 60.0 / bpm * 4 / quantize
    lead_ev = parse_events(lead_bars, quantize)
    att = attack_steps(lead_ev, ev, step_dur)
    if not att:                                   # nothing claimed -> leave the voice as transcribed
        return lead_bars, chord_bars, {"lead": None, "chords": None}
    lead_out, lead_info = shape_voice(lead_ev, att, att, ev, step_dur, nbars, quantize)
    chord_out, chord_info = None, None
    if chord_bars:
        chord_out, chord_info = shape_voice(parse_events(chord_bars, quantize), att, att, ev, step_dur, nbars,
                                            quantize)
    return lead_out, chord_out, {"lead": lead_info, "chords": chord_info, "attacks": len(att),
                                 "stem_attacks": int(len(ev.attacks))}
