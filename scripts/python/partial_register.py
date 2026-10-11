#!/usr/bin/env python3
"""Dominant-partial register of a melodic/vocal stem relative to its transcribed notes.

Why: a synth voice that plays the right PITCHES can still put its energy in the wrong BAND. A
sustained palette sound (``gm_voice_oohs``, a saw lead) concentrates energy at the fundamental,
while the original singer / lead concentrates it at a formant or harmonic 2-4x higher. For a
415 Hz vocal that is the difference between low_mid (250-500 Hz) and mid (500-2000 Hz), and no
gain or filter lever can move it (bug #11: stem melodic 15/64 low_mid/mid vs rendered 54/33).

``dominant_partial_log2(stem, midi, lo, hi)`` = energy-weighted mean of
log2(strongest partial in 100-2500 Hz during each note / that note's played fundamental).
0 = the fundamental dominates, 1 = the octave, 2 = two octaves up. The same statistic measured on a
chromatic SWEEP of a palette sound (``probe_sound_register.py`` -> ``data/partial_reference.json``,
track-independent, per note) gives that sound's reference;
``octave_shift`` = round(original - reference), clamped to 0..MAX_OCTAVE_SHIFT, i.e. how many
octaves the synth line must move up so its fundamental sits where the original's energy sits
(pitch class is preserved, so chroma is unaffected; the notes stay editable data).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "partial_reference.json"
MAX_OCTAVE_SHIFT = 2          # beyond two octaves the line stops being the same melody
MIN_NOTE_S = 0.12             # shorter notes do not hold a resolvable partial (STFT window ~0.19 s)
SEARCH_HZ = (100.0, 2500.0)   # partial search range: below = bass bleed, above = air
SR, NFFT, HOP = 22050, 4096, 512


def _fold(pitch: int, lo: int, hi: int) -> int:
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def _spans_log2(y, spans):
    """[(t0, t1, midi_pitch)] -> [(midi_pitch, log2(strongest partial / fundamental), weight)] on signal ``y``
    (SR Hz). The ONE implementation behind both the stem analysis and the sweep probe."""
    import librosa
    import numpy as np

    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    band = (f >= SEARCH_HZ[0]) & (f < SEARCH_HZ[1])
    out = []
    for t0, t1, pitch in spans:
        a, b = int(t0 * SR / HOP), int(t1 * SR / HOP)
        if b <= a or a >= S.shape[1]:
            continue
        spec = S[:, a:b].mean(1)
        if spec[band].sum() <= 0:
            continue
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        out.append((pitch, math.log2(f[band][int(np.argmax(spec[band]))] / f0), float(spec[band].sum())))
    return out


def dominant_partial_log2(audio_path: Path | str, midi_path: Path | str, *, lo: int = 0, hi: int = 127,
                          max_s: float = 135.0, detail: bool = False):
    """See module docstring. ``lo``/``hi`` fold the MIDI pitches exactly as the generator does, so the
    fundamental measured against is the one the synth line will actually play. None if unmeasurable.
    ``detail=True`` returns ``(stat, {played_pitch: energy_weight})`` for per-note reference lookup."""
    import librosa
    import pretty_midi

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True, duration=max_s)
    if y.size < SR:
        return (None, {}) if detail else None
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    spans = [(n.start, n.end, _fold(n.pitch, lo, hi)) for inst in pm.instruments for n in inst.notes
             if n.end - n.start >= MIN_NOTE_S and n.end <= max_s]
    rows = _spans_log2(y, spans)
    if not rows:
        return (None, {}) if detail else None
    wts = [w for _, _, w in rows]
    stat = sum(l * w for _, l, w in rows) / sum(wts)
    if not detail:
        return float(stat)
    pw: dict[int, float] = {}
    for pitch, _, w in rows:
        pw[pitch] = pw.get(pitch, 0.0) + w
    return float(stat), pw


def cached_register(cache_path: Path, name: str, stem: Path, midi: Path, lo: int, hi: int):
    """``dominant_partial_log2(detail=True)`` memoised in ``cache_path`` (JSON beside the sample pack),
    keyed by stem+midi mtime/size and the fold window, so calibration iterations 2..N do not re-analyse
    an unchanged stem. Returns ``(stat, pitch_weights, hit)``."""
    def sig(p: Path):
        st = Path(p).stat()
        return [int(st.st_mtime_ns), st.st_size]

    key = {"stem": sig(stem), "midi": sig(midi), "fold": [lo, hi]}
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    ent = cache.get(name)
    if ent and ent.get("key") == key:
        return ent["stat"], {int(k): v for k, v in ent["pitch_weights"].items()}, True
    stat, pw = dominant_partial_log2(stem, midi, lo=lo, hi=hi, detail=True)
    cache[name] = {"key": key, "stat": stat, "pitch_weights": {str(k): v for k, v in pw.items()}}
    try:
        cache_path.write_text(json.dumps(cache, indent=1) + "\n")
    except OSError as exc:
        print(f"WARNING register cache: cannot write {cache_path}: {exc}", file=sys.stderr)
    return stat, pw, False


def load_table(path: Path = REFERENCE_PATH) -> dict[str, dict]:
    """The reference table: {sound: {log2, per_note?, range?, source}}. Warns (naming the file) and
    returns {} when it cannot be read - the generator then applies no shift."""
    try:
        return dict(json.loads(Path(path).read_text())["sounds"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"WARNING partial reference: cannot read {path} ({type(exc).__name__}: {exc}); "
              "no octave shift will be derived", file=sys.stderr)
        return {}


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, float]:
    """Scalar view of the table (backward compatible: {sound: log2})."""
    return {k: float(v["log2"]) for k, v in load_table(path).items() if "log2" in v}


def reference_for(sound: str, pitch_weights: dict[int, float] | None, table: dict[str, dict]) -> float | None:
    """The sound's reference register for the notes actually played: the energy-weighted mean of its
    per-note sweep values over ``pitch_weights`` (nearest probed note when a pitch was not probed), else
    its scalar sweep mean. None = no measurement for this sound."""
    ent = table.get(sound)
    if not ent:
        return None
    per = {int(k): float(v) for k, v in (ent.get("per_note") or {}).items()}
    if per and pitch_weights:
        tot = sum(pitch_weights.values())
        if tot > 0:
            return sum(per[min(per, key=lambda q: abs(q - p))] * w for p, w in pitch_weights.items()) / tot
    return float(ent["log2"]) if "log2" in ent else None


def octave_decision(original_log2: float | None, sound: str, reference=None, pitch_weights=None) -> dict:
    """Choose the octave shift and say why. ``reference`` is a scalar table {sound: log2} (back-compat) or
    the full table.

    Rule (measurement-bounded): shifting the line up k octaves moves the synth's own dominant partial to
    ``ref(shifted notes) + k`` (the per-note sweep reference is evaluated at the SHIFTED note numbers, it
    varies a lot across the range). Candidates are k = 0..min(MAX_OCTAVE_SHIFT, floor(original)): the
    shifted FUNDAMENTAL may not exceed the original's dominant partial (k <= original), else the render's
    energy peak would land above the original's. Within that bound pick the k minimising
    |original - (ref_at_shifted_notes + k)|; ties go to the smaller k. No stem measurement or no reference
    for the sound -> shift 0 (the latter with a warning)."""
    table = load_table() if reference is None else {
        k: (v if isinstance(v, dict) else {"log2": v}) for k, v in reference.items()}
    out = {"shift": 0, "original_log2": original_log2, "bound": 0, "candidates": [], "reason": ""}
    if original_log2 is None:
        out["reason"] = "stem not measurable"
        return out
    if reference_for(sound, pitch_weights, table) is None:
        print(f"WARNING octave: no sweep reference for {sound!r} in {REFERENCE_PATH.name} "
              f"(run probe_sound_register.py --sound {sound}); no shift", file=sys.stderr)
        out["reason"] = "no sweep reference"
        return out
    bound = int(max(0, min(MAX_OCTAVE_SHIFT, math.floor(original_log2))))
    out["bound"] = bound
    for k in range(bound + 1):
        shifted = {p + 12 * k: w for p, w in (pitch_weights or {}).items()}
        ref = reference_for(sound, shifted, table)
        out["candidates"].append({"shift": k, "ref_log2": round(ref, 4), "predicted": round(ref + k, 4),
                                  "error": round(abs(original_log2 - (ref + k)), 4)})
    out["shift"] = min(out["candidates"], key=lambda c: (c["error"], c["shift"]))["shift"]
    out["reason"] = f"best of 0..{bound} (bound = floor(original))"
    return out


def octave_shift(original_log2: float | None, sound: str, reference=None, pitch_weights=None) -> int:
    return int(octave_decision(original_log2, sound, reference, pitch_weights)["shift"])


def octave_arg(text: str):
    """argparse ``type=`` for --lead-octave/--vocal-octave: 'auto' or an int in 0..MAX_OCTAVE_SHIFT."""
    import argparse
    if text == "auto":
        return "auto"
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or an integer 0..{MAX_OCTAVE_SHIFT}, got {text!r}")
    if not 0 <= n <= MAX_OCTAVE_SHIFT:
        raise argparse.ArgumentTypeError(f"octave shift must be 0..{MAX_OCTAVE_SHIFT} (or 'auto'), got {n}")
    return n


if __name__ == "__main__":
    # usage: partial_register.py <audio> <midi> [lo hi]   -> prints the stem statistic
    a = sys.argv
    lo, hi = (int(a[3]), int(a[4])) if len(a) > 4 else (0, 127)
    print(json.dumps({"log2": dominant_partial_log2(a[1], a[2], lo=lo, hi=hi)}))
