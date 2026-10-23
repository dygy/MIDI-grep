# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""The lead's note for a step is the LOUDEST transcribed note, not the highest.

Regression (2026-10-23, melodic-notes-in-time): the lead wrote the highest note of every 16th step. Measured against
the melodic stem's own chroma, the highest transcribed note names the stem's pitch class only 23% (Regime CLT) /
67% (VAGABUNDO) of the time, the loudest (Basic Pitch velocity) 52% / 80% - Basic Pitch hears overtones above the
played note. The chord voice must then exclude the lead's note, not the highest one.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_lead_loudest_pitch.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_dynamic_strudel as G  # noqa: E402

BPM = 120.0
STEP = 60.0 / BPM / 4


def _midi(tmp_path: Path, steps: dict[int, list[tuple[int, int]]]) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for s, notes in steps.items():
        for p, v in notes:
            inst.notes.append(pretty_midi.Note(v, p, s * STEP, s * STEP + STEP * 0.9))
    pm.instruments.append(inst)
    path = tmp_path / "m.mid"
    pm.write(str(path))
    return path


def test_lead_is_the_loudest_note_of_the_step_not_the_highest(tmp_path):
    # A4 played loudly; its 12th E6 is a quiet overtone the transcriber also reported
    mid = _midi(tmp_path, {0: [(69, 100), (88, 40)], 4: [(62, 90), (74, 30)]})
    bars = G.midi_to_bars(mid, BPM, 1, pick="loud", lo=48, hi=95)
    toks = bars[0].split()
    assert toks[0] == "a4" and toks[4] == "d4"
    assert G.midi_to_bars(mid, BPM, 1, pick="high", lo=48, hi=95)[0].split()[0] == "e6"   # old behaviour


def test_equal_loudness_keeps_the_highest(tmp_path):
    mid = _midi(tmp_path, {0: [(60, 80), (67, 80)]})
    assert G.midi_to_bars(mid, BPM, 1, pick="loud", lo=48, hi=95)[0].split()[0] == "g4"


def test_harmony_excludes_the_loudest_lead_note(tmp_path):
    mid = _midi(tmp_path, {0: [(60, 100), (67, 99)], 4: [(62, 90), (74, 90)]})
    steps, _ = G.harmony_notes(mid, BPM, 1, lo=48, hi=95)
    lead = G.midi_to_bars(mid, BPM, 1, pick="loud", lo=48, hi=95)[0].split()
    assert lead[0] == "c4" and 60 not in steps.get(0, [])      # the lead's C4 is not repeated in the chord
    assert steps[0] == [67]


def test_held_note_keeps_its_pitch_on_steps_without_an_onset(tmp_path):
    # one long A4 (steps 0-3, loud) and a short, quieter C5 that starts at step 2
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 69, 0.0, 4 * STEP))
    inst.notes.append(pretty_midi.Note(60, 72, 2 * STEP, 2.5 * STEP))
    pm.instruments.append(inst)
    path = tmp_path / "held.mid"
    pm.write(str(path))
    plain = G.midi_to_bars(path, BPM, 1, pick="loud", lo=48, hi=95)[0].split()
    held = G.midi_to_bars(path, BPM, 1, pick="loud", lo=48, hi=95, sustain=True)[0].split()
    assert plain[:4] == ["a4", "~", "c5", "~"]            # without sustain the held steps are rests
    assert held[:4] == ["a4", "a4", "c5", "a4"]           # with it the loudest SOUNDING note fills them; an onset still wins its step
