# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Harmony voice: the notes under the lead's top line are written as editable chord bars.

Regression (2026-10-20): the lead kept ONE note per 16th (the highest), throwing away 59% / 78% of the transcribed
notes (mean 2.6 / 7.8 simultaneous) - time-aligned chroma 0.58 / 0.47 vs 0.79 / 0.88 with the full roll. A two-note
chord in the transcription must come out as a chord token; a harmonic partial of a louder note and a ghost note
(salience from the stem's own velocity distribution) must not.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_harmony_voice.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_dynamic_strudel as G  # noqa: E402

BPM = 120.0
STEP = 60.0 / BPM / 4


def _midi(tmp_path: Path, steps: dict[int, list[tuple[int, int]]], ghosts: int = 40) -> Path:
    """steps: {step: [(pitch, velocity)]}. `ghosts` quiet notes elsewhere give the velocity distribution its
    low cluster (the stem's own bleed), so the derived salience floor sits between them and the played notes."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for s, notes in steps.items():
        for p, v in notes:
            inst.notes.append(pretty_midi.Note(v, p, s * STEP, s * STEP + STEP * 0.9))
    for i in range(ghosts):
        inst.notes.append(pretty_midi.Note(20, 60 + i % 12, (200 + i) * STEP, (200 + i) * STEP + STEP * 0.5))
    pm.instruments.append(inst)
    path = tmp_path / "m.mid"
    pm.write(str(path))
    return path


def test_two_note_chord_is_emitted_as_a_chord(tmp_path):
    # C major triad C4 E4 G4 at step 0, equally loud and well above the ghost cluster
    mid = _midi(tmp_path, {0: [(60, 90), (64, 90), (67, 90)], 4: [(62, 90), (65, 90), (69, 90)]})
    steps, _ = G.harmony_notes(mid, BPM, 2, lo=48, hi=79)
    bars = G.harmony_bars(steps, 2)
    assert bars is not None
    first = bars[0].split()
    assert first[0] == "[c4,e4]"          # top note (G4) stays the lead's; the rest is the chord
    assert first[4] == "[d4,f4]"          # A4 is the top of the second chord


def test_harmonic_partial_of_a_louder_note_is_rejected(tmp_path):
    # C3 (loudest = the lead's note) with its octave C4 (quieter overtone), a real E4 and its 12th G4
    mid = _midi(tmp_path, {0: [(48, 100), (60, 80), (67, 70), (64, 90)]})
    steps, info = G.harmony_notes(mid, BPM, 1, lo=36, hi=79)
    assert steps[0] == [64]              # E4 kept; C4 (octave) and G4 (12th) are C3's partials -> rejected; C3 is the lead's note
    assert info["rejected_partial"] >= 1


def test_ghost_notes_below_the_derived_salience_floor_are_rejected(tmp_path):
    mid = _midi(tmp_path, {0: [(72, 90), (64, 20)]})
    steps, info = G.harmony_notes(mid, BPM, 1, lo=48, hi=79)
    assert 0 not in steps and info["rejected_quiet"] >= 1
    assert 20 < info["salience_floor"] <= 90      # derived from the distribution, not typed


def test_partial_intervals_are_the_harmonic_series():
    assert {12, 19, 24, 28, 31, 34, 36} <= G.partial_intervals()
    assert 7 not in G.partial_intervals() and 5 not in G.partial_intervals()


def test_otsu_splits_two_clusters():
    assert 10 < G.otsu_threshold([1, 2, 2, 3, 50, 52, 55]) <= 50
