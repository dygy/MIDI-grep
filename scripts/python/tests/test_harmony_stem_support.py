# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""A quiet transcribed note the melodic stem DOES sound is written as harmony; one it does not sound stays out.

Regression (2026-10-25): ``harmony_notes`` rejected every transcribed note below an Otsu floor on the transcriber's
velocity. That is the transcriber's confidence, not the stem's content: on VAGABUNDO it discarded 1504 of 2332 notes,
and 24 % of the melodic stem's strong chroma that no written voice sounded was a pitch class the transcription held
and the floor threw away (audit: ``sound_quality`` melfill). A quiet note is now kept when the stem's own chroma
sounds its pitch class while it lasts, so the ghost note the stem never plays is still dropped.

The fixture is synthetic: a stem of A4 + C#5 sines struck once per bar, a transcription with A4 (loud, the lead),
C#5 (quiet, sounded by the stem) and F4 (quiet, never sounded), plus a cluster of ghost notes that puts the velocity
floor between them.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_harmony_stem_support.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
GEN = SCRIPTS / "generate_dynamic_strudel.py"
SR, BPM, NBARS = 22050, 120, 6
BAR = 60.0 / BPM * 4


def _fixture(tmp: Path) -> Path:
    (tmp / "pack").mkdir(parents=True)
    t = np.arange(int(NBARS * BAR * SR)) / SR
    y = np.zeros_like(t)
    for b in range(NBARS):
        m = (t >= b * BAR) & (t < b * BAR + 1.6)
        tt = t[m] - b * BAR
        env = np.minimum(1, tt / 0.01) * np.exp(-tt / 1.2)
        y[m] += env * (0.4 * np.sin(2 * np.pi * 440.0 * tt) + 0.4 * np.sin(2 * np.pi * 554.37 * tt))
    sf.write(str(tmp / "melodic.wav"), y.astype(np.float32), SR)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for b in range(NBARS):
        for pitch, vel in ((69, 100), (73, 30), (65, 30)):      # A4 lead / C#5 sounded / F4 never sounded
            inst.notes.append(pretty_midi.Note(vel, pitch, b * BAR, b * BAR + 1.5))
    step = BAR / 16
    for i in range(40):                                          # ghosts beyond the piece: they only shape the floor
        inst.notes.append(pretty_midi.Note(20, 60 + i % 12, (200 + i) * step, (200 + i) * step + step * 0.5))
    pm.instruments.append(inst)
    pm.write(str(tmp / "pack" / "melodic.mid"))
    return tmp


def _generate(tmp: Path, *extra: str) -> tuple[str, dict]:
    out = tmp / "o.strudel"
    cmd = [sys.executable, str(GEN), "--stems-dir", str(tmp), "--pack-dir", str(tmp / "pack"),
           "--bass-midi", str(tmp / "none.mid"), "--lead-midi", str(tmp / "pack" / "melodic.mid"),
           "--base-url", "http://x/", "--mode", "synth", "--bpm", str(BPM), "--key", "A major",
           "--genre", "brazilian_funk", "--num-bars", str(NBARS), *extra, "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    return out.read_text(), json.loads(r.stdout[r.stdout.index("{"):])


def _written(code: str, name: str) -> set[str]:
    m = re.search(r"let %s = \[\n(.*?)\n\]" % name, code, re.S)
    return set(re.findall(r"[a-g]s?-?\d+", m.group(1))) if m else set()


def test_quiet_note_the_stem_sounds_is_written_and_the_one_it_does_not_is_not(tmp_path):
    _fixture(tmp_path)
    code, summary = _generate(tmp_path)
    h = summary["harmony"]
    assert h["salience_floor"] > 30, h                       # both quiet notes are below the velocity floor
    assert _written(code, "lead") == {"a4"}
    chords = _written(code, "chords")
    assert "cs5" in chords, (chords, h)                      # the stem sounds C#5: written despite its low velocity
    assert not chords & {"f4", "f5", "f3"}, chords           # F4 is a transcription ghost: the stem has no F
    assert h.get("kept_quiet_stem_sounds", 0) >= 1, h


def test_without_stem_evidence_the_velocity_floor_still_decides(tmp_path):
    _fixture(tmp_path)
    code, summary = _generate(tmp_path, "--melodic-shape", "off")
    assert _written(code, "chords") == set(), summary["harmony"]
    assert summary["harmony"].get("kept_quiet_stem_sounds", 0) == 0
