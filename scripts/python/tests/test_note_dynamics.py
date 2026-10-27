# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""Each written lead/chord note carries the level the melodic stem plays it at (note_dynamics.py).

The sampled lead played every note at the voice's one gain (times a per-beat envelope); two notes of the same pitch
inside one beat - one loud, one quiet in the stem - came out equal. The generator now writes a parallel per-note
``.velocity()`` (superdough multiplies it into the voice's single ``.gain()``), measured as stem level over the
note / the level its grain plays at that pitch, divided by the median and clamped to median +- 3 robust sigmas.
The end-to-end test runs the generator on a synthetic stem and carries no flag: on the old code the lead has no
``.velocity()`` and the assertion on the loud/quiet ratio fails.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

SR, BPM, QUIET = 22050, 120.0, 0.4       # bar 2 s, beat 0.5 s, 8th 0.25 s


def _tone(n, amp):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * 440 * t) * np.hanning(n) ** 0.3).astype(np.float32)


def _make(tmp_path: Path):
    """4 bars, two a4 notes per beat: the first at the full level, the second at QUIET of it. Every beat is the same,
    so the per-beat envelope is flat and only the per-note velocity can tell the two notes apart."""
    import pretty_midi
    stems, pack = tmp_path / "stems", tmp_path / "pack"
    (pack / "instruments" / "lead").mkdir(parents=True)
    stems.mkdir()
    y = np.zeros(int(SR * 2.0 * 4), dtype=np.float32)
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    ins = pretty_midi.Instrument(0)
    for beat in range(16):
        for k, a in enumerate((1.0, QUIET)):
            t0 = beat * 0.5 + k * 0.25
            x = _tone(int(0.2 * SR), 0.3 * a)
            y[int(t0 * SR): int(t0 * SR) + len(x)] += x
            ins.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=t0, end=t0 + 0.2))
    pm.instruments.append(ins)
    pm.write(str(pack / "melodic.mid"))
    pm.write(str(pack / "bass.mid"))
    sf.write(str(stems / "melodic.wav"), y, SR)
    sf.write(str(stems / "bass.wav"), np.zeros_like(y), SR)
    sf.write(str(pack / "instruments" / "lead" / "a4.wav"), _tone(int(0.2 * SR), 0.3), SR)
    (pack / "instruments" / "samples.json").write_text(
        json.dumps({"_base": "https://example.invalid/i/", "lead": {"a4": "lead/a4.wav"}}))
    return stems, pack


def _generate(tmp_path, *extra):
    stems, pack = _make(tmp_path)
    out = tmp_path / "o.strudel"
    cmd = [sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"), "--stems-dir", str(stems), "--pack-dir", str(pack),
           "--bass-midi", str(pack / "bass.mid"), "--lead-midi", str(pack / "melodic.mid"),
           "--base-url", "https://example.invalid/i", "--samples-url", "https://example.invalid/i/instruments/samples.json",
           "--bass-sound", "lead", "--lead-sound", "lead", "--mode", "sample-instrument", "--bpm", str(BPM),
           "--key", "A minor", "--genre", "brazilian_funk", "--num-bars", "4", "--drum-mode", "extracted",
           *extra, "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-1500:]
    return out.read_text(encoding="utf-8")


def _array(code, name):
    body = code.split(f"let {name} = [")[1].split("\n]")[0]
    return re.findall(r'"([^"]*)"', body)


def _first_beat_values(code, name):
    """Velocity of every written note, per 16th step (bars are 16 slots; a `@n` hold advances n)."""
    out = {}
    for b, bar in enumerate(_array(code, name)):
        step = 0
        for tok in bar.split():
            v, _, n = tok.partition("@")
            if v != "~":
                out[b * 16 + step] = float(v)
            step += int(n) if n else 1
    return out


def test_two_notes_of_one_pitch_get_the_stems_loud_to_quiet_ratio(tmp_path):
    code = _generate(tmp_path)
    assert ".velocity(cat(...leadVel))" in code, "the lead must carry a per-note velocity"
    vel = _first_beat_values(code, "leadVel")
    lead = _array(code, "lead")
    # same shape as the notes: one value for every written note, rests stay rests
    assert [len(r.split()) for r in _array(code, "leadVel")] == [len(r.split()) for r in lead]
    # beat k holds steps 4k (loud, the 1st 8th) and 4k+2 (quiet, the 2nd 8th); the transcription may also place a
    # note a step early, so compare the notes that sit on the 8th grid
    loud = [v for s, v in vel.items() if s % 4 == 0]
    quiet = [v for s, v in vel.items() if s % 4 == 2]
    assert loud and quiet, vel
    ratio = float(np.median(quiet) / np.median(loud))
    assert ratio == pytest.approx(QUIET, rel=0.25), (ratio, vel)


def test_lead_dynamics_off_writes_every_note_flat(tmp_path):
    code = _generate(tmp_path, "--lead-dynamics", "off")
    assert ".velocity(" not in code.split("$:")[1] and "leadVel" not in code


def test_velocity_ratio_is_median_normalised_and_clamped():
    import note_dynamics as ND
    bars = ["c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4"] * 2
    sr, bpm = 1000, 120.0
    step = 60.0 / bpm * 4 / 16
    amp = np.full(int(sr * 4.0), 0.1)
    for s in range(32):
        a = 0.1 * (1000.0 if s == 5 else 1.0)            # one bleed spike 1000x the rest
        amp[int(s * step * sr): int((s + 1) * step * sr)] = a
    v = ND.note_velocities(amp, sr, bpm, {"lead": (bars, 1.0)}, lambda p, w: 1.0, lambda p: 1.0, lambda a, b: 1.0)
    vals = [float(t) for r in v["lead"] for t in r.split()]
    assert np.median(vals) == pytest.approx(1.0, abs=1e-6)
    assert max(vals) < 3.0, vals                           # the spike is clamped, not written 1000x
