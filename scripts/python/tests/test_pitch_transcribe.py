"""A1 data-driven pitch: tests that analyze_stem_pitch_by_section transcribes a tone to the right
Strudel note and degrades gracefully."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sound_timbre import analyze_stem_pitch_by_section, _PC_NAMES  # noqa: E402

SR = 22050


def _write_tone(path: Path, freq: float, secs: float = 4.0):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    sf.write(str(path), (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)


def test_transcribes_tone_to_correct_pitch_class(tmp_path):
    # A2 = 110 Hz → pitch class 'a', placed in target octave 2 → 'a2'
    f = tmp_path / "bass.wav"
    _write_tone(f, 110.0)
    cps = 2.0  # 2 cycles/sec → 4s = 8 cycles; keep sections small
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 4}], 8, cps, octave=2, steps_per_cycle=4)
    assert len(pats) == 1
    joined = pats[0]
    assert "a2" in joined, f"expected a2 in transcription, got: {joined[:80]}"
    # every emitted note must be a valid pitch-class name + the target octave
    import re
    for tok in re.findall(r"[a-g][#b]?2", joined):
        assert tok[:-1] in _PC_NAMES


def test_silence_transcribes_to_rests(tmp_path):
    f = tmp_path / "bass.wav"
    sf.write(str(f), np.zeros(SR * 3, dtype=np.float32), SR)
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 2}], 4, 2.0, octave=2, steps_per_cycle=4)
    assert pats and pats[0].count("~") > 0      # silence → rests, no spurious notes


def test_missing_file_returns_empty():
    assert analyze_stem_pitch_by_section("/no/such.wav", [{"cycles": 2}], 4, 2.0, 2) == []


def test_one_bar_per_cycle_structure(tmp_path):
    f = tmp_path / "bass.wav"
    _write_tone(f, 110.0)
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 3}], 6, 2.0, octave=2, steps_per_cycle=4)
    assert pats[0].count("[") == 3 and pats[0].startswith("<[") and pats[0].endswith("]>")


# --- A1 v2 sustain/legato (lead) -------------------------------------------

def test_sustain_legato_collapses_held_notes(tmp_path):
    # A held tone → sustain mode should produce a legato '@N' (one held note), not N re-triggers.
    f = tmp_path / "melodic.wav"
    _write_tone(f, 220.0, secs=4.0)  # A3
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 2}], 4, 2.0, octave=4,
                                         steps_per_cycle=8, sustain=True)
    assert "@" in pats[0], f"expected a legato @N hold, got: {pats[0][:80]}"


def test_sustain_fills_gaps_no_spurious_rests_when_loud(tmp_path):
    # A continuous tone is never silent → sustain mode should emit (almost) no rests.
    f = tmp_path / "melodic.wav"
    _write_tone(f, 220.0, secs=4.0)
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 2}], 4, 2.0, octave=4,
                                         steps_per_cycle=8, sustain=True)
    sustained_rests = pats[0].count("~")
    # without sustain (v1) a polyphonic-ish gap would emit rests; a steady tone here should be full
    assert sustained_rests <= 2, f"too many rests for a continuous tone: {pats[0][:80]}"


def test_sustain_true_silence_still_rests(tmp_path):
    f = tmp_path / "melodic.wav"
    sf.write(str(f), np.zeros(SR * 3, dtype=np.float32), SR)
    pats = analyze_stem_pitch_by_section(str(f), [{"cycles": 2}], 4, 2.0, octave=4,
                                         steps_per_cycle=8, sustain=True)
    assert pats and "~" in pats[0]   # genuine silence → rests (RMS-based), not held notes
