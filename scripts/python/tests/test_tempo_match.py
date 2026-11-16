"""compare_audio.tempo_match: the tempo similarity compares the raw beat-tracker reads too.

BibiBabydoll "Melhor Sobrar" synth (2026-11-15): original read 152 BPM, render 76 (the same pulse an octave down);
correct_octave_error mapped them to 101.3 and 114 (ratio 8/9), so the piece scored tempo 0.167 although it plays the
original's pulse - 0.021 of the overall score. Every other promoted piece kept its number.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_audio as CA  # noqa: E402


def test_same_pulse_corrected_to_different_factors_matches_through_the_raw_reads():
    o, r = CA.correct_octave_error(152.0), CA.correct_octave_error(76.0)
    assert (round(o, 1), round(r, 1)) == (101.3, 114.0)                     # the two corrections disagree
    assert CA.tempo_match(o, r)[0] == pytest.approx(12.67, abs=0.01)        # corrected reads alone: 8/9 apart
    assert CA.tempo_match(o, r, 152.0, 76.0) == (pytest.approx(0.0), 2.0)  # raw: an octave


def test_raw_reads_never_make_a_different_tempo_match():
    # 130 vs 112 BPM are no musical ratio apart, raw or corrected: the diff stays large
    o, r = CA.correct_octave_error(130.0), CA.correct_octave_error(112.0)
    assert CA.tempo_match(o, r, 130.0, 112.0)[0] > 0.1 * o
    # without raw reads (old cached features) the corrected comparison is unchanged
    assert CA.tempo_match(o, r) == CA.tempo_match(o, r, None, None)
