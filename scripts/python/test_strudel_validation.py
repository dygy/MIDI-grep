#!/usr/bin/env python3
"""Tests for strudel_validation.py — shared sound/bank correction utilities."""

import pytest
from strudel_validation import (
    SOUND_CORRECTIONS,
    BANK_CORRECTIONS,
    fix_sound_names,
    fix_bank_names,
)


class TestSoundCorrections:
    """Test the SOUND_CORRECTIONS dictionary."""

    def test_corrections_not_empty(self):
        assert len(SOUND_CORRECTIONS) > 20

    def test_no_identity_mappings(self):
        for wrong, correct in SOUND_CORRECTIONS.items():
            assert wrong != correct, f"Identity mapping: {wrong} → {correct}"

    def test_known_corrections(self):
        assert SOUND_CORRECTIONS["gm_acoustic_grand_piano"] == "gm_piano"
        assert SOUND_CORRECTIONS["gm_electric_guitar"] == "gm_electric_guitar_clean"
        assert SOUND_CORRECTIONS["gm_synth_bass"] == "gm_synth_bass_1"
        assert SOUND_CORRECTIONS["gm_electric_piano"] == "gm_epiano1"


class TestFixSoundNames:
    """Test fix_sound_names() function."""

    @pytest.mark.parametrize("input_code,expected", [
        # Basic correction
        ('.sound("gm_acoustic_grand_piano")', '.sound("gm_piano")'),
        ('.sound("gm_electric_guitar")', '.sound("gm_electric_guitar_clean")'),
        ('.sound("gm_synth_bass")', '.sound("gm_synth_bass_1")'),
        # Single quotes
        (".sound('gm_electric_piano')", ".sound('gm_epiano1')"),
        # Should NOT fix valid sounds (no partial match)
        ('.sound("gm_electric_guitar_clean")', '.sound("gm_electric_guitar_clean")'),
        ('.sound("gm_synth_bass_1")', '.sound("gm_synth_bass_1")'),
        ('.sound("sawtooth")', '.sound("sawtooth")'),
        # Multiple corrections in one code block
        (
            '.sound("gm_organ").gain(0.5)\n.sound("gm_brass")',
            '.sound("gm_drawbar_organ").gain(0.5)\n.sound("gm_brass_section")',
        ),
    ])
    def test_corrections(self, input_code, expected):
        assert fix_sound_names(input_code) == expected

    def test_empty_string(self):
        assert fix_sound_names("") == ""

    def test_no_sound_calls(self):
        code = 'note("c4 e4 g4").gain(0.5)'
        assert fix_sound_names(code) == code

    def test_idempotent(self):
        """Applying fix twice should produce the same result."""
        code = '.sound("gm_acoustic_grand_piano").sound("gm_electric_guitar")'
        once = fix_sound_names(code)
        twice = fix_sound_names(once)
        assert once == twice


class TestFixBankNames:
    """Test fix_bank_names() function."""

    @pytest.mark.parametrize("input_code,expected", [
        ('.bank("tr808")', '.bank("RolandTR808")'),
        ('.bank("TR808")', '.bank("RolandTR808")'),
        ('.bank("tr909")', '.bank("RolandTR909")'),
        ('.bank("linndrum")', '.bank("LinnDrum")'),
        ('.bank("linn")', '.bank("LinnDrum")'),
        ('.bank("dr110")', '.bank("BossDR110")'),
        ('.bank("mpc60")', '.bank("AkaiMPC60")'),
        # Already correct — no change
        ('.bank("RolandTR808")', '.bank("RolandTR808")'),
        ('.bank("LinnDrum")', '.bank("LinnDrum")'),
    ])
    def test_corrections(self, input_code, expected):
        assert fix_bank_names(input_code) == expected

    def test_empty_string(self):
        assert fix_bank_names("") == ""

    def test_idempotent(self):
        code = '.bank("tr808").bank("tr909")'
        once = fix_bank_names(code)
        twice = fix_bank_names(once)
        assert once == twice
