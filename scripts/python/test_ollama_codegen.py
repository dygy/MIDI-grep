#!/usr/bin/env python3
"""Tests for ollama_codegen.py — Strudel syntax fixing and voice enforcement."""

import pytest
from ollama_codegen import fix_strudel_syntax, enforce_three_voices


class TestFixStrudelSyntax:
    """Test fix_strudel_syntax() — the regex gauntlet every LLM output passes through."""

    def test_semicolon_removal(self):
        code = 'note("c4 e4 g4").gain(0.5);'
        assert ";" not in fix_strudel_syntax(code)

    def test_setcps_float_fix(self):
        code = "setcps(136.0/60/4)"
        assert fix_strudel_syntax(code) == "setcps(136/60/4)"

    def test_setcps_integer_unchanged(self):
        code = "setcps(136/60/4)"
        assert fix_strudel_syntax(code) == "setcps(136/60/4)"

    def test_gain_string_to_number(self):
        code = '.gain("0.5")'
        result = fix_strudel_syntax(code)
        assert '.gain(0.5)' in result

    def test_gain_pattern_string_preserved(self):
        """Gain patterns with < > should not be converted."""
        code = '.gain("<0.3 0.5 0.8>")'
        result = fix_strudel_syntax(code)
        assert '<0.3 0.5 0.8>' in result

    def test_empty_method_removal(self):
        code = '.room().delay(0.2)'
        result = fix_strudel_syntax(code)
        assert '.room()' not in result
        assert '.delay(0.2)' in result

    def test_bank_name_normalization(self):
        code = '.bank("tr808")'
        result = fix_strudel_syntax(code)
        assert 'RolandTR808' in result

    def test_drum_pattern_note_to_bd(self):
        """Note names in drum patterns (with .bank) should become drum sounds."""
        code = 's("c2 e2 g2 c2").bank("RolandTR808")'
        result = fix_strudel_syntax(code)
        assert "c2" not in result
        assert "bd" in result

    def test_sound_name_correction(self):
        code = '.sound("gm_acoustic_grand_piano")'
        result = fix_strudel_syntax(code)
        assert 'gm_piano' in result

    def test_arrange_comma_insertion(self):
        code = '  [4, note("c2")]\n  [8, note("e2")]'
        result = fix_strudel_syntax(code)
        assert '],\n' in result

    def test_arrange_comma_no_double(self):
        """Already-commaed entries should not get double commas."""
        code = '  [4, note("c2")],\n  [8, note("e2")]'
        result = fix_strudel_syntax(code)
        assert '],,' not in result

    def test_empty_input(self):
        assert fix_strudel_syntax("") == ""

    def test_idempotent(self):
        code = '.bank("tr808"); .gain("0.5"); .sound("gm_electric_guitar")'
        once = fix_strudel_syntax(code)
        twice = fix_strudel_syntax(once)
        assert once == twice

    def test_valid_code_unchanged(self):
        """Already-valid code should pass through unchanged."""
        code = '$: note("c4 e4 g4").sound("gm_piano").gain(0.5).lpf(4000)'
        result = fix_strudel_syntax(code)
        assert result == code


class TestEnforceThreeVoices:
    """Test enforce_three_voices() — ensures exactly 3 $: blocks."""

    def test_three_voices_unchanged(self):
        code = """setcps(136/60/4)

// Bass
$: arrange([4, note("c2")])

// Lead
$: arrange([4, note("c4")])

// Drums
$: arrange([4, s("bd sd")])"""
        result = enforce_three_voices(code)
        assert result.count("$:") == 3

    def test_six_voices_trimmed(self):
        blocks = []
        for i in range(6):
            blocks.append(f"// Voice {i}\n$: arrange([4, note(\"c{i}\")])")
        code = "setcps(136/60/4)\n\n" + "\n\n".join(blocks)
        result = enforce_three_voices(code)
        assert result.count("$:") == 3

    def test_header_preserved(self):
        code = """setcps(136/60/4)

// Some comment

// Bass
$: arrange([4, note("c2")])

// Lead
$: arrange([4, note("c4")])

// Drums
$: arrange([4, s("bd sd")])

// Extra
$: arrange([4, note("c5")])"""
        result = enforce_three_voices(code)
        assert "setcps(136/60/4)" in result
        assert result.count("$:") == 3

    def test_one_voice_unchanged(self):
        code = '$: note("c4").sound("gm_piano")'
        result = enforce_three_voices(code)
        assert result.count("$:") == 1

    def test_empty_input(self):
        assert enforce_three_voices("") == ""

    def test_no_dollar_blocks(self):
        code = 'note("c4 e4 g4")'
        result = enforce_three_voices(code)
        assert result == code
