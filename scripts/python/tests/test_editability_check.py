# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Editability / replay detector tests (spec 003 §2.2-A, rules R1–R6).

One positive + one negative case per rule, CLI exit codes via subprocess, and the verdicts the
tech spec pins on real artifacts (fixtures copied from the Regime CLT cache):

    v012 output.strudel            → fail (R1+R2, loop-only)
    v023 output.strudel            → fail (R1+R2 on line 196, no marker)
    v023 minus line 196            → pass, mode sample-instrument
    sample_pack/output_loops       → fail

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_editability_check.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from editability_check import (  # noqa: E402
    EditabilityResult,
    ParseError,
    check_editability,
    to_json_fields,
)
from strudel_validation import validate_code  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "editability"
CLI = SCRIPTS / "editability_check.py"
PY = sys.executable


def fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def rules(res: EditabilityResult) -> set[str]:
    return {d.rule for d in res.violation_details}


HEADER = 'setcps(0.5)\nlet bass = ["c2 ~ e2 ~", "f2 ~ g2 ~"]\nlet lead = ["c4 e4", "d4 f4"]\n'
TWO_VOICES = HEADER + '$: note(cat(...bass)).s("sawtooth")\n$: note(cat(...lead)).s("triangle")\n'


# ── R1 reconstruction-by-playback ─────────────────────────────────────────────────────────
def test_r1_array_slice_does_not_trip():
    code = HEADER + '$: note(cat(...bass.slice(0, 4))).s("sawtooth")\n$: s("bd sd")\n'
    res = check_editability(code)
    assert res.passed, res.violations
    assert "R1" not in rules(res)


@pytest.mark.parametrize("replay", [
    '$: s("vox").slice(16, run(16)).slow(16).clip(1)',           # literal N
    '$: s("vox").slice(nb, run(nb)).slow(nb)',                   # identifier N
    '$: s("vox").slice(nb, run(nb)).clip(1).slow(nb)',           # chain call in between
    '$: s("vox").loopAt(16)',
])
def test_r1_replay_shapes_are_violations(replay):
    res = check_editability(TWO_VOICES + "const nb = 16\n" + replay + "\n")
    assert not res.passed
    assert "R1" in rules(res)
    assert any(v.startswith("R1 line 7") for v in res.violations), res.violations


# ── R2 full-stem sound ────────────────────────────────────────────────────────────────────
def test_r2_instrument_sample_name_is_fine():
    code = 'await samples("https://x.invalid/instruments/samples.json")\n' + HEADER + \
        '$: note(cat(...bass)).s("regime_bass")\n$: note(cat(...lead)).s("regime_lead")\n'
    res = check_editability(code)
    assert res.passed, res.violations
    assert "R2" not in rules(res)


@pytest.mark.parametrize("name", ["originalfull", "vocalsfull", "origseg3", "drumsloop"])
def test_r2_full_stem_sound_is_violation(name):
    res = check_editability(TWO_VOICES + f'$: s("{name}").clip(1)\n')
    assert not res.passed
    assert any(v.startswith("R2 line 6") and name in v for v in res.violations), res.violations


def test_r2_replay_manifest_load_is_violation():
    code = 'await samples("https://x.invalid/samples_orig.json")\n' + TWO_VOICES
    res = check_editability(code)
    assert not res.passed
    assert any(v.startswith("R2 line 1") and "samples_orig.json" in v for v in res.violations)


# ── R3 editable voice counting ────────────────────────────────────────────────────────────
def test_r3_counts_bar_array_note_voices_literal_note_and_oneshot_drums():
    code = HEADER + (
        '$: note(cat(...bass)).s("sawtooth")\n'
        '$: note("c4 e4 g4").s("triangle")\n'
        '$: stack(\n  s("bd*2 sd"),\n  s("hh*8").gain(0.2)\n).bank("RolandTR808")\n'
    )
    res = check_editability(code)
    assert res.passed, res.violations
    assert [v.kind for v in res.editable_voices] == ["editable"] * 4
    assert [v.label for v in res.editable_voices] == ["bass", "triangle", "bd", "hh"]
    assert [v.line_start for v in res.editable_voices] == [4, 5, 7, 8]


def test_r3_note_without_pattern_data_is_not_editable():
    code = 'setcps(0.5)\n$: note(someVar).s("sawtooth")\n'
    res = check_editability(code)
    assert not res.passed
    assert res.editable_voices == []
    assert [v.kind for v in res.unclassified_voices] == ["unclassified"]
    assert "R5" in rules(res)


# ── R4 texture allowance ──────────────────────────────────────────────────────────────────
def test_r4_texture_loop_allowed_with_two_editable_voices():
    res = check_editability(fx("texture_pass.strudel"))
    assert res.passed, res.violations
    assert len(res.editable_voices) == 2
    assert len(res.texture_voices) == 1
    assert res.texture_voices[0].line_start == 19
    assert to_json_fields(res)["texture_voice_count"] == 1


def test_r4_texture_loop_rejected_with_one_editable_voice():
    res = check_editability(fx("texture_fail_one_voice.strudel"))
    assert not res.passed
    assert "R4" in rules(res)
    assert any(v.startswith("R4 line 14") for v in res.violations), res.violations
    assert len(res.editable_voices) == 1 and len(res.texture_voices) == 1


def test_r4_marker_must_be_on_the_replay_voice_line():
    code = TWO_VOICES + '// texture\n$: s("vocalsfull").loopAt(16)\n'
    res = check_editability(code)
    assert not res.passed
    assert res.texture_voices == [] and len(res.replay_voices) == 1


# ── R5 loop-only ──────────────────────────────────────────────────────────────────────────
def test_r5_passes_with_editable_voices():
    res = check_editability(fx("synth_pass.strudel"))
    assert res.passed, res.violations
    assert "R5" not in rules(res)
    assert len(res.editable_voices) == 3


def test_r5_loop_only_fails_even_with_texture_marker():
    code = 'setcps(0.5)\n$: s("vocalsfull").loopAt(16)  // texture\n'
    res = check_editability(code)
    assert not res.passed
    assert "R5" in rules(res) and "R4" in rules(res)
    assert res.editable_voices == []


# ── R6 generation mode ────────────────────────────────────────────────────────────────────
def test_r6_header_wins_over_hint_and_inference():
    code = "// generation_mode: synth\n" + 'await samples("https://x.invalid/instruments/samples.json")\n' + \
        HEADER + '$: note(cat(...bass)).s("regime_bass")\n'
    assert check_editability(code, mode_hint="sample-instrument").generation_mode == "synth"


def test_r6_hint_beats_inference():
    assert check_editability(TWO_VOICES, mode_hint="sample-instrument").generation_mode == "sample-instrument"


def test_r6_infers_sample_instrument_from_samples_load_plus_custom_name():
    code = 'await samples("https://x.invalid/instruments/samples.json")\n' + HEADER + \
        '$: note(cat(...bass)).s("regime_bass")\n'
    assert check_editability(code).generation_mode == "sample-instrument"


def test_r6_infers_synth_without_samples_load():
    res = check_editability(TWO_VOICES)
    assert res.passed and res.generation_mode == "synth"


def test_r6_loops_header_is_a_violation():
    res = check_editability("// generation_mode: loops — texture/diagnostic, NOT a deliverable\n" + TWO_VOICES)
    assert not res.passed
    assert res.generation_mode == "loops"
    assert any(v.startswith("R6 line 1") for v in res.violations), res.violations


# ── comment stripping / parse errors / JSON fields ────────────────────────────────────────
def test_comments_are_stripped_before_matching_and_urls_survive():
    code = (
        '// $: s("originalfull").slice(8, run(8)).slow(8)  <- commented-out replay is ignored\n'
        '/* s("vocalsfull").loopAt(4) */\n'
        'await samples("https://host.invalid/path//samples.json")  // url contains //\n'
        + TWO_VOICES
    )
    res = check_editability(code)
    assert res.passed, res.violations
    assert len(res.editable_voices) == 2


@pytest.mark.parametrize("bad", ["", "   \n// only a comment\n", 'setcps(0.5)\n$: note("c3"\n', '$: note("c3\n'])
def test_parse_errors(bad):
    with pytest.raises(ParseError):
        check_editability(bad)


def test_to_json_fields_shape():
    d = to_json_fields(check_editability(fx("v023_vocal_replay.strudel")))
    assert set(d) == {"editability", "generation_mode", "editability_violations",
                      "editable_voice_count", "texture_voice_count"}
    assert d["editability"] == "fail" and d["generation_mode"] == "sample-instrument"
    assert d["editable_voice_count"] == 7 and d["texture_voice_count"] == 0


# ── real-artifact verdicts (tech spec §2.2-A) ─────────────────────────────────────────────
def test_fixture_v012_loop_replay_fails_loop_only():
    res = check_editability(fx("v012_loop_replay.strudel"))
    assert not res.passed
    assert {"R1", "R2", "R5"} <= rules(res)
    assert res.editable_voices == [] and res.generation_mode == "loops"
    assert any(v.startswith("R1 line 5") for v in res.violations)


def test_fixture_v023_fails_on_vocal_line_196():
    res = check_editability(fx("v023_vocal_replay.strudel"))
    assert not res.passed
    assert rules(res) == {"R1", "R2"}
    assert all("line 196" in v for v in res.violations), res.violations
    assert res.generation_mode == "sample-instrument"
    assert len(res.editable_voices) == 7


def test_fixture_v023_minus_vocal_passes_as_sample_instrument():
    res = check_editability(fx("v023_minus_vocal.strudel"))
    assert res.passed, res.violations
    assert res.generation_mode == "sample-instrument"
    assert len(res.editable_voices) == 7 and res.texture_voices == []


def test_fixture_output_loops_fails():
    res = check_editability(fx("output_loops.strudel"))
    assert not res.passed
    assert {"R2", "R5"} <= rules(res)
    assert res.editable_voices == [] and len(res.replay_voices) == 4


# ── CLI exit codes ────────────────────────────────────────────────────────────────────────
def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(CLI), *args], capture_output=True, text=True)


def test_cli_exit_0_on_pass():
    p = run_cli(str(FIXTURES / "synth_pass.strudel"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("PASS")


def test_cli_exit_1_on_fail_and_json_names_line_196():
    p = run_cli(str(FIXTURES / "v023_vocal_replay.strudel"), "--json")
    assert p.returncode == 1
    d = json.loads(p.stdout)
    assert d["editability"] == "fail"
    assert all("line 196" in v for v in d["editability_violations"])


def test_cli_exit_2_on_missing_file_and_parse_error(tmp_path):
    assert run_cli(str(tmp_path / "nope.strudel")).returncode == 2
    empty = tmp_path / "empty.strudel"
    empty.write_text("// nothing here\n")
    p = run_cli(str(empty), "--json")
    assert p.returncode == 2
    assert json.loads(p.stdout)["editability"] == "error"
    assert run_cli().returncode == 2  # usage error


# ── strudel_validation generation-time reject (LLM codegen path) ──────────────────────────
@pytest.mark.parametrize("code,needle", [
    ('$: s("vox").loopAt(16)', ".loopAt("),
    ('$: s("originalfull").slice(8, run(8)).slow(8)', "originalfull"),
    ('$: s("vocalsfull").clip(1)', "vocalsfull"),
    ('$: s("drumsloop").n("<0 1 2>")', "drumsloop"),
])
def test_validate_code_rejects_replay(code, needle):
    _, err = validate_code(code)
    assert err and needle in err and "replay" in err


def test_validate_code_still_accepts_editable_code():
    _, err = validate_code('$: note("c3 e3").sound("sawtooth").lpf(400)\n$: s("bd sd").bank("RolandTR808")')
    assert err == ""
