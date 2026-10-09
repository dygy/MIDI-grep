"""Unit tests for validate_section_alignment in ollama_codegen."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ollama_codegen import validate_section_alignment, enforce_three_voices


def test_aligned_code_passes():
    code = '''setcps(130/60/4)

// Bass
$: arrange(
  [4, note("c2 c2 c2 c2").sound("gm_synth_bass_1")],
  [8, note("c2 e2 g2 e2").sound("gm_synth_bass_1")],
  [4, note("c2 ~ c2 ~").sound("gm_synth_bass_1")]
)

// Lead
$: arrange(
  [4, note("c4 e4 g4 c5").sound("gm_piano")],
  [8, note("e4 g4 c5 g4").sound("gm_piano")],
  [4, note("c4 ~ ~ ~").sound("gm_piano")]
)

// Drums
$: arrange(
  [4, s("bd ~ sd ~").bank("RolandTR808")],
  [8, s("bd sd hh oh").bank("RolandTR808")],
  [4, s("bd ~ ~ ~").bank("RolandTR808")]
)
'''
    ok, issues = validate_section_alignment(code, 3, expected_total_cycles=16)
    assert ok, f"expected aligned code to pass, got: {issues}"


def test_wrong_section_count_fails():
    code = '''$: arrange(
  [4, note("c2").sound("sawtooth")],
  [8, note("d2").sound("sawtooth")]
)

$: arrange(
  [4, note("c4").sound("gm_piano")]
)

$: arrange(
  [4, s("bd").bank("RolandTR808")]
)
'''
    ok, issues = validate_section_alignment(code, 3, expected_total_cycles=12)
    assert not ok
    assert any("voice 1" in i for i in issues), issues
    assert any("voice 2" in i for i in issues), issues


def test_cycle_count_drift_flagged():
    # 3 sections each, but cycle totals are way off from expected 16
    code = '''$: arrange(
  [1, note("c2").sound("sawtooth")],
  [1, note("d2").sound("sawtooth")],
  [1, note("e2").sound("sawtooth")]
)

$: arrange(
  [1, note("c4").sound("gm_piano")],
  [1, note("d4").sound("gm_piano")],
  [1, note("e4").sound("gm_piano")]
)

$: arrange(
  [1, s("bd").bank("RolandTR808")],
  [1, s("sd").bank("RolandTR808")],
  [1, s("hh").bank("RolandTR808")]
)
'''
    ok, issues = validate_section_alignment(code, 3, expected_total_cycles=16, tolerance=0.1)
    assert not ok
    assert any("total cycles" in i for i in issues), issues


def test_no_sections_short_circuits():
    ok, issues = validate_section_alignment("anything", 0)
    assert ok and issues == []


def test_enforce_three_voices_unchanged():
    code = '''$: $: $: $: $:'''  # parser-pathological input
    # Should not crash
    out = enforce_three_voices("$: a\n$: b\n$: c\n$: d\n$: e\n")
    assert "$: a" in out
    assert "$: e" not in out


if __name__ == "__main__":
    test_aligned_code_passes()
    test_wrong_section_count_fails()
    test_cycle_count_drift_flagged()
    test_no_sections_short_circuits()
    test_enforce_three_voices_unchanged()
    print("PASS: validate_section_alignment + enforce_three_voices")
