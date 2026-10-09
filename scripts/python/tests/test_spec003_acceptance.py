# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""Spec 003 acceptance tests — the WHOLE feature against functional-spec.md §2.1-§2.5.

The per-slice suites (test_editability_check, test_similarity_gate, test_generation_modes,
test_vocal_modes, test_compare_audio_stamping, test_report_editability) prove each component.
This file proves the contract holds ACROSS components, on whole deliverables:

  deliverable (committed fixture or cached v024/v025 render output)
      -> editability_check -> strudel_validation -> compare_audio --strudel
      -> eval/gate -> loop MCP -> ai_improver -> generate_report

Every positive case has a negative counterpart (see the comment above each test). Cases that need
a live BlackHole render are out of CI scope: they read the CACHED renders under
`.cache/stems/Regime CLT (Dj Brunin XM, Aurora Shukita)/v024|v025` and SKIP when absent.

Criterion -> test map
  2.1 editable data per voice     test_21_every_voice_is_its_own_block_of_editable_data
                                  test_21_undeclared_array_is_not_editable_data
      Editability Test (static)   test_21_single_note_edit_changes_exactly_one_token_and_stays_valid
                                  test_21_replay_voice_has_no_data_to_edit
      Sound-swap Test             test_21_sound_swap_revoices_but_keeps_the_pattern
                                  test_21_sound_swap_to_a_full_stem_replay_is_rejected
      Structure Test              test_21_reordering_and_duplicating_bars_stays_a_valid_deliverable
      FORBIDDEN replay            test_21_forbidden_replay_forms_are_rejected_at_every_gate
                                  test_21_user_bar_slicing_is_not_replay
  2.2 independent voices / mute   test_22_each_voice_can_be_commented_out_individually
                                  test_22_muting_everything_or_half_a_block_is_detected
      plays without manual fixes  test_22_output_passes_the_generation_time_validator
                                  test_22_invalid_code_is_caught_by_the_validator
      setcps from BPM             test_22_setcps_matches_the_declared_bpm
                                  test_22_setcps_mismatch_or_absence_is_detected
                                  test_22_generator_sets_cps_from_the_requested_bpm
  2.3 resemblance                 test_23_sample_instrument_realism_lives_in_declared_note_data
                                  test_23_synth_mode_is_pure_synthesis
                                  test_23_loops_are_texture_only_under_two_editable_voices
                                  test_23_a_loop_only_output_is_not_a_deliverable
  2.4 honest measurement          test_24_replay_is_refused_before_scoring_end_to_end
                                  test_24_editable_output_is_scored_and_stamped_end_to_end
                                  test_24_loop_mcp_refuses_replay_before_rendering
                                  test_24_ai_improver_stamps_fail_closed
                                  test_24_v023_is_disqualified
                                  test_24_two_modes_ship_and_loops_is_not_one_of_them
                                  test_24_mode_floors_are_reproducible_from_a_real_run
                                  test_24_a_tampered_measurement_does_not_reproduce
  2.5 reporting                   test_25_headline_states_the_generating_mode
                                  test_25_replay_score_never_reaches_the_report
                                  test_25_go_and_python_reports_agree

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_spec003_acceptance.py -q
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TESTS))

yaml = pytest.importorskip("yaml")

from editability_check import (  # noqa: E402
    DRUM_ONESHOTS,
    ParseError,
    check_editability,
    to_json_fields,
)
from strudel_validation import VALID_SOUNDS, validate_code  # noqa: E402
from eval.gate import evaluate_comparison, load_thresholds, main as gate_main  # noqa: E402

THRESHOLDS = load_thresholds()
FIX = TESTS / "fixtures" / "editability"
REPORT_FIX = TESTS / "fixtures" / "report"
CACHE = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
COMPARE = SCRIPTS / "compare_audio.py"

# ── deliverables under test ──────────────────────────────────────────────────────────────
#   (id, path, expected generation_mode).  Fixtures are committed and always run; the cached
#   renders are the real generator output measured in Slice 4 and skip when the cache is absent.
DELIVERABLES = [
    pytest.param(("fixture-sample-instrument", FIX / "v023_minus_vocal.strudel", "sample-instrument"),
                 id="fixture-sample-instrument"),
    pytest.param(("fixture-synth", FIX / "synth_pass.strudel", "synth"), id="fixture-synth"),
    pytest.param(("cache-v024", CACHE / "v024" / "output.strudel", "sample-instrument"), id="cache-v024"),
    pytest.param(("cache-v025", CACHE / "v025" / "output.strudel", "synth"), id="cache-v025"),
]


@dataclass(frozen=True)
class Deliverable:
    name: str
    path: Path
    code: str
    mode: str


@pytest.fixture(params=DELIVERABLES)
def deliverable(request) -> Deliverable:
    name, path, mode = request.param
    if not path.exists():
        pytest.skip(f"{path} not present (cached render absent)")
    return Deliverable(name, path, path.read_text(encoding="utf-8"), mode)


# ── helpers (pure text, no Strudel runtime) ──────────────────────────────────────────────
_BLOCK_START = re.compile(r"^_?\$\w*\s*:")
_NOTE_TOKEN = re.compile(r"(?<![\w#])[a-g][sb#]?-?\d(?![\w])")


def _blocks(code: str) -> list[tuple[int, int]]:
    """(first_line_idx, end_line_idx_exclusive) of every top-level `$:` block, 0-based."""
    lines = code.split("\n")
    starts = [i for i, ln in enumerate(lines) if _BLOCK_START.match(ln)]
    return [(s, starts[k + 1] if k + 1 < len(starts) else len(lines)) for k, s in enumerate(starts)]


def _comment_out(code: str, first: int, end: int) -> str:
    lines = code.split("\n")
    for i in range(first, end):
        lines[i] = "// " + lines[i]
    return "\n".join(lines)


def _array_decls(code: str) -> dict[str, list[str]]:
    """name -> list of bar strings for every `let name = [ "..." , ... ]` declaration."""
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"^let\s+(\w+)\s*=\s*\[(.*?)\n\]", code, re.S | re.M):
        out[m.group(1)] = re.findall(r'"([^"]*)"', m.group(2))
    return out


def _edit_first_note(code: str, array: str, new: str = "eb2") -> tuple[str, int, int]:
    """Replace the first note token of `array` with `new` (the spec's c2 -> eb2 edit).
    Returns (new_code, bar_index, token_index). Raises LookupError when `array` is not declared
    data or holds no note — i.e. there is nothing a user could edit."""
    m = re.search(r"^(let\s+%s\s*=\s*\[)(.*?)(\n\])" % re.escape(array), code, re.S | re.M)
    if not m:
        raise LookupError(f"no declared bar array named {array!r}")
    body = m.group(2)
    for bar_idx, bm in enumerate(re.finditer(r'"([^"]*)"', body)):
        toks = bm.group(1).split()
        for tok_idx, tok in enumerate(toks):
            if _NOTE_TOKEN.fullmatch(tok):
                toks[tok_idx] = new if tok != new else "e2"
                start = m.start(2) + bm.start(1)
                end = m.start(2) + bm.end(1)
                return code[:start] + " ".join(toks) + code[end:], bar_idx, tok_idx
    raise LookupError(f"array {array!r} holds no note token")


def _first_pitched_instrument(code: str) -> tuple[str, str]:
    """(line, old_sound) of the first `note(cat(...arr)) ... .s("old")` line."""
    for ln in code.split("\n"):
        if ln.lstrip().startswith("//"):
            continue  # the DJ-flow banner quotes example voices in comments
        if "note(cat(" in ln:
            m = re.search(r'\.s\("([^"]+)"\)', ln)
            if m:
                return ln, m.group(1)
    raise AssertionError("deliverable has no pitched note(cat(...)) voice with .s()")


def _setcps_value(code: str) -> float | None:
    m = re.search(r"(?m)^\s*setcps\(\s*([\d.]+)\s*\)", code)
    if m:
        return float(m.group(1))
    m = re.search(r"(?m)^\s*setcps\(\s*(\w+)\s*\)", code)
    if m:
        d = re.search(r"(?m)^\s*(?:const|let)\s+%s\s*=\s*([\d.]+)" % re.escape(m.group(1)), code)
        if d:
            return float(d.group(1))
    return None


def _declared_bpm(code: str) -> float | None:
    m = re.search(r"//\s*BPM:\s*([\d.]+)", code) or re.search(r"([\d.]+)\s*BPM", code)
    return float(m.group(1)) if m else None


def _violations_text(res) -> str:
    return " | ".join(res.violations)


# ═════════════════════════════════════════════════════════════════════════════════════════
# §2.1  Editability — the deliverable is editable source
# ═════════════════════════════════════════════════════════════════════════════════════════

# AC: "exposes per-voice editable data ... one identifiable block per voice"
def test_21_every_voice_is_its_own_block_of_editable_data(deliverable):
    res = check_editability(deliverable.code)
    assert res.passed, res.violations
    assert res.generation_mode == deliverable.mode
    blocks = _blocks(deliverable.code)
    assert len(blocks) >= 3, "bass / lead / drums must each be an identifiable `$:` block"
    # every block carries at least one editable voice (no decorative or empty channel)
    for first, end in blocks:
        in_block = [v for v in res.editable_voices if first + 1 <= v.line_start <= end]
        assert in_block, f"`$:` block at line {first + 1} contains no editable voice"
    # pitched voices are fed by DECLARED, non-empty bar arrays
    decls = _array_decls(deliverable.code)
    used = {a for v in res.editable_voices for a in v.arrays}
    assert used, "no pitched voice is driven by a bar array"
    for name in used:
        assert decls.get(name), f"bar array {name!r} is referenced but not declared with bars"
        assert any(_NOTE_TOKEN.search(bar) for bar in decls[name]), f"{name!r} holds no notes"
    assert any(v.line_start and re.match(r"^bd|^sd|^hh", v.label) for v in res.editable_voices) or any(
        set(v.sounds) & DRUM_ONESHOTS for v in res.editable_voices
    ), "drums must be editable one-shot patterns"


# negative: a pitched voice fed by an UNDECLARED array is baked/opaque, not editable data
def test_21_undeclared_array_is_not_editable_data(deliverable):
    base = check_editability(deliverable.code)
    broken = deliverable.code.replace("...lead", "...ghost")
    assert broken != deliverable.code, "fixture has no lead array to break"
    res = check_editability(broken)
    assert len(res.editable_voices) == len(base.editable_voices) - 1
    only = '$: note(cat(...ghost)).s("sawtooth")\n'
    assert not check_editability(only).passed, "a voice with no declared data must not pass"


# AC "Editability Test": changing one note (c2 -> eb2) changes THAT position and nothing else.
# (Static proxy for the audio test: the live render proof is editability_test.py, tasks.md Slice 4.)
def test_21_single_note_edit_changes_exactly_one_token_and_stays_valid(deliverable):
    array = next(iter(a for v in check_editability(deliverable.code).editable_voices for a in v.arrays))
    edited, bar_idx, tok_idx = _edit_first_note(deliverable.code, array)
    assert edited != deliverable.code
    old_bars = _array_decls(deliverable.code)[array]
    new_bars = _array_decls(edited)[array]
    assert len(old_bars) == len(new_bars)
    changed = [(i, a, b) for i, (a, b) in enumerate(zip(old_bars, new_bars)) if a != b]
    assert [c[0] for c in changed] == [bar_idx], "exactly the targeted bar changed"
    _, a, b = changed[0]
    ta, tb = a.split(), b.split()
    assert len(ta) == len(tb)
    assert [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y] == [tok_idx], "exactly one token changed"
    # every other line of the file is byte-identical
    diff = [d for d in difflib.unified_diff(deliverable.code.split("\n"), edited.split("\n"), lineterm="", n=0)
            if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    assert len(diff) == 2, diff  # one removed line, one added line
    after = check_editability(edited)
    assert after.passed, after.violations
    assert len(after.editable_voices) == len(check_editability(deliverable.code).editable_voices)


# negative: a replayed voice has no note data at all, so the same edit has nothing to act on
def test_21_replay_voice_has_no_data_to_edit():
    code = (FIX / "v023_vocal_replay.strudel").read_text(encoding="utf-8")
    res = check_editability(code)
    assert not res.passed
    replay = [v for v in res.replay_voices if "vocalsfull" in v.sounds or "vocalsfull" in v.snippet]
    assert replay and all(not v.arrays for v in replay), "a replay voice must carry no editable array"
    with pytest.raises(LookupError):
        _edit_first_note(code, "vocalsfull")
    with pytest.raises(LookupError):
        _edit_first_note(code, "vocal")  # the removed vocal never became data either


# AC "Sound-swap Test": .s("X") -> .s("Y") re-voices the part, the pattern data is untouched
def test_21_sound_swap_revoices_but_keeps_the_pattern(deliverable):
    line, old = _first_pitched_instrument(deliverable.code)
    new = "gm_piano" if old != "gm_piano" else "gm_epiano1"
    assert new in VALID_SOUNDS
    swapped = deliverable.code.replace(line, line.replace(f'.s("{old}")', f'.s("{new}")', 1), 1)
    assert swapped != deliverable.code
    assert _array_decls(swapped) == _array_decls(deliverable.code), "pattern data must be unchanged"
    res = check_editability(swapped)
    assert res.passed, res.violations
    assert new in {s for v in res.editable_voices for s in v.sounds}
    assert validate_code(swapped, autocorrect=False)[1] == ""


# negative: "swapping" the instrument to a full-stem sound is a replay, rejected twice over
@pytest.mark.parametrize("replay_sound", ["originalfull", "bassfull", "origseg3", "melodicloop"])
def test_21_sound_swap_to_a_full_stem_replay_is_rejected(deliverable, replay_sound):
    line, old = _first_pitched_instrument(deliverable.code)
    swapped = deliverable.code.replace(line, line.replace(f'.s("{old}")', f'.s("{replay_sound}")', 1), 1)
    res = check_editability(swapped)
    assert not res.passed, f"{replay_sound} swap must fail the detector"
    assert "R2" in _violations_text(res)
    assert validate_code(swapped, autocorrect=False)[1].startswith("replay sound")


# AC "Structure Test": slice / duplicate / reorder bars (the user's own arrangement edits)
@pytest.mark.parametrize(
    "template",
    [
        "cat(...{n}.slice(0, 2), ...{n}.slice(0, 2))",   # duplicate
        "cat(...{n}.slice(1, 2), ...{n}.slice(0, 1))",   # reorder
        "cat(...{n}.slice(0, 1))",                       # truncate
    ],
    ids=["duplicate", "reorder", "truncate"],
)
def test_21_reordering_and_duplicating_bars_stays_a_valid_deliverable(deliverable, template):
    base = check_editability(deliverable.code)
    name = next(n for v in base.editable_voices for n in v.arrays)
    m = re.search(r"cat\(\.\.\.%s(?:\.slice\([^)]*\))?\)" % re.escape(name), deliverable.code)
    assert m, f"no cat(...{name}) call to rearrange"
    rearranged = deliverable.code.replace(m.group(0), template.format(n=name), 1)
    res = check_editability(rearranged)
    assert res.passed, res.violations
    assert "R1" not in _violations_text(res)
    assert len(res.editable_voices) == len(base.editable_voices)
    assert name in {a for v in res.editable_voices for a in v.arrays}


# AC FORBIDDEN: replay by any equivalent — rejected by the detector, the gate, and (where the
# generation-time validator covers the form) validate_code. "regardless of similarity score".
_REPLAY_FORMS = [
    pytest.param('$: s("originalfull").slice(78, run(78)).slow(78)', True, id="originalfull-slice-run-slow"),
    pytest.param("let N = 16\n$: s(\"drumsfull\").slice(N, run(N)).slow(N)", True, id="stem-full-identifier-N"),
    pytest.param('$: note(cat(...bass)).s("sawtooth").loopAt(4)', True, id="loopAt-on-an-editable-voice"),
    pytest.param('$: s("bassloop")', True, id="stem-loop-sound"),
    pytest.param('$: s("origseg3")', True, id="origseg-sound"),
    pytest.param('await samples("https://x.example/samples_stems.json")', False, id="stem-manifest-load"),
]


@pytest.mark.parametrize("snippet,validator_covers", _REPLAY_FORMS)
def test_21_forbidden_replay_forms_are_rejected_at_every_gate(deliverable, snippet, validator_covers, tmp_path):
    assert check_editability(deliverable.code).passed
    replayed = deliverable.code.rstrip("\n") + "\n" + snippet + "\n"
    res = check_editability(replayed)
    assert not res.passed, "replay must be rejected whatever the rest of the file looks like"
    assert any(v.rule in ("R1", "R2", "R4") for v in res.violation_details), res.violations
    if validator_covers:
        assert validate_code(replayed, autocorrect=False)[1], "generation-time validator must reject it too"
    # the gate refuses it even with a near-perfect score attached
    comp = tmp_path / "comparison.json"
    comp.write_text(json.dumps({**to_json_fields(res), "comparison": {"overall_similarity": 0.99,
                                                                      "worst_band_diff": 1.0}}))
    verdict = evaluate_comparison(comp, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not verdict.passed and "editability: fail" in verdict.message
    # counterpart: the SAME score on the untouched editable deliverable is gate-eligible
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({**to_json_fields(check_editability(deliverable.code)),
                              "comparison": {"overall_similarity": 0.99, "worst_band_diff": 1.0}}))
    ok_verdict = evaluate_comparison(ok, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert ok_verdict.editability == "pass" and "editability: fail" not in ok_verdict.message


# counterpart: the user's own bar slicing / repeating (encouraged by the spec) is NOT replay
def test_21_user_bar_slicing_is_not_replay(deliverable):
    name = next(n for v in check_editability(deliverable.code).editable_voices for n in v.arrays)
    extra = f'\n$: note(cat(...{name}.slice(0, 2))).s("triangle").gain(0.3)  // user layer\n'
    res = check_editability(deliverable.code.rstrip("\n") + extra)
    assert res.passed, res.violations
    assert "R1" not in _violations_text(res)


# ═════════════════════════════════════════════════════════════════════════════════════════
# §2.2  Live-codeability
# ═════════════════════════════════════════════════════════════════════════════════════════

# AC independent voices + "Mute Test": comment out each `$:` block in turn
def test_22_each_voice_can_be_commented_out_individually(deliverable):
    base = check_editability(deliverable.code)
    total = len(base.editable_voices)
    blocks = _blocks(deliverable.code)
    assert len(blocks) >= 3
    for first, end in blocks:
        muted = _comment_out(deliverable.code, first, end)
        res = check_editability(muted)  # raises ParseError if commenting broke the structure
        dropped = [v for v in base.editable_voices if first + 1 <= v.line_start <= end]
        assert dropped, "block carries no voice"
        assert res.passed, f"muting block at line {first + 1}: {res.violations}"
        assert len(res.editable_voices) == total - len(dropped)
        assert len(res.editable_voices) >= 1, "the others keep playing"
        assert validate_code(muted, autocorrect=False)[1] == "", "muted file must remain valid Strudel"


# negative: muting EVERYTHING leaves no music (R5), and half-commenting a multi-line block is
# caught as a structural break rather than silently accepted
def test_22_muting_everything_or_half_a_block_is_detected(deliverable):
    code = deliverable.code
    for first, end in reversed(_blocks(code)):
        code = _comment_out(code, first, end)
    silent = check_editability(code)
    assert not silent.passed and "R5" in _violations_text(silent)

    lines = deliverable.code.split("\n")
    multi = [(f, e) for f, e in _blocks(deliverable.code)
             if sum(1 for ln in lines[f:e] if ln.strip() and not ln.lstrip().startswith("//")) > 1]
    if not multi:
        pytest.skip("deliverable has no multi-line voice block")
    for first, _ in multi:
        half = _comment_out(deliverable.code, first, first + 1)  # only the `$: stack(` opener
        with pytest.raises(ParseError):
            check_editability(half)


# AC "plays without manual fixes": syntax + sound names valid at generation time
def test_22_output_passes_the_generation_time_validator(deliverable):
    fixed, err = validate_code(deliverable.code, autocorrect=False)
    assert err == "", err
    assert fixed == deliverable.code
    check_editability(deliverable.code)  # no ParseError == balanced brackets / terminated strings


# negative: hallucinated methods / sound names are caught, an unterminated string breaks parsing
@pytest.mark.parametrize(
    "bad",
    ['.volume(0.5)', '.peak(1)', '.sound("sub_bass")', '.bank("RolandTR9000")', '.sound("gm_pad_4_choir")'],
)
def test_22_invalid_code_is_caught_by_the_validator(deliverable, bad):
    mutated = deliverable.code.rstrip("\n") + f'\n$: s("bd sd"){bad}\n'
    assert validate_code(mutated, autocorrect=False)[1] != ""
    with pytest.raises(ParseError):
        check_editability('$: note(cat(...bass)).s("saw)\n')


# AC setcps(): present, and consistent with the BPM the file declares
def test_22_setcps_matches_the_declared_bpm(deliverable):
    cps = _setcps_value(deliverable.code)
    bpm = _declared_bpm(deliverable.code)
    assert cps is not None, "setcps() missing — patterns would not play at the track's tempo"
    assert bpm is not None, "no BPM declared in the header"
    assert cps * 60 * 4 == pytest.approx(bpm, abs=0.05), (cps, bpm)


# negative: a wrong cps, or no setcps at all, is detectable by the same helpers
def test_22_setcps_mismatch_or_absence_is_detected(deliverable):
    cps = _setcps_value(deliverable.code)
    bpm = _declared_bpm(deliverable.code)
    wrong = re.sub(r"(?m)^(\s*const\s+cps\s*=\s*)[\d.]+", r"\g<1>0.9", deliverable.code, count=1)
    assert wrong != deliverable.code
    assert _setcps_value(wrong) * 240 != pytest.approx(bpm, abs=0.05)
    stripped = re.sub(r"(?m)^\s*setcps\([^)]*\)\s*$", "", deliverable.code)
    assert _setcps_value(stripped) is None
    assert cps is not None


# generator: the requested --bpm drives setcps, in BOTH modes (needs the cached sample pack)
_gen = pytest.importorskip("test_generation_modes")
_PACK_PRESENT = all(p.exists() for p in _gen._REQUIRED)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    if not _PACK_PRESENT:
        pytest.skip("Regime CLT sample_pack not present in .cache")
    d = tmp_path_factory.mktemp("spec003_generated")
    out = {}
    for mode, bpm in (("sample-instrument", 100), ("synth", 120)):
        args = list(_gen.BASE_ARGS)
        args[args.index("--bpm") + 1] = str(bpm)
        path = d / f"{mode}.strudel"
        proc = subprocess.run(
            [sys.executable, str(_gen.GEN), *args, "--mode", mode, "--out", str(path)],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        out[mode] = (bpm, path.read_text(encoding="utf-8"))
    return out


@pytest.mark.parametrize("mode", ["sample-instrument", "synth"])
def test_22_generator_sets_cps_from_the_requested_bpm(generated, mode):
    bpm, code = generated[mode]
    assert _setcps_value(code) * 240 == pytest.approx(bpm, abs=0.01)
    assert _declared_bpm(code) == pytest.approx(bpm)
    res = check_editability(code)
    assert res.passed and res.generation_mode == mode, res.violations
    assert validate_code(code, autocorrect=False)[1] == ""
    # negative: the OTHER mode's tempo is not what this file plays at
    other_bpm = generated["synth" if mode == "sample-instrument" else "sample-instrument"][0]
    assert _setcps_value(code) * 240 != pytest.approx(other_bpm, abs=0.5)


# ═════════════════════════════════════════════════════════════════════════════════════════
# §2.3  Resemblance via re-performance
# ═════════════════════════════════════════════════════════════════════════════════════════

def _custom_sounds(res) -> set[str]:
    allowed = VALID_SOUNDS | DRUM_ONESHOTS
    return {s for v in res.editable_voices for s in v.sounds if s not in allowed}


# AC (b) + "if sample-instruments are used, notes are editable data"
def test_23_sample_instrument_realism_lives_in_declared_note_data(deliverable):
    if deliverable.mode != "sample-instrument":
        pytest.skip("synth deliverable")
    res = check_editability(deliverable.code)
    assert res.generation_mode == "sample-instrument"
    assert re.search(r"(?m)^\s*await\s+samples\(", deliverable.code), "stem-derived instruments are loaded"
    custom = _custom_sounds(res)
    assert custom, "sample-instrument output must play stem-derived instruments"
    decls = _array_decls(deliverable.code)
    for v in res.editable_voices:
        if set(v.sounds) & custom:
            assert v.arrays and all(decls.get(a) for a in v.arrays), (
                f"custom instrument {sorted(set(v.sounds) & custom)} is not driven by declared note data"
            )
    # no recording is replayed: every custom name is an INSTRUMENT, not a *full/*loop/origseg take
    assert not any(re.search(r"(full|loop)$|^origseg\d+$", s) for s in custom), custom
    # negative: pointing the same instrument at undeclared data drops it from the editable set
    ghosted = deliverable.code.replace("...lead", "...ghost")
    assert len(check_editability(ghosted).editable_voices) < len(res.editable_voices)


# AC (a): synthesized voices carry no stem material at all; the modes are distinguishable
def test_23_synth_mode_is_pure_synthesis(deliverable):
    res = check_editability(deliverable.code)
    if deliverable.mode == "synth":
        assert res.generation_mode == "synth"
        assert "samples(" not in deliverable.code
        assert not _custom_sounds(res), _custom_sounds(res)
    else:
        # negative: the sample-instrument deliverable is not mistaken for pure synthesis
        assert res.generation_mode != "synth" and "samples(" in deliverable.code and _custom_sounds(res)


_TEXTURE = '$: s("vocalsfull").slice(16, run(16)).slow(16).gain(0.4)  // texture'


# AC: per-bar loops only as texture under editable voices, user-arrangeable
def test_23_loops_are_texture_only_under_two_editable_voices(deliverable):
    base = check_editability(deliverable.code)
    assert len(base.editable_voices) >= 2
    with_tex = deliverable.code.rstrip("\n") + "\n" + _TEXTURE + "\n"
    res = check_editability(with_tex)
    assert res.passed, res.violations
    assert len(res.texture_voices) == 1 and len(res.editable_voices) == len(base.editable_voices)
    # user-arrangeable: it is its own `$:` block and can be muted away
    first, end = _blocks(with_tex)[-1]
    assert "// texture" in with_tex.split("\n")[first]
    muted = check_editability(_comment_out(with_tex, first, end))
    assert muted.passed and not muted.texture_voices

    # negatives: unmarked loop, and a marked loop over too few editable voices
    unmarked = deliverable.code.rstrip("\n") + "\n" + _TEXTURE.replace("  // texture", "") + "\n"
    r2 = check_editability(unmarked)
    assert not r2.passed and ("R1" in _violations_text(r2) or "R2" in _violations_text(r2))
    only = next(iter(_array_decls(deliverable.code)))
    one_voice = (
        "setcps(0.5)\nlet %s = [\"c2 ~ e2 ~\", \"f2 ~ g2 ~\"]\n" % only
        + '$: note(cat(...%s)).s("sawtooth")\n' % only
    )
    assert len(check_editability(one_voice).editable_voices) == 1
    r3 = check_editability(one_voice + _TEXTURE + "\n")
    assert not r3.passed and "R4" in _violations_text(r3)


# AC: "a loop-only output is not a deliverable" — even if every line claims to be texture
@pytest.mark.parametrize("fixture", ["output_loops.strudel", "v012_loop_replay.strudel"])
def test_23_a_loop_only_output_is_not_a_deliverable(fixture):
    code = (FIX / fixture).read_text(encoding="utf-8")
    res = check_editability(code)
    assert not res.passed
    marked = "\n".join(ln + "  // texture" if ln.lstrip().startswith("$:") else ln for ln in code.split("\n"))
    res2 = check_editability(marked)
    assert not res2.passed and "R5" in _violations_text(res2), "texture markers cannot launder a loop-only file"


# ═════════════════════════════════════════════════════════════════════════════════════════
# §2.4  Honest, valid similarity measurement
# ═════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def wavs(tmp_path_factory):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    d = tmp_path_factory.mktemp("spec003_wavs")
    sr, secs = 22050, 2.0
    t = np.arange(int(secs * sr)) / sr
    rng = np.random.default_rng(7)
    a = (0.30 * np.sin(2 * np.pi * 110 * t) + 0.10 * np.sin(2 * np.pi * 880 * t)
         + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    b = (0.30 * np.sin(2 * np.pi * 115 * t) + 0.08 * np.sin(2 * np.pi * 900 * t)
         + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    sf.write(str(d / "orig.wav"), a, sr)
    sf.write(str(d / "rend.wav"), b, sr)
    return d / "orig.wav", d / "rend.wav"


def _compare(wavs, strudel: Path, out: Path) -> subprocess.CompletedProcess:
    orig, rend = wavs
    return subprocess.run(
        [sys.executable, str(COMPARE), str(orig), str(rend), "-d", "2.0", "-j",
         "--strudel", str(strudel), "-o", str(out)],
        capture_output=True, text=True, timeout=240, cwd=str(SCRIPTS),
    )


_V023_REPLAYS = [
    pytest.param(FIX / "v023_vocal_replay.strudel", id="fixture-v023"),
    pytest.param(CACHE / "v023" / "output.strudel", id="cache-v023"),
]


# AC: similarity only on detector-passing output; replay refused and NEVER reported
@pytest.mark.parametrize("replay_path", _V023_REPLAYS)
def test_24_replay_is_refused_before_scoring_end_to_end(wavs, tmp_path, replay_path):
    if not replay_path.exists():
        pytest.skip(f"{replay_path} absent")
    out = tmp_path / "comparison.json"
    proc = _compare(wavs, replay_path, out)
    assert proc.returncode == 3, proc.stderr[-600:]
    text = out.read_text()
    assert "overall_similarity" not in text and "overall_similarity" not in proc.stdout
    payload = json.loads(text)
    assert payload["editability"] == "fail" and payload["comparison"] is None
    assert any("vocalsfull" in v for v in payload["editability_violations"])
    # the gate and the loop MCP both read the refusal as a FAIL
    verdict = evaluate_comparison(out, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not verdict.passed and verdict.message.startswith("FAIL [brazilian_funk] editability: fail")
    assert gate_main([str(out), "--genre", "brazilian_funk", "--mode", "sample-instrument"]) == 1
    server = pytest.importorskip("mcp_servers.loop.server")
    got = server.eval_gate(str(out), genre="brazilian_funk")
    assert got["ok"] and got["gate_passed"] is False and got["editability"] == "fail"


# positive counterpart: the same chain on editable output scores and stamps
@pytest.mark.parametrize("name", ["v023_minus_vocal.strudel", "synth_pass.strudel"])
def test_24_editable_output_is_scored_and_stamped_end_to_end(wavs, tmp_path, name):
    out = tmp_path / "comparison.json"
    proc = _compare(wavs, FIX / name, out)
    assert proc.returncode == 0, proc.stderr[-600:]
    data = json.loads(out.read_text())
    expect_mode = "synth" if name.startswith("synth") else "sample-instrument"
    assert data["editability"] == "pass" and data["generation_mode"] == expect_mode
    assert data["editability_violations"] == [] and data["editable_voice_count"] >= 2
    assert 0.0 <= data["comparison"]["overall_similarity"] <= 1.0
    verdict = evaluate_comparison(out, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert verdict.editability == "pass" and verdict.mode == expect_mode
    assert "editability: fail" not in verdict.message
    assert verdict.floor_source == f"modes.{expect_mode.replace('-', '_')}"


# AC: pipeline can confirm "generated, not replayed" — loop MCP stops BEFORE any render
def test_24_loop_mcp_refuses_replay_before_rendering(tmp_path, monkeypatch):
    server = pytest.importorskip("mcp_servers.loop.server")
    calls = []

    def fake_render(*a, **k):
        calls.append((a, k))
        return {"ok": False, "error": "stub-render"}

    monkeypatch.setattr(server, "render_strudel", fake_render)
    replay = FIX / "v023_vocal_replay.strudel"
    res = server.verify_strudel(str(replay), original=str(tmp_path / "orig.wav"), genre="brazilian_funk")
    assert res["ok"] is False and res["stage"] == "editability" and res["gate_passed"] is False
    assert res["editability"] == "fail" and any("vocalsfull" in v for v in res["editability_violations"])
    assert calls == [], "a replay file must never reach the recorder"
    # counterpart: an editable file passes the detector stage and reaches the renderer
    ok = server.verify_strudel(str(FIX / "v023_minus_vocal.strudel"), original=str(tmp_path / "o.wav"))
    assert ok["stage"] == "render" and len(calls) == 1


# AC: the iteration loop records replay as a failed iteration; unparseable code fails closed
def test_24_ai_improver_stamps_fail_closed():
    ai = pytest.importorskip("ai_improver")
    replay = (FIX / "v023_vocal_replay.strudel").read_text(encoding="utf-8")
    assert ai._editability_fields(replay)["editability"] == "fail"
    ok = ai._editability_fields((FIX / "v023_minus_vocal.strudel").read_text(encoding="utf-8"))
    assert ok["editability"] == "pass" and ok["generation_mode"] == "sample-instrument"
    broken = ai._editability_fields('$: note(cat(...bass)).s("saw)\n')
    assert broken["editability"] == "fail" and broken["editability_violations"][0].startswith("parse error")


# AC: "v023 is disqualified" and may not seed any floor or reference entry
def test_24_v023_is_disqualified():
    res = check_editability((FIX / "v023_vocal_replay.strudel").read_text(encoding="utf-8"))
    assert not res.passed
    r1 = [v for v in res.violation_details if v.rule == "R1"]
    assert r1 and r1[0].line == 196 and "vocalsfull" in (r1[0].voice or r1[0].message)
    assert any(v.rule == "R2" and v.line == 196 for v in res.violation_details)
    runs = [m.get("run", "") for blk in (THRESHOLDS.get("modes") or {}).values()
            for m in ((blk or {}).get("measured") or {}).values()]
    ref = yaml.safe_load((REPO / "eval" / "datasets" / "reference_tracks.yaml").read_text()) or {}
    versions = [t.get("version") for t in (ref.get("tracks") or [])]
    assert not any(r.endswith("/v023") for r in runs), runs
    assert "v023" not in versions, versions
    # positive counterpart: the same file minus the replay line IS eligible
    assert check_editability((FIX / "v023_minus_vocal.strudel").read_text(encoding="utf-8")).passed
    # and the real cached v023 (if present) is the same verdict
    real = CACHE / "v023" / "output.strudel"
    if real.exists():
        assert not check_editability(real.read_text(encoding="utf-8")).passed


# AC: two modes ship, sample-instrument default; `loops` is not a mode a user can pick
def test_24_two_modes_ship_and_loops_is_not_one_of_them(tmp_path):
    gen = SCRIPTS / "generate_dynamic_strudel.py"
    helptext = subprocess.run([sys.executable, str(gen), "--help"], capture_output=True, text=True, timeout=120)
    assert helptext.returncode == 0
    flat = re.sub(r"\s+", " ", helptext.stdout)
    m = re.search(r"--mode \{([^}]*)\}", flat)
    assert m, flat[:400]
    assert set(m.group(1).split(",")) == {"sample-instrument", "synth"}
    assert re.search(r"--mode.*?default:? ?sample-instrument", flat) or "(default" in flat
    bad = subprocess.run(
        [sys.executable, str(gen), "--mode", "loops", "--bpm", "120", "--stems-dir", str(tmp_path),
         "--out", str(tmp_path / "x.strudel")],
        capture_output=True, text=True, timeout=120,
    )
    assert bad.returncode == 2 and "invalid choice" in bad.stderr
    # both modes have a measured floor, distinct from each other, and each differs from nothing
    modes = THRESHOLDS["modes"]
    assert set(modes) >= {"sample_instrument", "synth"}
    assert modes["sample_instrument"]["genres"]["brazilian_funk"] > modes["synth"]["genres"]["brazilian_funk"]


def _resolve_run(run: str) -> Path:
    return REPO / ".cache" / "stems" / run


# AC: "targets are measured per mode"; every reported number reproducible from a real run
def _reproduces(measured: dict, comparison: dict, mode: str, margin_floor: float | None) -> list[str]:
    problems = []
    got = comparison["comparison"]["overall_similarity"]
    if abs(got - measured["overall"]) > 5e-4:
        problems.append(f"overall {measured['overall']} != run {got:.4f}")
    if comparison.get("generation_mode", "").replace("-", "_") != mode:
        problems.append(f"run mode {comparison.get('generation_mode')} != {mode}")
    if comparison.get("editability") != "pass":
        problems.append("run is not detector-passing")
    if margin_floor is not None and margin_floor != pytest.approx(round(measured["overall"] - measured["margin"], 2)):
        problems.append("floor != measured - margin")
    return problems


def test_24_mode_floors_are_reproducible_from_a_real_run():
    dataset = yaml.safe_load((REPO / "eval" / "datasets" / "reference_tracks.yaml").read_text())["tracks"]
    checked = 0
    for mode, block in THRESHOLDS["modes"].items():
        for genre, m in ((block or {}).get("measured") or {}).items():
            run_dir = _resolve_run(m["run"])
            floor = block["genres"][genre]
            assert floor == pytest.approx(round(m["overall"] - m["margin"], 2))
            assert m["editability"] == "pass"
            entry = [t for t in dataset
                     if f"{t['cache_key']}/{t['version']}" == m["run"] and t["genre"] == genre
                     and t["mode"].replace("-", "_") == mode]
            assert entry, f"no reference_tracks.yaml entry for measured run {m['run']}"
            comp_path = run_dir / "comparison.json"
            if not comp_path.exists():
                continue  # cache absent: the static linkage above still holds
            comp = json.loads(comp_path.read_text())
            assert _reproduces(m, comp, mode, floor) == []
            strudel = run_dir / "output.strudel"
            res = check_editability(strudel.read_text(encoding="utf-8"))
            assert res.passed and res.generation_mode.replace("-", "_") == mode, res.violations
            assert evaluate_comparison(comp_path, genre=genre, thresholds=THRESHOLDS).passed
            checked += 1
    if checked == 0:
        pytest.skip("cached v024/v025 renders absent — static linkage verified only")


# negative: a figure that no run supports, or a run of another mode, does not reproduce
def test_24_a_tampered_measurement_does_not_reproduce():
    run = _resolve_run("Regime CLT (Dj Brunin XM, Aurora Shukita)/v024/comparison.json")
    if not run.exists():
        pytest.skip("cached v024 absent")
    comp = json.loads(run.read_text())
    m = dict(THRESHOLDS["modes"]["sample_instrument"]["measured"]["brazilian_funk"])
    assert _reproduces(m, comp, "sample_instrument", None) == []
    inflated = {**m, "overall": m["overall"] + 0.05}   # an aspirational number
    assert any("overall" in p for p in _reproduces(inflated, comp, "sample_instrument", None))
    assert any("run mode" in p for p in _reproduces(m, comp, "synth", None))   # cross-mode claim
    replay = {**comp, "editability": "fail"}
    assert "run is not detector-passing" in _reproduces(m, replay, "sample_instrument", None)


# ═════════════════════════════════════════════════════════════════════════════════════════
# §2.5  Reporting reflects the values
# ═════════════════════════════════════════════════════════════════════════════════════════

def _report_module():
    return pytest.importorskip("generate_report")


def _headline(html: str) -> str | None:
    m = re.search(r'<div class="overall-headline"[^>]*>(.*?)</div>\s*</div>', html, re.S)
    return m.group(1) if m else None


# AC: headline from a generated render, naming the mode
@pytest.mark.parametrize(
    "run,mode",
    [("v024", "sample-instrument"), ("v025", "synth")],
)
def test_25_headline_states_the_generating_mode(run, mode):
    gr = _report_module()
    path = CACHE / run / "comparison.json"
    if not path.exists():
        pytest.skip(f"{path} absent")
    data = json.loads(path.read_text())
    html = gr.generate_charts_html(data)
    head = _headline(html)
    assert head is not None
    pct = round(data["comparison"]["overall_similarity"] * 100)
    assert f"{pct}% — mode: {mode} · editable: pass" in head
    assert gr.EDITABILITY_BADGE_TEXT not in html
    # negative: the wrong mode label never appears
    other = "synth" if mode == "sample-instrument" else "sample-instrument"
    assert f"mode: {other}" not in head


# AC: a replay-derived score is never the deliverable's quality — detector verdict flows into the page
def test_25_replay_score_never_reaches_the_report():
    gr = _report_module()
    high = json.loads((REPORT_FIX / "comparison_pass_sample_instrument.json").read_text())
    assert round(high["comparison"]["overall_similarity"] * 100) >= 90
    replay_code = (FIX / "v023_vocal_replay.strudel").read_text(encoding="utf-8")
    stamped = {**high, **to_json_fields(check_editability(replay_code))}   # a ~94% replay run
    html = gr.generate_charts_html(stamped)
    assert gr.EDITABILITY_BADGE_TEXT in html
    assert _headline(html) is None
    assert "Similarity Scores" not in html and "94%" not in html
    assert any(gr.html.escape(v) in html for v in stamped["editability_violations"])
    stems = json.loads((REPORT_FIX / "stem_comparison_min.json").read_text())
    stem_html = gr.generate_stem_comparison_html(stems, {}, editability="fail")
    assert gr.STEM_SECTION_CAPTION in stem_html
    assert not re.search(r"\d+%", stem_html), "per-stem scores withheld for replay"
    # counterpart: the detector-passing stamp on the same numbers shows the headline and caption
    ok = {**high, **to_json_fields(check_editability((FIX / "v023_minus_vocal.strudel").read_text()))}
    ok_html = gr.generate_charts_html(ok)
    assert gr.EDITABILITY_BADGE_TEXT not in ok_html
    assert "mode: sample-instrument · editable: pass" in (_headline(ok_html) or "")
    assert gr.STEM_SECTION_CAPTION in gr.generate_stem_comparison_html(stems, {}, editability="pass")


# AC (cross-implementation): the Go report states the same verdict wording as the Python one
def test_25_go_and_python_reports_agree():
    gr = _report_module()
    go = (REPO / "internal" / "report" / "generator.go").read_text(encoding="utf-8")
    assert gr.EDITABILITY_BADGE_TEXT in go
    assert gr.STEM_SECTION_CAPTION in go
    assert "— mode: %s · editable: pass" in go
    assert "editability: not checked (pre-spec-003 run)" in go
    # negative: wording that exists in neither implementation is absent from both
    assert "REPLAY / VERIFIED" not in go and "REPLAY / VERIFIED" not in gr.EDITABILITY_BADGE_TEXT
