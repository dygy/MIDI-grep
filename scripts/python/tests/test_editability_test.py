# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Editability Test as code (spec 003 §4 E2E, tasks.md Slice 4) — the pure parts.

``editability_test.py`` changes ONE note in ``bass[0]`` of a sample-instrument output,
re-renders both files for N bars and asserts the difference signal rises above the noise floor
ONLY in the edited bar window. The render needs BlackHole; these tests cover the edit step, the
BPM/duration plumbing and the localisation math on synthetic signals — no audio devices.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_editability_test.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from editability_test import (  # noqa: E402
    EditError,
    bars_to_seconds,
    edit_one_note,
    localise,
    parse_bpm,
    per_bar_rms,
    transpose_token,
)

CLI = SCRIPTS / "editability_test.py"

CODE = (
    "// generation_mode: sample-instrument\n"
    "const cps = 0.566667   // 136 BPM, 4 beats/bar\n"
    "setcps(cps)\n"
    'await samples("https://x.invalid/samples.json")\n'
    "\n"
    "let bass = [\n"
    '  "~ ~ e2 c2 ~ ~ e2 ds2 ds2 e2 ~ ~ ~ ~ ~ ~",\n'
    '  "~ e2 ~ b2 g2 ~ c3 ~ ds2 ~ ~ b2 ~ b2 ~ cs2"\n'
    "]\n"
    "\n"
    "let lead = [\n"
    '  "e4 ~ g4 ~",\n'
    '  "~ e2 ~ b2"\n'
    "]\n"
    "\n"
    '$: note(cat(...bass)).s("regime_bass").gain(0.5)  // BASS\n'
    '$: note(cat(...lead)).s("regime_lead")  // LEAD\n'
)


# ── transpose ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tok,semis,expected", [
    ("cs2", 3, "e2"),
    ("a2", 3, "c3"),       # wraps the octave
    ("e2", 3, "g2"),
    ("b1", 1, "c2"),
    ("c#2", 3, "e2"),      # sharp spelled with '#'
    ("eb2", 3, "fs2"),     # flat spelled with 'b'
    ("c2", -1, "b1"),
])
def test_transpose_token(tok, semis, expected):
    assert transpose_token(tok, semis) == expected


def test_transpose_rejects_non_note():
    with pytest.raises(EditError):
        transpose_token("~", 3)
    with pytest.raises(EditError):
        transpose_token("bd", 3)


# ── edit step ─────────────────────────────────────────────────────────────────────────────
def test_edit_changes_exactly_the_first_non_rest_token_of_bar0():
    res = edit_one_note(CODE)
    assert res.bar == 0
    assert res.step == 2                 # "~ ~ e2 ..." → the third step
    assert res.old_token == "e2" and res.new_token == "g2"
    assert res.new_bar == "~ ~ g2 c2 ~ ~ e2 ds2 ds2 e2 ~ ~ ~ ~ ~ ~"
    # exactly one line differs and only that token on it
    old_lines, new_lines = CODE.splitlines(), res.code.splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    assert len(changed) == 1
    a, b = changed[0]
    assert a.split() != b.split() and sum(x != y for x, y in zip(a.split(), b.split())) == 1
    # the lead array and the voices are untouched
    assert 'let lead = [\n  "e4 ~ g4 ~",\n  "~ e2 ~ b2"\n]' in res.code
    assert '$: note(cat(...bass)).s("regime_bass")' in res.code


def test_edit_can_target_another_bar_and_interval():
    res = edit_one_note(CODE, bar=1, semitones=-3)
    assert res.bar == 1 and res.step == 1
    assert res.old_token == "e2" and res.new_token == "cs2"
    assert res.new_bar.startswith("~ cs2 ~ b2")


def test_edit_advances_to_first_bar_with_a_note_when_bar0_is_rests():
    code = CODE.replace('"~ ~ e2 c2 ~ ~ e2 ds2 ds2 e2 ~ ~ ~ ~ ~ ~"', '"~ ~ ~ ~"')
    res = edit_one_note(code)
    assert res.requested_bar == 0 and res.bar == 1
    assert res.old_token == "e2" and res.new_token == "g2"


def test_edit_fails_loudly_without_the_array_or_without_notes():
    with pytest.raises(EditError):
        edit_one_note(CODE, array="drums")
    with pytest.raises(EditError):
        edit_one_note(CODE.replace("e2", "~").replace("c2", "~").replace("ds2", "~")
                      .replace("b2", "~").replace("g2", "~").replace("c3", "~").replace("cs2", "~"))


# ── bpm / duration ────────────────────────────────────────────────────────────────────────
def test_parse_bpm_from_const_cps_and_literal():
    assert parse_bpm(CODE) == pytest.approx(136.0, abs=0.01)
    assert parse_bpm("setcps(0.5)\n") == pytest.approx(120.0)
    assert parse_bpm("// nothing\n$: s('bd')\n") is None


def test_bars_to_seconds():
    assert bars_to_seconds(8, 120.0) == pytest.approx(16.0)
    assert bars_to_seconds(8, 136.0) == pytest.approx(8 * 240 / 136)


# ── localisation math ─────────────────────────────────────────────────────────────────────
SR = 8000
BPM = 120.0            # 2 s per bar
NB = 8


def _tone(freq: float, n: int, amp: float = 0.3) -> np.ndarray:
    """A note-like signal: a tone re-struck every 8th note with an exponential decay (real
    renders have onsets; a steady sine has no envelope to align on)."""
    t = np.arange(n) / SR
    eighth = 60.0 / BPM / 2
    env = np.exp(-(t % eighth) / (eighth * 0.35))
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _two_renders(edited_bar: int | None, *, everywhere: bool = False, noise: float = 1e-3):
    rng = np.random.default_rng(0)
    bar_n = int(SR * 240 / BPM)
    a = _tone(110.0, bar_n * NB)
    b = a.copy()
    if everywhere:
        b = _tone(123.47, bar_n * NB)
    elif edited_bar is not None:
        s = edited_bar * bar_n
        b[s:s + bar_n] = _tone(130.81, bar_n)   # minor third up in one bar
    # independent recorder noise on each take
    a = a + rng.normal(0, noise, a.size).astype(np.float32)
    b = b + rng.normal(0, noise, b.size).astype(np.float32)
    return a, b


def test_per_bar_rms_shape_and_values():
    y = np.concatenate([np.zeros(SR * 2), np.ones(SR * 2) * 0.5]).astype(np.float32)
    r = per_bar_rms(y, SR, BPM, 2)
    assert len(r) == 2
    assert r[0] == pytest.approx(0.0, abs=1e-6) and r[1] == pytest.approx(0.5, abs=1e-3)


def test_localised_difference_in_bar0_is_detected():
    a, b = _two_renders(0)
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 1))
    assert v["localised"] is True
    assert len(v["diff_rms_per_bar"]) == NB
    assert v["diff_rms_per_bar"][0] > v["noise_floor"]
    assert all(x <= v["noise_floor"] for x in v["diff_rms_per_bar"][1:])


def test_difference_outside_the_window_is_not_localised():
    a, b = _two_renders(3)
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 1))
    assert v["localised"] is False
    assert v["offending_bars"] == [3]


def test_difference_everywhere_is_not_localised():
    a, b = _two_renders(None, everywhere=True)
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 1))
    assert v["localised"] is False


def test_no_difference_at_all_is_not_localised():
    a, b = _two_renders(None)
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 1))
    assert v["localised"] is False


def test_tail_bar_inside_window_is_tolerated_and_lengths_are_aligned():
    a, b = _two_renders(0)
    bar_n = int(SR * 240 / BPM)
    # release tail bleeding into bar 1 + takes of different length
    b[bar_n:bar_n + bar_n // 4] += _tone(130.81, bar_n // 4, amp=0.1)
    b = np.concatenate([b, np.zeros(SR, dtype=np.float32)])
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 2))
    assert v["localised"] is True
    assert v["bar_window"] == [0, 2]


def test_small_recorder_offset_is_realigned():
    a, b = _two_renders(0)
    b = np.concatenate([np.zeros(int(SR * 0.05), dtype=np.float32), b])  # 50 ms late start
    v = localise(a, b, SR, BPM, NB, bar_window=(0, 2), max_lag_s=0.2)
    assert abs(v["lag_s"] - 0.05) < 0.005
    assert v["localised"] is True


# ── CLI dry-run ───────────────────────────────────────────────────────────────────────────
def test_cli_no_render_writes_edited_file_and_prints_verdict(tmp_path):
    src = tmp_path / "in.strudel"
    src.write_text(CODE)
    out = tmp_path / "edited.strudel"
    proc = subprocess.run([sys.executable, str(CLI), str(src), "--no-render", "--edited-out", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "- " in proc.stdout and "+ " in proc.stdout   # the printed diff
    verdict = json.loads(proc.stdout[proc.stdout.rindex("{"):])
    assert verdict["edited"].startswith("bass[0]")
    assert "e2" in verdict["edited"] and "g2" in verdict["edited"]
    assert verdict["bar_window"] == [0, 2]
    assert verdict["rendered"] is False and verdict["localised"] is None
    assert verdict["bpm"] == pytest.approx(136.0, abs=0.01)
    assert verdict["render_seconds"] == pytest.approx(8 * 240 / verdict["bpm"])
    assert verdict["render_seconds"] == pytest.approx(8 * 240 / 136, rel=1e-4)


def test_cli_exit_2_when_nothing_to_edit(tmp_path):
    src = tmp_path / "in.strudel"
    src.write_text("setcps(0.5)\n$: s('bd sd')\n")
    proc = subprocess.run([sys.executable, str(CLI), str(src), "--no-render"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
