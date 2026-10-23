# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""Bass notes the stem plays WITHOUT a new attack must not re-strike the sample (2026-10-23).

Measured on Regime CLT: the sample bass lost to a synth with the SAME written notes (mix 0.875 vs 0.899) because of
the 1-step notes the stem does not re-attack (a pitch change inside its continuing 808 tone): the sample was
re-triggered from its start, so each 111 ms step played only the grain's transient / pitch sweep / one-period
fade-in (frame-wise chroma corr 0.55 vs 0.80 for the synth; 36 % of those frames < 0.5). Those notes are now their own
editable array ``bassLegato``, entered at the grain's loop (its steady part) with a one-period click guard.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_bass_legato_entry.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "tests"))

from test_bass_shape import PIECES, _generate, regime, vag  # noqa: E402


REGIME_SOUND = "regime_clt_dj_brunin_xm_aurora_shukita"   # the instrument name samples.json really holds


def _array(code: str, name: str) -> list[str]:
    body = code.split(f"let {name} = [")[1].split("\n]")[0]
    return re.findall(r"[a-g]s?-?\d+(?:@\d+)?", body)


@regime
def test_notes_without_a_stem_attack_are_a_separate_loop_entered_voice(tmp_path, monkeypatch):
    monkeypatch.setitem(PIECES["regime"], "sound", REGIME_SOUND)
    code, summary = _generate(tmp_path)
    legato = _array(code, "bassLegato")
    struck = _array(code, "bass")
    assert legato, "the stem has pitch changes without a re-attack: they must be written as bassLegato"
    assert len(struck) + len(legato) == summary["bass_shape"]["notes"]          # a split, nothing lost or invented
    assert "const bassLegatoEntry = " in code and "begin: l[0]" in code          # entered at the loop, not the grain start
    assert "bassLegatoEntry(note(cat(...bassLegato)))" in code
    assert "stack(note(cat(...bass)), note(cat(...bassLegato)))" in code         # the sine sub layer still plays every note
    assert summary["editability"] == "pass"


@regime
def test_synth_and_unshaped_bass_have_no_legato_voice(tmp_path):
    code, _ = _generate(tmp_path, mode="synth")
    assert "bassLegato" not in code                                              # no sample, no attack to avoid
    code, _ = _generate(tmp_path, "--bass-shape", "off", "--bass-duck", "off")
    assert "bassLegato" not in code


@vag
def test_every_written_bass_note_is_still_played_once(tmp_path):
    code, summary = _generate(tmp_path, piece="vag")
    n = len(_array(code, "bass")) + (len(_array(code, "bassLegato")) if "let bassLegato" in code else 0)
    assert n == summary["bass_shape"]["notes"]
