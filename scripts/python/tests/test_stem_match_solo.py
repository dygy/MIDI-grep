"""Tests for stem_match's solo-drums wiring: the drum voice is extracted cleanly so it can be
rendered in isolation (no demucs confound) for the drums verdict."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_match import _extract_drum_voice, _solo_drums_path  # noqa: E402

SAMPLE = """// header
setcps(0.5)

$: arrange(
  [4, note("c2 e2").sound("gm_synth_bass_1").gain(0.9)]
).gain("<0.1 0.9>")

$: arrange(
  [4, note("f4 a4").sound("gm_clarinet").gain(0.85)]
).gain("<0.3 0.6>")

$: arrange(
  [4, s("<[bd hh sd hh] [bd ~ sd ~]>").bank("RolandTR808").gain(1.0)]
).gain("<0.0 0.9>")
"""


def test_extract_drum_voice_keeps_only_drums(tmp_path):
    f = tmp_path / "output.strudel"
    f.write_text(SAMPLE)
    voice = _extract_drum_voice(f)
    assert voice is not None
    assert "setcps(0.5)" in voice          # preamble preserved
    assert ".bank(" in voice                # the drum voice
    assert voice.count("$:") == 1           # ONLY the drum block
    assert "gm_synth_bass_1" not in voice   # bass dropped
    assert "gm_clarinet" not in voice       # lead dropped


def test_extract_drum_voice_none_when_no_drums(tmp_path):
    f = tmp_path / "output.strudel"
    f.write_text('setcps(0.5)\n\n$: note("c2 e2").sound("gm_synth_bass_1")\n')
    assert _extract_drum_voice(f) is None


def test_extract_drum_voice_missing_file(tmp_path):
    assert _extract_drum_voice(tmp_path / "nope.strudel") is None


def test_solo_drums_path_prefers_wav(tmp_path):
    assert _solo_drums_path(tmp_path) is None
    (tmp_path / "render_drums_solo.mp3").write_bytes(b"x")
    assert _solo_drums_path(tmp_path).name == "render_drums_solo.mp3"
    (tmp_path / "render_drums_solo.wav").write_bytes(b"x")
    assert _solo_drums_path(tmp_path).name == "render_drums_solo.wav"  # wav preferred
