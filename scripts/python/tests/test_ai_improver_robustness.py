# @layer: unit
# @spec: 001-core-pipeline
# @regression
"""ai_improver: the instrumental reference comes from the STEM dir (not the input's folder); a renderer that
exits non-zero is never scored; renderer/pipeline_helpers are imported once at module level."""
from __future__ import annotations

import inspect
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import ai_improver  # noqa: E402


def _stems(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name in ("bass", "drums", "melodic", "vocals", "original"):
        soundfile.write(str(d / f"{name}.wav"), (0.1 * np.ones((2205, 2))).astype("float32"), 44100, subtype="FLOAT")


# --- #2 reference built from the stem dir ---------------------------------------------------------------
def test_reference_is_built_in_the_stem_dir_when_the_input_is_elsewhere(tmp_path: Path) -> None:
    user = tmp_path / "Music"
    user.mkdir()
    (user / "song.mp3").write_bytes(b"x")                # cache miss: Go handed over the user's own file
    track = tmp_path / "cache" / "stems" / "song"
    _stems(track)
    version = track / "v001"
    version.mkdir()
    got = ai_improver.build_instrumental_reference(str(user / "song.mp3"), str(version))
    assert Path(got) == track.resolve() / "instrumental.wav" and Path(got).is_file()
    assert not (user / "instrumental.wav").exists(), "must never write into the user's folder"


def test_reference_next_to_a_cached_original_still_works(tmp_path: Path) -> None:
    track = tmp_path / "t"
    _stems(track)
    got = ai_improver.build_instrumental_reference(str(track / "original.wav"), str(track / "v002"))
    assert Path(got) == track.resolve() / "instrumental.wav"


def test_reference_fails_loudly_when_no_stems_exist_anywhere(tmp_path: Path) -> None:
    user = tmp_path / "Music"
    user.mkdir()
    (user / "song.wav").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="no bass/drums/melodic stems"):
        ai_improver.build_instrumental_reference(str(user / "song.wav"), str(tmp_path / "out" / "v001"))
    assert not (user / "instrumental.wav").exists()


# --- #1 renderer exit status ----------------------------------------------------------------------------
def _stub(tmp_path: Path, body: str) -> Path:
    s = tmp_path / "rec.sh"
    s.write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n' + body)
    s.chmod(s.stat().st_mode | stat.S_IEXEC)
    return s


def test_render_that_exits_nonzero_is_not_scored_even_though_a_file_exists(tmp_path: Path) -> None:
    rec = _stub(tmp_path, 'echo wav > "$out"; echo "silent render" >&2; exit 3\n')
    wav = tmp_path / "render_v001.wav"
    assert ai_improver._render_candidate(rec, tmp_path / "x.strudel", wav, 5, node="/bin/bash") is False
    assert not wav.exists(), "the half-checked file must be deleted so nothing can score it"


def test_render_success_and_stale_file_removal(tmp_path: Path) -> None:
    wav = tmp_path / "render_v001.wav"
    wav.write_text("stale from an earlier attempt")
    rec = _stub(tmp_path, "exit 1\n")
    assert ai_improver._render_candidate(rec, tmp_path / "x.strudel", wav, 5, node="/bin/bash") is False
    assert not wav.exists()
    rec = _stub(tmp_path, 'echo fresh > "$out"\n')
    assert ai_improver._render_candidate(rec, tmp_path / "x.strudel", wav, 5, node="/bin/bash") is True
    assert wav.read_text().strip() == "fresh"


# --- #10 module-level imports ---------------------------------------------------------------------------
def test_no_sys_path_mutation_or_import_inside_the_iteration_loop() -> None:
    body = inspect.getsource(ai_improver.improve_strudel)
    assert "sys.path.insert" not in body and "from renderer import" not in body
    import renderer
    assert ai_improver.renderer_script is renderer.renderer_script
