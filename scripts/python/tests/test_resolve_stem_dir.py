"""Regression for the generalization bug: data-driven features silently skipped on FRESH separations
because stem_dir was derived from the piano path's parent (a workspace dir lacking the named stems).
_resolve_stem_dir must find the dir that actually contains bass/drums/melodic.wav."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen_orchestrator import _resolve_stem_dir  # noqa: E402


def _make_stems(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for s in ("bass.wav", "drums.wav", "melodic.wav"):
        (d / s).write_bytes(b"x")


def test_finds_cache_dir_when_piano_is_in_workspace(tmp_path):
    # the bug case: stems in the cache dir (= parent of the version output-dir),
    # but piano stem lives in an unrelated workspace dir.
    cache = tmp_path / "track"
    version = cache / "v030"
    version.mkdir(parents=True)
    _make_stems(cache)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "piano.wav").write_bytes(b"x")
    assert _resolve_stem_dir(str(workspace / "piano.wav"), str(version)) == str(cache)


def test_finds_when_piano_parent_has_stems(tmp_path):
    cache = tmp_path / "track"
    _make_stems(cache)
    assert _resolve_stem_dir(str(cache / "melodic.wav"), str(cache / "v1")) == str(cache)


def test_prefers_piano_parent_if_it_has_stems(tmp_path):
    # if BOTH piano-parent and output-parent have stems, piano-parent wins (first candidate)
    cache = tmp_path / "track"
    _make_stems(cache)
    assert _resolve_stem_dir(str(cache / "melodic.wav"), str(cache / "v1")) == str(cache)


def test_falls_back_to_piano_parent_when_no_stems_anywhere(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "piano.wav").write_bytes(b"x")
    out = tmp_path / "out" / "v1"
    out.mkdir(parents=True)
    assert _resolve_stem_dir(str(ws / "piano.wav"), str(out)) == str(ws)


def test_none_piano_and_output():
    assert _resolve_stem_dir(None, None) is None
