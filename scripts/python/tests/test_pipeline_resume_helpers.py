# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""pipeline_helpers: control lookup skips stem-less candidates; best.params format check; newer-than stamp."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import pipeline_helpers as ph  # noqa: E402

PH = SCRIPTS / "pipeline_helpers.py"


def _stems(d: Path, n: int = 4410) -> None:
    for name in ("original", "bass", "drums", "melodic", "vocals"):
        soundfile.write(str(d / f"{name}.wav"), (0.1 * np.ones((n, 2))).astype("float32"), 44100, subtype="FLOAT")


def _repo(tmp_path: Path, keys: str) -> Path:
    (tmp_path / "eval" / "datasets").mkdir(parents=True)
    rows = "".join(f"  - {{genre: g, cache_key: {k}}}\n" for k in keys)
    (tmp_path / "eval" / "datasets" / "reference_tracks.yaml").write_text("tracks:\n" + rows)
    return tmp_path


def _track(root: Path, key: str, *, stems: bool) -> Path:
    d = root / ".cache" / "stems" / key
    d.mkdir(parents=True)
    if stems:
        _stems(d)
    else:
        (d / "original.wav").write_bytes(b"RIFF")
    return d


# --- #6 control lookup ---------------------------------------------------------------------------------
def test_control_skips_a_candidate_without_stems_and_uses_the_next(tmp_path: Path) -> None:
    root = _repo(tmp_path, "ABC")
    _track(root, "A", stems=True)
    _track(root, "B", stems=False)       # original.wav only: the old code raised here and killed the lookup
    _track(root, "C", stems=True)
    assert ph.control_track_path("g", "A", repo_root=root) == root / ".cache" / "stems" / "C" / "instrumental.wav"


def test_control_fails_when_candidates_exist_but_none_has_stems(tmp_path: Path) -> None:
    root = _repo(tmp_path, "AB")
    _track(root, "A", stems=True)
    _track(root, "B", stems=False)
    with pytest.raises(FileNotFoundError, match="B"):
        ph.control_track_path("g", "A", repo_root=root)


def test_control_is_none_when_there_is_no_other_candidate(tmp_path: Path) -> None:
    root = _repo(tmp_path, "A")
    _track(root, "A", stems=True)
    assert ph.control_track_path("g", "A", repo_root=root) is None


# --- #5 best.params format -----------------------------------------------------------------------------
@pytest.mark.parametrize("text,ok", [
    ("0.4 0.7 1.0 5000 0.1 0.6", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 422", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 422 300\n", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300", False),   # pre-branch: ... MG CV BL LH
    ("0.4 0.7 1.0", False),
    ("", False),
    ("a b c d e f", False),
])
def test_params_valid(text: str, ok: bool) -> None:
    assert ph.params_valid(text) is ok


def test_params_valid_cli(tmp_path: Path) -> None:
    f = tmp_path / "best.params"
    f.write_text("0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300\n")
    run = lambda p: subprocess.run([sys.executable, str(PH), "params-valid", str(p)], capture_output=True, text=True).stdout.strip()  # noqa: E731
    assert run(f) == "stale" and run(tmp_path / "missing") == "stale"
    f.write_text("0.4 0.7 1.0 5000 0.1 0.6 422 300\n")
    assert run(f) == "ok"


# --- #3 newer-than stamp -------------------------------------------------------------------------------
def test_newer_than(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "instruments").mkdir(parents=True)
    old = pack / "instruments" / "bass.wav"
    old.write_bytes(b"x")
    stamp = pack / ".uploaded"
    stamp.write_text("base")
    os.utime(old, (1_000, 1_000))
    os.utime(stamp, (2_000, 2_000))
    assert not ph.newer_than(stamp, pack)             # the stamp itself never counts
    (pack / "instruments" / "lead.wav").write_bytes(b"y")
    os.utime(pack / "instruments" / "lead.wav", (3_000, 3_000))   # rebuilt after the upload
    assert ph.newer_than(stamp, pack)
    assert ph.newer_than(tmp_path / "no-stamp", pack)             # never uploaded
