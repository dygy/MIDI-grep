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


def _labels(tmp_path: Path, key: str, genre: str) -> Path:
    p = tmp_path / "genre_labels.yaml"
    p.write_text(f'labels:\n  - cache_key: "{key}"\n    genre: {genre}\n    source: owner\n    note: test\n')
    return p


def _stamped(tmp_path: Path, cache_key: str, **meta) -> Path:
    d = tmp_path / cache_key
    d.mkdir()
    (d / "metadata.json").write_text(__import__("json").dumps(meta))
    return d


def test_a_label_added_after_the_stamp_is_stale(tmp_path: Path) -> None:
    # Automotivo Holiday 2026-10-31: stamped 'detector-uncertain' / unreferenced, labelled brazilian_funk afterwards.
    d = _stamped(tmp_path, "Holiday", genre="brazilian_funk", genre_basis="detector-uncertain", genre_ref="unreferenced")
    assert ph.label_stale(d, _labels(tmp_path, "Holiday", "brazilian_funk"))


def test_a_label_that_changes_the_genre_is_stale(tmp_path: Path) -> None:
    d = _stamped(tmp_path, "T", genre="brazilian_funk", genre_basis="label", genre_ref="brazilian_funk")
    assert ph.label_stale(d, _labels(tmp_path, "T", "retro_wave"))


def test_a_stamp_from_the_label_an_unlabelled_track_and_an_override_are_fresh(tmp_path: Path) -> None:
    lab = _labels(tmp_path, "T", "brazilian_funk")
    assert not ph.label_stale(_stamped(tmp_path, "T", genre="brazilian_funk", genre_basis="label"), lab)
    assert not ph.label_stale(_stamped(tmp_path, "Other", genre="x", genre_basis="knn"), lab)
    lab2 = _labels(tmp_path, "O", "brazilian_funk")
    assert not ph.label_stale(_stamped(tmp_path, "O", genre="trance", genre_basis="override"), lab2)


def test_label_stale_cli_prints_the_verdict(tmp_path: Path) -> None:
    d = _stamped(tmp_path, "Holiday", genre="brazilian_funk", genre_basis="knn")
    out = subprocess.run([sys.executable, str(PH), "label-stale", str(d), "--labels", str(_labels(tmp_path, "Holiday", "brazilian_funk"))],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "stale"


def test_a_genre_only_restamp_keeps_the_refined_tempo(tmp_path: Path) -> None:
    # A re-stamp re-refines the bpm by the last digit (VAGABUNDO 129.989 -> 129.988) and every stage built at the old
    # bpm would rebuild; a label added later is about genre alone.
    d = _stamped(tmp_path, "VAG", bpm=129.989, bpm_method="phase-coherence", bpm_coarse=129.2, key="C# minor",
                 genre="brazilian_funk", duration=151.38)
    (d / "v001").mkdir()
    (d / "v001" / "metadata.json").write_text('{"bpm": 129.2, "key": "D minor", "style": "trance"}')
    m = ph.stamp_track_metadata(d, labels_path=_labels(tmp_path, "VAG", "retro_wave"), keep_tempo=True)
    assert (m["bpm"], m["key"], m["bpm_method"]) == (129.989, "C# minor", "phase-coherence")
    assert (m["genre"], m["genre_basis"], m["genre_ref"]) == ("retro_wave", "label", "retro_wave")
