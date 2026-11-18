"""A decided tempo (autopilot trial / editable-pipeline.sh --bpm) survives every later stamp.

Regression (BibiBabydoll, 2026-11-08): the extractor's tempogram locked onto a dotted subdivision (112.5 BPM,
phase-coherence 0.045; the drums are coherent at 150, 0.368). A fix that the next re-stamp silently reverted to the
extraction's estimate would be no fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline_helpers as PH  # noqa: E402

SR = 22050


def _track(tmp: Path, bpm: float) -> Path:
    t = tmp / "trk"
    (t / "v001").mkdir(parents=True)
    n = int(40 * SR)
    y = np.zeros(n)
    click = np.hanning(256) * 0.8
    for k in range(int(40 * bpm / 60)):
        i = int(k * 60 / bpm * SR)
        y[i:i + click.size] += click[: max(0, min(click.size, n - i))]
    sf.write(str(t / "drums.wav"), y.astype(np.float32), SR)
    (t / "metadata.json").write_text(json.dumps({"bpm": 112.5, "bpm_method": "x"}))
    (t / "v001" / "metadata.json").write_text(json.dumps({"bpm": 112.5, "key": "F minor"}))
    return t


def test_set_bpm_refines_around_the_decision_and_records_it(tmp_path):
    t = _track(tmp_path, 150.0)
    m = PH.set_decided_bpm(t, 150.0, by="test")
    assert abs(m["bpm"] - 150.0) < 0.2 and m["bpm_decided"]["bpm_coarse"] == 150.0
    assert json.loads((t / "metadata.json").read_text())["bpm_decided"]["by"] == "test"


def test_a_later_full_stamp_keeps_the_decided_tempo(tmp_path, monkeypatch):
    t = _track(tmp_path, 150.0)
    PH.set_decided_bpm(t, 150.0, by="test")
    labels = tmp_path / "labels.yaml"
    labels.write_text("{}\n")
    m = PH.stamp_track_metadata(t, genre_override="brazilian_funk", labels_path=labels)
    assert abs(m["bpm"] - 150.0) < 0.2           # not the extraction's 112.5 from v001/metadata.json


def _wipe_like_the_go_cache(t: Path, extraction_bpm: float = 112.5) -> None:
    """What stage 1 used to leave after a re-separation: metadata.json rewritten with the cache's four fields, and a
    fresh vNNN carrying the extraction's tempo estimate."""
    (t / "metadata.json").write_text(json.dumps({"title": "trk", "url": "u", "video_id": "v"}))
    (t / "v002").mkdir()
    (t / "v002" / "metadata.json").write_text(json.dumps({"bpm": extraction_bpm, "key": "F minor"}))


def test_the_decision_survives_a_rewritten_metadata_json(tmp_path):
    # Regression 2026-11-18 (owner: "the bpm setting is forgotten, restarts run at the wrong bpm"): bpm_decided lived only
    # in metadata.json; the Go cache rewrote that file and the next stamp went back to the extraction's 112.5.
    t = _track(tmp_path, 150.0)
    reg = tmp_path / "tempo_decisions.yaml"
    labels = tmp_path / "labels.yaml"
    labels.write_text("{}\n")
    PH.set_decided_bpm(t, 150.0, by="test", decisions_path=reg)
    _wipe_like_the_go_cache(t)
    m = PH.stamp_track_metadata(t, genre_override="brazilian_funk", labels_path=labels, decisions_path=reg)
    assert abs(m["bpm"] - 150.0) < 0.2 and m["bpm_decided"]["bpm_coarse"] == 150.0


def test_apply_restores_a_lost_decision_and_keeps_a_current_one(tmp_path):
    t = _track(tmp_path, 150.0)
    reg = tmp_path / "tempo_decisions.yaml"
    PH.set_decided_bpm(t, 150.0, by="test", decisions_path=reg)
    assert PH.apply_decided_bpm(t, reg)["status"] == "kept"
    (t / "metadata.json").write_text(json.dumps({"bpm": 112.5, "bpm_method": "x", "bpm_coarse": 112.5}))
    r = PH.apply_decided_bpm(t, reg)
    assert r["status"] == "applied" and abs(r["bpm"] - 150.0) < 0.2 and r["was"] == 112.5
    assert PH.apply_decided_bpm(t, reg)["status"] == "kept"
    other = _track(tmp_path / "o", 150.0).rename(tmp_path / "o" / "another")
    assert PH.apply_decided_bpm(other, reg)["status"] == "none"


def test_a_trial_copy_is_never_recorded(tmp_path):
    import tempo_decisions as TD
    reg = tmp_path / "tempo_decisions.yaml"
    t = _track(tmp_path, 150.0)
    trial = t.rename(t.parent / "trk ~trial 150-BPM")
    PH.set_decided_bpm(trial, 150.0, by="autopilot trial", decisions_path=reg)
    assert TD.load(reg) == {}
    assert json.loads((trial / "metadata.json").read_text())["bpm_decided"]["bpm_coarse"] == 150.0   # the trial still plays it
    TD.record("trk", 150.0, by="test", path=reg)
    TD.record("trk", 151.0, by="again", path=reg)
    assert TD.load(reg)["trk"]["bpm_coarse"] == 151.0 and len(TD.load(reg)) == 1
