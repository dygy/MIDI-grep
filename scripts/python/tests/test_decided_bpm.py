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
