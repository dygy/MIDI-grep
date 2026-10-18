# @layer: unit
# @spec: 001-core-pipeline
# @regression
"""Tempo must not be quantised to librosa's tempogram lag grid (spec 001 §2.3: plays at the correct speed).

``beat_track`` returns 60*sr/hop/k (129.1992 / 135.9992 at sr 22050, hop 512); a click train at a BPM
BETWEEN bins used to be reported ~0.8 BPM off. analyze.py now refines within one bin by phase coherence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

SR = 22050
TRUE_BPM = 129.99          # between the 129.1992 and 135.9992 tempogram bins


def _kick_train(bpm: float, seconds: float = 60.0, sr: int = SR) -> np.ndarray:
    y = np.zeros(int(seconds * sr))
    period = 60.0 / bpm * sr
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 60)) * t) * np.exp(-t * 40)
    k = 0
    while (pos := int(round(k * period))) + n < len(y):
        y[pos:pos + n] += kick
        k += 1
    return y


@pytest.fixture(scope="module")
def click_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("tempo") / "kicks.wav"
    soundfile.write(str(p), _kick_train(TRUE_BPM), SR)
    return p


def test_refine_bpm_recovers_between_bin_tempo() -> None:
    import librosa
    from tempo_refine import refine_bpm
    y = _kick_train(TRUE_BPM)
    coarse = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=SR, start_bpm=120.0)[0])[0])
    assert abs(coarse - TRUE_BPM) > 0.5, "premise: the tempogram grid value is off"
    r = refine_bpm(y, SR, coarse)
    assert abs(r["bpm"] - TRUE_BPM) < 0.05
    assert r["method"] == "phase-coherence" and r["confidence"] > 0.5


def test_analyze_script_reports_refined_not_grid_bpm(click_wav: Path, tmp_path: Path) -> None:
    out = tmp_path / "a.json"
    subprocess.run([sys.executable, str(SCRIPTS / "analyze.py"), str(click_wav), str(out)], check=True,
                   capture_output=True)
    d = json.loads(out.read_text())
    assert abs(d["bpm"] - TRUE_BPM) < 0.05, d["bpm"]
    assert d["bpm_method"] == "phase-coherence" and "bpm_coarse" in d


def test_stamp_refines_on_drums_and_records_provenance(click_wav: Path, tmp_path: Path) -> None:
    import pipeline_helpers as ph
    track = tmp_path / "track"
    (track / "v001").mkdir(parents=True)
    (track / "drums.wav").write_bytes(click_wav.read_bytes())
    (track / "v001" / "metadata.json").write_text(json.dumps({"bpm": 129.1992, "key": "C minor"}))
    m = ph.stamp_track_metadata(track)
    assert abs(m["bpm"] - TRUE_BPM) < 0.05
    assert m["bpm_coarse"] == pytest.approx(129.1992)
    assert m["bpm_method"].startswith("phase-coherence") and 0 < m["bpm_confidence"] <= 1
