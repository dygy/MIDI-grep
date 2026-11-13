"""detect_drums_bands.py: a bar the generator's gate calls loud never ends with no drum hits.

Regression (Creagina, 2026-11-08): bars 4-7 / 34-35 of drums.wav held a pitched riff at 500-2000 Hz (outside the kick,
snare-body and hat bands' reach under their global thresholds); the detector wrote no hit there, the generator wrote
empty drum bars, and the render was digitally silent (-167 dB) where the original played at -15 dB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_drums_bands as DB  # noqa: E402

BPM = 120.0
BAR = 240.0 / BPM


def _stem(tmp: Path) -> Path:
    n = int(6 * BAR * DB.SR)
    y = np.zeros(n)
    k = np.arange(int(0.15 * DB.SR)) / DB.SR
    kick = 0.8 * np.sin(2 * np.pi * (60 + 90 * np.exp(-k * 40)) * k) * np.exp(-k * 30)
    rng = np.random.default_rng(0)
    h = np.arange(int(0.05 * DB.SR)) / DB.SR
    hat = DB._bandpass(rng.standard_normal(h.size), 6000, 11000) * np.exp(-h * 80) * 3.0
    s = np.arange(int(0.12 * DB.SR)) / DB.SR
    snare = DB._bandpass(rng.standard_normal(s.size), 180, 1200) * np.exp(-s * 25) * 2.0
    for bar in (0, 1, 3, 4):           # a full kit (kicks, snares on 2 and 4, eighth hats); bar 2 = the riff, bar 5 silent
        for beat in range(4):
            a = int((bar * BAR + beat * BAR / 4) * DB.SR)
            y[a:a + kick.size] += kick
            if beat in (1, 3):
                y[a:a + snare.size] += snare
        for e in range(8):
            a = int((bar * BAR + e * BAR / 8) * DB.SR)
            y[a:a + hat.size] += hat
    note = np.arange(int(0.2 * DB.SR)) / DB.SR
    blip = 0.25 * np.sin(2 * np.pi * 1600.0 * note) * np.exp(-note * 12)  # above the snare body, below the hats
    for j in range(8):
        a = int((2 * BAR + j * BAR / 8) * DB.SR)
        y[a:a + blip.size] += blip
    p = tmp / "drums.wav"
    sf.write(str(p), y.astype(np.float32), DB.SR)
    return p


BANDS = {"bd": (20, 160), "sd": (180, 1200), "hh": (6000, 11000)}


def _band_hits() -> list[dict]:
    """What the bands detected on _stem: the kit in bars 0, 1, 3, 4 - nothing in the riff bar 2."""
    hits = []
    for bar in (0, 1, 3, 4):
        for beat in range(4):
            t = bar * BAR + beat * BAR / 4
            hits.append({"time": t, "type": "bd", "strength": 0.8})
            if beat in (1, 3):
                hits.append({"time": t + 0.001, "type": "sd", "strength": 0.6})
        hits += [{"time": bar * BAR + e * BAR / 8 + 0.002, "type": "hh", "strength": 0.5} for e in range(8)]
    return hits


def test_a_loud_bar_with_no_band_hit_gets_its_full_band_onsets(tmp_path):
    import librosa
    y, _ = librosa.load(str(_stem(tmp_path)), sr=DB.SR, mono=True)
    extra = DB._uncovered_bar_hits(y, BPM, _band_hits(), BANDS)
    assert len(extra) >= 6 and all(2 * BAR <= h["time"] < 3 * BAR and h["uncovered_bar"] for h in extra)
    assert {h["type"] for h in extra} <= set(BANDS)
    # the silent bar 5 gets nothing: below the generator's gate


def test_without_a_tempo_there_is_no_bar_grid_and_no_fallback(tmp_path):
    r = DB.detect(_stem(tmp_path), None)
    assert r["counts"]["uncovered_bar_hits"] == 0


CREAGINA = Path(__file__).resolve().parents[3] / ".cache" / "stems" / "Creagina"


@pytest.mark.skipif(not (CREAGINA / "drums.wav").is_file(), reason="Creagina cache absent")
def test_creagina_riff_bars_get_hits():
    import json
    bpm = json.loads((CREAGINA / "metadata.json").read_text())["bpm"]
    r = DB.detect(CREAGINA / "drums.wav", bpm)
    bar = 240.0 / bpm
    filled = {int(h["time"] // bar) for h in r["hits"] if h.get("uncovered_bar")}
    assert {4, 5, 6} <= filled
