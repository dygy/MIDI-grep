"""stem_repair.py: the sustained sub Demucs filed under drums goes back to the bass where the bass stem is silent.

Regression ("Automotivo Novinho Depressivo", 2026-11-01): bars 4-6 / 17-19 held the 808 in drums.wav (bass.wav at
-59 dB), so the generator muted the bass there and the drums envelope played the kick louder: sample-instrument
0.8515 / 0.8146 FAILED the gate; the same loop on the repaired view scored 0.8564 / 0.8442 PASS.
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
import bass_register as BR  # noqa: E402
import generate_dynamic_strudel as G  # noqa: E402
import stem_repair as SR  # noqa: E402

SRATE, BPM, BARS = 22050, 120.0, 8
BAR = 60.0 / BPM * 4
G1 = 440.0 * 2 ** ((31 - 69) / 12)


def _tone(t0: float, t1: float, n: int) -> np.ndarray:
    y = np.zeros(n)
    a, b = int(t0 * SRATE), int(t1 * SRATE)
    t = np.arange(b - a) / SRATE
    y[a:b] = 0.5 * np.sin(2 * np.pi * G1 * t)
    return y


def _kicks(n: int, bars: range) -> np.ndarray:
    y = np.zeros(n)
    k = np.arange(int(0.15 * SRATE)) / SRATE
    hit = 0.8 * np.sin(2 * np.pi * (60 + 90 * np.exp(-k * 40)) * k) * np.exp(-k * 30)
    for bar in bars:
        for beat in range(4):
            a = int((bar * BAR + beat * BAR / 4) * SRATE)
            y[a:a + hit.size] += hit[: max(0, min(hit.size, n - a))]
    return y


def _track(tmp: Path, misrouted: bool) -> Path:
    n = int(BARS * BAR * SRATE)
    bass = _tone(2 * BAR, BARS * BAR, n)                          # the bass stem enters at bar 2
    drums = _kicks(n, range(2, BARS))
    if misrouted:
        drums += _tone(0, 2 * BAR, n)                             # the intro 808, filed under drums
    else:
        drums += _kicks(n, range(0, 2))                           # an intro of kicks only
    d = tmp / "track"
    d.mkdir()
    for name, y in (("bass", bass), ("drums", drums), ("melodic", 0.01 * np.random.default_rng(0).standard_normal(n))):
        sf.write(str(d / f"{name}.wav"), np.stack([y, y], 1).astype(np.float32), SRATE, subtype="FLOAT")
    sf.write(str(d / "instrumental.wav"), np.zeros((n, 2), np.float32), SRATE, subtype="FLOAT")
    BR.measure(d / "bass.wav", d / ".pipeline" / "bass_f0.json")
    return d


def _build(d: Path) -> dict:
    return SR.build(d, d / ".pipeline" / "stems", BPM, BARS, d / ".pipeline" / "bass_f0.json")


def test_the_808_in_the_drums_stem_moves_to_the_bass_only_where_the_bass_is_silent(tmp_path):
    d = _track(tmp_path, misrouted=True)
    s = _build(d)
    assert s["moved_bars"] == [0, 1]
    view = Path(s["view"])
    # the generator now plays the bass in the intro it used to mute
    before = G.bar_env_pattern(d / "bass.wav", BARS, BPM, floor=0.15)
    after = G.bar_env_pattern(view / "bass.wav", BARS, BPM, floor=0.15)
    assert all(v == 0.0 for bar in before[:2] for v in bar)
    assert all(v > 0.0 for bar in after[:2] for v in bar)
    # bass + drums is the same signal: the scored instrumental does not change
    ob, _ = sf.read(str(d / "bass.wav")); od, _ = sf.read(str(d / "drums.wav"))
    nb, _ = sf.read(str(view / "bass.wav")); nd, _ = sf.read(str(view / "drums.wav"))
    assert np.max(np.abs((ob + od) - (nb + nd))) < 1e-5
    # the drums keep their kicks outside the moved bars (past the crossfade: one period of the register's lowest note)
    a = int((2 * BAR + 1.0 / SR._midi_hz(s["window"][0])) * SRATE)
    assert np.max(np.abs(od[a:] - nd[a:])) < 1e-3
    # the view reads the track's other stems and has its own bass f0 track
    assert (view / "melodic.wav").resolve() == (d / "melodic.wav").resolve()
    assert (view / "instrumental.wav").is_symlink()
    assert BR.Register.load(view / ".pipeline" / "bass_f0.json").window()


def test_a_kick_only_intro_is_not_bass(tmp_path):
    d = _track(tmp_path, misrouted=False)
    s = _build(d)
    assert s["moved_bars"] == [] and s["view"] is None
    assert not (d / ".pipeline" / "stems").exists()


def test_the_view_is_reused_until_a_stem_changes(tmp_path):
    d = _track(tmp_path, misrouted=True)
    first = _build(d)
    mtime = (Path(first["view"]) / "bass.wav").stat().st_mtime_ns
    assert _build(d) == json.loads((Path(first["view"]) / "repair.json").read_text())
    assert (Path(first["view"]) / "bass.wav").stat().st_mtime_ns == mtime
    y, _ = sf.read(str(d / "drums.wav"))
    sf.write(str(d / "drums.wav"), (y * 0.5).astype(np.float32), SRATE, subtype="FLOAT")
    assert _build(d)["stamp"] != first["stamp"]
