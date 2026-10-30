# @layer: unit
# @regression
"""The bass instrument pack holds only samples that sound, and keeps a loopable grain per pitch (Enter).

* Eleven bass samples (a2-g#4) were transcription ghosts cut from stretches where the stem is silent (peak 0.001,
  -60 dB); pitch verification is scale-free ('salient'), so they entered samples.json and played as silence.
* The grain chosen for d#1 was 'long enough' only through a digital-silence tail (0.1 s of tone in 0.44 s): it had
  no steady body, so five written pitches (25-29) had no loop and died after ~0.2 s.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_instrument_pack_sound.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bass_shape as BS  # noqa: E402
import pitched_grains as pg  # noqa: E402
from build_instruments import KIT_SOUNDS, build_instruments  # noqa: E402

SR = 22050


def _tone(midi: int, seconds: float, amp: float, sound_s: float | None = None) -> np.ndarray:
    """A shaped bass grain (fade-in, quarter release); ``sound_s`` < seconds leaves a digital-silence tail."""
    hz = pg.midi_to_hz(midi)
    t = np.arange(int(seconds * SR)) / SR
    y = amp * np.sin(2 * np.pi * hz * t)
    if sound_s is not None:
        y[int(sound_s * SR):] = 0.0
    return pg.shape_grain(y.astype(np.float32), SR, hz) if sound_s is None else y.astype(np.float32)


def _grain(midi: int, y: np.ndarray, quality: float) -> pg.Grain:
    g = pg.Grain(y, midi, midi, "salience", quality, 0.0, len(y) / SR, False, pg._rms(y))
    g.raw, g.sr = y, SR
    return g


def test_a_long_grain_with_a_silent_tail_loses_to_a_grain_that_can_sustain() -> None:
    tail = _grain(27, _tone(27, 0.5, 0.5, sound_s=0.1), quality=9.0)      # 'long enough', best quality, no body
    body = _grain(27, _tone(27, 0.3, 0.5), quality=1.0)                    # shorter, has a steady body
    assert BS.loop_points(body.audio, SR, pg.midi_to_hz(27)) is not None
    assert BS.loop_points(tail.audio, SR, pg.midi_to_hz(27)) is None
    chosen = pg.select_per_pitch([tail, body], target_len_s=0.4)
    assert chosen[27] is body


def test_a_grain_cut_where_the_stem_is_silent_is_never_selected() -> None:
    loud = _grain(33, _tone(33, 0.4, 0.5), quality=1.0)
    ghost = _grain(45, _tone(45, 0.4, 0.001), quality=9.0)                 # -54 dB: a transcription ghost
    floor = 0.02                                                            # the stem's sounding frame-RMS
    chosen = pg.select_per_pitch([loud, ghost], target_len_s=0.2, floor_rms=floor)
    assert set(chosen) == {33}


def _model(root: Path, name: str, samples: dict[str, np.ndarray], sounding_rms: float) -> Path:
    d = root / "models" / name
    (d / "pitched").mkdir(parents=True)
    pm = {}
    for note, y in samples.items():
        fn = note.replace("#", "_sharp_") + ".wav"
        sf.write(str(d / "pitched" / fn), y, SR)
        pm[note] = f"pitched/{fn}"
    (d / "metadata.json").write_text(json.dumps({"name": name, "pitched_map": pm, "grains": [],
                                                  "sounding_rms": sounding_rms}))
    return d


def test_a_silent_sample_never_enters_samples_json(tmp_path: Path) -> None:
    kit = tmp_path / "drums"
    kit.mkdir()
    for s in KIT_SOUNDS:
        sf.write(str(kit / f"{s}.wav"), np.zeros(256, dtype=np.float32), SR)
    model = _model(tmp_path, "t_bass", {"a1": _tone(33, 0.5, 0.5), "d#1": _tone(27, 0.5, 0.5),
                                        "a2": _tone(45, 0.3, 0.001)}, sounding_rms=0.02)
    out = tmp_path / "out"
    build_instruments([model], kit, out, "https://example.test/i/")
    doc = json.loads((out / "samples.json").read_text())
    assert set(doc["t_bass"]) == {"a1", "d#1"}
    rows = {r["label"]: r for r in json.loads((out / "verification.json").read_text())["samples"]["t_bass"]}
    assert rows["a2"]["action"] == "dropped" and "silent" in rows["a2"]["reason"]
