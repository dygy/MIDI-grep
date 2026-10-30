# @layer: unit
# @regression
"""The analysis lowpass of a voice (``effects_for`` <- ``stem_features['rolloff']``) follows how the stem SOUNDS.

It was the median over ALL frames of the 85 % rolloff: the rests' noise floor shifted it with the amount of rest, and
the 85 % point cuts 15 % of the stem's own energy by construction (Enter bass: 215 Hz over a stem whose sounding
frames reach ~3 kHz), so the sample layer was filtered more than the stem it imitates.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_stem_features_rolloff.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_hybrid_strudel as H  # noqa: E402

SR = H.SR


def _bass(tmp_path: Path, name: str, rests_s: float) -> Path:
    """0.4 s notes at 200 Hz with a 2nd (-10 dB) and 3rd (-17 dB) partial, separated by ``rests_s`` of faint noise."""
    rng = np.random.default_rng(0)
    t = np.arange(int(0.4 * SR)) / SR
    note = (np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t) + 0.15 * np.sin(2 * np.pi * 600 * t)) * 0.4
    rest = 0.0005 * rng.standard_normal(int(rests_s * SR))
    y = np.concatenate([np.concatenate([note, rest]) for _ in range(8)]).astype(np.float32)
    p = tmp_path / f"{name}.wav"
    sf.write(str(p), y, SR)
    return p


def test_the_cutoff_passes_the_partials_the_stem_sounds(tmp_path: Path) -> None:
    f = H.stem_features(_bass(tmp_path, "a", 0.05), 60.0)
    assert f is not None and f["rolloff"] >= 600          # old (85 %, all frames): ~400, the 3rd partial was filtered


def test_the_amount_of_rest_does_not_move_the_cutoff(tmp_path: Path) -> None:
    short = H.stem_features(_bass(tmp_path, "s", 0.1), 60.0)
    long_ = H.stem_features(_bass(tmp_path, "l", 1.2), 60.0)
    assert short and long_
    assert abs(short["rolloff"] - long_["rolloff"]) <= 50  # one STFT bin (~21 Hz) either side; old: moved by hundreds of Hz
