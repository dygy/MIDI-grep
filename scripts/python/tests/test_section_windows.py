# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""Bug #10: section-aware metric must expose its per-window numbers, and the calibrator
must turn them into per-bar multiplier curves."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from compare_audio import compare_audio  # noqa: E402
from calibrate_dynamic import ENV_LO, ENV_HI  # noqa: E402

SR = 22050


def _tone(seconds, f, seed):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    return 0.4 * np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size)


def _write(tmp_path, name, y):
    p = tmp_path / name
    sf.write(str(p), y, SR)
    return str(p)


def test_section_windows_emitted_and_consistent(tmp_path):
    o = _write(tmp_path, "o.wav", np.concatenate([_tone(15, 110, 1), _tone(15, 880, 2)]))
    r = _write(tmp_path, "r.wav", np.concatenate([_tone(15, 220, 3), _tone(15, 660, 4)]))
    res = compare_audio(o, r, duration=30)["comparison"]
    wins = res["section_windows"]
    assert len(wins) == res["section_aware_window_count"] == 3
    for k in ("t0", "t1", "mfcc", "band", "energy", "win", "orig_rms", "rend_rms", "band_diff"):
        assert k in wins[0]
    assert wins[1]["t0"] == pytest.approx(10.0) and wins[1]["t1"] == pytest.approx(20.0)
    assert set(wins[0]["band_diff"]) == {"sub_bass", "bass", "low_mid", "mid", "high_mid", "high"}
    assert abs(np.mean([w["win"] for w in wins]) - res["section_aware_similarity"]) < 1e-9


def test_section_windows_empty_on_short_audio(tmp_path):
    o = _write(tmp_path, "o.wav", _tone(8, 220, 1))
    r = _write(tmp_path, "r.wav", _tone(8, 220, 2))
    res = compare_audio(o, r, duration=8)["comparison"]
    assert res["section_windows"] == []
    assert res["section_aware_window_count"] == 0


def _synthetic_comparison(with_windows=True):
    # first half: bass under (rend < orig), lead over; second half the opposite.
    def win(i, bass_o, bass_r, mid_o, mid_r, orms, rrms):
        base = {"sub_bass": 0.0, "bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high_mid": 0.0, "high": 0.0}
        ob, rb = dict(base), dict(base)
        ob["bass"], rb["bass"], ob["mid"], rb["mid"] = bass_o, bass_r, mid_o, mid_r
        return {"t0": 10.0 * i, "t1": 10.0 * (i + 1), "mfcc": .9, "band": .8, "energy": .7, "win": .8,
                "orig_rms": orms, "rend_rms": rrms, "band_diff": {}, "orig_bands": ob, "rend_bands": rb}
    wins = [win(0, .5, .25, .1, .2, .2, .1), win(1, .5, .25, .1, .2, .2, .1),
            win(2, .25, .5, .2, .1, .1, .2), win(3, .25, .5, .2, .1, .1, .2)]
    bands = {"sub_bass": .1, "bass": .3, "low_mid": .2, "mid": .2, "high_mid": .1, "high": .1}
    d = {"original": {"bands": bands, "spectral": {"centroid_mean": 1000.0}, "rhythm": {"tempo": 120.0}},
         "rendered": {"bands": bands, "spectral": {"centroid_mean": 1000.0}},
         "comparison": {"overall_similarity": .9, "raw_rms_ratio": 1.0}}
    if with_windows:
        d["comparison"]["section_windows"] = wins
    return d


def _run(tmp_path, cmp_data, *extra):
    cp = tmp_path / "comparison.json"
    cp.write_text(json.dumps(cmp_data))
    out = tmp_path / "env.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "calibrate_dynamic.py"), "--comparison", str(cp),
         "--env-correction-out", str(out), *extra], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return out, proc


def test_env_correction_pattern(tmp_path):
    out, _ = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20")
    env = json.loads(out.read_text())
    assert env["bars"] == 20 and env["bpm"] == 120.0 and env["window_s"] == 10.0
    v = env["voices"]
    assert all(len(v[k]) == 20 for k in ("bass", "lead", "master"))
    # 120 BPM -> 2 s bars; first bar sits in the "bass under" half, last in the "bass over" half
    assert v["bass"][0] > 1 and v["bass"][-1] < 1
    assert v["lead"][0] < 1 and v["lead"][-1] > 1
    assert v["master"][0] > 1 and v["master"][-1] < 1
    for k in v:
        assert all(ENV_LO <= x <= ENV_HI for x in v[k])


def test_env_correction_composes(tmp_path):
    first, _ = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20")
    prev = tmp_path / "prev.json"
    prev.write_text(first.read_text())
    second, _ = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20",
                     "--env-correction-in", str(prev))
    a = json.loads(prev.read_text())["voices"]["bass"]
    b = json.loads(second.read_text())["voices"]["bass"]
    for x, y in zip(a, b):
        assert y == pytest.approx(min(ENV_HI, max(ENV_LO, x * x)), abs=2e-4)


def test_env_correction_absent_windows(tmp_path):
    out, proc = _run(tmp_path, _synthetic_comparison(with_windows=False))
    assert not out.exists()
    assert "no section_windows" in proc.stderr


def test_env_correction_emits_clamp_bounds(tmp_path):
    out, _ = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20")
    assert json.loads(out.read_text())["clamp"] == [0.5, 2.0]


def test_env_correction_resamples_prev_with_different_bars(tmp_path):
    single, _ = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20")
    single_bass = json.loads(single.read_text())["voices"]["bass"]
    # previous curve measured on a 10-bar grid (same bpm), not the 20-bar grid of this step
    prev = tmp_path / "prev.json"
    prev.write_text(json.dumps({"window_s": 10.0, "bpm": 120.0, "bars": 10, "source": "t",
                                "voices": {k: [1.5] * 10 for k in ("bass", "lead", "master")}}))
    out, proc = _run(tmp_path, _synthetic_comparison(), "--bpm", "120", "--bars", "20",
                     "--env-correction-in", str(prev))
    composed = json.loads(out.read_text())["voices"]["bass"]
    assert len(composed) == 20
    assert composed != single_bass                      # previous curve was not dropped
    assert "different bar count" in proc.stderr         # and the mismatch is visible
    for s, c in zip(single_bass, composed):
        assert c == pytest.approx(min(ENV_HI, max(ENV_LO, s * 1.5)), abs=2e-4)
