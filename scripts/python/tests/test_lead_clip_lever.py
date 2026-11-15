"""The lead articulation lever (calibrate_dynamic --lead-clip, the 12th calibration knob LC) and its measurement
(melodic_solo.articulation).

BibiBabydoll "Melhor Sobrar" synth at 150 BPM (2026-11-15): the melodic stem plays short stabs (energy-equivalent note
length 0.16 of the note window) while the lead at the generator's clip 1.3 rang 0.33; clip 0.5 by hand 0.787 -> 0.804.
The same clip on the references cost 0.007-0.026 (their stems ring: Regime 0.31 vs 0.31), so it is a measured, per-track
lever; on every reference's best the ratio sits inside SHARE_DEAD_BAND.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_loop as LOOP  # noqa: E402
import melodic_solo as MSO  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / "calibrate_dynamic.py"
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate"
SR = MSO.SR


def _cal(tmp_path: Path, orig: float | None, render: float | None, *extra: str, mode: str = "synth") -> dict[str, str]:
    solo = json.loads((FIX / "melhor_melodic_solo.json").read_text())
    solo["articulation"] = {"orig": orig, "render": render, "windows": 100}
    p = tmp_path / "solo.json"
    p.write_text(json.dumps(solo))
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(FIX / "tatuagem_synth_cal1.json"), "--mode", mode,
                          "--lead-lpf", "3500", "--cal-lead", "0.888", "--sub-gain", "0.474", "--bass-mult", "1.3",
                          "--sub-up", "0.743", "--master-gain", "0.78", "--melodic-solo", str(p), *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def test_notes_that_ring_too_long_shorten_the_clip_by_the_measured_ratio(tmp_path):
    assert float(_cal(tmp_path, 0.16, 0.32)["--lead-clip"]) == pytest.approx(1.3 * 0.5)       # from the default
    assert float(_cal(tmp_path, 0.16, 0.32, "--lead-clip", "0.65")["--lead-clip"]) == pytest.approx(0.325)
    assert float(_cal(tmp_path, 0.05, 0.40)["--lead-clip"]) == pytest.approx(0.65)            # at most 2x per step
    assert float(_cal(tmp_path, 0.40, 0.20, "--lead-clip", "0.5")["--lead-clip"]) == pytest.approx(1.0)   # too short


def test_a_matching_articulation_or_another_mode_leaves_the_clip_alone(tmp_path):
    assert _cal(tmp_path, 0.31, 0.30).get("--lead-clip") is None                              # Regime: inside the band
    assert _cal(tmp_path, 0.16, 0.32, mode="sample-instrument").get("--lead-clip") is None
    assert _cal(tmp_path, None, None).get("--lead-clip") is None


def test_synth_takes_only_the_articulation_from_the_melodic_solo(tmp_path):
    # the solo's level / brightness must not reach the synth lead (the mix decides there): same cal-lead and lead-lpf
    # with and without the solo
    base = subprocess.run([sys.executable, str(CAL), "--comparison", str(FIX / "tatuagem_synth_cal1.json"), "--mode", "synth",
                           "--lead-lpf", "3500", "--cal-lead", "0.888", "--sub-gain", "0.474", "--bass-mult", "1.3",
                           "--sub-up", "0.743", "--master-gain", "0.78"], capture_output=True, text=True, check=True).stdout.split()
    base = {base[i]: base[i + 1] for i in range(0, len(base), 2)}
    got = _cal(tmp_path, 0.31, 0.30)
    assert (got["--cal-lead"], got["--lead-lpf"]) == (base["--cal-lead"], base["--lead-lpf"])


def test_blend_moves_the_clip_as_a_ratio_with_zero_as_the_default():
    base = "1.3 0.474 0.888 3500 0 0.78 0 500 0.743 0.4237 0"
    assert float(LOOP.blend_knobs((base + " 0").split(), (base + " 0.65").split(), [11305, 500], 0.5)[11]) == \
        pytest.approx((1.3 * 0.65) ** 0.5, rel=1e-3)
    assert LOOP.blend_knobs(base.split(), (base + " 0.65").split(), [11305, 500], 1.0)[11] == "0.65"


def _notes(decay_s: float, seconds: float = 8.0) -> np.ndarray:
    """A 440 Hz note every 0.4 s decaying with time constant ``decay_s``."""
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for k in range(int(seconds / 0.4)):
        m = t >= k * 0.4
        y[m] += np.exp(-(t[m] - k * 0.4) / decay_s) * np.sin(2 * np.pi * 440 * (t[m] - k * 0.4)) * (t[m] < k * 0.4 + 0.4)
    return y


def test_articulation_reads_short_stabs_shorter_than_held_notes():
    stab, held = _notes(0.03), _notes(1.0)
    a = MSO.articulation(stab, held)
    assert a["orig"] < 0.1 < 0.25 < a["render"]          # a stab dies at once; a slow decay falls across the window
    assert MSO.articulation(held, held)["orig"] == pytest.approx(MSO.articulation(held, held)["render"])
