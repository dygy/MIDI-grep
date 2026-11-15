"""The synth drum low-pass lever (calibrate_dynamic --drum-lpf, the 11th calibration knob DL).

Tatuagem No Pescoço synth (2026-11-15): the drum machine's hats / snares lit the `high` band at 1.6 % vs 0.26 % with the
lead low-pass already at its floor and no hat layer; a drum low-pass of 3 kHz by hand: brightness 0.74 -> 0.98, overall
0.770 -> 0.792.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_loop as LOOP  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / "calibrate_dynamic.py"
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate" / "tatuagem_synth_cal1.json"


def _cal(*extra: str, comparison: Path = FIX, mode: str = "synth") -> dict[str, str]:
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(comparison), "--mode", mode,
                          "--lead-lpf", "3500", "--cal-lead", "0.888", "--sub-gain", "0.474", "--bass-mult", "1.3",
                          "--sub-up", "0.743", "--master-gain", "0.78", "--kick-boom-scale", "0.4237", *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def _with_high(tmp_path: Path, high: float) -> Path:
    d = json.loads(FIX.read_text())
    b = d["rendered"]["bands"]
    rest = 1.0 - b["high"]
    for k in b:
        b[k] = high if k == "high" else b[k] * (1.0 - high) / rest
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    return p


def test_a_too_bright_synth_mix_closes_the_drum_low_pass_by_at_most_an_octave():
    # render high 1.7 % vs 0.26 %: ratio 0.15, sqrt 0.39 -> the octave cap: open (10 kHz) -> 5 kHz
    assert float(_cal()["--drum-lpf"]) == pytest.approx(5000)
    assert float(_cal("--drum-lpf", "5000")["--drum-lpf"]) == pytest.approx(2500)
    assert float(_cal("--drum-lpf", "2500")["--drum-lpf"]) == pytest.approx(2000)   # never below high_mid's bottom


def test_the_drum_low_pass_waits_for_the_hats_and_the_lead_and_stays_out_of_sample_mode():
    assert _cal("--hat-gain", "0.1").get("--drum-lpf") is None            # a hat layer backs off first
    assert _cal(mode="sample-instrument").get("--drum-lpf") is None       # the stem's own kit: not this lever's
    out = _cal("--lead-lpf", "9000")                                       # the lead cutoff moves this step: one lever
    assert out["--lead-lpf"] != "9000" and out.get("--drum-lpf") is None


def test_a_dark_mix_opens_the_drum_low_pass_again_and_never_raises_hats_under_it(tmp_path):
    dark = _with_high(tmp_path, 0.0005)                                   # high 0.05 % vs 0.26 %: ratio 5.2
    assert float(_cal("--drum-lpf", "3000", comparison=dark)["--drum-lpf"]) == pytest.approx(6000)
    assert _cal("--drum-lpf", "6000", comparison=dark)["--drum-lpf"] == "0"   # past the top: open
    assert _cal("--drum-lpf", "3000", comparison=dark)["--hat-gain"] == "0.0"
    inside = _with_high(tmp_path, 0.0026)                                 # matched: held
    assert float(_cal("--drum-lpf", "3000", comparison=inside)["--drum-lpf"]) == pytest.approx(3000)


def test_blend_moves_the_drum_low_pass_in_octaves_with_open_at_the_top():
    base = "1.3 0.474 0.888 3500 0 0.78 0 500 0.743 0.4237"
    assert LOOP.blend_knobs((base + " 0").split(), (base + " 5000").split(), [11305, 500], 0.5)[10] == "7071"
    assert LOOP.blend_knobs(base.split(), (base + " 2500").split(), [11305, 500], 1.0)[10] == "2500"   # a 10-field parent
    assert LOOP.blend_knobs((base + " 5000").split(), (base + " 0").split(), [11305, 500], 1.0)[10] == "0"
