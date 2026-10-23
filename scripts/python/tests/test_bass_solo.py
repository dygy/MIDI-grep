# @layer: unit
# @regression
"""The bass layers are set by the bass voice rendered ALONE, not by band shares the kick also fills.

VAGABUNDO sample-instrument: the mix sub_bass share read 34.7 % vs 34.1 % (matched) while the bass voice alone sat
8.5 dB under the original bass stem (20-45 Hz -14.4 dB), and the loop LOWERED bass-mult (0.351 -> 0.331) because
the kick made the mix sub read "over".

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_bass_solo.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate"
CAL = SCRIPTS / "calibrate_dynamic.py"
sys.path.insert(0, str(SCRIPTS))

import bass_solo  # noqa: E402

KNOBS = ["--bass-mult", "0.5", "--sub-gain", "0.8", "--cal-lead", "1.1", "--lead-lpf", "6000",
         "--hat-gain", "0.1", "--master-gain", "0.65", "--sub-up", "1"]

STRUDEL = """// header
const cps = 0.5
let bass = ["c1 e1"]
const bassLoop = {25: [0.1, 0.2]}

$: stack(
  note(cat(...bass)).s("b"),
  note("c1").s("sine").gain(0)
)  // BASS — low end — root/sub. edit notes in `bass`

$: note(cat(...lead)).s("l")  // LEAD — melody

$: s("bd*4")  // DRUMS — kit
"""


def run_cal(cmp_path: Path, *extra: str) -> dict[str, str]:
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(cmp_path), *KNOBS, *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def solo_json(tmp_path: Path, sub_bass: float, bass: float) -> Path:
    p = tmp_path / "bass.json"
    p.write_text(json.dumps({"ratio": {"sub_bass": sub_bass, "bass": bass, "low_mid": None}}))
    return p


def test_solo_keeps_header_and_only_the_bass_channel():
    out = bass_solo.solo_strudel(STRUDEL)
    assert out is not None
    assert "const bassLoop" in out and "const cps" in out          # shared definitions stay
    assert "// BASS —" in out and "LEAD" not in out and "bd*4" not in out


def test_solo_is_none_without_a_separable_bass_channel():
    assert bass_solo.solo_strudel("$: stack(\n  note('c1'),\n  s('bd')\n)\n") is None
    assert bass_solo.solo_strudel("let x = 1\n") is None


def _write(tmp_path, name, y):
    import soundfile as sf
    p = tmp_path / f"{name}.wav"
    sf.write(p, y.astype(np.float32), bass_solo.SR)
    return p


def test_measure_is_an_absolute_level_that_follows_the_gain(tmp_path):
    n = bass_solo.SR * 4
    t = np.arange(n) / bass_solo.SR
    bass = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    stem = _write(tmp_path, "stem", bass)
    quiet = _write(tmp_path, "quiet", 0.25 * bass)                # 12 dB down in power
    louder = _write(tmp_path, "louder", 0.5 * bass)               # +6 dB: the ratio must see it (a mix share would not)
    r_quiet = bass_solo.measure(stem, quiet)["ratio"]
    r_louder = bass_solo.measure(stem, louder)["ratio"]
    assert r_quiet["sub_bass"] == pytest.approx(16.0, rel=0.05)   # (1 / 0.25)^2
    assert r_louder["sub_bass"] == pytest.approx(4.0, rel=0.05)
    assert bass_solo.measure(stem, quiet, master_gain=0.5)["ratio"]["sub_bass"] == pytest.approx(4.0, rel=0.05)
    assert r_quiet["high"] is None                                # a band the stem does not occupy never drives anything


def test_bass_mult_follows_the_bass_voice_not_the_mix_sub(tmp_path):
    cmp_path = FIX / "vagabundo_sample_v002.json"                 # mix sub_bass 26.9 % orig vs 29.9 % rend: "over"
    without = run_cal(cmp_path)
    assert float(without["--bass-mult"]) < 0.5                    # mix shares alone LOWER the sample layer (the bug)
    with_solo = run_cal(cmp_path, "--bass-solo", str(solo_json(tmp_path, sub_bass=12.8, bass=2.3)))
    assert float(with_solo["--bass-mult"]) > 0.5                  # the voice is 11 dB short: it must rise
    assert float(with_solo["--bass-mult"]) == pytest.approx(0.5 * 2.0)   # sqrt(12.8)=3.6 -> bounded to one 2x step
    assert with_solo["--sub-gain"] == without["--sub-gain"]       # the sine layer stays on the mix shares (both layers feed `bass`)


def test_a_voice_that_is_too_loud_is_lowered_and_a_missing_band_falls_back(tmp_path):
    cmp_path = FIX / "vagabundo_sample_v002.json"
    loud = run_cal(cmp_path, "--bass-solo", str(solo_json(tmp_path, sub_bass=0.25, bass=0.25)))
    assert float(loud["--bass-mult"]) == pytest.approx(0.5 * 0.5)         # sqrt(0.25) = 0.5
    p = tmp_path / "none.json"
    p.write_text(json.dumps({"ratio": {"sub_bass": None, "bass": None}}))
    assert run_cal(cmp_path, "--bass-solo", str(p)) == run_cal(cmp_path)  # no evidence = the mix-share behaviour, unchanged
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert run_cal(cmp_path, "--bass-solo", str(bad)) == run_cal(cmp_path)
