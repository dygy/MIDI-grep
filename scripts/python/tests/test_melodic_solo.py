# @layer: unit
# @regression
"""cal-lead and lead-lpf are set by the melodic voices (LEAD+CHORDS) rendered ALONE, not by whole-mix shares.

VAGABUNDO sample-instrument: the melodic voices rendered alone sat 7-9 dB under melodic.wav at 200 Hz-1.3 kHz while the
whole-mix low_mid+mid share and centroid (bass, kick and hats fill them too) did not show it.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_melodic_solo.py -q
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

import melodic_solo  # noqa: E402

KNOBS = ["--bass-mult", "0.5", "--sub-gain", "0.8", "--cal-lead", "1.0", "--lead-lpf", "5000",
         "--hat-gain", "0.1", "--master-gain", "0.65", "--sub-up", "1"]

STRUDEL = """// header
const cps = 0.5

$: stack(
  note(cat(...bass)).s("b")
)  // BASS — low end

$: note(cat(...lead)).s("l")  // LEAD — melody

$: note(cat(...chords)).s("l")  // CHORDS — harmony

$: s("bd*4")  // DRUMS — kit
"""


def run_cal(cmp_path: Path, *extra: str) -> dict[str, str]:
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(cmp_path), *KNOBS, *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def solo_json(tmp_path: Path, body_ratio: float, centroid_ratio: float) -> Path:
    """A measure JSON whose low_mid+mid power ratio is `body_ratio` and whose centroid orig/render is `centroid_ratio`."""
    p = tmp_path / "mel.json"
    p.write_text(json.dumps({
        "level_orig": {"low_mid": body_ratio, "mid": body_ratio}, "level_render": {"low_mid": 1.0, "mid": 1.0},
        "ratio": {"low_mid": body_ratio, "mid": body_ratio},
        "centroid": {"orig": 1000.0 * centroid_ratio, "render": 1000.0}}))
    return p


def test_solo_keeps_header_and_only_lead_and_chords():
    out = melodic_solo.solo_strudel(STRUDEL)
    assert out is not None
    assert "const cps" in out and "// LEAD —" in out and "// CHORDS —" in out
    assert "BASS" not in out and "bd*4" not in out


def test_solo_is_none_without_a_separable_melodic_channel():
    assert melodic_solo.solo_strudel("$: stack(\n  note('c3'),\n  s('bd')\n)\n") is None
    assert melodic_solo.solo_strudel("let x = 1\n") is None


def _write(tmp_path, name, y):
    import soundfile as sf
    p = tmp_path / f"{name}.wav"
    sf.write(p, y.astype(np.float32), melodic_solo.SR)
    return p


def test_measure_is_an_absolute_level_and_a_centroid(tmp_path):
    n = melodic_solo.SR * 4
    t = np.arange(n) / melodic_solo.SR
    stem_y = np.sin(2 * np.pi * 1000 * t)
    stem = _write(tmp_path, "stem", stem_y)
    quiet = _write(tmp_path, "quiet", 0.5 * stem_y)               # 6 dB down in power
    dark = _write(tmp_path, "dark", np.sin(2 * np.pi * 300 * t))
    m = melodic_solo.measure(stem, quiet)
    assert m["ratio"]["mid"] == pytest.approx(4.0, rel=0.05)      # (1 / 0.5)^2
    assert m["centroid"]["orig"] == pytest.approx(m["centroid"]["render"], rel=0.02)   # a level change keeps the tone
    assert melodic_solo.measure(stem, quiet, master_gain=0.5)["ratio"]["mid"] == pytest.approx(1.0, rel=0.05)
    assert melodic_solo.measure(stem, dark)["centroid"]["orig"] > 1.2 * melodic_solo.measure(stem, dark)["centroid"]["render"]


def test_cal_lead_follows_the_voice_level_not_the_mix_body_share(tmp_path):
    cmp_path = FIX / "vagabundo_sample_v002.json"
    without = run_cal(cmp_path)
    short = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, body_ratio=3.0, centroid_ratio=1.0)))
    assert float(short["--cal-lead"]) == pytest.approx(3.0 ** 0.5, rel=1e-3)   # the full measured sqrt step
    assert float(short["--cal-lead"]) != float(without["--cal-lead"])
    wild = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, body_ratio=40.0, centroid_ratio=1.0)))
    assert float(wild["--cal-lead"]) == pytest.approx(2.0)                 # one bounded 2x step, like the bass
    loud = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, body_ratio=0.25, centroid_ratio=1.0)))
    assert float(loud["--cal-lead"]) == pytest.approx(0.5)


def test_lead_lpf_follows_the_voice_centroid_not_the_mix_centroid(tmp_path):
    cmp_path = FIX / "vagabundo_sample_v002.json"
    dark = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, 1.0, centroid_ratio=1.3)))
    bright = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, 1.0, centroid_ratio=0.8)))
    flat = run_cal(cmp_path, "--melodic-solo", str(solo_json(tmp_path, 1.0, centroid_ratio=1.0)))
    assert int(dark["--lead-lpf"]) > 5000 and int(bright["--lead-lpf"]) < 5000
    assert int(flat["--lead-lpf"]) == 5000


def test_no_evidence_or_unreadable_file_keeps_the_mix_behaviour(tmp_path):
    cmp_path = FIX / "vagabundo_sample_v002.json"
    p = tmp_path / "none.json"
    p.write_text(json.dumps({"level_orig": {"low_mid": 1, "mid": 1}, "level_render": {"low_mid": 1, "mid": 1},
                             "ratio": {"low_mid": None, "mid": None}, "centroid": {"orig": 1000.0, "render": None}}))
    assert run_cal(cmp_path, "--melodic-solo", str(p)) == run_cal(cmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert run_cal(cmp_path, "--melodic-solo", str(bad)) == run_cal(cmp_path)
