# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Sub-sine layer under the calibrator: gain range from the measured sub_bass share, and a scored octave choice.

Regression (2026-10-19): `sub_gain` was clamped to 0.2-1.6 and saturated at 1.6 on VAGABUNDO while sub_bass was
still short (the lever went blind); and there was no way to try the layer an octave up, which recovered most of
the Regime CLT loss in a hand swap (0.7287 -> 0.7560) while a spectral-ratio derivation failed both tracks.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_sub_levers.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

import calibrate_loop as CL  # noqa: E402
import pipeline_helpers as PH  # noqa: E402

CAL = SCRIPTS / "calibrate_dynamic.py"


def _comparison(tmp_path: Path, orig_sub: float, rend_sub: float) -> Path:
    def bands(sub):   # the remaining share is split evenly so every other ratio is 1
        rest = (1.0 - sub) / 5
        return {"sub_bass": sub, "bass": rest, "low_mid": rest, "mid": rest, "high_mid": rest, "high": rest}
    d = {"original": {"bands": bands(orig_sub), "spectral": {"centroid_mean": 1000.0}},
         "rendered": {"bands": bands(rend_sub), "spectral": {"centroid_mean": 1000.0}},
         "comparison": {"overall_similarity": 0.7, "raw_rms_ratio": 1.0}}
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps(d))
    return p


def _cal(tmp_path, orig_sub, rend_sub, *extra):
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison(tmp_path, orig_sub, rend_sub)), *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def test_sub_gain_has_no_fixed_cap(tmp_path):
    """Render holds 1/3 of the original's sub share: the closing step (damped, per-step bounded) is x1.4 of 1.5 = 2.1.
    The old fixed 0.2-1.6 range returned 1.6."""
    got = _cal(tmp_path, 0.30, 0.10, "--sub-gain", "1.5")
    assert float(got["--sub-gain"]) == pytest.approx(2.1, abs=1e-3)
    assert got["--sub-up"] == "0"                        # sub_bass is SHORT: more gain, not an octave move


def test_sub_gain_can_fall_far_below_the_old_floor(tmp_path):
    got = _cal(tmp_path, 0.10, 0.12, "--sub-gain", "0.21", "--sub-up-locked", "1")   # slightly over, octave step already lost
    assert 0.0 < float(got["--sub-gain"]) < 0.21


def test_octave_up_is_tried_once_when_sub_bass_is_over(tmp_path):
    """Render sub share 2x the original's (beyond the share dead band): one step flips the octave, gain untouched."""
    got = _cal(tmp_path, 0.10, 0.20, "--sub-gain", "0.9")
    assert got["--sub-up"] == "1" and float(got["--sub-gain"]) == pytest.approx(0.9)


def test_octave_up_not_proposed_again_after_it_lost(tmp_path):
    got = _cal(tmp_path, 0.10, 0.20, "--sub-gain", "0.9", "--sub-up-locked", "1")
    assert got["--sub-up"] == "0" and float(got["--sub-gain"]) < 0.9          # falls back to the gain lever


def test_octave_up_stays_on_and_gain_corrects_when_already_up(tmp_path):
    over_bass = {"sub_bass": .10, "bass": .55, "low_mid": .09, "mid": .09, "high_mid": .09, "high": .08}
    orig = {"sub_bass": .10, "bass": .41, "low_mid": .13, "mid": .13, "high_mid": .12, "high": .11}
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, orig, over_bass)),
                          "--sub-gain", "0.9", "--sub-up", "1"], capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert got["--sub-up"] == "1" and float(got["--sub-gain"]) < 0.9        # bass over -> the octave-up sine comes down


def _comparison_bands(tmp_path: Path, orig: dict, rend: dict) -> Path:
    d = {"original": {"bands": orig, "spectral": {"centroid_mean": 1000.0}},
         "rendered": {"bands": rend, "spectral": {"centroid_mean": 1000.0}},
         "comparison": {"overall_similarity": 0.7, "raw_rms_ratio": 1.0}}
    p = tmp_path / "bands.json"
    p.write_text(json.dumps(d))
    return p


REGIME_ORIG = {"sub_bass": .404, "bass": .410, "low_mid": .096, "mid": .068, "high_mid": .007, "high": .006}
REGIME_CAL2 = {"sub_bass": .221, "bass": .552, "low_mid": .056, "mid": .133, "high_mid": .017, "high": .007}   # synth cal2, su=1


def test_with_the_octave_up_the_gains_follow_the_band_their_layer_feeds(tmp_path):
    """Regime CLT synth cal2: sine an octave up, render sub_bass 22% (< 40%) and bass 55% (> 41%). The sine now feeds
    the BASS band, so its gain must FALL (the old rule raised it 0.70 -> 0.946 and bass reached 79%), while the main
    layer, which feeds sub_bass, rises."""
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, REGIME_ORIG, REGIME_CAL2)),
                          "--sub-gain", "0.7", "--sub-up", "1", "--bass-mult", "0.784"],
                         capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert float(got["--sub-gain"]) < 0.7 and float(got["--bass-mult"]) > 0.784 and got["--sub-up"] == "1"


def test_octave_zero_keeps_the_original_assignment(tmp_path):
    """Octave 0: sub_bass short raises the sine, bass over lowers the main layer (unchanged behaviour)."""
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, REGIME_ORIG, REGIME_CAL2)),
                          "--sub-gain", "0.7", "--sub-up", "0", "--sub-up-locked", "1", "--bass-mult", "0.784"],
                         capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert float(got["--sub-gain"]) > 0.7 and float(got["--bass-mult"]) < 0.784


def test_the_flip_step_holds_bass_mult(tmp_path):
    """Regime CLT synth cal1: sub_bass 71% vs 40% over, bass 10% vs 41% short. The flip alone moved ~30 points of
    share; raising bass-mult in the same step (0.56 -> 0.784) compounded the overshoot, so the flip step holds it."""
    cal1 = {"sub_bass": .707, "bass": .104, "low_mid": .035, "mid": .115, "high_mid": .015, "high": .006}
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, REGIME_ORIG, cal1)),
                          "--sub-gain", "0.7", "--bass-mult", "0.56"], capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert got["--sub-up"] == "1" and float(got["--bass-mult"]) == pytest.approx(0.56) and float(got["--sub-gain"]) == pytest.approx(0.7)


def test_octave_up_is_tried_for_a_modest_excess_too(tmp_path):
    """Regime CLT sample: render sub 49.7% vs original 40.4% (ratio 0.81, inside the old 1.33 dead band) - the
    octave step is a scored trial, so it must be proposed."""
    got = _cal(tmp_path, 0.404, 0.497, "--sub-gain", "0.623")
    assert got["--sub-up"] == "1" and float(got["--sub-gain"]) == pytest.approx(0.623)


def test_no_octave_move_when_sub_bass_is_not_over(tmp_path):
    got = _cal(tmp_path, 0.10, 0.10, "--sub-gain", "0.9")
    assert got["--sub-up"] == "0"
    got = _cal(tmp_path, 0.10, 0.08, "--sub-gain", "0.9")        # render short of sub: gain lever only
    assert got["--sub-up"] == "0" and float(got["--sub-gain"]) > 0.9


def test_blend_takes_the_octave_whole_or_keeps_the_parents():
    parent = "0.4 0.7 1.0 5000 0.1 0.6 0 0 0".split()
    prop = "0.5 0.9 1.0 5000 0.1 0.6 0 0 1".split()
    assert CL.blend_knobs(parent, prop, [603, 300], 1.0)[8] == "1"
    shortened = CL.blend_knobs(parent, prop, [603, 300], 0.3)
    assert shortened[8] == "0" and float(shortened[1]) == pytest.approx(0.76, abs=1e-3)   # magnitudes still interpolate


@pytest.mark.parametrize("text,ok", [
    ("0.4 0.7 1.0 5000 0.1 0.6 0 0 1", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 422 300 0", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300", False),     # removed CV layout: 7th value ~1 is not a cutoff
    ("0.4 0.7 1.0 5000 0.1 0.6 0 0 2", False),
])
def test_params_valid_nine_fields(text, ok):
    assert PH.params_valid(text) is ok


# ---- generator: --sub-up on the cached Regime CLT pack (skipped without it) ----
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
needs_pack = pytest.mark.skipif(not (PACK / "bass.mid").exists(), reason="Regime CLT pack not cached")


def _gen(tmp_path, *extra):
    out = tmp_path / "o.strudel"
    r = subprocess.run([sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"), "--stems-dir", str(STEMS),
                        "--pack-dir", str(PACK), "--bass-midi", str(PACK / "bass.mid"),
                        "--lead-midi", str(PACK / "melodic.mid"), "--drums-json", str(PACK / "drums_bands.json"),
                        "--base-url", "http://localhost:1/", "--mode", "synth", "--bpm", "135.017",
                        "--genre", "brazilian_funk", "--num-bars", "77", "--out", str(out), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    return out.read_text(), r


@needs_pack
def test_generator_sub_up_adds_an_octave_to_the_sine_only(tmp_path):
    base, _ = _gen(tmp_path)
    up, r = _gen(tmp_path, "--sub-up", "1")
    sine = lambda t: next(ln for ln in t.splitlines() if '.s("sine")' in ln)       # noqa: E731
    assert ".add(note(12))" in sine(up) and ".add(" not in sine(base)
    assert ".lpf(180)" in sine(up) and ".lpf(90)" in sine(base)          # the layer's low-pass moves up with it
    assert json.loads(r.stdout[r.stdout.index("{"):])["sub_up"] == 1
