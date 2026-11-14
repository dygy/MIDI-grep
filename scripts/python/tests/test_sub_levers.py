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
    got = _cal(tmp_path, 0.10, 0.12, "--sub-gain", "0.21")   # slightly over (split inside the dead band): gain lever only
    assert 0.0 < float(got["--sub-gain"]) < 0.21


def test_split_off_moves_a_fraction_of_the_sine_up_and_holds_the_gain(tmp_path):
    """Render sub share 2x the original's (split ratio 0.44, beyond the dead band): f moves by a FRACTION (the share to
    carry over the layer's bound, 0.357 here), not the old whole octave; gain untouched."""
    got = _cal(tmp_path, 0.10, 0.20, "--sub-gain", "0.9")
    assert 0.0 < float(got["--sub-up"]) < 1.0 and float(got["--sub-up"]) == pytest.approx(0.357, abs=1e-3)
    assert float(got["--sub-gain"]) == pytest.approx(0.9)


def test_split_not_moved_when_the_generator_refused_the_up_sine(tmp_path):
    got = _cal(tmp_path, 0.10, 0.20, "--sub-gain", "0.9", "--sub-up-max", "0")
    assert got["--sub-up"] == "0" and float(got["--sub-gain"]) < 0.9          # falls back to the gain lever


def test_octave_up_stays_on_and_gain_corrects_when_already_up(tmp_path):
    over_bass = {"sub_bass": .11, "bass": .52, "low_mid": .09, "mid": .09, "high_mid": .09, "high": .10}   # split 1.15: inside the band
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
    """Sine an octave up (f = 1), split inside the dead band, render sub_bass 32% (< 40%) and bass 43% (~ 41%). The
    sine feeds the BASS band, so its gain must FALL (the old rule raised it 0.70 -> 0.946 on Regime CLT synth cal2 and
    bass reached 79%), while the main layer, which feeds sub_bass, rises."""
    short_sub = {"sub_bass": .323, "bass": .432, "low_mid": .096, "mid": .130, "high_mid": .010, "high": .009}
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, REGIME_ORIG, short_sub)),
                          "--sub-gain", "0.7", "--sub-up", "1", "--bass-mult", "0.784"],
                         capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert float(got["--sub-gain"]) < 0.7 and float(got["--bass-mult"]) > 0.784 and got["--sub-up"] == "1"


def test_octave_zero_keeps_the_original_assignment(tmp_path):
    """Octave 0: sub_bass short raises the sine, bass over lowers the main layer (unchanged behaviour)."""
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, REGIME_ORIG, REGIME_CAL2)),
                          "--sub-gain", "0.7", "--sub-up", "0", "--sub-up-max", "0", "--bass-mult", "0.784"],
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
    assert 0.0 < float(got["--sub-up"]) < 1.0 and float(got["--bass-mult"]) == pytest.approx(0.56) and float(got["--sub-gain"]) == pytest.approx(0.7)


def test_octave_up_is_tried_for_a_modest_excess_too(tmp_path):
    """Regime CLT sample: render sub 49.7% vs original 40.4% (ratio 0.81, inside the old 1.33 dead band) - the
    split step is a scored trial, so it must be proposed (as a fraction now)."""
    got = _cal(tmp_path, 0.404, 0.497, "--sub-gain", "0.623")
    assert 0.0 < float(got["--sub-up"]) < 1.0 and float(got["--sub-gain"]) == pytest.approx(0.623)


def test_no_octave_move_when_sub_bass_is_not_over(tmp_path):
    got = _cal(tmp_path, 0.10, 0.10, "--sub-gain", "0.9")
    assert got["--sub-up"] == "0"
    got = _cal(tmp_path, 0.10, 0.08, "--sub-gain", "0.9")        # render short of sub: gain lever only
    assert got["--sub-up"] == "0" and float(got["--sub-gain"]) > 0.9


def test_blend_interpolates_the_octave_fraction():
    parent = "0.4 0.7 1.0 5000 0.1 0.6 0 0 0".split()
    prop = "0.5 0.9 1.0 5000 0.1 0.6 0 0 1".split()
    assert CL.blend_knobs(parent, prop, [603, 300], 1.0)[8] == "1"
    shortened = CL.blend_knobs(parent, prop, [603, 300], 0.3)
    assert float(shortened[8]) == pytest.approx(0.3) and float(shortened[1]) == pytest.approx(0.76, abs=1e-3)


@pytest.mark.parametrize("text,ok", [
    ("0.4 0.7 1.0 5000 0.1 0.6 0 0 1", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 422 300 0", True),
    ("0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300", False),     # removed CV layout: 7th value ~1 is not a cutoff
    ("0.4 0.7 1.0 5000 0.1 0.6 0 0 0.357", True),         # a fraction of the sine layer an octave up
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


# ---- continuous split: the math ----
import calibrate_dynamic as CD  # noqa: E402


def _mix(s_other, b_other, e, f):
    """Band shares of a synthetic render: the sine layer holds E of the mix, (1-f) of it in sub_bass, f in bass."""
    return {"sub_bass": s_other + e * (1 - f), "bass": b_other + e * f}


def test_split_step_recovers_a_known_imbalance_with_a_measured_parent():
    """Original sub 45.2 / bass 45.3 (Sem Aliança). The render has the sine layer (E = 0.40 of the mix) all at the note
    (f = 0.2 -> parent f = 0.1: only f differed). With E measured from the pair the step lands on the original's split."""
    e, s_other, b_other = 0.40, 0.30, 0.10
    orig = {"sub_bass": 0.452, "bass": 0.453}
    parent, now = _mix(s_other, b_other, e, 0.1), _mix(s_other, b_other, e, 0.2)
    f_next = CD.sub_split_step(orig, now, 0.2, 1.0, {"f": 0.1, "rb": parent})
    after = _mix(s_other, b_other, e, f_next)
    assert after["sub_bass"] / (after["sub_bass"] + after["bass"]) == pytest.approx(0.452 / (0.452 + 0.453), abs=1e-6)


def test_split_step_without_a_parent_undershoots_and_never_passes_the_target():
    """No measurement: E is bounded by the donor band's share, so the step is no longer than the true one."""
    orig = {"sub_bass": 0.452, "bass": 0.453}
    for e in (0.2, 0.4, 0.6):                      # the layer's true share of the mix; it must fit in the donor band
        now = _mix(0.644 - e, 0.248, e, 0.0)       # f = 0: the layer sits in sub_bass (64.4 %), bass 24.8 % (Sem v005)
        f_next = CD.sub_split_step(orig, now, 0.0)
        assert 0.0 < f_next <= (0.644 - 0.4457 + 0.0) / e + 1e-9       # the true f is (Sr - target) / E
    # and in the other direction (Tatuagem-like: sub short, bass over, the sine layer up at f = 1)
    now = _mix(0.20, 0.30, 0.18, 1.0)              # sub 20 %, bass 48 %
    f_next = CD.sub_split_step({"sub_bass": 0.469, "bass": 0.379}, now, 1.0)
    assert 0.0 < f_next < 1.0


def test_split_step_is_inside_the_bounds_and_silent_in_the_dead_band():
    assert CD.sub_split_step({"sub_bass": .4, "bass": .4}, {"sub_bass": .41, "bass": .39}, 0.3) is None
    assert CD.sub_split_step({"sub_bass": .1, "bass": .6}, {"sub_bass": .6, "bass": .1}, 0.0, 0.0) == 0.0   # refused up sine
    assert CD.sub_split_step({"sub_bass": .1, "bass": .6}, {"sub_bass": .6, "bass": .1}, 1.0) is None       # sub over, all up already
    assert CD.sub_split_step({"sub_bass": .6, "bass": .1}, {"sub_bass": .1, "bass": .6}, 0.0) is None       # bass over, none up
    assert CD.sub_split_step({"sub_bass": .1, "bass": .6}, {"sub_bass": .6, "bass": .1}, 0.0) is not None


def test_measured_parent_is_ignored_when_the_levels_moved_too(tmp_path):
    """--parent-knobs with a different sub-gain/bass-mult is not a clean f step: the cold bound is used (same answer as no parent)."""
    parent = _comparison_bands(tmp_path, {"sub_bass": .1, "bass": .2}, {"sub_bass": .16, "bass": .12})
    base = ["--sub-gain", "0.9", "--bass-mult", "0.4"]
    args = [sys.executable, str(CAL), "--comparison", str(_comparison(tmp_path, 0.10, 0.20)), *base]
    cold = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    changed = subprocess.run([*args, "--parent-comparison", str(parent), "--parent-knobs", "0.4 0.5 1 5000 0.1 0.6 0 0 0"],
                             capture_output=True, text=True, check=True).stdout
    assert changed.split("--sub-up")[1].split()[0] == cold.split("--sub-up")[1].split()[0]


def test_old_binary_params_read_as_fractions(tmp_path):
    """A params/state JSON or --sub-up 0/1 from before the fraction reads as f = 0 / 1 and writes back the same text."""
    state = tmp_path / "prev.json"
    state.write_text(json.dumps({"sub_up": 1, "sub_gain": 0.9, "bass_mult": 0.5}))
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison(tmp_path, 0.10, 0.10)),
                          "--state", str(state)], capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert got["--sub-up"] == "1"
    assert CD.fmt_fraction(0.0) == "0" and CD.fmt_fraction(1.0) == "1" and CD.fmt_fraction(0.3571) == "0.357"


def test_calibration_memory_reads_old_rows_as_fractions():
    import calibration_memory as CM
    knobs = CM.parse_knobs("0.4 0.7 1.0 5000 0.1 0.6 0 0 1")
    assert knobs["sub_up"] == 1.0
    assert CM.format_knobs({**knobs, "sub_up": 0.357}).split()[-1] == "0.357"
    assert CM.format_knobs(knobs).split()[-1] == "1"
    assert CM._driving_bands("sub_gain", 1) == ("bass",) and CM._driving_bands("sub_gain", 0.2) == ("sub_bass",)
    assert CM._driving_bands("sub_gain", 0.7) == ("bass",)      # mostly up: the sine feeds the bass band


def test_bass_mult_has_no_absolute_ceiling(tmp_path):
    """bass-mult was clamped to 1.3 (pegged on every synth track incl. the references): bass share half of the original's
    must raise it from 1.3 by the per-step bound, not hold it."""
    got = _cal(tmp_path, 0.10, 0.10, "--bass-mult", "1.3")      # even shares: no move
    assert float(got["--bass-mult"]) == pytest.approx(1.3, abs=0.02)
    low_bass = {"sub_bass": .30, "bass": .15, "low_mid": .15, "mid": .15, "high_mid": .13, "high": .12}
    orig = {"sub_bass": .30, "bass": .30, "low_mid": .10, "mid": .10, "high_mid": .10, "high": .10}
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_bands(tmp_path, orig, low_bass)),
                          "--bass-mult", "1.3", "--sub-up-max", "0"], capture_output=True, text=True, check=True).stdout.split()
    got = {out[i]: out[i + 1] for i in range(0, len(out), 2)}
    assert float(got["--bass-mult"]) > 1.3


@needs_pack
def test_generator_fraction_makes_two_sines_with_the_energy_split(tmp_path):
    text, r = _gen(tmp_path, "--sub-up", "0.36")
    sines = [ln for ln in text.splitlines() if '.s("sine")' in ln and ".orbit" not in ln.split('.s("sine")')[0][:0]]
    sines = [ln for ln in sines if "lpf(" in ln]
    assert len(sines) == 2
    note, up = (next(ln for ln in sines if k(ln)) for k in (lambda l: ".add(" not in l, lambda l: ".add(note(12))" in l))
    assert ".lpf(90)" in note and ".lpf(180)" in up
    gains = lambda ln: [float(x) for x in re.findall(r"[0-9.]+", ln.split(".gain(")[1].split(")")[0])]   # noqa: E731
    gn, gu = gains(note), gains(up)
    assert sum(g * g for g in gu) / sum(g * g for g in gn) == pytest.approx(0.36 / 0.64, rel=0.02)    # energy f : 1-f
    assert json.loads(r.stdout[r.stdout.index("{"):])["sub_up"] == pytest.approx(0.36)


@needs_pack
def test_generator_endpoints_are_the_single_sine_layers(tmp_path):
    """f = 0 / 1 emit exactly one sine voice, as the binary lever did (the full-text equivalence with the commit before
    this change was diffed once by hand; this pins the voice shape)."""
    for f, lpf, add in (("0", 90, False), ("1", 180, True)):
        text, _ = _gen(tmp_path, "--sub-up", f)
        sines = [ln for ln in text.splitlines() if '.s("sine")' in ln and "lpf(" in ln]
        assert len(sines) == 1 and f".lpf({lpf})" in sines[0] and (".add(note(12))" in sines[0]) is add


@needs_pack
@pytest.mark.parametrize("f", ["0", "1"])
def test_generator_endpoints_equal_the_binary_levers_output(tmp_path, f):
    """The sine row written by the generator BEFORE the fraction (binary --sub-up 0 / 1, Regime CLT pack, synth) is stored
    in fixtures/calibrate/sub_sine_row_f<f>.txt; f = 0 / 1 must reproduce it byte for byte."""
    text, _ = _gen(tmp_path, "--sub-up", f)
    row = [ln for ln in text.splitlines() if '.s("sine")' in ln and "lpf(" in ln]
    want = (SCRIPTS / "tests" / "fixtures" / "calibrate" / f"sub_sine_row_f{f}.txt").read_text().splitlines()
    assert row == want


def _comparison_dark(tmp_path: Path, orig_sub: float, rend_sub: float) -> Path:
    """_comparison with the render twice too dark (centroid 1000 vs 2000): the hat lever wants to move."""
    p = _comparison(tmp_path, orig_sub, rend_sub)
    d = json.loads(p.read_text())
    d["original"]["spectral"]["centroid_mean"] = 2000.0
    p.write_text(json.dumps(d))
    return p


def _cal_dark(tmp_path, orig_sub, rend_sub, *extra):
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(_comparison_dark(tmp_path, orig_sub, rend_sub)), *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


KNOBS_DARK = ("--sub-gain", "0.9", "--lead-lpf", "8000", "--hat-gain", "0.1", "--cal-lead", "1.0")


def test_the_split_step_moves_no_other_lever(tmp_path):
    """Regression (Sem Aliança synth 2026-11-13): cal2 moved f together with the hats, lead-hpf and cal-lead and the split
    was judged on that mix (0.737 -> 0.707); the same f alone scored 0.747 / 0.765. While f moves, nothing else does."""
    control = _cal_dark(tmp_path, 0.10, 0.10, *KNOBS_DARK)          # split in balance: the dark mix moves the hats
    assert float(control["--hat-gain"]) > 0.1
    got = _cal_dark(tmp_path, 0.10, 0.20, *KNOBS_DARK)              # split off: f moves, the hats and the lead hold
    assert 0.0 < float(got["--sub-up"]) < 1.0
    assert float(got["--hat-gain"]) == pytest.approx(0.1)
    assert float(got["--lead-lpf"]) == pytest.approx(8000)
    assert float(got["--cal-lead"]) == pytest.approx(1.0)


def test_the_loop_holds_the_master_gain(tmp_path):
    """Regression (2026-11-13): a warm start carries the loudness stage's gain (2.299); a 0.78 cap left from the BlackHole
    limiter pulled it 9.4 dB down on the first step. The level is the loudness stage's lever; the loop holds it."""
    got = _cal(tmp_path, 0.10, 0.10, "--master-gain", "2.299")
    assert float(got["--master-gain"]) == pytest.approx(2.299)


def _solo(tmp_path: Path, r: float) -> Path:
    p = tmp_path / "bass_solo.json"
    p.write_text(json.dumps({"ratio": {"sub_bass": r, "bass": r}}))
    return p


def test_synth_bass_layer_follows_the_mix_low_end_not_the_solo_voice(tmp_path):
    """Regression (2026-11-13): on synth pieces the bass voice alone read 6-15x under its stem while the mix low end matched
    within a point; bass-mult 1.3 -> 2.6 lost on every track (Regime CLT 0.909 -> 0.834). In synth the layer moves only the
    way the mix low end (sub_bass + bass) asks, outside the share dead band."""
    solo = str(_solo(tmp_path, 15.0))
    base = ("--bass-mult", "1.3", "--sub-up-max", "0", "--bass-solo", solo)
    held = _cal(tmp_path, 0.40, 0.40, "--mode", "synth", *base)          # mix low end matched: hold
    assert float(held["--bass-mult"]) == pytest.approx(1.3)
    control = _cal(tmp_path, 0.40, 0.40, "--mode", "sample-instrument", *base)   # sample-instrument keeps the voice-solo step
    assert float(control["--bass-mult"]) > 1.3
    short = _cal(tmp_path, 0.60, 0.20, "--mode", "synth", *base)         # mix low end short beyond the dead band: raise
    assert float(short["--bass-mult"]) > 1.3
