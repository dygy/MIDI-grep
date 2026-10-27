# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""Calibration loop control: revert-on-regression with a reduced step, noise-aware BEST, early stop, cap.

Defects (VAGABUNDO synth cal1 0.7237 -> cal2 0.6930 -> cal3 0.7240; sample-instrument promoted cal3 0.7170 over
cal2 0.7214 for +0.0006 section-aware; a fixed 3 iterations). Stubbed driver runs, no render.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"
sys.path.insert(0, str(SCRIPTS))
import calibrate_loop as CL  # noqa: E402

STUB_GEN = r'''
import json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
open(out, "w").write("// stub\n")
open(out + ".args", "w").write(" ".join(a))
open(__import__("os").environ["AC_LOG"], "a").write("bm=" + a[a.index("--bass-mult") + 1] + "\n")
print(json.dumps({"out": out, "bass_lpf": 603, "bass_lpf_analysis": 603, "bass_note_max_hz": 131.0,
                  "lead_hpf": 300, "lead_note_min_hz": 130.8}))
'''
STUB_CMP = r'''
import json, os
f = os.environ["AC_SCORES"]
lines = open(f).read().split("\n")
i = int(open(f + ".n").read()) if os.path.exists(f + ".n") else 0
open(f + ".n", "w").write(str(i + 1))
o, s = (lines[min(i, len(lines) - 1)].split() + ["0"])[:2]
print(json.dumps({"comparison": {"overall_similarity": float(o), "section_aware_similarity": float(s)}}))
'''
STUB_GATE = r'''
import argparse, json
p = argparse.ArgumentParser(); p.add_argument("--comparison"); p.add_argument("--genre"); p.add_argument("--mode")
c = json.load(open(p.parse_args().comparison))["comparison"]
o, s = c["overall_similarity"], c["section_aware_similarity"]
m = min(o - 0.5, s - 0.5)
print(f"{m:.4f} {'PASS' if m >= 0 else 'FAIL'} {o:.4f} {s:.4f} 0.5000 0.5000")
'''
STUB_CAL = r'''
import sys
a = sys.argv[1:]
bm = float(a[a.index("--bass-mult") + 1])
print(f"--bass-mult {bm + 0.1:.4f} --sub-gain 0.8 --cal-lead 1.1 --lead-lpf 6000 --hat-gain 0.1 --master-gain 0.65")
'''
STUB_REC = '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo wav > "$2"; shift; done\n'


def drive(tmp_path, scores, *args, extra_env=None, cal=STUB_CAL):
    files = {"gen.py": STUB_GEN, "cmp.py": STUB_CMP, "cal.py": cal, "gate.py": STUB_GATE}
    for n, body in files.items():
        (tmp_path / n).write_text(body)
    (tmp_path / "rec.sh").write_text(STUB_REC)
    (tmp_path / "scores.txt").write_text("\n".join(f"{o} {s}" for o, s in scores))
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    (stems / "instrumental.wav").write_text("x")
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp_path / "gen.py"),
           "AC_CMP": str(tmp_path / "cmp.py"), "AC_CAL": str(tmp_path / "cal.py"), "AC_GATE": str(tmp_path / "gate.py"),
           "AC_REC": str(tmp_path / "rec.sh"), "AC_NODE": "/bin/bash", "AC_SETTLE_S": "0",
           "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0", "AC_REF": str(stems / "instrumental.wav"),
           "AC_SCORES": str(tmp_path / "scores.txt"), "AC_LOG": str(tmp_path / "gen.log"), "AC_NOISE": "0.005"}
    env.update(extra_env or {})
    r = subprocess.run(["/bin/bash", "-u", str(AUTO), "--stems", str(stems), "--base", "http://x/", "--mode", "synth",
                        "--start", "0.4 0.7 1.0 5000 0.1 0.6", "--workdir", str(tmp_path / "work"), *args],
                       capture_output=True, text=True, env=env, timeout=120)
    log = (tmp_path / "gen.log")
    bms = [float(ln.split("=")[1]) for ln in log.read_text().split()] if log.exists() else []
    return r, bms


def best_tag(r):
    return [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0].split()[1]


def test_regression_reverts_to_best_with_a_shorter_step(tmp_path):
    # cal1 0.80, cal2 0.70 (regression of 0.10 >> noise), cal3 must restart from cal1 (bm 0.4, proposal 0.5)
    r, bms = drive(tmp_path, [(0.80, 0.80), (0.70, 0.70), (0.82, 0.82)], "--iters", "3")
    assert r.returncode == 0, r.stderr
    assert bms[:2] == [0.4, 0.5]
    alpha = math.sqrt(0.005 / 0.10)                       # derived from the observed loss
    assert bms[2] == pytest.approx(0.4 + alpha * 0.1, abs=1e-3), bms
    assert "regressed vs best cal1" in r.stdout
    assert best_tag(r) == "cal3"


def test_tie_within_noise_prefers_higher_overall_then_section_aware(tmp_path):
    # margins 0.200 vs 0.203 (tie, < noise 0.005): overall 0.800 beats 0.796 although the margin is lower
    r, _ = drive(tmp_path, [(0.800, 0.700), (0.796, 0.703)], "--iters", "2")
    assert best_tag(r) == "cal1", r.stdout
    # equal overall: section-aware decides
    r, _ = drive(tmp_path / "b", [(0.800, 0.700), (0.800, 0.702)], "--iters", "2") if (tmp_path / "b").mkdir() is None else (None, None)
    assert best_tag(r) == "cal2"


def test_clear_margin_difference_beyond_noise_still_wins(tmp_path):
    r, _ = drive(tmp_path, [(0.800, 0.700), (0.790, 0.720)], "--iters", "2")   # margins 0.200 vs 0.220
    assert best_tag(r) == "cal2"


def test_stops_after_patience_steps_without_improvement(tmp_path):
    r, bms = drive(tmp_path, [(0.80, 0.80)], "--iters", "8")                     # flat forever
    assert r.returncode == 0, r.stderr
    assert len(bms) == 1 + CL.PATIENCE, bms
    assert "stop: no improvement beyond noise" in r.stdout


def test_cap_is_honoured_while_still_improving(tmp_path):
    r, bms = drive(tmp_path, [(0.60 + 0.02 * i, 0.60 + 0.02 * i) for i in range(10)], "--iters", "4")
    assert len(bms) == 4 and "iteration cap" not in r.stderr
    assert best_tag(r) == "cal4"


def test_default_cap_is_max_iters(tmp_path):
    r, bms = drive(tmp_path, [(0.50 + 0.02 * i, 0.50 + 0.02 * i) for i in range(12)])
    assert len(bms) == CL.MAX_ITERS and best_tag(r) == f"cal{CL.MAX_ITERS}"


def test_blend_env_interpolates_in_log_space():
    new = {"clamp": [0.5, 2.0], "voices": {"bass": [2.0, 1.0]}}
    used = {"voices": {"bass": [1.0, 1.0]}}
    out = CL.blend_env(used, new, 0.5)
    assert out["voices"]["bass"] == [round(math.sqrt(2), 4), 1.0]
    assert CL.blend_env(used, new, 1.0)["voices"] == new["voices"]


def test_measured_exponent_only_shortens_steps():
    from calibrate_dynamic import measured_exponent
    # VAGABUNDO synth cal1->cal2: hpf 300->428 moved low_mid-per-mid 0.49 -> 1.27
    assert measured_exponent(1.27, 0.49, 428 / 300, +1) == pytest.approx(2.67, abs=0.02)
    assert measured_exponent(1.0, None, 1.4, +1) == 1.0                  # no parent
    assert measured_exponent(0.8, 1.0, 1.4, +1) == 1.0                   # wrong-signed/low response: never lengthen
    assert measured_exponent(2.0, 1.0, 1.01, +1) == 1.0                  # move too small to measure


def test_lead_hpf_step_uses_measured_response(tmp_path):
    def cmp(lm, mid):
        b = {"sub_bass": .3, "bass": .4, "low_mid": lm, "mid": mid, "high_mid": .02, "high": .01}
        return {"original": {"bands": {**b, "low_mid": .06, "mid": .134}, "spectral": {"centroid_mean": 1600}},
                "rendered": {"bands": b, "spectral": {"centroid_mean": 1600}},
                "comparison": {"raw_rms_ratio": 1.0, "overall_similarity": .7}}
    (tmp_path / "parent.json").write_text(json.dumps(cmp(.089, .097)))
    (tmp_path / "now.json").write_text(json.dumps(cmp(.07, .095)))

    def hpf(*extra):
        out = subprocess.run([sys.executable, str(SCRIPTS / "calibrate_dynamic.py"), "--comparison", str(tmp_path / "now.json"),
                              "--mode", "synth", "--lead-hpf", "428", "--lead-note-min-hz", "130", *extra],
                             capture_output=True, text=True, check=True).stdout
        parts = out.split()
        return int(parts[parts.index("--lead-hpf") + 1]) if "--lead-hpf" in parts else 428
    # render still heavy in low_mid (ratio 0.61); the measured response (1.8 from 380->428 Hz) shortens the move
    plain = hpf()
    measured = hpf("--parent-comparison", str(tmp_path / "parent.json"), "--parent-lead-hpf", "380")
    assert abs(measured - 428) < abs(plain - 428), (plain, measured)


# a calibrator that proposes the sub-octave flip whenever it is allowed to (like a render with the sub_bass share over)
STUB_CAL_FLIP = r'''
import sys
a = sys.argv[1:]
bm = float(a[a.index("--bass-mult") + 1])
su, locked = a[a.index("--sub-up") + 1], a[a.index("--sub-up-locked") + 1]
open(__import__("os").environ["AC_LOG"] + ".cal", "a").write(locked + "\n")
flip = su == "0" and locked == "0"
print(f"--bass-mult {bm if flip else bm + 0.1:.4f} --sub-gain 0.8 --cal-lead 1.1 --lead-lpf 6000 --hat-gain 0.1 "
      f"--master-gain 0.65 --sub-up {1 if flip else 0}")
'''


def test_an_octave_flip_the_generator_refuses_is_locked_not_proposed_again(tmp_path):
    """The stub generator never reports sub_up=1 (it refused the flip). The calibrator must then be told the lever is
    locked; before, SULOCK was set only when a LOST step changed the octave, so the flip was re-proposed every step
    and each proposal froze sub-gain/bass-mult."""
    r, _ = drive(tmp_path, [(0.60 + 0.02 * i, 0.60 + 0.02 * i) for i in range(10)], "--iters", "4", cal=STUB_CAL_FLIP)
    assert r.returncode == 0, r.stderr
    locked = (tmp_path / "gen.log.cal").read_text().split()
    assert locked == ["0", "1", "1", "1"], locked
    asked = [(tmp_path / "work" / f"cal{i}.strudel.args").read_text().split() for i in range(1, 5)]
    assert [a[a.index("--sub-up") + 1] for a in asked] == ["0", "1", "0", "0"]
