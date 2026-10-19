# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""Bug #11 — synth-mode calibration: bass low-pass lever, hat engagement, 8-knob contract.

A sawtooth bass puts harmonics 2-5 in 250-500 Hz (low_mid) while the bass band drains; the only
lever that fixes it is the bass cutoff. These tests pin the lever's direction, bounds, CLI/shell
contract, and that the sample-instrument levers are untouched (golden values captured from the
pre-change calibrator on stored comparison.json files).

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_calibrate_bass_lpf.py -q
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate"
CAL = SCRIPTS / "calibrate_dynamic.py"
AUTO = REPO / "scripts" / "auto-calibrate.sh"

KNOBS = ["--bass-mult", "0.5", "--sub-gain", "0.8", "--cal-lead", "1.1", "--lead-lpf", "6000",
         "--hat-gain", "0.1", "--master-gain", "0.65"]
# top bass note ~131 Hz (C3) -> floor 262 Hz; analysis cutoff 603 (the v003 value)
BLPF = ["--mode", "synth", "--bass-lpf", "603", "--bass-lpf-max", "603", "--bass-note-max-hz", "131"]


def run_cal(cmp_path: Path, *extra: str) -> dict[str, str]:
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(cmp_path), *extra],
                         capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def with_bands(tmp_path: Path, lm_over_bass_rend: float) -> Path:
    """v003 with the rendered low_mid set so rend low_mid/bass == the given value (orig stays)."""
    d = json.loads((FIX / "vagabundo_synth_v003.json").read_text())
    d["rendered"]["bands"]["bass"] = 0.2
    d["rendered"]["bands"]["low_mid"] = 0.2 * lm_over_bass_rend
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    return p


def orig_ratio() -> float:
    b = json.loads((FIX / "vagabundo_synth_v003.json").read_text())["original"]["bands"]
    return b["low_mid"] / b["bass"]


def test_low_mid_excess_lowers_cutoff_damped():
    got = run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS, *BLPF)
    assert int(got["--bass-lpf"]) < 603
    assert int(got["--bass-lpf"]) >= int(603 * 0.7)          # one damped step, not a collapse


def test_low_mid_deficit_raises_cutoff(tmp_path):
    cmp_ = with_bands(tmp_path, orig_ratio() / 3.0)           # rendered low_mid/bass 3x too LOW
    got = run_cal(cmp_, *KNOBS, "--mode", "synth", "--bass-lpf", "400", "--bass-lpf-max", "603", "--bass-note-max-hz", "131")
    assert 400 < int(got["--bass-lpf"]) <= int(400 * 1.3)


def test_parity_unchanged_and_not_emitted(tmp_path):
    cmp_ = with_bands(tmp_path, orig_ratio())
    assert "--bass-lpf" not in run_cal(cmp_, *KNOBS, *BLPF)


def test_bounds_hold(tmp_path):
    # floor: current cutoff near the floor cannot drop below 2x the top bass fundamental
    got = run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS, "--mode", "synth",
                  "--bass-lpf", "300", "--bass-lpf-max", "603", "--bass-note-max-hz", "131")
    assert int(got["--bass-lpf"]) == 262
    # ceiling: a deficit cannot open past the analysis-derived cutoff
    cmp_ = with_bands(tmp_path, orig_ratio() / 5.0)
    got = run_cal(cmp_, *KNOBS, "--mode", "synth", "--bass-lpf", "580", "--bass-lpf-max", "603", "--bass-note-max-hz", "131")
    assert int(got["--bass-lpf"]) == 603


def test_lever_off_when_unmeasured():
    assert "--bass-lpf" not in run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS)                 # no cutoff
    assert "--bass-lpf" not in run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS, "--mode", "synth", "--bass-lpf", "603")  # no note max


def test_hat_engages_when_lead_already_past_high_mid():
    # v003: lead_lpf 5199 < 7000, centroid 1.38x too dark, high+high_mid share 4.5x too low
    got = run_cal(FIX / "vagabundo_synth_v003.json", "--lead-lpf", "5199", "--hat-gain", "0.0")
    assert float(got["--hat-gain"]) > 0.0


def test_hat_stays_off_when_lead_below_high_mid_edge():
    got = run_cal(FIX / "vagabundo_synth_v003.json", "--lead-lpf", "3000", "--hat-gain", "0.0")
    assert float(got["--hat-gain"]) == 0.0


# golden next-knob fragments from the calibrator BEFORE bug #11 (same inputs: KNOBS)
GOLDEN = {
    "vagabundo_sample_v004.json":
        "--bass-mult 0.521 --sub-gain 0.784 --cal-lead 1.082 --lead-lpf 6300 --hat-gain 0.1 "
        "--master-gain 0.728",
    "vagabundo_sample_v002.json":
        "--bass-mult 0.526 --sub-gain 0.759 --cal-lead 1.142 --lead-lpf 6255 --hat-gain 0.1 "
        "--master-gain 0.728",
    "regime_sample_v026.json":
        "--bass-mult 0.541 --sub-gain 0.732 --cal-lead 1.46 --lead-lpf 6652 --hat-gain 0.1 "
        "--master-gain 0.728",
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_sample_instrument_levers_unchanged(name):
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(FIX / name), *KNOBS],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == GOLDEN[name]


@pytest.mark.parametrize("name", ["vagabundo_sample_v002.json", "vagabundo_sample_v004.json",
                                  "regime_sample_v026.json"])
def test_sample_instrument_never_emits_bass_lpf_or_new_hat(name):
    got = run_cal(FIX / name, *KNOBS, "--bass-lpf", "603", "--bass-lpf-max", "603",
                  "--bass-note-max-hz", "131")
    assert "--bass-lpf" not in got                      # inside the dead band on stored comparisons


def test_auto_calibrate_syntax_and_8_field_contract():
    subprocess.run(["bash", "-n", str(AUTO)], check=True)
    src = AUTO.read_text()
    assert 'read -r BM SG CL LLPF HG MG BL LH' in src                       # 6, 7 or 8 fields, BL/LH default 0
    assert '--bass-lpf "$BL"' in src and '--bass-lpf) BL="$2"' in src and '--lead-hpf) LH="$2"' in src
    assert '"$BM $SG $CL $LLPF $HG $MG $BL $LH"' in src and "iters.tsv" in src          # 8 fields recorded per iteration
    assert '--cal-vocal "$CV"' not in src and '--vocal-mode "$VOCAL_MODE"' not in src      # the vocal knob is gone


def test_auto_calibrate_parse_step_round_trip():
    """Run the script's own parse loop + --start read on a calibrator fragment."""
    src = AUTO.read_text()
    start = src.index("  set -- $CLI")
    end = src.index("done", src.index("while [ $# -gt 0 ]; do case", start)) + len("done")
    loop = src[start:end]
    frag = ('--bass-mult 0.5 --sub-gain 0.8 --cal-lead 1.1 --lead-lpf 6000 --hat-gain 0.1 '
            '--master-gain 0.65 --bass-lpf 422 --lead-hpf 428')
    script = (f'CLI="{frag}"; BL=0; LH=0\n{loop}\necho "$BM $SG $CL $LLPF $HG $MG $BL $LH"\n'
              'for s in "1 2 3 4 5 6" "1 2 3 4 5 6 422"; do BL=0; '
              'read -r BM SG CL LLPF HG MG BL <<<"$s"; BL="${BL:-0}"; echo "$BL"; done')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split()
    assert out[:8] == "0.5 0.8 1.1 6000 0.1 0.65 422 428".split()
    assert out[8:] == ["0", "422"]


@pytest.mark.parametrize("old_start", [
    "0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300",     # old 9-field format (… MG CV BL LH)
    "0.4 0.7 1.0 5000 0.1 0.6 1.0",             # old 7-field (… MG CV): CV would land in BL
    "0.4 0.7 1.0 5000 0.1 0.6 1.0 422",         # old 8-field (… MG CV BL)
])
def test_start_in_the_old_cv_format_fails_loudly(old_start):
    r = subprocess.run(["bash", str(AUTO), "--stems", "x", "--base", "y", "--start", old_start],
                       capture_output=True, text=True)
    assert r.returncode == 64 and "--start" in r.stderr and "CV" in r.stderr, r.stderr


@pytest.mark.parametrize("new_start", ["0.4 0.7 1.0 5000 0.1 0.6", "0.4 0.7 1.0 5000 0.1 0.6 422", "0.4 0.7 1.0 5000 0.1 0.6 422 300"])
def test_start_in_the_new_format_is_accepted(new_start):
    r = subprocess.run(["bash", str(AUTO), "--stems", "/nonexistent", "--base", "y", "--start", new_start],
                       capture_output=True, text=True)
    assert "--start" not in r.stderr, r.stderr      # parsed fine (it fails later on the missing stems, not on --start)


def test_generator_reports_top_bass_note_hz():
    sys.path.insert(0, str(SCRIPTS))
    from generate_dynamic_strudel import bars_max_hz
    assert bars_max_hz(["c2 ~ e2 ~", "cs3 ~ ~ a1"]) == pytest.approx(138.59, abs=0.01)   # cs3
    assert bars_max_hz(["~ ~"]) == 0.0 and bars_max_hz(None) == 0.0


# --- driver-level stub run: catches `set -u` + empty-array failures (macOS bash 3.2) -------------
STUB_GEN = r'''
import json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
open(out, "w").write("// stub\n")
print(json.dumps({"out": out, "bass_lpf": 603, "bass_lpf_analysis": 603, "bass_note_max_hz": 131.0,
                  "lead_hpf": 300, "lead_note_min_hz": 130.8}))
'''
STUB_CMP = r'''
import json
print(json.dumps({"comparison": {"overall_similarity": 0.8, "section_aware_similarity": 0.8}}))
'''
STUB_CAL = r'''
print("--bass-mult 0.5 --sub-gain 0.8 --cal-lead 1.1 --lead-lpf 6000 --hat-gain 0.1 "
      "--master-gain 0.65 --bass-lpf 422")
'''
STUB_REC = '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo wav > "$2"; shift; done\n'


def run_driver(tmp_path: Path, script: Path, *args: str, gen: str = STUB_GEN,
               extra_env: dict | None = None) -> subprocess.CompletedProcess:
    for name, body in (("gen.py", gen), ("cmp.py", STUB_CMP), ("cal.py", STUB_CAL)):
        (tmp_path / name).write_text(body)
    (tmp_path / "rec.sh").write_text(STUB_REC)
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    (stems / "original.wav").write_text("x")
    (stems / "instrumental.wav").write_text("x")
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp_path / "gen.py"),
           "AC_CMP": str(tmp_path / "cmp.py"), "AC_CAL": str(tmp_path / "cal.py"),
           "AC_REC": str(tmp_path / "rec.sh"), "AC_NODE": "/bin/bash", "AC_SETTLE_S": "0",
           "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0",
           "AC_REF": str(stems / "instrumental.wav")}     # hermetic: never wait on a real recorder
    env.update(extra_env or {})
    return subprocess.run(["/bin/bash", "-u", str(script), "--stems", str(stems), "--base", "http://x/",
                           "--iters", "1", "--workdir", str(tmp_path / "work"), *args],
                          capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("extra", [
    ["--mode", "synth", "--start", "0.447 0.707 0.969 5199 0.0 0.672"],
    ["--mode", "sample-instrument", "--inst", "http://x/samples.json", "--key", "C# minor"],
])
def test_driver_runs_under_set_u_with_stubs(tmp_path, extra):
    r = run_driver(tmp_path, AUTO, *extra)
    assert r.returncode == 0, r.stderr
    best = [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]
    params = best.split("params:")[1].split()
    assert len(params) == 8, best
    assert params[-2:] == ["0", "0"]   # BL, LH (the overrides that went INTO the best render)


def test_driver_failure_exit_code_and_best_line(tmp_path):
    # generator succeeds on the first call, fails afterwards (marker file)
    gen = ("import os,sys\nm=os.path.join(os.path.dirname(__file__),'seen')\n"
           "if os.path.exists(m): sys.exit(3)\nopen(m,'w').write('x')\n" + STUB_GEN)
    r = run_driver(tmp_path, AUTO, "--mode", "synth", "--iters", "2", gen=gen)
    assert "GEN FAILED" in r.stdout and r.returncode == 70
    assert any(ln.startswith("BEST: cal1") for ln in r.stdout.splitlines())


def test_unparseable_generator_summary_warns(tmp_path):
    gen = STUB_GEN.replace("print(json.dumps(", "print('not json') if False else print(json.dumps(").replace(
        '"lead_hpf": 300, "lead_note_min_hz": 130.8', '"lead_hpf_MISSING": 1')
    r = run_driver(tmp_path, AUTO, "--mode", "synth", gen=gen)
    assert "WARNING: cannot parse the generator summary" in r.stdout and "levers are OFF" in r.stdout


# --- synth-only gating, lead-hpf lever, one-lever-per-step coordination (cal1/cal2 real runs) ------
CAL1 = FIX / "vagabundo_synth_cal1.json"
CAL2 = FIX / "vagabundo_synth_cal2.json"
LEAD = ["--lead-hpf", "300", "--lead-note-min-hz", "130.81"]
SYNTH = ["--mode", "synth"]
CAL2_KNOBS = ["--bass-mult", "0.474", "--sub-gain", "0.836", "--cal-lead", "0.735", "--lead-lpf", "5923",
              "--hat-gain", "0.135", "--master-gain", "0.753"]
CAL1_KNOBS = ["--bass-mult", "0.447", "--sub-gain", "0.707", "--cal-lead", "0.969", "--lead-lpf", "5199",
              "--hat-gain", "0.0", "--master-gain", "0.672"]
BASS = ["--bass-lpf", "422", "--bass-lpf-max", "603", "--bass-note-max-hz", "123.47"]


def test_levers_are_synth_only():
    # same v003 data that moves both levers in synth mode: silent in the default sample-instrument mode
    got = run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS, "--bass-lpf", "603", "--bass-lpf-max", "603",
                  "--bass-note-max-hz", "131", *LEAD)
    assert "--bass-lpf" not in got and "--lead-hpf" not in got
    got = run_cal(FIX / "vagabundo_synth_v003.json", *KNOBS, *SYNTH, "--bass-lpf", "603", "--bass-lpf-max", "603",
                  "--bass-note-max-hz", "131", *LEAD)
    assert "--lead-hpf" in got


def test_lead_hpf_direction_bounds_and_dead_band(tmp_path):
    got = run_cal(CAL2, *CAL2_KNOBS, *SYNTH, *LEAD)                      # render low_mid/mid 3.2x too high
    assert 300 < int(got["--lead-hpf"]) <= int(300 / 0.7) + 1            # raised, one damped step
    near = run_cal(CAL2, *CAL2_KNOBS, *SYNTH, "--lead-hpf", "495", "--lead-note-min-hz", "130.81")
    assert int(near["--lead-hpf"]) == 500                                # ceiling = low_mid band top
    d = json.loads(CAL2.read_text())                                     # deficit: render low_mid/mid 3x too LOW
    d["rendered"]["bands"]["low_mid"] = d["original"]["bands"]["low_mid"] * d["rendered"]["bands"]["mid"] / d["original"]["bands"]["mid"] / 3
    p = tmp_path / "d.json"; p.write_text(json.dumps(d))
    low = run_cal(p, *CAL2_KNOBS, *SYNTH, "--lead-hpf", "140", "--lead-note-min-hz", "130.81")
    assert int(low["--lead-hpf"]) == 131                                 # floor = lowest lead fundamental
    d["rendered"]["bands"]["low_mid"] = d["original"]["bands"]["low_mid"] * d["rendered"]["bands"]["mid"] / d["original"]["bands"]["mid"]
    p.write_text(json.dumps(d))
    assert "--lead-hpf" not in run_cal(p, *CAL2_KNOBS, *SYNTH, *LEAD)    # parity -> unchanged
    assert "--lead-hpf" not in run_cal(CAL2, *CAL2_KNOBS, *SYNTH)        # no lead info -> off


def test_one_lever_per_step_lead_hpf_has_priority_over_bass_lpf():
    got = run_cal(CAL2, *CAL2_KNOBS, *SYNTH, *LEAD, *BASS)
    assert "--lead-hpf" in got and "--bass-lpf" not in got


def test_bass_mult_not_raised_while_cutoff_moves():
    # cal2: bass-lpf alone moves 422 -> 295 and bass-mult must NOT rise (real cal3 raised it 0.474 -> 0.488)
    got = run_cal(CAL2, *CAL2_KNOBS, *SYNTH, *BASS)
    assert int(got["--bass-lpf"]) < 422
    assert float(got["--bass-mult"]) <= 0.474
    free = run_cal(CAL2, *CAL2_KNOBS, *SYNTH)                            # no cutoff move -> old behaviour
    assert float(free["--bass-mult"]) == pytest.approx(0.488)


def test_hats_via_drums_hold_lead_lpf_on_cal1():
    # cal1 -> real cal2 moved BOTH hat 0 -> 0.135 and lead-lpf 5199 -> 5923; one brightness lever per step
    got = run_cal(CAL1, *CAL1_KNOBS, *SYNTH)
    assert float(got["--hat-gain"]) == pytest.approx(0.135)
    assert int(got["--lead-lpf"]) == 5199
    # cal2 -> cal3: hats not engaged (centroid 1.11x), lead-lpf still opens as before
    assert int(run_cal(CAL2, *CAL2_KNOBS, *SYNTH)["--lead-lpf"]) == 6178


def test_lead_env_not_raised_on_a_lead_hpf_step(tmp_path):
    out = tmp_path / "env.json"
    run_cal(CAL2, *CAL2_KNOBS, *SYNTH, *LEAD, "--env-correction-out", str(out), "--bpm", "129", "--bars", "81")
    assert max(json.loads(out.read_text())["voices"]["lead"]) <= 1.0
    out2 = tmp_path / "env2.json"
    run_cal(CAL2, *CAL2_KNOBS, *SYNTH, "--env-correction-out", str(out2), "--bpm", "129", "--bars", "81")
    assert max(json.loads(out2.read_text())["voices"]["lead"]) > 1.0     # no hpf move -> lead may rise


def test_band_edges_come_from_compare_audio():
    sys.path.insert(0, str(SCRIPTS))
    import calibrate_dynamic as C
    from compare_audio import FREQ_BANDS
    assert C.HIGH_MID_TOP_HZ == FREQ_BANDS["high_mid"][1] and C.LOW_MID_TOP_HZ == FREQ_BANDS["low_mid"][1]


def test_llms_full_hat_rule_mentions_centroid_gate():
    txt = (REPO / "llms-full.txt").read_text()
    row = [ln for ln in txt.splitlines() if ln.startswith("| `calibrate_dynamic.py`")][0]
    assert "r_cent > 1.15" in row and "lead_hpf" in row


# --- pass 3: hat overshoot, gate-aware BEST, partial-register octave ----------------
RUN2_CAL1 = FIX / "vagabundo_synth_run2_cal1.json"
RUN2_KNOBS = ["--mode", "synth", "--bass-mult", "0.474", "--sub-gain", "0.836", "--cal-lead", "0.735",
              "--lead-lpf", "5923", "--hat-gain", "0.135", "--master-gain", "0.753"]


def test_hat_backs_off_on_drums_stem_high_overshoot(tmp_path):
    cmp_ = tmp_path / "comparison.json"
    cmp_.write_text(RUN2_CAL1.read_text())
    (tmp_path / "stem_comparison.json").write_text((FIX / "vagabundo_synth_v006_stem_drums.json").read_text())
    got = run_cal(cmp_, *RUN2_KNOBS)                         # drums high 21% vs 0.7% (v006): layer must shrink
    assert 0.0 < float(got["--hat-gain"]) <= 0.135 * 0.5 + 0.001


def test_hat_drums_path_shut_when_full_mix_high_already_above_original(tmp_path):
    d = json.loads((FIX / "vagabundo_synth_v003.json").read_text())
    d["rendered"]["bands"]["high"] = d["original"]["bands"]["high"] * 1.6      # hats cannot be the missing source
    p = tmp_path / "c.json"; p.write_text(json.dumps(d))
    assert float(run_cal(p, "--mode", "synth", "--lead-lpf", "5199", "--hat-gain", "0.0")["--hat-gain"]) == 0.0
    # control: same file with the original `high` -> the drums path engages (test_hat_engages_... above)
    assert float(run_cal(FIX / "vagabundo_synth_v003.json", "--mode", "synth", "--lead-lpf", "5199",
                         "--hat-gain", "0.0")["--hat-gain"]) > 0.0


def test_gate_margin_picks_nearest_to_floors():
    sys.path.insert(0, str(SCRIPTS))
    import gate_margin
    # real run 2: cal1 0.8229/0.7955 is nearer than cal2 0.8269/0.777 (floors read from thresholds.yaml, never retyped)
    from eval import gate
    thr = gate.load_thresholds()
    f, sf = gate.floor_for_genre("brazilian_funk", thr, "synth"), gate.section_aware_floor_for_genre("brazilian_funk", thr, "synth")
    def m(o, sa):
        return min(o - f, sa - sf)
    assert m(0.8229, 0.7955) > m(0.8269, 0.777) > m(0.7975, 0.7706)
    got = gate_margin.margins(RUN2_CAL1, "brazilian_funk", "synth")
    assert got["floor"] == pytest.approx(f) and got["sa_floor"] == pytest.approx(sf)
    assert got["margin"] == pytest.approx(min(got["overall"] - f, got["sa"] - sf))


def _seq_cmp(tmp_path: Path, rows: list[tuple[float, float]]) -> str:
    return ("import json,os\nd=os.path.dirname(__file__);f=os.path.join(d,'n');n=int(open(f).read()) if os.path.exists(f) else 0\n"
            "open(f,'w').write(str(n+1));rows=%r\no,s=rows[min(n,len(rows)-1)]\n"
            "print(json.dumps({'comparison':{'overall_similarity':o,'section_aware_similarity':s}}))\n" % (rows,))


def test_best_is_gate_aware_not_highest_overall(tmp_path):
    cmp_src = _seq_cmp(tmp_path, [(0.8229, 0.7955), (0.8269, 0.777)])
    global STUB_CMP
    old, STUB_CMP = STUB_CMP, cmp_src
    try:
        r = run_driver(tmp_path, AUTO, "--mode", "synth", "--iters", "2", "--genre", "brazilian_funk")
    finally:
        STUB_CMP = old
    assert r.returncode == 0, r.stderr
    best = [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]
    assert best.startswith("BEST: cal1 "), best                      # cal2 has the higher overall
    assert "FAIL" in r.stdout and "gate:" in r.stdout


def test_best_falls_back_to_overall_with_warning_when_gate_unreadable(tmp_path):
    global STUB_CMP
    old, STUB_CMP = STUB_CMP, _seq_cmp(tmp_path, [(0.8229, 0.7955), (0.8269, 0.777)])
    try:
        r = run_driver(tmp_path, AUTO, "--mode", "synth", "--iters", "2", extra_env={"AC_GATE": "/usr/bin/false"})
    finally:
        STUB_CMP = old
    assert "WARNING: gate margin unavailable" in r.stdout
    assert [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0].startswith("BEST: cal2 ")


def _tone_stem(tmp_path: Path, partial: int):
    import numpy as np
    import pretty_midi
    import soundfile as sf
    sr, f0 = 22050, 220.0
    t = np.arange(int(sr * 6)) / sr
    sf.write(tmp_path / f"s{partial}.wav", 0.3 * np.sin(2 * np.pi * f0 * partial * t), sr)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    for i in range(5):
        inst.notes.append(pretty_midi.Note(80, 57, 0.2 + i, 0.2 + i + 0.8))      # A3 = 220 Hz
    pm.instruments.append(inst); pm.write(str(tmp_path / "n.mid"))
    return tmp_path / f"s{partial}.wav", tmp_path / "n.mid"


def test_partial_register_statistic_and_octave_shift(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    wav, mid = _tone_stem(tmp_path, 1)
    assert PR.dominant_partial_log2(wav, mid) == pytest.approx(0.0, abs=0.1)        # fundamental dominates
    wav2, mid = _tone_stem(tmp_path, 2)
    assert PR.dominant_partial_log2(wav2, mid) == pytest.approx(1.0, abs=0.1)       # octave dominates
    ref = {"snd": -0.095, "lead": 0.472}                                             # scalar back-compat table
    assert PR.octave_shift(1.253, "snd", ref) == 1
    assert PR.octave_shift(0.776, "lead", ref) == 0
    assert PR.octave_shift(9.0, "snd", ref) == PR.MAX_OCTAVE_SHIFT                   # bounded
    assert PR.octave_shift(-1.0, "snd", ref) == 0                                    # never down
    assert PR.octave_shift(None, "snd", ref) == 0


def test_shipped_reference_has_nothing_per_track():
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    table = PR.load_table()
    assert all(str(v.get("source", "")).startswith("chromatic sweep probe") for v in table.values())


VAG = REPO / ".cache" / "stems" / "VAGABUNDO NÃO NAMORA"


@pytest.mark.skipif(not (VAG / "sample_pack" / "melodic.mid").exists(), reason="VAGABUNDO stems not cached")
@pytest.mark.parametrize("mode", ["synth", "sample-instrument"])
def test_generator_summary_has_no_vocal_fields_and_derives_lead_octave_only_in_synth(tmp_path, mode):
    pack = VAG / "sample_pack"
    ref = tmp_path / "ref.json"      # synthetic sweep table (a test double, not a measurement)
    ref.write_text(json.dumps({"sounds": {"gm_lead_2_sawtooth": {"log2": 0.5, "source": "chromatic sweep probe test"}}}))
    cmd = [sys.executable, "-W", "ignore", str(SCRIPTS / "generate_dynamic_strudel.py"), "--stems-dir", str(VAG),
           "--pack-dir", str(pack), "--bass-midi", str(pack / "bass.mid"), "--lead-midi", str(pack / "melodic.mid"),
           "--drums-json", str(pack / "drums_bands.json"), "--base-url", "http://x/", "--mode", mode, "--bpm", "129",
           "--genre", "brazilian_funk", "--num-bars", "24", "--drum-mode", "bank" if mode == "synth" else "extracted",
           "--partial-reference", str(ref), "--out", str(tmp_path / "o.strudel")]
    if mode != "synth":
        cmd += ["--samples-url", "http://x/instruments/samples.json"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip().startswith("{"):
        pytest.skip(f"generator not runnable here: {r.stderr[-200:]}")
    summ = json.loads(r.stdout)
    assert not [k for k in summ if "vocal" in k], "vocals are never generated: no vocal field in the summary"
    if mode != "synth":
        assert summ["lead_octave"] == 0


# --- pass 4: probe tool, reference hygiene, cache, argparse type, hat/vocal/gate honesty ---------------
def _sweep_wav(tmp_path: Path, lo: int, hi: int, harmonic: int, lead_in: float = 0.7) -> Path:
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(SCRIPTS))
    import probe_sound_register as P
    sr = 22050
    y = np.zeros(int(sr * (lead_in + (hi - lo + 1) * P.PROBE_NOTE_S + 0.5)))
    for i, m in enumerate(range(lo, hi + 1)):
        f0 = 440 * 2 ** ((m - 69) / 12)
        t = np.arange(int(sr * P.PROBE_NOTE_S)) / sr
        tone = 0.3 * np.sin(2 * np.pi * f0 * t) + 1.0 * np.sin(2 * np.pi * f0 * harmonic * t)   # known register
        a = int(sr * (lead_in + i * P.PROBE_NOTE_S))
        y[a:a + tone.size] += 0.2 * tone
    sf.write(tmp_path / f"sweep{harmonic}.wav", y, sr)
    return tmp_path / f"sweep{harmonic}.wav"


@pytest.mark.parametrize("harmonic,expect", [(1, 0.0), (2, 1.0), (4, 2.0)])
def test_sweep_probe_recovers_a_known_register(tmp_path, harmonic, expect):
    sys.path.insert(0, str(SCRIPTS))
    import probe_sound_register as P
    lo, hi = 57, 72
    # at harmonic 1 the 0.3-amplitude partial is ALSO the fundamental, so the dominant one is f0 itself
    wav = _sweep_wav(tmp_path, lo, hi, harmonic)
    per = P.analyse_sweep(wav, lo, hi)
    assert len(per) == hi - lo + 1
    assert all(v == pytest.approx(expect, abs=0.15) for v in per.values()), per
    ref = tmp_path / "ref.json"
    ent = P.merge_reference("snd", per, lo, hi, ref)
    assert ent["range"] == [lo, hi] and ent["source"].startswith("chromatic sweep probe")
    assert ent["log2"] == pytest.approx(expect, abs=0.15) and len(ent["per_note"]) == hi - lo + 1


def test_probe_cli_dry_run_writes_strudel_without_rendering(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPTS / "probe_sound_register.py"), "--sound", "gm_voice_oohs",
                        "--dry-run", "--workdir", str(tmp_path)], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    code = Path(info["strudel"]).read_text()
    assert '.s("gm_voice_oohs")' in code and "c2 cs2" in code and code.rstrip().endswith('.s("gm_voice_oohs")')
    assert not any(w in code for w in ("lpf", "hpf", "gain(", ".gain", "room"))     # no filters, no gain pattern
    assert "render-strudel-offline.js" in " ".join(info["render"]) and info["render"][-2] == "-d"
    assert not Path(info["wav"]).exists()                                            # nothing was rendered


def test_per_note_reference_is_used_where_available():
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    table = {"snd": {"log2": 0.5, "per_note": {"57": 0.0, "69": 1.0}, "source": "chromatic sweep probe"}}
    assert PR.reference_for("snd", {57: 1.0}, table) == pytest.approx(0.0)
    assert PR.reference_for("snd", {69: 3.0, 57: 1.0}, table) == pytest.approx(0.75)
    assert PR.reference_for("snd", {58: 1.0}, table) == pytest.approx(0.0)            # nearest probed note
    assert PR.reference_for("snd", None, table) == pytest.approx(0.5)                 # scalar fallback
    assert PR.reference_for("other", {57: 1.0}, table) is None


def test_empty_reference_warns_and_applies_no_shift(capsys):
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    assert PR.octave_shift(1.5, "gm_voice_oohs", {}, {60: 1.0}) == 0
    assert "no sweep reference for 'gm_voice_oohs'" in capsys.readouterr().err


@pytest.mark.parametrize("content", [None, "{not json", '{"nosounds": 1}'])
def test_load_table_warns_naming_the_file(tmp_path, capsys, content):
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    p = tmp_path / "missing_or_bad.json"
    if content is not None:
        p.write_text(content)
    assert PR.load_table(p) == {}
    err = capsys.readouterr().err
    assert "WARNING partial reference" in err and str(p) in err


def test_register_cache_hit_and_miss(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    stem, midi, cache = tmp_path / "s.wav", tmp_path / "m.mid", tmp_path / "register.json"
    stem.write_bytes(b"a"); midi.write_bytes(b"b")
    calls = []
    monkeypatch.setattr(PR, "dominant_partial_log2",
                        lambda *a, **k: calls.append(1) or (0.7, {60: 2.0}))
    assert PR.cached_register(cache, "melodic", stem, midi, 48, 79) == (0.7, {60: 2.0}, False)
    assert PR.cached_register(cache, "melodic", stem, midi, 48, 79) == (0.7, {60: 2.0}, True)
    assert len(calls) == 1
    PR.cached_register(cache, "melodic", stem, midi, 48, 91)                 # other fold window -> miss
    stem.write_bytes(b"changed!")                                            # stem changed -> miss
    PR.cached_register(cache, "melodic", stem, midi, 48, 91)
    assert len(calls) == 3


@pytest.mark.parametrize("bad", ["3", "-1", "x", "1.5"])
def test_octave_flag_rejects_out_of_range_with_argparse_error(bad):
    r = subprocess.run([sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"), "--stems-dir", "x",
                        "--pack-dir", "x", "--bass-midi", "x", "--lead-midi", "x", "--base-url", "x",
                        "--lead-octave", bad], capture_output=True, text=True)
    assert r.returncode == 2 and "--lead-octave" in r.stderr and "auto" in r.stderr
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    assert PR.octave_arg("auto") == "auto" and PR.octave_arg("2") == 2


def _hat_case(tmp_path: Path, rend_centroid_ratio: float):
    d = json.loads(RUN2_CAL1.read_text())
    d["rendered"]["bands"]["high"] = d["original"]["bands"]["high"] * 3.0            # full-mix `high` 3x the original
    d["rendered"]["spectral"]["centroid_mean"] = d["original"]["spectral"]["centroid_mean"] * rend_centroid_ratio
    p = tmp_path / "comparison.json"; p.write_text(json.dumps(d))
    return p


def test_hat_full_mix_proxy_caps_only_when_lead_lpf_stood_still(tmp_path):
    still = run_cal(_hat_case(tmp_path, 1.0), *RUN2_KNOBS)                           # centroid parity -> lead-lpf unchanged
    assert int(still["--lead-lpf"]) == 5923 and float(still["--hat-gain"]) < 0.135
    moved = run_cal(_hat_case(tmp_path, 0.8), *RUN2_KNOBS)                           # dull -> lead-lpf opens this step
    assert int(moved["--lead-lpf"]) > 5923 and float(moved["--hat-gain"]) == pytest.approx(0.135)


def test_hat_drums_stem_branch_acts_even_when_lead_lpf_moves(tmp_path):
    p = _hat_case(tmp_path, 0.8)
    (tmp_path / "stem_comparison.json").write_text((FIX / "vagabundo_synth_v006_stem_drums.json").read_text())
    got = run_cal(p, *RUN2_KNOBS)
    assert int(got["--lead-lpf"]) > 5923 and float(got["--hat-gain"]) < 0.135


GATE_STUB = ("import os,sys\nd=os.path.dirname(__file__);f=os.path.join(d,'g');n=int(open(f).read()) if os.path.exists(f) else 0\n"
             "open(f,'w').write(str(n+1))\nbad=%r\nif n in bad: sys.exit(2)\n"
             "print('0.0100 PASS 0.8000 0.8200 0.7700 0.8100')\n")


def _run_mixed_gate(tmp_path, rows, bad):
    global STUB_CMP
    old, STUB_CMP = STUB_CMP, _seq_cmp(tmp_path, rows)
    gate = tmp_path / "gate.py"; gate.write_text(GATE_STUB % (bad,))
    try:
        return run_driver(tmp_path, AUTO, "--mode", "synth", "--iters", "2",
                          extra_env={"AC_GATE": str(gate), "AC_PY": sys.executable})
    finally:
        STUB_CMP = old


def test_mixed_gate_readability_falls_back_to_overall_for_all(tmp_path):
    # cal1 gate readable (margin +0.01), cal2 gate exits 2 but has the higher overall -> overall ranks BOTH
    r = _run_mixed_gate(tmp_path, [(0.80, 0.82), (0.83, 0.78)], [1])
    best = [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]
    assert best.startswith("BEST: cal2 "), best
    assert "BEST selected by overall similarity across ALL iterations" in r.stdout
    assert "unreadable for >=1 iteration" in r.stdout


def test_mixed_gate_other_order(tmp_path):
    r = _run_mixed_gate(tmp_path, [(0.83, 0.78), (0.80, 0.82)], [0])
    best = [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]
    assert best.startswith("BEST: cal1 "), best          # first iteration unreadable, higher overall wins for ALL


# --- bounded octave rule: shift <= floor(original), reference evaluated at the SHIFTED notes ---------
@pytest.mark.parametrize("orig,ref,expect", [(1.25, -0.34, 1), (2.3, 0.0, 2), (0.9, 0.0, 0), (0.9, -0.5, 0)])
def test_octave_shift_is_bounded_by_the_measurement(orig, ref, expect):
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    dec = PR.octave_decision(orig, "snd", {"snd": ref}, {60: 1.0})
    assert dec["shift"] == expect and dec["bound"] == max(0, int(orig // 1))
    assert [c["shift"] for c in dec["candidates"]] == list(range(dec["bound"] + 1))
    assert all(c["shift"] <= orig for c in dec["candidates"])                  # fundamental never above the original's partial


def test_oohs_like_case_two_octaves_is_rejected_one_chosen():
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    dec = PR.octave_decision(1.253, "gm_voice_oohs", {"gm_voice_oohs": -0.34}, {60: 1.0, 68: 1.0})
    assert dec["shift"] == 1                                                    # naive round(1.253 + 0.34) would be 2
    assert {c["shift"] for c in dec["candidates"]} == {0, 1}


def test_reference_is_evaluated_at_the_shifted_notes():
    sys.path.insert(0, str(SCRIPTS))
    import partial_register as PR
    # low notes (<=48) have a +0.75 partial, high notes -0.3 (oohs-like). The line plays MIDI 40:
    # at k=0 the reference is +0.75 (error |1.2-0.75|=0.45); at k=1 the notes are 52 -> -0.3, predicted 0.7 (error 0.5)
    table = {"snd": {"log2": 0.0, "per_note": {"40": 0.75, "52": -0.3, "64": -0.3}, "source": "chromatic sweep probe"}}
    dec = PR.octave_decision(1.2, "snd", table, {40: 1.0})
    by = {c["shift"]: c for c in dec["candidates"]}
    assert by[0]["ref_log2"] == pytest.approx(0.75) and by[1]["ref_log2"] == pytest.approx(-0.3)
    assert dec["shift"] == 0
