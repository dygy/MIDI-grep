# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""Bass shape (--bass-shape stem|off) is a calibrator-scored choice, remembered in ClickHouse.

Defect: `stem` was a fixed generator default - it won on three of four pieces and lost on Regime sample-instrument
(v053 0.807/0.794 vs v051 0.815/0.803 without it). A per-track constant is forbidden, so auto-calibrate.sh renders the
BEST knobs once with the other shape, keeps the better one (a tie within calibration noise keeps the default), records
the answer per code version and restores it on the next warm start without re-testing. Stubbed driver, no render; the
stub's score depends only on the --bass-shape the piece was generated with.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_helpers as PH  # noqa: E402
import test_calibrate_loop as TCL  # noqa: E402  (calibrator / gate stubs)

MEM = SCRIPTS / "calibration_memory.py"

STUB_GEN = r'''
import json, os, sys
a = sys.argv[1:]
shape = a[a.index("--bass-shape") + 1] if "--bass-shape" in a else "NONE"
out = a[a.index("--out") + 1]
open(out, "w").write("shape=" + shape + "\n")
open(os.environ["AC_LOG"], "a").write(shape + "\n")
print(json.dumps({"out": out, "bass_lpf": 603, "bass_lpf_analysis": 603, "bass_note_max_hz": 131.0,
                  "lead_hpf": 300, "lead_note_min_hz": 130.8}))
'''
STUB_REC = '#!/bin/bash\ns="$1"; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && cp "$s" "$2"; shift; done\n'   # the "wav" is the strudel stub
STUB_CMP = r'''
import json, os, sys
shape = open(sys.argv[2]).read().strip().split("=")[1]
o = float(os.environ["SCORE_" + shape.upper()])
print(json.dumps({"comparison": {"overall_similarity": o, "section_aware_similarity": o}}))
'''


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import clickhouse_store as CS
    path = tmp_path / "chdb"
    monkeypatch.setenv("MIDIGREP_CLICKHOUSE_DB", str(path))
    monkeypatch.setattr(CS, "CLICKHOUSE_DB", path)
    monkeypatch.setattr(CS, "_clickhouse_initialized", False)
    monkeypatch.setenv("AC_NOISE", "0.005")
    import calibration_memory as M
    return M


def drive(tmp_path, stem, off, *args, run="a"):
    """One auto-calibrate.sh run against tmp_path/chdb (shared between runs); returns (result, shapes generated, work dir)."""
    wd = tmp_path / f"work_{run}"
    for n, body in {"gen.py": STUB_GEN, "cmp.py": STUB_CMP, "cal.py": TCL.STUB_CAL, "gate.py": TCL.STUB_GATE}.items():
        (tmp_path / n).write_text(body)
    (tmp_path / "rec.sh").write_text(STUB_REC)
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    (stems / "instrumental.wav").write_text("x")
    log = tmp_path / f"gen_{run}.log"
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp_path / "gen.py"), "AC_CMP": str(tmp_path / "cmp.py"),
           "AC_CAL": str(tmp_path / "cal.py"), "AC_GATE": str(tmp_path / "gate.py"), "AC_REC": str(tmp_path / "rec.sh"),
           "AC_NODE": "/bin/bash", "AC_SETTLE_S": "0", "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0",
           "AC_REF": str(stems / "instrumental.wav"), "AC_LOG": str(log), "AC_NOISE": "0.005", "AC_MEM": str(MEM),
           "MIDIGREP_CLICKHOUSE_DB": str(tmp_path / "chdb"), "SCORE_STEM": str(stem), "SCORE_OFF": str(off)}
    r = subprocess.run(["/bin/bash", "-u", str(AUTO), "--stems", str(stems), "--base", "http://x/", "--mode", "synth",
                        "--bpm", "130", "--key", "C# minor", "--bars", "81", "--workdir", str(wd), "--no-mix-dynamics", *args],
                       capture_output=True, text=True, env=env, timeout=300)
    return r, (log.read_text().split() if log.exists() else []), wd


def best_line(r):
    return [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]


def test_off_scoring_higher_is_chosen_recorded_and_used_by_best(db, tmp_path):
    r, shapes, wd = drive(tmp_path, 0.80, 0.84, "--iters", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert shapes[-1] == "off" and set(shapes[:-1]) == {"stem"}, shapes          # the loop ran stem, ONE extra render ran off
    assert best_line(r).endswith(" off"), best_line(r)                           # 10th token of the BEST params line
    assert (wd / "best.strudel").read_text() == "shape=off\n"
    rows = db.history("stems", "synth")
    shape = [x for x in rows if x["kind"] == "shape"]
    best = [x for x in rows if x["kind"] == "best"]
    assert len(shape) == 1 and shape[0]["bass_shape"] == "off" and shape[0]["overall"] == pytest.approx(0.84)
    assert len(best) == 1 and best[0]["bass_shape"] == "off" and best[0]["tag"] == "shape_off"
    assert {x["bass_shape"] for x in rows if x["kind"] == "step"} == {"stem"}


def test_warm_start_restores_the_choice_and_does_not_retest(db, tmp_path):
    drive(tmp_path, 0.80, 0.84, "--iters", "3", run="a")
    r, shapes, wd = drive(tmp_path, 0.80, 0.84, run="b")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warm start: track" in r.stdout and "bass shape: off (tested for this code version: 1" in r.stdout
    assert shapes and set(shapes) == {"off"}, shapes                              # cal1.. all off, and no extra trial render
    assert "bass-shape trial" not in r.stdout and "[shape_" not in r.stdout
    assert best_line(r).endswith(" off")


def test_default_kept_and_not_retested_when_the_other_shape_loses(db, tmp_path):
    r, shapes, _ = drive(tmp_path, 0.80, 0.70, "--iters", "3", run="a")
    assert shapes[-1] == "off" and best_line(r).endswith(" stem"), (shapes, r.stdout)
    assert [x["kind"] for x in db.history("stems", "synth") if x["bass_shape"] == "off"] == ["shape"]
    r2, shapes2, _ = drive(tmp_path, 0.80, 0.70, run="b")
    assert set(shapes2) == {"stem"} and "bass-shape trial" not in r2.stdout, shapes2   # recorded for this code version: no extra render
    assert best_line(r2).endswith(" stem")


def test_tie_within_calibration_noise_keeps_the_default(db, tmp_path):
    r, shapes, _ = drive(tmp_path, 0.800, 0.803, "--iters", "3")                  # +0.003 < noise 0.005
    assert shapes[-1] == "off" and best_line(r).endswith(" stem"), r.stdout
    assert "does not beat stem beyond noise" in r.stdout


def test_trial_when_the_run_started_from_off_defaults_back_to_stem_on_a_tie(db, tmp_path):
    drive(tmp_path, 0.80, 0.84, "--iters", "3", run="a")                          # off recorded
    import clickhouse_store as CS
    CS.clickhouse_query(f"ALTER TABLE midi_grep.calibration_steps UPDATE code_version = 'older' WHERE 1 SETTINGS mutations_sync = 2")
    r, shapes, _ = drive(tmp_path, 0.800, 0.801, run="b")                         # code changed: retest, the alternative is stem
    assert "bass shape: off (tested for this code version: 0" in r.stdout
    assert shapes[-1] == "stem" and best_line(r).endswith(" stem"), (shapes, r.stdout)


def test_pinned_shape_is_not_tested(db, tmp_path):
    r, shapes, _ = drive(tmp_path, 0.80, 0.90, "--iters", "3", "--bass-shape", "off")
    assert set(shapes) == {"off"} and best_line(r).endswith(" off")
    assert not [x for x in db.history("stems", "synth") if x["kind"] == "shape"]


def test_other_code_version_means_untested(db):
    def mk(kind, cv, shape, tag, ov):
        ctx = {"title": "t", "genre": "g", "bpm": 130.0, "key": "C# minor", "bars": 81, "code_version": cv}
        cj = {"comparison": {"overall_similarity": ov, "section_aware_similarity": ov}}
        return db.make_row(kind=kind, run_id="r", tag=tag, track_key="t", mode="synth", knobs=db.parse_knobs("0.5 0.6 0.9 5000 0 0.78 0 0 0"),
                           ctx=ctx, cmp_json=cj, margin=ov - 0.5, passed=True, bass_shape=shape, code=(cv, "abc"))
    db.write_rows([mk("step", "v1", "stem", "cal1", 0.80), mk("shape", "v1", "off", "shape_off", 0.90)])
    same = db.warm_start("t", "synth", "g", 130, "C# minor", 81, current_code="v1")
    assert same["shape_tested"] is True and same["bass_shape"] == "stem"          # a losing trial row never seeds a start
    new = db.warm_start("t", "synth", "g", 130, "C# minor", 81, current_code="v2")
    assert new["shape_tested"] is False and new["bass_shape"] == "stem"
    db.write_rows([mk("best", "v1", "off", "shape_off", 0.90)])
    assert db.warm_start("t", "synth", "g", 130, "C# minor", 81, current_code="v1")["bass_shape"] == "off"


@pytest.mark.parametrize("text,ok", [
    ("0.4 0.7 1 5000 0 0.6 0 0 0 stem", True), ("0.4 0.7 1 5000 0 0.6 300 450 1 off", True), ("0.4 0.7 1 5000 0 0.6 0 0 0", True),
    ("0.4 0.7 1 5000 0 0.6 0 0 0 maybe", False), ("0.4 0.7 1 5000 0 0.6 0 0 stem", False), ("0.4 0.7 1 5000 0 0.6 0 0 1 1", False)])
def test_params_valid_takes_the_bass_shape_token(text, ok):
    assert PH.params_valid(text) is ok


def test_promoted_note_with_a_shape_token_is_still_recorded(db, tmp_path):
    import json
    td = tmp_path / "track"; v = td / "v001"; v.mkdir(parents=True)
    (v / "metadata.json").write_text(json.dumps({"generation_mode": "synth", "generator": "g.py (knobs from calibrate_dynamic.py: 0.4 0.7 1 5000 0 0.6 0 0 0 off; final --master-gain 0.6; genre g; bars 81)",
                                                 "genre": "g", "bpm": 130, "key": "C# minor"}))
    (v / "comparison.json").write_text(json.dumps({"comparison": {"overall_similarity": 0.8, "section_aware_similarity": 0.8}}))
    row = db.promoted_row(td, "v001")
    assert row is not None and row["bass_shape"] == "off" and row["bass_mult"] == pytest.approx(0.4)
