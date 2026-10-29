# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""ClickHouse as the calibration memory: every step recorded, warm start from the recorded best (own track, else the
nearest same-genre track), iteration budget from history, learned damping exponents.

Defect: scripts/auto-calibrate.sh never touched ClickHouse; with no best.params it started cold and spent 5-8 renders.
Every database here is a temp directory (MIDIGREP_CLICKHOUSE_DB); the stubbed driver never renders.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_loop as CL  # noqa: E402
import test_calibrate_loop as TCL  # noqa: E402  (stub generator / comparer / calibrator / recorder)

MEM = SCRIPTS / "calibration_memory.py"
GENRE = "brazilian_funk"
BANDS = {"sub_bass": 0.34, "bass": 0.40, "low_mid": 0.06, "mid": 0.15, "high_mid": 0.03, "high": 0.02}


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


def row(M, *, track="other", mode="synth", tag="cal1", overall=0.80, margin=0.10, knobs="0.5 0.6 0.9 5000 0 0.78 400 450 1",
        bpm=130.0, key="C# minor", genre=GENRE, run="r1", kind="step", parent="", bars=81, bands=None, code="old", rb=None):
    ctx = {"title": track, "genre": genre, "bpm": bpm, "key": key, "bars": bars, "code_version": code}
    cmp_json = {"comparison": {"overall_similarity": overall, "section_aware_similarity": overall - 0.03},
                "original": {"bands": bands or BANDS}, "rendered": {"bands": rb or bands or BANDS}}
    return M.make_row(kind=kind, run_id=run, tag=tag, track_key=track, mode=mode, knobs=M.parse_knobs(knobs), ctx=ctx,
                      cmp_json=cmp_json, margin=margin, passed=margin >= 0, parent=parent,
                      env={"bars": bars, "voices": {"bass": [1.0] * bars}} if kind != "step" else None, code=(code, "abc"))


def run_driver(tmp_path, scores, *args, mem_db: Path | None = None, extra_env=None):
    for n, body in {"gen.py": TCL.STUB_GEN, "cmp.py": TCL.STUB_CMP, "cal.py": TCL.STUB_CAL, "gate.py": TCL.STUB_GATE}.items():
        (tmp_path / n).write_text(body)
    (tmp_path / "rec.sh").write_text(TCL.STUB_REC)
    (tmp_path / "scores.txt").write_text("\n".join(f"{o} {s}" for o, s in scores))
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    (stems / "instrumental.wav").write_text("x")
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp_path / "gen.py"), "AC_CMP": str(tmp_path / "cmp.py"),
           "AC_CAL": str(tmp_path / "cal.py"), "AC_GATE": str(tmp_path / "gate.py"), "AC_REC": str(tmp_path / "rec.sh"),
           "AC_NODE": "/bin/bash", "AC_SETTLE_S": "0", "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0",
           "AC_REF": str(stems / "instrumental.wav"), "AC_SCORES": str(tmp_path / "scores.txt"), "AC_LOG": str(tmp_path / "gen.log"),
           "AC_NOISE": "0.005", "AC_MEM": str(MEM), "MIDIGREP_CLICKHOUSE_DB": str(mem_db or tmp_path / "chdb"),
           "AC_SHAPE_TRIAL": "0"}   # these tests count generator calls; the bass-shape trial has its own tests (test_bass_shape_trial.py)
    env.update(extra_env or {})
    r = subprocess.run(["/bin/bash", "-u", str(AUTO), "--stems", str(stems), "--base", "http://x/", "--mode", "synth",
                        "--bpm", "130", "--key", "C# minor", "--bars", "81", "--workdir", str(tmp_path / "work"), *args],
                       capture_output=True, text=True, env=env, timeout=300)
    log = tmp_path / "gen.log"
    return r, [float(x.split("=")[1]) for x in log.read_text().split()] if log.exists() else []


def improving(n=12):
    return [(0.60 + 0.02 * i, 0.60 + 0.02 * i) for i in range(n)]


def test_a_migration_number_taken_by_another_branch_does_not_skip_this_migration(db):
    # 2026-10-27: the shared database had recorded version 5 for an unmerged experiment ("voice source column"), so the
    # write-order migration 5 never ran and every insert failed with "No such column seq". A migration is applied
    # unless ITS (version, description) is recorded.
    import clickhouse_store as CS
    CS.CLICKHOUSE_DB.mkdir(parents=True, exist_ok=True)
    for q in ("CREATE DATABASE IF NOT EXISTS midi_grep",
              CS.MIGRATIONS[0][2][1],                          # the schema_version table
              "INSERT INTO midi_grep.schema_version (version, description) VALUES (5, 'Calibration memory voice source column')"):
        assert CS._run_clickhouse_query(q).returncode == 0, q
    assert db.write_rows([row(db, tag="cal1")]) == 1
    assert [x["seq"] for x in db.history("other", "synth")] == [0]


def test_steps_recorded_in_separate_writes_keep_their_order_within_one_second(db):
    # auto-calibrate.sh records each step in its OWN write: seq (the position inside a write) is 0 for every one of them,
    # so steps written in the same second tied and came back in no defined order (CI: ['cal2', 'cal1', 'cal3']).
    # The write clock must increase across writes, and history must follow it even when the rows sit physically reversed.
    import clickhouse_store as CS
    t = "2026-10-29 05:00:00"
    for tag in ("cal1", "cal2", "cal3"):
        assert db.write_rows([{**row(db, tag=tag, run="rw"), "created_at": t}]) == 1
    rows = [x for x in db.history("other", "synth") if x["run_id"] == "rw"]
    assert [x["tag"] for x in rows] == ["cal1", "cal2", "cal3"]
    keys = [int(x["write_ns"]) for x in rows]
    assert keys == sorted(keys) and len(set(keys)) == 3, keys
    rev = [{**row(db, tag=tg, run="rv"), "created_at": t, "seq": 0, "write_ns": w} for tg, w in (("cal3", 30), ("cal1", 10), ("cal2", 20))]
    CS.clickhouse_insert_many(db.TABLE, rev)
    assert [x["tag"] for x in db.history("other", "synth") if x["run_id"] == "rv"] == ["cal1", "cal2", "cal3"]


def test_rows_of_one_write_come_back_in_write_order_although_they_tie_on_created_at(db):
    # created_at has one-second resolution: the steps of one run, recorded together, tie on it and ClickHouse returns ties
    # in no defined order (CI, ClickHouse 26.11: ['cal2', 'cal1', 'cal3']). The write order (`seq`) breaks the tie, so a
    # physically shuffled insert of one write still reads back cal1, cal2, cal3.
    import clickhouse_store as CS
    assert db.write_rows([row(db, tag="seed")]) == 1             # creates the table (migrations)
    t = "2026-10-27 05:00:00"
    rs = [{**row(db, tag=tg, run="r2"), "created_at": t, "seq": s} for tg, s in (("cal3", 2), ("cal1", 0), ("cal2", 1))]
    CS.clickhouse_insert_many(db.TABLE, rs)
    assert [x["tag"] for x in db.history("other", "synth") if x["run_id"] == "r2"] == ["cal1", "cal2", "cal3"]
    assert [x["seq"] for x in db.history("other", "synth") if x["run_id"] == "r1"] == [0]   # write_rows numbers its rows


def test_every_step_and_the_run_best_are_recorded(db, tmp_path):
    r, _ = run_driver(tmp_path, improving(), "--iters", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = db.history("stems", "synth")
    assert [x["tag"] for x in rows if x["kind"] == "step"] == ["cal1", "cal2", "cal3"]
    best = [x for x in rows if x["kind"] == "best"]
    assert len(best) == 1 and best[0]["tag"] == "cal3" and best[0]["is_run_best"] == 1
    s1 = rows[0]
    assert s1["genre"] == GENRE and s1["bpm"] == 130 and s1["key"] == "C# minor" and s1["mode"] == "synth" and s1["bars"] == 81
    assert s1["overall"] == pytest.approx(0.60) and s1["margin"] == pytest.approx(0.10) and s1["code_version"]
    assert s1["bass_mult"] == pytest.approx(0.4)                      # the knobs the step ran with
    assert [x["parent_tag"] for x in rows if x["kind"] == "step"] == ["", "cal1", "cal2"]


def test_recording_twice_is_a_noop(db):
    rs = [row(db, tag="cal1"), row(db, tag="cal2", overall=0.82)]
    assert db.write_rows(rs) == 2 and db.write_rows(rs) == 0
    assert len(db.history("other", "synth")) == 2


def test_warm_start_uses_the_recorded_best_of_this_track_and_caps_at_three(db, tmp_path):
    db.write_rows([row(db, track="stems", tag="cal1", overall=0.70, margin=0.0, knobs="0.40 0.7 1.0 5000 0 0.6 0 0 0"),
                   row(db, track="stems", tag="cal2", overall=0.80, margin=0.10, knobs="0.77 0.66 0.88 4100 0.1 0.7 300 410 1"),
                   row(db, track="stems", tag="cal3", overall=0.75, margin=0.05, knobs="0.50 0.50 0.50 4000 0 0.6 0 0 0")])
    r, bms = run_driver(tmp_path, improving())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warm start: track" in r.stdout and "iteration cap: 3" in r.stdout
    assert bms[0] == pytest.approx(0.77), bms          # cal1 renders the recorded best, not the cold default 0.40
    assert len(bms) == 3, bms                           # still improving, stopped by the warm budget


def test_cold_start_without_history_runs_the_full_budget(db, tmp_path):
    r, bms = run_driver(tmp_path, improving())
    assert "start: cold" in r.stdout and bms[0] == pytest.approx(0.40)
    assert len(bms) == CL.MAX_ITERS


def test_new_track_is_seeded_from_the_nearest_same_genre_track(db, tmp_path):
    far = {"sub_bass": 0.10, "bass": 0.20, "low_mid": 0.2, "mid": 0.3, "high_mid": 0.1, "high": 0.1}
    db.write_rows([row(db, track="near", tag="cal4", knobs="0.61 0.62 0.63 4200 0 0.7 300 400 1", bpm=129.0, key="C# minor"),
                   row(db, track="far", tag="cal4", knobs="0.99 0.98 0.97 3900 0 0.7 0 0 0", bpm=95.0, key="F major", bands=far),
                   row(db, track="jazzy", tag="cal4", knobs="0.11 0.12 0.13 3800 0 0.7 0 0 0", genre="jazz", bpm=130.0)])
    r, bms = run_driver(tmp_path, improving())
    assert "warm start: neighbour" in r.stdout and "'near'" in r.stdout, r.stdout
    assert bms[0] == pytest.approx(0.61) and len(bms) == 3
    # note-bound knobs (bass low-pass, lead high-pass) and the octave hypothesis are not copied from another track
    w = db.warm_start("new", "synth", GENRE, 130.0, "C# minor", 81, orig_bands=BANDS, current_code="x")
    assert w["source"] == "neighbour" and w["knobs"].split()[6:] == ["0", "0", "0"]


def test_genre_without_history_is_cold(db):
    db.write_rows([row(db, track="jazzy", genre="jazz")])
    assert db.warm_start("new", "synth", GENRE, 130.0, "C# minor", 81, orig_bands=BANDS)["source"] == "cold"


def test_current_code_version_wins_else_newest_recorded(db):
    old = {**row(db, tag="a", code="v1", overall=0.90, margin=0.2, knobs="0.9 0.9 0.9 5000 0 0.7 0 0 0"), "created_at": "2026-10-20 10:00:00"}
    new = {**row(db, tag="b", code="v2", overall=0.80, margin=0.1, knobs="0.2 0.2 0.2 5000 0 0.7 0 0 0", run="r2"), "created_at": "2026-10-21 10:00:00"}
    db.write_rows([old, new])
    assert db.warm_start("other", "synth", GENRE, 130, "", 81, current_code="v1")["knobs"].startswith("0.9 ")
    assert db.warm_start("other", "synth", GENRE, 130, "", 81, current_code="v3")["knobs"].startswith("0.2 ")


def test_env_correction_comes_back_only_for_the_same_bar_count(db):
    db.write_rows([row(db, kind="best", tag="cal2", bars=81)])
    assert db.warm_start("other", "synth", GENRE, 130, "", 81)["env"]["bars"] == 81
    assert db.warm_start("other", "synth", GENRE, 130, "", 60)["env"] is None


def test_memory_failure_never_breaks_a_run(db, tmp_path):
    bad = tmp_path / "bad_mem.py"
    bad.write_text("import sys; print('boom', file=sys.stderr); sys.exit(1)\n")
    r, bms = run_driver(tmp_path, improving(), "--iters", "2", extra_env={"AC_MEM": str(bad)})
    assert r.returncode == 0 and len(bms) == 2
    assert "[memory] WARNING" in r.stderr and "cold start" in r.stderr.lower()


def test_distance_is_zero_for_identical_and_orders_by_closeness(db):
    a = {"bpm": 130, "key": "C# minor", "bands": BANDS}
    assert db.distance(a, a) == 0
    assert db.distance(a, {**a, "bpm": 131}) < db.distance(a, {**a, "bpm": 95}) < 1.0 + 1e-9
    assert db.distance(a, {**a, "key": "E major"}) == 0           # relative major
    assert db.distance(a, {**a, "key": "F# major"}) > 0


def test_learned_response_exponent_is_inverse_median_elasticity(db):
    # bass_mult up by f -> render `bass` share up so orig/rend ratio falls by f**e (e = 1.6): exponent 1/1.6 = 0.625
    rows = []
    for i in range(4):
        f = 1.2
        p_bm = 0.4 + 0.1 * i
        o = {**BANDS, "bass": 0.40}
        rp = {**BANDS, "bass": 0.30}
        rc = {**BANDS, "bass": 0.30 * f ** 1.6}
        for tag, bm, rb, par in (("a", p_bm, rp, ""), ("b", p_bm * f, rc, "a")):
            rows.append(row(db, tag=tag, run=f"r{i}", parent=par, knobs=f"{bm} 0.7 1.0 5000 0 0.7 0 0 0", bands=o, rb=rb))
    resp, n = db.learn_responses([{**r, "created": ""} for r in rows])
    assert n["bass_mult"] == 4 and resp["bass_mult"] == pytest.approx(0.625, abs=0.01)
    assert "sub_gain" not in resp or n["sub_gain"] >= db.MIN_PAIRS      # unchanged knob: no pairs, sqrt damping stays


def test_calibrator_uses_a_learned_exponent_and_defaults_to_sqrt(tmp_path):
    cmp_ = {"original": {"bands": {**BANDS, "sub_bass": 0.4}, "spectral": {"centroid_mean": 1500}},
            "rendered": {"bands": {**BANDS, "sub_bass": 0.2}, "spectral": {"centroid_mean": 1500}},
            "comparison": {"raw_rms_ratio": 1.0}}
    (tmp_path / "c.json").write_text(json.dumps(cmp_))
    (tmp_path / "r.json").write_text(json.dumps({"sub_gain": 0.25}))

    def sub_gain(*extra):
        out = subprocess.run([sys.executable, str(SCRIPTS / "calibrate_dynamic.py"), "--comparison", str(tmp_path / "c.json"),
                              "--sub-gain", "0.7", *extra], capture_output=True, text=True).stdout
        return float(out.split("--sub-gain")[1].split()[0])

    ratio = (0.4 + 1e-4) / (0.2 + 1e-4)
    assert sub_gain() == pytest.approx(round(0.7 * min(1.4, ratio ** 0.5), 3))
    assert sub_gain("--responses", str(tmp_path / "r.json")) == pytest.approx(round(0.7 * ratio ** 0.25, 3))


def test_warm_budget_rule():
    assert CL.iteration_cap("track") == 3 == CL.iteration_cap("neighbour")
    assert CL.iteration_cap("cold") == CL.iteration_cap("seed") == CL.MAX_ITERS


def test_backfill_imports_new_scale_workdirs_and_skips_old_ones(db, tmp_path):
    t = tmp_path / "cache" / "Track A"
    wd = t / ".pipeline" / "synth"
    wd.mkdir(parents=True)
    (t / "metadata.json").write_text(json.dumps({"title": "Track A", "genre": GENRE, "bpm": 130, "key": "C# minor"}))
    (wd / "iters.tsv").write_text(
        "cal1\t0.7\t0.01\t0.01 PASS 0.7\t0.4 0.7 1.0 5000 0 0.6 0 0 0\t-\t0.68\t-\t1\n"
        "cal2\t0.75\t0.05\t0.05 PASS 0.75\t0.5 0.7 1.0 5000 0 0.6 0 0 0\t-\t0.7\tcal1\t1\n")
    (wd / "cal1.cmp.json").write_text(json.dumps({"comparison": {"overall_similarity": 0.7}}))
    c = db.backfill(tmp_path / "cache")
    assert c["workdir_rows"] == 3 and c["new"] == 3   # 2 steps + the run BEST
    assert db.backfill(tmp_path / "cache")["new"] == 0
    assert db.warm_start("Track A", "synth", GENRE, 130, "", 81)["knobs"].startswith("0.5 ")
    assert db.backfill(tmp_path / "cache", since="2999-01-01 00:00:00")["workdir_rows"] == 0


def test_a_promoted_row_carries_the_final_master_gain_its_score_was_measured_at(db, tmp_path):
    t = tmp_path / "Track A"
    v = t / "v001"
    v.mkdir(parents=True)
    (t / "metadata.json").write_text(json.dumps({"title": "Track A", "genre": GENRE, "bpm": 130, "key": "C# minor"}))
    gen = "generate_dynamic_strudel.py --mode synth (knobs from calibrate_dynamic.py: 0.4 0.7 1 5000 0 0.6 0 0 0 stem; %s; genre x; bars 81)"
    (v / "comparison.json").write_text(json.dumps({"comparison": {"overall_similarity": 0.8}}))
    for note, want in (("final --master-gain 1.1", 1.1), ("final --master-gain = the 6th knob", 0.6)):
        (v / "metadata.json").write_text(json.dumps({"generation_mode": "synth", "generator": gen % note}))
        assert db.promoted_row(t, "v001")["master_gain"] == pytest.approx(want), note
