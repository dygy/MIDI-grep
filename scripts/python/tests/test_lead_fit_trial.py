# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""The fitted synth lead is a calibrator-scored per-piece choice, remembered in ClickHouse (like --bass-shape).

Defect: a lead fitted on the full song gained +0.016/+0.008 on Regime synth and lost -0.008/-0.009 on VAGABUNDO synth,
so it can be neither a fixed default nor a per-track constant. auto-calibrate.sh (synth mode) searches the lead's
lpf/lpenv/detune/gain on the real score, keeps the fit only if overall beats the palette lead by MORE than calibration
noise AND section-aware is no worse than palette - 0.005, records decision + vector (kind 'leadfit', lead_fit column,
migration 4) and warm-starts from it (own vector: one verify render; recorded 'palette': none).
Stubbed driver, no render; the stub's scores depend only on whether the piece was generated with --lead-synth.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_loop as CL  # noqa: E402
import test_calibrate_loop as TCL  # noqa: E402

MEM = SCRIPTS / "calibration_memory.py"

STUB_GEN = r'''
import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
lead = "fitted" if "--lead-synth" in a else "palette"
open(out, "w").write("lead=" + lead + "\n")
open(os.environ["AC_LOG"], "a").write(lead + "\n")
print(json.dumps({"out": out, "bass_lpf": 603, "bass_lpf_analysis": 603, "bass_note_max_hz": 131.0,
                  "lead_hpf": 300, "lead_note_min_hz": 130.8}))
'''
STUB_REC = '#!/bin/bash\ns="$1"; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && cp "$s" "$2"; shift; done\n'
STUB_CMP = r'''
import json, os, sys
lead = open(sys.argv[2]).read().strip().split("=")[1]
o = float(os.environ["OV_" + lead.upper()]); s = float(os.environ["SA_" + lead.upper()])
print(json.dumps({"comparison": {"overall_similarity": o, "section_aware_similarity": s}}))
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


def drive(tmp_path, pal, fit, *args, run="a", sa_pal=None, sa_fit=None):
    wd = tmp_path / f"work_{run}"
    for n, body in {"gen.py": STUB_GEN, "cmp.py": STUB_CMP, "cal.py": TCL.STUB_CAL, "gate.py": TCL.STUB_GATE}.items():
        (tmp_path / n).write_text(body)
    (tmp_path / "rec.sh").write_text(STUB_REC)
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    (stems / "instrumental.wav").write_text("x")
    t = np.arange(22050 * 4) / 22050
    sf.write(stems / "melodic.wav", 0.2 * np.sign(np.sin(2 * np.pi * 220 * t)) * np.exp(-(t % 0.5) / 0.1), 22050)
    log = tmp_path / f"gen_{run}.log"
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp_path / "gen.py"), "AC_CMP": str(tmp_path / "cmp.py"),
           "AC_CAL": str(tmp_path / "cal.py"), "AC_GATE": str(tmp_path / "gate.py"), "AC_REC": str(tmp_path / "rec.sh"),
           "AC_NODE": "/bin/bash", "AC_SETTLE_S": "0", "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0",
           "AC_REF": str(stems / "instrumental.wav"), "AC_LOG": str(log), "AC_NOISE": "0.005", "AC_MEM": str(MEM),
           "AC_SHAPE_TRIAL": "0", "MIDIGREP_CLICKHOUSE_DB": str(tmp_path / "chdb"),
           "OV_PALETTE": str(pal), "OV_FITTED": str(fit),
           "SA_PALETTE": str(pal if sa_pal is None else sa_pal), "SA_FITTED": str(fit if sa_fit is None else sa_fit)}
    r = subprocess.run(["/bin/bash", "-u", str(AUTO), "--stems", str(stems), "--base", "http://x/", "--mode", "synth",
                        "--bpm", "130", "--key", "C# minor", "--bars", "81", "--workdir", str(wd), "--no-mix-dynamics", *args],
                       capture_output=True, text=True, env=env, timeout=600)
    gens = log.read_text().split() if log.exists() else []
    return r, gens, wd


def best_line(r):
    return [ln for ln in r.stdout.splitlines() if ln.startswith("BEST:")][0]


# ------------------------------------------------------------------------------ the keep / fallback gate
def row(overall, sa, tag="t"):
    return CL.Row("\t".join([tag, str(overall), "n/a", "", "", "", str(sa), "", "1", "stem"]))


@pytest.mark.parametrize("fit_o,fit_sa,kept", [
    (0.8060, 0.800, True),     # +0.006 > noise 0.005, section-aware equal
    (0.8040, 0.800, False),    # +0.004: inside noise
    (0.8049, 0.800, False),    # just under noise
    (0.8300, 0.7949, False),   # section-aware 0.0051 below the palette's: too much
    (0.8300, 0.7951, True),    # 0.0049 below: tolerated
])
def test_gate_keeps_the_fit_only_beyond_noise_and_without_a_section_regression(fit_o, fit_sa, kept):
    pal, fit = row(0.800, 0.800, "palette"), row(fit_o, fit_sa, "fit")
    assert (CL.choose_lead_fit(pal, fit, 0.005) is fit) is kept


# ------------------------------------------------------------------------------ the driver + ClickHouse
def test_fit_that_wins_is_kept_baked_into_best_and_recorded(db, tmp_path):
    r, gens, wd = drive(tmp_path, 0.78, 0.80, "--iters", "2", "--lead-fit-evals", "6")
    assert r.returncode == 0, r.stdout + r.stderr
    assert gens.count("fitted") == 6 and "palette" in gens[:-6], gens                 # the budget cap is a flag
    assert "-> --lead-synth" in r.stdout and (wd / "best.leadsynth.json").is_file()
    assert (wd / "best.strudel").read_text() == "lead=fitted\n"
    assert "leadsynth:" in r.stdout
    rows = db.history("stems", "synth")
    trial = [x for x in rows if x["kind"] == "leadfit"]
    best = [x for x in rows if x["kind"] == "best"]
    assert len(trial) == 1 and json.loads(trial[0]["lead_fit"])["decision"] == "keep"
    spec = json.loads(trial[0]["lead_fit"])["spec"]
    assert spec["sound"] in ("supersaw", "sawtooth", "square", "triangle") and {"lpf", "lpenv", "detune", "gain"} <= set(spec)
    assert len(best) == 1 and json.loads(best[0]["lead_fit"])["decision"] == "keep" and best[0]["tag"].startswith("leadfit_")


@pytest.mark.parametrize("fit,sa_fit", [(0.783, 0.783), (0.83, 0.74)])
def test_fit_that_does_not_clear_the_gate_leaves_the_palette_lead_and_is_recorded_as_palette(db, tmp_path, fit, sa_fit):
    r, gens, wd = drive(tmp_path, 0.78, fit, "--iters", "2", "--lead-fit-evals", "4", sa_fit=sa_fit)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "palette lead kept" in r.stdout and not (wd / "best.leadsynth.json").exists()
    assert (wd / "best.strudel").read_text() == "lead=palette\n"
    trial = [x for x in db.history("stems", "synth") if x["kind"] == "leadfit"]
    assert len(trial) == 1 and json.loads(trial[0]["lead_fit"])["decision"] == "palette"


def test_warm_start_own_vector_is_one_verify_render_and_a_recorded_palette_answer_is_no_render(db, tmp_path):
    drive(tmp_path, 0.78, 0.80, "--iters", "2", "--lead-fit-evals", "5", run="a")
    r, gens, wd = drive(tmp_path, 0.78, 0.80, run="b")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "lead fit: start track-keep" in r.stdout and gens.count("fitted") == 1, (r.stdout, gens)
    assert (wd / "best.strudel").read_text() == "lead=fitted\n"
    # a second piece answered 'palette' for this code version: nothing is re-rendered on the next run
    import shutil
    shutil.rmtree(tmp_path / "chdb")
    drive(tmp_path, 0.78, 0.70, "--iters", "2", "--lead-fit-evals", "3", run="c")
    r2, gens2, _ = drive(tmp_path, 0.78, 0.70, run="d")
    assert "lead fit: start track-palette" in r2.stdout and "fitted" not in gens2, (r2.stdout, gens2)


def test_flag_and_env_skip_the_fit(db, tmp_path):
    r, gens, _ = drive(tmp_path, 0.78, 0.80, "--iters", "2", "--no-lead-fit")
    assert "fitted" not in gens and "lead fit: skipped" in r.stdout
    assert not [x for x in db.history("stems", "synth") if x["kind"] == "leadfit"]


def test_memory_round_trip_and_neighbour_warm_start(db):
    def mk(track, decision, cv="v1", bpm=130.0):
        ctx = {"title": track, "genre": "g", "bpm": bpm, "key": "C# minor", "bars": 81, "code_version": cv}
        cj = {"comparison": {"overall_similarity": 0.8, "section_aware_similarity": 0.8}}
        lf = json.dumps({"decision": decision, "spec": {"sound": "supersaw", "lpf": 2500.0, "gain": 1.2, "marker": track}})
        return db.make_row(kind="leadfit", run_id="r" + cv, tag="leadfit_3", track_key=track, mode="synth",
                           knobs=db.parse_knobs("0.5 0.6 0.9 5000 0 0.78 0 0 0"), ctx=ctx, cmp_json=cj, margin=0.3, passed=True,
                           code=(cv, "abc"), lead_fit=lf)
    db.write_rows([mk("near", "keep", bpm=131.0), mk("far", "keep", bpm=95.0), mk("lost", "palette", bpm=130.0)])
    w = db.lead_warm("me", "synth", "g", 130.0, "C# minor", current_code="v1", orig_bands={})
    assert w["source"] == "neighbour" and w["spec"]["marker"] == "near"             # a 'palette' neighbour never lends a vector
    db.write_rows([mk("me", "keep", cv="v0")])
    assert db.lead_warm("me", "synth", "g", 130.0, "C# minor", current_code="v1", orig_bands={})["source"] == "track"
    db.write_rows([mk("me", "keep", cv="v1")])
    w = db.lead_warm("me", "synth", "g", 130.0, "C# minor", current_code="v1", orig_bands={})
    assert w["source"] == "track-keep" and w["spec"]["marker"] == "me"
    # kind 'leadfit' rows never seed a knob start
    assert db.warm_start("me", "synth", "g", 130.0, "C# minor", 81, current_code="v1")["source"] == "track"


def test_ask_tell_replays_deterministically_and_stops_at_the_cap(tmp_path):
    import lead_synth as LS
    wav = tmp_path / "m.wav"
    t = np.arange(22050 * 4) / 22050
    sf.write(wav, 0.2 * np.sign(np.sin(2 * np.pi * 220 * t)) * np.exp(-(t % 0.5) / 0.1), 22050)
    hist: dict = {}
    seen = []
    for k in range(7):
        st, spec = LS.ask(hist, melodic=wav, bpm=120.0, cap=6, free=list(LS.FREE_DEFAULT), warm=None)
        if st == "DONE":
            break
        seen.append(spec)
        LS.tell(hist, 0.5 + 0.01 * k)
    assert len(seen) == 6 and st == "DONE" and spec == seen[-1]                  # best = the highest score
    hist2 = json.loads(json.dumps(hist))
    st2, spec2 = LS.ask(hist2, melodic=wav, bpm=120.0, cap=6, free=list(LS.FREE_DEFAULT), warm=None)
    assert (st2, spec2) == ("DONE", spec)
    warm = dict(seen[2])
    h3: dict = {}
    _, first = LS.ask(h3, melodic=wav, bpm=120.0, cap=1, free=list(LS.FREE_DEFAULT), warm=warm)
    assert first["lpf"] == pytest.approx(warm["lpf"], rel=1e-3) and first["gain"] == pytest.approx(warm["gain"], rel=1e-3)   # verify = the stored vector
