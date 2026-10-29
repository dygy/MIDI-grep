# @layer: integration
# @spec: 004-second-reference-track
# @regression
"""The renders of one calibration iteration (mix, bass solo, melodic solo) run concurrently.

Defect: each iteration rendered the mix (~31 s), the bass voice alone (~29 s) and the melodic voices alone (~16 s) one
after the other although all three strudels come from the same generated piece - ~76 s where ~31 s is enough. Stubbed
driver runs: every stub render sleeps T and stamps its start/end, so overlap (or the lack of it) is measured, not assumed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

import render_jobs as RJ  # noqa: E402
import test_calibrate_loop as TCL  # noqa: E402  (stub generator / comparer / calibrator / gate)

T = 1.2                                    # seconds every stub render sleeps
MARK = "4243"                              # `sleep 4243` is the stub renderer's grandchild: nothing may outlive the driver

# A renderer stub: stamps start/end per strudel, spawns a grandchild (like node -> Chromium), sleeps, writes the wav.
STUB_RENDER = r'''
import os, subprocess, sys, time
src, out = sys.argv[1], sys.argv[sys.argv.index("-o") + 1]
name = os.path.basename(src)
kind = "bass" if ".bass." in name else "mel" if ".mel." in name else "mix"
stamps = os.environ["STUB_STAMPS"]
def stamp(what):
    with open(stamps, "a") as f:
        f.write(f"{kind}\t{what}\t{time.time():.3f}\n")
stamp("start")
child = subprocess.Popen(["sleep", os.environ.get("STUB_MARK", "4243")])
if os.environ.get("STUB_FAIL") == kind:
    time.sleep(float(os.environ.get("STUB_FAIL_AFTER", "0.2")))
    stamp("fail")
    child.terminate()          # a failing renderer cleans up its own browser; the driver only owns the OTHER jobs
    sys.exit(3)
time.sleep(float(os.environ["STUB_T_" + kind.upper()]))
child.terminate()
open(out, "w").write("wav")
stamp("end")
'''
STUB_CMP = r'''
import json, os, time
with open(os.environ["STUB_STAMPS"], "a") as f:
    f.write(f"cmp\tstart\t{time.time():.3f}\n")
time.sleep(float(os.environ.get("STUB_T_CMP", "0")))
with open(os.environ["STUB_STAMPS"], "a") as f:
    f.write(f"cmp\tend\t{time.time():.3f}\n")
print(json.dumps({"comparison": {"overall_similarity": 0.8, "section_aware_similarity": 0.8}}))
'''
STUB_SOLO = r'''
import json, sys
a = sys.argv[1:]
if a[0] == "solo":
    open(a[2], "w").write("// solo\n")
    sys.exit(0)
json.dump({"stub": a[0]}, open(a[a.index("--out") + 1], "w"))
print("solo stub: measured", file=sys.stderr)
'''


def drive(tmp: Path, *, jobs: str | None, t: float = T, extra_env: dict[str, str] | None = None, args: tuple[str, ...] = ("--iters", "1"),
          popen: bool = False):
    tmp.mkdir(parents=True, exist_ok=True)
    for n, body in {"gen.py": TCL.STUB_GEN, "cal.py": TCL.STUB_CAL, "gate.py": TCL.STUB_GATE, "cmp.py": STUB_CMP,
                    "rec.py": STUB_RENDER, "solo.py": STUB_SOLO}.items():
        (tmp / n).write_text(body)
    stems = tmp / "stems"
    (stems / "sample_pack").mkdir(parents=True, exist_ok=True)
    for f in ("instrumental.wav", "bass.wav", "melodic.wav"):
        (stems / f).write_text("x")
    env = {"PATH": "/usr/bin:/bin", "AC_PY": sys.executable, "AC_GEN": str(tmp / "gen.py"), "AC_CMP": str(tmp / "cmp.py"),
           "AC_CAL": str(tmp / "cal.py"), "AC_GATE": str(tmp / "gate.py"), "AC_REC": str(tmp / "rec.py"),
           "AC_NODE": sys.executable, "AC_BSOLO": str(tmp / "solo.py"), "AC_MSOLO": str(tmp / "solo.py"),
           "AC_SETTLE_S": "0", "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0", "AC_REF": str(stems / "instrumental.wav"),
           "AC_LOG": str(tmp / "gen.log"), "AC_NOISE": "0.005", "STUB_STAMPS": str(tmp / "stamps.tsv"),
           "STUB_T_MIX": str(t), "STUB_T_BASS": str(t), "STUB_T_MEL": str(t), "AC_SCORES": str(tmp / "scores.txt")}
    (tmp / "scores.txt").write_text("0.8 0.8")
    if jobs is not None:
        env["MIDIGREP_RENDER_JOBS"] = jobs
    env.update(extra_env or {})
    cmd = ["/bin/bash", "-u", str(AUTO), "--stems", str(stems), "--base", "http://x/", "--mode", "sample-instrument",
           "--start", "0.4 0.7 1.0 5000 0.1 0.6", "--workdir", str(tmp / "work"), "--no-mix-dynamics", *args]
    if popen:
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


def stamps(tmp: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    f = tmp / "stamps.tsv"
    for ln in (f.read_text().splitlines() if f.exists() else []):
        kind, what, ts = ln.split("\t")
        out.setdefault(kind, {})[what] = float(ts)
    return out


def leftovers() -> list[str]:
    r = subprocess.run(["pgrep", "-fl", f"sleep {MARK}"], capture_output=True, text=True)
    return r.stdout.splitlines()


@pytest.fixture(autouse=True)
def _no_stray_children():
    assert not leftovers(), "a previous run left processes behind"
    yield
    subprocess.run(["pkill", "-f", f"sleep {MARK}"], capture_output=True)


def test_three_renders_overlap_when_jobs_allow(tmp_path):
    t0 = time.time()
    r = drive(tmp_path, jobs="3")
    wall = time.time() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    s = stamps(tmp_path)
    assert {"mix", "bass", "mel"} <= set(s), s
    starts = [s[k]["start"] for k in ("mix", "bass", "mel")]
    ends = [s[k]["end"] for k in ("mix", "bass", "mel")]
    assert max(starts) < min(ends), f"renders did not overlap: starts {starts} ends {ends}"
    assert max(ends) - min(starts) < 1.6 * T, f"iteration took {max(ends) - min(starts):.2f}s for three {T}s renders (sequential = {3 * T}s)"
    assert wall < 3 * T + 2.5, wall
    assert "render jobs: 3 (MIDIGREP_RENDER_JOBS" in r.stdout


def test_compare_overlaps_the_solo_renders(tmp_path):
    # mix 0.4 s, solos 2.0 s, compare 0.6 s: the compare starts when the MIX is done, while the solos still render
    r = drive(tmp_path, jobs="3", extra_env={"STUB_T_MIX": "0.4", "STUB_T_BASS": "2.0", "STUB_T_MEL": "2.0", "STUB_T_CMP": "0.6"})
    assert r.returncode == 0, r.stdout + r.stderr
    s = stamps(tmp_path)
    assert s["cmp"]["start"] >= s["mix"]["end"] - 0.01
    assert s["bass"]["start"] < s["cmp"]["start"] and s["mel"]["start"] < s["cmp"]["start"], "the solos were already rendering"
    assert s["cmp"]["start"] < s["bass"]["end"] - 1.0 and s["cmp"]["end"] <= s["bass"]["end"] + 0.05, s
    assert "overall=0.8" in r.stdout and "solo stub: measured" in r.stdout


def test_jobs_1_is_sequential_in_the_old_order(tmp_path):
    t0 = time.time()
    r = drive(tmp_path, jobs="1")
    wall = time.time() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    s = stamps(tmp_path)
    order = sorted(("mix", "bass", "mel"), key=lambda k: s[k]["start"])
    assert order == ["mix", "bass", "mel"]
    assert s["bass"]["start"] >= s["mix"]["end"] and s["mel"]["start"] >= s["bass"]["end"], s
    assert s["cmp"]["start"] >= s["mix"]["end"] and s["cmp"]["end"] <= s["bass"]["start"] + 0.05, "old order: mix, compare, then the solos"
    assert wall >= 3 * T


def test_jobs_2_never_runs_three_at_once(tmp_path):
    r = drive(tmp_path, jobs="2")
    assert r.returncode == 0, r.stdout + r.stderr
    s = stamps(tmp_path)
    ev = sorted([(s[k]["start"], 1) for k in ("mix", "bass", "mel")] + [(s[k]["end"], -1) for k in ("mix", "bass", "mel")])
    live = peak = 0
    for _, d in ev:
        live += d
        peak = max(peak, live)
    assert peak == 2, (peak, s)
    assert max(s[k]["end"] for k in ("mix", "bass", "mel")) - min(s[k]["start"] for k in s if k != "cmp") < 2.6 * T


def test_the_concurrent_run_leaves_the_same_artefacts_and_calibration_input(tmp_path):
    a, b = tmp_path / "seq", tmp_path / "par"
    ra, rb = drive(a, jobs="1"), drive(b, jobs="3")
    assert ra.returncode == 0 and rb.returncode == 0, (ra.stderr, rb.stderr)
    names = lambda d: sorted(p.name for p in (d / "work").iterdir() if not p.name.endswith(".rc"))  # noqa: E731
    assert names(a) == names(b)
    for f in ("cal1.prop", "cal1.bass.json", "cal1.mel.json", "cal1.cmp.json", "cal1.bass.msg", "cal1.mel.msg"):
        assert (a / "work" / f).read_text() == (b / "work" / f).read_text(), f
    keep = lambda r, d: [ln.replace(str(d), "<d>") for ln in r.stdout.splitlines() if not ln.startswith(("render jobs:", "workdir:"))]  # noqa: E731
    assert keep(ra, a) == keep(rb, b)


def test_a_failed_mix_render_is_render_failed_rc_70_and_leaves_no_process(tmp_path):
    r = drive(tmp_path, jobs="3", extra_env={"STUB_FAIL": "mix", "STUB_FAIL_AFTER": "0.3", "STUB_T_BASS": "30", "STUB_T_MEL": "30"})
    assert r.returncode == 70, r.stdout + r.stderr
    assert "[cal1] RENDER FAILED (rc=3, see " in r.stdout
    assert not (tmp_path / "work" / "cal1.wav").exists()
    time.sleep(0.2)
    assert leftovers() == [], "background renders (and their grandchildren) outlived the failed run"
    s = stamps(tmp_path)
    assert "end" not in s.get("bass", {}), "the solo lanes were terminated, not waited out"


def test_a_failed_mix_render_with_jobs_1_is_the_same(tmp_path):
    r = drive(tmp_path, jobs="1", extra_env={"STUB_FAIL": "mix"})
    assert r.returncode == 70 and "[cal1] RENDER FAILED (rc=3," in r.stdout
    assert "bass" not in stamps(tmp_path), "sequentially, a failed mix render never starts the solos"


def test_a_failed_solo_render_only_warns(tmp_path):
    r = drive(tmp_path, jobs="3", extra_env={"STUB_FAIL": "bass"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[cal1] WARNING: bass-alone measurement failed (see " in r.stdout
    assert "melodic" not in r.stdout.split("WARNING")[1].split("\n")[0]
    assert not (tmp_path / "work" / "cal1.bass.json").exists() and (tmp_path / "work" / "cal1.mel.json").exists()
    time.sleep(0.2)
    assert leftovers() == []


def test_sigterm_mid_render_terminates_every_child(tmp_path):
    p = drive(tmp_path, jobs="3", t=30, popen=True)
    deadline = time.time() + 20
    while time.time() < deadline and len(stamps(tmp_path)) < 3:
        time.sleep(0.1)
    assert len(stamps(tmp_path)) == 3, "renders never started"
    assert leftovers(), "the stub grandchildren should be alive now"
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    assert p.returncode == 143
    time.sleep(0.3)
    assert leftovers() == []


def test_invalid_override_is_an_error_not_a_default(tmp_path):
    r = drive(tmp_path, jobs="zero")
    assert r.returncode == 64 and "MIDIGREP_RENDER_JOBS" in r.stderr


# ---- the derivation --------------------------------------------------------------------------------------------------------

def test_derivation_is_capped_by_the_independent_jobs():
    d = RJ.derive_jobs(3, env={}, cpus=64, available_mb=512 * 1024)
    assert d.jobs == 3 and d.independent == 3
    assert RJ.derive_jobs(1, env={}, cpus=64, available_mb=512 * 1024).jobs == 1


def test_derivation_follows_cores_and_memory():
    c = RJ.RENDER_CORES
    assert RJ.derive_jobs(3, env={}, cpus=int(c), available_mb=64 * 1024).jobs == 1                 # cores for one render
    assert RJ.derive_jobs(3, env={}, cpus=int(2 * c) + 1, available_mb=64 * 1024).jobs == 2         # cores for two
    need = RJ.HEADROOM_MB + 2 * RJ.RENDER_RSS_MB
    assert RJ.derive_jobs(3, env={}, cpus=64, available_mb=need + 1).jobs == 2                     # memory for two
    assert RJ.derive_jobs(3, env={}, cpus=64, available_mb=RJ.HEADROOM_MB).jobs == 1               # short of memory: one
    assert RJ.derive_jobs(3, env={}, cpus=64, available_mb=None).jobs == 1                         # machine unreadable: one
    assert RJ.derive_jobs(3, env={}, cpus=64, available_mb=None).source == "unknown-machine"


def test_derivation_env_override_and_validation():
    assert RJ.derive_jobs(3, env={"MIDIGREP_RENDER_JOBS": "2"}, cpus=1, available_mb=1).jobs == 2   # beats the machine
    assert RJ.derive_jobs(3, env={"MIDIGREP_RENDER_JOBS": "8"}, cpus=1, available_mb=1).jobs == 3   # but never above the independent jobs
    for bad in ("0", "-1", "x", "1.5"):
        with pytest.raises(ValueError, match="MIDIGREP_RENDER_JOBS"):
            RJ.derive_jobs(3, env={"MIDIGREP_RENDER_JOBS": bad})
    assert RJ.derive_jobs(3, env={"MIDIGREP_RENDER_RSS_MB": "100000"}, cpus=64, available_mb=64 * 1024).jobs == 1


def test_the_machine_is_readable_here():
    mem = RJ.available_memory_mb()
    assert mem is not None and mem > 100, mem


def test_cli_prints_the_number_and_the_reason(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPTS / "render_jobs.py"), "jobs", "--independent", "3"], capture_output=True,
                       text=True, env={**os.environ, "MIDIGREP_RENDER_JOBS": "2"})
    assert r.returncode == 0 and r.stdout.strip() == "2" and "MIDIGREP_RENDER_JOBS" in r.stderr
    r = subprocess.run([sys.executable, str(SCRIPTS / "render_jobs.py"), "jobs", "--independent", "3"], capture_output=True,
                       text=True, env={**os.environ, "MIDIGREP_RENDER_JOBS": "nope"})
    assert r.returncode == 2


# ---- stage 10: the report's three solo stem renders ------------------------------------------------------------------------

def test_report_stem_renders_run_concurrently(tmp_path, monkeypatch):
    import threading

    import pipeline_helpers as PH
    import soundfile as sf
    import numpy as np

    vdir = tmp_path / "v001"
    vdir.mkdir()
    (vdir / "output.strudel").write_text(
        '$: note("c2").s("sine").orbit(2)  // BASS — low end\n$: note("c4").s("sawtooth")  // LEAD — melody\n'
        '$: s("bd").bank("RolandTR808").duckorbit(2).duckdepth(0.5)  // DRUMS — groove\n')
    sf.write(str(vdir / "render.wav"), np.zeros(8000, dtype="float32"), 8000)
    live = peak = 0
    lock = threading.Lock()

    def run(argv):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.4)
        sf.write(str(argv[argv.index("-o") + 1]), np.zeros(8000, dtype="float32"), 8000)
        with lock:
            live -= 1

    monkeypatch.setenv("MIDIGREP_RENDER_JOBS", "3")
    PH.solo_render_stems(vdir, vdir / "render.wav", run=run)
    assert peak == 3, peak
    assert all((vdir / f"render_{g}.wav").is_file() for g in PH.RENDER_STEMS)
    live = peak = 0
    monkeypatch.setenv("MIDIGREP_RENDER_JOBS", "1")
    PH.solo_render_stems(vdir, vdir / "render.wav", run=run)
    assert peak == 1, peak


def test_report_stem_render_failure_propagates_and_waits(tmp_path, monkeypatch):
    import pipeline_helpers as PH
    import soundfile as sf
    import numpy as np

    vdir = tmp_path / "v001"
    vdir.mkdir()
    (vdir / "output.strudel").write_text('$: note("c2").s("sine")  // BASS — low end\n$: s("bd")  // DRUMS — groove\n')
    sf.write(str(vdir / "render.wav"), np.zeros(8000, dtype="float32"), 8000)

    def run(argv):
        if "bass" in Path(str(argv[2])).name:
            raise subprocess.CalledProcessError(3, "node")
        time.sleep(0.2)
        sf.write(str(argv[argv.index("-o") + 1]), np.zeros(8000, dtype="float32"), 8000)

    monkeypatch.setenv("MIDIGREP_RENDER_JOBS", "3")
    with pytest.raises(subprocess.CalledProcessError):
        PH.solo_render_stems(vdir, vdir / "render.wav", run=run)
