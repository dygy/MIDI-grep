# @layer: integration
# @spec: 004-second-reference-track
# @regression
"""editable-pipeline.sh resume logic, driven end-to-end against a hermetic copy of the repo whose heavy tools are
stubs (they log their argv): stage 3/4 follow the track's CURRENT bpm, stage 5/6 follow what was rebuilt, stage 7
reruns a best.params in the removed-CV format."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
SLUG, TITLE = "trk", "Trk"

STUBS = {
    "detect_drums_bands.py": 'import sys, json\nfrom pathlib import Path\n'
        'a = sys.argv[1:]\nout = a[a.index("--out") + 1]\n'
        'open(Path(__file__).parent / "calls.log", "a").write("drums_bands bpm=" + a[a.index("--bpm") + 1] + "\\n")\n'
        'open(out, "w").write("{}")\n',
    "build_sample_pack.py": 'import sys, json\nfrom pathlib import Path\n'
        'a = sys.argv[1:]\nout = Path(a[a.index("--out") + 1])\nbase = a[a.index("--base-url") + 1]\n'
        'open(Path(__file__).parent / "calls.log", "a").write("sample_pack bpm=" + a[a.index("--bpm") + 1] + "\\n")\n'
        '(out / "drums").mkdir(parents=True, exist_ok=True)\n(out / "drums" / "bd.wav").write_bytes(b"x")\n'
        '(out / "pack.json").write_text("{}")\n(out / "strudel.json").write_text("{}")\n'
        '(out / "samples.json").write_text(json.dumps({"_base": base}))\n',
    "build_instruments.py": 'import sys, json\nfrom pathlib import Path\n'
        'a = sys.argv[1:]\nout = Path(a[a.index("--out") + 1])\nbase = a[a.index("--base-url") + 1]\n'
        'open(Path(__file__).parent / "calls.log", "a").write("instruments\\n")\n'
        'out.mkdir(parents=True, exist_ok=True)\n(out / "samples.json").write_text(json.dumps({"_base": base, "bass": ["bass.wav"]}))\n'
        '(out / "verification.json").write_text("{}")\n'
        'model = Path(a[a.index("--models") + 1])\n(out / "bass.wav").write_bytes((model / "a.wav").read_bytes())\n',
    "bass_register.py": 'import sys\na = sys.argv[1:]\nopen(a[a.index("--out") + 1], "w").write("{}")\n',   # stage 3b (real one needs a real stem)
    # the real planner (pack_sync, symlinked in) decides; only the network step is stubbed
    "upload_r2.py": 'import sys\nfrom pathlib import Path\nimport pack_sync\n'
        'a = sys.argv[1:]\npack = Path(a[a.index("--pack-dir") + 1])\n'
        'base = a[a.index("--public-base") + 1].rstrip("/") + "/" + a[a.index("--prefix") + 1].strip("/")\n'
        'h = pack_sync.hash_served(pack)\nplan = pack_sync.plan_sync(pack, base, a[a.index("--bucket") + 1], force="--full" in a, hashes=h)\n'
        'if "--check" in a:\n    sys.exit(0 if plan.is_noop else 3)\n'
        'open(Path(__file__).parent / "calls.log", "a").write("upload\\n")\n'
        'pack_sync.write_state(pack, base, a[a.index("--bucket") + 1], h)\n',
    "compare_audio.py": 'import sys\nsys.exit(9)\n',        # stops the pipeline right after stage 7 (stage 8 dies)
    "transcribe.py": 'import sys\nopen(sys.argv[2], "w").write("mid")\n',
}


def _exe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture()
def repo(tmp_path: Path):
    root = tmp_path / "repo"
    sp = root / "scripts" / "python"
    sp.mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "editable-pipeline.sh", root / "scripts" / "editable-pipeline.sh")
    for f in SCRIPTS.glob("*.py"):
        if f.name not in STUBS and f.name != "renderer.py":
            (sp / f.name).symlink_to(f)
    shutil.copy(SCRIPTS / "renderer.py", sp / "renderer.py")      # copied: NODE_DIST must resolve inside the hermetic tree
    for name, body in STUBS.items():
        (sp / name).write_text(body)
    _exe(sp / ".venv" / "bin" / "python", f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    _exe(root / "scripts" / "auto-calibrate.sh",
         '#!/bin/bash\necho "auto-calibrate $*" >> "$(dirname "$0")/python/calls.log"\nexit 7\n')
    (root / "scripts" / "node" / "dist").mkdir(parents=True)
    (root / "scripts" / "node" / "dist" / "render-strudel-offline.js").write_text("//")
    (root / "eval" / "datasets").mkdir(parents=True)
    (root / "eval" / "datasets" / "reference_tracks.yaml").write_text("tracks: []\n")
    (root / "eval" / "gate.py").write_text("")
    bindir = tmp_path / "bin"
    _exe(bindir / "curl", "#!/bin/bash\nexit 0\n")
    track = root / ".cache" / "stems" / TITLE
    track.mkdir(parents=True)
    for n in ("original", "bass", "drums", "melodic", "vocals"):
        soundfile.write(str(track / f"{n}.wav"), (0.1 * np.ones((2205, 2))).astype("float32"), 44100, subtype="FLOAT")
    set_bpm(track, 129.989)
    pack = track / "sample_pack"
    pack.mkdir()
    (pack / "bass.mid").write_text("m")
    (pack / "melodic.mid").write_text("m")
    for role in ("bass", "lead"):
        m = root / "models" / f"{SLUG}_{role}"
        m.mkdir(parents=True)
        (m / "metadata.json").write_text(json.dumps({"grain_mode": "notes"}))
        (m / "a.wav").write_bytes(b"x")
    return type("Repo", (), {"root": root, "track": track, "pack": pack, "bin": bindir,
                             "calls": sp / "calls.log", "pw": track / ".pipeline"})


def set_bpm(track: Path, bpm: float) -> None:
    (track / "metadata.json").write_text(json.dumps(
        {"title": TITLE, "bpm": bpm, "bpm_method": "phase", "key": "C minor", "genre": "g", "duration": 100.0}))


def run(repo, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{repo.bin}:/usr/bin:/bin", "HOME": str(repo.root)}
    return subprocess.run(["/bin/bash", str(repo.root / "scripts" / "editable-pipeline.sh"), "--title", TITLE,
                           "--prefix", SLUG, *args], capture_output=True, text=True, env=env, timeout=120)


def calls(repo) -> list[str]:
    return repo.calls.read_text().splitlines() if repo.calls.exists() else []


# --- #4 stages 3/4 follow the track's current bpm ------------------------------------------------------
def test_stages_3_and_4_are_rebuilt_when_the_stamped_bpm_changes(repo):
    run(repo, "--local", "--mode", "synth")
    assert "drums_bands bpm=129.989" in calls(repo) and "sample_pack bpm=129.989" in calls(repo)
    n = len(calls(repo))
    r = run(repo, "--local", "--mode", "synth")
    assert [c for c in calls(repo)[n:] if c.startswith(("drums_bands", "sample_pack"))] == [], r.stderr   # unchanged bpm: reuse
    set_bpm(repo.track, 130.0)                                    # the forced re-stamp moved the bpm
    n = len(calls(repo))
    r = run(repo, "--local", "--mode", "synth")
    again = calls(repo)[n:]
    assert "drums_bands bpm=130.0" in again and "sample_pack bpm=130.0" in again, r.stderr


# --- #3 stage 6 follows what was rebuilt ---------------------------------------------------------------
def test_upload_is_repeated_when_the_pack_changed_after_the_last_upload(repo):
    run(repo, "--r2", "--mode", "synth")
    assert calls(repo).count("upload") == 1
    n = len(calls(repo))
    run(repo, "--r2", "--mode", "synth")
    assert calls(repo)[n:].count("upload") == 0, "nothing changed: the upload must be skipped"
    (repo.pack / "drums_bands.json").write_text('{"stage": 3}')    # an analysis file: never served, never re-uploaded
    r = run(repo, "--r2", "--mode", "synth")
    assert calls(repo)[n:].count("upload") == 0, "drums_bands.json is not served: the upload must be skipped"
    model_file = repo.root / "models" / f"{SLUG}_bass" / "a.wav"   # stage 5 retrained the bass model
    model_file.write_bytes(b"retrained")
    n = len(calls(repo))
    r = run(repo, "--r2", "--mode", "synth")
    again = calls(repo)[n:]
    assert "instruments" in again, r.stderr                        # the manifest follows its models
    assert again.count("upload") == 1, "R2 would keep serving the stale instruments"


# --- #5 old-format best.params ------------------------------------------------------------------------
OLD = "0.4 0.7 1.0 5000 0.1 0.6 1.0 422 300"
NEW = "0.4 0.7 1.0 5000 0.1 0.6 422 300"


def _code_version() -> str:
    return subprocess.run([sys.executable, str(SCRIPTS / "calibration_memory.py"), "code-version"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _seed_best(repo, mode: str, params: str, code: str | None = None) -> Path:
    w = repo.pw / mode
    w.mkdir(parents=True, exist_ok=True)
    for n, body in (("best.strudel", "//"), ("best.wav", "w"), ("best.cmp.json", "{}"), ("best.params", params + "\n"),
                    ("result.json", "{}"), ("best.code", (code if code is not None else _code_version()) + "\n")):
        (w / n).write_text(body)
    return w


def test_old_format_best_params_reruns_the_stage_and_invalidates_its_result(repo):
    w = _seed_best(repo, "sample-instrument", OLD)
    r = run(repo, "--local", "--mode", "sample-instrument")
    assert any(c.startswith("auto-calibrate") for c in calls(repo)), r.stderr
    assert "old (CV) format" in r.stderr and not (w / "result.json").exists()


def test_current_format_best_params_is_reused(repo):
    _seed_best(repo, "sample-instrument", NEW)
    r = run(repo, "--local", "--mode", "sample-instrument")
    assert not any(c.startswith("auto-calibrate") for c in calls(repo)), r.stderr
    assert "stage 7 skip (sample-instrument)" in r.stderr


def test_synth_does_not_start_from_an_old_format_sample_instrument_params(repo):
    _seed_best(repo, "sample-instrument", OLD)
    run(repo, "--local", "--mode", "synth")
    ac = [c for c in calls(repo) if c.startswith("auto-calibrate")]
    assert ac and "--seed" not in ac[0], ac            # auto-calibrate --seed would exit 64 on the 9-field line
    _seed_best(repo, "sample-instrument", NEW)
    (repo.calls).write_text("")
    run(repo, "--local", "--mode", "synth")
    assert f"--seed {NEW}" in [c for c in calls(repo) if c.startswith("auto-calibrate")][0]


def test_synth_is_seeded_with_the_sample_magnitudes_but_not_its_sub_octave(repo):
    """The 9th knob (sub-sine octave up) is a per-instrument decision: synth must re-trial it, not inherit su=1."""
    _seed_best(repo, "sample-instrument", NEW + " 1")                 # BM..MG BL LH SU=1 (9 fields)
    (repo.calls).write_text("")
    run(repo, "--local", "--mode", "synth")
    ac = [c for c in calls(repo) if c.startswith("auto-calibrate")][0]
    assert f"--seed {NEW}" in ac and f"--seed {NEW} 1" not in ac, ac


# --- a best render calibrated by other code is rerun (a code change must reach the output) ----------------
@pytest.mark.parametrize("stamp", ["0000000000", ""])
def test_best_from_other_code_reruns_the_stage_and_invalidates_its_result(repo, stamp):
    w = _seed_best(repo, "synth", NEW, code=stamp)
    if not stamp:
        (w / "best.code").unlink()                    # a best written before renders were stamped
    r = run(repo, "--local", "--mode", "synth")
    assert any(c.startswith("auto-calibrate") for c in calls(repo)), r.stderr
    assert "from other code" in r.stderr and not (w / "result.json").exists()
