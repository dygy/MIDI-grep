# @layer: unit
# @spec: 002-ml-customization
# @regression
"""rave/trainer.py RAVETrainer calls the REAL ``rave`` CLI (pip acids-rave 2.3.1) and fails loudly.

Defect: the trainer shelled ``python -m acids_rave ...``. The pip package ``acids-rave`` installs the console script ``rave`` and the
python packages ``rave`` / ``scripts`` - there is no ``acids_rave`` module (and this repo's own ``rave/`` package shadows the
installed one under ``python -m rave``). So every step failed, the preprocess fallback silently COPIED the wavs as a "preprocessed
dataset", the training exit status was ignored, and the flags (``--batch_size`` ...) were not the CLI's (``--batch``). RAVE is
never trained here: only the command lines, the failure handling, and - when the CLI is installed - the flags against ``--help``.
"""
from __future__ import annotations

import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from rave.trainer import RAVETrainer  # noqa: E402

RAVE_CLI = Path(sys.executable).parent / "rave"


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    """A stand-in `rave` executable beside a stand-in interpreter path; the script's exit code comes from $FAKE_RAVE_RC."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cli = bindir / "rave"
    cli.write_text('#!/bin/bash\necho "fake rave $@"\nexit ${FAKE_RAVE_RC:-0}\n')
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    return cli


def test_the_trainer_calls_the_rave_console_script_not_a_python_module(fake_cli, tmp_path):
    t = RAVETrainer(str(tmp_path / "models"))
    assert t.cli == str(fake_cli)
    for cmd in (t.preprocess_cmd(tmp_path / "d", tmp_path / "p"), t.train_cmd(tmp_path / "p", tmp_path / "c", "m", 100, 10),
                t.export_cmd(tmp_path / "r", "m", tmp_path)):
        assert cmd[0] == str(fake_cli) and "-m" not in cmd and "acids_rave" not in " ".join(cmd)
    assert t.preprocess_cmd(tmp_path / "d", tmp_path / "p")[1] == "preprocess"
    assert t.export_cmd(tmp_path / "r", "m", tmp_path)[1] == "export"


def test_train_flags_are_the_cli_flags(fake_cli, tmp_path):
    t = RAVETrainer(str(tmp_path / "models"), batch_size=4, n_signal=32768)
    cmd = t.train_cmd(tmp_path / "p", tmp_path / "c", "m", 500, 50, "v2")
    assert cmd[1] == "train"
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert flags["--batch"] == "4" and "--batch_size" not in flags             # the CLI's flag is --batch
    assert flags["--db_path"] == str(tmp_path / "p") and flags["--out_path"] == str(tmp_path / "c")
    assert flags["--config"] == "v2" and flags["--n_signal"] == "32768" and flags["--max_steps"] == "500"
    assert flags["--gpu"] == "-1"                                              # acids-rave exits on a Mac GPU: CPU is explicit
    cuda = RAVETrainer(str(tmp_path / "m2"), device="cuda").train_cmd(tmp_path / "p", tmp_path / "c", "m", 500, 50)
    assert "--gpu" not in cuda


def test_a_missing_cli_is_an_error_not_a_pip_install(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="acids-rave"):
        RAVETrainer(str(tmp_path / "models"))


def test_a_failing_preprocess_raises_instead_of_copying_the_wavs(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RAVE_RC", "3")
    t = RAVETrainer(str(tmp_path / "models"))
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.wav").write_bytes(b"RIFF")
    out = tmp_path / "pre"
    with pytest.raises(RuntimeError, match=r"preprocess failed \(exit 3\)"):
        t._run_preprocess(ds, out)
    assert not list(out.glob("*.wav"))                                         # nothing was faked as a dataset


def test_a_preprocess_that_writes_nothing_raises(fake_cli, tmp_path):
    t = RAVETrainer(str(tmp_path / "models"))
    ds = tmp_path / "ds"
    ds.mkdir()
    with pytest.raises(RuntimeError, match="wrote nothing"):
        t._run_preprocess(ds, tmp_path / "pre")


def test_a_failing_train_raises_and_a_train_without_checkpoints_raises(fake_cli, tmp_path, monkeypatch):
    t = RAVETrainer(str(tmp_path / "models"))
    ckpt = tmp_path / "ck"
    with pytest.raises(RuntimeError, match="left no checkpoint"):               # exit 0 but no run dir / .ckpt
        t._run_training(tmp_path / "p", ckpt, "m", 1, 10)
    monkeypatch.setenv("FAKE_RAVE_RC", "1")
    with pytest.raises(RuntimeError, match=r"train failed \(exit 1\)"):
        t._run_training(tmp_path / "p", ckpt, "m", 1, 10)


def test_the_run_dir_and_export_target_come_from_what_rave_wrote(fake_cli, tmp_path):
    t = RAVETrainer(str(tmp_path / "models"))
    ck = tmp_path / "ck"
    (ck / "m_abc123" / "version_0" / "checkpoints").mkdir(parents=True)
    (ck / "m_abc123" / "version_0" / "checkpoints" / "last.ckpt").write_bytes(b"x")
    assert t._run_training(tmp_path / "p", ck, "m", 1, 10) == ck / "m_abc123"
    with pytest.raises(RuntimeError, match="did not write"):                   # exit 0 but no .ts
        t._export_model(ck / "m_abc123", "m", tmp_path / "out" / "m.ts")


@pytest.mark.skipif(not RAVE_CLI.is_file(), reason="acids-rave CLI not installed in this venv")
@pytest.mark.parametrize("sub", ["preprocess", "train", "export"])
def test_every_flag_we_pass_exists_in_the_installed_cli(sub, tmp_path):
    out = subprocess.run([str(RAVE_CLI), sub, "--help"], capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    known = set(re.findall(r"--(?:\[no\])?([a-z_]+)", out.stdout + out.stderr))
    t = RAVETrainer(str(tmp_path / "models"))
    cmd = {"preprocess": t.preprocess_cmd(tmp_path / "d", tmp_path / "p"), "train": t.train_cmd(tmp_path / "p", tmp_path / "c", "m", 100, 10),
           "export": t.export_cmd(tmp_path / "r", "m", tmp_path)}[sub]
    ours = {a[2:] for a in cmd if a.startswith("--")}
    assert known and ours <= known, f"rave {sub}: flags not in `rave {sub} --help`: {sorted(ours - known)}"
