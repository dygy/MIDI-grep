# @layer: integration
# @spec: 001-core-pipeline
# @regression
"""One place decides which renderer runs: renderer.py (normalises case/whitespace, CLI for the shell drivers,
imported by the MCP server, owns the BlackHole busy regex)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))
import renderer  # noqa: E402


def cli(what: str, value: str | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["MIDIGREP_RENDERER"] = value
    return subprocess.run([sys.executable, str(SCRIPTS / "renderer.py"), what], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("value,name", [(None, "offline"), ("", "offline"), (" BlackHole\n", "blackhole"), ("OFFLINE", "offline")])
def test_cli_name_and_script_normalise_case_and_whitespace(value, name):
    assert cli("name", value).stdout.strip() == name
    assert Path(cli("script", value).stdout.strip()) == renderer.NODE_DIST / renderer.SCRIPTS[name]


def test_cli_rejects_a_bad_value_with_exit_2_and_a_message():
    r = cli("name", "node")
    assert r.returncode == 2 and "MIDIGREP_RENDERER" in r.stderr and r.stdout == ""


def test_cli_needs_device_and_busy_pattern():
    assert cli("needs-device", "blackhole").returncode == 0 and cli("needs-device").returncode == 1
    assert cli("busy-pattern").stdout.strip() == renderer.BLACKHOLE_BUSY


def test_the_busy_regex_and_renderer_names_live_only_in_renderer_py():
    needles = ("ffmpeg .* -f avfoundation", "record-strudel-blackhole\\.js")
    offenders = []
    for f in [*(ROOT / "scripts").glob("*.sh"), *SCRIPTS.glob("*.py"), ROOT / "mcp_servers" / "loop" / "server.py"]:
        if f.name == "renderer.py":
            continue
        text = f.read_text()
        for n in needles:
            if n in text:
                offenders.append(f"{f.name}: {n}")
    assert offenders == []


def test_auto_calibrate_accepts_a_padded_mixed_case_value_and_rejects_a_bad_one(tmp_path):
    from test_calibrate_bass_lpf import AUTO, run_driver
    ok = run_driver(tmp_path, AUTO, "--mode", "synth", "--start", "0.447 0.707 0.969 5199 0.0 0.672",
                    extra_env={"MIDIGREP_RENDERER": " Offline "})
    assert ok.returncode == 0, ok.stderr
    (tmp_path / "b").mkdir()
    bad = run_driver(tmp_path / "b", AUTO, "--mode", "synth", extra_env={"MIDIGREP_RENDERER": "node"})
    assert bad.returncode == 2 and "MIDIGREP_RENDERER" in bad.stderr


# --- MCP server ----------------------------------------------------------------------------------------
@pytest.fixture()
def server():
    pytest.importorskip("fastmcp")
    sys.path.insert(0, str(ROOT))
    from mcp_servers.loop import server as s
    return s


def test_mcp_uses_renderer_py_and_reads_the_env_at_call_time(server, monkeypatch):
    assert server.renderer_script is renderer.renderer_script
    monkeypatch.setenv("MIDIGREP_RENDERER", " BLACKHOLE ")
    assert server.renderer_script().name == "record-strudel-blackhole.js"


def test_mcp_returns_ok_false_instead_of_raising_on_a_bad_value(server, monkeypatch, tmp_path):
    monkeypatch.setenv("MIDIGREP_RENDERER", "node")
    monkeypatch.setattr(server, "NODE_MODULES", tmp_path)          # exists: reach the renderer check
    out = server.render_strudel(str(tmp_path / "a.strudel"), str(tmp_path / "a.wav"))
    assert out["ok"] is False and "MIDIGREP_RENDERER" in out["error"]
    v = server.verify_strudel(str(ROOT / "scripts" / "python" / "tests" / "fixtures" / "editability" / "v023_minus_vocal.strudel"),
                              original=str(tmp_path / "o.wav"))
    assert v["ok"] is False and v["stage"] == "render" and "MIDIGREP_RENDERER" in v["error"]
