# @layer: unit
# @spec: 001-core-pipeline
"""One switch picks the Strudel renderer: offline by default (exact timing and pitch), BlackHole on request."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import renderer  # noqa: E402

ROOT = SCRIPTS.parent.parent


def test_default_is_offline_and_env_selects_blackhole() -> None:
    assert renderer.renderer_name({}) == "offline"
    assert renderer.renderer_script({}).name == "render-strudel-offline.js"
    assert not renderer.needs_device_guard({})
    env = {"MIDIGREP_RENDERER": "blackhole"}
    assert renderer.renderer_name(env) == "blackhole"
    assert renderer.renderer_script(env).name == "record-strudel-blackhole.js"
    assert renderer.needs_device_guard(env)


def test_unknown_renderer_is_rejected() -> None:
    with pytest.raises(ValueError):
        renderer.renderer_name({"MIDIGREP_RENDERER": "node"})


@pytest.mark.parametrize("mod", ["probe_sound_register", "audition_drum_banks", "editability_test"])
def test_python_tools_default_to_the_offline_renderer(mod: str) -> None:
    m = __import__(mod)
    assert m.RECORDER.name == "render-strudel-offline.js"


def test_probe_does_not_wait_for_a_capture_device_when_offline() -> None:
    import probe_sound_register as p
    p.wait_for_idle_recorder(max_s=0.0)   # returns immediately (no pgrep loop) under the default renderer


@pytest.mark.parametrize("sh", ["scripts/auto-calibrate.sh", "scripts/editable-pipeline.sh", "scripts/sample-pipeline.sh"])
def test_shell_scripts_parse_and_default_to_offline(sh: str) -> None:
    assert subprocess.run(["bash", "-n", str(ROOT / sh)]).returncode == 0
    text = (ROOT / sh).read_text()
    assert 'MIDIGREP_RENDERER:-offline' in text and "render-strudel-offline.js" in text


def test_loop_server_defaults_to_offline() -> None:
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("fastmcp")
    from mcp_servers.loop import server
    assert server.RENDERER == "offline" and server._renderer_script() == server.NODE_OFFLINE
