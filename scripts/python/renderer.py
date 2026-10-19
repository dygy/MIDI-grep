#!/usr/bin/env python3
"""Which Strudel renderer the Python tools drive (one switch, one default).

``offline`` (default): ``scripts/node/dist/render-strudel-offline.js`` - Strudel's own renderPatternAudio on an
OfflineAudioContext, exact timing and pitch, no audio device. ``blackhole``: the real-time capture recorder
``record-strudel-blackhole.js`` - it needs the BlackHole device and renders one at a time; its raw capture
loses samples, so its pitch/timing are off (see CLAUDE.md). Select with ``MIDIGREP_RENDERER=blackhole``.
"""
from __future__ import annotations

import os
from pathlib import Path

NODE_DIST = Path(__file__).resolve().parent.parent / "node" / "dist"
SCRIPTS = {"offline": "render-strudel-offline.js", "blackhole": "record-strudel-blackhole.js"}
DEFAULT = "offline"
# live-recorder processes a BlackHole render must wait for (the offline renderer has none)
BLACKHOLE_BUSY = "node .*record-strudel-blackhole\\.js|ffmpeg .* -f avfoundation"


def renderer_name(env: dict[str, str] | None = None) -> str:
    name = ((os.environ if env is None else env).get("MIDIGREP_RENDERER") or DEFAULT).strip().lower()
    if name not in SCRIPTS:
        raise ValueError(f"MIDIGREP_RENDERER={name!r}: expected one of {sorted(SCRIPTS)}")
    return name


def renderer_script(env: dict[str, str] | None = None) -> Path:
    return NODE_DIST / SCRIPTS[renderer_name(env)]


def needs_device_guard(env: dict[str, str] | None = None) -> bool:
    """True when renders need the BlackHole device (serialise them, probe the device)."""
    return renderer_name(env) == "blackhole"
