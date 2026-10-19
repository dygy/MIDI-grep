#!/usr/bin/env python3
"""Which Strudel renderer the Python tools drive (one switch, one default).

``offline`` (default): ``scripts/node/dist/render-strudel-offline.js`` - Strudel's own renderPatternAudio on an
OfflineAudioContext, exact timing and pitch, no audio device. ``blackhole``: the real-time capture recorder
``record-strudel-blackhole.js`` - it needs the BlackHole device and renders one at a time; its raw capture
loses samples, so its pitch/timing are off (see CLAUDE.md). Select with ``MIDIGREP_RENDERER=blackhole``.

This module is the ONLY place that reads ``MIDIGREP_RENDERER`` (case and whitespace are normalised here). The
shell drivers ask it through the CLI (``renderer.py name|script|busy-pattern|needs-device``; exit 2 and a message
on stderr for a bad value) and the loop MCP server imports it.
"""
from __future__ import annotations

import argparse
import os
import sys
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Which Strudel renderer is selected (MIDIGREP_RENDERER)")
    ap.add_argument("what", choices=("name", "script", "busy-pattern", "needs-device"),
                    help="name: offline|blackhole; script: absolute path of the renderer .js; busy-pattern: pgrep -f "
                         "regex of live capture processes; needs-device: exit 0 when renders need BlackHole, else 1")
    what = ap.parse_args(argv).what
    try:
        if what == "name":
            print(renderer_name())
        elif what == "script":
            print(renderer_script())
        elif what == "busy-pattern":
            print(BLACKHOLE_BUSY)
        else:
            return 0 if needs_device_guard() else 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
