#!/usr/bin/env python3
"""Which Strudel renderer the Python tools drive (one switch, one default).

``offline`` (default): ``scripts/node/dist/render-strudel-offline.js`` - Strudel's own renderPatternAudio on an
OfflineAudioContext, exact timing and pitch, no audio device. ``blackhole``: the real-time capture recorder
``record-strudel-blackhole.js`` - it needs the BlackHole device and renders one at a time; its raw capture
loses samples, so its pitch/timing are off (see CLAUDE.md). Select with ``MIDIGREP_RENDERER=blackhole``.

``MIDIGREP_STRUDEL_BUNDLE=<@strudel/web dist/index.js>`` swaps the engine build the offline renderer loads (default: the
pinned npm ``@strudel/web``; the choice is logged in the render output). ``supports_master_bus`` says whether that engine
defines ``setMasterCompressor`` / ``setMasterLimiter`` (the user's fork does; npm 1.3.0 and the live site do not).

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


NPM_BUNDLE = NODE_DIST.parent / "node_modules" / "@strudel" / "web" / "dist" / "index.js"
MASTER_BUS_API = ("setMasterLimiter", "setMasterCompressor")


def engine_bundle(env: dict[str, str] | None = None) -> Path:
    """The @strudel/web bundle the offline renderer loads: ``MIDIGREP_STRUDEL_BUNDLE`` or the pinned npm one."""
    raw = ((os.environ if env is None else env).get("MIDIGREP_STRUDEL_BUNDLE") or "").strip()
    return Path(raw).expanduser().resolve() if raw else NPM_BUNDLE


def supports_master_bus(env: dict[str, str] | None = None) -> bool:
    """True when the engine that will render defines the master-bus functions (offline renderer only: the BlackHole
    recorder plays whatever the hosted site runs, which is not knowable here)."""
    if renderer_name(env) != "offline":
        return False
    try:
        text = engine_bundle(env).read_text(errors="ignore")
    except OSError:
        return False
    return all(fn in text for fn in MASTER_BUS_API)


def needs_device_guard(env: dict[str, str] | None = None) -> bool:
    """True when renders need the BlackHole device (serialise them, probe the device)."""
    return renderer_name(env) == "blackhole"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Which Strudel renderer is selected (MIDIGREP_RENDERER)")
    ap.add_argument("what", choices=("name", "script", "busy-pattern", "needs-device", "bundle", "master-bus"),
                    help="name: offline|blackhole; script: absolute path of the renderer .js; busy-pattern: pgrep -f "
                         "regex of live capture processes; needs-device: exit 0 when renders need BlackHole, else 1; "
                         "bundle: the engine bundle path; master-bus: exit 0 when that engine has setMasterLimiter/Compressor")
    what = ap.parse_args(argv).what
    try:
        if what == "name":
            print(renderer_name())
        elif what == "script":
            print(renderer_script())
        elif what == "busy-pattern":
            print(BLACKHOLE_BUSY)
        elif what == "bundle":
            print(engine_bundle())
        elif what == "master-bus":
            return 0 if supports_master_bus() else 1
        else:
            return 0 if needs_device_guard() else 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
