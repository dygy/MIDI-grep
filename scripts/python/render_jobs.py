#!/usr/bin/env python3
"""How many offline renders may run at once on THIS machine.

Several stages render independent strudels (the calibration iteration's mix + bass solo + melodic solo, the report's
three stem renders). Each render is one headless Chromium plus its node driver; running them together is faster only while
the machine has the cores and the memory for it, so the number is derived, never a constant::

    jobs = min(independent, cpus // CPU_CORES_PER_RENDER, (available_memory - HEADROOM) // RENDER_RSS)      (at least 1)

``independent`` is the number of renders the caller can start without waiting for another one's result - the cap.
``MIDIGREP_RENDER_JOBS=<n>`` overrides the derivation (still capped at ``independent``); ``0``/garbage is an error, not a
silent default. When the machine cannot be read (no ``vm_stat``, no ``/proc/meminfo``) the answer is 1.

The two per-render numbers are MEASURED (peak resident size of the node + Chromium process tree, and CPU seconds per wall
second, over full offline renders of a real calibration strudel; see ``RENDER_RSS_MB`` / ``RENDER_CORES``); the
environment variables ``MIDIGREP_RENDER_RSS_MB`` and ``MIDIGREP_RENDER_CORES`` replace them when a different engine
build is heavier.

CLI::

    render_jobs.py jobs --independent N      # prints the number; the derivation goes to stderr
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Measured 2026-10-29 on a real calibration strudel (VAGABUNDO cal1, 149.6 s of audio, darwin arm64, 14 cores, 24 GB):
RENDER_RSS_MB = 1500.0      # peak SUM of RSS over the node driver + Chromium tree of ONE render: 950 / 1070 / 1430 MB for the mix / bass solo / melodic
                            # solo (sampled every 0.5 s; an upper bound - shared pages count once per process); the largest single process peaks at 560 MB
RENDER_CORES = 1.1          # CPU seconds per wall second of the tree: `time -l` of a quiet render = 28.8 s user + 4.6 s sys over 32.3 s wall = 1.03
HEADROOM_MB = 3000.0        # what the compare_audio / python stages and the OS keep while renders run (one librosa compare ~1.5 GB)


@dataclass(frozen=True)
class JobsDecision:
    jobs: int
    independent: int
    cpus: int
    available_mb: float | None
    by_cpu: int
    by_memory: int | None
    source: str                 # "env" | "derived" | "unknown-machine"

    def describe(self) -> str:
        mem = "unknown" if self.available_mb is None else f"{self.available_mb / 1024:.1f} GB available"
        bym = "n/a" if self.by_memory is None else str(self.by_memory)
        if self.source == "env":
            return f"{self.jobs} (MIDIGREP_RENDER_JOBS, {self.independent} independent)"
        return (f"{self.jobs} (independent renders {self.independent}; cpu {self.cpus} cores -> {self.by_cpu}; "
                f"memory {mem} -> {bym})")


def _positive_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r}: expected a positive number") from exc
    if v <= 0:
        raise ValueError(f"{key}={raw!r}: expected a positive number")
    return v


def available_memory_mb() -> float | None:
    """Memory that can be handed to new processes without swapping: macOS free + inactive + speculative + purgeable
    pages, Linux MemAvailable. None when neither source is readable."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        m = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo.read_text(errors="ignore"), re.M)
        if m:
            return int(m.group(1)) / 1024
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    size = re.search(r"page size of (\d+) bytes", out)
    if not size:
        return None
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"^{label}:\s+(\d+)\.", out, re.M)
        if m:
            pages += int(m.group(1))
    return pages * int(size.group(1)) / (1024 * 1024) if pages else None


_READ = object()   # sentinel: "read the machine" (None means "known to be unreadable")


def derive_jobs(independent: int, *, env: dict[str, str] | None = None, cpus: int | None = None,
                available_mb: float | None | object = _READ) -> JobsDecision:
    """The concurrency for ``independent`` renders. ``cpus`` / ``available_mb`` are injectable for tests."""
    if independent < 1:
        raise ValueError(f"independent must be >= 1, got {independent}")
    env = dict(os.environ if env is None else env)
    ncpu = cpus if cpus is not None else (os.cpu_count() or 1)
    forced = env.get("MIDIGREP_RENDER_JOBS", "").strip()
    if forced:
        try:
            n = int(forced)
        except ValueError as exc:
            raise ValueError(f"MIDIGREP_RENDER_JOBS={forced!r}: expected a positive integer") from exc
        if n < 1:
            raise ValueError(f"MIDIGREP_RENDER_JOBS={forced!r}: expected a positive integer")
        return JobsDecision(min(n, independent), independent, ncpu, None, 0, None, "env")
    rss = _positive_float(env, "MIDIGREP_RENDER_RSS_MB", RENDER_RSS_MB)
    cores = _positive_float(env, "MIDIGREP_RENDER_CORES", RENDER_CORES)
    avail = available_memory_mb() if available_mb is _READ else available_mb
    by_cpu = max(1, int(ncpu // cores))
    if avail is None:
        return JobsDecision(1, independent, ncpu, None, by_cpu, None, "unknown-machine")
    by_mem = max(1, int((avail - HEADROOM_MB) // rss))
    return JobsDecision(max(1, min(independent, by_cpu, by_mem)), independent, ncpu, avail, by_cpu, by_mem, "derived")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs", help="print the concurrency for N independent renders")
    j.add_argument("--independent", type=int, required=True)
    args = ap.parse_args(argv)
    try:
        d = derive_jobs(args.independent)
    except ValueError as exc:
        print(f"render_jobs: {exc}", file=sys.stderr)
        return 2
    print(d.describe(), file=sys.stderr)
    print(d.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
