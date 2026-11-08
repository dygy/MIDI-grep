#!/usr/bin/env python3
"""Keep the pipeline's disk use under a hard budget, and refuse to start a run the disk cannot hold.

Why (owner, 2026-11-08): ``.cache`` grew to 42 GB (every track kept every vNNN at ~300 MB and every calibration
iteration's renders), macOS swap filled the rest, and the owner had to delete ~70 GB by hand - the venv and the model
caches included. Nothing regenerable may pile up again.

What is NEVER deleted: a track's stems (``*.wav`` beside ``metadata.json``), its metadata, ``sample_pack/``, and the
small calibration memory in ``.pipeline/`` (json / params / strudel / ``best.*``, which stage 7 warm-starts from).
Those are expensive to remake (download + Demucs + transcription) and small.

What is pruned, cheapest loss first, until ``.cache`` is under the budget:

1. ``always`` (after every run): calibration iteration renders in ``.pipeline/<mode>/`` - every ``*.wav`` / ``*.mp3``
   except ``best.wav`` (stage 7's skip key). Stage 10 already copied what the report needs into ``vNNN/iterations/``.
2. ``versions``: per track and mode, every ``vNNN`` except the PROTECTED ones - the newest, the best scoring (overall)
   and any version ``eval/datasets/reference_tracks.yaml`` pins.
3. ``slim``: in protected versions other than the newest per mode, the report's sources - ``render_<stem>.wav``,
   ``iterations/``, ``embed/`` (``report.html`` embeds them; ``render.wav`` stays so the piece can be re-scored).
4. ``caches``: ``.cache/compare-features`` and ``.cache/render-samples``, least recently used first.
5. ``views``: ``.pipeline/stems/`` (stage 3c's repaired view, rebuilt in seconds) of the least recently run tracks.

A track a run currently holds (``.pipeline/running.pid`` names a live process) is skipped entirely.

If the cache is still over budget after all five, the rest is stems of tracks - the tool lists them and exits 3; it
does not delete source material on its own.

Budget: ``MIDIGREP_CACHE_BUDGET_GB`` (default ``DEFAULT_BUDGET_GB``). Preflight additionally needs free disk >= the
largest growth a run has shown (recorded per run in ``.cache/runlogs/disk_growth.json``; ``FIRST_RUN_GROWTH_GB`` until
one is recorded) + the machine's RAM size (macOS swap files grow on the same disk, up to about the RAM size: 16.8 GB of
swap on this 24 GB machine on 2026-11-08).

Usage:
  cache_budget.py report                     sizes by category, nothing deleted
  cache_budget.py prune [--dry-run]          apply the tiers until under budget
  cache_budget.py preflight                  prune, then exit 4 when free disk cannot hold a run
  cache_budget.py record-growth <before_bytes> <after_peak_bytes>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = (ROOT / ".cache").resolve()     # resolved: in an autopilot snapshot .cache is a symlink to the main checkout
STEMS = CACHE / "stems"
GROWTH = CACHE / "runlogs" / "disk_growth.json"
REFERENCES = ROOT / "eval" / "datasets" / "reference_tracks.yaml"
GB = 1024 ** 3
# 19 tracks (2026-11-08) x (stems + pack ~0.45 GB + newest piece per mode 2 x 0.3 GB + one slimmed best per mode
# 2 x 0.12 GB) ~= 24 GB, + ~1 GB of compare/sample caches. Raise it with MIDIGREP_CACHE_BUDGET_GB as tracks are added;
# `report` prints what the kept material needs.
DEFAULT_BUDGET_GB = 25.0
FIRST_RUN_GROWTH_GB = 3.0     # 2026-11-07 batch: one track's .pipeline + 2 promoted pieces peaked at ~2-3 GB
VERSION_RE = re.compile(r"^v(\d{3,})$")
RENDER_STEM_WAVS = ("render_bass.wav", "render_drums.wav", "render_melodic.wav", "render_vocals.wav")
REPORT_SOURCES = ("iterations", "embed")
RUNNING = "running.pid"       # editable-pipeline.sh writes its PID here for the length of a run


def set_root(root: Path) -> None:
    """Point every path at ``root`` (the caller's checkout). editable-pipeline.sh passes its own ROOT: the script is
    reached through symlinks from hermetic test repos and autopilot snapshots, and ``__file__`` would name the MAIN
    checkout - a test run once measured (and could have pruned) the real cache."""
    global ROOT, CACHE, STEMS, GROWTH, REFERENCES
    ROOT = Path(root)
    CACHE = (ROOT / ".cache").resolve()
    STEMS = CACHE / "stems"
    GROWTH = CACHE / "runlogs" / "disk_growth.json"
    REFERENCES = ROOT / "eval" / "datasets" / "reference_tracks.yaml"


def size(p: Path) -> int:
    if p.is_symlink() or not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            fp = Path(dp) / f
            if not fp.is_symlink():
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
    return total


def budget_bytes(env: dict[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    raw = env.get("MIDIGREP_CACHE_BUDGET_GB")
    if raw is None:
        return int(DEFAULT_BUDGET_GB * GB)
    try:
        v = float(raw)
    except ValueError:
        raise SystemExit(f"MIDIGREP_CACHE_BUDGET_GB={raw!r} is not a number") from None
    if v <= 0:
        raise SystemExit(f"MIDIGREP_CACHE_BUDGET_GB={raw!r} must be > 0")
    return int(v * GB)


@dataclass
class Version:
    path: Path
    n: int
    mode: str
    overall: float | None


@dataclass
class Plan:
    actions: list[tuple[str, Path, int]] = field(default_factory=list)   # (tier, path, bytes)

    @property
    def freed(self) -> int:
        return sum(b for _, _, b in self.actions)


def _versions(track: Path) -> list[Version]:
    out = []
    for d in track.iterdir():
        m = VERSION_RE.match(d.name)
        if not (m and d.is_dir()):
            continue
        mode, overall = "unknown", None
        try:
            c = json.loads((d / "comparison.json").read_text())
            mode = str(c.get("generation_mode") or mode)
            overall = float(c["comparison"]["overall_similarity"])
        except (OSError, ValueError, KeyError, TypeError):
            try:
                mode = str(json.loads((d / "metadata.json").read_text()).get("generation_mode") or mode)
            except (OSError, ValueError):
                pass
        out.append(Version(d, int(m.group(1)), mode, overall))
    return sorted(out, key=lambda v: v.n)


def pinned() -> set[tuple[str, str]]:
    """(cache_key, version) pairs the reference dataset pins."""
    try:
        import yaml
        doc = yaml.safe_load(REFERENCES.read_text()) or {}
    except (OSError, ImportError):
        return set()
    return {(str(t.get("cache_key")), str(t.get("version"))) for t in doc.get("tracks") or [] if t.get("cache_key")}


def protected(track: Path, versions: list[Version], pins: set[tuple[str, str]]) -> tuple[set[Path], set[Path]]:
    """(protected versions, the newest per mode)."""
    keep, newest = set(), set()
    for mode in {v.mode for v in versions}:
        vs = [v for v in versions if v.mode == mode]
        newest.add(vs[-1].path)
        keep.add(vs[-1].path)
        scored = [v for v in vs if v.overall is not None]
        if scored:
            keep.add(max(scored, key=lambda v: v.overall).path)
    keep |= {v.path for v in versions if (track.name, v.path.name) in pins}
    return keep, newest


def active(track: Path) -> bool:
    """A pipeline run holds the track (``.pipeline/running.pid`` names a live process): never prune under it."""
    try:
        pid = int((track / ".pipeline" / RUNNING).read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def tracks() -> list[Path]:
    """Tracks no run currently holds, least recently stamped/run first."""
    if not STEMS.is_dir():
        return []
    ts = [t for t in STEMS.iterdir() if t.is_dir() and (t / "metadata.json").is_file() and not active(t)]
    return sorted(ts, key=lambda t: (t / "metadata.json").stat().st_mtime)


def plan(target: int, current: int) -> Plan:
    """The prune actions, tier by tier, stopping once ``current - freed`` <= ``target`` (tier 1 always runs)."""
    p = Plan()
    pins = pinned()

    def add(tier: str, path: Path) -> None:
        b = size(path)
        if b:
            p.actions.append((tier, path, b))

    ts = tracks()
    for t in ts:                                   # 1. iteration renders (always)
        for mode_dir in (t / ".pipeline").glob("*"):
            if mode_dir.is_dir() and mode_dir.name != "stems":
                for f in mode_dir.iterdir():
                    if f.is_file() and f.suffix in (".wav", ".mp3") and f.name != "best.wav":
                        add("always", f)
    over = lambda: current - p.freed > target      # noqa: E731
    if over():                                      # 2. unprotected versions
        for t in ts:
            vs = _versions(t)
            keep, _ = protected(t, vs, pins)
            for v in vs:
                if v.path not in keep:
                    add("versions", v.path)
    if over():                                      # 3. slim protected non-newest versions
        for t in ts:
            vs = _versions(t)
            keep, newest = protected(t, vs, pins)
            for v in vs:
                if v.path in keep and v.path not in newest and (v.path / "report.html").is_file():
                    for n in RENDER_STEM_WAVS + REPORT_SOURCES:
                        add("slim", v.path / n)
    if over():                                      # 4. global caches, least recently used first
        entries = [e for c in ("compare-features", "render-samples") for e in (CACHE / c).rglob("*") if e.is_file()]
        for e in sorted(entries, key=lambda e: e.stat().st_atime):
            if not over():
                break
            add("caches", e)
    if over():                                      # 5. stage-3c views of the least recently run tracks
        for t in ts:
            if not over():
                break
            add("views", t / ".pipeline" / "stems")
    return p


def apply(p: Plan) -> None:
    for _, path, _ in p.actions:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def report() -> dict:
    cats: dict[str, int] = {"stems+metadata+pack": 0, "versions(newest/mode)": 0, "versions(other)": 0,
                            "calibration workdirs": 0, "stage-3c views": 0}
    for t in tracks():
        vs = _versions(t)
        _, newest = protected(t, vs, pinned())
        for v in vs:
            cats["versions(newest/mode)" if v.path in newest else "versions(other)"] += size(v.path)
        cats["stage-3c views"] += size(t / ".pipeline" / "stems")
        cats["calibration workdirs"] += size(t / ".pipeline") - size(t / ".pipeline" / "stems")
        cats["stems+metadata+pack"] += sum(size(f) for f in t.iterdir()
                                           if not VERSION_RE.match(f.name) and f.name != ".pipeline")
    for c in sorted(p.name for p in CACHE.iterdir() if p.is_dir() and p.name != "stems") if CACHE.is_dir() else []:
        cats[f".cache/{c}"] = size(CACHE / c)
    return cats


def ram_bytes() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        import psutil  # Linux fallback (in requirements via other deps); fails loudly if absent
        return int(psutil.virtual_memory().total)


def run_growth() -> int:
    try:
        rows = json.loads(GROWTH.read_text())
        return max(int(r["growth"]) for r in rows) if rows else int(FIRST_RUN_GROWTH_GB * GB)
    except (OSError, ValueError, KeyError):
        return int(FIRST_RUN_GROWTH_GB * GB)


def record_growth(before: int, peak: int) -> None:
    GROWTH.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = json.loads(GROWTH.read_text())
    except (OSError, ValueError):
        rows = []
    rows.append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "before": before, "peak": peak,
                 "growth": max(0, peak - before)})
    tmp = GROWTH.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows[-50:], indent=1) + "\n")
    tmp.replace(GROWTH)


def _gb(b: int) -> str:
    return f"{b / GB:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="the checkout whose .cache is budgeted (default: this script's repo)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    pr = sub.add_parser("prune")
    pr.add_argument("--dry-run", action="store_true")
    sub.add_parser("preflight")
    rg = sub.add_parser("record-growth")
    rg.add_argument("before", type=int)
    rg.add_argument("peak", type=int)
    sub.add_parser("cache-bytes")
    a = ap.parse_args(argv)
    if a.root:
        set_root(a.root)
    if a.cmd == "cache-bytes":
        print(size(CACHE))
        return 0
    if a.cmd == "record-growth":
        record_growth(a.before, a.peak)
        return 0
    if a.cmd == "report":
        cats = report()
        for k, v in cats.items():
            print(f"{_gb(v):>9}  {k}")
        print(f"{_gb(sum(cats.values())):>9}  total (budget {_gb(budget_bytes())})")
        return 0
    target, current = budget_bytes(), size(CACHE)
    p = plan(target, current)
    by_tier: dict[str, int] = {}
    for tier, _, b in p.actions:
        by_tier[tier] = by_tier.get(tier, 0) + b
    verb = "would free" if a.cmd == "prune" and a.dry_run else "freed"
    print(f"cache {_gb(current)} / budget {_gb(target)}; {verb} {_gb(p.freed)} "
          + ", ".join(f"{t} {_gb(b)}" for t, b in by_tier.items()), file=sys.stderr)
    if not (a.cmd == "prune" and a.dry_run):
        apply(p)
    left = current - p.freed
    if left > target:
        print(f"cache budget: still {_gb(left)} > {_gb(target)} after every regenerable tier - the rest is track stems. "
              f"Remove tracks you no longer need from {STEMS} or raise MIDIGREP_CACHE_BUDGET_GB. Largest tracks:",
              file=sys.stderr)
        for t in sorted(tracks(), key=size, reverse=True)[:8]:
            print(f"  {_gb(size(t)):>9}  {t.name}", file=sys.stderr)
        if a.cmd == "prune":
            return 3
    if a.cmd == "preflight":
        free = shutil.disk_usage(ROOT).free
        need = run_growth() + ram_bytes()
        if free < need:
            print(f"disk preflight: {_gb(free)} free < {_gb(need)} needed (largest run growth {_gb(run_growth())} + "
                  f"swap headroom = RAM {_gb(ram_bytes())}). Free disk space before starting.", file=sys.stderr)
            return 4
        print(f"disk preflight: {_gb(free)} free >= {_gb(need)} needed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
