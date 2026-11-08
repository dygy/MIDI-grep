#!/usr/bin/env python3
"""Environment doctor: everything a pipeline run needs, checked in seconds - and repaired where it can be.

Why (2026-11-08): the venv was deleted while a batch ran; VAGABUNDO's report and every later track failed one by one
("No such file or directory: .venv/bin/python") and nobody noticed for an hour. The rebuilt venv then could not run
stage 1b (an unpinned torch 2.15 had no torchvision). The autopilot runs this before a batch and between tracks: a
broken environment is repaired (``--fix``) or the batch stops with the reason, instead of failing every track.

Checks (each ``ok`` / ``fixed`` / ``FAIL`` with the reason; ``warn`` = optional parts):
  venv         scripts/python/.venv exists and imports the pipeline's stack (librosa, demucs, basic_pitch, torch,
               torchvision, laion_clap, soundfile, yaml); fix: create it and install requirements.txt
  node         scripts/node/node_modules + dist/render-strudel-offline.js; fix: npm install / npm run build
  browser      the renderer's Chromium (renderer.py browser; ~/.cache/puppeteer) - deleted with the caches on
               2026-11-08, when "node ok" alone would have let every render fail; fix: npx puppeteer browsers install
  ffmpeg       on PATH (yt-dlp / audio conversion)
  clickhouse   bin/clickhouse answers on .clickhouse/db (calibration memory: warm starts, learned steps). Without it
               the store returns nothing and every calibration starts cold - silently worse (2026-11-08: bin/ was
               deleted with the caches); fix: the official installer (curl https://clickhouse.com/ | sh) into bin/
  midi-grep    warn: bin/midi-grep (stage 1 of a NEW track only); fix: go build
  disk         cache_budget.py preflight (prunes, then needs the largest run growth + swap headroom)
  memory       available RAM >= one render's measured footprint + the python headroom (render_jobs.py)
  ollama       warn: the server answers and mistral-small is pulled (stage 1b metadata genre, visuals)

Usage: doctor.py [--fix] [--json]   exit 0 = runnable, 1 = not runnable (the FAIL lines say why)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "scripts" / "python"
NODE = ROOT / "scripts" / "node"
VENV_PY = SP / ".venv" / "bin" / "python"
IMPORTS = ("librosa", "demucs", "basic_pitch", "torch", "torchvision", "laion_clap", "soundfile", "yaml")
OLLAMA = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "mistral-small"


def _run(cmd: list[str], cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def check_venv(fix: bool) -> tuple[str, str]:
    probe = f"import {', '.join(IMPORTS)}"
    if VENV_PY.exists():
        p = _run([str(VENV_PY), "-c", probe], timeout=300)
        if p.returncode == 0:
            return "ok", "imports " + ", ".join(IMPORTS)
        why = (p.stderr.strip().splitlines() or ["import failed"])[-1]
    else:
        why = f"{VENV_PY} missing"
    if not fix:
        return "FAIL", why
    py311 = shutil.which("python3.11") or "/opt/homebrew/bin/python3.11"
    if not VENV_PY.exists():
        p = _run([py311, "-m", "venv", str(SP / ".venv")])
        if p.returncode != 0:
            return "FAIL", f"{why}; venv creation failed: {p.stderr.strip()[-200:]}"
    p = _run([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(SP / "requirements.txt")], timeout=7200)
    if p.returncode != 0:
        return "FAIL", f"{why}; pip install failed: {p.stderr.strip()[-300:]}"
    p = _run([str(VENV_PY), "-c", probe], timeout=300)
    return ("fixed", f"{why} -> reinstalled from requirements.txt") if p.returncode == 0 else \
        ("FAIL", f"still broken after reinstall: {(p.stderr.strip().splitlines() or ['?'])[-1]}")


def check_node(fix: bool) -> tuple[str, str]:
    dist = NODE / "dist" / "render-strudel-offline.js"
    missing = [str(p.relative_to(ROOT)) for p in (NODE / "node_modules", dist) if not p.exists()]
    if not missing:
        return "ok", "node_modules + dist built"
    if not fix:
        return "FAIL", "missing " + ", ".join(missing)
    for cmd in (["npm", "install"], ["npm", "run", "build"]):
        p = _run(cmd, cwd=NODE, timeout=1800)
        if p.returncode != 0:
            return "FAIL", f"{' '.join(cmd)} failed: {p.stderr.strip()[-200:]}"
    return ("fixed", "npm install + build") if dist.exists() else ("FAIL", "build produced no dist")


def check_browser(fix: bool) -> tuple[str, str]:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    probe = lambda: _run([py, str(SP / "renderer.py"), "browser"], timeout=120)   # noqa: E731
    p = probe()
    if p.returncode == 0:
        return "ok", p.stdout.strip()[-120:]
    why = (p.stderr.strip().splitlines() or ["renderer browser missing"])[-1]
    if not fix:
        return "FAIL", why
    i = _run(["npx", "puppeteer", "browsers", "install", "chrome"], cwd=NODE, timeout=1800)
    p = probe()
    return ("fixed", f"{why} -> npx puppeteer browsers install chrome") if p.returncode == 0 else \
        ("FAIL", f"{why}; install: {(i.stderr or i.stdout).strip()[-200:]}")


def check_ffmpeg(_fix: bool) -> tuple[str, str]:
    return ("ok", shutil.which("ffmpeg")) if shutil.which("ffmpeg") else ("FAIL", "ffmpeg not on PATH (brew install ffmpeg)")


def check_clickhouse(fix: bool) -> tuple[str, str]:
    binp = ROOT / "bin" / "clickhouse"
    db = ROOT / ".clickhouse" / "db"

    def works() -> tuple[bool, str]:
        if not binp.exists():
            return False, f"{binp.relative_to(ROOT)} missing"
        p = _run([str(binp), "local", "--path", str(db), "--query", "SELECT 1"], timeout=120)
        return p.returncode == 0, (p.stderr.strip()[-200:] or "query failed")
    ok, why = works()
    if ok:
        return "ok", f"answers on {db.relative_to(ROOT)}"
    if not fix:
        return "FAIL", why + " - calibration memory would return nothing (cold starts)"
    binp.parent.mkdir(exist_ok=True)
    p = subprocess.run("curl -sSfL https://clickhouse.com/ | sh", shell=True, cwd=binp.parent, capture_output=True,
                       text=True, timeout=1800)
    ok, why2 = works()
    return ("fixed", f"{why} -> installed bin/clickhouse") if ok else ("FAIL", f"{why}; install failed: "
                                                                        f"{(p.stderr or why2).strip()[-200:]}")


def check_midigrep(fix: bool) -> tuple[str, str]:
    binp = ROOT / "bin" / "midi-grep"
    if binp.exists():
        return "ok", str(binp.relative_to(ROOT))
    if not fix:
        return "warn", "bin/midi-grep missing - only stage 1 of a NEW track needs it (go build -o bin/midi-grep ./cmd/midi-grep)"
    p = _run(["go", "build", "-o", str(binp), "./cmd/midi-grep"], cwd=ROOT, timeout=1800)
    return ("fixed", "go build") if p.returncode == 0 else ("warn", f"go build failed: {p.stderr.strip()[-200:]}")


def check_disk(_fix: bool) -> tuple[str, str]:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    p = _run([py, str(SP / "cache_budget.py"), "preflight"], timeout=1800)
    msg = " | ".join(l for l in p.stderr.strip().splitlines() if l.strip())[-400:]
    return ("ok" if p.returncode == 0 else "FAIL"), msg


def check_memory(_fix: bool) -> tuple[str, str]:
    import render_jobs as RJ
    avail = RJ.available_memory_mb()
    need = RJ.RENDER_RSS_MB + RJ.HEADROOM_MB
    if avail is None:
        return "warn", "cannot read available memory"
    return ("ok" if avail >= need else "FAIL"), f"{avail / 1024:.1f} GB available, a render needs {need / 1024:.1f} GB"


def check_ollama(_fix: bool) -> tuple[str, str]:
    try:
        with urllib.request.urlopen(OLLAMA, timeout=5) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except OSError as exc:
        return "warn", f"ollama not answering ({exc}); stage 1b runs without the metadata genre verdict"
    have = any(n.split(":")[0] == OLLAMA_MODEL for n in names)
    return ("ok", f"{OLLAMA_MODEL} pulled") if have else ("warn", f"{OLLAMA_MODEL} not pulled (ollama pull {OLLAMA_MODEL})")


CHECKS = {"venv": check_venv, "node": check_node, "browser": check_browser, "ffmpeg": check_ffmpeg, "clickhouse": check_clickhouse,
          "midi-grep": check_midigrep, "disk": check_disk,
          "memory": check_memory, "ollama": check_ollama}


def examine(fix: bool = False) -> dict[str, tuple[str, str]]:
    sys.path.insert(0, str(SP))
    out = {}
    for name, fn in CHECKS.items():
        try:
            out[name] = fn(fix)
        except Exception as exc:  # noqa: BLE001 - a check that crashes is a FAIL with its reason, never a silent pass
            out[name] = ("FAIL", f"{type(exc).__name__}: {exc}")
    return out


def runnable(res: dict[str, tuple[str, str]]) -> bool:
    return all(v[0] != "FAIL" for v in res.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = examine(a.fix)
    if a.json:
        print(json.dumps({k: {"status": s, "detail": d} for k, (s, d) in res.items()}, indent=1))
    else:
        for k, (s, d) in res.items():
            print(f"{s:5} {k:10} {d}")
    return 0 if runnable(res) else 1


if __name__ == "__main__":
    sys.exit(main())
