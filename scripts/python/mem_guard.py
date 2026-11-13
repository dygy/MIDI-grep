"""RAM guard for the heavy optional stages: load a model only when the memory for it is actually free.

Why (2026-11-08): mistral-small (~14 GB, Ollama) was loaded for stage 1b / stage 11 with ~6 GB available on the 24 GB
machine (Chromium renders, three IDEs, Docker); macOS pushed the rest into swap - 16.8 of 17 GB swap in use, the swap
files ate the disk, and every process crawled. A stage that needs a model waits a bounded time for the memory (renders
finishing free it) and is skipped with the reason when it does not come - the run continues without that stage's
evidence, which every caller already handles (genre_text: no metadata verdict; visuals: the run does not fail).

The model's footprint is Ollama's own size of the pulled model (``/api/tags``), not a number written here; available
memory is ``render_jobs.available_memory_mb`` (free + inactive + speculative + purgeable pages).
"""
from __future__ import annotations

import os
import time

import requests

DEFAULT_WAIT_S = 600.0     # MIDIGREP_MEM_WAIT_S: how long a stage waits for the memory before it is skipped
POLL_S = 5.0


class NotEnoughMemory(RuntimeError):
    pass


def ollama_model_mb(model: str, url: str) -> float | None:
    """The pulled model's size in MB (Ollama /api/tags), None when Ollama does not list it."""
    r = requests.get(f"{url}/api/tags", timeout=10)
    r.raise_for_status()
    base = model.split(":")[0]
    for m in r.json().get("models", []):
        name = m.get("name", "")
        if name == model or name.split(":")[0] == base:
            return float(m.get("size", 0)) / (1024 * 1024)
    return None


def loaded_mb(url: str, model: str) -> float:
    """MB of ``model`` Ollama already holds in memory (a loaded model needs no new memory)."""
    try:
        r = requests.get(f"{url}/api/ps", timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return 0.0
    base = model.split(":")[0]
    return sum(float(m.get("size", 0)) / (1024 * 1024) for m in r.json().get("models", [])
               if m.get("name", "").split(":")[0] == base)


def wait_for(need_mb: float, *, what: str, wait_s: float | None = None, available=None, sleep=time.sleep) -> float:
    """Block until ``need_mb`` is available (returns it) or raise NotEnoughMemory after ``wait_s``."""
    import render_jobs as RJ
    available = available or RJ.available_memory_mb
    wait_s = float(os.environ.get("MIDIGREP_MEM_WAIT_S", DEFAULT_WAIT_S)) if wait_s is None else wait_s
    deadline = time.monotonic() + wait_s
    while True:
        avail = available()
        if avail is None or avail >= need_mb:          # unreadable memory: do not block a run on a missing reading
            return avail if avail is not None else need_mb
        if time.monotonic() >= deadline:
            raise NotEnoughMemory(f"{what} needs {need_mb / 1024:.1f} GB, {avail / 1024:.1f} GB available after "
                                  f"waiting {wait_s:.0f} s - skipped so the machine does not swap")
        sleep(POLL_S)


def guard_ollama(model: str, url: str, **kw) -> None:
    """Raise NotEnoughMemory unless the model fits in the memory available (or Ollama already holds it)."""
    try:
        size = ollama_model_mb(model, url)
    except requests.RequestException:
        return                                          # Ollama down: the call itself fails with its own error
    if not size or loaded_mb(url, model) >= size:
        return
    wait_for(size, what=f"Ollama {model}", **kw)
