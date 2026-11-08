"""mem_guard.py: a 14 GB model is never loaded into ~6 GB of free memory (2026-11-08: 16.8 of 17 GB swap in use)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mem_guard as MG  # noqa: E402


def test_waits_then_skips_with_the_numbers_when_memory_never_comes():
    clock = iter(range(0, 10_000, 5))
    with pytest.raises(MG.NotEnoughMemory, match=r"needs 14\.0 GB, 5\.8 GB available"):
        MG.wait_for(14 * 1024, what="Ollama mistral-small", wait_s=0, available=lambda: 5.8 * 1024,
                    sleep=lambda _s: next(clock))


def test_proceeds_as_soon_as_memory_is_free():
    readings = iter([5000.0, 9000.0, 15000.0])
    slept = []
    got = MG.wait_for(14 * 1024, what="x", wait_s=3600, available=lambda: next(readings), sleep=slept.append)
    assert got == 15000.0 and len(slept) == 2


def test_an_unreadable_memory_reading_never_blocks_a_run():
    assert MG.wait_for(14 * 1024, what="x", wait_s=0, available=lambda: None) == 14 * 1024


def test_a_model_ollama_already_holds_needs_no_new_memory(monkeypatch):
    monkeypatch.setattr(MG, "ollama_model_mb", lambda m, u: 14 * 1024)
    monkeypatch.setattr(MG, "loaded_mb", lambda u, m: 14 * 1024)
    monkeypatch.setattr(MG, "wait_for", lambda *a, **k: pytest.fail("must not wait"))
    MG.guard_ollama("mistral-small", "http://x")


def test_genre_text_reports_the_skip_as_unavailable_evidence(tmp_path, monkeypatch):
    import genre_text as GT

    def ask(prompt, model):
        raise MG.NotEnoughMemory("Ollama mistral-small needs 14.0 GB, 5.8 GB available")
    monkeypatch.setattr(GT, "track_info", lambda d, fetch=None: {"title": "x"})
    res = GT.classify(tmp_path, ask=ask)
    assert res["status"] == "unavailable" and "NotEnoughMemory" in res["why"]
