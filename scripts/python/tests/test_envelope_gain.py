"""Regression tests for envelope-following gain automation and the silent-render bug class.

The silent-render bug: emitting `.gain("<…>".slow(87))` calls `String.prototype.slow` (undefined) →
Strudel throws at runtime → BlackHole records FULL-LENGTH SILENCE that the mini-parser can't catch.
These tests lock in that `_env_gain` emits a bare `<…>` (which already advances one element per
cycle) and that the orchestrator's method-on-string guard strips any such chain if it reappears.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen_orchestrator import _env_gain, _drum_bar_for_density, _per_cycle_drum_patterns  # noqa: E402


def test_env_gain_emits_bare_angle_pattern():
    g = _env_gain([0.0, 0.5, 0.95, 0.2], total_cycles=4)
    assert g.startswith('.gain("<')
    assert g.endswith('>")')
    assert "0.5" in g and "0.95" in g


def test_env_gain_never_calls_slow_on_a_string():
    # The exact bug: `"<…>".slow(N)` is a method on a string → runtime crash → silent render.
    g = _env_gain([0.1, 0.2, 0.3], total_cycles=3)
    assert ".slow(" not in g
    assert not re.search(r'"\s*\.\s*(slow|fast|ply|range)\s*\(', g)


def test_env_gain_empty_is_noop():
    assert _env_gain([], total_cycles=10) == ""
    assert _env_gain(None, total_cycles=10) == ""


def test_env_gain_pattern_is_balanced():
    g = _env_gain([0.0, 0.4, 0.9, 0.1, 0.6], total_cycles=5)
    assert g.count("<") == g.count(">") == 1
    assert g.count("(") == g.count(")")


@pytest.mark.parametrize("vals", [[0.0], [0.95] * 87, [round(i / 10, 2) for i in range(10)]])
def test_env_gain_only_numbers_and_spaces_inside_brackets(vals):
    g = _env_gain(vals, total_cycles=len(vals))
    inner = g[g.index("<") + 1:g.index(">")]
    assert re.fullmatch(r"[0-9. ]+", inner), f"non-numeric content in gain pattern: {inner!r}"


# --- per-cycle drum density -------------------------------------------------

def test_drum_bar_density_monotonic_busyness():
    # Higher density → more hits (more tokens); zero density → a rest.
    assert _drum_bar_for_density(0.0) == "~"
    counts = [len(_drum_bar_for_density(d).split()) for d in (0.05, 0.2, 0.35, 0.55, 0.8)]
    assert counts == sorted(counts), f"busyness not monotonic with density: {counts}"


def test_per_cycle_drum_patterns_one_bar_per_cycle():
    sections = [{"cycles": 3}, {"cycles": 2}]
    density = [0.0, 0.5, 0.9, 0.1, 0.7]  # 5 cycles total
    pats = _per_cycle_drum_patterns(sections, density)
    assert len(pats) == 2
    assert pats[0].count("[") == 3 and pats[1].count("[") == 2   # one bar per cycle
    assert pats[0].startswith("<[") and pats[0].endswith("]>")


def test_per_cycle_drum_patterns_balanced_and_rests_for_silence():
    sections = [{"cycles": 4}]
    density = [0.0, 0.0, 0.8, 0.3]
    pats = _per_cycle_drum_patterns(sections, density)
    p = pats[0]
    assert p.count("<") == p.count(">") == 1
    assert p.count("[") == p.count("]") == 4
    assert "[~]" in p  # silent cycles render as a rest bar


def test_per_cycle_drum_patterns_short_density_falls_back():
    # Fewer density values than cycles → later cycles use a sensible default, no IndexError.
    pats = _per_cycle_drum_patterns([{"cycles": 5}], [0.6])
    assert pats[0].count("[") == 5
