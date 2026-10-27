# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""A sampled lead/chord voice is high-passed at the lowest fundamental it plays, not at a fixed 300 Hz.

Regression (2026-10-23, vagabundo-melodic-mel): `--lead-hpf` defaulted to 300 for sample-instrument mode too. VAGABUNDO's
lead/chords reach cs3 (139 Hz); the melodic voice rendered alone matched the melodic stem's per-frame log-mel shape at
0.638 with hpf 300, 0.680 with hpf = lowest fundamental, 0.686 with the filter off.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_sample_lead_hpf.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_dynamic_strudel as G  # noqa: E402


def test_cutoff_is_the_lowest_fundamental_of_lead_and_chords():
    lead = ["cs4 ~ e4@2", "~ ~ cs3@7"]
    chords = ["[e3,gs3] ~", "a2@4"]
    assert G.sample_lead_hpf(lead, chords) == 110          # a2 = 110 Hz, below the lead's cs3 (138.6 Hz)


def test_legato_bars_count():
    assert G.sample_lead_hpf(["e4"], None, ["c3 ~"], None) == 130


def test_no_notes_gives_zero():
    assert G.sample_lead_hpf(None, ["~ ~ ~"]) == 0


def test_low_lead_is_not_cut_above_its_fundamental():
    hz = G.sample_lead_hpf(["cs3@7"])
    assert 0 < hz <= 138.6          # the fixed 300 would have removed this note's fundamental
