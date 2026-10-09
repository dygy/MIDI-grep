"""A1-for-drums: tests that analyze_stem_drum_pattern transcribes onsets to a per-cycle hit grid
and degrades gracefully."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sound_timbre import analyze_stem_drum_pattern  # noqa: E402

SR = 22050
VALID = {"bd", "sd", "hh", "oh", "cp", "~"}


def _click_track(path: Path, period_s: float = 0.5, secs: float = 4.0):
    """Synthetic percussion: a low-freq thump every period_s (reads as kicks)."""
    y = np.zeros(int(SR * secs), dtype=np.float32)
    for t in np.arange(0.0, secs, period_s):
        i = int(t * SR)
        n = int(0.04 * SR)
        env = np.exp(-np.linspace(0, 8, n))
        y[i:i + n] += (0.8 * np.sin(2 * np.pi * 80 * np.arange(n) / SR) * env).astype(np.float32)
    sf.write(str(path), y, SR)


def test_transcribes_onsets_to_grid(tmp_path):
    f = tmp_path / "drums.wav"
    _click_track(f)
    pats = analyze_stem_drum_pattern(str(f), [{"cycles": 4}], 8, 2.0, steps_per_cycle=16)
    assert len(pats) == 1
    toks = re.findall(r"[a-z~]+", pats[0])
    assert toks, "no tokens produced"
    assert all(t in VALID for t in toks), f"invalid drum tokens: {set(toks) - VALID}"
    assert any(t != "~" for t in toks), "expected at least one hit for a click track"


def test_one_bar_per_cycle_16_steps(tmp_path):
    f = tmp_path / "drums.wav"
    _click_track(f)
    pats = analyze_stem_drum_pattern(str(f), [{"cycles": 3}], 6, 2.0, steps_per_cycle=16)
    bars = pats[0].count("[")
    assert bars == 3
    # each bar has 16 step tokens
    first_bar = pats[0].split("] [")[0].lstrip("<[")
    assert len(first_bar.split()) == 16


def test_silence_is_all_rests(tmp_path):
    f = tmp_path / "drums.wav"
    sf.write(str(f), np.zeros(SR * 3, dtype=np.float32), SR)
    pats = analyze_stem_drum_pattern(str(f), [{"cycles": 2}], 4, 2.0, steps_per_cycle=16)
    assert pats and set(re.findall(r"[a-z~]+", pats[0])) <= {"~"}


def test_missing_file_returns_empty():
    assert analyze_stem_drum_pattern("/no/such.wav", [{"cycles": 2}], 4, 2.0) == []
