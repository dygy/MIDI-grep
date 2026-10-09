"""A3 infrastructure: resolve_drum_bank picks the timbre-nearest cached bank, falls back gracefully.

(The auto-swap is DISABLED in the orchestrator — it measured worse — but the resolver/audition stay
as infrastructure for genres whose original is machine-based, so the selection logic is tested here.)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sound_timbre import resolve_drum_bank, analyze_stem_timbre  # noqa: E402

SR = 22050


def _tone(path: Path, freq=110.0, secs=3.0):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    sf.write(str(path), (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)


def test_picks_nearest_cached_bank(tmp_path):
    f = tmp_path / "drums.wav"
    _tone(f)
    t = analyze_stem_timbre(str(f))  # the target vector
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "near": {k: t[k] for k in ("brightness", "warmth", "attack")},      # exact match
        "far": {"brightness": 1.0, "warmth": 0.0, "attack": 1.0},            # opposite corner
    }))
    assert resolve_drum_bank(str(f), ["near", "far"], str(cache)) == "near"


def test_fallback_when_no_cache(tmp_path):
    f = tmp_path / "drums.wav"
    _tone(f)
    assert resolve_drum_bank(str(f), ["RolandTR808", "LinnDrum"], str(tmp_path / "missing.json")) == "RolandTR808"


def test_fallback_when_candidate_uncached(tmp_path):
    f = tmp_path / "drums.wav"
    _tone(f)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"other": {"brightness": 0.1, "warmth": 0.1, "attack": 0.1}}))
    # neither candidate is in the cache → returns candidates[0]
    assert resolve_drum_bank(str(f), ["RolandTR909", "AkaiMPC60"], str(cache)) == "RolandTR909"


def test_empty_candidates_safe():
    assert resolve_drum_bank("/nope.wav", [], "/nope.json") == "RolandTR808"
