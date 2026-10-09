"""Self-test for Task E — per-section timbre-driven filter targets.

Proves:
1. analyze_stem_timbre_by_section returns one brightness per section.
2. Synthetic bright-late / dark-early stem yields rising per-section LPF targets.
3. assembled code still validates + has 3 voices.
4. Missing stem / zero cps / empty sections → [] (graceful degradation).
"""
from __future__ import annotations

import sys
import os

# Make scripts/python importable without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tempfile
import wave
import struct
import numpy as np

from sound_timbre import analyze_stem_timbre_by_section
from codegen_orchestrator import _brightness_to_lpf as co_brightness_to_lpf, assemble, _lpf_mod
from strudel_validation import validate_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: str, samples: list[float], sr: int = 22050) -> None:
    """Write a mono 16-bit WAV file from a list of float samples in [-1, 1]."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        raw = struct.pack(f"<{len(samples)}h", *[int(s * 32767) for s in samples])
        wf.writeframes(raw)


def _sine_tone(freq: float, duration: float, sr: int = 22050) -> list[float]:
    """Generate a pure sine tone at the given frequency."""
    n = int(duration * sr)
    return [math.sin(2 * math.pi * freq * t / sr) for t in range(n)]


def _make_bright_late_stem(sr: int = 22050) -> str:
    """Synthetic stem: first 2 s is a low-frequency (dark) tone, next 2 s is a high-frequency
    (bright) tone — so the brightness ARC rises over time."""
    dark   = _sine_tone(200.0,  2.0, sr)   # 200 Hz — very dark
    bright = _sine_tone(5000.0, 2.0, sr)   # 5 kHz — very bright
    samples = dark + bright
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    _write_wav(path, samples, sr)
    return path


# ---------------------------------------------------------------------------
# Test: guard cases return empty list
# ---------------------------------------------------------------------------

def test_guard_missing_file():
    sections = [{"cycles": 4}, {"cycles": 4}]
    result = analyze_stem_timbre_by_section("/nonexistent/file.wav", sections, 8, 1.0)
    assert result == [], f"Expected [] for missing file, got {result}"
    print("  PASS: missing file → []")


def test_guard_empty_sections():
    result = analyze_stem_timbre_by_section("/dev/null", [], 0, 1.0)
    assert result == [], f"Expected [] for empty sections, got {result}"
    print("  PASS: empty sections → []")


def test_guard_zero_cps():
    sections = [{"cycles": 4}]
    result = analyze_stem_timbre_by_section("/dev/null", sections, 4, 0.0)
    assert result == [], f"Expected [] for zero cps, got {result}"
    print("  PASS: zero cps → []")


# ---------------------------------------------------------------------------
# Test: bright-late stem yields rising LPF targets
# ---------------------------------------------------------------------------

def test_brightness_arc_rises():
    """Dark section first, bright section last → LPF targets should rise."""
    path = _make_bright_late_stem()
    try:
        sections = [
            {"name": "intro", "cycles": 2},   # covers the dark 200 Hz segment
            {"name": "drop",  "cycles": 2},   # covers the bright 5 kHz segment
        ]
        # cps = 1 cycle/s so each 2-cycle section is 2 s — matches stem segments exactly.
        cps = 1.0
        total_cycles = 4

        brightnesses = analyze_stem_timbre_by_section(path, sections, total_cycles, cps)
        assert len(brightnesses) == 2, f"Expected 2 brightness values, got {len(brightnesses)}"
        b_dark, b_bright = brightnesses
        print(f"  brightness: dark={b_dark:.3f}, bright={b_bright:.3f}")
        assert b_bright > b_dark, (
            f"Expected bright section brighter than dark section, "
            f"got dark={b_dark:.3f} bright={b_bright:.3f}"
        )

        # Map to lead LPF range (1500–9000).
        lpfs = [co_brightness_to_lpf(b, 1500, 9000) for b in brightnesses]
        print(f"  lead lpf targets: {lpfs}")
        assert lpfs[1] > lpfs[0], (
            f"Expected rising LPF targets, got {lpfs}"
        )
        # Values must stay within the declared range.
        for lpf in lpfs:
            assert 1500 <= lpf <= 9000, f"LPF {lpf} out of range [1500, 9000]"

        print("  PASS: bright-late stem yields rising LPF targets")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test: _brightness_to_lpf clamps and scales correctly
# ---------------------------------------------------------------------------

def test_brightness_to_lpf_bounds():
    assert co_brightness_to_lpf(0.0, 200, 1500) == 200,  "brightness=0 should give lo"
    assert co_brightness_to_lpf(1.0, 200, 1500) == 1500, "brightness=1 should give hi"
    mid = co_brightness_to_lpf(0.5, 200, 1500)
    assert 200 < mid < 1500, f"brightness=0.5 should be mid-range, got {mid}"
    # Clamp beyond [0, 1].
    assert co_brightness_to_lpf(-0.5, 200, 1500) == 200
    assert co_brightness_to_lpf(2.0, 200, 1500) == 1500
    print("  PASS: _brightness_to_lpf clamps and scales correctly")


# ---------------------------------------------------------------------------
# Test: assemble() with per-section LPFs produces valid 3-voice code
# ---------------------------------------------------------------------------

def _minimal_results():
    """Minimal JobResult-like dicts for the assembler."""
    from job_runner import JobResult, JobStatus

    structure_out = {"sections": [
        {"name": "intro", "cycles": 2},
        {"name": "drop",  "cycles": 4},
        {"name": "outro", "cycles": 2},
    ]}
    bass_out = {
        "sound": "gm_synth_bass_1",
        "patterns": ["c2 ~ ~ ~", "c2 e2 g2 e2", "c2 ~ ~ ~"],
        "gains": [0.3, 0.8, 0.4],
        "lpf": 800,
    }
    lead_out = {
        "sound": "gm_lead_2_sawtooth",
        "patterns": ["c4 ~ ~ ~", "c4 e4 g4 e4", "c4 ~ ~ ~"],
        "gains": [0.3, 0.8, 0.4],
        "lpf": 4000,
    }
    drums_out = {
        "bank": "RolandTR808",
        "patterns": ["bd ~ ~ ~", "bd hh sd hh", "bd ~ ~ ~"],
        "gains": [0.3, 0.8, 0.4],
    }
    return {
        "structure": JobResult("structure", JobStatus.OK, structure_out, 1),
        "voice.bass": JobResult("voice.bass", JobStatus.OK, bass_out, 1),
        "voice.lead": JobResult("voice.lead", JobStatus.OK, lead_out, 1),
        "drums":      JobResult("drums",      JobStatus.OK, drums_out, 1),
    }


def test_assemble_with_section_lpfs_valid():
    ctx = {"bpm": 120.0, "key": "C major", "genre": "electronic"}
    results = _minimal_results()

    # Simulate rising brightness: dark intro (200), bright drop (1200), medium outro (700).
    bass_lpfs = [200, 1200, 700]
    lead_lpfs = [2000, 8000, 4000]

    code = assemble(ctx, results, bass_section_lpfs=bass_lpfs, lead_section_lpfs=lead_lpfs)

    # 3 voices must be present.
    dollar_blocks = [ln for ln in code.splitlines() if ln.strip().startswith("$:")]
    assert len(dollar_blocks) == 3, f"Expected 3 $: blocks, got {len(dollar_blocks)}: {dollar_blocks}"

    # setcps must be present.
    assert "setcps(" in code, "Expected setcps() in assembled code"

    # Validate the code.
    _, err = validate_code(code, autocorrect=False)
    assert err == "", f"Validation failed: {err}\n\nCode:\n{code}"

    # The distinct lpf sweep values for intro (200 Hz → _lpf_mod(200, slow=16)) should appear
    # AND a sweep for drop (1200 Hz) — they must differ.
    intro_sweep = _lpf_mod(200, slow=16)
    drop_sweep  = _lpf_mod(1200, slow=16)
    assert intro_sweep in code, f"Expected bass intro sweep '{intro_sweep}' in code"
    assert drop_sweep  in code, f"Expected bass drop sweep '{drop_sweep}' in code"
    assert intro_sweep != drop_sweep, "Intro and drop sweeps should differ"

    print("  PASS: assembled code validates with 3 voices and per-section lpf sweeps")


def test_assemble_fallback_no_section_lpfs():
    """assemble() without section LPFs should still produce valid code (static lpf path)."""
    ctx = {"bpm": 120.0, "key": "C major", "genre": "electronic"}
    results = _minimal_results()
    code = assemble(ctx, results)
    _, err = validate_code(code, autocorrect=False)
    assert err == "", f"Fallback assemble validation failed: {err}"
    dollar_blocks = [ln for ln in code.splitlines() if ln.strip().startswith("$:")]
    assert len(dollar_blocks) == 3, f"Expected 3 $: blocks, got {len(dollar_blocks)}"
    print("  PASS: assemble() without section LPFs still produces valid 3-voice code")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Task E: per-section timbre-driven filter targets ===\n")

    print("Guard cases:")
    test_guard_missing_file()
    test_guard_empty_sections()
    test_guard_zero_cps()

    print("\n_brightness_to_lpf:")
    test_brightness_to_lpf_bounds()

    print("\nBrightness arc (requires librosa):")
    try:
        test_brightness_arc_rises()
    except ImportError as e:
        print(f"  SKIP: librosa not available ({e})")

    print("\nassemble() integration:")
    test_assemble_with_section_lpfs_valid()
    test_assemble_fallback_no_section_lpfs()

    print("\nAll tests passed.")
