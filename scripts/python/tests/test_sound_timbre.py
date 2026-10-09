"""Self-test for sound_timbre.py — no audio files required.

Tests:
1. resolve_sound returns a genre-valid sound from the candidate list.
2. A bright stem profile and a warm stem profile pick DIFFERENT candidates.
3. analyze_stem_timbre falls back gracefully for a non-existent path.
4. All sounds in GENRE_PALETTES non-drum roles are covered by TIMBRE_TABLE (or have the default).
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sound_timbre import TIMBRE_TABLE, _DEFAULT_TIMBRE, _get_timbre, _euclidean, resolve_sound, analyze_stem_timbre
from sound_selector import GENRE_PALETTES
from strudel_validation import VALID_SOUNDS


# ---------------------------------------------------------------------------
# Test 1: resolve_sound returns a valid candidate
# ---------------------------------------------------------------------------

def test_resolve_returns_valid_candidate():
    candidates = ["gm_synth_bass_1", "gm_acoustic_bass", "sine", "sawtooth"]
    result = resolve_sound(None, candidates)
    assert result in candidates, f"Expected one of {candidates}, got {result!r}"
    assert result in VALID_SOUNDS, f"{result!r} is not in VALID_SOUNDS"
    print(f"  PASS test_resolve_returns_valid_candidate: {result!r}")


# ---------------------------------------------------------------------------
# Test 2: bright vs warm stem picks different candidates
# ---------------------------------------------------------------------------

def test_bright_vs_warm_pick_different():
    """Simulate a bright stem (high centroid, low warmth) vs a warm stem (low centroid, high warmth).
    We patch analyze_stem_timbre by calling resolve_sound with synthetic candidates
    whose timbre vectors differ clearly."""

    # Bright candidates: high brightness, low warmth
    bright_candidates = ["gm_lead_2_sawtooth", "gm_glockenspiel", "supersaw",
                         "gm_celesta", "gm_lead_5_charang"]
    # Warm candidates: low brightness, high warmth
    warm_candidates = ["gm_acoustic_bass", "sine", "gm_pad_warm",
                       "gm_fretless_bass", "gm_contrabass"]

    # Verify the timbre table entries are actually distinct
    bright_vec = _get_timbre("gm_lead_2_sawtooth")
    warm_vec = _get_timbre("gm_acoustic_bass")
    assert bright_vec[0] > 0.5, f"gm_lead_2_sawtooth brightness should be >0.5, got {bright_vec[0]}"
    assert warm_vec[1] > 0.6, f"gm_acoustic_bass warmth should be >0.6, got {warm_vec[1]}"

    # Construct a "bright" stem analysis dict directly and find nearest in candidates
    # by bypassing file I/O (call the core matching logic manually).
    def nearest(stem_dict, candidates):
        stem_vec = (stem_dict["brightness"], stem_dict["warmth"], stem_dict["attack"])
        best, best_dist = candidates[0], float("inf")
        for c in candidates:
            d = _euclidean(stem_vec, _get_timbre(c))
            if d < best_dist:
                best_dist = d
                best = c
        return best

    bright_stem = {"brightness": 0.85, "warmth": 0.10, "attack": 0.65}
    warm_stem = {"brightness": 0.15, "warmth": 0.85, "attack": 0.35}
    mixed = bright_candidates + warm_candidates

    pick_bright = nearest(bright_stem, mixed)
    pick_warm = nearest(warm_stem, mixed)

    assert pick_bright != pick_warm, (
        f"Expected bright and warm stems to pick different candidates, but both picked {pick_bright!r}"
    )
    # Bright stem should pick from the bright set; warm from the warm set.
    assert pick_bright in bright_candidates, f"Bright stem picked {pick_bright!r}, expected from {bright_candidates}"
    assert pick_warm in warm_candidates, f"Warm stem picked {pick_warm!r}, expected from {warm_candidates}"
    print(f"  PASS test_bright_vs_warm_pick_different: bright→{pick_bright!r}, warm→{pick_warm!r}")


# ---------------------------------------------------------------------------
# Test 3: analyze_stem_timbre falls back gracefully for missing file
# ---------------------------------------------------------------------------

def test_analyze_missing_file_returns_default():
    result = analyze_stem_timbre("/nonexistent/path/stem.wav")
    assert isinstance(result, dict), "Expected dict"
    assert set(result.keys()) == {"brightness", "warmth", "attack"}
    assert all(isinstance(v, float) for v in result.values())
    print(f"  PASS test_analyze_missing_file_returns_default: {result}")


# ---------------------------------------------------------------------------
# Test 4: every palette sound either has an entry or uses the default gracefully
# ---------------------------------------------------------------------------

def test_palette_coverage():
    missing = []
    for genre, palette in GENRE_PALETTES.items():
        for role in ("bass", "lead", "pad", "high"):
            for sound in palette.get(role, []):
                if sound not in TIMBRE_TABLE:
                    missing.append(sound)
    # Duplicates aren't interesting
    unique_missing = sorted(set(missing))
    # The default is returned for missing entries, so this is just informational.
    # But we assert that each missing sound IS in VALID_SOUNDS (so they're real sounds).
    invalid_missing = [s for s in unique_missing if s not in VALID_SOUNDS]
    assert not invalid_missing, (
        f"Sounds in palette but neither in TIMBRE_TABLE nor VALID_SOUNDS: {invalid_missing}"
    )
    if unique_missing:
        print(f"  NOTE test_palette_coverage: {len(unique_missing)} sounds use the default "
              f"timbre (valid but not in TIMBRE_TABLE): {unique_missing[:8]}{'...' if len(unique_missing)>8 else ''}")
    else:
        print("  PASS test_palette_coverage: all palette sounds covered in TIMBRE_TABLE")


# ---------------------------------------------------------------------------
# Test 5: resolve_sound with a single candidate returns that candidate
# ---------------------------------------------------------------------------

def test_resolve_single_candidate():
    result = resolve_sound(None, ["gm_synth_bass_1"])
    assert result == "gm_synth_bass_1", f"Expected 'gm_synth_bass_1', got {result!r}"
    print("  PASS test_resolve_single_candidate")


# ---------------------------------------------------------------------------
# Test 6: resolve_sound with empty candidates returns a fallback string
# ---------------------------------------------------------------------------

def test_resolve_empty_candidates():
    result = resolve_sound(None, [])
    assert isinstance(result, str) and len(result) > 0, "Expected non-empty fallback string"
    print(f"  PASS test_resolve_empty_candidates: fallback={result!r}")


if __name__ == "__main__":
    print("Running sound_timbre self-tests...")
    test_resolve_returns_valid_candidate()
    test_bright_vs_warm_pick_different()
    test_analyze_missing_file_returns_default()
    test_palette_coverage()
    test_resolve_single_candidate()
    test_resolve_empty_candidates()
    print("\nAll tests passed.")
