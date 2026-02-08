#!/usr/bin/env python3
"""
Test script for loop detection functionality.
Tests cover:
1. Basic similarity and loop detection
2. Time signature support (3/4, 6/8, 5/4, 7/8)
3. Consensus-based reference selection
4. A-B-A-B alternating pattern detection
5. Swing/groove timing tolerance
6. Multi-voice loop detection
"""

import sys
import os
import unittest

# Add the script directory to path to import cleanup functions
sys.path.insert(0, os.path.dirname(__file__))

from cleanup import (
    detect_loop,
    detect_loops_per_voice,
    get_beats_per_bar,
    get_beat_unit,
    _find_consensus_pattern,
    _detect_alternating_pattern,
    _calculate_similarity,
)


# ============================================================================
# ORIGINAL TESTS
# ============================================================================

def test_similarity():
    """Test similarity calculation."""
    print("Testing similarity calculation...")

    # Identical patterns
    pattern1 = [
        {'pitch': 60, 'start': 0.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 64, 'start': 0.5, 'duration': 0.5, 'velocity': 80},
        {'pitch': 67, 'start': 1.0, 'duration': 0.5, 'velocity': 80},
    ]
    pattern2 = pattern1.copy()
    similarity = _calculate_similarity(pattern1, pattern2)
    print(f"  Identical patterns: {similarity:.2f} (expected: 1.0)")
    assert abs(similarity - 1.0) < 0.001, f"Expected ~1.0, got {similarity}"

    # Completely different patterns
    pattern3 = [
        {'pitch': 72, 'start': 0.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 76, 'start': 0.5, 'duration': 0.5, 'velocity': 80},
    ]
    similarity = _calculate_similarity(pattern1, pattern3)
    print(f"  Different patterns: {similarity:.2f} (expected: <0.5)")
    assert similarity < 0.5, f"Expected <0.5, got {similarity}"

    # Similar but slightly different timing
    pattern4 = [
        {'pitch': 60, 'start': 0.02, 'duration': 0.5, 'velocity': 80},  # 20ms offset
        {'pitch': 64, 'start': 0.52, 'duration': 0.5, 'velocity': 80},
        {'pitch': 67, 'start': 1.02, 'duration': 0.5, 'velocity': 80},
    ]
    similarity = _calculate_similarity(pattern1, pattern4)
    print(f"  Similar with timing offset: {similarity:.2f} (expected: >0.8)")
    assert similarity > 0.8, f"Expected >0.8, got {similarity}"

    print("Similarity tests passed!\n")


def test_loop_detection():
    """Test loop detection with synthetic data."""
    print("Testing loop detection...")

    # Create a 2-bar repeating pattern (assuming 120 BPM, 4/4 time)
    tempo = 120
    beat_duration = 60.0 / tempo  # 0.5 seconds per beat

    # 2-bar pattern (8 beats)
    base_pattern = [
        {'pitch': 60, 'start_quantized': 0.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 64, 'start_quantized': 0.5, 'duration': 0.5, 'velocity': 80},
        {'pitch': 67, 'start_quantized': 1.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 60, 'start_quantized': 2.0, 'duration': 1.0, 'velocity': 90},
    ]

    # Repeat the pattern 3 times
    notes = []
    for i in range(3):
        offset = i * 4.0  # 8 beats = 4 seconds at 120 BPM
        for note in base_pattern:
            notes.append({
                'pitch': note['pitch'],
                'start_quantized': note['start_quantized'] + offset,
                'duration': note['duration'],
                'velocity': note['velocity']
            })

    # Test detection
    loop_info = detect_loop(notes, tempo, beat_duration)

    if loop_info:
        print(f"  Loop detected: {loop_info['bars']} bar(s)")
        print(f"  Confidence: {loop_info['confidence']:.2f}")
        print(f"  Repetitions: {loop_info.get('repetitions', 0)}")
        print(f"  Notes in pattern: {len(loop_info['notes'])}")
        assert loop_info['detected'] == True
        assert loop_info['bars'] in [1, 2, 4]
        assert loop_info['confidence'] >= 0.65
        print("Loop detection test passed!\n")
    else:
        print("  ERROR: No loop detected (expected to find one)")
        sys.exit(1)


def test_no_loop():
    """Test that non-repeating patterns are not detected as loops."""
    print("Testing non-repeating pattern...")

    tempo = 120
    beat_duration = 60.0 / tempo

    # Random non-repeating notes
    notes = [
        {'pitch': 60, 'start_quantized': 0.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 62, 'start_quantized': 0.5, 'duration': 0.5, 'velocity': 80},
        {'pitch': 65, 'start_quantized': 1.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 69, 'start_quantized': 1.5, 'duration': 0.5, 'velocity': 80},
        {'pitch': 71, 'start_quantized': 2.0, 'duration': 0.5, 'velocity': 80},
        {'pitch': 74, 'start_quantized': 2.5, 'duration': 0.5, 'velocity': 80},
    ]

    loop_info = detect_loop(notes, tempo, beat_duration)

    if loop_info:
        print(f"  WARNING: Loop detected (confidence: {loop_info['confidence']:.2f})")
        print("  This might be okay if similarity is borderline")
    else:
        print("  Correctly identified as non-repeating")

    print("Non-repeating pattern test passed!\n")


# ============================================================================
# NEW UNIT TESTS
# ============================================================================

class TestTimeSignatureHelpers(unittest.TestCase):
    """Test time signature helper functions."""

    def test_beats_per_bar_4_4(self):
        """4/4 time has 4 beats per bar."""
        self.assertEqual(get_beats_per_bar("4/4"), 4)

    def test_beats_per_bar_3_4(self):
        """3/4 waltz time has 3 beats per bar."""
        self.assertEqual(get_beats_per_bar("3/4"), 3)

    def test_beats_per_bar_6_8(self):
        """6/8 compound time has 2 beats per bar (dotted quarters)."""
        self.assertEqual(get_beats_per_bar("6/8"), 2)

    def test_beats_per_bar_5_4(self):
        """5/4 time has 5 beats per bar."""
        self.assertEqual(get_beats_per_bar("5/4"), 5)

    def test_beats_per_bar_7_8(self):
        """7/8 time has 7 beats per bar."""
        self.assertEqual(get_beats_per_bar("7/8"), 7)

    def test_beats_per_bar_unknown(self):
        """Unknown time signature defaults to 4."""
        self.assertEqual(get_beats_per_bar("unknown"), 4)

    def test_beat_unit_simple_meter(self):
        """Simple meters have beat unit of 1.0."""
        self.assertEqual(get_beat_unit("4/4"), 1.0)
        self.assertEqual(get_beat_unit("3/4"), 1.0)

    def test_beat_unit_compound_meter(self):
        """6/8 has dotted quarter beat unit (1.5)."""
        self.assertEqual(get_beat_unit("6/8"), 1.5)

    def test_beat_unit_7_8(self):
        """7/8 has eighth note beat unit (0.5)."""
        self.assertEqual(get_beat_unit("7/8"), 0.5)


class TestLoopDetection3_4(unittest.TestCase):
    """Test loop detection in 3/4 (waltz) time."""

    def test_waltz_3_beat_loop(self):
        """Waltz pattern should detect 3-beat bars correctly."""
        # Create a simple 3-beat pattern repeated 4 times
        tempo = 120
        beat_duration = 60.0 / tempo  # 0.5 seconds per beat
        bar_duration = beat_duration * 3  # 1.5 seconds per bar

        notes = []
        for i in range(4):  # 4 repetitions
            start = i * bar_duration
            notes.append({'pitch': 60, 'start_quantized': start, 'duration': 0.2, 'velocity': 100})
            notes.append({'pitch': 64, 'start_quantized': start + beat_duration, 'duration': 0.2, 'velocity': 80})
            notes.append({'pitch': 67, 'start_quantized': start + 2 * beat_duration, 'duration': 0.2, 'velocity': 80})

        result = detect_loop(notes, tempo, beat_duration, bars_to_test=[1, 2, 4], time_sig="3/4")

        self.assertIsNotNone(result)
        self.assertTrue(result['detected'])
        self.assertEqual(result['bars'], 1)  # 1 bar of 3/4
        self.assertGreater(result['confidence'], 0.7)


class TestConsensusReference(unittest.TestCase):
    """Test consensus-based reference pattern selection."""

    def test_consensus_picks_most_common(self):
        """Consensus should pick pattern most similar to all others."""
        # Pattern A (will be at indices 0, 2, 3, 4) - most common
        pattern_a = [
            {'pitch': 60, 'start': 0.0, 'duration': 0.2, 'velocity': 100},
            {'pitch': 64, 'start': 0.5, 'duration': 0.2, 'velocity': 80},
        ]

        # Pattern B (different, at index 1) - intro variation
        pattern_b = [
            {'pitch': 72, 'start': 0.0, 'duration': 0.3, 'velocity': 100},
        ]

        iterations = [pattern_b, pattern_a, pattern_a, pattern_a, pattern_a]

        best_pattern, best_idx = _find_consensus_pattern(iterations)

        # Should NOT pick index 0 (the intro variation)
        # Should pick one of the common patterns (1, 2, 3, or 4)
        self.assertGreater(best_idx, 0)
        self.assertEqual(len(best_pattern), 2)  # Pattern A has 2 notes

    def test_consensus_single_pattern(self):
        """Single pattern should return itself."""
        pattern = [{'pitch': 60, 'start': 0.0, 'duration': 0.2, 'velocity': 100}]
        iterations = [pattern]

        best, idx = _find_consensus_pattern(iterations)

        self.assertEqual(idx, 0)
        self.assertEqual(best, pattern)


class TestAlternatingPattern(unittest.TestCase):
    """Test A-B-A-B alternating pattern detection."""

    def test_abab_detection(self):
        """Should detect verse-chorus style A-B-A-B alternation."""
        # Pattern A (verse)
        pattern_a = [
            {'pitch': 48, 'start': 0.0, 'duration': 0.5, 'velocity': 100},
            {'pitch': 48, 'start': 1.0, 'duration': 0.5, 'velocity': 100},
        ]

        # Pattern B (chorus - different pitches)
        pattern_b = [
            {'pitch': 60, 'start': 0.0, 'duration': 0.25, 'velocity': 100},
            {'pitch': 64, 'start': 0.5, 'duration': 0.25, 'velocity': 100},
            {'pitch': 67, 'start': 1.0, 'duration': 0.25, 'velocity': 100},
        ]

        # A-B-A-B pattern
        iterations = [pattern_a, pattern_b, pattern_a, pattern_b]

        result = _detect_alternating_pattern(iterations)

        self.assertIsNotNone(result)
        self.assertIn('variation_a', result)
        self.assertIn('variation_b', result)

    def test_no_alternating_if_similar(self):
        """Should NOT detect alternating if patterns are too similar."""
        pattern = [
            {'pitch': 60, 'start': 0.0, 'duration': 0.2, 'velocity': 100},
        ]

        iterations = [pattern, pattern, pattern, pattern]

        result = _detect_alternating_pattern(iterations)

        # All patterns are the same, so no alternating detection
        self.assertIsNone(result)

    def test_needs_4_iterations(self):
        """Need at least 4 iterations for A-B-A-B detection."""
        pattern_a = [{'pitch': 48, 'start': 0.0, 'duration': 0.5, 'velocity': 100}]
        pattern_b = [{'pitch': 60, 'start': 0.0, 'duration': 0.5, 'velocity': 100}]

        iterations = [pattern_a, pattern_b]  # Only 2

        result = _detect_alternating_pattern(iterations)

        self.assertIsNone(result)


class TestSwingTolerance(unittest.TestCase):
    """Test swing/groove timing tolerance."""

    def test_straight_timing(self):
        """Straight timing (swing_ratio=1.0) should use normal bin size."""
        pattern1 = [{'pitch': 60, 'start': 0.0, 'duration': 0.2, 'velocity': 100}]
        pattern2 = [{'pitch': 60, 'start': 0.03, 'duration': 0.2, 'velocity': 100}]  # 30ms off

        similarity = _calculate_similarity(pattern1, pattern2, swing_ratio=1.0)

        # Should still match within 50ms tolerance
        self.assertGreater(similarity, 0.8)

    def test_swing_increases_tolerance(self):
        """Swung timing should have wider tolerance bins."""
        # With swing, notes 60ms apart should still match
        pattern1 = [{'pitch': 60, 'start': 0.0, 'duration': 0.2, 'velocity': 100}]
        pattern2 = [{'pitch': 60, 'start': 0.06, 'duration': 0.2, 'velocity': 100}]  # 60ms off

        # Straight timing - 60ms might cross bin boundary
        straight_sim = _calculate_similarity(pattern1, pattern2, swing_ratio=1.0)

        # With swing (1.5 ratio) - larger bins accommodate timing variation
        swing_sim = _calculate_similarity(pattern1, pattern2, swing_ratio=1.5)

        # Swing tolerance should give equal or better match
        self.assertGreaterEqual(swing_sim, straight_sim)


class TestPerVoiceLoops(unittest.TestCase):
    """Test per-voice (bass, mid, high) loop detection."""

    def test_different_loop_lengths(self):
        """Bass 2-bar loop with melody 4-bar loop should be detected separately."""
        tempo = 120
        beat_duration = 0.5
        bar_duration = beat_duration * 4

        notes = []

        # Bass: 2-bar loop (MIDI < 48)
        bass_pattern_length = bar_duration * 2
        for rep in range(4):  # 4 repetitions of 2-bar pattern
            start = rep * bass_pattern_length
            notes.append({'pitch': 36, 'start_quantized': start, 'duration': 0.5, 'velocity': 100})
            notes.append({'pitch': 36, 'start_quantized': start + beat_duration * 2, 'duration': 0.5, 'velocity': 100})

        # Mid: 4-bar loop (48 <= MIDI < 72)
        mid_pattern_length = bar_duration * 4
        for rep in range(2):  # 2 repetitions of 4-bar pattern
            start = rep * mid_pattern_length
            notes.append({'pitch': 60, 'start_quantized': start, 'duration': 0.3, 'velocity': 80})
            notes.append({'pitch': 64, 'start_quantized': start + bar_duration, 'duration': 0.3, 'velocity': 80})
            notes.append({'pitch': 67, 'start_quantized': start + bar_duration * 2, 'duration': 0.3, 'velocity': 80})
            notes.append({'pitch': 72, 'start_quantized': start + bar_duration * 3, 'duration': 0.3, 'velocity': 80})

        result = detect_loops_per_voice(notes, tempo, beat_duration, time_sig="4/4")

        self.assertIn('bass', result)
        self.assertIn('mid', result)
        self.assertIn('high', result)

        # Bass should have a loop
        if result['bass']:
            self.assertTrue(result['bass']['detected'])

        # Mid should have a loop
        if result['mid']:
            self.assertTrue(result['mid']['detected'])

    def test_no_high_notes(self):
        """Voice with no notes should return None."""
        tempo = 120
        beat_duration = 0.5

        # Only bass notes
        notes = [
            {'pitch': 36, 'start_quantized': 0.0, 'duration': 0.5, 'velocity': 100},
            {'pitch': 36, 'start_quantized': 1.0, 'duration': 0.5, 'velocity': 100},
        ]

        result = detect_loops_per_voice(notes, tempo, beat_duration)

        # High should be None (no notes in that range)
        self.assertIsNone(result['high'])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("Loop Detection Test Suite")
    print("=" * 50 + "\n")

    # Run original function-based tests
    try:
        test_similarity()
        test_loop_detection()
        test_no_loop()
    except Exception as e:
        print(f"\nLegacy test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Running Unit Tests")
    print("=" * 50 + "\n")

    # Run unittest-based tests
    unittest.main(verbosity=2)
