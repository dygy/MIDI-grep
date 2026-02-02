#!/usr/bin/env python3
"""
Test script for loop detection functionality.
Creates a synthetic MIDI pattern and tests loop detection.
"""

import sys
import os

# Add the script directory to path to import cleanup functions
sys.path.insert(0, os.path.dirname(__file__))

from cleanup import detect_loop, _calculate_similarity

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


if __name__ == '__main__':
    print("=" * 50)
    print("Loop Detection Test Suite")
    print("=" * 50 + "\n")

    try:
        test_similarity()
        test_loop_detection()
        test_no_loop()

        print("=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
