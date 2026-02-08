#!/usr/bin/env python3
"""
MIDI cleanup and quantization using pretty_midi.
Outputs cleaned notes as JSON for Go to consume.
Separates into bass/mid/high voices for richer Strudel output.
"""

import sys
import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ============================================================================
# TIME SIGNATURE HELPERS
# ============================================================================

def get_beats_per_bar(time_sig: str) -> int:
    """Return number of beats per bar for given time signature."""
    return {
        "4/4": 4,
        "3/4": 3,
        "2/4": 2,
        "6/8": 2,  # Compound meter: 2 dotted quarter notes
        "5/4": 5,
        "7/8": 7,
    }.get(time_sig, 4)


def get_beat_unit(time_sig: str) -> float:
    """Return beat unit multiplier for compound meters.

    For simple meters (4/4, 3/4, etc.) returns 1.0.
    For compound meters (6/8, 7/8) returns the dotted note value.
    """
    return {
        "6/8": 1.5,  # Dotted quarter = 1.5 eighth notes
        "7/8": 0.5,  # Eighth note as beat unit
    }.get(time_sig, 1.0)


# ============================================================================
# LOOP DETECTION
# ============================================================================

def detect_loop(notes: List[Dict], tempo: float, beat_duration: float,
                bars_to_test: List[int] = [1, 2, 4, 8],
                time_sig: str = "4/4",
                swing_ratio: float = 1.0) -> Optional[Dict]:
    """
    Detect repeating patterns in MIDI notes.

    Args:
        notes: List of note dictionaries with 'pitch' and 'start_quantized' keys
        tempo: Tempo in BPM
        beat_duration: Duration of one beat in seconds
        bars_to_test: List of bar lengths to test (default: [1, 2, 4, 8])
        time_sig: Time signature string (default: "4/4")
        swing_ratio: Swing timing ratio (1.0=straight, 1.5-2.0=swing)

    Returns:
        Dictionary with loop information or None if no loop detected
    """
    if not notes or len(notes) < 4:
        return None

    # Calculate total length in beats
    max_time = max(n['start_quantized'] + n.get('duration', 0) for n in notes)
    total_beats = max_time / beat_duration

    # Minimum confidence threshold - lowered for tracks with variation
    MIN_CONFIDENCE = 0.45
    best_loop = None
    best_score = 0

    # Use time signature for beats per bar
    beats_per_bar = get_beats_per_bar(time_sig)
    beat_unit = get_beat_unit(time_sig)

    for bar_count in bars_to_test:
        loop_length_beats = bar_count * beats_per_bar * beat_unit
        loop_length_seconds = loop_length_beats * beat_duration

        # Need at least 2 repetitions to detect a loop
        if total_beats < loop_length_beats * 2:
            continue

        # Extract chunks and compare
        loop_info = _find_repeating_pattern(
            notes,
            loop_length_seconds,
            loop_length_beats,
            beat_duration,
            swing_ratio
        )

        if loop_info and loop_info['confidence'] > best_score:
            best_score = loop_info['confidence']
            best_loop = loop_info
            best_loop['bars'] = bar_count

    if best_loop and best_score >= MIN_CONFIDENCE:
        return best_loop

    return None


def _find_repeating_pattern(notes: List[Dict], loop_length_seconds: float,
                           loop_length_beats: float, beat_duration: float,
                           swing_ratio: float = 1.0) -> Optional[Dict]:
    """
    Find repeating pattern of given length.

    Args:
        notes: List of note dictionaries
        loop_length_seconds: Length of pattern to test in seconds
        loop_length_beats: Length of pattern in beats
        beat_duration: Duration of one beat in seconds
        swing_ratio: Swing timing ratio for tolerance adjustment

    Returns:
        Dictionary with loop info and confidence score or None
    """
    max_time = max(n['start_quantized'] + n.get('duration', 0) for n in notes)
    num_loops = int(max_time / loop_length_seconds)

    if num_loops < 2:
        return None

    # Extract notes for each loop iteration
    loop_iterations = []
    for i in range(num_loops):
        start_time = i * loop_length_seconds
        end_time = (i + 1) * loop_length_seconds

        chunk_notes = [
            n for n in notes
            if start_time <= n['start_quantized'] < end_time
        ]

        # Normalize to loop-relative time
        normalized = []
        for note in chunk_notes:
            normalized.append({
                'pitch': note['pitch'],
                'start': note['start_quantized'] - start_time,
                'duration': note.get('duration', 0),
                'velocity': note['velocity']
            })

        loop_iterations.append(normalized)

    # Find most common pattern by comparing all iterations
    if not loop_iterations:
        return None

    # Use consensus-based reference selection instead of always using first iteration
    reference, reference_idx = _find_consensus_pattern(loop_iterations, swing_ratio)

    # Check for A-B-A-B alternating pattern
    alternating = _detect_alternating_pattern(loop_iterations, swing_ratio)

    # Calculate similarities against consensus reference
    similarities = []
    for i in range(len(loop_iterations)):
        if i != reference_idx:
            similarity = _calculate_similarity(reference, loop_iterations[i], swing_ratio)
            similarities.append(similarity)

    if not similarities:
        return None

    # Average similarity is our confidence
    avg_similarity = sum(similarities) / len(similarities)

    # Find the most common pattern (use consensus reference if similarity is high)
    if avg_similarity >= 0.45:
        result = {
            'detected': True,
            'confidence': round(avg_similarity, 2),
            'start_beat': 0,
            'end_beat': round(loop_length_beats, 2),
            'notes': reference,
            'repetitions': num_loops,
            'reference_iteration': reference_idx,
            'pattern_type': 'single'
        }

        # Add alternating pattern info if detected
        if alternating:
            result['pattern_type'] = 'alternating'
            result['variation_a'] = alternating['variation_a']
            result['variation_b'] = alternating['variation_b']

        return result

    return None


def _find_consensus_pattern(iterations: List[List[Dict]], swing_ratio: float = 1.0) -> Tuple[List[Dict], int]:
    """
    Find the iteration that is most similar to all others (consensus pattern).

    Instead of always using the first iteration as reference (which can fail on intro
    variations), we build an NxN similarity matrix and pick the iteration with the
    highest total similarity to all others.

    Args:
        iterations: List of loop iterations (each is a list of notes)
        swing_ratio: Swing timing ratio for tolerance adjustment

    Returns:
        Tuple of (best pattern, best index)
    """
    n = len(iterations)

    if n == 0:
        return [], 0
    if n == 1:
        return iterations[0], 0

    # Build NxN similarity matrix
    sim_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)  # Perfect similarity with self
            else:
                row.append(_calculate_similarity(iterations[i], iterations[j], swing_ratio))
        sim_matrix.append(row)

    # Sum similarities for each iteration (total similarity to all others)
    totals = [sum(row) for row in sim_matrix]

    # Pick iteration with highest total similarity
    best_idx = max(range(n), key=lambda i: totals[i])

    return iterations[best_idx], best_idx


def _detect_alternating_pattern(iterations: List[List[Dict]], swing_ratio: float = 1.0) -> Optional[Dict]:
    """
    Detect A-B-A-B alternating patterns (like verse-chorus alternation).

    Algorithm:
    1. Group iterations: odd (0,2,4...) = A, even (1,3,5...) = B
    2. Check A-group internal similarity (>= 0.6)
    3. Check B-group internal similarity (>= 0.6)
    4. Check A-B cross-similarity (< 0.5 = different patterns)
    5. If all criteria met, return both patterns

    Args:
        iterations: List of loop iterations
        swing_ratio: Swing timing ratio

    Returns:
        Dictionary with variation_a and variation_b, or None
    """
    if len(iterations) < 4:
        return None  # Need at least 4 iterations for A-B-A-B

    # Split into A (even indices: 0,2,4...) and B (odd indices: 1,3,5...)
    a_group = [iterations[i] for i in range(0, len(iterations), 2)]
    b_group = [iterations[i] for i in range(1, len(iterations), 2)]

    if len(a_group) < 2 or len(b_group) < 2:
        return None

    # Calculate internal similarity within A group
    a_similarities = []
    for i in range(len(a_group)):
        for j in range(i + 1, len(a_group)):
            a_similarities.append(_calculate_similarity(a_group[i], a_group[j], swing_ratio))

    # Calculate internal similarity within B group
    b_similarities = []
    for i in range(len(b_group)):
        for j in range(i + 1, len(b_group)):
            b_similarities.append(_calculate_similarity(b_group[i], b_group[j], swing_ratio))

    # Calculate cross-similarity between A and B
    cross_similarities = []
    for a_iter in a_group:
        for b_iter in b_group:
            cross_similarities.append(_calculate_similarity(a_iter, b_iter, swing_ratio))

    # Check thresholds
    avg_a_similarity = sum(a_similarities) / len(a_similarities) if a_similarities else 0
    avg_b_similarity = sum(b_similarities) / len(b_similarities) if b_similarities else 0
    avg_cross_similarity = sum(cross_similarities) / len(cross_similarities) if cross_similarities else 0

    # A and B groups should be internally similar (>= 0.6)
    # A and B should be different from each other (< 0.5)
    if avg_a_similarity >= 0.6 and avg_b_similarity >= 0.6 and avg_cross_similarity < 0.5:
        # Use first of each group as representative pattern
        return {
            'variation_a': a_group[0],
            'variation_b': b_group[0]
        }

    return None


def _calculate_similarity(pattern1: List[Dict], pattern2: List[Dict],
                         swing_ratio: float = 1.0) -> float:
    """
    Calculate similarity between two note patterns.

    Uses a combination of:
    - Pitch sequence similarity (Jaccard index)
    - Timing alignment (normalized distance)
    - Note count ratio

    Swing awareness: When swing_ratio > 1.1, expands time bins to account
    for timing variations in swung rhythms.

    Args:
        pattern1: First pattern (list of notes)
        pattern2: Second pattern (list of notes)
        swing_ratio: Swing timing ratio (1.0=straight, 1.5-2.0=swing)

    Returns:
        Similarity score between 0 and 1
    """
    if not pattern1 and not pattern2:
        return 1.0

    if not pattern1 or not pattern2:
        return 0.0

    # 1. Count ratio similarity
    count_ratio = min(len(pattern1), len(pattern2)) / max(len(pattern1), len(pattern2))

    # 2. Pitch set similarity (Jaccard index)
    pitches1 = set(n['pitch'] for n in pattern1)
    pitches2 = set(n['pitch'] for n in pattern2)

    if pitches1 or pitches2:
        pitch_similarity = len(pitches1 & pitches2) / len(pitches1 | pitches2)
    else:
        pitch_similarity = 1.0

    # 3. Sequence similarity - compare note-by-note with timing tolerance
    # Adjust bin size for swing - swung notes have more timing variation
    base_bin_size = 0.05  # 50ms base tolerance
    if swing_ratio > 1.1:
        # Increase bin size proportionally to swing amount
        # Light swing (1.2): 62.5ms, Heavy swing (2.0): 100ms
        bin_size = base_bin_size * (1 + (swing_ratio - 1) * 0.5)
    else:
        bin_size = base_bin_size

    # Create time-binned representations for comparison
    def create_time_bins(pattern, bin_sz):
        """Group notes into time bins"""
        bins = defaultdict(list)
        for note in pattern:
            bin_idx = int(note['start'] / bin_sz)
            bins[bin_idx].append(note['pitch'])
        return bins

    bins1 = create_time_bins(pattern1, bin_size)
    bins2 = create_time_bins(pattern2, bin_size)

    all_bins = set(bins1.keys()) | set(bins2.keys())
    matching_bins = 0

    for bin_idx in all_bins:
        pitches_in_bin1 = set(bins1.get(bin_idx, []))
        pitches_in_bin2 = set(bins2.get(bin_idx, []))

        if pitches_in_bin1 or pitches_in_bin2:
            bin_similarity = len(pitches_in_bin1 & pitches_in_bin2) / len(pitches_in_bin1 | pitches_in_bin2)
            matching_bins += bin_similarity

    if all_bins:
        sequence_similarity = matching_bins / len(all_bins)
    else:
        sequence_similarity = 1.0

    # Weighted combination
    # Sequence similarity is most important, then pitch set, then count ratio
    final_score = (
        sequence_similarity * 0.6 +
        pitch_similarity * 0.3 +
        count_ratio * 0.1
    )

    return final_score


def detect_loops_per_voice(notes: List[Dict], tempo: float, beat_duration: float,
                           time_sig: str = "4/4", swing_ratio: float = 1.0,
                           bars_to_test: List[int] = [1, 2, 4, 8]) -> Dict[str, Optional[Dict]]:
    """
    Detect loops separately for each voice (bass, mid, high).

    Different musical parts often have different loop lengths - bass might have
    a 2-bar pattern while melody has a 4-bar pattern. This function runs loop
    detection independently on each voice range.

    Args:
        notes: List of all note dictionaries
        tempo: Tempo in BPM
        beat_duration: Duration of one beat in seconds
        time_sig: Time signature string
        swing_ratio: Swing timing ratio
        bars_to_test: List of bar lengths to test

    Returns:
        Dictionary with 'bass', 'mid', 'high' keys, each containing loop info or None
    """
    # Separate notes by voice range
    # Bass: C0-B2 (MIDI 24-47), Mid: C3-B4 (48-71), High: C5+ (72+)
    voices = {
        'bass': [n for n in notes if n.get('pitch', 60) < 48],
        'mid': [n for n in notes if 48 <= n.get('pitch', 60) < 72],
        'high': [n for n in notes if n.get('pitch', 60) >= 72]
    }

    result = {}
    for voice_name, voice_notes in voices.items():
        # Need at least 4 notes to detect a meaningful loop
        if len(voice_notes) >= 4:
            loop_info = detect_loop(
                voice_notes,
                tempo,
                beat_duration,
                bars_to_test,
                time_sig,
                swing_ratio
            )
            result[voice_name] = loop_info
        else:
            result[voice_name] = None

    return result


def remove_octave_duplicates(notes: List[Dict], preferred_octave: int = 4) -> List[Dict]:
    """
    Remove octave duplicates - if same note exists in multiple octaves at same time,
    keep only one (prefer middle octave around C4).
    """
    time_groups = defaultdict(list)
    for note in notes:
        time_key = round(note['start_quantized'], 4)
        time_groups[time_key].append(note)

    result = []
    for time_key, group in time_groups.items():
        note_classes = defaultdict(list)
        for note in group:
            note_class = note['pitch'] % 12
            note_classes[note_class].append(note)

        for note_class, duplicates in note_classes.items():
            if len(duplicates) == 1:
                result.append(duplicates[0])
            else:
                preferred_pitch = preferred_octave * 12 + note_class
                best = min(duplicates, key=lambda n: (
                    abs(n['pitch'] - preferred_pitch),
                    -n['velocity']
                ))
                result.append(best)

    return result


def merge_close_notes(notes: List[Dict], time_threshold: float = 0.05) -> List[Dict]:
    """
    Merge notes within time_threshold seconds of each other on same pitch.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: (n['pitch'], n['start_quantized']))

    result = []
    current = None

    for note in sorted_notes:
        if current is None:
            current = note.copy()
        elif (note['pitch'] == current['pitch'] and
              note['start_quantized'] - current['start_quantized'] <= time_threshold):
            current['duration'] = max(
                current['duration'],
                note['start_quantized'] + note['duration'] - current['start_quantized']
            )
            current['velocity'] = max(current['velocity'], note['velocity'])
        else:
            result.append(current)
            current = note.copy()

    if current is not None:
        result.append(current)

    return result


def reduce_chords(notes: List[Dict], max_chord_size: int = 3) -> List[Dict]:
    """
    Reduce complex chords - keep only loudest/most important notes per time.
    """
    time_groups = defaultdict(list)
    for note in notes:
        time_key = round(note['start_quantized'], 4)
        time_groups[time_key].append(note)

    result = []
    for time_key, group in time_groups.items():
        if len(group) <= max_chord_size:
            result.extend(group)
        else:
            sorted_group = sorted(group, key=lambda n: (
                -n['velocity'],
                abs(n['pitch'] - 60)
            ))
            result.extend(sorted_group[:max_chord_size])

    return result


def control_density(notes: List[Dict], tempo: float, max_notes_per_beat: int = 4) -> List[Dict]:
    """
    Limit note density to max_notes_per_beat.
    """
    if not notes:
        return []

    beat_duration = 60.0 / tempo

    beat_groups = defaultdict(list)
    for note in notes:
        beat_num = int(note['start_quantized'] / beat_duration)
        beat_groups[beat_num].append(note)

    result = []
    for beat_num, group in beat_groups.items():
        if len(group) <= max_notes_per_beat:
            result.extend(group)
        else:
            sorted_group = sorted(group, key=lambda n: (
                -n['velocity'],
                -n['duration']
            ))
            result.extend(sorted_group[:max_notes_per_beat])

    return result


def simplify_notes(notes: List[Dict], tempo: float, config: Optional[Dict] = None) -> List[Dict]:
    """
    Apply all simplification strategies to reduce note count while preserving musical content.
    """
    if not notes:
        return []

    config = config or {}
    merge_threshold = config.get('merge_threshold', 0.05)
    max_chord_size = config.get('max_chord_size', 3)
    max_notes_per_beat = config.get('max_notes_per_beat', 4)
    preferred_octave = config.get('preferred_octave', 4)

    result = notes
    original_count = len(result)

    print(f"\nSimplifying {original_count} notes:")

    result = remove_octave_duplicates(result, preferred_octave)
    print(f"  Octave deduplication: {len(result)} notes (removed {original_count - len(result)})")

    step2_count = len(result)
    result = merge_close_notes(result, merge_threshold)
    print(f"  Merging close notes: {len(result)} notes (merged {step2_count - len(result)})")

    step3_count = len(result)
    result = reduce_chords(result, max_chord_size)
    print(f"  Chord reduction: {len(result)} notes (removed {step3_count - len(result)})")

    step4_count = len(result)
    result = control_density(result, tempo, max_notes_per_beat)
    print(f"  Density control: {len(result)} notes (removed {step4_count - len(result)})")

    total_removed = original_count - len(result)
    reduction_pct = 100 * total_removed / original_count if original_count > 0 else 0
    print(f"Total simplification: {original_count} -> {len(result)} notes ({total_removed} removed, {reduction_pct:.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description='Clean and quantize MIDI file')
    parser.add_argument('input_midi', help='Input MIDI file')
    parser.add_argument('output_json', help='Output JSON file with notes')
    parser.add_argument('--quantize', type=int, default=16,
                        help='Quantization level (4, 8, or 16)')
    parser.add_argument('--velocity-threshold', type=int, default=10,
                        help='Minimum velocity to keep (default: 10)')
    parser.add_argument('--min-duration', type=float, default=0.02,
                        help='Minimum note duration in seconds (default: 0.02)')
    parser.add_argument('--detailed', action='store_true',
                        help='Output detailed voice separation')
    parser.add_argument('--simplify', action='store_true',
                        help='Apply aggressive note simplification (octave dedup, merge, chord reduction, density control)')
    parser.add_argument('--merge-threshold', type=float, default=0.05,
                        help='Time threshold for merging close notes in seconds (default: 0.05)')
    parser.add_argument('--max-chord-size', type=int, default=3,
                        help='Maximum notes in a chord (default: 3)')
    parser.add_argument('--max-notes-per-beat', type=int, default=4,
                        help='Maximum notes per beat (default: 4)')
    parser.add_argument('--preferred-octave', type=int, default=4,
                        help='Preferred octave for deduplication (default: 4, C4=middle C)')
    # Loop detection options
    parser.add_argument('--time-signature', type=str, default='4/4',
                        help='Time signature for loop detection (e.g., 4/4, 3/4, 6/8)')
    parser.add_argument('--swing-ratio', type=float, default=1.0,
                        help='Swing ratio for timing tolerance (1.0=straight, 1.5-2.0=swing)')
    parser.add_argument('--swing-confidence', type=float, default=0.0,
                        help='Confidence of swing detection (0.0-1.0)')
    parser.add_argument('--multi-voice-loops', action='store_true',
                        help='Detect loops separately for each voice (bass, mid, high)')

    args = parser.parse_args()

    if not os.path.exists(args.input_midi):
        print(f"Error: Input file not found: {args.input_midi}", file=sys.stderr)
        sys.exit(1)

    try:
        import pretty_midi
        import numpy as np

        # Load MIDI
        midi = pretty_midi.PrettyMIDI(args.input_midi)

        # Get tempo for quantization
        tempo = midi.estimate_tempo()
        if tempo <= 0:
            tempo = 120

        beat_duration = 60.0 / tempo
        grid_size = beat_duration / (args.quantize / 4)

        # Collect all notes with full info
        all_notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                })

        original_count = len(all_notes)
        print(f"Original notes: {original_count}")

        # Filter by velocity (very low threshold to keep most notes)
        filtered = [n for n in all_notes if n['velocity'] >= args.velocity_threshold]
        print(f"After velocity filter (>={args.velocity_threshold}): {len(filtered)}")

        # Filter by duration
        filtered = [n for n in filtered if n['duration'] >= args.min_duration]
        print(f"After duration filter (>={args.min_duration}s): {len(filtered)}")

        # Quantize start times to grid
        for note in filtered:
            note['start_quantized'] = round(note['start'] / grid_size) * grid_size
            note['duration_beats'] = note['duration'] / beat_duration

        # Remove exact duplicates at same quantized time/pitch (keep highest velocity)
        note_map = {}
        for note in filtered:
            key = (note['pitch'], round(note['start_quantized'], 4))
            if key not in note_map or note['velocity'] > note_map[key]['velocity']:
                note_map[key] = note

        unique = list(note_map.values())
        print(f"After deduplication: {len(unique)}")

        # Apply simplification if requested
        if args.simplify:
            simplify_config = {
                'merge_threshold': args.merge_threshold,
                'max_chord_size': args.max_chord_size,
                'max_notes_per_beat': args.max_notes_per_beat,
                'preferred_octave': args.preferred_octave
            }
            unique = simplify_notes(unique, tempo, simplify_config)

        # Sort by start time, then pitch
        unique.sort(key=lambda n: (n['start_quantized'], n['pitch']))

        # Separate into voice ranges for polyphonic output
        # Bass: C0-B2 (MIDI 24-47), Mid: C3-B4 (48-71), High: C5+ (72+)
        bass_notes = [n for n in unique if n['pitch'] < 48]
        mid_notes = [n for n in unique if 48 <= n['pitch'] < 72]
        high_notes = [n for n in unique if n['pitch'] >= 72]

        # Format output notes
        def format_notes(notes):
            return [
                {
                    'pitch': n['pitch'],
                    'start': round(n['start_quantized'], 4),
                    'duration': round(n['duration'], 4),
                    'duration_beats': round(n['duration_beats'], 4),
                    'velocity': n['velocity'],
                    'velocity_normalized': round(n['velocity'] / 127.0, 3)
                }
                for n in notes
            ]

        # Calculate total duration in beats
        if unique:
            max_end = max(n['start_quantized'] + n['duration'] for n in unique)
            total_beats = max_end / beat_duration
        else:
            total_beats = 0

        # Detect loops with time signature and swing awareness
        loop_info = detect_loop(
            unique, tempo, beat_duration,
            bars_to_test=[1, 2, 4, 8],
            time_sig=args.time_signature,
            swing_ratio=args.swing_ratio
        )

        # Format loop info for output
        if loop_info:
            # Format the loop notes
            loop_notes_formatted = [
                {
                    'pitch': n['pitch'],
                    'start': round(n['start'], 4),
                    'duration': round(n['duration'], 4),
                    'velocity': n['velocity'],
                    'velocity_normalized': round(n['velocity'] / 127.0, 3)
                }
                for n in loop_info['notes']
            ]

            loop_output = {
                'detected': loop_info['detected'],
                'bars': loop_info['bars'],
                'confidence': loop_info['confidence'],
                'start_beat': loop_info['start_beat'],
                'end_beat': loop_info['end_beat'],
                'notes': loop_notes_formatted,
                'repetitions': loop_info.get('repetitions', 0),
                'reference_iteration': loop_info.get('reference_iteration', 0),
                'pattern_type': loop_info.get('pattern_type', 'single')
            }

            # Add A-B variation notes if alternating pattern detected
            if loop_info.get('pattern_type') == 'alternating':
                loop_output['variation_a'] = [
                    {
                        'pitch': n['pitch'],
                        'start': round(n['start'], 4),
                        'duration': round(n['duration'], 4),
                        'velocity': n['velocity']
                    }
                    for n in loop_info.get('variation_a', [])
                ]
                loop_output['variation_b'] = [
                    {
                        'pitch': n['pitch'],
                        'start': round(n['start'], 4),
                        'duration': round(n['duration'], 4),
                        'velocity': n['velocity']
                    }
                    for n in loop_info.get('variation_b', [])
                ]
        else:
            loop_output = {
                'detected': False,
                'bars': 0,
                'confidence': 0.0,
                'start_beat': 0,
                'end_beat': 0,
                'notes': [],
                'pattern_type': 'none'
            }

        # Detect per-voice loops if requested
        voice_loops_output = None
        if args.multi_voice_loops:
            voice_loops = detect_loops_per_voice(
                unique, tempo, beat_duration,
                time_sig=args.time_signature,
                swing_ratio=args.swing_ratio
            )
            voice_loops_output = {}
            for voice_name, voice_loop in voice_loops.items():
                if voice_loop:
                    voice_loops_output[voice_name] = {
                        'bars': voice_loop['bars'],
                        'confidence': voice_loop['confidence'],
                        'pattern_type': voice_loop.get('pattern_type', 'single'),
                        'repetitions': voice_loop.get('repetitions', 0)
                    }
                else:
                    voice_loops_output[voice_name] = None

        result = {
            'notes': format_notes(unique),
            'retained': len(unique),  # Top-level for Go compatibility
            'removed': original_count - len(unique),  # Top-level for Go compatibility
            'voices': {
                'bass': format_notes(bass_notes),
                'mid': format_notes(mid_notes),
                'high': format_notes(high_notes)
            },
            'stats': {
                'total': len(unique),
                'bass_count': len(bass_notes),
                'mid_count': len(mid_notes),
                'high_count': len(high_notes),
                'original': original_count,
                'removed': original_count - len(unique)
            },
            'tempo': tempo,
            'beat_duration': beat_duration,
            'grid_size': grid_size,
            'total_beats': round(total_beats, 2),
            'quantize': args.quantize,
            'time_signature': args.time_signature,
            'swing_ratio': args.swing_ratio,
            'loop': loop_output
        }

        # Add voice loops if detected
        if voice_loops_output:
            result['voice_loops'] = voice_loops_output

        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nCleanup complete:")
        print(f"  Total: {len(unique)} notes")
        print(f"  Bass (<48): {len(bass_notes)} notes")
        print(f"  Mid (48-71): {len(mid_notes)} notes")
        print(f"  High (72+): {len(high_notes)} notes")
        print(f"  Duration: {total_beats:.1f} beats @ {tempo:.0f} BPM")

        if loop_info:
            print(f"\nLoop detected:")
            print(f"  Pattern: {loop_info['bars']} bar(s) ({args.time_signature})")
            print(f"  Confidence: {loop_info['confidence']:.0%}")
            print(f"  Repetitions: {loop_info.get('repetitions', 0)}")
            print(f"  Notes in pattern: {len(loop_info['notes'])}")
            print(f"  Reference iteration: {loop_info.get('reference_iteration', 0)}")
            if loop_info.get('pattern_type') == 'alternating':
                print(f"  Pattern type: A-B-A-B alternating")
            if args.swing_ratio > 1.1:
                print(f"  Swing-aware: ratio={args.swing_ratio:.2f}")
        else:
            print(f"\nNo repeating loop detected")

        # Print per-voice loops if detected
        if voice_loops_output:
            print(f"\nPer-voice loops:")
            for voice_name, voice_loop in voice_loops_output.items():
                if voice_loop:
                    print(f"  {voice_name}: {voice_loop['bars']} bar(s), {voice_loop['confidence']:.0%} confidence")
                else:
                    print(f"  {voice_name}: no loop detected")

    except ImportError:
        print("Error: pretty_midi not installed. Run: pip install pretty_midi", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: MIDI cleanup failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
