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

        result = {
            'notes': format_notes(unique),
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
            'quantize': args.quantize
        }

        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nCleanup complete:")
        print(f"  Total: {len(unique)} notes")
        print(f"  Bass (<48): {len(bass_notes)} notes")
        print(f"  Mid (48-71): {len(mid_notes)} notes")
        print(f"  High (72+): {len(high_notes)} notes")
        print(f"  Duration: {total_beats:.1f} beats @ {tempo:.0f} BPM")

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
