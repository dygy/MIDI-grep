#!/usr/bin/env python3
"""
MIDI cleanup and quantization using pretty_midi.
Outputs cleaned notes as JSON for Go to consume.
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
    parser.add_argument('--velocity-threshold', type=int, default=20,
                        help='Minimum velocity to keep (default: 20)')
    parser.add_argument('--min-duration', type=float, default=0.03,
                        help='Minimum note duration in seconds (default: 0.03)')

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

        # Collect all notes
        all_notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })

        original_count = len(all_notes)

        # Filter by velocity
        filtered = [n for n in all_notes if n['velocity'] >= args.velocity_threshold]

        # Filter by duration
        filtered = [n for n in filtered if (n['end'] - n['start']) >= args.min_duration]

        # Quantize start times
        for note in filtered:
            note['start'] = round(note['start'] / grid_size) * grid_size
            note['duration'] = note['end'] - note['start']

        # Remove duplicates at same time/pitch
        seen = set()
        unique = []
        for note in filtered:
            key = (note['pitch'], round(note['start'], 4))
            if key not in seen:
                seen.add(key)
                unique.append(note)

        # Sort by start time
        unique.sort(key=lambda n: (n['start'], n['pitch']))

        # Format output
        output_notes = [
            {
                'pitch': n['pitch'],
                'start': round(n['start'], 4),
                'duration': round(n['duration'], 4),
                'velocity': n['velocity']
            }
            for n in unique
        ]

        removed_count = original_count - len(output_notes)

        result = {
            'notes': output_notes,
            'retained': len(output_notes),
            'removed': removed_count,
            'tempo': tempo,
            'quantize': args.quantize
        }

        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Cleanup complete: {len(output_notes)} notes retained, {removed_count} removed")

    except ImportError:
        print("Error: pretty_midi not installed. Run: pip install pretty_midi", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: MIDI cleanup failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
