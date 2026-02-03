#!/usr/bin/env python3
"""
Generate Strudel code from smart analysis chord data.

This creates patterns based on detected chords, tempo, and sections -
much more accurate than note-by-note transcription for electronic music.
"""

import sys
import json


def main():
    if len(sys.argv) < 2:
        print("Usage: chord_to_strudel.py <analysis_json>", file=sys.stderr)
        sys.exit(1)

    analysis_path = sys.argv[1]

    with open(analysis_path, 'r') as f:
        data = json.load(f)

    tempo = data.get('tempo', 120)
    key = data.get('key', 'C')
    mode = data.get('mode', 'minor')
    chords = data.get('chords', [])
    sections = data.get('sections', [])
    genre_hints = data.get('genre_hints', [])

    # Simplify chords to bar-level (4 beats per bar)
    bar_chords = simplify_chords_to_bars(chords)

    # Detect chord progression pattern (most common 4-8 bar pattern)
    pattern = detect_chord_pattern(bar_chords)

    # Generate output
    output = []
    output.append(f"// Auto-generated from audio analysis")
    output.append(f"// Tempo: {tempo:.0f} BPM, Key: {key} {mode}")
    output.append(f"// Detected chord progression: {' → '.join(pattern[:8])}")
    output.append(f"// Genre hints: {', '.join(genre_hints)}")
    output.append("")
    output.append(f"setcps({tempo:.0f}/60/4)")
    output.append("")

    # Generate chord pattern
    chord_pattern = format_chord_pattern(pattern)
    output.append(f"let chords = \"{chord_pattern}\"")
    output.append("")

    # Generate bass pattern (root notes)
    bass_pattern = format_bass_pattern(pattern)
    output.append(f"let bassNotes = \"{bass_pattern}\"")
    output.append("")

    # Generate style-appropriate sounds based on genre hints
    if 'electronic' in genre_hints or 'high-energy' in genre_hints:
        output.append("// Electronic/Dance style")
        output.append("let bass = note(bassNotes)")
        output.append("  .sound(\"supersaw\")")
        output.append("  .lpf(400)")
        output.append("  .gain(0.9)")
        output.append("")
        output.append("let pad = note(chords)")
        output.append("  .sound(\"gm_pad_poly\")")
        output.append("  .lpf(2000)")
        output.append("  .attack(0.1)")
        output.append("  .release(0.5)")
        output.append("  .gain(0.6)")
        output.append("")
        output.append("let lead = note(chords)")
        output.append("  .sound(\"gm_lead_2_sawtooth\")")
        output.append("  .lpf(4000)")
        output.append("  .struct(\"t(5,16)\")")
        output.append("  .gain(0.4)")
        output.append("")
        output.append("let drums = stack(")
        output.append("  s(\"bd*4\").bank(\"RolandTR808\"),")
        output.append("  s(\"~ sd ~ sd\").bank(\"RolandTR808\"),")
        output.append("  s(\"hh*8\").bank(\"RolandTR808\").gain(0.5)")
        output.append(").room(0.1)")
    else:
        output.append("// Melodic style")
        output.append("let bass = note(bassNotes)")
        output.append("  .sound(\"gm_acoustic_bass\")")
        output.append("  .gain(0.9)")
        output.append("")
        output.append("let pad = note(chords)")
        output.append("  .sound(\"gm_piano\")")
        output.append("  .gain(0.7)")

    output.append("")
    output.append("// Play all together")
    output.append("$: stack(bass, pad, drums)")
    output.append("")
    output.append("// Or play individual parts:")
    output.append("// $: bass")
    output.append("// $: pad")
    output.append("// $: drums")

    print("\n".join(output))


def simplify_chords_to_bars(chords):
    """Convert beat-level chords to bar-level (one chord per bar)."""
    bar_chords = []
    current_bar = -1
    bar_chord_counts = {}

    for c in chords:
        bar = c['beat'] // 4
        if bar != current_bar:
            if bar_chord_counts:
                # Use most common chord in previous bar
                most_common = max(bar_chord_counts, key=bar_chord_counts.get)
                bar_chords.append(most_common)
            current_bar = bar
            bar_chord_counts = {}
        chord = c['chord']
        bar_chord_counts[chord] = bar_chord_counts.get(chord, 0) + 1

    # Don't forget last bar
    if bar_chord_counts:
        most_common = max(bar_chord_counts, key=bar_chord_counts.get)
        bar_chords.append(most_common)

    return bar_chords


def detect_chord_pattern(bar_chords, max_pattern_length=8):
    """Find the most likely repeating chord pattern."""
    if len(bar_chords) < 4:
        return bar_chords

    # Try pattern lengths from 4 to max_pattern_length
    best_pattern = bar_chords[:4]
    best_score = 0

    for pattern_len in range(4, min(max_pattern_length + 1, len(bar_chords) // 2 + 1)):
        pattern = bar_chords[:pattern_len]
        score = 0

        # Count how many times this pattern repeats
        for i in range(0, len(bar_chords) - pattern_len + 1, pattern_len):
            chunk = bar_chords[i:i + pattern_len]
            matches = sum(1 for a, b in zip(pattern, chunk) if a == b)
            score += matches / pattern_len

        # Normalize by number of possible repeats
        num_repeats = len(bar_chords) // pattern_len
        if num_repeats > 0:
            score /= num_repeats

        if score > best_score:
            best_score = score
            best_pattern = pattern

    return best_pattern


def format_chord_pattern(chords):
    """Format chords for Strudel mini-notation."""
    # Convert chord names to Strudel format
    formatted = []
    for chord in chords:
        # Convert chord name to note names
        root = chord.rstrip('m')
        is_minor = chord.endswith('m')

        # Build chord notes (root, third, fifth)
        if is_minor:
            # Minor chord: root, minor third (+3), fifth (+7)
            formatted.append(f"[{root}3,{get_note(root, 3)}3,{get_note(root, 7)}3]")
        else:
            # Major chord: root, major third (+4), fifth (+7)
            formatted.append(f"[{root}3,{get_note(root, 4)}3,{get_note(root, 7)}3]")

    return " | ".join(formatted)


def format_bass_pattern(chords):
    """Format bass notes (chord roots) for Strudel."""
    roots = []
    for chord in chords:
        root = chord.rstrip('m')
        roots.append(f"{root}1")  # Octave 1 for bass

    return " | ".join(roots)


def get_note(root, semitones):
    """Get note that is N semitones above root."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = notes.index(root) if root in notes else 0
    new_idx = (root_idx + semitones) % 12
    return notes[new_idx]


if __name__ == '__main__':
    main()
