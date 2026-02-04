#!/usr/bin/env python3
"""
Pattern Thinner - AI-driven pattern density control

This script takes Strudel code and AI-derived density parameters,
then thins out patterns to match the target onset density.

Key insight: Too many drum hits makes librosa detect a faster tempo.
We need to thin patterns to match the original's rhythmic feel.
"""

import argparse
import json
import re
import sys
import random
from typing import List, Dict, Tuple


def parse_bar_arrays(code: str) -> Dict[str, List[str]]:
    """Extract bar arrays from Strudel code."""
    arrays = {}

    # Match: let name = [\n  "pattern1",\n  "pattern2"\n]
    pattern = re.compile(r'let\s+(\w+)\s*=\s*\[\s*((?:"[^"]*"(?:,\s*)?)+)\s*\]', re.MULTILINE | re.DOTALL)

    for match in pattern.finditer(code):
        name = match.group(1)
        array_content = match.group(2)
        bars = re.findall(r'"([^"]*)"', array_content)
        if bars:
            arrays[name] = bars

    return arrays


def count_drum_hits(bar: str) -> Dict[str, int]:
    """Count drum hits in a bar pattern."""
    counts = {'bd': 0, 'sd': 0, 'hh': 0, 'oh': 0, 'cp': 0}

    # Expand ~*N notation
    tokens = []
    for token in bar.split():
        if token.startswith('~*'):
            try:
                count = int(token[2:])
                tokens.extend(['~'] * count)
            except ValueError:
                tokens.append(token)
        else:
            tokens.append(token)

    for token in tokens:
        # Handle chords [a,b]
        if token.startswith('[') and token.endswith(']'):
            notes = token[1:-1].split(',')
            for note in notes:
                note = note.strip().lower()
                if note in counts:
                    counts[note] += 1
        else:
            token_lower = token.lower()
            if token_lower in counts:
                counts[token_lower] += 1

    return counts


def thin_drum_bar(bar: str, max_kicks: int, max_snares: int, max_hihats: int) -> str:
    """
    Thin a drum bar to respect maximum hit counts.

    Strategy:
    - Keep hits on strong beats (1, 3)
    - Remove excess hits on weak beats
    - Prioritize kicks over hihats for thinning
    """
    # Expand ~*N notation
    tokens = []
    for token in bar.split():
        if token.startswith('~*'):
            try:
                count = int(token[2:])
                tokens.extend(['~'] * count)
            except ValueError:
                tokens.append(token)
        else:
            tokens.append(token)

    if not tokens:
        return "~"

    # Find positions of each drum type
    drum_positions = {'bd': [], 'sd': [], 'hh': [], 'oh': []}
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if token_lower in drum_positions:
            drum_positions[token_lower].append(i)

    # Calculate strong beat positions (assuming 16 slots per bar)
    num_slots = len(tokens)
    slots_per_beat = num_slots // 4 if num_slots >= 4 else 1
    strong_beats = [0, slots_per_beat * 2]  # Beats 1 and 3
    medium_beats = [slots_per_beat, slots_per_beat * 3]  # Beats 2 and 4

    def score_position(pos: int) -> int:
        """Score a position - higher = more important."""
        if pos in strong_beats:
            return 3
        if pos in medium_beats:
            return 2
        if pos % slots_per_beat == 0:  # On any beat
            return 1
        return 0  # Off-beat

    # Thin each drum type
    def thin_hits(positions: List[int], max_count: int) -> List[int]:
        if len(positions) <= max_count:
            return positions

        # Sort by importance (score), keep highest scoring
        scored = [(pos, score_position(pos)) for pos in positions]
        scored.sort(key=lambda x: -x[1])  # Descending by score

        # Keep top max_count, but also add some randomness for variation
        kept = []
        for pos, score in scored[:max_count]:
            kept.append(pos)

        return kept

    # Apply thinning
    kept_bd = set(thin_hits(drum_positions['bd'], max_kicks))
    kept_sd = set(thin_hits(drum_positions['sd'], max_snares))
    kept_hh = set(thin_hits(drum_positions['hh'], max_hihats))
    kept_oh = set(thin_hits(drum_positions['oh'], max(1, max_hihats // 2)))

    # Rebuild tokens
    new_tokens = []
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if token_lower == 'bd' and i not in kept_bd:
            new_tokens.append('~')
        elif token_lower == 'sd' and i not in kept_sd:
            new_tokens.append('~')
        elif token_lower == 'hh' and i not in kept_hh:
            new_tokens.append('~')
        elif token_lower == 'oh' and i not in kept_oh:
            new_tokens.append('~')
        else:
            new_tokens.append(token)

    # Simplify: collapse consecutive rests
    return simplify_pattern(new_tokens)


def simplify_pattern(tokens: List[str]) -> str:
    """Collapse consecutive rests into ~*N notation."""
    if not tokens:
        return "~"

    result = []
    rest_count = 0

    for token in tokens:
        if token == '~':
            rest_count += 1
        else:
            if rest_count > 0:
                if rest_count == 1:
                    result.append('~')
                else:
                    result.append(f'~*{rest_count}')
                rest_count = 0
            result.append(token)

    # Handle trailing rests (skip for cleaner output)
    # Actually, keep them to maintain bar length
    if rest_count > 0:
        if rest_count == 1:
            result.append('~')
        else:
            result.append(f'~*{rest_count}')

    if not result:
        return "~"

    return ' '.join(result)


def thin_melodic_bar(bar: str, target_density: float, bar_duration: float) -> str:
    """
    Thin a melodic bar to match target onset density.

    Args:
        bar: The pattern string
        target_density: Target onsets per second
        bar_duration: Duration of one bar in seconds
    """
    # Expand ~*N notation
    tokens = []
    for token in bar.split():
        if token.startswith('~*'):
            try:
                count = int(token[2:])
                tokens.extend(['~'] * count)
            except ValueError:
                tokens.append(token)
        else:
            tokens.append(token)

    if not tokens:
        return "~"

    # Count current notes
    note_positions = []
    for i, token in enumerate(tokens):
        if token != '~':
            note_positions.append(i)

    # Calculate target notes per bar
    target_notes = int(target_density * bar_duration)
    if target_notes < 1:
        target_notes = 1

    if len(note_positions) <= target_notes:
        # Already sparse enough
        return simplify_pattern(tokens)

    # Thin notes - keep notes on stronger positions
    num_slots = len(tokens)
    slots_per_beat = num_slots // 4 if num_slots >= 4 else 1

    def score_position(pos: int) -> float:
        """Score a position - higher = more important."""
        if pos == 0:  # Downbeat
            return 4
        if pos % (slots_per_beat * 2) == 0:  # Beats 1, 3
            return 3
        if pos % slots_per_beat == 0:  # Any beat
            return 2
        if pos % (slots_per_beat // 2) == 0:  # 8th notes
            return 1
        return 0.5  # 16th notes

    # Sort by importance
    scored = [(pos, score_position(pos)) for pos in note_positions]
    scored.sort(key=lambda x: -x[1])

    # Keep top target_notes
    kept = set(pos for pos, _ in scored[:target_notes])

    # Rebuild
    new_tokens = []
    for i, token in enumerate(tokens):
        if token != '~' and i not in kept:
            new_tokens.append('~')
        else:
            new_tokens.append(token)

    return simplify_pattern(new_tokens)


def thin_strudel_code(code: str, params: Dict) -> str:
    """
    Thin patterns in Strudel code to match AI-derived parameters.
    """
    # Extract parameters
    max_kicks = params.get('max_kicks_per_bar', 4)
    max_snares = params.get('max_snares_per_bar', 2)
    max_hihats = params.get('max_hihats_per_bar', 8)
    target_density = params.get('target_onset_density', 3.0)
    tempo = params.get('tempo', 120)

    bar_duration = 60.0 / tempo * 4  # 4 beats per bar

    # Parse arrays
    arrays = parse_bar_arrays(code)

    # Track modifications
    modified = code

    # Thin drums array
    if 'drums' in arrays:
        old_bars = arrays['drums']
        new_bars = []

        for bar in old_bars:
            new_bar = thin_drum_bar(bar, max_kicks, max_snares, max_hihats)
            new_bars.append(new_bar)

        # Replace in code
        modified = replace_array(modified, 'drums', new_bars)

        # Print stats
        old_hits = sum(sum(count_drum_hits(b).values()) for b in old_bars)
        new_hits = sum(sum(count_drum_hits(b).values()) for b in new_bars)
        print(f"Drums: {old_hits} hits -> {new_hits} hits ({len(old_bars)} bars)", file=sys.stderr)

    # ALWAYS thin melodic patterns to match target density
    # This is critical for tempo detection - too many 8th/16th notes
    # makes librosa detect double-time (2x BPM)
    for voice in ['bass', 'mid', 'high']:
        if voice in arrays:
            old_bars = arrays[voice]

            # Count notes
            total_notes = 0
            for bar in old_bars:
                tokens = bar.split()
                for token in tokens:
                    if token != '~' and not token.startswith('~*'):
                        total_notes += 1

            # Calculate current density
            total_duration = len(old_bars) * bar_duration
            current_density = total_notes / total_duration if total_duration > 0 else 0

            # Thin more aggressively for high voice (it has the most notes)
            voice_target = target_density
            if voice == 'high':
                voice_target = target_density * 0.5  # Much sparser for high voice
            elif voice == 'mid':
                voice_target = target_density * 0.8  # Moderately sparser

            # Thin if denser than target
            if current_density > voice_target:
                new_bars = []
                for bar in old_bars:
                    new_bar = thin_melodic_bar(bar, voice_target, bar_duration)
                    new_bars.append(new_bar)

                modified = replace_array(modified, voice, new_bars)

                new_notes = sum(1 for b in new_bars for t in b.split() if t != '~' and not t.startswith('~*'))
                print(f"{voice}: {total_notes} notes -> {new_notes} notes (target density: {voice_target:.2f}/sec)", file=sys.stderr)

    return modified


def replace_array(code: str, name: str, new_bars: List[str]) -> str:
    """Replace a bar array in the code."""
    # Build new array string
    new_array = f'let {name} = [\n'
    for i, bar in enumerate(new_bars):
        new_array += f'  "{bar}"'
        if i < len(new_bars) - 1:
            new_array += ','
        new_array += '\n'
    new_array += ']'

    # Replace in code using regex
    pattern = re.compile(
        rf'let\s+{name}\s*=\s*\[\s*((?:"[^"]*"(?:,\s*)?)+)\s*\]',
        re.MULTILINE | re.DOTALL
    )

    return pattern.sub(new_array, code)


def main():
    parser = argparse.ArgumentParser(description='Thin Strudel patterns to match target density')
    parser.add_argument('input', help='Input Strudel file')
    parser.add_argument('--params', '-p', required=True, help='AI params JSON file')
    parser.add_argument('--output', '-o', help='Output Strudel file (default: stdout)')
    args = parser.parse_args()

    # Load input
    with open(args.input, 'r') as f:
        code = f.read()

    # Load params
    with open(args.params, 'r') as f:
        params_data = json.load(f)

    # Extract params (might be nested under 'params')
    if 'params' in params_data:
        params = params_data['params']
    else:
        params = params_data

    # Add tempo from profile if available
    if 'profile' in params_data and 'tempo' in params_data['profile']:
        params['tempo'] = params_data['profile']['tempo']

    print(f"Thinning with params: max_kicks={params.get('max_kicks_per_bar', '?')}, "
          f"max_snares={params.get('max_snares_per_bar', '?')}, "
          f"max_hihats={params.get('max_hihats_per_bar', '?')}", file=sys.stderr)

    # Thin the code
    thinned = thin_strudel_code(code, params)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(thinned)
        print(f"Written: {args.output}", file=sys.stderr)
    else:
        print(thinned)


if __name__ == '__main__':
    main()
