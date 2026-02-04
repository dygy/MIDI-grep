#!/usr/bin/env python3
"""
Render Strudel-style patterns to WAV audio.
Synthesizes drums, bass, and melodic content.
Now with adjustable mix parameters based on analysis.
"""

import argparse
import json
import math
import re
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os

# Audio settings
SAMPLE_RATE = 44100

# Default mix levels - will be overridden by AI analysis (ai_params.json)
# These are fallback values if AI params not available
DEFAULT_MIX = {
    'kick_gain': 0.3,
    'snare_gain': 0.3,
    'hh_gain': 0.3,
    'bass_gain': 0.2,
    'vox_gain': 0.4,
    'stab_gain': 0.3,
    'lead_gain': 0.8,
}

def parse_strudel_code(code):
    """Extract patterns and BPM from Strudel code."""
    result = {
        'bpm': 120,
        'patterns': {},
        'bar_arrays': {},  # NEW: Store bar arrays separately for proper rendering
        'duration_bars': 16,
        'mix': DEFAULT_MIX.copy()
    }

    # Extract BPM from setcps
    cps_match = re.search(r'setcps\((\d+)/60/4\)', code)
    if cps_match:
        result['bpm'] = int(cps_match.group(1))

    # Extract let patterns - OLD format: let name = "pattern"
    let_patterns = re.findall(r'let\s+(\w+)\s*=\s*["`]([^"`]+)["`]', code)
    for name, pattern in let_patterns:
        result['patterns'][name] = pattern

    # Extract let patterns - NEW format: let name = ["pattern1", "pattern2", ...]
    # Match arrays like: let bass = [\n  "pattern1",\n  "pattern2"\n]
    array_pattern = re.compile(r'let\s+(\w+)\s*=\s*\[\s*((?:"[^"]*"(?:,\s*)?)+)\s*\]', re.MULTILINE | re.DOTALL)
    for match in array_pattern.finditer(code):
        name = match.group(1)
        array_content = match.group(2)
        # Extract all quoted strings from the array - keep them as separate bars
        patterns = re.findall(r'"([^"]*)"', array_content)
        if patterns:
            # NEW: Store as bar array for proper sequential rendering
            result['bar_arrays'][name] = patterns
            # Update duration_bars based on longest array
            if len(patterns) > result['duration_bars']:
                result['duration_bars'] = len(patterns)

    # Try to extract gain values from effect chains
    for pattern_type in ['kick', 'snare', 'hh', 'bass', 'vox', 'stab', 'lead']:
        gain_match = re.search(rf'{pattern_type}Fx.*?\.gain\(([0-9.]+)\)', code)
        if gain_match:
            # Scale the gain - Strudel uses 0-2+, we use 0-1
            strudel_gain = float(gain_match.group(1))
            result['mix'][f'{pattern_type}_gain'] = min(strudel_gain / 2.0, 1.0)

    return result

def parse_mini_notation(pattern, steps=16):
    """Parse mini-notation pattern into list of events."""
    events = []

    # Remove bar separators
    pattern = pattern.replace('|', ' ')

    # First, expand ~*N notation to individual rests
    # e.g., "~*3" becomes "~ ~ ~"
    expanded_tokens = []
    for token in pattern.split():
        if token.startswith('~*'):
            try:
                count = int(token[2:])
                expanded_tokens.extend(['~'] * count)
            except ValueError:
                expanded_tokens.append(token)
        else:
            expanded_tokens.append(token)

    if not expanded_tokens:
        return events

    step_duration = 1.0 / len(expanded_tokens)

    for i, token in enumerate(expanded_tokens):
        if token == '~':
            continue

        time = i * step_duration

        # Handle chords [a,b,c]
        if token.startswith('[') and token.endswith(']'):
            chord_notes = token[1:-1].split(',')
            for note in chord_notes:
                events.append({'time': time, 'note': note.strip(), 'duration': step_duration * 0.8})
        else:
            events.append({'time': time, 'note': token, 'duration': step_duration * 0.8})

    return events

def note_to_freq(note_str):
    """Convert note string like 'c#4' to frequency."""
    note_str = note_str.lower().strip()

    # Drum sounds
    if note_str in ['bd', 'kick']:
        return 60  # Low frequency for kick
    if note_str in ['sd', 'snare']:
        return 200  # Snare frequency
    if note_str in ['hh', 'hihat']:
        return 8000  # Hi-hat frequency
    if note_str in ['oh']:
        return 7000  # Open hi-hat

    # Parse note name
    note_map = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}

    if len(note_str) < 2:
        return 440

    note = note_str[0]
    if note not in note_map:
        return 440

    semitone = note_map[note]
    idx = 1

    # Handle sharp/flat
    if len(note_str) > 1 and note_str[1] == '#':
        semitone += 1
        idx = 2
    elif len(note_str) > 1 and note_str[1] == 'b':
        semitone -= 1
        idx = 2

    # Get octave
    octave = 4
    if idx < len(note_str):
        try:
            octave = int(note_str[idx:])
        except ValueError:
            pass

    # Calculate frequency (A4 = 440Hz)
    midi_note = semitone + (octave + 1) * 12
    freq = 440 * (2 ** ((midi_note - 69) / 12))

    return freq

def generate_kick(duration, sample_rate=SAMPLE_RATE):
    """Generate a kick drum sound - softer attack to reduce transient detection."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Pitch envelope (starts high, drops to base) - slower attack
    pitch_env = 150 * np.exp(-30 * t) + 55  # Gentler pitch drop

    # Generate sine with pitch envelope
    phase = np.cumsum(pitch_env) / sample_rate * 2 * np.pi
    kick = np.sin(phase)

    # Amplitude envelope with softer attack
    # Use raised cosine for smoother onset
    attack_samples = int(sample_rate * 0.005)  # 5ms attack ramp
    amp_env = np.exp(-10 * t)  # Slower decay
    if attack_samples < len(amp_env):
        attack_ramp = 0.5 * (1 - np.cos(np.pi * np.arange(attack_samples) / attack_samples))
        amp_env[:attack_samples] *= attack_ramp
    kick *= amp_env

    # REMOVE click/transient - this was causing tempo detection issues
    # The click creates sharp transients that librosa interprets as extra beats

    # Gentler distortion
    kick = np.tanh(kick * 1.2) * 0.6

    return kick

def generate_snare(duration, sample_rate=SAMPLE_RATE):
    """Generate a snare drum sound."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Body (tone)
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-20 * t)

    # Noise (snares)
    noise = np.random.randn(len(t)) * np.exp(-15 * t) * 0.5

    # High-pass the noise
    b, a = signal.butter(2, 2000 / (sample_rate / 2), 'high')
    noise = signal.filtfilt(b, a, noise)

    snare = body + noise
    snare = np.tanh(snare * 1.5) * 0.6

    return snare

def generate_hihat(duration, is_open=False, sample_rate=SAMPLE_RATE):
    """Generate a hi-hat sound."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # White noise
    noise = np.random.randn(len(t))

    # High-pass filter
    b, a = signal.butter(2, 6000 / (sample_rate / 2), 'high')
    noise = signal.filtfilt(b, a, noise)

    # Envelope
    decay = 5 if is_open else 30
    env = np.exp(-decay * t)

    hihat = noise * env * 0.5  # Increased base level

    return hihat

def generate_bass(freq, duration, sample_rate=SAMPLE_RATE):
    """Generate a bass sound (sawtooth with filter) with soft attack."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Sawtooth wave with sub-octave
    saw = 2 * (t * freq % 1) - 1
    sub = np.sin(2 * np.pi * freq * 0.5 * t)  # Sub-octave
    bass = saw * 0.6 + sub * 0.4

    # Low-pass filter
    cutoff = min(freq * 3, 800) / (sample_rate / 2)
    b, a = signal.butter(2, cutoff, 'low')
    bass = signal.filtfilt(b, a, bass)

    # Soft envelope with raised cosine attack
    attack = int(sample_rate * 0.015)
    release = int(sample_rate * 0.1)
    env = np.ones(len(t))
    if attack < len(env):
        x = np.linspace(0, np.pi, attack)
        env[:attack] = 0.5 * (1 - np.cos(x))
    if len(env) > release:
        env[-release:] = np.linspace(1, 0, release)

    bass *= env
    bass = np.tanh(bass * 1.5) * 0.6

    return bass

def generate_synth(freq, duration, waveform='square', sample_rate=SAMPLE_RATE):
    """Generate a synth sound with SOFT ATTACK for tempo stability.
    Uses additive synthesis for smooth onset without transient clicks."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Additive synthesis - pure sines for smooth onset
    wave = np.sin(2 * np.pi * freq * t) * 1.0

    # Add harmonics for richness
    for harmonic in range(2, 8):
        harmonic_freq = freq * harmonic
        wave += np.sin(2 * np.pi * harmonic_freq * t) * (0.5 / harmonic)

    # Gentle LP filter
    cutoff = min(freq * 8, 8000)
    b, a = signal.butter(2, cutoff / (sample_rate / 2), 'low')
    wave = signal.filtfilt(b, a, wave)

    # SOFT ATTACK envelope - critical for tempo detection
    attack = int(sample_rate * 0.04)  # 40ms attack
    decay = int(sample_rate * 0.08)
    sustain_level = 0.5
    release = int(sample_rate * 0.1)

    env = np.ones(len(t)) * sustain_level
    if attack > 0 and attack < len(env):
        # Raised cosine for very smooth attack
        x = np.linspace(0, np.pi, attack)
        env[:attack] = 0.5 * (1 - np.cos(x))
    if attack + decay < len(env):
        env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    if release > 0 and release < len(env):
        env[-release:] = np.linspace(sustain_level, 0, release)

    wave *= env * 0.4

    return wave

def generate_bright_synth(freq, duration, sample_rate=SAMPLE_RATE):
    """Generate bright synth with SOFT ATTACK for tempo stability.
    Adds harmonics for brightness while maintaining smooth onset."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Additive synthesis with more high harmonics for brightness
    wave = np.sin(2 * np.pi * freq * t) * 0.8

    # Add more harmonics for brightness
    for harmonic in range(2, 10):
        harmonic_freq = freq * harmonic
        # Boost harmonics in brightness range
        if 1000 <= harmonic_freq <= 4000:
            amp = 0.6 / harmonic
        else:
            amp = 0.3 / harmonic
        wave += np.sin(2 * np.pi * harmonic_freq * t) * amp

    # Gentle HP to add clarity
    hp_cutoff = 300 / (sample_rate / 2)
    b, a = signal.butter(2, hp_cutoff, 'high')
    wave = signal.filtfilt(b, a, wave)

    # Gentle LP
    lp_cutoff = min(freq * 10, 10000) / (sample_rate / 2)
    b, a = signal.butter(2, lp_cutoff, 'low')
    wave = signal.filtfilt(b, a, wave)

    # SOFT ATTACK envelope
    attack = int(sample_rate * 0.05)  # 50ms attack
    decay = int(sample_rate * 0.08)
    sustain_level = 0.55
    release = int(sample_rate * 0.12)

    env = np.ones(len(t)) * sustain_level
    if attack > 0 and attack < len(env):
        x = np.linspace(0, np.pi, attack)
        env[:attack] = 0.5 * (1 - np.cos(x))
    if attack + decay < len(env):
        env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    if release > 0 and release < len(env):
        env[-release:] = np.linspace(sustain_level, 0, release)

    wave *= env * 0.5

    return wave

def render_pattern(events, bar_duration, num_bars, sound_type='synth', sample_rate=SAMPLE_RATE):
    """Render a single-bar pattern repeated for num_bars."""
    total_duration = bar_duration * num_bars
    total_samples = int(sample_rate * total_duration)
    audio = np.zeros(total_samples)

    for bar in range(num_bars):
        bar_offset = bar * bar_duration

        for event in events:
            event_time = bar_offset + event['time'] * bar_duration
            event_samples = int(event_time * sample_rate)

            if event_samples >= total_samples:
                continue

            note = event['note'].lower()
            duration = event['duration'] * bar_duration

            # Generate appropriate sound
            if note in ['bd', 'kick']:
                sound = generate_kick(min(duration, 0.3))
            elif note in ['sd', 'snare']:
                sound = generate_snare(min(duration, 0.2))
            elif note == 'hh':
                sound = generate_hihat(min(duration, 0.1), is_open=False)
            elif note == 'oh':
                sound = generate_hihat(min(duration, 0.3), is_open=True)
            elif sound_type == 'bass':
                freq = note_to_freq(note)
                sound = generate_bass(freq, min(duration, 0.5))
            elif sound_type == 'synth_bright':
                freq = note_to_freq(note)
                sound = generate_bright_synth(freq, duration)
            else:
                freq = note_to_freq(note)
                sound = generate_synth(freq, duration)

            # Add to output
            end_sample = min(event_samples + len(sound), total_samples)
            available = end_sample - event_samples
            audio[event_samples:end_sample] += sound[:available]

    return audio


def render_bar_array(bar_patterns, bar_duration, sound_type='synth', sample_rate=SAMPLE_RATE, timing_jitter=0.015):
    """Render a bar array - each pattern string is one bar, played sequentially.

    Args:
        timing_jitter: Random timing variation as fraction of beat (0.015 = 15ms at 60 BPM)
    """
    num_bars = len(bar_patterns)
    total_duration = bar_duration * num_bars
    total_samples = int(sample_rate * total_duration)
    audio = np.zeros(total_samples)

    # Pre-seed random for reproducibility but still variable
    np.random.seed(int(bar_duration * 1000) % 12345)

    for bar_idx, pattern in enumerate(bar_patterns):
        bar_offset = bar_idx * bar_duration
        events = parse_mini_notation(pattern)

        for event in events:
            # Add humanizing timing jitter (±jitter_ms around the grid)
            jitter = (np.random.random() - 0.5) * 2 * timing_jitter * (bar_duration / 4)
            event_time = bar_offset + event['time'] * bar_duration + jitter
            event_time = max(0, event_time)  # Don't go negative
            event_samples = int(event_time * sample_rate)

            if event_samples >= total_samples:
                continue

            note = event['note'].lower()
            duration = event['duration'] * bar_duration

            # Generate appropriate sound
            if note in ['bd', 'kick']:
                sound = generate_kick(min(duration, 0.3))
            elif note in ['sd', 'snare']:
                sound = generate_snare(min(duration, 0.2))
            elif note == 'hh':
                sound = generate_hihat(min(duration, 0.1), is_open=False)
            elif note == 'oh':
                sound = generate_hihat(min(duration, 0.3), is_open=True)
            elif sound_type == 'bass':
                freq = note_to_freq(note)
                sound = generate_bass(freq, min(duration, 0.5))
            elif sound_type == 'synth_bright':
                freq = note_to_freq(note)
                sound = generate_bright_synth(freq, duration)
            else:
                freq = note_to_freq(note)
                sound = generate_synth(freq, duration)

            # Add to output
            end_sample = min(event_samples + len(sound), total_samples)
            available = end_sample - event_samples
            audio[event_samples:end_sample] += sound[:available]

    return audio, num_bars

def render_strudel_to_wav(code, output_path, duration_seconds=None, mix_overrides=None):
    """Render Strudel code to WAV file."""
    parsed = parse_strudel_code(code)
    bpm = parsed['bpm']
    mix = parsed['mix']
    bar_arrays = parsed.get('bar_arrays', {})

    # Apply any mix overrides (from comparison feedback)
    if mix_overrides:
        mix.update(mix_overrides)

    # Calculate bar duration
    bar_duration = 60.0 / bpm * 4  # 4 beats per bar

    # Calculate num_bars from bar arrays or duration
    if bar_arrays:
        # Use the longest bar array as reference
        num_bars = max(len(bars) for bars in bar_arrays.values())
    elif duration_seconds:
        num_bars = math.ceil(duration_seconds / bar_duration)
    else:
        num_bars = parsed.get('duration_bars', 16)

    # Override with explicit duration if provided
    if duration_seconds:
        num_bars = math.ceil(duration_seconds / bar_duration)

    total_duration = bar_duration * num_bars
    total_samples = int(SAMPLE_RATE * total_duration)

    print(f"Rendering: BPM={bpm}, {num_bars} bars, {total_duration:.1f}s")
    print(f"Mix levels: kick={mix['kick_gain']:.2f}, bass={mix['bass_gain']:.2f}, hh={mix['hh_gain']:.2f}, vox={mix['vox_gain']:.2f}")

    # Initialize stereo output
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    # First, render bar arrays (the modern format)
    for name, bars in bar_arrays.items():
        # Determine sound type and gain based on pattern name
        if name == 'drums':
            sound_type = 'drum'
            gain = mix['kick_gain']
            pan_l, pan_r = 1.0, 1.0
        elif name == 'bass':
            sound_type = 'bass'
            gain = mix['bass_gain']
            pan_l, pan_r = 1.0, 1.0
        elif name == 'mid':
            sound_type = 'synth'
            gain = mix['vox_gain']
            pan_l, pan_r = 1.1, 0.9
        elif name == 'high':
            sound_type = 'synth_bright'
            gain = mix['lead_gain']
            pan_l, pan_r = 0.8, 1.2
        elif 'kick' in name:
            sound_type = 'drum'
            gain = mix['kick_gain']
            pan_l, pan_r = 1.0, 1.0
        elif 'snare' in name:
            sound_type = 'drum'
            gain = mix['snare_gain']
            pan_l, pan_r = 1.0, 1.0
        elif 'hh' in name or 'hat' in name:
            sound_type = 'drum'
            gain = mix['hh_gain']
            pan_l, pan_r = 0.8, 1.2
        elif 'vox' in name or 'vocal' in name:
            sound_type = 'synth'
            gain = mix['vox_gain']
            pan_l, pan_r = 1.1, 0.9
        elif 'stab' in name:
            sound_type = 'synth'
            gain = mix['stab_gain']
            pan_l, pan_r = 0.9, 1.1
        elif 'lead' in name:
            sound_type = 'synth'
            gain = mix['lead_gain']
            pan_l, pan_r = 0.8, 1.2
        else:
            sound_type = 'synth'
            gain = mix['vox_gain']
            pan_l, pan_r = 1.0, 1.0

        # Extend bars to match num_bars by looping
        extended_bars = []
        for i in range(num_bars):
            extended_bars.append(bars[i % len(bars)])

        # Render the bar array
        audio, _ = render_bar_array(extended_bars, bar_duration, sound_type)

        # Pad or trim to match total_samples
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)))
        else:
            audio = audio[:total_samples]

        left += audio * gain * pan_l
        right += audio * gain * pan_r

    # Also render old-style patterns (single string patterns)
    patterns = parsed['patterns']
    for name, pattern in patterns.items():
        events = parse_mini_notation(pattern)
        if not events:
            continue

        # Skip if this pattern was already rendered as bar array
        if name in bar_arrays:
            continue

        # Determine sound type and panning based on pattern name
        if 'kick' in name or name.startswith('kick'):
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            gain = mix['kick_gain']
            left += audio * gain
            right += audio * gain
        elif 'snare' in name or name.startswith('snare'):
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            gain = mix['snare_gain']
            left += audio * gain
            right += audio * gain
        elif 'hh' in name or 'hat' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            gain = mix['hh_gain']
            left += audio * gain * 0.8
            right += audio * gain * 1.2
        elif name == 'drums':
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            gain = mix['kick_gain']
            left += audio * gain
            right += audio * gain
        elif name == 'bass' or 'bass' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'bass')
            gain = mix['bass_gain']
            left += audio * gain
            right += audio * gain
        elif name == 'mid':
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            gain = mix['vox_gain']
            left += audio * gain * 1.1
            right += audio * gain * 0.9
        elif name == 'high':
            audio = render_pattern(events, bar_duration, num_bars, 'synth_bright')
            gain = mix['lead_gain']
            left += audio * gain * 0.8
            right += audio * gain * 1.2
        elif 'vox' in name or 'vocal' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            gain = mix['vox_gain']
            left += audio * gain * 1.1
            right += audio * gain * 0.9
        elif 'stab' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            gain = mix['stab_gain']
            left += audio * gain * 0.9
            right += audio * gain * 1.1
        elif 'lead' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            gain = mix['lead_gain']
            left += audio * gain * 0.8
            right += audio * gain * 1.2

    # Mix to stereo
    stereo = np.column_stack([left, right])

    # Simple normalization - no aggressive filtering or compression
    max_val = np.max(np.abs(stereo))
    if max_val > 0:
        stereo = stereo / max_val * 0.9

    # Convert to 16-bit
    stereo_int = (stereo * 32767).astype(np.int16)

    # Write WAV
    wavfile.write(output_path, SAMPLE_RATE, stereo_int)
    print(f"Written: {output_path}")

    return output_path

def load_comparison_feedback(cache_dir):
    """Load previous comparison results to adjust mix."""
    feedback_file = os.path.join(cache_dir, 'comparison_feedback.json')
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description='Render Strudel patterns to WAV')
    parser.add_argument('input', help='Input Strudel file or - for stdin')
    parser.add_argument('-o', '--output', default='output.wav', help='Output WAV file')
    parser.add_argument('-d', '--duration', type=float, help='Duration in seconds')
    parser.add_argument('--feedback', help='Path to comparison feedback JSON')
    parser.add_argument('--bass-gain', type=float, help='Override bass gain (0-1)')
    parser.add_argument('--hh-gain', type=float, help='Override hi-hat gain (0-1)')
    parser.add_argument('--vox-gain', type=float, help='Override vox gain (0-1)')

    args = parser.parse_args()

    # Read input
    if args.input == '-':
        import sys
        code = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            code = f.read()

    # Build mix overrides from command line
    mix_overrides = {}
    if args.bass_gain is not None:
        mix_overrides['bass_gain'] = args.bass_gain
    if args.hh_gain is not None:
        mix_overrides['hh_gain'] = args.hh_gain
    if args.vox_gain is not None:
        mix_overrides['vox_gain'] = args.vox_gain

    # Load AI-derived feedback if provided
    if args.feedback and os.path.exists(args.feedback):
        print(f"Loading AI parameters from: {args.feedback}")
        with open(args.feedback, 'r') as f:
            feedback = json.load(f)
            # Use the renderer_mix from AI analysis
            if 'renderer_mix' in feedback:
                ai_mix = feedback['renderer_mix']
                print(f"AI-derived mix levels: {ai_mix}")
                for key, value in ai_mix.items():
                    mix_overrides[key] = value

    render_strudel_to_wav(code, args.output, args.duration, mix_overrides if mix_overrides else None)

if __name__ == '__main__':
    main()
