#!/usr/bin/env python3
"""
Render Strudel-style patterns to WAV audio.
Synthesizes drums, bass, and melodic content.
Now with adjustable mix parameters based on analysis.
"""

import argparse
import json
import re
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os

# Audio settings
SAMPLE_RATE = 44100

# Default mix levels - tuned to match original frequency distribution
# Voice naming is confusing: "mid" voice has low-mid notes, "high" voice has mid notes
# Original: 7% bass, 20% low-mid, 48% mid, 13% high-mid, 5% high
DEFAULT_MIX = {
    'kick_gain': 0.15,    # Minimal drums
    'snare_gain': 0.2,
    'hh_gain': 0.25,
    'bass_gain': 0.08,    # Very low bass
    'vox_gain': 0.3,      # "mid" voice produces 200-500Hz (reduce this)
    'stab_gain': 0.3,
    'lead_gain': 1.2,     # "high" voice produces 500-2000Hz (BOOST - this is the real mid!)
}

def parse_strudel_code(code):
    """Extract patterns and BPM from Strudel code."""
    result = {
        'bpm': 120,
        'patterns': {},
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
        # Extract all quoted strings from the array
        patterns = re.findall(r'"([^"]*)"', array_content)
        if patterns:
            # Join all bar patterns into one continuous pattern
            result['patterns'][name] = ' | '.join(patterns)

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
    """Generate a kick drum sound - tuned for less sub-bass to match piano stems."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Pitch envelope (starts higher, base pitch raised to reduce sub-bass)
    pitch_env = 200 * np.exp(-40 * t) + 80  # Higher base pitch (80Hz vs 50Hz)

    # Generate sine with pitch envelope
    phase = np.cumsum(pitch_env) / sample_rate * 2 * np.pi
    kick = np.sin(phase)

    # Amplitude envelope - shorter decay
    amp_env = np.exp(-12 * t)
    kick *= amp_env

    # Add click/attack for presence
    click = np.random.randn(int(sample_rate * 0.003)) * 0.4
    kick[:len(click)] += click[:len(kick[:len(click)])]

    # High-pass filter to remove sub-bass
    hpf_cutoff = 60 / (sample_rate / 2)
    b_hp, a_hp = signal.butter(2, hpf_cutoff, 'high')
    kick = signal.filtfilt(b_hp, a_hp, kick)

    # Light distortion
    kick = np.tanh(kick * 1.5) * 0.5  # Reduced gain

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
    """Generate a bass sound (sawtooth with filter) - filtered to reduce sub-bass."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Sawtooth wave only - no sub-octave to reduce low frequencies
    saw = 2 * (t * freq % 1) - 1
    bass = saw * 0.8

    # Band-pass filter - remove sub-bass below 80Hz, cap at 250Hz
    # This matches original stems which have almost no sub-bass energy
    low_cutoff = 80 / (sample_rate / 2)
    high_cutoff = min(freq * 2, 250) / (sample_rate / 2)
    # Ensure cutoffs are valid (between 0 and 1)
    low_cutoff = max(0.001, min(0.99, low_cutoff))
    high_cutoff = max(low_cutoff + 0.01, min(0.99, high_cutoff))
    b, a = signal.butter(2, [low_cutoff, high_cutoff], 'band')
    bass = signal.filtfilt(b, a, bass)

    # Envelope
    attack = int(sample_rate * 0.005)
    release = int(sample_rate * 0.1)
    env = np.ones(len(t))
    env[:attack] = np.linspace(0, 1, attack)
    if len(env) > release:
        env[-release:] = np.linspace(1, 0, release)

    bass *= env
    bass = np.tanh(bass * 1.2) * 0.3  # Reduced gain further

    return bass

def generate_synth(freq, duration, waveform='square', sample_rate=SAMPLE_RATE):
    """Generate a synth sound - with harmonics for brighter mid-range content."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate base waveform (keep fundamental)
    if waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == 'saw' or waveform == 'sawtooth':
        wave = 2 * (t * freq % 1) - 1
    else:  # sine with added harmonics for brightness
        wave = np.sin(2 * np.pi * freq * t)
        # Add harmonics to push energy into mid range (without filtering fundamental)
        wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)   # 2nd harmonic
        wave += 0.35 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic
        wave += 0.25 * np.sin(2 * np.pi * freq * 4 * t)  # 4th harmonic
        wave += 0.15 * np.sin(2 * np.pi * freq * 5 * t)  # 5th harmonic

    # Gentle low-pass filter (high cutoff to preserve brightness)
    cutoff = min(freq * 8, 8000)
    b, a = signal.butter(2, cutoff / (sample_rate / 2), 'low')
    wave = signal.filtfilt(b, a, wave)

    # ADSR envelope
    attack = int(sample_rate * 0.01)
    decay = int(sample_rate * 0.05)
    sustain_level = 0.6
    release = int(sample_rate * 0.05)

    env = np.ones(len(t)) * sustain_level
    if attack > 0 and attack < len(env):
        env[:attack] = np.linspace(0, 1, attack)
    if attack + decay < len(env):
        env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    if release > 0 and release < len(env):
        env[-release:] = np.linspace(sustain_level, 0, release)

    wave *= env * 0.4

    return wave

def generate_bright_synth(freq, duration, sample_rate=SAMPLE_RATE):
    """Generate a bright synth sound - sawtooth with more harmonics."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Sawtooth is naturally bright with rich harmonics
    wave = 2 * (t * freq % 1) - 1

    # Higher cutoff for brighter sound (but still keep fundamental)
    cutoff = min(freq * 6, 8000)
    b, a = signal.butter(2, cutoff / (sample_rate / 2), 'low')
    wave = signal.filtfilt(b, a, wave)

    # ADSR envelope - slightly faster attack for brightness
    attack = int(sample_rate * 0.005)
    decay = int(sample_rate * 0.03)
    sustain_level = 0.7
    release = int(sample_rate * 0.05)

    env = np.ones(len(t)) * sustain_level
    if attack > 0 and attack < len(env):
        env[:attack] = np.linspace(0, 1, attack)
    if attack + decay < len(env):
        env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    if release > 0 and release < len(env):
        env[-release:] = np.linspace(sustain_level, 0, release)

    wave *= env * 0.4

    return wave

def render_pattern(events, bar_duration, num_bars, sound_type='synth', sample_rate=SAMPLE_RATE):
    """Render a pattern to audio."""
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

def render_strudel_to_wav(code, output_path, duration_seconds=None, mix_overrides=None):
    """Render Strudel code to WAV file."""
    parsed = parse_strudel_code(code)
    bpm = parsed['bpm']
    mix = parsed['mix']

    # Apply any mix overrides (from comparison feedback)
    if mix_overrides:
        mix.update(mix_overrides)

    # Calculate bar duration
    bar_duration = 60.0 / bpm * 4  # 4 beats per bar

    # Default to 16 bars if duration not specified
    if duration_seconds:
        num_bars = int(duration_seconds / bar_duration)
    else:
        num_bars = 16

    total_duration = bar_duration * num_bars
    total_samples = int(SAMPLE_RATE * total_duration)

    print(f"Rendering: BPM={bpm}, {num_bars} bars, {total_duration:.1f}s")
    print(f"Mix levels: kick={mix['kick_gain']:.2f}, bass={mix['bass_gain']:.2f}, hh={mix['hh_gain']:.2f}, vox={mix['vox_gain']:.2f}")

    # Initialize stereo output
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    patterns = parsed['patterns']

    # Render each pattern type with appropriate gain
    for name, pattern in patterns.items():
        events = parse_mini_notation(pattern)
        if not events:
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
            # Combined drum pattern with bd, sd, hh, oh
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            gain = mix['kick_gain']  # Use kick gain for combined drums
            left += audio * gain
            right += audio * gain
        elif name == 'bass' or 'bass' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'bass')
            gain = mix['bass_gain']
            left += audio * gain
            right += audio * gain
        elif name == 'mid':
            # Mid voice - use synth with mid-range focus
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            gain = mix['vox_gain']  # Use vox gain for mids
            left += audio * gain * 1.1
            right += audio * gain * 0.9
        elif name == 'high':
            # High voice - use bright synth
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
