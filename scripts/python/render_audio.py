#!/usr/bin/env python3
"""
Render Strudel-style patterns to WAV audio.
Synthesizes drums, bass, and melodic content.
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

def parse_strudel_code(code):
    """Extract patterns and BPM from Strudel code."""
    result = {
        'bpm': 120,
        'patterns': {},
        'duration_bars': 16
    }

    # Extract BPM from setcps
    cps_match = re.search(r'setcps\((\d+)/60/4\)', code)
    if cps_match:
        result['bpm'] = int(cps_match.group(1))

    # Extract let patterns
    let_patterns = re.findall(r'let\s+(\w+)\s*=\s*["`]([^"`]+)["`]', code)
    for name, pattern in let_patterns:
        result['patterns'][name] = pattern

    return result

def parse_mini_notation(pattern, steps=16):
    """Parse mini-notation pattern into list of events."""
    events = []

    # Remove bar separators
    pattern = pattern.replace('|', ' ')

    # Split by whitespace
    tokens = pattern.split()

    if not tokens:
        return events

    step_duration = 1.0 / len(tokens)

    for i, token in enumerate(tokens):
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
    """Generate a kick drum sound."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Pitch envelope (starts high, drops quickly)
    pitch_env = 150 * np.exp(-30 * t) + 50

    # Generate sine with pitch envelope
    phase = np.cumsum(pitch_env) / sample_rate * 2 * np.pi
    kick = np.sin(phase)

    # Amplitude envelope
    amp_env = np.exp(-8 * t)
    kick *= amp_env

    # Add some click/attack
    click = np.random.randn(int(sample_rate * 0.005)) * 0.3
    kick[:len(click)] += click[:len(kick[:len(click)])]

    # Distortion
    kick = np.tanh(kick * 2) * 0.8

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

    hihat = noise * env * 0.3

    return hihat

def generate_bass(freq, duration, sample_rate=SAMPLE_RATE):
    """Generate a bass sound (sawtooth with filter)."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Sawtooth wave
    saw = 2 * (t * freq % 1) - 1

    # Add sub-octave
    sub = np.sin(2 * np.pi * freq * t)

    bass = saw * 0.6 + sub * 0.4

    # Low-pass filter
    cutoff = min(freq * 3, 500)
    b, a = signal.butter(2, cutoff / (sample_rate / 2), 'low')
    bass = signal.filtfilt(b, a, bass)

    # Envelope
    attack = int(sample_rate * 0.005)
    release = int(sample_rate * 0.1)
    env = np.ones(len(t))
    env[:attack] = np.linspace(0, 1, attack)
    if len(env) > release:
        env[-release:] = np.linspace(1, 0, release)

    bass *= env
    bass = np.tanh(bass * 2) * 0.7  # Distortion

    return bass

def generate_synth(freq, duration, waveform='square', sample_rate=SAMPLE_RATE):
    """Generate a synth sound."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    if waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == 'saw':
        wave = 2 * (t * freq % 1) - 1
    else:  # sine
        wave = np.sin(2 * np.pi * freq * t)

    # Low-pass filter
    cutoff = min(freq * 4, 4000)
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
            else:
                freq = note_to_freq(note)
                sound = generate_synth(freq, duration)

            # Add to output
            end_sample = min(event_samples + len(sound), total_samples)
            available = end_sample - event_samples
            audio[event_samples:end_sample] += sound[:available]

    return audio

def render_strudel_to_wav(code, output_path, duration_seconds=None):
    """Render Strudel code to WAV file."""
    parsed = parse_strudel_code(code)
    bpm = parsed['bpm']

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

    # Initialize stereo output
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    patterns = parsed['patterns']

    # Render each pattern type
    for name, pattern in patterns.items():
        events = parse_mini_notation(pattern)
        if not events:
            continue

        # Determine sound type and panning
        if 'kick' in name or name.startswith('kick'):
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            left += audio * 0.5
            right += audio * 0.5
        elif 'snare' in name or name.startswith('snare'):
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            left += audio * 0.5
            right += audio * 0.5
        elif 'hh' in name or 'hat' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'drum')
            left += audio * 0.4
            right += audio * 0.6
        elif 'bass' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'bass')
            left += audio * 0.5
            right += audio * 0.5
        elif 'vox' in name or 'vocal' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            left += audio * 0.6
            right += audio * 0.4
        elif 'stab' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            left += audio * 0.45
            right += audio * 0.55
        elif 'lead' in name:
            audio = render_pattern(events, bar_duration, num_bars, 'synth')
            left += audio * 0.4
            right += audio * 0.6

    # Mix to stereo
    stereo = np.column_stack([left, right])

    # Normalize
    max_val = np.max(np.abs(stereo))
    if max_val > 0:
        stereo = stereo / max_val * 0.9

    # Convert to 16-bit
    stereo_int = (stereo * 32767).astype(np.int16)

    # Write WAV
    wavfile.write(output_path, SAMPLE_RATE, stereo_int)
    print(f"Written: {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Render Strudel patterns to WAV')
    parser.add_argument('input', help='Input Strudel file or - for stdin')
    parser.add_argument('-o', '--output', default='output.wav', help='Output WAV file')
    parser.add_argument('-d', '--duration', type=float, help='Duration in seconds')

    args = parser.parse_args()

    # Read input
    if args.input == '-':
        import sys
        code = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            code = f.read()

    render_strudel_to_wav(code, args.output, args.duration)

if __name__ == '__main__':
    main()
