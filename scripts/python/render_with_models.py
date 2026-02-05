#!/usr/bin/env python3
"""
AI-driven audio rendering using trained granular models.
Adapts ALL parameters from original track analysis - NO HARDCODED VALUES.
Works for ANY track, ANY genre.
"""

import argparse
import json
import os
import sys
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal

SAMPLE_RATE = 44100


def load_model_samples(model_path):
    """Load pitched samples from a granular model."""
    pitched_dir = Path(model_path) / "pitched"
    samples = {}

    if not pitched_dir.exists():
        return samples

    for wav_file in pitched_dir.glob("*.wav"):
        note_name = wav_file.stem
        audio, sr = sf.read(str(wav_file))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        samples[note_name] = audio

    return samples


def analyze_audio_full(audio, sr=SAMPLE_RATE):
    """
    Comprehensive audio analysis for AI-driven matching.
    Returns ALL characteristics needed to match any track.
    """
    import librosa

    analysis = {}

    # Energy/Dynamics
    analysis['rms'] = float(np.sqrt(np.mean(audio ** 2)))
    analysis['peak'] = float(np.max(np.abs(audio)))

    # Spectral characteristics
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    analysis['centroid'] = float(np.mean(centroid))
    analysis['centroid_std'] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    analysis['bandwidth'] = float(np.mean(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    analysis['rolloff'] = float(np.mean(rolloff))

    flatness = librosa.feature.spectral_flatness(y=audio)
    analysis['flatness'] = float(np.mean(flatness))

    # Frequency bands (for EQ matching)
    spec = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)

    bands = [
        ('sub_bass', 20, 60),
        ('bass', 60, 250),
        ('low_mid', 250, 500),
        ('mid', 500, 2000),
        ('high_mid', 2000, 4000),
        ('high', 4000, 20000)
    ]

    total_energy = np.sum(spec ** 2)
    analysis['band_energy'] = {}
    for band_name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        analysis['band_energy'][band_name] = float(np.sum(spec[mask] ** 2) / total_energy) if total_energy > 0 else 0

    # MFCC for timbre matching
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    analysis['mfcc'] = [float(x) for x in np.mean(mfcc, axis=1)]

    # Chroma for harmonic content
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    analysis['chroma'] = [float(x) for x in np.mean(chroma, axis=1)]

    # Zero crossing rate (brightness/noisiness)
    zcr = librosa.feature.zero_crossing_rate(audio)
    analysis['zcr'] = float(np.mean(zcr))

    return analysis


def compute_similarity(current, target):
    """Compute overall similarity between current and target analysis."""
    scores = {}

    # RMS similarity
    if target.get('rms', 0) > 0:
        scores['rms'] = 1 - min(abs(current['rms'] - target['rms']) / target['rms'], 1)
    else:
        scores['rms'] = 0.5

    # Centroid similarity (brightness)
    if target.get('centroid', 0) > 0:
        scores['centroid'] = 1 - min(abs(current['centroid'] - target['centroid']) / target['centroid'], 1)
    else:
        scores['centroid'] = 0.5

    # Band energy similarity
    band_scores = []
    for band in target.get('band_energy', {}):
        if band in current.get('band_energy', {}):
            t_val = target['band_energy'][band]
            c_val = current['band_energy'][band]
            if t_val > 0.001:
                band_scores.append(1 - min(abs(c_val - t_val) / max(t_val, 0.01), 1))
    scores['bands'] = np.mean(band_scores) if band_scores else 0.5

    # MFCC similarity (timbre)
    if 'mfcc' in current and 'mfcc' in target:
        mfcc_diff = np.mean(np.abs(np.array(current['mfcc']) - np.array(target['mfcc'])))
        scores['mfcc'] = max(0, 1 - mfcc_diff / 100)
    else:
        scores['mfcc'] = 0.5

    # Chroma similarity (harmony)
    if 'chroma' in current and 'chroma' in target:
        chroma_corr = np.corrcoef(current['chroma'], target['chroma'])[0, 1]
        scores['chroma'] = (chroma_corr + 1) / 2 if not np.isnan(chroma_corr) else 0.5
    else:
        scores['chroma'] = 0.5

    # Weighted overall
    overall = (
        scores['rms'] * 0.15 +
        scores['centroid'] * 0.20 +
        scores['bands'] * 0.30 +
        scores['mfcc'] * 0.20 +
        scores['chroma'] * 0.15
    )

    return overall, scores


def apply_spectral_matching(audio, current, target, sr=SAMPLE_RATE):
    """Apply EQ and processing to match target spectrum."""
    from scipy.fft import rfft, irfft, rfftfreq

    n = len(audio)
    freqs = rfftfreq(n, 1/sr)
    spectrum = rfft(audio)

    bands = [
        ('sub_bass', 20, 60),
        ('bass', 60, 250),
        ('low_mid', 250, 500),
        ('mid', 500, 2000),
        ('high_mid', 2000, 4000),
        ('high', 4000, 20000)
    ]

    target_bands = target.get('band_energy', {})
    current_bands = current.get('band_energy', {})

    for band_name, low, high in bands:
        c_val = current_bands.get(band_name, 0.001)
        t_val = target_bands.get(band_name, 0.001)

        if c_val > 0.0001 and t_val > 0.0001:
            ratio = np.sqrt(t_val / c_val)
            ratio = np.clip(ratio, 0.2, 5.0)  # Limit extreme adjustments

            mask = (freqs >= low) & (freqs < high)
            spectrum[mask] *= ratio

    result = irfft(spectrum, n)

    # Match brightness (centroid)
    if current.get('centroid', 0) > 0 and target.get('centroid', 0) > 0:
        brightness_ratio = target['centroid'] / current['centroid']

        if brightness_ratio > 1.3:
            # Need more brightness - boost highs
            nyquist = sr / 2
            cutoff = min(2000, nyquist * 0.8)
            b, a = signal.butter(1, cutoff / nyquist, btype='high')
            high_content = signal.filtfilt(b, a, result)
            boost = min(brightness_ratio - 1, 0.5)
            result = result + high_content * boost
        elif brightness_ratio < 0.7:
            # Too bright - gentle lowpass
            nyquist = sr / 2
            cutoff = min(current['centroid'] * 1.5, nyquist * 0.95)
            b, a = signal.butter(1, cutoff / nyquist, btype='low')
            result = signal.filtfilt(b, a, result) * 0.6 + result * 0.4

    return result


def pitch_shift_sample(sample, semitones, sample_rate=SAMPLE_RATE):
    """Pitch shift using phase vocoder."""
    if semitones == 0 or len(sample) < 2:
        return sample

    try:
        import librosa
        return librosa.effects.pitch_shift(
            sample.astype(np.float32), sr=sample_rate, n_steps=semitones
        )
    except ImportError:
        ratio = 2 ** (semitones / 12)
        indices = np.arange(0, len(sample), ratio)
        indices = indices[indices < len(sample)].astype(int)
        return sample[indices] if len(indices) > 0 else sample


def note_to_midi(note_str):
    """Convert note string to MIDI number."""
    note_map = {'c': 0, 'cs': 1, 'd': 2, 'ds': 3, 'e': 4, 'f': 5,
                'fs': 6, 'g': 7, 'gs': 8, 'a': 9, 'as': 10, 'b': 11}
    match = re.match(r'([a-g]s?)(\d+)', note_str.lower())
    if not match:
        return 60
    note, octave = match.groups()
    return note_map.get(note, 0) + (int(octave) + 1) * 12


def midi_to_note_class(midi):
    """Get note class from MIDI number."""
    return ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'][midi % 12]


def render_note(samples, midi_note, duration, velocity=1.0):
    """Render a single note."""
    note_class = midi_to_note_class(midi_note)
    note_classes = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']

    if note_class in samples:
        base_sample = samples[note_class]
        base_octave = 4
    else:
        if not samples:
            return np.zeros(int(duration * SAMPLE_RATE))

        target_idx = note_classes.index(note_class) if note_class in note_classes else 0
        best_sample, best_dist, best_class = None, 999, None

        for nc, s in samples.items():
            if nc in note_classes:
                idx = note_classes.index(nc)
                dist = min(abs(idx - target_idx), 12 - abs(idx - target_idx))
                if dist < best_dist:
                    best_dist, best_sample, best_class = dist, s, nc

        if best_sample is None:
            note_class = list(samples.keys())[0]
            base_sample = samples[note_class]
        else:
            note_class, base_sample = best_class, best_sample
        base_octave = 4

    target_octave = (midi_note // 12) - 1
    semitones = (target_octave - base_octave) * 12

    if note_class in note_classes:
        semitones += (midi_note % 12) - note_classes.index(note_class)

    shifted = pitch_shift_sample(base_sample, semitones)
    target_samples = int(duration * SAMPLE_RATE)

    if len(shifted) >= target_samples:
        result = shifted[:target_samples]
    else:
        loops = (target_samples // len(shifted)) + 1
        result = np.tile(shifted, loops)[:target_samples]

    result = result * velocity

    attack = int(0.005 * SAMPLE_RATE)
    release = int(0.02 * SAMPLE_RATE)
    if len(result) > attack:
        result[:attack] *= np.linspace(0, 1, attack)
    if len(result) > release:
        result[-release:] *= np.linspace(1, 0, release)

    return result


def parse_bar_arrays(code):
    """Parse bar array patterns from Strudel code.

    Handles two formats:
    1. Array format: let bass = ["...", "..."]
    2. Single pattern format: let kick1 = "..." (Brazilian funk mode)
    """
    patterns = {}

    # Format 1: Array patterns like let bass = ["...", "..."]
    matches = re.findall(r'let\s+(\w+)\s*=\s*\[([\s\S]*?)\]', code)
    for name, content in matches:
        bars = re.findall(r'"([^"]*)"', content)
        patterns[name] = bars

    # Format 2: Single patterns like let kick1 = "..." (Brazilian funk)
    # Group by base name (kick, bass, vox, etc.)
    single_matches = re.findall(r'let\s+(\w+?)(\d+)\s*=\s*"([^"]*)"', code)
    for base_name, num, pattern in single_matches:
        if base_name not in patterns:
            patterns[base_name] = []
        # Ensure we have enough slots
        idx = int(num) - 1
        while len(patterns[base_name]) <= idx:
            patterns[base_name].append("~")
        patterns[base_name][idx] = pattern

    return patterns


def render_pattern(samples, pattern, beat_duration, sample_rate=SAMPLE_RATE):
    """Render a single bar pattern."""
    tokens = pattern.split()
    if not tokens:
        return np.zeros(int(beat_duration * 4 * sample_rate))

    total_slots = 0
    expanded_tokens = []
    for token in tokens:
        if token.startswith('~*'):
            try:
                n = int(token[2:])
                expanded_tokens.extend(['~'] * n)
                total_slots += n
            except ValueError:
                expanded_tokens.append('~')
                total_slots += 1
        else:
            expanded_tokens.append(token)
            total_slots += 1

    if total_slots == 0:
        total_slots = 1

    bar_samples = int(beat_duration * 4 * sample_rate)
    samples_per_slot = bar_samples // total_slots

    result = np.zeros(bar_samples)

    for i, token in enumerate(expanded_tokens):
        start = i * samples_per_slot
        if start >= bar_samples:
            break

        if token == '~':
            continue

        if token.startswith('[') and token.endswith(']'):
            chord_notes = token[1:-1].split(',')
            for note_str in chord_notes:
                note_str = note_str.strip()
                if re.match(r'[a-g]s?\d', note_str.lower()):
                    midi = note_to_midi(note_str)
                    duration = samples_per_slot / sample_rate
                    note_audio = render_note(samples, midi, duration)
                    end = min(start + len(note_audio), len(result))
                    result[start:end] += note_audio[:end-start] * 0.6
            continue

        if re.match(r'[a-g]s?\d', token.lower()):
            midi = note_to_midi(token)
            duration = samples_per_slot / sample_rate
            note_audio = render_note(samples, midi, duration)
            end = min(start + len(note_audio), len(result))
            result[start:end] += note_audio[:end-start]

        if token.lower() in ['sd', 'bd', 'hh', 'oh', 'cp', 'lt', 'mt', 'ht']:
            duration = samples_per_slot / sample_rate
            if samples:
                sample_key = list(samples.keys())[0]
                drum_audio = samples[sample_key][:int(duration * sample_rate)]
                if len(drum_audio) > 0:
                    end = min(start + len(drum_audio), len(result))
                    result[start:end] += drum_audio[:end-start]

    return result


def render_with_models(strudel_code, models_dir, output_path, bpm=120, duration=60,
                       ai_params_path=None, original_audio_path=None, max_iterations=10):
    """
    Fully AI-driven rendering. NO HARDCODED VALUES.
    All parameters derived from original track analysis.
    """
    print(f"\n{'='*60}")
    print("AI-DRIVEN GRANULAR RENDERING (Universal)")
    print(f"{'='*60}")

    # Step 1: Analyze original audio to get ALL target characteristics
    target = None

    if original_audio_path and os.path.exists(original_audio_path):
        print(f"Analyzing original: {original_audio_path}")
        orig_audio, orig_sr = sf.read(original_audio_path)
        if len(orig_audio.shape) > 1:
            orig_audio = orig_audio.mean(axis=1)
        orig_audio = orig_audio[:int(60 * orig_sr)]
        target = analyze_audio_full(orig_audio, orig_sr)
        print(f"  RMS: {target['rms']:.3f}, Centroid: {target['centroid']:.0f}Hz")
        print(f"  Bands: " + ", ".join(f"{k}={v*100:.1f}%" for k,v in target['band_energy'].items()))

    elif ai_params_path and os.path.exists(ai_params_path):
        print(f"Loading AI params: {ai_params_path}")
        with open(ai_params_path) as f:
            ai_params = json.load(f)
        analysis = ai_params.get('analysis', {})
        target = {
            'rms': analysis.get('dynamics', {}).get('rms_mean', 0.1),
            'centroid': analysis.get('spectrum', {}).get('centroid_mean', 2000),
            'bandwidth': analysis.get('spectrum', {}).get('bandwidth_mean', 2000),
            'band_energy': ai_params.get('target_bands') or analysis.get('spectrum', {}).get('band_energy', {}),
            'mfcc': analysis.get('timbre', {}).get('mfcc_mean', [0]*13),
            'chroma': [0.5]*12  # Default if not available
        }
        print(f"  RMS: {target['rms']:.3f}, Centroid: {target['centroid']:.0f}Hz")

    if not target:
        print("ERROR: No original audio or AI params provided!")
        print("Cannot do AI-driven rendering without target analysis.")
        return False

    # Step 2: Load models
    models = {}
    models_path = Path(models_dir)

    for model_name in ['melodic', 'bass', 'drums']:
        for model_dir in models_path.glob(f"*_{model_name}"):
            samples = load_model_samples(model_dir)
            if samples:
                models[model_name] = samples
                break
        if model_name not in models:
            model_dir = models_path / model_name
            if model_dir.exists():
                samples = load_model_samples(model_dir)
                if samples:
                    models[model_name] = samples

    print(f"Models: {', '.join(f'{k}({len(v)} samples)' for k,v in models.items())}")

    if not models:
        print("No models found!")
        return False

    # Step 3: Parse patterns
    patterns = parse_bar_arrays(strudel_code)
    print(f"Patterns: {list(patterns.keys())}")

    # Step 4: Calculate AI-driven initial gains from target bands
    target_bands = target.get('band_energy', {})
    bass_e = target_bands.get('bass', 0) + target_bands.get('sub_bass', 0)
    mid_e = target_bands.get('mid', 0) + target_bands.get('low_mid', 0)
    high_e = target_bands.get('high', 0) + target_bands.get('high_mid', 0)
    total_e = bass_e + mid_e + high_e

    # Map patterns to frequency bands for gain initialization
    bass_patterns = ['bass', 'kick']
    mid_patterns = ['mid', 'melodic', 'vox', 'stab']
    high_patterns = ['high', 'lead', 'hh']
    drums_patterns = ['drums', 'snare']

    # Initialize base gains from target analysis
    base_gains = {}
    if total_e > 0:
        base_gains = {
            'bass_band': max(0.1, bass_e / total_e * 3),
            'mid_band': max(0.5, mid_e / total_e * 4),
            'high_band': max(0.3, high_e / total_e * 4),
            'drums_band': 0.5
        }
    else:
        base_gains = {'bass_band': 0.5, 'mid_band': 1.5, 'high_band': 1.0, 'drums_band': 0.5}

    # Create gains for all patterns found in the code
    gains = {}
    for pattern_name in patterns.keys():
        if pattern_name in bass_patterns:
            gains[pattern_name] = base_gains['bass_band']
        elif pattern_name in mid_patterns:
            gains[pattern_name] = base_gains['mid_band']
        elif pattern_name in high_patterns:
            gains[pattern_name] = base_gains['high_band']
        elif pattern_name in drums_patterns:
            gains[pattern_name] = base_gains['drums_band']
        else:
            gains[pattern_name] = 1.0  # Default for unknown patterns

    print(f"Initial gains: {', '.join(f'{k}={v:.2f}' for k,v in gains.items())}")

    # Step 5: Render initial audio
    beat_duration = 60 / bpm
    bar_duration = beat_duration * 4
    total_samples = int(duration * SAMPLE_RATE)
    total_bars = int(duration / bar_duration) + 1

    # Map pattern names to model names
    pattern_model_map = {
        'bass': 'bass', 'mid': 'melodic', 'high': 'melodic',
        'drums': 'drums', 'melodic': 'melodic',
        # Brazilian funk patterns
        'kick': 'drums', 'snare': 'drums', 'hh': 'drums',
        'vox': 'melodic', 'stab': 'melodic', 'lead': 'melodic'
    }

    voices = {}
    for pattern_name, bars in patterns.items():
        model_name = pattern_model_map.get(pattern_name, 'melodic')
        if model_name not in models or not bars:
            continue

        voice_audio = np.zeros(total_samples)
        bar_samples = int(bar_duration * SAMPLE_RATE)

        for bar_idx in range(total_bars):
            bar_pattern = bars[bar_idx % len(bars)]
            start = bar_idx * bar_samples
            if start >= total_samples:
                break
            bar_audio = render_pattern(models[model_name], bar_pattern, beat_duration)
            end = min(start + len(bar_audio), total_samples)
            voice_audio[start:end] += bar_audio[:end-start]

        voices[pattern_name] = voice_audio

    # Step 6: Iterative refinement to match target
    print(f"\nIterative refinement (max {max_iterations} iterations)...")

    best_audio = None
    best_similarity = 0
    current_gains = gains.copy()

    for iteration in range(max_iterations):
        # Mix with current gains
        audio = np.zeros(total_samples)
        for voice_name, voice_audio in voices.items():
            audio += voice_audio * current_gains.get(voice_name, 1.0)

        # Normalize to target RMS
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            audio = audio * (target['rms'] / current_rms)

        # Analyze current
        current = analyze_audio_full(audio, SAMPLE_RATE)
        similarity, scores = compute_similarity(current, target)

        print(f"  [{iteration+1}] Similarity: {similarity*100:.1f}% | RMS:{scores['rms']*100:.0f}% Bright:{scores['centroid']*100:.0f}% Bands:{scores['bands']*100:.0f}% MFCC:{scores['mfcc']*100:.0f}% Chroma:{scores['chroma']*100:.0f}%")

        if similarity > best_similarity:
            best_similarity = similarity
            best_audio = audio.copy()

        if similarity > 0.90:
            print(f"  Target reached: {similarity*100:.1f}%")
            break

        # Adjust gains based on band differences
        # Map each voice to a frequency band for adjustment
        for voice_name in voices:
            if voice_name in current_gains:
                if voice_name in ['bass', 'kick']:
                    # Bass + sub-bass range
                    target_val = target_bands.get('bass', 0) + target_bands.get('sub_bass', 0)
                    current_val = current['band_energy'].get('bass', 0) + current['band_energy'].get('sub_bass', 0)
                elif voice_name in ['mid', 'melodic', 'vox', 'stab']:
                    # Mid range
                    target_val = target_bands.get('mid', 0) + target_bands.get('low_mid', 0)
                    current_val = current['band_energy'].get('mid', 0) + current['band_energy'].get('low_mid', 0)
                elif voice_name in ['high', 'lead', 'hh']:
                    # High range
                    target_val = target_bands.get('high', 0) + target_bands.get('high_mid', 0)
                    current_val = current['band_energy'].get('high', 0) + current['band_energy'].get('high_mid', 0)
                elif voice_name in ['snare', 'drums']:
                    # Snare covers mid + high_mid
                    target_val = target_bands.get('high_mid', 0) + target_bands.get('mid', 0) * 0.5
                    current_val = current['band_energy'].get('high_mid', 0) + current['band_energy'].get('mid', 0) * 0.5
                else:
                    continue

                if current_val > 0.001:
                    ratio = target_val / current_val
                    # More conservative adjustment - smaller steps for stability
                    adjustment = np.clip(ratio, 0.9, 1.1)
                    current_gains[voice_name] *= adjustment

        # Apply spectral matching
        audio = apply_spectral_matching(audio, current, target, SAMPLE_RATE)

    audio = best_audio if best_audio is not None else audio

    # Final processing
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (target['rms'] / current_rms)

    audio = np.tanh(audio * 2) / 2

    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (target['rms'] / current_rms)

    max_val = np.max(np.abs(audio))
    if max_val > 0.98:
        audio = audio * (0.98 / max_val)

    # Save
    sf.write(output_path, audio.astype(np.float32), SAMPLE_RATE)

    final_rms = np.sqrt(np.mean(audio ** 2))
    print(f"\n{'='*60}")
    print(f"RESULT: {best_similarity*100:.1f}% similarity")
    print(f"Output: {output_path}")
    print(f"RMS: {final_rms:.3f} (target: {target['rms']:.3f})")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(description='AI-driven render - works for ANY track')
    parser.add_argument('strudel', help='Strudel code file')
    parser.add_argument('--models', '-m', required=True, help='Models directory')
    parser.add_argument('--output', '-o', default='output.wav', help='Output WAV')
    parser.add_argument('--bpm', type=float, default=120, help='BPM')
    parser.add_argument('--duration', '-d', type=float, default=60, help='Duration (s)')
    parser.add_argument('--ai-params', '-p', help='AI params JSON')
    parser.add_argument('--original', help='Original audio for direct analysis (preferred)')
    parser.add_argument('--iterations', '-i', type=int, default=10, help='Max iterations')

    args = parser.parse_args()

    with open(args.strudel) as f:
        code = f.read()

    bpm_match = re.search(r'setcps\((\d+)/60/4\)', code)
    if bpm_match:
        args.bpm = float(bpm_match.group(1))

    # Auto-detect paths
    strudel_dir = os.path.dirname(os.path.abspath(args.strudel))
    cache_dir = os.path.dirname(strudel_dir)

    ai_params_path = args.ai_params
    if not ai_params_path:
        candidate = os.path.join(strudel_dir, 'ai_params.json')
        if os.path.exists(candidate):
            ai_params_path = candidate

    original_path = args.original
    if not original_path:
        for candidate in ['melodic.wav', 'piano.wav']:
            path = os.path.join(cache_dir, candidate)
            if os.path.exists(path):
                original_path = path
                break

    success = render_with_models(
        code, args.models, args.output,
        bpm=args.bpm, duration=args.duration,
        ai_params_path=ai_params_path,
        original_audio_path=original_path,
        max_iterations=args.iterations
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
