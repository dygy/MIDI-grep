#!/usr/bin/env python3
"""
AI-Driven Strudel Code Generator

This module analyzes original audio and generates Strudel code that INHERENTLY
matches the target characteristics - not by tweaking the renderer, but by
producing code with parameters that naturally achieve high similarity.

Key principles:
1. Analyze original audio for ALL characteristics (spectrum, dynamics, timing, timbre)
2. Generate Strudel code with parameters matched to target
3. NO hardcoded values - everything derived from analysis
4. Works universally for ANY track

This is the CODE GENERATION brain - the renderer just plays what this generates.
"""

import argparse
import json
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import sys
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Import sound selector for varied sound palette
try:
    from sound_selector import (
        analyze_and_suggest, get_genre_sounds, create_sound_alternation,
        GENRE_PALETTES, DRUM_BANKS, BASS_SOUNDS, LEAD_SOUNDS, PAD_SOUNDS,
        PERCUSSIVE_SOUNDS
    )
    SOUND_SELECTOR_AVAILABLE = True
except ImportError:
    SOUND_SELECTOR_AVAILABLE = False
    print("Warning: sound_selector not available, using default sounds", file=sys.stderr)


@dataclass
class AudioProfile:
    """Complete audio profile for a track."""
    # Spectral
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    spectral_flatness_mean: float
    brightness: str  # 'dark', 'neutral', 'bright', 'very_bright'

    # Frequency bands (normalized energy)
    sub_bass_energy: float   # 20-60 Hz
    bass_energy: float       # 60-250 Hz
    low_mid_energy: float    # 250-500 Hz
    mid_energy: float        # 500-2000 Hz
    high_mid_energy: float   # 2000-4000 Hz
    high_energy: float       # 4000-20000 Hz

    # Dynamics
    rms_mean: float
    rms_std: float
    dynamic_range_db: float
    crest_factor: float  # Peak/RMS ratio - indicates punchiness
    is_compressed: bool
    is_punchy: bool

    # Rhythm
    tempo: float
    tempo_confidence: float
    swing_amount: float  # 0 = straight, 0.5 = heavy swing
    has_syncopation: bool
    onset_density: float  # Onsets per second

    # Timbre (MFCCs)
    mfcc_mean: List[float]  # First 13 MFCCs
    mfcc_std: List[float]
    timbre_profile: str  # 'warm', 'harsh', 'clean', 'gritty'

    # Harmonic
    key: str
    mode: str  # 'major', 'minor'
    key_confidence: float
    dominant_pitch_class: int

    # Structure
    duration_seconds: float
    num_sections: int
    has_drops: bool
    has_builds: bool
    energy_curve: List[float]  # 16-point normalized energy envelope


def analyze_audio_complete(audio_path: str, duration: float = 60) -> AudioProfile:
    """
    Comprehensive audio analysis to build complete profile.
    This is the foundation for AI-driven code generation.
    """
    print(f"Analyzing: {audio_path}", file=sys.stderr)

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)

    # === SPECTRAL ANALYSIS ===
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Frequency band energy
    total_energy = np.sum(S ** 2) + 1e-10
    bands = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'high': (4000, 20000),
    }
    band_energy = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[band_name] = float(np.sum(S[mask, :] ** 2) / total_energy)

    # Brightness classification
    centroid_mean = float(np.mean(centroid))
    if centroid_mean > 3500:
        brightness = 'very_bright'
    elif centroid_mean > 2500:
        brightness = 'bright'
    elif centroid_mean > 1200:
        brightness = 'neutral'
    else:
        brightness = 'dark'

    # === DYNAMICS ANALYSIS ===
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    dynamic_range = float(np.max(rms_db) - np.min(rms_db))

    # Crest factor (punchiness)
    peak = np.max(np.abs(y))
    rms_total = np.sqrt(np.mean(y ** 2)) + 1e-10
    crest_factor = float(peak / rms_total)

    # === RHYTHM ANALYSIS ===
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Onset detection for density
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_density = len(onsets) / (len(y) / sr) if len(y) > 0 else 0

    # Swing detection
    if len(beat_frames) > 4:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        intervals = np.diff(beat_times)
        if len(intervals) > 2:
            # Check for alternating long-short pattern
            even_intervals = intervals[::2]
            odd_intervals = intervals[1::2]
            if len(even_intervals) > 0 and len(odd_intervals) > 0:
                ratio = np.mean(even_intervals) / (np.mean(odd_intervals) + 1e-10)
                swing_amount = float(max(0, min(0.5, (ratio - 1) / 2)))
            else:
                swing_amount = 0.0
        else:
            swing_amount = 0.0
    else:
        swing_amount = 0.0

    # Syncopation detection (offbeat emphasis)
    has_syncopation = onset_density > 4.0  # Rough heuristic

    # === TIMBRE ANALYSIS (MFCCs) ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]
    mfcc_std = [float(x) for x in np.std(mfccs, axis=1)]

    # Timbre classification based on MFCCs
    if mfcc_mean[1] > 50:
        timbre_profile = 'harsh'
    elif mfcc_mean[1] < -50:
        timbre_profile = 'warm'
    elif float(np.mean(flatness)) > 0.3:
        timbre_profile = 'gritty'
    else:
        timbre_profile = 'clean'

    # === HARMONIC ANALYSIS ===
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_pitch_class = int(np.argmax(chroma_mean))

    # Key detection
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = pitch_names[dominant_pitch_class]

    # Major/minor detection using Krumhansl-Schmuckler profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, dominant_pitch_class))[0, 1]
    minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, dominant_pitch_class))[0, 1]

    mode = 'major' if major_corr > minor_corr else 'minor'
    key_confidence = float(max(major_corr, minor_corr))

    # === STRUCTURE ANALYSIS ===
    duration_seconds = float(len(y) / sr)

    # Energy envelope (16 points)
    rms_normalized = rms / (np.max(rms) + 1e-10)
    energy_curve = list(np.interp(
        np.linspace(0, len(rms_normalized), 16),
        np.arange(len(rms_normalized)),
        rms_normalized
    ))

    # Detect energy drops/builds
    energy_diff = np.diff(energy_curve)
    has_drops = bool(np.any(energy_diff < -0.3))
    has_builds = bool(np.any(energy_diff > 0.3))

    # Section detection (rough estimate)
    num_sections = max(1, int(duration_seconds / 30))  # Roughly 30 sec sections

    return AudioProfile(
        spectral_centroid_mean=centroid_mean,
        spectral_centroid_std=float(np.std(centroid)),
        spectral_bandwidth_mean=float(np.mean(bandwidth)),
        spectral_rolloff_mean=float(np.mean(rolloff)),
        spectral_flatness_mean=float(np.mean(flatness)),
        brightness=brightness,
        sub_bass_energy=band_energy['sub_bass'],
        bass_energy=band_energy['bass'],
        low_mid_energy=band_energy['low_mid'],
        mid_energy=band_energy['mid'],
        high_mid_energy=band_energy['high_mid'],
        high_energy=band_energy['high'],
        rms_mean=float(np.mean(rms)),
        rms_std=float(np.std(rms)),
        dynamic_range_db=dynamic_range,
        crest_factor=crest_factor,
        is_compressed=dynamic_range < 15,
        is_punchy=crest_factor > 8,
        tempo=tempo,
        tempo_confidence=0.8,  # Placeholder
        swing_amount=swing_amount,
        has_syncopation=has_syncopation,
        onset_density=float(onset_density),
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        timbre_profile=timbre_profile,
        key=key,
        mode=mode,
        key_confidence=key_confidence,
        dominant_pitch_class=dominant_pitch_class,
        duration_seconds=duration_seconds,
        num_sections=num_sections,
        has_drops=has_drops,
        has_builds=has_builds,
        energy_curve=energy_curve
    )


def profile_to_strudel_params(profile: AudioProfile, genre: str = "") -> Dict:
    """
    Convert audio profile to optimal Strudel parameters.

    This is where the AI magic happens - we derive ALL effect parameters
    from the audio analysis, with NO hardcoded values.

    Args:
        profile: Complete audio analysis profile
        genre: Optional genre hint (e.g., 'brazilian_funk', 'electro_swing')
    """
    params = {}
    params['genre'] = genre  # Store for later use

    # === FILTER PARAMETERS (from spectral analysis) ===

    # LPF: Match brightness - brighter audio = higher cutoff
    if profile.brightness == 'very_bright':
        params['lpf_bass'] = 1200
        params['lpf_mid'] = 8000
        params['lpf_high'] = 16000
    elif profile.brightness == 'bright':
        params['lpf_bass'] = 800
        params['lpf_mid'] = 6000
        params['lpf_high'] = 12000
    elif profile.brightness == 'neutral':
        params['lpf_bass'] = 600
        params['lpf_mid'] = 4000
        params['lpf_high'] = 10000
    else:  # dark
        params['lpf_bass'] = 400
        params['lpf_mid'] = 2500
        params['lpf_high'] = 6000

    # HPF: Clean up based on sub-bass energy
    params['hpf_bass'] = 30 if profile.sub_bass_energy > 0.1 else 50
    params['hpf_mid'] = 150 if profile.bass_energy > 0.2 else 200
    params['hpf_high'] = 300 if profile.low_mid_energy > 0.2 else 400

    # === DYNAMICS PARAMETERS ===

    # Gain: Match RMS levels
    rms_scale = min(2.0, max(0.5, profile.rms_mean * 10))
    params['gain_bass'] = rms_scale * (1.2 if profile.bass_energy > 0.3 else 1.0)
    params['gain_mid'] = rms_scale * (1.0 if profile.mid_energy > 0.2 else 0.8)
    params['gain_high'] = rms_scale * (0.9 if profile.high_energy > 0.1 else 0.7)

    # Compression: Based on dynamic range
    if profile.is_compressed:
        params['clip_bass'] = 0.85
        params['clip_mid'] = 0.90
        params['clip_high'] = 0.95
    else:
        params['clip_bass'] = 1.0
        params['clip_mid'] = 1.1
        params['clip_high'] = 1.2

    # Distortion/saturation based on timbre
    if profile.timbre_profile == 'gritty':
        params['distort'] = 0.2
        params['crush'] = 12
        params['coarse'] = 2
    elif profile.timbre_profile == 'harsh':
        params['distort'] = 0.3
        params['crush'] = 10
        params['coarse'] = 4
    elif profile.timbre_profile == 'warm':
        params['distort'] = 0.05
        params['crush'] = 14
        params['coarse'] = 1
    else:  # clean
        params['distort'] = 0.0
        params['crush'] = 16
        params['coarse'] = 1

    # === SPATIAL PARAMETERS ===

    # Reverb: Larger for ambient, smaller for punchy
    if profile.is_punchy:
        params['room'] = 0.1
        params['size'] = 0.15
    else:
        room_amount = 0.15 + (1.0 - min(1.0, profile.crest_factor / 10)) * 0.25
        params['room'] = room_amount
        params['size'] = room_amount * 1.5

    # Delay: Based on tempo (sync to beat)
    beat_time = 60.0 / profile.tempo
    params['delay'] = 0.15 if profile.has_syncopation else 0.10
    params['delaytime'] = beat_time * 0.25  # 16th note
    params['delayfeedback'] = 0.3 if profile.has_syncopation else 0.2

    # Pan: Wider stereo for complex sounds
    params['pan_width'] = 0.3 if profile.spectral_bandwidth_mean > 2000 else 0.2

    # === MODULATION PARAMETERS ===

    # Vibrato: Based on spectral variation
    if profile.spectral_centroid_std > 500:
        params['vib'] = 4.0
        params['vibmod'] = 0.02
    else:
        params['vib'] = 0
        params['vibmod'] = 0

    # Phaser: For swirly sounds
    params['phaser'] = 0.3 if profile.spectral_flatness_mean > 0.2 else 0

    # === ENVELOPE PARAMETERS ===

    # Attack: Faster for punchy, slower for pads
    if profile.is_punchy:
        params['attack'] = 0.001
        params['decay'] = 0.1
        params['sustain'] = 0.7
        params['release'] = 0.2
    else:
        params['attack'] = 0.05
        params['decay'] = 0.2
        params['sustain'] = 0.6
        params['release'] = 0.5

    # === RHYTHM PARAMETERS ===

    params['swing'] = profile.swing_amount
    params['degradeBy'] = 0.05 if profile.timbre_profile == 'gritty' else 0

    # === PATTERN DENSITY PARAMETERS ===
    # These control how sparse/dense patterns should be to match original

    # Onset density per second (original has X onsets/sec)
    params['target_onset_density'] = profile.onset_density

    # Max drum hits per bar (derived from onset density and tempo)
    # At 89 BPM, one bar = 2.7 sec. If onset_density = 1.25, that's ~3.4 onsets/bar
    bar_duration = 60.0 / profile.tempo * 4  # 4 beats per bar
    target_hits_per_bar = profile.onset_density * bar_duration

    # Scale down for drums (drums are subset of all onsets)
    params['max_kicks_per_bar'] = max(1, int(target_hits_per_bar * 0.3))  # ~30% kicks
    params['max_snares_per_bar'] = max(1, int(target_hits_per_bar * 0.15))  # ~15% snares
    params['max_hihats_per_bar'] = max(1, int(target_hits_per_bar * 0.5))  # ~50% hihats

    # Pattern sparseness (0=dense, 1=very sparse)
    # Higher onset density = less sparse patterns
    if profile.onset_density > 5.0:
        params['pattern_sparseness'] = 0.0  # Dense
    elif profile.onset_density > 3.0:
        params['pattern_sparseness'] = 0.2  # Medium-dense
    elif profile.onset_density > 1.5:
        params['pattern_sparseness'] = 0.5  # Medium
    else:
        params['pattern_sparseness'] = 0.8  # Sparse

    # Beat emphasis (which beats have drums)
    # For 4/4 time: [1, 0, 0, 0] = only downbeat, [1, 0, 1, 0] = 1 and 3
    if profile.is_punchy:
        params['kick_beat_pattern'] = [1.0, 0.0, 0.8, 0.0]  # 4 on the floor variant
        params['snare_beat_pattern'] = [0.0, 0.0, 1.0, 0.0]  # Snare on 3
    else:
        params['kick_beat_pattern'] = [1.0, 0.0, 0.5, 0.0]  # Lighter kick
        params['snare_beat_pattern'] = [0.0, 1.0, 0.0, 1.0]  # Snare on 2 and 4

    # === SOUND SELECTION (using sound_selector for variety) ===

    if SOUND_SELECTOR_AVAILABLE:
        # Use sound selector for intelligent sound selection with variety
        # Calculate harmonic ratio from spectral flatness (higher flatness = more noise = less harmonic)
        harmonic_ratio = 1.0 - profile.spectral_flatness_mean

        # Analyze and get sound suggestions (use genre if provided)
        sound_suggestions = analyze_and_suggest(
            spectral_centroid=profile.spectral_centroid_mean,
            spectral_rolloff=profile.spectral_rolloff_mean,
            rms_mean=profile.rms_mean,
            attack_time=0.01 if profile.is_punchy else 0.05,  # Derived from punchiness
            harmonic_ratio=harmonic_ratio,
            genre=genre  # Pass genre for genre-specific sounds
        )

        # Use alternating sounds for variety (e.g., "<supersaw gm_synth_bass_1>")
        params['sound_bass'] = sound_suggestions['bass_sound']
        params['sound_mid'] = sound_suggestions['lead_sound']  # Lead for mid register
        params['sound_pad'] = sound_suggestions['pad_sound']  # Pad for sustained parts
        params['sound_high'] = sound_suggestions['high_sound']
        params['drum_bank'] = sound_suggestions['drum_bank']
        params['alt_drum_banks'] = sound_suggestions.get('alt_drum_banks', [])

        # Store timbre profile for debugging
        params['timbre_info'] = sound_suggestions.get('timbre_profile', {})
    else:
        # Fallback to original logic if sound_selector not available
        # Bass sound based on sub-bass and bass energy
        if profile.sub_bass_energy > 0.15:
            params['sound_bass'] = 'supersaw'
        elif profile.bass_energy > 0.25:
            params['sound_bass'] = 'sawtooth'
        else:
            params['sound_bass'] = 'gm_acoustic_bass'

        # Mid sound based on timbre
        if profile.timbre_profile == 'harsh':
            params['sound_mid'] = 'square'
        elif profile.timbre_profile == 'warm':
            params['sound_mid'] = 'gm_epiano1'
        elif profile.timbre_profile == 'gritty':
            params['sound_mid'] = 'gm_lead_2_sawtooth'
        else:
            params['sound_mid'] = 'gm_pad_poly'

        params['sound_pad'] = 'gm_pad_2_warm'

        # High sound based on brightness
        if profile.brightness in ['very_bright', 'bright']:
            params['sound_high'] = 'triangle'
        else:
            params['sound_high'] = 'gm_vibraphone'

        # Drums based on punchiness
        if profile.is_punchy:
            params['drum_bank'] = 'RolandTR808'
        else:
            params['drum_bank'] = 'RolandTR909'

        params['alt_drum_banks'] = []

    return params


def generate_effect_chain(params: Dict, voice: str) -> str:
    """
    Generate Strudel effect chain for a voice.
    Voice: 'bass', 'mid', 'high'

    Supports alternating sounds like "<supersaw gm_synth_bass_1>"
    """
    chain = []

    # Sound source - handle alternating sounds
    sound_key = f'sound_{voice}'
    if sound_key in params:
        sound_val = params[sound_key]
        # Alternating sounds already have < > syntax
        if sound_val.startswith('<'):
            chain.append(f'.sound("{sound_val}")')
        else:
            chain.append(f'.sound("{sound_val}")')

    # Gain with perlin modulation for organic feel
    gain_key = f'gain_{voice}'
    if gain_key in params:
        base_gain = params[gain_key]
        chain.append(f'.gain(perlin.range({base_gain*0.9:.2f}, {base_gain*1.1:.2f}).slow(8))')

    # Pan with subtle movement
    pan_width = params.get('pan_width', 0.2)
    if voice == 'bass':
        chain.append('.pan(0.5)')  # Bass centered
    elif voice == 'mid':
        chain.append(f'.pan(perlin.range({0.5-pan_width:.2f},{0.5+pan_width:.2f}).slow(4))')
    else:  # high
        chain.append(f'.pan(perlin.range({0.5-pan_width*1.5:.2f},{0.5+pan_width*1.5:.2f}).slow(3))')

    # Filters
    hpf_key = f'hpf_{voice}'
    lpf_key = f'lpf_{voice}'
    if hpf_key in params:
        chain.append(f'.hpf({params[hpf_key]})')
    if lpf_key in params:
        chain.append(f'.lpf({params[lpf_key]})')

    # Distortion/bitcrush (if applicable)
    if params.get('crush', 16) < 16:
        chain.append(f'.crush({params["crush"]})')
    if params.get('coarse', 1) > 1:
        chain.append(f'.coarse({params["coarse"]})')

    # Clip for dynamics
    clip_key = f'clip_{voice}'
    if clip_key in params:
        chain.append(f'.clip({params[clip_key]:.2f})')

    # Reverb
    if params.get('room', 0) > 0:
        room_mult = 1.0 if voice == 'bass' else 1.5 if voice == 'mid' else 2.0
        chain.append(f'.room({params["room"]*room_mult:.2f})')
        chain.append(f'.size({params["size"]*room_mult:.2f})')

    # Delay for high voice
    if voice == 'high' and params.get('delay', 0) > 0:
        chain.append(f'.delay({params["delay"]:.2f})')
        chain.append(f'.delaytime({params["delaytime"]:.3f})')
        chain.append(f'.delayfeedback({params["delayfeedback"]:.2f})')

    # Pattern transforms
    if params.get('swing', 0) > 0:
        chain.append(f'.swing({params["swing"]:.2f})')
    if params.get('degradeBy', 0) > 0:
        chain.append(f'.degradeBy({params["degradeBy"]:.2f})')

    return '\n    '.join(chain)


def main():
    parser = argparse.ArgumentParser(description='AI-driven Strudel code generation')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--duration', '-d', type=float, default=60, help='Analysis duration in seconds')
    parser.add_argument('--genre', '-g', default='', help='Genre hint for sound selection')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Analyze audio
    profile = analyze_audio_complete(args.audio_path, duration=args.duration)

    # Convert to Strudel parameters (with genre if provided)
    params = profile_to_strudel_params(profile, genre=args.genre)

    # Generate effect chains
    effect_chains = {
        'bass': generate_effect_chain(params, 'bass'),
        'mid': generate_effect_chain(params, 'mid'),
        'high': generate_effect_chain(params, 'high'),
    }

    # Compile output
    output = {
        'profile': asdict(profile),
        'params': params,
        'effect_chains': effect_chains,
        'metadata': {
            'source': args.audio_path,
            'analysis_duration': args.duration,
            'generator': 'ai_code_generator.py v1.0'
        }
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))

    if args.verbose:
        print(f"\n=== Audio Profile ===", file=sys.stderr)
        print(f"Key: {profile.key} {profile.mode}", file=sys.stderr)
        print(f"Tempo: {profile.tempo:.1f} BPM", file=sys.stderr)
        print(f"Brightness: {profile.brightness}", file=sys.stderr)
        print(f"Timbre: {profile.timbre_profile}", file=sys.stderr)
        print(f"Dynamic range: {profile.dynamic_range_db:.1f} dB", file=sys.stderr)
        print(f"Compressed: {profile.is_compressed}, Punchy: {profile.is_punchy}", file=sys.stderr)
        print(f"Swing: {profile.swing_amount:.2f}", file=sys.stderr)

        print(f"\n=== Derived Parameters ===", file=sys.stderr)
        for k, v in params.items():
            print(f"  {k}: {v}", file=sys.stderr)


if __name__ == '__main__':
    main()
