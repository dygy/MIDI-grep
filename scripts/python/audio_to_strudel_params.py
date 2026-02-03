#!/usr/bin/env python3
"""
AI-driven audio analysis to determine optimal Strudel parameters.
Analyzes frequency spectrum, dynamics, and timbre to suggest effects.
"""

import argparse
import json
import numpy as np
import librosa
from scipy import signal
import sys

# Strudel effect knowledge base
STRUDEL_EFFECTS = {
    # Filters
    'lpf': {'param': 'cutoff', 'range': (100, 20000), 'description': 'Low-pass filter cutoff frequency'},
    'hpf': {'param': 'cutoff', 'range': (20, 5000), 'description': 'High-pass filter cutoff frequency'},
    'bpf': {'param': 'cutoff', 'range': (100, 10000), 'description': 'Band-pass filter center frequency'},

    # Dynamics
    'gain': {'param': 'level', 'range': (0, 2), 'description': 'Volume level'},
    'distort': {'param': 'amount', 'range': (0, 1), 'description': 'Distortion/saturation'},
    'crush': {'param': 'bits', 'range': (4, 16), 'description': 'Bit crusher (lo-fi effect)'},
    'coarse': {'param': 'rate', 'range': (1, 32), 'description': 'Sample rate reduction'},

    # Spatial
    'room': {'param': 'size', 'range': (0, 1), 'description': 'Reverb room size'},
    'delay': {'param': 'mix', 'range': (0, 1), 'description': 'Delay effect mix'},
    'delaytime': {'param': 'time', 'range': (0.01, 1), 'description': 'Delay time in seconds'},
    'delayfeedback': {'param': 'feedback', 'range': (0, 0.95), 'description': 'Delay feedback'},
    'pan': {'param': 'position', 'range': (0, 1), 'description': 'Stereo panning'},

    # Modulation
    'phaser': {'param': 'rate', 'range': (0, 1), 'description': 'Phaser effect'},
    'vib': {'param': 'freq', 'range': (1, 20), 'description': 'Vibrato frequency'},
    'vibmod': {'param': 'depth', 'range': (0, 0.5), 'description': 'Vibrato depth'},
    'tremolo': {'param': 'freq', 'range': (1, 32), 'description': 'Tremolo/amplitude modulation frequency'},
    'tremolodepth': {'param': 'depth', 'range': (0, 1), 'description': 'Tremolo depth'},

    # Envelope
    'attack': {'param': 'time', 'range': (0.001, 2), 'description': 'Attack time'},
    'decay': {'param': 'time', 'range': (0.01, 2), 'description': 'Decay time'},
    'sustain': {'param': 'level', 'range': (0, 1), 'description': 'Sustain level'},
    'release': {'param': 'time', 'range': (0.01, 5), 'description': 'Release time'},

    # FM Synthesis
    'fm': {'param': 'index', 'range': (0, 10), 'description': 'FM modulation index'},
    'fmh': {'param': 'harmonicity', 'range': (0.5, 4), 'description': 'FM harmonicity ratio'},

    # Pattern transforms
    'swing': {'param': 'amount', 'range': (0, 0.5), 'description': 'Swing feel'},
    'degradeBy': {'param': 'amount', 'range': (0, 0.5), 'description': 'Random note removal'},

    # Sound sources
    'sounds': {
        'bass': ['sawtooth', 'supersaw', 'gm_synth_bass_1', 'gm_acoustic_bass'],
        'mid': ['square', 'gm_epiano1', 'gm_pad_poly', 'gm_lead_2_sawtooth'],
        'high': ['triangle', 'gm_vibraphone', 'gm_lead_5_charang'],
        'drums': ['RolandTR808', 'RolandTR909', 'LinnDrum']
    }
}

def analyze_frequency_spectrum(y, sr):
    """Analyze frequency distribution."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Calculate energy in frequency bands
    bands = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'high': (4000, 20000),
    }

    total_energy = np.sum(S ** 2)
    band_energy = {}

    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[band_name] = float(np.sum(S[mask, :] ** 2) / max(total_energy, 1e-10))

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Spectral flatness (noisiness)
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    return {
        'band_energy': band_energy,
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'flatness_mean': float(np.mean(flatness)),
        'brightness': 'bright' if np.mean(centroid) > 2500 else 'dark' if np.mean(centroid) < 1000 else 'neutral'
    }

def analyze_dynamics(y, sr):
    """Analyze dynamic characteristics over time."""
    rms = librosa.feature.rms(y=y)[0]

    # Dynamic range
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    dynamic_range = float(np.max(rms_db) - np.min(rms_db))

    # Transient sharpness
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Analyze dynamics over time (every 4 bars approximately)
    hop_length = 512
    frame_duration = hop_length / sr
    bar_duration = 60 / 130 * 4  # Assume ~130 BPM, 4 beats per bar
    frames_per_section = int(bar_duration * 4 / frame_duration)  # 4 bars per section

    # Split into sections and analyze
    sections = []
    for i in range(0, len(rms), frames_per_section):
        section_rms = rms[i:i+frames_per_section]
        if len(section_rms) > 0:
            sections.append({
                'rms_mean': float(np.mean(section_rms)),
                'rms_max': float(np.max(section_rms)),
                'rms_min': float(np.min(section_rms)),
            })

    # Find energy envelope shape
    rms_normalized = rms / (np.max(rms) + 1e-10)
    energy_curve = np.interp(
        np.linspace(0, len(rms_normalized), 16),
        np.arange(len(rms_normalized)),
        rms_normalized
    )

    return {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'rms_min': float(np.min(rms)),
        'rms_max': float(np.max(rms)),
        'dynamic_range_db': dynamic_range,
        'transient_strength': float(np.mean(onset_env)),
        'is_compressed': dynamic_range < 20,
        'is_punchy': float(np.mean(onset_env)) > 1.5,
        'sections': sections,
        'energy_curve': [float(x) for x in energy_curve],  # 16-point energy envelope
        'has_dynamics': dynamic_range > 12  # Meaningful dynamic variation
    }

def analyze_rhythm(y, sr):
    """Analyze rhythmic characteristics."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0

    # Tempogram for rhythm pattern
    tempogram = librosa.feature.tempogram(y=y, sr=sr)

    # Check for swing
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    swing_ratio = 1.0
    if len(onset_times) > 4:
        # Analyze eighth note timing
        intervals = np.diff(onset_times)
        if len(intervals) > 2:
            # Look for long-short pattern
            even_intervals = intervals[::2]
            odd_intervals = intervals[1::2]
            if len(even_intervals) > 0 and len(odd_intervals) > 0:
                ratio = np.mean(even_intervals) / np.mean(odd_intervals) if np.mean(odd_intervals) > 0 else 1
                if 1.2 < ratio < 2.0:
                    swing_ratio = ratio

    return {
        'tempo': float(tempo),
        'beat_count': len(beats),
        'rhythm_regularity': float(np.mean(np.max(tempogram, axis=0))),
        'swing_ratio': float(swing_ratio),
        'has_swing': bool(swing_ratio > 1.15)
    }

def analyze_timbre(y, sr):
    """Analyze timbral characteristics."""
    # MFCCs for timbre
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Zero crossing rate (correlates with noisiness/brightness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        'mfcc_mean': [float(x) for x in np.mean(mfcc, axis=1)],
        'contrast_mean': [float(x) for x in np.mean(contrast, axis=1)],
        'zcr_mean': float(np.mean(zcr)),
        'is_noisy': float(np.mean(zcr)) > 0.1,
        'is_tonal': float(np.mean(zcr)) < 0.05
    }

def suggest_strudel_params(analysis):
    """Convert audio analysis to Strudel effect parameters with dynamic expressions."""
    suggestions = {
        'bass': {},
        'mid': {},
        'high': {},
        'drums': {},
        'global': {},
        'dynamic_effects': {}  # Strudel expressions for time-varying effects
    }

    spectrum = analysis['spectrum']
    dynamics = analysis['dynamics']
    rhythm = analysis['rhythm']
    timbre = analysis['timbre']

    # Calculate base gains from spectrum
    bass_energy = spectrum['band_energy']['bass'] + spectrum['band_energy']['sub_bass']
    mid_energy = spectrum['band_energy']['mid'] + spectrum['band_energy']['low_mid']
    high_energy = spectrum['band_energy']['high'] + spectrum['band_energy']['high_mid']

    # === DYNAMIC EXPRESSIONS (Strudel LFO/perlin patterns) ===
    if dynamics.get('has_dynamics', False):
        # Use perlin noise for organic movement matching the track's dynamics
        energy_range = dynamics['rms_max'] - dynamics['rms_min']

        # Dynamic gain that follows the track's energy
        bass_min = max(bass_energy * 0.5, 0.1)
        bass_max = min(bass_energy * 1.5, 0.8)
        suggestions['dynamic_effects']['bass_gain'] = f'perlin.range({bass_min:.2f}, {bass_max:.2f}).slow(8)'

        mid_min = max(mid_energy * 0.7, 0.3)
        mid_max = min(mid_energy * 1.3, 1.0)
        suggestions['dynamic_effects']['mid_gain'] = f'perlin.range({mid_min:.2f}, {mid_max:.2f}).slow(4)'

        # Filter sweep that follows energy
        lpf_min = 500 if spectrum['brightness'] == 'dark' else 1000
        lpf_max = 3000 if spectrum['brightness'] == 'dark' else 6000
        suggestions['dynamic_effects']['lpf_sweep'] = f'perlin.range({lpf_min}, {lpf_max}).slow(8)'

        # Drums can be more punchy during high-energy sections
        suggestions['dynamic_effects']['drum_gain'] = f'sine.range(0.4, 0.7).slow(16)'
    else:
        # Static values for compressed/consistent tracks
        suggestions['dynamic_effects']['bass_gain'] = f'{bass_energy:.2f}'
        suggestions['dynamic_effects']['mid_gain'] = f'{mid_energy:.2f}'
        suggestions['dynamic_effects']['lpf_sweep'] = '2500'
        suggestions['dynamic_effects']['drum_gain'] = '0.5'

    # === BASS VOICE ===
    if bass_energy > 0.3:
        suggestions['bass']['lpf'] = 150  # Heavy sub bass
        suggestions['bass']['distort'] = 0.5
    else:
        suggestions['bass']['lpf'] = 400  # Lighter bass
        suggestions['bass']['distort'] = 0.2

    suggestions['bass']['gain'] = max(bass_energy, 0.15)  # Floor at 0.15 to prevent silence
    suggestions['bass']['attack'] = 0.001
    suggestions['bass']['decay'] = 0.15
    suggestions['bass']['sustain'] = 0.5
    suggestions['bass']['release'] = 0.1

    # === MID VOICE ===
    suggestions['mid']['lpf'] = 4000 if spectrum['brightness'] == 'bright' else 2500
    suggestions['mid']['gain'] = max(mid_energy, 0.4)  # Floor at 0.4 to prevent weak mids
    suggestions['mid']['room'] = 0.2 if mid_energy > 0.2 else 0.1

    if timbre['is_tonal']:
        suggestions['mid']['phaser'] = 0.3
        suggestions['mid']['vib'] = 4.0
        suggestions['mid']['vibmod'] = 0.08

    suggestions['mid']['attack'] = 0.01
    suggestions['mid']['decay'] = 0.1
    suggestions['mid']['sustain'] = 0.7
    suggestions['mid']['release'] = 0.15

    # === HIGH VOICE ===
    suggestions['high']['lpf'] = 8000 if spectrum['brightness'] == 'bright' else 5000
    suggestions['high']['gain'] = max(high_energy, 0.3)  # Floor at 0.3 to prevent weak highs
    suggestions['high']['room'] = 0.3
    suggestions['high']['delay'] = 0.2 if high_energy > 0.1 else 0
    suggestions['high']['delaytime'] = 60 / rhythm['tempo'] / 4  # Sixteenth note
    suggestions['high']['delayfeedback'] = 0.3

    suggestions['high']['attack'] = 0.02
    suggestions['high']['decay'] = 0.15
    suggestions['high']['sustain'] = 0.5
    suggestions['high']['release'] = 0.2

    # === DRUMS ===
    suggestions['drums']['gain'] = 0.7 if dynamics['is_punchy'] else 0.5
    suggestions['drums']['room'] = 0.1 if dynamics['is_compressed'] else 0.2

    if timbre['is_noisy']:
        suggestions['drums']['crush'] = 12  # Lo-fi drums

    # === GLOBAL ===
    if rhythm['has_swing']:
        suggestions['global']['swing'] = (rhythm['swing_ratio'] - 1) / 2

    if dynamics['is_compressed']:
        suggestions['global']['style'] = 'electronic'
    elif rhythm['has_swing']:
        suggestions['global']['style'] = 'jazz'
    else:
        suggestions['global']['style'] = 'electronic'

    # Sound selection based on timbre
    if spectrum['brightness'] == 'bright':
        suggestions['bass']['sound'] = 'sawtooth'
        suggestions['mid']['sound'] = 'gm_lead_2_sawtooth'
        suggestions['high']['sound'] = 'triangle'
    elif spectrum['brightness'] == 'dark':
        suggestions['bass']['sound'] = 'sine'
        suggestions['mid']['sound'] = 'gm_pad_poly'
        suggestions['high']['sound'] = 'gm_vibraphone'
    else:
        suggestions['bass']['sound'] = 'supersaw'
        suggestions['mid']['sound'] = 'square'
        suggestions['high']['sound'] = 'gm_lead_5_charang'

    # === ARRANGEMENT MASK (for builds/drops) ===
    # Generate a mask pattern based on energy curve
    if dynamics.get('has_dynamics', False) and 'energy_curve' in dynamics:
        energy = dynamics['energy_curve']
        # Convert energy curve to mask pattern
        mask_pattern = []
        for e in energy:
            if e > 0.7:
                mask_pattern.append('1')  # Full
            elif e > 0.4:
                mask_pattern.append('0.7')  # Medium
            else:
                mask_pattern.append('0.3')  # Low
        suggestions['dynamic_effects']['energy_mask'] = f'"<{" ".join(mask_pattern)}>"'

    return suggestions

def generate_effect_chain(voice, params):
    """Generate Strudel effect chain string for a voice."""
    chain = []

    # Order matters for Strudel
    effect_order = [
        'sound', 'lpf', 'hpf', 'gain', 'distort', 'crush', 'coarse',
        'attack', 'decay', 'sustain', 'release',
        'phaser', 'phaserdepth', 'vib', 'vibmod', 'tremolo', 'tremolodepth',
        'fm', 'fmh',
        'room', 'size', 'delay', 'delaytime', 'delayfeedback',
        'pan'
    ]

    for effect in effect_order:
        if effect in params:
            value = params[effect]
            if effect == 'sound':
                chain.append(f'.sound("{value}")')
            elif isinstance(value, float):
                chain.append(f'.{effect}({value:.3f})')
            else:
                chain.append(f'.{effect}({value})')

    return ''.join(chain)

def analyze_audio(audio_path, output_json=None):
    """Full audio analysis pipeline."""
    print(f"Analyzing: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

    # Run all analyses
    analysis = {
        'spectrum': analyze_frequency_spectrum(y, sr),
        'dynamics': analyze_dynamics(y, sr),
        'rhythm': analyze_rhythm(y, sr),
        'timbre': analyze_timbre(y, sr)
    }

    # Generate Strudel parameter suggestions
    suggestions = suggest_strudel_params(analysis)

    # Generate effect chains
    effect_chains = {}
    for voice in ['bass', 'mid', 'high', 'drums']:
        effect_chains[voice] = generate_effect_chain(voice, suggestions[voice])

    # Apply minimum gain floors to prevent silence
    def floor_gain(val, minimum, default):
        """Ensure gain is at least minimum value."""
        return max(val if val is not None else default, minimum)

    # === FULLY AI-DRIVEN RENDERER PARAMETERS ===
    # All values derived from analysis - no hardcoded constants
    bands = analysis['spectrum']['band_energy']
    dynamics = analysis['dynamics']
    rhythm = analysis['rhythm']
    spectrum = analysis['spectrum']

    # Original frequency distribution (these are the TARGET ratios)
    orig_sub_bass = bands['sub_bass']
    orig_bass = bands['bass']
    orig_low_mid = bands['low_mid']
    orig_mid = bands['mid']
    orig_high_mid = bands['high_mid']
    orig_high = bands['high']

    # Total energy for normalization
    total_energy = sum(bands.values()) + 1e-6

    # Derive renderer mix to match original frequency balance
    # Each renderer voice contributes to specific frequency bands:
    # - kick: sub_bass + bass (20-250Hz)
    # - snare: low_mid + some mid (200-1000Hz)
    # - hh: high_mid + high (2000Hz+)
    # - bass synth: bass + low_mid (60-500Hz)
    # - vox/mid synth: low_mid + mid (250-2000Hz)
    # - lead/high synth: mid + high_mid (500-4000Hz)

    renderer_mix = {
        # Drums - scale to match original's drum-like frequencies
        'kick_gain': (orig_sub_bass + orig_bass) / total_energy * 2.0,
        'snare_gain': (orig_low_mid * 0.5 + orig_mid * 0.3) / total_energy * 2.0,
        'hh_gain': (orig_high_mid + orig_high) / total_energy * 2.0,

        # Bass synth - match original bass content
        'bass_gain': (orig_bass + orig_low_mid * 0.5) / total_energy * 2.0,

        # Mid synths - match original mid content (usually dominant)
        'vox_gain': (orig_low_mid + orig_mid * 0.3) / total_energy * 2.0,
        'stab_gain': (orig_low_mid * 0.5 + orig_mid * 0.5) / total_energy * 2.0,

        # Lead synth - match original mid/high-mid (the melodic content)
        'lead_gain': (orig_mid + orig_high_mid * 0.5) / total_energy * 2.0,
    }

    # Derive synth parameters from spectral analysis
    centroid = spectrum['centroid_mean']
    brightness = spectrum['brightness']

    renderer_synth = {
        # Filter cutoffs derived from spectral centroid
        'bass_lpf': min(400, centroid * 0.15),
        'mid_lpf': min(4000, centroid * 1.5),
        'high_lpf': min(12000, centroid * 4),

        # Envelope from dynamics
        'attack': 0.005 if dynamics['is_punchy'] else 0.02,
        'decay': 0.1 if dynamics['is_compressed'] else 0.2,
        'sustain': 0.5 if dynamics['is_compressed'] else 0.7,
        'release': 0.1 if dynamics['is_punchy'] else 0.3,

        # Distortion from spectral flatness (noisy = more harmonics = distortion)
        'distortion': min(0.8, spectrum['flatness_mean'] * 10),

        # Reverb from dynamics range (more dynamic = more space)
        'reverb': 0.1 if dynamics['is_compressed'] else min(0.4, dynamics['dynamic_range_db'] / 60),
    }

    # Derive tempo-synced effects
    beat_duration = 60.0 / rhythm['tempo']
    renderer_timing = {
        'delay_time': beat_duration / 4,  # 16th note
        'delay_feedback': 0.3 if dynamics['is_compressed'] else 0.5,
        'tremolo_rate': rhythm['tempo'] / 60,  # Sync to tempo
    }

    result = {
        'analysis': analysis,
        'suggestions': suggestions,
        'effect_chains': effect_chains,
        'renderer_mix': renderer_mix,
        'renderer_synth': renderer_synth,
        'renderer_timing': renderer_timing,
        'target_bands': bands,  # Store original bands for comparison
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved to: {output_json}")

    return result

def print_report(result):
    """Print human-readable analysis report."""
    analysis = result['analysis']
    suggestions = result['suggestions']

    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS FOR STRUDEL")
    print("=" * 60)

    print("\n--- FREQUENCY SPECTRUM ---")
    spectrum = analysis['spectrum']
    print(f"Brightness: {spectrum['brightness']} (centroid: {spectrum['centroid_mean']:.0f} Hz)")
    print("Band energy distribution:")
    for band, energy in spectrum['band_energy'].items():
        bar = "█" * int(energy * 50)
        print(f"  {band:12} [{bar:<25}] {energy*100:5.1f}%")

    print("\n--- DYNAMICS ---")
    dynamics = analysis['dynamics']
    print(f"Dynamic range: {dynamics['dynamic_range_db']:.1f} dB")
    print(f"Compressed: {'Yes' if dynamics['is_compressed'] else 'No'}")
    print(f"Punchy transients: {'Yes' if dynamics['is_punchy'] else 'No'}")

    print("\n--- RHYTHM ---")
    rhythm = analysis['rhythm']
    print(f"Tempo: {rhythm['tempo']:.1f} BPM")
    print(f"Swing: {'Yes (ratio: {:.2f})'.format(rhythm['swing_ratio']) if rhythm['has_swing'] else 'No'}")

    print("\n--- SUGGESTED STRUDEL EFFECTS ---")
    for voice in ['bass', 'mid', 'high', 'drums']:
        chain = result['effect_chains'][voice]
        print(f"\n{voice.upper()}:")
        print(f"  {chain}")

    print("\n--- RENDERER MIX LEVELS ---")
    for key, value in result['renderer_mix'].items():
        print(f"  {key}: {value:.2f}")

    print("\n--- DYNAMIC STRUDEL EXPRESSIONS ---")
    if 'dynamic_effects' in suggestions:
        for key, expr in suggestions.get('dynamic_effects', {}).items():
            print(f"  {key}: {expr}")

    print("\n--- EXAMPLE STRUDEL CODE ---")
    print("// AI-generated dynamic effects:")
    if 'dynamic_effects' in suggestions:
        de = suggestions['dynamic_effects']
        print(f"let bassGain = {de.get('bass_gain', '0.5')}")
        print(f"let midGain = {de.get('mid_gain', '0.5')}")
        print(f"let filterSweep = {de.get('lpf_sweep', '2500')}")
        if 'energy_mask' in de:
            print(f"let energyMask = {de.get('energy_mask')}")

    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Analyze audio for optimal Strudel parameters')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('-j', '--json', action='store_true', help='Output only JSON')

    args = parser.parse_args()

    result = analyze_audio(args.input, args.output)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

if __name__ == '__main__':
    main()
