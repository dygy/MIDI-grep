#!/usr/bin/env python3
"""
Aggregate analysis results from multiple tracks to find common genre characteristics.
Used to improve Brazilian funk/phonk detection and generation.
"""

import argparse
import json
import os
import glob
import numpy as np
from collections import defaultdict

def load_analyses(cache_dir=".cache/stems"):
    """Load all analysis.json files from cache directories."""
    analyses = []

    for analysis_file in glob.glob(f"{cache_dir}/*/analysis.json"):
        try:
            with open(analysis_file, 'r') as f:
                data = json.load(f)
                video_id = os.path.basename(os.path.dirname(analysis_file))
                data['video_id'] = video_id
                analyses.append(data)
                print(f"Loaded: {video_id}")
        except Exception as e:
            print(f"Error loading {analysis_file}: {e}")

    return analyses

def aggregate_characteristics(analyses):
    """Find common characteristics across all tracks."""
    if not analyses:
        return None

    result = {
        'track_count': len(analyses),
        'bpm': {'values': [], 'mean': 0, 'std': 0, 'range': [0, 0]},
        'brightness': {'values': [], 'distribution': {}},
        'dynamics': {'compressed_count': 0, 'punchy_count': 0, 'dynamic_range_mean': 0},
        'rhythm': {'swing_count': 0, 'regularity_mean': 0},
        'spectrum': {
            'sub_bass': {'mean': 0, 'std': 0},
            'bass': {'mean': 0, 'std': 0},
            'low_mid': {'mean': 0, 'std': 0},
            'mid': {'mean': 0, 'std': 0},
            'high_mid': {'mean': 0, 'std': 0},
            'high': {'mean': 0, 'std': 0},
        },
        'common_characteristics': [],
        'detection_rules': {}
    }

    # Collect values
    bpm_values = []
    brightness_values = []
    dynamic_ranges = []
    regularities = []
    band_energies = defaultdict(list)

    for a in analyses:
        # BPM
        if 'analysis' in a and 'rhythm' in a['analysis']:
            bpm = a['analysis']['rhythm'].get('tempo', 0)
            if bpm > 0:
                bpm_values.append(bpm)
                # Also check for double-time (common in funk)
                if bpm < 100:
                    bpm_values.append(bpm * 2)  # Add double-time interpretation

        # Brightness
        if 'analysis' in a and 'spectrum' in a['analysis']:
            brightness = a['analysis']['spectrum'].get('brightness', 'neutral')
            brightness_values.append(brightness)

        # Dynamics
        if 'analysis' in a and 'dynamics' in a['analysis']:
            dyn = a['analysis']['dynamics']
            if dyn.get('is_compressed', False):
                result['dynamics']['compressed_count'] += 1
            if dyn.get('is_punchy', False):
                result['dynamics']['punchy_count'] += 1
            if 'dynamic_range_db' in dyn:
                dynamic_ranges.append(dyn['dynamic_range_db'])

        # Rhythm
        if 'analysis' in a and 'rhythm' in a['analysis']:
            rhythm = a['analysis']['rhythm']
            if rhythm.get('has_swing', False):
                result['rhythm']['swing_count'] += 1
            if 'rhythm_regularity' in rhythm:
                regularities.append(rhythm['rhythm_regularity'])

        # Spectrum bands
        if 'analysis' in a and 'spectrum' in a['analysis']:
            bands = a['analysis']['spectrum'].get('band_energy', {})
            for band, energy in bands.items():
                band_energies[band].append(energy)

    # Calculate statistics
    if bpm_values:
        result['bpm']['values'] = sorted(set([int(b) for b in bpm_values]))
        result['bpm']['mean'] = float(np.mean(bpm_values))
        result['bpm']['std'] = float(np.std(bpm_values))
        result['bpm']['range'] = [float(min(bpm_values)), float(max(bpm_values))]

    if brightness_values:
        result['brightness']['values'] = brightness_values
        for b in brightness_values:
            result['brightness']['distribution'][b] = result['brightness']['distribution'].get(b, 0) + 1

    if dynamic_ranges:
        result['dynamics']['dynamic_range_mean'] = float(np.mean(dynamic_ranges))

    if regularities:
        result['rhythm']['regularity_mean'] = float(np.mean(regularities))

    for band, values in band_energies.items():
        if values:
            result['spectrum'][band] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(min(values)),
                'max': float(max(values))
            }

    # Determine common characteristics
    n = len(analyses)

    # BPM range for Brazilian funk
    if result['bpm']['mean'] > 0:
        bpm_low = result['bpm']['mean'] - 2 * result['bpm']['std']
        bpm_high = result['bpm']['mean'] + 2 * result['bpm']['std']
        result['common_characteristics'].append(f"BPM typically {bpm_low:.0f}-{bpm_high:.0f}")

    # Compression
    if result['dynamics']['compressed_count'] > n * 0.5:
        result['common_characteristics'].append("Usually compressed (limited dynamic range)")

    # Punchy transients
    if result['dynamics']['punchy_count'] > n * 0.5:
        result['common_characteristics'].append("Punchy transients (strong attacks)")

    # Brightness
    most_common_brightness = max(result['brightness']['distribution'].items(),
                                  key=lambda x: x[1], default=('neutral', 0))[0]
    result['common_characteristics'].append(f"Typically {most_common_brightness} brightness")

    # Frequency balance
    if result['spectrum'].get('mid', {}).get('mean', 0) > 0.3:
        result['common_characteristics'].append("Mid-heavy frequency balance")
    if result['spectrum'].get('bass', {}).get('mean', 0) > 0.2:
        result['common_characteristics'].append("Strong bass presence")

    # Generate detection rules
    result['detection_rules'] = {
        'bpm_range': [
            max(60, result['bpm']['mean'] - 2 * max(result['bpm']['std'], 10)),
            min(180, result['bpm']['mean'] + 2 * max(result['bpm']['std'], 10))
        ],
        'mid_energy_threshold': max(0.2, result['spectrum'].get('mid', {}).get('mean', 0.3) - 0.1),
        'bass_energy_threshold': max(0.05, result['spectrum'].get('bass', {}).get('mean', 0.1) - 0.05),
        'compression_likely': result['dynamics']['compressed_count'] > n * 0.5,
        'typical_brightness': most_common_brightness,
    }

    return result

def generate_go_detection_code(characteristics):
    """Generate Go code for improved Brazilian funk detection."""
    rules = characteristics.get('detection_rules', {})

    code = f'''
// Auto-generated Brazilian Funk detection rules based on {characteristics['track_count']} tracks
// BPM range: {rules.get('bpm_range', [125, 145])}
// Mid energy threshold: {rules.get('mid_energy_threshold', 0.3):.2f}
// Bass energy threshold: {rules.get('bass_energy_threshold', 0.05):.2f}

func shouldUseBrazilianFunkMode(analysis *analysis.Result, cleanup *midi.CleanupResult) bool {{
    score := 0.0

    // BPM check (expanded range based on analysis)
    bpmLow := {rules.get('bpm_range', [125, 145])[0]:.0f}
    bpmHigh := {rules.get('bpm_range', [125, 145])[1]:.0f}
    if analysis.BPM >= bpmLow && analysis.BPM <= bpmHigh {{
        score += 2.0
    }}

    // Also check half-time
    if analysis.BPM >= bpmLow/2 && analysis.BPM <= bpmHigh/2 {{
        score += 1.5  // Half-time feel
    }}

    // Mid-range dominance (characteristic of vocal chops)
    // Based on average mid energy: {characteristics['spectrum'].get('mid', {}).get('mean', 0):.1%}
    midThreshold := {rules.get('mid_energy_threshold', 0.3):.2f}
    if cleanup.MidRatio > midThreshold {{
        score += 2.5
    }}

    // Sparse bass (not bass-heavy like other electronic)
    // Based on average bass energy: {characteristics['spectrum'].get('bass', {}).get('mean', 0):.1%}
    if cleanup.BassRatio < 0.15 {{
        score += 1.5
    }}

    // Note density check (sparse patterns typical)
    // Average regularity: {characteristics['rhythm'].get('regularity_mean', 0):.2f}
    avgDuration := cleanup.TotalDuration / float64(len(cleanup.Notes))
    if avgDuration > 0.2 {{  // Sparse, not busy
        score += 1.5
    }}

    return score >= 5.0
}}
'''
    return code

def print_report(characteristics):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("BRAZILIAN FUNK/PHONK GENRE ANALYSIS")
    print(f"Based on {characteristics['track_count']} tracks")
    print("=" * 70)

    print("\n--- BPM ANALYSIS ---")
    print(f"Mean BPM: {characteristics['bpm']['mean']:.1f}")
    print(f"Std Dev: {characteristics['bpm']['std']:.1f}")
    print(f"Range: {characteristics['bpm']['range'][0]:.0f} - {characteristics['bpm']['range'][1]:.0f}")
    print(f"Common values: {characteristics['bpm']['values'][:10]}")

    print("\n--- FREQUENCY SPECTRUM ---")
    for band in ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']:
        if band in characteristics['spectrum']:
            s = characteristics['spectrum'][band]
            bar = "█" * int(s.get('mean', 0) * 50)
            print(f"  {band:12} [{bar:<25}] {s.get('mean', 0)*100:5.1f}% (±{s.get('std', 0)*100:.1f}%)")

    print("\n--- DYNAMICS ---")
    print(f"Compressed tracks: {characteristics['dynamics']['compressed_count']}/{characteristics['track_count']}")
    print(f"Punchy tracks: {characteristics['dynamics']['punchy_count']}/{characteristics['track_count']}")
    print(f"Avg dynamic range: {characteristics['dynamics']['dynamic_range_mean']:.1f} dB")

    print("\n--- RHYTHM ---")
    print(f"Tracks with swing: {characteristics['rhythm']['swing_count']}/{characteristics['track_count']}")
    print(f"Avg regularity: {characteristics['rhythm']['regularity_mean']:.2f}")

    print("\n--- COMMON CHARACTERISTICS ---")
    for c in characteristics['common_characteristics']:
        print(f"  • {c}")

    print("\n--- DETECTION RULES ---")
    for rule, value in characteristics['detection_rules'].items():
        print(f"  {rule}: {value}")

    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Aggregate genre analysis from multiple tracks')
    parser.add_argument('--cache-dir', default='.cache/stems', help='Cache directory')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('--go-code', action='store_true', help='Generate Go detection code')

    args = parser.parse_args()

    analyses = load_analyses(args.cache_dir)

    if not analyses:
        print("No analysis files found!")
        return

    characteristics = aggregate_characteristics(analyses)

    print_report(characteristics)

    if args.go_code:
        print("\n--- GENERATED GO CODE ---")
        print(generate_go_detection_code(characteristics))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(characteristics, f, indent=2)
        print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()
