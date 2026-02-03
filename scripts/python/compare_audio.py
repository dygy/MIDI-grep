#!/usr/bin/env python3
"""
Compare rendered audio with original stems.
Analyzes spectral, rhythmic, and timbral differences.
"""

import argparse
import json
import numpy as np
import librosa
import sys
from scipy import signal
from scipy.spatial.distance import cosine

def load_audio(path, sr=22050, duration=None):
    """Load audio file and return mono signal."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None

def compute_spectral_features(y, sr=22050):
    """Compute spectral features for comparison."""
    features = {}

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['centroid_mean'] = float(np.mean(centroid))
    features['centroid_std'] = float(np.std(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['bandwidth_mean'] = float(np.mean(bandwidth))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = float(np.mean(rolloff))

    # Spectral flatness (noise vs tonal)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['flatness_mean'] = float(np.mean(flatness))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    return features

def compute_mfcc_features(y, sr=22050, n_mfcc=13):
    """Compute MFCC features for timbral comparison."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def compute_chroma_features(y, sr=22050):
    """Compute chroma features for harmonic comparison."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return np.mean(chroma, axis=1)

def compute_rhythm_features(y, sr=22050):
    """Compute rhythm/tempo features."""
    features = {}

    # Tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # Handle both old and new librosa versions
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    features['tempo'] = float(tempo)
    features['beat_count'] = len(beats)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength_mean'] = float(np.mean(onset_env))
    features['onset_strength_std'] = float(np.std(onset_env))

    # Tempogram for rhythm pattern
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    features['rhythm_regularity'] = float(np.mean(np.max(tempogram, axis=0)))

    return features

def compute_frequency_bands(y, sr=22050):
    """Analyze energy in frequency bands (sub-bass, bass, mids, highs)."""
    # Compute spectrogram
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Define bands
    bands = {
        'sub_bass': (20, 60),      # Sub bass
        'bass': (60, 250),         # Bass
        'low_mid': (250, 500),     # Low mids
        'mid': (500, 2000),        # Mids
        'high_mid': (2000, 4000),  # High mids
        'high': (4000, 10000),     # Highs
    }

    band_energy = {}
    total_energy = np.sum(S ** 2)

    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[band_name] = float(np.sum(S[mask, :] ** 2) / max(total_energy, 1e-10))

    return band_energy

def compare_audio(original_path, rendered_path, duration=60):
    """Compare original and rendered audio files."""
    print(f"Loading original: {original_path}")
    original = load_audio(original_path, duration=duration)

    print(f"Loading rendered: {rendered_path}")
    rendered = load_audio(rendered_path, duration=duration)

    if original is None or rendered is None:
        return None

    # Normalize lengths
    min_len = min(len(original), len(rendered))
    original = original[:min_len]
    rendered = rendered[:min_len]

    print(f"Analyzing {min_len / 22050:.1f} seconds of audio...")

    results = {
        'duration_analyzed': min_len / 22050,
        'original': {},
        'rendered': {},
        'comparison': {}
    }

    # Compute features for both
    print("Computing spectral features...")
    results['original']['spectral'] = compute_spectral_features(original)
    results['rendered']['spectral'] = compute_spectral_features(rendered)

    print("Computing MFCC features...")
    orig_mfcc = compute_mfcc_features(original)
    rend_mfcc = compute_mfcc_features(rendered)

    print("Computing chroma features...")
    orig_chroma = compute_chroma_features(original)
    rend_chroma = compute_chroma_features(rendered)

    print("Computing rhythm features...")
    results['original']['rhythm'] = compute_rhythm_features(original)
    results['rendered']['rhythm'] = compute_rhythm_features(rendered)

    print("Computing frequency band energy...")
    results['original']['bands'] = compute_frequency_bands(original)
    results['rendered']['bands'] = compute_frequency_bands(rendered)

    # Compute similarity scores
    print("Computing similarity scores...")

    # MFCC similarity (timbre)
    mfcc_sim = 1 - cosine(orig_mfcc, rend_mfcc)
    results['comparison']['mfcc_similarity'] = float(mfcc_sim)

    # Chroma similarity (harmony)
    chroma_sim = 1 - cosine(orig_chroma, rend_chroma)
    results['comparison']['chroma_similarity'] = float(chroma_sim)

    # Spectral centroid difference (brightness)
    centroid_diff = abs(results['original']['spectral']['centroid_mean'] -
                       results['rendered']['spectral']['centroid_mean'])
    centroid_ratio = min(results['original']['spectral']['centroid_mean'],
                        results['rendered']['spectral']['centroid_mean']) / \
                    max(results['original']['spectral']['centroid_mean'],
                        results['rendered']['spectral']['centroid_mean'])
    results['comparison']['brightness_similarity'] = float(centroid_ratio)

    # Tempo similarity
    tempo_diff = abs(results['original']['rhythm']['tempo'] -
                    results['rendered']['rhythm']['tempo'])
    tempo_sim = 1 - min(tempo_diff / 20, 1)  # 20 BPM tolerance
    results['comparison']['tempo_similarity'] = float(tempo_sim)

    # Energy distribution similarity
    orig_bands = np.array(list(results['original']['bands'].values()))
    rend_bands = np.array(list(results['rendered']['bands'].values()))
    band_sim = 1 - cosine(orig_bands, rend_bands) if np.sum(orig_bands) > 0 else 0
    results['comparison']['frequency_balance_similarity'] = float(band_sim)

    # RMS energy ratio
    rms_ratio = min(results['original']['spectral']['rms_mean'],
                   results['rendered']['spectral']['rms_mean']) / \
               max(results['original']['spectral']['rms_mean'],
                   results['rendered']['spectral']['rms_mean'], 1e-10)
    results['comparison']['energy_similarity'] = float(rms_ratio)

    # Overall similarity (weighted average)
    weights = {
        'mfcc_similarity': 0.25,
        'chroma_similarity': 0.20,
        'brightness_similarity': 0.15,
        'tempo_similarity': 0.15,
        'frequency_balance_similarity': 0.15,
        'energy_similarity': 0.10
    }

    overall = sum(results['comparison'][k] * w for k, w in weights.items())
    results['comparison']['overall_similarity'] = float(overall)

    # Generate insights
    results['insights'] = generate_insights(results)

    return results

def generate_insights(results):
    """Generate human-readable insights from comparison."""
    insights = []
    comp = results['comparison']
    orig = results['original']
    rend = results['rendered']

    overall = comp['overall_similarity'] * 100
    insights.append(f"Overall similarity: {overall:.1f}%")

    # Tempo
    tempo_diff = abs(orig['rhythm']['tempo'] - rend['rhythm']['tempo'])
    if tempo_diff < 2:
        insights.append(f"Tempo: Excellent match ({orig['rhythm']['tempo']:.0f} vs {rend['rhythm']['tempo']:.0f} BPM)")
    elif tempo_diff < 5:
        insights.append(f"Tempo: Good match ({orig['rhythm']['tempo']:.0f} vs {rend['rhythm']['tempo']:.0f} BPM)")
    else:
        insights.append(f"Tempo: Mismatch ({orig['rhythm']['tempo']:.0f} vs {rend['rhythm']['tempo']:.0f} BPM)")

    # Timbre
    mfcc_pct = comp['mfcc_similarity'] * 100
    if mfcc_pct > 80:
        insights.append(f"Timbre: Very similar ({mfcc_pct:.0f}%)")
    elif mfcc_pct > 60:
        insights.append(f"Timbre: Moderately similar ({mfcc_pct:.0f}%)")
    else:
        insights.append(f"Timbre: Different character ({mfcc_pct:.0f}%)")

    # Harmony
    chroma_pct = comp['chroma_similarity'] * 100
    if chroma_pct > 80:
        insights.append(f"Harmony: Strong match ({chroma_pct:.0f}%)")
    elif chroma_pct > 60:
        insights.append(f"Harmony: Partial match ({chroma_pct:.0f}%)")
    else:
        insights.append(f"Harmony: Different key/chords ({chroma_pct:.0f}%)")

    # Frequency balance
    insights.append("")
    insights.append("Frequency balance:")
    for band in ['sub_bass', 'bass', 'mid', 'high']:
        orig_val = orig['bands'].get(band, 0) * 100
        rend_val = rend['bands'].get(band, 0) * 100
        diff = rend_val - orig_val
        direction = "+" if diff > 0 else ""
        insights.append(f"  {band}: orig {orig_val:.1f}% vs rend {rend_val:.1f}% ({direction}{diff:.1f}%)")

    # Suggestions
    insights.append("")
    insights.append("Suggestions to improve similarity:")

    if rend['bands']['bass'] < orig['bands']['bass'] * 0.7:
        insights.append("  - Increase bass presence (add sub-octave, boost LPF cutoff)")
    if rend['bands']['bass'] > orig['bands']['bass'] * 1.5:
        insights.append("  - Reduce bass (lower gain, reduce distortion)")

    if rend['bands']['high'] < orig['bands']['high'] * 0.5:
        insights.append("  - Add more high frequency content (hi-hats, bright synths)")
    if rend['bands']['high'] > orig['bands']['high'] * 2:
        insights.append("  - Reduce highs (lower LPF, reduce hi-hat volume)")

    if comp['chroma_similarity'] < 0.6:
        insights.append("  - Check key detection - harmony doesn't match well")

    if rend['spectral']['rms_mean'] < orig['spectral']['rms_mean'] * 0.5:
        insights.append("  - Rendered output is too quiet - increase overall gain")

    return insights

def print_report(results):
    """Print formatted comparison report."""
    print("\n" + "=" * 60)
    print("AUDIO COMPARISON REPORT")
    print("=" * 60)

    print(f"\nDuration analyzed: {results['duration_analyzed']:.1f} seconds")

    print("\n--- SIMILARITY SCORES ---")
    for key, value in results['comparison'].items():
        label = key.replace('_', ' ').title()
        bar_len = int(value * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{label:30} [{bar}] {value*100:5.1f}%")

    print("\n--- INSIGHTS ---")
    for insight in results['insights']:
        print(insight)

    print("\n--- DETAILED METRICS ---")
    print("\nOriginal audio:")
    print(f"  Tempo: {results['original']['rhythm']['tempo']:.1f} BPM")
    print(f"  Brightness: {results['original']['spectral']['centroid_mean']:.0f} Hz")
    print(f"  RMS Energy: {results['original']['spectral']['rms_mean']:.4f}")

    print("\nRendered audio:")
    print(f"  Tempo: {results['rendered']['rhythm']['tempo']:.1f} BPM")
    print(f"  Brightness: {results['rendered']['spectral']['centroid_mean']:.0f} Hz")
    print(f"  RMS Energy: {results['rendered']['spectral']['rms_mean']:.4f}")

    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Compare rendered audio with original')
    parser.add_argument('original', help='Path to original audio file')
    parser.add_argument('rendered', help='Path to rendered audio file')
    parser.add_argument('-d', '--duration', type=float, default=60,
                       help='Max duration to analyze in seconds')
    parser.add_argument('-j', '--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    results = compare_audio(args.original, args.rendered, args.duration)

    if results is None:
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

if __name__ == '__main__':
    main()
