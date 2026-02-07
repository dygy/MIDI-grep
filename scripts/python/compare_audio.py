#!/usr/bin/env python3
"""
Compare rendered audio with original stems.
Analyzes spectral, rhythmic, and timbral differences.
"""

import argparse
import json
import numpy as np
import librosa
import librosa.display
import sys
from scipy import signal
from scipy.spatial.distance import cosine

# Global flag for quiet mode (suppress status messages to stdout)
_QUIET_MODE = False

def log(msg):
    """Print message to stderr in quiet mode, stdout otherwise."""
    if _QUIET_MODE:
        print(msg, file=sys.stderr)
    else:
        print(msg)

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

def compare_audio(original_path, rendered_path, duration=60, synth_config_path=None):
    """Compare original and rendered audio files.

    Args:
        original_path: Path to original audio
        rendered_path: Path to rendered audio
        duration: Max duration to analyze
        synth_config_path: Optional path to synth_config.json for AI-derived tolerance
    """
    # Load AI-derived tempo tolerance from synth config if available
    ai_tempo_tolerance = None
    if synth_config_path:
        try:
            with open(synth_config_path, 'r') as f:
                synth_config = json.load(f)
            ai_tempo_tolerance = synth_config.get('tempo', {}).get('tempo_tolerance')
            if ai_tempo_tolerance:
                log(f"Using AI-derived tempo tolerance: {ai_tempo_tolerance:.1%}")
        except Exception as e:
            log(f"Warning: Could not load synth config: {e}")

    log(f"Loading original: {original_path}")
    original = load_audio(original_path, duration=duration)

    log(f"Loading rendered: {rendered_path}")
    rendered = load_audio(rendered_path, duration=duration)

    if original is None or rendered is None:
        return None

    # Normalize lengths
    min_len = min(len(original), len(rendered))
    original = original[:min_len]
    rendered = rendered[:min_len]

    log(f"Analyzing {min_len / 22050:.1f} seconds of audio...")

    results = {
        'duration_analyzed': min_len / 22050,
        'original': {},
        'rendered': {},
        'comparison': {}
    }

    # Compute features for both
    log("Computing spectral features...")
    results['original']['spectral'] = compute_spectral_features(original)
    results['rendered']['spectral'] = compute_spectral_features(rendered)

    log("Computing MFCC features...")
    orig_mfcc = compute_mfcc_features(original)
    rend_mfcc = compute_mfcc_features(rendered)

    log("Computing chroma features...")
    orig_chroma = compute_chroma_features(original)
    rend_chroma = compute_chroma_features(rendered)

    log("Computing rhythm features...")
    results['original']['rhythm'] = compute_rhythm_features(original)
    results['rendered']['rhythm'] = compute_rhythm_features(rendered)

    log("Computing frequency band energy...")
    results['original']['bands'] = compute_frequency_bands(original)
    results['rendered']['bands'] = compute_frequency_bands(rendered)

    # Compute similarity scores
    log("Computing similarity scores...")

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

    # Tempo similarity with octave handling (half-time/double-time are musically equivalent)
    orig_tempo = results['original']['rhythm']['tempo']
    rend_tempo = results['rendered']['rhythm']['tempo']

    # Check tempo at various musical ratios (including non-standard ones for synthesized audio)
    # Synthesized audio often triggers beat tracker at faster rates due to transients
    tempo_ratios = [1.0, 2.0, 0.5, 3.0, 1/3, 4.0, 0.25, 1.5, 0.67, 1.6, 0.625]
    best_tempo_diff = float('inf')
    best_ratio = 1.0

    for ratio in tempo_ratios:
        adjusted_rend = rend_tempo * ratio
        diff = abs(orig_tempo - adjusted_rend)
        if diff < best_tempo_diff:
            best_tempo_diff = diff
            best_ratio = ratio

    # Store the best match info
    results['comparison']['tempo_ratio_used'] = best_ratio
    results['comparison']['tempo_diff_bpm'] = best_tempo_diff

    # Calculate similarity using AI-derived or default tolerance
    # AI-derived tolerance accounts for beat regularity and tempo estimate variance
    if ai_tempo_tolerance:
        tolerance = orig_tempo * ai_tempo_tolerance  # AI-derived
    else:
        tolerance = orig_tempo * 0.15  # Default 15% if no config
    tempo_sim = max(0, 1 - best_tempo_diff / tolerance)
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

def setup_chart_style():
    """Setup matplotlib style for dark theme charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Enhanced dark theme styling
    plt.rcParams['text.color'] = '#c9d1d9'
    plt.rcParams['axes.labelcolor'] = '#c9d1d9'
    plt.rcParams['axes.edgecolor'] = '#30363d'
    plt.rcParams['xtick.color'] = '#8b949e'
    plt.rcParams['ytick.color'] = '#8b949e'
    plt.rcParams['axes.titlecolor'] = '#c9d1d9'
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.family'] = 'sans-serif'

    return plt

def generate_frequency_bands_chart(results, output_path):
    """Generate frequency band distribution chart with enhanced styling."""
    plt = setup_chart_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    bands = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']
    band_labels = ['Sub Bass\n(20-60Hz)', 'Bass\n(60-250Hz)', 'Low Mid\n(250-500Hz)',
                   'Mid\n(500-2kHz)', 'High Mid\n(2-4kHz)', 'High\n(4-10kHz)']
    x = np.arange(len(bands))
    width = 0.35

    orig_vals = [results['original']['bands'].get(b, 0) * 100 for b in bands]
    rend_vals = [results['rendered']['bands'].get(b, 0) * 100 for b in bands]

    # Use gradient colors for better visual appeal
    bars1 = ax.bar(x - width/2, orig_vals, width, label='Original', color='#58a6ff',
                   alpha=0.85, edgecolor='#79c0ff', linewidth=1)
    bars2 = ax.bar(x + width/2, rend_vals, width, label='Rendered', color='#3fb950',
                   alpha=0.85, edgecolor='#7ee787', linewidth=1)

    # Add value labels on top of bars
    for bar, val in zip(bars1, orig_vals):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#58a6ff')
    for bar, val in zip(bars2, rend_vals):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#3fb950')

    ax.set_ylabel('Energy Distribution (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=10)
    ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=11)
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    # Add subtle difference indicators
    for i, (o, r) in enumerate(zip(orig_vals, rend_vals)):
        diff = r - o
        if abs(diff) > 2:  # Only show significant differences
            color = '#3fb950' if diff > 0 else '#f85149'
            sign = '+' if diff > 0 else ''
            ax.annotate(f'{sign}{diff:.1f}%', xy=(x[i], max(o, r) + 2),
                       ha='center', fontsize=8, color=color, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()

def generate_similarity_chart(results, output_path):
    """Generate similarity scores chart with enhanced styling and overall score."""
    plt = setup_chart_style()
    fig, (ax_main, ax_overall) = plt.subplots(1, 2, figsize=(12, 5),
                                               gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')

    # Main metrics chart
    ax_main.set_facecolor('#161b22')
    metrics = ['mfcc_similarity', 'chroma_similarity', 'brightness_similarity',
               'tempo_similarity', 'frequency_balance_similarity', 'energy_similarity']
    metric_labels = ['Timbre (MFCC)', 'Harmony (Chroma)', 'Brightness',
                     'Tempo', 'Frequency Balance', 'Energy']
    values = [results['comparison'].get(m, 0) * 100 for m in metrics]

    # Gradient colors based on value
    def get_color(v):
        if v >= 80:
            return '#238636'  # Dark green
        elif v >= 70:
            return '#3fb950'  # Green
        elif v >= 60:
            return '#9e6a03'  # Dark yellow
        elif v >= 50:
            return '#d29922'  # Yellow
        else:
            return '#f85149'  # Red

    colors = [get_color(v) for v in values]

    # Add weight indicators
    weights = {'Timbre (MFCC)': 25, 'Harmony (Chroma)': 20, 'Brightness': 15,
               'Tempo': 15, 'Frequency Balance': 15, 'Energy': 10}

    bars = ax_main.barh(metric_labels, values, color=colors, alpha=0.9,
                        edgecolor=[c.replace('f8', 'ff').replace('3f', '5f') for c in colors],
                        linewidth=1.5)
    ax_main.set_xlim(0, 105)
    ax_main.set_xlabel('Similarity Score (%)', fontsize=12)
    ax_main.grid(True, alpha=0.15, axis='x', linestyle='--')
    ax_main.set_axisbelow(True)

    # Add value and weight labels
    for bar, val, label in zip(bars, values, metric_labels):
        # Value label
        ax_main.text(val + 1.5, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
                     va='center', fontsize=11, color='#c9d1d9', fontweight='bold')
        # Weight indicator (smaller, gray)
        weight = weights.get(label, 0)
        ax_main.text(2, bar.get_y() + bar.get_height()/2, f'({weight}%)',
                     va='center', fontsize=9, color='#6e7681', alpha=0.8)

    # Add threshold lines
    ax_main.axvline(x=70, color='#3fb950', linestyle=':', alpha=0.4, linewidth=1)
    ax_main.axvline(x=50, color='#d29922', linestyle=':', alpha=0.4, linewidth=1)

    # Overall score gauge
    ax_overall.set_facecolor('#161b22')
    overall = results['comparison'].get('overall_similarity', 0) * 100

    # Create a semi-circular gauge
    theta = np.linspace(np.pi, 0, 100)
    r = 1

    # Background arc (gray)
    ax_overall.plot(r * np.cos(theta), r * np.sin(theta), color='#30363d', linewidth=20, solid_capstyle='round')

    # Value arc
    filled_theta = np.linspace(np.pi, np.pi - (overall/100) * np.pi, int(overall))
    arc_color = get_color(overall)
    if len(filled_theta) > 1:
        ax_overall.plot(r * np.cos(filled_theta), r * np.sin(filled_theta),
                       color=arc_color, linewidth=18, solid_capstyle='round')

    # Center text
    ax_overall.text(0, 0.15, f'{overall:.1f}%', ha='center', va='center',
                    fontsize=28, fontweight='bold', color='#c9d1d9')
    ax_overall.text(0, -0.15, 'OVERALL', ha='center', va='center',
                    fontsize=10, color='#8b949e')

    ax_overall.set_xlim(-1.3, 1.3)
    ax_overall.set_ylim(-0.5, 1.2)
    ax_overall.set_aspect('equal')
    ax_overall.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()

def generate_spectrogram_chart(audio_path, output_path, title, duration=30):
    """Generate mel spectrogram chart for a single audio file with colorbar."""
    plt = setup_chart_style()
    y = load_audio(audio_path, duration=duration)
    if y is None:
        return False

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    try:
        # Use mel spectrogram for better perceptual relevance
        S = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Plot with improved colormap
        img = librosa.display.specshow(
            S_db, sr=22050, x_axis='time', y_axis='mel',
            ax=ax, cmap='inferno', vmin=-80, vmax=0
        )

        # Add colorbar with proper styling
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        cbar.outline.set_edgecolor('#30363d')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False

def generate_chromagram_chart(audio_path, output_path, title, duration=30):
    """Generate chromagram chart for a single audio file with colorbar."""
    plt = setup_chart_style()
    y = load_audio(audio_path, duration=duration)
    if y is None:
        return False

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    try:
        # Use CQT chroma with better parameters
        chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512, n_chroma=12)

        # Plot with improved colormap (BuPu works well for chroma)
        img = librosa.display.specshow(
            chroma, sr=22050, x_axis='time', y_axis='chroma',
            ax=ax, cmap='BuPu', vmin=0, vmax=1
        )

        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, format='%.1f', pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        cbar.outline.set_edgecolor('#30363d')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')
        cbar.set_label('Intensity', color='#8b949e')

        ax.set_ylabel('Pitch Class', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False

def generate_waveform_chart(original_path, rendered_path, output_path, duration=30):
    """Generate side-by-side waveform comparison chart."""
    plt = setup_chart_style()
    orig_y = load_audio(original_path, duration=duration)
    rend_y = load_audio(rendered_path, duration=duration)

    if orig_y is None or rend_y is None:
        return False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor('#0d1117')

    # Normalize lengths
    min_len = min(len(orig_y), len(rend_y))
    orig_y = orig_y[:min_len]
    rend_y = rend_y[:min_len]

    # Time axis
    time = np.linspace(0, min_len / 22050, min_len)

    # Original waveform
    ax1.set_facecolor('#161b22')
    ax1.fill_between(time, orig_y, alpha=0.7, color='#58a6ff', linewidth=0)
    ax1.plot(time, orig_y, color='#79c0ff', linewidth=0.3, alpha=0.5)
    ax1.set_ylabel('Original', fontsize=11)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.15, linestyle='--')
    ax1.axhline(y=0, color='#30363d', linewidth=0.5)

    # Rendered waveform
    ax2.set_facecolor('#161b22')
    ax2.fill_between(time, rend_y, alpha=0.7, color='#3fb950', linewidth=0)
    ax2.plot(time, rend_y, color='#7ee787', linewidth=0.3, alpha=0.5)
    ax2.set_ylabel('Rendered', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.axhline(y=0, color='#30363d', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()
    return True


def generate_onset_chart(original_path, rendered_path, output_path, duration=30):
    """Generate onset strength comparison chart."""
    plt = setup_chart_style()
    orig_y = load_audio(original_path, duration=duration)
    rend_y = load_audio(rendered_path, duration=duration)

    if orig_y is None or rend_y is None:
        return False

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Compute onset envelopes
    orig_onset = librosa.onset.onset_strength(y=orig_y, sr=22050)
    rend_onset = librosa.onset.onset_strength(y=rend_y, sr=22050)

    # Time axis (onset frames)
    times_orig = librosa.times_like(orig_onset, sr=22050)
    times_rend = librosa.times_like(rend_onset, sr=22050)

    # Normalize for comparison
    orig_onset = orig_onset / (np.max(orig_onset) + 1e-10)
    rend_onset = rend_onset / (np.max(rend_onset) + 1e-10)

    # Plot
    ax.fill_between(times_orig, orig_onset, alpha=0.4, color='#58a6ff', label='Original')
    ax.plot(times_orig, orig_onset, color='#79c0ff', linewidth=1, alpha=0.8)

    ax.fill_between(times_rend, rend_onset, alpha=0.4, color='#3fb950', label='Rendered')
    ax.plot(times_rend, rend_onset, color='#7ee787', linewidth=1, alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Onset Strength (normalized)', fontsize=11)
    ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()
    return True


def generate_mfcc_comparison_chart(original_path, rendered_path, output_path, duration=30):
    """Generate MFCC comparison heatmaps."""
    plt = setup_chart_style()
    orig_y = load_audio(original_path, duration=duration)
    rend_y = load_audio(rendered_path, duration=duration)

    if orig_y is None or rend_y is None:
        return False

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5),
                                         gridspec_kw={'width_ratios': [1, 1, 0.4]})
    fig.patch.set_facecolor('#0d1117')

    # Compute MFCCs
    orig_mfcc = librosa.feature.mfcc(y=orig_y, sr=22050, n_mfcc=13)
    rend_mfcc = librosa.feature.mfcc(y=rend_y, sr=22050, n_mfcc=13)

    # Original MFCC
    ax1.set_facecolor('#161b22')
    img1 = librosa.display.specshow(orig_mfcc, sr=22050, x_axis='time', ax=ax1, cmap='viridis')
    ax1.set_ylabel('MFCC Coefficient', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('Original', fontsize=12, color='#58a6ff')

    # Rendered MFCC
    ax2.set_facecolor('#161b22')
    img2 = librosa.display.specshow(rend_mfcc, sr=22050, x_axis='time', ax=ax2, cmap='viridis')
    ax2.set_ylabel('')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title('Rendered', fontsize=12, color='#3fb950')

    # Add colorbar
    cbar = fig.colorbar(img2, ax=[ax1, ax2], format='%+2.0f', pad=0.02, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    cbar.outline.set_edgecolor('#30363d')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

    # MFCC mean comparison (bar chart)
    ax3.set_facecolor('#161b22')
    orig_mean = np.mean(orig_mfcc, axis=1)
    rend_mean = np.mean(rend_mfcc, axis=1)

    y_pos = np.arange(13)
    width = 0.35

    ax3.barh(y_pos - width/2, orig_mean, width, label='Original', color='#58a6ff', alpha=0.8)
    ax3.barh(y_pos + width/2, rend_mean, width, label='Rendered', color='#3fb950', alpha=0.8)
    ax3.set_xlabel('Mean Value', fontsize=10)
    ax3.set_ylabel('MFCC #', fontsize=10)
    ax3.set_yticks(y_pos)
    ax3.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=9)
    ax3.grid(True, alpha=0.15, axis='x', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()
    return True


def generate_comparison_charts(results, original_path, rendered_path, output_dir):
    """Generate all comparison charts as separate files."""
    import os
    from pathlib import Path

    output_dir = Path(output_dir)
    charts = {}

    print("Generating frequency bands chart...")
    freq_path = output_dir / "chart_frequency.png"
    generate_frequency_bands_chart(results, str(freq_path))
    charts['frequency'] = str(freq_path)

    print("Generating similarity chart...")
    sim_path = output_dir / "chart_similarity.png"
    generate_similarity_chart(results, str(sim_path))
    charts['similarity'] = str(sim_path)

    print("Generating original spectrogram (mel)...")
    spec_orig_path = output_dir / "chart_spectrogram_original.png"
    if generate_spectrogram_chart(original_path, str(spec_orig_path), "Original - Mel Spectrogram"):
        charts['spectrogram_original'] = str(spec_orig_path)

    print("Generating rendered spectrogram (mel)...")
    spec_rend_path = output_dir / "chart_spectrogram_rendered.png"
    if generate_spectrogram_chart(rendered_path, str(spec_rend_path), "Rendered - Mel Spectrogram"):
        charts['spectrogram_rendered'] = str(spec_rend_path)

    print("Generating original chromagram...")
    chroma_orig_path = output_dir / "chart_chromagram_original.png"
    if generate_chromagram_chart(original_path, str(chroma_orig_path), "Original - Chromagram"):
        charts['chromagram_original'] = str(chroma_orig_path)

    print("Generating rendered chromagram...")
    chroma_rend_path = output_dir / "chart_chromagram_rendered.png"
    if generate_chromagram_chart(rendered_path, str(chroma_rend_path), "Rendered - Chromagram"):
        charts['chromagram_rendered'] = str(chroma_rend_path)

    print("Generating waveform comparison...")
    wave_path = output_dir / "chart_waveform.png"
    if generate_waveform_chart(original_path, rendered_path, str(wave_path)):
        charts['waveform'] = str(wave_path)

    print("Generating onset strength comparison...")
    onset_path = output_dir / "chart_onset.png"
    if generate_onset_chart(original_path, rendered_path, str(onset_path)):
        charts['onset'] = str(onset_path)

    print("Generating MFCC comparison...")
    mfcc_path = output_dir / "chart_mfcc.png"
    if generate_mfcc_comparison_chart(original_path, rendered_path, str(mfcc_path)):
        charts['mfcc'] = str(mfcc_path)

    print(f"Generated {len(charts)} charts in {output_dir}")
    return charts

def save_comparison_json(results, output_path):
    """Save comparison results as JSON for HTML report generation."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Comparison JSON saved: {output_path}")

def generate_comparison_chart(results, original_path, rendered_path, output_path):
    """Generate a comprehensive comparison chart as PNG (legacy single-image mode)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    # If output_path is a directory or ends with /, generate multiple charts
    output_dir = Path(output_path).parent
    if output_path.endswith('/') or Path(output_path).is_dir():
        output_dir = Path(output_path)
        save_comparison_json(results, str(output_dir / "comparison.json"))
        return generate_comparison_charts(results, original_path, rendered_path, str(output_dir))

    # Save comparison JSON for HTML report
    save_comparison_json(results, str(output_dir / "comparison.json"))

    # Generate individual charts in same directory
    generate_comparison_charts(results, original_path, rendered_path, str(output_dir))

    # Also generate combined chart for backwards compatibility
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0d1117')

    # Style setup
    plt.rcParams['text.color'] = '#c9d1d9'
    plt.rcParams['axes.labelcolor'] = '#c9d1d9'
    plt.rcParams['axes.edgecolor'] = '#30363d'
    plt.rcParams['xtick.color'] = '#8b949e'
    plt.rcParams['ytick.color'] = '#8b949e'

    # Load audio for visualizations
    orig_y = load_audio(original_path, duration=30)
    rend_y = load_audio(rendered_path, duration=30)

    # 1. Frequency band comparison
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    bands = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']
    band_labels = ['Sub\nBass', 'Bass', 'Low\nMid', 'Mid', 'High\nMid', 'High']
    x = np.arange(len(bands))
    width = 0.35

    orig_vals = [results['original']['bands'].get(b, 0) * 100 for b in bands]
    rend_vals = [results['rendered']['bands'].get(b, 0) * 100 for b in bands]

    ax1.bar(x - width/2, orig_vals, width, label='Original', color='#58a6ff', alpha=0.8)
    ax1.bar(x + width/2, rend_vals, width, label='Rendered', color='#3fb950', alpha=0.8)

    ax1.set_ylabel('Energy %', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, fontsize=9)
    ax1.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d')
    ax1.grid(True, alpha=0.2, axis='y')

    # 2. Similarity scores
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    metrics = ['mfcc_similarity', 'chroma_similarity', 'brightness_similarity',
               'tempo_similarity', 'frequency_balance_similarity', 'energy_similarity']
    metric_labels = ['Timbre', 'Harmony', 'Brightness', 'Tempo', 'Freq Balance', 'Energy']
    values = [results['comparison'].get(m, 0) * 100 for m in metrics]
    colors = ['#3fb950' if v >= 70 else '#d29922' if v >= 50 else '#f85149' for v in values]

    bars = ax2.barh(metric_labels, values, color=colors, alpha=0.8)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Similarity %', fontsize=10)
    ax2.grid(True, alpha=0.2, axis='x')

    for bar, val in zip(bars, values):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
                va='center', fontsize=9, color='#c9d1d9')

    # 3. Spectrogram - Original
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    try:
        if orig_y is not None:
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig_y)), ref=np.max)
            img = librosa.display.specshow(D_orig, sr=22050, x_axis='time', y_axis='hz', ax=ax3, cmap='magma')
            ax3.set_ylim(0, 8000)
    except Exception as e:
        ax3.text(0.5, 0.5, f'Spectrogram unavailable', ha='center', va='center',
                transform=ax3.transAxes, color='#8b949e')

    # 4. Spectrogram - Rendered
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    try:
        if rend_y is not None:
            D_rend = librosa.amplitude_to_db(np.abs(librosa.stft(rend_y)), ref=np.max)
            librosa.display.specshow(D_rend, sr=22050, x_axis='time', y_axis='hz', ax=ax4, cmap='magma')
            ax4.set_ylim(0, 8000)
    except Exception as e:
        ax4.text(0.5, 0.5, f'Spectrogram unavailable', ha='center', va='center',
                transform=ax4.transAxes, color='#8b949e')

    # 5. Chromagram - Original
    ax5 = axes[2, 0]
    ax5.set_facecolor('#161b22')
    try:
        if orig_y is not None:
            chroma_orig = librosa.feature.chroma_cqt(y=orig_y, sr=22050)
            librosa.display.specshow(chroma_orig, sr=22050, x_axis='time', y_axis='chroma', ax=ax5, cmap='coolwarm')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Chromagram unavailable', ha='center', va='center',
                transform=ax5.transAxes, color='#8b949e')

    # 6. Chromagram - Rendered
    ax6 = axes[2, 1]
    ax6.set_facecolor('#161b22')
    try:
        if rend_y is not None:
            chroma_rend = librosa.feature.chroma_cqt(y=rend_y, sr=22050)
            librosa.display.specshow(chroma_rend, sr=22050, x_axis='time', y_axis='chroma', ax=ax6, cmap='coolwarm')
    except Exception as e:
        ax6.text(0.5, 0.5, f'Chromagram unavailable', ha='center', va='center',
                transform=ax6.transAxes, color='#8b949e')

    # Overall score is shown in the report, not embedded in image
    overall = results['comparison'].get('overall_similarity', 0) * 100

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {output_path}")

def main():
    global _QUIET_MODE

    parser = argparse.ArgumentParser(description='Compare rendered audio with original')
    parser.add_argument('original', help='Path to original audio file')
    parser.add_argument('rendered', help='Path to rendered audio file')
    parser.add_argument('-d', '--duration', type=float, default=60,
                       help='Max duration to analyze in seconds')
    parser.add_argument('-j', '--json', action='store_true',
                       help='Output as JSON')
    parser.add_argument('-c', '--chart', help='Output path for comparison chart PNG')
    parser.add_argument('--config', help='Path to synth_config.json for AI-derived tolerance')

    args = parser.parse_args()

    # Enable quiet mode for JSON output (status to stderr, JSON to stdout)
    if args.json:
        _QUIET_MODE = True

    results = compare_audio(args.original, args.rendered, args.duration, args.config)

    if results is None:
        sys.exit(1)

    # Generate chart if requested
    if args.chart:
        generate_comparison_chart(results, args.original, args.rendered, args.chart)

    if args.json:
        print(json.dumps(results, indent=2))
    elif not args.chart:
        print_report(results)

if __name__ == '__main__':
    main()
