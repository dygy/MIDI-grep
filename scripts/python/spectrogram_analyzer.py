#!/usr/bin/env python3
"""
Mel Spectrogram Analyzer - Deep analysis for AI learning.

Compares original vs rendered spectrograms to identify:
1. Which frequency bands differ at which times
2. Envelope/amplitude differences over time
3. Harmonic content differences
4. Transient/attack differences

These insights are fed to the AI to generate better code.
"""

import numpy as np
import librosa
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path


def compute_mel_spectrogram(audio_path: str, sr: int = 22050,
                            n_mels: int = 128, n_fft: int = 2048,
                            hop_length: int = 512, duration: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mel spectrogram from audio file.

    Returns:
        (mel_spec, times) - mel spectrogram in dB and time axis
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Get time axis
    times = librosa.times_like(mel_spec_db, sr=sr, hop_length=hop_length)

    return mel_spec_db, times


def get_frequency_bands() -> Dict[str, Tuple[int, int]]:
    """Define frequency bands for analysis (mel bin ranges for 128 mels)."""
    return {
        'sub_bass': (0, 8),      # ~20-60 Hz
        'bass': (8, 20),         # ~60-250 Hz
        'low_mid': (20, 40),     # ~250-500 Hz
        'mid': (40, 64),         # ~500-2000 Hz
        'high_mid': (64, 90),    # ~2000-6000 Hz
        'high': (90, 110),       # ~6000-12000 Hz
        'air': (110, 128),       # ~12000-20000 Hz
    }


def analyze_band_over_time(orig_mel: np.ndarray, rend_mel: np.ndarray,
                           band_start: int, band_end: int) -> Dict[str, Any]:
    """
    Analyze a specific frequency band over time.

    Returns insights about:
    - Average energy difference
    - Time-varying differences (attack, sustain, decay)
    - Peaks and troughs
    """
    # Extract band
    orig_band = orig_mel[band_start:band_end, :]
    rend_band = rend_mel[band_start:band_end, :]

    # Average energy over frequency bins
    orig_energy = np.mean(orig_band, axis=0)
    rend_energy = np.mean(rend_band, axis=0)

    # Normalize to 0-1 range
    orig_norm = (orig_energy - orig_energy.min()) / (orig_energy.max() - orig_energy.min() + 1e-10)
    rend_norm = (rend_energy - rend_energy.min()) / (rend_energy.max() - rend_energy.min() + 1e-10)

    # Calculate differences
    energy_diff = np.mean(orig_energy) - np.mean(rend_energy)  # Positive = rendered is quieter

    # Analyze attack (first 10% of audio)
    n_frames = len(orig_energy)
    attack_frames = int(n_frames * 0.1)
    attack_diff = np.mean(orig_energy[:attack_frames]) - np.mean(rend_energy[:attack_frames])

    # Analyze sustain (middle 60%)
    sustain_start = int(n_frames * 0.2)
    sustain_end = int(n_frames * 0.8)
    sustain_diff = np.mean(orig_energy[sustain_start:sustain_end]) - np.mean(rend_energy[sustain_start:sustain_end])

    # Analyze decay (last 20%)
    decay_frames = int(n_frames * 0.2)
    decay_diff = np.mean(orig_energy[-decay_frames:]) - np.mean(rend_energy[-decay_frames:])

    # Correlation (how similar is the shape)
    if np.std(orig_norm) > 0.01 and np.std(rend_norm) > 0.01:
        correlation = float(np.corrcoef(orig_norm, rend_norm)[0, 1])
    else:
        correlation = 1.0 if np.allclose(orig_norm, rend_norm) else 0.0

    # Peak analysis - find if rendered has peaks where original doesn't
    orig_peaks = orig_energy > (np.mean(orig_energy) + np.std(orig_energy))
    rend_peaks = rend_energy > (np.mean(rend_energy) + np.std(rend_energy))
    peak_alignment = float(np.mean(orig_peaks == rend_peaks))

    return {
        'energy_diff_db': float(energy_diff),
        'attack_diff_db': float(attack_diff),
        'sustain_diff_db': float(sustain_diff),
        'decay_diff_db': float(decay_diff),
        'shape_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'peak_alignment': float(peak_alignment),
    }


def analyze_transients(orig_mel: np.ndarray, rend_mel: np.ndarray,
                       times: np.ndarray) -> Dict[str, Any]:
    """
    Analyze transient/attack characteristics.

    Transients are rapid changes in energy - important for drums and percussive sounds.
    """
    # Sum across all frequencies
    orig_energy = np.sum(orig_mel, axis=0)
    rend_energy = np.sum(rend_mel, axis=0)

    # Compute onset strength (derivative of energy)
    orig_onset = np.diff(orig_energy)
    rend_onset = np.diff(rend_energy)

    # Find strong transients (large positive changes)
    orig_transients = orig_onset > np.percentile(orig_onset, 90)
    rend_transients = rend_onset > np.percentile(rend_onset, 90)

    # Count transients
    orig_count = np.sum(orig_transients)
    rend_count = np.sum(rend_transients)

    # Transient strength (average height of detected transients)
    orig_strength = float(np.mean(orig_onset[orig_transients])) if orig_count > 0 else 0
    rend_strength = float(np.mean(rend_onset[rend_transients])) if rend_count > 0 else 0

    # Transient alignment (do they occur at the same times?)
    if len(orig_transients) == len(rend_transients):
        alignment = float(np.mean(orig_transients == rend_transients))
    else:
        alignment = 0.5  # Can't compare

    return {
        'orig_transient_count': int(orig_count),
        'rend_transient_count': int(rend_count),
        'transient_count_ratio': float(rend_count / max(orig_count, 1)),
        'orig_transient_strength': float(orig_strength),
        'rend_transient_strength': float(rend_strength),
        'transient_alignment': float(alignment),
    }


def analyze_harmonic_content(orig_mel: np.ndarray, rend_mel: np.ndarray) -> Dict[str, Any]:
    """
    Analyze harmonic content by looking at relationships between frequency bands.

    Harmonics appear as parallel lines in spectrogram at octave intervals.
    """
    n_mels = orig_mel.shape[0]

    # Compare ratios between frequency bands (fundamental vs harmonics)
    # For a 128-mel spectrogram, check bins at roughly octave intervals
    octave_pairs = [
        (16, 32),   # ~125Hz vs ~250Hz
        (32, 48),   # ~250Hz vs ~500Hz
        (48, 64),   # ~500Hz vs ~1000Hz
        (64, 80),   # ~1000Hz vs ~2000Hz
    ]

    harmonic_diffs = []
    for low, high in octave_pairs:
        if high < n_mels:
            orig_ratio = np.mean(orig_mel[high, :]) / (np.mean(orig_mel[low, :]) + 1e-10)
            rend_ratio = np.mean(rend_mel[high, :]) / (np.mean(rend_mel[low, :]) + 1e-10)
            harmonic_diffs.append(abs(orig_ratio - rend_ratio))

    avg_harmonic_diff = float(np.mean(harmonic_diffs)) if harmonic_diffs else 0.0

    # Check odd vs even harmonics (important for waveform type)
    # Square waves have strong odd harmonics, saw waves have both
    odd_bins = [24, 40, 56, 72]   # ~3rd, 5th, 7th, 9th harmonics (approximate)
    even_bins = [32, 48, 64, 80]  # ~2nd, 4th, 6th, 8th harmonics (approximate)

    orig_odd = np.mean([np.mean(orig_mel[b, :]) for b in odd_bins if b < n_mels])
    orig_even = np.mean([np.mean(orig_mel[b, :]) for b in even_bins if b < n_mels])
    rend_odd = np.mean([np.mean(rend_mel[b, :]) for b in odd_bins if b < n_mels])
    rend_even = np.mean([np.mean(rend_mel[b, :]) for b in even_bins if b < n_mels])

    orig_odd_even_ratio = orig_odd / (orig_even + 1e-10)
    rend_odd_even_ratio = rend_odd / (rend_even + 1e-10)

    # Determine waveform suggestion
    if rend_odd_even_ratio > orig_odd_even_ratio * 1.3:
        waveform_suggestion = 'Use sawtooth instead of square (need more even harmonics)'
    elif rend_odd_even_ratio < orig_odd_even_ratio * 0.7:
        waveform_suggestion = 'Use square instead of sawtooth (need more odd harmonics)'
    else:
        waveform_suggestion = None

    return {
        'harmonic_structure_diff': avg_harmonic_diff,
        'orig_odd_even_ratio': float(orig_odd_even_ratio),
        'rend_odd_even_ratio': float(rend_odd_even_ratio),
        'waveform_suggestion': waveform_suggestion,
    }


def generate_ai_insights(analysis: Dict[str, Any]) -> List[str]:
    """
    Convert spectrogram analysis into actionable insights for the AI.

    These insights are specific and actionable, not generic.
    """
    insights = []

    # Band-specific insights
    for band_name, band_data in analysis.get('bands', {}).items():
        energy_diff = band_data.get('energy_diff_db', 0)
        attack_diff = band_data.get('attack_diff_db', 0)
        shape_corr = band_data.get('shape_correlation', 1.0)

        if abs(energy_diff) > 3:  # More than 3dB difference
            if energy_diff > 0:
                insights.append(f"INCREASE_{band_name.upper()}_GAIN: Rendered {band_name} is {abs(energy_diff):.1f}dB quieter than original. Multiply {band_name} voice gain by {10**(energy_diff/20):.2f}x")
            else:
                insights.append(f"DECREASE_{band_name.upper()}_GAIN: Rendered {band_name} is {abs(energy_diff):.1f}dB louder than original. Multiply {band_name} voice gain by {10**(energy_diff/20):.2f}x")

        if abs(attack_diff) > 5:  # Attack differs significantly
            if attack_diff > 0:
                insights.append(f"FASTER_{band_name.upper()}_ATTACK: {band_name} attack is too slow. Reduce attack time by {min(50, abs(attack_diff)*5):.0f}%")
            else:
                insights.append(f"SLOWER_{band_name.upper()}_ATTACK: {band_name} attack is too fast. Increase attack time by {min(50, abs(attack_diff)*5):.0f}%")

        if shape_corr < 0.7:
            insights.append(f"RESHAPE_{band_name.upper()}_ENVELOPE: {band_name} envelope shape differs significantly (correlation: {shape_corr:.2f}). Consider adjusting ADSR or adding modulation")

    # Transient insights
    transients = analysis.get('transients', {})
    trans_ratio = transients.get('transient_count_ratio', 1.0)
    trans_align = transients.get('transient_alignment', 1.0)

    if trans_ratio < 0.7:
        insights.append(f"ADD_TRANSIENTS: Rendered has {(1-trans_ratio)*100:.0f}% fewer transients. Add click/attack transients or reduce attack time")
    elif trans_ratio > 1.3:
        insights.append(f"REDUCE_TRANSIENTS: Rendered has {(trans_ratio-1)*100:.0f}% more transients. Increase attack time or add compression")

    if trans_align < 0.6:
        insights.append(f"FIX_TRANSIENT_TIMING: Transients are misaligned (only {trans_align*100:.0f}% match). Check tempo/quantization")

    # Harmonic insights
    harmonics = analysis.get('harmonics', {})
    waveform_sug = harmonics.get('waveform_suggestion')
    if waveform_sug:
        insights.append(f"CHANGE_WAVEFORM: {waveform_sug}")

    harm_diff = harmonics.get('harmonic_structure_diff', 0)
    if harm_diff > 0.5:
        insights.append(f"ADJUST_HARMONICS: Harmonic structure differs significantly. Consider using FM synthesis or additive harmonics")

    return insights


def analyze_spectrograms(original_audio: str, rendered_audio: str,
                         duration: float = 60.0) -> Dict[str, Any]:
    """
    Main analysis function - compare two audio files using mel spectrograms.

    Returns detailed analysis with actionable insights for AI learning.
    """
    print("Computing mel spectrograms...", file=__import__('sys').stderr)

    # Compute spectrograms
    orig_mel, times = compute_mel_spectrogram(original_audio, duration=duration)
    rend_mel, _ = compute_mel_spectrogram(rendered_audio, duration=duration)

    # Ensure same length
    min_len = min(orig_mel.shape[1], rend_mel.shape[1])
    orig_mel = orig_mel[:, :min_len]
    rend_mel = rend_mel[:, :min_len]
    times = times[:min_len]

    print("Analyzing frequency bands...", file=__import__('sys').stderr)

    # Analyze each frequency band
    bands = get_frequency_bands()
    band_analysis = {}
    for band_name, (start, end) in bands.items():
        band_analysis[band_name] = analyze_band_over_time(orig_mel, rend_mel, start, end)

    print("Analyzing transients...", file=__import__('sys').stderr)

    # Analyze transients
    transient_analysis = analyze_transients(orig_mel, rend_mel, times)

    print("Analyzing harmonic content...", file=__import__('sys').stderr)

    # Analyze harmonics
    harmonic_analysis = analyze_harmonic_content(orig_mel, rend_mel)

    # Compile full analysis
    analysis = {
        'bands': band_analysis,
        'transients': transient_analysis,
        'harmonics': harmonic_analysis,
    }

    # Generate AI insights
    print("Generating AI insights...", file=__import__('sys').stderr)
    insights = generate_ai_insights(analysis)
    analysis['ai_insights'] = insights

    # Overall similarity (based on spectrogram MSE)
    mse = np.mean((orig_mel - rend_mel) ** 2)
    max_mse = np.mean(orig_mel ** 2)  # Max possible error
    similarity = 1 - (mse / max_mse) if max_mse > 0 else 0
    analysis['spectrogram_similarity'] = float(np.clip(similarity, 0, 1))

    return analysis


def format_for_llm(analysis: Dict[str, Any]) -> str:
    """
    Format analysis as a prompt for the LLM.
    """
    lines = [
        "## Mel Spectrogram Analysis Results",
        "",
        f"Overall spectrogram similarity: {analysis.get('spectrogram_similarity', 0)*100:.1f}%",
        "",
        "### Frequency Band Analysis:",
    ]

    for band_name, data in analysis.get('bands', {}).items():
        energy_diff = data.get('energy_diff_db', 0)
        direction = "quieter" if energy_diff > 0 else "louder"
        lines.append(f"- **{band_name}**: {abs(energy_diff):.1f}dB {direction}, shape correlation: {data.get('shape_correlation', 0):.2f}")

    lines.extend([
        "",
        "### Transient Analysis:",
        f"- Original transients: {analysis.get('transients', {}).get('orig_transient_count', 0)}",
        f"- Rendered transients: {analysis.get('transients', {}).get('rend_transient_count', 0)}",
        f"- Alignment: {analysis.get('transients', {}).get('transient_alignment', 0)*100:.0f}%",
        "",
        "### Harmonic Analysis:",
        f"- Original odd/even ratio: {analysis.get('harmonics', {}).get('orig_odd_even_ratio', 0):.2f}",
        f"- Rendered odd/even ratio: {analysis.get('harmonics', {}).get('rend_odd_even_ratio', 0):.2f}",
    ])

    waveform_sug = analysis.get('harmonics', {}).get('waveform_suggestion')
    if waveform_sug:
        lines.append(f"- Waveform suggestion: {waveform_sug}")

    lines.extend([
        "",
        "### Actionable Insights (use these to modify the code):",
    ])

    for insight in analysis.get('ai_insights', []):
        lines.append(f"- {insight}")

    if not analysis.get('ai_insights'):
        lines.append("- No significant issues detected")

    return "\n".join(lines)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mel Spectrogram Analyzer')
    parser.add_argument('original', help='Path to original audio')
    parser.add_argument('rendered', help='Path to rendered audio')
    parser.add_argument('-o', '--output', help='Output JSON path')
    parser.add_argument('-d', '--duration', type=float, default=60.0, help='Duration to analyze')
    parser.add_argument('--llm', action='store_true', help='Output formatted for LLM')

    args = parser.parse_args()

    analysis = analyze_spectrograms(args.original, args.rendered, args.duration)

    if args.llm:
        print(format_for_llm(analysis))
    elif args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {args.output}")
    else:
        print(json.dumps(analysis, indent=2))
