#!/usr/bin/env python3
"""
Iterative audio rendering with AI-driven parameter refinement.
Renders, compares, adjusts, repeats until quality target is met.
"""

import argparse
import json
import os
import sys
import subprocess
import numpy as np
import librosa
from pathlib import Path

# Import our analysis functions
from compare_audio import compute_frequency_bands

MAX_ITERATIONS = 20
TARGET_SIMILARITY = 0.70  # 70% overall similarity target
MIN_IMPROVEMENT = 0.005   # Stop if improvement < 0.5%


def compute_similarity(original_path, rendered_path, duration=60):
    """Compute similarity between original and rendered audio."""
    # Load both audio files
    y_orig, sr = librosa.load(original_path, sr=22050, mono=True, duration=duration)
    y_rend, sr = librosa.load(rendered_path, sr=22050, mono=True, duration=duration)

    # Compute frequency bands for both
    orig_bands = compute_frequency_bands(y_orig, sr)
    rend_bands = compute_frequency_bands(y_rend, sr)

    # Compute band-by-band differences
    band_diffs = {}
    for band in orig_bands:
        diff = rend_bands[band] - orig_bands[band]
        band_diffs[band] = diff

    # Compute MFCC similarity
    mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13)
    mfcc_rend = librosa.feature.mfcc(y=y_rend, sr=sr, n_mfcc=13)
    mfcc_sim = 1 - np.mean(np.abs(mfcc_orig.mean(axis=1) - mfcc_rend.mean(axis=1))) / 100

    # Compute chroma similarity
    chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=sr)
    chroma_rend = librosa.feature.chroma_cqt(y=y_rend, sr=sr)
    chroma_sim = np.corrcoef(chroma_orig.flatten(), chroma_rend.flatten())[0, 1]
    chroma_sim = max(0, chroma_sim)  # Clamp negative correlations

    # Frequency balance similarity
    freq_sim = 1 - sum(abs(d) for d in band_diffs.values()) / 2

    # Overall similarity (weighted)
    # Focus on frequency balance (what rendering controls), less on chroma (transcription quality)
    overall = (mfcc_sim * 0.25 + chroma_sim * 0.15 + freq_sim * 0.6)

    return {
        'overall': overall,
        'mfcc': mfcc_sim,
        'chroma': chroma_sim,
        'frequency': freq_sim,
        'band_diffs': band_diffs,
        'orig_bands': orig_bands,
        'rend_bands': rend_bands,
    }


def adjust_mix(current_mix, band_diffs, learning_rate=0.7):
    """Adjust mix parameters based on frequency band differences."""
    new_mix = current_mix.copy()

    # Band to mix parameter mapping
    # If a band has too much energy (positive diff), reduce the corresponding gain
    # If a band has too little energy (negative diff), increase the gain

    # Sub-bass and bass → kick_gain, bass_gain
    bass_diff = band_diffs.get('sub_bass', 0) + band_diffs.get('bass', 0)
    if abs(bass_diff) > 0.01:
        adjustment = -bass_diff * learning_rate
        new_mix['kick_gain'] = max(0.01, min(1.0, current_mix.get('kick_gain', 0.3) + adjustment * 0.5))
        new_mix['bass_gain'] = max(0.01, min(1.0, current_mix.get('bass_gain', 0.2) + adjustment * 0.5))

    # Low-mid → vox_gain, stab_gain
    low_mid_diff = band_diffs.get('low_mid', 0)
    if abs(low_mid_diff) > 0.01:
        adjustment = -low_mid_diff * learning_rate
        new_mix['vox_gain'] = max(0.01, min(1.0, current_mix.get('vox_gain', 0.4) + adjustment))
        new_mix['stab_gain'] = max(0.01, min(1.0, current_mix.get('stab_gain', 0.3) + adjustment * 0.5))

    # Mid → lead_gain (this is usually the dominant range)
    mid_diff = band_diffs.get('mid', 0)
    if abs(mid_diff) > 0.01:
        adjustment = -mid_diff * learning_rate
        new_mix['lead_gain'] = max(0.1, min(2.0, current_mix.get('lead_gain', 0.8) + adjustment))

    # High-mid and high → hh_gain
    high_diff = band_diffs.get('high_mid', 0) + band_diffs.get('high', 0)
    if abs(high_diff) > 0.01:
        adjustment = -high_diff * learning_rate
        new_mix['hh_gain'] = max(0.01, min(1.0, current_mix.get('hh_gain', 0.3) + adjustment))

    # Snare is in low-mid/mid range
    snare_diff = low_mid_diff * 0.5 + mid_diff * 0.3
    if abs(snare_diff) > 0.01:
        adjustment = -snare_diff * learning_rate * 0.5
        new_mix['snare_gain'] = max(0.01, min(1.0, current_mix.get('snare_gain', 0.3) + adjustment))

    return new_mix


def render_with_params(strudel_path, output_path, duration, mix_params, ai_params_path=None):
    """Render audio with given mix parameters."""
    # Write temporary params file
    temp_params = {
        'renderer_mix': mix_params
    }

    # If we have existing AI params, merge them
    if ai_params_path and os.path.exists(ai_params_path):
        with open(ai_params_path) as f:
            existing = json.load(f)
            temp_params.update(existing)
            temp_params['renderer_mix'] = mix_params

    temp_params_path = output_path + '.params.json'
    with open(temp_params_path, 'w') as f:
        json.dump(temp_params, f)

    # Run renderer
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'render_audio.py'),
        strudel_path,
        '-o', output_path,
        '-d', str(duration),
        '--feedback', temp_params_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Render error: {result.stderr}")
        return False

    # Cleanup temp file
    os.remove(temp_params_path)
    return True


def iterative_render(original_path, strudel_path, output_path, ai_params_path=None,
                     duration=60, max_iterations=MAX_ITERATIONS, target_similarity=TARGET_SIMILARITY):
    """
    Iteratively render and refine until quality target is met.

    Returns:
        dict with final similarity scores and mix parameters
    """
    print(f"\n{'='*60}")
    print("ITERATIVE AUDIO RENDERING")
    print(f"{'='*60}")
    print(f"Original: {original_path}")
    print(f"Target similarity: {target_similarity*100:.0f}%")
    print(f"Max iterations: {max_iterations}")

    # Load initial mix from AI params
    current_mix = {
        'kick_gain': 0.3,
        'snare_gain': 0.3,
        'hh_gain': 0.3,
        'bass_gain': 0.2,
        'vox_gain': 0.4,
        'stab_gain': 0.3,
        'lead_gain': 0.8,
    }

    if ai_params_path and os.path.exists(ai_params_path):
        with open(ai_params_path) as f:
            ai_params = json.load(f)
            if 'renderer_mix' in ai_params:
                current_mix.update(ai_params['renderer_mix'])
                print(f"\nLoaded initial mix from AI analysis")

    best_similarity = 0
    best_mix = current_mix.copy()
    history = []

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Mix: kick={current_mix['kick_gain']:.3f}, bass={current_mix['bass_gain']:.3f}, "
              f"lead={current_mix['lead_gain']:.3f}, hh={current_mix['hh_gain']:.3f}")

        # Render with current parameters
        temp_output = output_path + f'.iter{iteration}.wav'
        if not render_with_params(strudel_path, temp_output, duration, current_mix, ai_params_path):
            print("Render failed, stopping")
            break

        # Compare to original
        similarity = compute_similarity(original_path, temp_output, duration)
        overall = similarity['overall']
        history.append({'iteration': iteration, 'similarity': overall, 'mix': current_mix.copy()})

        print(f"Similarity: {overall*100:.1f}% (mfcc={similarity['mfcc']*100:.0f}%, "
              f"chroma={similarity['chroma']*100:.0f}%, freq={similarity['frequency']*100:.0f}%)")

        # Check if we've reached target
        if overall >= target_similarity:
            print(f"\n✓ Target similarity reached!")
            best_mix = current_mix.copy()
            best_similarity = overall
            # Keep this as final output
            os.rename(temp_output, output_path)
            break

        # Track best result
        if overall > best_similarity:
            best_similarity = overall
            best_mix = current_mix.copy()
            # Save as best output so far
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
        else:
            os.remove(temp_output)

        # Check for convergence
        if iteration > 0:
            improvement = overall - history[-2]['similarity']
            if improvement < MIN_IMPROVEMENT and improvement >= 0:
                print(f"\nConverged (improvement {improvement*100:.1f}% < {MIN_IMPROVEMENT*100:.0f}%)")
                break

        # Adjust mix for next iteration
        print(f"Band diffs: " + ", ".join(f"{k}={v*100:+.1f}%" for k, v in similarity['band_diffs'].items()))
        current_mix = adjust_mix(current_mix, similarity['band_diffs'])

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Best similarity: {best_similarity*100:.1f}%")
    print(f"Iterations: {len(history)}")
    print(f"Final mix: {json.dumps(best_mix, indent=2)}")

    return {
        'similarity': best_similarity,
        'mix': best_mix,
        'history': history,
        'iterations': len(history),
    }


def main():
    parser = argparse.ArgumentParser(description='Iterative audio rendering with AI refinement')
    parser.add_argument('original', help='Original audio file to match')
    parser.add_argument('strudel', help='Strudel code file')
    parser.add_argument('-o', '--output', default='output.wav', help='Output WAV file')
    parser.add_argument('-p', '--params', help='AI params JSON file')
    parser.add_argument('-d', '--duration', type=float, default=60, help='Duration to analyze (seconds)')
    parser.add_argument('-i', '--iterations', type=int, default=MAX_ITERATIONS, help='Max iterations')
    parser.add_argument('-t', '--target', type=float, default=TARGET_SIMILARITY, help='Target similarity (0-1)')

    args = parser.parse_args()

    result = iterative_render(
        args.original,
        args.strudel,
        args.output,
        ai_params_path=args.params,
        duration=args.duration,
        max_iterations=args.iterations,
        target_similarity=args.target
    )

    # Save result metadata
    result_path = args.output + '.result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {result_path}")


if __name__ == '__main__':
    main()
