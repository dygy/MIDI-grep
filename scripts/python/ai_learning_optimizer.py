#!/usr/bin/env python3
"""
AI Learning Optimizer for Energy and Timbre.

Uses gradient-based learning to improve synthesis parameters based on comparison feedback.
Stores successful learnings for reuse across tracks.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Learning rate for parameter adjustments
LEARNING_RATES = {
    "energy": {
        "master_gain": 0.3,      # Aggressive for loudness
        "voice_gains": 0.2,      # Per-voice gain adjustments
        "compression": 0.1,      # Compression ratio
    },
    "timbre": {
        "fm_index": 0.2,         # FM modulation depth
        "detune": 0.15,          # Detuning amount
        "filter_cutoff": 0.1,    # Filter adjustments
        "harmonic_mix": 0.15,    # Harmonic content blend
    }
}

# Target improvements per iteration
TARGET_IMPROVEMENT = 0.02  # 2% per iteration


def load_comparison(comparison_path: str) -> Dict[str, Any]:
    """Load comparison results."""
    with open(comparison_path, 'r') as f:
        return json.load(f)


def load_synth_config(config_path: str) -> Dict[str, Any]:
    """Load synthesis config."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_synth_config(config: Dict[str, Any], config_path: str):
    """Save synthesis config."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def compute_energy_gradient(comparison: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute gradient for energy improvement.

    Returns adjustments needed based on RMS and energy differences.
    """
    orig = comparison.get('original', {})
    rend = comparison.get('rendered', {})
    comp = comparison.get('comparison', {})

    orig_rms = orig.get('spectral', {}).get('rms_mean', 0.3)
    rend_rms = rend.get('spectral', {}).get('rms_mean', 0.1)

    # RMS ratio tells us how much louder we need to be
    rms_ratio = orig_rms / max(rend_rms, 0.01)

    # Energy similarity (0-1)
    energy_sim = comp.get('energy_similarity', 0.5)
    energy_gap = 1.0 - energy_sim

    # Compute gradients
    gradients = {}

    if rms_ratio > 1.1:  # Rendered is too quiet
        # Need to increase gain
        gradients['master_gain_mult'] = min(2.0, rms_ratio ** 0.5)  # Square root for stability
        gradients['increase_gains'] = True
    elif rms_ratio < 0.9:  # Rendered is too loud
        gradients['master_gain_mult'] = max(0.5, rms_ratio ** 0.5)
        gradients['increase_gains'] = False
    else:
        gradients['master_gain_mult'] = 1.0
        gradients['increase_gains'] = None

    # Per-band energy adjustments
    orig_bands = orig.get('bands', {})
    rend_bands = rend.get('bands', {})

    band_adjustments = {}
    for band in ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']:
        orig_e = orig_bands.get(band, 0.1)
        rend_e = rend_bands.get(band, 0.1)
        if orig_e > 0.01:
            ratio = rend_e / orig_e
            if ratio < 0.8:  # Need more energy in this band
                band_adjustments[band] = 'increase'
            elif ratio > 1.2:  # Need less energy
                band_adjustments[band] = 'decrease'

    gradients['band_adjustments'] = band_adjustments
    gradients['energy_gap'] = energy_gap

    return gradients


def compute_timbre_gradient(comparison: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute gradient for timbre (MFCC) improvement.

    MFCC captures spectral envelope - we adjust synthesis parameters
    to better match the original's spectral characteristics.
    """
    orig = comparison.get('original', {})
    rend = comparison.get('rendered', {})
    comp = comparison.get('comparison', {})

    mfcc_sim = comp.get('mfcc_similarity', 0.4)
    mfcc_gap = 1.0 - mfcc_sim

    gradients = {
        'mfcc_gap': mfcc_gap,
    }

    # Spectral characteristics
    orig_spec = orig.get('spectral', {})
    rend_spec = rend.get('spectral', {})

    orig_centroid = orig_spec.get('centroid_mean', 2000)
    rend_centroid = rend_spec.get('centroid_mean', 1500)

    orig_flatness = orig_spec.get('flatness_mean', 0.02)
    rend_flatness = rend_spec.get('flatness_mean', 0.02)

    # Centroid difference suggests brightness adjustment
    centroid_ratio = orig_centroid / max(rend_centroid, 100)
    if centroid_ratio > 1.2:
        gradients['brightness'] = 'increase'
        gradients['brightness_amount'] = min(0.5, (centroid_ratio - 1) * 0.3)
    elif centroid_ratio < 0.8:
        gradients['brightness'] = 'decrease'
        gradients['brightness_amount'] = min(0.5, (1 - centroid_ratio) * 0.3)

    # Flatness difference suggests harmonic content adjustment
    # Higher flatness = more noise-like, lower = more tonal
    flatness_ratio = orig_flatness / max(rend_flatness, 0.001)
    if flatness_ratio > 1.3:
        # Original has more noise-like content
        gradients['add_noise'] = True
        gradients['noise_amount'] = min(0.2, (flatness_ratio - 1) * 0.1)
    elif flatness_ratio < 0.7:
        # Original is more tonal
        gradients['add_noise'] = False
        gradients['reduce_harmonics'] = True

    # FM synthesis adjustment based on MFCC gap
    # If MFCC is low, try adjusting FM parameters
    if mfcc_gap > 0.5:
        gradients['adjust_fm'] = True
        gradients['fm_index_delta'] = 0.5  # Try more modulation
    elif mfcc_gap > 0.3:
        gradients['adjust_fm'] = True
        gradients['fm_index_delta'] = 0.2

    return gradients


def apply_energy_learning(config: Dict[str, Any], gradients: Dict[str, float],
                          iteration: int, comparison: Dict[str, Any] = None) -> Dict[str, Any]:
    """Apply energy-based learning to config."""
    synth = config.get('synth_config', {})

    # Learning rate decay
    lr_decay = 1.0 / (1 + iteration * 0.1)

    # Set target RMS from original audio (most important for energy matching)
    if comparison:
        orig_rms = comparison.get('original', {}).get('spectral', {}).get('rms_mean', 0.3)
        dynamics = synth.get('dynamics', {})
        dynamics['target_rms'] = orig_rms
        synth['dynamics'] = dynamics
        print(f"  Energy: target_rms set to {orig_rms:.4f}")

    # Apply master gain adjustment
    if 'master_gain_mult' in gradients:
        master = synth.get('master', {})
        current_gain = master.get('gain', 1.0)
        mult = gradients['master_gain_mult']

        # Damped adjustment
        target_gain = current_gain * mult
        new_gain = current_gain + (target_gain - current_gain) * LEARNING_RATES['energy']['master_gain'] * lr_decay
        new_gain = max(0.5, min(3.0, new_gain))  # Clamp

        if 'master' not in synth:
            synth['master'] = {}
        synth['master']['gain'] = new_gain
        print(f"  Energy: master_gain {current_gain:.2f} → {new_gain:.2f}")

    # Apply per-band adjustments via voice gains
    band_adj = gradients.get('band_adjustments', {})
    voices = synth.get('voices', {})

    voice_band_map = {
        'bass': ['sub_bass', 'bass'],
        'mid': ['low_mid', 'mid'],
        'high': ['high_mid', 'high']
    }

    for voice_name, bands in voice_band_map.items():
        if voice_name in voices:
            voice = voices[voice_name]
            current_gain = voice.get('gain', 0.5)

            # Check if any mapped band needs adjustment
            needs_increase = any(band_adj.get(b) == 'increase' for b in bands)
            needs_decrease = any(band_adj.get(b) == 'decrease' for b in bands)

            if needs_increase and not needs_decrease:
                delta = LEARNING_RATES['energy']['voice_gains'] * lr_decay
                new_gain = min(2.0, current_gain + delta)
                voice['gain'] = new_gain
                print(f"  Energy: {voice_name}_gain {current_gain:.2f} → {new_gain:.2f} (band increase)")
            elif needs_decrease and not needs_increase:
                delta = LEARNING_RATES['energy']['voice_gains'] * lr_decay
                new_gain = max(0.1, current_gain - delta)
                voice['gain'] = new_gain
                print(f"  Energy: {voice_name}_gain {current_gain:.2f} → {new_gain:.2f} (band decrease)")

    config['synth_config'] = synth
    return config


def apply_timbre_learning(config: Dict[str, Any], gradients: Dict[str, float],
                          iteration: int) -> Dict[str, Any]:
    """Apply timbre-based learning to config."""
    synth = config.get('synth_config', {})

    # Learning rate decay
    lr_decay = 1.0 / (1 + iteration * 0.1)

    # Adjust FM synthesis
    if gradients.get('adjust_fm'):
        fm = synth.get('fm', {})
        current_index = fm.get('modulation_index', 2.0)
        delta = gradients.get('fm_index_delta', 0.2) * LEARNING_RATES['timbre']['fm_index'] * lr_decay

        # Try increasing modulation for richer harmonics
        new_index = current_index + delta
        new_index = max(0.5, min(6.0, new_index))  # Clamp

        if 'fm' not in synth:
            synth['fm'] = {'enabled': True, 'modulator_ratio': 1.0}
        synth['fm']['modulation_index'] = new_index
        synth['fm']['enabled'] = True
        print(f"  Timbre: fm_index {current_index:.2f} → {new_index:.2f}")

    # Adjust brightness via high-shelf
    if 'brightness' in gradients:
        master = synth.get('master', {})
        current_shelf = master.get('high_shelf_boost', 0)

        if gradients['brightness'] == 'increase':
            delta = gradients.get('brightness_amount', 0.2)
            new_shelf = min(4.0, current_shelf + delta)
        else:
            delta = gradients.get('brightness_amount', 0.2)
            new_shelf = max(-2.0, current_shelf - delta)

        if 'master' not in synth:
            synth['master'] = {}
        synth['master']['high_shelf_boost'] = new_shelf
        print(f"  Timbre: high_shelf {current_shelf:.1f} → {new_shelf:.1f} dB")

    # Adjust detune for richer timbre
    voices = synth.get('voices', {})
    if gradients.get('mfcc_gap', 0) > 0.4:
        for voice_name in ['mid', 'high']:
            if voice_name in voices:
                voice = voices[voice_name]
                current_detune = voice.get('detune_cents', 5)
                delta = 3 * LEARNING_RATES['timbre']['detune'] * lr_decay
                new_detune = min(25, current_detune + delta)
                voice['detune_cents'] = new_detune
                print(f"  Timbre: {voice_name}_detune {current_detune:.0f} → {new_detune:.0f} cents")

    # Adjust filter for spectral shaping
    if gradients.get('brightness') == 'increase':
        for voice_name in ['mid', 'high']:
            if voice_name in voices:
                voice = voices[voice_name]
                current_lpf = voice.get('lpf', 4000)
                delta = 500 * LEARNING_RATES['timbre']['filter_cutoff'] * lr_decay
                new_lpf = min(12000, current_lpf + delta)
                voice['lpf'] = new_lpf
                print(f"  Timbre: {voice_name}_lpf {current_lpf:.0f} → {new_lpf:.0f} Hz")

    # Add noise component for more realistic timbre
    if gradients.get('add_noise'):
        noise_amount = gradients.get('noise_amount', 0.1)
        for voice_name in ['mid', 'high']:
            if voice_name in voices:
                voice = voices[voice_name]
                current_noise = voice.get('noise_mix', 0)
                new_noise = min(0.3, current_noise + noise_amount * lr_decay)
                voice['noise_mix'] = new_noise
                print(f"  Timbre: {voice_name}_noise {current_noise:.2f} → {new_noise:.2f}")

    config['synth_config'] = synth
    return config


def render_audio(strudel_path: str, output_path: str, config_path: str,
                 duration: int, scripts_dir: str) -> bool:
    """Render audio using Node.js renderer."""
    renderer = os.path.abspath(os.path.join(scripts_dir, "node", "dist", "render-strudel-node.js"))

    # Use absolute paths to avoid issues with spaces
    strudel_abs = os.path.abspath(strudel_path)
    output_abs = os.path.abspath(output_path)
    config_abs = os.path.abspath(config_path)

    cmd = [
        "node", renderer,
        strudel_abs,
        "-o", output_abs,
        "-d", str(duration),
        "--config", config_abs
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Render stderr: {result.stderr[:200] if result.stderr else 'none'}")
        return result.returncode == 0
    except Exception as e:
        print(f"Render error: {e}")
        return False


def compare_audio(original_path: str, rendered_path: str, config_path: str,
                  output_path: str, scripts_dir: str) -> Optional[Dict[str, Any]]:
    """Run audio comparison."""
    compare_script = os.path.join(scripts_dir, "python", "compare_audio.py")
    venv_python = os.path.join(scripts_dir, "python", ".venv", "bin", "python")

    cmd = [
        venv_python, compare_script,
        original_path, rendered_path,
        "--json",
        "--config", config_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            comparison = json.loads(result.stdout)
            # Save comparison
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            return comparison
    except Exception as e:
        print(f"Comparison error: {e}")

    return None


def run_learning_loop(
    original_path: str,
    strudel_path: str,
    config_path: str,
    output_dir: str,
    scripts_dir: str,
    max_iterations: int = 10,
    target_energy: float = 0.6,
    target_mfcc: float = 0.5
) -> Tuple[float, float]:
    """
    Run AI learning loop to improve energy and timbre.

    Returns final (energy_similarity, mfcc_similarity).
    """
    print("=" * 60)
    print("AI LEARNING OPTIMIZER")
    print("=" * 60)
    print(f"Target Energy: {target_energy:.0%}")
    print(f"Target MFCC:   {target_mfcc:.0%}")
    print(f"Max iterations: {max_iterations}")
    print()

    # Load initial config
    config = load_synth_config(config_path)

    best_energy = 0
    best_mfcc = 0
    best_overall = 0

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Render
        render_path = os.path.join(output_dir, f"learn_iter_{iteration}.wav")
        print("Rendering...")
        if not render_audio(strudel_path, render_path, config_path, 60, scripts_dir):
            print("  Render failed, stopping")
            break

        # Compare
        comparison_path = os.path.join(output_dir, f"learn_comparison_{iteration}.json")
        print("Comparing...")
        comparison = compare_audio(original_path, render_path, config_path,
                                   comparison_path, scripts_dir)

        if not comparison:
            print("  Comparison failed, stopping")
            break

        comp = comparison.get('comparison', {})
        energy_sim = comp.get('energy_similarity', 0)
        mfcc_sim = comp.get('mfcc_similarity', 0)
        overall = comp.get('overall_similarity', 0)

        print(f"\nResults:")
        print(f"  Energy: {energy_sim:.1%} (target: {target_energy:.0%})")
        print(f"  MFCC:   {mfcc_sim:.1%} (target: {target_mfcc:.0%})")
        print(f"  Overall: {overall:.1%}")

        # Track best
        if overall > best_overall:
            best_overall = overall
            best_energy = energy_sim
            best_mfcc = mfcc_sim
            # Save best config
            best_config_path = os.path.join(output_dir, "best_config.json")
            save_synth_config(config, best_config_path)
            print(f"  New best! Saved to {best_config_path}")

        # Check if targets reached
        if energy_sim >= target_energy and mfcc_sim >= target_mfcc:
            print(f"\nTargets reached!")
            break

        # Compute gradients
        print("\nLearning adjustments:")

        if energy_sim < target_energy:
            energy_grad = compute_energy_gradient(comparison)
            config = apply_energy_learning(config, energy_grad, iteration, comparison)

        if mfcc_sim < target_mfcc:
            timbre_grad = compute_timbre_gradient(comparison)
            config = apply_timbre_learning(config, timbre_grad, iteration)

        # Save updated config
        save_synth_config(config, config_path)

    print("\n" + "=" * 60)
    print("LEARNING COMPLETE")
    print("=" * 60)
    print(f"Best Energy: {best_energy:.1%}")
    print(f"Best MFCC:   {best_mfcc:.1%}")
    print(f"Best Overall: {best_overall:.1%}")

    return best_energy, best_mfcc


def main():
    parser = argparse.ArgumentParser(description='AI Learning Optimizer')
    parser.add_argument('original', help='Original audio path')
    parser.add_argument('strudel', help='Strudel code path')
    parser.add_argument('config', help='Synth config path')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--scripts-dir', default='.', help='Scripts directory')
    parser.add_argument('--iterations', type=int, default=10, help='Max iterations')
    parser.add_argument('--target-energy', type=float, default=0.6, help='Target energy similarity')
    parser.add_argument('--target-mfcc', type=float, default=0.5, help='Target MFCC similarity')

    args = parser.parse_args()

    # Find scripts dir
    scripts_dir = args.scripts_dir
    if scripts_dir == '.':
        # Try to find it
        for candidate in ['scripts', '../scripts', '../../scripts']:
            if os.path.exists(os.path.join(candidate, 'python', 'compare_audio.py')):
                scripts_dir = candidate
                break

    run_learning_loop(
        args.original,
        args.strudel,
        args.config,
        args.output_dir,
        scripts_dir,
        args.iterations,
        args.target_energy,
        args.target_mfcc
    )


if __name__ == '__main__':
    main()
