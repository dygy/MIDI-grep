#!/usr/bin/env python3
"""
AI-driven iterative parameter optimizer.

Uses comparison feedback to adjust synthesis parameters toward target similarity.
No hardcoded adjustments - learns adjustment rates from the error gradient.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import copy


def load_config(config_path: str) -> Dict:
    """Load synthesis config."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict, config_path: str):
    """Save synthesis config."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_render(strudel_path: str, output_path: str, config_path: str,
               duration: float, node_script: str) -> bool:
    """Run the Node.js renderer with current config."""
    cmd = [
        "node", node_script,
        strudel_path,
        "-o", output_path,
        "--config", config_path,
        "-d", str(int(duration))
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_comparison(original_path: str, rendered_path: str, config_path: str,
                   python_path: str, compare_script: str) -> Optional[Dict]:
    """Run comparison and get results."""
    cmd = [
        python_path, compare_script,
        original_path, rendered_path,
        "--config", config_path,
        "-j"  # JSON output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Comparison failed: {result.stderr}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse comparison output", file=sys.stderr)
        return None


def compute_adjustments(comparison: Dict, config: Dict) -> Dict[str, float]:
    """
    Compute parameter adjustments based on comparison feedback.

    Uses the error between original and rendered to determine adjustment direction and magnitude.
    This is gradient-free optimization based on direct feedback signals.
    """
    adjustments = {}

    orig_bands = comparison.get('original', {}).get('bands', {})
    rend_bands = comparison.get('rendered', {}).get('bands', {})
    comp = comparison.get('comparison', {})

    # Calculate band errors (positive = rendered has too much, negative = too little)
    sub_bass_error = rend_bands.get('sub_bass', 0) - orig_bands.get('sub_bass', 0)
    bass_error = rend_bands.get('bass', 0) - orig_bands.get('bass', 0)
    mid_error = rend_bands.get('mid', 0) - orig_bands.get('mid', 0)
    high_error = (rend_bands.get('high_mid', 0) + rend_bands.get('high', 0)) - \
                 (orig_bands.get('high_mid', 0) + orig_bands.get('high', 0))

    # Learning rate based on current similarity (higher error = larger adjustments)
    freq_sim = comp.get('frequency_balance_similarity', 0.5)
    learning_rate = 0.5 * (1 - freq_sim)  # 0-0.5 based on error

    # Adjust voice gains based on band errors
    # Negative error means we need MORE of that band

    # Bass adjustment
    total_bass_error = sub_bass_error + bass_error
    if abs(total_bass_error) > 0.02:  # 2% threshold
        # If we have too little bass (negative error), increase gain
        bass_adj = -total_bass_error * learning_rate * 2  # Scale factor for bass
        adjustments['bass_gain'] = bass_adj

        # Also adjust sub-octave gain for sub-bass
        if abs(sub_bass_error) > 0.05:
            adjustments['sub_octave_gain'] = -sub_bass_error * learning_rate

    # Mid adjustment
    if abs(mid_error) > 0.02:
        mid_adj = -mid_error * learning_rate * 1.5
        adjustments['mid_gain'] = mid_adj

    # High adjustment
    if abs(high_error) > 0.01:
        high_adj = -high_error * learning_rate * 3  # Highs are more sensitive
        adjustments['high_gain'] = high_adj

    # Brightness adjustment based on spectral centroid
    brightness_sim = comp.get('brightness_similarity', 0.5)
    if brightness_sim < 0.8:
        orig_centroid = comparison.get('original', {}).get('spectral', {}).get('centroid_mean', 2000)
        rend_centroid = comparison.get('rendered', {}).get('spectral', {}).get('centroid_mean', 2000)

        if rend_centroid < orig_centroid * 0.9:
            # Too dark - increase LPF cutoff
            adjustments['lpf_increase'] = 0.1 * (1 - brightness_sim)
        elif rend_centroid > orig_centroid * 1.1:
            # Too bright - decrease LPF cutoff
            adjustments['lpf_decrease'] = 0.1 * (1 - brightness_sim)

    # Energy/loudness adjustment - use direct RMS ratio for aggressive correction
    orig_rms = comparison.get('original', {}).get('spectral', {}).get('rms_mean', 0.3)
    rend_rms = comparison.get('rendered', {}).get('spectral', {}).get('rms_mean', 0.3)
    rms_ratio = orig_rms / max(rend_rms, 0.01)

    if rms_ratio > 1.2:  # Rendered is >20% quieter
        # Calculate gain multiplier needed (with dampening to avoid overshoot)
        gain_mult = min(2.0, rms_ratio * 0.5)  # Aim for 50% of the difference per iteration
        adjustments['master_gain_mult'] = gain_mult
        adjustments['voice_gain_boost'] = min(0.3, (rms_ratio - 1) * 0.15)  # Boost all voices

    return adjustments


def apply_adjustments(config: Dict, adjustments: Dict[str, float]) -> Dict:
    """Apply computed adjustments to config."""
    new_config = copy.deepcopy(config)

    voices = new_config.get('synth_config', {}).get('voices', {})

    # Apply bass gain adjustment
    if 'bass_gain' in adjustments:
        current = voices.get('bass', {}).get('gain', 0.5)
        new_val = max(0.01, min(1.5, current + adjustments['bass_gain']))
        if 'bass' in voices:
            voices['bass']['gain'] = new_val
        print(f"  bass_gain: {current:.3f} → {new_val:.3f} ({adjustments['bass_gain']:+.3f})")

    # Apply sub-octave gain adjustment
    if 'sub_octave_gain' in adjustments:
        current = voices.get('bass', {}).get('sub_octave_gain', 0.2)
        new_val = max(0, min(0.8, current + adjustments['sub_octave_gain']))
        if 'bass' in voices:
            voices['bass']['sub_octave_gain'] = new_val
        print(f"  sub_octave: {current:.3f} → {new_val:.3f} ({adjustments['sub_octave_gain']:+.3f})")

    # Apply mid gain adjustment
    if 'mid_gain' in adjustments:
        current = voices.get('mid', {}).get('gain', 0.8)
        new_val = max(0.1, min(2.0, current + adjustments['mid_gain']))
        if 'mid' in voices:
            voices['mid']['gain'] = new_val
        print(f"  mid_gain: {current:.3f} → {new_val:.3f} ({adjustments['mid_gain']:+.3f})")

    # Apply high gain adjustment
    if 'high_gain' in adjustments:
        current = voices.get('high', {}).get('gain', 0.5)
        new_val = max(0.05, min(1.5, current + adjustments['high_gain']))
        if 'high' in voices:
            voices['high']['gain'] = new_val
        print(f"  high_gain: {current:.3f} → {new_val:.3f} ({adjustments['high_gain']:+.3f})")

    # Apply LPF adjustment
    if 'lpf_increase' in adjustments:
        current = voices.get('mid', {}).get('lpf', 4000)
        new_val = min(12000, current * (1 + adjustments['lpf_increase']))
        if 'mid' in voices:
            voices['mid']['lpf'] = new_val
        print(f"  mid_lpf: {current:.0f} → {new_val:.0f} Hz (+{adjustments['lpf_increase']*100:.0f}%)")

    if 'lpf_decrease' in adjustments:
        current = voices.get('mid', {}).get('lpf', 4000)
        new_val = max(1000, current * (1 - adjustments['lpf_decrease']))
        if 'mid' in voices:
            voices['mid']['lpf'] = new_val
        print(f"  mid_lpf: {current:.0f} → {new_val:.0f} Hz (-{adjustments['lpf_decrease']*100:.0f}%)")

    # Apply master gain multiplier (for loudness matching)
    if 'master_gain_mult' in adjustments:
        master = new_config.get('synth_config', {}).get('master', {})
        current = master.get('gain', 1.0)
        new_val = min(3.0, current * adjustments['master_gain_mult'])
        if 'master' in new_config.get('synth_config', {}):
            new_config['synth_config']['master']['gain'] = new_val
        print(f"  master_gain: {current:.3f} → {new_val:.3f} (×{adjustments['master_gain_mult']:.2f})")

    # Apply voice gain boost (boost all voices for loudness)
    if 'voice_gain_boost' in adjustments:
        boost = adjustments['voice_gain_boost']
        for voice_name in ['bass', 'mid', 'high']:
            if voice_name in voices:
                current = voices[voice_name].get('gain', 0.5)
                new_val = min(2.5, current + boost)
                voices[voice_name]['gain'] = new_val
        print(f"  all_voices: +{boost:.3f} gain boost")

    return new_config


def optimize(
    original_path: str,
    strudel_path: str,
    output_dir: str,
    config_path: str,
    max_iterations: int = 10,
    target_similarity: float = 0.85,
    duration: float = 60
) -> Tuple[float, int]:
    """
    Run iterative optimization loop.

    Returns: (best_similarity, iterations_used)
    """
    scripts_dir = Path(__file__).parent
    node_script = scripts_dir.parent / "node" / "dist" / "render-strudel-node.js"
    compare_script = scripts_dir / "compare_audio.py"
    python_path = sys.executable

    if not node_script.exists():
        print(f"Error: Node renderer not found at {node_script}", file=sys.stderr)
        return 0, 0

    config = load_config(config_path)
    best_similarity = 0
    best_config = config

    print(f"\n{'='*60}")
    print("AI-DRIVEN ITERATIVE OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Target similarity: {target_similarity*100:.0f}%")
    print(f"Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Render with current config
        render_path = os.path.join(output_dir, f"render_opt_{iteration:02d}.wav")
        iter_config_path = os.path.join(output_dir, f"config_opt_{iteration:02d}.json")

        save_config(config, iter_config_path)

        print(f"Rendering...")
        if not run_render(strudel_path, render_path, iter_config_path, duration, str(node_script)):
            print("Render failed, stopping")
            break

        # Compare
        print(f"Comparing...")
        comparison = run_comparison(original_path, render_path, iter_config_path,
                                   python_path, str(compare_script))
        if comparison is None:
            print("Comparison failed, stopping")
            break

        current_similarity = comparison.get('comparison', {}).get('overall_similarity', 0)
        freq_balance = comparison.get('comparison', {}).get('frequency_balance_similarity', 0)

        print(f"Similarity: {current_similarity*100:.1f}% (freq balance: {freq_balance*100:.1f}%)")

        # Track best
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_config = copy.deepcopy(config)
            print(f"  ★ New best!")

        # Check if target reached
        if current_similarity >= target_similarity:
            print(f"\n✓ Target similarity {target_similarity*100:.0f}% reached!")
            break

        # Compute and apply adjustments
        print(f"Computing adjustments...")
        adjustments = compute_adjustments(comparison, config)

        if not adjustments:
            print("No adjustments needed, converged")
            break

        config = apply_adjustments(config, adjustments)

    # Save best config
    best_config_path = os.path.join(output_dir, "synth_config_optimized.json")
    save_config(best_config, best_config_path)
    print(f"\nBest config saved: {best_config_path}")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best similarity: {best_similarity*100:.1f}%")
    print(f"{'='*60}")

    return best_similarity, iteration


def main():
    parser = argparse.ArgumentParser(description='AI-driven iterative parameter optimizer')
    parser.add_argument('original', help='Path to original audio')
    parser.add_argument('strudel', help='Path to Strudel code file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('config', help='Path to initial synth_config.json')
    parser.add_argument('-n', '--iterations', type=int, default=10,
                       help='Maximum iterations (default: 10)')
    parser.add_argument('-t', '--target', type=float, default=0.85,
                       help='Target similarity (default: 0.85)')
    parser.add_argument('-d', '--duration', type=float, default=60,
                       help='Render duration in seconds (default: 60)')

    args = parser.parse_args()

    best_sim, iters = optimize(
        args.original,
        args.strudel,
        args.output_dir,
        args.config,
        args.iterations,
        args.target,
        args.duration
    )

    # Output result as JSON for pipeline integration
    result = {
        "best_similarity": best_sim,
        "iterations_used": iters,
        "target_reached": best_sim >= args.target
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
