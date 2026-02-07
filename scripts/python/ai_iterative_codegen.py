#!/usr/bin/env python3
"""
AI-driven iterative CODE GENERATION optimizer.

Unlike the parameter optimizer, this actually modifies the Strudel code
based on comparison feedback. The goal is for the AI to learn to generate
better code, not just tweak renderer parameters.

Flow:
1. Render current Strudel code
2. Compare with original audio
3. AI analyzes gaps (bass too low, brightness mismatch, etc.)
4. AI modifies the Strudel code itself (.gain, .lpf, voice patterns)
5. Re-render with improved code
6. Track which code changes improved similarity
7. Feed learnings back for future code generation
"""

import argparse
import json
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import copy
import re

# Import from our other modules
from ai_code_improver import analyze_comparison_gaps, improve_strudel_code

# Import learning functions from ai_improver if available
try:
    from ai_improver import (
        get_track_hash, get_previous_run, get_best_run,
        get_learned_knowledge, store_run, store_knowledge
    )
    HAS_LEARNING = True
except ImportError:
    HAS_LEARNING = False
    def get_track_hash(path):
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read(1024*1024)).hexdigest()[:16]


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_strudel(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def save_strudel(code: str, path: str):
    with open(path, 'w') as f:
        f.write(code)


def run_render(strudel_path: str, output_path: str, config_path: str,
               duration: float, node_script: str) -> bool:
    """Run the Node.js renderer."""
    cmd = [
        "node", node_script,
        strudel_path,
        "-o", output_path,
        "--config", config_path,
        "-d", str(int(duration))
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Render error: {result.stderr}", file=sys.stderr)
    return result.returncode == 0


def run_comparison(original_path: str, rendered_path: str, output_path: str,
                   config_path: str = None) -> Optional[Dict]:
    """Run comparison and save results."""
    scripts_dir = Path(__file__).parent
    compare_script = scripts_dir / "compare_audio.py"

    # compare_audio.py outputs JSON to stdout with -j flag
    cmd = [
        sys.executable, str(compare_script),
        original_path, rendered_path,
        "-j"
    ]
    if config_path:
        cmd.extend(["--config", config_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Comparison failed: {result.stderr}", file=sys.stderr)
        return None

    try:
        comparison = json.loads(result.stdout)
        # Save to file for reference
        save_json(comparison, output_path)
        return comparison
    except json.JSONDecodeError:
        print(f"Failed to parse comparison output", file=sys.stderr)
        return None


def analyze_and_improve_code(strudel_code: str, comparison: Dict,
                              synth_config: Dict) -> Tuple[str, Dict, Dict]:
    """
    Analyze comparison and improve Strudel code.

    Returns: (improved_code, improved_config, gaps_found)
    """
    gaps = analyze_comparison_gaps(comparison)

    if not gaps['issues']:
        return strudel_code, synth_config, gaps

    improved_code, improved_config = improve_strudel_code(
        strudel_code, gaps, synth_config
    )

    return improved_code, improved_config, gaps


def apply_smart_code_improvements(code: str, comparison: Dict) -> str:
    """
    Apply smarter code improvements based on deeper analysis.

    This goes beyond simple regex replacements to actually understand
    what's happening in the code and make structural improvements.
    """
    orig = comparison.get('original', {})
    rend = comparison.get('rendered', {})
    comp = comparison.get('comparison', {})

    improved = code

    # Bass improvement: if bass is lacking, boost bass voice patterns
    orig_bass = orig.get('bands', {}).get('bass', 0) + orig.get('bands', {}).get('sub_bass', 0)
    rend_bass = rend.get('bands', {}).get('bass', 0) + rend.get('bands', {}).get('sub_bass', 0)

    if rend_bass < orig_bass * 0.7:
        bass_boost = min(1.5, orig_bass / max(rend_bass, 0.01))
        # Add or boost .gain() in bassFx function
        if 'bassFx' in improved:
            if '.gain(' in improved:
                # Boost existing gain
                improved = re.sub(
                    r'(bassFx.*?\.gain\()(\d+\.?\d*)(\))',
                    lambda m: f'{m.group(1)}{float(m.group(2)) * bass_boost:.2f}{m.group(3)}',
                    improved
                )
            else:
                # Add gain after sound
                improved = re.sub(
                    r'(bassFx.*?\.sound\([^)]+\))',
                    lambda m: f'{m.group(1)}.gain({bass_boost:.2f})',
                    improved
                )

    # Mid improvement: if mid is too hot, reduce
    orig_mid = orig.get('bands', {}).get('mid', 0)
    rend_mid = rend.get('bands', {}).get('mid', 0)

    if rend_mid > orig_mid * 1.3:
        mid_reduction = max(0.5, orig_mid / max(rend_mid, 0.01))
        if 'midFx' in improved:
            if '.gain(' in improved and 'midFx' in improved:
                improved = re.sub(
                    r'(midFx.*?\.gain\()(\d+\.?\d*)(\))',
                    lambda m: f'{m.group(1)}{float(m.group(2)) * mid_reduction:.2f}{m.group(3)}',
                    improved
                )

    # Brightness: adjust lpf based on spectral centroid
    brightness_sim = comp.get('brightness_similarity', 1.0)
    if brightness_sim < 0.8:
        orig_centroid = orig.get('spectral', {}).get('centroid_mean', 2000)
        rend_centroid = rend.get('spectral', {}).get('centroid_mean', 2000)

        if rend_centroid < orig_centroid * 0.85:
            # Too dark - increase LPF
            lpf_increase = int((orig_centroid - rend_centroid) * 1.5)
            improved = re.sub(
                r'\.lpf\((\d+)\)',
                lambda m: f'.lpf({int(m.group(1)) + lpf_increase})',
                improved
            )
        elif rend_centroid > orig_centroid * 1.15:
            # Too bright - decrease LPF
            lpf_decrease = int((rend_centroid - orig_centroid) * 1.5)
            improved = re.sub(
                r'\.lpf\((\d+)\)',
                lambda m: f'.lpf({max(200, int(m.group(1)) - lpf_decrease)})',
                improved
            )

    # Add room/reverb if timbre doesn't match
    mfcc_sim = comp.get('mfcc_similarity', 1.0)
    if mfcc_sim < 0.7 and '.room(' not in improved:
        # Add subtle room to help blend
        improved = re.sub(
            r'(let (?:bass|mid|high)Fx = p => p\.sound\([^)]+\))',
            r'\1.room(0.1)',
            improved
        )

    return improved


def record_learning(iteration: int, gaps: Dict, before_sim: float,
                    after_sim: float, learnings_path: str):
    """Record what worked for future learning."""
    learning = {
        'iteration': iteration,
        'gaps_identified': gaps.get('issues', []),
        'parameters_adjusted': gaps.get('parameters', {}),
        'similarity_before': before_sim,
        'similarity_after': after_sim,
        'improvement': after_sim - before_sim,
        'effective': after_sim > before_sim
    }

    # Load existing learnings or start fresh
    learnings = []
    if os.path.exists(learnings_path):
        try:
            learnings = load_json(learnings_path)
        except:
            learnings = []

    learnings.append(learning)
    save_json(learnings, learnings_path)


def iterative_codegen_optimize(
    original_path: str,
    strudel_path: str,
    config_path: str,
    output_dir: str,
    max_iterations: int = 10,
    target_similarity: float = 0.90,
    duration: float = 60
) -> Tuple[float, int, str]:
    """
    Run AI-driven iterative code generation optimization.

    Returns: (best_similarity, iterations_used, best_strudel_path)
    """
    scripts_dir = Path(__file__).parent
    node_script = scripts_dir.parent / "node" / "dist" / "render-strudel-node.js"

    if not node_script.exists():
        print(f"Error: Node renderer not found at {node_script}", file=sys.stderr)
        return 0, 0, strudel_path

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load initial files first (needed for genre/BPM lookup)
    current_code = load_strudel(strudel_path)
    current_config = load_json(config_path)

    # Get track hash for learning
    track_hash = get_track_hash(original_path)
    track_name = Path(original_path).stem

    # Check for previous runs (learning from past iterations)
    knowledge = []
    if HAS_LEARNING:
        previous_run = get_previous_run(track_hash)
        if previous_run:
            prev_sim = previous_run.get('similarity_overall', 0)
            prev_ver = previous_run.get('version', 0)
            print(f"Found previous run: v{prev_ver} with {prev_sim*100:.1f}% similarity")

            # Get learned knowledge for this context
            best_run = get_best_run(track_hash)
            if best_run and best_run.get('similarity_overall', 0) > prev_sim:
                print(f"Best historical run: v{best_run.get('version')} with {best_run.get('similarity_overall')*100:.1f}%")
        else:
            print("No previous runs found - starting fresh")

        # Get learned knowledge for this genre/BPM/key
        synth_cfg = current_config.get('synth_config', {})
        genre = synth_cfg.get('genre', '')
        bpm = synth_cfg.get('tempo', {}).get('bpm', 120)
        key = synth_cfg.get('key', '')
        key_type = 'minor' if 'minor' in key.lower() else 'major'

        knowledge = get_learned_knowledge(genre, bpm, key_type)
        if knowledge:
            print(f"📚 Found {len(knowledge)} learned adjustments for genre={genre or 'any'}, BPM~{bpm:.0f}, key={key_type}")
            for k in knowledge[:3]:
                print(f"   - {k.get('parameter_name')}: {k.get('parameter_new_value')} (+{k.get('similarity_improvement', 0)*100:.1f}%)")
    else:
        print("Learning system not available")

    best_similarity = 0
    best_code = current_code
    best_config = current_config
    best_strudel_path = strudel_path

    learnings_path = os.path.join(output_dir, "ai_learnings.json")

    print(f"\n{'='*60}")
    print("AI-DRIVEN ITERATIVE CODE GENERATION")
    print(f"{'='*60}")
    print(f"Target similarity: {target_similarity*100:.0f}%")
    print(f"Max iterations: {max_iterations}")
    print()

    for iteration in range(1, max_iterations + 1):
        print(f"--- Iteration {iteration}/{max_iterations} ---")

        # Save current code and config
        iter_strudel = os.path.join(output_dir, f"strudel_v{iteration:02d}.js")
        iter_config = os.path.join(output_dir, f"config_v{iteration:02d}.json")
        iter_render = os.path.join(output_dir, f"render_v{iteration:02d}.wav")
        iter_comparison = os.path.join(output_dir, f"comparison_v{iteration:02d}.json")

        save_strudel(current_code, iter_strudel)
        save_json(current_config, iter_config)

        # Render
        print(f"  Rendering...")
        if not run_render(iter_strudel, iter_render, iter_config, duration, str(node_script)):
            print("  Render failed, stopping")
            break

        # Compare
        print(f"  Comparing...")
        comparison = run_comparison(original_path, iter_render, iter_comparison, iter_config)
        if comparison is None:
            print("  Comparison failed, stopping")
            break

        current_similarity = comparison.get('comparison', {}).get('overall_similarity', 0)
        mfcc_sim = comparison.get('comparison', {}).get('mfcc_similarity', 0)
        energy_sim = comparison.get('comparison', {}).get('energy_similarity', 0)
        brightness_sim = comparison.get('comparison', {}).get('brightness_similarity', 0)
        freq_sim = comparison.get('comparison', {}).get('frequency_balance_similarity', 0)

        print(f"  Similarity: {current_similarity*100:.1f}%")
        print(f"    MFCC={mfcc_sim*100:.0f}% Energy={energy_sim*100:.0f}% "
              f"Brightness={brightness_sim*100:.0f}% FreqBal={freq_sim*100:.0f}%")

        # Track best
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_code = current_code
            best_config = current_config
            best_strudel_path = iter_strudel
            print(f"  ★ New best!")
        elif iteration > 1 and current_similarity < best_similarity - 0.01:
            # Similarity dropped by more than 1% - revert to best code
            print(f"  ⚠ Similarity dropped, reverting to best code")
            current_code = best_code
            current_config = best_config
            # Skip further improvement attempts on this branch
            continue

        # Check if target reached
        if current_similarity >= target_similarity:
            print(f"\n✓ Target similarity {target_similarity*100:.0f}% reached!")
            break

        # AI: Analyze gaps and improve code
        print(f"  AI analyzing gaps...")
        improved_code, improved_config, gaps = analyze_and_improve_code(
            current_code, comparison, current_config
        )

        if gaps['issues']:
            print(f"  Issues found: {', '.join(gaps['issues'])}")
            print(f"  Suggestions: {'; '.join(gaps['suggestions'][:2])}")

        # Apply smarter improvements
        improved_code = apply_smart_code_improvements(improved_code, comparison)

        # Check if code actually changed
        if improved_code == current_code and improved_config == current_config:
            print(f"  No further improvements found, converged")
            break

        # Record what we learned (local file)
        record_learning(iteration, gaps, best_similarity, current_similarity, learnings_path)

        # Store knowledge if this iteration IMPROVED similarity
        improvement = current_similarity - best_similarity
        if HAS_LEARNING and improvement > 0.005:  # >0.5% improvement
            # Extract genre/BPM/key from config
            synth_cfg = current_config.get('synth_config', {})
            genre = synth_cfg.get('genre', '')
            bpm = synth_cfg.get('tempo', {}).get('bpm', 120)
            key = synth_cfg.get('key', '')
            key_type = 'minor' if 'minor' in key.lower() else 'major'

            # Store each parameter adjustment that worked
            for param_name, param_value in gaps.get('parameters', {}).items():
                try:
                    store_knowledge(
                        genre=genre,
                        bpm=bpm,
                        key_type=key_type,
                        parameter_name=param_name,
                        old_value='default',
                        new_value=str(param_value),
                        improvement=improvement
                    )
                    print(f"  📚 Stored learning: {param_name}={param_value} improved {improvement*100:.1f}% (genre={genre or 'any'})")
                except Exception as e:
                    pass  # Don't fail on learning storage errors

        # Store run in ClickHouse for cross-session learning
        if HAS_LEARNING:
            try:
                # Extract genre from config (try multiple locations)
                cfg_genre = (current_config.get('genre', '') or
                            current_config.get('synth_config', {}).get('genre', ''))
                cfg_style = (current_config.get('style', '') or
                            current_config.get('synth_config', {}).get('style', ''))
                cfg_bpm = (current_config.get('synth_config', {}).get('tempo', {}).get('bpm', 0) or
                          current_config.get('tempo', {}).get('tempo_bpm', 120))
                cfg_key = (current_config.get('synth_config', {}).get('key', '') or
                          current_config.get('key', ''))

                store_run(
                    track_hash=track_hash,
                    track_name=track_name,
                    version=iteration,
                    strudel_code=current_code,
                    bpm=cfg_bpm,
                    key=cfg_key,
                    style=cfg_style,
                    genre=cfg_genre,
                    comparison=comparison,
                    mix_params=gaps.get('parameters', {}),
                    improved_from=iteration - 1 if iteration > 1 else None,
                    ai_suggestions=gaps.get('suggestions', [])
                )
            except Exception as e:
                print(f"  Warning: Could not store run in ClickHouse: {e}")

        # Update for next iteration
        current_code = improved_code
        current_config = improved_config

        print()

    # Save final best results
    final_strudel = os.path.join(output_dir, "strudel_optimized.js")
    final_config = os.path.join(output_dir, "synth_config_optimized.json")
    save_strudel(best_code, final_strudel)
    save_json(best_config, final_config)

    print(f"\n{'='*60}")
    print(f"AI CODE OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best similarity: {best_similarity*100:.1f}%")
    print(f"Iterations used: {iteration}")
    print(f"Optimized code: {final_strudel}")
    print(f"Optimized config: {final_config}")

    return best_similarity, iteration, final_strudel


def main():
    parser = argparse.ArgumentParser(
        description='AI-driven iterative Strudel code optimization'
    )
    parser.add_argument('original', help='Path to original audio (melodic stem)')
    parser.add_argument('strudel', help='Path to initial Strudel code')
    parser.add_argument('config', help='Path to synth_config.json')
    parser.add_argument('output_dir', help='Output directory for iterations')
    parser.add_argument('-n', '--iterations', type=int, default=10,
                       help='Maximum iterations (default: 10)')
    parser.add_argument('-t', '--target', type=float, default=0.90,
                       help='Target similarity (default: 0.90)')
    parser.add_argument('-d', '--duration', type=float, default=60,
                       help='Render duration in seconds (default: 60)')

    args = parser.parse_args()

    best_sim, iters, best_path = iterative_codegen_optimize(
        args.original,
        args.strudel,
        args.config,
        args.output_dir,
        args.iterations,
        args.target,
        args.duration
    )

    # Output result as JSON for pipeline integration
    result = {
        "best_similarity": best_sim,
        "iterations_used": iters,
        "target_reached": best_sim >= args.target,
        "optimized_strudel": best_path
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
