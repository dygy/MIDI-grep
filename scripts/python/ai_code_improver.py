#!/usr/bin/env python3
"""
AI Code Improver - Learns from comparison feedback and improves Strudel code.

Instead of just tweaking renderer parameters, this actually modifies the generated
Strudel code based on what the comparison tells us is wrong.

Now enhanced with Mel Spectrogram analysis for deeper insights.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Try to import spectrogram analyzer
try:
    from spectrogram_analyzer import analyze_spectrograms, format_for_llm
    HAS_SPECTROGRAM = True
except ImportError:
    HAS_SPECTROGRAM = False


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_strudel(path: str) -> str:
    """Load Strudel code."""
    with open(path, 'r') as f:
        return f.read()


def save_strudel(code: str, path: str):
    """Save Strudel code."""
    with open(path, 'w') as f:
        f.write(code)


def analyze_with_spectrogram(original_audio: str, rendered_audio: str,
                              duration: float = 60.0) -> Optional[Dict[str, Any]]:
    """
    Perform deep spectrogram analysis if available.

    Returns detailed insights about frequency/time differences.
    """
    if not HAS_SPECTROGRAM:
        return None

    try:
        analysis = analyze_spectrograms(original_audio, rendered_audio, duration)
        return analysis
    except Exception as e:
        print(f"Spectrogram analysis failed: {e}", file=sys.stderr)
        return None


def spectrogram_to_gaps(spec_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert spectrogram analysis into gap parameters for code modification.

    This translates dB differences into gain multipliers and envelope suggestions.
    """
    gaps = {
        'issues': [],
        'suggestions': [],
        'parameters': {},
        'envelope_changes': {},
    }

    if not spec_analysis:
        return gaps

    bands = spec_analysis.get('bands', {})

    # Map spectrogram bands to our voice structure
    band_to_voice = {
        'sub_bass': 'bass',
        'bass': 'bass',
        'low_mid': 'mid',
        'mid': 'mid',
        'high_mid': 'high',
        'high': 'high',
        'air': 'high',
    }

    # Aggregate adjustments per voice
    voice_adjustments = {'bass': [], 'mid': [], 'high': []}

    for band_name, band_data in bands.items():
        energy_diff = band_data.get('energy_diff_db', 0)
        attack_diff = band_data.get('attack_diff_db', 0)
        shape_corr = band_data.get('shape_correlation', 1.0)

        voice = band_to_voice.get(band_name, 'mid')

        # Energy adjustment (dB to linear multiplier)
        if abs(energy_diff) > 2:  # More than 2dB difference
            # Positive diff = rendered is quieter, need to boost
            # Negative diff = rendered is louder, need to cut
            multiplier = 10 ** (energy_diff / 20)  # dB to linear
            voice_adjustments[voice].append(('gain', multiplier))

            if energy_diff > 0:
                gaps['issues'].append(f'{band_name}_too_quiet')
            else:
                gaps['issues'].append(f'{band_name}_too_loud')

        # Attack adjustment
        if abs(attack_diff) > 4:
            # Positive diff = rendered attack too soft, need faster attack
            # Negative diff = rendered attack too hard, need slower attack
            if attack_diff > 0:
                gaps['envelope_changes'].setdefault(voice, {})['attack_faster'] = min(0.5, abs(attack_diff) / 20)
            else:
                gaps['envelope_changes'].setdefault(voice, {})['attack_slower'] = min(0.5, abs(attack_diff) / 20)

        # Envelope shape issues
        if shape_corr < 0.5:
            gaps['issues'].append(f'{band_name}_envelope_mismatch')
            gaps['envelope_changes'].setdefault(voice, {})['needs_reshape'] = True

    # Calculate average gain adjustment per voice
    for voice, adjustments in voice_adjustments.items():
        gain_mults = [mult for (adj_type, mult) in adjustments if adj_type == 'gain']
        if gain_mults:
            # Use geometric mean for gain multipliers
            import math
            avg_mult = math.exp(sum(math.log(max(0.01, m)) for m in gain_mults) / len(gain_mults))
            # Clamp to reasonable range
            avg_mult = max(0.1, min(3.0, avg_mult))
            gaps['parameters'][f'{voice}_gain_mult'] = avg_mult

            if avg_mult < 0.8:
                gaps['suggestions'].append(f'Reduce {voice} gain to {avg_mult:.0%}')
            elif avg_mult > 1.2:
                gaps['suggestions'].append(f'Increase {voice} gain to {avg_mult:.0%}')

    # Transient analysis
    transients = spec_analysis.get('transients', {})
    trans_ratio = transients.get('transient_count_ratio', 1.0)
    if trans_ratio < 0.7:
        gaps['issues'].append('missing_transients')
        gaps['suggestions'].append('Add more transient attack (reduce attack time)')
        gaps['parameters']['global_attack_mult'] = 0.5
    elif trans_ratio > 1.5:
        gaps['issues'].append('excess_transients')
        gaps['suggestions'].append('Soften attacks (increase attack time)')
        gaps['parameters']['global_attack_mult'] = 2.0

    # Harmonic analysis
    harmonics = spec_analysis.get('harmonics', {})
    waveform_sug = harmonics.get('waveform_suggestion')
    if waveform_sug:
        gaps['issues'].append('wrong_waveform')
        gaps['suggestions'].append(waveform_sug)
        if 'sawtooth' in waveform_sug.lower():
            gaps['parameters']['prefer_saw'] = True
        elif 'square' in waveform_sug.lower():
            gaps['parameters']['prefer_square'] = True

    return gaps


def analyze_comparison_gaps(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze comparison results and identify what needs improvement.
    Returns actionable insights for code modification.
    """
    comp = comparison.get('comparison', {})
    orig = comparison.get('original', {})
    rend = comparison.get('rendered', {})

    gaps = {
        'issues': [],
        'suggestions': [],
        'parameters': {}
    }

    # Energy analysis
    energy_sim = comp.get('energy_similarity', 0)
    if energy_sim < 0.8:
        orig_rms = orig.get('spectral', {}).get('rms_mean', 0.3)
        rend_rms = rend.get('spectral', {}).get('rms_mean', 0.1)
        if rend_rms < orig_rms * 0.8:
            gaps['issues'].append('too_quiet')
            gaps['suggestions'].append('Increase gain values in effects')
            gaps['parameters']['gain_multiplier'] = orig_rms / max(rend_rms, 0.01)

    # Brightness analysis
    brightness_sim = comp.get('brightness_similarity', 0)
    if brightness_sim < 0.8:
        orig_centroid = orig.get('spectral', {}).get('centroid_mean', 2000)
        rend_centroid = rend.get('spectral', {}).get('centroid_mean', 1500)
        if rend_centroid < orig_centroid * 0.85:
            gaps['issues'].append('too_dark')
            gaps['suggestions'].append('Increase LPF cutoff or add high-frequency content')
            gaps['parameters']['lpf_increase'] = int((orig_centroid - rend_centroid) * 2)
            # Also add high-shelf boost to brighten the mix
            gaps['parameters']['high_shelf_boost'] = min(4.0, (orig_centroid - rend_centroid) / 300)
        elif rend_centroid > orig_centroid * 1.15:
            gaps['issues'].append('too_bright')
            gaps['suggestions'].append('Decrease LPF cutoff')
            gaps['parameters']['lpf_decrease'] = int((rend_centroid - orig_centroid) * 2)

    # Frequency balance analysis - use direct band comparison for better sensitivity
    freq_sim = comp.get('frequency_balance_similarity', 0)
    # Always analyze bands - the similarity score can mask significant imbalances
    orig_bands = orig.get('bands', {})
    rend_bands = rend.get('bands', {})
    if orig_bands and rend_bands:

        # Check bass - more sensitive threshold (85% of original triggers adjustment)
        orig_bass = orig_bands.get('bass', 0) + orig_bands.get('sub_bass', 0)
        rend_bass = rend_bands.get('bass', 0) + rend_bands.get('sub_bass', 0)
        bass_ratio = rend_bass / max(orig_bass, 0.01)
        if bass_ratio < 0.85:
            gaps['issues'].append('needs_more_bass')
            gaps['suggestions'].append(f'Increase bass voice gain (currently {bass_ratio:.0%} of original)')
            # Conservative adjustment - aim for 80% of the difference per iteration
            target_mult = 1.0 / max(bass_ratio, 0.3)
            gaps['parameters']['bass_gain_mult'] = 1.0 + (target_mult - 1.0) * 0.5  # 50% of full correction

        # Check mids - more sensitive threshold (115% of original triggers adjustment)
        orig_mid = orig_bands.get('mid', 0) + orig_bands.get('low_mid', 0)
        rend_mid = rend_bands.get('mid', 0) + rend_bands.get('low_mid', 0)
        mid_ratio = rend_mid / max(orig_mid, 0.01)
        if mid_ratio > 1.15:
            gaps['issues'].append('too_much_mid')
            gaps['suggestions'].append(f'Decrease mid voice gain (currently {mid_ratio:.0%} of original)')
            # Conservative adjustment - aim for 50% of the difference per iteration
            # to avoid over-correction that hurts MFCC
            target_mult = 1.0 / mid_ratio
            gaps['parameters']['mid_gain_mult'] = max(0.6, 1.0 + (target_mult - 1.0) * 0.5)

        # Check highs
        orig_high = orig_bands.get('high', 0) + orig_bands.get('high_mid', 0)
        rend_high = rend_bands.get('high', 0) + rend_bands.get('high_mid', 0)
        high_ratio = rend_high / max(orig_high, 0.01) if orig_high > 0.01 else 1.0
        if high_ratio < 0.7:
            gaps['issues'].append('needs_more_highs')
            gaps['suggestions'].append(f'Increase high voice gain (currently {high_ratio:.0%} of original)')
            gaps['parameters']['high_gain_mult'] = min(2.0, 1.0 / max(high_ratio, 0.3))

    # MFCC (timbre) analysis
    mfcc_sim = comp.get('mfcc_similarity', 0)
    if mfcc_sim < 0.6:
        gaps['issues'].append('timbre_mismatch')
        gaps['suggestions'].append('Try different sound/instrument')

    return gaps


def improve_strudel_code(code: str, gaps: Dict[str, Any], synth_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Modify Strudel code based on identified gaps.
    Returns (improved_code, updated_synth_config).

    Key insight: We modify the ACTUAL Strudel code (the effect functions),
    not just the synth config parameters. This is what the user wants -
    the AI should learn to generate better CODE.
    """
    improved_code = code
    params = gaps.get('parameters', {})
    issues = gaps.get('issues', [])

    # ============================================
    # STRUDEL CODE MODIFICATIONS
    # ============================================

    # 1. Handle bass gain (needs_more_bass issue)
    if 'bass_gain_mult' in params:
        mult = min(2.0, params['bass_gain_mult'])
        # Find bassFx and boost its gain
        if 'bassFx' in improved_code:
            # Try to modify existing .gain()
            pattern = r'(bassFx.*?\.sound\([^)]+\))\.gain\(([0-9.]+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_gain = float(match.group(2))
                new_gain = min(1.5, current_gain * mult)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.gain({new_gain:.2f})',
                    improved_code
                )
            else:
                # Add .gain() if not present
                improved_code = re.sub(
                    r'(bassFx.*?\.sound\([^)]+\))',
                    lambda m: f'{m.group(1)}.gain({min(1.0, 0.5 * mult):.2f})',
                    improved_code
                )

    # 2. Handle mid gain (too_much_mid issue)
    if 'mid_gain_mult' in params:
        mult = max(0.3, params['mid_gain_mult'])
        if 'midFx' in improved_code:
            # Try to modify existing .gain()
            pattern = r'(midFx.*?\.sound\([^)]+\))\.gain\(([0-9.]+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_gain = float(match.group(2))
                new_gain = max(0.3, current_gain * mult)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.gain({new_gain:.2f})',
                    improved_code
                )
            else:
                # Add .gain() if not present (assume default gain of 1.0)
                new_gain = max(0.3, 1.0 * mult)
                improved_code = re.sub(
                    r'(midFx.*?\.sound\([^)]+\))',
                    lambda m: f'{m.group(1)}.gain({new_gain:.2f})',
                    improved_code
                )

    # 2b. Handle high gain (needs_more_highs issue)
    if 'high_gain_mult' in params:
        mult = min(2.0, params['high_gain_mult'])
        if 'highFx' in improved_code:
            pattern = r'(highFx.*?\.sound\([^)]+\))\.gain\(([0-9.]+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_gain = float(match.group(2))
                new_gain = min(2.0, current_gain * mult)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.gain({new_gain:.2f})',
                    improved_code
                )
            else:
                # Add .gain() if not present
                improved_code = re.sub(
                    r'(highFx.*?\.sound\([^)]+\))',
                    lambda m: f'{m.group(1)}.gain({min(1.5, 0.8 * mult):.2f})',
                    improved_code
                )

    # 3. Handle overall gain (too_quiet issue)
    if 'gain_multiplier' in params:
        mult = min(1.5, params['gain_multiplier'])
        # Boost all Fx gains proportionally
        for voice in ['bassFx', 'midFx', 'highFx']:
            pattern = rf'({voice}.*?\.sound\([^)]+\))\.gain\(([0-9.]+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_gain = float(match.group(2))
                new_gain = min(2.0, current_gain * mult)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.gain({new_gain:.2f})',
                    improved_code,
                    count=1  # Only first match to avoid double-replacing
                )

    # 4. Handle brightness (too_dark / too_bright issues)
    if 'lpf_increase' in params:
        increase = params['lpf_increase']
        # Open up LPF for mid and high voices
        for voice in ['midFx', 'highFx']:
            pattern = rf'({voice}[^)]+)\.lpf\((\d+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_lpf = int(match.group(2))
                new_lpf = min(16000, current_lpf + increase)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.lpf({new_lpf})',
                    improved_code,
                    count=1
                )

    if 'lpf_decrease' in params:
        decrease = params['lpf_decrease']
        for voice in ['midFx', 'highFx']:
            pattern = rf'({voice}[^)]+)\.lpf\((\d+)\)'
            match = re.search(pattern, improved_code)
            if match:
                current_lpf = int(match.group(2))
                new_lpf = max(500, current_lpf - decrease)
                improved_code = re.sub(
                    pattern,
                    lambda m: f'{m.group(1)}.lpf({new_lpf})',
                    improved_code,
                    count=1
                )

    # ============================================
    # SYNTH CONFIG MODIFICATIONS (for renderer)
    # ============================================

    synth = synth_config.get('synth_config', synth_config)
    voices = synth.get('voices', {})

    if 'bass_gain_mult' in params:
        if 'bass' in voices:
            current = voices['bass'].get('gain', 0.5)
            voices['bass']['gain'] = min(2.0, current * params['bass_gain_mult'])

    if 'mid_gain_mult' in params:
        if 'mid' in voices:
            current = voices['mid'].get('gain', 0.5)
            voices['mid']['gain'] = max(0.1, current * params['mid_gain_mult'])

    if 'high_shelf_boost' in params:
        # Add high shelf boost to master
        master = synth.get('master', {})
        current_shelf = master.get('high_shelf_boost', 0)
        master['high_shelf_boost'] = min(6.0, current_shelf + params['high_shelf_boost'])
        synth['master'] = master

    return improved_code, synth_config


def generate_improvement_report(gaps: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any] = None) -> str:
    """Generate human-readable improvement report."""
    lines = [
        "=" * 60,
        "AI CODE IMPROVEMENT REPORT",
        "=" * 60,
        "",
        "Issues identified:"
    ]

    for issue in gaps.get('issues', []):
        lines.append(f"  - {issue.replace('_', ' ').title()}")

    lines.append("")
    lines.append("Suggestions:")
    for suggestion in gaps.get('suggestions', []):
        lines.append(f"  - {suggestion}")

    lines.append("")
    lines.append("Parameter adjustments:")
    for param, value in gaps.get('parameters', {}).items():
        if isinstance(value, float):
            lines.append(f"  - {param}: {value:.2f}")
        else:
            lines.append(f"  - {param}: {value}")

    if after:
        lines.append("")
        lines.append("Results:")
        before_comp = before.get('comparison', {})
        after_comp = after.get('comparison', {})
        for metric in ['overall_similarity', 'mfcc_similarity', 'energy_similarity', 'brightness_similarity']:
            b = before_comp.get(metric, 0) * 100
            a = after_comp.get(metric, 0) * 100
            delta = a - b
            sign = '+' if delta > 0 else ''
            lines.append(f"  - {metric.replace('_', ' ').title()}: {b:.1f}% → {a:.1f}% ({sign}{delta:.1f}%)")

    lines.append("=" * 60)
    return "\n".join(lines)


def merge_gaps(basic_gaps: Dict[str, Any], spec_gaps: Dict[str, Any]) -> Dict[str, Any]:
    """Merge gaps from basic comparison and spectrogram analysis."""
    merged = {
        'issues': list(set(basic_gaps.get('issues', []) + spec_gaps.get('issues', []))),
        'suggestions': basic_gaps.get('suggestions', []) + spec_gaps.get('suggestions', []),
        'parameters': {**basic_gaps.get('parameters', {}), **spec_gaps.get('parameters', {})},
        'envelope_changes': spec_gaps.get('envelope_changes', {}),
    }

    # Spectrogram analysis is more precise, so prefer its gain values
    for key in ['bass_gain_mult', 'mid_gain_mult', 'high_gain_mult']:
        if key in spec_gaps.get('parameters', {}):
            merged['parameters'][key] = spec_gaps['parameters'][key]

    return merged


def main():
    parser = argparse.ArgumentParser(description='AI Code Improver')
    parser.add_argument('comparison', help='Path to comparison.json')
    parser.add_argument('strudel', help='Path to Strudel code')
    parser.add_argument('config', help='Path to synth_config.json')
    parser.add_argument('--output-strudel', help='Output path for improved Strudel code')
    parser.add_argument('--output-config', help='Output path for improved config')
    parser.add_argument('--report', action='store_true', help='Print improvement report')
    parser.add_argument('--original-audio', help='Path to original audio for spectrogram analysis')
    parser.add_argument('--rendered-audio', help='Path to rendered audio for spectrogram analysis')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration to analyze')

    args = parser.parse_args()

    # Load inputs
    comparison = load_json(args.comparison)
    code = load_strudel(args.strudel)
    synth_config = load_json(args.config)

    # Basic gap analysis from comparison.json
    basic_gaps = analyze_comparison_gaps(comparison)

    # Enhanced analysis with spectrogram if audio provided
    spec_gaps = {'issues': [], 'suggestions': [], 'parameters': {}}
    if args.original_audio and args.rendered_audio and HAS_SPECTROGRAM:
        print("Performing deep spectrogram analysis...", file=sys.stderr)
        spec_analysis = analyze_with_spectrogram(
            args.original_audio, args.rendered_audio, args.duration
        )
        if spec_analysis:
            spec_gaps = spectrogram_to_gaps(spec_analysis)
            print(f"  Spectrogram similarity: {spec_analysis.get('spectrogram_similarity', 0)*100:.1f}%", file=sys.stderr)
            print(f"  Found {len(spec_gaps.get('issues', []))} issues from spectrogram", file=sys.stderr)

            # Store the full analysis for LLM
            if args.output_config:
                # Save spectrogram insights to a separate file for AI learning
                spec_output = args.output_config.replace('.json', '_spectrogram.json')
                save_json(spec_analysis, spec_output)
                print(f"  Saved spectrogram analysis to: {spec_output}", file=sys.stderr)

    # Merge gaps (spectrogram analysis takes precedence for gain values)
    gaps = merge_gaps(basic_gaps, spec_gaps)

    if args.report:
        print(generate_improvement_report(gaps, comparison))

    if not gaps['issues']:
        print("No significant issues found - code is already good!")
        return

    # Improve code
    improved_code, improved_config = improve_strudel_code(code, gaps, synth_config)

    # Save outputs
    if args.output_strudel:
        save_strudel(improved_code, args.output_strudel)
        print(f"Saved improved Strudel code to: {args.output_strudel}")

    if args.output_config:
        save_json(improved_config, args.output_config)
        print(f"Saved improved config to: {args.output_config}")


if __name__ == '__main__':
    main()
