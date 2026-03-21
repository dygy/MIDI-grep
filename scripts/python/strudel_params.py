#!/usr/bin/env python3
"""
Strudel parameter manipulation utilities.

Extracted from ai_improver.py — contains all functions for extracting,
applying, merging, and optimizing Strudel code effect parameters.
"""

import re
import json
import sys
from typing import Optional, Dict, Any, List

# Import orchestrator for multi-prompt generation
try:
    from ai_orchestrator import (
        prompt_sections, prompt_voice, prompt_drums, prompt_mix,
        assemble_code, normalize_pattern,
        BASS_SOUNDS, MID_SOUNDS, HIGH_SOUNDS, DRUM_KITS
    )
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False


def extract_parameters_from_code(code: str) -> Dict[str, Dict[str, Any]]:
    """Extract effect parameters from Strudel code for learning."""
    import re
    params = {}

    # Pattern to match effect functions and their parameters
    fx_pattern = r'let\s+(\w+Fx)\s*=\s*p\s*=>\s*p([^\n]*(?:\n\s+\.[^\n]*)*)'

    for match in re.finditer(fx_pattern, code, re.MULTILINE):
        fx_name = match.group(1)  # e.g., "bassFx"
        fx_body = match.group(2)  # e.g., ".sound("supersaw").gain(0.15)..."

        fx_params = {}

        # Extract .gain(value)
        gain_m = re.search(r'\.gain\(([0-9.]+)\)', fx_body)
        if gain_m:
            fx_params['gain'] = float(gain_m.group(1))

        # Extract .hpf(value)
        hpf_m = re.search(r'\.hpf\(([0-9.]+)\)', fx_body)
        if hpf_m:
            fx_params['hpf'] = float(hpf_m.group(1))

        # Extract .lpf(value)
        lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', fx_body)
        if lpf_m:
            fx_params['lpf'] = float(lpf_m.group(1))

        # Extract .attack(value)
        attack_m = re.search(r'\.attack\(([0-9.]+)\)', fx_body)
        if attack_m:
            fx_params['attack'] = float(attack_m.group(1))

        # Extract .decay(value)
        decay_m = re.search(r'\.decay\(([0-9.]+)\)', fx_body)
        if decay_m:
            fx_params['decay'] = float(decay_m.group(1))

        # Extract .sustain(value)
        sustain_m = re.search(r'\.sustain\(([0-9.]+)\)', fx_body)
        if sustain_m:
            fx_params['sustain'] = float(sustain_m.group(1))

        # Extract .release(value)
        release_m = re.search(r'\.release\(([0-9.]+)\)', fx_body)
        if release_m:
            fx_params['release'] = float(release_m.group(1))

        # Extract .room(value)
        room_m = re.search(r'\.room\(([0-9.]+)\)', fx_body)
        if room_m:
            fx_params['room'] = float(room_m.group(1))

        # Extract .sound("value")
        sound_m = re.search(r'\.sound\("([^"]+)"\)', fx_body)
        if sound_m:
            fx_params['sound'] = sound_m.group(1)

        if fx_params:
            params[fx_name] = fx_params

    return params


def apply_genre_presets(code: str, presets: Dict[str, Dict]) -> str:
    """Apply genre presets to Strudel code as starting point."""
    import re

    if not presets:
        return code

    result = code

    for fx_name, params in presets.items():
        for param, value in params.items():
            # Find and replace parameter in the effect function
            # Pattern: .param(old_value) -> .param(new_value)
            pattern = rf'(let\s+{fx_name}\s*=.*?\.{param}\()([0-9.]+)(\))'

            def replacer(m):
                return f"{m.group(1)}{value}{m.group(3)}"

            result = re.sub(pattern, replacer, result, flags=re.DOTALL)

    return result


def extract_effect_functions(code: str) -> str:
    """Extract just the effect functions from Strudel code for concise prompts."""
    import re
    # Find all let *Fx = ... patterns (multiline)
    fx_pattern = r'let\s+(\w+Fx)\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
    matches = re.findall(fx_pattern, code, re.MULTILINE)
    if matches:
        # Return the full function definitions
        full_matches = re.findall(r'let\s+\w+Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*', code, re.MULTILINE)
        return '\n'.join(full_matches)
    # Fallback: return first 2000 chars if no pattern matches
    return code[:2000] + ('...' if len(code) > 2000 else '')


def merge_effect_functions(original_code: str, improved_effects: str) -> str:
    """Merge improved effect functions back into the original code.

    Also handles setcps() tempo control at the top of the code.
    """
    import re

    result = original_code

    # Handle setcps() tempo control
    setcps_match = re.search(r'setcps\([^)]+\)', improved_effects)
    if setcps_match:
        new_setcps = setcps_match.group()
        # Check if original has setcps
        if re.search(r'setcps\([^)]+\)', result):
            # Replace existing setcps
            result = re.sub(r'setcps\([^)]+\)', new_setcps, result)
        else:
            # Add setcps at the beginning (after any comments)
            lines = result.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('//') and line.strip():
                    insert_idx = i
                    break
            lines.insert(insert_idx, new_setcps)
            result = '\n'.join(lines)

    # Parse improved effects into a dict
    improved_dict = {}
    fx_pattern = r'let\s+(\w+Fx)\s*=\s*(p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*)'
    for match in re.finditer(fx_pattern, improved_effects, re.MULTILINE):
        name, body = match.groups()
        improved_dict[name] = f'let {name} = {body}'

    if not improved_dict:
        return result

    # Replace each effect function in the original
    for name, new_def in improved_dict.items():
        # Pattern to match the full function definition
        old_pattern = rf'let\s+{name}\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
        # Escape backslashes in the replacement to avoid regex errors
        safe_new_def = new_def.replace('\\', '\\\\')
        try:
            result = re.sub(old_pattern, safe_new_def, result, flags=re.MULTILINE)
        except re.error as e:
            print(f"Warning: regex error merging {name}: {e}")
            continue

    return result


def generate_orchestrated_effects(
    code: str,
    metadata: Dict,
    comparison: Optional[Dict] = None
) -> str:
    """
    Use the orchestrator to generate dynamic effects with beat-synced patterns.
    Replaces the effect functions in the code with properly dynamic ones.
    """
    if not HAS_ORCHESTRATOR:
        print("  Warning: Orchestrator not available, skipping dynamic effects")
        return code

    import re

    # Extract bar_energy from code
    bar_energy_match = re.search(r'let bar_energy = \[([\d., ]+)\]', code)
    bar_energy = [0.5] * 16
    if bar_energy_match:
        try:
            bar_energy = [float(x.strip()) for x in bar_energy_match.group(1).split(',')]
        except:
            pass

    bpm = metadata.get('bpm', 120)
    genre = metadata.get('genre', 'electronic')

    # Extract spectrum info from comparison
    spectrum = {'brightness': 0.5, 'energy': 0.7}
    freq_balance = {}
    if comparison:
        freq_bands = comparison.get('frequency_bands', {})
        freq_balance = {
            'sub_bass_orig': freq_bands.get('sub_bass', {}).get('original', 0.1),
            'sub_bass_curr': freq_bands.get('sub_bass', {}).get('rendered', 0.1),
            'bass_orig': freq_bands.get('bass', {}).get('original', 0.2),
            'bass_curr': freq_bands.get('bass', {}).get('rendered', 0.2),
            'mid_orig': freq_bands.get('mid', {}).get('original', 0.4),
            'mid_curr': freq_bands.get('mid', {}).get('rendered', 0.4),
            'high_orig': freq_bands.get('high', {}).get('original', 0.2),
            'high_curr': freq_bands.get('high', {}).get('rendered', 0.2),
        }
        spectrum['brightness'] = comparison.get('brightness', {}).get('original', 0.5)
        spectrum['energy'] = comparison.get('energy', {}).get('original', 0.7)

    print("\n  [Orchestrator] Generating dynamic effects with beat-synced patterns...")

    # Step 1: Analyze sections
    print("    [1/6] Analyzing sections...")
    sections_result = prompt_sections(bar_energy, bpm)
    sections = sections_result.get('sections', [
        {"name": "intro", "start_bar": 0, "end_bar": 8, "energy": 0.4},
        {"name": "main", "start_bar": 8, "end_bar": 24, "energy": 0.8},
        {"name": "outro", "start_bar": 24, "end_bar": 32, "energy": 0.4}
    ])
    print(f"      Found {len(sections)} sections: {[s['name'] for s in sections]}")

    # Step 2-4: Generate voice configs
    print("    [2/6] Generating bass...")
    bass_config = prompt_voice("bass", BASS_SOUNDS, spectrum, sections, genre)
    print(f"      Sound: {bass_config.get('sound')}, Gain: {bass_config.get('gain_pattern')}")

    print("    [3/6] Generating mid...")
    mid_config = prompt_voice("mid", MID_SOUNDS, spectrum, sections, genre)
    print(f"      Sound: {mid_config.get('sound')}, Gain: {mid_config.get('gain_pattern')}")

    print("    [4/6] Generating high...")
    high_config = prompt_voice("high", HIGH_SOUNDS, spectrum, sections, genre)
    print(f"      Sound: {high_config.get('sound')}, Gain: {high_config.get('gain_pattern')}")

    # Step 5: Drums
    print("    [5/6] Generating drums...")
    rhythm_density = comparison.get('rhythm', {}).get('density', 0.6) if comparison else 0.6
    drums_config = prompt_drums(sections, rhythm_density, genre)
    print(f"      Kit: {drums_config.get('kit')}, Gain: {drums_config.get('gain_pattern')}")

    # Step 6: Mix balance
    print("    [6/6] Balancing mix...")
    mix_config = prompt_mix(bass_config, mid_config, high_config, drums_config, freq_balance)
    print(f"      Multipliers: bass={mix_config.get('bass_mult')}, mid={mix_config.get('mid_mult')}")

    # Apply mix multipliers and normalize patterns
    def apply_mult(pattern, mult):
        pattern = normalize_pattern(pattern)
        if mult == 1.0:
            return pattern
        if pattern.startswith("<") and pattern.endswith(">"):
            values = pattern[1:-1].split()
            new_values = [f"{float(v) * mult:.2f}" for v in values]
            return f"<{' '.join(new_values)}>"
        return pattern

    bass_gain = apply_mult(bass_config.get('gain_pattern', '<0.3 0.5 0.7 0.5>'), mix_config.get('bass_mult', 1.0))
    mid_gain = apply_mult(mid_config.get('gain_pattern', '<0.3 0.5 0.9 0.5>'), mix_config.get('mid_mult', 1.0))
    high_gain = apply_mult(high_config.get('gain_pattern', '<0.2 0.4 0.7 0.4>'), mix_config.get('high_mult', 1.0))
    drums_gain = apply_mult(drums_config.get('gain_pattern', '<0.5 0.7 1.0 0.7>'), mix_config.get('drums_mult', 1.0))
    drums_room = normalize_pattern(drums_config.get('room_pattern', '<0.2 0.15 0.1 0.15>'))

    # Build new effect functions
    new_effects = f'''// Effects (applied at playback) - AI Orchestrated with beat-synced dynamics
let bassFx = p => p.sound("{bass_config.get('sound', 'sawtooth')}")
    .gain("{bass_gain}".slow({bass_config.get('gain_slow', 16)}))
    .lpf({bass_config.get('lpf', 600)})
    .hpf({bass_config.get('hpf', 40)})
    .room({bass_config.get('room', 0.1)})

let midFx = p => p.sound("{mid_config.get('sound', 'triangle')}")
    .gain("{mid_gain}".slow({mid_config.get('gain_slow', 16)}))
    .lpf({mid_config.get('lpf', 4000)})
    .hpf({mid_config.get('hpf', 200)})
    .room({mid_config.get('room', 0.2)})

let highFx = p => p.sound("{high_config.get('sound', 'square')}")
    .gain("{high_gain}".slow({high_config.get('gain_slow', 16)}))
    .lpf({high_config.get('lpf', 8000)})
    .hpf({high_config.get('hpf', 500)})
    .room({high_config.get('room', 0.2)})

let drumsFx = p => p.bank("{drums_config.get('kit', 'RolandTR808')}")
    .gain("{drums_gain}".slow({drums_config.get('gain_slow', 16)}))
    .room("{drums_room}".slow({drums_config.get('room_slow', 16)}))'''

    # Replace effect functions in code
    # Find the effects section
    effects_pattern = r'// Effects.*?let drumsFx = p => p\.bank\([^)]+\)[^\n]*(?:\n[^\n]*)*?(?=\n\n|\n// Play|$)'
    new_code = re.sub(effects_pattern, new_effects, code, flags=re.DOTALL)

    # If replacement didn't work, try simpler pattern
    if new_code == code:
        # Replace individual effect functions
        new_code = re.sub(r'let bassFx = p => p[^\n]+(?:\n    \.[^\n]+)*',
                         f'''let bassFx = p => p.sound("{bass_config.get('sound', 'sawtooth')}")
    .gain("{bass_gain}".slow({bass_config.get('gain_slow', 16)}))
    .lpf({bass_config.get('lpf', 600)})
    .hpf({bass_config.get('hpf', 40)})
    .room({bass_config.get('room', 0.1)})''', code)
        new_code = re.sub(r'let midFx = p => p[^\n]+(?:\n    \.[^\n]+)*',
                         f'''let midFx = p => p.sound("{mid_config.get('sound', 'triangle')}")
    .gain("{mid_gain}".slow({mid_config.get('gain_slow', 16)}))
    .lpf({mid_config.get('lpf', 4000)})
    .hpf({mid_config.get('hpf', 200)})
    .room({mid_config.get('room', 0.2)})''', new_code)
        new_code = re.sub(r'let highFx = p => p[^\n]+(?:\n    \.[^\n]+)*',
                         f'''let highFx = p => p.sound("{high_config.get('sound', 'square')}")
    .gain("{high_gain}".slow({high_config.get('gain_slow', 16)}))
    .lpf({high_config.get('lpf', 8000)})
    .hpf({high_config.get('hpf', 500)})
    .room({high_config.get('room', 0.2)})''', new_code)
        new_code = re.sub(r'let drumsFx = p => p\.bank\([^\n]+',
                         f'''let drumsFx = p => p.bank("{drums_config.get('kit', 'RolandTR808')}")
    .gain("{drums_gain}".slow({drums_config.get('gain_slow', 16)}))
    .room("{drums_room}".slow({drums_config.get('room_slow', 16)}))''', new_code)

    print("  [Orchestrator] Done - effects now have beat-synced dynamics")
    return new_code


def _scale_fx_gain(fx_text: str, multiplier: float) -> str:
    """Scale gain value(s) in an Fx function definition by a multiplier.

    Handles three gain patterns:
    1. .gain(0.6) - simple numeric
    2. .gain(perlin.range(0.63, 0.77).slow(8)) - perlin range
    3. .gain("<0.3 0.5 0.8 0.5>".slow(16)) - string pattern
    """
    # Pattern 1: perlin.range(low, high)
    def scale_perlin(m):
        low = round(min(1.5, max(0.01, float(m.group(1)) * multiplier)), 2)
        high = round(min(1.5, max(0.01, float(m.group(2)) * multiplier)), 2)
        return f'.gain(perlin.range({low}, {high})'

    result = re.sub(r'\.gain\(perlin\.range\(([0-9.]+),\s*([0-9.]+)\)', scale_perlin, fx_text)
    if result != fx_text:
        return result

    # Pattern 2: string pattern "<values>"
    def scale_string_gain(m):
        prefix = m.group(1)
        text = m.group(2)
        suffix = m.group(3)
        def scale_val(vm):
            v = round(min(1.5, max(0.01, float(vm.group()) * multiplier)), 2)
            return str(v)
        new_text = re.sub(r'[0-9]+\.?[0-9]*', scale_val, text)
        return f'{prefix}{new_text}{suffix}'

    result = re.sub(r'(\.gain\(")([<>0-9. ]+)(")', scale_string_gain, fx_text)
    if result != fx_text:
        return result

    # Pattern 3: simple .gain(N)
    def scale_simple(m):
        v = round(min(1.5, max(0.01, float(m.group(1)) * multiplier)), 3)
        return f'.gain({v})'

    return re.sub(r'\.gain\(([0-9.]+)\)', scale_simple, fx_text)


def _set_fx_numeric_param(fx_text: str, param: str, new_value) -> str:
    """Set a simple numeric parameter value in an Fx function definition."""
    pattern = rf'\.{param}\(([0-9.]+)\)'
    match = re.search(pattern, fx_text)
    if match:
        return fx_text[:match.start(1)] + str(new_value) + fx_text[match.end(1):]
    return fx_text


def _classify_arrange_line(line: str) -> Optional[str]:
    """Classify an arrange() entry line as 'bass', 'drums', or 'melodic'.

    Returns None if line can't be classified.
    """
    stripped = line.strip()
    # Drums: uses s("...") pattern (not note())
    if re.search(r'\bs\s*\(', stripped) and not re.search(r'\bnote\s*\(', stripped):
        return 'drums'
    # Bass: note patterns with octave 1-2
    if re.search(r'note\s*\(\s*"[^"]*[a-gA-G]#?[12]\b', stripped):
        return 'bass'
    # Melodic: note patterns with octave 3-7
    if re.search(r'note\s*\(\s*"[^"]*[a-gA-G]#?[3-7]\b', stripped):
        return 'melodic'
    return None


def _optimize_arrange_parameters(code: str, comparison: dict, stem_comparison: dict = None) -> tuple:
    """Optimize parameters in arrange() format code (inline .gain/.lpf/.hpf/.room).

    Same 6 optimization rules as optimize_parameters(), but operates on arrange() entries
    where effects are inline rather than in separate Fx definitions.

    Returns (optimized_code, list_of_change_descriptions).
    """
    # Extract metrics from comparison
    orig = comparison.get("original", {})
    rend = comparison.get("rendered", {})

    orig_rms = orig.get("spectral", {}).get("rms_mean", 0.1)
    rend_rms = rend.get("spectral", {}).get("rms_mean", 0.1)
    energy_ratio = rend_rms / max(orig_rms, 0.001)

    orig_centroid = orig.get("spectral", {}).get("centroid_mean", 1)
    rend_centroid = rend.get("spectral", {}).get("centroid_mean", 1)
    brightness_ratio = rend_centroid / max(orig_centroid, 1) if orig_centroid else 1.0

    orig_bands = orig.get("bands", {})
    rend_bands = rend.get("bands", {})

    changes = []
    result = code

    # Classify each line in the code by voice type
    lines = result.split('\n')
    voice_lines = {}  # {line_index: 'bass'|'drums'|'melodic'}
    for i, line in enumerate(lines):
        classification = _classify_arrange_line(line)
        if classification:
            voice_lines[i] = classification

    stem_voice_map = {'bass': 'bass', 'drums': 'drums', 'melodic': 'melodic'}
    modified_voices = set()

    # --- 1. PER-STEM ENERGY FIX ---
    if stem_comparison:
        stems_data = stem_comparison.get("stems", {})
        for stem_name in ['bass', 'drums', 'melodic']:
            stem_info = stems_data.get(stem_name, {})
            if not stem_info:
                continue
            s_orig_rms = stem_info.get("original", {}).get("spectral", {}).get("rms_mean", 0)
            s_rend_rms = stem_info.get("rendered", {}).get("spectral", {}).get("rms_mean", 0)
            if s_orig_rms < 0.005:
                continue
            stem_energy = s_rend_rms / s_orig_rms
            if stem_energy < 0.6 or stem_energy > 1.5:
                mult = 1 + (1 / stem_energy - 1) * 0.5
                mult = max(0.5, min(2.0, mult))
                for line_idx, voice_type in voice_lines.items():
                    if voice_type == stem_name:
                        old_line = lines[line_idx]
                        new_line = _scale_fx_gain(old_line, mult)
                        if new_line != old_line:
                            lines[line_idx] = new_line
                            modified_voices.add(stem_name)
                            changes.append(f"arrange {stem_name} gain *= {mult:.2f} ({stem_name} energy {stem_energy:.0%})")

    # --- 2. OVERALL ENERGY FIX (for voices not adjusted by stem) ---
    if energy_ratio < 0.7 or energy_ratio > 1.4:
        mult = 1 + (1 / energy_ratio - 1) * 0.5
        mult = max(0.5, min(2.0, mult))
        for line_idx, voice_type in voice_lines.items():
            if voice_type not in modified_voices:
                old_line = lines[line_idx]
                new_line = _scale_fx_gain(old_line, mult)
                if new_line != old_line:
                    lines[line_idx] = new_line
                    modified_voices.add(voice_type)
                    changes.append(f"arrange {voice_type} gain *= {mult:.2f} (overall energy {energy_ratio:.0%})")

    # --- 3. BRIGHTNESS FIX (adjust LPF on melodic voices) ---
    if brightness_ratio > 1.2 or brightness_ratio < 0.8:
        lpf_factor = orig_centroid / max(rend_centroid, 1)
        lpf_factor = 1 + (lpf_factor - 1) * 0.5
        for line_idx, voice_type in voice_lines.items():
            if voice_type == 'melodic':
                old_line = lines[line_idx]
                lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', old_line)
                if lpf_m:
                    old_lpf = float(lpf_m.group(1))
                    new_lpf = round(max(200, min(16000, old_lpf * lpf_factor)))
                    if abs(new_lpf - old_lpf) > 50:
                        lines[line_idx] = _set_fx_numeric_param(old_line, 'lpf', new_lpf)
                        changes.append(f"arrange melodic.lpf {old_lpf:.0f} -> {new_lpf} (brightness {brightness_ratio:.0%})")

    # --- 4. REVERB CUT (if too quiet, reverb eats energy) ---
    if energy_ratio < 0.7:
        for line_idx, voice_type in voice_lines.items():
            old_line = lines[line_idx]
            room_m = re.search(r'\.room\(([0-9.]+)\)', old_line)
            if room_m:
                old_room = float(room_m.group(1))
                if old_room > 0.05:
                    new_room = round(max(0.02, old_room * 0.7), 3)
                    if abs(new_room - old_room) > 0.01:
                        lines[line_idx] = _set_fx_numeric_param(old_line, 'room', new_room)
                        changes.append(f"arrange {voice_type}.room {old_room} -> {new_room} (preserve energy)")

    # --- 5. SUB-BASS FIX (adjust bass HPF) ---
    sub_orig = orig_bands.get("sub_bass", 0)
    sub_rend = rend_bands.get("sub_bass", 0)
    sub_diff_pct = (sub_rend - sub_orig) * 100
    if abs(sub_diff_pct) > 10:
        for line_idx, voice_type in voice_lines.items():
            if voice_type == 'bass':
                old_line = lines[line_idx]
                hpf_m = re.search(r'\.hpf\(([0-9.]+)\)', old_line)
                if hpf_m:
                    old_hpf = float(hpf_m.group(1))
                    if sub_diff_pct < -10:
                        new_hpf = round(max(15, old_hpf * 0.7))
                    else:
                        new_hpf = round(min(200, old_hpf * 1.3))
                    if abs(new_hpf - old_hpf) > 3:
                        lines[line_idx] = _set_fx_numeric_param(old_line, 'hpf', new_hpf)
                        changes.append(f"arrange bass.hpf {old_hpf:.0f} -> {new_hpf} (sub-bass {sub_diff_pct:+.0f}%)")

    # --- 6. HIGH FREQUENCY FIX (reduce LPF on bright voices) ---
    high_orig = orig_bands.get("high", 0)
    high_rend = rend_bands.get("high", 0)
    high_diff_pct = (high_rend - high_orig) * 100
    if high_diff_pct > 10:
        for line_idx, voice_type in voice_lines.items():
            if voice_type in ('melodic', 'drums'):
                old_line = lines[line_idx]
                lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', old_line)
                if lpf_m:
                    old_lpf = float(lpf_m.group(1))
                    new_lpf = round(max(2000, old_lpf * 0.8))
                    if abs(new_lpf - old_lpf) > 100:
                        lines[line_idx] = _set_fx_numeric_param(old_line, 'lpf', new_lpf)
                        changes.append(f"arrange {voice_type}.lpf {old_lpf:.0f} -> {new_lpf} (high freq +{high_diff_pct:.0f}%)")

    if changes:
        result = '\n'.join(lines)

    return result, changes


def optimize_parameters(code: str, comparison: dict, stem_comparison: dict = None) -> tuple:
    """Deterministic parameter optimizer. Adjusts gain/lpf/hpf/room based on comparison data.

    No LLM call needed - pure math with 50% dampening to prevent oscillation.
    Only modifies effect function parameters, never notes/sounds/structure.

    Returns (optimized_code, list_of_change_descriptions).
    """
    fx_pattern = r'let\s+(\w+Fx)\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
    fx_matches = list(re.finditer(fx_pattern, code, re.MULTILINE))

    if not fx_matches:
        # Fallback: try arrange() format where effects are inline
        if 'arrange(' in code:
            return _optimize_arrange_parameters(code, comparison, stem_comparison)
        return code, []

    # Extract metrics from comparison
    orig = comparison.get("original", {})
    rend = comparison.get("rendered", {})
    comp = comparison.get("comparison", {})

    orig_rms = orig.get("spectral", {}).get("rms_mean", 0.1)
    rend_rms = rend.get("spectral", {}).get("rms_mean", 0.1)
    energy_ratio = rend_rms / max(orig_rms, 0.001)

    orig_centroid = orig.get("spectral", {}).get("centroid_mean", 1)
    rend_centroid = rend.get("spectral", {}).get("centroid_mean", 1)
    brightness_ratio = rend_centroid / max(orig_centroid, 1) if orig_centroid else 1.0

    orig_bands = orig.get("bands", {})
    rend_bands = rend.get("bands", {})

    changes = []
    result = code

    # Voice classification
    bass_voices = {'bassFx', 'kickFx'}
    drum_voices = {'drumsFx', 'snareFx', 'hhFx'}
    melodic_voices = {'midFx', 'highFx', 'voxFx', 'stabFx', 'leadFx'}
    all_voices = bass_voices | drum_voices | melodic_voices

    modified_gains = set()

    # --- 1. PER-STEM ENERGY FIX (most targeted) ---
    stem_voice_map = {
        'bass': bass_voices,
        'drums': drum_voices,
        'melodic': melodic_voices,
    }

    if stem_comparison:
        stems_data = stem_comparison.get("stems", {})
        for stem_name, voice_names in stem_voice_map.items():
            stem_info = stems_data.get(stem_name, {})
            if not stem_info:
                continue
            s_orig_rms = stem_info.get("original", {}).get("spectral", {}).get("rms_mean", 0)
            s_rend_rms = stem_info.get("rendered", {}).get("spectral", {}).get("rms_mean", 0)

            if s_orig_rms < 0.005:
                continue  # near-silent stem, skip

            stem_energy = s_rend_rms / s_orig_rms
            if stem_energy < 0.6 or stem_energy > 1.5:
                mult = 1 + (1 / stem_energy - 1) * 0.5  # 50% dampened
                mult = max(0.5, min(2.0, mult))

                fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
                for match in fx_matches:
                    fx_name = match.group(1)
                    if fx_name in voice_names:
                        fx_text = match.group()
                        new_fx = _scale_fx_gain(fx_text, mult)
                        if new_fx != fx_text:
                            result = result.replace(fx_text, new_fx, 1)
                            modified_gains.add(fx_name)
                            changes.append(f"{fx_name} gain *= {mult:.2f} ({stem_name} energy {stem_energy:.0%})")

    # --- 2. OVERALL ENERGY FIX (for voices not adjusted by stem) ---
    if energy_ratio < 0.7 or energy_ratio > 1.4:
        mult = 1 + (1 / energy_ratio - 1) * 0.5
        mult = max(0.5, min(2.0, mult))

        fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
        for match in fx_matches:
            fx_name = match.group(1)
            if fx_name not in modified_gains and fx_name in all_voices:
                fx_text = match.group()
                new_fx = _scale_fx_gain(fx_text, mult)
                if new_fx != fx_text:
                    result = result.replace(fx_text, new_fx, 1)
                    modified_gains.add(fx_name)
                    changes.append(f"{fx_name} gain *= {mult:.2f} (overall energy {energy_ratio:.0%})")

    # --- 3. BRIGHTNESS FIX (adjust LPF on melodic voices) ---
    if brightness_ratio > 1.2 or brightness_ratio < 0.8:
        lpf_factor = orig_centroid / max(rend_centroid, 1)
        lpf_factor = 1 + (lpf_factor - 1) * 0.5  # dampened

        fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
        for match in fx_matches:
            fx_name = match.group(1)
            if fx_name in melodic_voices:
                fx_text = match.group()
                lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', fx_text)
                if lpf_m:
                    old_lpf = float(lpf_m.group(1))
                    new_lpf = round(max(200, min(16000, old_lpf * lpf_factor)))
                    if abs(new_lpf - old_lpf) > 50:
                        new_fx = _set_fx_numeric_param(fx_text, 'lpf', new_lpf)
                        result = result.replace(fx_text, new_fx, 1)
                        changes.append(f"{fx_name}.lpf {old_lpf:.0f} -> {new_lpf} (brightness {brightness_ratio:.0%})")

    # --- 4. REVERB CUT (if too quiet, reverb eats energy) ---
    if energy_ratio < 0.7:
        fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
        for match in fx_matches:
            fx_name = match.group(1)
            fx_text = match.group()
            room_m = re.search(r'\.room\(([0-9.]+)\)', fx_text)
            if room_m:
                old_room = float(room_m.group(1))
                if old_room > 0.05:
                    new_room = round(max(0.02, old_room * 0.7), 3)
                    if abs(new_room - old_room) > 0.01:
                        new_fx = _set_fx_numeric_param(fx_text, 'room', new_room)
                        result = result.replace(fx_text, new_fx, 1)
                        changes.append(f"{fx_name}.room {old_room} -> {new_room} (preserve energy)")

    # --- 5. SUB-BASS FIX (adjust bass HPF) ---
    sub_orig = orig_bands.get("sub_bass", 0)
    sub_rend = rend_bands.get("sub_bass", 0)
    sub_diff_pct = (sub_rend - sub_orig) * 100

    if abs(sub_diff_pct) > 10:
        fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
        for match in fx_matches:
            fx_name = match.group(1)
            if fx_name in bass_voices:
                fx_text = match.group()
                hpf_m = re.search(r'\.hpf\(([0-9.]+)\)', fx_text)
                if hpf_m:
                    old_hpf = float(hpf_m.group(1))
                    if sub_diff_pct < -10:
                        new_hpf = round(max(15, old_hpf * 0.7))
                    else:
                        new_hpf = round(min(200, old_hpf * 1.3))
                    if abs(new_hpf - old_hpf) > 3:
                        new_fx = _set_fx_numeric_param(fx_text, 'hpf', new_hpf)
                        result = result.replace(fx_text, new_fx, 1)
                        changes.append(f"{fx_name}.hpf {old_hpf:.0f} -> {new_hpf} (sub-bass {sub_diff_pct:+.0f}%)")

    # --- 6. HIGH FREQUENCY FIX (reduce LPF on bright voices) ---
    high_orig = orig_bands.get("high", 0)
    high_rend = rend_bands.get("high", 0)
    high_diff_pct = (high_rend - high_orig) * 100

    if high_diff_pct > 10:
        fx_matches = list(re.finditer(fx_pattern, result, re.MULTILINE))
        for match in fx_matches:
            fx_name = match.group(1)
            if fx_name in {'hhFx', 'highFx'}:
                fx_text = match.group()
                lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', fx_text)
                if lpf_m:
                    old_lpf = float(lpf_m.group(1))
                    new_lpf = round(max(2000, old_lpf * 0.8))
                    if abs(new_lpf - old_lpf) > 100:
                        new_fx = _set_fx_numeric_param(fx_text, 'lpf', new_lpf)
                        result = result.replace(fx_text, new_fx, 1)
                        changes.append(f"{fx_name}.lpf {old_lpf:.0f} -> {new_lpf} (high freq +{high_diff_pct:.0f}%)")

    return result, changes
