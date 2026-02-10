#!/usr/bin/env python3
"""
AI Orchestrator - Multi-prompt pipeline for Strudel code generation.

6 focused prompts instead of 1 massive prompt:
1. Sections - analyze energy curve → section boundaries
2. Bass - sound + dynamics for bass voice
3. Mid - sound + dynamics for mid voice
4. High - sound + dynamics for high voice
5. Drums - kit + dynamics for drums
6. Mix - balance all voices together
"""

import json
import sys
import os
import argparse
import requests
from typing import Dict, List, Optional, Any

# Sound catalogs for LLM context
BASS_SOUNDS = """BASS SOUNDS (pick one):
- sawtooth: warm, rich harmonics, aggressive
- supersaw: thick, detuned, modern EDM
- square: hollow, retro, 8-bit
- triangle: soft, mellow, subtle
- gm_synth_bass_1: punchy, electronic, snappy
- gm_synth_bass_2: rubbery, soft, smooth
- gm_electric_bass_finger: natural, funk, groove
- gm_electric_bass_pick: bright, rock, cutting
- gm_acoustic_bass: warm, jazz, organic
- gm_slap_bass_1: funky, percussive, pop
- gm_fretless_bass: smooth, expressive, jazz fusion
"""

MID_SOUNDS = """MID/PAD SOUNDS (pick one):
- gm_pad_1_fantasia: lush, evolving, dreamy
- gm_pad_2_warm: soft, ambient, cozy
- gm_pad_3_polysynth: bright, synthetic, modern
- gm_pad_4_choir: vocal, ethereal, atmospheric
- gm_pad_sweep: moving, filter sweep, EDM
- gm_string_ensemble_1: orchestral, lush, cinematic
- gm_synth_strings_1: synthetic, smooth, pad-like
- gm_electric_piano_1: warm, keys, soulful
- gm_electric_piano_2: bright, digital, modern
- gm_organ_1: full, church, powerful
- triangle: soft, pure, simple
- sine: pure, sub, minimal
"""

HIGH_SOUNDS = """HIGH/LEAD SOUNDS (pick one):
- gm_lead_1_square: bright, cutting, retro
- gm_lead_2_sawtooth: rich, aggressive, lead
- gm_lead_5_charang: distorted, aggressive, rock
- gm_lead_6_voice: vocal, expressive, unique
- gm_synth_lead: modern, bright, EDM
- gm_trumpet: brass, punchy, bold
- gm_alto_sax: jazzy, smooth, expressive
- gm_flute: airy, light, delicate
- gm_clarinet: warm, woody, mellow
- gm_violin: expressive, orchestral, emotional
- square: bright, hollow, chiptune
- triangle: soft, flute-like, gentle
"""

DRUM_KITS = """DRUM KITS (pick one):
- RolandTR808: punchy, hip-hop, deep kick, classic
- RolandTR909: bright, house/techno, punchy, dance
- RolandCR78: vintage, disco, lo-fi, analog
- LinnDrum: crisp, 80s, iconic, pop
- OberheimDMX: punchy, electro, fat, funky
- AkaiLinn: jazzy, organic, natural, fusion
- RolandTR707: digital, clean, bright, pop
- RolandTR606: thin, acid, TB-303 companion
- BossDR110: lo-fi, cheap, charming, quirky
- KorgKR55: vintage, preset rhythms, retro
- KorgMini: small, punchy, portable
- AlesisHR16: digital, 80s, gated, snappy
"""


def call_ollama(prompt: str, model: str = "llama3:8b") -> str:
    """Call Ollama API with a prompt."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return ""


def prompt_sections(bar_energy: List[float], bpm: float) -> Dict:
    """Prompt 1: Analyze energy curve → section boundaries."""

    # Simplify energy to 8 values for prompt
    chunk_size = max(1, len(bar_energy) // 8)
    energy_summary = []
    for i in range(0, len(bar_energy), chunk_size):
        chunk = bar_energy[i:i+chunk_size]
        avg = sum(chunk) / len(chunk) if chunk else 0
        energy_summary.append(round(avg, 2))

    prompt = f"""Analyze this energy curve and identify sections.

ENERGY PER SECTION (0-1, where 1=loudest):
{energy_summary}

BPM: {bpm}
Total bars: {len(bar_energy)}

Identify section boundaries. Output JSON only:
{{
  "sections": [
    {{"name": "intro", "start_bar": 0, "end_bar": 8, "energy": 0.3}},
    {{"name": "verse", "start_bar": 8, "end_bar": 24, "energy": 0.5}},
    {{"name": "drop", "start_bar": 24, "end_bar": 40, "energy": 0.9}},
    {{"name": "outro", "start_bar": 40, "end_bar": 48, "energy": 0.4}}
  ]
}}

Rules:
- Use names: intro, verse, buildup, drop, chorus, bridge, outro
- Energy 0.0-0.3 = quiet, 0.4-0.6 = medium, 0.7-1.0 = loud
- Output ONLY valid JSON, no explanation"""

    response = call_ollama(prompt)

    try:
        # Extract JSON from response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    # Default sections if parsing fails
    total_bars = len(bar_energy)
    return {
        "sections": [
            {"name": "intro", "start_bar": 0, "end_bar": total_bars // 4, "energy": 0.4},
            {"name": "main", "start_bar": total_bars // 4, "end_bar": total_bars * 3 // 4, "energy": 0.8},
            {"name": "outro", "start_bar": total_bars * 3 // 4, "end_bar": total_bars, "energy": 0.4}
        ]
    }


def prompt_voice(
    voice_name: str,
    sound_catalog: str,
    spectrum: Dict,
    sections: List[Dict],
    genre: str
) -> Dict:
    """Prompt 2-4: Generate sound + dynamics for a voice."""

    # Build section summary
    section_summary = ", ".join([
        f"{s['name']}({s['energy']:.1f})" for s in sections
    ])

    prompt = f"""Pick a sound and dynamics for {voice_name.upper()} voice.

{sound_catalog}

GENRE: {genre}
SECTIONS: {section_summary}
SPECTRUM ANALYSIS:
- Brightness: {spectrum.get('brightness', 0.5):.2f} (0=dark, 1=bright)
- Energy: {spectrum.get('energy', 0.5):.2f}

Generate {voice_name} settings. Output JSON only:
{{
  "sound": "sound_name_from_catalog",
  "gain_pattern": "<v1 v2 v3 v4>",
  "gain_slow": 16,
  "lpf": 2000,
  "hpf": 40,
  "room": 0.2
}}

RULES:
- gain_pattern: 4 values matching sections (intro/verse/drop/outro)
- Values must differ by at least 0.2 (e.g., 0.2, 0.4, 0.8, 0.5)
- Low energy section = low gain (0.1-0.3)
- High energy section = high gain (0.6-0.9)
- {voice_name} lpf range: {"300-800" if voice_name == "bass" else "2000-8000" if voice_name == "mid" else "4000-12000"}
- Output ONLY valid JSON"""

    response = call_ollama(prompt)

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    # Defaults per voice
    defaults = {
        "bass": {"sound": "sawtooth", "gain_pattern": "<0.2 0.3 0.6 0.4>", "gain_slow": 16, "lpf": 600, "hpf": 40, "room": 0.1},
        "mid": {"sound": "gm_pad_2_warm", "gain_pattern": "<0.3 0.5 0.9 0.5>", "gain_slow": 16, "lpf": 4000, "hpf": 200, "room": 0.3},
        "high": {"sound": "gm_lead_1_square", "gain_pattern": "<0.1 0.3 0.7 0.4>", "gain_slow": 16, "lpf": 8000, "hpf": 500, "room": 0.2}
    }
    return defaults.get(voice_name, defaults["mid"])


def prompt_drums(
    sections: List[Dict],
    rhythm_density: float,
    genre: str
) -> Dict:
    """Prompt 5: Generate drum kit + dynamics."""

    section_summary = ", ".join([
        f"{s['name']}({s['energy']:.1f})" for s in sections
    ])

    prompt = f"""Pick a drum kit and dynamics.

{DRUM_KITS}

GENRE: {genre}
SECTIONS: {section_summary}
RHYTHM DENSITY: {rhythm_density:.2f} (0=sparse, 1=dense)

Generate drum settings. Output JSON only:
{{
  "kit": "kit_name_from_catalog",
  "gain_pattern": "<v1 v2 v3 v4>",
  "gain_slow": 16,
  "room_pattern": "<v1 v2 v3 v4>",
  "room_slow": 16
}}

RULES:
- gain_pattern: match section energy (quiet intro, loud drop)
- room_pattern: LESS reverb at drop (punchy), MORE at intro (spacey)
- Values must differ by at least 0.2
- Output ONLY valid JSON"""

    response = call_ollama(prompt)

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    return {
        "kit": "RolandTR808",
        "gain_pattern": "<0.5 0.6 1.0 0.7>",
        "gain_slow": 16,
        "room_pattern": "<0.3 0.2 0.1 0.2>",
        "room_slow": 16
    }


def prompt_mix(
    bass_config: Dict,
    mid_config: Dict,
    high_config: Dict,
    drums_config: Dict,
    freq_balance: Dict
) -> Dict:
    """Prompt 6: Balance all voices together."""

    prompt = f"""Adjust mix balance based on frequency analysis.

CURRENT VOICES:
- Bass: {bass_config.get('sound')}, gain {bass_config.get('gain_pattern')}
- Mid: {mid_config.get('sound')}, gain {mid_config.get('gain_pattern')}
- High: {high_config.get('sound')}, gain {high_config.get('gain_pattern')}
- Drums: {drums_config.get('kit')}, gain {drums_config.get('gain_pattern')}

FREQUENCY BALANCE (original vs current):
- Sub-bass: original {freq_balance.get('sub_bass_orig', 0.1):.0%}, current {freq_balance.get('sub_bass_curr', 0.1):.0%}
- Bass: original {freq_balance.get('bass_orig', 0.2):.0%}, current {freq_balance.get('bass_curr', 0.2):.0%}
- Mid: original {freq_balance.get('mid_orig', 0.4):.0%}, current {freq_balance.get('mid_curr', 0.4):.0%}
- High: original {freq_balance.get('high_orig', 0.2):.0%}, current {freq_balance.get('high_curr', 0.2):.0%}

Output gain multipliers to fix balance. JSON only:
{{
  "bass_mult": 1.0,
  "mid_mult": 1.0,
  "high_mult": 1.0,
  "drums_mult": 1.0,
  "master_gain": 0.8
}}

RULES:
- If bass too loud, reduce bass_mult (e.g., 0.7)
- If mid too quiet, increase mid_mult (e.g., 1.3)
- Keep multipliers between 0.5 and 1.5
- Output ONLY valid JSON"""

    response = call_ollama(prompt)

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    return {
        "bass_mult": 1.0,
        "mid_mult": 1.0,
        "high_mult": 1.0,
        "drums_mult": 1.0,
        "master_gain": 0.8
    }


def normalize_pattern(pattern: str) -> str:
    """Convert various LLM output formats to proper Strudel mini-notation."""
    if not pattern:
        return "<0.5>"

    # Already in correct format
    if pattern.startswith("<") and pattern.endswith(">"):
        return pattern

    # Remove quotes if present
    pattern = pattern.strip('"\'')

    # Handle "0.2, 0.5, 0.8" format
    if "," in pattern:
        values = [v.strip() for v in pattern.split(",")]
        return f"<{' '.join(values)}>"

    # Handle "0.2 0.5 0.8" format (space-separated)
    values = pattern.split()
    if len(values) > 1:
        return f"<{' '.join(values)}>"

    # Single value
    return f"<{pattern}>"


def assemble_code(
    bpm: float,
    key: str,
    bass_patterns: List[str],
    mid_patterns: List[str],
    high_patterns: List[str],
    drum_patterns: List[str],
    bass_config: Dict,
    mid_config: Dict,
    high_config: Dict,
    drums_config: Dict,
    mix_config: Dict
) -> str:
    """Assemble all outputs into final Strudel code."""

    cps = bpm / 60 / 4

    # Format patterns
    bass_arr = json.dumps(bass_patterns, indent=2)
    mid_arr = json.dumps(mid_patterns, indent=2)
    high_arr = json.dumps(high_patterns, indent=2)
    drums_arr = json.dumps(drum_patterns, indent=2)

    # Apply mix multipliers to gain patterns
    def apply_mult(pattern: str, mult: float) -> str:
        pattern = normalize_pattern(pattern)
        if mult == 1.0:
            return pattern
        # Parse pattern like "<0.2 0.4 0.8>"
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

    # Get config values with defaults
    bass_sound = bass_config.get('sound', 'sawtooth')
    bass_lpf = bass_config.get('lpf', 600)
    bass_hpf = bass_config.get('hpf', 40)
    bass_room = bass_config.get('room', 0.1)
    bass_slow = bass_config.get('gain_slow', 16)

    mid_sound = mid_config.get('sound', 'triangle')
    mid_lpf = mid_config.get('lpf', 4000)
    mid_hpf = mid_config.get('hpf', 200)
    mid_room = mid_config.get('room', 0.2)
    mid_slow = mid_config.get('gain_slow', 16)

    high_sound = high_config.get('sound', 'square')
    high_lpf = high_config.get('lpf', 8000)
    high_hpf = high_config.get('hpf', 500)
    high_room = high_config.get('room', 0.2)
    high_slow = high_config.get('gain_slow', 16)

    drums_kit = drums_config.get('kit', 'RolandTR808')
    drums_slow = drums_config.get('gain_slow', 16)
    drums_room_slow = drums_config.get('room_slow', 16)

    code = f'''// MIDI-grep output (AI Orchestrated)
// BPM: {bpm}, Key: {key}

setcps({cps:.4f})

let bass = {bass_arr}

let mid = {mid_arr}

let high = {high_arr}

let drums = {drums_arr}

// Effect functions with beat-synced dynamics
let bassFx = p => p.sound("{bass_sound}")
  .gain("{bass_gain}".slow({bass_slow}))
  .lpf({bass_lpf})
  .hpf({bass_hpf})
  .room({bass_room})

let midFx = p => p.sound("{mid_sound}")
  .gain("{mid_gain}".slow({mid_slow}))
  .lpf({mid_lpf})
  .hpf({mid_hpf})
  .room({mid_room})

let highFx = p => p.sound("{high_sound}")
  .gain("{high_gain}".slow({high_slow}))
  .lpf({high_lpf})
  .hpf({high_hpf})
  .room({high_room})

let drumsFx = p => p.bank("{drums_kit}")
  .gain("{drums_gain}".slow({drums_slow}))
  .room("{drums_room}".slow({drums_room_slow}))

$: stack(
  bassFx(cat(...bass.map(b => note(b)))),
  midFx(cat(...mid.map(b => note(b)))),
  highFx(cat(...high.map(b => note(b)))),
  drumsFx(cat(...drums.map(b => s(b))))
)
'''
    return code


def run_orchestrated_pipeline(
    metadata_path: str,
    patterns_path: str,
    comparison_path: str,
    output_path: str
) -> None:
    """Run the full orchestrated pipeline."""

    print("=" * 60)
    print("AI ORCHESTRATOR - Multi-Prompt Pipeline")
    print("=" * 60)

    # Load inputs
    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(patterns_path) as f:
        patterns = json.load(f)

    comparison = {}
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)

    bpm = metadata.get('bpm', 120)
    key = metadata.get('key', 'C major')
    genre = metadata.get('genre', 'electronic')
    bar_energy = metadata.get('bar_energy', [0.5] * 16)

    bass_patterns = patterns.get('bass', ['c2'])
    mid_patterns = patterns.get('mid', ['c4 e4 g4'])
    high_patterns = patterns.get('high', ['c5'])
    drum_patterns = patterns.get('drums', ['bd sd'])

    # Extract spectrum info from comparison
    freq_bands = comparison.get('frequency_bands', {})
    spectrum = {
        'brightness': comparison.get('brightness', {}).get('original', 0.5),
        'energy': comparison.get('energy', {}).get('original', 0.5)
    }
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
    rhythm_density = comparison.get('rhythm', {}).get('density', 0.5)

    # Prompt 1: Sections
    print("\n[1/6] Analyzing sections...")
    sections_result = prompt_sections(bar_energy, bpm)
    sections = sections_result.get('sections', [])
    print(f"  Found {len(sections)} sections: {[s['name'] for s in sections]}")

    # Prompts 2-4: Voices (can run in parallel in future)
    print("\n[2/6] Generating bass voice...")
    bass_config = prompt_voice("bass", BASS_SOUNDS, spectrum, sections, genre)
    print(f"  Sound: {bass_config.get('sound')}, Gain: {bass_config.get('gain_pattern')}")

    print("\n[3/6] Generating mid voice...")
    mid_config = prompt_voice("mid", MID_SOUNDS, spectrum, sections, genre)
    print(f"  Sound: {mid_config.get('sound')}, Gain: {mid_config.get('gain_pattern')}")

    print("\n[4/6] Generating high voice...")
    high_config = prompt_voice("high", HIGH_SOUNDS, spectrum, sections, genre)
    print(f"  Sound: {high_config.get('sound')}, Gain: {high_config.get('gain_pattern')}")

    # Prompt 5: Drums
    print("\n[5/6] Generating drums...")
    drums_config = prompt_drums(sections, rhythm_density, genre)
    print(f"  Kit: {drums_config.get('kit')}, Gain: {drums_config.get('gain_pattern')}")

    # Prompt 6: Mix
    print("\n[6/6] Balancing mix...")
    mix_config = prompt_mix(bass_config, mid_config, high_config, drums_config, freq_balance)
    print(f"  Multipliers: bass={mix_config.get('bass_mult')}, mid={mix_config.get('mid_mult')}, high={mix_config.get('high_mult')}")

    # Assemble final code
    print("\nAssembling final code...")
    code = assemble_code(
        bpm, key,
        bass_patterns, mid_patterns, high_patterns, drum_patterns,
        bass_config, mid_config, high_config, drums_config, mix_config
    )

    with open(output_path, 'w') as f:
        f.write(code)

    print(f"\nSaved: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AI Orchestrator for Strudel generation")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--patterns", required=True, help="Path to patterns.json")
    parser.add_argument("--comparison", default="", help="Path to comparison.json")
    parser.add_argument("--output", required=True, help="Output Strudel file")

    args = parser.parse_args()

    run_orchestrated_pipeline(
        args.metadata,
        args.patterns,
        args.comparison,
        args.output
    )


if __name__ == "__main__":
    main()
