#!/usr/bin/env python3
"""
AI-Driven Strudel Code Improver

Uses Claude to analyze comparison results and generate improved Strudel code.
Stores all runs in ClickHouse for incremental learning.
"""

import argparse
import json
import os
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

# ClickHouse connection
CLICKHOUSE_BIN = Path(__file__).parent.parent.parent / "bin" / "clickhouse"
CLICKHOUSE_DB = Path(__file__).parent.parent.parent / ".clickhouse" / "db"

# LLM APIs
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import our smarter gap analysis
try:
    from ai_code_improver import analyze_comparison_gaps, improve_strudel_code
    HAS_CODE_IMPROVER = True
except ImportError:
    HAS_CODE_IMPROVER = False

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "tinyllama:latest"


def get_audio_duration(audio_path: str) -> float:
    """Get exact audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    # Fallback: use librosa
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        return len(y) / sr
    except Exception:
        pass
    return 60.0  # Default fallback


def get_track_hash(audio_path: str) -> str:
    """Generate a hash for the audio file."""
    with open(audio_path, 'rb') as f:
        # Read first 1MB for hash (faster than full file)
        content = f.read(1024 * 1024)
    return hashlib.sha256(content).hexdigest()[:16]


def clickhouse_query(query: str, format: str = "JSONEachRow") -> List[Dict]:
    """Execute a ClickHouse query and return results."""
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query,
        "--format", format
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ClickHouse error: {result.stderr}", file=sys.stderr)
        return []

    if not result.stdout.strip():
        return []

    rows = []
    for line in result.stdout.strip().split('\n'):
        if line:
            rows.append(json.loads(line))
    return rows


def clickhouse_insert(table: str, data: Dict[str, Any]):
    """Insert a row into ClickHouse."""
    # Escape string values
    values = []
    for k, v in data.items():
        if v is None:
            values.append(f"{k} = NULL")
        elif isinstance(v, str):
            escaped = v.replace("'", "\\'").replace("\\", "\\\\")
            values.append(f"{k} = '{escaped}'")
        elif isinstance(v, bool):
            values.append(f"{k} = {1 if v else 0}")
        elif isinstance(v, (int, float)):
            values.append(f"{k} = {v}")
        else:
            values.append(f"{k} = '{json.dumps(v)}'")

    query = f"INSERT INTO {table} SETTINGS input_format_skip_unknown_fields=1 FORMAT Values ({', '.join(values)})"

    # Use INSERT with columns approach instead
    cols = list(data.keys())
    vals = []
    for k in cols:
        v = data[k]
        if v is None:
            vals.append("NULL")
        elif isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace("'", "\\'")
            vals.append(f"'{escaped}'")
        elif isinstance(v, bool):
            vals.append(str(1 if v else 0))
        elif isinstance(v, (int, float)):
            vals.append(str(v))
        else:
            escaped = json.dumps(v).replace("\\", "\\\\").replace("'", "\\'")
            vals.append(f"'{escaped}'")

    query = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals)})"

    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ClickHouse insert error: {result.stderr}", file=sys.stderr)
        return False
    return True


def get_previous_run(track_hash: str) -> Optional[Dict]:
    """Get the most recent run for a track."""
    query = f"""
        SELECT *
        FROM midi_grep.runs
        WHERE track_hash = '{track_hash}'
        ORDER BY version DESC
        LIMIT 1
    """
    rows = clickhouse_query(query)
    return rows[0] if rows else None


def get_best_run(track_hash: str) -> Optional[Dict]:
    """Get the best-scoring run for a track."""
    query = f"""
        SELECT *
        FROM midi_grep.runs
        WHERE track_hash = '{track_hash}'
        ORDER BY similarity_overall DESC
        LIMIT 1
    """
    rows = clickhouse_query(query)
    return rows[0] if rows else None


def get_learned_knowledge(genre: str, bpm: float, key_type: str) -> List[Dict]:
    """Get relevant learned knowledge for this track context.

    Priority:
    1. Exact genre match with BPM in range
    2. Any genre with standard Fx patterns (bassFx, midFx, etc.)
    3. Universal learnings (empty genre)
    """
    # First try exact genre match
    query = f"""
        SELECT parameter_name, parameter_new_value, similarity_improvement, confidence, genre
        FROM midi_grep.knowledge
        WHERE (
            (genre = '{genre}' AND bpm_range_low <= {bpm} AND bpm_range_high >= {bpm})
            OR (parameter_name LIKE 'bassFx%' OR parameter_name LIKE 'midFx%'
                OR parameter_name LIKE 'highFx%' OR parameter_name LIKE 'drumsFx%')
            OR genre = ''
        )
          AND confidence > 0.5
          AND similarity_improvement > 0.1
        ORDER BY similarity_improvement DESC
        LIMIT 20
    """
    results = clickhouse_query(query)

    # Log what we found
    if results:
        genres_found = set(r.get('genre', '') for r in results)
        print(f"       Found {len(results)} knowledge items from genres: {genres_found}")

    return results


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


def learn_from_improvement(
    genre: str,
    bpm: float,
    key_type: str,
    old_code: str,
    new_code: str,
    old_similarity: float,
    new_similarity: float
) -> int:
    """
    Compare before/after code and store successful parameter changes as knowledge.
    Returns number of knowledge entries stored.
    """
    if new_similarity <= old_similarity:
        return 0  # No improvement, nothing to learn

    improvement = new_similarity - old_similarity
    if improvement < 0.01:  # Less than 1% improvement, skip
        return 0

    old_params = extract_parameters_from_code(old_code)
    new_params = extract_parameters_from_code(new_code)

    entries_stored = 0

    for fx_name in new_params:
        if fx_name not in old_params:
            continue

        for param_name in new_params[fx_name]:
            if param_name not in old_params[fx_name]:
                continue

            old_val = old_params[fx_name][param_name]
            new_val = new_params[fx_name][param_name]

            # Skip if values are the same
            if old_val == new_val:
                continue

            # Store the learned improvement
            full_param_name = f"{fx_name}.{param_name}"
            success = store_knowledge(
                genre=genre,
                bpm=bpm,
                key_type=key_type,
                parameter_name=full_param_name,
                old_value=str(old_val),
                new_value=str(new_val),
                improvement=improvement
            )
            if success:
                entries_stored += 1
                print(f"       📚 Learned: {full_param_name}: {old_val} → {new_val} (+{improvement*100:.1f}%)")

    return entries_stored


def normalize_artist(name: str) -> str:
    """Normalize artist name for consistent matching."""
    import re
    normalized = name.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    return normalized


def get_artist_presets(artist: str) -> Dict[str, Dict]:
    """
    Get artist-specific parameter presets from learned knowledge.
    Uses the artist_knowledge table populated by learn_artist.py.
    """
    if not artist:
        return {}

    artist_normalized = normalize_artist(artist)

    query = f"""
        SELECT parameter_name, parameter_value, avg_similarity, track_count
        FROM midi_grep.artist_knowledge
        WHERE artist_normalized = '{artist_normalized}'
          AND confidence > 0.3
        ORDER BY avg_similarity DESC
    """

    try:
        rows = clickhouse_query(query)
    except:
        return {}  # Table might not exist yet

    presets = {}
    for row in rows:
        param_name = row.get('parameter_name', '')
        value = row.get('parameter_value', '')

        if '.' in param_name:
            fx_name, param = param_name.split('.', 1)
            if fx_name not in presets:
                presets[fx_name] = {}
            try:
                presets[fx_name][param] = float(value)
            except ValueError:
                presets[fx_name][param] = value

    return presets


def detect_artist_from_path(path: str) -> Optional[str]:
    """Try to detect artist name from file path or cache directory."""
    import re

    # Common patterns in YouTube titles: "Artist - Song Title"
    # or "Song Title - Artist"
    path_str = str(path)

    # Look for patterns like "Artist Name - "
    match = re.search(r'/([^/]+)\s*-\s*[^/]+/(?:melodic|v\d+)', path_str)
    if match:
        potential_artist = match.group(1).strip()
        # Filter out common non-artist words
        skip_words = ['official', 'video', 'audio', 'lyric', 'hd', '4k', 'remix']
        if not any(w in potential_artist.lower() for w in skip_words):
            return potential_artist

    return None


def get_genre_presets(genre: str) -> Dict[str, Any]:
    """
    Get genre-specific parameter presets from successful runs.
    Analyzes high-performing runs and extracts average parameters.
    """
    # Query successful runs for this genre (>80% similarity)
    query = f"""
        SELECT
            strudel_code,
            similarity_overall,
            bpm
        FROM midi_grep.runs
        WHERE genre = '{genre}'
          AND similarity_overall > 0.80
        ORDER BY similarity_overall DESC
        LIMIT 10
    """
    rows = clickhouse_query(query)

    if not rows:
        # Fall back to all high-performing runs
        query = """
            SELECT
                strudel_code,
                similarity_overall,
                bpm
            FROM midi_grep.runs
            WHERE similarity_overall > 0.85
            ORDER BY similarity_overall DESC
            LIMIT 10
        """
        rows = clickhouse_query(query)

    if not rows:
        return {}

    # Aggregate parameters from successful runs
    param_sums = {}
    param_counts = {}

    for row in rows:
        code = row.get('strudel_code', '')
        params = extract_parameters_from_code(code)

        for fx_name, fx_params in params.items():
            if fx_name not in param_sums:
                param_sums[fx_name] = {}
                param_counts[fx_name] = {}

            for param, value in fx_params.items():
                if isinstance(value, (int, float)):
                    if param not in param_sums[fx_name]:
                        param_sums[fx_name][param] = 0
                        param_counts[fx_name][param] = 0
                    param_sums[fx_name][param] += value
                    param_counts[fx_name][param] += 1

    # Calculate averages
    presets = {}
    for fx_name in param_sums:
        presets[fx_name] = {}
        for param in param_sums[fx_name]:
            if param_counts[fx_name][param] > 0:
                avg = param_sums[fx_name][param] / param_counts[fx_name][param]
                presets[fx_name][param] = round(avg, 3)

    return presets


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
    """Merge improved effect functions back into the original code."""
    import re

    # Parse improved effects into a dict
    improved_dict = {}
    fx_pattern = r'let\s+(\w+Fx)\s*=\s*(p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*)'
    for match in re.finditer(fx_pattern, improved_effects, re.MULTILINE):
        name, body = match.groups()
        improved_dict[name] = f'let {name} = {body}'

    if not improved_dict:
        return original_code

    # Replace each effect function in the original
    result = original_code
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


def build_improvement_prompt(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = ""
) -> str:
    """Build the prompt for LLM analysis - let LLM decide everything."""

    # Extract full effect functions (the LLM needs to see all effects)
    effects_code = extract_effect_functions(original_code)

    # Get frequency band differences
    band_bass = previous_run.get('band_bass', 0)
    band_mid = previous_run.get('band_mid', 0)
    band_high = previous_run.get('band_high', 0)

    # Get spectral metrics
    brightness_ratio = previous_run.get('brightness_ratio', 1.0)
    energy_ratio = previous_run.get('energy_ratio', 1.0)

    # Include learned knowledge if available - MAKE IT MANDATORY
    knowledge_str = ""
    if learned_knowledge:
        knowledge_str = """

⚠️ MANDATORY KNOWLEDGE FROM DATABASE - YOU MUST USE THESE VALUES ⚠️
These are PROVEN improvements from previous tracks. DO NOT IGNORE.
If the current code doesn't have these values, ADD THEM.
"""
        for k in learned_knowledge[:10]:  # Show more items
            param = k.get('parameter_name', '')
            new_val = k.get('parameter_new_value', '')
            improvement = k.get('similarity_improvement', 0) * 100
            knowledge_str += f"  ✓ SET {param} = {new_val} (proven +{improvement:.0f}% improvement)\n"
        knowledge_str += "\nFAILURE TO APPLY THESE WILL RESULT IN LOWER SIMILARITY SCORES.\n"

    # Include spectrogram insights if available (deep analysis)
    spectrogram_str = ""
    if spectrogram_insights:
        spectrogram_str = f"\n\nDEEP SPECTROGRAM ANALYSIS:\n{spectrogram_insights}\n"

    # Include genre/artist context
    context_str = ""
    if genre:
        context_str += f"\nGENRE: {genre}"
    if artist:
        context_str += f"\nARTIST: {artist}"
    if context_str:
        context_str = f"\nCONTEXT:{context_str}\n"

    # Per-stem issues (the REAL problems, not hidden by overall score)
    per_stem_issues = previous_run.get('per_stem_issues', [])
    stem_issues_str = ""
    if per_stem_issues:
        stem_issues_str = "\n\nPER-STEM ISSUES (these are the REAL problems to fix):\n"
        for issue in per_stem_issues:
            stem_issues_str += f"- {issue}\n"
        stem_issues_str += "\nIMPORTANT: Overall similarity may look OK but individual stems need fixing!\n"

    # Build the research-oriented prompt - give LLM full control
    return f'''You are a STRUDEL MUSIC EXPERT and audio mixing AI.

STRUDEL KNOWLEDGE:
Strudel is a live coding music environment. Code generates audio patterns in real-time.
- Patterns use mini-notation: "c3 d3 e3" plays notes, "~" is rest, "~*4" is 4 rests
- Effects chain: p.sound("...").gain(...).lpf(...).attack(...) etc.
- The code has 4 voices: bass, mid, high, drums - each with effect functions (bassFx, midFx, etc.)

MIXING PRINCIPLES:
1. FREQUENCY BALANCE: Each voice should occupy its own frequency range
   - Bass (20-250Hz): .lpf(300-500), .hpf(30-60) - keep it low and punchy
   - Mid (250-2kHz): .lpf(4000-6000), .hpf(200-400) - main melodic content
   - High (2k-10kHz): .lpf(10000-15000), .hpf(400-800) - brightness and air
   - Avoid overlap! If bass is bleeding into mid, lower bass .lpf()

2. GAIN STAGING:
   - Bass should be felt, not heard: gain 0.1-0.4
   - Mid carries the melody: gain 0.5-1.5
   - High adds sparkle: gain 0.3-0.8
   - If a band is too loud, REDUCE gain. If too quiet, INCREASE gain.

3. ENVELOPES (ADSR):
   - attack: 0.001-0.01 for punchy, 0.05-0.2 for soft
   - decay: 0.05-0.3 for tight, 0.3-1.0 for sustained
   - sustain: 0.3-0.9 (how loud during hold)
   - release: 0.1-0.5 for clean, 0.5-2.0 for ambient

4. EFFECTS:
   - room(0-0.4): reverb - more on highs, less on bass
   - delay(0-0.3): echo - good for highs, careful on bass
   - distort(0-0.3): saturation - adds harmonics
   - phaser(0-0.5): movement - good for pads
   - crush(8-16): lo-fi grit
{context_str}
{stem_issues_str}
COMPARISON DATA (rendered minus original):
- Bass: {band_bass*100:+.0f}% (positive=too loud, negative=too quiet)
- Mid: {band_mid*100:+.0f}%
- High: {band_high*100:+.0f}%
- Brightness: {brightness_ratio:.0%} of original
- Energy: {energy_ratio:.0%} of original
{spectrogram_str}
CURRENT EFFECT FUNCTIONS:
{effects_code}
{knowledge_str}
AVAILABLE EFFECTS:
- .gain(0.01-2.0) - volume
- .hpf(20-2000) - high pass filter (removes bass)
- .lpf(200-20000) - low pass filter (removes treble)
- .attack(0.001-0.5) - envelope attack time
- .decay(0.01-1.0) - envelope decay time
- .sustain(0-1) - envelope sustain level
- .release(0.01-2.0) - envelope release time
- .crush(1-16) - bit depth (lower=more distortion)
- .coarse(1-16) - sample rate reduction
- .room(0-1) - reverb amount
- .delay(0-1) - delay wet mix
- .distort(0-1) - distortion amount
- .phaser(0-1) - phaser effect

SOUND PALETTE (choose sounds that match the genre/timbre):
WAVEFORMS: sine, sawtooth, square, triangle, supersaw
BASS: gm_acoustic_bass, gm_electric_bass_finger, gm_electric_bass_pick, gm_fretless_bass, gm_slap_bass_1, gm_synth_bass_1, gm_synth_bass_2
LEAD/MID: gm_lead_1_square, gm_lead_2_sawtooth, gm_lead_3_calliope, gm_lead_5_charang, gm_lead_6_voice, gm_lead_7_fifths
PADS: gm_pad_1_new_age, gm_pad_2_warm, gm_pad_3_polysynth, gm_pad_4_choir, gm_string_ensemble_1, gm_synth_strings_1
KEYS: gm_acoustic_grand_piano, gm_bright_acoustic_piano, gm_electric_piano_1, gm_electric_piano_2, gm_harpsichord, gm_clavinet
BRASS: gm_trumpet, gm_trombone, gm_alto_sax, gm_tenor_sax, gm_brass_section, gm_synth_brass_1
PERCUSSION: gm_glockenspiel, gm_music_box, gm_vibraphone, gm_marimba, gm_xylophone, gm_celesta
FX: gm_fx_1_rain, gm_fx_3_crystal, gm_fx_4_atmosphere, gm_fx_7_echoes
DRUM BANKS: RolandTR808, RolandTR909, RolandTR707, LinnDrum, OberheimDMX, AlesisHR16, BossDR110

SOUND ALTERNATION - Use "<sound1 sound2>" for variety:
Example: .sound("<supersaw gm_synth_bass_1>") alternates between sounds
Example: .bank("<RolandTR808 RolandTR909>") alternates drum kits

IMPORTANT: Don't use the same sounds for every track! Choose sounds that match the genre:
- Brazilian Funk: RolandTR808, gm_synth_bass_1, gm_lead_2_sawtooth
- Electro Swing: LinnDrum, gm_acoustic_bass, gm_trumpet, gm_clarinet
- Trance: RolandTR909, supersaw, gm_pad_7_halo, gm_lead_7_fifths
- LoFi: BossDR110, triangle, gm_electric_piano_1, gm_vibraphone

EXAMPLE OUTPUT (what I expect):
let bassFx = p => p.sound("<gm_electric_bass_finger gm_synth_bass_1>").gain(0.15).hpf(60).lpf(500).attack(0.01)
let midFx = p => p.sound("<gm_electric_piano_2 gm_lead_3_calliope>").gain(1.2).lpf(5000).attack(0.05).release(0.3)
let highFx = p => p.sound("<gm_music_box gm_vibraphone>").gain(0.8).lpf(12000)
let drumsFx = p => p.bank("<RolandTR909 LinnDrum>").gain(0.9)

TASK: Fix the frequency balance issues shown in COMPARISON DATA.
- If bass is -20%, INCREASE bass gain by ~25% (e.g., 0.3 → 0.38)
- If mid is +30%, DECREASE mid gain by ~25% (e.g., 1.0 → 0.75)
- Use the spectrogram insights for precise dB-to-gain conversions
- Choose sounds that match the genre (not the same sounds every time!)

CRITICAL RULES (MUST FOLLOW):
1. **APPLY ALL MANDATORY KNOWLEDGE ABOVE** - These are proven improvements from the database. If it says "SET bassFx.gain = 0.6", then bassFx MUST have .gain(0.6). No exceptions.
2. Bass MUST have .lpf(500) or lower to stay out of mid range
3. Each voice needs .hpf() and .lpf() to prevent frequency overlap
4. Use perlin.range() for natural gain variation: perlin.range(0.3, 0.5).slow(8)
5. Conservative changes - adjust by 20-30% per iteration, not 200%

Output ONLY these 4 lines (no explanation, no comments):
let bassFx = p => p.sound("...").gain(...).hpf(...).lpf(...)...
let midFx = p => p.sound("...").gain(...).hpf(...).lpf(...)...
let highFx = p => p.sound("...").gain(...).hpf(...).lpf(...)...
let drumsFx = p => p.bank("...").gain(...)...'''


def analyze_with_ollama(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    model: str = None,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = ""
) -> Dict[str, Any]:
    """Use Ollama (local LLM) to analyze results and suggest improvements."""

    if not HAS_REQUESTS:
        print("Warning: requests package not installed", file=sys.stderr)
        return {"suggestions": [], "improved_code": original_code, "reasoning": "No requests"}

    ollama_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 4096,
                }
            },
            timeout=300  # 5 minutes for generation
        )
        response.raise_for_status()

        result_text = response.json().get("response", "")

        # Try to parse as JSON first
        import re

        # Method 1: Look for ```json block
        if "```json" in result_text:
            json_text = result_text.split("```json")[1].split("```")[0]
            try:
                return json.loads(json_text)
            except:
                pass

        # Method 2: Look for JSON object pattern
        json_match = re.search(r'\{[\s\S]*?"improved_code"[\s\S]*?\}', result_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # Method 3: Extract code from markdown code blocks
        code_match = re.search(r'```(?:scss|javascript|js)?\n?([\s\S]*?)```', result_text)
        if code_match:
            extracted_code = code_match.group(1).strip()
            # Check if it looks like effect functions
            if 'Fx' in extracted_code or 'sound(' in extracted_code:
                return {
                    "analysis": "Extracted from response",
                    "suggestions": ["Code extracted from markdown"],
                    "improved_code": extracted_code
                }

        # Method 4: Look for let *Fx patterns directly in text
        fx_matches = re.findall(r'let\s+\w+Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*', result_text)
        if fx_matches:
            return {
                "analysis": "Extracted effect functions",
                "suggestions": ["Effect functions extracted from response"],
                "improved_code": '\n'.join(fx_matches)
            }

        # Method 5: Parse natural language reasoning and extract values
        # Look for mentions of gain values, reduce/increase, etc.
        analysis = result_text[:500] if len(result_text) > 500 else result_text
        suggestions = []

        # Extract any numeric gain suggestions from text
        gain_mentions = re.findall(r'(bass|mid|high|drums?)\s*(?:gain|Fx)?\s*(?:to|=|:)?\s*(0?\.[0-9]+|[0-2]\.[0-9]+)', result_text.lower())
        if gain_mentions:
            suggestions = [f"{m[0]} gain -> {m[1]}" for m in gain_mentions]

        # Look for reduce/increase keywords
        reduce_matches = re.findall(r'reduce\s+(bass|mid|high|drums?)', result_text.lower())
        increase_matches = re.findall(r'increase\s+(bass|mid|high|drums?)', result_text.lower())

        if reduce_matches:
            suggestions.extend([f"reduce {m}" for m in reduce_matches])
        if increase_matches:
            suggestions.extend([f"increase {m}" for m in increase_matches])

        # Look for "too loud" / "too quiet" mentions
        if re.search(r'bass.*too\s*(loud|much|high)', result_text.lower()):
            suggestions.append("reduce bass")
        if re.search(r'bass.*too\s*(quiet|low|soft)', result_text.lower()):
            suggestions.append("increase bass")
        if re.search(r'mid.*too\s*(loud|much|high)', result_text.lower()):
            suggestions.append("reduce mid")
        if re.search(r'mid.*too\s*(quiet|low|soft|missing)', result_text.lower()):
            suggestions.append("increase mid")

        # Return whatever we found - the caller will generate code from band data
        return {
            "analysis": analysis,
            "suggestions": suggestions if suggestions else ["LLM reasoning parsed"],
            "improved_code": "",
            "text_response": result_text,
            "parsed_from_text": True
        }

    except requests.exceptions.ConnectionError:
        print(f"Warning: Ollama not running at {OLLAMA_URL}", file=sys.stderr)
        print("Start with: ollama serve", file=sys.stderr)
        return {"suggestions": [], "improved_code": original_code, "reasoning": "Ollama not running"}
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return {"suggestions": [f"Error: {e}"], "improved_code": original_code, "reasoning": str(e)}


def analyze_with_claude(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = ""
) -> Dict[str, Any]:
    """Use Claude API to analyze results and suggest improvements."""

    if Anthropic is None:
        return None  # Signal to try Ollama

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # Signal to try Ollama

    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist
    )
    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)

    except Exception as e:
        print(f"Claude API error: {e}", file=sys.stderr)
        return None  # Signal to try Ollama


def analyze_with_llm(
    previous_run: Dict,
    comparison_data: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    use_ollama: bool = False,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = ""
) -> Dict[str, Any]:
    """
    Analyze and improve code using available LLM.

    Priority:
    1. If use_ollama=True, use Ollama directly
    2. Try Claude API if ANTHROPIC_API_KEY is set
    3. Fall back to Ollama (local)
    4. Return unchanged code if nothing works

    Now enhanced with spectrogram_insights for deeper analysis.
    Genre and artist context helps with sound selection.
    """

    if use_ollama:
        print("       Using Ollama (local LLM)...")
        return analyze_with_ollama(
            previous_run, learned_knowledge, original_code,
            spectrogram_insights=spectrogram_insights, genre=genre, artist=artist
        )

    # Try Claude first
    result = analyze_with_claude(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist
    )
    if result is not None:
        return result

    # Fall back to Ollama
    print("       Claude unavailable, using Ollama (local LLM)...")
    return analyze_with_ollama(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist
    )


def store_run(
    track_hash: str,
    track_name: str,
    version: int,
    strudel_code: str,
    bpm: float,
    key: str,
    style: str,
    genre: str,
    comparison: Dict,
    mix_params: Dict,
    improved_from: Optional[int] = None,
    ai_suggestions: Optional[List[str]] = None
):
    """Store a run in ClickHouse."""
    # Extract data from comparison structure
    comp_scores = comparison.get("comparison", {})
    orig_bands = comparison.get("original", {}).get("bands", {})
    rend_bands = comparison.get("rendered", {}).get("bands", {})

    # Compute band differences (rendered - original)
    band_diffs = {}
    for band in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
        band_diffs[band] = rend_bands.get(band, 0) - orig_bands.get(band, 0)

    data = {
        "track_hash": track_hash,
        "track_name": track_name,
        "version": version,
        "bpm": bpm,
        "key": key,
        "style": style,
        "genre": genre,
        "strudel_code": strudel_code,
        "similarity_overall": comp_scores.get("overall_similarity", 0),
        "similarity_mfcc": comp_scores.get("mfcc_similarity", 0),
        "similarity_chroma": comp_scores.get("chroma_similarity", 0),
        "similarity_frequency": comp_scores.get("frequency_balance_similarity", 0),
        "similarity_rhythm": comp_scores.get("tempo_similarity", 0),
        "band_sub_bass": band_diffs.get("sub_bass", 0),
        "band_bass": band_diffs.get("bass", 0),
        "band_low_mid": band_diffs.get("low_mid", 0),
        "band_mid": band_diffs.get("mid", 0),
        "band_high_mid": band_diffs.get("high_mid", 0),
        "band_high": band_diffs.get("high", 0),
        "mix_params": json.dumps(mix_params),
        "improved_from_version": improved_from,
        "ai_suggestions": json.dumps(ai_suggestions) if ai_suggestions else ""
    }

    return clickhouse_insert("midi_grep.runs", data)


def store_knowledge(
    genre: str,
    bpm: float,
    key_type: str,
    parameter_name: str,
    old_value: str,
    new_value: str,
    improvement: float
):
    """Store learned knowledge."""
    data = {
        "genre": genre,
        "bpm_range_low": max(0, bpm - 20),
        "bpm_range_high": bpm + 20,
        "key_type": key_type,
        "parameter_name": parameter_name,
        "parameter_old_value": old_value,
        "parameter_new_value": new_value,
        "similarity_improvement": improvement,
        "confidence": 1.0,
        "run_ids": "[]"
    }
    return clickhouse_insert("midi_grep.knowledge", data)


def analyze_original_audio(audio_path: str, output_dir: str) -> Optional[Dict]:
    """Analyze original audio to extract synthesis parameters."""
    config_path = Path(output_dir) / "synth_config.json"

    analyze_script = Path(__file__).parent / "analyze_synth_params.py"
    if not analyze_script.exists():
        print(f"Warning: analyze_synth_params.py not found", file=sys.stderr)
        return None

    cmd = [
        sys.executable,
        str(analyze_script),
        audio_path,
        "-o", str(config_path),
        "-d", "60"  # Analyze first 60 seconds
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Audio analysis failed: {result.stderr[:200]}", file=sys.stderr)
        return None

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def improve_strudel(
    original_audio: str,
    strudel_path: str,
    output_dir: str,
    metadata: Dict,
    max_iterations: int = 5,
    target_similarity: float = 0.70,
    use_ollama: bool = False
) -> Dict:
    """
    Main improvement loop with AI-driven synthesis parameter extraction.

    Args:
        original_audio: Path to original audio file
        strudel_path: Path to current Strudel code
        output_dir: Directory to save outputs
        metadata: Track metadata (bpm, key, style, genre)
        max_iterations: Maximum improvement iterations
        target_similarity: Target similarity to reach
        use_ollama: Force using Ollama (local LLM) instead of Claude

    Returns:
        Dict with final results
    """
    track_hash = get_track_hash(original_audio)
    track_name = Path(original_audio).stem

    print(f"\n{'='*60}")
    print("AI-DRIVEN STRUDEL IMPROVEMENT")
    print(f"{'='*60}")
    print(f"Track: {track_name}")
    print(f"Hash: {track_hash}")
    print(f"Target: {target_similarity*100:.0f}% similarity")
    print(f"Max iterations: {max_iterations}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # PHASE 1: Analyze original audio to extract synthesis parameters
    print(f"\n--- Phase 1: Analyzing original audio ---")
    synth_config = analyze_original_audio(original_audio, output_dir)
    synth_config_path = Path(output_dir) / "synth_config.json"

    if synth_config:
        synth_params = synth_config.get("synth_config", synth_config)
        tempo_info = synth_params.get("tempo", {})
        analyzed_bpm = tempo_info.get("bpm", metadata.get("bpm", 120))
        print(f"  Extracted BPM: {analyzed_bpm:.1f} (confidence: {tempo_info.get('confidence', 0)*100:.0f}%)")
        print(f"  Waveform suggestion: {synth_params.get('oscillator', {}).get('waveform', 'saw')}")
        print(f"  Attack: {synth_params.get('envelope', {}).get('attack', 0.01)*1000:.1f}ms")
        print(f"  Harmonics: {'tonal' if synth_params.get('harmonics', {}).get('harmonic_ratio', 0) > 0.5 else 'percussive'}")
        # Update metadata with analyzed values
        metadata["bpm"] = analyzed_bpm
    else:
        print("  Warning: Could not analyze original audio, using defaults")
        synth_config = None

    # Read current code
    with open(strudel_path) as f:
        current_code = f.read()

    # Apply genre-specific presets as starting point
    genre = metadata.get("genre", "")
    if genre:
        presets = get_genre_presets(genre)
        if presets:
            print(f"\n--- Applying {genre} genre presets ---")
            for fx_name, params in presets.items():
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  {fx_name}: {param_str}")
            current_code = apply_genre_presets(current_code, presets)
            # Save preset-applied code
            with open(strudel_path, 'w') as f:
                f.write(current_code)

    # Try to detect and apply artist-specific presets
    artist = metadata.get("artist", "")
    if not artist:
        artist = detect_artist_from_path(original_audio)
    if artist:
        artist_presets = get_artist_presets(artist)
        if artist_presets:
            print(f"\n--- Applying {artist} artist presets ---")
            for fx_name, params in artist_presets.items():
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  {fx_name}: {param_str}")
            current_code = apply_genre_presets(current_code, artist_presets)
            with open(strudel_path, 'w') as f:
                f.write(current_code)

    # Track previous code for learning
    previous_code = current_code
    previous_similarity = 0.0

    # Check for previous runs (most recent for version numbering)
    previous_run = get_previous_run(track_hash)
    if previous_run:
        current_version = previous_run["version"] + 1
        print(f"\nFound previous run: v{previous_run['version']} with {previous_run['similarity_overall']*100:.1f}% similarity")
    else:
        current_version = 1
        print(f"\nNo previous runs found, starting fresh")

    # CRITICAL: Get the BEST run ever (from ClickHouse, survives cache clear)
    # This ensures v228 knows everything about the best code from all 227 previous versions
    best_run = get_best_run(track_hash)
    if best_run and best_run.get("strudel_code"):
        best_ever_similarity = best_run.get("similarity_overall", 0)
        best_ever_code = best_run.get("strudel_code", "")
        best_ever_version = best_run.get("version", 0)
        print(f"📊 Best ever: v{best_ever_version} with {best_ever_similarity*100:.1f}% similarity (from ClickHouse)")

        # Start from best code if it's better than what we'd start with
        if best_ever_code and best_ever_similarity > 0:
            current_code = best_ever_code
            previous_similarity = best_ever_similarity
            print(f"   Starting from best known code (not fresh)")
    else:
        best_ever_similarity = 0
        best_ever_code = None

    # Get learned knowledge
    key_type = "minor" if "minor" in metadata.get("key", "").lower() else "major"
    knowledge = get_learned_knowledge(
        metadata.get("genre", ""),
        metadata.get("bpm", 120),
        key_type
    )
    if knowledge:
        print(f"Loaded {len(knowledge)} knowledge items for this context")

    best_similarity = previous_similarity if best_run else 0
    best_code = current_code

    # Get exact duration from original audio (millisecond precision)
    exact_duration = get_audio_duration(original_audio)
    print(f"Original audio duration: {exact_duration:.6f}s")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} (v{current_version}) ---")

        # 1. Render current code using Node.js renderer (better harmonics)
        render_path = Path(output_dir) / f"render_v{current_version:03d}.wav"
        node_renderer = Path(__file__).parent.parent / "node" / "dist" / "render-strudel-node.js"

        if node_renderer.exists():
            # Prefer Node.js renderer with dynamic synthesis config
            render_cmd = [
                "node", str(node_renderer),
                strudel_path,
                str(render_path),
                "--duration", f"{exact_duration:.2f}"
            ]
            # Pass synth config if available (AI-analyzed parameters)
            if synth_config_path.exists():
                render_cmd.extend(["--config", str(synth_config_path)])
                print(f"       Using AI-analyzed synthesis config")
        else:
            # Fallback to Python renderer
            render_cmd = [
                sys.executable,
                str(Path(__file__).parent / "render_audio.py"),
                strudel_path,
                "-o", str(render_path),
                "-d", f"{exact_duration:.6f}"
            ]
        subprocess.run(render_cmd, capture_output=True)

        if not render_path.exists():
            print("Render failed, stopping")
            break

        # 2. Compare to original
        compare_cmd = [
            sys.executable,
            str(Path(__file__).parent / "compare_audio.py"),
            original_audio,
            str(render_path),
            "-j"  # Output JSON to stdout
        ]
        compare_result = subprocess.run(compare_cmd, capture_output=True, text=True)

        comparison_path = Path(output_dir) / f"comparison_v{current_version:03d}.json"
        if compare_result.returncode == 0 and compare_result.stdout.strip():
            comparison = json.loads(compare_result.stdout)
            # Save for reference
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
        else:
            print("Comparison failed, using defaults")
            comparison = {"overall": 0, "mfcc": 0, "chroma": 0, "frequency": 0}

        # Extract similarity from nested comparison structure
        comp_scores = comparison.get("comparison", {})
        current_similarity = comp_scores.get("overall_similarity", 0)
        print(f"Similarity: {current_similarity*100:.1f}%")

        # Load per-stem comparison if available (shows real issues even when overall looks good)
        stem_comparison = {}
        stem_comparison_path = Path(output_dir) / "stem_comparison.json"
        if stem_comparison_path.exists():
            with open(stem_comparison_path) as f:
                stem_comparison = json.load(f)
            agg = stem_comparison.get("aggregate", {}).get("per_stem", {})
            worst = stem_comparison.get("aggregate", {}).get("worst_sections", [])
            print(f"       Per-stem: bass={agg.get('bass', {}).get('overall', 0)*100:.0f}% drums={agg.get('drums', {}).get('overall', 0)*100:.0f}% melodic={agg.get('melodic', {}).get('overall', 0)*100:.0f}%")
            if worst:
                print(f"       Worst: {worst[0].get('stem', '?')} {worst[0].get('time_range', '?')}: {worst[0].get('issues', '?')}")

        # Learn from improvement (if this iteration improved over previous)
        if iteration > 0 and current_similarity > previous_similarity:
            key_type = "minor" if "minor" in metadata.get("key", "").lower() else "major"
            learned = learn_from_improvement(
                genre=metadata.get("genre", ""),
                bpm=metadata.get("bpm", 120),
                key_type=key_type,
                old_code=previous_code,
                new_code=current_code,
                old_similarity=previous_similarity,
                new_similarity=current_similarity
            )
            if learned > 0:
                print(f"       Stored {learned} knowledge entries")

        # Update tracking for next iteration
        previous_code = current_code
        previous_similarity = current_similarity

        # 3. Store this run
        store_run(
            track_hash=track_hash,
            track_name=track_name,
            version=current_version,
            strudel_code=current_code,
            bpm=metadata.get("bpm", 120),
            key=metadata.get("key", ""),
            style=metadata.get("style", ""),
            genre=metadata.get("genre", ""),
            comparison=comparison,
            mix_params={},
            improved_from=current_version - 1 if current_version > 1 else None,
            ai_suggestions=None
        )

        # Track best
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_code = current_code

        # 4. ALWAYS run LLM analysis - even if overall looks good, per-stem may have issues
        # The overall score can hide problems (e.g., bass 68%, drums 72% while melodic 93%)
        print("Analyzing with LLM...")

        # Extract data from comparison structure
        comp_scores = comparison.get("comparison", {})
        orig_bands = comparison.get("original", {}).get("bands", {})
        rend_bands = comparison.get("rendered", {}).get("bands", {})

        # Compute band differences (rendered - original)
        band_diffs = {}
        for band in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
            band_diffs[band] = rend_bands.get(band, 0) - orig_bands.get(band, 0)

        # Compute brightness and energy ratios
        orig_spectral = comparison.get("original", {}).get("spectral", {})
        rend_spectral = comparison.get("rendered", {}).get("spectral", {})

        orig_centroid = orig_spectral.get("centroid_mean", 1)
        rend_centroid = rend_spectral.get("centroid_mean", 1)
        brightness_ratio = rend_centroid / max(orig_centroid, 1) if orig_centroid else 1.0

        orig_rms = orig_spectral.get("rms_mean", 0.1)
        rend_rms = rend_spectral.get("rms_mean", 0.1)
        energy_ratio = rend_rms / max(orig_rms, 0.001) if orig_rms else 1.0

        # Build per-stem issues for LLM (the real problems, not hidden by overall score)
        per_stem_issues = []
        if stem_comparison:
            agg = stem_comparison.get("aggregate", {})
            per_stem = agg.get("per_stem", {})
            for stem_name, stem_data in per_stem.items():
                stem_overall = stem_data.get("overall", 0)
                if stem_overall < 0.80:  # Only report issues below 80%
                    per_stem_issues.append(f"{stem_name}: {stem_overall*100:.0f}% (needs improvement)")
            worst = agg.get("worst_sections", [])
            for w in worst[:3]:  # Top 3 worst sections
                per_stem_issues.append(f"WORST: {w.get('stem', '?')} at {w.get('time_range', '?')} - {w.get('issues', '')}")

        run_data = {
            "similarity_overall": current_similarity,
            "similarity_mfcc": comp_scores.get("mfcc_similarity", 0),
            "similarity_chroma": comp_scores.get("chroma_similarity", 0),
            "similarity_frequency": comp_scores.get("frequency_balance_similarity", 0),
            "band_sub_bass": band_diffs.get("sub_bass", 0),
            "band_bass": band_diffs.get("bass", 0),
            "band_low_mid": band_diffs.get("low_mid", 0),
            "band_mid": band_diffs.get("mid", 0),
            "band_high_mid": band_diffs.get("high_mid", 0),
            "band_high": band_diffs.get("high", 0),
            "brightness_ratio": brightness_ratio,
            "energy_ratio": energy_ratio,
            "bpm": metadata.get("bpm", 120),
            "key": metadata.get("key", ""),
            "style": metadata.get("style", ""),
            "genre": metadata.get("genre", ""),
            "per_stem_issues": per_stem_issues  # The REAL problems
        }

        print(f"       Bands: bass={band_diffs.get('bass',0)*100:+.0f}% mid={band_diffs.get('mid',0)*100:+.0f}% high={band_diffs.get('high',0)*100:+.0f}%")
        print(f"       Brightness: {brightness_ratio:.0%}  Energy: {energy_ratio:.0%}")

        # Deep spectrogram analysis for AI learning
        spectrogram_insights = None
        try:
            from spectrogram_analyzer import analyze_spectrograms, format_for_llm
            print(f"       Running deep spectrogram analysis...")
            spec_analysis = analyze_spectrograms(original_audio, str(render_path), duration=min(30, exact_duration))
            spectrogram_insights = format_for_llm(spec_analysis)
            spec_sim = spec_analysis.get('spectrogram_similarity', 0)
            print(f"       Spectrogram similarity: {spec_sim*100:.1f}%")

            # Save spectrogram analysis for learning
            spec_path = Path(output_dir) / f"spectrogram_v{current_version:03d}.json"
            with open(spec_path, 'w') as f:
                json.dump(spec_analysis, f, indent=2)

            # Add spectrogram insights to run_data for LLM
            run_data['spectrogram_insights'] = spectrogram_insights
            run_data['spectrogram_similarity'] = spec_sim
        except ImportError:
            pass
        except Exception as e:
            print(f"       Spectrogram analysis failed: {e}")

        # Extract genre and artist from metadata for sound selection
        genre = metadata.get("genre", "")
        artist_context = artist if artist else metadata.get("artist", "")
        ai_result = analyze_with_llm(
            run_data, comparison, knowledge, current_code,
            use_ollama=use_ollama, spectrogram_insights=spectrogram_insights,
            genre=genre, artist=artist_context
        )

        print(f"Analysis: {ai_result.get('analysis', 'N/A')[:100]}...")
        print(f"Suggestions: {ai_result.get('suggestions', [])[:3]}")

        # 6. Update code for next iteration
        improved_effects = ai_result.get("improved_code", "")

        # NO FALLBACKS - LLM MUST return code or we fail
        if not improved_effects or not improved_effects.strip():
            raise RuntimeError(
                f"LLM did not return code! This is a bug.\n"
                f"AI result: {ai_result}\n"
                f"Knowledge items: {len(knowledge)}\n"
                f"Make sure Ollama is running: curl http://localhost:11434/api/tags"
            )

        if improved_effects and improved_effects.strip():
            # Merge improved effects back into full code
            improved_code = merge_effect_functions(current_code, improved_effects)
            if improved_code == current_code:
                print("       No effective changes this iteration, continuing...")
                # DON'T break - keep iterating, LLM might give different suggestions next time
            else:
                current_code = improved_code
                # Save improved code
                improved_path = Path(output_dir) / f"output_v{current_version + 1:03d}.strudel"
                with open(improved_path, 'w') as f:
                    f.write(improved_code)
                # Update the main strudel file
                with open(strudel_path, 'w') as f:
                    f.write(improved_code)
                print(f"Saved improved code to {improved_path}")
        else:
            print("       No code changes this iteration, continuing...")
            # DON'T break - keep iterating

        current_version += 1

        # NEVER exit early - ALWAYS run ALL iterations
        # The LLM and full analysis must run every single time
        # Per-stem issues need multiple iterations to fix properly

    # Generate final comparison charts and report
    print("\nGenerating comparison charts and report...")

    # Find the latest render file in output directory
    render_files = sorted(Path(output_dir).glob("render_v*.wav"))
    if render_files:
        best_render = render_files[-1]  # Latest version
    else:
        best_render = Path(output_dir) / "render.wav"

    if best_render.exists():
        print(f"Using render: {best_render.name}")

        # Generate charts
        chart_cmd = [
            sys.executable,
            str(Path(__file__).parent / "compare_audio.py"),
            original_audio,
            str(best_render),
            "-c", str(Path(output_dir) / "comparison.png")
        ]
        chart_result = subprocess.run(chart_cmd, capture_output=True, text=True)
        if chart_result.returncode != 0:
            print(f"Chart generation warning: {chart_result.stderr[:200]}")

        # Find and copy best strudel file
        strudel_files = sorted(Path(output_dir).glob("output_v*.strudel"))
        if strudel_files:
            best_strudel = strudel_files[-1]
            with open(best_strudel) as f:
                code = f.read()
            with open(Path(output_dir) / "output.strudel", 'w') as f:
                f.write(code)

        # Copy render
        import shutil
        if best_render.name != "render.wav":
            shutil.copy(best_render, Path(output_dir) / "render.wav")

        # Create metadata
        meta = {
            "bpm": metadata.get("bpm", 120),
            "key": metadata.get("key", ""),
            "style": metadata.get("style", ""),
            "genre": metadata.get("genre", ""),
            "ai_improved": True,
            "iterations": current_version,
            "similarity": best_similarity
        }
        with open(Path(output_dir) / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Generate HTML report
        cache_dir = Path(output_dir).parent
        dir_name = Path(output_dir).name
        # Extract version number (e.g., "v001" -> "1", or use directory if no version pattern)
        import re
        version_match = re.search(r'v(\d+)', dir_name)
        version_num = version_match.group(1) if version_match else "1"

        report_cmd = [
            sys.executable,
            str(Path(__file__).parent / "generate_report.py"),
            str(cache_dir),
            "-v", version_num,
            "-o", str(Path(output_dir) / "report.html")
        ]
        result = subprocess.run(report_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Report generated: {Path(output_dir) / 'report.html'}")
        else:
            print(f"Report generation failed: {result.stderr}")

    print(f"\n{'='*60}")
    print("IMPROVEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Best similarity: {best_similarity*100:.1f}%")
    print(f"Total versions: {current_version}")

    return {
        "best_similarity": best_similarity,
        "best_code": best_code,
        "versions": current_version,
        "track_hash": track_hash
    }


def main():
    parser = argparse.ArgumentParser(description='AI-driven Strudel code improvement')
    parser.add_argument('original', help='Original audio file')
    parser.add_argument('strudel', help='Strudel code file')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('-i', '--iterations', type=int, default=5, help='Max iterations')
    parser.add_argument('-t', '--target', type=float, default=0.70, help='Target similarity (0-1)')
    parser.add_argument('--bpm', type=float, default=120, help='Track BPM')
    parser.add_argument('--key', default='', help='Track key')
    parser.add_argument('--style', default='auto', help='Sound style')
    parser.add_argument('--genre', default='', help='Genre')
    parser.add_argument('--artist', default='', help='Artist name (for artist-specific presets)')
    parser.add_argument('--ollama', action='store_true', help='Use Ollama (local LLM) instead of Claude API')
    parser.add_argument('--ollama-model', default=DEFAULT_OLLAMA_MODEL, help=f'Ollama model to use (default: {DEFAULT_OLLAMA_MODEL})')

    args = parser.parse_args()

    # Set Ollama model in environment for nested calls
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model

    metadata = {
        "bpm": args.bpm,
        "key": args.key,
        "style": args.style,
        "genre": args.genre,
        "artist": args.artist
    }

    result = improve_strudel(
        args.original,
        args.strudel,
        args.output_dir,
        metadata,
        max_iterations=args.iterations,
        target_similarity=args.target,
        use_ollama=args.ollama
    )

    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == '__main__':
    main()
