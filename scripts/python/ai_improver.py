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

# Import agentic Ollama wrapper
try:
    from ollama_agent import OllamaAgent
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


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


def _run_clickhouse_query(query: str) -> subprocess.CompletedProcess:
    """Run a ClickHouse query and return result."""
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _get_schema_version() -> int:
    """Get current schema version from DB, or 0 if not initialized."""
    result = _run_clickhouse_query(
        "SELECT version FROM midi_grep.schema_version ORDER BY version DESC LIMIT 1"
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0


# Migrations: list of (version, description, queries)
# Each migration is atomic - all queries must succeed
MIGRATIONS = [
    (1, "Initial schema", [
        "CREATE DATABASE IF NOT EXISTS midi_grep",
        """CREATE TABLE midi_grep.schema_version (
            version UInt32,
            applied_at DateTime DEFAULT now(),
            description String
        ) ENGINE = MergeTree()
        ORDER BY version""",
        """CREATE TABLE midi_grep.runs (
            track_hash String,
            track_name String,
            version UInt32,
            created_at DateTime DEFAULT now(),
            bpm Float64,
            key String,
            key_type String,
            style String,
            genre String,
            strudel_code String,
            similarity_overall Float64,
            similarity_mfcc Float64,
            similarity_chroma Float64,
            similarity_frequency Float64,
            similarity_rhythm Float64,
            band_sub_bass Float64,
            band_bass Float64,
            band_low_mid Float64,
            band_mid Float64,
            band_high_mid Float64,
            band_high Float64,
            mix_params String,
            improved_from_version Nullable(UInt32),
            ai_suggestions String
        ) ENGINE = MergeTree()
        ORDER BY (track_hash, version)""",
        """CREATE TABLE midi_grep.knowledge (
            id UUID DEFAULT generateUUIDv4(),
            created_at DateTime DEFAULT now(),
            parameter_name String,
            parameter_old_value String,
            parameter_new_value String,
            similarity_improvement Float64,
            confidence Float64,
            genre String,
            bpm_range_low Float64,
            bpm_range_high Float64,
            key_type String,
            track_hash String
        ) ENGINE = MergeTree()
        ORDER BY (parameter_name, created_at)""",
    ]),
    # Add future migrations here:
    # (2, "Add new column", ["ALTER TABLE midi_grep.runs ADD COLUMN foo String"]),
]


def init_clickhouse():
    """Initialize ClickHouse with migrations. Fails fast if any migration fails."""
    CLICKHOUSE_DB.mkdir(parents=True, exist_ok=True)

    # Ensure database exists first
    result = _run_clickhouse_query("CREATE DATABASE IF NOT EXISTS midi_grep")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create database: {result.stderr}")

    current_version = _get_schema_version()

    # Apply pending migrations
    for version, description, queries in MIGRATIONS:
        if version <= current_version:
            continue  # Already applied

        print(f"  [ClickHouse] Applying migration {version}: {description}")

        # Run all queries for this migration
        for query in queries:
            result = _run_clickhouse_query(query)
            if result.returncode != 0:
                # Check if it's just "already exists" for CREATE statements
                if "already exists" in result.stderr.lower():
                    continue
                raise RuntimeError(
                    f"Migration {version} failed on query:\n{query}\n\nError: {result.stderr}"
                )

        # Record successful migration
        record_query = f"INSERT INTO midi_grep.schema_version (version, description) VALUES ({version}, '{description}')"
        result = _run_clickhouse_query(record_query)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to record migration {version}: {result.stderr}")

        print(f"  [ClickHouse] Migration {version} applied successfully")

    final_version = _get_schema_version()
    if final_version > 0:
        print(f"  [ClickHouse] Schema at version {final_version}")


# Initialize ClickHouse on module load
try:
    init_clickhouse()
except Exception as e:
    print(f"ClickHouse initialization failed: {e}", file=sys.stderr)
    raise


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
    """Insert a row into ClickHouse. Raises RuntimeError on failure."""
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

    result = _run_clickhouse_query(query)
    if result.returncode != 0:
        raise RuntimeError(f"ClickHouse insert failed: {result.stderr}\nQuery: {query[:500]}")


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
    track_hash: str,
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
                track_hash=track_hash,
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


def build_improvement_prompt(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = "",
    bpm: float = 120
) -> str:
    """Build the prompt for LLM analysis with INTERACTIVE CONTROLS."""

    # Extract full effect functions (the LLM needs to see all effects)
    effects_code = extract_effect_functions(original_code)

    # Get frequency band differences
    band_bass = previous_run.get('band_bass', 0) + previous_run.get('band_sub_bass', 0)
    band_mid = previous_run.get('band_mid', 0) + previous_run.get('band_low_mid', 0)
    band_high = previous_run.get('band_high', 0) + previous_run.get('band_high_mid', 0)

    # Get spectral metrics
    brightness_ratio = previous_run.get('brightness_ratio', 1.0)
    energy_ratio = previous_run.get('energy_ratio', 1.0)

    # Include learned knowledge if available
    knowledge_str = ""
    if learned_knowledge:
        knowledge_str = "\n## PROVEN IMPROVEMENTS (from database)\n"
        for k in learned_knowledge[:5]:
            param = k.get('parameter_name', '')
            new_val = k.get('parameter_new_value', '')
            improvement = k.get('similarity_improvement', 0) * 100
            knowledge_str += f"- {param} = {new_val} (+{improvement:.0f}%)\n"

    # Include spectrogram insights if available
    spectrogram_str = ""
    if spectrogram_insights:
        spectrogram_str = f"\n## SPECTROGRAM ANALYSIS\n{spectrogram_insights}\n"

    # Per-stem issues
    per_stem_issues = previous_run.get('per_stem_issues', [])
    stem_issues_str = ""
    if per_stem_issues:
        stem_issues_str = "\n## PER-STEM ISSUES\n"
        for issue in per_stem_issues[:5]:
            stem_issues_str += f"- {issue}\n"

    # Calculate suggested gains based on comparison
    bass_status = "too loud" if band_bass > 0.05 else "too quiet" if band_bass < -0.05 else "OK"
    mid_status = "too loud" if band_mid > 0.05 else "too quiet" if band_mid < -0.05 else "OK"
    high_status = "too loud" if band_high > 0.05 else "too quiet" if band_high < -0.05 else "OK"

    # CPS calculation for tempo
    cps = bpm / 60 / 4  # cycles per second for 4 beats/cycle

    return f'''# STRUDEL LIVE CODING AI

You are an expert at Strudel, the live coding music environment.
Your output will be used for LIVE PERFORMANCE - make it INTERACTIVE!

## STRUDEL INTERACTIVE CONTROLS (USE THESE!)

### Inline Sliders - Draggable UI controls in code
```javascript
slider(default, min, max, step)
// Example: .lpf(slider(500, 100, 2000, 1))
// Example: .gain(slider(0.5, 0, 1, 0.01))
```

### Tempo Control
```javascript
setcps({cps:.3f})  // Current: {bpm:.0f} BPM
// Or with slider: setcps(slider({cps:.3f}, 0.3, 1.0, 0.01))
```

### Dynamic Modulation (LFOs)
```javascript
sine.range(min, max).slow(cycles)    // Smooth LFO
perlin.range(min, max).slow(cycles)  // Organic noise
// Example: .lpf(sine.range(400, 4000).slow(8))
// Example: .gain(perlin.range(0.3, 0.7).slow(16))
```

### Probability Variation
```javascript
.sometimes(x => x.speed(2))    // 50% chance
.rarely(x => x.rev())          // 25% chance
.degradeBy(slider(0.3, 0, 1))  // Slider-controlled note drop
```

## CURRENT ANALYSIS

| Voice | Status | Difference |
|-------|--------|------------|
| Bass  | {bass_status} | {band_bass*100:+.0f}% |
| Mid   | {mid_status} | {band_mid*100:+.0f}% |
| High  | {high_status} | {band_high*100:+.0f}% |

Brightness: {brightness_ratio:.0%} of original
Energy: {energy_ratio:.0%} of original
{spectrogram_str}
{stem_issues_str}
{knowledge_str}

## CURRENT EFFECT FUNCTIONS
```javascript
{effects_code}
```

## GENRE CONTEXT
Genre: {genre or 'auto'}
Artist: {artist or 'unknown'}
BPM: {bpm:.0f}

## GENRE-SPECIFIC SOUNDS
- **Brazilian Funk**: RolandTR808, gm_synth_bass_1, gm_lead_2_sawtooth
- **Electro Swing**: LinnDrum, gm_acoustic_bass, gm_trumpet, gm_clarinet
- **Trance**: RolandTR909, supersaw, gm_pad_7_halo, gm_lead_7_fifths
- **LoFi**: BossDR110, triangle, gm_electric_piano_1, gm_vibraphone
- **House**: RolandTR909, gm_synth_bass_2, supersaw
- **Jazz**: AkaiLinn, gm_acoustic_bass, gm_electric_piano_1

## TASK: Generate BEAT-SYNCED Strudel code

Fix frequency balance and add BEAT-SYNCED dynamics using Strudel patterns.

### BEAT-SYNCED VALUES (MANDATORY - DO NOT USE PERLIN FOR GAIN!)

**CRITICAL RULES:**
- ❌ FORBIDDEN: `perlin.range(0.5, 0.6)` - ranges too narrow, inaudible
- ❌ FORBIDDEN: Any gain range smaller than 0.3 difference
- ✅ REQUIRED: Use `"<v1 v2 v3 v4>".slow(16)` for section dynamics
- ✅ REQUIRED: Wide ranges (0.2 to 0.8, not 0.5 to 0.6)

```javascript
// CORRECT - Beat-synced with wide range:
.gain("<0.2 0.4 0.8 0.5>".slow(16))  // 4 sections, dramatic changes

// WRONG - Too subtle, inaudible:
.gain(perlin.range(0.5, 0.6).slow(8))  // Only 10% variation - USELESS!

// CORRECT - Filter sweep with wide range:
.lpf("<400 1000 4000 2000>".slow(16))

// For buildups use saw:
.lpf(saw.range(400, 4000).slow(16))  // Ramps 400→4000 over 16 beats
```

### SECTION STRUCTURE (64 beats total)

```
Beats 0-16:   INTRO    - quiet, filtered (gain 0.2-0.3, lpf low)
Beats 16-32:  BUILDUP  - rising energy (gain 0.4-0.5, lpf sweeping up)
Beats 32-48:  DROP     - full power (gain 0.7-1.0, lpf open)
Beats 48-64:  OUTRO    - wind down (gain decreasing)
```

### Example Effect Functions

```javascript
setcps({cps:.3f})

// Bass: Quiet intro → loud drop
let bassFx = p => p.sound("sawtooth")
  .gain("<0.2 0.3 0.5 0.4>".slow(16))  // Section-based gain
  .lpf(saw.range(300, 800).slow(32))   // Filter opens over 32 beats
  .hpf(40)

// Mid: Buildup with filter sweep
let midFx = p => p.sound("triangle")
  .gain("<0.3 0.5 0.9 0.6>".slow(16))  // Intro→build→drop→outro
  .lpf("<2000 3000 6000 4000>".slow(16))

// High: Comes in at drop
let highFx = p => p.sound("square")
  .gain("<0 0.2 0.6 0.3>".slow(16))    // Silent intro, appears at drop
  .lpf(8000)

// Drums: Builds energy
let drumsFx = p => p.bank("RolandTR808")
  .gain("<0.5 0.7 1.0 0.8>".slow(16))
  .room("<0.3 0.2 0.1 0.2>".slow(16))  // Less reverb at drop (punchy)
```

### PATTERN SYNTAX (USE IN THIS ORDER OF PREFERENCE):
1. `"<v1 v2 v3 v4>".slow(16)` - **PRIMARY** - Discrete section values (ALWAYS use for gain!)
2. `saw.range(min, max).slow(n)` - Buildups only (min→max ramp)
3. `sine.range(min, max).slow(n)` - Filter movement only, NOT for gain
4. ~~perlin~~ - DO NOT USE for gain (too subtle)

### MANDATORY RULES:
1. **GAIN MUST use `"<...>".slow(16)`** - NO perlin, NO sine for gain!
2. **MINIMUM RANGE 0.3** - Values must differ by at least 0.3 (e.g., 0.2→0.5→0.8)
3. Bass: gain `"<0.1 0.2 0.5 0.3>".slow(16)`, lpf 300-800
4. Mid: gain `"<0.3 0.5 0.9 0.6>".slow(16)`, lpf 2000-8000
5. High: gain `"<0 0.2 0.7 0.4>".slow(16)` (can start silent)
6. Drums: gain `"<0.5 0.7 1.0 0.7>".slow(16)`
7. Filter OPENS as energy increases (low lpf = dark intro, high lpf = bright drop)

Output ONLY the effect functions in a javascript code block, no explanation.'''


def analyze_with_ollama(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    model: str = None,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = "",
    bpm: float = 120
) -> Dict[str, Any]:
    """Use Ollama (local LLM) to analyze results and suggest improvements."""

    if not HAS_REQUESTS:
        print("Warning: requests package not installed", file=sys.stderr)
        return {"suggestions": [], "improved_code": original_code, "reasoning": "No requests"}

    ollama_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist, bpm=bpm
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

        # Method 3: Extract code AND automation from markdown code blocks
        # Look for javascript/js block (Strudel code)
        code_match = re.search(r'```(?:scss|javascript|js)?\n?([\s\S]*?)```', result_text)
        # Look for json block (automation timeline)
        automation_match = re.search(r'```json\n?([\s\S]*?)```', result_text)

        if code_match:
            extracted_code = code_match.group(1).strip()
            # Check if it looks like Strudel code (effect functions, sound, or setcps)
            if 'Fx' in extracted_code or 'sound(' in extracted_code or 'setcps(' in extracted_code:
                result = {
                    "analysis": "Extracted from response",
                    "suggestions": ["Code extracted from markdown"],
                    "improved_code": extracted_code
                }

                # Try to extract automation timeline if present
                if automation_match:
                    try:
                        automation_text = automation_match.group(1).strip()
                        automation_json = json.loads(automation_text)
                        result["automation"] = automation_json
                        result["suggestions"].append("Automation timeline extracted")
                    except json.JSONDecodeError as e:
                        print(f"       Warning: Could not parse automation JSON: {e}")

                return result

        # Method 4: Look for let *Fx patterns and setcps directly in text
        fx_matches = re.findall(r'let\s+\w+Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*', result_text)
        setcps_match = re.search(r'setcps\([^)]+\)', result_text)

        if fx_matches or setcps_match:
            code_parts = []
            if setcps_match:
                code_parts.append(setcps_match.group())
            code_parts.extend(fx_matches)

            result = {
                "analysis": "Extracted effect functions with tempo control",
                "suggestions": ["Effect functions and setcps extracted from response"],
                "improved_code": '\n'.join(code_parts)
            }

            # Try to extract automation from json block
            if automation_match:
                try:
                    automation_text = automation_match.group(1).strip()
                    automation_json = json.loads(automation_text)
                    result["automation"] = automation_json
                except json.JSONDecodeError:
                    pass

            return result

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
    artist: str = "",
    bpm: float = 120
) -> Dict[str, Any]:
    """Use Claude API to analyze results and suggest improvements."""

    if Anthropic is None:
        return None  # Signal to try Ollama

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # Signal to try Ollama

    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist, bpm=bpm
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
    artist: str = "",
    bpm: float = 120,
    agent: 'OllamaAgent' = None  # Agentic mode with memory
) -> Dict[str, Any]:
    """
    Analyze and improve code using available LLM.

    Priority:
    1. If agent provided, use agentic Ollama with memory (PREFERRED)
    2. If use_ollama=True, use Ollama directly (stateless)
    3. Try Claude API if ANTHROPIC_API_KEY is set
    4. Fall back to Ollama (local)

    Now enhanced with spectrogram_insights for deeper analysis.
    Genre and artist context helps with sound selection.
    BPM is used for tempo control in generated code.
    """

    # AGENTIC MODE - uses persistent memory, ClickHouse queries
    if agent is not None and HAS_AGENT:
        print("       Using Agentic Ollama (with memory + ClickHouse)...")
        context = {
            "genre": genre,
            "bpm": bpm,
            "similarity": previous_run.get("similarity_overall", 0),
            "band_bass": previous_run.get("band_bass", 0) + previous_run.get("band_sub_bass", 0),
            "band_mid": previous_run.get("band_mid", 0) + previous_run.get("band_low_mid", 0),
            "band_high": previous_run.get("band_high", 0) + previous_run.get("band_high_mid", 0),
        }
        response = agent.generate(context)
        code = agent.extract_code(response)
        # Check if code was rejected by validation
        analysis = response[:500]
        if not code and hasattr(agent, 'last_validation_error') and agent.last_validation_error:
            analysis = f"REJECTED: {agent.last_validation_error}"
        return {
            "analysis": analysis,
            "suggestions": ["Agentic generation with memory"],
            "improved_code": code
        }

    if use_ollama:
        print("       Using Ollama (local LLM)...")
        return analyze_with_ollama(
            previous_run, learned_knowledge, original_code,
            spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm
        )

    # Try Claude first
    result = analyze_with_claude(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm
    )
    if result is not None:
        return result

    # Fall back to Ollama
    print("       Claude unavailable, using Ollama (local LLM)...")
    return analyze_with_ollama(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm
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
    track_hash: str,
    genre: str,
    bpm: float,
    key_type: str,
    parameter_name: str,
    old_value: str,
    new_value: str,
    improvement: float
):
    """Store learned knowledge for this specific track."""
    data = {
        "track_hash": track_hash,
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

    # Create agentic Ollama wrapper with memory
    agent = None
    if HAS_AGENT and use_ollama:
        print(f"  [Agentic Mode] Creating agent with ClickHouse access...")
        agent = OllamaAgent(track_hash)
        print(f"  [Agentic Mode] Agent ready (history: {len(agent.messages)} messages)")

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

    # PHASE 2: Generate dynamic effects with orchestrator (beat-synced patterns)
    if HAS_ORCHESTRATOR:
        print("\n--- Phase 2: Generating beat-synced dynamic effects ---")
        current_code = generate_orchestrated_effects(current_code, metadata, comparison=None)
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
            # Write best code to strudel_path so renders use it
            with open(strudel_path, 'w') as f:
                f.write(best_ever_code)
            print(f"   Updated {strudel_path} with best known code")
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

    # Track automation timeline across iterations (starts None, updated by LLM)
    current_automation_path = None

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
            # Pass automation timeline if available (beat-synced parameter changes)
            if current_automation_path and current_automation_path.exists():
                render_cmd.extend(["--automation", str(current_automation_path)])
                print(f"       Using automation timeline: {current_automation_path.name}")
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
                track_hash=track_hash,
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

        # Extract genre, artist, and BPM from metadata for sound/tempo selection
        genre = metadata.get("genre", "")
        artist_context = artist if artist else metadata.get("artist", "")
        track_bpm = metadata.get("bpm", 120)

        # Update agent with iteration results (for learning)
        if agent is not None:
            improved = current_similarity > best_similarity
            agent.add_iteration_result(
                iteration=iteration + 1,
                version=current_version,
                similarity=current_similarity,
                band_diffs=band_diffs,
                code_generated=current_code,
                improved=improved
            )

        ai_result = analyze_with_llm(
            run_data, comparison, knowledge, current_code,
            use_ollama=use_ollama, spectrogram_insights=spectrogram_insights,
            genre=genre, artist=artist_context, bpm=track_bpm,
            agent=agent  # Pass agent for agentic mode
        )

        print(f"Analysis: {ai_result.get('analysis', 'N/A')[:100]}...")
        print(f"Suggestions: {ai_result.get('suggestions', [])[:3]}")

        # Save automation timeline if LLM generated one
        if ai_result.get("automation"):
            automation_path = Path(output_dir) / f"automation_v{current_version:03d}.json"
            with open(automation_path, 'w') as f:
                json.dump(ai_result["automation"], f, indent=2)
            print(f"       Saved automation timeline to {automation_path.name}")
            # Also save as latest automation.json
            latest_automation = Path(output_dir) / "automation.json"
            with open(latest_automation, 'w') as f:
                json.dump(ai_result["automation"], f, indent=2)
            # Update for next iteration's render
            current_automation_path = automation_path

        # 6. Update code for next iteration
        improved_effects = ai_result.get("improved_code", "")

        # Handle validation failures or empty code gracefully
        if not improved_effects or not improved_effects.strip():
            # Check if this is a validation failure (code was rejected)
            if "REJECTED" in ai_result.get("analysis", ""):
                print("       ⚠ LLM generated invalid Strudel code - skipping iteration")
                current_version += 1
                continue  # Skip this iteration, try again
            else:
                # This is a real bug - LLM didn't return any code
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
            else:
                # REGRESSION PREVENTION: Check if this is actually better
                # We need to render and compare before accepting
                test_path = Path(output_dir) / f"render_test.wav"
                test_strudel = Path(output_dir) / f"output_test.strudel"
                with open(test_strudel, 'w') as f:
                    f.write(improved_code)

                # Quick render to test
                node_renderer = Path(__file__).parent.parent / "node" / "dist" / "render-strudel-node.js"
                if node_renderer.exists():
                    test_cmd = ["node", str(node_renderer), str(test_strudel), str(test_path),
                               "--duration", f"{min(30, exact_duration):.2f}"]
                    if synth_config_path.exists():
                        test_cmd.extend(["--config", str(synth_config_path)])
                    # Include automation for test render too
                    if current_automation_path and current_automation_path.exists():
                        test_cmd.extend(["--automation", str(current_automation_path)])
                    subprocess.run(test_cmd, capture_output=True)

                    # Quick compare
                    if test_path.exists():
                        test_compare_cmd = [
                            sys.executable,
                            str(Path(__file__).parent / "compare_audio.py"),
                            original_audio, str(test_path),
                            "-d", f"{min(30, exact_duration):.2f}",
                            "--json", "--quiet"  # Output JSON, suppress logs
                        ]
                        test_result = subprocess.run(test_compare_cmd, capture_output=True, text=True)
                        try:
                            test_comparison = json.loads(test_result.stdout)
                            test_similarity = test_comparison.get("comparison", {}).get("overall_similarity", 0)

                            if test_similarity > best_similarity:
                                print(f"       ✓ IMPROVEMENT: {best_similarity*100:.1f}% → {test_similarity*100:.1f}%")
                                best_similarity = test_similarity
                                best_code = improved_code
                                current_code = improved_code
                                # Save improved code
                                improved_path = Path(output_dir) / f"output_v{current_version + 1:03d}.strudel"
                                with open(improved_path, 'w') as f:
                                    f.write(improved_code)
                                with open(strudel_path, 'w') as f:
                                    f.write(improved_code)
                                print(f"       Saved improved code to {improved_path}")
                            else:
                                print(f"       ✗ REGRESSION: {best_similarity*100:.1f}% → {test_similarity*100:.1f}% - REVERTING")
                                current_code = best_code  # Revert to best
                                with open(strudel_path, 'w') as f:
                                    f.write(best_code)
                        except json.JSONDecodeError:
                            print("       ⚠ Could not parse test comparison, accepting change")
                            current_code = improved_code
                            # BUG FIX: Actually save the improved code to file
                            with open(strudel_path, 'w') as f:
                                f.write(improved_code)
                    else:
                        print("       ⚠ Test render failed, accepting change")
                        current_code = improved_code
                        # BUG FIX: Actually save the improved code to file
                        with open(strudel_path, 'w') as f:
                            f.write(improved_code)
                else:
                    # No Node renderer, accept change blindly
                    current_code = improved_code
                    improved_path = Path(output_dir) / f"output_v{current_version + 1:03d}.strudel"
                    with open(improved_path, 'w') as f:
                        f.write(improved_code)
                    with open(strudel_path, 'w') as f:
                        f.write(improved_code)
                    print(f"       Saved improved code to {improved_path}")
        else:
            print("       No code changes this iteration, continuing...")

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
