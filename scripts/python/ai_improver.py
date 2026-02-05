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
    """Get relevant learned knowledge for this track context."""
    query = f"""
        SELECT parameter_name, parameter_new_value, similarity_improvement, confidence
        FROM midi_grep.knowledge
        WHERE (genre = '{genre}' OR genre = '')
          AND bpm_range_low <= {bpm}
          AND bpm_range_high >= {bpm}
          AND (key_type = '{key_type}' OR key_type = '')
          AND confidence > 0.5
        ORDER BY similarity_improvement DESC
        LIMIT 10
    """
    return clickhouse_query(query)


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
    original_code: str
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

    # Include learned knowledge if available
    knowledge_str = ""
    if learned_knowledge:
        knowledge_str = "\n\nPREVIOUS LEARNINGS:\n"
        for k in learned_knowledge[:5]:
            knowledge_str += f"- {k.get('parameter_name')}: {k.get('parameter_old_value')} -> {k.get('parameter_new_value')} (improved {k.get('similarity_improvement', 0)*100:.0f}%)\n"

    # Build the research-oriented prompt - give LLM full control
    return f'''You are an audio mixing AI. Your goal: make the rendered audio match the original.

COMPARISON DATA (rendered minus original):
- Bass: {band_bass*100:+.0f}% (positive=too loud, negative=too quiet)
- Mid: {band_mid*100:+.0f}%
- High: {band_high*100:+.0f}%
- Brightness: {brightness_ratio:.0%} of original
- Energy: {energy_ratio:.0%} of original

CURRENT EFFECT FUNCTIONS:
{effects_code}
{knowledge_str}
AVAILABLE EFFECTS:
- .gain(0.01-2.0) - volume
- .hpf(20-2000) - high pass filter (removes bass)
- .lpf(200-20000) - low pass filter (removes treble)
- .crush(1-16) - bit depth (lower=more distortion)
- .coarse(1-16) - sample rate reduction
- .room(0-1) - reverb amount
- .delay(0-1) - delay wet mix

EXAMPLE OUTPUT (what I expect):
let bassFx = p => p.sound("gm_electric_bass_finger").gain(0.15).hpf(60).lpf(500)
let midFx = p => p.sound("gm_epiano2").gain(1.2).lpf(5000)
let highFx = p => p.sound("gm_music_box").gain(0.8).lpf(12000)
let drumsFx = p => p.bank("RolandTR808").gain(0.9)

TASK: Output your improved effect functions with SPECIFIC NUMERIC VALUES.
Think about the frequency band data and adjust accordingly.

Output ONLY the 4 lines (no comments, no explanation):'''


def analyze_with_ollama(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    model: str = None
) -> Dict[str, Any]:
    """Use Ollama (local LLM) to analyze results and suggest improvements."""

    if not HAS_REQUESTS:
        print("Warning: requests package not installed", file=sys.stderr)
        return {"suggestions": [], "improved_code": original_code, "reasoning": "No requests"}

    ollama_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    prompt = build_improvement_prompt(previous_run, learned_knowledge, original_code)

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
    original_code: str
) -> Dict[str, Any]:
    """Use Claude API to analyze results and suggest improvements."""

    if Anthropic is None:
        return None  # Signal to try Ollama

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # Signal to try Ollama

    prompt = build_improvement_prompt(previous_run, learned_knowledge, original_code)
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
    use_ollama: bool = False
) -> Dict[str, Any]:
    """
    Analyze and improve code using available LLM.

    Priority:
    1. If use_ollama=True, use Ollama directly
    2. Try Claude API if ANTHROPIC_API_KEY is set
    3. Fall back to Ollama (local)
    4. Return unchanged code if nothing works
    """

    if use_ollama:
        print("       Using Ollama (local LLM)...")
        return analyze_with_ollama(previous_run, learned_knowledge, original_code)

    # Try Claude first
    result = analyze_with_claude(previous_run, learned_knowledge, original_code)
    if result is not None:
        return result

    # Fall back to Ollama
    print("       Claude unavailable, using Ollama (local LLM)...")
    return analyze_with_ollama(previous_run, learned_knowledge, original_code)


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

    # Check for previous runs
    previous_run = get_previous_run(track_hash)
    if previous_run:
        current_version = previous_run["version"] + 1
        print(f"\nFound previous run: v{previous_run['version']} with {previous_run['similarity_overall']*100:.1f}% similarity")
    else:
        current_version = 1
        print(f"\nNo previous runs found, starting fresh")

    # Get learned knowledge
    key_type = "minor" if "minor" in metadata.get("key", "").lower() else "major"
    knowledge = get_learned_knowledge(
        metadata.get("genre", ""),
        metadata.get("bpm", 120),
        key_type
    )
    if knowledge:
        print(f"Loaded {len(knowledge)} knowledge items for this context")

    best_similarity = 0
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

        # 4. Check if target reached
        if current_similarity >= target_similarity:
            print(f"\n✓ Target similarity reached!")
            best_similarity = current_similarity
            best_code = current_code
            break

        # Track best
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_code = current_code

        # 5. Get AI suggestions for improvement
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
            "genre": metadata.get("genre", "")
        }

        print(f"       Bands: bass={band_diffs.get('bass',0)*100:+.0f}% mid={band_diffs.get('mid',0)*100:+.0f}% high={band_diffs.get('high',0)*100:+.0f}%")
        print(f"       Brightness: {brightness_ratio:.0%}  Energy: {energy_ratio:.0%}")

        ai_result = analyze_with_llm(run_data, comparison, knowledge, current_code, use_ollama=use_ollama)

        print(f"Analysis: {ai_result.get('analysis', 'N/A')[:100]}...")
        print(f"Suggestions: {ai_result.get('suggestions', [])[:3]}")

        # 6. Update code for next iteration
        improved_effects = ai_result.get("improved_code", "")

        # If LLM gave suggestions but no code, generate code from frequency band data
        if not improved_effects or not improved_effects.strip():
            import re as regex

            # Extract current gains from code (try multiple patterns)
            bass_gain_m = regex.search(r'bassFx.*?\.gain\(([0-9.]+)\)', current_code)
            mid_gain_m = regex.search(r'midFx.*?\.gain\(([0-9.]+)\)', current_code)
            high_gain_m = regex.search(r'highFx.*?\.gain\(([0-9.]+)\)', current_code)
            drums_gain_m = regex.search(r'drumsFx.*?\.gain\(([0-9.]+)\)', current_code)

            bass_g = float(bass_gain_m.group(1)) if bass_gain_m else 0.15
            mid_g = float(mid_gain_m.group(1)) if mid_gain_m else 1.2
            high_g = float(high_gain_m.group(1)) if high_gain_m else 0.8
            drums_g = float(drums_gain_m.group(1)) if drums_gain_m else 0.9

            # Aggressive adjustments based on band differences
            # Positive band_diff = too loud, reduce gain
            # Negative band_diff = too quiet, increase gain
            # For large deficits (like -76%), we need aggressive changes
            bass_diff = band_diffs.get('bass', 0) + band_diffs.get('sub_bass', 0) * 0.5
            mid_diff = band_diffs.get('mid', 0) + band_diffs.get('low_mid', 0) * 0.5
            high_diff = band_diffs.get('high', 0) + band_diffs.get('high_mid', 0) * 0.5

            # Scale adjustment by deficit magnitude (more aggressive for large deficits)
            bass_adj = max(0.5, min(2.0, 1.0 - bass_diff * 1.5))
            mid_adj = max(0.5, min(2.5, 1.0 - mid_diff * 1.2))  # More aggressive for mid
            high_adj = max(0.5, min(2.0, 1.0 - high_diff * 1.0))
            energy_adj = max(0.7, min(1.5, 1.0 / max(energy_ratio, 0.3))) if energy_ratio < 0.7 else 1.0

            new_bass = max(0.05, min(0.5, bass_g * bass_adj))
            new_mid = max(0.5, min(3.0, mid_g * mid_adj * energy_adj))  # Allow higher mid gain
            new_high = max(0.3, min(1.5, high_g * high_adj))
            new_drums = max(0.5, min(1.2, drums_g * energy_adj))

            print(f"       Computing from bands: bass={new_bass:.2f} mid={new_mid:.2f} high={new_high:.2f} drums={new_drums:.2f}")

            # Extract current sounds
            bass_sound_m = regex.search(r'bassFx.*?\.sound\("([^"]+)"\)', current_code)
            mid_sound_m = regex.search(r'midFx.*?\.sound\("([^"]+)"\)', current_code)
            high_sound_m = regex.search(r'highFx.*?\.sound\("([^"]+)"\)', current_code)
            drums_bank_m = regex.search(r'drumsFx.*?\.bank\("([^"]+)"\)', current_code)

            bass_sound = bass_sound_m.group(1) if bass_sound_m else "gm_electric_bass_finger"
            mid_sound = mid_sound_m.group(1) if mid_sound_m else "gm_epiano2"
            high_sound = high_sound_m.group(1) if high_sound_m else "gm_music_box"
            drums_bank = drums_bank_m.group(1) if drums_bank_m else "RolandTR808"

            # Extract current filter values
            bass_hpf_m = regex.search(r'bassFx.*?\.hpf\(([0-9]+)\)', current_code)
            bass_lpf_m = regex.search(r'bassFx.*?\.lpf\(([0-9]+)\)', current_code)
            mid_lpf_m = regex.search(r'midFx.*?\.lpf\(([0-9]+)\)', current_code)
            high_lpf_m = regex.search(r'highFx.*?\.lpf\(([0-9]+)\)', current_code)

            bass_hpf = int(bass_hpf_m.group(1)) if bass_hpf_m else 60
            bass_lpf = int(bass_lpf_m.group(1)) if bass_lpf_m else 500
            mid_lpf = int(mid_lpf_m.group(1)) if mid_lpf_m else 5000
            high_lpf = int(high_lpf_m.group(1)) if high_lpf_m else 12000

            # Adjust filters based on brightness
            if brightness_ratio < 0.8:  # Too dark, open up filters
                mid_lpf = min(8000, mid_lpf + 1000)
                high_lpf = min(16000, high_lpf + 2000)
            elif brightness_ratio > 1.3:  # Too bright, close filters
                mid_lpf = max(3000, mid_lpf - 1000)
                high_lpf = max(8000, high_lpf - 2000)

            # Generate improved effects
            improved_effects = f'''let bassFx = p => p.sound("{bass_sound}").gain({new_bass:.2f}).hpf({bass_hpf}).lpf({bass_lpf})
let midFx = p => p.sound("{mid_sound}").gain({new_mid:.2f}).lpf({mid_lpf})
let highFx = p => p.sound("{high_sound}").gain({new_high:.2f}).lpf({high_lpf})
let drumsFx = p => p.bank("{drums_bank}").gain({new_drums:.2f})'''

        if improved_effects and improved_effects.strip():
            # Merge improved effects back into full code
            improved_code = merge_effect_functions(current_code, improved_effects)
            if improved_code == current_code:
                print("No effective changes made, stopping")
                break
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
            print("No code changes suggested, stopping")
            break

        current_version += 1

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
        version_num = Path(output_dir).name.replace("v", "")
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
        "genre": args.genre
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
