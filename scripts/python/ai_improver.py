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


def build_improvement_prompt(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str
) -> str:
    """Build the prompt for LLM analysis."""
    return f"""You are an expert in Strudel live coding and audio production. Analyze the following comparison results and improve the Strudel code.

## Previous Run Results
- Overall Similarity: {previous_run.get('similarity_overall', 0)*100:.1f}%
- MFCC (Timbre): {previous_run.get('similarity_mfcc', 0)*100:.1f}%
- Chroma (Pitch): {previous_run.get('similarity_chroma', 0)*100:.1f}%
- Frequency Balance: {previous_run.get('similarity_frequency', 0)*100:.1f}%

## Frequency Band Differences (positive = rendered has MORE than original)
- Sub-bass (20-60Hz): {previous_run.get('band_sub_bass', 0)*100:+.1f}%
- Bass (60-250Hz): {previous_run.get('band_bass', 0)*100:+.1f}%
- Low-mid (250-500Hz): {previous_run.get('band_low_mid', 0)*100:+.1f}%
- Mid (500-2kHz): {previous_run.get('band_mid', 0)*100:+.1f}%
- High-mid (2-4kHz): {previous_run.get('band_high_mid', 0)*100:+.1f}%
- High (4-20kHz): {previous_run.get('band_high', 0)*100:+.1f}%

## Track Context
- BPM: {previous_run.get('bpm', 120)}
- Key: {previous_run.get('key', 'unknown')}
- Style: {previous_run.get('style', 'auto')}
- Genre: {previous_run.get('genre', 'unknown')}

## Learned Knowledge (what has worked before for similar tracks)
{json.dumps(learned_knowledge, indent=2) if learned_knowledge else "No prior knowledge available"}

## Current Strudel Code
```javascript
{original_code}
```

## Your Task
1. Analyze what's causing the similarity gaps
2. Suggest specific changes to the Strudel code to improve similarity
3. Generate improved Strudel code

Focus on:
- If bass is too loud (+), reduce bass voice gain or add HPF
- If highs are missing (-), reduce LPF cutoff or add high frequencies
- If rhythm similarity is low, adjust note timing or swing
- If timbre doesn't match, adjust effects (reverb, filter, distortion)

Respond in JSON format:
{{
    "analysis": "Brief analysis of what's wrong",
    "suggestions": ["Specific change 1", "Specific change 2", ...],
    "improved_code": "The full improved Strudel code",
    "expected_improvement": "What metrics should improve"
}}
"""


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

        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        # Try to find JSON object in text
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_text = json_match.group()

        result = json.loads(result_text)
        return result

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
    data = {
        "track_hash": track_hash,
        "track_name": track_name,
        "version": version,
        "bpm": bpm,
        "key": key,
        "style": style,
        "genre": genre,
        "strudel_code": strudel_code,
        "similarity_overall": comparison.get("overall", 0),
        "similarity_mfcc": comparison.get("mfcc", 0),
        "similarity_chroma": comparison.get("chroma", 0),
        "similarity_frequency": comparison.get("frequency", 0),
        "similarity_rhythm": comparison.get("rhythm", 0),
        "band_sub_bass": comparison.get("band_diffs", {}).get("sub_bass", 0),
        "band_bass": comparison.get("band_diffs", {}).get("bass", 0),
        "band_low_mid": comparison.get("band_diffs", {}).get("low_mid", 0),
        "band_mid": comparison.get("band_diffs", {}).get("mid", 0),
        "band_high_mid": comparison.get("band_diffs", {}).get("high_mid", 0),
        "band_high": comparison.get("band_diffs", {}).get("high", 0),
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
    Main improvement loop.

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

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} (v{current_version}) ---")

        # 1. Render current code
        render_path = Path(output_dir) / f"render_v{current_version:03d}.wav"
        render_cmd = [
            sys.executable,
            str(Path(__file__).parent / "render_audio.py"),
            strudel_path,
            "-o", str(render_path),
            "-d", "60"
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
            "--output-json", str(Path(output_dir) / f"comparison_v{current_version:03d}.json")
        ]
        subprocess.run(compare_cmd, capture_output=True)

        comparison_path = Path(output_dir) / f"comparison_v{current_version:03d}.json"
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
        else:
            print("Comparison failed, using defaults")
            comparison = {"overall": 0, "mfcc": 0, "chroma": 0, "frequency": 0}

        current_similarity = comparison.get("overall", 0)
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
        run_data = {
            "similarity_overall": current_similarity,
            "similarity_mfcc": comparison.get("mfcc", 0),
            "similarity_chroma": comparison.get("chroma", 0),
            "similarity_frequency": comparison.get("frequency", 0),
            "band_sub_bass": comparison.get("band_diffs", {}).get("sub_bass", 0),
            "band_bass": comparison.get("band_diffs", {}).get("bass", 0),
            "band_low_mid": comparison.get("band_diffs", {}).get("low_mid", 0),
            "band_mid": comparison.get("band_diffs", {}).get("mid", 0),
            "band_high_mid": comparison.get("band_diffs", {}).get("high_mid", 0),
            "band_high": comparison.get("band_diffs", {}).get("high", 0),
            "bpm": metadata.get("bpm", 120),
            "key": metadata.get("key", ""),
            "style": metadata.get("style", ""),
            "genre": metadata.get("genre", "")
        }

        ai_result = analyze_with_llm(run_data, comparison, knowledge, current_code, use_ollama=use_ollama)

        print(f"Analysis: {ai_result.get('analysis', 'N/A')[:100]}...")
        print(f"Suggestions: {ai_result.get('suggestions', [])[:3]}")

        # 6. Update code for next iteration
        improved_code = ai_result.get("improved_code", current_code)
        if improved_code and improved_code != current_code:
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
