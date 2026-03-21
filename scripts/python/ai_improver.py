#!/usr/bin/env python3
"""
AI-Driven Strudel Code Improver — Orchestrator

Coordinates the iterative improvement loop. Delegates to:
- clickhouse_store.py: Database operations and learning storage
- llm_client.py: LLM prompt building and API calls
- strudel_params.py: Strudel code parameter extraction and optimization
"""

import argparse
import json
import os
import re
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

# ClickHouse connection (kept for backward compat, primary definitions in clickhouse_store.py)
CLICKHOUSE_BIN = Path(__file__).parent.parent.parent / "bin" / "clickhouse"
CLICKHOUSE_DB = Path(__file__).parent.parent.parent / ".clickhouse" / "db"

# Import extracted modules (with fallbacks for backward compat)
try:
    from clickhouse_store import (
        get_track_hash, sanitize_sql_value, clickhouse_query, clickhouse_insert,
        get_previous_run, get_best_run, get_learned_knowledge,
        retrieve_relevant_knowledge, find_similar_tracks,
        store_run, store_knowledge, learn_from_improvement,
        normalize_artist, get_artist_presets, detect_artist_from_path,
        get_genre_presets, _ensure_clickhouse, HAS_CLICKHOUSE,
    )
    HAS_CH_MODULE = True
except ImportError:
    HAS_CH_MODULE = False

try:
    from strudel_params import (
        extract_parameters_from_code, apply_genre_presets,
        extract_effect_functions, merge_effect_functions,
        optimize_parameters, generate_orchestrated_effects,
    )
    HAS_PARAMS_MODULE = True
except ImportError:
    HAS_PARAMS_MODULE = False

try:
    from llm_client import (
        build_improvement_prompt, analyze_with_ollama,
        analyze_with_claude, analyze_with_llm,
        build_constrained_llm_prompt, OLLAMA_URL, DEFAULT_OLLAMA_MODEL,
    )
    HAS_LLM_MODULE = True
except ImportError:
    HAS_LLM_MODULE = False

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

# ai_code_improver removed (dead code) — gap analysis is now inline

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

# Import syntax fixer and voice enforcer from codegen
try:
    from ollama_codegen import fix_strudel_syntax, enforce_three_voices
    HAS_SYNTAX_FIXER = True
except ImportError:
    HAS_SYNTAX_FIXER = False

# Import agentic Ollama wrapper
try:
    from ollama_agent import OllamaAgent, INVALID_GM_PATTERNS, VALID_SOUNDS
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False
    INVALID_GM_PATTERNS = []
    VALID_SOUNDS = set()

# Import genre-aware sound RAG
try:
    from sound_selector import retrieve_genre_context
    HAS_SOUND_SELECTOR = True
except ImportError:
    HAS_SOUND_SELECTOR = False


def _has_invalid_sounds(code: str) -> bool:
    """Check if Strudel code contains hallucinated/invalid sound names."""
    if not VALID_SOUNDS:
        return False
    for pattern in INVALID_GM_PATTERNS:
        if re.search(pattern, code):
            return True
    # Check .sound("...") calls against valid sounds
    for m in re.finditer(r'\.sound\(["\']([^"\']+)["\']\)', code):
        sound = m.group(1)
        # Skip alternation patterns like "<sound1 sound2>"
        if sound.startswith('<'):
            for s in sound.strip('<>').split():
                if s and s not in VALID_SOUNDS:
                    return True
        elif sound not in VALID_SOUNDS:
            return True
    return False

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "midi-grep-strudel"


def map_windows_to_sections(windowed: dict, sections: list) -> list:
    """Map per-window comparison data to section indices.

    Args:
        windowed: dict of stem_name -> list of window dicts with time_start, time_end, similarity, issues
        sections: list of section dicts with start, end

    Returns:
        list of dicts sorted worst-first: section_idx, stem, similarity, issues
    """
    section_scores = []
    for sec_idx, sec in enumerate(sections):
        s, e = sec.get("start", 0), sec.get("end", 0)
        for stem_name, windows in windowed.items():
            if not isinstance(windows, list):
                continue
            overlapping = [w for w in windows if w.get("time_start", 0) < e and w.get("time_end", 0) > s]
            if overlapping:
                avg_sim = sum(w.get("similarity", 0) for w in overlapping) / len(overlapping)
                issues = list(set(
                    issue for w in overlapping for issue in w.get("issues", [])
                ))[:3]
                section_scores.append({
                    "section_idx": sec_idx,
                    "stem": stem_name,
                    "similarity": round(avg_sim, 3),
                    "issues": issues
                })
    # Sort worst first
    section_scores.sort(key=lambda x: x["similarity"])
    return section_scores


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



# Functions moved to clickhouse_store.py, strudel_params.py, llm_client.py
# Import them at the top of this file.


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



# generate_orchestrated_effects, optimize_parameters, etc. moved to strudel_params.py
# build_improvement_prompt, analyze_with_*, etc. moved to llm_client.py


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

    # Apply genre/artist presets (only for old effect-function format, not arrange())
    genre = metadata.get("genre", "")
    artist = metadata.get("artist", "")
    if not artist:
        artist = detect_artist_from_path(original_audio)

    if 'arrange(' not in current_code:
        if genre:
            presets = get_genre_presets(genre)
            if presets:
                print(f"\n--- Applying {genre} genre presets ---")
                for fx_name, params in presets.items():
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    print(f"  {fx_name}: {param_str}")
                current_code = apply_genre_presets(current_code, presets)
                with open(strudel_path, 'w') as f:
                    f.write(current_code)

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
    else:
        print("\n--- Skipping genre/artist presets (code uses arrange() format) ---")

    # PHASE 2: Generate dynamic effects with orchestrator (only for old effect-function format)
    if HAS_ORCHESTRATOR and 'arrange(' not in current_code:
        print("\n--- Phase 2: Generating beat-synced dynamic effects ---")
        current_code = generate_orchestrated_effects(current_code, metadata, comparison=None)
        with open(strudel_path, 'w') as f:
            f.write(current_code)
    elif 'arrange(' in current_code:
        print("\n--- Phase 2: Skipped (code already uses arrange() format) ---")

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

    # Get the BEST run ever from ClickHouse for regression baseline only.
    # IMPORTANT: Do NOT overwrite current_code with ClickHouse best!
    # Fresh codegen output (with Genre RAG sounds) must be the starting point.
    # ClickHouse best is used only to set the regression baseline so that
    # if fresh code scores worse, regression revert kicks in naturally.
    best_run = get_best_run(track_hash)
    if best_run and best_run.get("strudel_code"):
        best_ever_similarity = best_run.get("similarity_overall", 0)
        best_ever_code = best_run.get("strudel_code", "")
        best_ever_version = best_run.get("version", 0)
        print(f"  Best ever: v{best_ever_version} with {best_ever_similarity*100:.1f}% similarity (from ClickHouse, baseline only)")
    else:
        best_ever_similarity = 0
        best_ever_code = None

    # Get learned knowledge (legacy genre-based)
    key_type = "minor" if "minor" in metadata.get("key", "").lower() else "major"
    knowledge = get_learned_knowledge(
        metadata.get("genre", ""),
        metadata.get("bpm", 120),
        key_type
    )
    if knowledge:
        print(f"Loaded {len(knowledge)} knowledge items for this context")

    # Cross-track RAG: find spectrally similar tracks for code inspiration
    if HAS_CH_MODULE and not best_run:
        initial_comparison_path = Path(output_dir) / "comparison.json"
        if initial_comparison_path.exists():
            with open(initial_comparison_path) as f:
                init_comp = json.load(f)
            orig_bands = init_comp.get("original", {}).get("bands", {})
            if orig_bands:
                similar = find_similar_tracks(
                    orig_bands, metadata.get("genre", ""),
                    metadata.get("bpm", 120), track_hash
                )
                if similar:
                    print(f"  Found {len(similar)} spectrally similar tracks for reference")

    best_similarity = previous_similarity if best_run else 0
    best_code = current_code
    best_render_path = None  # Track render file for best code

    # Get exact duration from original audio (millisecond precision)
    exact_duration = get_audio_duration(original_audio)
    print(f"Original audio duration: {exact_duration:.6f}s")

    # Load initial stem comparison for per-stem feedback (used in iteration prompts)
    # NOTE: Do NOT use weighted_overall as best_similarity baseline — it's a different metric
    # than compare_audio.py overall_similarity. Iteration 0 renders the initial code and
    # establishes the baseline using the same metric as subsequent iterations.
    stem_comparison_path = Path(output_dir) / "stem_comparison.json"
    if stem_comparison_path.exists():
        with open(stem_comparison_path) as f:
            initial_stem_comparison = json.load(f)
    else:
        initial_stem_comparison = {}

    # Track automation timeline across iterations (starts None, updated by LLM)
    current_automation_path = None

    # Deterministic optimizer state
    deterministic_converged = False
    deterministic_prev_similarity = 0.0

    # Load section boundaries for temporal mapping
    sections = []
    smart_analysis_path = Path(output_dir).parent / "smart_analysis.json"
    if not smart_analysis_path.exists():
        # Also check inside the output dir
        smart_analysis_path = Path(output_dir) / "smart_analysis.json"
    if smart_analysis_path.exists():
        try:
            with open(smart_analysis_path) as f:
                sections = json.load(f).get("sections", [])
            if sections:
                print(f"Loaded {len(sections)} sections for temporal mapping")
        except Exception as e:
            print(f"Warning: Could not load smart_analysis.json: {e}")

    # Iteration manifest for report
    iterations_data = []
    similarity_history = []  # Track similarity per iteration for convergence detection
    consecutive_no_change = 0  # Count iterations with no code change

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} (v{current_version}) ---")

        # 1. Render current code using BlackHole recorder (real Strudel playback)
        # Node.js renderer was deleted and Python renderer can't parse arrange() format
        # BlackHole gives real audio - use full duration to compare the whole track
        render_path = Path(output_dir) / f"render_v{current_version:03d}.wav"
        blackhole_recorder = Path(__file__).parent.parent / "node" / "dist" / "record-strudel-blackhole.js"
        iter_duration = exact_duration

        if blackhole_recorder.exists():
            # Write current code to a temp strudel file for this iteration
            iter_strudel = Path(output_dir) / f"output_iter_{current_version:03d}.strudel"
            with open(iter_strudel, 'w') as f:
                f.write(current_code)

            render_cmd = [
                "node", str(blackhole_recorder),
                str(iter_strudel),
                "-o", str(render_path),
                "-d", f"{iter_duration:.0f}"
            ]
            print(f"       Rendering {iter_duration:.0f}s via BlackHole...")
            result = subprocess.run(render_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"       BlackHole render failed: {result.stderr[:200]}")
        else:
            print("       BlackHole recorder not found, skipping render")

        # 2. Compare to original (only if render succeeded)
        if render_path.exists():
            compare_cmd = [
                sys.executable,
                str(Path(__file__).parent / "compare_audio.py"),
                original_audio,
                str(render_path),
                "-j",  # Output JSON to stdout
                "-d", f"{iter_duration:.2f}"
            ]
            if synth_config_path.exists():
                compare_cmd.extend(["--config", str(synth_config_path)])
            compare_result = subprocess.run(compare_cmd, capture_output=True, text=True)

            comparison_path = Path(output_dir) / f"comparison_v{current_version:03d}.json"
            if compare_result.returncode == 0 and compare_result.stdout.strip():
                comparison = json.loads(compare_result.stdout)
                with open(comparison_path, 'w') as f:
                    json.dump(comparison, f, indent=2)
            else:
                print("       Comparison failed, using initial data")
                comparison = {"comparison": {}, "original": {}, "rendered": {}}
        else:
            # Render failed — fall back to initial BlackHole comparison
            print("       Render failed, using initial comparison data")
            initial_comparison_path = Path(output_dir) / "comparison.json"
            if initial_comparison_path.exists():
                with open(initial_comparison_path) as f:
                    comparison = json.load(f)
            else:
                comparison = {"comparison": {}, "original": {}, "rendered": {}}

        # Extract similarity from comparison
        comp_scores = comparison.get("comparison", {})
        current_similarity = comp_scores.get("overall_similarity", 0)
        print(f"       Similarity: {current_similarity*100:.1f}%")

        # Track iteration data for manifest
        iter_data = {
            "version": current_version,
            "iteration": iteration + 1,
            "similarity": current_similarity,
            "code": current_code,
            "render_path": str(render_path) if render_path.exists() else None,
            "comparison": {
                "overall": current_similarity,
                "mfcc": comp_scores.get("mfcc_similarity", 0),
                "chroma": comp_scores.get("chroma_similarity", 0),
                "frequency_balance": comp_scores.get("frequency_balance_similarity", 0),
                "brightness": comp_scores.get("brightness_similarity", 0),
                "energy": comp_scores.get("energy_similarity", 0),
                "tempo": comp_scores.get("tempo_similarity", 0),
            },
            "phase": None,  # Set below
            "changes": [],  # Set below
            "was_best": False,  # Set below
            "reverted": False,  # Set below
        }

        # Node.js renderer gives ~16% for arrange() code — use initial BlackHole comparison
        # for LLM feedback since it reflects the REAL audio quality
        initial_comparison_path = Path(output_dir) / "comparison.json"
        if current_similarity < 0.30 and initial_comparison_path.exists():
            with open(initial_comparison_path) as f:
                blackhole_comparison = json.load(f)
            blackhole_sim = blackhole_comparison.get("comparison", {}).get("overall_similarity", 0)
            if blackhole_sim > current_similarity:
                print(f"       Node.js score too low ({current_similarity*100:.1f}%), using BlackHole comparison ({blackhole_sim*100:.1f}%) for LLM")
                comparison = blackhole_comparison

        # REGRESSION CHECK: If this iteration is worse than best, revert code
        if iteration > 0 and current_similarity < best_similarity and best_code:
            print(f"       ✗ REGRESSION: {best_similarity*100:.1f}% → {current_similarity*100:.1f}% - reverting to best")
            iter_data["reverted"] = True
            current_code = best_code
            with open(strudel_path, 'w') as f:
                f.write(best_code)

        # Load per-stem comparison if available (shows real issues even when overall looks good)
        stem_comparison = {}
        worst = []
        section_scores = []
        stem_comparison_path = Path(output_dir) / "stem_comparison.json"
        if stem_comparison_path.exists():
            with open(stem_comparison_path) as f:
                stem_comparison = json.load(f)
            agg = stem_comparison.get("aggregate", {}).get("per_stem", {})
            worst = stem_comparison.get("aggregate", {}).get("worst_sections", [])
            print(f"       Per-stem: bass={agg.get('bass', {}).get('overall', 0)*100:.0f}% drums={agg.get('drums', {}).get('overall', 0)*100:.0f}% melodic={agg.get('melodic', {}).get('overall', 0)*100:.0f}%")
            if worst:
                print(f"       Worst: {worst[0].get('stem', '?')} {worst[0].get('time_range', '?')}: {worst[0].get('issues', '?')}")

            # Map windowed comparison data to section indices
            windowed = stem_comparison.get("windowed", {})
            if windowed and sections:
                section_scores = map_windows_to_sections(windowed, sections)
                if section_scores:
                    print(f"       Section scores: worst={section_scores[0]['stem']} S{section_scores[0]['section_idx']+1} ({section_scores[0]['similarity']*100:.0f}%)")

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

        # Track best (code AND render file)
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_code = current_code
            best_render_path = render_path
            iter_data["was_best"] = True

        # 4. OPTIMIZATION (Deterministic parameter tuning, then constrained LLM)

        # Extract metrics for debugging output
        comp_scores = comparison.get("comparison", {})
        orig_bands = comparison.get("original", {}).get("bands", {})
        rend_bands = comparison.get("rendered", {}).get("bands", {})

        band_diffs = {}
        for band in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
            band_diffs[band] = rend_bands.get(band, 0) - orig_bands.get(band, 0)

        orig_spectral = comparison.get("original", {}).get("spectral", {})
        rend_spectral = comparison.get("rendered", {}).get("spectral", {})
        orig_centroid = orig_spectral.get("centroid_mean", 1)
        rend_centroid = rend_spectral.get("centroid_mean", 1)
        brightness_ratio = rend_centroid / max(orig_centroid, 1) if orig_centroid else 1.0
        orig_rms = orig_spectral.get("rms_mean", 0.1)
        rend_rms = rend_spectral.get("rms_mean", 0.1)
        energy_ratio = rend_rms / max(orig_rms, 0.001) if orig_rms else 1.0

        print(f"       Bands: sub_bass={band_diffs.get('sub_bass',0)*100:+.0f}% bass={band_diffs.get('bass',0)*100:+.0f}% low_mid={band_diffs.get('low_mid',0)*100:+.0f}% mid={band_diffs.get('mid',0)*100:+.0f}% high={band_diffs.get('high',0)*100:+.0f}%")
        print(f"       Brightness: {brightness_ratio:.0%}  Energy: {energy_ratio:.0%}")

        # Always call the LLM agent to generate improved code
        if agent is not None:
            improved = current_similarity > best_similarity

            # Build per-stem scores dict for the agent
            per_stem_scores = {}
            agg = stem_comparison.get("aggregate", {}).get("per_stem", {})
            for stem_name in ["bass", "drums", "melodic"]:
                stem_data = agg.get(stem_name, {})
                per_stem_scores[stem_name] = stem_data.get("overall", 0)

            agent.add_iteration_result(
                iteration=iteration + 1,
                version=current_version,
                similarity=current_similarity,
                band_diffs=band_diffs,
                code_generated=current_code,
                improved=improved,
                genre=genre,
                worst_sections=worst[:5] if worst else None,
                section_scores=section_scores[:8] if section_scores else None,
                per_stem_scores=per_stem_scores if per_stem_scores else None,
                orig_bands=orig_bands,
                rend_bands=rend_bands,
            )

            # LLM generates improved code every iteration
            print("       LLM generating improved code...")
            iter_data["phase"] = "llm_agent"

            # RAG: retrieve knowledge relevant to CURRENT problem (not just genre)
            rag_context = ""
            if HAS_CH_MODULE:
                tried = getattr(agent, 'tried_values', None)
                rag_context = retrieve_relevant_knowledge(
                    comparison, genre, metadata.get("bpm", 120), tried
                )
                if rag_context:
                    print(f"       RAG: retrieved {rag_context.count(chr(10))} proven fixes")

            context = {
                "genre": genre,
                "bpm": metadata.get("bpm", 120),
                "similarity": current_similarity,
                "band_sub_bass": band_diffs.get("sub_bass", 0),
                "band_bass": band_diffs.get("bass", 0),
                "band_low_mid": band_diffs.get("low_mid", 0),
                "band_mid": band_diffs.get("mid", 0),
                "band_high_mid": band_diffs.get("high_mid", 0),
                "band_high": band_diffs.get("high", 0),
                "rag_context": rag_context,  # Injected into agent prompt
            }
            response = agent.generate(context)
            new_code = agent.extract_code(response, previous_code=current_code)

            if new_code and new_code != current_code:
                # Validate and fix syntax
                if HAS_SYNTAX_FIXER:
                    new_code = fix_strudel_syntax(new_code)
                    new_code = enforce_three_voices(new_code)

                # Check for invalid sounds
                if not _has_invalid_sounds(new_code):
                    current_code = new_code
                    improved_path = Path(output_dir) / f"output_v{current_version + 1:03d}.strudel"
                    with open(improved_path, 'w') as f:
                        f.write(new_code)
                    with open(strudel_path, 'w') as f:
                        f.write(new_code)
                    print(f"       LLM generated {len(new_code)} chars of improved code")
                    iter_data["changes"] = ["LLM rewrote code"]
                else:
                    print("       LLM code has invalid sounds, keeping current")
                    iter_data["changes"] = ["LLM code rejected (invalid sounds)"]
                    # Feed rejection reason back to agent so it doesn't repeat the mistake
                    agent.messages.append({"role": "user", "content":
                        "Your code was REJECTED: it contains invalid sound names. "
                        "Use ONLY sounds from the available sounds list. Do NOT invent sound names."
                    })
            elif new_code:
                print("       LLM returned same code, no change")
                iter_data["changes"] = ["LLM: no change"]
            else:
                validation_err = getattr(agent, 'last_validation_error', None)
                print(f"       LLM failed to generate code{f': {validation_err}' if validation_err else ''}")
                iter_data["changes"] = [f"LLM failed{f': {validation_err}' if validation_err else ''}"]
                # Feed validation error back to agent
                if validation_err and agent:
                    agent.messages.append({"role": "user", "content":
                        f"Your previous output was REJECTED: {validation_err}. "
                        f"Fix this issue in your next attempt."
                    })
        else:
            # Fallback: deterministic optimization when no agent available
            print("       Deterministic parameter optimization (no agent)...")
            iter_data["phase"] = "deterministic"
            optimized_code, opt_changes = optimize_parameters(current_code, comparison, stem_comparison)

            if opt_changes:
                for change in opt_changes:
                    print(f"         {change}")
                iter_data["changes"] = list(opt_changes)
                current_code = optimized_code
                improved_path = Path(output_dir) / f"output_v{current_version + 1:03d}.strudel"
                with open(improved_path, 'w') as f:
                    f.write(optimized_code)
                with open(strudel_path, 'w') as f:
                    f.write(optimized_code)
                print(f"       Applied {len(opt_changes)} deterministic changes")
            else:
                print("       No deterministic changes possible")
                iter_data["changes"] = ["No deterministic changes possible"]

        # Record iteration data (without large code field for JSON - code saved in .strudel files)
        iterations_data.append(iter_data)

        current_version += 1

        # --- CONVERGENCE DETECTION ---
        similarity_history.append(current_similarity)

        # Check if target reached
        if current_similarity >= target_similarity:
            print(f"\n    Target similarity {target_similarity*100:.0f}% reached ({current_similarity*100:.1f}%). Stopping.")
            break

        # Detect plateau: std dev < 1% over last 5 iterations
        if len(similarity_history) >= 5:
            recent = similarity_history[-5:]
            std_dev = (sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)) ** 0.5
            if std_dev < 0.01:
                print(f"\n    Converged: similarity plateau (std dev {std_dev*100:.2f}% over last 5 iterations). Stopping.")
                break

        # Detect oscillation: same similarity repeating (within 0.5%)
        if len(similarity_history) >= 4:
            last4 = similarity_history[-4:]
            if abs(last4[0] - last4[2]) < 0.005 and abs(last4[1] - last4[3]) < 0.005 and abs(last4[0] - last4[1]) > 0.01:
                print(f"\n    Oscillation detected ({last4[0]*100:.1f}% <-> {last4[1]*100:.1f}%). Stopping.")
                break

        # Detect no-change stalls
        if iter_data.get("changes") and any("no change" in str(c).lower() or "no deterministic" in str(c).lower() for c in iter_data["changes"]):
            consecutive_no_change += 1
        else:
            consecutive_no_change = 0

        if consecutive_no_change >= 3:
            print(f"\n    No changes for {consecutive_no_change} consecutive iterations. Stopping.")
            break

    # Save iteration manifest (strip code to keep JSON small - it's in .strudel files)
    iterations_manifest = []
    for it in iterations_data:
        manifest_entry = {k: v for k, v in it.items() if k != "code"}
        iterations_manifest.append(manifest_entry)
    iterations_json_path = Path(output_dir) / "iterations.json"
    with open(iterations_json_path, 'w') as f:
        json.dump({"iterations": iterations_manifest, "best_similarity": best_similarity}, f, indent=2)
    print(f"Saved iteration manifest: {iterations_json_path}")

    # Batch stem separation for all iteration renders
    separate_script = Path(__file__).parent / "separate.py"
    if separate_script.exists():
        print("\nSeparating iteration renders into stems...")
        for entry in iterations_manifest:
            render_path = entry.get("render_path")
            if not render_path or not Path(render_path).exists():
                continue
            ver = entry.get("version", 0)
            prefix = f"render_v{ver:03d}"
            # Find stem file with either .wav or .mp3 extension (demucs output varies by quality)
            def _find_stem(stem_name):
                for ext in [".wav", ".mp3"]:
                    p = Path(output_dir) / f"{prefix}_{stem_name}{ext}"
                    if p.exists():
                        return p
                return None
            # Check if stems already exist (idempotent re-runs)
            existing = [_find_stem(s) for s in ["melodic", "drums", "bass"]]
            if all(existing):
                entry["stem_melodic_path"] = str(existing[0])
                entry["stem_drums_path"] = str(existing[1])
                entry["stem_bass_path"] = str(existing[2])
                continue
            print(f"  Separating v{ver:03d}...")
            sep_cmd = [
                sys.executable, str(separate_script),
                render_path, str(output_dir),
                "--mode", "full", "--prefix", prefix
            ]
            sep_result = subprocess.run(sep_cmd, capture_output=True, text=True)
            if sep_result.returncode == 0:
                entry["stem_melodic_path"] = str(_find_stem("melodic") or "")
                entry["stem_drums_path"] = str(_find_stem("drums") or "")
                entry["stem_bass_path"] = str(_find_stem("bass") or "")
            else:
                print(f"  Warning: stem separation failed for v{ver:03d}: {sep_result.stderr[:200]}")
        # Re-save iterations.json with stem paths
        with open(iterations_json_path, 'w') as f:
            json.dump({"iterations": iterations_manifest, "best_similarity": best_similarity}, f, indent=2)
        print("Updated iterations.json with stem paths")

    # Generate final comparison charts and report
    print("\nGenerating comparison charts and report...")

    # Use the render file that corresponds to best_code (not just the latest!)
    # The latest render might be from a regressed iteration
    if best_render_path and best_render_path.exists():
        best_render = best_render_path
    else:
        # Fallback: find any render file
        render_files = sorted(Path(output_dir).glob("render_v*.wav"))
        best_render = render_files[-1] if render_files else Path(output_dir) / "render.wav"

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

        # Write BEST code (not last iteration) to output.strudel
        # The iteration loop tracks best_code based on Node.js comparison
        if best_code:
            with open(Path(output_dir) / "output.strudel", 'w') as f:
                f.write(best_code)
            print(f"Wrote best code ({best_similarity*100:.1f}% similarity) to output.strudel")

        # Copy render
        import shutil
        if best_render.name != "render.wav":
            shutil.copy(best_render, Path(output_dir) / "render.wav")

        # Separate render into stems for per-stem comparison in report
        final_render = Path(output_dir) / "render.wav"
        separate_script = Path(__file__).parent / "separate.py"
        if final_render.exists() and separate_script.exists():
            print("Separating render into stems...")
            sep_cmd = [
                sys.executable, str(separate_script),
                str(final_render), str(output_dir),
                "--mode", "full", "--prefix", "render"
            ]
            sep_result = subprocess.run(sep_cmd, capture_output=True, text=True)
            if sep_result.returncode != 0:
                print(f"Warning: render stem separation failed: {sep_result.stderr[:200]}")
            else:
                print("Render stems ready: render_melodic.wav, render_drums.wav, render_bass.wav")

        # Create metadata (preserve existing fields like notes/drum_hits from pipeline)
        existing_meta = {}
        meta_path = Path(output_dir) / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    existing_meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        meta = {
            **existing_meta,  # Preserve existing fields (notes, drum_hits, etc.)
            "bpm": metadata.get("bpm", existing_meta.get("bpm", 120)),
            "key": metadata.get("key", existing_meta.get("key", "")),
            "style": metadata.get("style", existing_meta.get("style", "")),
            "genre": metadata.get("genre", existing_meta.get("genre", "")),
            "notes": metadata.get("notes", existing_meta.get("notes", 0)),
            "drum_hits": metadata.get("drum_hits", existing_meta.get("drum_hits", 0)),
            "ai_improved": True,
            "iterations": current_version,
            "similarity": best_similarity
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Generate HTML report
        # Report expects: cache_dir/melodic.wav (original stems) + cache_dir/vNNN/ (version outputs)
        # When run standalone, output_dir may be flat (not vNNN pattern).
        # In that case, create a temporary structure for the report.
        import re
        dir_name = Path(output_dir).name
        version_match = re.search(r'^v(\d+)$', dir_name)

        if version_match:
            # Standard cache structure: output_dir is already vNNN inside cache_dir
            cache_dir = Path(output_dir).parent
            version_num = version_match.group(1)
        else:
            # Standalone mode: output_dir has everything flat.
            # Create a temp version subdir and symlink/copy what the report needs.
            cache_dir = Path(output_dir)
            version_num = "1"
            version_subdir = cache_dir / "v001"
            version_subdir.mkdir(exist_ok=True)
            # Copy/link version-specific files into v001/
            for fname in ["render.wav", "render_melodic.wav", "render_melodic.mp3",
                          "render_drums.wav", "render_drums.mp3",
                          "render_bass.wav", "render_bass.mp3",
                          "comparison.json", "comparison.png", "output.strudel",
                          "metadata.json", "synth_config.json",
                          "stem_comparison.json", "iterations.json"]:
                src = cache_dir / fname
                dst = version_subdir / fname
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
            # Copy iteration render files and stems
            for render_file in cache_dir.glob("render_v*.wav"):
                dst = version_subdir / render_file.name
                if not dst.exists():
                    shutil.copy(render_file, dst)
            for render_file in cache_dir.glob("render_v*.mp3"):
                dst = version_subdir / render_file.name
                if not dst.exists():
                    shutil.copy(render_file, dst)
            # Copy chart images
            for chart in cache_dir.glob("chart_*.png"):
                dst = version_subdir / chart.name
                if not dst.exists():
                    shutil.copy(chart, dst)

        report_cmd = [
            sys.executable,
            str(Path(__file__).parent / "generate_report.py"),
            str(cache_dir),
            "-v", version_num,
            "-o", str(Path(output_dir) / "report.html")
        ]
        # Pass iterations manifest if available
        iter_json = Path(output_dir) / "iterations.json"
        if iter_json.exists():
            report_cmd.extend(["--iterations-file", str(iter_json)])
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
