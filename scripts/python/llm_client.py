#!/usr/bin/env python3
"""
LLM Client for AI-Driven Strudel Code Improvement

Extracted from ai_improver.py.  Contains all LLM-facing functions:
  - build_improvement_prompt
  - analyze_with_ollama
  - analyze_with_claude
  - analyze_with_llm
  - build_constrained_llm_prompt
  - _apply_constrained_llm_choice
  - _call_constrained_llm

Helper functions included because they are called from within the above:
  - map_windows_to_sections
  - _classify_arrange_line
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

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


def build_improvement_prompt(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = "",
    bpm: float = 120,
    key: str = "C minor",
    stem_comparison: Dict = None,
    comparison_data: Dict = None
) -> str:
    """Build the prompt for LLM to improve arrange()-based Strudel code.

    Enhanced with 6-band frequency breakdown, per-stem data, dynamics info.
    """

    # Get ALL 6 frequency band differences (don't combine - LLM needs granular data)
    band_sub_bass = previous_run.get('band_sub_bass', 0)
    band_bass = previous_run.get('band_bass', 0)
    band_low_mid = previous_run.get('band_low_mid', 0)
    band_mid = previous_run.get('band_mid', 0)
    band_high_mid = previous_run.get('band_high_mid', 0)
    band_high = previous_run.get('band_high', 0)

    # Get spectral metrics
    brightness_ratio = previous_run.get('brightness_ratio', 1.0)
    energy_ratio = previous_run.get('energy_ratio', 1.0)
    overall_similarity = previous_run.get('similarity_overall', 0)

    # Build 6-band frequency table with specific Hz ranges
    def band_row(name, hz_range, diff):
        if diff > 0.05:
            return f"| {name} | {hz_range} | TOO LOUD | {diff*100:+.0f}% | Reduce gain by ~{diff*100:.0f}% |"
        elif diff < -0.05:
            return f"| {name} | {hz_range} | TOO QUIET | {diff*100:+.0f}% | Increase gain by ~{abs(diff)*100:.0f}% |"
        else:
            return f"| {name} | {hz_range} | OK | {diff*100:+.0f}% | No change needed |"

    freq_table = f"""| Band | Hz Range | Status | Diff | Action |
|------|----------|--------|------|--------|
{band_row("Sub-bass", "20-60 Hz", band_sub_bass)}
{band_row("Bass", "60-250 Hz", band_bass)}
{band_row("Low-mid", "250-500 Hz", band_low_mid)}
{band_row("Mid", "500-2k Hz", band_mid)}
{band_row("High-mid", "2k-4k Hz", band_high_mid)}
{band_row("High", "4k-20k Hz", band_high)}"""

    # Extract original/rendered band values from comparison_data for context
    band_context = ""
    if comparison_data:
        orig_bands = comparison_data.get("original", {}).get("bands", {})
        rend_bands = comparison_data.get("rendered", {}).get("bands", {})
        if orig_bands and rend_bands:
            band_context = "\n**Absolute band energy (original → rendered):**\n"
            for band_name in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
                o_val = orig_bands.get(band_name, 0) * 100
                r_val = rend_bands.get(band_name, 0) * 100
                band_context += f"  {band_name}: {o_val:.1f}% → {r_val:.1f}%\n"

    # Dynamics and energy analysis
    dynamics_str = ""
    if comparison_data:
        orig_spec = comparison_data.get("original", {}).get("spectral", {})
        rend_spec = comparison_data.get("rendered", {}).get("spectral", {})
        orig_rms = orig_spec.get("rms_mean", 0)
        rend_rms = rend_spec.get("rms_mean", 0)
        orig_rms_std = orig_spec.get("rms_std", 0)
        rend_rms_std = rend_spec.get("rms_std", 0)
        dynamics_str = f"""
## DYNAMICS ANALYSIS
- **Overall energy**: rendered is {energy_ratio:.0%} of original ({rend_rms:.3f} vs {orig_rms:.3f} RMS)
- **Dynamic variation**: rendered {rend_rms_std:.3f} vs original {orig_rms_std:.3f} std dev"""
        if energy_ratio < 0.5:
            dynamics_str += f"\n- **CRITICAL: Output is {(1-energy_ratio)*100:.0f}% too quiet!** Increase all gains significantly."
        elif energy_ratio < 0.8:
            dynamics_str += f"\n- Output is too quiet. Increase gains by ~{(1/energy_ratio - 1)*100:.0f}%."
        if orig_rms_std > 0 and rend_rms_std / max(orig_rms_std, 0.001) < 0.5:
            dynamics_str += f"\n- **Dynamic range is too flat!** Add gain variation with `.gain(\"<0.3 0.5 0.8 0.5>\".slow(16))` patterns."

    # Per-stem detailed breakdown from stem_comparison
    stem_detail_str = ""
    if stem_comparison:
        stems_data = stem_comparison.get("stems", {})
        if stems_data:
            stem_detail_str = "\n## PER-STEM FREQUENCY ANALYSIS\n"
            for stem_name in ["bass", "drums", "melodic"]:
                stem_info = stems_data.get(stem_name, {})
                if not stem_info:
                    continue
                s_orig = stem_info.get("original", {}).get("bands", {})
                s_rend = stem_info.get("rendered", {}).get("bands", {})
                s_comp = stem_info.get("comparison", {})
                s_overall = s_comp.get("overall_similarity", 0)
                s_energy = s_comp.get("energy_similarity", 0)

                # RMS comparison
                s_orig_rms = stem_info.get("original", {}).get("spectral", {}).get("rms_mean", 0)
                s_rend_rms = stem_info.get("rendered", {}).get("spectral", {}).get("rms_mean", 0)

                stem_detail_str += f"\n### {stem_name.upper()} stem ({s_overall*100:.0f}% similar)\n"
                stem_detail_str += f"Energy: {s_rend_rms:.3f} vs {s_orig_rms:.3f} RMS"
                if s_orig_rms > 0:
                    stem_detail_str += f" ({s_rend_rms/s_orig_rms:.0%} of original)"
                stem_detail_str += "\n"

                # Show worst band differences for this stem
                s_band_diffs = s_comp.get("band_differences", {})
                if s_band_diffs:
                    worst_band = max(s_band_diffs, key=lambda k: abs(s_band_diffs[k]))
                    stem_detail_str += f"Worst band: {worst_band} ({s_band_diffs[worst_band]:+.1f}%)\n"
                    for b_name in ["sub_bass", "bass", "low_mid", "mid"]:
                        if b_name in s_band_diffs and abs(s_band_diffs[b_name]) > 5:
                            o_pct = s_orig.get(b_name, 0) * 100
                            r_pct = s_rend.get(b_name, 0) * 100
                            stem_detail_str += f"  {b_name}: {o_pct:.0f}% → {r_pct:.0f}% (diff: {s_band_diffs[b_name]:+.1f}%)\n"

    # Per-stem issues (summary)
    per_stem_issues = previous_run.get('per_stem_issues', [])
    stem_issues_str = ""
    if per_stem_issues:
        stem_issues_str = "\n## WORST TIME RANGES (fix these arrange() sections)\n"
        stem_issues_str += "Map: Section 1 = first [cycles, pattern] pair in arrange(), Section 2 = second, etc.\n"
        for issue in per_stem_issues[:5]:
            stem_issues_str += f"- {issue}\n"

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

    return f'''You are improving Strudel live coding audio to match an original track.
Current similarity: **{overall_similarity*100:.1f}%** — target is 80%+.

## FREQUENCY BALANCE (6-band analysis)

{freq_table}
{band_context}
Brightness: {brightness_ratio:.0%} of original
{dynamics_str}
{stem_detail_str}
{stem_issues_str}
{spectrogram_str}
{knowledge_str}

## CURRENT CODE
```javascript
{original_code}
```

## GENRE CONTEXT
Genre: {genre or 'auto'}, Artist: {artist or 'unknown'}, BPM: {bpm:.0f}, Key: {key}

## GENRE-SPECIFIC SOUNDS
{retrieve_genre_context(genre) if HAS_SOUND_SELECTOR and genre else "Use sounds appropriate for the genre."}

## TASK: Fix the frequency balance and dynamics

Based on the analysis above, modify the code to fix the issues. Focus on:
1. **Sub-bass** — if sub_bass is too quiet, bass notes must go LOWER (a1, e1 instead of a2, e2) and use deeper sounds
2. **Energy/loudness** — if rendered is too quiet, increase ALL .gain() values proportionally
3. **Dynamic variation** — use gain patterns like `.gain("<0.3 0.6 0.9 0.6>".slow(8))` for natural dynamics
4. **Per-stem balance** — fix the worst-scoring stem first

**RULES:**
1. Output the COMPLETE Strudel code (setcps + all $: voices) in a ```javascript block
2. Keep EXACTLY 3 `$:` blocks (bass, lead, drums) — NO MORE, NO LESS
3. Each `$:` has ONE arrange() with ALL sections as [cycles, pattern] pairs inside it
4. DO NOT create separate `$:` blocks per section — sections go INSIDE arrange()
5. Keep the SAME key ({key}) and BPM ({bpm:.0f}) — do NOT change them
6. Use `note()` + `.sound()` for bass and lead voices, `s()` + `.bank()` for drums
7. Drums ONLY use `bd sd hh oh cp lt mt ht` names, NEVER note names like `c2 e2`
8. Adjust `.gain()` values to fix the frequency balance issues shown above
9. NO semicolons (Strudel is NOT JavaScript)
10. NO `.peak()`, `.volume()`, `.eq()`, `.filter()` — these don't exist

**VALID SOUNDS:** sine, triangle, square, sawtooth, supersaw,
  gm_acoustic_bass, gm_synth_bass_1, gm_synth_bass_2, gm_electric_bass_finger,
  gm_piano, gm_epiano1, gm_epiano2, gm_lead_2_sawtooth, gm_lead_5_charang,
  gm_pad_warm, gm_pad_poly, gm_string_ensemble_1, gm_brass_section, gm_trumpet,
  gm_acoustic_guitar_nylon, gm_electric_guitar_clean, gm_overdriven_guitar

**VALID DRUM BANKS:** RolandTR808, RolandTR909, LinnDrum, BossDR110

Output ONLY the improved code in a ```javascript block, no explanation.'''


def analyze_with_ollama(
    previous_run: Dict,
    learned_knowledge: List[Dict],
    original_code: str,
    model: str = None,
    spectrogram_insights: str = None,
    genre: str = "",
    artist: str = "",
    bpm: float = 120,
    stem_comparison: Dict = None,
    comparison_data: Dict = None
) -> Dict[str, Any]:
    """Use Ollama (local LLM) to analyze results and suggest improvements."""

    if not HAS_REQUESTS:
        print("Warning: requests package not installed", file=sys.stderr)
        return {"suggestions": [], "improved_code": original_code, "reasoning": "No requests"}

    ollama_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    key = previous_run.get("key", "C minor")
    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist, bpm=bpm, key=key,
        stem_comparison=stem_comparison, comparison_data=comparison_data
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
                    "num_ctx": 32768,
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
            # Check if it looks like Strudel code (arrange, effect functions, sound, or setcps)
            if 'arrange(' in extracted_code or '$:' in extracted_code or 'Fx' in extracted_code or 'sound(' in extracted_code or 'setcps(' in extracted_code:
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
    bpm: float = 120,
    stem_comparison: Dict = None,
    comparison_data: Dict = None
) -> Dict[str, Any]:
    """Use Claude API to analyze results and suggest improvements."""

    if Anthropic is None:
        return None  # Signal to try Ollama

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # Signal to try Ollama

    key = previous_run.get("key", "C minor")
    prompt = build_improvement_prompt(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights, genre=genre, artist=artist, bpm=bpm, key=key,
        stem_comparison=stem_comparison, comparison_data=comparison_data
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
    agent: 'OllamaAgent' = None,  # Agentic mode with memory
    stem_comparison: Dict = None
) -> Dict[str, Any]:
    """
    Analyze and improve code using available LLM.

    Priority:
    1. If agent provided, use agentic Ollama with memory (PREFERRED)
    2. If use_ollama=True, use Ollama directly (stateless)
    3. Try Claude API if ANTHROPIC_API_KEY is set
    4. Fall back to Ollama (local)

    Now enhanced with spectrogram_insights for deeper analysis,
    stem_comparison for per-stem frequency data,
    and comparison_data for dynamics/energy context.
    """

    # AGENTIC MODE - uses persistent memory, ClickHouse queries
    if agent is not None and HAS_AGENT:
        print("       Using Agentic Ollama (with memory + ClickHouse)...")
        context = {
            "genre": genre,
            "bpm": bpm,
            "similarity": previous_run.get("similarity_overall", 0),
            "band_sub_bass": previous_run.get("band_sub_bass", 0),
            "band_bass": previous_run.get("band_bass", 0),
            "band_low_mid": previous_run.get("band_low_mid", 0),
            "band_mid": previous_run.get("band_mid", 0),
            "band_high_mid": previous_run.get("band_high_mid", 0),
            "band_high": previous_run.get("band_high", 0),
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
            spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm,
            stem_comparison=stem_comparison, comparison_data=comparison_data
        )

    # Try Claude first
    result = analyze_with_claude(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm,
        stem_comparison=stem_comparison, comparison_data=comparison_data
    )
    if result is not None:
        return result

    # Fall back to Ollama
    print("       Claude unavailable, using Ollama (local LLM)...")
    return analyze_with_ollama(
        previous_run, learned_knowledge, original_code,
        spectrogram_insights=spectrogram_insights, genre=genre, artist=artist, bpm=bpm,
        stem_comparison=stem_comparison, comparison_data=comparison_data
    )


def build_constrained_llm_prompt(code: str, comparison: dict, stem_comparison: dict = None, genre: str = "", sections: list = None) -> tuple:
    """Build constrained prompt for Phase 2 - LLM picks from menu, doesn't write code.

    Returns (prompt_text, options_list).
    """
    sim = comparison.get("comparison", {}).get("overall_similarity", 0)

    # Extract current sounds/banks from Fx definitions
    sounds = {}
    for m in re.finditer(r'let\s+(\w+Fx)\s*=.*?\.sound\("([^"]+)"\)', code, re.DOTALL):
        sounds[m.group(1)] = m.group(2)
    banks = {}
    for m in re.finditer(r'let\s+(\w+Fx)\s*=.*?\.bank\("([^"]+)"\)', code, re.DOTALL):
        banks[m.group(1)] = m.group(2)

    # Fallback: extract sounds from arrange() format (inline .sound/.bank calls)
    if not sounds and 'arrange(' in code:
        for line in code.split('\n'):
            classification = _classify_arrange_line(line)
            if not classification:
                continue
            sound_m = re.search(r'\.sound\("([^"]+)"\)', line)
            bank_m = re.search(r'\.bank\("([^"]+)"\)', line)
            if classification == 'bass' and sound_m and 'bassFx' not in sounds:
                sounds['bassFx'] = sound_m.group(1)
            elif classification == 'melodic' and sound_m and 'leadFx' not in sounds:
                sounds['leadFx'] = sound_m.group(1)
            elif classification == 'drums' and bank_m and 'drumsFx' not in banks:
                banks['drumsFx'] = bank_m.group(1)

    # Per-stem scores
    stem_lines = ""
    worst_stem = None
    worst_sim = 1.0
    if stem_comparison:
        agg = stem_comparison.get("aggregate", {}).get("per_stem", {})
        for stem, data in agg.items():
            s = data.get('overall', 1.0)
            stem_lines += f"\n- {stem}: {s*100:.0f}%"
            if s < worst_sim:
                worst_sim = s
                worst_stem = stem

    # Build options based on worst stem
    options = []
    bass_alts = ['gm_synth_bass_1', 'gm_synth_bass_2', 'gm_acoustic_bass', 'gm_electric_bass_finger', 'sawtooth']
    lead_alts = ['gm_lead_2_sawtooth', 'gm_lead_5_charang', 'supersaw', 'square', 'triangle', 'gm_pad_warm']
    drum_alts = ['RolandTR808', 'RolandTR909', 'LinnDrum', 'BossDR110', 'AkaiLinn']

    current_bass = sounds.get('bassFx', '')
    current_lead = sounds.get('midFx', '') or sounds.get('highFx', '') or sounds.get('leadFx', '')
    current_drums = banks.get('drumsFx', '') or banks.get('kickFx', '')

    bass_alts = [a for a in bass_alts if a != current_bass][:2]
    lead_alts = [a for a in lead_alts if a != current_lead][:2]
    drum_alts = [a for a in drum_alts if a != current_drums][:2]

    if worst_stem in ('bass', None):
        for alt in bass_alts:
            options.append(f"Change bass sound from '{current_bass}' to '{alt}'")

    if worst_stem in ('melodic', None):
        for alt in lead_alts:
            options.append(f"Change lead sound from '{current_lead}' to '{alt}'")

    if worst_stem in ('drums', None):
        for alt in drum_alts:
            options.append(f"Change drum bank from '{current_drums}' to '{alt}'")

    options.append("No change needed - keep current sounds")

    numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    # Add temporal section info if available
    section_info = ""
    if sections and stem_comparison:
        windowed = stem_comparison.get("windowed", {})
        if windowed:
            sec_scores = map_windows_to_sections(windowed, sections)
            if sec_scores:
                section_info = "\nWorst sections:"
                for ss in sec_scores[:3]:
                    issues_str = ", ".join(ss.get("issues", [])[:2]) or "low similarity"
                    section_info += f"\n- Section {ss['section_idx']+1} ({ss['stem']}): {ss['similarity']*100:.0f}% — {issues_str}"

    prompt = f"""Similarity: {sim*100:.1f}% after parameter optimization.
Per-stem scores:{stem_lines}
Worst stem: {worst_stem or 'N/A'} ({worst_sim*100:.0f}%)
{section_info}

Current sounds: bass='{current_bass}', lead='{current_lead}', drums='{current_drums}'
Genre: {genre or 'auto'}

Choose ONE change to improve the worst-scoring stem.
Respond with the number ONLY, nothing else.

{numbered}"""

    return prompt, options


def _apply_constrained_llm_choice(code: str, choice: int, options: list) -> tuple:
    """Apply a constrained LLM choice to the code programmatically.

    Returns (modified_code, description).
    """
    if choice < 1 or choice > len(options):
        return code, "Invalid choice"

    option_text = options[choice - 1]

    if "Change bass sound" in option_text:
        m = re.search(r"to '([^']+)'", option_text)
        if m:
            new_sound = m.group(1)
            # Try Fx definition format first
            fx_pattern = r'(let\s+bassFx\s*=\s*p\s*=>\s*p.*?\.sound\(")[^"]+(")'
            new_code = re.sub(fx_pattern, rf'\g<1>{new_sound}\g<2>', code, flags=re.DOTALL)
            if new_code != code:
                return new_code, f"Changed bass sound to {new_sound}"
            # Fallback: arrange() format - replace .sound() on bass lines (octave 1-2)
            if 'arrange(' in code:
                lines = code.split('\n')
                changed = False
                for i, line in enumerate(lines):
                    if _classify_arrange_line(line) == 'bass':
                        new_line = re.sub(r'(\.sound\(")[^"]+(")', rf'\g<1>{new_sound}\g<2>', line)
                        if new_line != line:
                            lines[i] = new_line
                            changed = True
                if changed:
                    return '\n'.join(lines), f"Changed bass sound to {new_sound}"

    elif "Change lead sound" in option_text:
        m = re.search(r"to '([^']+)'", option_text)
        if m:
            new_sound = m.group(1)
            # Try Fx definition format first
            modified = code
            for fx in ['midFx', 'highFx', 'leadFx', 'voxFx', 'stabFx']:
                fx_pattern = rf'(let\s+{fx}\s*=\s*p\s*=>\s*p.*?\.sound\(")[^"]+(")'
                modified = re.sub(fx_pattern, rf'\g<1>{new_sound}\g<2>', modified, flags=re.DOTALL)
            if modified != code:
                return modified, f"Changed lead/melodic sound to {new_sound}"
            # Fallback: arrange() format - replace .sound() on melodic lines (octave 3+)
            if 'arrange(' in code:
                lines = code.split('\n')
                changed = False
                for i, line in enumerate(lines):
                    if _classify_arrange_line(line) == 'melodic':
                        new_line = re.sub(r'(\.sound\(")[^"]+(")', rf'\g<1>{new_sound}\g<2>', line)
                        if new_line != line:
                            lines[i] = new_line
                            changed = True
                if changed:
                    return '\n'.join(lines), f"Changed lead/melodic sound to {new_sound}"

    elif "Change drum bank" in option_text:
        m = re.search(r"to '([^']+)'", option_text)
        if m:
            new_bank = m.group(1)
            # Try Fx definition format first
            modified = code
            for fx in ['drumsFx', 'kickFx', 'snareFx', 'hhFx']:
                fx_pattern = rf'(let\s+{fx}\s*=\s*p\s*=>\s*p.*?\.bank\(")[^"]+(")'
                modified = re.sub(fx_pattern, rf'\g<1>{new_bank}\g<2>', modified, flags=re.DOTALL)
            if modified != code:
                return modified, f"Changed drum bank to {new_bank}"
            # Fallback: arrange() format - replace .bank() on drum lines
            if 'arrange(' in code:
                lines = code.split('\n')
                changed = False
                for i, line in enumerate(lines):
                    if _classify_arrange_line(line) == 'drums':
                        new_line = re.sub(r'(\.bank\(")[^"]+(")', rf'\g<1>{new_bank}\g<2>', line)
                        if new_line != line:
                            lines[i] = new_line
                            changed = True
                if changed:
                    return '\n'.join(lines), f"Changed drum bank to {new_bank}"

    elif "No change" in option_text:
        return code, "No structural change needed"

    return code, "Could not apply change"


def _call_constrained_llm(prompt: str, use_ollama: bool = False) -> Optional[int]:
    """Call LLM with constrained prompt, expecting a single number response."""
    # Try Ollama first if requested
    if use_ollama and HAS_REQUESTS:
        ollama_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10, "num_ctx": 8192}
                },
                timeout=60
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            m = re.search(r'\d+', text)
            if m:
                return int(m.group())
        except Exception as e:
            print(f"       Constrained LLM (Ollama) error: {e}")

    # Try Claude API
    if Anthropic is not None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                m = re.search(r'\d+', text)
                if m:
                    return int(m.group())
            except Exception as e:
                print(f"       Constrained LLM (Claude) error: {e}")

    return None
