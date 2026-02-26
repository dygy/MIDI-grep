#!/usr/bin/env python3
"""
Agentic Ollama with ClickHouse Integration

One agent per track with:
- Persistent conversation history
- SQL queries to ClickHouse for data-driven decisions
- Context compression when approaching limits
- Memory of what was tried and what failed
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import genre-aware sound retrieval
try:
    from sound_selector import retrieve_genre_context
    HAS_SOUND_SELECTOR = True
except ImportError:
    HAS_SOUND_SELECTOR = False

# ClickHouse connection
CLICKHOUSE_BIN = Path(__file__).parent.parent.parent / "bin" / "clickhouse"
CLICKHOUSE_DB = Path(__file__).parent.parent.parent / ".clickhouse" / "db"
AGENTS_DIR = Path(__file__).parent.parent.parent / ".cache" / "agents"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "midi-grep-strudel")

# ============================================================================
# VALID STRUDEL SOUNDS (from gm.mjs + synth.mjs)
# ============================================================================

# Basic waveforms and synths
VALID_SYNTHS = {
    "sine", "sin", "triangle", "tri", "square", "sqr", "sawtooth", "saw",
    "supersaw", "pulse", "sbd", "bytebeat",
    "pink", "white", "brown", "crackle",
    "zzfx", "z_sine", "z_sawtooth", "z_triangle", "z_square", "z_tan", "z_noise"
}

# Valid GM instrument names (from Strudel's gm.mjs)
VALID_GM_SOUNDS = {
    "gm_piano", "gm_epiano1", "gm_epiano2", "gm_harpsichord", "gm_clavinet",
    "gm_celesta", "gm_glockenspiel", "gm_music_box", "gm_vibraphone",
    "gm_marimba", "gm_xylophone", "gm_tubular_bells", "gm_dulcimer",
    "gm_drawbar_organ", "gm_percussive_organ", "gm_rock_organ", "gm_church_organ",
    "gm_reed_organ", "gm_accordion", "gm_harmonica", "gm_bandoneon",
    "gm_acoustic_guitar_nylon", "gm_acoustic_guitar_steel",
    "gm_electric_guitar_jazz", "gm_electric_guitar_clean",
    "gm_electric_guitar_muted", "gm_overdriven_guitar",
    "gm_distortion_guitar", "gm_guitar_harmonics",
    "gm_acoustic_bass", "gm_electric_bass_finger", "gm_electric_bass_pick",
    "gm_fretless_bass", "gm_slap_bass_1", "gm_slap_bass_2",
    "gm_synth_bass_1", "gm_synth_bass_2",
    "gm_violin", "gm_viola", "gm_cello", "gm_contrabass",
    "gm_tremolo_strings", "gm_pizzicato_strings", "gm_orchestral_harp", "gm_timpani",
    "gm_string_ensemble_1", "gm_string_ensemble_2",
    "gm_synth_strings_1", "gm_synth_strings_2",
    "gm_choir_aahs", "gm_voice_oohs", "gm_synth_choir", "gm_orchestra_hit",
    "gm_trumpet", "gm_trombone", "gm_tuba", "gm_muted_trumpet",
    "gm_french_horn", "gm_brass_section", "gm_synth_brass_1", "gm_synth_brass_2",
    "gm_soprano_sax", "gm_alto_sax", "gm_tenor_sax", "gm_baritone_sax",
    "gm_oboe", "gm_english_horn", "gm_bassoon", "gm_clarinet",
    "gm_piccolo", "gm_flute", "gm_recorder", "gm_pan_flute",
    "gm_blown_bottle", "gm_shakuhachi", "gm_whistle", "gm_ocarina",
    "gm_lead_1_square", "gm_lead_2_sawtooth", "gm_lead_3_calliope",
    "gm_lead_4_chiff", "gm_lead_5_charang", "gm_lead_6_voice",
    "gm_lead_7_fifths", "gm_lead_8_bass_lead",
    "gm_pad_new_age", "gm_pad_warm", "gm_pad_poly", "gm_pad_choir",
    "gm_pad_bowed", "gm_pad_metallic", "gm_pad_halo", "gm_pad_sweep",
    "gm_fx_rain", "gm_fx_soundtrack", "gm_fx_crystal", "gm_fx_atmosphere",
    "gm_fx_brightness", "gm_fx_goblins", "gm_fx_echoes", "gm_fx_sci_fi",
    "gm_sitar", "gm_banjo", "gm_shamisen", "gm_koto",
    "gm_kalimba", "gm_bagpipe", "gm_fiddle", "gm_shanai",
    "gm_tinkle_bell", "gm_agogo", "gm_steel_drums", "gm_woodblock",
    "gm_taiko_drum", "gm_melodic_tom", "gm_synth_drum",
    "gm_reverse_cymbal", "gm_guitar_fret_noise", "gm_breath_noise",
    "gm_seashore", "gm_bird_tweet", "gm_telephone",
    "gm_helicopter", "gm_applause", "gm_gunshot"
}

# Valid drum banks (from strudel-client/website/.vercel/output/static/tidal-drum-machines.json)
VALID_DRUM_BANKS = {
    # Roland
    "RolandTR505", "RolandTR606", "RolandTR626", "RolandTR707", "RolandTR727",
    "RolandTR808", "RolandTR909",
    "RolandCompurhythm78", "RolandCompurhythm1000", "RolandCompurhythm8000",
    "RolandD110", "RolandD70", "RolandDDR30", "RolandJD990",
    "RolandMC202", "RolandMC303", "RolandMT32", "RolandR8",
    "RolandS50", "RolandSH09", "RolandSystem100",
    # Linn
    "LinnDrum", "Linn9000", "LinnLM1", "LinnLM2",
    # Akai
    "AkaiLinn", "AkaiMPC60", "AkaiXR10",
    # Boss
    "BossDR55", "BossDR110", "BossDR220", "BossDR550",
    # Korg
    "KorgDDM110", "KorgKPR77", "KorgKR55", "KorgKRZ",
    "KorgM1", "KorgMinipops", "KorgPoly800", "KorgT3",
    # Casio
    "CasioRZ1", "CasioSK1", "CasioVL1",
    # Emu
    "EmuDrumulator", "EmuModular", "EmuSP12",
    # Alesis / Oberheim
    "AlesisHR16", "AlesisSR16", "OberheimDMX",
    # Sequential Circuits
    "SequentialCircuitsDrumtracks", "SequentialCircuitsTom",
    # Yamaha
    "YamahaRM50", "YamahaRX21", "YamahaRX5", "YamahaRY30", "YamahaTG33",
    # Simmons
    "SimmonsSDS400", "SimmonsSDS5",
    # Others
    "AJKPercusyn", "DoepferMS404", "MFB512", "MPC1000",
    "MoogConcertMateMG1", "RhodesPolaris", "RhythmAce",
    "SakataDPM48", "SergeModular", "SoundmastersR88",
    "UnivoxMicroRhythmer12", "ViscoSpaceDrum", "XdrumLM8953",
}

# All valid sounds combined
VALID_SOUNDS = VALID_SYNTHS | VALID_GM_SOUNDS | VALID_DRUM_BANKS

# Invalid GM sound patterns that LLMs hallucinate (with numbers that don't exist)
INVALID_GM_PATTERNS = [
    r'gm_pad_\d+_',      # e.g., gm_pad_4_choir (should be gm_pad_choir)
    r'gm_fx_\d+_',       # e.g., gm_fx_1_rain (should be gm_fx_rain)
    r'gm_electric_piano_\d+',  # e.g., gm_electric_piano_1 (should be gm_epiano1)
    r'gm_acoustic_grand',      # Not in Strudel (use gm_piano)
    r'gm_bright_acoustic',     # Not in Strudel
    r'gm_honkytonk',           # Not in Strudel
]


class OllamaAgent:
    """
    Agentic Ollama wrapper with:
    - Persistent chat history per track
    - ClickHouse SQL tool via ReAct pattern
    - Iteration memory (what was tried, what failed)
    - Context compression
    - Voice splicing (only send bad voices to LLM, keep good ones intact)
    """

    def __init__(self, track_hash: str, model: str = None):
        self.track_hash = track_hash
        self.model = model or DEFAULT_MODEL
        self.messages: List[Dict] = []
        self.iteration_history: List[Dict] = []
        self.best_similarity = 0.0
        self.best_code = ""
        self.tried_values: Dict[str, List[str]] = {}  # param -> [values tried]
        self.max_context_tokens = 6000  # Conservative limit
        self.last_validation_error: Optional[str] = None  # Track validation failures
        self._previous_code: str = ""  # For voice splicing
        self._voices_to_fix: List[str] = []  # Which voices need changes

        # Ensure agents directory exists
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.history_file = AGENTS_DIR / f"{track_hash}.json"

        # Load existing history if available
        self.load_history()

        # Initialize with system prompt if new session
        if not self.messages:
            self.messages = [{"role": "system", "content": self._system_prompt()}]

    def _system_prompt(self) -> str:
        return f"""You are a Strudel live coding AI that makes SURGICAL improvements to match a target audio.

Track hash: {self.track_hash}

## DATABASE ACCESS

Query with: <sql>YOUR QUERY</sql>

Tables: midi_grep.runs (track_hash, version, similarity_overall, similarity_mfcc, similarity_chroma, strudel_code, genre, bpm, band_bass, band_mid, band_high), midi_grep.knowledge (track_hash, parameter_name, parameter_old_value, parameter_new_value, similarity_improvement, genre)

## CRITICAL RULES

1. You will be told which voice(s) to fix (bass, lead, or drums). Output ONLY those voice blocks.
2. Each voice block starts with a comment (// Bass) followed by $: arrange(...)
3. NEVER change the arrange() structure (keep same number of [cycles, pattern] pairs)
4. Change AT MOST one parameter per iteration (gain OR lpf OR sound, not multiple)
5. Maximum gain change: 0.2 per step (0.8→0.6, not 0.8→0.3)
6. NEVER use '...' or 'no changes' — always output the full voice block code
7. Gain range: 0.05 to 1.0
8. NEVER repeat failed parameter values — check what was already tried

## FORMAT

Output in a ```javascript block. Only include the voice blocks you were asked to fix.
Example for fixing bass only:
```javascript
// Bass
$: arrange(
  [7, note("cs1 ~ ~ cs1").sound("sawtooth").gain(0.3).lpf(200)],
  [1, note("cs2 ~ ~ ~").sound("sawtooth").gain(0.2).lpf(150)]
)
```"""

    def load_history(self):
        """Load conversation history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.messages = data.get("messages", [])
                self.iteration_history = data.get("iteration_history", [])
                self.best_similarity = data.get("best_similarity", 0.0)
                self.best_code = data.get("best_code", "")
                self.tried_values = data.get("tried_values", {})
                print(f"  [Agent] Loaded history: {len(self.messages)} messages, best={self.best_similarity*100:.1f}%")
            except Exception as e:
                print(f"  [Agent] Could not load history: {e}")

    def save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "iteration_history": self.iteration_history,
                    "best_similarity": self.best_similarity,
                    "best_code": self.best_code,
                    "tried_values": self.tried_values,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"  [Agent] Could not save history: {e}")

    def estimate_tokens(self) -> int:
        """Rough token count estimation (4 chars per token)."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        return total_chars // 4

    def compress_context(self):
        """Compress old messages when context too long."""
        if self.estimate_tokens() < self.max_context_tokens * 0.8:
            return

        print(f"  [Agent] Compressing context ({self.estimate_tokens()} tokens)...")

        # Summarize iteration history
        history_summary = self._summarize_history()

        # Keep system prompt + summary + last 4 messages
        self.messages = [
            self.messages[0],  # System prompt
            {"role": "user", "content": f"## SESSION HISTORY\n{history_summary}"},
            {"role": "assistant", "content": "I understand the history. I will try different approaches."},
            *self.messages[-4:]  # Last 2 turns
        ]

        print(f"  [Agent] Compressed to {self.estimate_tokens()} tokens")

    def _summarize_history(self) -> str:
        """Create a summary of what was tried."""
        lines = [
            f"Track: {self.track_hash}",
            f"Best similarity achieved: {self.best_similarity*100:.1f}%",
            f"Total iterations: {len(self.iteration_history)}",
            "",
            "## What Was Tried (DO NOT REPEAT):"
        ]

        for param, values in self.tried_values.items():
            lines.append(f"- {param}: {', '.join(values[:5])}{'...' if len(values) > 5 else ''}")

        lines.append("")
        lines.append("## Recent Iterations:")
        for h in self.iteration_history[-5:]:
            status = "✓" if h.get("improved") else "✗"
            lines.append(f"- {status} v{h.get('version', '?')}: {h.get('similarity', 0)*100:.1f}% - {h.get('notes', '')}")

        return "\n".join(lines)

    def execute_sql(self, sql: str) -> str:
        """Execute SQL query on ClickHouse."""
        sql = sql.strip()
        if not sql:
            return "ERROR: Empty query"

        # Security: only allow SELECT
        if not sql.upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries allowed"

        cmd = [
            str(CLICKHOUSE_BIN), "local",
            "--path", str(CLICKHOUSE_DB),
            "--query", sql,
            "--format", "PrettyCompact"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return f"ERROR: {result.stderr[:500]}"
            output = result.stdout.strip()
            if not output:
                return "(no results)"
            # Limit output size
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Query timeout"
        except Exception as e:
            return f"ERROR: {str(e)[:200]}"

    def add_iteration_result(
        self,
        iteration: int,
        version: int,
        similarity: float,
        band_diffs: Dict[str, float],
        code_generated: str,
        improved: bool,
        genre: str = "",
        worst_sections: Optional[List[Dict]] = None,
        section_scores: Optional[List[Dict]] = None,
        per_stem_scores: Optional[Dict[str, float]] = None,
        orig_bands: Optional[Dict[str, float]] = None,
        rend_bands: Optional[Dict[str, float]] = None
    ):
        """Add result of an iteration for learning."""
        # Track what was tried
        self._extract_tried_values(code_generated)

        # Update best
        if similarity > self.best_similarity:
            self.best_similarity = similarity
            self.best_code = code_generated

        # Add to history
        notes = []
        if band_diffs.get("bass", 0) > 0.1:
            notes.append("bass too loud")
        elif band_diffs.get("bass", 0) < -0.1:
            notes.append("bass too quiet")
        if band_diffs.get("mid", 0) > 0.1:
            notes.append("mid too loud")
        elif band_diffs.get("mid", 0) < -0.1:
            notes.append("mid too quiet")

        self.iteration_history.append({
            "iteration": iteration,
            "version": version,
            "similarity": similarity,
            "band_diffs": band_diffs,
            "improved": improved,
            "notes": ", ".join(notes) if notes else "balanced",
            "timestamp": datetime.now().isoformat()
        })

        # Add feedback message
        status = "IMPROVEMENT!" if improved else "NO IMPROVEMENT"
        sounds_ctx = ""
        if HAS_SOUND_SELECTOR and genre:
            sounds_ctx = f"\n\n{retrieve_genre_context(genre)}\nUse ONLY these sounds in .sound() and .bank() calls."

        # Build per-stem feedback — this tells the LLM WHICH voices to fix
        stem_str = ""
        if per_stem_scores:
            stem_str = "\n\n## PER-STEM SIMILARITY (this maps to your 3 $: blocks):"
            sorted_stems = sorted(per_stem_scores.items(), key=lambda x: x[1])
            for stem_name, score in sorted_stems:
                voice_map = {"bass": "Bass ($: arrange #1)", "drums": "Drums ($: arrange #3)", "melodic": "Lead ($: arrange #2)"}
                voice = voice_map.get(stem_name, stem_name)
                if score >= 0.80:
                    stem_str += f"\n- {voice}: {score*100:.0f}% — GOOD, do NOT change this voice"
                elif score >= 0.60:
                    stem_str += f"\n- {voice}: {score*100:.0f}% — NEEDS IMPROVEMENT, adjust gain/lpf/sounds"
                else:
                    stem_str += f"\n- {voice}: {score*100:.0f}% — BAD, this voice needs major changes"

        # Build actionable per-band advice from actual band values
        band_advice = self._build_band_advice(band_diffs, orig_bands, rend_bands)

        # Build temporal feedback from worst sections and section scores
        temporal_str = ""
        if worst_sections:
            temporal_str += "\n\nWorst time ranges (fix these in arrange()):"
            for w in worst_sections[:3]:
                issues = ", ".join(w.get("issues", [])[:2]) or "low similarity"
                temporal_str += f"\n- {w.get('stem', '?')} {w.get('time_start', 0):.0f}-{w.get('time_end', 0):.0f}s: {w.get('similarity', 0)*100:.0f}% ({issues})"
        if section_scores:
            temporal_str += "\n\nPer-section scores (Section N = Nth [cycles, pattern] pair in arrange()):"
            for ss in section_scores[:4]:
                issues_str = ", ".join(ss.get("issues", [])[:2]) or "low similarity"
                temporal_str += f"\n- Section {ss['section_idx']+1} ({ss['stem']}): {ss['similarity']*100:.0f}% — {issues_str}"

        # Determine which voices need fixing
        voices_to_fix = []
        voices_good = []
        if per_stem_scores:
            stem_to_voice = {"bass": "bass", "drums": "drums", "melodic": "lead"}
            for stem_name, score in per_stem_scores.items():
                voice = stem_to_voice.get(stem_name, stem_name)
                if score < 0.80:
                    voices_to_fix.append(voice)
                else:
                    voices_good.append(voice)

        if not voices_to_fix:
            voices_to_fix = ["bass"]  # Default to bass if no per-stem data

        # Store for splice_voices later
        self._previous_code = code_generated
        self._voices_to_fix = voices_to_fix

        # Split the code into voices — only show the bad voice(s) to the LLM
        prev_voices = self._split_voices(code_generated)

        # Build the voice-specific prompt
        bad_voice_code = ""
        section_counts = {}
        for v in voices_to_fix:
            block = prev_voices.get(v, "")
            if block:
                bad_voice_code += f"\n{block}\n"
                section_counts[v] = self._count_arrange_entries(block)

        good_voice_note = ""
        if voices_good:
            good_voice_note = f"\nThe following voices are GOOD (>=80%) and will be kept as-is: {', '.join(voices_good)}. Do NOT output them."

        # Build section count requirement string
        count_str = ""
        for v, cnt in section_counts.items():
            count_str += f"\nThe {v} voice has {cnt} sections — your output MUST also have exactly {cnt} [cycles, pattern] pairs."

        feedback = f"""## Iteration {iteration} Result: {status}

Similarity: {similarity*100:.1f}% (best: {self.best_similarity*100:.1f}%)
{stem_str}

## FREQUENCY BAND ANALYSIS (from THIS iteration's render vs original):
{band_advice}
{temporal_str}

## VOICE TO FIX: {', '.join(voices_to_fix)}
{good_voice_note}
{count_str}

## RULES:
1. Output ONLY the modified voice block(s): {', '.join(voices_to_fix)}
2. Each voice must start with a comment (// Bass, // Lead, or // Drums) and $: arrange(...)
3. Your output MUST have the EXACT SAME number of [cycles, pattern] pairs as the input — if it has 7 pairs, output 7 pairs
4. Only change .gain(), .lpf(), .hpf(), .sound(), or note names within existing pairs — do NOT add or remove pairs
5. Change AT MOST one parameter per pair per iteration (e.g. only .gain OR only .lpf, not both)
6. Maximum gain change: 0.2 per iteration (e.g. 0.8→0.6, not 0.8→0.3)
7. If sub_bass is too quiet, try: lower note octave (2→1), lower .lpf(), or use a deeper bass sound
8. If low_mid is too loud, try: lower .lpf() on bass voice (400→200), or raise .hpf()
{'The code you generated worked! Build on this approach.' if improved else ''}
{sounds_ctx}

Current {', '.join(voices_to_fix)} voice code to improve:
```javascript
{bad_voice_code.strip()}
```

Output ONLY the modified {', '.join(voices_to_fix)} voice block(s) in a ```javascript block.
CRITICAL: Must have EXACTLY {', '.join(f'{cnt} pairs' for cnt in section_counts.values())} in arrange().
Do NOT output the other voices. I will handle splicing them back together."""

        self.messages.append({"role": "user", "content": feedback})

        # Compress if needed
        self.compress_context()

        # Save
        self.save_history()

    def _build_band_advice(self, band_diffs: Dict[str, float],
                          orig_bands: Optional[Dict[str, float]],
                          rend_bands: Optional[Dict[str, float]]) -> str:
        """Build actionable per-band advice from comparison data."""
        lines = []
        bands_info = [
            ("sub_bass", "20-60Hz", "Bass $: voice — affects kick sub and bass fundamental"),
            ("bass", "60-250Hz", "Bass $: voice — main bass body"),
            ("low_mid", "250-500Hz", "Bass $: and Lead $: voices — bass harmonics, lead low end"),
            ("mid", "500-2kHz", "Lead $: voice — main melodic content"),
            ("high_mid", "2-4kHz", "Lead $: and Drums $: voices — presence, hi-hat detail"),
            ("high", "4-20kHz", "Drums $: voice — cymbals, brightness"),
        ]

        for band_key, freq_range, affected_voice in bands_info:
            diff = band_diffs.get(band_key, 0)
            orig_val = orig_bands.get(band_key, 0) if orig_bands else 0
            rend_val = rend_bands.get(band_key, 0) if rend_bands else 0

            if abs(diff) < 0.03:
                continue  # Skip bands that are close enough

            direction = "TOO LOUD" if diff > 0 else "TOO QUIET"
            ratio = rend_val / max(orig_val, 0.001)

            line = f"- {band_key} ({freq_range}): {direction} — orig={orig_val*100:.1f}%, yours={rend_val*100:.1f}%"

            # Add specific actionable advice
            if band_key == "sub_bass" and diff < -0.10:
                line += f"\n  FIX: Use deeper bass notes (octave 1), lower .lpf() to ~150Hz, or try sound with more sub (sawtooth, gm_synth_bass_2)"
            elif band_key == "sub_bass" and diff > 0.10:
                line += f"\n  FIX: Raise bass .hpf() to cut sub-bass, or lower .gain() on bass voice"
            elif band_key == "low_mid" and diff > 0.10:
                line += f"\n  FIX: Lower .lpf() on bass voice (e.g. 400→200), this band has too much energy from bass harmonics"
            elif band_key == "low_mid" and diff < -0.10:
                line += f"\n  FIX: Raise .lpf() on bass voice, or raise bass .gain() slightly"
            elif band_key == "bass" and diff > 0.10:
                line += f"\n  FIX: Lower .gain() on bass voice by 0.1-0.2"
            elif band_key == "bass" and diff < -0.10:
                line += f"\n  FIX: Raise .gain() on bass voice by 0.1-0.2"
            elif band_key == "mid" and abs(diff) > 0.05:
                line += f"\n  FIX: Adjust .gain() on lead voice by 0.1-0.2"
            elif band_key in ("high_mid", "high") and abs(diff) > 0.05:
                line += f"\n  FIX: Adjust .gain() on drums or raise/lower .lpf() on lead"

            lines.append(line)

        if not lines:
            return "All frequency bands are well-balanced. Focus on timbre/sound choices."

        return "\n".join(lines)

    @staticmethod
    def _split_voices(code: str) -> Dict[str, str]:
        """Split Strudel code into named voice blocks.

        Returns {"header": "// comments\\nsetcps(...)\\n",
                 "bass": "// Bass\\n$: arrange(...)\\n",
                 "lead": "// Lead\\n$: arrange(...)\\n",
                 "drums": "// Drums\\n$: arrange(...)\\n"}
        """
        result = {"header": "", "bass": "", "lead": "", "drums": ""}

        # Find all $: block start positions
        lines = code.split('\n')
        block_starts = []  # (line_index, voice_name)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('$:'):
                # Look for comment line above (skipping blanks)
                comment = ""
                for j in range(i - 1, max(i - 3, -1), -1):
                    if lines[j].strip().startswith('//'):
                        comment = lines[j].strip().lower()
                        break
                    elif lines[j].strip():
                        break

                if 'bass' in comment:
                    name = "bass"
                elif 'lead' in comment or 'mid' in comment or 'melody' in comment:
                    name = "lead"
                elif 'drum' in comment:
                    name = "drums"
                else:
                    # Infer from content
                    rest = '\n'.join(lines[i:i+5]).lower()
                    if 's(' in rest and ('bd' in rest or 'sd' in rest):
                        name = "drums"
                    elif re.search(r'note\(["\'][a-g]s?[12]', rest):
                        name = "bass"
                    else:
                        name = "lead"

                # Include the comment line(s) above as part of this block
                start_line = i
                for j in range(i - 1, max(i - 3, -1), -1):
                    if lines[j].strip().startswith('//') or not lines[j].strip():
                        start_line = j
                    else:
                        break

                block_starts.append((start_line, name))

        if not block_starts:
            return result

        # Header is everything before first voice block
        result["header"] = '\n'.join(lines[:block_starts[0][0]]).rstrip()

        # Extract each voice block
        for idx, (start, name) in enumerate(block_starts):
            if idx + 1 < len(block_starts):
                end = block_starts[idx + 1][0]
            else:
                end = len(lines)
            block_code = '\n'.join(lines[start:end]).strip()
            result[name] = block_code

        return result

    @staticmethod
    def _count_arrange_entries(voice_code: str) -> int:
        """Count the number of [cycles, pattern] entries in an arrange() block."""
        return len(re.findall(r'\[\s*\d+\s*,', voice_code))

    def splice_voices(self, new_code: str, previous_code: str) -> str:
        """Splice modified voice(s) from new_code into previous_code.

        If new_code has all 3 valid voices with matching section counts, returns new_code.
        If new_code is incomplete (missing voices, has '...', wrong section count),
        fills in missing/broken voices from previous_code.
        """
        # Parse both into voice blocks
        prev_voices = self._split_voices(previous_code)
        new_voices = self._split_voices(new_code)

        # Use header from previous (LLM sometimes drops it)
        header = prev_voices.get("header") or new_voices.get("header", "")

        # For each voice: use new if valid AND has correct section count, else keep previous
        parts = [header] if header else []
        spliced = []
        for voice_name in ["bass", "lead", "drums"]:
            new_block = new_voices.get(voice_name, "")
            prev_block = prev_voices.get(voice_name, "")

            # Count arrange entries in both
            prev_count = self._count_arrange_entries(prev_block) if prev_block else 0
            new_count = self._count_arrange_entries(new_block) if new_block else 0

            # New block is valid if it has arrange() with actual content
            new_valid = (
                new_block
                and 'arrange(' in new_block
                and '...' not in new_block
                and 'no changes' not in new_block.lower()
                and len(new_block) > 30  # Not just a stub
            )

            # Also reject if section count differs (LLM deleted sections)
            if new_valid and prev_count > 0 and new_count != prev_count:
                reason = f"section count mismatch: {new_count} vs {prev_count}"
                print(f"  [Agent] Rejected LLM {voice_name}: {reason}")
                new_valid = False

            if new_valid:
                parts.append(new_block)
            elif prev_block:
                parts.append(prev_block)
                spliced.append(voice_name)

        if spliced:
            print(f"  [Agent] Spliced original voices: {', '.join(spliced)} (LLM output was incomplete)")

        return '\n\n'.join(parts)

    def _extract_tried_values(self, code: str):
        """Extract parameter values from code to track what was tried."""
        # Extract gain values
        gain_matches = re.findall(r'\.gain\(([^)]+)\)', code)
        for g in gain_matches:
            self.tried_values.setdefault("gain", []).append(g[:50])

        # Extract lpf values
        lpf_matches = re.findall(r'\.lpf\((\d+)\)', code)
        for l in lpf_matches:
            self.tried_values.setdefault("lpf", []).append(l)

        # Extract sounds
        sound_matches = re.findall(r'\.sound\("([^"]+)"\)', code)
        for s in sound_matches:
            self.tried_values.setdefault("sound", []).append(s)

        # Keep only last 20 values per param
        for param in self.tried_values:
            self.tried_values[param] = self.tried_values[param][-20:]

    def generate(self, context: Dict = None) -> str:
        """
        Generate improved Strudel code using agentic loop.

        The agent can execute multiple SQL queries before generating code.
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests package not installed")

        # Add context if provided
        if context:
            genre = context.get('genre', 'unknown')
            sounds_ctx = ""
            if HAS_SOUND_SELECTOR:
                sounds_ctx = f"\n\n{retrieve_genre_context(genre)}\nUse ONLY these sounds in .sound() and .bank() calls."

            context_msg = f"""## Current Track Context

Genre: {genre}
BPM: {context.get('bpm', 120)}
Current similarity: {context.get('similarity', 0)*100:.1f}%

Frequency issues:
- Bass: {context.get('band_bass', 0)*100:+.1f}%
- Mid: {context.get('band_mid', 0)*100:+.1f}%
- High: {context.get('band_high', 0)*100:+.1f}%
{sounds_ctx}

Query the database using track_hash='{self.track_hash}' to find what worked for THIS track, then generate improved code."""

            self.messages.append({"role": "user", "content": context_msg})

        # Agentic loop - allow multiple SQL queries
        max_turns = 5
        for turn in range(max_turns):
            # Call Ollama
            response = self._call_ollama()

            if not response:
                print(f"  [Agent] No response from Ollama")
                break

            # Check for SQL queries
            sql_matches = re.findall(r'<sql>(.*?)</sql>', response, re.DOTALL | re.IGNORECASE)

            if sql_matches:
                # Execute queries and continue
                results = []
                for sql in sql_matches:
                    print(f"  [Agent] Executing SQL: {sql[:80]}...")
                    result = self.execute_sql(sql)
                    results.append(f"Query:\n{sql}\n\nResult:\n{result}")

                # Add results and continue
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": "SQL Results:\n\n" + "\n\n---\n\n".join(results) + "\n\nNow generate the improved code based on these results."
                })
            else:
                # No SQL, this is the final response
                self.messages.append({"role": "assistant", "content": response})
                self.save_history()
                return response

        self.save_history()
        return self.messages[-1].get("content", "") if self.messages else ""

    def _call_ollama(self) -> str:
        """Make a single call to Ollama."""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4096,
                        "num_ctx": 32768,
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            print(f"  [Agent] Ollama not running at {OLLAMA_URL}")
            return ""
        except Exception as e:
            print(f"  [Agent] Ollama error: {e}")
            return ""

    def extract_code(self, response: str, previous_code: str = None) -> str:
        """Extract Strudel code from agent response, splicing with previous code if needed."""
        self.last_validation_error = None  # Reset validation error

        # Use stored previous code if not provided
        if previous_code is None:
            previous_code = self._previous_code

        raw_code = self._extract_raw_code(response)
        if not raw_code:
            return ""

        # If we have previous code, splice the LLM output back into the full code
        if previous_code:
            spliced = self.splice_voices(raw_code, previous_code)
            if spliced and spliced != raw_code:
                # Validate the spliced result
                result = self._validate_code(spliced)
                if result:
                    return result
            # Fall through to validate raw code if splice didn't help

        return self._validate_code(raw_code) or ""

    def _extract_raw_code(self, response: str) -> str:
        """Extract raw Strudel code from LLM response (before validation/splicing)."""
        # 1. Prefer explicitly tagged javascript/js code blocks
        js_match = re.search(r'```(?:javascript|js)\n?([\s\S]*?)```', response)
        if js_match:
            code = js_match.group(1).strip()
            if self._looks_like_strudel(code):
                return code

        # 2. Try all code blocks, pick the one with Strudel patterns
        all_blocks = re.findall(r'```(?:\w*)\n?([\s\S]*?)```', response)
        for block in all_blocks:
            block = block.strip()
            # Skip SQL, JSON, or very short blocks
            if re.match(r'(?i)^\s*(SELECT|INSERT|DELETE|CREATE|\{)', block):
                continue
            if len(block) < 20:
                continue
            if self._looks_like_strudel(block):
                return block

        # 3. Look for effect function patterns (let bassFx = p => p...)
        fx_pattern = r'let\s+\w+Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
        fx_matches = re.findall(fx_pattern, response)
        if fx_matches:
            code = '\n\n'.join(fx_matches)
            if self._looks_like_strudel(code):
                return code

        # 4. Look for bare Strudel code (lines starting with $: or setcps)
        strudel_lines = []
        in_strudel = False
        for line in response.split('\n'):
            stripped = line.strip()
            if stripped.startswith('setcps(') or stripped.startswith('$:') or stripped.startswith('//'):
                in_strudel = True
            if in_strudel:
                if stripped and not stripped.startswith('#') and not stripped.startswith('*'):
                    strudel_lines.append(line)
                elif not stripped and strudel_lines:
                    strudel_lines.append(line)
                elif stripped and strudel_lines:
                    break
        if strudel_lines:
            code = '\n'.join(strudel_lines).strip()
            if len(code) > 20 and self._looks_like_strudel(code):
                return code

        return ""

    @staticmethod
    def _looks_like_strudel(code: str) -> bool:
        """Check if code looks like Strudel (has at least one valid pattern)."""
        indicators = ['setcps(', '$:', '.sound(', '.gain(', '.bank(', 'note(', 's(', '.lpf(', '.hpf(', 'arrange(']
        return any(ind in code for ind in indicators)

    def _fix_bank_names(self, code: str) -> str:
        """Auto-correct common LLM drum bank name hallucinations."""
        BANK_CORRECTIONS = {
            'tr808': 'RolandTR808',
            'TR808': 'RolandTR808',
            'tr909': 'RolandTR909',
            'TR909': 'RolandTR909',
            'tr707': 'RolandTR707',
            'TR707': 'RolandTR707',
            'tr606': 'RolandTR606',
            'TR606': 'RolandTR606',
            'linndrum': 'LinnDrum',
            'linn': 'LinnDrum',
            'dr110': 'BossDR110',
            'mpc60': 'AkaiMPC60',
        }
        for wrong, correct in BANK_CORRECTIONS.items():
            pattern = r'(\.bank\(["\'])' + re.escape(wrong) + r'(["\'])'
            if re.search(pattern, code):
                code = re.sub(pattern, r'\g<1>' + correct + r'\2', code)
                print(f"  [Agent] Auto-corrected bank: {wrong} → {correct}")
        return code

    def _fix_sound_names(self, code: str) -> str:
        """Auto-correct common LLM sound name hallucinations before validation."""
        # Map of common wrong names → correct Strudel names
        SOUND_CORRECTIONS = {
            'gm_electric_guitar': 'gm_electric_guitar_clean',
            'gm_electric_piano': 'gm_epiano1',
            'gm_acoustic_guitar': 'gm_acoustic_guitar_nylon',
            'gm_acoustic_piano': 'gm_piano',
            'gm_acoustic_grand_piano': 'gm_piano',
            'gm_grand_piano': 'gm_piano',
            'gm_electric_bass': 'gm_electric_bass_finger',
            'gm_electric_lead': 'gm_lead_2_sawtooth',
            'gm_synth_lead': 'gm_lead_2_sawtooth',
            'gm_synth_pad': 'gm_pad_warm',
            'gm_synth_bass': 'gm_synth_bass_1',
            'gm_organ': 'gm_drawbar_organ',
            'gm_strings': 'gm_string_ensemble_1',
            'gm_synth_strings': 'gm_synth_strings_1',
            'gm_brass': 'gm_brass_section',
            'gm_synth_brass': 'gm_synth_brass_1',
            'gm_choir': 'gm_choir_aahs',
            'gm_flute': 'gm_flute',  # This one is actually correct
            'gm_slap_bass': 'gm_slap_bass_1',
            'gm_bass': 'gm_acoustic_bass',
            'gm_lead': 'gm_lead_2_sawtooth',
            'gm_pad': 'gm_pad_warm',
            'gm_fx': 'gm_fx_atmosphere',
            'gm_drum': 'gm_synth_drum',
            'gm_acoustic_electric': 'gm_electric_guitar_clean',
            'gm_electric': 'gm_electric_guitar_clean',
            'gm_guitar': 'gm_acoustic_guitar_nylon',
            'gm_piano1': 'gm_piano',
            'gm_piano2': 'gm_epiano1',
        }
        for wrong, correct in SOUND_CORRECTIONS.items():
            if wrong == correct:
                continue
            # Only replace exact matches in .sound("...") calls, not partial matches
            # e.g., fix gm_electric_guitar but not gm_electric_guitar_clean
            pattern = r'(\.sound\(["\'])' + re.escape(wrong) + r'(["\'])'
            if re.search(pattern, code):
                code = re.sub(pattern, r'\g<1>' + correct + r'\2', code)
                print(f"  [Agent] Auto-corrected sound: {wrong} → {correct}")
        return code

    def _validate_code(self, code: str) -> str:
        """
        Validate Strudel code and reject if it contains invalid methods or sounds.

        Returns empty string if code is invalid, otherwise returns the code.
        """
        # Auto-correct common sound/bank name mistakes before validation
        code = self._fix_sound_names(code)
        code = self._fix_bank_names(code)

        # Known invalid methods that LLMs sometimes hallucinate
        INVALID_METHODS = [
            '.peak(',      # Doesn't exist - maybe confused with .hpf or EQ peak
            '.eq(',        # Not a Strudel method (use .lpf/.hpf)
            '.volume(',    # Should be .gain()
            '.filter(',    # Too generic, use specific filters
            '.bass(',      # Not a method
            '.treble(',    # Not a method
            '.mid(',       # Not a method
            '.high(',      # Not a method (voice selector, not effect)
            '.low(',       # Not a method (voice selector, not effect)
            '.boost(',     # Not a method
            '.cut(',       # Not a method (use .lpf/.hpf)
            '.compress(',  # Not a method (use .compressor)
            '.limit(',     # Not a method
            '.normalize(', # Not a method
        ]

        for invalid in INVALID_METHODS:
            if invalid in code:
                msg = f"REJECTED: Code contains invalid method {invalid} - LLM hallucinated a non-existent Strudel method"
                print(f"  [Agent] {msg}")
                self.last_validation_error = msg
                return ""

        # Check for invalid GM sound patterns (LLM often hallucinates numbered names)
        for pattern in INVALID_GM_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                msg = f"REJECTED: Code contains invalid sound pattern {matches[0]} - use correct Strudel GM names (e.g., gm_pad_choir not gm_pad_4_choir)"
                print(f"  [Agent] {msg}")
                self.last_validation_error = msg
                return ""

        # Extract and validate all sounds in .sound("...") calls
        sound_matches = re.findall(r'\.sound\(["\']([^"\']+)["\']', code)
        for sound in sound_matches:
            # Handle alternation patterns like "<sound1 sound2>"
            sound_names = sound.strip('<>').split()
            for s in sound_names:
                s = s.strip()
                if s and s not in VALID_SOUNDS:
                    msg = f"REJECTED: Unknown sound '{s}' - not in Strudel's sound library"
                    print(f"  [Agent] {msg}")
                    self.last_validation_error = msg
                    return ""

        # Extract and validate all banks in .bank("...") calls
        bank_matches = re.findall(r'\.bank\(["\']([^"\']+)["\']', code)
        for bank in bank_matches:
            bank = bank.strip()
            if bank and bank not in VALID_DRUM_BANKS:
                msg = f"REJECTED: Unknown drum bank '{bank}' - not in Strudel's drum library"
                print(f"  [Agent] {msg}")
                self.last_validation_error = msg
                return ""

        # Additional check: ensure at least one valid Strudel pattern
        valid_patterns = [
            '.sound(', '.gain(', '.lpf(', '.hpf(', '.room(', '.delay(', '.bank(',
            '.attack(', '.release(', '.decay(', '.sustain(',
            '.crush(', '.distort(', '.phaser(', '.vibrato(',
            'note(', 's(', 'setcps(', '$:',
        ]
        has_valid = any(p in code for p in valid_patterns)

        if not has_valid:
            msg = "REJECTED: Code has no recognizable Strudel patterns"
            print(f"  [Agent] {msg}")
            self.last_validation_error = msg
            return ""

        return code

    def reset(self):
        """Reset agent state for fresh start."""
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self.iteration_history = []
        self.tried_values = {}
        # Keep best_similarity and best_code
        self.save_history()


def test_agent():
    """Quick test of agent functionality."""
    print("Testing OllamaAgent...")

    agent = OllamaAgent("test_track_123")

    # Test SQL execution
    print("\n1. Testing SQL execution:")
    result = agent.execute_sql("SELECT count() FROM midi_grep.runs")
    print(f"   Result: {result}")

    # Test generation with context
    print("\n2. Testing code generation:")
    response = agent.generate({
        "genre": "brazilian_funk",
        "bpm": 136,
        "similarity": 0.75,
        "band_bass": 0.15,
        "band_mid": -0.10,
        "band_high": 0.05
    })
    print(f"   Response length: {len(response)} chars")

    # Extract code
    code = agent.extract_code(response)
    print(f"   Extracted code length: {len(code)} chars")
    if code:
        print(f"   First 200 chars:\n{code[:200]}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_agent()
