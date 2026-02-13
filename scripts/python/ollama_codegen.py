#!/usr/bin/env python3
"""
Ollama-powered Strudel code generator.

Replaces the old rule-based ai_code_generator.py.
Uses Ollama LLM to generate dynamic, section-aware Strudel code
directly from audio analysis data.
"""

import argparse
import json
import sys
import os
import re

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Import sound validation from ollama_agent
try:
    from ollama_agent import VALID_SOUNDS, VALID_DRUM_BANKS, INVALID_GM_PATTERNS
except ImportError:
    VALID_SOUNDS = set()
    VALID_DRUM_BANKS = set()
    INVALID_GM_PATTERNS = []

# Import ClickHouse best-run lookup
try:
    from ai_improver import get_best_run, get_track_hash
    HAS_LEARNING = True
except ImportError:
    HAS_LEARNING = False


def build_prompt(bpm, key, genre, style, drum_kit, sections, duration, notes_json=None):
    """Build the LLM prompt with all audio context."""

    # Load transcribed notes if available
    notes_text = ""
    if notes_json and os.path.exists(notes_json):
        try:
            with open(notes_json) as f:
                notes = json.load(f)
            if isinstance(notes, list) and len(notes) > 0:
                pitches = sorted(set(n.get("note", n.get("pitch", "")) for n in notes[:200] if n.get("note") or n.get("pitch")))
                if pitches:
                    notes_text = f"\nTranscribed notes: {', '.join(pitches[:20])}"
        except Exception:
            pass

    root = key.split()[0].lower() if key else "c"
    kit = drum_kit or 'RolandTR808'

    # Build section entries for arrange()
    section_entries = []
    if sections:
        max_energy = max(s.get("energy", 0.5) for s in sections) or 1.0
        for i, s in enumerate(sections):
            dur = s.get("duration", 30)
            cycles = max(1, round(dur * bpm / 240))
            rel = s.get("energy", 0.5) / max_energy
            if rel > 0.7:
                level = "HIGH"
            elif rel > 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"
            section_entries.append((cycles, level, i + 1))
    else:
        section_entries = [(4, "LOW", 1), (8, "HIGH", 2), (4, "MEDIUM", 3), (2, "LOW", 4)]

    # Build the example arrange() with all sections for each voice
    def build_voice_arrange(voice_type):
        lines = []
        for cycles, level, idx in section_entries:
            if voice_type == "bass":
                if level == "HIGH":
                    lines.append(f'  [{cycles}, note("{root}2 {root}2 ~ {root}2").sound("gm_synth_bass_1").gain(0.8).lpf(400)]')
                elif level == "MEDIUM":
                    lines.append(f'  [{cycles}, note("{root}2 ~ {root}2 ~").sound("gm_synth_bass_1").gain(0.5).lpf(300)]')
                else:
                    lines.append(f'  [{cycles}, note("{root}2 ~ ~ ~").sound("gm_acoustic_bass").gain(0.3).lpf(200).room(0.3)]')
            elif voice_type == "lead":
                if level == "HIGH":
                    lines.append(f'  [{cycles}, note("{root}4 {root}4 {root}4 {root}4").sound("gm_lead_2_sawtooth").gain(0.7).lpf(6000)]')
                elif level == "MEDIUM":
                    lines.append(f'  [{cycles}, note("{root}4 ~ {root}4 ~").sound("gm_piano").gain(0.5).lpf(5000)]')
                else:
                    lines.append(f'  [{cycles}, note("{root}4 ~ ~ ~").sound("gm_piano").gain(0.3).lpf(4000).room(0.3)]')
            else:  # drums
                if level == "HIGH":
                    lines.append(f'  [{cycles}, s("bd sd hh hh bd sd hh oh").bank("{kit}").gain(0.8)]')
                elif level == "MEDIUM":
                    lines.append(f'  [{cycles}, s("bd ~ sd hh").bank("{kit}").gain(0.6)]')
                else:
                    lines.append(f'  [{cycles}, s("bd ~ ~ hh").bank("{kit}").gain(0.4).room(0.3)]')
        return ",\n".join(lines)

    prompt = f"""Generate Strudel live coding music.

BPM: {bpm}, Key: {key}, Genre: {genre or 'electronic'}{notes_text}

The code MUST have EXACTLY this structure — 3 `$:` blocks, one per voice.
Each voice has ONE arrange() containing ALL {len(section_entries)} sections as [cycles, pattern] pairs.

EXAMPLE (modify the note patterns, sounds, and gains to match the genre):
```javascript
setcps({bpm}/60/4)

// Bass
$: arrange(
{build_voice_arrange("bass")}
)

// Lead
$: arrange(
{build_voice_arrange("lead")}
)

// Drums
$: arrange(
{build_voice_arrange("drums")}
)
```

CRITICAL RULES:
1. EXACTLY 3 `$:` blocks — bass, lead, drums. NO MORE, NO LESS.
2. Each `$:` has ONE arrange() with ALL sections inside it.
3. DO NOT create separate `$:` blocks per section — sections go INSIDE arrange().
4. Bass: note() in octave 2 with .sound()
5. Lead: note() in octave 4 with .sound()
6. Drums: s("bd sd hh oh") with .bank("{kit}") — NEVER use note names for drums
7. LOW energy: sparse (use ~), gain 0.2-0.4, add .room(0.3)
8. HIGH energy: dense, gain 0.7-0.9, add .distort(0.3) or .crush(4)
9. NO semicolons, NO .peak(), NO .volume(), NO .eq()
10. Valid sounds: gm_acoustic_bass, gm_synth_bass_1, gm_piano, gm_epiano1, gm_lead_2_sawtooth, gm_pad_warm, gm_string_ensemble_1
11. Valid banks: {kit} (use full name, NOT "tr808")

Output ONLY the code in a ```javascript block. No explanation."""

    return prompt


def call_ollama(prompt, model=DEFAULT_MODEL):
    """Call Ollama and return the response text."""
    if not HAS_REQUESTS:
        print("ERROR: requests package not installed", file=sys.stderr)
        return None

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 4096,
                }
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Ollama not running at {OLLAMA_URL}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Ollama call failed: {e}", file=sys.stderr)
        return None


def fix_strudel_syntax(code):
    """Fix common LLM syntax mistakes in Strudel code."""
    # Remove semicolons (Strudel is not JS)
    code = code.replace(';', '')

    # Fix missing commas between arrange() array entries:
    # ] // comment\n  [ → ], // comment\n  [
    # Only add comma if not already present
    def add_arrange_commas(match):
        bracket = match.group(1)  # ]
        after = match.group(2) or ''  # optional comment
        ws = match.group(3)  # whitespace before next [
        return f'],{after}\n{ws}['
    code = re.sub(r'(\])(?!,)(\s*//[^\n]*)?\n(\s*)\[', add_arrange_commas, code)

    # Fix drum bank names: tr808 → RolandTR808, tr909 → RolandTR909
    code = re.sub(r'\.bank\(["\']tr808["\']\)', '.bank("RolandTR808")', code, flags=re.IGNORECASE)
    code = re.sub(r'\.bank\(["\']tr909["\']\)', '.bank("RolandTR909")', code, flags=re.IGNORECASE)
    code = re.sub(r'\.bank\(["\']tr707["\']\)', '.bank("RolandTR707")', code, flags=re.IGNORECASE)
    code = re.sub(r'\.bank\(["\']tr606["\']\)', '.bank("RolandTR606")', code, flags=re.IGNORECASE)

    # Remove empty method calls: .room() → remove, .lpf() → remove, etc.
    code = re.sub(r'\.(room|size|lpf|hpf|delay|delaytime|delayfeedback|crush|distort|pan|attack|decay|sustain|release|vibrato|phaser)\(\)', '', code)

    # Fix .gain("0.5") → .gain(0.5) (string number to actual number)
    code = re.sub(r'\.gain\("(\d+\.?\d*)"\)', r'.gain(\1)', code)

    # Fix setcps with float: setcps(136.0/60/4) → setcps(136/60/4)
    code = re.sub(r'setcps\((\d+)\.0/', r'setcps(\1/', code)

    # Fix invalid drum sounds in s() patterns with .bank()
    # Valid sounds: bd, sd, hh, oh, cp, lt, mt, ht, cb, cy, rs, rim, cl, ma, cow, tom, ride, crash, clap
    # Invalid: note names like c2, e3; or hallucinated names like kss, snr, etc.
    VALID_DRUM_TOKENS = {
        'bd', 'sd', 'hh', 'oh', 'cp', 'lt', 'mt', 'ht', 'cb', 'cy', 'rs',
        'rim', 'cl', 'ma', 'cow', 'tom', 'ride', 'crash', 'clap',
        'ch',  # closed hat alias
        '~',  # rest
    }

    def fix_drum_pattern(match):
        pattern = match.group(1)
        tokens = pattern.split()
        fixed_tokens = []
        changed = False
        for t in tokens:
            t_stripped = t.strip()
            if not t_stripped or t_stripped in VALID_DRUM_TOKENS:
                fixed_tokens.append(t)
            elif re.match(r'^[a-g]#?\d$', t_stripped):
                # Note name — replace with drum sound
                fixed_tokens.append('bd')
                changed = True
            else:
                # Unknown drum name — replace with closest match
                changed = True
                if t_stripped.startswith('k') or t_stripped.startswith('b'):
                    fixed_tokens.append('bd')
                elif t_stripped.startswith('s') or t_stripped.startswith('r'):
                    fixed_tokens.append('sd')
                elif t_stripped.startswith('h') or t_stripped.startswith('c'):
                    fixed_tokens.append('hh')
                else:
                    fixed_tokens.append('hh')
        if changed:
            return f's("{" ".join(fixed_tokens)}")'
        return match.group(0)

    code = re.sub(r's\("([^"]+)"\)(?=\s*\.bank\()', fix_drum_pattern, code)

    # Fix common sound name shortcuts
    sound_fixes = {
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
        'gm_slap_bass': 'gm_slap_bass_1',
        'gm_bass': 'gm_acoustic_bass',
        'gm_lead': 'gm_lead_2_sawtooth',
        'gm_pad': 'gm_pad_warm',
        'gm_fx': 'gm_fx_atmosphere',
        'gm_drum': 'gm_synth_drum',
    }
    for wrong, correct in sound_fixes.items():
        # Only replace exact matches (not partial)
        pattern = r'(\.sound\(["\'])' + re.escape(wrong) + r'(["\'])'
        code = re.sub(pattern, r'\g<1>' + correct + r'\2', code)

    return code


def enforce_three_voices(code):
    """Ensure code has exactly 3 $: blocks (bass, lead, drums).

    LLMs often generate extra $: blocks (one per section instead of one per voice).
    This function keeps only the first 3 $: blocks and discards the rest.
    """
    # Split code into $: blocks
    # Pattern: find each $: and its content until the next $: or end
    blocks = re.split(r'(?=^\$:)', code, flags=re.MULTILINE)

    # First element is everything before the first $: (header/setcps)
    header = blocks[0] if blocks else ""
    voice_blocks = [b for b in blocks[1:] if b.strip()]

    if len(voice_blocks) <= 3:
        return code  # Already 3 or fewer, no change needed

    print(f"  [enforce_three_voices] Found {len(voice_blocks)} $: blocks, keeping first 3", file=sys.stderr)

    # Keep first 3 blocks (typically bass, lead, drums)
    result = header.rstrip('\n') + '\n\n'
    for i, block in enumerate(voice_blocks[:3]):
        result += block.rstrip('\n') + '\n\n'

    return result.rstrip('\n') + '\n'


def extract_code(response):
    """Extract Strudel code from LLM response."""
    if not response:
        return None

    # Try javascript/js code blocks first
    js_match = re.search(r'```(?:javascript|js)\n?([\s\S]*?)```', response)
    if js_match:
        return js_match.group(1).strip()

    # Try any code block that looks like Strudel
    for block in re.findall(r'```(?:\w*)\n?([\s\S]*?)```', response):
        block = block.strip()
        if re.match(r'(?i)^\s*(SELECT|INSERT|DELETE|CREATE|\{)', block):
            continue
        strudel_indicators = ['setcps(', '$:', '.sound(', '.gain(', '.bank(', 'note(', 's(', 'arrange(']
        if any(ind in block for ind in strudel_indicators):
            return block

    # Try bare Strudel code
    lines = []
    in_code = False
    for line in response.split('\n'):
        s = line.strip()
        if s.startswith('setcps(') or s.startswith('$:') or s.startswith('//'):
            in_code = True
        if in_code:
            if s or lines:
                lines.append(line)
    if lines:
        code = '\n'.join(lines).strip()
        if len(code) > 30:
            return code

    return None


def generate_fallback(bpm, key, genre, drum_kit, sections):
    """Generate minimal dynamic code without LLM (fallback)."""
    kit = drum_kit or "RolandTR808"

    # Build arrange sections based on energy
    if sections and len(sections) > 1:
        max_energy = max(s.get("energy", 0.5) for s in sections) or 1.0
        bass_sections = []
        drum_sections = []
        lead_sections = []

        root = key.replace(" minor", "").replace(" major", "").lower()
        is_minor = "minor" in key.lower()
        # Simple scale degrees
        if is_minor:
            third = f"{root}3"  # approximate
            fifth = f"{root}3"
        else:
            third = f"{root}3"
            fifth = f"{root}3"

        for s in sections:
            rel = s.get("energy", 0.5) / max_energy
            dur = s.get("duration", 30)
            cycles = max(1, round(dur * bpm / 240))
            gain = f"{min(0.9, 0.3 + rel * 0.6):.2f}"

            if rel > 0.7:
                bass_sections.append(f'  [{cycles}, note("{root}2 {root}2 {root}2 {root}2").sound("gm_acoustic_bass").gain({gain}).lpf(400)]')
                drum_sections.append(f'  [{cycles}, s("bd sd hh hh bd sd hh oh").bank("{kit}").gain({gain})]')
                lead_sections.append(f'  [{cycles}, note("{root}4 {root}4 {root}4 {root}4").sound("gm_piano").gain({gain}).lpf(5000)]')
            elif rel > 0.4:
                bass_sections.append(f'  [{cycles}, note("{root}2 ~ {root}2 ~").sound("gm_acoustic_bass").gain({gain}).lpf(400)]')
                drum_sections.append(f'  [{cycles}, s("bd ~ sd hh").bank("{kit}").gain({gain})]')
                lead_sections.append(f'  [{cycles}, note("{root}4 ~ {root}4 ~").sound("gm_piano").gain({gain}).lpf(5000).room(0.2)]')
            else:
                bass_sections.append(f'  [{cycles}, note("{root}2 ~ ~ ~").sound("gm_acoustic_bass").gain({gain}).lpf(300)]')
                drum_sections.append(f'  [{cycles}, s("bd ~ ~ hh").bank("{kit}").gain({gain}).room(0.3)]')
                lead_sections.append(f'  [{cycles}, note("{root}4 ~ ~ ~").sound("gm_piano").gain({gain}).lpf(4000).room(0.3)]')

        bass_arr = ",\n".join(bass_sections)
        drum_arr = ",\n".join(drum_sections)
        lead_arr = ",\n".join(lead_sections)

        return f"""// MIDI-grep AI-generated output
// BPM: {bpm}, Key: {key}
// Sections: {len(sections)} detected

setcps({bpm}/60/4)

// Bass
$: arrange(
{bass_arr}
)

// Lead
$: arrange(
{lead_arr}
)

// Drums
$: arrange(
{drum_arr}
)"""
    else:
        root = key.replace(" minor", "").replace(" major", "").lower()
        return f"""// MIDI-grep AI-generated output
// BPM: {bpm}, Key: {key}

setcps({bpm}/60/4)

// Bass
$: note("{root}2 ~ {root}2 ~")
  .sound("gm_acoustic_bass")
  .gain(0.60)
  .lpf(400)

// Lead
$: note("{root}4 ~ {root}4 ~")
  .sound("gm_piano")
  .gain(0.50)
  .lpf(5000)
  .room(0.25)

// Drums
$: s("bd sd hh hh bd sd hh oh")
  .bank("{kit}")
  .gain(0.70)"""


def main():
    parser = argparse.ArgumentParser(description="Ollama-powered Strudel code generator")
    parser.add_argument("audio_path", help="Path to audio file (for context)")
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--key", default="C minor")
    parser.add_argument("--genre", default="")
    parser.add_argument("--style", default="auto")
    parser.add_argument("--drum-kit", default="RolandTR808")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--track-hash", default="")
    parser.add_argument("--sections-json", help="Path to smart_analysis.json with section data")
    parser.add_argument("--notes-json", help="Path to transcribed notes JSON")
    parser.add_argument("--drums-only", action="store_true")
    parser.add_argument("--drums-json", help="Path to drum detection JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    # Load sections from smart analysis
    sections = []
    if args.sections_json and os.path.exists(args.sections_json):
        try:
            with open(args.sections_json) as f:
                data = json.load(f)
            sections = data.get("sections", [])
            print(f"Loaded {len(sections)} sections from analysis", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load sections: {e}", file=sys.stderr)

    # Check ClickHouse for best previous run
    best_previous = None
    if args.track_hash and HAS_LEARNING:
        try:
            best = get_best_run(args.track_hash)
            if best and best.get("strudel_code") and best.get("similarity_overall", 0) > 0.5:
                # Validate the code doesn't have hallucinated sounds
                code = best["strudel_code"]
                has_invalid = False
                if VALID_SOUNDS:
                    for m in re.finditer(r'\.sound\(["\']([^"\']+)["\']', code):
                        for s in m.group(1).strip('<>').split():
                            if s.strip() and s.strip() not in VALID_SOUNDS:
                                has_invalid = True
                                break
                if not has_invalid:
                    best_previous = {
                        "code": code,
                        "similarity": best["similarity_overall"],
                        "version": best.get("version", 0),
                    }
                    print(f"Found best previous run: v{best_previous['version']} ({best_previous['similarity']*100:.0f}%)", file=sys.stderr)
                else:
                    print("Best previous run has invalid sounds, ignoring", file=sys.stderr)
        except Exception as e:
            print(f"Warning: ClickHouse lookup failed: {e}", file=sys.stderr)

    # Drums-only mode
    if args.drums_only:
        kit = args.drum_kit or "RolandTR808"
        if sections and len(sections) > 1:
            prompt = f"""Generate ONLY a drum pattern in Strudel for BPM {args.bpm}, genre {args.genre or 'unknown'}.
Use `arrange()` with different density per section. Use `.bank("{kit}")`.

Sections:
"""
            max_energy = max(s.get("energy", 0.5) for s in sections) or 1.0
            for i, s in enumerate(sections):
                rel = s.get("energy", 0.5) / max_energy
                cycles = max(1, round(s.get("duration", 30) * args.bpm / 240))
                prompt += f"  Section {i+1}: {cycles} cycles, {'HIGH' if rel > 0.7 else 'MEDIUM' if rel > 0.4 else 'LOW'} energy\n"
            prompt += f"\nOutput ONLY the drum code in a ```javascript block. Start with setcps({args.bpm}/60/4)."

            response = call_ollama(prompt, args.model)
            code = extract_code(response) if response else None
            if code:
                code = fix_strudel_syntax(code)
            if not code:
                code = f'setcps({args.bpm}/60/4)\n\n$: s("bd sd hh hh bd sd hh oh")\n  .bank("{kit}")\n  .gain(0.70)'
        else:
            code = f'setcps({args.bpm}/60/4)\n\n$: s("bd sd hh hh bd sd hh oh")\n  .bank("{kit}")\n  .gain(0.70)'

        output = {"code": code}
        if best_previous:
            output["best_previous"] = best_previous
        print(json.dumps(output))
        return

    # Build prompt and call Ollama
    prompt = build_prompt(
        args.bpm, args.key, args.genre, args.style,
        args.drum_kit, sections, args.duration, args.notes_json
    )

    print(f"Calling Ollama ({args.model}) for code generation...", file=sys.stderr)
    response = call_ollama(prompt, args.model)
    code = extract_code(response) if response else None

    if code:
        code = fix_strudel_syntax(code)
        code = enforce_three_voices(code)
        print(f"Ollama generated {len(code)} chars of Strudel code", file=sys.stderr)
    else:
        print("Ollama failed or returned no code, using fallback", file=sys.stderr)
        code = generate_fallback(args.bpm, args.key, args.genre, args.drum_kit, sections)

    # Add header comment
    if not code.startswith("//"):
        header = f"// MIDI-grep AI-generated output\n// BPM: {args.bpm:.0f}, Key: {args.key}\n\n"
        code = header + code

    output = {"code": code}
    if best_previous:
        output["best_previous"] = best_previous

    print(json.dumps(output))


if __name__ == "__main__":
    main()
