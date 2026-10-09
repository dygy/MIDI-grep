#!/usr/bin/env python3
"""Orchestrated (job-based) Strudel code generation.

Drop-in alternative to ollama_codegen.py: accepts the same CLI args and prints Strudel to stdout,
so the Go pipeline can switch to it behind `--codegen orchestrated`.

Instead of one giant prompt, generation is decomposed into small validated jobs:
  - structure  : section layout (cycle counts) from BPM/sections
  - voice.bass : sound + per-section note pattern + dynamics (octave 2)
  - voice.lead : sound + per-section note pattern + dynamics (octave 4)
  - drums      : drum bank + per-section pattern (bd/sd/hh/oh)

Each job's JSON is validated (sounds checked against strudel_validation's single source of truth),
retried with error feedback, or falls back to a logged deterministic default. A pure assembler then
builds the 3-voice arrange() Strudel — so structure, setcps(), and the 3-voice contract are
guaranteed by construction, not by the LLM. A job_run.json is written next to the output for
observability (which jobs retried / fell back).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

from job_runner import JobRunner, JobSpec
from strudel_validation import VALID_SOUNDS, VALID_DRUM_BANKS, validate_code, fix_names

try:
    from ollama_codegen import fix_strudel_syntax, enforce_three_voices
    HAS_FIXERS = True
except ImportError:
    HAS_FIXERS = False

try:
    from sound_selector import retrieve_genre_context, GENRE_PALETTES
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    GENRE_PALETTES = {}

try:
    from sound_timbre import (
        resolve_sound, analyze_stem_timbre_by_section, analyze_stem_activity_by_section,
        analyze_stem_envelope, analyze_drum_density_per_cycle, analyze_stem_pitch_by_section,
        resolve_drum_bank, analyze_stem_drum_pattern,
    )
    HAS_TIMBRE = True
except ImportError:
    HAS_TIMBRE = False
    def analyze_stem_timbre_by_section(*_a, **_kw):  # type: ignore[misc]
        return []
    def analyze_stem_activity_by_section(*_a, **_kw):  # type: ignore[misc]
        return []
    def analyze_stem_envelope(*_a, **_kw):  # type: ignore[misc]
        return []
    def analyze_drum_density_per_cycle(*_a, **_kw):  # type: ignore[misc]
        return []
    def analyze_stem_pitch_by_section(*_a, **_kw):  # type: ignore[misc]
        return []
    def resolve_drum_bank(_p, candidates, _c):  # type: ignore[misc]
        return candidates[0] if candidates else "RolandTR808"
    def analyze_stem_drum_pattern(*_a, **_kw):  # type: ignore[misc]
        return []

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "midi-grep-strudel-mistral")

# Octave-appropriate, genre-neutral fallback sounds (valid by construction).
FALLBACK_SOUNDS = {"bass": "gm_synth_bass_1", "lead": "gm_lead_2_sawtooth"}
DRUM_LETTERS = {"bd", "sd", "hh", "oh", "cp", "rs", "lt", "mt", "ht", "cy", "~"}


def log(msg: str):
    print(msg, file=sys.stderr)


# The default model's Modelfile SYSTEM prompt tells it to emit full Strudel code in a ```javascript
# block — which directly fights the job prompts' "JSON only". Override it per-job with a JSON-first
# system prompt AND constrain Ollama to valid-JSON output (format=json). This is the single biggest
# reliability win for the job path (was causing non-JSON responses + retries).
JOB_SYSTEM = (
    "You are a music-arrangement assistant. You produce ONE small part of a Strudel arrangement at a "
    "time. ALWAYS respond with a single valid JSON object and NOTHING else — no prose, no explanation, "
    "no markdown, no code fences. Use only the exact keys the user asks for."
)


def llm_call(prompt: str) -> str:
    """Single Ollama generate call. keep_alive 5m so the model stays warm across the job sequence;
    the caller unloads it before the render phase."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": DEFAULT_MODEL,
                "system": JOB_SYSTEM,        # override the code-focused Modelfile system prompt
                "prompt": prompt,
                "format": "json",            # constrain Ollama to emit valid JSON
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.6, "num_predict": 700, "num_ctx": 8192},
            },
            timeout=180,
        )
        return r.json().get("response", "")
    except Exception as e:  # noqa: BLE001
        log(f"  [llm] error: {e}")
        return ""


_STRUDEL_VALIDATOR = Path(__file__).resolve().parent.parent / "node" / "strudel-validate.mjs"


def _strudel_parse_errors(code: str) -> list:
    """Validate code with Strudel's REAL parser (node @strudel/mini). Returns a list of
    {pattern, error} dicts (empty = valid). Degrades to [] if node/the validator is unavailable."""
    if not _STRUDEL_VALIDATOR.exists():
        return []
    try:
        import subprocess
        out = subprocess.run(["node", str(_STRUDEL_VALIDATOR)], input=code,
                             capture_output=True, text=True, timeout=30)
        # @strudel/core prints a banner to stdout before our JSON — slice from the first '{'.
        raw = out.stdout
        i = raw.find("{")
        data = json.loads(raw[i:]) if i >= 0 else {}
        return data.get("errors", []) if not data.get("ok", True) else []
    except Exception:  # noqa: BLE001
        return []


def _unload_model():
    """Free the ~13GB model before the render phase (keep_alive 0)."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": DEFAULT_MODEL, "keep_alive": 0}, timeout=10)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Full-song timeline: derive how many cycles fill the real track duration so the
# arrange() spans the whole song instead of looping a short fragment.
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> float:
    """Real duration (seconds) of the input stem via ffprobe; 0.0 if unknown."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip()) if out.stdout.strip() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _total_cycles(bpm: float, duration: float) -> int:
    """Cycles that span `duration` seconds (cps = bpm/60/4). Clamped to a sane range."""
    if duration <= 0:
        return 0
    cps = (bpm / 60.0) / 4.0
    return max(16, min(400, round(duration * cps)))


def _scale_sections_to_duration(sections: list, total_cycles: int) -> list:
    """Scale each section's `cycles` proportionally so they sum to total_cycles — making the
    arrange() cover the whole track instead of a short loop. Preserves the section shape."""
    if total_cycles <= 0 or not sections:
        return sections
    cur = sum(max(1, int(s.get("cycles", 1))) for s in sections)
    if cur <= 0:
        return sections
    scaled, used = [], 0
    for i, s in enumerate(sections):
        if i == len(sections) - 1:
            cyc = max(1, total_cycles - used)  # last section absorbs rounding
        else:
            cyc = max(1, round(int(s.get("cycles", 1)) * total_cycles / cur))
            used += cyc
        scaled.append({**s, "cycles": cyc})
    return scaled


# ---------------------------------------------------------------------------
# Job: structure
# ---------------------------------------------------------------------------

_ALL_VOICES = ["bass", "lead", "drums"]


def _structure_prompt(ctx, deps):
    return f"""Plan the full-song section arc for a {ctx['genre']} track at {ctx['bpm']} BPM.
Output JSON ONLY:
{{"sections": [
  {{"name": "intro", "cycles": 2, "voices": ["lead"]}},
  {{"name": "build", "cycles": 3, "voices": ["lead", "bass"]}},
  {{"name": "drop", "cycles": 4, "voices": ["bass", "lead", "drums"]}},
  {{"name": "break", "cycles": 2, "voices": ["lead", "drums"]}},
  {{"name": "drop2", "cycles": 4, "voices": ["bass", "lead", "drums"]}},
  {{"name": "outro", "cycles": 2, "voices": ["lead"]}}
]}}
RULES:
- 4-6 sections forming a real arc (intro → build → drop → break → drop → outro).
- cycles are RELATIVE weights (integers 1-6) — scaled to the real track length later.
- "voices": which of bass/lead/drums ACTUALLY PLAY in that section. Real arrangements drop voices
  in/out — an intro is often lead-only, a drop has all three, a break thins out. Do NOT put all
  three voices in every section; make sections differ by WHO plays, not just volume."""


def _structure_validate(p, ctx):
    secs = p.get("sections")
    if not isinstance(secs, list) or not (3 <= len(secs) <= 7):
        return ["need 3-7 sections forming a full-song arc"]
    for s in secs:
        if not isinstance(s.get("cycles"), int) or not (1 <= s["cycles"] <= 8):
            return ["each section needs an integer relative weight 1-8 in 'cycles'"]
        v = s.get("voices")
        if not isinstance(v, list) or not v or any(x not in _ALL_VOICES for x in v):
            return ["each section needs a non-empty 'voices' list from bass/lead/drums"]
    if all(set(s.get("voices", [])) == set(_ALL_VOICES) for s in secs):
        return ["vary which voices play per section — not all three everywhere (drop voices in/out)"]
    return []


def _structure_fallback(ctx):
    return {"sections": [
        {"name": "intro", "cycles": 2, "voices": ["lead"]},
        {"name": "build", "cycles": 3, "voices": ["lead", "bass"]},
        {"name": "drop", "cycles": 4, "voices": ["bass", "lead", "drums"]},
        {"name": "break", "cycles": 2, "voices": ["lead", "drums"]},
        {"name": "drop2", "cycles": 4, "voices": ["bass", "lead", "drums"]},
        {"name": "outro", "cycles": 2, "voices": ["lead"]},
    ]}


# ---------------------------------------------------------------------------
# Job: a melodic voice (bass / lead)
# ---------------------------------------------------------------------------

def _role_sounds(genre, role):
    """Role-targeted sound RAG for this voice (bass job sees only bass sounds, etc.)."""
    if HAS_RAG and genre:
        try:
            return retrieve_genre_context(genre, role=role)
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _voice_prompt(voice, octave, ctx, deps):
    secs = deps.get("structure", {}).get("sections", [])
    names = ", ".join(s.get("name", f"section {i+1}") for i, s in enumerate(secs))
    n = len(secs)
    rag = _role_sounds(ctx.get("genre", ""), voice)
    # Resolved sound from timbre matching (set by main() before jobs run).
    resolved = (ctx.get("resolved_sounds") or {}).get(voice)
    # Few-shot: a worked example in a DIFFERENT key/genre (so it adapts rather than copies),
    # demonstrating dynamics (gains differ) + density variation (sparse intro -> busy drop).
    if voice == "bass":
        ex_sound = resolved or "gm_synth_bass_1"
        example = (f'{{"sound": "{ex_sound}", "patterns": '
                   '["<[a1 ~ ~ ~] [a1 ~ a1 ~]>", "<[a1 a1 e2 a1] [a1 e2 a1 c2] [a1 a1 g1 a1] [c2 ~ a1 e2]>", "<[a1 ~ e2 ~] [a1 ~ ~ ~]>"], '
                   '"gains": [0.35, 0.85, 0.55], "lpf": 600}')
    else:
        ex_sound = resolved or "gm_lead_2_sawtooth"
        example = (f'{{"sound": "{ex_sound}", "patterns": '
                   '["<[a4 ~ ~ ~] [c5 ~ a4 ~]>", "<[a4 c5 e5 c5] [e5 c5 a4 g4] [a4 b4 c5 d5] [e5 ~ c5 a4]>", "<[e5 c5 a4 ~] [a4 ~ ~ ~]>"], '
                   '"gains": [0.3, 0.8, 0.5], "lpf": 4500}')

    sound_rule = (
        f'- sound: MUST be exactly "{resolved}" (timbre-matched to the original stem).'
        if resolved else
        f'- sound: a real Strudel sound (e.g. {FALLBACK_SOUNDS[voice]}, sawtooth, gm_acoustic_bass).'
    )

    return f"""Write the {voice.upper()} voice for a {ctx['genre']} track in {ctx['key']}, octave {octave}.
{rag}
This track has {n} sections: {names}.

EXAMPLE (A minor, different track — adapt to THIS key/genre, do not copy):
{example}

Now produce the {voice} voice as JSON with EXACTLY these keys: sound, patterns, gains, lpf.
RULES:
{sound_rule}
- patterns: {n} strings (one per section), notes in octave {octave} from the key {ctx['key']}.
- EACH pattern MUST EVOLVE, not loop one bar. Write it as a per-cycle alternation of 2-4 DIFFERENT
  bars: "<[bar1] [bar2] [bar3]>" — Strudel plays one bracketed bar per cycle, so the section changes
  bar-to-bar instead of repeating. Busy sections (drops) = more bars / more notes; quiet sections =
  fewer bars / more ~. Each bar holds note names and ~ ONLY. NO ".slow(", NO parentheses, NO gains.
- gains: {n} values (one per section) for DYNAMICS — quiet intro (0.2-0.4), loud drop/main (0.7-0.9).
  They MUST differ to create a build/drop envelope.
- Make the DROP sections clearly busier (more bars, more notes) than intro/break. lpf: one integer."""


# Note patterns are mini-notation: note names, ~, _, <>, [], *, /, ,, !, @, numbers.
# They must NOT contain method calls or parens — a common failure is the LLM leaking a
# gain/automation pattern like "<0.2 0.8>.slow(16)" into the note string, which is invalid
# mini-notation and renders to silence.
_BAD_PATTERN_TOKEN = re.compile(r'[()]|\.[a-zA-Z]')


def _balanced(s: str) -> bool:
    """Check <> and [] are balanced — an unclosed group (e.g. '<[a4] [b4]' missing '>') is the
    #1 cause of a Strudel mini-notation parse error → silent render."""
    stack = []
    pairs = {")": "(", "]": "[", ">": "<"}
    openers = set(pairs.values())
    for ch in s:
        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _valid_note_pattern(pat) -> bool:
    s = str(pat).strip()
    if not s:
        return False
    if _BAD_PATTERN_TOKEN.search(s) is not None:
        return False
    return _balanced(s)  # reject unclosed <> / [] that crash Strudel's parser


def _voice_validate_factory(n_sections):
    def _v(p, ctx):
        errs = []
        if p.get("sound") not in VALID_SOUNDS:
            errs.append(f"unknown sound '{p.get('sound')}'")
        pats = p.get("patterns")
        if not isinstance(pats, list) or len(pats) < 1:
            errs.append("patterns must be a non-empty list")
        else:
            bad = [pat for pat in pats if not _valid_note_pattern(pat)]
            if bad:
                errs.append(
                    f"pattern {bad[0]!r} is not valid mini-notation — note patterns hold ONLY "
                    f"note names and ~ (no method calls, no '.slow(', no parentheses, no gain numbers)"
                )
        gains = p.get("gains")
        if not isinstance(gains, list) or not gains:
            errs.append("gains must be a non-empty list (one per section, for dynamics)")
        elif any(not isinstance(g, (int, float)) or not (0.05 <= g <= 1.0) for g in gains):
            errs.append("each gain must be 0.05-1.0")
        elif len(set(round(float(g), 2) for g in gains)) == 1 and len(gains) > 1:
            errs.append("gains are all identical — vary them across sections for a build/drop envelope")
        return errs
    return _v


def _section_envelope(n: int) -> list:
    """A simple build/drop gain envelope across n sections (quiet ends, loud middle)."""
    if n <= 1:
        return [0.6]
    if n == 2:
        return [0.5, 0.8]
    base = [0.35] + [0.8] * (n - 2) + [0.5]  # intro quiet, body loud, outro mid
    return base[:n]


def _voice_fallback_factory(voice, octave):
    def _f(ctx):
        secs = ctx.get("_sections", [{"cycles": 4}])
        r, third, fifth = f"c{octave}", f"e{octave}", f"g{octave}"
        # Multi-bar EVOLVING patterns even in fallback (alternating bars per cycle), so a fallback
        # section isn't a static loop. Busy sections get more/denser bars.
        sparse = f"<[{r} ~ ~ ~] [{r} ~ {fifth} ~]>"
        dense = f"<[{r} {third} {fifth} {r}] [{fifth} {third} {r} {third}] [{r} {fifth} {third} {r}] [{third} {r} {fifth} ~]>"
        env = _section_envelope(len(secs))
        return {"sound": FALLBACK_SOUNDS[voice],
                "patterns": [sparse if env[i] < 0.5 else dense for i in range(len(secs))],
                "gains": env,
                "lpf": 800 if voice == "bass" else 4000}
    return _f


# ---------------------------------------------------------------------------
# Job: drums
# ---------------------------------------------------------------------------

def _drums_prompt(ctx, deps):
    secs = deps.get("structure", {}).get("sections", [])
    names = ", ".join(s.get("name", f"section {i+1}") for i, s in enumerate(secs))
    kit = ctx.get("drum_kit") or "RolandTR808"
    n = len(secs)
    banks = _role_sounds(ctx.get("genre", ""), "drums")
    return f"""Write DRUMS for a {ctx['genre']} track using bank {kit}.
{banks}
This track has {n} sections: {names}.

EXAMPLE (different track — adapt, do not copy):
{{"bank": "{kit}", "patterns": ["<[bd ~ ~ ~] [bd ~ sd ~]>", "<[bd hh sd hh] [bd bd sd hh] [bd hh sd hh bd hh sd hh] [bd hh sd cp]>", "<[bd ~ sd ~] [bd ~ ~ ~]>"], "gains": [0.4, 0.85, 0.6]}}

Now produce the drums as JSON with EXACTLY these keys: bank, patterns, gains.
RULES:
- bank: a real Strudel drum bank (e.g. RolandTR808, RolandTR909, LinnDrum). NEVER 'tr808'.
- EACH pattern MUST EVOLVE, not loop: write it as a per-cycle alternation of 2-4 DIFFERENT bars
  "<[bar1] [bar2] [bar3]>" so the beat changes bar-to-bar (add fills, vary kicks/snares/hats).
- Use ONLY these tokens inside bars: bd sd hh oh cp ~ (no 'rs' — TR808 lacks it).
- DROP sections busier (more bars, more hits) than intro/break.
- gains: {n} values (one per section), 0.3-0.9, MUST differ to build/drop."""


# rs (rimshot) is absent from many banks (incl. RolandTR808) and renders silent — exclude it.
DRUM_LETTERS_STRICT = {"bd", "sd", "hh", "oh", "cp", "lt", "mt", "ht", "cy", "~"}


def _gains_validate(gains, n_label="section"):
    if not isinstance(gains, list) or not gains:
        return [f"gains must be a non-empty list (one per {n_label}, for dynamics)"]
    if any(not isinstance(g, (int, float)) or not (0.05 <= g <= 1.0) for g in gains):
        return ["each gain must be 0.05-1.0"]
    if len(gains) > 1 and len(set(round(float(g), 2) for g in gains)) == 1:
        return ["gains are all identical — vary them across sections for a build/drop envelope"]
    return []


def _drums_validate(p, ctx):
    errs = []
    if p.get("bank") not in VALID_DRUM_BANKS:
        errs.append(f"unknown drum bank '{p.get('bank')}'")
    pats = p.get("patterns")
    if not isinstance(pats, list) or len(pats) < 1:
        errs.append("patterns must be a non-empty list")
    else:
        for pat in pats:
            if not _balanced(str(pat)):
                errs.append(f"unbalanced <> or [] in drum pattern {str(pat)[:40]!r} (will crash the parser)")
                break
            cleaned = str(pat)
            for ch in "[]<>*":
                cleaned = cleaned.replace(ch, " ")
            toks = cleaned.split()
            bad = [t for t in toks if not t.isdigit() and t not in DRUM_LETTERS_STRICT]
            if bad:
                errs.append(f"invalid/silent drum tokens {bad[:3]} (use only bd/sd/hh/oh/cp/~)")
                break
    errs += _gains_validate(p.get("gains"))
    return errs


def _drum_pattern_for_density(density: float) -> str:
    """Pick an EVOLVING multi-bar drum pattern whose busyness matches a measured density (0-1).
    Tiers are shifted busy-ward: section-length windows under-read onset rate, and real beats
    (esp. drops) are denser than the metric suggests, so we map mid densities to busy patterns."""
    if density < 0.12:
        return "<[bd ~ ~ ~] [bd ~ sd ~]>"
    if density < 0.28:
        return "<[bd ~ sd ~] [bd hh sd ~] [bd ~ sd hh]>"
    if density < 0.45:
        return "<[bd hh sd hh] [bd bd sd hh] [bd hh sd cp]>"
    if density < 0.62:
        return "<[bd hh sd hh bd hh sd hh] [bd hh sd hh bd bd sd hh] [bd hh sd hh hh bd sd hh]>"
    return "<[bd hh sd hh bd hh sd cp] [bd hh hh sd hh bd sd hh] [hh bd hh sd hh hh sd cp] [bd hh sd hh bd hh sd hh]>"


def _drum_bar_for_density(density: float) -> str:
    """A SINGLE drum bar whose busyness matches a per-cycle density (0-1). Used to build a per-cycle
    drum sequence `<[bar_c0] [bar_c1] …]>` so the drums thin out in breaks and fill in drops exactly
    when the original drums do — the per-cycle DENSITY lever the volume envelope can't provide.
    density==0 → a rest (the cycle's drums drop out), matching the original's silent cycles."""
    if density <= 0.0:
        return "~"
    if density < 0.12:
        return "bd ~ ~ ~"
    if density < 0.28:
        return "bd ~ sd ~"
    if density < 0.45:
        return "bd hh sd hh"
    if density < 0.62:
        return "bd hh sd hh bd hh sd hh"
    return "bd hh sd hh bd bd sd hh"


def _per_cycle_drum_patterns(sections: list, density_per_cycle: list) -> list:
    """One drum pattern per SECTION, each a per-cycle sequence `<[bar] [bar] …>` (one bar per cycle in
    that section) chosen from the original drums' per-cycle density. `<…>` advances one element per
    cycle, so a section of N cycles needs N bars to give each cycle its own density-matched bar."""
    patterns = []
    cursor = 0
    for sec in sections:
        n = max(1, int(sec.get("cycles", 1) or 1))
        bars = []
        for k in range(n):
            d = density_per_cycle[cursor + k] if (cursor + k) < len(density_per_cycle) else 0.3
            bars.append(f"[{_drum_bar_for_density(d)}]")
        cursor += n
        patterns.append("<" + " ".join(bars) + ">")
    return patterns


def _drums_fallback(ctx):
    secs = ctx.get("_sections", [{"cycles": 4}])
    kit = ctx.get("drum_kit") or "RolandTR808"
    env = _section_envelope(len(secs))
    # Evolving multi-bar patterns (alternate per cycle) so even fallback drums aren't a static loop.
    sparse = "<[bd ~ ~ ~] [bd ~ sd ~]>"
    busy = "<[bd hh sd hh] [bd bd sd hh] [bd hh sd hh bd hh sd cp] [bd hh sd hh]>"
    pats = [sparse if env[i] < 0.5 else busy for i in range(len(secs))]
    return {"bank": kit, "patterns": pats, "gains": env}


# ---------------------------------------------------------------------------
# Deterministic assembler — structure guaranteed here, never by the LLM.
# ---------------------------------------------------------------------------

def _arrange(sections, patterns, render_entry):
    """Build an arrange(...) body: one [cycles, <rendered pattern>] per section.
    render_entry(pattern, section_index) lets each section apply its own dynamics."""
    lines = []
    for i, sec in enumerate(sections):
        pat = patterns[i % len(patterns)] if patterns else "~"
        lines.append(f"  [{sec['cycles']}, {render_entry(pat, i)}]")
    return "arrange(\n" + ",\n".join(lines) + "\n)"


def _gain_at(cfg, i, default):
    """Per-section gain — gives each arrange section its own level (build/drop envelope)."""
    gains = cfg.get("gains")
    if isinstance(gains, list) and gains:
        return round(float(gains[i % len(gains)]), 3)
    return cfg.get("gain", default)  # back-compat with a scalar gain


def _brightness_to_lpf(brightness: float, lo: int, hi: int) -> int:
    """Map a 0-1 brightness value linearly to an LPF cutoff in [lo, hi].

    High brightness → filter OPENS (high cutoff = brighter sound).
    Low brightness  → filter CLOSES (low cutoff = darker sound).
    """
    b = max(0.0, min(1.0, brightness))
    return int(lo + b * (hi - lo))


def _lpf_mod(base_lpf: int, slow: int = 8) -> str:
    """Continuous filter movement: a slow sine sweep AROUND the base cutoff, instead of a frozen
    value. Emitted on the METHOD side (valid Strudel signal) — never inside note()."""
    base = max(80, int(base_lpf))
    lo, hi = int(base * 0.55), int(base * 1.35)
    return f"sine.range({lo}, {hi}).slow({slow})"


def _expand_env(env: list, gamma: float) -> list:
    """A2 (low-parameter): nonlinear dynamic-range expansion to pre-compensate the synth's measured
    compression (rendered dynamic range ≈ half the original; soft-clip flattens peaks NON-linearly,
    so Pearson corr caps ~0.8 even after the open-loop envelope). Pearson is invariant to LINEAR
    scaling, so a linear expansion can't move it — we apply `(v/max)**gamma * max` (gamma>1 drops the
    mids more than the peaks → expands range nonlinearly), the inverse shape of a compressive
    soft-clip. gamma==1.0 is a no-op. Single global parameter → NO per-cycle feedback noise (that's
    what sank the per-cycle A2).

    MEASURED NULL (June 2026): on Caravan bass, gamma 1.0→1.8 left shape unchanged (0.716→0.713).
    The macro envelope corr (~0.72) is already near-affine; the residual decorrelation is render/
    separation noise + sub-resolution detail, NOT a gain nonlinearity gamma can invert. Left at the
    1.0 default (no-op); the gain-envelope lever is exhausted for shape. Do not re-enable without a
    new measured win."""
    if gamma == 1.0 or not env:
        return env
    mx = max(env) or 1.0
    return [round((v / mx) ** gamma * mx, 3) for v in env]


def _resolve_stem_dir(piano: str | None, output_dir: str | None) -> str | None:
    """Find the directory that actually contains the original stems (bass/drums/melodic.wav).

    The data-driven features (envelope, A1 pitch, transcribed drums, timbre sound-match) all read
    those three named stems. They USED to assume `Path(args.piano).parent`, but on a FRESH separation
    the pipeline passes a piano stem from a WORKSPACE dir that lacks the named stems — so every
    analysis silently returned [] and the whole data-driven layer was skipped (the generalization bug:
    Caravan worked only because its cached run happened to pass the cache-dir path). The named stems
    live in the CACHE dir = parent of the version `--output-dir`. Try all plausible locations and
    return the first that has the stems; None if none qualify.
    """
    candidates = []
    if piano:
        candidates.append(str(Path(piano).parent))
    if output_dir:
        candidates.append(output_dir)                       # version dir
        candidates.append(str(Path(output_dir).parent))     # cache dir (where stems live)
        candidates.append(str(Path(output_dir).parent.parent))
    seen = set()
    for d in candidates:
        if not d or d in seen:
            continue
        seen.add(d)
        if all(os.path.exists(os.path.join(d, s)) for s in ("bass.wav", "drums.wav", "melodic.wav")):
            return d
    # last resort: piano's parent (back-compat) even if incomplete
    return str(Path(piano).parent) if piano else None


def _env_gain(env: list | None, total_cycles: int, gamma: float = 1.0) -> str:
    """A Strudel gain-automation pattern from the original stem's envelope: one value per cycle,
    spread over the whole track. Makes the rendered voice rise/fall WHEN the original does.

    `<a b c …>` (angle brackets) already advances ONE element per cycle, so N values span N cycles
    with no `.slow()` — and crucially NO `.slow()` on a raw string (that calls String.prototype.slow,
    which is undefined → a runtime TypeError that crashes the whole program → a SILENT render).

    gamma>1 applies the A2 nonlinear range expansion (see _expand_env)."""
    if not env:
        return ""
    env = _expand_env(env, gamma)
    vals = " ".join(str(round(float(v), 3)) for v in env)
    return f'.gain("<{vals}>")'


def assemble(
    ctx,
    results,
    bass_section_lpfs: list[int] | None = None,
    lead_section_lpfs: list[int] | None = None,
    bass_env: list | None = None,
    lead_env: list | None = None,
    drums_env: list | None = None,
) -> str:
    """Build final Strudel code from job results.

    Args:
        ctx:               Orchestrator context (bpm, key, …).
        results:           Dict of JobResult objects keyed by job id.
        bass_section_lpfs: Optional per-section LPF targets for the bass voice.
                           When provided, each section's _lpf_mod sweep is centred
                           on the section's brightness-derived cutoff rather than a
                           single static LPF from the LLM job output.
        lead_section_lpfs: Same for the lead voice.
    """
    bpm = ctx["bpm"]
    sections = results["structure"].output["sections"]
    bass, lead, drums = results["voice.bass"].output, results["voice.lead"].output, results["drums"].output

    bass_lpf_default = int(bass.get("lpf", 800))
    lead_lpf_default = int(lead.get("lpf", 4000))

    def _section_lpf(section_lpfs: list[int] | None, default: int, i: int) -> int:
        """Return the per-section LPF target if available, else fall back to the voice default."""
        if section_lpfs and i < len(section_lpfs):
            return section_lpfs[i]
        return default

    def _present(i, voice):
        # Per-section voice presence: a voice can be ABSENT in a section (silent rest), giving real
        # arrangement contrast (lead-only intro, full drop, thinned break). Defaults to present so
        # older structure outputs without a "voices" key keep playing all three (back-compat).
        v = sections[i].get("voices") if i < len(sections) else None
        return (voice in v) if isinstance(v, list) and v else True

    # When a per-cycle envelope drives a voice (see _env_gain), DON'T also apply the per-section
    # data-driven gain — both come from the original stem, so multiplying them SQUARES the dynamics:
    # quiet sections get pushed below the silence threshold and the shape decorrelates. With an
    # envelope active, the per-entry gain is a flat base and the envelope does all the shaping.
    BASE_GAIN = {"bass": 0.9, "lead": 0.85, "drums": 1.0}

    def bass_entry(pat, i):
        if not _present(i, "bass"):
            return 'note("~")'  # bass silent in this section
        base = _section_lpf(bass_section_lpfs, bass_lpf_default, i)
        g = BASE_GAIN["bass"] if bass_env else _gain_at(bass, i, 0.6)
        return (f'note("{pat}").sound("{bass["sound"]}").gain({g})'
                f'.lpf({_lpf_mod(base, slow=16)})')

    def lead_entry(pat, i):
        if not _present(i, "lead"):
            return 'note("~")'  # lead silent in this section
        base = _section_lpf(lead_section_lpfs, lead_lpf_default, i)
        g = BASE_GAIN["lead"] if lead_env else _gain_at(lead, i, 0.5)
        return (f'note("{pat}").sound("{lead["sound"]}").gain({g})'
                f'.lpf({_lpf_mod(base, slow=8)})')

    def drum_entry(pat, i):
        if not _present(i, "drums"):
            return 's("~")'  # drums silent in this section
        g = BASE_GAIN["drums"] if drums_env else _gain_at(drums, i, 0.7)
        # NB: drum-groove humanization (swing/room/velocity) was tried here and MEASURABLY HURT the
        # stem-shape correlation (0.345 → 0.25–0.30): swing shifts energy off the original's grid,
        # room smears it across the ~0.6s envelope grid, perlin velocity adds uncorrelated variation.
        # Groove is a feel improvement the shape metric penalises, so it is deliberately NOT applied.
        return f's("{pat}").bank("{drums["bank"]}").gain({g})'

    cps = round(bpm / 60 / 4, 4)
    total_cycles = sum(int(s.get("cycles", 4) or 4) for s in sections) or 1
    bass_eg = _env_gain(bass_env, total_cycles)
    lead_eg = _env_gain(lead_env, total_cycles)
    drums_eg = _env_gain(drums_env, total_cycles)
    code = f"""// MIDI-grep orchestrated output
// BPM: {int(bpm)}, Key: {ctx['key']}

setcps({cps})

$: {_arrange(sections, bass["patterns"], bass_entry)}{bass_eg}

$: {_arrange(sections, lead["patterns"], lead_entry)}{lead_eg}

$: {_arrange(sections, drums["patterns"], drum_entry)}{drums_eg}
"""
    # Final safety net: shared name corrections + syntax fixes + 3-voice enforcement.
    code = fix_names(code)
    if HAS_FIXERS:
        code = fix_strudel_syntax(code)
        code = enforce_three_voices(code)
    return code


def build_jobs(ctx) -> list:
    nsec = 3
    return [
        JobSpec("structure", _structure_prompt, _structure_validate, _structure_fallback),
        JobSpec("voice.bass", lambda c, d: _voice_prompt("bass", 2, c, d),
                _voice_validate_factory(nsec), _voice_fallback_factory("bass", 2), deps=["structure"]),
        JobSpec("voice.lead", lambda c, d: _voice_prompt("lead", 4, c, d),
                _voice_validate_factory(nsec), _voice_fallback_factory("lead", 4), deps=["structure"]),
        JobSpec("drums", _drums_prompt, _drums_validate, _drums_fallback, deps=["structure"]),
    ]


# ---------------------------------------------------------------------------
# Phase 4: targeted iteration — re-run ONE voice job and splice it into existing code.
# ---------------------------------------------------------------------------

# Orchestrated output is exactly 3 `$: arrange(...)` blocks in order: bass, lead, drums.
_VOICE_INDEX = {"bass": 0, "lead": 1, "drums": 2, "melodic": 1}
_VOICE_OCTAVE = {"bass": 2, "lead": 4}


def _arrange_blocks(code: str) -> list:
    """Return (start, end) spans of each top-level `$: arrange(...)` block, in order."""
    spans = []
    for m in re.finditer(r'\$:\s*arrange\(', code):
        i = code.index('(', m.start())
        depth, j = 0, i
        while j < len(code):
            if code[j] == '(':
                depth += 1
            elif code[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        spans.append((m.start(), j + 1))
    return spans


def regenerate_voice(voice: str, ctx: dict, sections: list, gap_hint: str = "") -> dict:
    """Re-run a single voice/drums job (validated + retried), optionally with a gap hint.
    Returns the validated job output dict (or a logged fallback)."""
    ctx = {**ctx, "_sections": sections}
    if voice == "drums":
        spec = JobSpec("drums",
                       lambda c, d: _drums_prompt(c, {"structure": {"sections": sections}})
                       + (f"\nFIX THIS GAP: {gap_hint}" if gap_hint else ""),
                       _drums_validate, _drums_fallback)
    else:
        octave = _VOICE_OCTAVE.get(voice, 4)
        spec = JobSpec(f"voice.{voice}",
                       lambda c, d: _voice_prompt(voice, octave, c, {"structure": {"sections": sections}})
                       + (f"\nFIX THIS GAP: {gap_hint}" if gap_hint else ""),
                       _voice_validate_factory(len(sections)), _voice_fallback_factory(voice, octave))
    res = JobRunner(llm_call, log=log).run([spec], ctx)
    return res[spec.id].output


def splice_voice(code: str, voice: str, new_output: dict, ctx: dict, sections: list) -> str:
    """Replace one voice's `$: arrange(...)` block in existing orchestrated code with a freshly
    rendered block from new_output. Returns the spliced code (validated/fixed)."""
    idx = _VOICE_INDEX.get(voice)
    spans = _arrange_blocks(code)
    if idx is None or idx >= len(spans):
        return code  # can't locate the block — leave code unchanged

    if voice == "drums":
        def entry(pat, i):
            return f's("{pat}").bank("{new_output["bank"]}").gain({_gain_at(new_output,i,0.7)})'
    else:
        lpf_def = 800 if voice == "bass" else 4000
        def entry(pat, i):
            return (f'note("{pat}").sound("{new_output["sound"]}")'
                    f'.gain({_gain_at(new_output,i,0.6)}).lpf({int(new_output.get("lpf",lpf_def))})')

    new_block = "$: " + _arrange(sections, new_output["patterns"], entry)
    s, e = spans[idx]
    spliced = code[:s] + new_block + code[e:]
    spliced = fix_names(spliced)
    if HAS_FIXERS:
        spliced = fix_strudel_syntax(spliced)
    return spliced


def main():
    ap = argparse.ArgumentParser(description="Orchestrated (job-based) Strudel codegen")
    ap.add_argument("piano", nargs="?", help="melodic stem path (unused; for arg-compat)")
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--key", default="C major")
    ap.add_argument("--style", default="electronic")
    ap.add_argument("--genre", default="")
    ap.add_argument("--duration", default="30")
    ap.add_argument("--sections-json", default=None)
    ap.add_argument("--track-hash", default=None)
    ap.add_argument("--drum-kit", default=None)
    ap.add_argument("--notes-json", default=None)
    ap.add_argument("--analysis-json", default=None)
    ap.add_argument("--output-dir", default=None, help="where to write job_run.json")
    args = ap.parse_args()

    # Per-role sound RAG is retrieved inside each voice/drums prompt from ctx["genre"]
    # (see _role_sounds), so no shared "rag" string is needed here.
    ctx = {
        "bpm": args.bpm,
        "key": args.key,
        "genre": args.genre or args.style or "electronic",
        "style": args.style,
        "drum_kit": args.drum_kit,
    }
    # Pre-resolve a section list for fallbacks (in case the structure job itself falls back).
    ctx["_sections"] = _structure_fallback(ctx)["sections"]

    # Timbre-based sound resolution: pick the genre-RAG candidate that best matches the
    # original stem's measured timbre. This replaces a pure genre/LLM guess with an
    # acoustically grounded choice. Stored in ctx so _voice_prompt can pin the sound.
    ctx["resolved_sounds"] = {}
    if HAS_TIMBRE and HAS_RAG:
        genre = ctx["genre"]
        # Derive stem directory from the melodic stem path (args.piano).
        stem_dir = _resolve_stem_dir(args.piano, args.output_dir)
        for voice, stem_name in (("bass", "bass.wav"), ("lead", "melodic.wav")):
            stem_path = os.path.join(stem_dir, stem_name) if stem_dir else None
            rag_ctx = retrieve_genre_context(genre, role=voice)
            # Extract candidate sound names from the RAG string (comma-separated after ": ").
            candidates: list[str] = []
            if ": " in rag_ctx:
                raw_candidates = rag_ctx.split(": ", 1)[1].split(", ")
                from strudel_validation import VALID_SOUNDS
                candidates = [c.strip() for c in raw_candidates if c.strip() in VALID_SOUNDS]
            if not candidates:
                from sound_selector import GENRE_PALETTES
                palette = GENRE_PALETTES.get(genre, GENRE_PALETTES["default"])
                role_map = {"bass": "bass", "lead": "lead"}
                candidates = list(palette.get(role_map[voice], []))
            if candidates:
                resolved = resolve_sound(stem_path, candidates)
                ctx["resolved_sounds"][voice] = resolved
                log(f"  [timbre] {voice}: {resolved} (stem={stem_name}, candidates={candidates[:4]})")

    runner = JobRunner(llm_call, log=log)
    results = runner.run(build_jobs(ctx), ctx)
    _unload_model()  # free RAM before the render phase

    # Full-song timeline: scale the structure's relative weights to fill the REAL track duration,
    # so arrange() spans the whole song instead of looping a ~37s fragment.
    duration = _audio_duration(args.piano) or float(args.duration or 0)
    total = _total_cycles(args.bpm, duration)
    if total and "structure" in results:
        secs = results["structure"].output.get("sections", [])
        scaled = _scale_sections_to_duration(secs, total)
        results["structure"].output["sections"] = scaled
        log(f"  [orchestrator] scaled {len(scaled)} sections to {total} cycles "
            f"(~{duration:.0f}s) — total now {sum(s['cycles'] for s in scaled)}")

    # Per-section timbre-driven filter targets: measure brightness arc over the original stems
    # and map each section's brightness to a voice-appropriate LPF cutoff, so the filter opens
    # in bright sections and closes in dark ones (intro/outro).  Wrapped in try/except so any
    # analysis failure falls back gracefully to the static per-voice LPF.
    bass_section_lpfs: list[int] | None = None
    lead_section_lpfs: list[int] | None = None
    bass_env: list | None = None
    lead_env: list | None = None
    drums_env: list | None = None
    try:
        stem_dir = _resolve_stem_dir(args.piano, args.output_dir)
        scaled_sections = results["structure"].output.get("sections", []) if "structure" in results else []
        cps = args.bpm / 60.0 / 4.0

        if stem_dir and scaled_sections and cps > 0:
            for voice, stem_name, lo, hi, attr in (
                ("bass", "bass.wav",     200,  1500, "bass_section_lpfs"),
                ("lead", "melodic.wav", 1500,  9000, "lead_section_lpfs"),
            ):
                stem_path = os.path.join(stem_dir, stem_name)
                brightnesses = analyze_stem_timbre_by_section(
                    stem_path, scaled_sections, total or 0, cps
                )
                if brightnesses and len(brightnesses) == len(scaled_sections):
                    lpfs = [_brightness_to_lpf(b, lo, hi) for b in brightnesses]
                    if attr == "bass_section_lpfs":
                        bass_section_lpfs = lpfs
                    else:
                        lead_section_lpfs = lpfs
                    log(
                        f"  [timbre-arc] {voice}: {len(lpfs)} section lpf targets "
                        f"min={min(lpfs)} max={max(lpfs)} "
                        f"(brightness {min(brightnesses):.2f}–{max(brightnesses):.2f})"
                    )
                else:
                    log(f"  [timbre-arc] {voice}: analysis unavailable, using static lpf")
    except Exception as _e:  # noqa: BLE001
        log(f"  [timbre-arc] error computing per-section lpf: {_e} — falling back to static lpf")
        bass_section_lpfs = None
        lead_section_lpfs = None

    # DATA-DRIVEN arrangement: match the ORIGINAL stems' real per-section activity instead of a
    # generic template. For each section, a voice PLAYS only if the original stem is active there,
    # and its per-section GAIN follows the original stem's dynamic envelope. This fixes the
    # "sparse/blocky vs dense original" mismatch (e.g. bass that plays throughout no longer gets
    # dropped by a template intro). Falls back to the LLM/structure values on any analysis failure.
    try:
        stem_dir = _resolve_stem_dir(args.piano, args.output_dir)
        scaled_sections = results["structure"].output.get("sections", []) if "structure" in results else []
        cps = args.bpm / 60.0 / 4.0
        if stem_dir and scaled_sections and cps > 0:
            VOICE_STEM = {"bass": "bass.wav", "lead": "melodic.wav", "drums": "drums.wav"}
            activity = {}
            for voice, stem in VOICE_STEM.items():
                act = analyze_stem_activity_by_section(os.path.join(stem_dir, stem),
                                                       scaled_sections, total or 0, cps)
                if act and len(act) == len(scaled_sections):
                    activity[voice] = act
            if activity:
                # 1) Presence: a voice plays in a section iff its original stem is active there.
                for i, sec in enumerate(scaled_sections):
                    present = [v for v in ("bass", "lead", "drums")
                               if v in activity and activity[v][i]["active"]]
                    if not present:  # never leave a section fully silent — keep the loudest voice
                        present = [max(activity, key=lambda v: activity[v][i]["gain"])] if activity else ["lead"]
                    sec["voices"] = present
                # 2) Gain: each voice's per-section gain follows the original stem's envelope.
                #    Drums get a boost — TR808/909 samples render quieter than the synth voices, so
                #    the raw envelope gain leaves them ~5x too low; scale up (capped) to sit forward.
                GAIN_BOOST = {"drums": 1.6, "bass": 1.15, "lead": 1.0}
                for voice, jobkey in (("bass", "voice.bass"), ("lead", "voice.lead"), ("drums", "drums")):
                    if voice in activity and jobkey in results:
                        boost = GAIN_BOOST.get(voice, 1.0)
                        results[jobkey].output["gains"] = [
                            round(min(0.97, a["gain"] * boost), 3) for a in activity[voice]
                        ]
                # 3) Drum PATTERN (A1 for drums): prefer the ORIGINAL drums' ACTUAL transcribed hit
                #    pattern (onset-detected + classified bd/sd/hh/oh/cp on a 16-step grid) — lands hits
                #    where the original does. Measured strictly better than density tiers on every dim
                #    (rhythm 0.09→0.14, timbre 0.15→0.22, shape 0.30→0.31). Falls back to per-cycle
                #    density, then per-section density, on analysis failure.
                if "drums" in activity and "drums" in results:
                    drum_stem = os.path.join(stem_dir, VOICE_STEM["drums"])
                    n_cyc = sum(int(s.get("cycles", 1) or 1) for s in scaled_sections)
                    transcribed = analyze_stem_drum_pattern(drum_stem, scaled_sections, int(total or 0), cps)
                    if transcribed and len(transcribed) == len(scaled_sections):
                        results["drums"].output["patterns"] = transcribed
                        log(f"  [data-driven] A1 DRUMS: transcribed actual hit pattern "
                            f"({len(transcribed)} sections, 16-step grid)")
                    elif (per_cycle := analyze_drum_density_per_cycle(drum_stem, int(total or 0), cps)) \
                            and len(per_cycle) >= n_cyc:
                        results["drums"].output["patterns"] = _per_cycle_drum_patterns(scaled_sections, per_cycle)
                        nz = sum(1 for d in per_cycle if d > 0)
                        log(f"  [data-driven] PER-CYCLE drum density (transcription unavailable): "
                            f"{len(per_cycle)} cycles, {nz} active")
                    else:
                        dens = [a["density"] for a in activity["drums"]]
                        results["drums"].output["patterns"] = [_drum_pattern_for_density(d) for d in dens]
                        log(f"  [data-driven] drum density per section (fallback): "
                            f"{[round(d,2) for d in dens]}")
                # 4) PITCH (A1): replace the LLM's guessed notes with the ORIGINAL stem's ACTUAL
                #    transcribed melody (per-cycle note patterns) — pitch is what makes it the SAME song.
                #    BASS: sustain=False (sparse/rhythmic, rests match). Measured win: shape 0.64→0.72,
                #      pitch 0.82→0.89, rhythm 0.19→0.29.
                #    LEAD (A1 v2): sustain=True. v1 (no sustain) collapsed the lead's shape gate
                #      (0.64→0.21) — the melodic stem is POLYPHONIC so pyin marks sustained chords as
                #      "unvoiced" and v1 emitted rests there. v2 holds the note through (rest only on
                #      true RMS-silence) + legato-collapses. Measured win vs the LLM lead: shape
                #      0.52→0.72, pitch 0.73→0.76, rhythm 0.11→0.17, timbre 0.02→0.06.
                for voice, stem, jobkey, octv, steps, sustain in (
                    ("bass", "bass.wav", "voice.bass", _VOICE_OCTAVE.get("bass", 2), 4, False),
                    ("lead", "melodic.wav", "voice.lead", _VOICE_OCTAVE.get("lead", 4), 8, True),
                ):
                    if jobkey in results:
                        pit = analyze_stem_pitch_by_section(
                            os.path.join(stem_dir, stem), scaled_sections, int(total or 0), cps,
                            octv, steps, sustain=sustain)
                        if pit and len(pit) == len(scaled_sections):
                            results[jobkey].output["patterns"] = pit
                            nonrest = sum(p.count(f"{octv}") for p in pit)
                            log(f"  [data-driven] A1 PITCH: {voice} transcribed from {stem} "
                                f"(sustain={sustain}, {len(pit)} sections, ~{nonrest} notes)")
                # 5) DRUM SOUND-MATCH (A3): TRIED + MEASURED + DISABLED (negative result). Auditioning
                #    drum banks (audition_drum_banks.py → eval/drum_bank_timbre.json) and selecting the
                #    timbre-nearest via resolve_drum_bank() did NOT improve the timbre metric and HURT
                #    shape (LinnDrum 0.27 vs TR808 0.35 on Caravan). Root cause: the original is an
                #    ACOUSTIC kit and NO sample machine in the catalog reproduces that (all warmth≈1.0,
                #    similar attack/MFCC) — consistent with T4 finding timbre NON-discriminating. The
                #    resolver/audition stay as infrastructure (may help genres whose original IS
                #    machine-based), but the bank is NOT auto-swapped here. Re-enable per-genre only
                #    with a measured win. See orchestration-infra memory.
                pres_summary = [(s.get("name", f"s{i}"), s.get("voices")) for i, s in enumerate(scaled_sections)]
                log(f"  [data-driven] presence/gain from original stems: {pres_summary}")
    except Exception as _e:  # noqa: BLE001
        log(f"  [data-driven] error: {_e} — falling back to template presence/gains")

    # Envelope-following gain automation: sample each ORIGINAL stem's loudness envelope to one value
    # per cycle and apply it as a `.gain("<...>".slow(total))` on the voice. This is what makes the
    # rendered stem rise/fall WHEN the original does — i.e. pass stem_match.py (envelope correlation),
    # the honest shape self-test. Per-section gain is coarse (≈6 steps); this is per-cycle (≈87 steps),
    # so the rendered temporal SHAPE tracks the original instead of being a uniform block.
    try:
        stem_dir = _resolve_stem_dir(args.piano, args.output_dir)
        if stem_dir and total and total > 0:
            for voice, stem in (("bass", "bass.wav"), ("lead", "melodic.wav"), ("drums", "drums.wav")):
                env = analyze_stem_envelope(os.path.join(stem_dir, stem), int(total))
                if env:
                    if voice == "bass":
                        bass_env = env
                    elif voice == "lead":
                        lead_env = env
                    else:
                        drums_env = env
            log(f"  [envelope] per-cycle gain automation from stems "
                f"(bass={len(bass_env or [])}, lead={len(lead_env or [])}, drums={len(drums_env or [])} steps)")
    except Exception as _e:  # noqa: BLE001
        log(f"  [envelope] error: {_e} — no envelope-following gain")

    code = assemble(ctx, results, bass_section_lpfs=bass_section_lpfs, lead_section_lpfs=lead_section_lpfs,
                    bass_env=bass_env, lead_env=lead_env, drums_env=drums_env)
    _, error = validate_code(code, autocorrect=False)
    if error:
        log(f"  [orchestrator] assembled code still invalid: {error}")

    # Guard the silent-render class the mini-parser CAN'T see: a method called directly on a string
    # literal — `"<…>".slow(87)`, `"…".fast(2)` — is `String.prototype.slow`, which is undefined, so
    # Strudel throws at runtime and records full-length SILENCE. (In valid Strudel a string only ever
    # sits INSIDE note()/s(); methods chain off the resulting pattern, never off the raw string.)
    _str_method = re.findall(r'"[^"]*"\s*\.\s*(slow|fast|ply|range|rev|every|off|jux)\s*\(', code)
    if _str_method:
        log(f"  [orchestrator] ⚠ method-on-string would crash Strudel → silent render: {_str_method[:3]}")
        code = re.sub(r'("[^"]*")\s*\.\s*(?:slow|fast|ply|range|rev|every|off|jux)\s*\([^)]*\)', r"\1", code)

    # Authoritative pre-render gate: validate with Strudel's REAL parser (@strudel/mini). A
    # mini-notation parse error (unbalanced "<[a] [b]", a stray "♭", etc.) renders SILENT, so we
    # must not let it through. CORRECTIVE: if the parser rejects patterns, replace the offending
    # voices with deterministic fallback patterns (always parse-safe) and re-assemble — guaranteeing
    # a non-silent render. Belt-and-suspenders on the Python balance/accidental checks upstream.
    parse_errors = _strudel_parse_errors(code)
    if parse_errors:
        bad = {e.get("pattern", "") for e in parse_errors}
        log(f"  [orchestrator] ⚠ REAL PARSER rejected {len(parse_errors)} pattern(s) — "
            f"repairing with fallbacks before render: {list(bad)[:2]}")
        for voice, jobkey, octv in (("bass", "voice.bass", 2), ("lead", "voice.lead", 4)):
            out = results.get(jobkey)
            if out and any(any(b[:20] in str(p) for b in bad) for p in out.output.get("patterns", [])):
                out.output["patterns"] = _voice_fallback_factory(voice, octv)(ctx)["patterns"]
        dr = results.get("drums")
        if dr and any(any(b[:20] in str(p) for b in bad) for p in dr.output.get("patterns", [])):
            dr.output["patterns"] = _drums_fallback(ctx)["patterns"]
        code = assemble(ctx, results, bass_section_lpfs=bass_section_lpfs, lead_section_lpfs=lead_section_lpfs,
                        bass_env=bass_env, lead_env=lead_env, drums_env=drums_env)
        parse_errors = _strudel_parse_errors(code)
        # Last resort: if STILL broken, fall back every voice (deterministic patterns always parse).
        if parse_errors:
            for voice, jobkey, octv in (("bass", "voice.bass", 2), ("lead", "voice.lead", 4)):
                if jobkey in results:
                    results[jobkey].output["patterns"] = _voice_fallback_factory(voice, octv)(ctx)["patterns"]
            results["drums"].output["patterns"] = _drums_fallback(ctx)["patterns"]
            code = assemble(ctx, results, bass_section_lpfs=bass_section_lpfs, lead_section_lpfs=lead_section_lpfs,
                            bass_env=bass_env, lead_env=lead_env, drums_env=drums_env)
            parse_errors = _strudel_parse_errors(code)
        log(f"  [orchestrator] after repair: {len(parse_errors)} parse error(s) remain")

    # Observability: job_run.json next to the output.
    out_dir = args.output_dir
    if not out_dir and args.sections_json:
        out_dir = str(Path(args.sections_json).parent)
    if out_dir:
        try:
            manifest = runner.manifest()
            manifest["assembled_valid"] = (error == "")
            manifest["real_parser_ok"] = (len(parse_errors) == 0)
            if parse_errors:
                manifest["parse_errors"] = parse_errors
            Path(out_dir, "job_run.json").write_text(json.dumps(manifest, indent=2))
            log(f"  [orchestrator] wrote job_run.json ({len(manifest['fallbacks'])} fallbacks)")
        except Exception as e:  # noqa: BLE001
            log(f"  [orchestrator] could not write job_run.json: {e}")

    # Strudel to stdout (Go captures this, same contract as ollama_codegen.py).
    print(code)


if __name__ == "__main__":
    main()
