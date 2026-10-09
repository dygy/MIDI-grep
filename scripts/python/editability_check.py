#!/usr/bin/env python3
"""Editability / replay detector for generated Strudel code (spec 003 §2.2-A).

Pure regex + structural analysis over the code TEXT — no Strudel runtime, no audio deps.
Importable from ``ai_improver.py``, ``compare_audio.py`` and the ``loop`` MCP server.

Contract (context/spec/003-editable-strudel-generation/technical-considerations.md §2.2-A):

  R1 reconstruction-by-playback  ``.slice(N, run(N)).slow(N)`` (same N, identifier or literal,
                                 simple chain calls allowed in between) and ``loopAt(`` → the
                                 enclosing voice is a REPLAY voice.
  R2 full-stem sound             ``s("originalfull")`` / ``s("<stem>full")`` / ``s("origseg<N>")``
                                 → replay voice; a ``samples(...)`` load of
                                 ``samples_orig.json`` / ``samples_segs.json`` / ``samples_stems.json``
                                 → violation on that line.
  R3 editable voice              a ``$:`` block (or ``stack(...)`` member) containing ``note(`` fed
                                 by a bar array (``cat(...name)`` with ``let name = [`` declared) or
                                 a mini-notation string, OR a bare ``s("…")`` whose tokens are
                                 one-shot names (not matching R2) → counts toward ``editable_voices``.
                                 A chained ``.s("…")`` only selects the instrument — it is not
                                 pattern data, so ``note(unknownVar).s("saw")`` is NOT editable.
  R4 texture allowance           a replay voice is tolerated only when a line of that voice carries
                                 a ``// texture`` marker AND ``len(editable_voices) >= 2``.
  R5 loop-only                   ``len(editable_voices) == 0`` → fail regardless of markers.
  R6 mode                        ``// generation_mode:`` header wins; else ``mode_hint``; else infer
                                 ``sample-instrument`` when a ``samples(`` load exists and an
                                 editable voice plays a non-builtin sample name, else ``synth``.

Ambiguities resolved with the STRICTER reading (documented here on purpose):

  * The ``// texture`` marker lives in a comment, so markers and the ``// generation_mode:``
    header are collected from the ORIGINAL text before comments are stripped for rule matching.
  * Per-bar stem-loop banks (``s("<stem>loop")`` — ``drumsloop``, ``bassloop``, … as emitted by
    ``build_sample_pack.py`` / ``generate_sample_strudel.py --mode loops``) are treated as
    R2 replay sounds. values.md §3 only allows per-bar loops as texture under editable voices,
    so they fall under the R4 allowance exactly like a full-stem loop. Without this
    ``sample_pack/output_loops.strudel`` (which the spec says must FAIL) would slip through.
  * A ``samples(...)`` load of a replay manifest is a violation on its own line (not a voice).
  * A voice that is both replay-shaped and note-shaped is classified as REPLAY.
  * A ``// generation_mode: loops`` header is itself a violation (spec §2.2-C: loops output is
    "texture/diagnostic, NOT a deliverable"); an unknown header value is also a violation.
  * When nothing editable exists but replay voices do, the inferred mode is ``loops`` (more
    informative than ``synth`` for a loop-only file). The header / ``mode_hint`` still win.
  * Empty input (after comment stripping) and unbalanced delimiters / unterminated strings are
    PARSE errors (CLI exit 2) rather than a silent fail.
  * ``bass.slice(0, 4)`` on a bar ARRAY never trips R1 — the rule requires the
    ``run(N)…slow(N)`` shape (functional-spec §2.1 Structure Test encourages array slicing).

CLI:  ``editability_check.py <file.strudel> [--json] [--mode-hint MODE]``
      exit 0 = pass, 1 = fail, 2 = usage or parse error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # single source of truth for builtin sound names (same directory)
    from strudel_validation import VALID_SOUNDS as _BUILTIN_SOUNDS
except ImportError:  # pragma: no cover - repo-root import path (MCP server)
    try:
        from scripts.python.strudel_validation import VALID_SOUNDS as _BUILTIN_SOUNDS
    except ImportError:
        _BUILTIN_SOUNDS = frozenset()

__all__ = [
    "EditabilityResult",
    "VoiceInfo",
    "Violation",
    "ParseError",
    "check_editability",
    "to_json_fields",
    "KNOWN_MODES",
]

KNOWN_MODES = ("sample-instrument", "synth", "loops")

# Strudel's default drum one-shot vocabulary (tidal-drum-machines abbreviations). Used only
# for R6 inference: these are builtin, not sample-instrument names.
DRUM_ONESHOTS = frozenset(
    "bd sd sn hh oh ch cp cr rd ride lt mt ht rim cb tb sh perc misc fx cl click clap tom "
    "crash hat kick snare hc ho hat_closed hat_open cowbell tabla bass".split()
)

# ── rule regexes ──────────────────────────────────────────────────────────────────────────
# R1: .slice(N, run(N)) … .slow(N) with the same N (identifier or literal). Simple chain calls
# (e.g. .clip(1)) are allowed between the slice and the slow — stricter than the bare spec regex.
_R1_SLICE_RUN_SLOW = re.compile(
    r"\.slice\(\s*(\w+)\s*,\s*run\(\s*\1\s*\)\s*\)(?:\s*\.\w+\([^()]*\))*\s*\.slow\(\s*\1\s*\)"
)
_R1_LOOPAT = re.compile(r"\bloopAt\s*\(")

# R2: replay sample names — full stems, original segments, per-bar stem loop banks.
_R2_REPLAY_NAME = re.compile(r"^(originalfull|[a-z]+full|origseg\d+|[a-z]+loop)$")
_R2_MANIFEST = re.compile(r"samples\(\s*[\"'`][^\"'`]*samples_(orig|segs|stems)\.json[\"'`]")

# sound calls: s("…"), .s("…"), sound("…"), .sound("…") — first string argument only
_SOUND_CALL = re.compile(r"(?<![\w$])(?:\.\s*)?(?:s|sound)\(\s*([\"'`])(.*?)\1", re.S)
# a BARE s("…")/sound("…") is pattern data (drum hits); a chained .s("…") only picks the instrument
_SOUND_PATTERN_CALL = re.compile(r"(?<![\w$.])(?:s|sound)\(\s*([\"'`])(.*?)\1", re.S)
# a bare s(...)/sound(...) whose argument is an expression (e.g. s(cat(...vox)) on a bar array)
_SOUND_PATTERN_OPEN = re.compile(r"(?<![\w$.])(?:s|sound)\(")
_NOTE_CALL = re.compile(r"(?<![\w$.])note\(")
_SPREAD_NAME = re.compile(r"\.\.\.\s*([A-Za-z_$][\w$]*)")
_LET_ARRAY = re.compile(r"^\s*(?:let|const|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\[", re.M)
_SAMPLES_LOAD = re.compile(r"(?<![\w$])samples\s*\(")
_STRING_LITERAL = re.compile(r"([\"'`])(.*?)\1", re.S)
_TOKEN = re.compile(r"[A-Za-z_][\w]*")

# comment-carried metadata (read from the ORIGINAL text, before stripping)
_TEXTURE_MARKER = re.compile(r"//\s*texture\b", re.I)
_MODE_HEADER = re.compile(r"^\s*//\s*generation_mode\s*:\s*([\w-]+)", re.M)

_VOICE_PREFIX = re.compile(r"^_?\$[\w]*\s*:")
_DECL_PREFIX = re.compile(r"^(?:let|const|var|await|import|export|function|return|if|for|while)\b")
_EXPR_VOICE_START = re.compile(r"^(?:stack|note|s|sound|n|cat|seq|arrange|silence)\s*\(")


class ParseError(ValueError):
    """The code could not be structurally analysed (empty, unbalanced, unterminated string)."""


@dataclass
class VoiceInfo:
    index: int
    label: str
    kind: str  # "editable" | "replay" | "texture" | "unclassified"
    line_start: int
    line_end: int
    sounds: list[str] = field(default_factory=list)
    arrays: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    snippet: str = ""


@dataclass
class Violation:
    rule: str
    line: int | None
    message: str
    voice: str | None = None

    def __str__(self) -> str:
        where = f" line {self.line}" if self.line is not None else ""
        who = f" [{self.voice}]" if self.voice else ""
        return f"{self.rule}{where}{who}: {self.message}"


@dataclass
class EditabilityResult:
    passed: bool
    generation_mode: str | None
    editable_voices: list[VoiceInfo] = field(default_factory=list)
    texture_voices: list[VoiceInfo] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    summary: str = ""
    # extras beyond the §2.2-A contract (structured form of `violations` + leftovers)
    violation_details: list[Violation] = field(default_factory=list)
    replay_voices: list[VoiceInfo] = field(default_factory=list)
    unclassified_voices: list[VoiceInfo] = field(default_factory=list)


def to_json_fields(res: EditabilityResult) -> dict:
    """The five keys that get merged into comparison.json / metadata.json."""
    return {
        "editability": "pass" if res.passed else "fail",
        "generation_mode": res.generation_mode,
        "editability_violations": list(res.violations),
        "editable_voice_count": len(res.editable_voices),
        "texture_voice_count": len(res.texture_voices),
    }


# ── lexical helpers ───────────────────────────────────────────────────────────────────────
def _strip_comments(code: str) -> str:
    """Blank out // and /* */ comments outside string literals, preserving every newline so
    line numbers survive. Raises ParseError on an unterminated string."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c in "\"'`":
            quote = c
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote:
                    break
                if code[j] == "\n" and quote != "`":
                    raise ParseError(f"unterminated string literal at line {code.count(chr(10), 0, i) + 1}")
                j += 1
            else:
                raise ParseError(f"unterminated string literal at line {code.count(chr(10), 0, i) + 1}")
            out.append(code[i : j + 1])
            i = j + 1
        elif code.startswith("//", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", code[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _split_statements(code: str) -> list[tuple[int, int, str]]:
    """Top-level statements as (start_offset, end_offset, text). A statement ends at a newline
    at bracket depth 0 unless the next non-blank line continues a method chain with '.'.
    Raises ParseError on unbalanced delimiters."""
    stmts: list[tuple[int, int, str]] = []
    depth = 0
    i, n = 0, len(code)
    start: int | None = None
    opens, closes = "([{", ")]}"
    while i < n:
        c = code[i]
        if c in "\"'`":  # strings were validated by _strip_comments; skip them
            j = i + 1
            while j < n and code[j] != c:
                j += 2 if code[j] == "\\" else 1
            if start is None:
                start = i
            i = j + 1
            continue
        if start is None and not c.isspace():
            start = i
        if c in opens:
            depth += 1
        elif c in closes:
            depth -= 1
            if depth < 0:
                raise ParseError(f"unbalanced '{c}' at line {_line_of(code, i)}")
        elif depth == 0 and start is not None and (c == "\n" or c == ";"):
            nxt = code[i + 1 :].lstrip()
            if c == "\n" and nxt.startswith("."):
                i += 1
                continue
            stmts.append((start, i, code[start:i]))
            start = None
        i += 1
    if depth != 0:
        raise ParseError("unbalanced brackets at end of file")
    if start is not None:
        stmts.append((start, n, code[start:n]))
    return [s for s in stmts if s[2].strip()]


def _split_top_level_args(text: str) -> list[tuple[int, str]]:
    """Split the inside of a call on top-level commas → [(offset_in_text, arg_text)]."""
    args: list[tuple[int, str]] = []
    depth = 0
    i, n = 0, len(text)
    start = 0
    while i < n:
        c = text[i]
        if c in "\"'`":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append((start, text[start:i]))
            start = i + 1
        i += 1
    args.append((start, text[start:n]))
    return [(o, a) for o, a in args if a.strip()]


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    i, n = open_idx, len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _sound_tokens(arg: str) -> list[str]:
    """Sample names referenced by a mini-notation sound string ('hh*8 <bd sd>' → hh, bd, sd)."""
    cleaned = re.sub(r":\d+", "", arg)  # drop sample-index suffixes bd:3
    return [t for t in _TOKEN.findall(cleaned)]


def _array_literal_tokens(stripped: str) -> dict[str, list[str]]:
    """``let name = ["a ~ b", "c"]`` → {name: [a, b, c]}: the mini-notation tokens inside each
    declared bar array's string literals (what a ``s(cat(...name))`` voice actually plays)."""
    out: dict[str, list[str]] = {}
    for m in _LET_ARRAY.finditer(stripped):
        open_idx = stripped.index("[", m.end() - 1)
        close = _matching_bracket(stripped, open_idx)
        body = stripped[open_idx + 1 : close] if close > 0 else stripped[open_idx + 1 :]
        toks: list[str] = []
        for lit in _STRING_LITERAL.finditer(body):
            toks.extend(_sound_tokens(lit.group(2)))
        out[m.group(1)] = toks
    return out


def _matching_bracket(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    in_str: str | None = None
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == in_str:
                in_str = None
        elif c in "\"'`":
            in_str = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# ── voice classification ──────────────────────────────────────────────────────────────────
def _classify_voice(
    idx: int,
    text: str,
    line_start: int,
    line_end: int,
    declared_arrays: set[str],
    marker_lines: set[int],
    array_tokens: dict[str, list[str]] | None = None,
) -> VoiceInfo:
    reasons: list[str] = []
    sounds: list[str] = []
    replay = False
    array_tokens = array_tokens or {}

    m = _R1_SLICE_RUN_SLOW.search(text)
    if m:
        replay = True
        reasons.append(f"R1 reconstruction-by-playback `{m.group(0)}`")
    if _R1_LOOPAT.search(text):
        replay = True
        reasons.append("R1 reconstruction-by-playback `loopAt(`")

    for sm in _SOUND_CALL.finditer(text):
        for tok in _sound_tokens(sm.group(2)):
            sounds.append(tok)
            if _R2_REPLAY_NAME.match(tok):
                replay = True
                reasons.append(f"R2 full-stem/loop sample `s(\"{tok}\")`")

    arrays = [n for n in _SPREAD_NAME.findall(text) if n in declared_arrays]
    has_note = bool(_NOTE_CALL.search(text))
    note_fed = False
    if has_note:
        nm = _NOTE_CALL.search(text)
        close = _matching_paren(text, nm.end() - 1)
        note_arg = text[nm.end() : close] if close > 0 else text[nm.end() :]
        note_fed = bool(_STRING_LITERAL.search(note_arg)) or any(
            n in declared_arrays for n in _SPREAD_NAME.findall(note_arg)
        )
    pattern_sounds = [t for pm in _SOUND_PATTERN_CALL.finditer(text) for t in _sound_tokens(pm.group(2))]
    # R3 mirror of the note(cat(...arr)) rule: a bare s(...) fed by a declared bar array of
    # one-shot names (the chops voice `s(cat(...vox))`). The array literal's tokens are the
    # pattern data, so they take the same R2 replay-name screening as a string literal would.
    sound_arrays: list[str] = []
    for om in _SOUND_PATTERN_OPEN.finditer(text):
        close = _matching_paren(text, om.end() - 1)
        s_arg = text[om.end() : close] if close > 0 else text[om.end() :]
        if _STRING_LITERAL.search(s_arg):
            continue  # literal form, handled by _SOUND_PATTERN_CALL above
        for n in _SPREAD_NAME.findall(s_arg):
            if n in declared_arrays and n not in sound_arrays:
                sound_arrays.append(n)
    for n in sound_arrays:
        for tok in array_tokens.get(n, []):
            pattern_sounds.append(tok)
            sounds.append(tok)
            if _R2_REPLAY_NAME.match(tok):
                replay = True
                reasons.append(f"R2 full-stem/loop sample `{tok}` in bar array {n}")
    has_oneshot_sound = bool(pattern_sounds) and not any(_R2_REPLAY_NAME.match(t) for t in pattern_sounds)
    has_sound_call = bool(_SOUND_PATTERN_CALL.search(text)) or bool(sound_arrays)

    label = arrays[0] if arrays else (sounds[0] if sounds else f"voice{idx}")
    snippet = " ".join(text.split())
    if len(snippet) > 90:
        snippet = snippet[:87] + "..."

    if replay:
        kind = "texture" if any(ln in marker_lines for ln in range(line_start, line_end + 1)) else "replay"
    elif note_fed:
        kind = "editable"
        reasons.append("R3 note() fed by " + ("bar array " + ",".join(arrays) if arrays else "mini-notation"))
    elif has_oneshot_sound or (has_sound_call and not pattern_sounds):
        kind = "editable"
        fed = f" fed by bar array {','.join(sound_arrays)}" if sound_arrays else ""
        reasons.append("R3 s() one-shot pattern" + fed
                       + (f" ({', '.join(dict.fromkeys(pattern_sounds))})" if pattern_sounds else " (rests only)"))
    else:
        kind = "unclassified"
        reasons.append("no note()/s() pattern data found")

    return VoiceInfo(idx, label, kind, line_start, line_end, sounds, arrays, reasons, snippet)


def _voices_from_statement(start: int, text: str, stripped: str) -> list[tuple[str, int, int]]:
    """Split a `$:` block (or bare expression statement) into voices → [(text, line_start, line_end)].
    A top-level stack(...) yields one voice per member; anything else is a single voice."""
    body = text
    body_off = start
    pm = _VOICE_PREFIX.match(body)
    if pm:
        body = body[pm.end() :]
        body_off += pm.end()
    lead_ws = len(body) - len(body.lstrip())
    body = body[lead_ws:]
    body_off += lead_ws

    if body.startswith("stack") and re.match(r"stack\s*\(", body):
        open_idx = body.index("(")
        close_idx = _matching_paren(body, open_idx)
        if close_idx > 0:
            inner = body[open_idx + 1 : close_idx]
            members = []
            for off, arg in _split_top_level_args(inner):
                a_lead = len(arg) - len(arg.lstrip())
                abs_start = body_off + open_idx + 1 + off + a_lead
                abs_end = body_off + open_idx + 1 + off + len(arg.rstrip())
                members.append((arg.strip(), _line_of(stripped, abs_start), _line_of(stripped, max(abs_start, abs_end - 1))))
            if members:
                return members
    return [(body, _line_of(stripped, body_off), _line_of(stripped, body_off + max(0, len(body.rstrip()) - 1)))]


# ── main entry point ──────────────────────────────────────────────────────────────────────
def check_editability(code: str, *, mode_hint: str | None = None) -> EditabilityResult:
    """Apply R1–R6 to Strudel source text. Raises ParseError when the text cannot be analysed."""
    if not isinstance(code, str):
        raise ParseError("code must be a string")

    marker_lines = {i for i, ln in enumerate(code.splitlines(), 1) if _TEXTURE_MARKER.search(ln)}
    header_mode = None
    hm = _MODE_HEADER.search(code)
    if hm:
        header_mode = hm.group(1).strip()

    stripped = _strip_comments(code)
    if not stripped.strip():
        raise ParseError("empty input (no code after comment stripping)")

    statements = _split_statements(stripped)
    if not statements:
        raise ParseError("no statements found")

    declared_arrays = set(_LET_ARRAY.findall(stripped))
    array_tokens = _array_literal_tokens(stripped)
    details: list[Violation] = []
    voices: list[VoiceInfo] = []
    has_samples_load = bool(_SAMPLES_LOAD.search(stripped))

    for sm in _R2_MANIFEST.finditer(stripped):
        details.append(Violation("R2", _line_of(stripped, sm.start()),
                                 f"samples() load of replay manifest samples_{sm.group(1)}.json"))

    idx = 0
    for s_off, _e_off, s_text in statements:
        head = s_text.lstrip()
        is_voice_block = bool(_VOICE_PREFIX.match(head))
        if not is_voice_block:
            if _DECL_PREFIX.match(head) or head.startswith("setcps") or head.startswith("samples("):
                continue
            if not _EXPR_VOICE_START.match(head):
                continue
        for v_text, l0, l1 in _voices_from_statement(s_off + (len(s_text) - len(head)), head, stripped):
            idx += 1
            voices.append(_classify_voice(idx, v_text, l0, l1, declared_arrays, marker_lines, array_tokens))

    editable = [v for v in voices if v.kind == "editable"]
    texture = [v for v in voices if v.kind == "texture"]
    replay = [v for v in voices if v.kind == "replay"]
    unclassified = [v for v in voices if v.kind == "unclassified"]

    for v in replay:
        for r in v.reasons:
            rule = r.split()[0]
            details.append(Violation(rule, v.line_start, r[len(rule) + 1 :] + " — replay voice without `// texture` marker", v.label))
    if len(editable) < 2:
        for v in texture:
            details.append(Violation("R4", v.line_start,
                                     f"`// texture` loop needs >= 2 editable voices (found {len(editable)}); "
                                     + "; ".join(r for r in v.reasons if r.startswith(("R1", "R2"))), v.label))
    if not editable:
        details.append(Violation("R5", None, f"loop-only output: 0 editable voices (R3) among {len(voices)} voice(s)"))

    # R6 — mode: header > hint > inference
    if header_mode is not None:
        mode: str | None = header_mode
        if header_mode not in KNOWN_MODES:
            details.append(Violation("R6", _line_of(code, hm.start()), f"unknown generation_mode header '{header_mode}'"))
    elif mode_hint:
        mode = mode_hint
    elif not editable and (replay or texture):
        mode = "loops"
    else:
        custom = [t for v in editable for t in v.sounds
                  if t not in _BUILTIN_SOUNDS and t not in DRUM_ONESHOTS]
        mode = "sample-instrument" if (has_samples_load and custom) else "synth"
    if mode == "loops":
        details.append(Violation("R6", _line_of(code, hm.start()) if hm else None,
                                 "generation_mode 'loops' is texture/diagnostic output, not a deliverable"))

    passed = not details
    violations = [str(d) for d in details]
    summary = (
        f"{'PASS' if passed else 'FAIL'} — mode={mode}; {len(editable)} editable voice(s), "
        f"{len(texture)} texture, {len(replay)} replay, {len(unclassified)} unclassified; "
        f"{len(violations)} violation(s)"
    )
    return EditabilityResult(passed, mode, editable, texture, violations, summary,
                             details, replay, unclassified)


# ── CLI ───────────────────────────────────────────────────────────────────────────────────
def _result_to_dict(res: EditabilityResult) -> dict:
    d = to_json_fields(res)
    d["summary"] = res.summary
    d["voices"] = [asdict(v) for v in (res.editable_voices + res.texture_voices + res.replay_voices + res.unclassified_voices)]
    d["violation_details"] = [asdict(v) for v in res.violation_details]
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strudel editability / replay detector (spec 003 R1-R6).")
    ap.add_argument("file", help="Strudel source file")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--mode-hint", default=None, choices=KNOWN_MODES,
                    help="generation mode to assume when the file has no // generation_mode: header")
    args = ap.parse_args(argv)

    path = Path(args.file)
    try:
        code = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 2
    try:
        res = check_editability(code, mode_hint=args.mode_hint)
    except ParseError as e:
        if args.json:
            print(json.dumps({"editability": "error", "error": str(e), "file": str(path)}))
        else:
            print(f"PARSE ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_result_to_dict(res), indent=2))
    else:
        print(res.summary)
        for v in res.editable_voices:
            print(f"  editable  L{v.line_start}: {v.label} — {'; '.join(v.reasons)}")
        for v in res.texture_voices:
            print(f"  texture   L{v.line_start}: {v.label} — {'; '.join(v.reasons)}")
        for viol in res.violations:
            print(f"  VIOLATION {viol}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
