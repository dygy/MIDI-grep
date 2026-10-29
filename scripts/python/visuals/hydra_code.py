"""The Hydra side: live controls read from the piece's bar arrays, the LLM-written chain, and its validator.

Controls are JavaScript functions evaluated every Hydra frame against Strudel's clock (``getTime()``, in cycles), so
they follow the patterns as written, not a recording of them:

  kick / snare / hat  onset pulse of that drum row: 1 at the hit, halving every grid step
  bassHit / leadHit   onset pulse of the bass / lead voice
  bassWave            held bass pitch as a wave count: 1 at the piece's lowest bass note, x2 per octave above it
  leadHue             held lead pitch class / 12 (0..1, a turn of the colour wheel per octave of pitch classes)
  chordDepth          notes sounding in the chord voice now / the most it ever stacks (0..1)
  energy              onsets per bar over every voice / the busiest bar (0..1)

A voice the piece does not have yields a control that is always 0.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

CONTROLS = {
    "kick": "kick drum onset pulse, 0..1 (1 at the hit, halves every grid step)",
    "snare": "snare/clap onset pulse, 0..1",
    "hat": "hi-hat onset pulse, 0..1",
    "bassHit": "bass note onset pulse, 0..1",
    "leadHit": "lead note onset pulse, 0..1",
    "bassWave": "held bass pitch as a wave count: 1 at the lowest bass note, doubles per octave (about 1..4)",
    "leadHue": "held lead pitch class / 12, 0..1",
    "chordDepth": "chord notes sounding now / the most the chord voice stacks, 0..1",
    "energy": "this bar's onset count over all voices / the busiest bar, 0..1",
}

# Hydra functions a chain may call. Left out on purpose: kaleid, pixelate, posterize, voronoi, shape, repeat* —
# they turn the photo into an abstract pattern, and the scene must stay recognizable.
HYDRA_FUNCS = {
    "src", "osc", "noise", "gradient", "solid",
    "modulate", "modulateScale", "modulateRotate", "modulateScrollX", "modulateScrollY",
    "scale", "rotate", "scrollX", "scrollY",
    "hue", "saturate", "contrast", "brightness", "luma", "colorama", "color", "invert", "thresh",
    "blend", "add", "mult", "diff", "layer", "mask", "out",
}
JS_NAMES = {"Math", "PI", "sin", "cos", "abs", "min", "max", "pow", "sqrt", "s0", "o0", "time"}
ALLOWED = HYDRA_FUNCS | JS_NAMES | set(CONTROLS)
REQUIRED = ("kick", "bassWave")
SIGNATURES = "\n".join("    " + x for x in (
    "osc(frequency, sync, offset)  noise(scale, offset)  gradient(speed)  solid(r, g, b, a)  src(s0 | o0)",
    ".modulate(SOURCE, amount)  .modulateScale(SOURCE, multiple, offset)  .modulateRotate(SOURCE, multiple, offset)",
    ".modulateScrollX(SOURCE, scrollX, speed)  .modulateScrollY(SOURCE, scrollY, speed)",
    ".scale(amount)  .rotate(angle, speed)  .scrollX(scrollX, speed)  .scrollY(scrollY, speed)",
    ".hue(amount)  .saturate(amount)  .contrast(amount)  .brightness(amount)  .luma(threshold, tolerance)",
    ".colorama(amount)  .color(r, g, b)  .invert(amount)  .thresh(threshold, tolerance)",
    ".blend(SOURCE, amount)  .add(SOURCE, amount)  .mult(SOURCE, amount)  .diff(SOURCE)  .layer(SOURCE)  .mask(SOURCE)",
    ".out()",
))  # the image must move with the kick and wave with the bass

_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_FORBIDDEN = re.compile(r"[;{}\[\]`'\"\\]|\bawait\b|\bnew\b|\bfunction\b|\bthis\b")


def validate_chain(chain: str) -> list[str]:
    """Problems with an LLM-written Hydra chain; empty = acceptable to embed in the piece."""
    c = " ".join(chain.split())
    errs = []
    if not c.startswith("src(s0)"):
        errs.append("must start with src(s0) (the scene image is the base layer)")
    if not re.search(r"\.out\((o0)?\)$", c):
        errs.append("must end with .out()")
    if _FORBIDDEN.search(c):
        errs.append("only one expression: no ; {} [] quotes, await, new, function or this")
    depth = 0
    for ch in c:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            break
    if depth != 0:
        errs.append("unbalanced parentheses")
    unknown = sorted({w for w in _IDENT.findall(re.sub(r"(?<![\w$.])\d*\.?\d+(e-?\d+)?", " ", c)) if w not in ALLOWED})
    if unknown:
        errs.append(f"unknown names: {', '.join(unknown)} (allowed: Hydra {sorted(HYDRA_FUNCS)}, controls {sorted(CONTROLS)}, Math)")
    for name in REQUIRED:
        if not re.search(rf"\b{name}\b", c):
            errs.append(f"must use the {name} control")
    if c.count(".out(") != 1:
        errs.append("exactly one .out()")
    return errs


def fallback_chain() -> str:
    """The chain used when the LLM is off or its answer fails validation: the same wiring the prompt asks for."""
    return ("src(s0)"
            ".modulate(osc(() => 2 * bassWave(), 0.05, 0).rotate(Math.PI / 2), () => 0.003 + 0.02 * kick() + 0.01 * bassHit())"
            ".scale(() => 1 + 0.03 * kick())"
            ".modulateScale(noise(4, 0.3), () => 0.03 * snare())"
            ".hue(() => 0.08 * leadHue())"
            ".saturate(() => 1 + 0.4 * chordDepth())"
            ".brightness(() => 0.05 * hat())"
            ".blend(src(o0).scale(1.005), () => 0.2 + 0.3 * energy())"
            ".out()")


def fallback_image_prompt(vision: dict) -> str:
    clap = vision.get("clap") or {}
    scene = (clap.get("scenes") or [{"text": "a city street at night"}])[0]["text"]
    mood = (clap.get("moods") or [{"text": vision.get("mode") or "cinematic"}])[0]["text"]
    genre = (vision.get("genre") or "").replace("_", " ")
    return f"{scene}, {mood} atmosphere, {genre} music video still, photorealistic, 35mm film photo, natural light"


def build_prompt(vision: dict) -> str:
    controls = "\n".join(f"- {k}(): {v}" for k, v in CONTROLS.items())
    return f"""You design the visuals for a piece of music: a REAL-LOOKING photo of a scene, animated by Hydra (hydra.ojack.xyz)
so that it moves with the music's voices.

Measured track vision (JSON):
{json.dumps(vision, ensure_ascii=False, indent=1)}

"clap" ranks scene and mood descriptions by how well the audio matches them (higher score = closer).
"centroid_hz" is the mix brightness (low = dark/warm, high = bright).

Return ONE JSON object with keys:
- "image_prompt": a text-to-image prompt (max 60 words) for a photorealistic scene that fits the vision: a real place,
  real lighting, the colours the key, mood and brightness suggest. No text, no logos, no people's faces in close-up.
- "negative_prompt": what to avoid (cartoon, illustration, text, watermark ...).
- "chain": ONE Hydra expression over the photo. Rules:
  * starts with src(s0) and ends with .out()
  * allowed functions: {", ".join(sorted(HYDRA_FUNCS))}; Math.sin/cos/abs/min/max/pow/PI
  * signatures (every number may be an arrow function returning a number; a SOURCE is src(s0), src(o0), osc(...),
    noise(...), gradient(...) or solid(...)):
{SIGNATURES}
  * example of the shape wanted (do not copy the numbers):
    {fallback_chain()}
  * dynamic arguments are arrow functions using these controls (each returns a number):
{controls}
  * must use kick and bassWave; the photo must WAVE with the bass: modulate it with osc(() => ... bassWave() ...)
    rotated or not, and a small amount
  * keep the scene recognizable: modulate amounts <= 0.05, scale within 1 +- 0.08, hue shifts <= 0.15,
    feedback via blend(src(o0)...) amount <= 0.5
  * no strings, no semicolons, no variables
- "rationale": one sentence on why this scene and this motion fit the track.
"""


def ask_ollama(vision: dict, model: str, seed: int, url: str | None = None) -> dict:
    import requests
    url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    r = requests.post(f"{url}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(vision)}],
        "format": "json",
        "stream": False,
        "keep_alive": "30s",  # free the RAM for the diffusion model that runs next
        "options": {"temperature": 0.4, "seed": seed, "num_predict": 1024, "num_ctx": 8192},
    }, timeout=600)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# ── reading the piece ─────────────────────────────────────────────────────────

_LET_ARRAY = re.compile(r"^let (\w+) = \[", re.M)
_DRUM_ROW = re.compile(r"^\s*cat\(s\(\"")
_S_STR = re.compile(r's\("([^"]*)"\)')
DRUM_CONTROL = {"bd": "kick", "sd": "snare", "cp": "snare", "rim": "snare", "hh": "hat", "oh": "hat"}


def array_names(code: str) -> set[str]:
    return set(_LET_ARRAY.findall(code))


def drum_rows(code: str) -> dict[str, list[str]]:
    """The drum stack's rows (``cat(s("..."), ...)``, one per drum sound), keyed by the sound most of its hits play."""
    rows: dict[str, list[str]] = {}
    for line in code.splitlines():
        if not _DRUM_ROW.match(line):
            continue
        bars = _S_STR.findall(line)
        sounds = Counter(re.split(r"[:@*!]", tok)[0] for bar in bars for tok in bar.split() if tok != "~")
        if not sounds:
            continue
        name, k = sounds.most_common(1)[0][0], 2
        key = name
        while key in rows:
            key, k = f"{name}{k}", k + 1
        rows[key] = bars
    return rows


def grid_steps(code: str, rows: dict[str, list[str]]) -> int:
    """Steps per bar of the piece's grid: the most common step count of the drum bars (16 for the generator's 16ths)."""
    counts = Counter(len(b.split()) for bars in rows.values() for b in bars if "@" not in b)
    return counts.most_common(1)[0][0] if counts else 16


def js_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def controls_js(code: str) -> tuple[str, dict]:
    """The Hydra block's control definitions for this piece, and what each control is wired to."""
    names = array_names(code)
    rows = drum_rows(code)
    steps = grid_steps(code, rows)

    def voice(base: str) -> str | None:
        parts = [f"note(cat(...{n}))" for n in (base, base + "Legato") if n in names]
        return None if not parts else parts[0] if len(parts) == 1 else f"stack({', '.join(parts)})"

    bass, lead, chords = voice("bass"), voice("lead"), voice("chords")
    lines = [f"const visGrid = 1 / {steps}  // one grid step in cycles: pulses halve every step"]
    if rows:
        lines.append("// the drum stack's rows, copied: a change to a drum row above must be made here too to move the picture")
        lines.append("let visDrums = {")
        for k, bars in rows.items():
            lines.append(f"  {k}: [{', '.join(js_str(b) for b in bars)}],")
        lines.append("}")
    lines += [
        "const visMidi = v => typeof v.note === 'number' ? v.note : noteToMidi(v.note)",
        "const visPulse = pat => () => {",
        "  const t = getTime(), hs = pat.queryArc(t - 1, t).filter(h => h.hasOnset() && h.whole.begin <= t)",
        "  if (!hs.length) return 0",
        "  return Math.pow(0.5, (t - Math.max(...hs.map(h => h.whole.begin.valueOf()))) / visGrid)",
        "}",
        "const visHeld = (pat, f) => { let last = 0; return () => {",
        "  const t = getTime(), hs = pat.queryArc(t, t + 1e-4)",
        "  if (hs.length) last = f(hs)",
        "  return last",
        "} }",
        "const visNone = () => 0",
    ]
    wiring = {}
    drum_of = {}
    for k in rows:
        ctl = DRUM_CONTROL.get(re.sub(r"\d+$", "", k))
        if ctl and ctl not in drum_of:
            drum_of[ctl] = k
    for ctl in ("kick", "snare", "hat"):
        if ctl in drum_of:
            lines.append(f"const {ctl} = visPulse(cat(...visDrums.{drum_of[ctl]}.map(b => s(b))))")
            wiring[ctl] = f"drum row {drum_of[ctl]}"
        else:
            lines.append(f"const {ctl} = visNone")
            wiring[ctl] = None
    if bass:
        lines += [f"const visBass = {bass}",
                  "const visBassLow = Math.min(...visBass.queryArc(0, bass.length).map(h => visMidi(h.value)))",
                  "const bassHit = visPulse(visBass)",
                  "const bassWave = visHeld(visBass, hs => Math.pow(2, (visMidi(hs[0].value) - visBassLow) / 12))"]
        wiring.update(bassHit=bass, bassWave=bass)
    else:
        lines += ["const bassHit = visNone", "const bassWave = () => 1"]
        wiring.update(bassHit=None, bassWave=None)
    if lead:
        lines += [f"const visLead = {lead}",
                  "const leadHit = visPulse(visLead)",
                  "const leadHue = visHeld(visLead, hs => (visMidi(hs[0].value) % 12) / 12)"]
        wiring.update(leadHit=lead, leadHue=lead)
    else:
        lines += ["const leadHit = visNone", "const leadHue = visNone"]
        wiring.update(leadHit=None, leadHue=None)
    if chords:
        lines += [f"const visChords = {chords}",
                  "const visChordMax = Math.max(1, ...Array.from({ length: chords.length }, (_, i) => {",
                  "  const at = {}",
                  "  for (const h of visChords.queryArc(i, i + 1)) if (h.hasOnset()) at[h.whole.begin] = (at[h.whole.begin] || 0) + 1",
                  "  return Math.max(0, ...Object.values(at))",
                  "}))",
                  "const chordDepth = () => visChords.queryArc(getTime(), getTime() + 1e-4).length / visChordMax"]
        wiring["chordDepth"] = chords
    else:
        lines.append("const chordDepth = visNone")
        wiring["chordDepth"] = None
    voices = [v for v in (bass, lead, chords) if v] + ([f"cat(...visDrums.{k}.map(b => s(b)))" for k in rows])
    bars_of = "bass" if "bass" in names else ("lead" if "lead" in names else None)
    if voices and bars_of:
        lines += [f"const visAll = stack({', '.join(voices)})",
                  f"const visDensity = Array.from({{ length: {bars_of}.length }}, (_, i) => visAll.queryArc(i, i + 1).filter(h => h.hasOnset()).length)",
                  "const visDensityMax = Math.max(1, ...visDensity)",
                  "const energy = () => visDensity[Math.floor(getTime()) % visDensity.length] / visDensityMax"]
        wiring["energy"] = "onsets per bar over every voice"
    else:
        lines.append("const energy = visNone")
        wiring["energy"] = None
    return "\n".join(lines), wiring
