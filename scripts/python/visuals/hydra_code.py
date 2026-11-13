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
import math
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
# they turn the photo into an abstract pattern, and the scene must stay recognizable; and scroll* / modulateScroll*,
# whose GLSL ends in `return fract(_st)`: they wrap the frame themselves, past any edge margin (Enter v011, 2026-11-02:
# modulateScrollX by up to 0.05 put a hard seam at x = 1 - shift, a column jump 3.5-4x the median, in the live REPL).
HYDRA_FUNCS = {
    "src", "osc", "noise", "gradient", "solid",
    "modulate", "modulateScale", "modulateRotate",
    "scale", "rotate",
    "hue", "saturate", "contrast", "brightness", "luma", "colorama", "color", "invert", "thresh",
    "blend", "add", "mult", "diff", "layer", "mask", "out",
}
_MATH_NAMES = {"PI", "sin", "cos", "abs", "min", "max", "pow", "sqrt"}  # allowed only as Math.<name>
JS_NAMES = {"Math", "s0", "o0", "time"} | _MATH_NAMES
ALLOWED = HYDRA_FUNCS | JS_NAMES | set(CONTROLS)
REQUIRED = ("kick", "bassWave")
MAX_WARP = 0.05  # the largest modulate offset a chain may use (told to the LLM; the fallback stays under it)
# Hydra's src() samples fract(uv): a coordinate warped past an edge shows the OPPOSITE edge (a stripe of the image's
# top along the bottom, seen in the live REPL 2026-10-30). Zooming the scene in by 1 + 2*MAX_WARP before any warp keeps
# every offset within the cap inside the image.
EDGE_ZOOM = round(1 + 2 * MAX_WARP, 4)
# the same margin bounds a rotation of the photo itself: a turn by a (rad) about the centre moves the frame's corners,
# hypot(0.5, 0.5) from it in uv, by ~0.707 * a, so a <= MAX_WARP / 0.707 keeps them inside it (~4 degrees). The LLM
# turned the photo by pi/4 and pi/2 (VAGABUNDO v073, Enter v010/v011; 2026-10-30..11-02): scene corr ~0
MAX_PHOTO_TURN = round(MAX_WARP / math.hypot(0.5, 0.5), 3)


# Warps on the photo add up past the margin each one fits: Enter v011 (2026-11-02) turned it by 0.071 rad (corners
# 0.05) and modulated it by 0.05, and the live REPL showed a sliver of sky in the bottom-right corner (1.8 % of the
# bottom rows in one check frame). The scene is read through `scene`: src with the coordinate MIRRORED at the edges,
# so past an edge shows that same edge, never the opposite one (0 % in every frame; scene corr 0.720 -> 0.724).
SCENE_SOURCE = ("setFunction({ name: 'scene', type: 'src', inputs: [{ type: 'sampler2D', name: 'tex', default: NaN }], "
                "glsl: 'return texture2D(tex, 1.0 - abs(mod(_st, 2.0) - 1.0));' })")


def with_edge_margin(chain: str) -> str:
    """The chain with the scene read through `scene` (mirrored edges) and zoomed by EDGE_ZOOM at its source (the first
    src(s0))."""
    return chain.replace("src(s0)", f"scene(s0).scale({EDGE_ZOOM})", 1)
SIGNATURES = "\n".join("    " + x for x in (
    "osc(frequency, speed, offset)  noise(scale, offset)  gradient(speed)  solid(r, g, b, a)  src(s0 | o0)",
    ".modulate(SOURCE, amount)  .modulateScale(SOURCE, multiple, offset)  .modulateRotate(SOURCE, multiple, offset)",
    ".scale(amount)  .rotate(angle, speed)",
    # the value that leaves the photo unchanged, from hydra-synth's GLSL: mistral-small read contrast/saturate as
    # amounts around 0 (contrast(() => 0.3 * snare()) = flat grey between hits; 9 of 10 chains failed, 2026-11-02)
    ".hue(shift: 0 = unchanged)  .saturate(factor: 1 = unchanged, 0 = grey)  .contrast(factor: 1 = unchanged, 0 = flat grey)",
    ".brightness(offset: 0 = unchanged)  .color(r, g, b: 1, 1, 1 = unchanged)  .invert(0 = unchanged, 1 = negative)",
    ".colorama(amount)  .luma(threshold, tolerance)  .thresh(threshold, tolerance)",
    ".blend(SOURCE, amount)  .add(SOURCE, amount)  .mult(SOURCE, amount)  .diff(SOURCE)  .layer(SOURCE)  .mask(SOURCE)",
    ".out()",
))  # the image must move with the kick and wave with the bass

_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_FORBIDDEN = re.compile(r"[;{}\[\]`'\"\\]|\bawait\b|\bnew\b|\bfunction\b|\bthis\b")


def validate_chain(chain: str, bpm: float | None = None) -> list[str]:
    """Problems with an LLM-written Hydra chain; empty = acceptable to embed in the piece. ``bpm`` bounds the speeds
    (one pattern period per bar); without it only their constancy is checked."""
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
    if re.search(r"\b(true|false)\b", c):
        # mistral-small read osc(frequency, sync, offset) as a boolean sync (4 of 10 answers, 2026-11-01)
        errs.append("true/false are not Hydra arguments: osc(frequency, speed, offset) takes three numbers")
    bare = sorted({n for n in _MATH_NAMES if re.search(rf"(?<!Math\.)\b{n}\b", c)})
    if bare:
        # VAGABUNDO v073 (2026-11-02): rotate(... * PI) - no error, but PI is not pi in the REPL: scene corr -0.065,
        # 0.788 with Math.PI
        errs.append(f"write Math.{bare[0]}, not a bare {bare[0]} ({', '.join(bare)}): the bare names are not Math's in the REPL")
    scrolls = sorted(set(re.findall(r"\b((?:modulateS|s)croll[XY]?)\(", c)))
    if scrolls:
        errs.append(f"{', '.join(scrolls)}: Hydra wraps a scrolled frame around (its right edge shows the left one); "
                    f"move the photo with .modulate(SOURCE, amount) instead")
    unknown = sorted({w for w in _IDENT.findall(re.sub(r"(?<![\w$.])\d*\.?\d+(e-?\d+)?", " ", c)) if w not in ALLOWED and w not in scrolls})
    if unknown:
        errs.append(f"unknown names: {', '.join(unknown)} (allowed: Hydra {sorted(HYDRA_FUNCS)}, controls {sorted(CONTROLS)}, Math)")
    for name in REQUIRED:
        if not re.search(rf"\b{name}\b", c):
            errs.append(f"must use the {name} control")
    if c.count(".out(") != 1:
        errs.append("exactly one .out()")
    bare = sorted({m.group(1) for m in re.finditer(rf"\b({'|'.join(CONTROLS)})\(\)", c) if not _in_arrow(c, m.start())})
    if bare:
        # VAGABUNDO v073 (2026-11-01): modulate(osc(...), bassHit()) - read once when the block runs, never moves
        errs.append(f"controls outside an arrow function (read once, never move): {', '.join(bare)}; write () => ... {bare[0]}() ...")
    misplaced = sorted({name for name, args in _photo_steps(c) if name in _NUMERIC_STEPS
                        and any(_SOURCE_CALL.match(a.strip()) for a in args)})
    if misplaced:
        # Dumb Blond Bitches v020, Enter v010 (2026-11-01): .scrollX(osc(...), 0) - the shader does not compile, black frame
        errs.append(f"a SOURCE where a number is expected ({', '.join(misplaced)}): those arguments take numbers or "
                    f"() => ... functions; warp with .modulate*(SOURCE, amount) instead")
    drifting = sorted({name for name, args in _photo_steps(c) if name in _SPEED_ARG and len(args) > _SPEED_ARG[name]
                       and args[_SPEED_ARG[name]].strip() not in ("0", "0.0")})
    if drifting:
        # VAGABUNDO v073 (2026-11-01): .scrollY(..., () => energy() * 0.02) on the photo; the offset accumulates with
        # time, so the scene slid away section by section (corr 0.89 -> 0.19; 0.86 with the speed at 0)
        errs.append(f"a speed argument on the photo itself ({', '.join(drifting)}) accumulates and slides the scene away: "
                    f"set it to 0 or move the motion into a SOURCE")
    errs += _speed_errors(c, bpm)
    return errs


# Hydra multiplies these arguments by `time`, the seconds since Hydra started (glsl-functions.js: osc
# sin((x + time*sync) * frequency), noise(_st*scale, offset*time), gradient sin(time*speed), rotate angle + speed*time).
# A control there moves the phase by time * (its change) on every hit: 9 of the 10 chains of 2026-11-02 drove osc's
# speed or frequency with kick/bassWave, so the waves jumped further the longer the piece played - at minute 2 a kick
# shifts osc(bassWave(), kick()) by ~120 * bassWave rad, dozens of wave periods in one frame. index -> default
_TIME_ARGS = {"osc": ((1, 0.1),), "noise": ((1, 0.1),), "gradient": ((0, 0.0),), "rotate": ((1, 0.0),)}
_OSC_FREQ_DEFAULT = 60.0
# how far one unit of a time argument moves the pattern per second, in wave periods: osc and gradient are sines of
# speed*time (2 pi per period); noise's offset*time walks through value noise whose features are ~1 unit apart; a turn
# (2 pi) of rotate brings stripes back twice, but a full turn per bar is the bound kept there too
_PERIOD = {"osc": 2 * math.pi, "noise": 1.0, "gradient": 2 * math.pi, "rotate": 2 * math.pi}


def bar_seconds(bpm: float) -> float:
    """One 4/4 bar of the piece (the pipeline's bars*240/bpm)."""
    return 240.0 / bpm


def max_speed(fn: str, bpm: float, frequency: float = 1.0) -> float:
    """The largest speed argument that moves ``fn``'s pattern by at most one period per bar of the piece: the visuals
    keep the music's pace, and the hits (pulses on the amounts) carry the faster motion."""
    return _PERIOD[fn] / (bar_seconds(bpm) * (frequency if fn == "osc" else 1.0))


def _const(expr: str) -> float | None:
    """A numeric argument's value when it is a constant (digits, + - * / ( ), Math.PI); None when it moves."""
    e = expr.strip().replace("Math.PI", repr(math.pi))
    if not e or not re.fullmatch(r"[\d.eE+\-*/() ]+", e):
        return None
    try:
        return float(eval(e, {"__builtins__": {}}, {}))  # only the characters above reach eval
    except Exception:
        return None


def _speed_errors(chain: str, bpm: float | None) -> list[str]:
    errs = []
    for fn, args in _calls(chain, set(_TIME_ARGS)):
        for i, default in _TIME_ARGS[fn]:
            raw = args[i] if len(args) > i else repr(default)
            speed = _const(raw)
            if speed is None:
                errs.append(f"{fn}'s speed argument ({raw.strip()}) must be a plain number: Hydra multiplies it by the "
                            f"seconds since start, so a control there makes the {fn} jump further the longer the piece "
                            f"plays; drive the AMOUNT of the step that uses it instead")
                continue
            freq = 1.0
            if fn == "osc":
                f = _const(args[0]) if args else _OSC_FREQ_DEFAULT
                if f is None:
                    if speed != 0:
                        errs.append(f"osc's frequency ({args[0].strip()}) moves while its speed is {speed:g}: frequency "
                                    f"also multiplies time; give the osc speed 0 to let a control drive its frequency")
                    continue
                freq = abs(f)
            if bpm and abs(speed) > max_speed(fn, bpm, freq) + 1e-9:
                errs.append(f"{fn} speed {speed:g} moves its pattern {abs(speed) / max_speed(fn, bpm, freq):.1f} periods "
                            f"per bar; at {bpm:g} BPM keep it <= {max_speed(fn, bpm, freq):.3g}"
                            + (f" for frequency {freq:g}" if fn == "osc" else ""))
    return errs


def _calls(chain: str, names: set[str]) -> list[tuple[str, list[str]]]:
    """Every call of one of ``names`` anywhere in the chain (sources and steps) with its top-level arguments."""
    out = []
    for m in re.finditer(rf"(?<![\w$])({'|'.join(sorted(names))})\(", chain):
        out.append((m.group(1), _args_at(chain, m.end())[0]))
    return out


def _args_at(chain: str, start: int) -> tuple[list[str], int]:
    """The top-level arguments of the call whose '(' ends at ``start``, and the index after its ')'."""
    depth, j, args, s = 1, start, [], start
    while j < len(chain) and depth:
        ch = chain[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if (ch == "," and depth == 1) or depth == 0:
            args.append(chain[s:j])
            s = j + 1
        j += 1
    return [a for a in args if a.strip()], j


# steps whose arguments are all numbers (the SOURCE-taking ones are modulate*, blend, add, mult, diff, layer, mask)
_NUMERIC_STEPS = {"scale", "rotate", "hue", "saturate", "contrast", "brightness", "luma",
                  "colorama", "color", "invert", "thresh"}
_SOURCE_CALL = re.compile(r"(src|osc|noise|gradient|solid)\(")
# the speed argument's index: a non-zero speed advances the offset/angle every frame, without bound
_SPEED_ARG = {"rotate": 1}


def _photo_steps(chain: str) -> list[tuple[str, list[str]]]:
    """The methods called on the chain itself (outside every argument list) with their top-level arguments."""
    steps, i = [], 0
    while (m := re.compile(r"\.([A-Za-z_]\w*)\(").search(chain, i)):
        args, i = _args_at(chain, m.end())
        steps.append((m.group(1), args))
    return steps


def _in_arrow(chain: str, pos: int) -> bool:
    """Whether ``pos`` lies in the body of an arrow function: an arrow's body runs to the end of its argument, so the
    argument holding ``pos``, or an argument enclosing it, has '=>' between its start and the nested call."""
    depth, end, skipping = 0, pos, False
    for i in range(pos - 1, -1, -1):
        ch = chain[i]
        if ch == ")":
            depth += 1
        elif ch == "(" and depth > 0:
            depth -= 1
        elif depth == 0 and ch == ",":
            if not skipping and "=>" in chain[i + 1:end]:
                return True
            skipping = True  # earlier sibling arguments do not cover pos
        elif depth == 0 and ch == "(":
            if not skipping and "=>" in chain[i + 1:end]:
                return True
            end, skipping = i, False  # step out: the enclosing argument runs up to this call
    return False


# music words the diffusion model cannot paint: the LLM wrote "The key of C# minor suggests deep blues ..." into the
# scene prompt (Regime CLT, Enter, Automotivo XM; 2026-10-30)
_MUSIC_TERMS = re.compile(r"\b[A-G][#b♯♭]?\s*(major|minor)\b|\bkey\b|\bbpm\b|\btempo\b", re.I)


def scene_prompt(text: str) -> str:
    """The LLM's image prompt without the sentences that talk about the music instead of the picture ('' if none left).

    A one-sentence prompt keeps its clauses that describe the picture ("..., real lighting, colors in G minor key.")."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    kept = [x for x in sentences if not _MUSIC_TERMS.search(x)]
    if not kept and len(sentences) == 1:
        kept = [", ".join(c.strip() for c in sentences[0].rstrip(".").split(",") if c.strip() and not _MUSIC_TERMS.search(c))]
    return " ".join(kept).strip()


def fallback_chain() -> str:
    """The chain used when the LLM is off or its answer fails validation: the same wiring the prompt asks for."""
    return ("src(s0)"
            # speed 0: the frequency follows the bass, and a moving frequency with a speed would jump (validate_chain)
            ".modulate(osc(() => 2 * bassWave(), 0, 0).rotate(Math.PI / 2), () => 0.003 + 0.02 * kick() + 0.01 * bassHit())"
            ".scale(() => 1 + 0.03 * kick())"
            ".modulateScale(noise(4, 0.3), () => 0.03 * snare())"
            ".hue(() => 0.08 * leadHue())"
            ".saturate(() => 1 + 0.4 * chordDepth())"
            ".brightness(() => 0.05 * hat())"
            ".blend(src(o0).scale(1.005), () => 0.2 + 0.3 * energy())"
            ".out()")


def copies_example(chain: str) -> bool:
    """The chain is the prompt's example (the fallback chain) verbatim, whitespace aside: the LLM designed nothing."""
    return "".join(chain.split()) == "".join(fallback_chain().split())


def fallback_image_prompt(vision: dict) -> str:
    clap = vision.get("clap") or {}
    scene = (clap.get("scenes") or [{"text": "a city street at night"}])[0]["text"]
    mood = (clap.get("moods") or [{"text": vision.get("mode") or "cinematic"}])[0]["text"]
    genre = (vision.get("genre") or "").replace("_", " ")
    return f"{scene}, {mood} atmosphere, {genre} music video still, photorealistic, 35mm film photo, natural light"


def build_prompt(vision: dict, wiring: dict) -> str:
    """The LLM prompt. ``wiring`` (controls_js) says which controls the piece drives: no example chain is shown, since
    mistral-small returned the one it was given verbatim on 2 of 5 tracks (2026-11-01)."""
    controls = "\n".join(f"- {k}(): {v}" for k, v in CONTROLS.items())
    bpm = float(vision["bpm"])
    live = "\n".join(f"    - {k}: {'live, from ' + v if v else 'always 0 (no such voice)'}" for k, v in wiring.items())
    return f"""You design the visuals for a piece of music: a REAL-LOOKING photo of a scene, animated by Hydra (hydra.ojack.xyz)
so that it moves with the music's voices.

Measured track vision (JSON):
{json.dumps(vision, ensure_ascii=False, indent=1)}

"clap" ranks scene and mood descriptions by how well the audio matches them (higher score = closer).
"centroid_hz" is the mix brightness (low = dark/warm, high = bright).

Return ONE JSON object with keys:
- "image_prompt": a text-to-image prompt (max 60 words) for a photorealistic scene that fits the vision: a real place,
  real lighting, the colours the key, mood and brightness suggest. Describe only what a camera would see: name the
  colours themselves, never the key, tempo or any other music term (the image model cannot paint "C# minor").
  No text, no logos, no people's faces in close-up.
- "negative_prompt": what to avoid (cartoon, illustration, text, watermark ...).
- "chain": ONE Hydra expression over the photo. Rules:
  * starts with src(s0) and ends with .out()
  * allowed functions: {", ".join(sorted(HYDRA_FUNCS))}; Math.sin/cos/abs/min/max/pow/PI
  * signatures (every number may be an arrow function returning a number; a SOURCE is src(s0), src(o0), osc(...),
    noise(...), gradient(...) or solid(...)):
{SIGNATURES}
  * shape: src(s0), then a warp of the photo, then colour/scale moves, optionally feedback, then .out() - e.g.
    src(s0).modulate(<osc or noise SOURCE driven by bassWave>, <small amount driven by a pulse>)
           .<colour or scale step driven by another voice>( ... ) ... .out()
    (the numbers, the SOURCEs, which steps and their order are yours to choose)
  * dynamic arguments are arrow functions using these controls (each returns a number):
{controls}
    write every control inside an arrow function, e.g. .hue(() => 0.1 * leadHue()); a bare leadHue() is read once
    and never moves. Arguments are numbers or such arrow functions only (no true/false, no strings); a SOURCE goes
    only where the signature says SOURCE (.scale(osc(...)) does not compile: the frame goes black).
  * in THIS piece:
{live}
    build the motion from the live ones; a control that is always 0 here moves nothing
  * speeds are plain numbers, never controls: osc's speed, noise's offset, gradient's speed and rotate's speed are
    multiplied by the seconds since start, so a control there makes the pattern jump further the longer the piece
    plays. Keep the pace of the music, at most one wave period per bar ({bar_seconds(bpm):.2f} s here): osc
    frequency x speed <= {max_speed("osc", bpm):.2f}, noise offset <= {max_speed("noise", bpm):.2f}, rotate/gradient
    speed <= {max_speed("rotate", bpm):.2f}. The hits move the picture through the AMOUNTS, not the speeds
  * must use kick and bassWave; the photo must WAVE with the bass: modulate it with a SOURCE whose frequency follows
    bassWave() - give that osc speed 0, a moving frequency also multiplies time - or whose amount follows it, and a
    small amount; a .rotate on the photo itself turns it by at most {MAX_PHOTO_TURN} rad
    (about 4 degrees: more shows the image's edges); larger turns belong inside a SOURCE
  * the speed argument of .rotate on the photo must be 0 (a speed accumulates every frame and turns the scene away);
    put continuous motion inside a SOURCE instead
  * every step must answer the vision: which voice drives it, and why its colour/speed fits the mood and brightness
  * a colour step moves AROUND its unchanged value (see the signatures): .contrast(() => 1 + 0.2 * snare()), not
    .contrast(() => 0.2 * snare())
  * keep the scene recognizable: modulate amounts <= {MAX_WARP}, scale within 1 +- 0.08, hue shifts <= 0.15,
    feedback via blend(src(o0)...) amount <= 0.5
  * no strings, no semicolons, no variables
- "rationale": one sentence on why this scene and this motion fit the track.
"""


def ask_ollama(vision: dict, wiring: dict, model: str, seed: int, url: str | None = None,
               repair: tuple[dict, list[str]] | None = None) -> dict:
    """The LLM's answer; ``repair`` = (its previous answer, the errors: the validator's or the check's) asks it to fix
    that chain."""
    url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    messages = [{"role": "user", "content": build_prompt(vision, wiring)}]
    if repair:
        prev, errs = repair
        messages += [{"role": "assistant", "content": json.dumps(prev, ensure_ascii=False)},
                     {"role": "user", "content": "The chain was rejected:\n" + "\n".join(f"- {e}" for e in errs)
                      + "\nReturn the same JSON object with a corrected chain that keeps your design."}]
    try:
        return _chat(url, model, messages, seed)
    except json.JSONDecodeError as exc:
        # Automotivo XM v018 (2026-11-01): "Expecting ',' delimiter" - the error goes back once, like a rejected chain
        return _chat(url, model, messages + [{"role": "assistant", "content": exc.doc},
                                             {"role": "user", "content": f"That is not valid JSON ({exc.msg} at char "
                                                                         f"{exc.pos}). Return the JSON object only."}], seed)


def _chat(url: str, model: str, messages: list[dict], seed: int) -> dict:
    import requests
    r = requests.post(f"{url}/api/chat", json={
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
        # unload as soon as it answers: the diffusion model loads seconds later, and a 30 s keep-alive held mistral-small
        # (~14 GB) resident beside it, which on a 24 GB machine filled swap and got the matrix run killed (2026-10-30)
        "keep_alive": 0,
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
        "  const t = getTime(), hs = pat.queryArc(t, t + 0.0001)",
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
                  "const chordDepth = () => visChords.queryArc(getTime(), getTime() + 0.0001).length / visChordMax"]
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
