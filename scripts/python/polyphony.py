#!/usr/bin/env python3
"""Peak simultaneous sound sources of a generated .strudel file (live-player voice budget).

superdough keeps one ``activeSoundSources`` entry per hap chain (a 7-voice supersaw = 1) from the moment the scheduler
TRIGGERS the hap — up to ``latency + interval + overlap`` before it sounds — until its source node ends
(hold + release; a sample without ``clip``/``release`` plays its whole slice). When the cap (``maxPolyphony``) is
a number, the OLDEST entries are killed once the count reaches it. This module reads the generated code (bar
arrays + the layers that play them) and sweeps those lifetimes to get the peak count.

Model, per layer (one source per note of a chord, per hit):
  * sample sound (name in the pack's samples.json): alive = min(hold, slice length / playback rate) + release + 0.01,
    where hold = hap * clip if ``.clip`` is set, else the whole slice
  * oscillator / soundfont sound: alive = hap * (clip or 1) + release     (default release 0.01 s)
  * drums: hits of ``bd/sd/hh/oh`` play their whole slice (extracted kit length when known, else the longest kit file)
``burst`` is the most onsets sharing one instant — the granularity of the estimate, used as headroom.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import soundfile as sf

# superdough / @strudel/core Cyclist constants: the hap's source is created `deadline` before it sounds, up to
# latency (0.1) + callback interval (0.1) + overlap (0.1) — core/cyclist.mjs, core/zyklus.mjs
TRIGGER_LEAD_S = 0.1 + 0.1 + 0.1
DEFAULT_RELEASE_S = 0.01      # getADSRValues releaseMin
SAMPLE_END_PAD_S = 0.01       # sampler: bufferSource.stop(holdEnd + release + 0.01)

_NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def note_midi(name: str) -> float | None:
    m = re.fullmatch(r"([a-g])([#sf]*)(-?\d+)", name.lower())
    if not m:
        return None
    acc = sum(1 if c in "#s" else -1 for c in m.group(2))
    return 12 * (int(m.group(3)) + 1) + _NOTE[m.group(1)] + acc


# ---- mini-notation (the subset the generator emits: rests, notes, [a,b] stacks, [a b] groups, @w, *n, _, <a b>) ----
def _split_top(s: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[<":
            depth += 1
        elif ch in "]>":
            depth -= 1
        if depth == 0 and ((sep == " " and ch.isspace()) or ch == sep):
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _events(seq: str, t0: float, span: float) -> list[tuple[float, float, str]]:
    """(start, duration, name) of every sounding step of a mini-notation sequence inside [t0, t0+span)."""
    toks = _split_top(seq.strip(), " ")
    items = []                                     # [weight, token]
    for tok in toks:
        if tok == "_" and items:
            items[-1][0] += 1
            continue
        w = 1.0
        m = re.search(r"@([\d.]+)", tok)
        if m:
            w = float(m.group(1))
            tok = tok.replace(m.group(0), "")
        items.append([w, tok])
    total = sum(w for w, _ in items) or 1.0
    out, pos = [], 0.0
    for w, tok in items:
        s, d = t0 + span * pos / total, span * w / total
        pos += w
        rep = 1
        m = re.search(r"\*(\d+)", tok)
        if m:
            rep = int(m.group(1))
            tok = tok.replace(m.group(0), "")
        tok = tok.rstrip("?!")
        if tok in ("~", ".", ""):
            continue
        for r in range(rep):
            rs, rd = s + d * r / rep, d / rep
            if tok.startswith("[") and tok.endswith("]"):
                for part in _split_top(tok[1:-1], ","):
                    out += _events(part, rs, rd)
            elif tok.startswith("<") and tok.endswith(">"):
                for alt in _split_top(tok[1:-1], " "):      # conservative: every alternative sounds
                    out += _events(alt, rs, rd)
            else:
                out.append((rs, rd, re.sub(r"\(.*\)$", "", tok)))
    return out


# ---- reading the generated code ----
def _arrays(code: str) -> dict[str, list[str]]:
    arrs = {}
    for m in re.finditer(r"^let (\w+) = \[\n(.*?)\n\]", code, re.S | re.M):
        arrs[m.group(1)] = re.findall(r'"([^"]*)"', m.group(2))
    return arrs


def _slice_lengths(samples_json: Path | None) -> dict[str, list[tuple[float | None, float]]]:
    """{sound: [(midi or None, seconds)]} from a samples.json whose files sit next to it."""
    if not samples_json or not Path(samples_json).exists():
        return {}
    d = json.loads(Path(samples_json).read_text())
    out = {}
    for name, v in d.items():
        if name.startswith("_"):
            continue
        entries = list(v.items()) if isinstance(v, dict) else [(None, f) for f in v]
        lst = []
        for key, f in entries:
            for f1 in (f if isinstance(f, list) else [f]):
                p = Path(samples_json).parent / f1
                if p.exists():
                    info = sf.info(str(p))
                    lst.append((note_midi(key) if key else None, info.frames / info.samplerate))
        if lst:
            out[name] = lst
    return out


def estimate(code: str, samples_json: Path | None = None) -> dict:
    cps_m = re.search(r"^const cps = ([\d.]+)", code, re.M)
    if not cps_m:
        raise ValueError("no `const cps` in code")
    bar = 1.0 / float(cps_m.group(1))
    arrs = _arrays(code)
    lens = _slice_lengths(samples_json)
    kit_max = max((s for n in ("bd", "sd", "hh", "oh") for _, s in lens.get(n, [])), default=1.0)
    lines = [ln for ln in code.splitlines() if not ln.lstrip().startswith("//")]
    spans: list[tuple[float, float]] = []          # (alive from, alive until)
    starts: list[float] = []
    layers: dict[str, int] = {}

    def add(layer: str, start: float, alive: float):
        spans.append((start - TRIGGER_LEAD_S, start + alive))
        starts.append(round(start, 4))
        layers[layer] = layers.get(layer, 0) + 1

    def chain_param(chain: str, name: str) -> float | None:
        m = re.search(rf"\.{name}\(([\d.]+)\)", chain)
        return float(m.group(1)) if m else None

    # drum block: the `stack(` holding cat(s("..."), ...) lines; its sound comes from the tokens themselves
    for i, ln in enumerate(lines):
        for m in re.finditer(r"cat\(\.\.\.(\w+)\)", ln):
            bars = arrs.get(m.group(1))
            if not bars:
                continue
            chain = ln[m.end():]
            sm = re.search(r'\.(?:s|sound)\("([^"]+)"\)', chain)
            sound = sm.group(1) if sm else None
            clip, rel = chain_param(chain, "clip"), chain_param(chain, "release")
            release = rel if rel is not None else DEFAULT_RELEASE_S
            for b, bs in enumerate(bars):
                for s, d, name in _events(bs, b * bar, bar):
                    if sound in lens:
                        mi = note_midi(name)
                        cand = lens[sound]
                        if mi is not None:
                            k = min(cand, key=lambda c: abs((c[0] if c[0] is not None else mi) - mi))
                            rate = 2 ** ((mi - k[0]) / 12) if k[0] is not None else 1.0
                        else:
                            k, rate = cand[0], 1.0
                        full = k[1] / rate
                        hold = min(d * clip, full) if clip is not None else (d if rel is not None else full)
                        add(m.group(1), s, min(hold + release, full) + SAMPLE_END_PAD_S)
                    else:
                        add(m.group(1), s, d * (clip if clip is not None else 1.0) + release)
        if re.match(r'\s*cat\(s\("', ln) or re.match(r'\s*s\("(?:hh|bd|sd|oh)\*', ln):
            if re.match(r'\s*cat\(', ln):
                bars = re.findall(r's\("([^"]*)"\)', ln)
            else:
                bars = [re.match(r'\s*s\("([^"]*)"', ln).group(1)]
            for b, bs in enumerate(bars):
                for s, d, name in _events(bs, b * bar if len(bars) > 1 else 0.0, bar):
                    if len(bars) > 1:
                        add("drums", s, max((x for _, x in lens.get(name, [])), default=kit_max) + SAMPLE_END_PAD_S)
                    else:
                        # a steady layer (`s("hh*8")`) repeats every bar for the whole piece
                        total_bars = max((len(v) for v in arrs.values()), default=1)
                        for bb in range(total_bars):
                            add("drums", s + bb * bar, max((x for _, x in lens.get(name, [])), default=kit_max) + SAMPLE_END_PAD_S)
    if not spans:
        return {"peak": 0, "burst": 0, "layers": {}, "events": 0}
    ev = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, dlt in ev:
        cur += dlt
        peak = max(peak, cur)
    burst = max(starts.count(x) for x in set(starts))
    return {"peak": peak, "burst": burst, "layers": layers, "events": len(spans)}


def main() -> int:
    code = Path(sys.argv[1]).read_text()
    sj = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(estimate(code, sj)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
