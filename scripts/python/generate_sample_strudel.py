#!/usr/bin/env python3
"""
Generate Strudel code that plays a sample-pack hosted at a base URL.

The pack (from ``build_sample_pack.py``) is loaded in Strudel with a single
``samples({...}, "<base>/")`` call. The base URL is the ONLY thing that changes
between local serving (``http://localhost:5555/<prefix>``) and Cloudflare R2
(``https://pub-xxxx.r2.dev/<prefix>``) — the code is identical otherwise.

Why the inline map + explicit base (instead of ``samples("<url>/strudel.json")``):
``@strudel/webaudio`` resolves a fetched JSON's base as
``url.split('/').slice(0,-1).join('/')`` — i.e. WITHOUT a trailing slash — then
prepends it to each path by raw string concat (``base + path``), yielding
``.../regime-cltloops/x.wav``. It also only accepts ARRAY-valued entries. Passing
the map inline with an explicit base that ends in ``/`` sidesteps both issues and
keeps a single portable knob: the base URL.

Modes:
  loops      (default) - reconstruct the track from the real per-bar stem loops,
                         one bar per cycle. Highest similarity (~99% on tests):
                         it IS the original audio, re-arrangeable live.
  instrument           - play the classified drum one-shots + a pitched
                         representative bass/lead sample. Full live control.
  hybrid               - real drum + bass loops under a pitched lead line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_sample_map(sm: dict, mode: str) -> dict:
    """Build an inline, array-valued Strudel sample map for the chosen mode.

    Every value is a list of relative paths (the only shape this Strudel build's
    ``samples()`` accepts). The base URL is applied by Strudel at load time.
    """
    out: dict[str, list[str]] = {}

    # Loops are arrays already — keep the ones that exist.
    for name in ("drumsloop", "bassloop", "melodicloop", "vocalsloop"):
        if isinstance(sm.get(name), list) and sm[name]:
            out[name] = sm[name]

    if mode in ("instrument", "hybrid"):
        # Drum one-shots: wrap each string path as a single-element array.
        for name in ("bd", "sd", "hh", "oh"):
            if isinstance(sm.get(name), str):
                out[name] = [sm[name]]
        # Pitched voices: pick ONE representative sample and let note() pitch it
        # (the note-keyed object form is not supported by this Strudel build).
        for voice in ("trackbass", "tracklead"):
            rep = _representative(sm.get(voice))
            if rep:
                out[voice] = [rep]

    return out


def _representative(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and entry:
        return entry[len(entry) // 2]
    if isinstance(entry, dict) and entry:
        vals = list(entry.values())
        return vals[len(vals) // 2]
    return None


def seq_indices(n: int) -> str:
    """A slow-changing mini-notation pattern: one index per cycle."""
    return "<" + " ".join(str(i) for i in range(n)) + ">"


def write_resolved_json(pack_dir: Path, sample_map: dict, base_url: str) -> str:
    """Write a host-resolved sample map (arrays + absolute ``_base``).

    Strudel's ``samples(jsonUrl)`` prepends ``_base`` to each path by raw concat,
    so ``_base`` MUST be absolute and end with ``/``. We bake the target host in
    here; re-run codegen with a different --base-url to retarget (localhost↔R2).
    The file is named per-host-hash-free ``samples.json`` and lives in the pack
    dir, so the uploader ships it alongside the WAVs.
    """
    base = base_url.rstrip("/") + "/"
    resolved = {"_base": base, **sample_map}
    out = pack_dir / "samples.json"
    out.write_text(json.dumps(resolved, indent=2) + "\n")
    return base + "samples.json"


def header(samples_url: str, base_url: str, bpm: float, key: str | None) -> list[str]:
    cps = bpm / 60 / 4  # 1 cycle == 1 bar (4/4)
    return [
        "// MIDI-grep sample-pack playback",
        f"// Samples hosted at: {base_url.rstrip('/')}/",
        f"// BPM {bpm:.0f}" + (f"  Key {key}" if key else ""),
        "// Retarget by re-running codegen with a different --base-url (localhost <-> R2).",
        "",
        f"setcps({cps:.6f})",
        f'await samples("{samples_url}")',
        "",
    ]


def loops_block(pack: dict, sm: dict) -> list[str]:
    nbars = int(pack.get("num_bars") or len(sm.get("drumsloop", [])) or 16)
    idx = seq_indices(nbars)
    lines = ["// --- LOOPS: real stem audio, one bar per cycle (max similarity) ---"]
    voices = []
    # Raw stem loops carry the original inter-stem balance, so play each at unity.
    for name in ("drumsloop", "bassloop", "melodicloop", "vocalsloop"):
        if isinstance(sm.get(name), list) and sm[name]:
            voices.append(f'  s("{name}").n("{idx}").clip(1)')
    lines.append("$: stack(\n" + ",\n".join(voices) + "\n)")
    lines.append("")
    return lines


def instrument_block(sm: dict, key: str | None) -> list[str]:
    """Genre pattern from the real one-shots + a pitched representative sample."""
    lines = ["// --- INSTRUMENT: real one-shots + pitched samples, live-codeable ---"]
    drum_rows = []
    if isinstance(sm.get("bd"), str):
        drum_rows.append('  s("bd ~ ~ bd ~ ~ bd ~")')
    if isinstance(sm.get("sd"), str):
        drum_rows.append('  s("~ ~ sd ~ ~ ~ sd ~")')
    if isinstance(sm.get("hh"), str):
        hats = "hh*6 oh hh" if isinstance(sm.get("oh"), str) else "hh*8"
        drum_rows.append(f'  s("{hats}").gain(0.7)')
    if drum_rows:
        lines.append("$: stack(\n" + ",\n".join(drum_rows) + "\n)")
    if _representative(sm.get("trackbass")):
        root = _root_note(sm["trackbass"], key)
        lines.append(f'$: note("{root} ~ {root} {root}").s("trackbass").clip(0.9)')
    if _representative(sm.get("tracklead")):
        pick = _mid_note(sm["tracklead"], key)
        lines.append(f'$: note("~ {pick} ~ {pick}").s("tracklead").clip(0.8).gain(0.8)')
    lines.append("")
    return lines


def hybrid_block(pack: dict, sm: dict, key: str | None) -> list[str]:
    nbars = int(pack.get("num_bars") or len(sm.get("drumsloop", [])) or 16)
    idx = seq_indices(nbars)
    lines = ["// --- HYBRID: real drum+bass loops under a pitched lead ---"]
    voices = []
    for name in ("drumsloop", "bassloop"):
        if isinstance(sm.get(name), list) and sm[name]:
            voices.append(f'  s("{name}").n("{idx}").clip(1)')
    if _representative(sm.get("tracklead")):
        pick = _mid_note(sm["tracklead"], key)
        voices.append(f'  note("~ {pick} ~ {pick}").s("tracklead").clip(0.8).gain(0.8)')
    lines.append("$: stack(\n" + ",\n".join(voices) + "\n)")
    lines.append("")
    return lines


def _midi_of(note: str) -> int:
    names = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
    name, octave = note[:-1], int(note[-1])
    return names.index(name) + (octave + 1) * 12


def _root_note(pitched: object, key: str | None) -> str:
    if key:
        return key.strip().split()[0].lower().replace("#", "s") + "1"
    if isinstance(pitched, dict) and pitched:
        return min(pitched.keys(), key=_midi_of)
    return "c1"


def _mid_note(pitched: object, key: str | None) -> str:
    if isinstance(pitched, dict) and pitched:
        notes = sorted(pitched.keys(), key=_midi_of)
        return notes[len(notes) // 2]
    if key:
        return key.strip().split()[0].lower().replace("#", "s") + "3"
    return "c3"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Strudel code for a hosted sample-pack.")
    ap.add_argument("--pack-dir", required=True, type=Path)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://localhost:5555/regime-clt or https://pub-xxxx.r2.dev/regime-clt")
    ap.add_argument("--mode", choices=["loops", "instrument", "hybrid"], default="loops")
    ap.add_argument("--out", type=Path, help="output .strudel (default: pack-dir/output_<mode>.strudel)")
    args = ap.parse_args()

    pack = json.loads((args.pack_dir / "pack.json").read_text())
    sm = json.loads((args.pack_dir / "strudel.json").read_text())

    # The resolved samples.json is a superset (all modes) so a single upload
    # serves every mode; per-mode code just references the sound names it needs.
    full_map = build_sample_map(sm, "instrument")  # instrument == superset
    samples_url = write_resolved_json(args.pack_dir, full_map, args.base_url)
    lines = header(samples_url, args.base_url, float(pack.get("bpm", 120)), pack.get("key"))
    if args.mode == "loops":
        lines += loops_block(pack, sm)
    elif args.mode == "instrument":
        lines += instrument_block(sm, pack.get("key"))
    else:
        lines += hybrid_block(pack, sm, pack.get("key"))

    out = args.out or (args.pack_dir / f"output_{args.mode}.strudel")
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "mode": args.mode, "base_url": args.base_url,
                      "samples_url": samples_url, "sounds": list(full_map)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
