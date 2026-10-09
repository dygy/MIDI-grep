#!/usr/bin/env python3
"""A3 — audition drum banks: render a fixed pattern per bank, measure its timbre, cache the vectors.

Drum banks are sample sets (not synths), so we can't read a static timbre vector for them — we render
each bank playing the SAME standard pattern via BlackHole, analyze (brightness/warmth/attack), and
cache the result. `sound_timbre.resolve_drum_bank()` then picks the bank nearest the original drums.
The cache is track-independent — build it ONCE. Run with no other BlackHole render in flight.

Usage: python audition_drum_banks.py [--banks RolandTR808 RolandTR909 …] [--secs 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sound_timbre import analyze_stem_timbre

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDER = ROOT / "scripts" / "node" / "dist" / "record-strudel-blackhole.js"
CACHE = ROOT / "eval" / "drum_bank_timbre.json"
PATTERN = 'bd hh sd hh bd hh sd oh'   # a standard kit groove, same for every bank

# A diverse default set spanning the character space (deep/punchy/vintage-sampled/electronic).
DEFAULT_BANKS = [
    "RolandTR808", "RolandTR909", "RolandTR707", "LinnDrum",
    "AkaiMPC60", "AkaiLinn", "OberheimDMX", "EmuSP12",
]


def audition(bank: str, secs: int) -> dict | None:
    if not RECORDER.exists():
        print(f"  recorder missing: {RECORDER}", file=sys.stderr)
        return None
    code = f'setcps(0.5)\n\n$: s("{PATTERN}").bank("{bank}").gain(0.9)\n'
    with tempfile.NamedTemporaryFile("w", suffix=".strudel", delete=False) as tf:
        tf.write(code)
        strudel = tf.name
    out = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(["node", str(RECORDER), strudel, "-o", out, "-d", str(secs)],
                       check=False, capture_output=True, text=True, timeout=secs + 120)
        if not Path(out).exists():
            print(f"  {bank}: render produced no file", file=sys.stderr)
            return None
        t = analyze_stem_timbre(out)
        return {k: round(v, 4) for k, v in t.items()}
    finally:
        for p in (strudel, out):
            Path(p).unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audition drum banks → timbre cache")
    ap.add_argument("--banks", nargs="*", default=DEFAULT_BANKS)
    ap.add_argument("--secs", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-audition even if cached")
    args = ap.parse_args(argv)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for bank in args.banks:
        if bank in cache and not args.force:
            print(f"  {bank}: cached {cache[bank]}")
            continue
        print(f"  auditioning {bank}…", file=sys.stderr)
        vec = audition(bank, args.secs)
        if vec:
            cache[bank] = vec
            print(f"  {bank}: {vec}")
        else:
            print(f"  {bank}: SKIPPED (render failed)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"\nCache written: {CACHE} ({len(cache)} banks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
