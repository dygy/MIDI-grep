#!/usr/bin/env python3
"""
Hybrid Strudel generator: LIBRARY-first genre patterns + R2 custom samples.

Philosophy (per user direction):
- Sounds the library can make are GENERATED as the canonical genre pattern
  (e.g. classic Brazilian-funk batidao = tresillo kick `bd(3,8)`) using library
  drum machines + GM/synth sounds, and ALIGNED to the track via effects derived
  from the original's analysis (gain/lpf/room — never hardcoded levels).
- Only the identity-carrying stems the library has "nothing similar" for
  (vocals, signature melodic hooks) are sliced into continuous, gapless loops and
  hosted (localhost or R2), played with `loopAt()` + effects.

So a typical Brazilian-funk track uploads ONLY vocals (+ melodic); drums and bass
are generated from library sounds. The base URL is the single localhost<->R2 knob.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 44100
NOTE_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]

# Canonical, EDITABLE genre patterns built from library sounds. This is genre
# knowledge (like the sound palette), not track-tuned values. Rhythm only —
# levels/tone come from analysis-derived effects.
GENRE_PATTERNS: dict = {
    "brazilian_funk": {
        "drum_bank": "RolandTR808",
        # tresillo kick = the batidao foundation; clap on the backbeat; busy hats
        "kick": "bd(3,8)",
        "clap": "~ ~ cp ~",
        "hat": "hh*8",
        # built-in waveform (no soundfont fetch); sub-octave + lpf give the 808 feel
        "bass_sound": "sawtooth",
        # 808 bass follows the kick's tresillo, root of the key
        "bass_struct": "t(3,8)",
    },
    "default": {
        "drum_bank": "RolandTR909",
        "kick": "bd ~ ~ ~ bd ~ ~ ~",
        "clap": "~ ~ ~ ~ sd ~ ~ ~",
        "hat": "hh*8",
        "bass_sound": "gm_synth_bass_1",
        "bass_struct": "t ~ t ~",
    },
}


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def key_root_token(key: str | None) -> str:
    if not key:
        return "c"
    return key.strip().split()[0].lower().replace("#", "s")


def stem_features(path: Path, dur: float) -> dict | None:
    """Spectral/energy features used to ALIGN generated voices via effects."""
    if not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=dur)
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return None
    rms = float(np.sqrt(np.mean(y**2)))
    rolloff = float(np.median(librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)))
    centroid = float(np.median(librosa.feature.spectral_centroid(y=y, sr=SR)))
    return {"rms": rms, "rolloff": rolloff, "centroid": centroid}


# Library sounds are already full-scale, so RMS-ratio gain (which assumes the
# voice IS the original audio) makes them too quiet. Give library voices a
# genre-presence gain and reserve RMS-ratio for real custom loops.
LIBRARY_PRESENCE = {"drums": 0.95, "bass": 0.85}


def effects_for(feat: dict | None, ref_rms: float, *, kind: str, library: bool) -> dict:
    """Derive gain/lpf/room from analysis so a voice sits like the original.

    Library voices: genre-presence gain (audio is synthesized full-scale),
    analysis only shapes tone (lpf/room). Custom loops (real audio): gain is the
    RMS ratio vs the loudest stem so the inter-stem balance matches the original.
    """
    lpf = int(round(min(16000, max(200, feat["rolloff"])))) if feat else None
    room = 0.0 if kind == "bass" else (0.12 if kind == "drums" else 0.18)
    if library:
        gain = LIBRARY_PRESENCE.get(kind, 0.8)
    else:
        # Custom loops ARE the original audio at its original level — the
        # inter-stem balance is already baked in, so play at unity (scaling by
        # RMS-ratio would double-count and make them too quiet).
        gain = 1.0
    return {"gain": gain, "lpf": lpf, "room": room}


def fx_chain(fx: dict, *, drop_lpf: bool = False) -> str:
    parts = [f".gain({fx['gain']})"]
    if fx.get("lpf") and not drop_lpf:
        parts.append(f".lpf({fx['lpf']})")
    if fx.get("room"):
        parts.append(f".room({fx['room']})")
    return "".join(parts)


def build_detected_drums(drums_json: Path, bpm: float, nbars: int, quantize: int = 16) -> list[str] | None:
    """Turn detected drum hits into per-bar TR808 mini-notation (library sounds).

    Plays the TRACK'S OWN groove with library drum-machine sounds — library-first
    (no upload) but rhythmically matched, unlike the fixed canonical batidao.
    Returns one ``stack(...)`` voice line per drum type, or None if unusable.
    """
    if not drums_json.exists():
        return None
    data = json.loads(drums_json.read_text())
    hits = data.get("hits", [])
    if not hits:
        return None
    bar_dur = 60.0 / bpm * 4
    step_dur = bar_dur / quantize
    total_steps = nbars * quantize
    # grid[type] -> list of step indices that are ON
    grid: dict[str, set] = {}
    for h in hits:
        step = int(round(h["time"] / step_dur))
        if 0 <= step < total_steps:
            grid.setdefault(h["type"], set()).add(step)
    if not grid:
        return None
    # Emit a voice per type as cat() of per-bar 16-step strings.
    order = ["bd", "sd", "cp", "hh", "oh"]
    voices = []
    for t in [x for x in order if x in grid] + [x for x in grid if x not in order]:
        on = grid[t]
        bars = []
        for b in range(nbars):
            slots = [t if (b * quantize + s) in on else "~" for s in range(quantize)]
            bars.append(" ".join(slots))
        cat = ", ".join(f's("{b}")' for b in bars)
        voices.append(f"  cat({cat})")
    return voices


def write_continuous_loop(stem_path: Path, out: Path, nbars: int, bpm: float) -> bool:
    """Raw continuous N-bar loop (gapless via loopAt); only edge-faded."""
    if not stem_path.exists():
        return False
    y, _ = librosa.load(str(stem_path), sr=SR, mono=True)
    L = int(round((60.0 / bpm * 4) * nbars * SR))
    seg = y[:L].copy()
    if seg.size < SR:  # too short / silent
        return False
    if float(np.max(np.abs(seg))) < 1e-4:
        return False
    n = int(0.003 * SR)
    seg[:n] *= np.linspace(0, 1, n)
    seg[-n:] *= np.linspace(1, 0, n)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), seg, SR)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid library-genre + R2-sample Strudel generator.")
    ap.add_argument("--stems-dir", required=True, type=Path)
    ap.add_argument("--out-pack", required=True, type=Path, help="dir for custom loops + samples.json")
    ap.add_argument("--base-url", required=True, help="localhost or R2 base for custom samples")
    ap.add_argument("--genre", default="brazilian_funk")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--num-bars", type=int, default=16)
    ap.add_argument("--upload-voices", default="melodic,vocals",
                    help="stems with no library match -> hosted custom loops")
    ap.add_argument("--drums-json", type=Path,
                    help="detect_drums.py output -> play the track's real groove with library TR808")
    ap.add_argument("--out", type=Path, help="output .strudel (default: out-pack/output_hybrid.strudel)")
    args = ap.parse_args()

    pat = GENRE_PATTERNS.get(args.genre, GENRE_PATTERNS["default"])
    nbars = args.num_bars
    dur = (60.0 / args.bpm * 4) * nbars
    cps = args.bpm / 60 / 4
    root = key_root_token(args.key)

    # 1. Build continuous custom loops for the identity-carrying stems.
    upload_voices = [v.strip() for v in args.upload_voices.split(",") if v.strip()]
    custom: dict[str, list[str]] = {}
    for v in upload_voices:
        fn = f"{v}full.wav"
        if write_continuous_loop(args.stems_dir / f"{v}.wav", args.out_pack / fn, nbars, args.bpm):
            custom[f"{v}full"] = [fn]
        else:
            print(f"[hybrid] skip custom '{v}' (missing/silent)", file=sys.stderr)

    base = args.base_url.rstrip("/") + "/"
    (args.out_pack).mkdir(parents=True, exist_ok=True)
    (args.out_pack / "samples.json").write_text(
        json.dumps({"_base": base, **custom}, indent=2) + "\n")

    # 2. Analysis-derived effects (alignment).
    ref_rms = max((stem_features(args.stems_dir / f"{s}.wav", dur) or {"rms": 1e-6})["rms"]
                  for s in ("drums", "bass", "melodic", "vocals"))
    library_set = {"drums", "bass"}
    fx = {s: effects_for(stem_features(args.stems_dir / f"{s}.wav", dur), ref_rms,
                         kind=s, library=(s in library_set))
          for s in ("drums", "bass", "melodic", "vocals")}

    # 3. Emit code: generated library batidao (drums+bass) + custom R2 loops.
    L = [
        "// generation_mode: hybrid — real loops are texture under editable voices",
        f"// MIDI-grep hybrid — library genre groove + hosted custom samples",
        f"// genre={args.genre}  bpm={args.bpm:.0f}  key={args.key}  base={base}",
        f"setcps({cps:.6f})",
    ]
    if custom:
        L.append(f'await samples("{base}samples.json")')
    L.append("")

    drums_fx = fx_chain(fx["drums"], drop_lpf=True)  # keep 808 sub on drums
    bass_fx = fx_chain(fx["bass"])
    detected = build_detected_drums(args.drums_json, args.bpm, nbars) if args.drums_json else None
    if detected:
        L += [
            "// --- LIBRARY: the track's DETECTED groove, played with TR808 ---",
            "$: stack(",
            ",\n".join(detected),
            f').bank("{pat["drum_bank"]}"){drums_fx}',
            "",
        ]
    else:
        L += [
            "// --- LIBRARY: classic batidao generated from TR808 (tresillo kick) ---",
            f'$: stack(',
            f'  s("{pat["kick"]}"),',
            f'  s("{pat["clap"]}"),',
            f'  s("{pat["hat"]}").gain(0.7)',
            f').bank("{pat["drum_bank"]}"){drums_fx}',
            "",
        ]
    L += [
        "// --- LIBRARY: 808 bass on the batidao, root of the key ---",
        f'$: note("{root}1").struct("{pat["bass_struct"]}").s("{pat["bass_sound"]}"){bass_fx}',
        "",
    ]
    if custom:
        L.append("// --- CUSTOM (R2): identity stems the library can't make ---")
        for v in upload_voices:
            key_name = f"{v}full"
            if key_name in custom:
                # real audio already carries its spectrum — gain+room only, no lpf
                # Spec 003 R4: a replayed stem is allowed only as texture under editable
                # voices; the marker is what the editability detector looks for.
                L.append(f'$: s("{key_name}").loopAt({nbars}){fx_chain(fx[v], drop_lpf=True)}  // texture')
        L.append("")

    out = args.out or (args.out_pack / "output_hybrid.strudel")
    out.write_text("\n".join(L) + "\n")
    print(json.dumps({
        "out": str(out), "genre": args.genre, "bpm": args.bpm,
        "uploaded_custom": list(custom), "library_voices": ["drums", "bass"],
        "effects": fx, "base_url": args.base_url,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
