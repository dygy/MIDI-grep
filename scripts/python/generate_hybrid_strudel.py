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


# The lowpass is a CEILING for a sample layer, not a typical value: it must not remove what the stem itself sounds.
# Energy fraction kept below a frame's rolloff: the conventional "bandwidth" (all but the top 1 %). Measured on the five
# reference bass stems, the median per-frame 85 % rolloff leaves 0.8-3.0 % of the sounding energy above the cutoff
# (VAGABUNDO 3.0 %), the 99 % one < 0.005 % on all five. Enter: lpf 215 Hz -> ~2.8 kHz (a bass-stem A/B at 2918 Hz: +0.022).
ROLLOFF_ENERGY = 0.99
ANALYSIS_SR = 22050          # the rate the pipeline analyses stems at (bass_shape.StemEvidence); no content above it is musical


def stem_features(path: Path, dur: float, *, oscillator: bool = False) -> dict | None:
    """Spectral/energy features used to ALIGN generated voices via effects.

    ``oscillator=True`` (synth mode, library voices): the voice is a raw oscillator whose harmonics have nothing to
    do with the stem's own spectrum, so the cutoff keeps its original derivation (median 85 % rolloff over all
    frames); the sounding-energy rolloff below is valid only for a sample cut from the stem.

    ``rolloff`` = median, over the stem's SOUNDING frames (Otsu over its own log frame RMS, as bass_shape), of each
    frame's :data:`ROLLOFF_ENERGY` rolloff at the analysis rate. Rests / noise-floor frames are excluded: their rolloff
    is the noise's (Enter, quiet third of the frames: 5.8 kHz, sounding 0.2-2.8 kHz), so the all-frame median depended on
    how much of the stem is rest rather than on how it sounds."""
    if not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=dur)
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return None
    rms = float(np.sqrt(np.mean(y**2)))
    if oscillator:
        return {"rms": rms, "rolloff": float(np.median(librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85))),
                "centroid": float(np.median(librosa.feature.spectral_centroid(y=y, sr=SR)))}
    import bass_shape as BS
    ya = librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR)
    S = np.abs(librosa.stft(ya, n_fft=2048, hop_length=512))
    frame_rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    ro = librosa.feature.spectral_rolloff(S=S, sr=ANALYSIS_SR, roll_percent=ROLLOFF_ENERGY)[0]
    sounding = frame_rms >= BS.sounding_level(frame_rms)
    rolloff = float(np.median(ro[sounding] if sounding.any() else ro))
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


VELOCITY_LO_PCT, VELOCITY_HI_PCT = 10, 90   # the band the dynamic span is reported over (p10-p90)


def grid_phase(times: list[float], step_dur: float) -> float:
    """Seconds by which the hits sit LATE of the step grid (negative = early), the circular mean of frac(t / step).

    ``round(t / step)`` assumes the grid starts at t = 0; a stem whose beat grid starts elsewhere (Dumb Blond Bitches:
    -51 ms = 0.46 step) sent about half its hits to the neighbouring slot (mean error 49 ms, against 17-23 ms on
    tracks whose grid does start near 0). 0 when the hits give no phase (spread evenly over the step) - the estimate
    is used only if it lowers the mean quantisation error."""
    if len(times) < 2 or step_dur <= 0:
        return 0.0
    t = np.asarray(times, dtype=float)
    mean = np.exp(2j * np.pi * (t / step_dur)).mean()
    phase = float(np.angle(mean) / (2 * np.pi) * step_dur)

    def err(off: float) -> float:
        r = (t - off) / step_dur
        return float(np.abs(r - np.round(r)).mean())

    return phase if err(phase) < err(0.0) else 0.0


def playable_phase(times: list[float], step_dur: float) -> float:
    """`grid_phase` taken modulo a step, in [0, step): the grid point nearest the piece start is at or after 0.

    The slots are shifted by ``.late(phase)`` over a cyclic ``cat``. A negative shift sends step 0 to negative time
    (never heard) and replays the next repetition's step 0 just before the end of the piece; a shift in [0, step)
    moves every slot later, and the previous repetition's last slot (at -step + phase) stays before 0."""
    return grid_phase(times, step_dur) % step_dur if step_dur > 0 else 0.0


def slot_of(t: float, phase: float, step_dur: float) -> int:
    """Step a hit at *t* is written on: the nearest grid point, but never one before the piece start (such a hit
    takes the first playable step instead of being dropped)."""
    return max(0, int(round((t - phase) / step_dur)))


class DetectedDrums(list):
    """The drum voice lines of `build_detected_drums`, with the ``.late()`` (cycles) that puts their slots back on the
    stem's times: the phase they were slotted with, not a second estimate."""
    late: float = 0.0


def drums_late(drums_json: Path, bpm: float, quantize: int = 16) -> float:
    """The grid phase of the detected drums in CYCLES (one bar = one cycle): what `build_detected_drums` returns as
    ``.late`` for the same hits and *quantize*. 0.0 when there is nothing to correct."""
    if not drums_json.exists():
        return 0.0
    hits = json.loads(drums_json.read_text()).get("hits", [])
    bar_dur = 60.0 / bpm * 4
    return playable_phase([h["time"] for h in hits], bar_dur / quantize) / bar_dur


def late_suffix(late: float) -> str:
    """``.late(x)`` for a nonzero phase (cycles), else nothing."""
    return f".late({late:.6f})" if abs(late) >= 1e-6 else ""


def hit_velocities(hits: list[dict], bpm: float, nbars: int, quantize: int = 16,
                   mode: str = "bar", phase: float | None = None) -> dict[tuple[str, int], float]:
    """Measured per-hit velocity: {(drum type, step) -> strength / reference strength of that type}.

    ``strength`` comes from detect_drums_bands (linear band peak at the onset). The reference is the MEDIAN
    strength of the hit's own class — within the same bar (``mode="bar"``: the bar-to-bar level is already carried
    by the drum envelope, so only the within-bar accent is added) or over the whole piece (``mode="global"``).
    Ratios are winsorised per class to their own p10..p90, then scaled to unit mean power. A bar holding fewer than 3 hits of a class has no meaningful median of its own and uses the piece median.
    Hits without a strength (older drums_bands.json) get no entry. Two hits on one step keep the stronger.
    Steps are counted from the grid phase (``phase`` seconds; measured from the hits when None), as in `build_detected_drums`."""
    step_dur = 60.0 / bpm * 4 / quantize
    total = nbars * quantize
    if phase is None:
        phase = playable_phase([h["time"] for h in hits], step_dur)
    placed: dict[tuple[str, int], float] = {}
    for h in hits:
        st = h.get("strength")
        step = slot_of(h["time"], phase, step_dur)
        if st is None or not step < total or not st > 0:
            continue
        k = (h["type"], step)
        placed[k] = max(placed.get(k, 0.0), float(st))
    by_class: dict[str, list[float]] = {}
    by_bar: dict[tuple[str, int], list[float]] = {}
    for (t, step), st in placed.items():
        by_class.setdefault(t, []).append(st)
        by_bar.setdefault((t, step // quantize), []).append(st)
    med = {t: float(np.median(v)) for t, v in by_class.items()}
    out = {}
    for (t, step), st in placed.items():
        ref = med[t]
        if mode == "bar" and len(by_bar[(t, step // quantize)]) >= 3:
            ref = float(np.median(by_bar[(t, step // quantize)]))
        out[(t, step)] = st / ref
    # winsorise each class to the p10..p90 of its own measured ratios: the extremes of a band-peak detector are
    # bleed (a quiet bar's one real hit reads 26x its neighbours) and a raw 28 dB velocity swing scored -0.11
    for t in by_class:
        ks = [k for k in out if k[0] == t]
        lo, hi = np.percentile([out[k] for k in ks], [VELOCITY_LO_PCT, VELOCITY_HI_PCT])
        for k in ks:
            out[k] = float(min(max(out[k], lo), hi))
        # energy-preserving: the drum gain is calibrated for velocity 1, so the class's mean power must stay 1
        # (a median reference reads a bimodal class — VAGABUNDO's hats — at mean velocity 3.3 and scored -0.11)
        rms = float(np.sqrt(np.mean([out[k] ** 2 for k in ks])))
        for k in ks:
            out[k] /= rms
    return out


def build_detected_drums(drums_json: Path, bpm: float, nbars: int, quantize: int = 16,
                         velocity: str | None = None,
                         pitch: dict[str, dict[float, float]] | None = None) -> DetectedDrums | None:
    """Turn detected drum hits into per-bar TR808 mini-notation (library sounds).

    Plays the TRACK'S OWN groove with library drum-machine sounds — library-first
    (no upload) but rhythmically matched, unlike the fixed canonical batidao.
    Returns one ``stack(...)`` voice line per drum type, or None if unusable.

    ``pitch`` ({class: {hit time -> playback-rate ratio}}, `drum_pitch.hit_ratios`) is the measured pitch each hit of a
    tuned class (the 808 kick) sounds in the stem relative to its one-shot: written as a parallel ``.speed()`` pattern
    (1 on a hit whose pitch could not be read), only for a class where at least one hit differs from the one-shot.

    Hits are slotted relative to the measured grid phase (`playable_phase`); the caller plays the stack ``.late(result.late)``
    so the slots land on the stem's own times. ``pitch`` is still keyed by the hits' original times.
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
    phase = playable_phase([h["time"] for h in hits], step_dur)
    grid: dict[str, set] = {}
    for h in hits:
        step = slot_of(h["time"], phase, step_dur)
        if step < total_steps:
            grid.setdefault(h["type"], set()).add(step)
    if not grid:
        return None
    vel = hit_velocities(hits, bpm, nbars, quantize, velocity, phase) if velocity else {}
    # measured pitch ratio per (class, step); the first hit on a step wins
    rate: dict[tuple[str, int], float] = {}
    for h in sorted(hits, key=lambda h: h["time"]):
        r = (pitch or {}).get(h["type"], {}).get(h["time"])
        step = slot_of(h["time"], phase, step_dur)
        if r is not None and step < total_steps:
            rate.setdefault((h["type"], step), r)
    tuned = {t for (t, _), r in rate.items() if abs(r - 1.0) > 1e-9}
    # Emit a voice per type as cat() of per-bar 16-step strings.
    order = ["bd", "sd", "cp", "hh", "oh"]
    voices = DetectedDrums()
    voices.late = phase / bar_dur
    for t in [x for x in order if x in grid] + [x for x in grid if x not in order]:
        on = grid[t]
        bars = []
        for b in range(nbars):
            slots = [t if (b * quantize + s) in on else "~" for s in range(quantize)]
            line = f's("{" ".join(slots)}")'
            if vel:
                # parallel velocity pattern, one value per hit (same steps, rests stay rests): velocity MULTIPLIES
                # the bar's .gain() envelope in superdough, so the accents ride on the level instead of replacing it
                vs = [f"{vel[(t, b * quantize + s)]:.3g}" if slot != "~" and (t, b * quantize + s) in vel else "~"
                      for s, slot in enumerate(slots)]
                if all((v != "~") == (sl != "~") for v, sl in zip(vs, slots)) and any(v != "~" for v in vs):
                    line += f'.velocity("{" ".join(vs)}")'
            if t in tuned:
                rs = [f"{rate.get((t, b * quantize + s), 1.0):.4g}" if slot != "~" else "~"
                      for s, slot in enumerate(slots)]
                if any(r not in ("~", "1") for r in rs):
                    line += f'.speed("{" ".join(rs)}")'
            bars.append(line)
        cat = ", ".join(bars)
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
    fx = {s: effects_for(stem_features(args.stems_dir / f"{s}.wav", dur, oscillator=(s in library_set)), ref_rms,
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
            f'){late_suffix(detected.late)}.bank("{pat["drum_bank"]}"){drums_fx}',
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
