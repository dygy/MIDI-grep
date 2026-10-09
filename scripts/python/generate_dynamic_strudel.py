#!/usr/bin/env python3
"""
Dynamic, NOTE-MATERIAL Strudel generator.

This is the real-technology path (not loop-replay): transcribed stems become
editable per-bar ``note("…")`` bar arrays, played by instruments that are either
library sounds OR custom PITCHED sample-instruments hosted on R2 (a sound mapped
across notes, driven by ``note(...).s("trackbass")`` — Strudel pitch-shifts the
nearest sample). Drums use the track's DETECTED groove on a library drum machine.

Output is the project's bar-array format so voices/bars are freely editable:

    let bass = ["cs2 ~ e2 ~", ...]
    let lead = ["e3 g3 ~ b3", ...]
    $: stack(
      note(cat(...bass)).s("trackbass").lpf(700),
      note(cat(...lead)).s("tracklead").room(0.2),
      <detected groove>.bank("RolandTR808"),
    )

The R2 sample-instrument is referenced via ``await samples("<base>/samples.json")``
where samples.json carries note-keyed maps + an absolute ``_base`` (trailing /).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

# reuse the drum-groove converter + effects helpers
sys.path.insert(0, str(Path(__file__).parent))
import generate_hybrid_strudel as H  # noqa: E402

NOTE_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]


def midi_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def fold_pitch(pitch: int, lo: int, hi: int) -> int:
    """Fold an out-of-range pitch into [lo,hi] by octaves (kills basic-pitch
    octave errors while preserving the pitch class / melodic contour)."""
    while pitch > hi:
        pitch -= 12
    while pitch < lo:
        pitch += 12
    return max(lo, min(hi, pitch))


def midi_to_bars(midi_path: Path, bpm: float, nbars: int, *, quantize: int = 16,
                 pick: str = "low", lo: int = 0, hi: int = 127) -> list[str] | None:
    """Quantise a MIDI file into per-bar 16-step note strings (monophonic line).

    pick='low' keeps the lowest note per step (bass line); 'high' keeps the
    highest (melody/lead). Out-of-range pitches are FOLDED into [lo,hi] by
    octaves so transcription octave-jumps don't create spurious highs/lows.
    Returns one string per bar, or None if empty.
    """
    if not midi_path.exists():
        return None
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return None
    bar_dur = 60.0 / bpm * 4
    step_dur = bar_dur / quantize
    total = nbars * quantize
    chosen: dict[int, int] = {}  # step -> folded pitch
    for n in notes:
        step = int(round(n.start / step_dur))
        if not (0 <= step < total):
            continue
        p = fold_pitch(n.pitch, lo, hi)
        if step not in chosen:
            chosen[step] = p
        else:
            better = p < chosen[step] if pick == "low" else p > chosen[step]
            if better:
                chosen[step] = p
    if not chosen:
        return None
    bars = []
    for b in range(nbars):
        slots = [midi_name(chosen[b * quantize + s]) if (b * quantize + s) in chosen else "~"
                 for s in range(quantize)]
        bars.append(" ".join(slots))
    return bars


def onset_driven_bars(stem_path: Path, midi_path: Path, bpm: float, nbars: int, *,
                      quantize: int = 16, pick: str = "low", lo: int = 0, hi: int = 127) -> list[str] | None:
    """Per-bar note strings whose RHYTHM comes from the stem's actual audio onsets and whose
    PITCH comes from the transcription.

    basic-pitch over-produces notes (~5x the real onsets — 1238 vs 237 for this bass), so
    grid-quantising ALL of them fills nearly every step → a continuous line where the original
    is sparse → the groove (onset correlation) is wrong. Instead we detect the stem's real
    onsets (the true rhythm), and at each onset read the transcribed pitch active then. Result:
    the original's sparse groove with the right notes.
    """
    if not stem_path.exists() or not midi_path.exists():
        return None
    y, _ = librosa.load(str(stem_path), sr=22050, mono=True)
    if y.size == 0:
        return None
    onsets = librosa.onset.onset_detect(y=y, sr=22050, hop_length=256, units="time", backtrack=True)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes or len(onsets) == 0:
        return None
    bar_dur = 60.0 / bpm * 4
    step_dur = bar_dur / quantize
    total = nbars * quantize
    chosen: dict[int, int] = {}
    for t in onsets:
        step = int(round(t / step_dur))
        if not (0 <= step < total):
            continue
        active = [n.pitch for n in notes if n.start - 0.06 <= t < n.end + 0.03]
        if not active:
            active = [min(notes, key=lambda n: abs(n.start - t)).pitch]
        raw = min(active) if pick == "low" else max(active)
        chosen[step] = fold_pitch(raw, lo, hi)
    if not chosen:
        return None
    bars = []
    for b in range(nbars):
        slots = [midi_name(chosen[b * quantize + s]) if (b * quantize + s) in chosen else "~"
                 for s in range(quantize)]
        bars.append(" ".join(slots))
    return bars


def js_array(name: str, bars: list[str]) -> str:
    items = ",\n  ".join(f'"{b}"' for b in bars)
    return f"let {name} = [\n  {items}\n]"


def bar_env_pattern(stem_path: Path, nbars: int, bpm: float, *,
                    floor: float = 0.0, gamma: float = 1.0,
                    silence: float = 0.10, steps: int = 8) -> list[list[float]] | None:
    """Per-step loudness envelope of an original stem, as ``nbars`` lists of ``steps`` floats
    (0..1). Consumed by ``gain_pattern`` which bakes it into one ``.gain("<[..] [..]>")``.

    This is the DYNAMICS lever for the stem-match SHAPE metric. The render plays every voice
    flat; the original drops voices in breakdowns, slams them in drops, and — crucially —
    varies WITHIN the bar (accented downbeats, note swells). We measure each original stem's
    RMS at ``steps`` sub-divisions per bar, normalise to its own peak, and emit one value per
    sub-step. Chained as a ``.gain()`` it MULTIPLIES the voice's base gain (Strudel composes
    gains), so the render's loudness contour tracks the original's at sub-bar resolution —
    raising the time-aligned envelope correlation. ``steps=1`` recovers the old per-bar
    behaviour; higher ``steps`` (4=beats, 8=8ths, 16=16ths) carries intra-bar accent/dynamics
    (e.g. a bassline whose downbeats are louder than its offbeats).

    ``floor`` keeps a *present* step from going fully silent; ``gamma``<1 lifts quiet steps,
    >1 deepens dynamics. Each bar == one cycle, aligned with ``cat(...)`` bar arrays.

    TRUE SILENCE is honoured: a step whose RMS is below ``silence`` × the *median active step*
    is emitted as a hard ``0`` (the voice drops out exactly where the original stem is silent
    — e.g. a bass that doesn't enter until the intro ends). The floor only applies to steps
    that are genuinely present, so it never resurrects a section the original leaves empty.
    The reference is the MEDIAN active step, not the peak, so one loud fill (common in drum
    stems, ~10× the groove) can't make the steady groove read as "silent".
    """
    if not stem_path.exists():
        return None
    y, _ = librosa.load(str(stem_path), sr=22050, mono=True)
    if y.size == 0:
        return None
    steps = max(1, int(steps))
    Lbar = (60.0 / bpm * 4) * 22050
    Ls = int(round(Lbar / steps))
    nsub = nbars * steps
    env = np.array([
        float(np.sqrt(np.mean(y[i * Ls:(i + 1) * Ls] ** 2))) if y[i * Ls:(i + 1) * Ls].size else 0.0
        for i in range(nsub)
    ])
    peak = float(env.max())
    if peak < 1e-6:
        return None
    # Robust silence gate: reference = median of steps that carry real signal (above the
    # stem's noise/bleed floor at ~1% of peak), so it tracks the groove level, not the peak.
    active = env[env > peak * 0.01]
    ref = float(np.median(active)) if active.size else peak
    silent_mask = env < (silence * ref)
    norm = (env / peak) ** gamma
    norm = floor + (1.0 - floor) * norm
    norm[silent_mask] = 0.0   # honour true gain-0 sections of the original stem
    return [[float(norm[b * steps + s]) for s in range(steps)] for b in range(nbars)]


# librosa imported lazily to keep the module importable without the audio stack
import librosa  # noqa: E402


def section_map(mix_path: Path, nbars: int, bpm: float):
    """Label bar ranges as a DJ arrangement (intro/build/drop/break/outro) from the mix's
    per-bar loudness, plus an 8-level sparkline. Returns (sparkline:str, ranges:list[(name,a,b)]).

    A live coder reads this to know which bars are the drop (full energy) vs the break
    (stripped) so they can solo/jump to a section, e.g. ``cat(...bass.slice(32,48))``.
    Purely informational (emitted as comments) — it changes no audio."""
    if not mix_path.exists():
        return None, None
    y, _ = librosa.load(str(mix_path), sr=22050, mono=True)
    if y.size == 0:
        return None, None
    L = int(round((60.0 / bpm * 4) * 22050))
    env = np.array([
        float(np.sqrt(np.mean(y[b * L:(b + 1) * L] ** 2))) if y[b * L:(b + 1) * L].size else 0.0
        for b in range(nbars)
    ])
    peak = float(env.max()) or 1.0
    norm = env / peak
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(7, int(v * 8))] for v in norm)
    # Smooth the per-bar energy (moving average, window 4 bars) before tiering so a DJ section
    # map reflects real arrangement blocks, not per-bar flicker.
    k = 4
    sm = np.convolve(norm, np.ones(k) / k, mode="same")
    tiers = [(0 if v < 0.30 else 1 if v < 0.62 else 2) for v in sm]  # quiet / mid / full
    ranges = []
    a = 0
    for b in range(1, nbars + 1):
        if b == nbars or tiers[b] != tiers[a]:
            ranges.append([tiers[a], a, b])  # [tier, start, end)
            a = b
    # Merge sections shorter than 4 bars into the previous section (kill sub-phrase noise).
    merged = []
    for r in ranges:
        if merged and (r[2] - r[1]) < 4:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    ranges = merged
    # name sections: first quiet=intro, last quiet=outro, full=drop, mid=build/verse, quiet mid=break
    named = []
    full_seen = False
    for i, (t, s, e) in enumerate(ranges):
        if t == 2:
            name = "drop"; full_seen = True
        elif t == 1:
            name = "build" if not full_seen else "verse"
        else:  # quiet
            if i == 0:
                name = "intro"
            elif i == len(ranges) - 1:
                name = "outro"
            else:
                name = "break"
        named.append((name, s, e))
    return spark, named


def pitched_map(pack_dir: Path, base: str) -> dict:
    """Note-keyed pitched-instrument map (trackbass/tracklead) + absolute _base."""
    sm = json.loads((pack_dir / "strudel.json").read_text())
    out = {"_base": base}
    for k in ("trackbass", "tracklead"):
        if isinstance(sm.get(k), dict) and sm[k]:
            out[k] = sm[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamic note-material Strudel generator.")
    ap.add_argument("--stems-dir", required=True, type=Path)
    ap.add_argument("--pack-dir", required=True, type=Path, help="build_sample_pack output (pitched maps)")
    ap.add_argument("--bass-midi", required=True, type=Path)
    ap.add_argument("--lead-midi", required=True, type=Path)
    ap.add_argument("--drums-json", type=Path, help="detect_drums output for the real groove")
    ap.add_argument("--base-url", required=True, help="R2/localhost base for the pitched instruments")
    ap.add_argument("--samples-url", default=None,
                    help="use a pre-built samples.json manifest at this URL (e.g. trained instruments) "
                         "instead of writing one from the pack; the generator references it verbatim")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--genre", default="brazilian_funk")
    ap.add_argument("--num-bars", type=int, default=16)
    ap.add_argument("--bass-sound", default="trackbass", help="trackbass (custom) or a library sound")
    ap.add_argument("--lead-sound", default="tracklead", help="tracklead (custom) or a library sound")
    ap.add_argument("--vocal-loop", action="store_true", default=True,
                    help="layer the vocal stem as a hosted loop (the one sound the library can't make)")
    ap.add_argument("--no-vocal-loop", dest="vocal_loop", action="store_false")
    ap.add_argument("--drum-mode", choices=["extracted", "bank", "realstem"], default="extracted",
                    help="extracted=DRUMS GENERATOR — the track's OWN extracted kick/snare/hat one-shots "
                         "(bd/sd/hh/oh) triggered by the detected, editable rhythm pattern (real sounds + "
                         "real notes); bank=same groove on a library drum machine; realstem=replay the whole "
                         "drum stem (not a generator — identical audio but un-editable)")
    ap.add_argument("--drum-stem-gain", type=float, default=1.0,
                    help="gain for the realstem drum channel (balances the real drums in the mix)")
    ap.add_argument("--drum-lpf", type=int, default=0,
                    help="low-pass the drum bus toward the original drum stem brightness (0=off)")
    # per-voice gain calibration (multiplies the voice's base gain) — set from a render's
    # measured stem RMS vs the original stem RMS so the inter-stem balance matches the original.
    ap.add_argument("--cal-bass", type=float, default=1.0)
    ap.add_argument("--cal-lead", type=float, default=1.0)
    ap.add_argument("--cal-drums", type=float, default=1.0)
    ap.add_argument("--cal-vocal", type=float, default=1.0)
    ap.add_argument("--master-gain", type=float, default=1.0,
                    help="master gain on the whole stack to back off the recorder's 0 dBFS limiter (~0.6)")
    ap.add_argument("--env-steps", type=int, default=4,
                    help="sub-bar resolution of the per-voice loudness envelope baked into .gain() "
                         "(1=per-bar arrangement only, 4=per-beat [default], 8=8ths, 16=16ths). Higher "
                         "carries more of the original's intra-bar dynamics/accents and true silence; "
                         "note that the 23ms energy metric can't fully reward it (render loudness is "
                         "note-event-driven), but per-beat is musically faithful without bloating the "
                         "gain pattern. 1 keeps the gain block compact/most editable.")
    ap.add_argument("--bass-mult", type=float, default=0.40,
                    help="trackbass band gain multiplier (lower = less 60-250Hz boom)")
    ap.add_argument("--bass-lpf", type=int, default=0,
                    help="override trackbass low-pass (Hz); low (~140) cuts upper-bass harmonics → shifts energy toward sub")
    ap.add_argument("--drum-hpf", type=int, default=0,
                    help="high-pass the drum bus (Hz); ~55 removes the 808 kick's stray sub (orig drums carry almost no sub)")
    ap.add_argument("--lead-lpf", type=int, default=5000,
                    help="lead low-pass (Hz); raise toward ~7000 to restore himid sparkle")
    ap.add_argument("--lead-hpf", type=int, default=300,
                    help="lead high-pass (Hz); lower toward ~150 to restore lowmid body")
    ap.add_argument("--sub-gain", type=float, default=0.7,
                    help="sub-sine layer gain (raise to fill the 20-60Hz sub_bass band)")
    ap.add_argument("--bass-cut", type=int, default=0,
                    help="cut-group id for the bass (e.g. 1) → MONOPHONIC bass: each note chokes the "
                         "previous so a flat sustained sample gives a continuous line with no "
                         "polyphonic pile-up. Pair with a long, flat bass sample.")
    ap.add_argument("--sub-octave", type=int, default=0,
                    help="octaves to drop the sub-sine so its fundamental lands in sub_bass (1 = -12 semis)")
    ap.add_argument("--dj-flow", action="store_true", default=True,
                    help="emit mutable per-voice $: channels + a section/arrangement map (normal "
                         "live-play DJ layout: comment a channel to mute, solo by soloing, slice to a drop)")
    ap.add_argument("--no-dj-flow", dest="dj_flow", action="store_false")
    ap.add_argument("--drum-floor", type=float, default=0.2,
                    help="drum envelope floor (0=deepest dynamics→best shape corr; higher=louder average)")
    ap.add_argument("--hat-gain", type=float, default=0.0,
                    help="steady TR808 hh*8 layer inside the drum bus (smooths envelope for shape + lifts centroid)")
    ap.add_argument("--rhythm", choices=["onset", "grid"], default="grid",
                    help="grid=quantise MIDI (better pitch/timbre fidelity); onset=stem onsets drive timing "
                         "(sparser but librosa bass onset detection is unreliable -> worse mfcc)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    nbars = args.num_bars
    cps = args.bpm / 60 / 4
    dur = (60.0 / args.bpm * 4) * nbars
    base = args.base_url.rstrip("/") + "/"

    # fold to musical ranges: bass E1-C3, lead C3-G5 (kills octave-error screech)
    # RHYTHM from the stem's real onsets, PITCH from the transcription (onset-driven) — the
    # transcription over-produces notes ~5x, so plain grid-quantising makes a too-busy line
    # that loses the original's groove. Fall back to midi_to_bars if onset detection fails.
    if args.rhythm == "onset":
        bass_bars = onset_driven_bars(args.stems_dir / "bass.wav", args.bass_midi, args.bpm, nbars,
                                      pick="low", lo=28, hi=48) \
            or midi_to_bars(args.bass_midi, args.bpm, nbars, pick="low", lo=28, hi=48)
        lead_bars = onset_driven_bars(args.stems_dir / "melodic.wav", args.lead_midi, args.bpm, nbars,
                                      pick="high", lo=48, hi=79) \
            or midi_to_bars(args.lead_midi, args.bpm, nbars, pick="high", lo=48, hi=79)
    else:
        bass_bars = midi_to_bars(args.bass_midi, args.bpm, nbars, pick="low", lo=28, hi=48)
        lead_bars = midi_to_bars(args.lead_midi, args.bpm, nbars, pick="high", lo=48, hi=79)
    if not bass_bars and not lead_bars:
        print("ERROR: no note material extracted", file=sys.stderr)
        return 1

    # effects aligned to the original stems
    ref_rms = max((H.stem_features(args.stems_dir / f"{s}.wav", dur) or {"rms": 1e-6})["rms"]
                  for s in ("drums", "bass", "melodic"))
    bass_fx = H.effects_for(H.stem_features(args.stems_dir / "bass.wav", dur), ref_rms,
                            kind="bass", library=(args.bass_sound != "trackbass"))
    lead_fx = H.effects_for(H.stem_features(args.stems_dir / "melodic.wav", dur), ref_rms,
                            kind="melodic", library=(args.lead_sound != "tracklead"))
    drums_fx = H.effects_for(H.stem_features(args.stems_dir / "drums.wav", dur), ref_rms,
                             kind="drums", library=True)

    # vocal stem as a hosted loop — the one sound the library has nothing for;
    # it also restores the original's mid/high/air that note-instruments lack.
    vocal_ok = False
    if args.vocal_loop:
        voc_out = args.pack_dir / "vocalsfull.wav"
        if args.samples_url:
            # An external manifest is already HOSTED; its vocalsfull.wav is authoritative. Do NOT
            # rewrite the local copy (write_continuous_loop would make a full nbars-long loop and
            # desync from the shorter hosted loop, so slice()/slow() would use the wrong bar count
            # and the vocal would play at the wrong rate). Use the existing local copy as-is —
            # vocalBars is measured from it below, so keep local == hosted.
            vocal_ok = voc_out.exists()
        else:
            vocal_ok = H.write_continuous_loop(
                args.stems_dir / "vocals.wav", voc_out, nbars, args.bpm)

    # realstem drums: play the track's OWN drum stem as a Strudel sample so the rendered drums
    # are the real drums (identical sound + spectrogram), via legit Strudel samples() — not a
    # library drum machine. The drum stem already carries its own dynamics, so no env is applied.
    drum_realstem = args.drum_mode == "realstem" and (args.pack_dir / "drumsfull.wav").exists()

    # If a pre-built manifest URL is given (e.g. trained instruments), reference it verbatim and
    # don't write a pack manifest. Otherwise build one from the pack's pitched maps / loops.
    samples_url = args.samples_url
    needs_samples = bool(samples_url) or (
        args.bass_sound == "trackbass" or args.lead_sound == "tracklead"
        or vocal_ok or drum_realstem)
    if needs_samples and not samples_url:
        smap = pitched_map(args.pack_dir, base)
        if vocal_ok:
            smap["vocalsfull"] = ["vocalsfull.wav"]
        if drum_realstem:
            smap["drumsfull"] = ["drumsfull.wav"]
        (args.pack_dir / "samples.json").write_text(json.dumps(smap, indent=2) + "\n")
        samples_url = f"{base}samples.json"

    # When an EXTERNAL manifest is referenced (e.g. trained instruments/samples.json), it carries
    # only the pitched instruments + drum one-shots — NOT the vocal/realstem-drum loops, which live
    # in the PACK manifest. The vocal voice's s("vocalsfull") would then resolve to nothing (silent
    # singer). Strudel merges successive samples() calls, so we add a SECOND samples() that loads
    # the co-hosted pack manifest ({base}samples.json, which always carries vocalsfull/drumsfull).
    # Use the proven STRING-URL form — Strudel's embed transpiler rejects the inline-object form
    # `samples({...}, base)` here with "Unexpected string". The pack manifest's other entries
    # (trackbass/tracklead) are fetched lazily, so loading it costs nothing unless they're played.
    extra_samples_url = None
    if args.samples_url and (
            (vocal_ok and (args.pack_dir / "vocalsfull.wav").exists())
            or drum_realstem):
        extra_samples_url = f"{base}samples.json"

    # slice()/run()/slow() of a hosted LOOP must use the bar-span of THAT SAMPLE, not the
    # arrangement length: x = round(sample_seconds * cps). A vocal/drum loop that is e.g. 16 bars
    # long (28s @136bpm) sliced by nbars=78 plays at the wrong rate (mangled). Measure the sample
    # and emit the count as a derived const so it tracks cps and the real loop length.
    def _sample_bars(p: Path) -> int:
        try:
            secs = float(librosa.get_duration(path=str(p)))
        except Exception:
            return nbars
        return max(1, round(secs * cps))

    vocal_bars = _sample_bars(args.pack_dir / "vocalsfull.wav") if vocal_ok else nbars
    drum_bars = _sample_bars(args.pack_dir / "drumsfull.wav") if drum_realstem else nbars

    L = [
        "// MIDI-grep dynamic — transcribed note material on pitched instruments",
        f"// genre={args.genre}",
        f"// BPM: {args.bpm:.0f}",
        f"// Key: {args.key}",
        f"// Bars: {nbars}  Duration: {dur:.0f}s",
        f"const cps = {cps:.6f}   // {args.bpm:.0f} BPM, 4 beats/bar",
        "setcps(cps)",
    ]
    if needs_samples:
        L.append(f'await samples("{samples_url}")')
    if extra_samples_url:
        # second manifest (pack) for loops missing from an external samples_url (e.g. vocalsfull).
        L.append(f'await samples("{extra_samples_url}")')
    if vocal_ok:
        L.append(f"const vocalBars = {vocal_bars}   // round(vocal_sample_secs * cps): bars the vocal loop spans")
    if drum_realstem:
        L.append(f"const drumBars = {drum_bars}   // round(drum_sample_secs * cps)")
    L.append("")

    # --- Spectral alignment (data-driven from measured band diffs) ---
    # rendered ran HOT in bass/low-mid and (especially) high-mid, and SHORT on
    # high/top air. With only gain/lpf/hpf + layering available in Strudel:
    bass_fx["gain"] = round(bass_fx["gain"] * args.bass_mult * args.cal_bass, 2)   # tame bass/low-mid boom
    if args.bass_lpf:
        bass_fx["lpf"] = args.bass_lpf   # cut trackbass upper-bass harmonics → energy shifts toward sub
    # NOTE: raising lead gain adds high-mid faster than lpf removes it (lead lives
    # in 2-3.5kHz), and a lower lpf clips needed hi/top air — 0.6x + lpf 5000 is
    # the measured optimum (overall 76.5%); going higher/lower both regress.
    lead_g = round(lead_fx["gain"] * 0.60 * args.cal_lead, 2)            # restore some mid body
    drums_fx["gain"] = round(drums_fx["gain"] * args.cal_drums, 2)       # inter-stem balance
    sub_g = round(args.sub_gain * args.cal_bass, 2)
    sub_transpose = f".add(note({-12 * args.sub_octave}))" if args.sub_octave else ""
    vocal_g = round(1.0 * args.cal_vocal, 2)

    # --- Arrangement dynamics: bake each original stem's per-bar loudness envelope into
    # its render voice so the render follows the original's drops/breakdowns (SHAPE metric).
    # floor keeps voices musically present; drums get floor 0 so they truly drop out in
    # breakdowns (the original drum stem is ~70% silent vs the render's flat 0%).
    bass_env = bar_env_pattern(args.stems_dir / "bass.wav", nbars, args.bpm, floor=0.15, steps=args.env_steps)
    # lead floor low so the render tracks the original melodic CONTOUR (its intrinsic
    # note-gaps add uncorrelated envelope variation; a lower floor lets the matched
    # per-bar shape dominate the correlation).
    lead_env = bar_env_pattern(args.stems_dir / "melodic.wav", nbars, args.bpm, floor=0.10, steps=args.env_steps)
    # drums: gamma 1.0 (linear) is the most faithful reproduction of the original drum
    # envelope -> highest SHAPE correlation (the primary gate). NOTE: the silence sub-gate
    # is NOT chasable on this track: the original drum STEM is dominated by one loud fill
    # (~10x the groove), so 70% of it reads "silent" vs that peak at 0.25 s resolution;
    # no gamma gets a genre-appropriate continuous groove past ~0.33 silence (measured),
    # and forcing it would mean deleting the drum groove. So we optimise shape, not silence.
    # drum floor 0.2 keeps the drums more consistently present (raises the demucs-recovered
    # average drum-stem level, which is gain-limited by clipping not boostable at the master)
    # while still ducking in breakdowns. gamma 1.0 = faithful envelope.
    drum_env = bar_env_pattern(args.stems_dir / "drums.wav", nbars, args.bpm, floor=args.drum_floor, gamma=1.0, steps=args.env_steps)
    voc_env = bar_env_pattern(args.stems_dir / "vocals.wav", nbars, args.bpm, floor=0.10, steps=args.env_steps)

    # IMPORTANT: Strudel's chained .gain() REPLACES (last wins), it does NOT multiply (verified
    # empirically). So a voice must carry exactly ONE .gain(): the per-bar envelope pattern with
    # the voice's base gain BAKED IN. gain_pattern folds base*env into a single `.gain("<...>")`.
    def gain_pattern(env_bars: list[list[float]] | None, base: float) -> str:
        base = round(float(base), 4)
        if not env_bars:
            return f".gain({base})"
        bars = []
        for steps in env_bars:
            if len(steps) == 1:                       # per-bar (one value per cycle)
                bars.append(f"{round(base * steps[0], 3)}")
            else:                                     # sub-bar: a bracketed sequence per cycle
                inner = " ".join(f"{round(base * v, 3)}" for v in steps)
                bars.append(f"[{inner}]")
        return f'.gain("<{" ".join(bars)}>")'

    # master gain folded INTO each voice's single baked gain (a separate trailing .gain() would
    # REPLACE the envelope — Strudel chained gain is last-wins). It backs the summed mix off the
    # recorder's 0 dBFS limiter without disturbing inter-voice balance.
    M = args.master_gain

    def fx_no_gain(fx, *, drop_lpf=False):
        parts = []
        if fx.get("lpf") and not drop_lpf:
            parts.append(f'.lpf({fx["lpf"]})')
        if fx.get("room"):
            parts.append(f'.room({fx["room"]})')
        return "".join(parts)

    # Each channel is an independent, mutable Strudel track: one ``$:`` block the performer can
    # comment out to mute, solo, or hot-swap live — the normal live-coding DJ layout. The note
    # material stays in the editable ``let`` arrays above; channels just voice it.
    channels = []  # (label, comment, code)  — code is a top-level pattern (no outer indent)
    if bass_bars:
        L += [js_array("bass", bass_bars), ""]
        # bass + sub-sine reinforcement kept together as ONE channel so muting bass mutes its sub.
        # Each line carries ONE baked gain = base * master * per-bar envelope.
        # cut(N): a cut group makes the bass MONOPHONIC — each note chokes the previous, so a
        # flat sustained sample plays at full level until the next note cuts it → a continuous,
        # non-piled-up bass line (matches a real mono synth; without it, long samples overlap
        # polyphonically and build into a loud, muddy cluster on a fast-retriggered pattern).
        bass_cut = f'.cut({args.bass_cut})' if args.bass_cut else ''
        bass_code = (
            "stack(\n"
            f'  note(cat(...bass)).s("{args.bass_sound}"){fx_no_gain(bass_fx)}{bass_cut}{gain_pattern(bass_env, bass_fx["gain"] * M)},\n'
            f'  note(cat(...bass)){sub_transpose}.s("sine").lpf(90){gain_pattern(bass_env, sub_g * M)}\n'
            ")"
        )
        channels.append(("BASS", "low end — root/sub. edit notes in `bass`", bass_code))
    if lead_bars:
        L += [js_array("lead", lead_bars), ""]
        # lpf tames high-mid top while keeping mid body; hpf clears low bleed.
        # clip(1.3) sustains notes slightly past their step so the monophonic lead's envelope
        # is smoother (closer to the original melodic's sustained contour, lifting shape corr).
        lead_code = (f'note(cat(...lead)).s("{args.lead_sound}")'
                     f'.lpf({args.lead_lpf}).hpf({args.lead_hpf}).clip(1.3).room({lead_fx["room"]})'
                     f'{gain_pattern(lead_env, lead_g * M)}')
        channels.append(("LEAD", "melody/harmony — edit notes in `lead`", lead_code))

    if drum_realstem:
        # The real drum stem, played gaplessly via Strudel slice playback → rendered drums are
        # the ORIGINAL drums (identical sound & spectrogram). Carries its own dynamics (no env).
        drum_code = ('s("drumsfull").slice(drumBars, run(drumBars)).slow(drumBars).clip(1)'
                     f'.gain({round(args.drum_stem_gain * M, 3)})')
        channels.append(("DRUMS", "REAL drum stem (your track's drums, 100% — via Strudel samples)", drum_code))
        detected = True  # for the summary flag
    else:
        detected = H.build_detected_drums(args.drums_json, args.bpm, nbars) if args.drums_json else None
        if detected:
            drum_lpf = f'.lpf({args.drum_lpf})' if args.drum_lpf else ''
            drum_lpf += f'.hpf({args.drum_hpf})' if args.drum_hpf else ''
            dlines = list(detected)
            if args.hat_gain > 0:
                dlines.append(f'  s("hh*8").gain({args.hat_gain})')
            if args.drum_mode == "extracted":
                # DRUMS GENERATOR: the detected pattern triggers the track's OWN extracted
                # bd/sd/hh one-shots (mapped in samples.json) — no library bank. Real sounds,
                # editable pattern, reusable in any song.
                bank_suffix = ""
                label = "your extracted kit (bd/sd/hh) triggered by the detected pattern — edit it"
            else:
                bank = H.GENRE_PATTERNS.get(args.genre, H.GENRE_PATTERNS["default"])["drum_bank"]
                bank_suffix = f'.bank("{bank}")'
                label = f'detected groove on {bank} — edit the patterns'
            drum_code = ("stack(\n" + ",\n".join("  " + d for d in dlines)
                         + f'\n){bank_suffix}{drum_lpf}{fx_no_gain(drums_fx, drop_lpf=True)}'
                         + f'{gain_pattern(drum_env, drums_fx["gain"] * M)}')
            channels.append(("DRUMS", label, drum_code))

    if vocal_ok:
        # loopAt(N) renders SILENT for large N on a long sample (embed bug). slice(N,run(N)).slow(N)
        # .clip(1) plays the full hosted vocal in order, one bar-slice/cycle, gaplessly.
        vocal_code = ('s("vocalsfull").slice(vocalBars, run(vocalBars)).slow(vocalBars).clip(1)'
                      f'.room(0.18){gain_pattern(voc_env, vocal_g * M)}')
        channels.append(("VOCAL", "real vocal stem (the one sound a synth can't make)", vocal_code))

    if args.dj_flow:
        # --- Performance header: arrangement map + how to play it live ---
        spark, sections = section_map(args.pack_dir / "originalfull.wav", nbars, args.bpm)
        L.append("// ┌── LIVE-PLAY DJ FLOW " + "─" * 40)
        L.append("// │ Each $: below is a channel — comment a line to MUTE it, or comment all-but-one to SOLO.")
        L.append("// │ Voices already build & drop with the song (per-bar gain envelopes track the original).")
        if spark and sections:
            L.append(f"// │ energy : {spark}")
            secstr = "  ".join(f"{n}[{a}-{b - 1}]" for n, a, b in sections)
            L.append(f"// │ sections: {secstr}")
            drops = [f"{a}-{b}" for n, a, b in sections if n == "drop"]
            if drops:
                L.append(f"// │ jump to a drop, e.g.:  $: note(cat(...bass.slice({drops[0].split('-')[0]}, {drops[0].split('-')[1]}))).s(\"{args.bass_sound}\")")
        L.append("// └" + "─" * 60)
        L.append("")
        for label, comment, code in channels:
            L.append(f"$: {code}  // {label} — {comment}")
            L.append("")
    else:
        body = ",\n".join("  " + c.replace("\n", "\n  ") for _, _, c in channels)
        L += [f"$: stack(", body, ")", ""]

    voices = channels  # for the summary count below

    out = args.out or (args.pack_dir / "output_dynamic.strudel")
    code = "\n".join(L) + "\n"
    out.write_text(code)

    # Spec 003 Slice 1: the same validator the LLM codegen path uses now also sees this
    # generator's output, so a replay voice (values.md A1 — e.g. the `vocalsfull` loop) is
    # reported at generation time instead of being discovered after a render. Warn-only until
    # Slice 3 makes the vocal voice editable; Slice 2 makes compare/gate refuse to score it.
    try:
        from strudel_validation import validate_code
        _, validation_error = validate_code(code, autocorrect=False)
    except Exception as exc:  # pragma: no cover - validator unavailable must not block output
        validation_error = f"validator unavailable: {exc}"
    if validation_error:
        print(f"WARNING editability: {validation_error}", file=sys.stderr)

    print(json.dumps({"out": str(out), "bass_bars": len(bass_bars or []),
                      "lead_bars": len(lead_bars or []), "drums": bool(detected),
                      "bass_sound": args.bass_sound, "lead_sound": args.lead_sound,
                      "uses_custom_samples": needs_samples,
                      "validation": validation_error or "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
