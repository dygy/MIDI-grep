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

Vocals are NEVER generated (product decision): a vocal is a person singing, not part of the generated
music. The voices are bass, lead and drums; ``vocals.wav`` is not read here.

Generation modes (spec 003 Slice 4 — ``--mode``):
  sample-instrument (default)  the pack's pitched instruments (``trackbass``/``tracklead``/
                               or a hosted manifest) via ``await samples(...)``
  synth                        NO ``samples(...)`` and no custom sample names: bass/lead
                               sounds and the drum machine come from the ``sound_selector`` genre
                               palette (``pick_palette_sound``), drums take the ``--drum-mode bank``
                               path. Same editable
                               bar arrays, envelopes and ``setcps``.
Both write ``// generation_mode: <mode>`` so the detector/report never guess.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pretty_midi

# reuse the drum-groove converter + effects helpers
sys.path.insert(0, str(Path(__file__).parent))
import generate_hybrid_strudel as H  # noqa: E402
import partial_register as PR  # noqa: E402
import bass_register as BR  # noqa: E402  (stem-measured register + octave snap)
import bass_shape as BS  # noqa: E402  (bass note shape from the stem: hold, glide, sample loop, kick duck)
import melodic_shape as MS  # noqa: E402  (lead/chords: retrigger only on stem attacks, hold between)
import drum_pitch as DP  # noqa: E402  (the pitch each stem kick sounds, written per hit)
from editability_check import check_editability  # noqa: E402  (spec 003 detector)
from sound_selector import pick_palette_sound  # noqa: E402  (genre palette — no hardcoded names)
from strudel_validation import VALID_SOUNDS  # noqa: E402  (library sound catalog)

GENERATION_MODES = ("sample-instrument", "synth")

# Typed registers of the pre-2026-10-19 generator (bass E1-C3, lead C3-G5). Used ONLY when the stem
# measurement (bass_register.py) is missing/stale, with a warning: they put a C#1 sub an octave up.
LEGACY_BASS_WINDOW = (28, 48)
LEGACY_LEAD_WINDOW = (48, 79)
SYNTH_LEAD_HPF_START = 300   # synth mode only: first guess of the calibrator's lead-hpf lever (sample modes derive it)
BASS_ORBIT = 2             # the bass' own audio bus: the kick's duck targets it (everything else stays on orbit 1)
SUB_LPF_HZ = 90            # low-pass of the sine sub layer (the sub_bass band is 20-60 Hz; 90 keeps one octave of margin)

NOTE_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]


def midi_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _bars_pitches(bars: list[str] | None) -> list[int]:
    """MIDI pitches of the note names in ``bars`` (rests skipped)."""
    pitches = []
    for bar in bars or []:
        for tok in bar.split():
            m = re.fullmatch(r"([a-g]s?)(-?\d+)(?:@\d+)?", tok)
            if m and m.group(1) in NOTE_NAMES:
                pitches.append((int(m.group(2)) + 1) * 12 + NOTE_NAMES.index(m.group(1)))
    return pitches


def _chord_bars_pitches(bars: list[str] | None) -> list[int]:
    """MIDI pitches of every note name in ``bars``, chord tokens (``[c3,e3]``) included (rests skipped)."""
    pitches = []
    for bar in bars or []:
        for name, octv in re.findall(r"([a-g]s?)(-?\d+)", bar.replace("@", " @")):
            pitches.append((int(octv) + 1) * 12 + NOTE_NAMES.index(name))
    return pitches


def _bars_hz(bars: list[str] | None, pick) -> float:
    pitches = _bars_pitches(bars)
    return round(440.0 * 2 ** ((pick(pitches) - 69) / 12), 2) if pitches else 0.0


def sample_lead_hpf(*bar_sets: list[str] | None) -> int:
    """High-pass (Hz) for a sampled lead/chord voice: the lowest fundamental among everything it plays (0 = no notes).

    A cutoff above that removes the body of the lowest notes. VAGABUNDO's lead/chords reach cs3 (139 Hz) while the
    old fixed 300 Hz cut the 100-300 Hz region where the melodic stem is strongest: the voice rendered alone matched
    the stem's per-frame log-mel shape at 0.638 with hpf 300, 0.680 with hpf = the lowest fundamental, 0.686 with
    the filter off. The grains already carry the instrument's own spectrum; nothing here is shaped by hand."""
    pitches = [p for bars in bar_sets for p in _chord_bars_pitches(bars)]
    return int(440.0 * 2 ** ((min(pitches) - 69) / 12)) if pitches else 0


def bars_max_hz(bars: list[str] | None) -> float:
    """Highest fundamental (Hz) among the note names of ``bars`` (0.0 if none). Reported so the
    calibrator can derive the bass low-pass floor from the actual material, not a typed value."""
    return _bars_hz(bars, max)


def bars_min_hz(bars: list[str] | None) -> float:
    """Lowest fundamental (Hz) among the note names of ``bars`` (0.0 if none); floor of the lead
    high-pass lever (a cutoff below the lowest note filters nothing)."""
    return _bars_hz(bars, min)


def fold_pitch(pitch: int, lo: int, hi: int) -> int:
    """Fold an out-of-range pitch into [lo,hi] by octaves (the unvoiced / no-measurement fallback;
    the window itself comes from the stem, see bass_register.py)."""
    return BR.fold_into(pitch, lo, hi)


def _step_pitch(cands: list[int], t: float, step_dur: float, pick: str, lo: int, hi: int,
                register: "BR.Register | None") -> int:
    """The note for one step. With a measured stem register: snap to the stem's local octave and take the
    candidate nearest the stem f0 (BR.resolve); without (lead, or no measurement): fold into [lo,hi], then
    lowest/highest. The f0 is read around the step: a quarter step before to 1.5 steps after the onset."""
    loc = register.local(t - step_dur / 4, t + 1.5 * step_dur) if register is not None else None
    return BR.resolve(cands, loc, (lo, hi), pick)


def loudest_pitches(group) -> list[int]:
    """The transcribed notes of one step that carry the most amplitude (Basic Pitch velocity), as pitches.

    A polyphonic stem transcribes 2-8 notes per step; the one the stem actually sounds is the loudest, not the
    highest. Measured against the melodic stem's own chroma (Regime CLT / VAGABUNDO): the highest note names the
    stem's pitch class 23% / 67% of the time, the loudest 52% / 80%. ``group`` holds objects with ``pitch`` and
    ``velocity``; ties keep every tied pitch (the window fold then takes the highest)."""
    top = max(n.velocity for n in group)
    return [n.pitch for n in group if n.velocity == top]


def midi_to_bars(midi_path: Path, bpm: float, nbars: int, *, quantize: int = 16,
                 pick: str = "low", lo: int = 0, hi: int = 127,
                 register: "BR.Register | None" = None, sustain: bool = False) -> list[str] | None:
    """Quantise a MIDI file into per-bar 16-step note strings (monophonic line).

    All notes quantised to a step are candidates; ``_step_pitch`` decides the octave and the winner
    (stem-measured register when ``register`` is given, else window fold + lowest/highest per ``pick``).
    ``sustain``: a step no note STARTS on is written with the loudest note still SOUNDING there (a held note keeps
    its pitch for its duration, as the stem does) instead of a rest/the next onset's pitch.
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
    groups: dict[int, list] = {}
    for n in notes:
        step = int(round(n.start / step_dur))
        if 0 <= step < total:
            groups.setdefault(step, []).append(n)
    if not groups:
        return None
    if sustain:
        for st in range(total):
            if st in groups:
                continue
            t_now = st * step_dur
            held = [n for n in notes if n.start < t_now < n.end]
            if held:
                groups[st] = held
    # pick="loud": the loudest transcribed note names the step (see loudest_pitches); the window fold decides its octave
    cands = {s: loudest_pitches(g) if pick == "loud" else [n.pitch for n in g] for s, g in groups.items()}
    fold_pick = "high" if pick == "loud" else pick
    chosen = {s: _step_pitch(c, s * step_dur, step_dur, fold_pick, lo, hi, register) for s, c in cands.items()}
    bars = []
    for b in range(nbars):
        slots = [midi_name(chosen[b * quantize + s]) if (b * quantize + s) in chosen else "~"
                 for s in range(quantize)]
        bars.append(" ".join(slots))
    return bars


def onset_driven_bars(stem_path: Path, midi_path: Path, bpm: float, nbars: int, *,
                      quantize: int = 16, pick: str = "low", lo: int = 0, hi: int = 127,
                      register: "BR.Register | None" = None) -> list[str] | None:
    """Per-bar note strings whose RHYTHM comes from the stem's actual audio onsets and whose
    PITCH comes from the transcription (octave/winner decided as in ``midi_to_bars``).

    basic-pitch over-produces notes (~5x the real onsets — 1238 vs 237 for this bass), so
    grid-quantising ALL of them fills nearly every step → a continuous line where the original
    is sparse → the groove (onset correlation) is wrong. Instead we detect the stem's real
    onsets (the true rhythm), and at each onset read the transcribed pitch active then.
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
        active_notes = [n for n in notes if n.start - 0.06 <= t < n.end + 0.03]
        if not active_notes:
            active_notes = [min(notes, key=lambda n: abs(n.start - t))]
        active = loudest_pitches(active_notes) if pick == "loud" else [n.pitch for n in active_notes]
        chosen[step] = _step_pitch(active, float(t), step_dur, "high" if pick == "loud" else pick, lo, hi, register)
    if not chosen:
        return None
    bars = []
    for b in range(nbars):
        slots = [midi_name(chosen[b * quantize + s]) if (b * quantize + s) in chosen else "~"
                 for s in range(quantize)]
        bars.append(" ".join(slots))
    return bars


def partial_intervals(max_partial: int | None = None) -> set[int]:
    """Semitone intervals (rounded) from a note's fundamental to its harmonic partials 2..K: round(12*log2(k)) =
    12 (octave), 19 (12th), 24, 28, 31, 34, 36 ... K is not typed: it is the first partial whose interval no
    longer rounds to a NEW semitone distance below the next one's, i.e. the partials that are distinguishable
    from each other (k and k+1 map to different semitones only for k < 1/(2^(1/12)-1) ~ 16.8). ``max_partial``
    overrides it for tests."""
    k_max = max_partial or int(1.0 / (2 ** (1.0 / 12.0) - 1.0))
    return {int(round(12.0 * np.log2(k))) for k in range(2, k_max + 1)}


def otsu_threshold(values) -> float:
    """Otsu's between-class-variance threshold of ``values`` (the split that best separates a quiet cluster from a
    loud one). No parameters: the cut comes from the distribution itself. Returns min(values) when the sample is
    constant/too small to split."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size < 2 or v[0] == v[-1]:
        return float(v[0]) if v.size else 0.0
    best_t, best_var = float(v[0]), -1.0
    for i in range(1, v.size):
        if v[i] == v[i - 1]:
            continue
        w0, w1 = i / v.size, 1.0 - i / v.size
        var = w0 * w1 * (v[:i].mean() - v[i:].mean()) ** 2
        if var > best_var:
            best_var, best_t = var, float(v[i])
    return best_t


def harmony_notes(midi_path: Path, bpm: float, nbars: int, *, quantize: int = 16, lo: int = 0, hi: int = 127
                  ) -> tuple[dict[int, list[int]], dict]:
    """The REAL harmony under the lead's top line, per quantised step: {step: [pitches below the top note]}.

    The lead keeps ONE note per step (the highest, after folding into [lo,hi]); the transcription holds
    2.6-7.8 on average. A transcribed note is kept as harmony only when it is
      1. salient: its velocity (the transcriber's amplitude) is at/above the Otsu split of the stem's own velocity
         distribution (the quiet cluster is bleed / ghost notes), and
      2. not a harmonic partial of a LOUDER simultaneous note: pitch - other pitch in ``partial_intervals()``
         (octave, 12th, 2 octaves ...) with the other note louder - a partial is the same instrument's overtone,
         not a second voice.
    The kept notes are octave-folded into [lo,hi] (the lead's own window) and deduplicated against each other and
    the top note. Returns (steps, info) - info carries the derived salience floor and counts for the summary.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    info = {"salience_floor": None, "notes": len(notes), "kept": 0, "rejected_quiet": 0, "rejected_partial": 0,
            "gain_ratio": 1.0}
    kept_vel: list[int] = []
    top_vel: list[int] = []
    if not notes:
        return {}, info
    # Otsu on log-velocity: loudness is perceived (and the transcriber's amplitude is distributed) on a log scale,
    # so the split sits between the ghost-note cluster and the played notes instead of inside the played ones.
    floor = float(np.exp(otsu_threshold(np.log([max(1, n.velocity) for n in notes]))))
    info["salience_floor"] = floor
    ivals = partial_intervals()
    step_dur = 60.0 / bpm * 4 / quantize
    total = nbars * quantize
    groups: dict[int, list] = {}
    for n in notes:
        s = int(round(n.start / step_dur))
        if 0 <= s < total:
            groups.setdefault(s, []).append(n)
    out: dict[int, list[int]] = {}
    for s, g in groups.items():
        # the lead's note (the step's loudest, as written by midi_to_bars(pick="loud")) is not harmony
        top = max(BR.fold_into(p, lo, hi) for p in loudest_pitches(g))
        keep: list[int] = []
        top_vel.append(max(n.velocity for n in g if BR.fold_into(n.pitch, lo, hi) == top))
        for n in g:
            if n.velocity < floor:
                info["rejected_quiet"] += 1
                continue
            if any(m is not n and m.velocity > n.velocity and (n.pitch - m.pitch) in ivals for m in g):
                info["rejected_partial"] += 1
                continue
            p = BR.fold_into(n.pitch, lo, hi)
            if p != top and p not in keep:
                keep.append(p)
                kept_vel.append(n.velocity)
        if keep:
            out[s] = sorted(keep)
            info["kept"] += len(keep)
    if kept_vel and top_vel:
        # Each chord note plays at the lead sample's level, so the voice's loudness relative to the top line is the
        # ratio of the transcriber's amplitudes (median kept harmony velocity / median top-note velocity).
        info["gain_ratio"] = float(np.median(kept_vel) / np.median(top_vel))
    return out, info


def harmony_bars(steps: dict[int, list[int]], nbars: int, *, quantize: int = 16) -> list[str] | None:
    """Per-bar 16-step strings of chord tokens (``[c3,e3,g3]``; one pitch -> a plain name; ``~`` rests)."""
    if not steps:
        return None
    bars = []
    for b in range(nbars):
        slots = []
        for s in range(quantize):
            ps = steps.get(b * quantize + s)
            slots.append("~" if not ps else (midi_name(ps[0]) if len(ps) == 1
                                             else "[" + ",".join(midi_name(p) for p in ps) + "]"))
        bars.append(" ".join(slots))
    return bars


def fmt_bpm(bpm: float) -> str:
    """BPM with the precision of the value ``cps`` is derived from (4 decimals, trailing zeros trimmed)."""
    return f"{bpm:.4f}".rstrip("0").rstrip(".")


def count_note_events(*voices: list[str] | None) -> int:
    """Note events the bar arrays actually play: every non-rest token across all bars of all voices."""
    return sum(1 for bars in voices for bar in (bars or []) for tok in bar.split() if tok != "~")


_DRUM_STEP_RE = re.compile(r's\("([^"]*)"\)')


def count_drum_hits(drum_voices: list[str] | None) -> int:
    """Drum hits in the detected drum pattern lines (``cat(s("bd ~ ..."), ...)`` per drum type): non-rest
    tokens inside each ``s("...")`` bar. Steady layers added outside the detected pattern are not counted."""
    return sum(1 for line in (drum_voices or []) for bar in _DRUM_STEP_RE.findall(line)
               for tok in bar.split() if tok != "~")


def count_header_lines(genre: str, notes: int, drum_hits: int) -> list[str]:
    """Header comments the report parsers read (generate_report.py ``extract_info_from_strudel`` and
    internal/report ``extractStrudelInfo``): ``Notes: N``, ``Drums: N hits``, ``Genre: x``. A zero count is
    omitted (reports show N/A rather than a false 0)."""
    out = [f"// Genre: {genre}"]
    if notes > 0:
        out.append(f"// Notes: {notes}")
    if drum_hits > 0:
        out.append(f"// Drums: {drum_hits} hits")
    return out


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


ENV_CORRECTION_VOICES = ("bass", "lead", "master")
ENV_CORRECTION_CLAMP = "clamp"   # reserved key in load_env_correction()'s result


def load_env_correction(path: Path, nbars: int) -> dict[str, list[float]]:
    """Read a ``calibrate_dynamic.py --env-correction-out`` JSON into per-voice, per-bar
    multipliers aligned to ``nbars`` (index-aligned; the last multiplier is held when the file
    has fewer bars, extras are dropped, with a warning on any mismatch).

    The returned dict also carries the bound for the COMPOSED (voice x master) multiplier under
    the reserved key ``"clamp"`` = ``[lo, hi]``: the file's own ``clamp`` when present, else the
    ``[min, max]`` over every voice array in the file (the correction's own range). The bound is
    always data, never a generator constant.

    Raises ``ValueError`` with the offending path/field on malformed input.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--env-correction {path}: cannot read JSON ({exc})") from exc
    voices = raw.get("voices") if isinstance(raw, dict) else None
    if not isinstance(voices, dict):
        raise ValueError(f"--env-correction {path}: missing object field 'voices'")
    out: dict[str, list[float]] = {}
    for name in ENV_CORRECTION_VOICES:
        vals = voices.get(name)
        if vals is None:
            continue
        if (not isinstance(vals, list) or not vals
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in vals)):
            raise ValueError(f"--env-correction {path}: voices.{name} must be a non-empty list of "
                             "non-negative numbers")
        out[name] = [float(v) for v in vals]
    if not out:
        raise ValueError(f"--env-correction {path}: voices has none of {list(ENV_CORRECTION_VOICES)}")
    file_bars = raw.get("bars", max(len(v) for v in out.values()))
    if file_bars != nbars or any(len(v) != nbars for v in out.values()):
        print(f"WARNING env-correction: {Path(path).name} has bars={file_bars} but --num-bars={nbars}; "
              "aligning by index and holding the last multiplier", file=sys.stderr)
    result: dict[str, list[float]] = {k: [v[min(i, len(v) - 1)] for i in range(nbars)] for k, v in out.items()}
    clamp = raw.get("clamp")
    if clamp is not None:
        if (not isinstance(clamp, list) or len(clamp) != 2
                or not all(isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0 for c in clamp)
                or clamp[0] > clamp[1]):
            raise ValueError(f"--env-correction {path}: clamp must be [lo, hi] with 0 <= lo <= hi")
        result[ENV_CORRECTION_CLAMP] = [float(clamp[0]), float(clamp[1])]
    else:
        flat = [x for v in out.values() for x in v]
        result[ENV_CORRECTION_CLAMP] = [min(flat), max(flat)]
    return result


def master_bus_lines(spec: dict | None) -> list[str]:
    """Strudel lines for a measured master bus (mix_dynamics.derive_master_bus). Each call is guarded with ``typeof``, so the
    SAME file plays on engines that define ``setMasterCompressor`` / ``setMasterLimiter`` (the fork) and on those that
    do not (npm @strudel/web 1.3.0, the hosted site), where the lines are no-ops and the voice gains alone decide the
    level."""
    if not spec:
        return []
    def obj(d: dict) -> str:
        return "{ " + ", ".join(f"{k}: {v:g}" for k, v in d.items()) + " }"
    out = ["// master bus (measured from the original; no-ops on engines without setMasterCompressor/setMasterLimiter)"]
    if spec.get("compressor"):
        out.append(f"typeof setMasterCompressor === 'function' && setMasterCompressor({obj(spec['compressor'])})")
    if spec.get("ceiling") is not None:
        out.append(f"typeof setMasterLimiter === 'function' && setMasterLimiter({obj({'ceiling': spec['ceiling'], **({'release': spec['limiter_release']} if spec.get('limiter_release') else {})})})")
    return out + [""]


def compose_env_multipliers(voice_m: list[float] | None, master_m: list[float] | None,
                            clamp: list[float] | None) -> list[float] | None:
    """Per-bar multiplier for one voice: voice x master, bounded to ``clamp`` = ``[lo, hi]``.

    Either input may be None (only the other applies); both None -> None (no correction).
    """
    if voice_m and master_m:
        m = [a * b for a, b in zip(voice_m, master_m)]
    else:
        m = voice_m or master_m
    if not m:
        return None
    if clamp:
        lo, hi = clamp
        m = [min(hi, max(lo, x)) for x in m]
    return m


def apply_env_correction(env_bars: list[list[float]] | None,
                         mults: list[float] | None) -> list[list[float]] | None:
    """Scale each bar of a per-step envelope by that bar's multiplier.

    Hard-silence steps (0.0) stay 0.0 (the voice is absent where the original stem is). The
    envelopes are 0..1 (normalised to the stem's peak), so a step never exceeds the largest
    multiplier; bounding the multipliers themselves is ``compose_env_multipliers``' job.
    """
    if not env_bars or not mults:
        return env_bars
    return [[0.0 if v == 0.0 else v * mults[b] for v in steps]
            for b, steps in enumerate(env_bars)]


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


def measured_drum_pitch(args) -> dict | None:
    """{class: {hit time: rate ratio}} for the kick, measured on the drums stem against the extracted one-shot the
    pattern plays (the kit entry of the pack's instruments/samples.json, else drums/bd.wav); None when either is missing."""
    import soundfile as sf
    stem = args.stems_dir / "drums.wav"
    shot = None
    sj = args.pack_dir / "instruments" / "samples.json"
    if sj.exists():
        ent = json.loads(sj.read_text()).get("bd")
        if ent:
            shot = args.pack_dir / "instruments" / (ent[0] if isinstance(ent, list) else ent)
    if shot is None or not shot.exists():
        shot = args.pack_dir / "drums" / "bd.wav"
    if not stem.exists() or not shot.exists() or not args.drums_json.exists():
        return None
    y, sr = sf.read(str(stem), always_2d=True)
    x, sr2 = sf.read(str(shot), always_2d=True)
    y, x = y.mean(1), x.mean(1)
    if sr2 != sr:
        import scipy.signal as sps
        x = sps.resample_poly(x, sr, sr2)
    hits = json.loads(args.drums_json.read_text()).get("hits", [])
    ratios = DP.hit_ratios(y, sr, hits, "bd", x)
    return {"bd": ratios} if ratios else None


def pitched_map(pack_dir: Path, base: str) -> dict:
    """Note-keyed pitched-instrument maps (trackbass/tracklead) + absolute _base."""
    sm = json.loads((pack_dir / "strudel.json").read_text())
    out = {"_base": base}
    for k, v in sm.items():
        if k in ("trackbass", "tracklead") and isinstance(v, dict) and v:
            out[k] = v
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
    ap.add_argument("--mode", choices=GENERATION_MODES, default="sample-instrument",
                    help="sample-instrument=the pack's pitched instruments via samples() [default]; "
                         "synth=no samples() at all — bass/lead sounds + drum machine from the "
                         "sound_selector genre palette, drums on the --drum-mode bank path")
    ap.add_argument("--bass-sound", default=None,
                    help="trackbass (custom, sample-instrument default) or a library sound "
                         "(synth default: the genre palette's bass pick)")
    ap.add_argument("--lead-sound", default=None,
                    help="tracklead (custom, sample-instrument default) or a library sound "
                         "(synth default: the genre palette's lead pick)")
    ap.add_argument("--lead-synth", type=Path, default=None,
                    help="synth mode: JSON lead spec fitted by fit_lead_synth.py (lead_synth.spec_from) — the lead and "
                         "harmony voices play a plain oscillator with the fitted unison/detune/ADSR/filter-envelope "
                         "controls instead of the palette preset; its lpf replaces --lead-lpf, its gain scales the lead")
    ap.add_argument("--drum-mode", choices=["extracted", "bank", "realstem"], default="extracted",
                    help="extracted=DRUMS GENERATOR — the track's OWN extracted kick/snare/hat one-shots "
                         "(bd/sd/hh/oh) triggered by the detected, editable rhythm pattern (real sounds + "
                         "real notes); bank=same groove on a library drum machine; realstem=replay the whole "
                         "drum stem (not a generator — identical audio but un-editable)")
    ap.add_argument("--drum-pitch", choices=["off", "auto"], default="auto",
                    help="auto: a tuned kick (808) is written with the pitch each stem hit sounds, as a per-hit .speed() "
                         "pattern relative to the extracted one-shot (drum_pitch.py); hits whose pitch cannot be read play it as is")
    ap.add_argument("--drum-velocity", choices=["off", "bar", "global"], default="bar",
                    help="per-hit drum velocity measured from the drums stem (detect_drums_bands 'strength'): "
                         "bar=hit strength / its class median in the same bar, global=/ the piece median, off=none")
    ap.add_argument("--drum-stem-gain", type=float, default=1.0,
                    help="gain for the realstem drum channel (balances the real drums in the mix)")
    ap.add_argument("--drum-lpf", type=int, default=0,
                    help="low-pass the drum bus toward the original drum stem brightness (0=off)")
    # per-voice gain calibration (multiplies the voice's base gain) — set from a render's
    # measured stem RMS vs the original stem RMS so the inter-stem balance matches the original.
    ap.add_argument("--cal-bass", type=float, default=1.0)
    ap.add_argument("--cal-lead", type=float, default=1.0)
    ap.add_argument("--cal-drums", type=float, default=1.0)
    ap.add_argument("--cal-chord", type=float, default=1.0,
                    help="multiplier on the harmony voice's derived gain (calibrator lever; 1.0 = as derived)")
    ap.add_argument("--no-harmony", action="store_true",
                    help="omit the harmony (chord) voice: the lead keeps one note per step, as before")
    ap.add_argument("--master-gain", type=float, default=1.0,
                    help="master gain on the whole stack to back off the recorder's 0 dBFS limiter (~0.6)")
    ap.add_argument("--master-bus", type=Path, default=None,
                    help="JSON from mix_dynamics.py derive-bus/next-bus: emit the measured master-bus lines (bus compressor + "
                         "brickwall limiter, each guarded so engines without them ignore the line). Voice gains are NOT touched")
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
    ap.add_argument("--lead-hpf", type=int, default=0,
                    help="lead high-pass (Hz); 0 = derive: sample-instrument modes use the lowest fundamental the lead "
                         "and chords play (sample_lead_hpf); synth mode starts at SYNTH_LEAD_HPF_START and the "
                         "calibrator moves it")
    ap.add_argument("--partial-reference", type=Path, default=None,
                    help="sound register table (default data/partial_reference.json, written by "
                         "probe_sound_register.py); empty/missing -> warn and apply no octave shift")
    ap.add_argument("--lead-octave", type=PR.octave_arg, default="auto",
                    help="octaves (int) to move the lead line up, or 'auto' (default): synth mode only, "
                         "derived by partial_register.py from the original stem's dominant-partial register "
                         "minus the palette sound's measured reference; sample modes always 0")
    ap.add_argument("--sub-gain", type=float, default=0.7,
                    help="sub-sine layer gain (raise to fill the 20-60Hz sub_bass band)")
    ap.add_argument("--bass-cut", type=int, default=0,
                    help="cut-group id for the bass (e.g. 1) → MONOPHONIC bass: each note chokes the "
                         "previous so a flat sustained sample gives a continuous line with no "
                         "polyphonic pile-up. Pair with a long, flat bass sample.")
    ap.add_argument("--sub-octave", type=int, default=0,
                    help="octaves to drop the sub-sine so its fundamental lands in sub_bass (1 = -12 semis)")
    ap.add_argument("--sub-up", type=int, choices=[0, 1], default=0,
                    help="1 = play the sub-sine an octave ABOVE the written bass note (a calibrator lever: it moves the "
                         "layer's energy from the sub_bass band into the bass band; its low-pass moves up an octave too). "
                         "Refused (0, warning) when the median written note an octave up would leave the bass band")
    ap.add_argument("--dj-flow", action="store_true", default=True,
                    help="emit mutable per-voice $: channels + a section/arrangement map (normal "
                         "live-play DJ layout: comment a channel to mute, solo by soloing, slice to a drop)")
    ap.add_argument("--no-dj-flow", dest="dj_flow", action="store_false")
    ap.add_argument("--drum-floor", type=float, default=0.2,
                    help="drum envelope floor (0=deepest dynamics→best shape corr; higher=louder average)")
    ap.add_argument("--hat-gain", type=float, default=0.0,
                    help="steady TR808 hh*8 layer inside the drum bus (smooths envelope for shape + lifts centroid)")
    ap.add_argument("--bass-f0", type=Path, default=None, metavar="PATH",
                    help="bass_register.py f0 track of bass.wav (default <stems-dir>/.pipeline/bass_f0.json, written by "
                         "editable-pipeline.sh). Gives the bass window + per-note octave; absent/stale -> loud "
                         "fallback to the legacy typed windows")
    ap.add_argument("--bass-shape", choices=["stem", "off"], default="stem",
                    help="stem=BASS note shape measured from bass.wav (bass_shape.py): held notes (c2@3) instead of 1-step "
                         "hits, glide() where the stem's f0 slides, sample loop points so held notes sustain; "
                         "off=the old one-step note per transcribed trigger")
    ap.add_argument("--bass-duck", choices=["auto", "off"], default="auto",
                    help="auto=kick-triggered sidechain on the bass orbit when the original bass stem measurably dips "
                         "after kick onsets (duckdepth/duckonset/duckattack measured); no dip -> none emitted")
    ap.add_argument("--rhythm", choices=["onset", "grid"], default="grid",
                    help="grid=quantise MIDI (better pitch/timbre fidelity); onset=stem onsets drive timing "
                         "(sparser but librosa bass onset detection is unreliable -> worse mfcc)")
    ap.add_argument("--melodic-shape", choices=["stem", "off"], default="stem",
                    help="stem=LEAD/CHORDS retrigger only where the melodic stem has a strong onset and hold the note "
                         "(c4@3) until the next onset / the stem going quiet / the bar end (melodic_shape.py), sample "
                         "loop points so a held note sustains; off=the 1-step retrigger stream as transcribed")
    ap.add_argument("--env-correction", type=Path, default=None, metavar="PATH",
                    help="JSON from calibrate_dynamic.py --env-correction-out: per-BAR multipliers "
                         "(voices.bass/lead/master) applied to the per-bar gain envelopes, so "
                         "section-level bass/lead balance follows the measured render-vs-original ratios")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # --- generation mode: which instruments voice the (identical) note material ------------
    # synth: every sound is a library sound DERIVED from the genre palette (sound_selector), never
    # a pack sample and never a name typed here (CLAUDE.md ZERO HARDCODING). An explicit
    # --bass-sound/--lead-sound is honoured only if it is a known library sound.
    synth = args.mode == "synth"

    def _library_sound(explicit: str | None, role: str) -> str:
        pick = pick_palette_sound(args.genre, role)
        if explicit is None or explicit == pick:
            return pick
        if explicit in VALID_SOUNDS:
            return explicit
        print(f"WARNING mode=synth: --{role}-sound {explicit!r} is not a library sound (a pack sample?) — "
              f"using the {args.genre} palette pick {pick!r}", file=sys.stderr)
        return pick

    if synth:
        bass_sound = _library_sound(args.bass_sound, "bass")
        lead_sound = _library_sound(args.lead_sound, "lead")
        if args.drum_mode != "bank":
            print(f"WARNING mode=synth: --drum-mode {args.drum_mode} uses the pack's drum samples — "
                  "using the library drum machine path (--drum-mode bank)", file=sys.stderr)
            args.drum_mode = "bank"
    else:
        bass_sound = args.bass_sound or "trackbass"
        lead_sound = args.lead_sound or "tracklead"

    # the lead's sound + low-pass as one chain: the palette preset, or the fitted synth controls
    lead_voice = f'.s("{lead_sound}").lpf({args.lead_lpf})'
    lead_synth_gain = 1.0
    if args.lead_synth is not None:
        if not synth:
            ap.error("--lead-synth needs --mode synth")
        import lead_synth as LS
        spec = json.loads(args.lead_synth.read_text())
        lead_voice = LS.lead_chain(spec)
        lead_synth_gain = float(spec.get("gain", 1.0))

    nbars = args.num_bars
    cps = args.bpm / 60 / 4
    dur = (60.0 / args.bpm * 4) * nbars
    base = args.base_url.rstrip("/") + "/"

    # fold to musical ranges: bass E1-C3, lead C3-G5 (kills octave-error screech)
    # RHYTHM from the stem's real onsets, PITCH from the transcription (onset-driven) — the
    # transcription over-produces notes ~5x, so plain grid-quantising makes a too-busy line
    # that loses the original's groove. Fall back to midi_to_bars if onset detection fails.
    def _octave(arg, stem: Path, midi: Path, lo: int, hi: int, sound: str):
        """(octaves, measured original dominant-partial log2 or None, decision dict or None).
        Sample modes: always (0, None, None)."""
        if str(arg) != "auto":
            return int(arg), None, {"shift": int(arg), "reason": "explicit flag"}
        if not synth:
            return 0, None, None
        try:
            # cached beside the pack: calibration iterations 2..N reuse an unchanged stem's analysis
            stat, pw, _hit = PR.cached_register(args.pack_dir / "register.json", stem.stem, stem, midi, lo, hi)
        except Exception as exc:  # noqa: BLE001 - unmeasurable stem/midi -> no evidence, no shift
            print(f"WARNING octave: could not measure {stem.name} ({exc}); no shift", file=sys.stderr)
            return 0, None, {"shift": 0, "reason": f"unmeasurable: {exc}"}
        table = PR.load_table(args.partial_reference) if args.partial_reference else PR.load_table()
        dec = PR.octave_decision(stat, sound, table, pw)
        return dec["shift"], stat, dec

    # --- registers MEASURED from the stems (bass_register.py); legacy typed windows only as a loud fallback
    bass_f0_path = args.bass_f0 or (args.stems_dir / ".pipeline" / "bass_f0.json")
    bass_reg = None
    bass_win = LEGACY_BASS_WINDOW
    try:
        data = json.loads(bass_f0_path.read_text())
        want = BR.stem_hash(args.stems_dir / "bass.wav")
        if data.get("key", {}).get("sha1") != want:
            raise ValueError("cached f0 is for a different bass.wav (stale)")
        bass_reg = BR.Register(data)
        bass_win = bass_reg.window()
    except Exception as exc:  # noqa: BLE001 - any failure -> the old behaviour, never silently
        print(f"WARNING bass register: {bass_f0_path} unusable ({exc}); FALLING BACK to the legacy typed bass "
              f"window {LEGACY_BASS_WINDOW} (octave errors expected). Run: bass_register.py measure --stem "
              f"{args.stems_dir / 'bass.wav'} --out {bass_f0_path}", file=sys.stderr)
        bass_reg = None
        bass_win = LEGACY_BASS_WINDOW
    try:
        lead_win = BR.melodic_window(args.lead_midi, args.stems_dir / "melodic.wav",
                                     bass_win[1] if bass_reg is not None else LEGACY_BASS_WINDOW[1])
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING lead register: could not derive from the melodic stem ({exc}); using the legacy "
              f"window {LEGACY_LEAD_WINDOW}", file=sys.stderr)
        lead_win = LEGACY_LEAD_WINDOW
    blo, bhi = bass_win

    lead_oct, lead_partial, lead_dec = _octave(args.lead_octave, args.stems_dir / "melodic.wav", args.lead_midi,
                                     lead_win[0], lead_win[1], lead_sound)
    llo, lhi = lead_win[0] + 12 * lead_oct, lead_win[1] + 12 * lead_oct
    if args.rhythm == "onset":
        bass_bars = onset_driven_bars(args.stems_dir / "bass.wav", args.bass_midi, args.bpm, nbars,
                                      pick="low", lo=blo, hi=bhi, register=bass_reg) \
            or midi_to_bars(args.bass_midi, args.bpm, nbars, pick="low", lo=blo, hi=bhi, register=bass_reg)
        lead_bars = onset_driven_bars(args.stems_dir / "melodic.wav", args.lead_midi, args.bpm, nbars,
                                      pick="loud", lo=llo, hi=lhi) \
            or midi_to_bars(args.lead_midi, args.bpm, nbars, pick="loud", lo=llo, hi=lhi)
    else:
        bass_bars = midi_to_bars(args.bass_midi, args.bpm, nbars, pick="low", lo=blo, hi=bhi, register=bass_reg)
        lead_bars = midi_to_bars(args.lead_midi, args.bpm, nbars, pick="loud", lo=llo, hi=lhi)
    if not bass_bars and not lead_bars:
        print("ERROR: no note material extracted", file=sys.stderr)
        return 1
    # HARMONY: the notes under the lead's top line (editable chord bars), derived from the same transcription.
    chord_bars, harmony_info = None, None
    if lead_bars and not args.no_harmony and args.lead_midi.exists():
        try:
            hsteps, harmony_info = harmony_notes(args.lead_midi, args.bpm, nbars, lo=llo, hi=lhi)
            chord_bars = harmony_bars(hsteps, nbars)
        except Exception as exc:  # noqa: BLE001 - unreadable midi must not lose the rest of the piece
            print(f"WARNING harmony: could not derive ({exc}); no harmony voice", file=sys.stderr)

    # MELODIC NOTE SHAPE (melodic_shape.py): the written lead retriggers every 16th; the stem plays a note and lets it ring.
    lead_loop_table, melodic_info, lead_loop_stats = {}, None, None
    lead_legato, chord_legato = None, None
    if args.melodic_shape == "stem" and lead_bars:
        try:
            mev = MS.load_evidence(args.stems_dir / "melodic.wav")
            lead_bars, chord_bars, melodic_info = MS.shape_melodic(lead_bars, chord_bars, args.bpm, mev)
            lead_legato = ((melodic_info or {}).get("lead") or {}).pop("legato_bars", None)
            chord_legato = ((melodic_info or {}).get("chords") or {}).pop("legato_bars", None)
        except Exception as exc:  # noqa: BLE001 - an unmeasurable stem must not lose the piece
            print(f"WARNING melodic-shape: cannot measure melodic.wav ({exc}); lead stays a 1-step stream",
                  file=sys.stderr)
        if not synth and melodic_info and melodic_info.get("lead"):
            try:
                lead_loop_table, lead_loop_stats = BS.sample_loops(
                    args.pack_dir / "instruments" / "samples.json", lead_sound,
                    _chord_bars_pitches(lead_bars) + _chord_bars_pitches(chord_bars)
                    + _chord_bars_pitches(lead_legato) + _chord_bars_pitches(chord_legato))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING melodic-shape: no loop points ({exc}); held lead notes play their grain once",
                      file=sys.stderr)

    if args.lead_hpf <= 0:
        args.lead_hpf = (SYNTH_LEAD_HPF_START if synth else
                         sample_lead_hpf(lead_bars, chord_bars, lead_legato, chord_legato) or SYNTH_LEAD_HPF_START)

    # BASS NOTE SHAPE (bass_shape.py): the original bass is one continuous tone, the transcription a row of hits.
    bass_glide_bars, bass_loop_table, bass_duck, shape_info, loop_stats, duck_diag = None, {}, None, None, None, {}
    bass_legato_bars, bass_attack_bars, bass_shape_notes = None, None, None
    if args.bass_shape == "stem" and bass_bars:
        try:
            ev = BS.StemEvidence.load(args.stems_dir / "bass.wav")
            notes, shape_info = BS.shape_bars(bass_bars, args.bpm, ev, bass_reg)
            bass_bars = BS.format_bars(notes, nbars, 16, midi_name)
            bass_shape_notes = notes
            if any(n.glide > 0 for n in notes):
                bass_glide_bars = BS.format_bars(notes, nbars, 16, midi_name, glide=True)
        except Exception as exc:  # noqa: BLE001 - an unmeasurable stem must not lose the piece
            print(f"WARNING bass-shape: cannot measure bass.wav ({exc}); bass notes stay 1-step", file=sys.stderr)
        if not synth:
            sj = args.pack_dir / "instruments" / "samples.json"
            try:
                bass_loop_table, loop_stats = BS.sample_loops(sj, bass_sound, _bars_pitches(bass_bars))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING bass-shape: no loop points ({exc}); held sample notes play their grain once",
                      file=sys.stderr)
    # A bass note the stem plays WITHOUT a new attack (a pitch change inside its continuing tone) must not re-strike the
    # sample: a 1-step note then plays only the grain's start - transient, pitch sweep, one-period fade-in - while the
    # stem is on its steady tone (Regime: chroma 0.55 on those frames vs 0.80 for a held-tone synth). Those notes go
    # to their own array, entered at the grain's loop (steady part) with a click-guard attack of one period of the note.
    if bass_shape_notes and bass_loop_table and any(not n.attack for n in bass_shape_notes):
        bass_legato_bars = BS.format_bars([n for n in bass_shape_notes if not n.attack], nbars, 16, midi_name)
        bass_attack_bars = BS.format_bars([n for n in bass_shape_notes if n.attack], nbars, 16, midi_name)
    if args.bass_duck == "auto" and bass_bars and args.drums_json and args.drums_json.exists() \
            and args.drum_mode != "realstem":
        try:
            kicks = sorted({h["time"] for h in json.loads(args.drums_json.read_text()).get("hits", [])
                            if h.get("type") == "bd"})
            rms_f, hop_f = BS.stem_rms_fine(args.stems_dir / "bass.wav")
            bass_duck = BS.measure_duck(rms_f, hop_f, kicks, diag=duck_diag)
            if bass_duck and kicks and kicks[0] < BS.DUCK_MIN_T:
                print(f"WARNING bass-duck: a kick at {kicks[0]:.3f}s precedes the engine's {BS.DUCK_MIN_T}s duck "
                      f"scheduling bound; no ducking emitted", file=sys.stderr)
                bass_duck = None
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING bass-duck: cannot measure ({exc}); no ducking", file=sys.stderr)

    # effects aligned to the original stems
    ref_rms = max((H.stem_features(args.stems_dir / f"{s}.wav", dur) or {"rms": 1e-6})["rms"]
                  for s in ("drums", "bass", "melodic"))
    bass_fx = H.effects_for(H.stem_features(args.stems_dir / "bass.wav", dur), ref_rms,
                            kind="bass", library=(bass_sound != "trackbass"))
    lead_fx = H.effects_for(H.stem_features(args.stems_dir / "melodic.wav", dur), ref_rms,
                            kind="melodic", library=(lead_sound != "tracklead"))
    drums_fx = H.effects_for(H.stem_features(args.stems_dir / "drums.wav", dur), ref_rms,
                             kind="drums", library=True)

    # realstem drums: play the track's OWN drum stem as a Strudel sample so the rendered drums
    # are the real drums (identical sound + spectrogram), via legit Strudel samples() — not a
    # library drum machine. The drum stem already carries its own dynamics, so no env is applied.
    drum_realstem = args.drum_mode == "realstem" and (args.pack_dir / "drumsfull.wav").exists()

    # If a pre-built manifest URL is given (e.g. trained instruments), reference it verbatim and
    # don't write a pack manifest. Otherwise build one from the pack's pitched maps / loops.
    # synth mode: no manifest at all — every voice is a library sound (an external --samples-url
    # is ignored on purpose, loudly).
    samples_url = None if synth else args.samples_url
    if synth and args.samples_url:
        print("WARNING mode=synth: --samples-url ignored (synth emits no samples() load)", file=sys.stderr)
    needs_samples = (not synth) and (bool(samples_url) or (
        bass_sound == "trackbass" or lead_sound == "tracklead"
        or drum_realstem))
    if needs_samples and not samples_url:
        smap = pitched_map(args.pack_dir, base)
        if drum_realstem:
            smap["drumsfull"] = ["drumsfull.wav"]
        (args.pack_dir / "samples.json").write_text(json.dumps(smap, indent=2) + "\n")
        samples_url = f"{base}samples.json"

    # When an EXTERNAL manifest is referenced (e.g. trained instruments/samples.json) it carries only the
    # pitched instruments + drum one-shots, not the realstem-drum loop, which lives in the PACK manifest.
    # Strudel merges successive samples() calls, so a SECOND samples() loads the co-hosted pack manifest.
    extra_samples_url = None
    if samples_url and args.samples_url and drum_realstem:
        extra_samples_url = f"{base}samples.json"

    # slice()/run()/slow() of a hosted LOOP must use the bar-span of THAT SAMPLE, not the
    # arrangement length: x = round(sample_seconds * cps). A drum loop that is e.g. 16 bars
    # long (28s @136bpm) sliced by nbars=78 plays at the wrong rate (mangled). Measure the sample
    # and emit the count as a derived const so it tracks cps and the real loop length.
    def _sample_bars(p: Path) -> int:
        try:
            secs = float(librosa.get_duration(path=str(p)))
        except Exception:
            return nbars
        return max(1, round(secs * cps))

    drum_bars = _sample_bars(args.pack_dir / "drumsfull.wav") if drum_realstem else nbars

    # Effective mode: `synth` when requested; a sample-instrument request that ends up needing no
    # pack samples (all library sounds, bank drums) is honestly labelled synth too.
    generation_mode = "synth" if synth else ("sample-instrument" if needs_samples else "synth")
    L = [
        "// MIDI-grep dynamic — transcribed note material on pitched instruments",
        f"// genre={args.genre}",
        f"// generation_mode: {generation_mode}",
        "// editability: checked-by editability_check.py",
        *([f"// env_correction: {args.env_correction.name}"] if args.env_correction else []),
        f"// BPM: {fmt_bpm(args.bpm)}",
        f"// Key: {args.key}",
        f"// Bars: {nbars}  Duration: {dur:.0f}s",
        f"const cps = {cps:.6f}   // {fmt_bpm(args.bpm)} BPM, 4 beats/bar",
        "setcps(cps)",
    ]
    if bass_glide_bars:
        # portamento exists only in Strudel builds with glide(); everywhere else it is accepted and ignored
        L.append("if (typeof Pattern.prototype.glide !== 'function') { register('glide', (t, pat) => pat, false) }")
    if needs_samples:
        L.append(f'await samples("{samples_url}")')
    if extra_samples_url:
        # second manifest (pack) for loops missing from an external samples_url (drumsfull).
        L.append(f'await samples("{extra_samples_url}")')
    if drum_realstem:
        L.append(f"const drumBars = {drum_bars}   // round(drum_sample_secs * cps)")
    L.append("")

    # --- Spectral alignment (data-driven from measured band diffs) ---
    # rendered ran HOT in bass/low-mid and (especially) high-mid, and SHORT on
    # high/top air. With only gain/lpf/hpf + layering available in Strudel:
    bass_fx["gain"] = round(bass_fx["gain"] * args.bass_mult * args.cal_bass, 2)   # tame bass/low-mid boom
    bass_lpf_analysis = int(bass_fx.get("lpf") or 0)   # analysis-derived cutoff, before any override
    if args.bass_lpf:
        bass_fx["lpf"] = args.bass_lpf   # cut trackbass upper-bass harmonics → energy shifts toward sub
    # NOTE: raising lead gain adds high-mid faster than lpf removes it (lead lives
    # in 2-3.5kHz), and a lower lpf clips needed hi/top air — 0.6x + lpf 5000 is
    # the measured optimum (overall 76.5%); going higher/lower both regress.
    lead_g = round(lead_fx["gain"] * 0.60 * args.cal_lead, 2)            # restore some mid body
    drums_fx["gain"] = round(drums_fx["gain"] * args.cal_drums, 2)       # inter-stem balance
    sub_g = round(args.sub_gain * args.cal_bass, 2)
    # The sub layer may drop octaves only while its LOWEST written note stays at/above the project's sub-bass
    # band floor (compare_audio.FREQ_BANDS sub_bass = 20 Hz, BR.SUB_FLOOR_HZ): below it nothing is audible or
    # measured, so a layer there is wasted (a C#1 sub dropped an octave = 17 Hz).
    sub_octave = args.sub_octave
    if sub_octave and bass_bars:
        lowest = min(_bars_pitches(bass_bars))
        floor_midi = 69.0 + 12.0 * np.log2(BR.SUB_FLOOR_HZ / 440.0)
        while sub_octave > 0 and lowest - 12 * sub_octave < floor_midi:
            sub_octave -= 1
        if sub_octave != args.sub_octave:
            print(f"WARNING --sub-octave {args.sub_octave} would put the sub layer under {BR.SUB_FLOOR_HZ:g} Hz "
                  f"(lowest written MIDI {lowest}); using {sub_octave}", file=sys.stderr)
    # sub-up: the whole sine layer (note AND its low-pass) moves one octave up, so it is the same layer shifted into
    # the bass band. Refused only when the median written note an octave up would leave the bass band (its top edge,
    # BR.BASS_CEIL_HZ = compare_audio.FREQ_BANDS bass 250 Hz): then there is no band left for it to land in.
    sub_up = args.sub_up
    if sub_up and bass_bars:
        med = float(np.median(_bars_pitches(bass_bars)))
        up_hz = 440.0 * 2 ** ((med + 12 - sub_octave * 12 - 69) / 12.0)
        if up_hz > BR.BASS_CEIL_HZ:
            print(f"WARNING --sub-up refused: the median bass note an octave up ({up_hz:.0f} Hz) is above the bass band "
                  f"({BR.BASS_CEIL_HZ:g} Hz)", file=sys.stderr)
            sub_up = 0
    sub_lpf = SUB_LPF_HZ * 2 ** sub_up
    sub_shift = 12 * sub_up - 12 * sub_octave
    sub_transpose = f".add(note({sub_shift}))" if sub_shift else ""
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

    # Measured per-section correction (calibrate_dynamic.py --env-correction-out): scales each
    # voice's per-bar envelope BEFORE gain_pattern folds base*env into the single .gain(). The
    # bass line and its sub layer share bass_env; "master" scales every voice.
    if args.env_correction:
        try:
            corr = load_env_correction(args.env_correction, nbars)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        master_m = corr.get("master")
        clamp = corr.get(ENV_CORRECTION_CLAMP)

        def _corrected(env, voice):
            m = compose_env_multipliers(corr.get(voice), master_m, clamp)
            if env is None and m:
                # missing/silent stem: a flat all-ones envelope so the per-bar curve still shapes it
                env = [[1.0] * max(1, int(args.env_steps)) for _ in range(nbars)]
            return apply_env_correction(env, m)

        bass_env = _corrected(bass_env, "bass")
        lead_env = _corrected(lead_env, "lead")
        drum_env = _corrected(drum_env, "drums")

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
        L += [js_array("bass", bass_attack_bars if bass_legato_bars else bass_bars), ""]
        if bass_legato_bars:
            L += [js_array("bassLegato", bass_legato_bars), ""]
        # bass + sub-sine reinforcement kept together as ONE channel so muting bass mutes its sub.
        # Each line carries ONE baked gain = base * master * per-bar envelope.
        # cut(N): a cut group makes the bass MONOPHONIC — each note chokes the previous, so a
        # flat sustained sample plays at full level until the next note cuts it → a continuous,
        # non-piled-up bass line (matches a real mono synth; without it, long samples overlap
        # polyphonically and build into a loud, muddy cluster on a fast-retriggered pattern).
        bass_cut = f'.cut({args.bass_cut})' if args.bass_cut else ''
        # note shape (bass_shape.py): glide seconds per note (parallel to `bass`), sample loop table, own orbit
        if bass_glide_bars:
            L += [js_array("bassGlide", bass_glide_bars), ""]
        glide = '.glide(cat(...bassGlide))' if bass_glide_bars else ''
        orbit = f'.orbit({BASS_ORBIT})' if bass_duck else ''
        main_row = f'note(cat(...bass)).s("{bass_sound}")'
        if bass_loop_table:
            L += [f"// sample loop [begin, end] (fractions of the grain) per written pitch (midi), measured on the grains:",
                  f"// held notes sustain inside it. Delete a pitch to play its grain once.",
                  "const bassLoop = {" + ", ".join(f"{p}: [{b}, {e}]" for p, (b, e) in sorted(bass_loop_table.items())) + "}",
                  "const withLoop = p => p.withValue(v => { const l = bassLoop[Math.round(noteToMidi(v.note))]; "
                  "return l ? { ...v, loop: 1, loopBegin: l[0], loopEnd: l[1] } : v })", ""]
            if bass_legato_bars:
                L += ["// bassLegato: pitch changes inside a continuing tone - the sample entered at its loop (no new attack).",
                      "// Move a note to `bass` to strike it instead. The click guard is one period of the note.",
                      "const bassLegatoEntry = p => p.withValue(v => { const m = noteToMidi(v.note); const l = bassLoop[Math.round(m)]; "
                      "const a = 1 / (440 * 2 ** ((m - 69) / 12)); return l ? { ...v, begin: l[0], attack: a } : { ...v, attack: a } })", ""]
                main_row = f'stack(note(cat(...bass)), bassLegatoEntry(note(cat(...bassLegato)))).s("{bass_sound}")'
            main_row = f'withLoop({main_row})'
        sub_notes = "stack(note(cat(...bass)), note(cat(...bassLegato)))" if bass_legato_bars else "note(cat(...bass))"
        rows = [f'  {main_row}{fx_no_gain(bass_fx)}{bass_cut}{glide}{orbit}{gain_pattern(bass_env, bass_fx["gain"] * M)}',
                f'  {sub_notes}{sub_transpose}.s("sine").lpf({sub_lpf}){glide}{orbit}{gain_pattern(bass_env, sub_g * M)}']
        if bass_duck:
            # the duck target orbit must exist before the first kick of any render window: a silent keeper voice
            rows.append(f'  note("c1").s("sine").orbit({BASS_ORBIT}).gain(0)')
        bass_code = "stack(\n" + ",\n".join(rows) + "\n)"
        channels.append(("BASS", "low end — root/sub. edit notes in `bass`", bass_code))
    lead_row, chord_row = "note(cat(...lead))", "note(cat(...chords))"
    if lead_legato:
        lead_row = f"stack({lead_row}, noLoopAttack(note(cat(...leadLegato))))"
    if chord_legato:
        chord_row = f"stack({chord_row}, noLoopAttack(note(cat(...chordsLegato))))"
    if lead_bars and (lead_loop_table or lead_legato or chord_legato):
        L += ["// sample loop [begin, end] (fractions of the grain) per written pitch (midi), measured on the grains:",
              "// held lead/chord notes sustain inside it. Delete a pitch to play its grain once.",
              "const leadLoop = {" + ", ".join(f"{p}: [{b}, {e}]" for p, (b, e) in sorted(lead_loop_table.items())) + "}",
              "const withLeadLoop = p => p.withValue(v => { const l = leadLoop[Math.round(noteToMidi(v.note))]; "
              "return l ? { ...v, loop: 1, loopBegin: l[0], loopEnd: l[1] } : v })"]
        if lead_legato or chord_legato:
            L += ["// a pitch change inside a held note (leadLegato / chordsLegato): the same instrument entered at the",
                  "// start of its loop, so it has no new attack. Delete a pitch from leadLoop to retrigger it instead.",
                  f"const legatoAttack = {round(60.0 / args.bpm / 4, 4)}  // s: one grid step to fade the entry in",
                  "const noLoopAttack = p => p.withValue(v => { const l = leadLoop[Math.round(noteToMidi(v.note))]; "
                  "return l ? { ...v, begin: l[0], attack: legatoAttack } : { ...v, attack: legatoAttack } })"]
        L += [""]
        lead_row, chord_row = f"withLeadLoop({lead_row})", f"withLeadLoop({chord_row})"
    if lead_bars:
        L += [js_array("lead", lead_bars), ""]
        if lead_legato:
            L += [js_array("leadLegato", lead_legato), ""]
        # lpf tames high-mid top while keeping mid body; hpf clears low bleed.
        # clip(1.3) sustains notes slightly past their step so the monophonic lead's envelope
        # is smoother (closer to the original melodic's sustained contour, lifting shape corr).
        lead_code = (f'{lead_row}{lead_voice}'
                     f'.hpf({args.lead_hpf}).clip(1.3).room({lead_fx["room"]})'
                     f'{gain_pattern(lead_env, lead_g * lead_synth_gain * M)}')
        channels.append(("LEAD", "melody/harmony — edit notes in `lead`", lead_code))
    if chord_bars:
        L += [js_array("chords", chord_bars), ""]
        if chord_legato:
            L += [js_array("chordsLegato", chord_legato), ""]
        # same instrument, filters, room and loudness contour as the lead; the base gain is the lead's scaled by the
        # transcribed harmony-vs-melody amplitude ratio (harmony_notes), then the calibrator's cal-chord lever.
        chord_g = round(lead_g * harmony_info["gain_ratio"] * args.cal_chord, 4)
        chord_code = (f'{chord_row}{lead_voice}'
                      f'.hpf({args.lead_hpf}).clip(1.3).room({lead_fx["room"]})'
                      f'{gain_pattern(lead_env, chord_g * lead_synth_gain * M)}')
        channels.append(("CHORDS", "harmony under the lead — edit chords in `chords`", chord_code))

    if drum_realstem:
        # The real drum stem, played gaplessly via Strudel slice playback → rendered drums are
        # the ORIGINAL drums (identical sound & spectrogram). Carries its own dynamics (no env).
        drum_code = ('s("drumsfull").slice(drumBars, run(drumBars)).slow(drumBars).clip(1)'
                     f'.gain({round(args.drum_stem_gain * M, 3)})')
        channels.append(("DRUMS", "REAL drum stem (your track's drums, 100% — via Strudel samples)", drum_code))
        detected = True  # for the summary flag
    else:
        pitch = measured_drum_pitch(args) if args.drums_json and args.drum_mode == "extracted" and args.drum_pitch == "auto" else None
        detected = H.build_detected_drums(args.drums_json, args.bpm, nbars, velocity=None if args.drum_velocity == "off" else args.drum_velocity, pitch=pitch) if args.drums_json else None
        if detected:
            drum_lpf = f'.lpf({args.drum_lpf})' if args.drum_lpf else ''
            drum_lpf += f'.hpf({args.drum_hpf})' if args.drum_hpf else ''
            dlines = list(detected)
            if bass_duck:
                # kick -> bass sidechain: every bd hit ducks the bass orbit by the measured depth/onset/attack
                duck = (f'.duckorbit({BASS_ORBIT}).duckdepth({bass_duck.depth}).duckonset({bass_duck.onset_s})'
                        f'.duckattack({bass_duck.attack_s})')
                dlines = [d + duck if re.search(r's\("[^"]*\bbd\b', d) else d for d in dlines]
            if args.hat_gain > 0:
                dlines.append(f'  s("hh*8").gain({args.hat_gain})')
            if args.drum_mode == "extracted":
                # DRUMS GENERATOR: the detected pattern triggers the track's OWN extracted
                # bd/sd/hh one-shots (mapped in samples.json) — no library bank. Real sounds,
                # editable pattern, reusable in any song.
                bank_suffix = ""
                label = "your extracted kit (bd/sd/hh) triggered by the detected pattern — edit it"
            else:
                # the genre palette's first-ranked drum machine (sound_selector) — one source of
                # truth for "the genre drum machine" across the LLM prompts and this generator
                bank = pick_palette_sound(args.genre, "drums")
                bank_suffix = f'.bank("{bank}")'
                label = f'detected groove on {bank} — edit the patterns'
            drum_code = ("stack(\n" + ",\n".join("  " + d for d in dlines)
                         + f'\n){bank_suffix}{drum_lpf}{fx_no_gain(drums_fx, drop_lpf=True)}'
                         + f'{gain_pattern(drum_env, drums_fx["gain"] * M)}')
            channels.append(("DRUMS", label, drum_code))

    # Each `$:` is a channel; a (label, comment, code, marker) tuple → one top-level pattern.
    # `marker` is a comment token the detector reads (only `// texture` today).
    def assemble(chs: list[tuple]) -> str:
        lines = list(L)
        bus_lines = master_bus_lines(json.loads(args.master_bus.read_text()) if args.master_bus else None)
        # counts of what THIS piece plays (bars / detected drum pattern are final by now)
        lines[2:2] = count_header_lines(args.genre, count_note_events(bass_bars, lead_bars, chord_bars),
                                        count_drum_hits(detected if isinstance(detected, list) else None))
        lines += bus_lines
        if args.dj_flow:
            # --- Performance header: arrangement map + how to play it live ---
            spark, sections = section_map(args.stems_dir / "instrumental.wav" if (args.stems_dir / "instrumental.wav").exists()
                                         else args.pack_dir / "originalfull.wav", nbars, args.bpm)
            lines.append("// ┌── LIVE-PLAY DJ FLOW " + "─" * 40)
            lines.append("// │ Each $: below is a channel — comment a line to MUTE it, or comment all-but-one to SOLO.")
            lines.append("// │ Voices already build & drop with the song (per-bar gain envelopes track the original).")
            if spark and sections:
                lines.append(f"// │ energy : {spark}")
                secstr = "  ".join(f"{n}[{a}-{b - 1}]" for n, a, b in sections)
                lines.append(f"// │ sections: {secstr}")
                drops = [f"{a}-{b}" for n, a, b in sections if n == "drop"]
                if drops:
                    lines.append(f"// │ jump to a drop, e.g.:  $: note(cat(...bass.slice({drops[0].split('-')[0]}, {drops[0].split('-')[1]}))).s(\"{bass_sound}\")")
            lines.append("// └" + "─" * 60)
            lines.append("")
            for label, comment, code, *marker in chs:
                tag = f"// {marker[0]} — " if marker else "// "
                lines.append(f"$: {code}  {tag}{label} — {comment}")
                lines.append("")
        else:
            members = []
            for i, (_, _, c, *marker) in enumerate(chs):
                m = "  " + c.replace("\n", "\n  ")
                if i < len(chs) - 1:
                    m += ","                       # separator BEFORE any trailing comment
                if marker:
                    m += f"  // {marker[0]}"
                members.append(m)
            lines += ["$: stack(", "\n".join(members), ")", ""]
        return "\n".join(lines) + "\n"

    code = assemble(channels)

    voices = channels  # for the summary count below

    out = args.out or (args.pack_dir / "output_dynamic.strudel")
    out.write_text(code)

    # Spec 003: the detector is the contract (it understands the `// texture` allowance); the
    # LLM-path validator (strudel_validation.validate_code) also runs — it flags every replay
    # sound, so its message is only surfaced when the detector fails too.
    try:
        res = check_editability(code)
        editability = "pass" if res.passed else "fail"
        violations = list(res.violations)
    except Exception as exc:  # pragma: no cover
        editability, violations = "error", [str(exc)]
    try:
        from strudel_validation import validate_code
        _, validation_error = validate_code(code, autocorrect=False)
    except Exception as exc:  # pragma: no cover - validator unavailable must not block output
        validation_error = f"validator unavailable: {exc}"
    if editability != "pass":
        for v in violations:
            print(f"WARNING editability: {v}", file=sys.stderr)
        if validation_error:
            print(f"WARNING editability: {validation_error}", file=sys.stderr)

    # live-player voice budget: peak simultaneous sources of THIS code (report only — stock web REPLs run with the
    # superdough cap disabled, so no setMaxPolyphony line is emitted)
    try:
        import polyphony
        poly = polyphony.estimate(code, args.pack_dir / "instruments" / "samples.json")
    except Exception as exc:  # pragma: no cover - an estimate must never block the generator
        poly = {"error": str(exc)}

    print(json.dumps({"out": str(out), "mode": args.mode, "polyphony": poly, "bass_bars": len(bass_bars or []),
                      "lead_bars": len(lead_bars or []), "drums": bool(detected),
                      "drum_mode": args.drum_mode,
                      "bass_sound": bass_sound, "lead_sound": lead_sound,
                      "bass_shape": shape_info, "melodic_shape": melodic_info, "lead_loops": lead_loop_stats, "bass_loops": loop_stats, "bass_duck_diag": duck_diag,
                      "bass_duck": ({k: getattr(bass_duck, k) for k in ("depth_db", "onset_s", "attack_s", "depth", "sigma", "kicks")}
                                    if bass_duck else None),
                      # effective bass low-pass + its bounds, fed back to calibrate_dynamic.py
                      "bass_lpf": int(bass_fx.get("lpf") or 0), "bass_lpf_analysis": bass_lpf_analysis,
                      "bass_note_max_hz": bars_max_hz(bass_bars),
                      "sub_up": int(sub_up),
                      "harmony": harmony_info, "chord_bars": len(chord_bars or []),
                      "lead_octave": lead_oct, "lead_octave_decision": lead_dec,
                      "lead_partial_log2": lead_partial,
                      "lead_hpf": int(args.lead_hpf), "lead_note_min_hz": bars_min_hz(lead_bars),
                      "generation_mode": generation_mode,
                      "env_correction": str(args.env_correction) if args.env_correction else None,
                      "uses_custom_samples": needs_samples,
                      "editability": editability, "editability_violations": violations,
                      "validation": (validation_error or "ok") if editability != "pass" else "ok"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
