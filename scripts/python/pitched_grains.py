#!/usr/bin/env python3
"""Note-aligned, pitch-verified grains for the sample-instrument pipeline (spec 003 §2.3b).

A stem-derived instrument must play the notes that are WRITTEN in the pattern, so each pitched sample must

  1. be one whole transcribed note (onset -> end, clipped at the next onset), so its length comes from the
     music rather than from a constant,
  2. carry a pitch label that was MEASURED on the sample itself, with an analysis window sized from the
     lowest frequency searched (a pyin frame that spans ~1 period of a 30 Hz note cannot label it), and
  3. be re-verified before it enters a manifest (:func:`verify_samples`).

Every length / threshold below is derived from the audio, the transcription, or a cited convention.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# YIN (de Cheveigne & Kawahara 2002) integrates over a window that must exceed the largest lag searched;
# pyin's win_length is frame_length // 2. PERIODS_IN_WINDOW periods of the lowest f0 in that window keeps
# the difference function's dip well defined (one period gives a single, noise-prone dip).
PERIODS_IN_WINDOW = 3
HOP_DIVISOR = 16            # pyin hop = frame_length / 16 (>= 93 % frame overlap -> smooth pitch track)
VOICED_PROB_MIN = 0.5       # pyin's own voicing probability: > 0.5 is the natural decision boundary (noise sits ~0.01)
VOICED_FRACTION_MIN = 0.40  # the trainer's pre-existing "trust only well-pitched grains" convention
STABLE_TOLERANCE_ST = 0.5   # a frame belongs to the stable part when within half a semitone of the median
MIN_STABLE_FRAMES = 2
RANGE_TRIM_PCT = 2.0        # robust register: ignore the outer 2 % of transcribed pitches (transcriber ghosts)
RANGE_MARGIN_ST = 1.0       # search one semitone beyond that register (a flat/sharp note must still be heard)
VERIFY_TOLERANCE_ST = 0.5   # a label is right when the measured pitch rounds to it (half a semitone)
VERIFY_OCTAVE_MARGIN_ST = 12.0  # models without a recorded register: search one octave either side of the labels

# Harmonic-sum salience (Klapuri 2006 style): partials 1..SALIENCE_HARMONICS weighted 1/h, each read as the
# spectral maximum within +-half a semitone of h*f0. Outlier fence = Tukey's Q3 + 1.5*IQR of the grain's own
# salience over the instrument register (the grain's spectrum decides what "salient" means, not a constant).
SALIENCE_HARMONICS = 6
FENCE_IQR = 1.5
SUBHARMONICS = (2, 3, 4)    # octave, octave+fifth, two octaves below
FINE_STEP_ST = 0.1          # sub-semitone grid used to locate the salience peak around a label
FFT_PAD = 8                 # zero-padding factor (interpolates the spectrum; adds no resolution)

NOTE_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def hz_to_midi(f: float) -> float:
    return 12.0 * math.log2(f / 440.0) + 69.0


def midi_to_note(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int


@dataclass(frozen=True)
class PitchMeasure:
    midi: float          # median pitch of the stable part, in (fractional) MIDI
    hz: float
    voiced_fraction: float
    stability_st: float  # std of the stable frames, semitones
    stable_frames: int

    @property
    def note(self) -> int:
        return int(round(self.midi))

    @property
    def quality(self) -> float:
        return self.voiced_fraction / (1.0 + self.stability_st)


def read_midi_notes(path: str | Path) -> list[Note]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [Note(float(n.start), float(n.end), int(n.pitch)) for inst in pm.instruments if not inst.is_drum
             for n in inst.notes]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def register_hz(pitches: list[int]) -> tuple[float, float]:
    """Search range from the transcription: trimmed register +- a semitone margin (no instrument constants)."""
    lo, hi = np.percentile(pitches, [RANGE_TRIM_PCT, 100.0 - RANGE_TRIM_PCT])
    return midi_to_hz(float(lo) - RANGE_MARGIN_ST), midi_to_hz(float(hi) + RANGE_MARGIN_ST)


def stub_note_length(notes: list[Note]) -> float:
    """25th percentile of the stem's note durations: the shortest quarter of the notes are transcriber stubs
    (clipped or ghost notes). A grain that short is a click, not the note, so it never becomes a sample."""
    return float(np.percentile([n.end - n.start for n in notes], 25.0))


def typical_note_length(notes: list[Note]) -> float:
    """The 75th percentile of the stem's note durations.

    A sample only reproduces a written note over its whole length if it is at least as long as the note.
    The median would leave half of all notes cut short; P75 covers three of four while still being a
    length the stem demonstrably produces (the maximum is an outlier of the transcriber)."""
    return float(np.percentile([n.end - n.start for n in notes], 75.0))


def pyin_geometry(sr: int, fmin: float) -> tuple[int, int]:
    """``(frame_length, hop_length)`` so the pyin window covers PERIODS_IN_WINDOW periods of ``fmin``."""
    win = math.ceil(PERIODS_IN_WINDOW * sr / fmin)
    frame = 1 << max(1, math.ceil(math.log2(2 * win)))
    return frame, max(64, frame // HOP_DIVISOR)


def measure_pitch(y: np.ndarray, sr: int, fmin: float, fmax: float) -> PitchMeasure | None:
    """Measured pitch of ``y`` = median of the stable part of the voiced pyin track, or None if unvoiced."""
    import librosa
    fmax = min(fmax, sr / 2 - 1)
    if len(y) < 16 or fmin >= fmax:
        return None
    frame, hop = pyin_geometry(sr, fmin)
    f0, voiced, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop)
    n_frames = len(f0)
    ok = ~np.isnan(f0)
    if ok.sum() < MIN_STABLE_FRAMES or n_frames == 0:
        return None
    vfrac = float(np.mean(voiced)) if voiced is not None else float(ok.mean())
    m = 12.0 * np.log2(f0[ok] / 440.0) + 69.0
    keep = np.abs(m - np.median(m)) <= STABLE_TOLERANCE_ST
    stable = m[keep]
    if (len(stable) < MIN_STABLE_FRAMES or len(stable) < 0.5 * len(m) or vfrac < VOICED_FRACTION_MIN
            or float(np.mean(np.asarray(vprob)[ok][keep])) < VOICED_PROB_MIN):
        return None
    med = float(np.median(stable))
    return PitchMeasure(med, midi_to_hz(med), vfrac, float(np.std(stable)), int(len(stable)))


def _spectrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    nfft = 1 << max(10, math.ceil(math.log2(n * FFT_PAD)))
    mag = np.abs(np.fft.rfft(y.astype(np.float64) * np.hanning(n), nfft))
    return np.fft.rfftfreq(nfft, 1.0 / sr), mag


def _salience(freqs: np.ndarray, mag: np.ndarray, sr: int, midi: float) -> float:
    f0 = midi_to_hz(midi)
    total = 0.0
    for h in range(1, SALIENCE_HARMONICS + 1):
        fh = h * f0
        if fh * 2 ** (1 / 24) >= sr / 2:
            break
        lo, hi = np.searchsorted(freqs, [fh * 2 ** (-1 / 24), fh * 2 ** (1 / 24)])
        if hi > lo:
            total += float(mag[lo:hi].max()) / h
    return total


@dataclass(frozen=True)
class Salience:
    accepted: bool
    reason: str
    peak_midi: float | None   # fine-grid salience peak within +-1 semitone of the label
    prominence: float         # S(label) / fence (>1 = outlier of the grain's own salience distribution)


def check_label(y: np.ndarray, sr: int, label: int, lo_midi: float, hi_midi: float) -> Salience:
    """Polyphony-robust verification of a TRANSCRIBED label (pyin is monophonic and declines polyphonic grains).

    Accept when, in the grain's own spectrum, (1) the labelled pitch's harmonic salience is an outlier above
    the Tukey fence of the salience over the register, (2) the fine-grid salience peak within +-1 semitone lies
    within half a semitone of the label (not a neighbouring note), and (3) it beats the octave below AND above
    (octave errors are rejected explicitly: the octave below owns all of the label's partials, the octave
    above owns only the even ones). The grain must span PERIODS_IN_WINDOW periods of the label."""
    f0 = midi_to_hz(label)
    if len(y) < PERIODS_IN_WINDOW * sr / f0 or float(np.max(np.abs(y))) < 1e-6:
        return Salience(False, "grain shorter than 3 periods / silent", None, 0.0)
    freqs, mag = _spectrum(y, sr)
    grid = np.arange(math.floor(lo_midi), math.ceil(hi_midi) + 1)
    sal = np.array([_salience(freqs, mag, sr, m) for m in grid])
    q1, q3 = np.percentile(sal, [25, 75])
    fence = q3 + FENCE_IQR * (q3 - q1)
    s_lab = _salience(freqs, mag, sr, label)
    prom = s_lab / fence if fence > 0 else 0.0
    fine = np.arange(label - 1.0, label + 1.0 + 1e-9, FINE_STEP_ST)
    fs = np.array([_salience(freqs, mag, sr, m) for m in fine])
    peak = float(fine[int(np.argmax(fs))])
    if s_lab <= fence:
        return Salience(False, "label not salient in the grain's spectrum", peak, prom)
    if abs(peak - label) > VERIFY_TOLERANCE_ST:
        return Salience(False, f"salience peaks {peak - label:+.1f} st from the label", peak, prom)
    if s_lab <= _salience(freqs, mag, sr, label + 12):
        return Salience(False, "octave error: the octave above is more salient", peak, prom)
    for h in SUBHARMONICS:   # label would just be the h-th partial of a more salient lower fundamental
        if s_lab <= _salience(freqs, mag, sr, label - round(12 * math.log2(h))):
            return Salience(False, f"label is partial {h} of a more salient lower fundamental", peak, prom)
    return Salience(True, "salient", peak, prom)


@dataclass(frozen=True)
class Assessment:
    note: int | None    # accepted label (may differ from the transcription), None = reject
    method: str         # "salience" | "pyin" | "rejected"
    quality: float
    detail: str


def assess(y: np.ndarray, sr: int, label: int, fmin: float, fmax: float) -> Assessment:
    """Decide the label of one grain. 1) a confident monophonic pyin reading is authoritative (it relabels a
    wrong transcription; it works on bass and solo lines); 2) otherwise pyin has declined (polyphony, short
    grain) and the TRANSCRIBED label is accepted if its salience checks out; 3) otherwise reject."""
    pm = measure_pitch(y, sr, fmin, fmax)
    if pm is not None:
        return Assessment(pm.note, "pyin", pm.quality, f"pyin {pm.midi:.2f}")
    sal = check_label(y, sr, label, hz_to_midi(fmin), hz_to_midi(fmax))
    if sal.accepted:
        return Assessment(label, "salience", sal.prominence, sal.reason)
    return Assessment(None, "rejected", 0.0, sal.reason)


def note_segments(notes: list[Note], total_s: float) -> list[tuple[Note, float, float]]:
    """Each note's audio span ``(note, start_s, end_s)``: its own onset to its end, clipped at the next
    LATER onset (simultaneous onsets are a chord and do not clip each other)."""
    onsets = sorted({round(n.start, 4) for n in notes})
    out = []
    for n in notes:
        i = bisect_right(onsets, round(n.start, 4) + 1e-4)   # first onset strictly later than this note's (O(log N))
        nxt = onsets[i] if i < len(onsets) else None
        end = min(n.end, nxt if nxt is not None else n.end, total_s)
        if end > n.start:
            out.append((n, float(n.start), float(end)))
    return out


def shape_grain(grain: np.ndarray, sr: int, f0_hz: float) -> np.ndarray:
    """Fade-in of one f0 period (click guard, keeps the attack) and a cosine release of a quarter of the
    grain (Tukey taper; the note was cut at its transcribed end / next onset, so it must not stop dead).
    Both lengths are derived from the grain / its pitch."""
    g = grain.astype(np.float64).copy()
    n = len(g)
    a = min(int(round(sr / f0_hz)), n // 4)
    r = n // 4
    if a > 1:
        g[:a] *= np.linspace(0.0, 1.0, a)
    if r > 1:
        t = np.linspace(0.0, 1.0, r)
        g[-r:] *= 1.0 - (t * t * (3 - 2 * t))
    return g.astype(np.float32)


@dataclass
class Grain:
    audio: np.ndarray
    note: int                # accepted label (transcription if verified, else pyin's reading)
    transcribed: int         # the transcribed midi note this segment came from
    method: str              # how the label was established: "salience" | "pyin"
    quality: float
    onset_s: float
    duration_s: float
    ringed: bool = False     # cut where the stem stops sounding the note, not at the transcribed end
    rms: float = 0.0         # level of the grain as the stem plays it


RING_PITCH_TOLERANCE_ST = 1.0   # a ring that leaves the label by more than a semitone is the next note / a glide
RING_DEVIANT_FRAMES = 2         # ... for at least this many pyin frames (one frame is a pyin glitch)


def ring_end(onset: float, note_end: float, attacks: np.ndarray, own_tol: float, sounding_end: float) -> float:
    """Where the stem's note struck at ``onset`` really stops: the earlier of the stem going quiet
    (``sounding_end``) and its next audible re-attack, never earlier than the transcribed end.

    The transcriber cuts an 808 at ~one 16th step (median 0.07-0.09 s on both reference stems) while the stem
    rings 0.2-0.3 s (and seconds when held), so a grain cut at the transcribed end loses the sub tail.
    ``own_tol`` (seconds) = the attack detector's resolution: an attack within it of ``onset`` is this
    note's own. A note with no attack of its own is a transcription echo: it keeps its transcribed span."""
    own = attacks[np.abs(attacks - onset) <= own_tol]
    if own.size == 0:
        return note_end
    after = attacks[attacks > float(own.max()) + own_tol]
    nxt = float(after[0]) if after.size else sounding_end
    return max(note_end, min(nxt, sounding_end))


def pitch_stable_end(y: np.ndarray, sr: int, label: int, fmin: float, fmax: float, start_s: float = 0.0) -> float:
    """Seconds into ``y`` after which the pyin track leaves ``label`` (a legato note / slide begins).

    Frames before the track first reaches the label are the note's own pitch drop and are kept; from there
    the grain ends at the first run of RING_DEVIANT_FRAMES voiced frames further than RING_PITCH_TOLERANCE_ST
    from the label, or at its end. ``start_s`` frames earlier than it are never trimmed (the transcribed span)."""
    import librosa
    fmax = min(fmax, sr / 2 - 1)
    if len(y) < 16 or fmin >= fmax:
        return len(y) / sr
    frame, hop = pyin_geometry(sr, fmin)
    f0, _v, _p = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop)
    dev = np.full(len(f0), np.nan)
    ok = ~np.isnan(f0)
    dev[ok] = np.abs(12.0 * np.log2(f0[ok] / midi_to_hz(label)))
    reached = np.nonzero(dev <= STABLE_TOLERANCE_ST)[0]
    first = int(reached[0]) if reached.size else 0
    run = 0
    for i in range(first, len(dev)):
        if not np.isnan(dev[i]) and dev[i] > RING_PITCH_TOLERANCE_ST:
            run += 1
            if run >= RING_DEVIANT_FRAMES:
                cut = (i - run + 1) * hop / sr
                return max(cut, start_s)
        elif not np.isnan(dev[i]):
            run = 0
    return len(y) / sr


class StemRing:
    """The stem's own attacks and decay (bass_shape.StemEvidence), as ``ring_end`` inputs."""

    def __init__(self, y: np.ndarray, sr: int):
        import librosa
        import bass_shape as BS
        y22 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=BS.StemEvidence.SR)
        self.ev = BS.StemEvidence(y22)
        self.own_tol = 4 * self.ev.hop_s            # backtracked onsets sit a few hops before the audible attack
        gaps = np.diff(self.ev.attacks)
        # longest hold the stem itself shows: P95 of its inter-attack intervals
        self.horizon = float(np.percentile(gaps, 95.0)) if gaps.size else 0.0

    def end(self, note: Note, seg_end: float) -> float:
        if self.horizon <= 0.0:
            return seg_end
        sounding = self.ev.end_after(note.start, self.horizon)
        return ring_end(note.start, seg_end, self.ev.attacks, self.own_tol, sounding)


def extract_note_grains(y: np.ndarray, sr: int, notes: list[Note], fmin: float, fmax: float,
                        ring: "StemRing | None" = None) -> list[Grain]:
    """One verified grain per transcribed note (onset -> end/next onset); unverifiable segments are skipped.

    With ``ring`` a note that has an audible attack of its own is cut where the STEM stops sounding it
    (:func:`ring_end`), trimmed where its pitch moves away (:func:`pitch_stable_end`); the shorter
    transcribed span is kept when the longer cut cannot be verified."""
    grains: list[Grain] = []
    total = len(y) / sr
    for n, s, e in note_segments(notes, total):
        if ring is not None:
            e_ring = min(ring.end(n, e), total)
            if e_ring > e + 1e-3:
                long_seg = y[int(s * sr):int(e_ring * sr)]
                cut = pitch_stable_end(long_seg, sr, n.pitch, fmin, fmax, start_s=e - s)
                long_seg = long_seg[:int(cut * sr)]
                a = assess(long_seg, sr, n.pitch, fmin, fmax)
                if a.note is not None and len(long_seg) > int((e - s) * sr):
                    grains.append(Grain(shape_grain(attack_aligned(long_seg, sr), sr, midi_to_hz(a.note)), a.note, n.pitch, a.method,
                                        a.quality, s, len(long_seg) / sr, True, _rms(long_seg)))
                    continue
        seg = y[int(s * sr):int(e * sr)]
        a = assess(seg, sr, n.pitch, fmin, fmax)
        if a.note is None:
            continue
        grains.append(Grain(shape_grain(attack_aligned(seg, sr), sr, midi_to_hz(a.note)), a.note, n.pitch, a.method, a.quality, s,
                            len(seg) / sr, False, _rms(seg)))
    return grains


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def select_per_pitch(grains: list[Grain], target_len_s: float, min_len_s: float = 0.0) -> dict[int, Grain]:
    """Best grain per note: among those at least ``target_len_s`` long the highest quality; if none is long
    enough, the longest (ties -> quality). Grains shorter than ``min_len_s`` (stub_note_length) are never used."""
    by: dict[int, list[Grain]] = {}
    for g in grains:
        if g.duration_s >= min_len_s:
            by.setdefault(g.note, []).append(g)
    best: dict[int, Grain] = {}
    for note, gs in by.items():
        long_enough = [g for g in gs if g.duration_s >= target_len_s]
        pool = long_enough or gs
        # A sample stands for every time the stem plays this pitch: take the loudness the stem usually plays it
        # at (at or above the pitch's median grain level), not the quietest passage that happened to be cleanest.
        med = float(np.median([g.rms for g in pool]))
        pool = [g for g in pool if g.rms >= med]
        best[note] = (max(pool, key=lambda g: g.quality) if long_enough
                      else max(pool, key=lambda g: (g.duration_s, g.quality)))
    return best


def coverage(transcribed: list[int], samples: list[int]) -> dict:
    """Repitch distance (semitones) from each transcribed note to its nearest sample note."""
    if not samples:
        return {"samples": 0, "max_repitch_st": None}
    d = np.array([min(abs(t - m) for m in samples) for t in transcribed], dtype=float)
    lo, hi = np.percentile(transcribed, [RANGE_TRIM_PCT, 100 - RANGE_TRIM_PCT])
    core = d[[lo <= t <= hi for t in transcribed]]
    return {"samples": len(samples), "sample_notes": sorted(samples),
            "transcribed_range": [int(min(transcribed)), int(max(transcribed))],
            "transcribed_core_range": [int(round(lo)), int(round(hi))],
            "max_repitch_st": float(d.max()), "max_repitch_core_st": float(core.max()) if len(core) else None,
            "median_repitch_st": float(np.median(d)), "p95_repitch_st": float(np.percentile(d, 95))}


def verify_samples(samples: dict[str, Path], *, fmin: float | None = None, fmax: float | None = None,
                   tolerance_st: float = VERIFY_TOLERANCE_ST) -> tuple[dict[str, Path], list[dict]]:
    """Re-measure every ``{note name: wav}`` sample; keep, relabel or drop. Returns ``(final map, report)``.

    keep: |measured - label| <= tolerance; relabel: it measures as another note that no verified sample
    already covers; drop: unvoiced, or its measured note is already covered. The search range is
    ``[fmin, fmax]`` or, absent that, one octave either side of the labels (octave errors stay audible)."""
    import soundfile as sf
    names = list(samples)
    labels = {k: _note_midi(k) for k in names}
    if fmin is None or fmax is None:
        fmin = midi_to_hz(min(labels.values()) - VERIFY_OCTAVE_MARGIN_ST)
        fmax = midi_to_hz(max(labels.values()) + VERIFY_OCTAVE_MARGIN_ST)
    rows: dict[str, dict] = {}
    for k in names:
        y, sr = sf.read(str(samples[k]), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        a = assess(y, sr, labels[k], fmin, fmax)
        row = {"label": k, "file": samples[k].name, "duration_s": round(len(y) / sr, 4)}
        if a.note is None:
            row.update(measured=None, error_st=None, action="dropped", reason=a.detail)
        else:
            row.update(measured=midi_to_note(a.note), method=a.method, error_st=float(a.note - labels[k]),
                       quality=round(a.quality, 3), reason=a.detail)
            row["action"] = "kept" if a.note == labels[k] else "pending"
        rows[k] = row
    final: dict[str, Path] = {k: samples[k] for k in names if rows[k]["action"] == "kept"}
    covered = set(final)
    pend = sorted((k for k in names if rows[k]["action"] == "pending"), key=lambda k: -rows[k]["quality"])
    for k in pend:
        new = rows[k]["measured"]
        if new not in covered:
            rows[k].update(action="relabelled", new_label=new, reason=f"measured {new}, labelled {k}")
            final[new] = samples[k]
            covered.add(new)
        else:
            rows[k].update(action="dropped", reason=f"measured {new} (labelled {k}); {new} is already covered")
    return dict(sorted(final.items(), key=lambda kv: _note_midi(kv[0]))), [rows[k] for k in names]


def _note_midi(note: str) -> int:
    """'c#4' or Strudel's 'cs4' -> MIDI."""
    idx = len(note.rstrip("0123456789-"))
    pc = note[:idx].replace("s", "#")
    return NOTE_NAMES.index(pc) + (int(note[idx:]) + 1) * 12


def ring_target_length(grains: list["Grain"], fallback: float) -> float:
    """P75 of the ringing grains' lengths (the same convention as :func:`typical_note_length`, but measured on
    what the stem sustains); ``fallback`` when no note rang past its transcribed end."""
    ringed = [g.duration_s for g in grains if g.ringed]
    return float(np.percentile(ringed, 75.0)) if ringed else fallback


ATTACK_FRACTION = 0.1   # the audible attack = first sample at 10 % of the grain's peak (-20 dB)


def attack_aligned(grain: np.ndarray, sr: int) -> np.ndarray:
    """Drop the silence before the audible attack so the attack sits at sample 0 (the transcribed onset can
    lead the stem's hit by tens of ms: VAGABUNDO c#1 started 79 ms late, every hit of that note landed late).
    One sample's worth of look-back is kept so the onset is not clipped."""
    pk = float(np.max(np.abs(grain))) if len(grain) else 0.0
    if pk <= 0.0:
        return grain
    first = int(np.argmax(np.abs(grain) >= ATTACK_FRACTION * pk))
    return grain[max(0, first - 1):] if first > 1 else grain

