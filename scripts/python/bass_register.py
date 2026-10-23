#!/usr/bin/env python3
"""Register of the bass stem, measured from the audio, and the octave decision built on it.

Why: ``fold_pitch(p, 28, 48)`` (a typed E1-C3 window) moved 43% (VAGABUNDO) / 24% (Regime CLT) of the
transcribed bass notes an octave UP, because the real bass of both tracks sits at MIDI 25-31 (a held C#1
sub). Basic Pitch's lowest note per step was already right ~75% of the time; after the typed fold only
45% / 59% (BASS.md, 2026-10-19). The octave belongs to the stem, the pitch class to the transcription:

1. ``measure``  pyin f0 of ``bass.wav`` down to the sub-bass floor, cached as JSON keyed by the stem's
                content hash (``<track>/.pipeline/bass_f0.json``; the pipeline driver writes it).
2. ``Register.window``  the smallest semitone interval holding ``MASS_FRACTION`` of the loudness-weighted
                f0 mass = the stem's own register (VAGABUNDO [25,38], Regime CLT [27,31]).
3. ``Register.resolve``  per step: snap each transcribed note to the octave of the LOCAL stem f0 when its
                pitch class agrees (within ``PITCH_CLASS_TOL_ST``), then pick the candidate nearest that f0;
                where the stem is unvoiced fall back to folding into the derived window.

Everything numeric below is either a named, justified constant or taken from the project's band table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

# Lowest / highest f0 searched = the project's own sub-bass floor and the top of its bass band
# (compare_audio.FREQ_BANDS: sub_bass 20-60, bass 60-250 Hz).
SUB_FLOOR_HZ = 20.0
BASS_CEIL_HZ = 250.0
SR = 11025                # bass content is < 250 Hz; a low rate keeps pyin cheap and the window long
MASS_FRACTION = 0.96      # keep the central 96% of the loudness-weighted f0 mass; the outer 2% per side are
                          # pyin glide/attack outliers (measured: p2..p98 gave the best exact-pitch rate on both
                          # reference tracks, 70.3% / 81.1% vs 73.7% / 74.4% for a window 11 semitones wider)
ACTIVE_RMS_FRACTION = 0.2  # a frame counts when its RMS >= this fraction of the median non-silent RMS
                          # (below it: decay tails / noise floor, f0 is not a musical pitch)
PITCH_CLASS_TOL_ST = 1.0  # a transcribed note whose pitch class is within this of the stem f0 (mod octave)
                          # is "the same note, maybe in the wrong octave" (808 glides span ~1 semitone)
MIN_FRAMES = 2            # voiced frames needed to call a local f0
VERSION = 1


def _frame_geometry(sr: int, fmin: float) -> tuple[int, int]:
    """pyin frame covering >= 2 periods of ``fmin`` (power of two), hop = frame/8."""
    frame = 1 << math.ceil(math.log2(2 * sr / fmin))
    return frame, frame // 8


def stem_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(stem: Path, cache: Path | None = None) -> dict:
    """pyin f0 track of ``stem``; reuses ``cache`` when its hash/version/params match."""
    key = {"version": VERSION, "sha1": stem_hash(stem), "sr": SR, "fmin": SUB_FLOOR_HZ, "fmax": BASS_CEIL_HZ}
    if cache and cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("key") == key:
                return d
        except (OSError, ValueError):
            pass
    import librosa
    y, _ = librosa.load(str(stem), sr=SR, mono=True)
    frame, hop = _frame_geometry(SR, SUB_FLOOR_HZ)
    f0, _vf, _vp = librosa.pyin(y, fmin=SUB_FLOOR_HZ, fmax=BASS_CEIL_HZ, sr=SR, frame_length=frame,
                                hop_length=hop, fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0][: len(f0)]
    midi = np.where(np.isnan(f0), np.nan, 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0))
    d = {"key": key, "hop_s": hop / SR,
         "midi": [None if np.isnan(m) else round(float(m), 2) for m in midi],
         "rms": [round(float(r), 6) for r in rms]}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d))
    return d


def smallest_interval(weights: dict[int, float], mass: float) -> tuple[int, int]:
    """Smallest [lo,hi] (integer pitches) whose total weight >= ``mass`` of the whole."""
    ks = sorted(weights)
    total = sum(weights.values())
    best = (ks[0], ks[-1])
    j = 0
    acc = 0.0
    for i, k in enumerate(ks):
        acc += weights[k]
        while acc - weights[ks[j]] >= mass * total:
            acc -= weights[ks[j]]
            j += 1
        if acc >= mass * total and ks[i] - ks[j] < best[1] - best[0]:
            best = (ks[j], ks[i])
    return best


class Register:
    def __init__(self, data: dict):
        self.hop = float(data["hop_s"])
        self.midi = np.array([np.nan if m is None else m for m in data["midi"]], dtype=float)
        self.rms = np.asarray(data["rms"], dtype=float)
        n = min(len(self.midi), len(self.rms))
        self.midi, self.rms = self.midi[:n], self.rms[:n]
        nz = self.rms[self.rms > 0]
        gate = ACTIVE_RMS_FRACTION * float(np.median(nz)) if nz.size else 0.0
        self.active = ~np.isnan(self.midi) & (self.rms >= gate)
        self.times = np.arange(n) * self.hop

    @classmethod
    def load(cls, path: Path) -> "Register":
        return cls(json.loads(Path(path).read_text()))

    def window(self) -> tuple[int, int]:
        """The stem's register: smallest semitone interval with MASS_FRACTION of the loudness-weighted f0 mass."""
        if not self.active.any():
            raise ValueError("no voiced bass frames")
        w: dict[int, float] = {}
        for m, r in zip(np.rint(self.midi[self.active]).astype(int), self.rms[self.active]):
            w[int(m)] = w.get(int(m), 0.0) + float(r)
        return smallest_interval(w, MASS_FRACTION)

    def local(self, t0: float, t1: float) -> float | None:
        """Median f0 (MIDI, float) of the active frames in [t0,t1]; None when unvoiced there."""
        s = self.active & (self.times >= t0) & (self.times <= t1)
        return float(np.median(self.midi[s])) if s.sum() >= MIN_FRAMES else None


def fold_into(p: int, lo: int, hi: int) -> int:
    """Octave-fold ``p`` into [lo,hi]; if the interval is narrower than an octave and holds no copy of the
    pitch class, take the nearest edge (never leaves the window)."""
    while p > hi:
        p -= 12
    while p < lo:
        p += 12
    return max(lo, min(hi, p))


def snap_octave(p: int, loc: float) -> int | None:
    """``p`` moved to the octave nearest ``loc`` if its pitch class agrees within PITCH_CLASS_TOL_ST, else None."""
    d = p - loc
    o = round(d / 12.0)
    return p - 12 * o if abs(d - 12 * o) <= PITCH_CLASS_TOL_ST else None


def harmonic_of(p: float, loc: float) -> int | None:
    """k (>= 2) when ``p`` sits within PITCH_CLASS_TOL_ST of the k-th harmonic of the fundamental ``loc``
    (k*f0 = 12*log2(k) semitones up), else None. Only harmonics distinguishable from their neighbours count:
    the gap to BOTH neighbours, 12*log2((k+1)/k) and 12*log2(k/(k-1)), must be >= 2 tolerances (k <= 8 at 1 semitone) - above that
    a "harmonic" would match almost any pitch. Octaves (k=2,4) never reach here (``snap_octave`` takes them)."""
    d = p - loc
    if d <= 0:
        return None
    k = round(2.0 ** (d / 12.0))
    if k < 2 or 12.0 * math.log2((k + 1) / k) < 2 * PITCH_CLASS_TOL_ST or 12.0 * math.log2(k / (k - 1)) < 2 * PITCH_CLASS_TOL_ST:
        return None
    return k if abs(d - 12.0 * math.log2(k)) <= PITCH_CLASS_TOL_ST else None


def resolve(cands: list[int], loc: float | None, window: tuple[int, int], pick: str = "low") -> int:
    """The note written for one step from the transcribed ``cands``.

    loc known   -> each candidate snapped to the stem's octave; a candidate that is a (non-octave) harmonic of the
                   stem f0 is a transcription of the sub's PARTIAL (Basic Pitch hears the 3rd/5th of an 808), so
                   it becomes the fundamental round(loc); any other disagreeing pitch class is only folded into
                   the window. The candidate nearest ``loc`` wins (ties: the lower).
    loc unknown -> window fold, then lowest (``pick='low'``) / highest.
    """
    lo, hi = window
    if loc is None:
        folded = [fold_into(p, lo, hi) for p in cands]
        return min(folded) if pick == "low" else max(folded)
    out = []
    for p in cands:
        s = snap_octave(p, loc)
        if s is None and harmonic_of(p, loc) is not None:
            s = int(round(loc))
        out.append(s if s is not None else fold_into(p, lo, hi))
    return min(out, key=lambda q: (abs(q - loc), q))


def weighted_window(pitches, weights) -> tuple[int, int]:
    """Smallest interval holding MASS_FRACTION of the weight (used for the polyphonic lead, which has no f0)."""
    w: dict[int, float] = {}
    for p, x in zip(pitches, weights):
        w[int(p)] = w.get(int(p), 0.0) + float(x)
    return smallest_interval(w, MASS_FRACTION)


def melodic_window(midi_path: Path, stem_path: Path, floor_pitch: int) -> tuple[int, int]:
    """Lead register: smallest MASS_FRACTION interval of the transcribed melodic notes (weight = duration x stem RMS
    over the note) that lie ABOVE ``floor_pitch`` (the top of the bass register; the melodic stem carries bass bleed
    below it, which is not lead material). VAGABUNDO [44,81], Regime CLT [40,80]."""
    import librosa
    import pretty_midi
    y, sr = librosa.load(str(stem_path), sr=22050, mono=True)
    notes = [n for i in pretty_midi.PrettyMIDI(str(midi_path)).instruments for n in i.notes if n.pitch > floor_pitch]
    if not notes:
        raise ValueError("no melodic notes above the bass register")
    def rms(n):
        a, b = int(n.start * sr), max(int(n.end * sr), int(n.start * sr) + 1)
        return float(np.sqrt(np.mean(y[a:b] ** 2))) if a < len(y) else 0.0
    return weighted_window([n.pitch for n in notes], [(n.end - n.start) * rms(n) for n in notes])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="write the cached f0 track of a bass stem")
    m.add_argument("--stem", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = measure(a.stem, a.out)
    reg = Register(d)
    print(json.dumps({"out": str(a.out), "frames": len(reg.midi), "voiced": int(reg.active.sum()),
                      "window": list(reg.window())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
