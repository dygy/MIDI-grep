#!/usr/bin/env python3
"""Listening EQ profile of a promoted piece, for the owner's Strudel player (10-band EQ, spec 020/021).

The player shows "this track recommends: ..." with an Apply button. This module produces the data: how much
each octave band of the RENDER (``vNNN/render.wav``) must be raised on playback so it sounds like the ORIGINAL
INSTRUMENTAL (``instrumental.wav``, the score's reference). Everything is measured; nothing is typed per track.

Contract (fixed; the player parses exactly this):

    // @eq-profile {"v":1,"preset":..,"bands":[{"frequency":31,"gain":g},..10..],"lowCut":..,"highCut":..,
                    "outputGain":0,"basis":"instrumental vs render","why":".."}

one line in ``vNNN/output.strudel`` right after the leading header comment block (replaced on rerun), and the
same object pretty-printed in ``vNNN/eq_profile.json`` with a ``diagnostics`` key (not in the line).

Measurement
-----------
* Both signals are loaded the way ``compare_audio`` loads them (``load_audio``, from t=0, the length
  ``comparison.json`` analysed) at 44.1 kHz so the 16 kHz band exists, trimmed to the shorter, and
  ``compare_audio.rms_normalize``d (the score is level-invariant, so is the profile).
* Long-term average power spectrum (Welch) per octave band centred at 31 ... 16000 Hz; band edges are the
  geometric means between neighbouring centres (the bands tile the spectrum). gain_b = 10 log10(P_original_b /
  P_render_b).
* Loudness match: all gains shift by one common offset so the render's power, played through the profile, equals
  its power without it (energy-weighted mean change 0 dB; the player keeps every preset within 3 dB of Flat).
  Each gain is then clamped to the player's band range (+-24 dB, ``EQ_GAIN_LIMIT`` in AudioEqualizer.ts).
  ``outputGain`` stays 0.

Cuts (no typed dB threshold; the original's own spectrum is the yardstick)
--------------------------------------------------------------------------
A cut is a CLIFF: the original's bands beyond the cut edge all sit lower than the band next to the edge by MORE
than the largest octave-to-octave step the original shows anywhere inside its own body (the bands on the other
side of the edge), AND the render has more energy than the original there (its gain is negative). The body needs
>= 3 bands (to have steps to compare against). Of several valid edges the one that removes the fewest bands wins.
The cut frequency is the band edge. The cut bands' own gains are then 0 (the cut does that job) and they do not
count in the loudness match. Otherwise ``lowCut`` / ``highCut`` are null.

Preset
------
The player's genre preset whose id matches the track's genre (``metadata.json`` ``genre``) by normalised name
(lower case, no separators, a trailing "genre" dropped): brazilian_funk -> brazilianFunk. null when the genre is
uncertain / unreferenced or no id matches. ``GENRE_PRESET_IDS`` is the id list of the player's genre group
(``packages/equalizer/presets.ts``, group 'genre'); update it when the player adds presets.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.signal import welch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

LINE_TAG = "// @eq-profile "
PROFILE_JSON = "eq_profile.json"
BASIS = "instrumental vs render"
SCHEMA_VERSION = 1
SAMPLE_RATE = 44100                       # high enough for the 16 kHz band (22.6 kHz upper edge is clipped at Nyquist)
BAND_FREQUENCIES: tuple[int, ...] = (31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)   # the player's 10 bands
EQ_GAIN_LIMIT = 24.0                      # dB, AudioEqualizer.ts EQ_GAIN_LIMIT (read from the player's source)
LOW_CUT_RANGE = (20.0, 1000.0)            # Hz, AudioEqualizer.ts
HIGH_CUT_RANGE = (1000.0, 20000.0)
MIN_BODY_BANDS = 3                        # a cliff is judged against the steps inside >= 3 bands of body
WELCH_NPERSEG = 16384                     # 2.7 Hz bins at 44.1 kHz: ~8 bins in the 31 Hz band
POWER_FLOOR = 1e-12                       # of the total band power: keeps log10 finite for an empty band

# Genre-group preset ids of the player (strudel-client packages/equalizer/presets.ts, spec 020/021).
GENRE_PRESET_IDS: tuple[str, ...] = (
    "brazilianFunk", "automotivo", "montagem", "funkAgressivo", "mandelao", "bruxaria", "mandelaoSlowed",
    "funkCariocaOldSchool", "baileFunk", "reggaeton", "dembow", "cumbia", "afroHouse", "amapiano",
    "phonk", "driftPhonk", "trap", "brazilianPhonk", "housePhonk", "memphis", "boomBap", "lofiHipHop",
    "ukDrill", "nyDrill", "rage", "plugg", "cloudRap", "jerseyClub", "electro", "techno", "drumAndBass",
    "dubstep", "house", "deepHouse", "techHouse", "minimal", "hardTechno", "trance", "psytrance",
    "hardstyle", "gabber", "ukGarage", "jungle", "breakbeat", "synthwave", "hyperpop", "futureBass",
    "riddim", "rockGenre", "metal", "punk", "usFunk", "disco", "soul", "reggae", "dub", "ambient",
    "chillout", "lofi",
)


class Band(BaseModel):
    model_config = ConfigDict(frozen=True)
    frequency: int
    gain: float


class Cut(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool
    frequency: float


class EqProfile(BaseModel):
    """The contract object. Field order is the order in the line."""
    model_config = ConfigDict(frozen=True)
    v: int
    preset: str | None
    bands: list[Band]
    lowCut: Cut | None
    highCut: Cut | None
    outputGain: int
    basis: str
    why: str

    def line(self) -> str:
        """The compact single-line ``// @eq-profile {...}`` comment."""
        return LINE_TAG + self.model_dump_json()


# ---------------------------------------------------------------- measurement

def band_edges() -> list[float]:
    """11 edges tiling the spectrum: geometric means between centres, f0/sqrt2 and f9*sqrt2 outside."""
    f = BAND_FREQUENCIES
    inner = [math.sqrt(a * b) for a, b in zip(f, f[1:])]
    return [f[0] / math.sqrt(2), *inner, f[-1] * math.sqrt(2)]


def band_powers(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Long-term average power per band (Welch PSD summed over the band's bins; clipped at Nyquist)."""
    nper = min(WELCH_NPERSEG, len(y))
    freqs, psd = welch(y, fs=sr, nperseg=nper)
    edges = band_edges()
    out = np.zeros(len(BAND_FREQUENCIES))
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        out[i] = float(psd[(freqs >= lo) & (freqs < hi)].sum())
    return out


def _db(ratio: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(ratio)


def _levels(p: np.ndarray) -> np.ndarray:
    return _db(np.maximum(p, p.sum() * POWER_FLOOR))


def _find_cut(level_o: np.ndarray, gains: np.ndarray, side: str) -> int | None:
    """Edge index of the cliff on ``side`` ('low' | 'high'), or None. Edge j sits between bands j-1 and j.

    High side: tail = bands >= j, body = bands < j. Low side: tail = bands < j, body = bands >= j.
    The fewest removed bands wins (high: the largest j; low: the smallest j).
    """
    n = len(level_o)
    for j in (range(1, n) if side == "low" else range(n - 1, 0, -1)):
        body = level_o[j:] if side == "low" else level_o[:j]
        tail = level_o[:j] if side == "low" else level_o[j:]
        if len(body) < MIN_BODY_BANDS:
            continue
        steps = np.abs(np.diff(body))
        edge_level = body[0] if side == "low" else body[-1]
        cliff = bool(np.all(tail < edge_level - steps.max()))
        render_heavier = bool(np.all((gains[:j] if side == "low" else gains[j:]) < 0))
        if cliff and render_heavier:
            return j
    return None


def _cut(edge_index: int | None, side: str) -> Cut | None:
    if edge_index is None:
        return None
    lo, hi = LOW_CUT_RANGE if side == "low" else HIGH_CUT_RANGE
    freq = band_edges()[edge_index]
    return Cut(enabled=True, frequency=round(min(max(freq, lo), hi), 1))


def _fmt_hz(f: int) -> str:
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f} Hz"


def explain(gains: list[float], low: Cut | None, high: Cut | None) -> str:
    """One English sentence from the measured gains (positive = the render is light there)."""
    parts: list[str] = []
    ranked = sorted(zip(BAND_FREQUENCIES, gains), key=lambda fg: -abs(fg[1]))
    for f, g in ranked:
        if len(parts) == 2 or round(g, 1) == 0:
            break
        word = "light" if g > 0 else "heavy"
        parts.append(f"{abs(g):.1f} dB {word} at {_fmt_hz(f)}")
    cuts = [f"no energy {side} {_fmt_hz(int(c.frequency))}" for side, c in (("below", low), ("above", high)) if c]
    if not parts and not cuts:
        return "The render already matches the original's spectral balance."
    body = " and ".join(parts)
    if body:
        body = f"The render is {body} versus the original"
    if cuts:
        tail = "the original has " + " and ".join(cuts)
        body = f"{body}; {tail}" if body else f"The render differs from the original: {tail}"
    return body + "."


def compute_profile(original: np.ndarray, render: np.ndarray, sr: int = SAMPLE_RATE,
                    preset: str | None = None) -> tuple[EqProfile, dict]:
    """The profile for two mono signals (as loaded), plus a diagnostics dict for ``eq_profile.json``."""
    from compare_audio import rms_normalize
    n = min(len(original), len(render))
    if n < sr:
        raise ValueError(f"need at least 1 s of audio in both signals, got {n / sr:.2f} s")
    p_o = band_powers(rms_normalize(original[:n]), sr)
    p_r = band_powers(rms_normalize(render[:n]), sr)
    lev_o, lev_r = _levels(p_o), _levels(p_r)
    raw = lev_o - lev_r                                   # dB the render must be raised, per band

    low_j, high_j = _find_cut(lev_o, raw, "low"), _find_cut(lev_o, raw, "high")
    cut_mask = np.zeros(len(raw), dtype=bool)
    if low_j is not None:
        cut_mask[:low_j] = True
    if high_j is not None:
        cut_mask[high_j:] = True

    # Loudness match over the bands the cuts leave: the render's power through the profile equals without it.
    keep = ~cut_mask
    w = p_r[keep] / p_r[keep].sum()
    offset = float(-10.0 * math.log10(float(np.sum(w * 10.0 ** (raw[keep] / 10.0)))))
    gains = np.where(cut_mask, 0.0, np.clip(raw + offset, -EQ_GAIN_LIMIT, EQ_GAIN_LIMIT))
    rounded = [round(float(g), 1) + 0.0 for g in gains]   # + 0.0 turns -0.0 into 0.0

    low, high = _cut(low_j, "low"), _cut(high_j, "high")
    profile = EqProfile(
        v=SCHEMA_VERSION, preset=preset,
        bands=[Band(frequency=f, gain=g) for f, g in zip(BAND_FREQUENCIES, rounded)],
        lowCut=low, highCut=high, outputGain=0, basis=BASIS, why=explain(rounded, low, high))
    diagnostics = {
        "band_level_original_db": [round(float(x), 2) for x in lev_o],
        "band_level_render_db": [round(float(x), 2) for x in lev_r],
        "raw_gain_db": [round(float(x), 2) for x in raw],
        "loudness_offset_db": round(offset, 2),
        "clamped_bands": [f for f, r in zip(BAND_FREQUENCIES, raw + offset) if abs(r) > EQ_GAIN_LIMIT],
        "seconds": round(n / sr, 2), "sample_rate": sr,
    }
    return profile, diagnostics


# ---------------------------------------------------------------- preset

def _norm(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key[: -len("genre")] if key.endswith("genre") and len(key) > len("genre") else key


def preset_for_genre(genre: str | None, *, uncertain: bool = False, genre_ref: str | None = None) -> str | None:
    """The player's genre preset id for a track's genre, or None (uncertain / unreferenced / no match)."""
    if not genre or uncertain or genre_ref == "unreferenced":
        return None
    by_name = {_norm(i): i for i in GENRE_PRESET_IDS}
    candidates = [_norm(genre)]
    try:
        from sound_selector import GENRE_ALIASES
        alias = GENRE_ALIASES.get(genre.lower().strip().replace(" ", "_").replace("-", "_"))
        if alias:
            candidates.append(_norm(alias))
    except ImportError:
        pass
    return next((by_name[c] for c in candidates if c in by_name), None)


# ---------------------------------------------------------------- the line in output.strudel

def insert_profile_line(code: str, line: str) -> str:
    """``code`` with exactly one ``// @eq-profile`` line, right after the leading ``//`` header block."""
    lines = [ln for ln in code.split("\n") if not ln.startswith(LINE_TAG.rstrip())]
    end = 0
    while end < len(lines) and lines[end].lstrip().startswith("//"):
        end += 1
    lines.insert(end, line)
    return "\n".join(lines)


# ---------------------------------------------------------------- piece-level API

def write_profile(track_dir: Path, version: str) -> EqProfile:
    """Measure ``track_dir/version`` and write ``eq_profile.json`` + the line into its ``output.strudel``."""
    from compare_audio import load_audio
    vdir = track_dir / version
    strudel, render, ref = vdir / "output.strudel", vdir / "render.wav", track_dir / "instrumental.wav"
    for p in (strudel, render, ref):
        if not p.is_file():
            raise FileNotFoundError(f"eq_profile: missing {p}")
    seconds: float | None = None
    cmp_path = vdir / "comparison.json"
    if cmp_path.is_file():
        seconds = json.loads(cmp_path.read_text(encoding="utf-8")).get("duration_analyzed")
    original = load_audio(str(ref), sr=SAMPLE_RATE, duration=seconds)
    rendered = load_audio(str(render), sr=SAMPLE_RATE, duration=seconds)
    if original is None or rendered is None:
        raise ValueError("eq_profile: could not load the audio")
    meta_path = track_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    preset = preset_for_genre(meta.get("genre"), uncertain=bool(meta.get("genre_uncertain")),
                              genre_ref=meta.get("genre_ref"))
    profile, diagnostics = compute_profile(original, rendered, SAMPLE_RATE, preset)
    doc = profile.model_dump()
    (vdir / PROFILE_JSON).write_text(json.dumps({**doc, "diagnostics": diagnostics}, indent=2) + "\n", encoding="utf-8")
    code = strudel.read_text(encoding="utf-8")
    updated = insert_profile_line(code, profile.line())
    if updated != code:
        strudel.write_text(updated, encoding="utf-8")
    return profile


def main() -> int:
    ap = argparse.ArgumentParser(description="Write the listening EQ profile of a promoted piece.")
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("version", help="vNNN")
    args = ap.parse_args()
    try:
        profile = write_profile(args.track_dir, args.version)
    except (OSError, ValueError) as e:
        print(f"eq_profile: {e}", file=sys.stderr)
        return 1
    print(profile.line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
