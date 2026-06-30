#!/usr/bin/env python3
"""
build_sample_pack.py — Build a Strudel-playable sample pack from Demucs stems.

Turns extracted stems (melodic.wav, bass.wav, drums.wav, vocals.wav) into:
  - drums/      : kick (bd), snare (sd), closed-hat (hh), open-hat (oh) one-shots
  - bass/       : pitched multi-samples (pyin) + representative bass.wav
  - melodic/    : pitched multi-samples + representative melodic.wav
  - loops/      : beat-aligned per-bar WAV slices for each stem
  - strudel.json: Strudel ``samples()`` manifest (relative paths only)
  - pack.json   : metadata — bpm, key, counts, bar_duration

All classification thresholds are derived from THIS track's feature
distributions (percentile-based). No absolute values hardcoded to a
specific recording.

Usage::

    build_sample_pack.py --stems-dir DIR --out DIR \\
        [--bpm FLOAT] [--key STR] [--num-bars INT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR: int = 44_100          # canonical sample rate
FADE_MS: int = 10         # fade-in/out in ms to avoid clicks
PEAK_TARGET: float = 0.95 # normalisation target


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at SR=44100."""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    return y, sr


def peak_normalize(y: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """Peak-normalise array to *target* amplitude."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y
    return y * (target / peak)


def apply_fade(y: np.ndarray, fade_ms: int = FADE_MS, sr: int = SR) -> np.ndarray:
    """Apply short linear fade-in and fade-out to avoid clicks."""
    n = max(1, int(fade_ms * sr / 1000))
    n = min(n, len(y) // 4)  # never exceed a quarter of the clip
    out = y.copy()
    out[:n] *= np.linspace(0.0, 1.0, n)
    out[-n:] *= np.linspace(1.0, 0.0, n)
    return out


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    """Normalise, fade, and write a 16-bit WAV file (for one-shots / pitched samples)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = peak_normalize(apply_fade(y, sr=sr))
    sf.write(str(path), out.astype(np.float32), sr, subtype="PCM_16")


def write_wav_raw(path: Path, y: np.ndarray, sr: int = SR, edge_fade_ms: int = 2) -> None:
    """
    Write a WAV file preserving the original amplitude (no peak normalisation).

    Only a short linear edge fade (default 2 ms) is applied to avoid boundary
    clicks at loop seams.  Use this for loop slices so that inter-stem balance
    and per-bar dynamics are kept intact for faithful mix reconstruction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(edge_fade_ms * sr / 1000))
    n = min(n, len(y) // 4)
    out = y.copy()
    out[:n]  *= np.linspace(0.0, 1.0, n)
    out[-n:] *= np.linspace(1.0, 0.0, n)
    sf.write(str(path), out.astype(np.float32), sr, subtype="PCM_16")


def midi_to_note_name(midi: int) -> str:
    """Convert a MIDI note number to a Strudel note name (e.g. 60 → 'c4')."""
    names = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
    octave = (midi // 12) - 1
    return f"{names[midi % 12]}{octave}"


def estimate_bpm(y: np.ndarray, sr: int = SR) -> float:
    """Estimate BPM via librosa beat tracker."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0])
    return float(tempo)


# ---------------------------------------------------------------------------
# Drum onset detection and classification
# ---------------------------------------------------------------------------

def _extract_window(y: np.ndarray, onset_sample: int, window_samples: int) -> np.ndarray:
    """Return a fixed-length window at onset_sample, zero-padded at tail if needed."""
    end = onset_sample + window_samples
    if end <= len(y):
        return y[onset_sample:end].copy()
    chunk = y[onset_sample:].copy()
    return np.pad(chunk, (0, end - len(y)))


def _compute_onset_features(
    y: np.ndarray,
    sr: int,
    onset_samples: np.ndarray,
    window_samples: int,
) -> list[dict]:
    """
    Compute spectral features for every onset window.

    Returns a list of feature dicts (silent onsets are filtered out).
    Values are raw — thresholds applied in ``_classify_onsets``.
    """
    features: list[dict] = []
    for onset in onset_samples:
        win = _extract_window(y, int(onset), window_samples)
        if float(np.max(np.abs(win))) < 1e-6:
            continue  # silent window — skip

        fft_mag = np.abs(np.fft.rfft(win))
        freqs   = np.fft.rfftfreq(len(win), d=1.0 / sr)
        total_energy = float(np.sum(fft_mag ** 2)) + 1e-12

        low_mask  = freqs < 150
        high_mask = freqs >= 5000

        low_ratio  = float(np.sum(fft_mag[low_mask]  ** 2) / total_energy)
        high_ratio = float(np.sum(fft_mag[high_mask] ** 2) / total_energy)
        centroid   = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12))
        zcr        = float(librosa.feature.zero_crossing_rate(win)[0].mean())
        rms        = float(np.sqrt(np.mean(win ** 2)))

        # Rough decay proxy: tail-RMS / head-RMS within window
        half = len(win) // 2
        head_rms = float(np.sqrt(np.mean(win[:half] ** 2))) + 1e-12
        tail_rms = float(np.sqrt(np.mean(win[half:] ** 2)))
        decay_ratio = tail_rms / head_rms  # high = longer sustain → open hat

        features.append({
            "onset":       int(onset),
            "low_ratio":   low_ratio,
            "high_ratio":  high_ratio,
            "centroid":    centroid,
            "zcr":         zcr,
            "rms":         rms,
            "decay_ratio": decay_ratio,
        })
    return features


def _classify_onsets(features: list[dict]) -> dict[str, list[dict]]:
    """
    Classify onset features into kick/snare/hh/oh using percentile-derived
    thresholds relative to this track's own distributions.

    Rules (all cuts derived from this set — not hardcoded):
      - bd  : low_ratio  > P65  AND  centroid  < P40
      - oh  : high_ratio > P60  AND  decay_ratio > P65
      - hh  : high_ratio > P60  (but decay_ratio ≤ P65)
      - sd  : everything else
    """
    classes: dict[str, list[dict]] = {"bd": [], "sd": [], "hh": [], "oh": []}
    if not features:
        return classes

    arr_low    = np.array([f["low_ratio"]   for f in features])
    arr_cent   = np.array([f["centroid"]    for f in features])
    arr_high   = np.array([f["high_ratio"]  for f in features])
    arr_decay  = np.array([f["decay_ratio"] for f in features])

    th_low   = float(np.percentile(arr_low,   65))
    th_cent  = float(np.percentile(arr_cent,  40))
    th_high  = float(np.percentile(arr_high,  60))
    th_decay = float(np.percentile(arr_decay, 65))

    for f in features:
        is_kick = f["low_ratio"] > th_low and f["centroid"] < th_cent
        is_high = f["high_ratio"] > th_high

        if is_kick and not is_high:
            classes["bd"].append(f)
        elif is_high and f["decay_ratio"] > th_decay:
            classes["oh"].append(f)
        elif is_high:
            classes["hh"].append(f)
        else:
            classes["sd"].append(f)

    return classes


def _pick_representative(
    group: list[dict],
    y: np.ndarray,
    window_samples: int,
) -> Optional[np.ndarray]:
    """
    Return the median-energy onset from *group* as the representative sample.
    Median energy is a robust choice — avoids both the quietest (bleed) and
    loudest (clipping) examples.
    """
    if not group:
        return None
    rms_arr = np.array([f["rms"] for f in group])
    idx = int(np.argsort(rms_arr)[len(rms_arr) // 2])
    return _extract_window(y, group[idx]["onset"], window_samples)


def build_drums(stems_dir: Path, out_dir: Path) -> dict[str, str]:
    """
    Detect drum onsets in drums.wav, classify them, and write one representative
    WAV per class into ``out_dir/drums/``.

    Returns a dict suitable for strudel.json: ``{"bd": "drums/bd.wav", ...}``.
    """
    drums_path = stems_dir / "drums.wav"
    if not drums_path.exists():
        print("[drums] drums.wav not found — skipping", file=sys.stderr)
        return {}

    print("[drums] loading …", file=sys.stderr)
    y, sr = load_mono(drums_path)

    window_samples = int(0.250 * sr)  # 250 ms onset window

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="samples", backtrack=True)
    print(f"[drums] {len(onset_frames)} onsets detected", file=sys.stderr)

    if len(onset_frames) < 4:
        print("[drums] too few onsets — skipping", file=sys.stderr)
        return {}

    features = _compute_onset_features(y, sr, onset_frames, window_samples)
    classes  = _classify_onsets(features)

    counts = {k: len(v) for k, v in classes.items()}
    print(f"[drums] classification counts: {counts}", file=sys.stderr)

    drum_out = out_dir / "drums"
    drum_out.mkdir(parents=True, exist_ok=True)

    out_map: dict[str, str] = {}
    for label, group in classes.items():
        sample = _pick_representative(group, y, window_samples)
        if sample is None:
            continue
        fpath = drum_out / f"{label}.wav"
        write_wav(fpath, sample, sr=sr)
        out_map[label] = f"drums/{label}.wav"
        print(f"[drums]   {label}.wav  ({len(group)} onsets)", file=sys.stderr)

    return out_map


# ---------------------------------------------------------------------------
# Pitched multi-sampling (shared by bass and melodic)
# ---------------------------------------------------------------------------

def _extract_pitched_samples(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    min_duration_sec: float = 0.15,
    sample_dur_sec: float = 0.45,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> dict[int, np.ndarray]:
    """
    Run pyin pitch detection and return one audio excerpt per distinct MIDI pitch.

    Picks the longest sustained segment for each note and extracts up to
    *sample_dur_sec* from its stable interior (skipping the attack transient).

    Returns:
        ``{midi_note: audio_chunk}``
    """
    min_frames = max(1, int(min_duration_sec * sr / hop_length))

    pitches, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    voiced = np.where(voiced_flag, pitches, np.nan)

    # Segment consecutive voiced frames sharing the same MIDI pitch
    segments: list[tuple[int, int, int]] = []  # (start_frame, end_frame, midi)
    i = 0
    while i < len(voiced):
        if np.isnan(voiced[i]):
            i += 1
            continue
        seg_start = i
        midi_ref   = int(round(12 * np.log2(voiced[i] / 440) + 69))
        while i < len(voiced) and not np.isnan(voiced[i]):
            midi_cur = int(round(12 * np.log2(voiced[i] / 440) + 69))
            if midi_cur != midi_ref:
                break
            i += 1
        seg_end = i
        if (seg_end - seg_start) >= min_frames:
            segments.append((seg_start, seg_end, midi_ref))

    # Keep the longest segment per MIDI pitch
    best: dict[int, tuple[int, int]] = {}
    for s_start, s_end, midi in segments:
        length = s_end - s_start
        if midi not in best or length > (best[midi][1] - best[midi][0]):
            best[midi] = (s_start, s_end)

    sample_samples = int(sample_dur_sec * sr)
    result: dict[int, np.ndarray] = {}

    for midi, (sf_s, sf_e) in best.items():
        onset   = librosa.frames_to_samples(sf_s, hop_length=hop_length)
        seg_len = librosa.frames_to_samples(sf_e - sf_s, hop_length=hop_length)
        # Skip the first 50 ms of the segment to avoid attack artefacts
        offset  = min(int(0.05 * sr), max(0, seg_len - sample_samples))
        start   = onset + offset
        end     = min(len(y), start + sample_samples)
        start   = max(0, end - sample_samples)
        chunk   = y[start:end]
        if len(chunk) < int(0.05 * sr):
            continue
        result[midi] = chunk

    return result


def build_bass(stems_dir: Path, out_dir: Path) -> tuple[dict[str, str], Optional[str]]:
    """
    Build pitched bass samples from bass.wav.

    Returns:
        (strudel_pitched_map, representative_rel_path)
        ``strudel_pitched_map``: ``{"c2": "bass/bass_<midi>.wav", ...}``
    """
    bass_path = stems_dir / "bass.wav"
    if not bass_path.exists():
        print("[bass] bass.wav not found — skipping", file=sys.stderr)
        return {}, None

    print("[bass] loading …", file=sys.stderr)
    y, sr = load_mono(bass_path)

    samples = _extract_pitched_samples(y, sr, fmin=30.0, fmax=400.0)
    print(f"[bass] {len(samples)} distinct pitches: {sorted(samples)}", file=sys.stderr)

    bass_out = out_dir / "bass"
    bass_out.mkdir(parents=True, exist_ok=True)

    strudel_map: dict[str, str] = {}
    for midi, chunk in sorted(samples.items()):
        note_name = midi_to_note_name(midi)
        fname     = f"bass_{midi}.wav"
        write_wav(bass_out / fname, chunk, sr=sr)
        strudel_map[note_name] = f"bass/{fname}"
        print(f"[bass]   {fname}  ({note_name})", file=sys.stderr)

    rep_path: Optional[str] = None
    if samples:
        # Pick median-index pitch as representative
        keys_sorted = sorted(samples.keys())
        rep_midi    = keys_sorted[len(keys_sorted) // 2]
        write_wav(bass_out / "bass.wav", samples[rep_midi], sr=sr)
        rep_path = "bass/bass.wav"
        print(f"[bass]   bass.wav (representative, midi {rep_midi})", file=sys.stderr)

    return strudel_map, rep_path


def build_melodic(stems_dir: Path, out_dir: Path) -> tuple[dict[str, str], Optional[str]]:
    """
    Build pitched melodic samples from melodic.wav.

    Falls back to a 2-second excerpt when no pitched segments are found
    (e.g. dense polyphonic content that defeats pyin).

    Returns:
        (strudel_pitched_map, representative_rel_path)
    """
    mel_path = stems_dir / "melodic.wav"
    if not mel_path.exists():
        print("[melodic] melodic.wav not found — skipping", file=sys.stderr)
        return {}, None

    print("[melodic] loading …", file=sys.stderr)
    y, sr = load_mono(mel_path)

    samples = _extract_pitched_samples(y, sr, fmin=80.0, fmax=2000.0)
    print(f"[melodic] {len(samples)} distinct pitches: {sorted(samples)}", file=sys.stderr)

    mel_out = out_dir / "melodic"
    mel_out.mkdir(parents=True, exist_ok=True)

    strudel_map: dict[str, str] = {}
    for midi, chunk in sorted(samples.items()):
        note_name = midi_to_note_name(midi)
        fname     = f"melodic_{midi}.wav"
        write_wav(mel_out / fname, chunk, sr=sr)
        strudel_map[note_name] = f"melodic/{fname}"
        print(f"[melodic]   {fname}  ({note_name})", file=sys.stderr)

    rep_path: Optional[str] = None
    if samples:
        keys_sorted = sorted(samples.keys())
        rep_midi    = keys_sorted[len(keys_sorted) // 2]
        write_wav(mel_out / "melodic.wav", samples[rep_midi], sr=sr)
        rep_path = "melodic/melodic.wav"
        print(f"[melodic]   melodic.wav (representative, midi {rep_midi})", file=sys.stderr)
    else:
        # Fallback: 2-second excerpt from the first quarter of the track
        excerpt_len = min(int(2.0 * sr), len(y))
        start       = min(len(y) // 4, max(0, len(y) - excerpt_len))
        chunk       = y[start: start + excerpt_len]
        write_wav(mel_out / "melodic.wav", chunk, sr=sr)
        rep_path = "melodic/melodic.wav"
        print("[melodic]   melodic.wav (fallback 2s excerpt — no pitched segments found)", file=sys.stderr)

    return strudel_map, rep_path


# ---------------------------------------------------------------------------
# Beat-aligned loop slicing
# ---------------------------------------------------------------------------

def build_loops(
    stems_dir: Path,
    out_dir: Path,
    bpm: float,
    num_bars: int,
) -> dict[str, list[str]]:
    """
    Slice each stem at bar boundaries for the first *num_bars* bars (4/4 assumed).

    Each slice is EXACTLY ``bar_samples`` samples long (zero-padded if needed)
    so that Strudel can loop them frame-accurately at matching cps.

    Returns strudel loop map, e.g.::

        {"drumloop": ["loops/drums_bar00.wav", ...], ...}
    """
    bar_duration_sec = (60.0 / bpm) * 4.0
    bar_samples      = int(round(bar_duration_sec * SR))

    loop_out = out_dir / "loops"
    loop_out.mkdir(parents=True, exist_ok=True)

    # Include vocals if present — important for genres where vocals are prominent
    stem_configs = [
        ("drums.wav",   "drums"),
        ("bass.wav",    "bass"),
        ("melodic.wav", "melodic"),
        ("vocals.wav",  "vocals"),
    ]

    result: dict[str, list[str]] = {}

    for stem_file, stem_label in stem_configs:
        stem_path = stems_dir / stem_file
        if not stem_path.exists():
            continue

        print(f"[loops] slicing {stem_file} …", file=sys.stderr)
        y, sr = load_mono(stem_path)

        available_bars = int(len(y) / bar_samples)
        n_bars         = min(num_bars, available_bars)

        if n_bars == 0:
            print(
                f"[loops] {stem_file}: too short "
                f"(need {bar_duration_sec:.2f}s, have {len(y)/sr:.2f}s) — skipping",
                file=sys.stderr,
            )
            continue

        paths: list[str] = []
        for bar_idx in range(n_bars):
            start = bar_idx * bar_samples
            end   = start + bar_samples
            chunk = y[start:end]
            # Guarantee EXACT length (zero-pad if at end of file)
            if len(chunk) < bar_samples:
                chunk = np.pad(chunk, (0, bar_samples - len(chunk)))
            fname = f"{stem_label}_bar{bar_idx:02d}.wav"
            # Write RAW (no peak-normalisation) to preserve original amplitude,
            # dynamics, and inter-stem balance.  Only a 2ms edge fade is applied
            # to avoid click artefacts at loop boundaries.
            write_wav_raw(loop_out / fname, chunk, sr=sr)
            paths.append(f"loops/{fname}")

        result[f"{stem_label}loop"] = paths
        print(
            f"[loops]   {stem_label}: {n_bars} bars × {bar_duration_sec:.3f}s",
            file=sys.stderr,
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Strudel-playable sample pack from Demucs stems."
    )
    parser.add_argument("--stems-dir", required=True,
                        help="Directory containing melodic.wav, bass.wav, drums.wav")
    parser.add_argument("--out", required=True,
                        help="Output pack directory (created if absent)")
    parser.add_argument("--bpm", type=float, default=None,
                        help="BPM (auto-detected from drums.wav if omitted)")
    parser.add_argument("--key", type=str, default=None,
                        help="Musical key, stored in pack.json (informational)")
    parser.add_argument("--num-bars", type=int, default=16,
                        help="Number of loop bars to slice (default: 16)")
    args = parser.parse_args()

    stems_dir = Path(args.stems_dir)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- BPM ---
    bpm = args.bpm
    if bpm is None:
        for probe in ["drums.wav", "melodic.wav", "bass.wav"]:
            probe_path = stems_dir / probe
            if probe_path.exists():
                print(f"[bpm] detecting from {probe} …", file=sys.stderr)
                y_probe, sr_probe = load_mono(probe_path)
                bpm = estimate_bpm(y_probe, sr_probe)
                print(f"[bpm] detected {bpm:.1f} BPM", file=sys.stderr)
                break
        if bpm is None:
            bpm = 120.0
            print("[bpm] fallback to 120.0 BPM", file=sys.stderr)

    bar_duration_sec = (60.0 / bpm) * 4.0
    print(f"\n=== BPM={bpm:.1f}  bar={bar_duration_sec:.3f}s ===\n", file=sys.stderr)

    # --- Build sections ---
    drum_map              = build_drums(stems_dir, out_dir)
    bass_pitched, bass_rep = build_bass(stems_dir, out_dir)
    mel_pitched,  mel_rep  = build_melodic(stems_dir, out_dir)
    loop_map              = build_loops(stems_dir, out_dir, bpm=bpm, num_bars=args.num_bars)

    # --- strudel.json ---
    strudel: dict = {}

    # Drum one-shots (each a plain path string)
    strudel.update(drum_map)

    # Pitched maps (Strudel note-keyed dicts)
    if bass_pitched:
        strudel["trackbass"] = bass_pitched
    elif bass_rep:
        strudel["bass"] = bass_rep

    if mel_pitched:
        strudel["tracklead"] = mel_pitched
    elif mel_rep:
        strudel["lead"] = mel_rep

    # Loop arrays
    strudel.update(loop_map)

    strudel_path = out_dir / "strudel.json"
    strudel_path.write_text(json.dumps(strudel, indent=2))
    print(f"\n[strudel.json] keys: {list(strudel.keys())}", file=sys.stderr)

    # --- pack.json ---
    files_written = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.wav"))
    files_written += ["strudel.json", "pack.json"]

    pack_meta = {
        "bpm":              round(bpm, 2),
        "key":              args.key,
        "num_bars":         args.num_bars,
        "bar_duration_sec": round(bar_duration_sec, 4),
        "sample_rate":      SR,
        "counts": {
            "drums":            len(drum_map),
            "bass_pitches":     len(bass_pitched),
            "melodic_pitches":  len(mel_pitched),
            "loop_bars":        {k: len(v) for k, v in loop_map.items()},
        },
        "note": (
            f"Sample pack from Demucs stems. BPM={bpm:.1f}. "
            f"Drums: {list(drum_map)}. "
            f"Bass pitches: {list(bass_pitched)}. "
            f"Melodic pitches: {list(mel_pitched)}."
        ),
    }
    (out_dir / "pack.json").write_text(json.dumps(pack_meta, indent=2))

    # --- stdout summary ---
    summary = {
        "out_dir":          str(out_dir.resolve()),
        "bpm":              round(bpm, 2),
        "files_written":    files_written,
        "strudel_json_path": str(strudel_path.resolve()),
        "sounds":           list(strudel.keys()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
