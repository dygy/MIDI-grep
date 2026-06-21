"""Sound timbre resolver for Strudel voice matching.

Provides:
  - TIMBRE_TABLE  : static curated (brightness, warmth, attack) for every sound
                    in GENRE_PALETTES non-drum roles. Values are 0-1 floats based
                    on each instrument's well-known spectral character.
  - analyze_stem_timbre(path) -> dict  : librosa-based per-stem measurement
  - analyze_stem_timbre_by_section(path, sections, total_cycles, cps)
                      -> list[float]   : per-section brightness [0-1], one per section
  - resolve_sound(stem_path, candidates) -> str : pick candidate nearest to stem

Design note: brightness is normalised spectral centroid (high = 1), warmth is
low/high energy ratio (high = 1 means bass-heavy), attack is onset sharpness
(high = 1 means percussive/fast). Same three axes are used in the static table.
"""
from __future__ import annotations

import math
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Static timbre feature table — (brightness, warmth, attack)
# Sources: General MIDI spec + acoustic instrument knowledge.
# Default for unknown sounds: (0.5, 0.4, 0.5).
# ---------------------------------------------------------------------------

TIMBRE_TABLE: dict[str, tuple[float, float, float]] = {
    # --- Waveforms ---
    "sine":           (0.15, 0.80, 0.30),
    "triangle":       (0.20, 0.70, 0.35),
    "square":         (0.55, 0.40, 0.50),
    "sawtooth":       (0.70, 0.35, 0.55),
    "supersaw":       (0.80, 0.30, 0.60),

    # --- GM Bass ---
    "gm_acoustic_bass":       (0.25, 0.85, 0.45),
    "gm_electric_bass_finger":(0.30, 0.80, 0.50),
    "gm_electric_bass_pick":  (0.40, 0.70, 0.65),
    "gm_fretless_bass":       (0.28, 0.82, 0.40),
    "gm_slap_bass_1":         (0.50, 0.65, 0.80),
    "gm_synth_bass_1":        (0.45, 0.60, 0.55),
    "gm_synth_bass_2":        (0.40, 0.65, 0.50),
    "gm_contrabass":          (0.20, 0.90, 0.35),
    "gm_tuba":                (0.30, 0.80, 0.40),
    "gm_lead_8_bass_lead":    (0.60, 0.40, 0.60),

    # --- GM Strings / Ensemble ---
    "gm_violin":              (0.65, 0.30, 0.40),
    "gm_cello":               (0.40, 0.55, 0.35),
    "gm_string_ensemble_1":   (0.50, 0.50, 0.35),
    "gm_string_ensemble_2":   (0.55, 0.45, 0.35),
    "gm_synth_strings_1":     (0.60, 0.40, 0.45),
    "gm_choir_aahs":          (0.45, 0.55, 0.25),
    "gm_voice_oohs":          (0.40, 0.55, 0.25),

    # --- GM Pads ---
    "gm_pad_new_age":   (0.50, 0.50, 0.15),
    "gm_pad_warm":      (0.35, 0.65, 0.15),
    "gm_pad_poly":      (0.55, 0.45, 0.20),
    "gm_pad_choir":     (0.45, 0.55, 0.20),
    "gm_pad_bowed":     (0.40, 0.55, 0.15),
    "gm_pad_metallic":  (0.70, 0.30, 0.35),
    "gm_pad_halo":      (0.60, 0.40, 0.15),
    "gm_pad_sweep":     (0.55, 0.45, 0.10),

    # --- GM Synth Leads ---
    "gm_lead_1_square":    (0.55, 0.40, 0.55),
    "gm_lead_2_sawtooth":  (0.72, 0.30, 0.60),
    "gm_lead_5_charang":   (0.75, 0.25, 0.65),
    "gm_lead_6_voice":     (0.40, 0.50, 0.30),
    "gm_lead_7_fifths":    (0.65, 0.45, 0.55),

    # --- GM Brass / Reed ---
    "gm_trumpet":         (0.78, 0.25, 0.70),
    "gm_trombone":        (0.60, 0.40, 0.55),
    "gm_french_horn":     (0.55, 0.45, 0.45),
    "gm_synth_brass_1":   (0.72, 0.30, 0.65),
    "gm_alto_sax":        (0.68, 0.35, 0.55),
    "gm_tenor_sax":       (0.60, 0.40, 0.55),
    "gm_clarinet":        (0.58, 0.35, 0.50),
    "gm_oboe":            (0.65, 0.30, 0.50),
    "gm_bassoon":         (0.40, 0.55, 0.40),

    # --- GM Pipe ---
    "gm_flute":           (0.65, 0.25, 0.50),
    "gm_piccolo":         (0.80, 0.15, 0.55),

    # --- GM FX ---
    "gm_fx_atmosphere":   (0.40, 0.50, 0.10),
    "gm_fx_crystal":      (0.80, 0.20, 0.75),
    "gm_fx_echoes":       (0.50, 0.40, 0.20),
    "gm_fx_sci_fi":       (0.60, 0.35, 0.30),

    # --- GM Piano / Keys ---
    "gm_piano":           (0.60, 0.45, 0.75),
    "gm_epiano1":         (0.55, 0.45, 0.70),
    "gm_harpsichord":     (0.70, 0.30, 0.85),
    "gm_drawbar_organ":   (0.50, 0.55, 0.30),
    "gm_rock_organ":      (0.60, 0.45, 0.35),

    # --- GM Chromatic Perc ---
    "gm_glockenspiel":    (0.88, 0.10, 0.90),
    "gm_vibraphone":      (0.75, 0.20, 0.80),
    "gm_marimba":         (0.65, 0.30, 0.85),
    "gm_celesta":         (0.85, 0.10, 0.85),
    "gm_music_box":       (0.82, 0.10, 0.80),
    "gm_tubular_bells":   (0.78, 0.15, 0.75),
    "gm_kalimba":         (0.72, 0.25, 0.85),
    "gm_woodblock":       (0.75, 0.10, 0.95),

    # --- GM Guitar ---
    "gm_acoustic_guitar_nylon": (0.55, 0.40, 0.70),
    "gm_electric_guitar_clean": (0.65, 0.30, 0.75),
    "gm_electric_guitar_muted": (0.60, 0.25, 0.85),
    "gm_overdriven_guitar":     (0.75, 0.30, 0.70),
    "gm_distortion_guitar":     (0.80, 0.35, 0.65),

    # --- GM Harmonica ---
    "gm_harmonica":       (0.60, 0.40, 0.55),
}

# Default for any sound not in the table.
_DEFAULT_TIMBRE: tuple[float, float, float] = (0.50, 0.40, 0.50)


def _get_timbre(sound: str) -> tuple[float, float, float]:
    return TIMBRE_TABLE.get(sound, _DEFAULT_TIMBRE)


def _euclidean(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Stem analysis
# ---------------------------------------------------------------------------

def analyze_stem_timbre(path: str) -> dict[str, float]:
    """Measure (brightness, warmth, attack) for an audio file via librosa.

    Returns a dict with keys 'brightness', 'warmth', 'attack' in [0, 1].
    Falls back to the default timbre on any error (missing file, import fail, etc.).
    """
    default = {"brightness": 0.5, "warmth": 0.4, "attack": 0.5}
    if not path or not os.path.exists(path):
        return default
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=22050, mono=True, duration=30.0)
        if y is None or len(y) < sr * 0.5:  # less than 0.5s — not enough signal
            return default

        # Brightness: normalised mean spectral centroid.
        # Centroid range roughly 80–8000 Hz; normalise to [0,1].
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(np.clip(np.mean(centroid) / 6000.0, 0.0, 1.0))

        # Warmth: ratio of low-band energy (0–400 Hz) to total energy.
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs <= 400
        high_mask = freqs > 400
        low_energy = float(np.sum(S[low_mask] ** 2)) + 1e-9
        high_energy = float(np.sum(S[high_mask] ** 2)) + 1e-9
        warmth = float(np.clip(low_energy / (low_energy + high_energy) * 3.5, 0.0, 1.0))

        # Attack: mean onset strength normalised by peak (percussive = high value).
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) > 0:
            attack = float(np.clip(np.mean(onset_env) / (np.max(onset_env) + 1e-9), 0.0, 1.0))
        else:
            attack = 0.5

        return {"brightness": brightness, "warmth": warmth, "attack": attack}
    except Exception:  # noqa: BLE001 — librosa not installed, file unreadable, etc.
        return default


def analyze_stem_timbre_by_section(
    path: str,
    sections: list,
    total_cycles: int,
    cps: float,
) -> list[float]:
    """Return one brightness value (0-1) per section by windowing the stem over time.

    Each section's window length is proportional to its ``cycles`` field:
        window_seconds = section_cycles / cps

    Args:
        path:          Absolute path to the audio stem (wav/mp3).
        sections:      List of section dicts, each with a ``cycles`` key (int).
        total_cycles:  Total cycles that span the whole track.
        cps:           Cycles per second (bpm / 60 / 4).

    Returns:
        A list[float] of the same length as ``sections``, each brightness in [0, 1].
        Returns [] (empty) on any error so the caller can fall back gracefully.
    """
    if not sections or total_cycles <= 0 or cps <= 0:
        return []
    if not path or not os.path.exists(path):
        return []
    try:
        import librosa
        import numpy as np

        # Load full stem (mono, 22050 Hz).  Cap at 600 s to avoid OOM on long tracks.
        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []

        results: list[float] = []
        cursor = 0.0  # running start-time cursor (seconds)

        for sec in sections:
            cyc = max(1, int(sec.get("cycles", 1)))
            window_sec = cyc / cps

            start_s = cursor
            end_s = cursor + window_sec
            cursor = end_s

            # Convert to sample indices, clamp to file length.
            start_i = int(start_s * sr)
            end_i = int(end_s * sr)
            start_i = min(start_i, len(y))
            end_i = min(end_i, len(y))

            segment = y[start_i:end_i]
            if len(segment) < sr * 0.1:
                # Window is too short (file ended early or section is tiny) — use full-file.
                segment = y

            # Brightness: normalised mean spectral centroid, same scale as analyze_stem_timbre.
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            brightness = float(np.clip(np.mean(centroid) / 6000.0, 0.0, 1.0))
            results.append(brightness)

        return results
    except Exception:  # noqa: BLE001 — librosa not available, file error, etc.
        return []


# A stem is "active" in a window if its loudest frame clears the noise floor.
_ACTIVE_PEAK = 0.012


def analyze_stem_activity_by_section(
    path: str,
    sections: list,
    total_cycles: int,
    cps: float,
) -> list[dict]:
    """Per-section activity of an original stem, for DATA-DRIVEN arrangement.

    For each section's time window (window_seconds = cycles / cps) returns:
        {"active": bool,    # is the stem actually playing here?
         "gain":   float,   # relative loudness in [0.15, 0.95] (this window's RMS vs the stem's max)
         "density":float}   # onset rate in [0,1] (how busy the part is here)

    This lets the orchestrator match the ORIGINAL's real arrangement (which voices play when, how
    loud, how busy) instead of imposing a generic intro→drop→outro template. Returns [] on any error.
    """
    if not sections or total_cycles <= 0 or cps <= 0 or not path or not os.path.exists(path):
        return []
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []

        # Reference loudness for relative gain: the loudest window-sized RMS across the stem.
        full_env = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        ref_peak = float(np.max(full_env)) if len(full_env) else 0.0
        if ref_peak < 1e-9:
            return []

        results: list[dict] = []
        cursor = 0.0
        for sec in sections:
            cyc = max(1, int(sec.get("cycles", 1)))
            window_sec = cyc / cps
            start_i = min(int(cursor * sr), len(y))
            end_i = min(int((cursor + window_sec) * sr), len(y))
            cursor += window_sec
            seg = y[start_i:end_i]
            if len(seg) < sr * 0.1:
                results.append({"active": False, "gain": 0.2, "density": 0.0})
                continue
            env = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
            peak = float(np.max(env)) if len(env) else 0.0
            seg_rms = float(np.sqrt(np.mean(seg ** 2)))
            active = peak >= _ACTIVE_PEAK
            # Relative loudness vs the stem's own loudest part → the original's dynamic envelope.
            gain = float(np.clip(seg_rms / ref_peak, 0.15, 0.95))
            onsets = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
            dur = max(0.5, len(seg) / sr)
            density = float(np.clip((len(onsets) / dur) / 6.0, 0.0, 1.0))  # ~6 onsets/s = full
            results.append({"active": active, "gain": round(gain, 3), "density": round(density, 3)})
        return results
    except Exception:  # noqa: BLE001
        return []


def analyze_stem_envelope(path: str, n_steps: int) -> list[float]:
    """Sample the original stem's loudness envelope into n_steps gain values (0-1).

    Used as a per-voice gain-automation pattern so the RENDERED stem rises/falls WHEN the original
    does — the thing that makes the stem-shape self-test (envelope correlation) pass. Returns [] on
    error. Values are scaled so the loudest step ~= 0.95 and silence ~= 0.0 (preserving the SHAPE).
    """
    if not path or not os.path.exists(path) or n_steps < 1:
        return []
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []
        env = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(env) == 0:
            return []
        # Resample the envelope to exactly n_steps points (mean over each chunk).
        idx = np.linspace(0, len(env), n_steps + 1).astype(int)
        steps = [float(env[idx[i]:max(idx[i] + 1, idx[i + 1])].mean()) for i in range(n_steps)]
        peak = max(steps) or 1.0
        # Normalise to shape: loudest → ~0.95, scale the rest proportionally, floor tiny values to 0.
        out = []
        for s in steps:
            g = s / peak * 0.95
            out.append(round(g if g >= 0.05 else 0.0, 3))
        return out
    except Exception:  # noqa: BLE001
        return []


def analyze_drum_density_per_cycle(path: str, total_cycles: int, cps: float) -> list[float]:
    """Per-CYCLE onset density of the original drums, in [0,1] (≈6 onsets/s = 1.0).

    The per-cycle GAIN envelope (analyze_stem_envelope) makes our drums get louder/quieter when the
    original does, but a steady `bd hh sd hh` has near-uniform onset energy regardless of gain, so it
    can't track the original's fills-vs-breaks DENSITY. This returns one busyness value per cycle so
    the assembler can pick a sparser/denser drum bar per cycle — the per-cycle DENSITY lever that the
    volume envelope alone can't provide. Returns [] on error (caller falls back to per-section density).
    """
    if not path or not os.path.exists(path) or total_cycles < 1 or cps <= 0:
        return []
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []
        # Onset times once, then bucket into per-cycle windows (cycle = 1/cps seconds).
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        cycle_sec = 1.0 / cps
        out = []
        for c in range(total_cycles):
            t0, t1 = c * cycle_sec, (c + 1) * cycle_sec
            n = int(np.sum((onset_times >= t0) & (onset_times < t1)))
            # onsets-per-second / 6 → [0,1]; a cycle is ~cycle_sec long.
            density = float(np.clip((n / max(0.25, cycle_sec)) / 6.0, 0.0, 1.0))
            out.append(round(density, 3))
        return out
    except Exception:  # noqa: BLE001
        return []


def analyze_stem_drum_pattern(path: str, sections: list, total_cycles: int, cps: float,
                              steps_per_cycle: int = 16) -> list:
    """DATA-DRIVE DRUMS (A1 for drums): transcribe the ORIGINAL drums' ACTUAL hit pattern — detect
    each onset, classify it (bd/sd/hh/oh/cp by spectral band), and place it on a per-cycle step grid —
    instead of the generic density-tier templates. Lands hits WHERE the original does, so the rendered
    drums track the original's rhythm/shape (the drum analog of A1's actual-notes win for bass/lead).

    Returns one pattern per section, each `<[bd hh sd hh …] …]>` (one bar per cycle, steps_per_cycle
    steps). A step with no onset is a rest; with one or more, the LOUDEST hit's class. [] on error.
    """
    if not path or not os.path.exists(path) or total_cycles < 1 or cps <= 0 or not sections:
        return []
    try:
        import librosa
        import numpy as np
        from detect_drums import classify_drum_hit
        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []
        hop = 512
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
        # classify + amplitude for each onset
        hits = []  # (time, type, strength)
        for s, t in zip(onset_samples, onset_times):
            dtype, conf = classify_drum_hit(y, sr, int(s), hop)
            amp = float(np.max(np.abs(y[int(s):int(s) + int(0.05 * sr)])) or 0.0)
            hits.append((float(t), dtype, amp))

        cycle_sec = 1.0 / cps
        step_sec = cycle_sec / steps_per_cycle
        total_steps = total_cycles * steps_per_cycle
        grid = ["~"] * total_steps
        best = [-1.0] * total_steps   # keep the loudest hit per step
        for t, dtype, amp in hits:
            idx = int(t / step_sec)
            if 0 <= idx < total_steps and amp > best[idx]:
                best[idx] = amp
                grid[idx] = dtype

        patterns = []
        cursor = 0
        for sec in sections:
            n = max(1, int(sec.get("cycles", 1) or 1))
            bars = []
            for k in range(n):
                start = (cursor + k) * steps_per_cycle
                bars.append("[" + " ".join(grid[start:start + steps_per_cycle]) + "]")
            cursor += n
            patterns.append("<" + " ".join(bars) + ">")
        return patterns
    except Exception:  # noqa: BLE001
        return []


# Flat spelling to match the codebase (ab4 / eb4 / bb4 …), index = pitch class 0..11
_PC_NAMES = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]


def _rle_collapse(steps: list) -> str:
    """Run-length-collapse a bar's steps into `tok@count` for sustained (legato) notes — one attack
    held across N steps instead of N re-triggers, so the loudness envelope stays full like the
    original's sustained texture instead of choppy hit+decay."""
    out = []
    i = 0
    while i < len(steps):
        j = i
        while j < len(steps) and steps[j] == steps[i]:
            j += 1
        run = j - i
        out.append(steps[i] if run == 1 else f"{steps[i]}@{run}")
        i = j
    return "[" + " ".join(out) + "]"


def analyze_stem_pitch_by_section(
    path: str,
    sections: list,
    total_cycles: int,
    cps: float,
    octave: int,
    steps_per_cycle: int = 8,
    sustain: bool = False,
) -> list:
    """DATA-DRIVE PITCH (A1): transcribe the ORIGINAL stem's melody and emit Strudel note patterns —
    one pattern string per section, each a per-cycle sequence `<[step step …] …]>` (one bar per cycle,
    `steps_per_cycle` steps per bar). So the rendered voice plays the track's ACTUAL notes.

    Pitch is tracked with librosa.pyin (monophonic f0). Each step takes the median voiced f0 → nearest
    pitch CLASS → placed in the target octave (chroma is octave-invariant).

    sustain=False (BASS): unvoiced steps → rests (the bass is sparse/rhythmic; rests match it).
    sustain=True (LEAD, A1 v2): the melodic stem is POLYPHONIC, so pyin marks sustained chords as
      "unvoiced" — emitting rests there (v1) gutted the loudness envelope (melodic silence 0.23→0.46,
      shape 0.64→0.21). v2 instead distinguishes by RMS: a step is a REST only when the original is
      genuinely SILENT (RMS below floor); when sound is present but pyin found no pitch, it HOLDS the
      previous note. Consecutive equal steps are then RLE-collapsed to a single sustained note (@N).
      This keeps the envelope full while still hitting the right pitch classes.

    Returns [] on any error so the caller keeps the LLM patterns.
    """
    if not path or not os.path.exists(path) or total_cycles < 1 or cps <= 0 or not sections:
        return []
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=22050, mono=True, duration=600.0)
        if y is None or len(y) < sr * 0.3:
            return []
        fmin = float(librosa.note_to_hz(f"C{max(1, octave - 1)}"))
        fmax = float(librosa.note_to_hz(f"C{octave + 2}"))
        hop = 512
        f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        rms_floor = float(np.max(rms)) * 0.10 if len(rms) else 0.0   # below this = genuinely silent

        cycle_sec = 1.0 / cps
        step_sec = cycle_sec / steps_per_cycle
        HOLD = "\x00"  # sentinel: sound present but no pitch → sustain previous note

        def step_token(t0: float, t1: float) -> str:
            mask = (times >= t0) & (times < t1)
            if not np.any(mask):
                return "~"
            seg_f0 = f0[mask]
            seg_v = voiced[mask]
            good = seg_f0[np.isfinite(seg_f0) & (seg_v if seg_v.dtype == bool else seg_v > 0.5)]
            if len(good) >= max(1, int(0.4 * mask.sum())):
                midi = float(librosa.hz_to_midi(float(np.median(good))))
                return f"{_PC_NAMES[int(round(midi)) % 12]}{octave}"
            # no pitch found: rest vs hold depends on whether there's actually sound here
            if sustain:
                seg_rms = rms[mask] if len(rms) else np.array([0.0])
                return HOLD if float(np.mean(seg_rms)) >= rms_floor else "~"
            return "~"

        # Flat token stream across all cycles, then resolve HOLDs left→right (carry last real note).
        total_steps = total_cycles * steps_per_cycle
        toks = [step_token(s * step_sec, (s + 1) * step_sec) for s in range(total_steps)]
        last = "~"
        for i, t in enumerate(toks):
            if t == HOLD:
                toks[i] = last           # sustain the previous note through the unvoiced-but-loud gap
            elif t != "~":
                last = t

        patterns = []
        cursor = 0
        for sec in sections:
            n = max(1, int(sec.get("cycles", 1) or 1))
            bars = []
            for k in range(n):
                start = (cursor + k) * steps_per_cycle
                steps = toks[start:start + steps_per_cycle]
                bars.append(_rle_collapse(steps) if sustain else "[" + " ".join(steps) + "]")
            cursor += n
            patterns.append("<" + " ".join(bars) + ">")
        return patterns
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_sound(stem_path: Optional[str], candidates: list[str]) -> str:
    """Return the candidate whose static timbre vector is euclidean-nearest to the stem.

    Falls back to candidates[0] on any error (empty list, analysis fail, etc.).
    The returned sound is guaranteed to be in the candidates list.
    """
    if not candidates:
        return "gm_synth_bass_1"

    stem_timbre = analyze_stem_timbre(stem_path or "")
    stem_vec = (stem_timbre["brightness"], stem_timbre["warmth"], stem_timbre["attack"])

    best: str = candidates[0]
    best_dist: float = float("inf")
    for c in candidates:
        dist = _euclidean(stem_vec, _get_timbre(c))
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def resolve_drum_bank(drums_path: Optional[str], candidates: list, cache_path: str) -> str:
    """A3 — pick the drum BANK whose tone-colour best matches the ORIGINAL drums (euclidean-nearest
    on brightness/warmth/attack). Drum banks are sample sets, not synths, so their timbre vectors are
    AUDITIONED once (render a fixed pattern per bank → analyze_stem_timbre) and cached track-independently
    in `cache_path` (build it with audition_drum_banks.py). Returns candidates[0] if nothing is cached.

    This is what the calibrated TIMBRE dimension was built to credit: TR808's deep synth tone scored
    near chance vs an acoustic-ish swing kit; matching the bank to the original raises that dimension.
    """
    if not candidates:
        return "RolandTR808"
    try:
        import json
        if not os.path.exists(cache_path):
            return candidates[0]
        cache = json.loads(open(cache_path).read())
        t = analyze_stem_timbre(drums_path or "")
        tgt = (t["brightness"], t["warmth"], t["attack"])
        best, best_d = candidates[0], float("inf")
        for c in candidates:
            v = cache.get(c)
            if not v:
                continue
            d = _euclidean(tgt, (v["brightness"], v["warmth"], v["attack"]))
            if d < best_d:
                best_d, best = d, c
        return best
    except Exception:  # noqa: BLE001
        return candidates[0]
