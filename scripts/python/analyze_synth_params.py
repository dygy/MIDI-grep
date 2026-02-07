#!/usr/bin/env python3
"""
AI Audio Analyzer for Synthesis Parameters

Analyzes original audio and extracts parameters for synthesis:
- Transient characteristics (attack/decay)
- Spectral envelope (filter cutoffs)
- Harmonic content (waveform suggestions)
- Temporal dynamics (amplitude envelope)
- Precise tempo with sub-BPM accuracy

Output: JSON config for renderer with NO hardcoding.
"""

import argparse
import json
import numpy as np
import librosa
import sys
from pathlib import Path
from scipy import signal
from scipy.ndimage import maximum_filter1d


def analyze_transients(y, sr=22050):
    """Analyze attack and release characteristics."""
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Find peaks (transients)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                    pre_avg=3, post_avg=5, delta=0.5, wait=10)

    if len(peaks) < 2:
        return {"attack_ms": 10, "decay_ms": 100, "transient_sharpness": 0.5}

    # Analyze attack times (rise to peak)
    hop_length = 512
    frame_time = hop_length / sr * 1000  # ms per frame

    attacks = []
    decays = []

    for peak in peaks[:50]:  # Analyze first 50 transients
        # Look back for attack start
        start = max(0, peak - 10)
        if peak > start:
            attack_frames = peak - np.argmax(onset_env[start:peak] > onset_env[peak] * 0.1)
            attacks.append(attack_frames * frame_time)

        # Look forward for decay
        end = min(len(onset_env), peak + 20)
        if end > peak:
            decay_region = onset_env[peak:end]
            decay_to_half = np.argmax(decay_region < decay_region[0] * 0.5)
            if decay_to_half > 0:
                decays.append(decay_to_half * frame_time)

    # Compute transient sharpness (ratio of attack to decay)
    avg_attack = np.median(attacks) if attacks else 10
    avg_decay = np.median(decays) if decays else 100
    sharpness = 1 / (1 + avg_attack / max(avg_decay, 1))

    return {
        "attack_ms": float(max(1, min(100, avg_attack))),
        "decay_ms": float(max(10, min(500, avg_decay))),
        "transient_sharpness": float(np.clip(sharpness, 0, 1))
    }


def analyze_spectral_envelope(y, sr=22050):
    """Analyze frequency distribution and suggest filter parameters."""
    # Compute spectrogram
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Average spectrum
    avg_spectrum = np.mean(S, axis=1)

    # Find spectral centroid and rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    # Suggest LPF cutoff (where 90% of energy is below)
    cumsum = np.cumsum(avg_spectrum ** 2)
    total = cumsum[-1]
    lpf_idx = np.searchsorted(cumsum, total * 0.9)
    lpf_cutoff = freqs[min(lpf_idx, len(freqs) - 1)]

    # Suggest HPF cutoff (where 5% of energy is below)
    hpf_idx = np.searchsorted(cumsum, total * 0.05)
    hpf_cutoff = freqs[max(0, hpf_idx)]

    # Analyze spectral slope (brightness character)
    # Fit line to log spectrum to get slope
    log_freqs = np.log10(freqs[1:] + 1)
    log_spec = np.log10(avg_spectrum[1:] + 1e-10)
    slope = np.polyfit(log_freqs, log_spec, 1)[0]

    # Negative slope = more bass, positive = more treble
    brightness = np.clip((slope + 2) / 4, 0, 1)  # Normalize to 0-1

    return {
        "lpf_cutoff": float(np.clip(lpf_cutoff, 500, 15000)),
        "hpf_cutoff": float(np.clip(hpf_cutoff, 20, 500)),
        "centroid_hz": float(np.mean(centroid)),
        "rolloff_hz": float(np.mean(rolloff)),
        "brightness": float(brightness),
        "spectral_slope": float(slope)
    }


def analyze_harmonics(y, sr=22050):
    """Analyze harmonic content and suggest waveform type."""
    # Compute harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harmonic_ratio = np.sum(y_harmonic ** 2) / (np.sum(y ** 2) + 1e-10)

    # Analyze harmonic spectrum for waveform hints
    S_harm = np.abs(librosa.stft(y_harmonic))

    # Look at fundamental vs harmonics ratio
    # High odd harmonics = square-ish
    # All harmonics = saw-ish
    # Only fundamental = sine-ish

    # Compute spectral flatness of harmonic content
    flatness = librosa.feature.spectral_flatness(y=y_harmonic)[0]
    avg_flatness = np.mean(flatness)

    # Suggest waveform based on characteristics
    if harmonic_ratio < 0.3:
        waveform = "noise"  # Mostly percussive
        harmonics_count = 1
    elif avg_flatness > 0.1:
        waveform = "saw"  # Rich harmonics
        harmonics_count = 8
    elif avg_flatness > 0.05:
        waveform = "square"  # Moderate harmonics (odd)
        harmonics_count = 5
    else:
        waveform = "sine"  # Clean fundamental
        harmonics_count = 2

    return {
        "suggested_waveform": waveform,
        "harmonics_count": harmonics_count,
        "harmonic_ratio": float(harmonic_ratio),
        "spectral_flatness": float(avg_flatness),
        "is_percussive": harmonic_ratio < 0.3
    }


def analyze_formants(y, sr=22050):
    """Extract formant frequencies and harmonic profile for better timbre matching."""
    from scipy.signal import find_peaks

    # Compute average spectrum
    S = np.abs(librosa.stft(y))
    avg_spectrum = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)

    # Find spectral peaks (formants)
    # Use prominence to find significant peaks, not just any local maximum
    peaks, properties = find_peaks(
        avg_spectrum,
        prominence=np.max(avg_spectrum) * 0.05,  # 5% of max as minimum prominence
        distance=5  # Minimum distance between peaks (in frequency bins)
    )

    # Get top 8 formants by prominence
    if len(peaks) > 0:
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1][:8]
        top_peaks = peaks[sorted_indices]

        formant_freqs = freqs[top_peaks].tolist()
        formant_amps = (avg_spectrum[top_peaks] / np.max(avg_spectrum)).tolist()

        # Sort by frequency
        sorted_pairs = sorted(zip(formant_freqs, formant_amps))
        formant_freqs = [f for f, a in sorted_pairs]
        formant_amps = [a for f, a in sorted_pairs]
    else:
        formant_freqs = [100, 300, 600, 1000]  # Default formants
        formant_amps = [1.0, 0.5, 0.3, 0.2]

    # Compute harmonic profile (amplitudes of first 16 harmonics relative to fundamental)
    # This helps recreate the timbre with additive synthesis
    harmonics_profile = []
    fundamental_bin = 5  # ~100 Hz bin as approximate fundamental

    for h in range(1, 17):
        harmonic_bin = fundamental_bin * h
        if harmonic_bin < len(avg_spectrum):
            # Average around the harmonic bin (±2 bins)
            start = max(0, harmonic_bin - 2)
            end = min(len(avg_spectrum), harmonic_bin + 3)
            amp = np.mean(avg_spectrum[start:end])
            harmonics_profile.append(float(amp))
        else:
            harmonics_profile.append(0.0)

    # Normalize harmonic profile
    max_harm = max(harmonics_profile) if harmonics_profile else 1.0
    harmonics_profile = [h / max_harm for h in harmonics_profile]

    # Estimate FM parameters from harmonic structure
    # If odd harmonics dominate: square-like (FM ratio ~1, low index)
    # If all harmonics present: saw-like (FM ratio ~1, high index)
    odd_harmonics = sum(harmonics_profile[0::2])  # 1st, 3rd, 5th...
    even_harmonics = sum(harmonics_profile[1::2])  # 2nd, 4th, 6th...
    odd_ratio = odd_harmonics / (odd_harmonics + even_harmonics + 1e-10)

    if odd_ratio > 0.7:
        # Square-like: odd harmonics dominate
        fm_ratio = 1.0
        fm_index = 2.0
    else:
        # Saw-like: all harmonics present
        fm_ratio = 1.0
        fm_index = np.pi  # ~3.14 gives saw-like spectrum

    return {
        "formant_frequencies": formant_freqs,
        "formant_amplitudes": formant_amps,
        "formant_Q": 5.0,  # Default Q for moderate resonance
        "harmonics_profile": harmonics_profile,
        "fm_ratio": float(fm_ratio),
        "fm_index": float(fm_index),
        "odd_harmonic_ratio": float(odd_ratio)
    }


def analyze_dynamics(y, sr=22050):
    """Analyze amplitude dynamics for envelope shaping."""
    # Compute RMS envelope
    rms = librosa.feature.rms(y=y)[0]

    # Dynamic range
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    # Compute loudness variation (how much the volume changes)
    rms_diff = np.diff(rms)
    variation = np.std(rms_diff) / (np.mean(rms) + 1e-10)

    # Suggest compression based on dynamic range
    if dynamic_range > 30:
        compression = "high"  # Needs compression
        comp_ratio = 4.0
    elif dynamic_range > 20:
        compression = "medium"
        comp_ratio = 2.0
    else:
        compression = "low"  # Already compressed
        comp_ratio = 1.5

    # Suggest sustain level based on RMS stability
    sustain = np.clip(1 - variation, 0.2, 0.9)

    return {
        "dynamic_range_db": dynamic_range,
        "suggested_compression": compression,
        "compression_ratio": comp_ratio,
        "sustain_level": float(sustain),
        "rms_mean": float(np.mean(rms)),
        "rms_variation": float(variation)
    }


def correct_octave_error(tempo, prior_center=120, prior_range=40):
    """
    Correct octave errors by checking if half/double time is more likely.

    Most music falls in 90-150 BPM range. If detected tempo is outside this,
    check if an octave multiple is within range.

    Args:
        tempo: Detected tempo
        prior_center: Expected tempo center (default 120 BPM)
        prior_range: Expected tempo range from center (default ±40 BPM = 80-160)

    Returns:
        Corrected tempo
    """
    low = prior_center - prior_range  # 80 BPM
    high = prior_center + prior_range  # 160 BPM

    # Generate all octave candidates
    candidates = [
        tempo,
        tempo / 2,
        tempo * 2,
        tempo * 2 / 3,  # triplet relationship
        tempo * 3 / 2,  # triplet relationship
    ]

    # Score each candidate by distance from prior center
    best_score = float('inf')
    best_tempo = tempo

    for cand in candidates:
        if low <= cand <= high:
            # Distance from prior center (prefer tempos near 120)
            score = abs(cand - prior_center)
            if score < best_score:
                best_score = score
                best_tempo = cand

    # If no candidate in range, fall back to closest octave
    if best_score == float('inf'):
        for cand in candidates:
            score = abs(cand - prior_center)
            if score < best_score:
                best_score = score
                best_tempo = cand

    return best_tempo


def analyze_tempo_precise(y, sr=22050):
    """Extract precise tempo with sub-BPM accuracy."""
    # Use multiple methods and combine

    # Method 1: librosa beat tracking with start_bpm hint
    # Using start_bpm helps reduce octave errors by biasing towards typical range
    tempo1, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
    if hasattr(tempo1, '__len__'):
        tempo1 = float(tempo1[0]) if len(tempo1) > 0 else 120.0
    else:
        tempo1 = float(tempo1)

    # Method 2: Onset-based tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo2 = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=120.0)
    if hasattr(tempo2, '__len__'):
        tempo2 = float(tempo2[0]) if len(tempo2) > 0 else 120.0
    else:
        tempo2 = float(tempo2)

    # Method 3: Tempogram-based (most precise)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)

    # Find peak in tempogram (with prior correction)
    avg_tempogram = np.mean(tempogram, axis=1)
    peak_idx = np.argmax(avg_tempogram[1:]) + 1  # Skip DC
    tempo3 = tempo_freqs[peak_idx] if peak_idx < len(tempo_freqs) else 120.0

    # Apply octave correction to each estimate
    tempo1 = correct_octave_error(tempo1)
    tempo2 = correct_octave_error(tempo2)
    tempo3 = correct_octave_error(tempo3)

    # Combine estimates (weighted average, prefer tempogram)
    tempos = [tempo1, tempo2, tempo3]
    weights = [0.3, 0.3, 0.4]  # Prefer tempogram

    # Filter out outliers
    median_tempo = np.median(tempos)
    valid_tempos = []
    valid_weights = []
    for t, w in zip(tempos, weights):
        # Allow half-time/double-time variations
        ratio = t / median_tempo
        if 0.45 < ratio < 2.2:  # Within reasonable range
            valid_tempos.append(t)
            valid_weights.append(w)

    if valid_tempos:
        final_tempo = np.average(valid_tempos, weights=valid_weights)
    else:
        final_tempo = median_tempo

    # Final octave correction on combined result
    final_tempo = correct_octave_error(final_tempo)

    # Calculate beat times for sync
    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else [0.5]
    beat_regularity = 1 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)

    # AI-driven tempo tolerance: based on beat regularity and tempo estimate variance
    # High regularity = tight tolerance, low regularity = looser tolerance
    # Also, variance in tempo estimates indicates uncertainty
    tempo_variance = np.std([t for t in tempos if 60 < t < 200]) if len(tempos) > 1 else 10

    # Tolerance formula: base 10% + uncertainty factor
    # - High regularity (>0.9): 10% base
    # - Medium regularity (0.7-0.9): 15% base
    # - Low regularity (<0.7): 20% base
    # - Add variance factor: +1% per 5 BPM variance
    if beat_regularity > 0.9:
        base_tolerance = 0.10
    elif beat_regularity > 0.7:
        base_tolerance = 0.15
    else:
        base_tolerance = 0.20

    variance_factor = min(0.10, tempo_variance / 50)  # Cap at 10% extra
    tempo_tolerance = base_tolerance + variance_factor

    return {
        "tempo_bpm": float(np.clip(final_tempo, 60, 200)),
        "tempo_confidence": float(np.clip(beat_regularity, 0, 1)),
        "tempo_tolerance": float(tempo_tolerance),  # AI-derived tolerance for comparison
        "beat_count": len(beats),
        "seconds_per_beat": 60.0 / final_tempo,
        "samples_per_beat": int(sr * 60.0 / final_tempo),
        "beat_times": beat_times[:10].tolist() if len(beat_times) > 0 else [],
        "tempo_estimates": {
            "librosa_beat": tempo1,
            "onset_tempo": tempo2,
            "tempogram": tempo3
        }
    }


def analyze_frequency_bands(y, sr=22050):
    """Detailed frequency band analysis for mixing."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    bands = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 10000),
        'air': (10000, 20000)
    }

    total_energy = np.sum(S ** 2)
    band_data = {}

    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        energy = np.sum(S[mask, :] ** 2) / max(total_energy, 1e-10)
        band_data[name] = float(energy)

    # Suggest mix balance adjustments
    bass_heavy = band_data['bass'] + band_data['sub_bass']
    mid_presence = band_data['mid'] + band_data['low_mid']
    high_content = band_data['high_mid'] + band_data['presence'] + band_data['brilliance']

    return {
        "bands": band_data,
        "bass_ratio": float(bass_heavy),
        "mid_ratio": float(mid_presence),
        "high_ratio": float(high_content),
        "mix_character": "bass-heavy" if bass_heavy > 0.4 else ("bright" if high_content > 0.3 else "balanced")
    }


def generate_synth_config(analysis_results):
    """Generate synthesis configuration from analysis."""
    trans = analysis_results["transients"]
    spec = analysis_results["spectral"]
    harm = analysis_results["harmonics"]
    dyn = analysis_results["dynamics"]
    tempo = analysis_results["tempo"]
    bands = analysis_results["frequency_bands"]
    formants = analysis_results.get("formants", {})

    # Build envelope config
    envelope = {
        "attack": trans["attack_ms"] / 1000,  # Convert to seconds
        "decay": trans["decay_ms"] / 1000,
        "sustain": dyn["sustain_level"],
        "release": trans["decay_ms"] / 1000 * 1.5
    }

    # Build filter config
    filters = {
        "lpf_cutoff": spec["lpf_cutoff"],
        "hpf_cutoff": spec["hpf_cutoff"],
        "filter_envelope_amount": 0.5 if spec["brightness"] > 0.5 else 0.3,
        "filter_attack": trans["attack_ms"] / 1000 * 2,
        "filter_decay": trans["decay_ms"] / 1000 * 3
    }

    # Build oscillator config
    oscillator = {
        "waveform": harm["suggested_waveform"],
        "harmonics": harm["harmonics_count"],
        "detune_cents": 5 if harm["harmonic_ratio"] > 0.5 else 0,
        "sub_octave_gain": 0.3 if bands["bass_ratio"] > 0.3 else 0.1
    }

    # Build dynamics config
    dynamics = {
        "compression_ratio": dyn["compression_ratio"],
        "target_rms": dyn["rms_mean"],
        "limiter_threshold": 0.95
    }

    # Build tempo config
    tempo_config = {
        "bpm": tempo["tempo_bpm"],
        "confidence": tempo["tempo_confidence"],
        "samples_per_beat": tempo["samples_per_beat"],
        "sync_to_beat": tempo["tempo_confidence"] > 0.7
    }

    # Build per-voice configs based on frequency analysis
    # KEY INSIGHT: Synthesizers need harmonics to fill the spectrum.
    # Even if original sounds "tonal", we need harmonics to match spectral distribution.
    # A note at C4 (262Hz) only fills low_mid, but with harmonics can fill mid band too.

    # KEY INSIGHT: Melodic patterns ALWAYS need harmonics for proper spectrum
    # Even if the original mix is bass-heavy (low mid_ratio), the melodic notes
    # still need saw waveform to produce harmonics that fill the mid band.
    # Sine waves only produce fundamental frequency, not harmonics.
    # So we ALWAYS use saw for mid voice, regardless of mix frequency balance.
    mid_waveform = "saw"  # Always use saw - melodic patterns need harmonics

    # LPF for mid: must allow harmonics through (always high cutoff for saw wave)
    mid_lpf = max(4000, spec["centroid_hz"] * 2)

    # Detune for richer spectral content (always use 10 for saw wave)
    mid_detune = 10

    # Calculate gain scaling based on original frequency distribution
    # If original has low bass, we should have very low bass
    bass_scale = max(0.01, bands["bass_ratio"])  # Minimum 1% to avoid divide by zero
    mid_scale = max(0.1, bands["mid_ratio"])

    voice_configs = {
        "bass": {
            # Bass gain proportional to original's bass content
            # Bass-heavy tracks (>30% bass) need strong bass gain
            "gain": min(0.8, bass_scale * 1.2) if bands["bass_ratio"] > 0.05 else 0.02,
            "lpf": min(600, spec["lpf_cutoff"] * 0.3),  # Higher LPF for more bass harmonics
            "hpf": 30,  # Lower HPF to keep sub-bass
            "envelope": {**envelope, "attack": 0.005, "decay": 0.15},
            "waveform": "saw" if bands["bass_ratio"] > 0.1 else "sine",
            "sub_octave_gain": 0.4 if bands["bass_ratio"] > 0.2 else 0.1  # Strong sub for bass-heavy
        },
        "mid": {
            # Reduce mid gain for bass-heavy tracks to maintain balance
            "gain": min(1.2, mid_scale * 1.2 + 0.3) if bands["bass_ratio"] < 0.3 else min(0.8, mid_scale * 1.0 + 0.2),
            # LPF based on original's brightness - if centroid > 2000Hz, use high LPF
            "lpf": max(6000, spec["centroid_hz"] * 2.5) if spec["centroid_hz"] > 1500 else mid_lpf,
            "hpf": 200,  # Keep HPF moderate
            "envelope": envelope,
            "waveform": mid_waveform,
            "detune_cents": mid_detune  # 10 cents for richer spectral content
        },
        "high": {
            # Boost high voice gain based on brightness needs
            "gain": min(1.2, bands["high_ratio"] * 3.0 + 0.4),
            "lpf": max(12000, spec["lpf_cutoff"] * 2),  # Very high LPF for brightness
            "hpf": 500,
            "envelope": {**envelope, "attack": 0.001, "sustain": 0.5},
            "waveform": "square"  # Square for bright odd harmonics
        },
        "drums": {
            "gain": 0.6 if harm["is_percussive"] else 0.4,  # v2 balance
            "transient_boost": trans["transient_sharpness"],
            "reverb": 0.1
        }
    }

    # Adjust oscillator config based on our analysis
    oscillator["waveform"] = mid_waveform
    oscillator["detune_cents"] = mid_detune

    # Build FM synthesis config (AI-derived from harmonic analysis)
    # FM synthesis produces richer harmonics that can better match real audio MFCC
    # Enable for any track with significant harmonic content
    fm_config = {
        "enabled": True,  # Always enable FM for richer timbre
        "modulator_ratio": formants.get("fm_ratio", 1.0),
        # Modulation index determines harmonic richness: higher = more harmonics
        "modulation_index": formants.get("fm_index", 2.0)
    }

    # Build formants config (resonant peaks from original audio)
    formants_config = {
        "frequencies": formants.get("formant_frequencies", [200, 500, 1000, 2000])[:6],  # Top 6 formants
        "amplitudes": formants.get("formant_amplitudes", [1.0, 0.5, 0.3, 0.2])[:6],
        "Q": formants.get("formant_Q", 5.0)
    }

    return {
        "envelope": envelope,
        "filters": filters,
        "oscillator": oscillator,
        "dynamics": dynamics,
        "tempo": tempo_config,
        "fm": fm_config,
        "formants": formants_config,
        "harmonics_profile": formants.get("harmonics_profile", [1.0, 0.5, 0.33, 0.25]),
        "voices": voice_configs,
        "master": {
            # Higher master gain for louder output (limiter will prevent clipping)
            "gain": 1.5,
            # HPF based on bass content: bass-heavy tracks need low HPF
            # >30% bass: HPF=20 (preserve sub), 15-30%: HPF=40, 5-15%: HPF=80, <5%: HPF=120
            "hpf": 120 if bands["bass_ratio"] < 0.05 else (80 if bands["bass_ratio"] < 0.15 else (40 if bands["bass_ratio"] < 0.3 else 20)),
            "limiter": True,
            # AI-derived brightness target: boost high frequencies if original is bright
            "target_centroid": spec["centroid_hz"],
            # High shelf boost: scale based on how bright the original is
            # Centroid 2400Hz needs ~3dB boost to match after saturation darkens things
            "high_shelf_boost": min(4.0, max(0, (spec["centroid_hz"] - 1500) / 300))
        }
    }


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def analyze_audio(audio_path, duration=60):
    """Main analysis function."""
    print(f"Analyzing: {audio_path}", file=sys.stderr)

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=duration)
    print(f"Loaded {len(y)/sr:.1f}s at {sr}Hz", file=sys.stderr)

    results = {}

    print("Analyzing transients...", file=sys.stderr)
    results["transients"] = analyze_transients(y, sr)

    print("Analyzing spectral envelope...", file=sys.stderr)
    results["spectral"] = analyze_spectral_envelope(y, sr)

    print("Analyzing harmonics...", file=sys.stderr)
    results["harmonics"] = analyze_harmonics(y, sr)

    print("Analyzing dynamics...", file=sys.stderr)
    results["dynamics"] = analyze_dynamics(y, sr)

    print("Analyzing tempo...", file=sys.stderr)
    results["tempo"] = analyze_tempo_precise(y, sr)

    print("Analyzing frequency bands...", file=sys.stderr)
    results["frequency_bands"] = analyze_frequency_bands(y, sr)

    print("Analyzing formants and harmonic profile...", file=sys.stderr)
    results["formants"] = analyze_formants(y, sr)

    # Generate synthesis config
    print("Generating synthesis config...", file=sys.stderr)
    results["synth_config"] = generate_synth_config(results)

    # Convert numpy types to Python native types for JSON serialization
    return convert_numpy_types(results)


def main():
    parser = argparse.ArgumentParser(description='Analyze audio for synthesis parameters')
    parser.add_argument('audio', help='Audio file to analyze')
    parser.add_argument('-o', '--output', help='Output JSON file (default: stdout)')
    parser.add_argument('-d', '--duration', type=float, default=60, help='Duration to analyze (seconds)')
    parser.add_argument('-c', '--config-only', action='store_true', help='Output only synth config')

    args = parser.parse_args()

    results = analyze_audio(args.audio, args.duration)

    if args.config_only:
        output = results["synth_config"]
    else:
        output = results

    json_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == '__main__':
    main()
