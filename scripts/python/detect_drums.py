#!/usr/bin/env python3
"""
Drum onset detection and classification using librosa.

Detects drum hits in an audio file and classifies them by type:
  - bd (kick): Low frequency energy (20-150 Hz)
  - sd (snare): Mid frequency energy (200-2000 Hz) with noise
  - hh (hi-hat): High frequency energy (4000+ Hz), short decay
  - oh (open hat): High frequency energy, longer decay
  - cp (clap): Mid-high frequency, very short attack

Outputs JSON with hits, statistics, and detected tempo.
"""

import sys
import os
import json
import argparse
import numpy as np
import librosa


def classify_drum_hit(y, sr, onset_sample, hop_length=512):
    """
    Classify a drum hit based on its spectral content.

    Args:
        y: Audio signal
        sr: Sample rate
        onset_sample: Sample index of the onset
        hop_length: Hop length for analysis

    Returns:
        Tuple of (drum_type, confidence)
    """
    # Extract a window around the onset (100ms)
    window_samples = int(0.1 * sr)
    start = max(0, onset_sample)
    end = min(len(y), onset_sample + window_samples)

    if end - start < 256:
        return "bd", 0.5  # Default to kick if too short

    segment = y[start:end]

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1/sr)

    # Define frequency bands
    low_mask = (freqs >= 20) & (freqs < 150)
    mid_mask = (freqs >= 200) & (freqs < 2000)
    high_mask = (freqs >= 4000) & (freqs < 12000)

    # Compute energy in each band
    total_energy = np.sum(spectrum ** 2) + 1e-10
    low_energy = np.sum(spectrum[low_mask] ** 2) / total_energy if np.any(low_mask) else 0
    mid_energy = np.sum(spectrum[mid_mask] ** 2) / total_energy if np.any(mid_mask) else 0
    high_energy = np.sum(spectrum[high_mask] ** 2) / total_energy if np.any(high_mask) else 0

    # Compute decay characteristics
    envelope = np.abs(segment)
    if len(envelope) > 10:
        # Simple decay: compare first half to second half energy
        first_half = np.sum(envelope[:len(envelope)//2])
        second_half = np.sum(envelope[len(envelope)//2:])
        decay_ratio = second_half / (first_half + 1e-10)
    else:
        decay_ratio = 0.5

    # Classification logic
    # Kick: Strong low frequency, weak high
    if low_energy > 0.3 and high_energy < 0.2:
        return "bd", min(1.0, low_energy + 0.3)

    # Hi-hat: Strong high frequency, short decay
    if high_energy > 0.3 and decay_ratio < 0.4:
        return "hh", min(1.0, high_energy + 0.2)

    # Open hi-hat: Strong high frequency, longer decay
    if high_energy > 0.25 and decay_ratio > 0.4:
        return "oh", min(1.0, high_energy + 0.1)

    # Snare: Mid frequency with some high (noise), moderate decay
    if mid_energy > 0.25 and high_energy > 0.15:
        return "sd", min(1.0, mid_energy + 0.2)

    # Clap: Mid-high, very sharp attack
    if mid_energy > 0.2 and high_energy > 0.2 and decay_ratio < 0.3:
        return "cp", min(1.0, (mid_energy + high_energy) / 2)

    # Default classification based on dominant frequency band
    if low_energy >= mid_energy and low_energy >= high_energy:
        return "bd", 0.5
    elif high_energy >= mid_energy:
        return "hh", 0.5
    else:
        return "sd", 0.5


def estimate_velocity(y, sr, onset_sample, window_ms=50):
    """
    Estimate the velocity (intensity) of a drum hit.

    Args:
        y: Audio signal
        sr: Sample rate
        onset_sample: Sample index of the onset
        window_ms: Window size in milliseconds

    Returns:
        Velocity as integer 0-127
    """
    window_samples = int(window_ms / 1000 * sr)
    start = max(0, onset_sample)
    end = min(len(y), onset_sample + window_samples)

    if end <= start:
        return 64

    segment = y[start:end]
    rms = np.sqrt(np.mean(segment ** 2))

    # Map RMS to velocity (0-127)
    # Using a simple mapping; could be improved with calibration
    velocity = int(min(127, max(1, rms * 1000)))

    return velocity


def detect_drums(audio_path, output_json, quantize=16, bpm=None):
    """
    Detect and classify drum hits in an audio file.

    Args:
        audio_path: Path to audio file
        output_json: Path for output JSON file
        quantize: Quantization grid (4, 8, 16, 32)
        bpm: BPM to use (if None, will be estimated)

    Returns:
        Detection result dictionary
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Estimate tempo if not provided
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Handle array return from newer librosa versions
        if hasattr(tempo, '__len__'):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)

    # Detect onsets using multiple methods for better accuracy
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env,
        backtrack=True, units='frames'
    )

    # Convert frames to samples and times
    hop_length = 512
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Calculate quantization grid
    beat_duration = 60.0 / bpm  # seconds per beat
    grid_size = beat_duration / (quantize / 4)  # grid step in seconds

    # Classify each onset and build hits list
    hits = []
    for i, (onset_sample, onset_time) in enumerate(zip(onset_samples, onset_times)):
        # Classify the hit
        drum_type, confidence = classify_drum_hit(y, sr, onset_sample)

        # Estimate velocity
        velocity = estimate_velocity(y, sr, onset_sample)

        # Quantize time to grid
        quantized_time = round(onset_time / grid_size) * grid_size

        hits.append({
            "type": drum_type,
            "time": round(quantized_time, 4),
            "velocity": velocity,
            "velocity_normalized": round(velocity / 127.0, 3),
            "confidence": round(confidence, 3),
            "original_time": round(onset_time, 4)
        })

    # Remove duplicate hits at the same quantized time (keep highest velocity)
    hits_by_time = {}
    for hit in hits:
        key = (hit["type"], hit["time"])
        if key not in hits_by_time or hit["velocity"] > hits_by_time[key]["velocity"]:
            hits_by_time[key] = hit

    hits = sorted(hits_by_time.values(), key=lambda x: x["time"])

    # Compute statistics
    stats = {
        "total": len(hits),
        "by_type": {}
    }
    for hit in hits:
        t = hit["type"]
        stats["by_type"][t] = stats["by_type"].get(t, 0) + 1

    # Build result
    result = {
        "hits": hits,
        "stats": stats,
        "tempo": round(bpm, 1),
        "quantize": quantize,
        "duration": round(len(y) / sr, 2)
    }

    # Write JSON output
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Detect and classify drum hits in audio'
    )
    parser.add_argument('audio_path', help='Input audio file (drums stem)')
    parser.add_argument('output_json', help='Output JSON file path')
    parser.add_argument('--quantize', type=int, default=16, choices=[4, 8, 16, 32],
                        help='Quantization grid (default: 16)')
    parser.add_argument('--bpm', type=float, default=None,
                        help='BPM to use (default: auto-detect)')

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Input file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = detect_drums(
            args.audio_path,
            args.output_json,
            quantize=args.quantize,
            bpm=args.bpm
        )

        print(f"Detected {result['stats']['total']} drum hits")
        print(f"Tempo: {result['tempo']} BPM")
        print(f"Types: {result['stats']['by_type']}")
        print(f"Output saved to: {args.output_json}")

    except Exception as e:
        print(f"Error: Drum detection failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
