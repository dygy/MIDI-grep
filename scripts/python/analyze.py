#!/usr/bin/env python3
"""
Audio analysis for BPM, key, time signature, and swing detection using librosa.
"""

import sys
import os
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: analyze.py <input_audio> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import librosa
        import numpy as np

        # Load audio
        y, sr = librosa.load(input_path, sr=22050)

        # BPM Detection with start_bpm hint to reduce octave errors
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)

        # Handle tempo array (librosa may return array)
        if hasattr(tempo, '__len__'):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)

        # Correct octave errors - most music is in 80-160 BPM range
        def correct_octave(t, center=120, range_=40):
            low, high = center - range_, center + range_
            candidates = [t, t/2, t*2, t*2/3, t*3/2]
            best = min(candidates, key=lambda c: abs(c - center) if low <= c <= high else float('inf'))
            return best if low <= best <= high else min(candidates, key=lambda c: abs(c - center))

        bpm = correct_octave(bpm)

        # Estimate BPM confidence based on beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        bpm_confidence = min(float(np.max(pulse)), 1.0)

        # Key Detection using Krumhansl-Schmuckler algorithm
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Collect all key correlations
        key_scores = []
        for i in range(12):
            # Rotate chroma to test each key
            rotated = np.roll(chroma_mean, -i)

            # Correlate with major profile
            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            if not np.isnan(major_corr):
                key_scores.append((f"{key_names[i]} major", float(major_corr)))

            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
            if not np.isnan(minor_corr):
                key_scores.append((f"{key_names[i]} minor", float(minor_corr)))

        # Sort by correlation and get top 5
        key_scores.sort(key=lambda x: x[1], reverse=True)
        key_candidates = [{"key": k, "confidence": round(max(0, min(c, 1.0)), 3)} for k, c in key_scores[:5]]

        best_key_info = key_scores[0] if key_scores else ("C major", 0.0)
        detected_key = best_key_info[0]
        key_confidence = max(0, min(best_key_info[1], 1.0))

        # Time Signature Detection
        time_sig, time_sig_confidence, time_sig_candidates = detect_time_signature(y, sr, beat_frames, onset_env)

        # Swing Detection
        swing_ratio, swing_confidence = detect_swing(y, sr, beat_frames, bpm)

        # BPM alternatives (half/double time)
        bpm_candidates = [
            {"bpm": round(bpm, 1), "confidence": round(bpm_confidence, 3)},
            {"bpm": round(bpm / 2, 1), "confidence": round(bpm_confidence * 0.7, 3)},
            {"bpm": round(bpm * 2, 1), "confidence": round(bpm_confidence * 0.6, 3)},
        ]
        # Filter to reasonable range and sort by confidence
        bpm_candidates = [c for c in bpm_candidates if 60 <= c["bpm"] <= 200]
        bpm_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # Write results
        result = {
            "bpm": bpm,
            "bpm_confidence": bpm_confidence,
            "bpm_candidates": bpm_candidates[:3],
            "key": detected_key,
            "key_confidence": key_confidence,
            "key_candidates": key_candidates,
            "time_signature": time_sig,
            "time_signature_confidence": time_sig_confidence,
            "time_signature_candidates": time_sig_candidates,
            "swing_ratio": swing_ratio,
            "swing_confidence": swing_confidence
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Analysis complete: BPM={bpm:.1f}, Key={detected_key}, Time={time_sig}, Swing={swing_ratio:.2f}")

    except ImportError as e:
        print(f"Error: Required library not installed: {e}", file=sys.stderr)
        print("Run: pip install librosa numpy", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


def detect_time_signature(y, sr, beat_frames, onset_env):
    """
    Detect time signature by analyzing beat groupings and accent patterns.

    Returns:
        tuple: (time_signature string like "4/4", confidence 0-1, candidates list)
    """
    import numpy as np
    import librosa

    if len(beat_frames) < 8:
        return "4/4", 0.5, [{"time_signature": "4/4", "confidence": 0.5}]

    # Get beat times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Analyze onset strength at each beat
    beat_strengths = []
    hop_length = 512
    for frame in beat_frames:
        if frame < len(onset_env):
            beat_strengths.append(onset_env[frame])
        else:
            beat_strengths.append(0)

    beat_strengths = np.array(beat_strengths)

    if len(beat_strengths) < 8:
        return "4/4", 0.5, [{"time_signature": "4/4", "confidence": 0.5}]

    # Normalize beat strengths
    if np.max(beat_strengths) > 0:
        beat_strengths = beat_strengths / np.max(beat_strengths)

    # Test different meters by looking at accent patterns
    # For each meter, check if downbeats (every N beats) are stronger

    meters = {
        "4/4": 4,
        "3/4": 3,
        "6/8": 6,
        "2/4": 2,
        "5/4": 5,
        "7/8": 7,
    }

    meter_scores = []

    for meter_name, beats_per_bar in meters.items():
        if len(beat_strengths) < beats_per_bar * 2:
            continue

        # Calculate accent strength ratio for this meter
        # Downbeats should be stronger than other beats
        downbeat_strengths = []
        other_strengths = []

        for i, strength in enumerate(beat_strengths):
            if i % beats_per_bar == 0:
                downbeat_strengths.append(strength)
            else:
                other_strengths.append(strength)

        if len(downbeat_strengths) < 2 or len(other_strengths) < 2:
            continue

        # Score is ratio of downbeat strength to average other beat strength
        avg_downbeat = np.mean(downbeat_strengths)
        avg_other = np.mean(other_strengths) + 0.001  # Avoid division by zero

        score = avg_downbeat / avg_other

        # Bonus for common meters
        if meter_name in ["4/4", "3/4"]:
            score *= 1.1

        meter_scores.append((meter_name, score))

    # Sort by score
    meter_scores.sort(key=lambda x: x[1], reverse=True)

    if not meter_scores:
        return "4/4", 0.5, [{"time_signature": "4/4", "confidence": 0.5}]

    best_meter = meter_scores[0][0]
    best_score = meter_scores[0][1]

    # Calculate confidence based on how clear the accent pattern is
    confidence = min(1.0, (best_score - 1.0) / 0.5) if best_score > 1.0 else 0.3

    # Build candidates list
    candidates = []
    for meter, score in meter_scores[:3]:
        conf = min(1.0, (score - 1.0) / 0.5) if score > 1.0 else 0.3
        candidates.append({"time_signature": meter, "confidence": round(conf, 3)})

    return best_meter, confidence, candidates


def detect_swing(y, sr, beat_frames, bpm):
    """
    Detect swing feel by analyzing timing deviations from straight grid.

    Swing ratio:
    - 1.0 = straight (no swing)
    - 1.5 = light swing
    - 2.0 = heavy swing (triplet feel)

    Returns:
        tuple: (swing_ratio, confidence 0-1)
    """
    import numpy as np
    import librosa

    if len(beat_frames) < 4:
        return 1.0, 0.0  # No swing detected with low confidence

    # Get beat times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Calculate expected beat duration
    beat_duration = 60.0 / bpm

    # Detect onsets at a finer resolution
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if len(onset_times) < 8:
        return 1.0, 0.0

    # Analyze eighth note subdivisions
    # For each beat, find onsets that fall on eighth note positions
    eighth_duration = beat_duration / 2
    tolerance = eighth_duration * 0.25  # 25% tolerance

    # Collect timing deviations for "off-beat" eighth notes
    swing_ratios = []

    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        expected_offbeat = beat_start + eighth_duration

        # Find onset closest to expected offbeat position
        offbeat_onsets = [t for t in onset_times
                         if expected_offbeat - tolerance < t < expected_offbeat + tolerance]

        if offbeat_onsets:
            actual_offbeat = min(offbeat_onsets, key=lambda t: abs(t - expected_offbeat))
            # Calculate swing ratio
            # Ratio = (time from beat to offbeat) / (time from offbeat to next beat)
            first_half = actual_offbeat - beat_start
            second_half = beat_end - actual_offbeat

            if second_half > 0.01:  # Avoid division by very small numbers
                ratio = first_half / second_half
                if 0.5 < ratio < 3.0:  # Sanity check
                    swing_ratios.append(ratio)

    if len(swing_ratios) < 4:
        return 1.0, 0.3  # Not enough data, assume straight

    # Calculate mean swing ratio
    mean_ratio = np.mean(swing_ratios)
    std_ratio = np.std(swing_ratios)

    # Confidence based on consistency of swing feel
    consistency = 1.0 - min(1.0, std_ratio / 0.3)

    # Normalize to typical swing values
    # Straight = 1.0, triplet swing = 2.0
    if mean_ratio < 1.1:
        swing_ratio = 1.0  # Straight
        confidence = consistency * 0.8
    elif mean_ratio < 1.3:
        swing_ratio = round(mean_ratio, 2)  # Light swing
        confidence = consistency * 0.7
    elif mean_ratio < 1.7:
        swing_ratio = round(mean_ratio, 2)  # Medium swing
        confidence = consistency * 0.8
    else:
        swing_ratio = min(2.0, round(mean_ratio, 2))  # Heavy swing
        confidence = consistency * 0.9

    return swing_ratio, confidence


if __name__ == '__main__':
    main()
