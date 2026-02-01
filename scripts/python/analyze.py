#!/usr/bin/env python3
"""
Audio analysis for BPM and key detection using librosa.
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

        # BPM Detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Handle tempo array (librosa may return array)
        if hasattr(tempo, '__len__'):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)

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

        best_corr = -1
        best_key = 'C'
        best_mode = 'major'

        for i in range(12):
            # Rotate chroma to test each key
            rotated = np.roll(chroma_mean, -i)

            # Correlate with major profile
            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = key_names[i]
                best_mode = 'major'

            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = key_names[i]
                best_mode = 'minor'

        key_confidence = max(0, min(best_corr, 1.0))
        detected_key = f"{best_key} {best_mode}"

        # Write results
        result = {
            "bpm": bpm,
            "bpm_confidence": bpm_confidence,
            "key": detected_key,
            "key_confidence": key_confidence
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Analysis complete: BPM={bpm:.1f}, Key={detected_key}")

    except ImportError as e:
        print(f"Error: Required library not installed: {e}", file=sys.stderr)
        print("Run: pip install librosa numpy", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
