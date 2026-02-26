#!/usr/bin/env python3
"""
Smart audio analysis - extracts musical features for accurate recreation.

Features extracted:
- Chords (via chromagram)
- Beat positions and tempo
- Key and scale
- Sections (verse, chorus, etc.)
- Melodic contour
- Bass pattern
"""

import sys
import os
import json
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: smart_analyze.py <input_audio> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import librosa

        # Load audio
        print(f"Loading: {input_path}", file=sys.stderr)
        y, sr = librosa.load(input_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)

        result = {
            "duration": duration,
            "sample_rate": sr,
        }

        # 1. Tempo and beat detection
        print("Detecting tempo and beats...", file=sys.stderr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Get tempo as float
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        result["tempo"] = tempo
        result["beat_times"] = beat_times.tolist()
        result["num_beats"] = len(beat_times)

        # 2. Key detection using Krumhansl-Schmuckler algorithm
        print("Detecting key...", file=sys.stderr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        # Key profiles (major and minor)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_corr = -1
        best_key = 'C'
        best_mode = 'major'

        for i in range(12):
            rotated = np.roll(chroma_avg, -i)
            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = note_names[i]
                best_mode = 'major'
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = note_names[i]
                best_mode = 'minor'

        result["key"] = best_key
        result["mode"] = best_mode
        result["key_confidence"] = float(best_corr)

        # 3. Chord detection per beat
        print("Detecting chords...", file=sys.stderr)
        chords = []
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Get chroma for each beat
        hop_length = 512
        for i, beat_time in enumerate(beat_times):
            frame = librosa.time_to_frames(beat_time, sr=sr, hop_length=hop_length)
            if frame < chroma.shape[1]:
                beat_chroma = chroma[:, max(0, frame-2):min(chroma.shape[1], frame+2)]
                if beat_chroma.size > 0:
                    avg_chroma = np.mean(beat_chroma, axis=1)
                    # Find root note
                    root_idx = np.argmax(avg_chroma)
                    root = chord_names[root_idx]

                    # Determine chord quality (major/minor)
                    third_major = avg_chroma[(root_idx + 4) % 12]  # Major third
                    third_minor = avg_chroma[(root_idx + 3) % 12]  # Minor third
                    fifth = avg_chroma[(root_idx + 7) % 12]  # Perfect fifth

                    if third_minor > third_major:
                        chord = f"{root}m"
                    else:
                        chord = root

                    chords.append({
                        "time": float(beat_time),
                        "beat": i,
                        "chord": chord,
                        "root": root,
                        "is_minor": bool(third_minor > third_major)
                    })

        result["chords"] = chords

        # 4. Section detection using self-similarity matrix
        print("Detecting sections...", file=sys.stderr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Compute self-similarity
        S = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)

        # Find segment boundaries using novelty function
        bounds = librosa.segment.agglomerative(mfcc, 8)  # Max 8 sections
        bound_times = librosa.frames_to_time(bounds, sr=sr)

        sections = []
        for i, (start, end) in enumerate(zip(bound_times[:-1], bound_times[1:])):
            # Estimate section energy
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            section_audio = y[start_sample:end_sample]
            energy = float(np.sqrt(np.mean(section_audio**2)))

            sections.append({
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "energy": energy,
                "index": i
            })

        result["sections"] = sections

        # 5. Onset detection for rhythm pattern
        print("Detecting onsets...", file=sys.stderr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Convert to beat-relative positions
        if len(beat_times) > 1:
            beat_duration = np.median(np.diff(beat_times))
            onset_beats = []
            for onset in onset_times:
                # Find which beat this onset belongs to
                beat_idx = np.searchsorted(beat_times, onset) - 1
                if beat_idx >= 0 and beat_idx < len(beat_times):
                    beat_offset = (onset - beat_times[beat_idx]) / beat_duration
                    onset_beats.append({
                        "time": float(onset),
                        "beat": int(beat_idx),
                        "offset": float(beat_offset)  # 0-1 within beat
                    })
            result["onsets"] = onset_beats[:500]  # Limit to 500

        # 6. Spectral centroid for brightness/timbre
        print("Analyzing timbre...", file=sys.stderr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        result["brightness_mean"] = float(np.mean(spectral_centroid))
        result["brightness_std"] = float(np.std(spectral_centroid))

        # 6b. Enrich sections with per-section chord, density, brightness
        onset_times_list = onset_times.tolist() if len(onset_times) > 0 else []
        for section in sections:
            start_t, end_t = section["start"], section["end"]

            # Dominant chord(s) in this section
            section_chords = [c["chord"] for c in chords if start_t <= c["time"] < end_t]
            if section_chords:
                from collections import Counter
                counts = Counter(section_chords)
                section["dominant_chord"] = counts.most_common(1)[0][0]
                section["chord_progression"] = [c[0] for c in counts.most_common(2)]

            # Onset density (onsets per beat)
            sec_onsets = [o for o in onset_times_list if start_t <= o < end_t]
            sec_beats = [b for b in beat_times if start_t <= b < end_t]
            section["onset_density"] = round(len(sec_onsets) / max(len(sec_beats), 1), 2)

            # Average brightness (spectral centroid)
            start_frame = librosa.time_to_frames(start_t, sr=sr)
            end_frame = librosa.time_to_frames(end_t, sr=sr)
            sec_brightness = spectral_centroid[start_frame:end_frame]
            section["avg_brightness"] = round(float(np.mean(sec_brightness)), 0) if len(sec_brightness) > 0 else 0

        # 7. Estimate genre characteristics
        # High tempo + high brightness = electronic/dance
        # Low tempo + low brightness = ambient/chill
        # Mid tempo + varied brightness = pop/rock
        genre_hints = []
        if tempo > 125:
            genre_hints.append("electronic")
            if result["brightness_mean"] > 3000:
                genre_hints.append("high-energy")
        elif tempo > 100:
            genre_hints.append("upbeat")
        elif tempo > 70:
            genre_hints.append("moderate")
        else:
            genre_hints.append("slow")

        if best_mode == "minor":
            genre_hints.append("dark")
        else:
            genre_hints.append("bright")

        result["genre_hints"] = genre_hints

        # Save result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Smart analysis saved to: {output_path}", file=sys.stderr)
        print(f"Tempo: {tempo:.1f} BPM, Key: {best_key} {best_mode}", file=sys.stderr)
        print(f"Chords: {len(chords)}, Sections: {len(sections)}", file=sys.stderr)

    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Error: Analysis failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
