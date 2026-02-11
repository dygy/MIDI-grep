#!/usr/bin/env python3
"""
Audio to MIDI transcription using Basic Pitch.
"""

import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: transcribe.py <input_audio> <output_midi>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Check if audio is essentially silent (skip transcription for silent stems)
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(input_path, sr=22050, mono=True, duration=60)
        rms = np.sqrt(np.mean(y**2))
        SILENCE_THRESHOLD = 0.001
        if rms < SILENCE_THRESHOLD:
            print(f"  Audio is silent (RMS={rms:.6f} < {SILENCE_THRESHOLD}), creating empty MIDI")
            # Create empty MIDI file
            import pretty_midi
            empty_midi = pretty_midi.PrettyMIDI()
            empty_midi.write(output_path)
            print(f"Empty MIDI saved to: {output_path}")
            sys.exit(0)
    except Exception as e:
        print(f"  Warning: Could not check silence: {e}", file=sys.stderr)

    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import tempfile
        import shutil

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run Basic Pitch with high sensitivity for maximum note capture
            predict_and_save(
                [input_path],
                temp_dir,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.3,      # Lower = more onsets detected (default 0.5)
                frame_threshold=0.2,      # Lower = more notes detected (default 0.3)
                minimum_note_length=50,   # Minimum note length in ms (default 127.7)
                minimum_frequency=27.5,   # A0 - capture full piano range
                maximum_frequency=4186.0, # C8 - capture full piano range
            )

            # Find the generated MIDI file
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            midi_file = os.path.join(temp_dir, f"{input_name}_basic_pitch.mid")

            if os.path.exists(midi_file):
                shutil.copy2(midi_file, output_path)
                print(f"MIDI saved to: {output_path}")
            else:
                # Search for any .mid file
                for f in os.listdir(temp_dir):
                    if f.endswith('.mid'):
                        shutil.copy2(os.path.join(temp_dir, f), output_path)
                        print(f"MIDI saved to: {output_path}")
                        break
                else:
                    print("Error: No MIDI file generated", file=sys.stderr)
                    sys.exit(1)

    except ImportError:
        print("Error: Basic Pitch not installed. Run: pip install basic-pitch", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
