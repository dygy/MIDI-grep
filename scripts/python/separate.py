#!/usr/bin/env python3
"""
Stem separation using Demucs.
Extracts piano/other stem from mixed audio.
"""

import sys
import os
import shutil
from pathlib import Path
import subprocess

def main():
    if len(sys.argv) < 3:
        print("Usage: separate.py <input_audio> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Use demucs for separation
        # htdemucs separates into: drums, bass, other, vocals
        # Piano typically ends up in "other" stem

        demucs_out = os.path.join(output_dir, 'demucs_temp')
        os.makedirs(demucs_out, exist_ok=True)

        # Run demucs with mp3 output (more compatible)
        result = subprocess.run([
            sys.executable, '-m', 'demucs',
            '--two-stems=vocals',  # Split into vocals and no_vocals (instruments)
            '-n', 'htdemucs',
            '--mp3',  # Use mp3 output for compatibility
            '-o', demucs_out,
            input_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            # Try without two-stems for full separation
            result = subprocess.run([
                sys.executable, '-m', 'demucs',
                '-n', 'htdemucs',
                '--mp3',
                '-o', demucs_out,
                input_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error: Demucs failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)

        # Find the output stem
        input_name = Path(input_path).stem
        piano_dst = os.path.join(output_dir, 'piano.wav')  # Will rename mp3 to wav

        # Look for the separated stems (mp3 format)
        possible_paths = [
            # Two-stem mode
            os.path.join(demucs_out, 'htdemucs', input_name, 'no_vocals.mp3'),
            # Full separation - "other" contains piano
            os.path.join(demucs_out, 'htdemucs', input_name, 'other.mp3'),
            # Also check wav just in case
            os.path.join(demucs_out, 'htdemucs', input_name, 'no_vocals.wav'),
            os.path.join(demucs_out, 'htdemucs', input_name, 'other.wav'),
        ]

        found = False
        for src_path in possible_paths:
            if os.path.exists(src_path):
                # Copy and rename to .wav (or keep as is for downstream)
                if src_path.endswith('.mp3'):
                    piano_dst = os.path.join(output_dir, 'piano.mp3')
                shutil.copy2(src_path, piano_dst)
                print(f"Piano/instrumental stem saved to: {piano_dst}")
                found = True
                break

        if not found:
            # Search for any audio file in output
            for root, dirs, files in os.walk(demucs_out):
                for f in files:
                    if (f.endswith('.wav') or f.endswith('.mp3')) and ('other' in f or 'no_vocals' in f or 'instrumental' in f):
                        ext = Path(f).suffix
                        piano_dst = os.path.join(output_dir, f'piano{ext}')
                        shutil.copy2(os.path.join(root, f), piano_dst)
                        print(f"Instrumental stem saved to: {piano_dst}")
                        found = True
                        break
                if found:
                    break

        if not found:
            print("Error: Could not find instrumental stem in output", file=sys.stderr)
            sys.exit(1)

        # Cleanup temp directory
        shutil.rmtree(demucs_out, ignore_errors=True)

    except Exception as e:
        print(f"Error: Stem separation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
