#!/usr/bin/env python3
"""
Stem separation using Demucs.
Extracts piano/other stem and/or drum stem from mixed audio.

Modes:
  - piano (default): Extract instrumental/piano stem only
  - drums: Extract drum stem only
  - full: Extract both piano and drums stems
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
import subprocess


def check_gpu_available():
    """Check if CUDA GPU is available for acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Separate audio into stems using Demucs')
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('output_dir', help='Output directory for stems')
    parser.add_argument('--mode', choices=['piano', 'drums', 'full'], default='piano',
                        help='Separation mode: piano (default), drums, or full')
    parser.add_argument('--quality', choices=['fast', 'normal', 'high', 'best'], default='normal',
                        help='Quality preset: fast (quick), normal (default), high (better, slower), best (highest quality, slowest)')

    args = parser.parse_args()

    # Quality presets
    quality_settings = {
        'fast': {
            'model': 'htdemucs',
            'shifts': 0,
            'overlap': 0.25,
            'format': 'mp3',
        },
        'normal': {
            'model': 'htdemucs',
            'shifts': 1,
            'overlap': 0.25,
            'format': 'mp3',
        },
        'high': {
            'model': 'htdemucs_ft',  # Fine-tuned model
            'shifts': 3,              # Test-time augmentation
            'overlap': 0.5,           # More overlap
            'format': 'wav',          # Lossless
        },
        'best': {
            'model': 'htdemucs_ft',  # Fine-tuned model
            'shifts': 5,              # Maximum TTA
            'overlap': 0.75,          # Maximum overlap
            'format': 'wav',          # Lossless
        },
    }

    quality = quality_settings[args.quality]
    use_gpu = check_gpu_available()

    if args.quality in ['high', 'best']:
        print(f"Using high-quality mode: model={quality['model']}, shifts={quality['shifts']}, overlap={quality['overlap']}")
        if use_gpu:
            print("GPU acceleration: enabled")
        else:
            print("GPU acceleration: disabled (no CUDA GPU found)")

    input_path = args.input_audio
    output_dir = args.output_dir
    mode = args.mode

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        demucs_out = os.path.join(output_dir, 'demucs_temp')
        os.makedirs(demucs_out, exist_ok=True)

        input_name = Path(input_path).stem

        # Build base demucs command with quality settings
        def build_demucs_cmd(extra_args=None):
            cmd = [
                sys.executable, '-m', 'demucs',
                '-n', quality['model'],
                '-o', demucs_out,
            ]
            # Add quality options
            if quality['shifts'] > 0:
                cmd.extend(['--shifts', str(quality['shifts'])])
            if quality['overlap'] > 0:
                cmd.extend(['--overlap', str(quality['overlap'])])
            if quality['format'] == 'mp3':
                cmd.append('--mp3')
            # GPU acceleration
            if use_gpu:
                cmd.extend(['--device', 'cuda'])
            # Extra args
            if extra_args:
                cmd.extend(extra_args)
            cmd.append(input_path)
            return cmd

        if mode == 'piano':
            # Try two-stem first, then full separation
            result = subprocess.run(
                build_demucs_cmd(['--two-stems=vocals']),
                capture_output=True, text=True
            )

            if result.returncode != 0:
                # Fallback to full separation
                result = subprocess.run(
                    build_demucs_cmd(),
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    print(f"Error: Demucs failed: {result.stderr}", file=sys.stderr)
                    sys.exit(1)

            # Find piano stem
            piano_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['no_vocals', 'other'],
                'piano'
            )
            if piano_dst:
                print(f"Piano/instrumental stem saved to: {piano_dst}")
            else:
                print("Error: Could not find instrumental stem in output", file=sys.stderr)
                sys.exit(1)

        elif mode == 'drums':
            # Full separation to get drums
            result = subprocess.run(
                build_demucs_cmd(),
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"Error: Demucs failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)

            # Find drums stem
            drums_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['drums'],
                'drums'
            )
            if drums_dst:
                print(f"Drums stem saved to: {drums_dst}")
            else:
                print("Error: Could not find drums stem in output", file=sys.stderr)
                sys.exit(1)

        else:  # mode == 'full'
            # Full separation to get all stems
            result = subprocess.run(
                build_demucs_cmd(),
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"Error: Demucs failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)

            found_any = False

            # Find and copy melodic stem (from 'other')
            piano_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['other'],
                'piano'  # Keep as 'piano' for backwards compat
            )
            if piano_dst:
                print(f"Melodic stem saved to: {piano_dst}")
                found_any = True

            # Find and copy bass stem
            bass_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['bass'],
                'bass'
            )
            if bass_dst:
                print(f"Bass stem saved to: {bass_dst}")
                found_any = True

            # Find and copy drums stem
            drums_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['drums'],
                'drums'
            )
            if drums_dst:
                print(f"Drums stem saved to: {drums_dst}")
                found_any = True

            # Find and copy vocals stem
            vocals_dst = find_and_copy_stem(
                demucs_out, input_name, output_dir,
                ['vocals'],
                'vocals'
            )
            if vocals_dst:
                print(f"Vocals stem saved to: {vocals_dst}")
                found_any = True

            if not found_any:
                print("Error: Could not find any stems in output", file=sys.stderr)
                sys.exit(1)

        # Cleanup temp directory
        shutil.rmtree(demucs_out, ignore_errors=True)

    except Exception as e:
        print(f"Error: Stem separation failed: {e}", file=sys.stderr)
        sys.exit(1)


def find_and_copy_stem(demucs_out, input_name, output_dir, stem_names, output_name):
    """
    Find a stem file from demucs output and copy it to the output directory.

    Args:
        demucs_out: Demucs output directory
        input_name: Original input file name (without extension)
        output_dir: Where to copy the stem
        stem_names: List of possible stem names to look for (in order of preference)
        output_name: Name to use for output file (e.g., 'piano', 'drums')

    Returns:
        Path to copied file, or None if not found
    """
    # Build list of possible paths
    possible_paths = []
    for model_name in ['htdemucs_ft', 'htdemucs']:  # Check both models
        for stem in stem_names:
            for ext in ['.mp3', '.wav']:
                possible_paths.append(
                    os.path.join(demucs_out, model_name, input_name, f'{stem}{ext}')
                )

    # Try each path
    for src_path in possible_paths:
        if os.path.exists(src_path):
            ext = Path(src_path).suffix
            dst_path = os.path.join(output_dir, f'{output_name}{ext}')
            shutil.copy2(src_path, dst_path)
            return dst_path

    # Search recursively as fallback
    for root, dirs, files in os.walk(demucs_out):
        for f in files:
            if any(stem in f.lower() for stem in stem_names):
                if f.endswith('.wav') or f.endswith('.mp3'):
                    ext = Path(f).suffix
                    dst_path = os.path.join(output_dir, f'{output_name}{ext}')
                    shutil.copy2(os.path.join(root, f), dst_path)
                    return dst_path

    return None


if __name__ == '__main__':
    main()
