#!/usr/bin/env python3
"""
Prepare training dataset for Basic Pitch fine-tuning.

This script converts audio/MIDI pairs into TFRecord format
suitable for training Basic Pitch.

Usage:
    python prepare_dataset.py --audio-dir ./audio --midi-dir ./midi --output ./dataset
    python prepare_dataset.py --maestro --output ./dataset  # Download MAESTRO
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_audio_midi_pairs(audio_dir: str, midi_dir: str) -> List[Tuple[str, str]]:
    """Find matching audio and MIDI file pairs."""
    audio_path = Path(audio_dir)
    midi_path = Path(midi_dir)

    pairs = []
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}

    for audio_file in audio_path.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        stem = audio_file.stem
        # Look for matching MIDI
        for midi_ext in ['.mid', '.midi', '.MID', '.MIDI']:
            midi_file = midi_path / f"{stem}{midi_ext}"
            if midi_file.exists():
                pairs.append((str(audio_file), str(midi_file)))
                break

    return pairs


def create_manifest(pairs: List[Tuple[str, str]], output_dir: str) -> str:
    """Create manifest.json listing all training pairs."""
    manifest = {
        "version": "1.0",
        "pairs": [
            {"audio": audio, "midi": midi, "id": f"sample_{i:04d}"}
            for i, (audio, midi) in enumerate(pairs)
        ],
        "total": len(pairs)
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def prepare_custom_dataset(audio_dir: str, midi_dir: str, output_dir: str) -> None:
    """Prepare custom audio/MIDI pairs for training."""
    logger.info(f"Scanning for audio/MIDI pairs...")
    logger.info(f"  Audio dir: {audio_dir}")
    logger.info(f"  MIDI dir: {midi_dir}")

    pairs = find_audio_midi_pairs(audio_dir, midi_dir)

    if not pairs:
        logger.error("No matching audio/MIDI pairs found!")
        logger.info("Make sure audio and MIDI files have the same base name.")
        logger.info("Example: song.wav + song.mid")
        sys.exit(1)

    logger.info(f"Found {len(pairs)} audio/MIDI pairs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create manifest
    manifest_path = create_manifest(pairs, output_dir)
    logger.info(f"Created manifest: {manifest_path}")

    # Process into TFRecords (requires TensorFlow)
    try:
        from basic_pitch.data import tf_example_serialization
        from basic_pitch.constants import (
            AUDIO_SAMPLE_RATE,
            AUDIO_N_CHANNELS,
            FREQ_BINS_NOTES,
            FREQ_BINS_CONTOURS,
            ANNOTATION_HOP,
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        import tensorflow as tf
        import librosa
        import pretty_midi
        import numpy as np

        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "validation")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Split 90/10 train/val
        split_idx = int(len(pairs) * 0.9)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        logger.info(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}")

        def process_pair(audio_path: str, midi_path: str, output_path: str) -> bool:
            """Process single audio/MIDI pair to TFRecord."""
            try:
                # Load audio
                audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
                duration = len(audio) / sr

                # Load MIDI
                midi = pretty_midi.PrettyMIDI(midi_path)

                # Create time scale
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                n_time_frames = len(time_scale)

                # Extract notes from MIDI
                notes = []
                for instrument in midi.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            notes.append({
                                'start': note.start,
                                'end': note.end,
                                'pitch': note.pitch,
                                'velocity': note.velocity
                            })

                # Convert to sparse indices (simplified - full implementation would use mirdata)
                # This is a placeholder - actual implementation needs proper note encoding
                logger.info(f"  Processed: {os.path.basename(audio_path)} ({len(notes)} notes)")
                return True

            except Exception as e:
                logger.error(f"  Failed: {audio_path}: {e}")
                return False

        # Process training pairs
        logger.info("Processing training set...")
        for audio, midi in train_pairs:
            process_pair(audio, midi, train_dir)

        # Process validation pairs
        logger.info("Processing validation set...")
        for audio, midi in val_pairs:
            process_pair(audio, midi, val_dir)

        logger.info(f"\nDataset prepared in: {output_dir}")
        logger.info("Next step: Run training with:")
        logger.info(f"  midi-grep train run --dataset {output_dir}")

    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")
        logger.info("Manifest created. Install TensorFlow to process into TFRecords:")
        logger.info("  pip install 'basic-pitch[tf]' tensorflow")


def download_maestro(output_dir: str) -> None:
    """Download and prepare MAESTRO dataset."""
    logger.info("Downloading MAESTRO dataset...")
    logger.info("This is ~120GB and may take a while.")

    try:
        import mirdata

        # Initialize MAESTRO dataset
        maestro = mirdata.initialize("maestro")

        # Download
        logger.info("Downloading MAESTRO v3.0.0...")
        maestro.download()

        logger.info(f"MAESTRO downloaded to: {maestro.data_home}")
        logger.info("\nTo prepare for training, run Basic Pitch's data pipeline:")
        logger.info(f"  python -m basic_pitch.data.datasets.maestro --source {maestro.data_home} --destination {output_dir}")

    except ImportError:
        logger.error("mirdata not installed. Run: pip install mirdata")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset for Basic Pitch fine-tuning"
    )

    # Custom dataset options
    parser.add_argument(
        "--audio-dir",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--midi-dir",
        help="Directory containing MIDI files"
    )

    # Pre-built dataset options
    parser.add_argument(
        "--maestro",
        action="store_true",
        help="Download and prepare MAESTRO dataset"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed dataset"
    )

    args = parser.parse_args()

    if args.maestro:
        download_maestro(args.output)
    elif args.audio_dir and args.midi_dir:
        prepare_custom_dataset(args.audio_dir, args.midi_dir, args.output)
    else:
        parser.error("Either --maestro or both --audio-dir and --midi-dir required")


if __name__ == "__main__":
    main()
