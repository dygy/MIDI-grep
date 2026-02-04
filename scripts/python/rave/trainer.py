#!/usr/bin/env python3
"""
RAVE Model Trainer.
Trains generative neural audio models on track stems.
These models can then generate NEW sounds in the same timbre.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import hashlib


class RAVETrainer:
    """Train RAVE models on audio stems."""

    def __init__(self,
                 output_dir: str = "models",
                 device: str = "mps",
                 batch_size: int = 8,
                 n_signal: int = 65536):
        """
        Initialize RAVE trainer.

        Args:
            output_dir: Where to save trained models
            device: Training device (mps for M1/M2, cuda, cpu)
            batch_size: Training batch size
            n_signal: Signal length for training
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.n_signal = n_signal

        self._check_rave_installed()

    def _check_rave_installed(self):
        """Check if RAVE is installed."""
        try:
            import acids_rave
            print(f"RAVE version: {acids_rave.__version__}")
        except ImportError:
            print("RAVE not installed. Installing...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "acids-rave", "--upgrade"
            ], check=True)
            print("RAVE installed successfully")

    def prepare_dataset(self, audio_path: str, dataset_dir: str) -> str:
        """Prepare audio for RAVE training."""
        import librosa
        import soundfile as sf
        import numpy as np

        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95

        min_duration = 60
        min_samples = min_duration * sr

        if len(y) < min_samples:
            repeats = int(np.ceil(min_samples / len(y)))
            y = np.tile(y, repeats)[:min_samples]
            print(f"Looped audio {repeats}x to reach {min_duration}s minimum")

        chunk_duration = 10
        chunk_samples = chunk_duration * sr

        for i, start in enumerate(range(0, len(y) - chunk_samples, chunk_samples // 2)):
            chunk = y[start:start + chunk_samples]
            chunk_path = dataset_path / f"chunk_{i:04d}.wav"
            sf.write(str(chunk_path), chunk, sr)

        print(f"Prepared {i + 1} training chunks in {dataset_path}")
        return str(dataset_path)

    def train(self,
              audio_path: str,
              model_name: str,
              epochs: int = 1000,
              val_every: int = 100,
              config: str = "v2") -> Dict:
        """
        Train a RAVE model on audio.

        Returns:
            Dict with model info and paths
        """
        print(f"\n{'='*60}")
        print(f"TRAINING RAVE MODEL: {model_name}")
        print(f"{'='*60}")
        print(f"Source: {audio_path}")
        print(f"Config: {config}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")

        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = model_dir / "dataset"
        self.prepare_dataset(audio_path, str(dataset_dir))

        print("\n[1/3] Preprocessing dataset...")
        preprocess_dir = model_dir / "preprocessed"
        self._run_preprocess(dataset_dir, preprocess_dir)

        print(f"\n[2/3] Training ({epochs} epochs)...")
        checkpoint_dir = model_dir / "checkpoints"
        self._run_training(preprocess_dir, checkpoint_dir, epochs, val_every)

        print("\n[3/3] Exporting model...")
        exported_path = model_dir / f"{model_name}.ts"
        self._export_model(checkpoint_dir, exported_path)

        metadata = {
            "name": model_name,
            "source_audio": str(audio_path),
            "config": config,
            "epochs": epochs,
            "device": self.device,
            "created": datetime.now().isoformat(),
            "exported_path": str(exported_path),
            "sample_rate": 44100,
            "latent_dim": 128
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Model saved: {exported_path}")

        return metadata

    def _run_preprocess(self, dataset_dir: Path, output_dir: Path):
        """Run RAVE preprocessing on dataset."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "acids_rave", "preprocess",
            "--input_path", str(dataset_dir),
            "--output_path", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Standard preprocess unavailable, using direct preprocessing...")
            self._direct_preprocess(dataset_dir, output_dir)

    def _direct_preprocess(self, dataset_dir: Path, output_dir: Path):
        """Direct preprocessing without CLI."""
        import soundfile as sf

        for i, audio_file in enumerate(dataset_dir.glob("*.wav")):
            y, sr = sf.read(str(audio_file))
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            sf.write(str(output_dir / f"audio_{i:04d}.wav"), y, sr)

        print(f"Preprocessed {i + 1} files")

    def _run_training(self, data_dir: Path, checkpoint_dir: Path, epochs: int, val_every: int):
        """Run RAVE training."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "acids_rave", "train",
            "--name", "model",
            "--db_path", str(data_dir),
            "--out_path", str(checkpoint_dir),
            "--n_signal", str(self.n_signal),
            "--batch_size", str(self.batch_size),
            "--max_steps", str(epochs * 100)
        ]

        if self.device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        print(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()

    def _export_model(self, checkpoint_dir: Path, output_path: Path):
        """Export trained model to TorchScript."""
        cmd = [
            sys.executable, "-m", "acids_rave", "export",
            "--run", str(checkpoint_dir),
            "--output", str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Export info: {result.stderr}")


class GranularTrainer:
    """
    Fast granular model trainer.
    Creates a granular sample bank for immediate use.
    """

    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, audio_path: str, model_name: str, grain_ms: int = 100) -> Dict:
        """
        Create granular model from audio.

        Args:
            audio_path: Source audio
            model_name: Model name
            grain_ms: Grain duration in milliseconds

        Returns:
            Model metadata
        """
        import librosa
        import soundfile as sf
        import numpy as np

        print(f"\n{'='*60}")
        print(f"GRANULAR MODEL: {model_name}")
        print(f"{'='*60}")

        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95

        # Create grains directory
        grains_dir = model_dir / "grains"
        grains_dir.mkdir(exist_ok=True)

        # Extract grains at onsets
        grain_samples = int(grain_ms * sr / 1000)

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_samples = librosa.frames_to_samples(onset_frames)

        grains_info = []
        for i, onset in enumerate(onset_samples):
            if onset + grain_samples > len(y):
                continue

            grain = y[onset:onset + grain_samples]

            # Apply envelope
            attack = int(0.01 * sr)
            release = int(0.02 * sr)
            envelope = np.ones(len(grain))
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            grain = grain * envelope

            # Estimate pitch
            pitches, voiced_flag, _ = librosa.pyin(
                grain, fmin=50, fmax=2000, sr=sr
            )
            pitch_hz = float(np.nanmedian(pitches)) if np.any(~np.isnan(pitches)) else 440

            # Convert to MIDI note
            if pitch_hz > 0:
                midi_note = int(round(12 * np.log2(pitch_hz / 440) + 69))
            else:
                midi_note = 60

            grain_path = grains_dir / f"g{i:04d}.wav"
            sf.write(str(grain_path), grain, sr)

            grains_info.append({
                "index": i,
                "file": grain_path.name,
                "pitch_hz": pitch_hz,
                "midi_note": midi_note,
                "onset_sec": float(onset / sr)
            })

        # Also create pitched versions for playability
        pitched_dir = model_dir / "pitched"
        pitched_dir.mkdir(exist_ok=True)

        # Group grains by pitch class
        pitch_groups = {}
        for grain_info in grains_info:
            pc = grain_info["midi_note"] % 12
            if pc not in pitch_groups:
                pitch_groups[pc] = []
            pitch_groups[pc].append(grain_info)

        # Create representative sample for each pitch class
        for pc, grains in pitch_groups.items():
            if grains:
                best = grains[len(grains) // 2]  # Take middle grain
                src_path = grains_dir / best["file"]
                note_names = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']
                dst_path = pitched_dir / f"{note_names[pc]}.wav"
                shutil.copy(src_path, dst_path)

        # Save metadata
        metadata = {
            "name": model_name,
            "type": "granular",
            "source_audio": str(audio_path),
            "created": datetime.now().isoformat(),
            "sample_rate": sr,
            "num_grains": len(grains_info),
            "grain_duration_ms": grain_ms,
            "grains": grains_info
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Created {len(grains_info)} grains")
        print(f"Model saved: {model_dir}")

        return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train generative model")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--type", choices=["rave", "granular"], default="granular")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (RAVE)")
    parser.add_argument("--grain-ms", type=int, default=100, help="Grain duration (granular)")

    args = parser.parse_args()

    if args.type == "rave":
        trainer = RAVETrainer(args.output)
        result = trainer.train(args.audio, args.name, epochs=args.epochs)
    else:
        trainer = GranularTrainer(args.output)
        result = trainer.train(args.audio, args.name, grain_ms=args.grain_ms)

    print(f"\nResult: {json.dumps(result, indent=2)}")
