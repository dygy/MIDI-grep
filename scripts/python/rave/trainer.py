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
    """Train RAVE models on audio stems with the real ``rave`` CLI (pip ``acids-rave``).

    The pip package ``acids-rave`` installs a console script ``rave`` (``rave preprocess|train|export ...``) and the
    python packages ``rave`` and ``scripts`` - there is NO ``acids_rave`` module, and this repo's own ``rave/`` package
    shadows the installed ``rave`` for a ``python -m rave`` run from ``scripts/python``. So the CLI executable beside the
    running interpreter is called directly, from the model directory, and every step fails LOUDLY: a missing CLI, a
    non-zero exit, or a missing output raises ``RuntimeError`` (nothing is copied or faked as a "preprocessed" dataset).
    Flags are the ones ``rave <command> --help`` lists (acids-rave 2.3.1): ``--batch`` (not ``--batch_size``),
    ``--config``, ``--sampling_rate``/``--num_signal`` for preprocess, ``--run``/``--name``/``--output`` for export.
    """

    CLI_NAME = "rave"

    def __init__(self,
                 output_dir: str = "models",
                 device: str = "cpu",
                 batch_size: int = 8,
                 n_signal: int = 65536,
                 sample_rate: int = 44100):
        """
        Args:
            output_dir: Where to save trained models
            device: ``cuda`` or ``cpu``. ``mps`` is accepted but acids-rave's train script exits on a Mac GPU ("Training
                on mac is not available yet"), so it trains on the CPU (``--gpu -1``), loudly.
            batch_size: Training batch size
            n_signal: Signal length (samples) for training and for the preprocessed chunks
            sample_rate: Sampling rate the dataset is preprocessed at
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.n_signal = n_signal
        self.sample_rate = sample_rate
        self.cli = self._find_cli()

    @classmethod
    def _find_cli(cls) -> str:
        """Path of the installed ``rave`` console script (beside this interpreter, else on PATH); RuntimeError if absent."""
        beside = Path(sys.executable).parent / cls.CLI_NAME
        if beside.is_file():
            return str(beside)
        found = shutil.which(cls.CLI_NAME)
        if found:
            return found
        raise RuntimeError(
            f"the '{cls.CLI_NAME}' CLI is not installed (pip package acids-rave); install it into the venv with "
            f"`pip install acids-rave` - it pins torch, so check `pip list | grep torch` afterwards")

    def preprocess_cmd(self, dataset_dir: Path, output_dir: Path) -> list:
        return [self.cli, "preprocess",
                "--input_path", str(dataset_dir), "--output_path", str(output_dir),
                "--sampling_rate", str(self.sample_rate), "--channels", "1",
                "--num_signal", str(self.n_signal)]

    def train_cmd(self, data_dir: Path, out_dir: Path, name: str, max_steps: int, val_every: int,
                  config: str = "v2") -> list:
        cmd = [self.cli, "train",
               "--name", name, "--db_path", str(data_dir), "--out_path", str(out_dir),
               "--config", config, "--n_signal", str(self.n_signal), "--batch", str(self.batch_size),
               "--max_steps", str(max_steps), "--val_every", str(val_every), "--save_every", str(val_every)]
        if self.device != "cuda":
            cmd += ["--gpu", "-1"]   # train.py: -1 = CPU; without it a Mac exits and a box with no GPU probes for one
        return cmd

    def export_cmd(self, run_dir: Path, name: str, output_dir: Path) -> list:
        return [self.cli, "export", "--run", str(run_dir), "--name", name, "--output", str(output_dir)]

    def _run(self, cmd: list, cwd: Path, what: str) -> None:
        """Run one CLI step from ``cwd`` (never scripts/python: its ``rave/`` would shadow the installed package),
        streaming its output; a non-zero exit raises."""
        print(f"Running: {' '.join(cmd)}")
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: list = []
        for line in proc.stdout:
            print(line, end="")
            tail = (tail + [line])[-20:]
        if proc.wait() != 0:
            raise RuntimeError(f"rave {what} failed (exit {proc.returncode}); last output:\n{''.join(tail)}")

    def prepare_dataset(self, audio_path: str, dataset_dir: str) -> str:
        """Prepare audio for RAVE training."""
        import librosa
        import soundfile as sf
        import numpy as np

        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95

        min_duration = 60
        min_samples = min_duration * sr

        if len(y) < min_samples:
            repeats = int(np.ceil(min_samples / len(y)))
            y = np.tile(y, repeats)[:min_samples]
            print(f"Looped audio {repeats}x to reach {min_duration}s minimum")

        chunk_duration = 10
        chunk_samples = chunk_duration * sr

        n_chunks = 0
        for i, start in enumerate(range(0, len(y) - chunk_samples, chunk_samples // 2)):
            chunk = y[start:start + chunk_samples]
            chunk_path = dataset_path / f"chunk_{i:04d}.wav"
            sf.write(str(chunk_path), chunk, sr)
            n_chunks = i + 1

        if n_chunks == 0:
            raise RuntimeError(f"{audio_path}: too short to cut a single {chunk_duration}s training chunk")
        print(f"Prepared {n_chunks} training chunks in {dataset_path}")
        return str(dataset_path)

    def train(self,
              audio_path: str,
              model_name: str,
              epochs: int = 1000,
              val_every: int = 100,
              config: str = "v2") -> Dict:
        """
        Train a RAVE model on audio (preprocess -> train -> export, each a real ``rave`` CLI call).

        Returns:
            Dict with model info and paths
        """
        print(f"\n{'='*60}")
        print(f"TRAINING RAVE MODEL: {model_name}")
        print(f"{'='*60}")
        print(f"Source: {audio_path}")
        print(f"Config: {config}")
        print(f"Epochs: {epochs}")
        device = self.device
        if device == "mps":
            print("Device: mps requested, but acids-rave cannot train on a Mac GPU -> CPU (--gpu -1)")
            device = "cpu"
        print(f"Device: {device}")

        model_dir = (self.output_dir / model_name).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = model_dir / "dataset"
        self.prepare_dataset(audio_path, str(dataset_dir))

        print("\n[1/3] Preprocessing dataset...")
        preprocess_dir = model_dir / "preprocessed"
        self._run_preprocess(dataset_dir, preprocess_dir)

        print(f"\n[2/3] Training ({epochs} epochs)...")
        checkpoint_dir = model_dir / "checkpoints"
        run_dir = self._run_training(preprocess_dir, checkpoint_dir, model_name, epochs, val_every, config)

        print("\n[3/3] Exporting model...")
        exported_path = model_dir / f"{model_name}.ts"
        self._export_model(run_dir, model_name, exported_path)

        metadata = {
            "name": model_name,
            "source_audio": str(audio_path),
            "config": config,
            "epochs": epochs,
            "device": device,
            "created": datetime.now().isoformat(),
            "exported_path": str(exported_path),
            "sample_rate": self.sample_rate,
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
        """``rave preprocess``: wavs -> the lmdb dataset ``rave train`` reads. Raises if it fails or writes nothing."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run(self.preprocess_cmd(dataset_dir, output_dir), output_dir.parent, "preprocess")
        if not any(output_dir.iterdir()):
            raise RuntimeError(f"rave preprocess wrote nothing to {output_dir}")

    def _run_training(self, data_dir: Path, checkpoint_dir: Path, name: str, epochs: int, val_every: int,
                      config: str = "v2") -> Path:
        """``rave train``; returns the run directory it created (``<out_path>/<name>_<config hash>``)."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self._run(self.train_cmd(data_dir, checkpoint_dir, name, epochs * 100, val_every, config),
                  checkpoint_dir.parent, "train")
        runs = sorted((p for p in checkpoint_dir.glob(f"{name}_*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not runs or not any(runs[-1].rglob("*.ckpt")):
            raise RuntimeError(f"rave train left no checkpoint under {checkpoint_dir}/{name}_*")
        return runs[-1]

    def _export_model(self, run_dir: Path, name: str, output_path: Path):
        """``rave export``: the run's last checkpoint -> TorchScript ``output_path`` (<dir>/<name>.ts)."""
        self._run(self.export_cmd(run_dir, name, output_path.parent), output_path.parent, "export")
        if not output_path.is_file():
            raise RuntimeError(f"rave export did not write {output_path}")


class GranularTrainer:
    """
    Fast granular model trainer.
    Creates a granular sample bank for immediate use.
    """

    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, audio_path: str, model_name: str, grain_ms: Optional[float] = None,
              pitch_fmin: Optional[float] = None, pitch_fmax: Optional[float] = None,
              midi_path: Optional[str] = None) -> Dict:
        """
        Create granular model from audio.

        With ``midi_path`` (the stem's transcribed notes) every grain is ONE note, from its onset to its
        end (clipped at the next onset), so grain length comes from the music; the pitch label is measured
        on the grain (pitched_grains.measure_pitch) and the search range comes from the transcribed
        register. Without MIDI (onset fallback) the grain length is ``grain_ms`` if the caller sets it, else the
        lower quartile of the stem's own inter-onset intervals (so most grains hold a single onset), and the
        pitch range is derived from the stem's long-term spectrum unless ``pitch_fmin``/``pitch_fmax`` are given.

        Args:
            audio_path: Source audio
            model_name: Model name
            grain_ms: Grain duration in milliseconds
            attack_ms: per-grain fade-in (ms)
            release_ms: per-grain fade-out (ms). A LONG release (e.g. 150ms on a 600ms grain)
                gives a sustained sample whose tail overlaps the next retriggered note, so a
                densely-retriggered pattern (short MIDI notes) reads as a continuous, sustained
                line instead of a string of attack transients — matching a real synth bass.

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

        # Create grains directory (fresh: a retrain must not leave stale samples that metadata no longer names)
        grains_dir = model_dir / "grains"
        pitched_dir = model_dir / "pitched"
        for d in (grains_dir, pitched_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(exist_ok=True)
        NN = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']

        def key_and_file(midi_note: int):
            octave = midi_note // 12 - 1
            return (f"{NN[midi_note % 12]}{octave}".replace('s', '#'),
                    f"{NN[midi_note % 12]}{octave}.wav".replace('s', '_sharp_'))

        grains_info = []
        pitched_map = {}
        if midi_path:
            import pitched_grains as pg
            notes = pg.read_midi_notes(midi_path)
            if not notes:
                raise ValueError(f"{midi_path} has no notes - cannot cut note-aligned grains")
            fmin, fmax = pg.register_hz([n.pitch for n in notes])
            target = pg.typical_note_length(notes)
            print(f"note-aligned grains from {len(notes)} transcribed notes; search {fmin:.1f}-{fmax:.1f} Hz; "
                  f"target length P75 = {target:.3f}s")
            grains = pg.extract_note_grains(y, sr, notes, fmin, fmax, ring=pg.StemRing(y, sr))
            target = pg.ring_target_length(grains, target)
            chosen = pg.select_per_pitch(grains, target, pg.stub_note_length(notes))
            for i, g in enumerate(grains):
                grains_info.append({
                    "index": i, "pitch_hz": pg.midi_to_hz(g.note), "midi_note": g.note,
                    "transcribed_note": g.transcribed, "onset_sec": g.onset_s, "duration_sec": g.duration_s,
                    "label_method": g.method, "quality": g.quality,
                })
            for midi_note, g in sorted(chosen.items()):
                key, fn = key_and_file(midi_note)
                sf.write(str(pitched_dir / fn), g.audio, sr)
                pitched_map[key] = f"pitched/{fn}"
            grain_mode = "notes"
            extra = {"pitch_range_hz": [fmin, fmax], "target_length_sec": target,
                     "pitched_durations_sec": {key_and_file(m)[0]: round(g.duration_s, 4) for m, g in sorted(chosen.items())}}
        else:
            import pitched_grains as pg
            onset_samples = librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr))
            if grain_ms is None:
                ioi = np.diff(onset_samples) / sr
                if len(ioi) == 0:
                    raise ValueError("no onsets found - cannot derive a grain length from this stem")
                grain_ms = float(np.percentile(ioi, 25.0)) * 1000.0   # P25 IOI: 3 of 4 grains contain one onset
            grain_samples = int(grain_ms * sr / 1000)
            if pitch_fmin is None or pitch_fmax is None:
                spec = np.abs(librosa.stft(y)) ** 2
                cum = np.cumsum(spec.mean(axis=1)); cum /= cum[-1]
                freqs = librosa.fft_frequencies(sr=sr)
                d_lo = float(freqs[int(np.searchsorted(cum, 0.01))])     # 1 % of energy lies below
                d_hi = float(freqs[int(np.searchsorted(cum, 0.85))])     # 85 % roll-off
                pitch_fmin = pitch_fmin or max(d_lo, 20.0)               # 20 Hz: lower edge of hearing
                pitch_fmax = pitch_fmax or d_hi
            fmin, fmax = float(pitch_fmin), float(pitch_fmax)
            by_note = {}
            for i, onset in enumerate(onset_samples):
                if onset + grain_samples > len(y):
                    continue
                grain = y[onset:onset + grain_samples]
                pm = pg.measure_pitch(grain, sr, fmin, fmax)   # voiced + stable, label = measured median
                if pm is None:
                    continue  # percussive/noisy grain - not a usable pitched sample
                grain_path = grains_dir / f"g{i:04d}.wav"
                sf.write(str(grain_path), pg.shape_grain(grain, sr, pm.hz), sr)
                row = {"index": i, "file": grain_path.name, "pitch_hz": pm.hz, "midi_note": pm.note,
                       "onset_sec": float(onset / sr), "vfrac": pm.voiced_fraction, "pstd": pm.stability_st,
                       "quality": pm.quality}
                grains_info.append(row)
                by_note.setdefault(pm.note, []).append(row)
            for midi_note, gs in by_note.items():
                best = max(gs, key=lambda g: g["quality"])
                key, fn = key_and_file(midi_note)
                shutil.copy(grains_dir / best["file"], pitched_dir / fn)
                pitched_map[key] = f"pitched/{fn}"
            grain_mode = "onsets"
            extra = {"pitch_range_hz": [fmin, fmax]}
        print(f"Clean pitched instrument: {len(pitched_map)} notes -> {sorted(pitched_map)}")

        # Save metadata
        metadata = {
            "name": model_name,
            "type": "granular",
            "source_audio": str(audio_path),
            "created": datetime.now().isoformat(),
            "sample_rate": sr,
            "num_grains": len(grains_info),
            "grain_mode": grain_mode,
            "grain_duration_ms": round(grain_ms, 2) if grain_mode == "onsets" else None,
            "midi_path": str(midi_path) if midi_path else None,
            "pitched_map": pitched_map,
            "grains": grains_info,
            **extra,
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
    parser.add_argument("--grain-ms", type=float, default=None, help="Grain duration (granular, onset fallback)")
    parser.add_argument("--midi", default=None, help="Transcribed notes (.mid): cut one grain per note")

    args = parser.parse_args()

    if args.type == "rave":
        trainer = RAVETrainer(args.output)   # raises RuntimeError (no 'rave' CLI / a failed step) - never a fake model
        result = trainer.train(args.audio, args.name, epochs=args.epochs)
    else:
        trainer = GranularTrainer(args.output)
        result = trainer.train(args.audio, args.name, grain_ms=args.grain_ms, midi_path=args.midi)

    print(f"\nResult: {json.dumps(result, indent=2)}")
