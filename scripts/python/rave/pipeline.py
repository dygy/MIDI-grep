#!/usr/bin/env python3
"""
Generative Sound Model Pipeline.
End-to-end processing from audio to playable Strudel with trained models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rave.timbre_embeddings import TimbreAnalyzer, TimbreIndex, generate_model_id
from rave.trainer import RAVETrainer, GranularTrainer
from rave.repository import ModelRepository, generate_model_id as gen_model_id


class GenerativePipeline:
    """
    Main pipeline for creating generative sound models from audio.

    Flow:
    1. Separate stems (uses existing demucs separation)
    2. Analyze timbre of each stem
    3. Check if similar model exists in repository
    4. If match: use existing model
    5. If no match: train new model
    6. Generate Strudel code with note() control
    """

    def __init__(self,
                 models_path: str = "models",
                 github_repo: Optional[str] = None,
                 similarity_threshold: float = 0.88,
                 training_mode: str = "granular"):  # or "rave"
        """
        Initialize pipeline.

        Args:
            models_path: Path to model repository
            github_repo: Optional GitHub repo for sync
            similarity_threshold: Min similarity to reuse model
            training_mode: "granular" (fast) or "rave" (quality)
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.repository = ModelRepository(
            str(self.models_path),
            github_repo=github_repo
        )

        self.timbre_analyzer = TimbreAnalyzer()
        self.similarity_threshold = similarity_threshold
        self.training_mode = training_mode

        # Initialize trainer
        if training_mode == "rave":
            self.trainer = RAVETrainer(str(self.models_path))
        else:
            self.trainer = GranularTrainer(str(self.models_path))

    def process_track(self,
                      stems_dir: str,
                      track_id: str,
                      output_dir: str) -> Dict:
        """
        Process a track's stems into playable models.

        Args:
            stems_dir: Directory containing separated stems
            track_id: Unique track identifier
            output_dir: Where to save output

        Returns:
            Dict with model assignments and Strudel code
        """
        stems_dir = Path(stems_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"GENERATIVE PIPELINE: {track_id}")
        print(f"{'='*70}")
        print(f"Stems: {stems_dir}")
        print(f"Mode: {self.training_mode}")
        print(f"Similarity threshold: {self.similarity_threshold:.0%}")

        # Find stems
        stems = self._find_stems(stems_dir)
        print(f"\nFound stems: {list(stems.keys())}")

        # Process each stem
        models_used = {}
        for voice, stem_path in stems.items():
            print(f"\n--- Processing {voice} ---")
            model_id, is_new = self._process_stem(stem_path, voice, track_id)
            models_used[voice] = {
                "model_id": model_id,
                "is_new": is_new,
                "url": self.repository.get_strudel_url(model_id)
            }

        # Transcribe MIDI from melodic stem
        print("\n--- Transcribing notes ---")
        midi_data = self._transcribe_stem(stems.get("melodic", stems.get("piano")))

        # Generate Strudel code
        print("\n--- Generating Strudel code ---")
        strudel_code = self._generate_strudel(models_used, midi_data, track_id)

        # Save outputs
        strudel_path = output_dir / "output.strudel"
        with open(strudel_path, 'w') as f:
            f.write(strudel_code)

        metadata = {
            "track_id": track_id,
            "models": models_used,
            "training_mode": self.training_mode,
            "created": datetime.now().isoformat()
        }

        metadata_path = output_dir / "generative_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Sync to GitHub if configured
        if self.repository.github_repo:
            print("\n--- Syncing to GitHub ---")
            self.repository.sync_to_github()

        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Strudel code: {strudel_path}")
        print(f"Models used: {list(models_used.keys())}")

        return {
            "strudel_path": str(strudel_path),
            "models": models_used,
            "metadata": metadata
        }

    def _find_stems(self, stems_dir: Path) -> Dict[str, Path]:
        """Find stem files in directory."""
        stems = {}

        stem_names = {
            "melodic": ["melodic.wav", "melodic.mp3", "piano.wav", "piano.mp3", "other.wav"],
            "bass": ["bass.wav", "bass.mp3"],
            "drums": ["drums.wav", "drums.mp3"],
            "vocals": ["vocals.wav", "vocals.mp3"]
        }

        for voice, filenames in stem_names.items():
            for filename in filenames:
                path = stems_dir / filename
                if path.exists():
                    stems[voice] = path
                    break

        return stems

    def _process_stem(self,
                      stem_path: Path,
                      voice: str,
                      track_id: str) -> Tuple[str, bool]:
        """
        Process a single stem - find or create model.

        Returns:
            (model_id, is_new) tuple
        """
        # Extract timbre embedding
        print(f"Extracting timbre embedding...")
        embedding = self.timbre_analyzer.extract_embedding(str(stem_path))

        # Search for existing model
        print(f"Searching for similar models...")
        matches = self.repository.find_similar(
            embedding.tolist(),
            threshold=self.similarity_threshold
        )

        if matches:
            model_id, similarity = matches[0]
            print(f"Found match: {model_id} (similarity: {similarity:.1%})")
            return model_id, False

        # No match - train new model
        print(f"No match found. Training new model...")
        model_id = f"{voice}_{track_id[:8]}"

        if self.training_mode == "rave":
            metadata = self.trainer.train(
                str(stem_path),
                model_id,
                epochs=500
            )
        else:
            metadata = self.trainer.train(
                str(stem_path),
                model_id
            )

        # Add to repository
        model_path = self.models_path / model_id
        self.repository.add_model(
            str(model_path),
            model_id,
            embedding.tolist(),
            metadata
        )

        print(f"Created new model: {model_id}")
        return model_id, True

    def _transcribe_stem(self, stem_path: Optional[Path]) -> Dict:
        """Transcribe MIDI from melodic stem."""
        if stem_path is None:
            return {"notes": [], "bpm": 120, "key": "C major"}

        # Use basic_pitch for transcription
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            print(f"Running Basic Pitch on {stem_path}...")
            model_output, midi_data, note_events = predict(str(stem_path))

            notes = []
            for note in note_events:
                notes.append({
                    "pitch": note[2],  # MIDI pitch
                    "start": note[0],  # Start time
                    "end": note[1],    # End time
                    "velocity": note[3] if len(note) > 3 else 100
                })

            # Detect BPM
            import librosa
            y, sr = librosa.load(str(stem_path), sr=22050)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            return {
                "notes": notes,
                "bpm": float(tempo) if hasattr(tempo, '__float__') else float(tempo[0]),
                "key": "detected"  # Would need key detection
            }

        except ImportError:
            print("Basic Pitch not available, using onset detection...")
            return self._simple_transcribe(stem_path)

    def _simple_transcribe(self, stem_path: Path) -> Dict:
        """Simple transcription using librosa."""
        import librosa
        import numpy as np

        y, sr = librosa.load(str(stem_path), sr=22050)

        # Detect tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = tempo[0]

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Estimate pitches at onsets
        notes = []
        for onset_time in onset_times:
            onset_sample = int(onset_time * sr)
            chunk_duration = 0.1  # 100ms
            chunk = y[onset_sample:onset_sample + int(chunk_duration * sr)]

            if len(chunk) > 0:
                pitches, voiced, _ = librosa.pyin(
                    chunk, fmin=50, fmax=2000, sr=sr
                )
                pitch_hz = np.nanmedian(pitches) if np.any(~np.isnan(pitches)) else 440

                if pitch_hz > 0:
                    midi_pitch = int(round(12 * np.log2(pitch_hz / 440) + 69))
                else:
                    midi_pitch = 60

                notes.append({
                    "pitch": midi_pitch,
                    "start": float(onset_time),
                    "end": float(onset_time + chunk_duration),
                    "velocity": 100
                })

        return {
            "notes": notes,
            "bpm": float(tempo),
            "key": "C major"
        }

    def _generate_strudel(self,
                          models: Dict,
                          midi_data: Dict,
                          track_id: str) -> str:
        """Generate Strudel code using trained models."""
        bpm = midi_data.get("bpm", 120)
        notes = midi_data.get("notes", [])

        # Group notes into bars
        bar_duration = 60 / bpm * 4  # 4 beats per bar
        bars = []
        current_bar = []
        current_bar_start = 0

        for note in sorted(notes, key=lambda n: n["start"]):
            bar_index = int(note["start"] / bar_duration)

            while bar_index >= len(bars):
                if current_bar:
                    bars.append(current_bar)
                current_bar = []
                current_bar_start = len(bars) * bar_duration

            # Convert MIDI pitch to note name
            note_names = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']
            pitch = note["pitch"]
            note_name = note_names[pitch % 12]
            octave = (pitch // 12) - 1

            current_bar.append(f"{note_name}{octave}")

        if current_bar:
            bars.append(current_bar)

        # Build Strudel code
        lines = [
            f"// Generated by MIDI-grep Generative Pipeline",
            f"// Track: {track_id}",
            f"// BPM: {bpm:.0f}",
            f"// Models: {', '.join(models.keys())}",
            f"//",
            f"// FULL NOTE CONTROL - edit any pitch live!",
            f"",
            f"setcps({bpm:.0f}/60/4)",
            f""
        ]

        # Load samples/models
        if "melodic" in models:
            url = models["melodic"]["url"]
            lines.append(f'await samples("{url}")')
            lines.append("")

        # Create note patterns from transcription
        if bars:
            lines.append("// Melodic pattern (edit these notes freely!)")
            lines.append("let melody = [")
            for i, bar in enumerate(bars[:16]):  # First 16 bars
                bar_str = " ".join(bar) if bar else "~"
                lines.append(f'  "{bar_str}",')
            lines.append("]")
            lines.append("")

        # Playback with model
        melodic_model = models.get("melodic", {}).get("model_id", "melodic")
        bass_model = models.get("bass", {}).get("model_id", "bass")
        drums_model = models.get("drums", {}).get("model_id", "drums")

        lines.extend([
            "// Play with the trained model - sounds like the original!",
            f'$: note(cat(...melody)).sound("{melodic_model}")',
            "",
            "// You can change ANY note and it still sounds right:",
            f'// $: note("c3 e3 g3 b3").sound("{melodic_model}")',
            "",
            "// Bass (if model trained)",
        ])

        if "bass" in models:
            lines.append(f'// $: note("c2 ~ g2 ~").sound("{bass_model}")')
        else:
            lines.append(f'// No bass model for this track')

        lines.extend([
            "",
            "// Drums",
        ])

        if "drums" in models:
            lines.append(f'$: s("{drums_model}:0 {drums_model}:1 {drums_model}:0 {drums_model}:2")')
        else:
            lines.append('// $: s("bd sd bd sd")')

        lines.extend([
            "",
            "// Effects (adjust to taste)",
            "// .lpf(2000)",
            "// .room(0.3)",
            "// .delay(0.25)",
        ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generative sound model pipeline")
    parser.add_argument("stems_dir", help="Directory with separated stems")
    parser.add_argument("--track-id", required=True, help="Unique track identifier")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--models", default="models", help="Models directory")
    parser.add_argument("--github", help="GitHub repo (username/repo)")
    parser.add_argument("--mode", choices=["granular", "rave"], default="granular",
                        help="Training mode")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="Similarity threshold for reusing models")

    args = parser.parse_args()

    pipeline = GenerativePipeline(
        models_path=args.models,
        github_repo=args.github,
        similarity_threshold=args.threshold,
        training_mode=args.mode
    )

    result = pipeline.process_track(
        args.stems_dir,
        args.track_id,
        args.output
    )

    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
