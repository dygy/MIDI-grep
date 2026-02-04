#!/usr/bin/env python3
"""
Timbre Embedding Extraction and Comparison.
Uses OpenL3 or CLAP to create embeddings that represent "what it sounds like".
These embeddings enable finding similar timbres across your model library.
"""

import numpy as np
import librosa
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class TimbreAnalyzer:
    """Extract and compare timbre embeddings from audio."""

    def __init__(self, model_type: str = "openl3"):
        """
        Initialize the timbre analyzer.

        Args:
            model_type: "openl3" (lightweight) or "clap" (more powerful)
        """
        self.model_type = model_type
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        if self.model_type == "openl3":
            try:
                import openl3
                self.model = openl3
                self.embedding_dim = 512  # OpenL3 default
                print("Loaded OpenL3 embedding model")
            except ImportError:
                print("OpenL3 not available, falling back to librosa features")
                self.model_type = "librosa"
                self.model = None
                self.embedding_dim = 128

        elif self.model_type == "clap":
            try:
                import laion_clap
                self.model = laion_clap.CLAP_Module(enable_fusion=False)
                self.model.load_ckpt()
                self.embedding_dim = 512
                print("Loaded CLAP embedding model")
            except ImportError:
                print("CLAP not available, trying OpenL3...")
                self.model_type = "openl3"
                self._load_model()
        else:
            # Fallback: use librosa features as pseudo-embedding
            self.model = None
            self.embedding_dim = 128
            print("Using librosa-based timbre features")

    def extract_embedding(self, audio_path: str, duration: float = 30.0) -> np.ndarray:
        """
        Extract timbre embedding from audio file.

        Args:
            audio_path: Path to audio file
            duration: Max duration to analyze (seconds)

        Returns:
            Embedding vector (normalized)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=duration)

        if self.model_type == "openl3" and self.model is not None:
            # Use OpenL3
            emb, ts = self.model.get_audio_embedding(y, sr, embedding_size=512)
            # Average over time
            embedding = np.mean(emb, axis=0)

        elif self.model_type == "clap" and self.model is not None:
            # Use CLAP
            # CLAP expects file path or specific format
            embedding = self.model.get_audio_embedding_from_data([y], sr)[0]

        else:
            # Fallback: librosa features
            embedding = self._librosa_embedding(y, sr)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def _librosa_embedding(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Create embedding from librosa features (fallback)."""
        features = []

        # MFCCs (timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        # Chroma (harmonic content)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # Zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # RMS energy dynamics
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms) - np.min(rms))

        # Onset strength (attack character)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        features.append(np.mean(onset))
        features.append(np.std(onset))

        # Pad or truncate to fixed size
        embedding = np.array(features[:self.embedding_dim])
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        return embedding

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))

    def find_similar(self,
                     query_embedding: np.ndarray,
                     index: Dict[str, np.ndarray],
                     threshold: float = 0.85,
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar models in the index.

        Args:
            query_embedding: Embedding to search for
            index: Dict of model_id -> embedding
            threshold: Minimum similarity to consider a match
            top_k: Maximum number of results

        Returns:
            List of (model_id, similarity) tuples, sorted by similarity
        """
        results = []

        for model_id, embedding in index.items():
            embedding = np.array(embedding)
            similarity = self.cosine_similarity(query_embedding, embedding)

            if similarity >= threshold:
                results.append((model_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


class TimbreIndex:
    """Manage a searchable index of timbre embeddings."""

    def __init__(self, index_path: str):
        """
        Initialize or load timbre index.

        Args:
            index_path: Path to index.json file
        """
        self.index_path = Path(index_path)
        self.index = self._load_index()
        self.analyzer = TimbreAnalyzer()

    def _load_index(self) -> Dict:
        """Load existing index or create empty one."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {
            "version": "1.0",
            "embedding_model": "openl3",
            "embedding_dim": 512,
            "models": {}
        }

    def save(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_model(self, model_id: str, audio_path: str, metadata: Dict = None):
        """
        Add a model to the index.

        Args:
            model_id: Unique identifier for the model
            audio_path: Path to the source audio
            metadata: Optional metadata dict
        """
        embedding = self.analyzer.extract_embedding(audio_path)

        self.index["models"][model_id] = {
            "embedding": embedding.tolist(),
            "source": audio_path,
            "metadata": metadata or {}
        }

        self.save()
        print(f"Added model {model_id} to index")

    def find_match(self, audio_path: str, threshold: float = 0.88) -> Optional[Tuple[str, float]]:
        """
        Find a matching model for the given audio.

        Args:
            audio_path: Path to audio to match
            threshold: Minimum similarity for a match

        Returns:
            (model_id, similarity) if match found, None otherwise
        """
        query_embedding = self.analyzer.extract_embedding(audio_path)

        # Build embedding dict
        embeddings = {
            model_id: np.array(data["embedding"])
            for model_id, data in self.index["models"].items()
        }

        matches = self.analyzer.find_similar(query_embedding, embeddings, threshold)

        if matches:
            return matches[0]
        return None

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get metadata for a model."""
        return self.index["models"].get(model_id)


def generate_model_id(audio_path: str, prefix: str = "model") -> str:
    """
    Generate a unique model ID based on audio content.

    Args:
        audio_path: Path to audio file
        prefix: Prefix for the ID

    Returns:
        Unique model ID like "model_a3f2b1c4"
    """
    # Hash the file content
    with open(audio_path, 'rb') as f:
        content_hash = hashlib.md5(f.read()).hexdigest()[:8]

    return f"{prefix}_{content_hash}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Timbre embedding tools")
    parser.add_argument("command", choices=["extract", "compare", "search"])
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("--audio2", help="Second audio for comparison")
    parser.add_argument("--index", default="models/index.json", help="Index path")

    args = parser.parse_args()

    analyzer = TimbreAnalyzer()

    if args.command == "extract":
        emb = analyzer.extract_embedding(args.audio)
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding (first 10): {emb[:10]}")

    elif args.command == "compare":
        if not args.audio2:
            print("Need --audio2 for comparison")
            exit(1)
        emb1 = analyzer.extract_embedding(args.audio)
        emb2 = analyzer.extract_embedding(args.audio2)
        sim = analyzer.cosine_similarity(emb1, emb2)
        print(f"Similarity: {sim:.2%}")

    elif args.command == "search":
        index = TimbreIndex(args.index)
        match = index.find_match(args.audio)
        if match:
            print(f"Found match: {match[0]} (similarity: {match[1]:.2%})")
        else:
            print("No match found")
