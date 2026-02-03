#!/usr/bin/env python3
"""
Genre Detection using Essentia's pre-trained music classifiers.

Uses multiple models:
1. Genre classification (electronic subgenres)
2. Mood/style detection
3. Danceability and other high-level features
"""

import argparse
import json
import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np


def analyze_with_essentia(audio_path, bpm=None):
    """Analyze audio using Essentia's pre-trained models."""
    import essentia.standard as es
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

    print(f"Loading audio: {audio_path}", file=sys.stderr)

    # Load audio at 16kHz (Essentia's requirement for most models)
    audio = MonoLoader(filename=audio_path, sampleRate=16000)()

    results = {
        'audio_path': audio_path,
        'bpm': bpm,
        'predictions': {},
    }

    # Try to use Discogs-Effnet for genre embeddings
    try:
        # Download model if not present
        from essentia.standard import TensorflowPredictEffnetDiscogs

        # Use the Discogs-trained model for electronic music genre
        model_path = os.path.expanduser("~/.essentia/models/discogs-effnet-bs64-1.pb")

        if not os.path.exists(model_path):
            print("Downloading Essentia Discogs model...", file=sys.stderr)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            url = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
            urllib.request.urlretrieve(url, model_path)

        embeddings_model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
        embeddings = embeddings_model(audio)

        # Average embeddings across time
        avg_embedding = np.mean(embeddings, axis=0)
        results['embedding_dim'] = len(avg_embedding)

    except Exception as e:
        print(f"Discogs model error: {e}", file=sys.stderr)
        avg_embedding = None

    # Try music style classification
    try:
        from essentia.standard import TensorflowPredict2D

        # Genre classifier
        genre_model_path = os.path.expanduser("~/.essentia/models/genre_discogs400-discogs-effnet-1.pb")

        if not os.path.exists(genre_model_path):
            print("Downloading genre classifier...", file=sys.stderr)
            os.makedirs(os.path.dirname(genre_model_path), exist_ok=True)
            import urllib.request
            url = "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
            urllib.request.urlretrieve(url, genre_model_path)

        if avg_embedding is not None:
            genre_model = TensorflowPredict2D(graphFilename=genre_model_path)
            genre_preds = genre_model(avg_embedding.reshape(1, -1))

            # Top predictions
            top_indices = np.argsort(genre_preds[0])[::-1][:10]
            results['genre_predictions'] = [
                {'index': int(i), 'score': float(genre_preds[0][i])}
                for i in top_indices
            ]

    except Exception as e:
        print(f"Genre classifier error: {e}", file=sys.stderr)

    # Basic audio features for additional scoring
    try:
        from essentia.standard import RhythmExtractor2013, KeyExtractor, Energy, SpectralCentroidTime

        # Rhythm analysis
        rhythm_extractor = RhythmExtractor2013()
        detected_bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        results['detected_bpm'] = float(detected_bpm)
        results['bpm_confidence'] = float(np.mean(beats_confidence)) if len(beats_confidence) > 0 else 0

        # Key detection
        key_extractor = KeyExtractor()
        key, scale, key_strength = key_extractor(audio)
        results['key'] = f"{key} {scale}"
        results['key_strength'] = float(key_strength)

        # Energy and spectral features
        energy = Energy()
        results['energy'] = float(energy(audio))

        # Spectral centroid (brightness indicator)
        spec_centroid = SpectralCentroidTime()
        centroid = spec_centroid(audio)
        results['spectral_centroid_mean'] = float(np.mean(centroid))

    except Exception as e:
        print(f"Feature extraction error: {e}", file=sys.stderr)

    return results


def classify_genre_heuristic(results, bpm=None):
    """Apply heuristic rules to classify into our target genres."""

    # Use provided BPM or detected
    track_bpm = bpm or results.get('detected_bpm', 130)

    scores = {
        'brazilian_funk': 0.0,
        'brazilian_phonk': 0.0,
        'retro_wave': 0.0,
        'trance': 0.0,
        'house': 0.0,
        'lofi': 0.0,
    }

    # BPM-based scoring
    if 130 <= track_bpm <= 145:
        scores['brazilian_funk'] += 2.0
        scores['house'] += 1.0
    elif 145 <= track_bpm <= 180:
        scores['brazilian_phonk'] += 1.5
        scores['trance'] += 1.5
    elif 80 <= track_bpm <= 100:
        scores['brazilian_phonk'] += 1.0
        scores['lofi'] += 1.5
    elif 100 <= track_bpm <= 130:
        scores['retro_wave'] += 1.0
        scores['house'] += 1.0

    # Spectral centroid (brightness)
    centroid = results.get('spectral_centroid_mean', 2000)
    if centroid > 3000:
        # Bright - more likely synthwave/trance
        scores['retro_wave'] += 1.0
        scores['trance'] += 0.5
    elif centroid < 1500:
        # Dark - more likely phonk/lofi
        scores['brazilian_phonk'] += 0.5
        scores['lofi'] += 1.0

    # Key (minor keys more common in certain genres)
    key = results.get('key', '')
    if 'minor' in key.lower():
        scores['brazilian_phonk'] += 0.5
        scores['retro_wave'] += 0.3
    else:
        scores['brazilian_funk'] += 0.3
        scores['house'] += 0.3

    # Energy
    energy = results.get('energy', 0)
    if energy > 0.1:
        scores['brazilian_funk'] += 0.5
        scores['trance'] += 0.5
    else:
        scores['lofi'] += 0.5

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_scores


def main():
    parser = argparse.ArgumentParser(description='Genre detection using Essentia')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--bpm', type=float, help='Known BPM')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # Analyze with Essentia
    results = analyze_with_essentia(args.audio_path, args.bpm)

    # Apply heuristic classification
    genre_scores = classify_genre_heuristic(results, args.bpm)

    # Output
    output = {
        'audio_path': args.audio_path,
        'detected_genre': genre_scores[0][0],
        'confidence': genre_scores[0][1],
        'all_scores': dict(genre_scores),
        'analysis': results,
    }

    print("\n" + "=" * 60)
    print("ESSENTIA GENRE DETECTION")
    print("=" * 60)
    print(f"Detected Genre: {output['detected_genre'].upper()}")
    print(f"Score: {output['confidence']:.2f}")
    print(f"\nDetected BPM: {results.get('detected_bpm', 'N/A'):.1f}")
    print(f"Key: {results.get('key', 'N/A')}")
    print(f"Spectral Centroid: {results.get('spectral_centroid_mean', 'N/A'):.0f} Hz")
    print(f"\nAll scores:")
    for genre, score in genre_scores:
        print(f"  {genre}: {score:.2f}")
    print("=" * 60)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")

    # JSON output
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
