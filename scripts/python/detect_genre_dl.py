#!/usr/bin/env python3
"""
Deep Learning Genre Detection using CLAP (Contrastive Language-Audio Pretraining).

Zero-shot audio classification - no training required.
Compares audio against text descriptions of genres.
"""

import argparse
import json
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import librosa

# Genre definitions with multiple text descriptions for better matching
GENRE_DEFINITIONS = {
    'brazilian_funk': {
        'descriptions': [
            "Brazilian funk carioca tamborzão drums MC vocals Portuguese singing",
            "Funk carioca Brazilian dance music chopped female vocals",
            "Brazilian baile funk 808 bass rhythm vocal samples Portuguese",
            "Tamborzão beat Brazilian funk female MC vocals dance",
            "Brazilian funk proibidão bass vocal samples Latin rhythm",
        ],
        'bpm_range': (130, 145),
        'bpm_weight': 1.8,  # Strong bonus for matching BPM
        'negative_descriptions': [
            "synthesizer arpeggios 80s retro",
            "Russian electronic hardbass",
        ],
    },
    'brazilian_phonk': {
        'descriptions': [
            "Brazilian phonk dark aggressive bass cowbell distorted",
            "Phonk drift dark trap distorted 808 bass aggressive",
            "Brazilian phonk aggressive bass cowbell dark electronic",
            "Dark phonk Memphis rap influence heavy bass",
            "Kordhell aggressive phonk electronic bass distortion",
        ],
        'bpm_range': (75, 180),
        'bpm_weight': 1.0,
        'negative_descriptions': [
            "synthesizer arpeggios 80s",
            "clean melodic synth pads",
        ],
    },
    'retro_wave': {
        'descriptions': [
            "Synthwave 80s synthesizer arpeggios electronic retro nostalgic",
            "Retrowave electronic 80s synth pads arpeggiated melody",
            "80s style synthwave clean synthesizers arpeggios",
            "Outrun synthwave neon 80s electronic arpeggios pads",
            "Darksynth retrowave 80s electronic synthesizer melody",
            "Eastern European synthwave electronic 80s style",
            "Italo disco inspired synthwave arpeggios 80s",
        ],
        'bpm_range': (100, 170),
        'bpm_weight': 1.3,
        'negative_descriptions': [
            "Brazilian vocals chopped",
            "tamborzão drums MC",
        ],
    },
    'trance': {
        'descriptions': [
            "Trance electronic euphoric synthesizer buildup breakdown",
            "Progressive trance uplifting melody arpeggios",
            "Psytrance psychedelic fast electronic beats",
            "Uplifting trance emotional electronic melody",
        ],
        'bpm_range': (130, 150),
        'bpm_weight': 1.0,
    },
    'house': {
        'descriptions': [
            "House electronic four on the floor beat dance",
            "Deep house smooth groovy bass synthesizer",
            "Tech house minimal electronic beats",
            "Progressive house melodic electronic dance",
        ],
        'bpm_range': (120, 130),
        'bpm_weight': 1.0,
    },
    'lofi': {
        'descriptions': [
            "Lo-fi hip hop chill relaxing beats vinyl crackle",
            "Lofi chillhop jazzy relaxing music",
            "Lo-fi beats study music calm",
        ],
        'bpm_range': (70, 95),
        'bpm_weight': 1.2,
    },
    'jazz': {
        'descriptions': [
            "Jazz improvisation saxophone piano acoustic",
            "Smooth jazz relaxing instrumental",
            "Jazz fusion electronic elements",
        ],
        'bpm_range': (60, 140),
        'bpm_weight': 0.8,
    },
}


def load_clap_model():
    """Load CLAP model for audio-text similarity."""
    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # Load default checkpoint
        return model, 'laion_clap'
    except ImportError:
        pass

    # Fallback to transformers CLAP
    try:
        from transformers import ClapModel, ClapProcessor
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        return (model, processor), 'transformers'
    except ImportError:
        pass

    return None, None


def get_audio_embedding_laion(model, audio_path):
    """Get audio embedding using laion_clap."""
    # Load audio at 48kHz (CLAP requirement)
    y, sr = librosa.load(audio_path, sr=48000, mono=True)

    # CLAP expects audio in specific format
    # Limit to 10 seconds for efficiency
    max_samples = 48000 * 10
    if len(y) > max_samples:
        # Take middle section
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]

    # Get embedding
    audio_embed = model.get_audio_embedding_from_data(x=[y], use_tensor=False)
    return audio_embed[0]


def get_text_embeddings_laion(model, texts):
    """Get text embeddings using laion_clap."""
    text_embed = model.get_text_embedding(texts, use_tensor=False)
    return text_embed


def get_audio_embedding_transformers(model_tuple, audio_path):
    """Get audio embedding using transformers CLAP."""
    model, processor = model_tuple

    # Load audio
    y, sr = librosa.load(audio_path, sr=48000, mono=True)

    # Limit length
    max_samples = 48000 * 10
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]

    # Process (use 'audio' instead of deprecated 'audios')
    inputs = processor(audio=y, sampling_rate=48000, return_tensors="pt")
    outputs = model.get_audio_features(**inputs)

    # Handle different return types
    if hasattr(outputs, 'pooler_output'):
        audio_embed = outputs.pooler_output
    elif hasattr(outputs, 'last_hidden_state'):
        audio_embed = outputs.last_hidden_state.mean(dim=1)
    else:
        # Direct tensor output
        audio_embed = outputs

    return audio_embed.detach().numpy()[0]


def get_text_embeddings_transformers(model_tuple, texts):
    """Get text embeddings using transformers CLAP."""
    model, processor = model_tuple
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)

    # Handle different return types
    if hasattr(outputs, 'pooler_output'):
        text_embed = outputs.pooler_output
    elif hasattr(outputs, 'last_hidden_state'):
        text_embed = outputs.last_hidden_state.mean(dim=1)
    else:
        text_embed = outputs

    return text_embed.detach().numpy()


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def detect_genre_clap(audio_path, model, model_type, bpm=None):
    """Detect genre using CLAP model."""
    # Get audio embedding
    if model_type == 'laion_clap':
        audio_embed = get_audio_embedding_laion(model, audio_path)
    else:
        audio_embed = get_audio_embedding_transformers(model, audio_path)

    results = {}

    for genre, info in GENRE_DEFINITIONS.items():
        # Get text embeddings for all descriptions
        if model_type == 'laion_clap':
            text_embeds = get_text_embeddings_laion(model, info['descriptions'])
        else:
            text_embeds = get_text_embeddings_transformers(model, info['descriptions'])

        # Calculate similarity with each description
        similarities = []
        for text_embed in text_embeds:
            sim = cosine_similarity(audio_embed, text_embed)
            similarities.append(sim)

        # Use max similarity (best matching description)
        max_sim = max(similarities)
        avg_sim = np.mean(similarities)

        # Combined score (weighted towards max)
        score = 0.7 * max_sim + 0.3 * avg_sim

        # Apply negative descriptions penalty
        negative_penalty = 0.0
        if 'negative_descriptions' in info:
            if model_type == 'laion_clap':
                neg_embeds = get_text_embeddings_laion(model, info['negative_descriptions'])
            else:
                neg_embeds = get_text_embeddings_transformers(model, info['negative_descriptions'])

            neg_sims = []
            for neg_embed in neg_embeds:
                neg_sim = cosine_similarity(audio_embed, neg_embed)
                neg_sims.append(neg_sim)

            # If audio is similar to negative descriptions, penalize
            max_neg = max(neg_sims) if neg_sims else 0
            if max_neg > 0.3:  # Threshold for penalty
                negative_penalty = (max_neg - 0.3) * 0.5

        score -= negative_penalty

        # BPM bonus if provided
        if bpm is not None:
            bpm_low, bpm_high = info['bpm_range']
            if bpm_low <= bpm <= bpm_high:
                score *= info['bpm_weight']
            # Also check half-time
            elif bpm_low <= bpm * 2 <= bpm_high:
                score *= (info['bpm_weight'] * 0.8)
            # Penalty for being outside range
            elif abs(bpm - (bpm_low + bpm_high) / 2) > 30:
                score *= 0.8

        results[genre] = {
            'score': float(score),
            'max_similarity': float(max_sim),
            'avg_similarity': float(avg_sim),
            'negative_penalty': float(negative_penalty),
            'best_description': info['descriptions'][similarities.index(max_sim)],
        }

    # Sort by score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)

    return sorted_results


def detect_genre_fallback(audio_path, bpm=None):
    """Fallback genre detection using audio features (no deep learning)."""
    print("Warning: CLAP not available, using fallback feature-based detection", file=sys.stderr)

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Extract features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Simple heuristic scoring
    results = {}

    for genre, info in GENRE_DEFINITIONS.items():
        score = 0.5  # Base score

        # BPM matching
        if bpm is not None:
            bpm_low, bpm_high = info['bpm_range']
            if bpm_low <= bpm <= bpm_high:
                score += 0.3 * info['bpm_weight']

        # Spectral characteristics (rough heuristics)
        if genre == 'brazilian_funk':
            # Mid-heavy, moderate brightness
            if 1000 < spectral_centroid < 3000:
                score += 0.2
        elif genre == 'brazilian_phonk':
            # Can be darker or brighter
            if 800 < spectral_centroid < 4000:
                score += 0.15
        elif genre == 'retro_wave':
            # Typically brighter (synths)
            if spectral_centroid > 2000:
                score += 0.2
        elif genre == 'lofi':
            # Darker, warmer
            if spectral_centroid < 2000:
                score += 0.2

        results[genre] = {
            'score': float(score),
            'max_similarity': float(score),
            'avg_similarity': float(score),
            'best_description': info['descriptions'][0],
            'fallback': True,
        }

    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    return sorted_results


def main():
    parser = argparse.ArgumentParser(description='Deep learning genre detection using CLAP')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--bpm', type=float, help='BPM for additional scoring')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--top', type=int, default=3, help='Number of top results to show')

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {args.audio_path}", file=sys.stderr)

    # Try to load CLAP model
    model, model_type = load_clap_model()

    if model is not None:
        print(f"Using CLAP model ({model_type})", file=sys.stderr)
        results = detect_genre_clap(args.audio_path, model, model_type, args.bpm)
    else:
        results = detect_genre_fallback(args.audio_path, args.bpm)

    # Prepare output
    output = {
        'audio_path': args.audio_path,
        'bpm': args.bpm,
        'detected_genre': results[0][0],
        'confidence': results[0][1]['score'],
        'rankings': [
            {
                'genre': genre,
                'score': info['score'],
                'similarity': info['max_similarity'],
                'description': info['best_description'],
            }
            for genre, info in results
        ],
    }

    # Print results
    print("\n" + "=" * 60)
    print("DEEP LEARNING GENRE DETECTION")
    print("=" * 60)
    print(f"\nDetected Genre: {output['detected_genre'].upper()}")
    print(f"Confidence: {output['confidence']:.3f}")
    print(f"\nTop {args.top} matches:")
    for i, (genre, info) in enumerate(results[:args.top], 1):
        print(f"  {i}. {genre}: {info['score']:.3f}")
        print(f"     Best match: \"{info['best_description'][:50]}...\"")
    print("=" * 60)

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")

    # Also print JSON to stdout for parsing
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
