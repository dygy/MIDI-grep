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

# Genre definitions: an ENSEMBLE of text prompts per genre (their normalised embeddings are averaged into one
# class prototype, the CLIP-style prompt ensemble). There is NO tempo prior and NO penalty term here: a bonus
# multiplier from a BPM range let the tempo overrule the audio (a 1.8x on 'brazilian_funk' turned a 0.65 cosine
# into the 1.10 "confidence" of one reference track), and no labelled data supports any particular range
# (Brazilian funk is played from ~125 to ~150 BPM). `bpm_range` is used ONLY by the no-CLAP fallback, where it is
# the single piece of evidence there is - and that result is always flagged uncertain.
# Known limit (measured, see context/spec): these prompts name vocals/Portuguese for funk, so CLAP partly answers
# "sung/rapped dance music"; a track like that is told apart from funk only if the prompts separate it, and the
# margin rule below reports when they do not.
GENRE_DEFINITIONS = {
    'brazilian_funk': {
        'descriptions': [
            "Brazilian funk carioca tamborzão drums MC vocals Portuguese singing",
            "Funk carioca Brazilian dance music chopped female vocals",
            "Brazilian baile funk 808 bass rhythm vocal samples Portuguese",
            "Tamborzão beat Brazilian funk female MC vocals dance",
            "Brazilian funk proibidão bass vocal samples Latin rhythm",
        ],
        'bpm_range': (125, 150),
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
    },
    'trance': {
        'descriptions': [
            "Trance electronic euphoric synthesizer buildup breakdown",
            "Progressive trance uplifting melody arpeggios",
            "Psytrance psychedelic fast electronic beats",
            "Uplifting trance emotional electronic melody",
        ],
        'bpm_range': (130, 150),
    },
    'house': {
        'descriptions': [
            "House electronic four on the floor beat dance",
            "Deep house smooth groovy bass synthesizer",
            "Tech house minimal electronic beats",
            "Progressive house melodic electronic dance",
        ],
        'bpm_range': (120, 130),
    },
    'lofi': {
        'descriptions': [
            "Lo-fi hip hop chill relaxing beats vinyl crackle",
            "Lofi chillhop jazzy relaxing music",
            "Lo-fi beats study music calm",
        ],
        'bpm_range': (70, 95),
    },
    'jazz': {
        'descriptions': [
            "Jazz improvisation saxophone piano acoustic",
            "Smooth jazz relaxing instrumental",
            "Jazz fusion electronic elements",
        ],
        'bpm_range': (60, 140),
    },
}

# --- evidence handling (nothing below is tuned to a track) ---------------------------------------------------
WINDOW_S = 10.0          # CLAP's own input length (the processor truncates/pads to 10 s)
MAX_WINDOWS = 24         # evenly spaced non-overlapping windows over the WHOLE track (cost cap: ~0.3 s each on CPU)
SILENCE_DB = -40.0       # a window this far under the track's RMS carries no genre evidence
Z_MIN = 2.0              # top-2 margin must exceed this many standard errors of its own per-window noise (~95 %)
MIN_VOTE = 0.5           # ... and the winner must be the best genre in at least this share of the windows
LOGIT_SCALE = 100.0      # CLAP's trained logit scale (exp(4.6052)); only used for the reported probabilities


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
        # safetensors avoids torch.load entirely: transformers >= 4.5x refuses torch.load on
        # torch < 2.6 (CVE-2025-32434) and this venv pins torch 2.5 for acids-rave. Without this
        # the detector silently failed and the pipeline fell back to BPM heuristics (2026-10-10).
        src = os.environ.get("MIDIGREP_CLAP_PATH") or "laion/clap-htsat-unfused"   # local dir = offline use
        try:
            model = ClapModel.from_pretrained(src, use_safetensors=True)
        except (OSError, ValueError):
            model = ClapModel.from_pretrained(src)
        processor = ClapProcessor.from_pretrained(src)
        return (model, processor), 'transformers'
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        # Review #6: the plain-load fallback raises the torch.load ValueError on torch < 2.6; any
        # load failure must reach the heuristic fallback, not crash the detector.
        print(f"Warning: CLAP unavailable ({type(exc).__name__}: {str(exc)[:120]})", file=sys.stderr)

    return None, None


def split_windows(y, sr, window_s=WINDOW_S, max_windows=MAX_WINDOWS, silence_db=SILENCE_DB):
    """Non-overlapping ``window_s`` windows spread evenly over the WHOLE track (not one excerpt), silent ones
    dropped. A track shorter than one window gives that one (padded by the CLAP processor)."""
    seg = int(window_s * sr)
    if len(y) <= seg:
        return [y]
    n = len(y) // seg
    starts = [i * seg for i in range(n)]
    if n > max_windows:
        starts = [starts[int(round(k))] for k in np.linspace(0, n - 1, max_windows)]
    rms_all = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) or 1.0
    floor = rms_all * 10 ** (silence_db / 20.0)
    wins = [y[a:a + seg] for a in starts]
    kept = [w for w in wins if float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) >= floor]
    return kept or wins


def _pooled(outputs):
    if hasattr(outputs, 'pooler_output'):
        return outputs.pooler_output
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state.mean(dim=1)
    return outputs


def embed_windows(model, model_type, windows, sr=48000):
    """One CLAP audio embedding per window -> array (n_windows, dim)."""
    if model_type == 'laion_clap':
        return np.asarray(model.get_audio_embedding_from_data(x=list(windows), use_tensor=False))
    import torch
    m, processor = model
    out = []
    for w in windows:
        inputs = processor(audio=w, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out.append(_pooled(m.get_audio_features(**inputs)).detach().numpy()[0])
    return np.asarray(out)


def get_text_embeddings_laion(model, texts):
    """Get text embeddings using laion_clap."""
    return model.get_text_embedding(texts, use_tensor=False)


def get_text_embeddings_transformers(model_tuple, texts):
    """Get text embeddings using transformers CLAP."""
    model, processor = model_tuple
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    return _pooled(model.get_text_features(**inputs)).detach().numpy()


def text_prompt_embeddings(model, model_type, definitions=None):
    """{genre: (n_prompts, dim) array of unit-norm prompt embeddings}."""
    definitions = GENRE_DEFINITIONS if definitions is None else definitions
    fn = get_text_embeddings_laion if model_type == 'laion_clap' else get_text_embeddings_transformers
    out = {}
    for genre, info in definitions.items():
        e = np.asarray(fn(model, info['descriptions']), dtype=np.float64)
        out[genre] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rank_genres(window_embeds, prompt_embeds, definitions=None):
    """Score a track from per-window audio embeddings and per-genre prompt embeddings. Pure numpy (no model).

    Each genre is the unit-norm mean of its prompt embeddings (prompt ensemble); a window's evidence for a genre
    is the cosine to that prototype; the track's evidence is the mean over ALL windows. The decision is then
    tested against the measured noise of exactly that quantity:

    * ``margin`` = mean over windows of (top1 - top2) cosine, ``margin_se`` = its standard error across windows,
      ``margin_z`` = margin / margin_se (the windows are the replications of the measurement);
    * ``vote_share`` = share of windows in which the winner is the best genre;
    * ``uncertain`` = margin_z < Z_MIN, or vote_share < MIN_VOTE, or fewer than 3 windows (no noise estimate);
    * ``candidates`` = the winner plus every genre whose paired margin to the winner is under Z_MIN standard errors.

    Returns the dict written to genre.json minus the file-level fields.
    """
    definitions = GENRE_DEFINITIONS if definitions is None else definitions
    E = np.asarray(window_embeds, dtype=np.float64)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    genres = list(prompt_embeds)
    protos = []
    for g in genres:
        c = prompt_embeds[g].mean(axis=0)
        protos.append(c / np.linalg.norm(c))
    S = E @ np.stack(protos).T                       # (windows, genres) cosine
    n = S.shape[0]
    logits = LOGIT_SCALE * S
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    mean_cos, mean_prob = S.mean(axis=0), prob.mean(axis=0)
    order = list(np.argsort(-mean_cos))
    t1, t2 = order[0], order[1]

    def paired(a, b):
        d = S[:, a] - S[:, b]
        if n < 3:
            return float(d.mean()), None, None
        se = float(d.std(ddof=1) / np.sqrt(n))
        return float(d.mean()), se, (float(d.mean() / se) if se > 0 else float('inf'))

    margin, se, z = paired(t1, t2)
    vote = float((S.argmax(axis=1) == t1).mean())
    reasons = []
    if z is None:
        reasons.append(f"only {n} analysis window(s): no estimate of the score noise")
    elif z < Z_MIN:
        reasons.append(f"top-2 margin {margin:.3f} is {z:.1f} standard errors (<{Z_MIN:g}) over {n} windows")
    if vote < MIN_VOTE:
        reasons.append(f"winner is best in only {vote:.0%} of the windows (<{MIN_VOTE:.0%})")
    candidates = [genres[t1]]
    for j in order[1:]:
        _, _, zj = paired(t1, j)
        if zj is None or zj < Z_MIN:
            candidates.append(genres[j])
    rankings = []
    for j in order:
        sims = prompt_embeds[genres[j]] @ E.T        # (prompts, windows)
        best = int(np.argmax(sims.mean(axis=1)))
        rankings.append({
            'genre': genres[j],
            'score': float(mean_cos[j]),             # mean cosine to the prompt-ensemble prototype
            'probability': float(mean_prob[j]),      # window-averaged softmax at CLAP's logit scale (reporting only)
            'similarity': float(sims[best].mean()),  # mean cosine to the single best prompt
            'description': definitions[genres[j]]['descriptions'][best] if genres[j] in definitions else '',
        })
    return {
        'detected_genre': genres[t1],
        'confidence': float(mean_prob[t1]),
        'uncertain': bool(reasons),
        'uncertain_reasons': reasons,
        'candidates': candidates,
        'margin': margin, 'margin_se': se, 'margin_z': z,
        'vote_share': vote, 'n_windows': int(n),
        'rankings': rankings,
    }


def expected_model_type():
    """The model type load_clap_model() will return, without loading a checkpoint ('fallback' = no CLAP at all).
    The import is tried for real: a package that is installed but cannot import (a missing torchvision) is no CLAP."""
    try:
        import laion_clap  # noqa: F401
        return 'laion_clap'
    except Exception:  # noqa: BLE001 - any import failure means load_clap_model() falls through to transformers
        pass
    import importlib.util
    return 'transformers' if importlib.util.find_spec('transformers') is not None else 'fallback'


def embedding_model_id(model_type):
    """Identifies the audio embedding space (model + checkpoint + windowing). Embeddings from different ids are never
    compared or mixed (genre_knn.py caches them per id)."""
    src = "default" if model_type == 'laion_clap' else (os.environ.get("MIDIGREP_CLAP_PATH") or "laion/clap-htsat-unfused")
    return f"{model_type}:{os.path.basename(src.rstrip('/'))}|win{WINDOW_S:g}s-max{MAX_WINDOWS}-sil{SILENCE_DB:g}dB"


def mean_embedding(window_embeds):
    """The track embedding: the unit-norm mean of the unit-norm window embeddings (all windows weigh the same)."""
    E = np.asarray(window_embeds, dtype=np.float64)
    m = (E / np.linalg.norm(E, axis=1, keepdims=True)).mean(axis=0)
    return m / np.linalg.norm(m)


def track_embedding(audio_path, model, model_type):
    """(embedding, embedding_model_id) of one track, windowed exactly as the genre detector windows it."""
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    return mean_embedding(embed_windows(model, model_type, split_windows(y, sr), sr)), embedding_model_id(model_type)


def detect_genre_clap(audio_path, model, model_type, bpm=None):
    """CLAP genre detection over every window of the track. ``bpm`` is accepted for the caller's convenience and
    ignored: it is not evidence about the audio. The result also carries the track's mean audio embedding
    (``audio_embedding``) - the input of the kNN against the owner-labelled tracks (genre_knn.py)."""
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    windows = split_windows(y, sr)
    W = embed_windows(model, model_type, windows, sr)
    result = rank_genres(W, text_prompt_embeddings(model, model_type))
    result['window_s'] = WINDOW_S
    result['audio_embedding'] = [float(v) for v in mean_embedding(W)]
    result['embedding_model'] = embedding_model_id(model_type)
    return result


def detect_genre_fallback(audio_path, bpm=None):
    """Fallback genre detection using audio features (no deep learning): BPM range + centroid heuristics.
    It is not audio evidence of the genre, so the result is ALWAYS uncertain."""
    print("Warning: CLAP not available, using fallback feature-based detection", file=sys.stderr)

    y, sr = librosa.load(audio_path, sr=22050)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    results = {}
    for genre, info in GENRE_DEFINITIONS.items():
        score = 0.5
        if bpm is not None:
            lo, hi = info['bpm_range']
            if lo <= bpm <= hi:
                score += 0.3
        if genre == 'brazilian_funk' and 1000 < spectral_centroid < 3000:
            score += 0.2
        elif genre == 'brazilian_phonk' and 800 < spectral_centroid < 4000:
            score += 0.15
        elif genre == 'retro_wave' and spectral_centroid > 2000:
            score += 0.2
        elif genre == 'lofi' and spectral_centroid < 2000:
            score += 0.2
        results[genre] = score
    order = sorted(results, key=results.get, reverse=True)
    return {
        'detected_genre': order[0], 'confidence': float(results[order[0]]),
        'uncertain': True,
        'uncertain_reasons': ['CLAP unavailable: BPM/centroid heuristic only, no audio-text evidence'],
        'candidates': order[:3], 'margin': float(results[order[0]] - results[order[1]]),
        'margin_se': None, 'margin_z': None, 'vote_share': None, 'n_windows': 0,
        'rankings': [{'genre': g, 'score': float(results[g]), 'similarity': float(results[g]),
                      'probability': None, 'description': GENRE_DEFINITIONS[g]['descriptions'][0],
                      'fallback': True} for g in order],
    }


def main():
    parser = argparse.ArgumentParser(description='Deep learning genre detection using CLAP')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--bpm', type=float, help='Recorded in the output only; it does not influence the score')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--top', type=int, default=3, help='Number of top results to show')

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {args.audio_path}", file=sys.stderr)
    model, model_type = load_clap_model()
    if model is not None:
        print(f"Using CLAP model ({model_type})", file=sys.stderr)
        res = detect_genre_clap(args.audio_path, model, model_type, args.bpm)
    else:
        res = detect_genre_fallback(args.audio_path, args.bpm)

    output = {
        'audio_path': args.audio_path,
        'bpm': args.bpm,
        'model_type': model_type or 'fallback',   # laion_clap | transformers | fallback (BPM/spectral heuristic)
        'scoring': 'window-mean-cosine/prompt-ensemble/no-bpm-prior' if model is not None else 'heuristic',
        **res,
    }

    print("\n" + "=" * 60)
    print("DEEP LEARNING GENRE DETECTION")
    print("=" * 60)
    flag = "  [UNCERTAIN]" if output['uncertain'] else ""
    print(f"\nDetected Genre: {output['detected_genre'].upper()}{flag}")
    print(f"Confidence (window-mean probability): {output['confidence']:.3f}")
    if output['margin_z'] is not None:
        print(f"Top-2 margin {output['margin']:.3f} = {output['margin_z']:.1f} s.e. over {output['n_windows']} windows, "
              f"winner best in {output['vote_share']:.0%}")
    for r in output['uncertain_reasons']:
        print(f"  uncertain: {r}")
    print(f"\nTop {args.top} matches:")
    for i, r in enumerate(output['rankings'][:args.top], 1):
        print(f"  {i}. {r['genre']}: {r['score']:.3f}")
        print(f"     Best match: \"{r['description'][:50]}...\"")
    print("=" * 60)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")

    print("\n--- JSON OUTPUT ---")
    print(json.dumps({k: v for k, v in output.items() if k != 'audio_embedding'}, indent=2))


if __name__ == '__main__':
    main()
