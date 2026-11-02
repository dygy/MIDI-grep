"""The track's "vision": what the scene should show, measured from the track rather than chosen per track.

Sources: the track's ``metadata.json`` (title, genre, key, BPM), the piece's own header (sections), CLAP zero-shot
similarity of ``instrumental.wav`` to scene and mood descriptions (averaged over one 10 s window per section, so a
quiet intro and a loud drop both count), and the instrumental's spectral centroid (dark vs bright mix).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

CLAP_SR = 48000
CLAP_WINDOW_S = 10.0

# Candidate vocabularies for CLAP zero-shot (the same role as detect_genre_dl.GENRE_DESCRIPTIONS): CLAP ranks them
# against the audio, nothing here is picked per track. Scenes are real places so the picture reminds of something real.
SCENES = [
    "a neon-lit city street at night in the rain",
    "a crowded favela street party at night",
    "a car driving fast on a highway at night",
    "a packed nightclub dance floor with strobe lights",
    "a beach at sunset with palm trees",
    "a rooftop overlooking a city skyline at dusk",
    "an underground parking garage with car headlights",
    "a sound system stacked with speakers at an outdoor party",
    "a tropical rainforest with mist",
    "an empty desert road under a starry sky",
    "a calm ocean shore at dawn",
    "a snowy mountain landscape",
    "a concert stage with lasers and smoke",
    "a subway train rushing through a tunnel",
    "a quiet coffee shop on a rainy afternoon",
    "a sunny summer festival field with a crowd",
    "an industrial warehouse with flickering lights",
    "a cathedral interior with stained glass light",
]
MOODS = [
    "aggressive", "euphoric", "dark", "melancholic", "playful", "dreamy",
    "tense", "romantic", "energetic", "calm", "menacing", "nostalgic",
]

_SECTIONS_RE = re.compile(r"(\w+)\[(\d+)-(\d+)\]")


def piece_sections(strudel_code: str) -> list[dict]:
    """The sections the generator wrote into the piece's DJ-flow header (``// │ sections: intro[0-15] ...``)."""
    for line in strudel_code.splitlines():
        if "sections:" in line:
            return [{"name": n, "bars": [int(a), int(b)]} for n, a, b in _SECTIONS_RE.findall(line)]
    return []


def piece_cps(strudel_code: str) -> float | None:
    m = re.search(r"const cps = ([0-9.]+)", strudel_code) or re.search(r"setcps\(([0-9.]+)\)", strudel_code)
    return float(m.group(1)) if m else None


def _clap_windows(y: np.ndarray, sections: list[dict], cps: float | None) -> list[np.ndarray]:
    """One CLAP_WINDOW_S window per section (starting at its first bar); the middle of the file without sections."""
    n = int(CLAP_WINDOW_S * CLAP_SR)
    starts = []
    if sections and cps:
        starts = [int(s["bars"][0] / cps * CLAP_SR) for s in sections]
    if not starts:
        starts = [max(0, (len(y) - n) // 2)]
    return [y[s:s + n] for s in starts if len(y[s:s + n]) >= CLAP_SR]  # at least 1 s of audio


def clap_tags(audio: Path, sections: list[dict], cps: float | None, top: int = 3) -> dict:
    """Top scene and mood descriptions by CLAP cosine similarity, mean over the section windows."""
    import librosa
    from transformers import ClapModel, ClapProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()

    # use_safetensors: transformers refuses torch.load on torch < 2.6 (see detect_genre_dl.load_clap_model)
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True)
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    y, _ = librosa.load(str(audio), sr=CLAP_SR, mono=True)
    windows = _clap_windows(y, sections, cps)

    def unit(t):
        t = getattr(t, "pooler_output", t)
        v = t.detach().numpy()
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    a = unit(model.get_audio_features(**proc(audio=windows, sampling_rate=CLAP_SR, return_tensors="pt")))
    out = {}
    for key, texts, prefix in (("scenes", SCENES, "music for "), ("moods", MOODS, "music that sounds ")):
        t = unit(model.get_text_features(**proc(text=[prefix + x for x in texts], return_tensors="pt", padding=True)))
        sim = (a @ t.T).mean(axis=0)
        order = np.argsort(-sim)[:top]
        out[key] = [{"text": texts[i], "score": round(float(sim[i]), 4)} for i in order]
    out["windows"] = len(windows)
    return out


def centroid_hz(audio: Path) -> float:
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))


def build_vision(track_dir: Path, strudel_code: str, use_clap: bool = True) -> dict:
    meta_path = track_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    audio = track_dir / "instrumental.wav"
    sections = piece_sections(strudel_code)
    cps = piece_cps(strudel_code)
    key = meta.get("key") or ""
    vision = {
        "title": meta.get("title") or track_dir.name,
        "genre": meta.get("genre") or meta.get("genre_detected") or "",
        "key": key,
        "mode": "minor" if "minor" in key.lower() else ("major" if "major" in key.lower() else ""),
        "bpm": meta.get("bpm"),
        "sections": sections,
        "centroid_hz": round(centroid_hz(audio), 1) if audio.exists() else None,
    }
    if use_clap and audio.exists():
        try:
            vision["clap"] = clap_tags(audio, sections, cps)
        except Exception as exc:  # CLAP is optional evidence; the LLM still gets metadata + centroid
            print(f"Warning: CLAP tags unavailable ({type(exc).__name__}: {str(exc)[:160]})", file=sys.stderr)
    return vision
