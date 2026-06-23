"""Genre-specific synthesis profiles.

The base synth config is 100% audio-analysis driven (see ``analyze_synth_params.py``).
Audio analysis captures the *spectrum* of a track but not its *genre character* —
e.g. Brazilian funk wants a heavier 808 sub and punchier transients than the raw
spectrum alone implies, while lo-fi wants the highs rolled off.

These profiles apply a thin, transparent layer of genre nudges AFTER the
analysis-derived config is built. They are intentionally conservative
(multipliers near 1.0, gentle absolute overrides) so the audio analysis still
drives the baseline and the genre only shapes character.

Profile schema (all keys optional):

    {
        "voices": {
            "<bass|mid|high|drums>": {
                "gain_mult":        float,   # multiply the analysis-derived gain
                "sub_octave_gain":  float,   # absolute override (bass only)
                "lpf":              float,   # absolute override (Hz)
                "transient_boost":  float,   # absolute override (drums only)
            }
        },
        "master": {
            "hpf":              float,   # absolute override (Hz)
            "high_shelf_boost": float,   # absolute override (dB)
        },
        "sidechain": {
            "depth": float,   # 0.0-1.0, consumed by codegen prompt (kick ducks bass)
        },
    }

The fields the Node renderer reads (``voices.*.gain``, ``sub_octave_gain``,
``lpf``, ``transient_boost``, ``master.hpf``, ``master.high_shelf_boost``) are
exactly what these profiles touch, so genre tuning takes effect even through the
already-compiled ``dist/render-strudel-node.js`` — no source changes needed.
"""

from copy import deepcopy
from typing import Any, Dict

# Sensible clamps so a profile can never produce a runaway gain.
_GAIN_MIN, _GAIN_MAX = 0.02, 2.0


GENRE_PROFILES: Dict[str, Dict[str, Any]] = {
    # Heavy 808 sub, punchy kick, minimal mids. Low master HPF preserves sub.
    "brazilian_funk": {
        "voices": {
            "bass": {"gain_mult": 1.25, "sub_octave_gain": 0.60},
            "drums": {"gain_mult": 1.15, "transient_boost": 0.50},
        },
        "master": {"hpf": 20},
        "sidechain": {"depth": 0.7},
    },
    # Darker, sub-heavy, rolled-off highs (Memphis).
    "brazilian_phonk": {
        "voices": {
            "bass": {"gain_mult": 1.20, "sub_octave_gain": 0.65},
            "high": {"gain_mult": 0.85},
            "drums": {"gain_mult": 1.10, "transient_boost": 0.45},
        },
        "master": {"hpf": 20},
        "sidechain": {"depth": 0.6},
    },
    "phonk": {
        "voices": {
            "bass": {"gain_mult": 1.20, "sub_octave_gain": 0.65},
            "high": {"gain_mult": 0.85},
        },
        "master": {"hpf": 20},
        "sidechain": {"depth": 0.6},
    },
    # Brass/piano mids, swing, light sub, bright top.
    "electro_swing": {
        "voices": {
            "bass": {"gain_mult": 0.90, "sub_octave_gain": 0.25},
            "high": {"gain_mult": 1.10},
        },
        "master": {"hpf": 40, "high_shelf_boost": 2.0},
        "sidechain": {"depth": 0.0},
    },
    # 4-on-floor kick, sidechain bass, bright leads.
    "house": {
        "voices": {
            "bass": {"gain_mult": 1.0, "sub_octave_gain": 0.40},
            "drums": {"gain_mult": 1.10, "transient_boost": 0.40},
        },
        "master": {"hpf": 30},
        "sidechain": {"depth": 0.7},
    },
    "techno": {
        "voices": {
            "bass": {"gain_mult": 1.0, "sub_octave_gain": 0.40},
            "drums": {"gain_mult": 1.15, "transient_boost": 0.45},
        },
        "master": {"hpf": 30},
        "sidechain": {"depth": 0.75},
    },
    # Heavy sub, fast breaks.
    "dnb": {
        "voices": {
            "bass": {"gain_mult": 1.20, "sub_octave_gain": 0.65},
            "drums": {"gain_mult": 1.10, "transient_boost": 0.50},
        },
        "master": {"hpf": 20},
        "sidechain": {"depth": 0.5},
    },
    # Uplifting, supersaw-heavy, bright.
    "trance": {
        "voices": {
            "bass": {"gain_mult": 1.0, "sub_octave_gain": 0.40},
            "high": {"gain_mult": 1.10},
        },
        "master": {"hpf": 30},
        "sidechain": {"depth": 0.6},
    },
    # Warm analog bass, retro 80s.
    "synthwave": {
        "voices": {
            "bass": {"gain_mult": 1.05, "sub_octave_gain": 0.45},
        },
        "master": {"hpf": 30},
        "sidechain": {"depth": 0.3},
    },
    "retro_wave": {
        "voices": {
            "bass": {"gain_mult": 1.05, "sub_octave_gain": 0.45},
        },
        "master": {"hpf": 30},
        "sidechain": {"depth": 0.3},
    },
    # Warm, dusty, rolled-off highs, tape character.
    "lofi": {
        "voices": {
            "bass": {"gain_mult": 1.0, "sub_octave_gain": 0.30},
            "high": {"gain_mult": 0.70},
        },
        "master": {"hpf": 40, "high_shelf_boost": 0.0},
        "sidechain": {"depth": 0.0},
    },
    # Acoustic, natural — almost no synthetic sub, no transient hyping.
    "jazz": {
        "voices": {
            "bass": {"sub_octave_gain": 0.15},
            "drums": {"transient_boost": 0.10},
        },
        "master": {"hpf": 40},
        "sidechain": {"depth": 0.0},
    },
    "classical": {
        "voices": {
            "bass": {"sub_octave_gain": 0.10},
            "drums": {"transient_boost": 0.05},
        },
        "master": {"hpf": 40},
        "sidechain": {"depth": 0.0},
    },
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_sidechain_depth(genre: str) -> float:
    """Return the kick-ducks-bass depth (0.0-1.0) for a genre, 0.0 if unknown/none."""
    profile = GENRE_PROFILES.get((genre or "").strip().lower())
    if not profile:
        return 0.0
    return float(profile.get("sidechain", {}).get("depth", 0.0))


def sidechain_instruction(genre: str) -> str:
    """Return a copy-pasteable kick-ducks-bass instruction for genres that want it.

    Empty string when the genre has no sidechain character (jazz, lofi, swing).
    Uses the verified Strudel idiom: the bass sits on a named orbit, the kick
    fires .duckorbit() at that orbit. Ducking modulates the orbit-bus gain, so
    it works through arrange() and regardless of per-event .gain().
    """
    depth = get_sidechain_depth(genre)
    if depth <= 0:
        return ""
    return (
        f"\n\n## SIDECHAIN (kick ducks bass — essential for {genre} punch):\n"
        f"- Put the Bass voice on its own orbit: append `.orbit(2)` to the bass pattern's effect chain.\n"
        f"- Make the kick duck it: append `.duckorbit(2).duckdepth({depth:.2f}).duckattack(0.15)` "
        f'to the Drums voice (the s("bd...") layer). Use the SAME orbit number (2) on both.\n'
        f"- duckorbit goes on the KICK, .orbit(2) goes on the BASS — never swap them, or it silently won't duck."
    )


def apply_genre_profile(config: Dict[str, Any], genre: str) -> Dict[str, Any]:
    """Apply genre-specific nudges to an analysis-derived synth config.

    Returns a NEW config dict (the input is not mutated). Unknown or empty
    genres are a no-op. Defensive against missing voices/master keys so a
    partially-populated config never raises.
    """
    key = (genre or "").strip().lower()
    profile = GENRE_PROFILES.get(key)
    if not profile:
        return config

    cfg = deepcopy(config)
    cfg["genre_profile"] = key

    voices = cfg.get("voices")
    if isinstance(voices, dict):
        for voice_name, tweaks in profile.get("voices", {}).items():
            voice = voices.get(voice_name)
            if not isinstance(voice, dict):
                continue
            for field, value in tweaks.items():
                if field == "gain_mult":
                    base = voice.get("gain")
                    if isinstance(base, (int, float)):
                        voice["gain"] = _clamp(base * value, _GAIN_MIN, _GAIN_MAX)
                else:
                    # Absolute overrides: sub_octave_gain, lpf, transient_boost.
                    voice[field] = value

    master = cfg.get("master")
    if isinstance(master, dict):
        for field, value in profile.get("master", {}).items():
            master[field] = value

    sidechain_depth = profile.get("sidechain", {}).get("depth", 0.0)
    if sidechain_depth:
        cfg.setdefault("sidechain", {})["depth"] = float(sidechain_depth)

    return cfg


if __name__ == "__main__":
    # Tiny self-check: apply each profile to a representative base config and
    # confirm gains stay in range and overrides land.
    import json

    base = {
        "voices": {
            "bass": {"gain": 0.10, "lpf": 400, "hpf": 40, "sub_octave_gain": 0.30},
            "mid": {"gain": 0.60, "lpf": 6000, "hpf": 200},
            "high": {"gain": 0.80, "lpf": 12000, "hpf": 400},
            "drums": {"gain": 0.70, "transient_boost": 0.0019},
        },
        "master": {"gain": 1.5, "hpf": 120, "limiter": True, "high_shelf_boost": 0},
    }
    for g in list(GENRE_PROFILES) + ["unknown_genre", ""]:
        out = apply_genre_profile(base, g)
        bass_gain = out["voices"]["bass"]["gain"]
        assert _GAIN_MIN <= bass_gain <= _GAIN_MAX, (g, bass_gain)
        print(f"{g or '(empty)':18s} sidechain={get_sidechain_depth(g):.2f} "
              f"bass.gain={bass_gain:.3f} master.hpf={out['master']['hpf']}")
    print("\nbrazilian_funk profile applied:")
    print(json.dumps(apply_genre_profile(base, "brazilian_funk"), indent=2))
