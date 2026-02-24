#!/usr/bin/env python3
"""
Intelligent Sound Selector for Strudel

Analyzes original audio's spectral characteristics and selects appropriate
sounds from Strudel's COMPLETE palette. Supports sound alternation patterns.

Contains:
- All 67+ drum machines from tidal-drum-machines
- All 128 General MIDI instruments (gm_* prefix)
- Basic waveforms and ZZFX synths
- Genre-specific palettes
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# COMPLETE STRUDEL SOUND CATALOG
# ============================================================================

# Basic waveforms
WAVEFORMS = ["sine", "sawtooth", "square", "triangle", "supersaw"]

# ZZFX synth waveforms
ZZFX_WAVES = ["z_sine", "z_sawtooth", "z_square", "z_tan", "z_noise"]

# Noise types
NOISE = ["white", "pink", "brown", "crackle"]

# ============================================================================
# ALL 67 DRUM MACHINES (from tidal-drum-machines)
# ============================================================================

DRUM_BANKS = [
    # Roland
    "RolandTR505", "RolandTR606", "RolandTR626", "RolandTR707", "RolandTR727",
    "RolandTR808", "RolandTR909",
    "RolandCompurhythm78", "RolandCompurhythm1000", "RolandCompurhythm8000",
    "RolandD110", "RolandD70", "RolandDDR30", "RolandJD990",
    "RolandMC202", "RolandMC303", "RolandMT32", "RolandR8",
    "RolandS50", "RolandSH09", "RolandSystem100",
    # Linn
    "LinnDrum", "Linn9000", "LinnLM1", "LinnLM2",
    # Akai
    "AkaiLinn", "AkaiMPC60", "AkaiXR10",
    # Boss
    "BossDR55", "BossDR110", "BossDR220", "BossDR550",
    # Korg
    "KorgDDM110", "KorgKPR77", "KorgKR55", "KorgKRZ",
    "KorgM1", "KorgMinipops", "KorgPoly800", "KorgT3",
    # Casio
    "CasioRZ1", "CasioSK1", "CasioVL1",
    # Emu
    "EmuDrumulator", "EmuModular", "EmuSP12",
    # Alesis
    "AlesisHR16", "AlesisSR16",
    # Oberheim
    "OberheimDMX",
    # Sequential Circuits
    "SequentialCircuitsDrumtracks", "SequentialCircuitsTom",
    # Yamaha
    "YamahaRM50",
    # Simmons
    "SimmonsSDS400", "SimmonsSDS5",
    # Others
    "AJKPercusyn", "DoepferMS404", "MFB512", "MPC1000",
    "MoogConcertMateMG1", "RhodesPolaris", "RhythmAce",
    "SakataDPM48", "SergeModular", "SoundmastersR88",
    "UnivoxMicroRhythmer12", "ViscoSpaceDrum", "XdrumLM8953",
]

# ============================================================================
# ALL 128 GENERAL MIDI INSTRUMENTS (gm_* prefix)
# ============================================================================

# Piano (1-8) - NOTE: Strudel uses simplified names, not full GM names
GM_PIANO = [
    "gm_piano", "gm_epiano1", "gm_epiano2",
    "gm_harpsichord", "gm_clavinet"
]

# Chromatic Percussion (9-16)
GM_CHROMATIC_PERCUSSION = [
    "gm_celesta", "gm_glockenspiel", "gm_music_box", "gm_vibraphone",
    "gm_marimba", "gm_xylophone", "gm_tubular_bells", "gm_dulcimer"
]

# Organ (17-24)
GM_ORGAN = [
    "gm_drawbar_organ", "gm_percussive_organ", "gm_rock_organ", "gm_church_organ",
    "gm_reed_organ", "gm_accordion", "gm_harmonica", "gm_bandoneon"
]

# Guitar (25-32)
GM_GUITAR = [
    "gm_acoustic_guitar_nylon", "gm_acoustic_guitar_steel",
    "gm_electric_guitar_jazz", "gm_electric_guitar_clean",
    "gm_electric_guitar_muted", "gm_overdriven_guitar",
    "gm_distortion_guitar", "gm_guitar_harmonics"
]

# Bass (33-40)
GM_BASS = [
    "gm_acoustic_bass", "gm_electric_bass_finger", "gm_electric_bass_pick",
    "gm_fretless_bass", "gm_slap_bass_1", "gm_slap_bass_2",
    "gm_synth_bass_1", "gm_synth_bass_2"
]

# Strings (41-48)
GM_STRINGS = [
    "gm_violin", "gm_viola", "gm_cello", "gm_contrabass",
    "gm_tremolo_strings", "gm_pizzicato_strings", "gm_orchestral_harp", "gm_timpani"
]

# Ensemble (49-56)
GM_ENSEMBLE = [
    "gm_string_ensemble_1", "gm_string_ensemble_2",
    "gm_synth_strings_1", "gm_synth_strings_2",
    "gm_choir_aahs", "gm_voice_oohs", "gm_synth_choir", "gm_orchestra_hit"
]

# Brass (57-64)
GM_BRASS = [
    "gm_trumpet", "gm_trombone", "gm_tuba", "gm_muted_trumpet",
    "gm_french_horn", "gm_brass_section", "gm_synth_brass_1", "gm_synth_brass_2"
]

# Reed (65-72)
GM_REED = [
    "gm_soprano_sax", "gm_alto_sax", "gm_tenor_sax", "gm_baritone_sax",
    "gm_oboe", "gm_english_horn", "gm_bassoon", "gm_clarinet"
]

# Pipe (73-80)
GM_PIPE = [
    "gm_piccolo", "gm_flute", "gm_recorder", "gm_pan_flute",
    "gm_blown_bottle", "gm_shakuhachi", "gm_whistle", "gm_ocarina"
]

# Synth Lead (81-88)
GM_SYNTH_LEAD = [
    "gm_lead_1_square", "gm_lead_2_sawtooth", "gm_lead_3_calliope",
    "gm_lead_4_chiff", "gm_lead_5_charang", "gm_lead_6_voice",
    "gm_lead_7_fifths", "gm_lead_8_bass_lead"
]

# Synth Pad (89-96) - NOTE: Strudel uses names without numbers
GM_SYNTH_PAD = [
    "gm_pad_new_age", "gm_pad_warm", "gm_pad_poly", "gm_pad_choir",
    "gm_pad_bowed", "gm_pad_metallic", "gm_pad_halo", "gm_pad_sweep"
]

# Synth Effects (97-104) - NOTE: Strudel uses names without numbers
GM_SYNTH_FX = [
    "gm_fx_rain", "gm_fx_soundtrack", "gm_fx_crystal", "gm_fx_atmosphere",
    "gm_fx_brightness", "gm_fx_goblins", "gm_fx_echoes", "gm_fx_sci_fi"
]

# Ethnic (105-112)
GM_ETHNIC = [
    "gm_sitar", "gm_banjo", "gm_shamisen", "gm_koto",
    "gm_kalimba", "gm_bagpipe", "gm_fiddle", "gm_shanai"
]

# Percussive (113-119)
GM_PERCUSSIVE = [
    "gm_tinkle_bell", "gm_agogo", "gm_steel_drums", "gm_woodblock",
    "gm_taiko_drum", "gm_melodic_tom", "gm_synth_drum"
]

# Sound Effects (120-128)
GM_SOUND_FX = [
    "gm_reverse_cymbal", "gm_guitar_fret_noise", "gm_breath_noise",
    "gm_seashore", "gm_bird_tweet", "gm_telephone",
    "gm_helicopter", "gm_applause", "gm_gunshot"
]

# ============================================================================
# AGGREGATED LISTS BY CHARACTER
# ============================================================================

# All bass sounds
BASS_SOUNDS = GM_BASS + ["gm_tuba", "gm_contrabass"] + WAVEFORMS[:3]

# All lead sounds (bright, cutting)
LEAD_SOUNDS = (GM_SYNTH_LEAD + GM_BRASS[:4] + GM_REED[:4] +
               ["supersaw", "sawtooth", "square"])

# All pad sounds (sustained, ambient)
PAD_SOUNDS = GM_SYNTH_PAD + GM_ENSEMBLE[:4] + GM_STRINGS[:4]

# All high/bright sounds
HIGH_SOUNDS = (GM_CHROMATIC_PERCUSSION + GM_PIPE[:4] +
               ["gm_music_box", "gm_celesta", "gm_glockenspiel"])

# All piano/keys sounds
PIANO_SOUNDS = GM_PIANO + GM_ORGAN[:4]

# All guitar sounds
GUITAR_SOUNDS = GM_GUITAR

# All ethnic/world sounds
ETHNIC_SOUNDS = GM_ETHNIC

# All percussive sounds
PERCUSSIVE_SOUNDS = GM_PERCUSSIVE + GM_CHROMATIC_PERCUSSION

# All FX sounds
FX_SOUNDS = GM_SYNTH_FX + GM_SOUND_FX

# Complete list of all GM instruments
ALL_GM_INSTRUMENTS = (
    GM_PIANO + GM_CHROMATIC_PERCUSSION + GM_ORGAN + GM_GUITAR + GM_BASS +
    GM_STRINGS + GM_ENSEMBLE + GM_BRASS + GM_REED + GM_PIPE +
    GM_SYNTH_LEAD + GM_SYNTH_PAD + GM_SYNTH_FX + GM_ETHNIC +
    GM_PERCUSSIVE + GM_SOUND_FX
)


# ============================================================================
# GENRE -> SOUND MAPPINGS
# ============================================================================

GENRE_PALETTES = {
    "brazilian_funk": {
        "drums": ["RolandTR808", "RolandTR909", "AkaiMPC60"],
        "bass": ["sawtooth", "gm_synth_bass_1", "gm_synth_bass_2", "gm_slap_bass_1"],
        "lead": ["gm_lead_2_sawtooth", "gm_lead_1_square", "supersaw", "gm_synth_brass_1"],
        "pad": ["gm_pad_poly", "gm_synth_brass_1", "gm_pad_metallic"],
        "high": ["gm_lead_5_charang", "square", "gm_music_box", "gm_synth_voice"],
        "character": "aggressive, punchy, 808-heavy"
    },
    "electro_swing": {
        "drums": ["LinnDrum", "RolandTR909", "RolandTR808", "RolandCompurhythm78"],
        "bass": ["gm_acoustic_bass", "gm_electric_bass_finger", "triangle", "gm_contrabass"],
        "lead": ["gm_trumpet", "gm_clarinet", "gm_alto_sax", "gm_violin", "gm_trombone"],
        "pad": ["gm_string_ensemble_1", "gm_pad_warm", "gm_choir_aahs"],
        "high": ["gm_glockenspiel", "gm_vibraphone", "gm_music_box", "gm_celesta"],
        "character": "jazzy, brass, vintage samples"
    },
    "russian_hardbass": {
        "drums": ["RolandTR909", "RolandTR808", "AlesisHR16", "KorgM1"],
        "bass": ["supersaw", "gm_synth_bass_1", "gm_lead_8_bass_lead", "sawtooth"],
        "lead": ["supersaw", "gm_lead_2_sawtooth", "gm_lead_7_fifths", "gm_synth_brass_1"],
        "pad": ["gm_pad_metallic", "gm_pad_poly", "gm_synth_strings_1"],
        "high": ["gm_lead_1_square", "gm_fx_crystal", "gm_lead_5_charang"],
        "character": "heavy, distorted, eurobeat"
    },
    "phonk": {
        "drums": ["RolandTR808", "CasioRZ1", "EmuSP12", "MPC1000"],
        "bass": ["sine", "triangle", "gm_synth_bass_2", "gm_fretless_bass"],
        "lead": ["gm_fx_echoes", "gm_lead_voice", "gm_pad_choir", "gm_voice_oohs"],
        "pad": ["gm_pad_sweep", "gm_fx_atmosphere", "gm_pad_bowed"],
        "high": ["gm_music_box", "gm_celesta", "gm_glockenspiel", "gm_kalimba"],
        "character": "dark, lo-fi, Memphis rap samples"
    },
    "house": {
        "drums": ["RolandTR909", "RolandTR808", "LinnDrum", "OberheimDMX"],
        "bass": ["sine", "triangle", "gm_synth_bass_1", "gm_electric_bass_finger"],
        "lead": ["gm_lead_2_sawtooth", "supersaw", "gm_pad_poly", "gm_synth_brass_1"],
        "pad": ["gm_pad_warm", "gm_string_ensemble_1", "gm_pad_halo"],
        "high": ["gm_epiano1", "gm_vibraphone", "gm_marimba"],
        "character": "4-on-floor, warm, classic"
    },
    "jpop": {
        "drums": ["RolandTR909", "RolandTR707", "YamahaRM50", "KorgM1"],
        "bass": ["gm_electric_bass_pick", "gm_synth_bass_1", "sawtooth", "gm_slap_bass_1"],
        "lead": ["gm_electric_guitar_clean", "gm_lead_2_sawtooth", "supersaw", "gm_bright_acoustic_piano"],
        "pad": ["gm_pad_new_age", "gm_string_ensemble_1", "gm_synth_strings_1", "gm_choir_aahs"],
        "high": ["gm_glockenspiel", "gm_music_box", "gm_epiano1", "gm_celesta"],
        "character": "bright, layered, energetic"
    },
    "trance": {
        "drums": ["RolandTR909", "RolandTR808", "RolandTR707"],
        "bass": ["sawtooth", "supersaw", "gm_synth_bass_1", "gm_lead_8_bass_lead"],
        "lead": ["supersaw", "gm_lead_2_sawtooth", "gm_lead_7_fifths", "gm_synth_brass_1"],
        "pad": ["gm_pad_halo", "gm_pad_poly", "supersaw", "gm_pad_new_age"],
        "high": ["gm_fx_crystal", "gm_lead_5_charang", "gm_glockenspiel"],
        "character": "uplifting, supersaw-heavy, layered"
    },
    "lofi": {
        "drums": ["RolandTR707", "BossDR110", "CasioVL1", "RolandCompurhythm78"],
        "bass": ["triangle", "sine", "gm_acoustic_bass", "gm_fretless_bass"],
        "lead": ["gm_epiano1", "gm_vibraphone", "gm_acoustic_guitar_nylon", "gm_flute"],
        "pad": ["gm_pad_warm", "gm_pad_new_age", "gm_string_ensemble_1"],
        "high": ["gm_music_box", "gm_celesta", "gm_kalimba", "gm_glockenspiel"],
        "character": "warm, dusty, jazzy chords"
    },
    "synthwave": {
        "drums": ["RolandTR808", "LinnDrum", "OberheimDMX", "SimmonsSDS5"],
        "bass": ["sawtooth", "gm_synth_bass_1", "gm_synth_bass_2", "supersaw"],
        "lead": ["gm_lead_2_sawtooth", "supersaw", "gm_lead_5_charang", "gm_synth_brass_1"],
        "pad": ["gm_pad_poly", "gm_pad_halo", "gm_synth_strings_1", "gm_pad_warm"],
        "high": ["gm_fx_crystal", "gm_lead_1_square", "gm_epiano1"],
        "character": "retro 80s, analog synths, neon"
    },
    "dnb": {
        "drums": ["RolandTR909", "AkaiMPC60", "EmuSP12", "AlesisHR16"],
        "bass": ["sawtooth", "gm_synth_bass_1", "supersaw", "gm_lead_8_bass_lead"],
        "lead": ["gm_lead_2_sawtooth", "supersaw", "gm_pad_metallic", "gm_synth_brass_1"],
        "pad": ["gm_pad_poly", "gm_fx_atmosphere", "gm_pad_sweep"],
        "high": ["gm_fx_crystal", "gm_lead_5_charang", "gm_glockenspiel"],
        "character": "fast breaks, heavy bass, atmospheric"
    },
    "techno": {
        "drums": ["RolandTR909", "RolandTR808", "RolandTR606", "OberheimDMX"],
        "bass": ["sine", "sawtooth", "gm_synth_bass_1", "triangle"],
        "lead": ["gm_lead_1_square", "gm_lead_2_sawtooth", "gm_fx_sci_fi"],
        "pad": ["gm_pad_metallic", "gm_fx_atmosphere", "gm_pad_sweep"],
        "high": ["gm_fx_crystal", "gm_percussion", "gm_lead_5_charang"],
        "character": "minimal, industrial, hypnotic"
    },
    "jazz": {
        "drums": ["LinnDrum", "RolandCompurhythm78", "BossDR110"],
        "bass": ["gm_acoustic_bass", "gm_electric_bass_finger", "gm_fretless_bass"],
        "lead": ["gm_trumpet", "gm_alto_sax", "gm_tenor_sax", "gm_clarinet", "gm_flute"],
        "pad": ["gm_string_ensemble_1", "gm_pad_warm", "gm_choir_aahs"],
        "high": ["gm_vibraphone", "gm_epiano1", "gm_acoustic_grand_piano"],
        "character": "acoustic, swing, sophisticated"
    },
    "classical": {
        "drums": ["gm_timpani", "gm_orchestral_harp"],
        "bass": ["gm_contrabass", "gm_cello", "gm_tuba", "gm_bassoon"],
        "lead": ["gm_violin", "gm_flute", "gm_oboe", "gm_french_horn", "gm_trumpet"],
        "pad": ["gm_string_ensemble_1", "gm_string_ensemble_2", "gm_choir_aahs"],
        "high": ["gm_piccolo", "gm_glockenspiel", "gm_celesta", "gm_tubular_bells"],
        "character": "orchestral, dynamic, acoustic"
    },
    "ambient": {
        "drums": ["RolandTR707", "CasioVL1"],
        "bass": ["sine", "triangle", "gm_pad_bowed"],
        "lead": ["gm_pad_halo", "gm_fx_atmosphere", "gm_fx_echoes"],
        "pad": ["gm_pad_new_age", "gm_pad_halo", "gm_fx_atmosphere", "gm_pad_warm"],
        "high": ["gm_fx_crystal", "gm_music_box", "gm_celesta", "gm_tubular_bells"],
        "character": "atmospheric, evolving, spacious"
    },
    "metal": {
        "drums": ["RolandTR909", "SimmonsSDS5", "AlesisHR16"],
        "bass": ["gm_distortion_guitar", "gm_electric_bass_pick", "sawtooth"],
        "lead": ["gm_distortion_guitar", "gm_overdriven_guitar", "gm_lead_5_charang"],
        "pad": ["gm_pad_metallic", "gm_synth_strings_1"],
        "high": ["gm_distortion_guitar", "gm_lead_1_square"],
        "character": "heavy, distorted, aggressive"
    },
    "reggae": {
        "drums": ["RolandTR808", "LinnDrum", "RolandCompurhythm78"],
        "bass": ["gm_electric_bass_finger", "gm_acoustic_bass", "sine"],
        "lead": ["gm_electric_guitar_clean", "gm_trumpet", "gm_harmonica"],
        "pad": ["gm_drawbar_organ", "gm_rock_organ", "gm_pad_warm"],
        "high": ["gm_electric_guitar_muted", "gm_glockenspiel"],
        "character": "offbeat, laid-back, warm"
    },
    "default": {
        "drums": ["RolandTR808", "RolandTR909", "LinnDrum"],
        "bass": ["sawtooth", "gm_synth_bass_1", "triangle", "gm_electric_bass_finger"],
        "lead": ["gm_lead_2_sawtooth", "supersaw", "gm_epiano1", "gm_trumpet"],
        "pad": ["gm_pad_warm", "gm_string_ensemble_1", "gm_pad_poly"],
        "high": ["gm_music_box", "gm_glockenspiel", "gm_lead_5_charang", "gm_vibraphone"],
        "character": "balanced, versatile"
    }
}


# ============================================================================
# TIMBRE ANALYSIS -> SOUND SELECTION
# ============================================================================

@dataclass
class TimbreProfile:
    """Describes the timbral characteristics of audio."""
    brightness: float  # 0-1, higher = brighter
    warmth: float      # 0-1, higher = more low-mids
    attack: float      # 0-1, higher = faster attack
    sustain: float     # 0-1, higher = more sustained
    harmonic_richness: float  # 0-1, higher = more harmonics
    noise_content: float  # 0-1, higher = more noise
    genre_hint: str = ""


def select_sounds_for_timbre(profile: TimbreProfile) -> Dict[str, List[str]]:
    """Select appropriate sounds based on timbre analysis."""

    # Start with genre palette if available
    genre = profile.genre_hint.lower().replace(" ", "_").replace("-", "_")
    base_palette = GENRE_PALETTES.get(genre, GENRE_PALETTES["default"])

    sounds = {
        "drums": list(base_palette["drums"]),
        "bass": [],
        "lead": [],
        "pad": [],
        "high": []
    }

    # Select bass sounds based on warmth and attack
    if profile.warmth > 0.7:
        sounds["bass"] = ["sine", "triangle", "gm_acoustic_bass", "gm_fretless_bass"]
    elif profile.attack > 0.7:
        sounds["bass"] = ["sawtooth", "gm_synth_bass_1", "gm_slap_bass_1", "gm_electric_bass_pick"]
    else:
        sounds["bass"] = list(base_palette["bass"])

    # Select lead sounds based on brightness and harmonics
    if profile.brightness > 0.7 and profile.harmonic_richness > 0.6:
        sounds["lead"] = ["supersaw", "gm_lead_7_fifths", "gm_brass_section", "gm_synth_brass_1"]
    elif profile.brightness < 0.4:
        sounds["lead"] = ["triangle", "gm_flute", "gm_pad_warm", "gm_recorder"]
    else:
        sounds["lead"] = list(base_palette["lead"])

    # Select pad sounds based on sustain and warmth
    if profile.sustain > 0.7:
        sounds["pad"] = ["gm_string_ensemble_1", "gm_pad_halo", "gm_pad_bowed", "gm_choir_aahs"]
    elif profile.warmth > 0.6:
        sounds["pad"] = ["gm_pad_warm", "gm_pad_new_age", "gm_synth_strings_1"]
    else:
        sounds["pad"] = list(base_palette["pad"])

    # Select high sounds based on brightness
    if profile.brightness > 0.8:
        sounds["high"] = ["gm_glockenspiel", "gm_celesta", "gm_fx_crystal", "gm_piccolo"]
    elif profile.noise_content > 0.5:
        sounds["high"] = ["gm_fx_echoes", "gm_fx_atmosphere", "gm_breath_noise"]
    else:
        sounds["high"] = list(base_palette["high"])

    return sounds


def create_sound_alternation(sounds: List[str], variety: float = 0.5) -> str:
    """
    Create Strudel sound alternation pattern.

    variety: 0-1, higher = more sound changes

    Examples:
    - variety=0: "supersaw"
    - variety=0.3: "<supersaw gm_lead_2_sawtooth>"
    - variety=0.7: "<supersaw gm_lead_2_sawtooth square>"
    """
    if not sounds:
        return "sawtooth"

    if variety < 0.2 or len(sounds) == 1:
        return sounds[0]

    # Select 2-4 sounds based on variety
    if variety < 0.4:
        num_sounds = 2
    elif variety < 0.7:
        num_sounds = min(3, len(sounds))
    else:
        num_sounds = min(4, len(sounds))

    selected = sounds[:num_sounds]
    return f"<{' '.join(selected)}>"


def create_drum_pattern_with_variety(bank: str, alt_banks: List[str] = None) -> str:
    """Create drum pattern with optional bank switching."""
    if not alt_banks or len(alt_banks) < 2:
        return f'.bank("{bank}")'

    # Alternate between banks
    banks = [bank] + [b for b in alt_banks if b != bank][:1]
    return f'.bank("<{" ".join(banks)}")'


# ============================================================================
# ANALYZE AUDIO AND SUGGEST SOUNDS
# ============================================================================

def analyze_and_suggest(
    spectral_centroid: float,
    spectral_rolloff: float,
    rms_mean: float,
    attack_time: float,
    harmonic_ratio: float,
    genre: str = ""
) -> Dict:
    """
    Analyze audio characteristics and suggest sounds.

    Args:
        spectral_centroid: Average spectral centroid in Hz
        spectral_rolloff: Average rolloff frequency in Hz
        rms_mean: Average RMS energy
        attack_time: Average attack time in seconds
        harmonic_ratio: Ratio of harmonic to noise content
        genre: Optional genre hint

    Returns:
        Dict with sound suggestions for each voice
    """

    # Normalize to 0-1 range
    brightness = min(1.0, spectral_centroid / 4000)
    warmth = 1.0 - min(1.0, spectral_rolloff / 8000)
    attack = 1.0 - min(1.0, attack_time / 0.1)  # Faster attack = higher value
    sustain = min(1.0, rms_mean / 0.1)
    harmonic_richness = harmonic_ratio
    noise_content = 1.0 - harmonic_ratio

    profile = TimbreProfile(
        brightness=brightness,
        warmth=warmth,
        attack=attack,
        sustain=sustain,
        harmonic_richness=harmonic_richness,
        noise_content=noise_content,
        genre_hint=genre
    )

    sounds = select_sounds_for_timbre(profile)

    # Create alternation patterns with varying degrees of variety
    result = {
        "bass_sound": create_sound_alternation(sounds["bass"], variety=0.4),
        "lead_sound": create_sound_alternation(sounds["lead"], variety=0.6),
        "pad_sound": create_sound_alternation(sounds["pad"], variety=0.5),
        "high_sound": create_sound_alternation(sounds["high"], variety=0.7),
        "drum_bank": sounds["drums"][0],
        "alt_drum_banks": sounds["drums"][1:] if len(sounds["drums"]) > 1 else [],
        "timbre_profile": {
            "brightness": round(brightness, 2),
            "warmth": round(warmth, 2),
            "attack": round(attack, 2),
            "sustain": round(sustain, 2),
            "harmonic_richness": round(harmonic_richness, 2)
        }
    }

    return result


def get_genre_sounds(genre: str) -> Dict:
    """Get sound palette for a specific genre."""
    genre_key = genre.lower().replace(" ", "_").replace("-", "_")
    palette = GENRE_PALETTES.get(genre_key, GENRE_PALETTES["default"])

    return {
        "bass_sound": create_sound_alternation(palette["bass"], variety=0.5),
        "lead_sound": create_sound_alternation(palette["lead"], variety=0.6),
        "pad_sound": create_sound_alternation(palette["pad"], variety=0.4),
        "high_sound": create_sound_alternation(palette["high"], variety=0.6),
        "drum_bank": palette["drums"][0],
        "alt_drum_banks": palette["drums"][1:],
        "character": palette.get("character", "")
    }


def retrieve_genre_context(genre: str, bpm: float = 0, key: str = "") -> str:
    """Return a compact string of genre-appropriate sounds for LLM prompts.

    ~40 tokens instead of 800 for the full catalog. Falls back to "default" palette.
    """
    genre_key = genre.lower().replace(" ", "_").replace("-", "_") if genre else "default"
    palette = GENRE_PALETTES.get(genre_key, GENRE_PALETTES["default"])

    parts = []
    for role in ("bass", "lead", "pad", "high", "drums"):
        items = palette.get(role, [])
        if items:
            parts.append(f"{role.capitalize()}: {', '.join(items)}")

    character = palette.get("character", "")
    header = f"Available sounds for {genre or 'default'}"
    if character:
        header += f" ({character})"

    return f"{header} — {' | '.join(parts)}"


def get_random_sounds(seed: int = None) -> Dict:
    """Get random sounds from the full catalog for maximum variety."""
    if seed is not None:
        random.seed(seed)

    return {
        "bass_sound": create_sound_alternation(random.sample(BASS_SOUNDS, min(3, len(BASS_SOUNDS))), variety=0.5),
        "lead_sound": create_sound_alternation(random.sample(LEAD_SOUNDS, min(4, len(LEAD_SOUNDS))), variety=0.6),
        "pad_sound": create_sound_alternation(random.sample(PAD_SOUNDS, min(3, len(PAD_SOUNDS))), variety=0.5),
        "high_sound": create_sound_alternation(random.sample(HIGH_SOUNDS, min(4, len(HIGH_SOUNDS))), variety=0.7),
        "drum_bank": random.choice(DRUM_BANKS),
        "alt_drum_banks": random.sample(DRUM_BANKS, 2),
    }


def list_all_sounds() -> Dict[str, List[str]]:
    """Return the complete sound catalog."""
    return {
        "waveforms": WAVEFORMS,
        "zzfx": ZZFX_WAVES,
        "noise": NOISE,
        "drum_banks": DRUM_BANKS,
        "gm_piano": GM_PIANO,
        "gm_chromatic_percussion": GM_CHROMATIC_PERCUSSION,
        "gm_organ": GM_ORGAN,
        "gm_guitar": GM_GUITAR,
        "gm_bass": GM_BASS,
        "gm_strings": GM_STRINGS,
        "gm_ensemble": GM_ENSEMBLE,
        "gm_brass": GM_BRASS,
        "gm_reed": GM_REED,
        "gm_pipe": GM_PIPE,
        "gm_synth_lead": GM_SYNTH_LEAD,
        "gm_synth_pad": GM_SYNTH_PAD,
        "gm_synth_fx": GM_SYNTH_FX,
        "gm_ethnic": GM_ETHNIC,
        "gm_percussive": GM_PERCUSSIVE,
        "gm_sound_fx": GM_SOUND_FX,
        "total_gm_instruments": len(ALL_GM_INSTRUMENTS),
        "total_drum_machines": len(DRUM_BANKS),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sound selector for Strudel")
    parser.add_argument("--genre", default="", help="Genre hint")
    parser.add_argument("--brightness", type=float, default=0.5)
    parser.add_argument("--warmth", type=float, default=0.5)
    parser.add_argument("--attack", type=float, default=0.5)
    parser.add_argument("--list-genres", action="store_true", help="List available genre palettes")
    parser.add_argument("--list-all", action="store_true", help="List all available sounds")
    parser.add_argument("--random", action="store_true", help="Get random sound selection")

    args = parser.parse_args()

    if args.list_genres:
        print("Available genre palettes:")
        for genre, palette in GENRE_PALETTES.items():
            print(f"\n  {genre}:")
            print(f"    Character: {palette.get('character', 'N/A')}")
            print(f"    Drums: {', '.join(palette['drums'][:3])}")
            print(f"    Bass: {', '.join(palette['bass'][:3])}")
            print(f"    Lead: {', '.join(palette['lead'][:3])}")
    elif args.list_all:
        catalog = list_all_sounds()
        print(f"STRUDEL SOUND CATALOG")
        print(f"=====================")
        print(f"Total GM Instruments: {catalog['total_gm_instruments']}")
        print(f"Total Drum Machines: {catalog['total_drum_machines']}")
        print(f"\nDrum Banks ({len(catalog['drum_banks'])}):")
        for i in range(0, len(catalog['drum_banks']), 5):
            print(f"  {', '.join(catalog['drum_banks'][i:i+5])}")
        print(f"\nGM Piano: {', '.join(catalog['gm_piano'])}")
        print(f"GM Bass: {', '.join(catalog['gm_bass'])}")
        print(f"GM Synth Lead: {', '.join(catalog['gm_synth_lead'])}")
        print(f"GM Synth Pad: {', '.join(catalog['gm_synth_pad'])}")
        print(f"GM Brass: {', '.join(catalog['gm_brass'])}")
        print(f"GM Strings: {', '.join(catalog['gm_strings'])}")
    elif args.random:
        sounds = get_random_sounds()
        print(json.dumps(sounds, indent=2))
    else:
        if args.genre:
            sounds = get_genre_sounds(args.genre)
        else:
            sounds = analyze_and_suggest(
                spectral_centroid=2000,
                spectral_rolloff=4000,
                rms_mean=0.05,
                attack_time=0.02,
                harmonic_ratio=0.7,
                genre=args.genre
            )

        print(json.dumps(sounds, indent=2))
