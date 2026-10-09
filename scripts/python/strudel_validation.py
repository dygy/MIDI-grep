"""Shared Strudel code validation, correction, and extraction utilities.

Single source of truth for sound/bank name corrections, code extraction
from LLM responses, and Strudel-specific validation logic.
"""

import re


# ============================================================================
# SOUND NAME CORRECTIONS (LLM hallucination → valid Strudel name)
# ============================================================================

SOUND_CORRECTIONS = {
    'gm_electric_guitar': 'gm_electric_guitar_clean',
    'gm_electric_piano': 'gm_epiano1',
    'gm_acoustic_guitar': 'gm_acoustic_guitar_nylon',
    'gm_acoustic_piano': 'gm_piano',
    'gm_acoustic_grand_piano': 'gm_piano',
    'gm_grand_piano': 'gm_piano',
    'gm_bright_acoustic_piano': 'gm_piano',
    'gm_electric_bass': 'gm_electric_bass_finger',
    'gm_electric_lead': 'gm_lead_2_sawtooth',
    'gm_synth_lead': 'gm_lead_2_sawtooth',
    'gm_synth_pad': 'gm_pad_warm',
    'gm_synth_bass': 'gm_synth_bass_1',
    'gm_organ': 'gm_drawbar_organ',
    'gm_strings': 'gm_string_ensemble_1',
    'gm_synth_strings': 'gm_synth_strings_1',
    'gm_brass': 'gm_brass_section',
    'gm_synth_brass': 'gm_synth_brass_1',
    'gm_choir': 'gm_choir_aahs',
    'gm_slap_bass': 'gm_slap_bass_1',
    'gm_bass': 'gm_acoustic_bass',
    'gm_lead': 'gm_lead_2_sawtooth',
    'gm_pad': 'gm_pad_warm',
    'gm_fx': 'gm_fx_atmosphere',
    'gm_drum': 'gm_synth_drum',
    'gm_acoustic_electric': 'gm_electric_guitar_clean',
    'gm_electric': 'gm_electric_guitar_clean',
    'gm_guitar': 'gm_acoustic_guitar_nylon',
    'gm_piano1': 'gm_piano',
    'gm_piano2': 'gm_epiano1',
}


# ============================================================================
# BANK NAME CORRECTIONS (LLM hallucination → valid Strudel bank name)
# ============================================================================

BANK_CORRECTIONS = {
    'tr808': 'RolandTR808',
    'TR808': 'RolandTR808',
    'tr909': 'RolandTR909',
    'TR909': 'RolandTR909',
    'tr707': 'RolandTR707',
    'TR707': 'RolandTR707',
    'tr606': 'RolandTR606',
    'TR606': 'RolandTR606',
    'linndrum': 'LinnDrum',
    'linn': 'LinnDrum',
    'dr110': 'BossDR110',
    'mpc60': 'AkaiMPC60',
}


# ============================================================================
# CORRECTION FUNCTIONS
# ============================================================================

def fix_sound_names(code: str, verbose: bool = False) -> str:
    """Auto-correct common LLM sound name hallucinations."""
    for wrong, correct in SOUND_CORRECTIONS.items():
        if wrong == correct:
            continue
        pattern = r'(\.sound\(["\'])' + re.escape(wrong) + r'(["\'])'
        if re.search(pattern, code):
            code = re.sub(pattern, r'\g<1>' + correct + r'\2', code)
            if verbose:
                print(f"  [Validation] Auto-corrected sound: {wrong} → {correct}")
    return code


def fix_bank_names(code: str, verbose: bool = False) -> str:
    """Auto-correct common LLM drum bank name hallucinations."""
    for wrong, correct in BANK_CORRECTIONS.items():
        pattern = r'(\.bank\(["\'])' + re.escape(wrong) + r'(["\'])'
        if re.search(pattern, code):
            code = re.sub(pattern, r'\g<1>' + correct + r'\2', code)
            if verbose:
                print(f"  [Validation] Auto-corrected bank: {wrong} → {correct}")
    return code


def fix_accidentals(code: str) -> str:
    """Map unicode music accidentals to ASCII the Strudel parser accepts: ♭→b, ♯→s.
    The LLM sometimes emits 'a♭4' which crashes the mini-notation parser → silent render."""
    return code.replace("♭", "b").replace("♯", "s").replace("♭", "b").replace("♯", "s")


def fix_names(code: str, verbose: bool = False) -> str:
    """Apply sound/bank-name corrections + accidental sanitisation. The one entry point all codegen
    paths should call so corrections (e.g. tr808 → RolandTR808, ♭ → b) are applied consistently."""
    return fix_accidentals(fix_bank_names(fix_sound_names(code, verbose), verbose))


# ============================================================================
# VALID NAMES — single source of truth (was duplicated in ollama_agent.py)
# ============================================================================

VALID_SYNTHS = {
    "sine", "sin", "triangle", "tri", "square", "sqr", "sawtooth", "saw",
    "supersaw", "pulse", "sbd", "bytebeat",
    "pink", "white", "brown", "crackle",
    "zzfx", "z_sine", "z_sawtooth", "z_triangle", "z_square", "z_tan", "z_noise",
}

VALID_GM_SOUNDS = {
    "gm_piano", "gm_epiano1", "gm_epiano2", "gm_harpsichord", "gm_clavinet",
    "gm_celesta", "gm_glockenspiel", "gm_music_box", "gm_vibraphone",
    "gm_marimba", "gm_xylophone", "gm_tubular_bells", "gm_dulcimer",
    "gm_drawbar_organ", "gm_percussive_organ", "gm_rock_organ", "gm_church_organ",
    "gm_reed_organ", "gm_accordion", "gm_harmonica", "gm_bandoneon",
    "gm_acoustic_guitar_nylon", "gm_acoustic_guitar_steel",
    "gm_electric_guitar_jazz", "gm_electric_guitar_clean",
    "gm_electric_guitar_muted", "gm_overdriven_guitar",
    "gm_distortion_guitar", "gm_guitar_harmonics",
    "gm_acoustic_bass", "gm_electric_bass_finger", "gm_electric_bass_pick",
    "gm_fretless_bass", "gm_slap_bass_1", "gm_slap_bass_2",
    "gm_synth_bass_1", "gm_synth_bass_2",
    "gm_violin", "gm_viola", "gm_cello", "gm_contrabass",
    "gm_tremolo_strings", "gm_pizzicato_strings", "gm_orchestral_harp", "gm_timpani",
    "gm_string_ensemble_1", "gm_string_ensemble_2",
    "gm_synth_strings_1", "gm_synth_strings_2",
    "gm_choir_aahs", "gm_voice_oohs", "gm_synth_choir", "gm_orchestra_hit",
    "gm_trumpet", "gm_trombone", "gm_tuba", "gm_muted_trumpet",
    "gm_french_horn", "gm_brass_section", "gm_synth_brass_1", "gm_synth_brass_2",
    "gm_soprano_sax", "gm_alto_sax", "gm_tenor_sax", "gm_baritone_sax",
    "gm_oboe", "gm_english_horn", "gm_bassoon", "gm_clarinet",
    "gm_piccolo", "gm_flute", "gm_recorder", "gm_pan_flute",
    "gm_blown_bottle", "gm_shakuhachi", "gm_whistle", "gm_ocarina",
    "gm_lead_1_square", "gm_lead_2_sawtooth", "gm_lead_3_calliope",
    "gm_lead_4_chiff", "gm_lead_5_charang", "gm_lead_6_voice",
    "gm_lead_7_fifths", "gm_lead_8_bass_lead",
    "gm_pad_new_age", "gm_pad_warm", "gm_pad_poly", "gm_pad_choir",
    "gm_pad_bowed", "gm_pad_metallic", "gm_pad_halo", "gm_pad_sweep",
    "gm_fx_rain", "gm_fx_soundtrack", "gm_fx_crystal", "gm_fx_atmosphere",
    "gm_fx_brightness", "gm_fx_goblins", "gm_fx_echoes", "gm_fx_sci_fi",
    "gm_sitar", "gm_banjo", "gm_shamisen", "gm_koto",
    "gm_kalimba", "gm_bagpipe", "gm_fiddle", "gm_shanai",
    "gm_tinkle_bell", "gm_agogo", "gm_steel_drums", "gm_woodblock",
    "gm_taiko_drum", "gm_melodic_tom", "gm_synth_drum",
    "gm_reverse_cymbal", "gm_guitar_fret_noise", "gm_breath_noise",
    "gm_seashore", "gm_bird_tweet", "gm_telephone",
    "gm_helicopter", "gm_applause", "gm_gunshot",
}

VALID_DRUM_BANKS = {
    "RolandTR505", "RolandTR606", "RolandTR626", "RolandTR707", "RolandTR727",
    "RolandTR808", "RolandTR909",
    "RolandCompurhythm78", "RolandCompurhythm1000", "RolandCompurhythm8000",
    "RolandD110", "RolandD70", "RolandDDR30", "RolandJD990",
    "RolandMC202", "RolandMC303", "RolandMT32", "RolandR8",
    "RolandS50", "RolandSH09", "RolandSystem100",
    "LinnDrum", "Linn9000", "LinnLM1", "LinnLM2",
    "AkaiLinn", "AkaiMPC60", "AkaiXR10",
    "BossDR55", "BossDR110", "BossDR220", "BossDR550",
    "KorgDDM110", "KorgKPR77", "KorgKR55", "KorgKRZ",
    "KorgM1", "KorgMinipops", "KorgPoly800", "KorgT3",
    "CasioRZ1", "CasioSK1", "CasioVL1",
    "EmuDrumulator", "EmuModular", "EmuSP12",
    "AlesisHR16", "AlesisSR16", "OberheimDMX",
    "SequentialCircuitsDrumtracks", "SequentialCircuitsTom",
    "YamahaRM50", "YamahaRX21", "YamahaRX5", "YamahaRY30", "YamahaTG33",
    "SimmonsSDS400", "SimmonsSDS5",
    "AJKPercusyn", "DoepferMS404", "MFB512", "MPC1000",
    "MoogConcertMateMG1", "RhodesPolaris", "RhythmAce",
    "SakataDPM48", "SergeModular", "SoundmastersR88",
    "UnivoxMicroRhythmer12", "ViscoSpaceDrum", "XdrumLM8953",
}

VALID_SOUNDS = VALID_SYNTHS | VALID_GM_SOUNDS | VALID_DRUM_BANKS

# Numbered/aliased GM names LLMs hallucinate that have no valid Strudel equivalent.
INVALID_GM_PATTERNS = [
    r'gm_pad_\d+_',
    r'gm_fx_\d+_',
    r'gm_electric_piano_\d+',
    r'gm_acoustic_grand',
    r'gm_bright_acoustic',
    r'gm_honkytonk',
]

# Methods LLMs invent that don't exist in Strudel.
INVALID_METHODS = [
    '.peak(', '.eq(', '.volume(', '.filter(', '.bass(', '.treble(',
    '.mid(', '.high(', '.low(', '.boost(', '.cut(', '.compress(',
    '.limit(', '.normalize(',
]

# Replay / reconstruction-by-playback — FORBIDDEN as a deliverable (context/product/values.md A1,
# spec 003 §2.2-A R1/R2). These are real Strudel features, not hallucinations, so they live in
# their own lists with their own error message. The full structural check (texture allowance,
# voice counting, mode) is editability_check.py; this is the cheap generation-time reject that
# the LLM codegen path (ollama_agent._validate_code) gets for free via validate_code().
REPLAY_METHODS = [
    '.loopAt(',
]

# s("originalfull") / s("<stem>full") / s("origseg<N>") / s("<stem>loop") — also .s( / sound(.
REPLAY_SOUND_PATTERN = re.compile(
    r'(?<![\w$])(?:\.\s*)?(?:s|sound)\(\s*["\'`]\s*(originalfull|[a-z]+full|origseg\d+|[a-z]+loop)\b'
)

_VALID_STRUDEL_PATTERNS = [
    '.sound(', '.gain(', '.lpf(', '.hpf(', '.room(', '.delay(', '.bank(',
    '.attack(', '.release(', '.decay(', '.sustain(',
    '.crush(', '.distort(', '.phaser(', '.vibrato(',
    'note(', 's(', 'setcps(', '$:',
]


def validate_code(code: str, autocorrect: bool = True):
    """Validate Strudel code against the single source of truth.

    Returns (corrected_code, error). error == "" means valid. If autocorrect is True,
    sound/bank name hallucinations (e.g. tr808 → RolandTR808) are fixed before validation,
    and the corrected code is returned so callers can persist the fixed version.
    """
    if autocorrect:
        code = fix_names(code)

    for invalid in INVALID_METHODS:
        if invalid in code:
            return code, f"invalid method {invalid} (non-existent Strudel method)"

    for replay in REPLAY_METHODS:
        if replay in code:
            return code, f"replay method {replay} (audio replay is forbidden — values.md A1; use note()/s() patterns)"

    rm = REPLAY_SOUND_PATTERN.search(code)
    if rm:
        return code, f"replay sound '{rm.group(1)}' (full-stem replay is forbidden — values.md A1; use a sample-instrument or synth)"

    for pattern in INVALID_GM_PATTERNS:
        m = re.findall(pattern, code)
        if m:
            return code, f"invalid sound pattern {m[0]} (use correct Strudel GM names)"

    for sound in re.findall(r'\.sound\(["\']([^"\']+)["\']', code):
        for s in sound.strip('<>').split():
            s = s.strip()
            if s and s not in VALID_SOUNDS:
                return code, f"unknown sound '{s}' (not in Strudel's sound library)"

    for bank in re.findall(r'\.bank\(["\']([^"\']+)["\']', code):
        bank = bank.strip()
        if bank and bank not in VALID_DRUM_BANKS:
            return code, f"unknown drum bank '{bank}' (not in Strudel's drum library)"

    if not any(p in code for p in _VALID_STRUDEL_PATTERNS):
        return code, "no recognizable Strudel patterns"

    return code, ""
