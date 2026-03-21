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
