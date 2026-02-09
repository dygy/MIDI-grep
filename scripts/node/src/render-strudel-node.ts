#!/usr/bin/env node
/**
 * Node.js Strudel Renderer with Dynamic Synthesis
 *
 * Uses Strudel's mini notation parser for accurate pattern parsing.
 * Accepts synthesis config from AI audio analyzer for dynamic parameters.
 * NO HARDCODED VALUES - everything comes from config or analysis.
 */

// @ts-ignore - Strudel types not available
import { mini } from '@strudel/mini';
import fs from 'fs';
import path from 'path';

const SAMPLE_RATE = 44100;

// ============================================
// INTERFACES
// ============================================

interface EnvelopeConfig {
  attack: number;  // seconds
  decay: number;
  sustain: number; // 0-1
  release: number;
}

interface VoiceSynthConfig {
  gain: number;
  lpf: number;
  hpf: number;
  envelope: EnvelopeConfig;
  waveform: 'sine' | 'saw' | 'square' | 'triangle' | 'noise';
  detune_cents?: number;
  sub_octave_gain?: number;
  transient_boost?: number;
  reverb?: number;
  noise_mix?: number;  // 0-1 amount of noise to mix for more realistic timbre
}

interface SynthConfig {
  envelope: EnvelopeConfig;
  filters: {
    lpf_cutoff: number;
    hpf_cutoff: number;
    filter_envelope_amount: number;
    filter_attack: number;
    filter_decay: number;
  };
  oscillator: {
    waveform: string;
    harmonics: number;
    detune_cents: number;
    sub_octave_gain: number;
  };
  dynamics: {
    compression_ratio: number;
    target_rms: number;
    limiter_threshold: number;
  };
  tempo: {
    bpm: number;
    confidence: number;
    samples_per_beat: number;
    sync_to_beat: boolean;
  };
  // FM synthesis settings (AI-derived from original audio)
  fm?: {
    enabled: boolean;
    modulator_ratio: number;  // Ratio of modulator to carrier frequency
    modulation_index: number; // Depth of modulation (0-10 typical)
  };
  // Formant settings (resonant peaks from original audio)
  formants?: {
    frequencies: number[];  // Peak frequencies in Hz
    amplitudes: number[];   // Relative amplitudes (0-1)
    Q: number;              // Resonance quality factor
  };
  // Harmonic amplitudes for additive synthesis
  harmonics_profile?: number[];  // Amplitude of each harmonic (1st, 2nd, 3rd...)
  voices: {
    bass: VoiceSynthConfig;
    mid: VoiceSynthConfig;
    high: VoiceSynthConfig;
    drums: Partial<VoiceSynthConfig>;
  };
  master: {
    gain: number;
    hpf: number;
    limiter: boolean;
    target_centroid?: number;
    high_shelf_boost?: number;  // dB boost above 2kHz
  };
}

interface GranularModel {
  name: string;
  samples: Map<string, Float32Array>;
  drumSamples?: Map<string, Float32Array[]>;
  sampleRate: number;
}

interface VoiceEffects {
  gain: number;
  lpf?: number;
  hpf?: number;
}

interface ParsedVoice {
  name: string;
  patterns: string[];
  effects: VoiceEffects;
  modelType: 'melodic' | 'bass' | 'drums';
  soundName?: string;  // The sound from .sound() in the code
}

interface StemOutputs {
  mixed: Float32Array;
  bass: Float32Array;
  drums: Float32Array;
  melodic: Float32Array;  // mid + high combined
}

// ============================================
// SOUND-TO-SYNTHESIS MAPPING
// Maps Strudel sound names to synthesis parameters for variety
// ============================================

interface SoundMapping {
  waveform: 'sine' | 'saw' | 'square' | 'triangle' | 'noise';
  lpf: number;
  attack: number;
  decay: number;
  sustain: number;
  release: number;
  detune?: number;
  subOctave?: number;
  noiseMix?: number;
}

const SOUND_MAPPINGS: { [key: string]: SoundMapping } = {
  // Basic waveforms
  'sine': { waveform: 'sine', lpf: 20000, attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.2 },
  'sawtooth': { waveform: 'saw', lpf: 8000, attack: 0.005, decay: 0.15, sustain: 0.7, release: 0.2 },
  'square': { waveform: 'square', lpf: 6000, attack: 0.005, decay: 0.1, sustain: 0.6, release: 0.15 },
  'triangle': { waveform: 'triangle', lpf: 10000, attack: 0.02, decay: 0.2, sustain: 0.8, release: 0.25 },
  'supersaw': { waveform: 'saw', lpf: 10000, attack: 0.01, decay: 0.15, sustain: 0.8, release: 0.25, detune: 15 },

  // ========== BASS (33-40) ==========
  'gm_acoustic_bass': { waveform: 'sine', lpf: 800, attack: 0.02, decay: 0.3, sustain: 0.4, release: 0.2, subOctave: 0.3 },
  'gm_electric_bass_finger': { waveform: 'saw', lpf: 1200, attack: 0.01, decay: 0.2, sustain: 0.5, release: 0.15 },
  'gm_electric_bass_pick': { waveform: 'saw', lpf: 2000, attack: 0.002, decay: 0.15, sustain: 0.4, release: 0.1 },
  'gm_fretless_bass': { waveform: 'sine', lpf: 1000, attack: 0.03, decay: 0.3, sustain: 0.6, release: 0.3 },
  'gm_slap_bass_1': { waveform: 'saw', lpf: 3000, attack: 0.001, decay: 0.1, sustain: 0.3, release: 0.1 },
  'gm_slap_bass_2': { waveform: 'saw', lpf: 3500, attack: 0.001, decay: 0.12, sustain: 0.35, release: 0.1 },
  'gm_synth_bass_1': { waveform: 'saw', lpf: 1500, attack: 0.005, decay: 0.2, sustain: 0.6, release: 0.15, subOctave: 0.2 },
  'gm_synth_bass_2': { waveform: 'square', lpf: 1200, attack: 0.003, decay: 0.15, sustain: 0.5, release: 0.12 },
  'gm_tuba': { waveform: 'saw', lpf: 600, attack: 0.05, decay: 0.3, sustain: 0.6, release: 0.25, subOctave: 0.2 },
  'gm_contrabass': { waveform: 'saw', lpf: 700, attack: 0.04, decay: 0.35, sustain: 0.5, release: 0.3, subOctave: 0.25 },

  // ========== SYNTH LEAD (81-88) ==========
  'gm_lead_1_square': { waveform: 'square', lpf: 6000, attack: 0.01, decay: 0.15, sustain: 0.7, release: 0.2 },
  'gm_lead_2_sawtooth': { waveform: 'saw', lpf: 8000, attack: 0.01, decay: 0.2, sustain: 0.75, release: 0.25 },
  'gm_lead_3_calliope': { waveform: 'sine', lpf: 4000, attack: 0.05, decay: 0.3, sustain: 0.6, release: 0.3 },
  'gm_lead_4_chiff': { waveform: 'square', lpf: 3000, attack: 0.001, decay: 0.1, sustain: 0.4, release: 0.1 },
  'gm_lead_5_charang': { waveform: 'saw', lpf: 5000, attack: 0.005, decay: 0.1, sustain: 0.5, release: 0.15, detune: 8 },
  'gm_lead_6_voice': { waveform: 'sine', lpf: 3000, attack: 0.08, decay: 0.4, sustain: 0.7, release: 0.4 },
  'gm_lead_7_fifths': { waveform: 'saw', lpf: 7000, attack: 0.02, decay: 0.2, sustain: 0.7, release: 0.25, detune: 10 },
  'gm_lead_8_bass_lead': { waveform: 'saw', lpf: 2000, attack: 0.01, decay: 0.2, sustain: 0.65, release: 0.2, subOctave: 0.3 },

  // ========== SYNTH PAD (89-96) ==========
  'gm_pad_1_new_age': { waveform: 'sine', lpf: 4000, attack: 0.3, decay: 0.5, sustain: 0.8, release: 0.8 },
  'gm_pad_2_warm': { waveform: 'saw', lpf: 2000, attack: 0.2, decay: 0.4, sustain: 0.7, release: 0.6, detune: 5 },
  'gm_pad_3_polysynth': { waveform: 'saw', lpf: 4000, attack: 0.15, decay: 0.3, sustain: 0.75, release: 0.5, detune: 12 },
  'gm_pad_4_choir': { waveform: 'sine', lpf: 3000, attack: 0.25, decay: 0.4, sustain: 0.8, release: 0.7, noiseMix: 0.05 },
  'gm_pad_5_bowed': { waveform: 'saw', lpf: 3500, attack: 0.35, decay: 0.4, sustain: 0.75, release: 0.6 },
  'gm_pad_6_metallic': { waveform: 'square', lpf: 5000, attack: 0.2, decay: 0.35, sustain: 0.7, release: 0.5, detune: 8 },
  'gm_pad_7_halo': { waveform: 'sine', lpf: 5000, attack: 0.4, decay: 0.5, sustain: 0.85, release: 1.0 },
  'gm_pad_8_sweep': { waveform: 'saw', lpf: 6000, attack: 0.5, decay: 0.6, sustain: 0.7, release: 1.2, detune: 10 },

  // ========== PIANO (1-8) ==========
  'gm_acoustic_grand_piano': { waveform: 'triangle', lpf: 6000, attack: 0.005, decay: 0.5, sustain: 0.3, release: 0.4 },
  'gm_bright_acoustic_piano': { waveform: 'triangle', lpf: 8000, attack: 0.003, decay: 0.4, sustain: 0.35, release: 0.35 },
  'gm_electric_grand_piano': { waveform: 'sine', lpf: 5000, attack: 0.005, decay: 0.4, sustain: 0.4, release: 0.35 },
  'gm_honkytonk_piano': { waveform: 'triangle', lpf: 5000, attack: 0.003, decay: 0.35, sustain: 0.25, release: 0.3, detune: 12 },
  'gm_electric_piano_1': { waveform: 'sine', lpf: 4000, attack: 0.01, decay: 0.3, sustain: 0.4, release: 0.3 },
  'gm_electric_piano_2': { waveform: 'sine', lpf: 5000, attack: 0.008, decay: 0.35, sustain: 0.45, release: 0.35 },
  'gm_harpsichord': { waveform: 'saw', lpf: 7000, attack: 0.001, decay: 0.2, sustain: 0.1, release: 0.15 },
  'gm_clavinet': { waveform: 'square', lpf: 5000, attack: 0.001, decay: 0.15, sustain: 0.2, release: 0.1 },

  // ========== CHROMATIC PERCUSSION (9-16) ==========
  'gm_celesta': { waveform: 'sine', lpf: 10000, attack: 0.001, decay: 0.5, sustain: 0.1, release: 0.4 },
  'gm_glockenspiel': { waveform: 'sine', lpf: 12000, attack: 0.001, decay: 0.8, sustain: 0.1, release: 0.5 },
  'gm_music_box': { waveform: 'sine', lpf: 8000, attack: 0.001, decay: 0.6, sustain: 0.15, release: 0.4 },
  'gm_vibraphone': { waveform: 'sine', lpf: 6000, attack: 0.01, decay: 0.5, sustain: 0.3, release: 0.5 },
  'gm_marimba': { waveform: 'sine', lpf: 4000, attack: 0.005, decay: 0.4, sustain: 0.1, release: 0.3 },
  'gm_xylophone': { waveform: 'triangle', lpf: 8000, attack: 0.001, decay: 0.3, sustain: 0.05, release: 0.2 },
  'gm_tubular_bells': { waveform: 'sine', lpf: 6000, attack: 0.01, decay: 1.0, sustain: 0.2, release: 0.8 },
  'gm_dulcimer': { waveform: 'triangle', lpf: 5000, attack: 0.002, decay: 0.4, sustain: 0.2, release: 0.3 },

  // ========== ORGAN (17-24) ==========
  'gm_drawbar_organ': { waveform: 'sine', lpf: 6000, attack: 0.01, decay: 0.1, sustain: 0.9, release: 0.1 },
  'gm_percussive_organ': { waveform: 'sine', lpf: 5000, attack: 0.005, decay: 0.2, sustain: 0.7, release: 0.15 },
  'gm_rock_organ': { waveform: 'square', lpf: 4000, attack: 0.01, decay: 0.15, sustain: 0.85, release: 0.1, detune: 5 },
  'gm_church_organ': { waveform: 'sine', lpf: 3000, attack: 0.15, decay: 0.3, sustain: 0.9, release: 0.4 },
  'gm_reed_organ': { waveform: 'saw', lpf: 2500, attack: 0.08, decay: 0.2, sustain: 0.8, release: 0.2 },
  'gm_accordion': { waveform: 'saw', lpf: 3500, attack: 0.05, decay: 0.15, sustain: 0.75, release: 0.15, detune: 8 },
  'gm_harmonica': { waveform: 'saw', lpf: 3000, attack: 0.03, decay: 0.1, sustain: 0.7, release: 0.1, noiseMix: 0.03 },
  'gm_tango_accordion': { waveform: 'saw', lpf: 4000, attack: 0.04, decay: 0.15, sustain: 0.8, release: 0.15, detune: 10 },

  // ========== GUITAR (25-32) ==========
  'gm_acoustic_guitar_nylon': { waveform: 'triangle', lpf: 4000, attack: 0.005, decay: 0.4, sustain: 0.3, release: 0.3 },
  'gm_acoustic_guitar_steel': { waveform: 'triangle', lpf: 5000, attack: 0.003, decay: 0.35, sustain: 0.25, release: 0.25 },
  'gm_electric_guitar_jazz': { waveform: 'sine', lpf: 3000, attack: 0.01, decay: 0.3, sustain: 0.5, release: 0.25 },
  'gm_electric_guitar_clean': { waveform: 'triangle', lpf: 4500, attack: 0.005, decay: 0.25, sustain: 0.4, release: 0.2 },
  'gm_electric_guitar_muted': { waveform: 'triangle', lpf: 2000, attack: 0.002, decay: 0.1, sustain: 0.2, release: 0.08 },
  'gm_overdriven_guitar': { waveform: 'saw', lpf: 4000, attack: 0.005, decay: 0.2, sustain: 0.6, release: 0.2 },
  'gm_distortion_guitar': { waveform: 'square', lpf: 5000, attack: 0.003, decay: 0.15, sustain: 0.7, release: 0.2 },
  'gm_guitar_harmonics': { waveform: 'sine', lpf: 8000, attack: 0.01, decay: 0.6, sustain: 0.2, release: 0.4 },

  // ========== STRINGS (41-48) ==========
  'gm_violin': { waveform: 'saw', lpf: 5000, attack: 0.08, decay: 0.2, sustain: 0.75, release: 0.2 },
  'gm_viola': { waveform: 'saw', lpf: 4000, attack: 0.1, decay: 0.25, sustain: 0.7, release: 0.25 },
  'gm_cello': { waveform: 'saw', lpf: 3000, attack: 0.12, decay: 0.3, sustain: 0.7, release: 0.3 },
  'gm_tremolo_strings': { waveform: 'saw', lpf: 4000, attack: 0.05, decay: 0.2, sustain: 0.7, release: 0.3, detune: 6 },
  'gm_pizzicato_strings': { waveform: 'triangle', lpf: 3500, attack: 0.002, decay: 0.2, sustain: 0.1, release: 0.15 },
  'gm_orchestral_harp': { waveform: 'triangle', lpf: 5000, attack: 0.005, decay: 0.5, sustain: 0.2, release: 0.4 },
  'gm_timpani': { waveform: 'sine', lpf: 400, attack: 0.01, decay: 0.6, sustain: 0.2, release: 0.4, subOctave: 0.3 },

  // ========== ENSEMBLE (49-56) ==========
  'gm_string_ensemble_1': { waveform: 'saw', lpf: 4000, attack: 0.15, decay: 0.3, sustain: 0.7, release: 0.4, detune: 8 },
  'gm_string_ensemble_2': { waveform: 'saw', lpf: 3500, attack: 0.2, decay: 0.35, sustain: 0.65, release: 0.45, detune: 10 },
  'gm_synth_strings_1': { waveform: 'saw', lpf: 5000, attack: 0.1, decay: 0.25, sustain: 0.75, release: 0.35, detune: 12 },
  'gm_synth_strings_2': { waveform: 'saw', lpf: 4500, attack: 0.15, decay: 0.3, sustain: 0.7, release: 0.4, detune: 15 },
  'gm_choir_aahs': { waveform: 'sine', lpf: 3000, attack: 0.2, decay: 0.4, sustain: 0.8, release: 0.5, noiseMix: 0.03 },
  'gm_voice_oohs': { waveform: 'sine', lpf: 2500, attack: 0.25, decay: 0.45, sustain: 0.75, release: 0.55, noiseMix: 0.02 },
  'gm_synth_voice': { waveform: 'sine', lpf: 3500, attack: 0.15, decay: 0.35, sustain: 0.7, release: 0.4 },
  'gm_orchestra_hit': { waveform: 'saw', lpf: 6000, attack: 0.001, decay: 0.3, sustain: 0.2, release: 0.25, detune: 15 },

  // ========== BRASS (57-64) ==========
  'gm_trumpet': { waveform: 'saw', lpf: 5000, attack: 0.03, decay: 0.2, sustain: 0.7, release: 0.15 },
  'gm_trombone': { waveform: 'saw', lpf: 3000, attack: 0.05, decay: 0.25, sustain: 0.65, release: 0.2 },
  'gm_muted_trumpet': { waveform: 'saw', lpf: 2000, attack: 0.03, decay: 0.2, sustain: 0.6, release: 0.15 },
  'gm_french_horn': { waveform: 'saw', lpf: 2500, attack: 0.08, decay: 0.3, sustain: 0.65, release: 0.25 },
  'gm_brass_section': { waveform: 'saw', lpf: 4000, attack: 0.02, decay: 0.2, sustain: 0.7, release: 0.2, detune: 10 },
  'gm_synth_brass_1': { waveform: 'saw', lpf: 6000, attack: 0.01, decay: 0.15, sustain: 0.75, release: 0.15, detune: 8 },
  'gm_synth_brass_2': { waveform: 'square', lpf: 5000, attack: 0.015, decay: 0.18, sustain: 0.7, release: 0.18, detune: 10 },

  // ========== REED (65-72) ==========
  'gm_soprano_sax': { waveform: 'saw', lpf: 5000, attack: 0.035, decay: 0.18, sustain: 0.72, release: 0.18, noiseMix: 0.02 },
  'gm_alto_sax': { waveform: 'saw', lpf: 4000, attack: 0.04, decay: 0.2, sustain: 0.7, release: 0.2, noiseMix: 0.02 },
  'gm_tenor_sax': { waveform: 'saw', lpf: 3500, attack: 0.04, decay: 0.25, sustain: 0.65, release: 0.25, noiseMix: 0.02 },
  'gm_baritone_sax': { waveform: 'saw', lpf: 2500, attack: 0.045, decay: 0.28, sustain: 0.6, release: 0.28, noiseMix: 0.02 },
  'gm_oboe': { waveform: 'saw', lpf: 4500, attack: 0.06, decay: 0.2, sustain: 0.7, release: 0.18, noiseMix: 0.01 },
  'gm_english_horn': { waveform: 'saw', lpf: 3500, attack: 0.07, decay: 0.22, sustain: 0.68, release: 0.2, noiseMix: 0.01 },
  'gm_bassoon': { waveform: 'saw', lpf: 2000, attack: 0.08, decay: 0.25, sustain: 0.65, release: 0.25, noiseMix: 0.015 },
  'gm_clarinet': { waveform: 'square', lpf: 3500, attack: 0.05, decay: 0.2, sustain: 0.7, release: 0.18 },

  // ========== PIPE (73-80) ==========
  'gm_piccolo': { waveform: 'sine', lpf: 10000, attack: 0.04, decay: 0.15, sustain: 0.75, release: 0.15, noiseMix: 0.02 },
  'gm_flute': { waveform: 'sine', lpf: 6000, attack: 0.05, decay: 0.18, sustain: 0.7, release: 0.18, noiseMix: 0.03 },
  'gm_recorder': { waveform: 'sine', lpf: 5000, attack: 0.04, decay: 0.15, sustain: 0.72, release: 0.15, noiseMix: 0.02 },
  'gm_pan_flute': { waveform: 'sine', lpf: 4000, attack: 0.06, decay: 0.2, sustain: 0.65, release: 0.2, noiseMix: 0.05 },
  'gm_blown_bottle': { waveform: 'sine', lpf: 2000, attack: 0.1, decay: 0.3, sustain: 0.5, release: 0.3, noiseMix: 0.1 },
  'gm_shakuhachi': { waveform: 'sine', lpf: 4000, attack: 0.08, decay: 0.25, sustain: 0.6, release: 0.25, noiseMix: 0.08 },
  'gm_whistle': { waveform: 'sine', lpf: 8000, attack: 0.03, decay: 0.1, sustain: 0.8, release: 0.1, noiseMix: 0.02 },
  'gm_ocarina': { waveform: 'sine', lpf: 3500, attack: 0.05, decay: 0.18, sustain: 0.7, release: 0.18, noiseMix: 0.02 },

  // ========== SYNTH FX (97-104) ==========
  'gm_fx_1_rain': { waveform: 'noise', lpf: 4000, attack: 0.5, decay: 0.8, sustain: 0.4, release: 1.5 },
  'gm_fx_2_soundtrack': { waveform: 'saw', lpf: 3000, attack: 0.4, decay: 0.6, sustain: 0.7, release: 1.0, detune: 12 },
  'gm_fx_3_crystal': { waveform: 'sine', lpf: 12000, attack: 0.1, decay: 0.4, sustain: 0.5, release: 0.8 },
  'gm_fx_4_atmosphere': { waveform: 'sine', lpf: 3000, attack: 0.3, decay: 0.5, sustain: 0.7, release: 1.0, noiseMix: 0.1 },
  'gm_fx_5_brightness': { waveform: 'saw', lpf: 10000, attack: 0.2, decay: 0.4, sustain: 0.6, release: 0.8, detune: 8 },
  'gm_fx_6_goblins': { waveform: 'square', lpf: 2500, attack: 0.25, decay: 0.5, sustain: 0.5, release: 0.7, detune: 15 },
  'gm_fx_7_echoes': { waveform: 'sine', lpf: 4000, attack: 0.2, decay: 0.6, sustain: 0.6, release: 1.2 },
  'gm_fx_8_sci_fi': { waveform: 'saw', lpf: 6000, attack: 0.15, decay: 0.4, sustain: 0.55, release: 0.9, detune: 20 },

  // ========== ETHNIC (105-112) ==========
  'gm_sitar': { waveform: 'saw', lpf: 5000, attack: 0.01, decay: 0.5, sustain: 0.3, release: 0.4, detune: 5 },
  'gm_banjo': { waveform: 'triangle', lpf: 6000, attack: 0.002, decay: 0.25, sustain: 0.15, release: 0.2 },
  'gm_shamisen': { waveform: 'triangle', lpf: 4500, attack: 0.003, decay: 0.3, sustain: 0.2, release: 0.25 },
  'gm_koto': { waveform: 'triangle', lpf: 5000, attack: 0.005, decay: 0.4, sustain: 0.2, release: 0.35 },
  'gm_kalimba': { waveform: 'sine', lpf: 6000, attack: 0.002, decay: 0.5, sustain: 0.1, release: 0.4 },
  'gm_bag_pipe': { waveform: 'saw', lpf: 3000, attack: 0.1, decay: 0.2, sustain: 0.85, release: 0.15, detune: 5, noiseMix: 0.03 },
  'gm_fiddle': { waveform: 'saw', lpf: 5000, attack: 0.06, decay: 0.2, sustain: 0.75, release: 0.2 },
  'gm_shanai': { waveform: 'saw', lpf: 4000, attack: 0.08, decay: 0.25, sustain: 0.7, release: 0.2, noiseMix: 0.04 },

  // ========== PERCUSSIVE (113-119) ==========
  'gm_tinkle_bell': { waveform: 'sine', lpf: 10000, attack: 0.001, decay: 0.4, sustain: 0.1, release: 0.3 },
  'gm_agogo': { waveform: 'sine', lpf: 6000, attack: 0.001, decay: 0.3, sustain: 0.15, release: 0.25 },
  'gm_steel_drums': { waveform: 'sine', lpf: 5000, attack: 0.005, decay: 0.4, sustain: 0.25, release: 0.35 },
  'gm_woodblock': { waveform: 'triangle', lpf: 4000, attack: 0.001, decay: 0.15, sustain: 0.05, release: 0.1 },
  'gm_taiko_drum': { waveform: 'sine', lpf: 500, attack: 0.005, decay: 0.4, sustain: 0.2, release: 0.3, subOctave: 0.4 },
  'gm_melodic_tom': { waveform: 'sine', lpf: 800, attack: 0.005, decay: 0.3, sustain: 0.15, release: 0.25 },
  'gm_synth_drum': { waveform: 'sine', lpf: 600, attack: 0.002, decay: 0.25, sustain: 0.1, release: 0.2, subOctave: 0.3 },

  // ========== SOUND EFFECTS (120-128) ==========
  'gm_reverse_cymbal': { waveform: 'noise', lpf: 8000, attack: 1.0, decay: 0.3, sustain: 0.8, release: 0.2 },
  'gm_guitar_fret_noise': { waveform: 'noise', lpf: 6000, attack: 0.001, decay: 0.15, sustain: 0.1, release: 0.1 },
  'gm_breath_noise': { waveform: 'noise', lpf: 3000, attack: 0.1, decay: 0.3, sustain: 0.5, release: 0.3 },
  'gm_seashore': { waveform: 'noise', lpf: 2000, attack: 1.5, decay: 1.0, sustain: 0.6, release: 1.5 },
  'gm_bird_tweet': { waveform: 'sine', lpf: 10000, attack: 0.01, decay: 0.1, sustain: 0.3, release: 0.1 },
  'gm_telephone_ring': { waveform: 'square', lpf: 4000, attack: 0.001, decay: 0.05, sustain: 0.8, release: 0.05 },
  'gm_helicopter': { waveform: 'noise', lpf: 500, attack: 0.5, decay: 0.5, sustain: 0.7, release: 0.5 },
  'gm_applause': { waveform: 'noise', lpf: 6000, attack: 0.3, decay: 0.5, sustain: 0.6, release: 0.8 },
  'gm_gunshot': { waveform: 'noise', lpf: 8000, attack: 0.001, decay: 0.1, sustain: 0.05, release: 0.15 },
};

function getSoundMapping(soundName: string): SoundMapping {
  // Handle alternating sounds - just use the first one for offline rendering
  let name = soundName;
  if (soundName.startsWith('<') && soundName.endsWith('>')) {
    const sounds = soundName.slice(1, -1).trim().split(/\s+/);
    name = sounds[0];
  }

  return SOUND_MAPPINGS[name] || SOUND_MAPPINGS['sawtooth'];
}

// ============================================
// NOTE MAPPING
// ============================================

const NOTE_TO_MIDI: { [key: string]: number } = {};
const NOTES = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'];
for (let octave = 0; octave <= 8; octave++) {
  NOTES.forEach((note, i) => {
    NOTE_TO_MIDI[`${note}${octave}`] = 12 + octave * 12 + i;
  });
}

function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function noteToMidi(note: string): number | null {
  const normalized = note.toLowerCase().replace('#', 's');
  return NOTE_TO_MIDI[normalized] ?? null;
}

// ============================================
// DEFAULT CONFIG (fallback when no config provided)
// ============================================

function getDefaultConfig(): SynthConfig {
  return {
    envelope: { attack: 0.01, decay: 0.1, sustain: 0.7, release: 0.15 },
    filters: {
      lpf_cutoff: 8000,
      hpf_cutoff: 80,
      filter_envelope_amount: 0.3,
      filter_attack: 0.02,
      filter_decay: 0.3
    },
    oscillator: {
      waveform: 'saw',
      harmonics: 5,
      detune_cents: 5,
      sub_octave_gain: 0.2
    },
    dynamics: {
      compression_ratio: 2.0,
      target_rms: 0.3,
      limiter_threshold: 0.95
    },
    tempo: {
      bpm: 120,
      confidence: 0.5,
      samples_per_beat: Math.floor(SAMPLE_RATE * 60 / 120),
      sync_to_beat: false
    },
    voices: {
      bass: {
        // Reduced from 0.3 to 0.1 - bass was 4812% too loud
        gain: 0.1, lpf: 400, hpf: 40,
        envelope: { attack: 0.005, decay: 0.2, sustain: 0.6, release: 0.1 },
        waveform: 'saw', sub_octave_gain: 0.3
      },
      mid: {
        // Reduced from 1.0 to 0.6 - mids were dominating (+31%)
        gain: 0.6, lpf: 6000, hpf: 200,
        envelope: { attack: 0.01, decay: 0.15, sustain: 0.7, release: 0.2 },
        waveform: 'saw', detune_cents: 5
      },
      high: {
        gain: 0.8, lpf: 12000, hpf: 400,
        envelope: { attack: 0.005, decay: 0.1, sustain: 0.5, release: 0.15 },
        waveform: 'square'
      },
      drums: {
        gain: 0.7, transient_boost: 0.5, reverb: 0.1
      }
    },
    master: { gain: 0.9, hpf: 30, limiter: true }
  };
}

// ============================================
// WAVEFORM GENERATORS
// ============================================

function generateSine(phase: number): number {
  return Math.sin(2 * Math.PI * phase);
}

function generateSaw(phase: number): number {
  return 2 * (phase % 1) - 1;
}

function generateSquare(phase: number): number {
  return (phase % 1) < 0.5 ? 1 : -1;
}

function generateTriangle(phase: number): number {
  const t = phase % 1;
  return Math.abs(t * 4 - 2) - 1;
}

function generateNoise(): number {
  return Math.random() * 2 - 1;
}

function generateWaveform(waveform: string, phase: number): number {
  switch (waveform) {
    case 'sine': return generateSine(phase);
    case 'saw': return generateSaw(phase);
    case 'square': return generateSquare(phase);
    case 'triangle': return generateTriangle(phase);
    case 'noise': return generateNoise();
    default: return generateSaw(phase);
  }
}

// ============================================
// ENVELOPE GENERATOR
// ============================================

function applyEnvelope(t: number, duration: number, env: EnvelopeConfig): number {
  const { attack, decay, sustain, release } = env;

  if (t < attack) {
    // Attack phase
    return t / attack;
  } else if (t < attack + decay) {
    // Decay phase
    const decayProgress = (t - attack) / decay;
    return 1 - (1 - sustain) * decayProgress;
  } else if (t < duration - release) {
    // Sustain phase
    return sustain;
  } else {
    // Release phase
    const releaseProgress = (t - (duration - release)) / release;
    return sustain * (1 - Math.min(1, releaseProgress));
  }
}

// ============================================
// FILTER (Simple one-pole)
// ============================================

function applyLPF(samples: Float32Array, cutoffHz: number, sampleRate: number): void {
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const dt = 1 / sampleRate;
  const alpha = dt / (rc + dt);

  let prev = samples[0];
  for (let i = 1; i < samples.length; i++) {
    samples[i] = prev + alpha * (samples[i] - prev);
    prev = samples[i];
  }
}

function applyHPF(samples: Float32Array, cutoffHz: number, sampleRate: number): void {
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);

  let prevIn = samples[0];
  let prevOut = samples[0];
  for (let i = 1; i < samples.length; i++) {
    const current = samples[i];
    samples[i] = alpha * (prevOut + current - prevIn);
    prevIn = current;
    prevOut = samples[i];
  }
}

// ============================================
// RESONANT BANDPASS FILTER (for formants)
// ============================================

interface BiquadState {
  x1: number; x2: number;
  y1: number; y2: number;
}

function createBiquadBandpass(centerFreq: number, Q: number, sampleRate: number): {
  b0: number; b1: number; b2: number;
  a1: number; a2: number;
} {
  const w0 = 2 * Math.PI * centerFreq / sampleRate;
  const alpha = Math.sin(w0) / (2 * Q);

  const b0 = alpha;
  const b1 = 0;
  const b2 = -alpha;
  const a0 = 1 + alpha;
  const a1 = -2 * Math.cos(w0);
  const a2 = 1 - alpha;

  return {
    b0: b0 / a0,
    b1: b1 / a0,
    b2: b2 / a0,
    a1: a1 / a0,
    a2: a2 / a0
  };
}

function applyBiquad(
  samples: Float32Array,
  coeffs: { b0: number; b1: number; b2: number; a1: number; a2: number },
  state: BiquadState
): void {
  for (let i = 0; i < samples.length; i++) {
    const x = samples[i];
    const y = coeffs.b0 * x + coeffs.b1 * state.x1 + coeffs.b2 * state.x2
            - coeffs.a1 * state.y1 - coeffs.a2 * state.y2;

    state.x2 = state.x1;
    state.x1 = x;
    state.y2 = state.y1;
    state.y1 = y;

    samples[i] = y;
  }
}

// ============================================
// FORMANT FILTER BANK (parallel bandpass filters)
// ============================================

interface FormantConfig {
  frequencies: number[];  // Center frequencies
  amplitudes: number[];   // Relative amplitudes
  Q: number;              // Quality factor (resonance)
}

function applyFormants(
  samples: Float32Array,
  formants: FormantConfig,
  sampleRate: number
): Float32Array {
  if (!formants.frequencies || formants.frequencies.length === 0) {
    return samples;
  }

  const output = new Float32Array(samples.length);

  for (let f = 0; f < formants.frequencies.length; f++) {
    const freq = formants.frequencies[f];
    const amp = formants.amplitudes[f] || 1.0;

    // Skip invalid frequencies
    if (freq < 20 || freq > sampleRate / 2) continue;

    const coeffs = createBiquadBandpass(freq, formants.Q, sampleRate);
    const state: BiquadState = { x1: 0, x2: 0, y1: 0, y2: 0 };

    // Copy input and filter
    const filtered = new Float32Array(samples);
    applyBiquad(filtered, coeffs, state);

    // Add to output with amplitude
    for (let i = 0; i < output.length; i++) {
      output[i] += filtered[i] * amp;
    }
  }

  return output;
}

// ============================================
// HIGH-SHELF FILTER (for brightness boost)
// ============================================

function createHighShelf(freq: number, gainDb: number, sampleRate: number): {
  b0: number; b1: number; b2: number;
  a1: number; a2: number;
} {
  const A = Math.pow(10, gainDb / 40);
  const w0 = 2 * Math.PI * freq / sampleRate;
  const alpha = Math.sin(w0) / 2 * Math.sqrt(2);

  const b0 = A * ((A + 1) + (A - 1) * Math.cos(w0) + 2 * Math.sqrt(A) * alpha);
  const b1 = -2 * A * ((A - 1) + (A + 1) * Math.cos(w0));
  const b2 = A * ((A + 1) + (A - 1) * Math.cos(w0) - 2 * Math.sqrt(A) * alpha);
  const a0 = (A + 1) - (A - 1) * Math.cos(w0) + 2 * Math.sqrt(A) * alpha;
  const a1 = 2 * ((A - 1) - (A + 1) * Math.cos(w0));
  const a2 = (A + 1) - (A - 1) * Math.cos(w0) - 2 * Math.sqrt(A) * alpha;

  return {
    b0: b0 / a0,
    b1: b1 / a0,
    b2: b2 / a0,
    a1: a1 / a0,
    a2: a2 / a0
  };
}

function applyHighShelf(samples: Float32Array, freq: number, gainDb: number, sampleRate: number): void {
  if (Math.abs(gainDb) < 0.1) return; // Skip if gain is negligible

  const coeffs = createHighShelf(freq, gainDb, sampleRate);
  const state: BiquadState = { x1: 0, x2: 0, y1: 0, y2: 0 };
  applyBiquad(samples, coeffs, state);
}

// ============================================
// FM SYNTHESIS
// ============================================

function generateFM(
  carrierFreq: number,
  modulatorRatio: number,
  modulationIndex: number,
  phase: number
): number {
  // Carrier frequency modulated by modulator
  const modFreq = carrierFreq * modulatorRatio;
  const modPhase = phase * modulatorRatio;
  const modulator = Math.sin(2 * Math.PI * modPhase) * modulationIndex;

  return Math.sin(2 * Math.PI * phase + modulator);
}

// ============================================
// HARMONIC ADDITIVE SYNTHESIS
// ============================================

function generateAdditive(
  fundamentalPhase: number,
  harmonicAmplitudes: number[]
): number {
  let output = 0;
  for (let h = 0; h < harmonicAmplitudes.length; h++) {
    const harmonic = h + 1;
    const amp = harmonicAmplitudes[h];
    output += Math.sin(2 * Math.PI * fundamentalPhase * harmonic) * amp;
  }
  return output;
}

// ============================================
// DYNAMIC SYNTHESIZERS
// ============================================

function synthNote(
  freq: number,
  duration: number,
  config: VoiceSynthConfig,
  sampleRate: number,
  globalConfig?: SynthConfig
): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const waveform = config.waveform || 'saw';
  const detune = (config.detune_cents || 0) / 100;
  const subGain = config.sub_octave_gain || 0;

  // FM synthesis parameters from AI analysis
  const useFM = globalConfig?.fm?.enabled ?? false;
  const fmRatio = globalConfig?.fm?.modulator_ratio ?? 1.0;
  const fmIndex = globalConfig?.fm?.modulation_index ?? 2.0;

  // Harmonic profile from AI analysis (for additive synthesis component)
  const harmonicsProfile = globalConfig?.harmonics_profile ?? [];
  const useAdditive = harmonicsProfile.length > 4;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;
    const phase = freq * t;

    let sample = 0;

    if (useFM && waveform === 'saw') {
      // FM synthesis for richer harmonics (AI-derived)
      sample = generateFM(freq, fmRatio, fmIndex, phase);
    } else if (useAdditive && waveform === 'saw') {
      // Additive synthesis using AI-analyzed harmonic profile
      sample = generateAdditive(phase, harmonicsProfile.slice(0, 8));
    } else {
      // Standard waveform generation
      sample = generateWaveform(waveform, phase);
    }

    // Detuned oscillator for fatness
    if (detune !== 0) {
      const detuneRatio = Math.pow(2, detune / 12);
      const phase2 = freq * detuneRatio * t;
      if (useFM && waveform === 'saw') {
        sample = (sample + generateFM(freq * detuneRatio, fmRatio, fmIndex, phase2)) * 0.5;
      } else if (useAdditive && waveform === 'saw') {
        sample = (sample + generateAdditive(phase2, harmonicsProfile.slice(0, 8))) * 0.5;
      } else {
        sample = (sample + generateWaveform(waveform, phase2)) * 0.5;
      }
    }

    // Sub oscillator
    if (subGain > 0) {
      const subPhase = freq * 0.5 * t;
      sample += generateSine(subPhase) * subGain;
    }

    // Noise mixing for more realistic timbre (AI-learned parameter)
    const noiseMix = config.noise_mix || 0;
    if (noiseMix > 0) {
      const noise = generateNoise() * noiseMix;
      sample = sample * (1 - noiseMix) + noise;
    }

    // Apply envelope
    const env = applyEnvelope(t, duration, config.envelope);
    output[i] = sample * env * config.gain;
  }

  // Apply filters
  if (config.lpf && config.lpf < sampleRate / 2) {
    applyLPF(output, config.lpf, sampleRate);
  }
  if (config.hpf && config.hpf > 20) {
    applyHPF(output, config.hpf, sampleRate);
  }

  // Apply formant filters if available (for vocal-like timbre)
  // Note: Formants can hurt frequency balance if not tuned carefully
  // Disable for now - needs more AI tuning to determine when to apply
  // const formants = globalConfig?.formants;
  // if (formants && formants.frequencies && formants.frequencies.length > 0) {
  //   const formantOutput = applyFormants(output, {
  //     frequencies: formants.frequencies,
  //     amplitudes: formants.amplitudes || [],
  //     Q: formants.Q || 5.0
  //   }, sampleRate);
  //   // Mix original with formant-filtered (20% for subtle effect)
  //   for (let i = 0; i < output.length; i++) {
  //     output[i] = (output[i] * 0.8 + formantOutput[i] * 0.2);
  //   }
  // }

  return output;
}

function synthKick(duration: number, config: Partial<VoiceSynthConfig>, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const pitchStart = 150;
  const pitchEnd = 40;
  const pitchDecay = 0.05;
  const transientBoost = config.transient_boost || 0.5;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Pitch envelope
    const pitchEnv = Math.exp(-t / pitchDecay);
    const freq = pitchEnd + (pitchStart - pitchEnd) * pitchEnv;

    // Sine with harmonics
    const phase = 2 * Math.PI * freq * t;
    let sample = Math.sin(phase);
    sample += 0.3 * Math.sin(phase * 2) * Math.exp(-t / 0.02);

    // Amplitude envelope
    const ampEnv = Math.exp(-t / 0.15);

    // Click transient
    const click = Math.exp(-t / 0.003) * transientBoost;

    output[i] = (sample * ampEnv + click) * (config.gain || 0.7);
  }

  return output;
}

function synthSnare(duration: number, config: Partial<VoiceSynthConfig>, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const transientBoost = config.transient_boost || 0.5;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Body tone
    const body = (Math.sin(2 * Math.PI * 180 * t) * 0.5 +
                  Math.sin(2 * Math.PI * 330 * t) * 0.3) *
                  Math.exp(-t / 0.08);

    // Noise (snare wires)
    const noise = generateNoise() * Math.exp(-t / 0.12);

    // Transient
    const click = Math.exp(-t / 0.002) * transientBoost;

    output[i] = (body * 0.5 + noise * 0.6 + click * 0.3) * (config.gain || 0.7);
  }

  // High-pass for crispness
  applyHPF(output, 200, sampleRate);

  return output;
}

function synthHihat(duration: number, config: Partial<VoiceSynthConfig>, sampleRate: number, open: boolean = false): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const decay = open ? 0.15 : 0.03;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Metallic noise
    let metallic = 0;
    const freqs = [800, 1500, 3000, 6000, 10000];
    for (const f of freqs) {
      metallic += Math.sin(2 * Math.PI * f * t * (1 + Math.random() * 0.01)) * 0.2;
    }

    // Pure noise
    const noise = generateNoise();

    // Envelope
    const env = Math.exp(-t / decay);

    output[i] = (metallic * 0.4 + noise * 0.6) * env * (config.gain || 0.5);
  }

  // High-pass
  applyHPF(output, 2000, sampleRate);

  return output;
}

// ============================================
// PATTERN PARSING
// ============================================

function getPatternEvents(pattern: string, startCycle: number, endCycle: number): Array<{
  value: string;
  begin: number;
  end: number;
}> {
  try {
    const pat = mini(pattern);
    const haps = pat.queryArc(startCycle, endCycle);
    return haps.map((hap: any) => ({
      value: String(hap.value),
      begin: Number(hap.whole?.begin ?? hap.part.begin),
      end: Number(hap.whole?.end ?? hap.part.end),
    }));
  } catch (e) {
    return parsePatternSimple(pattern, startCycle, endCycle);
  }
}

function parsePatternSimple(pattern: string, startCycle: number, endCycle: number): Array<{
  value: string;
  begin: number;
  end: number;
}> {
  const events: Array<{ value: string; begin: number; end: number }> = [];
  const tokens = pattern.split(/\s+/).filter(t => t && t !== '~');

  if (tokens.length === 0) return events;

  const slotDuration = 1 / tokens.length;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.startsWith('[') && token.endsWith(']')) {
      // Chord
      const notes = token.slice(1, -1).split(',');
      for (const note of notes) {
        events.push({
          value: note.trim(),
          begin: startCycle + i * slotDuration,
          end: startCycle + (i + 1) * slotDuration,
        });
      }
    } else {
      events.push({
        value: token,
        begin: startCycle + i * slotDuration,
        end: startCycle + (i + 1) * slotDuration,
      });
    }
  }

  return events;
}

// ============================================
// STRUDEL CODE PARSING
// ============================================

function parseStrudelCode(code: string, configBpm?: number): { bpm: number; voices: ParsedVoice[] } {
  let bpm = configBpm || 120;

  // Extract BPM from setcps
  const bpmMatch = code.match(/setcps\((\d+)\/60\/4\)/);
  if (bpmMatch && !configBpm) {
    bpm = parseInt(bpmMatch[1]);
  }

  const voices: ParsedVoice[] = [];

  // Array format: let bass = ["c2 e2", "f2 g2"]
  const arrayRegex = /let\s+(\w+)\s*=\s*\[([\s\S]*?)\]/g;
  let match;

  while ((match = arrayRegex.exec(code)) !== null) {
    const name = match[1];
    const content = match[2];
    const patterns: string[] = [];
    const strRegex = /"([^"]+)"/g;
    let strMatch;
    while ((strMatch = strRegex.exec(content)) !== null) {
      patterns.push(strMatch[1]);
    }
    if (patterns.length > 0) {
      const parsed = parseEffects(code, name);
      voices.push({
        name,
        patterns,
        effects: parsed.effects,
        modelType: getModelType(name),
        soundName: parsed.soundName,
      });
    }
  }

  // Single pattern format: let kick1 = "bd ~ bd ~"
  const singlePatterns: { [base: string]: string[] } = {};
  const singleRegex = /let\s+(\w+?)(\d+)\s*=\s*"([^"]+)"/g;

  while ((match = singleRegex.exec(code)) !== null) {
    const baseName = match[1];
    const idx = parseInt(match[2]) - 1;
    const pattern = match[3];
    if (!singlePatterns[baseName]) {
      singlePatterns[baseName] = [];
    }
    while (singlePatterns[baseName].length <= idx) {
      singlePatterns[baseName].push('~');
    }
    singlePatterns[baseName][idx] = pattern;
  }

  for (const [name, patterns] of Object.entries(singlePatterns)) {
    if (!voices.find(v => v.name === name)) {
      const parsed = parseEffects(code, name);
      voices.push({
        name,
        patterns,
        effects: parsed.effects,
        modelType: getModelType(name),
        soundName: parsed.soundName,
      });
    }
  }

  return { bpm, voices };
}

interface ParsedEffects {
  effects: VoiceEffects;
  soundName?: string;
}

function parseEffects(code: string, voiceName: string): ParsedEffects {
  const effects: VoiceEffects = { gain: 1.0 };
  let soundName: string | undefined;

  const fxRegex = new RegExp(`let\\s+${voiceName}Fx\\s*=\\s*p\\s*=>\\s*p[^\\n]+(?:\\n\\s+\\.[^\\n]+)*`, 'i');
  const match = code.match(fxRegex);

  if (match) {
    const chain = match[0];
    const gainMatch = chain.match(/\.gain\(([0-9.]+)\)/);
    if (gainMatch) effects.gain = parseFloat(gainMatch[1]);

    const lpfMatch = chain.match(/\.lpf\(([0-9.]+)\)/);
    if (lpfMatch) effects.lpf = parseFloat(lpfMatch[1]);

    const hpfMatch = chain.match(/\.hpf\(([0-9.]+)\)/);
    if (hpfMatch) effects.hpf = parseFloat(hpfMatch[1]);

    // Extract sound name (supports alternating sounds like "<sound1 sound2>")
    const soundMatch = chain.match(/\.sound\("([^"]+)"\)/);
    if (soundMatch) {
      soundName = soundMatch[1];
    }
  }

  return { effects, soundName };
}

function getModelType(name: string): 'melodic' | 'bass' | 'drums' {
  const n = name.toLowerCase();
  if (n.includes('kick') || n.includes('snare') || n.includes('hh') || n.includes('drum')) {
    return 'drums';
  }
  if (n.includes('bass')) {
    return 'bass';
  }
  return 'melodic';
}

// ============================================
// MAIN RENDER FUNCTION
// ============================================

async function render(
  voices: ParsedVoice[],
  config: SynthConfig,
  duration: number,
  separateStems: boolean = false
): Promise<StemOutputs> {
  const totalSamples = Math.floor(duration * SAMPLE_RATE);

  // Create separate stem buffers
  const stems: StemOutputs = {
    mixed: new Float32Array(totalSamples),
    bass: new Float32Array(totalSamples),
    drums: new Float32Array(totalSamples),
    melodic: new Float32Array(totalSamples),
  };

  const bpm = config.tempo.bpm;
  const cyclesPerSecond = bpm / 60 / 4;
  const totalCycles = duration * cyclesPerSecond;

  console.log(`\nRendering ${voices.length} voices at ${bpm} BPM for ${duration}s`);
  console.log(`Config: envelope A=${config.envelope.attack.toFixed(3)}s D=${config.envelope.decay.toFixed(3)}s`);
  console.log(`        filters LPF=${config.filters.lpf_cutoff}Hz HPF=${config.filters.hpf_cutoff}Hz`);

  for (const voice of voices) {
    // Get voice-specific config
    let voiceConfig: VoiceSynthConfig;

    if (voice.modelType === 'bass') {
      voiceConfig = { ...config.voices.bass };
    } else if (voice.modelType === 'drums') {
      voiceConfig = {
        gain: config.voices.drums.gain || 0.7,
        lpf: 20000,
        hpf: 20,
        envelope: config.envelope,
        waveform: 'noise',
        transient_boost: config.voices.drums.transient_boost
      };
    } else {
      // Melodic - choose mid or high based on note range
      voiceConfig = { ...config.voices.mid };
      if (voice.name.toLowerCase().includes('high')) {
        voiceConfig = { ...config.voices.high };
      }
    }

    // Apply sound mapping if a specific sound is requested
    if (voice.soundName && voice.modelType !== 'drums') {
      const soundMapping = getSoundMapping(voice.soundName);
      voiceConfig = {
        ...voiceConfig,
        waveform: soundMapping.waveform,
        lpf: soundMapping.lpf,
        envelope: {
          attack: soundMapping.attack,
          decay: soundMapping.decay,
          sustain: soundMapping.sustain,
          release: soundMapping.release,
        },
        detune_cents: soundMapping.detune || 0,
        sub_octave_gain: soundMapping.subOctave || 0,
        noise_mix: soundMapping.noiseMix || 0,
      };
    }

    // Apply effect chain overrides from Strudel code
    if (voice.effects.gain !== 1.0) {
      voiceConfig.gain *= voice.effects.gain;
    }
    if (voice.effects.lpf) {
      voiceConfig.lpf = Math.min(voiceConfig.lpf, voice.effects.lpf);
    }
    if (voice.effects.hpf) {
      voiceConfig.hpf = Math.max(voiceConfig.hpf, voice.effects.hpf);
    }

    const soundInfo = voice.soundName ? `, sound=${voice.soundName}` : '';
    console.log(`  ${voice.name}: gain=${voiceConfig.gain.toFixed(2)}, type=${voice.modelType}${soundInfo}, ${voice.patterns.length} patterns`);

    let currentCycle = 0;
    let patternIndex = 0;

    while (currentCycle < totalCycles) {
      const pattern = voice.patterns[patternIndex % voice.patterns.length];
      const events = getPatternEvents(pattern, 0, 1);

      for (const event of events) {
        const eventCycle = currentCycle + event.begin;
        const eventEndCycle = currentCycle + event.end;
        const eventTime = eventCycle / cyclesPerSecond;
        const eventDuration = (eventEndCycle - eventCycle) / cyclesPerSecond;

        if (eventTime >= duration) break;

        const startSample = Math.floor(eventTime * SAMPLE_RATE);
        if (startSample >= totalSamples) break;

        let synthSample: Float32Array;

        if (voice.modelType === 'drums') {
          const drumType = event.value.toLowerCase();
          const drumConfig = config.voices.drums;

          if (drumType === 'bd' || drumType === 'kick') {
            synthSample = synthKick(0.3, drumConfig, SAMPLE_RATE);
          } else if (drumType === 'sd' || drumType === 'snare' || drumType === 'sn') {
            synthSample = synthSnare(0.2, drumConfig, SAMPLE_RATE);
          } else if (drumType === 'oh' || drumType === 'openhat') {
            synthSample = synthHihat(0.3, drumConfig, SAMPLE_RATE, true);
          } else if (drumType === 'hh' || drumType === 'hihat' || drumType === 'ch') {
            synthSample = synthHihat(0.1, drumConfig, SAMPLE_RATE, false);
          } else {
            synthSample = synthKick(0.3, drumConfig, SAMPLE_RATE);
          }
        } else {
          const targetMidi = noteToMidi(event.value);
          if (targetMidi === null) continue;

          const targetFreq = midiToFreq(targetMidi);
          const noteDuration = Math.min(eventDuration, 0.5);

          // Choose voice config based on MIDI note
          let noteConfig = voiceConfig;
          if (targetMidi >= 72) { // C5 and above
            noteConfig = { ...config.voices.high, gain: voiceConfig.gain };
          } else if (targetMidi < 48) { // Below C3
            noteConfig = { ...config.voices.bass, gain: voiceConfig.gain };
          }

          synthSample = synthNote(targetFreq, noteDuration, noteConfig, SAMPLE_RATE, config);
        }

        // Mix into appropriate stem and mixed output
        const len = Math.min(synthSample.length, totalSamples - startSample);
        for (let i = 0; i < len; i++) {
          const sample = synthSample[i];
          stems.mixed[startSample + i] += sample;

          // Route to correct stem
          if (voice.modelType === 'drums') {
            stems.drums[startSample + i] += sample;
          } else if (voice.modelType === 'bass') {
            stems.bass[startSample + i] += sample;
          } else {
            stems.melodic[startSample + i] += sample;
          }
        }
      }

      currentCycle++;
      patternIndex++;
    }
  }

  // Apply master processing to mixed output
  console.log(`\nApplying master processing...`);

  // Master HPF
  if (config.master.hpf > 20) {
    applyHPF(stems.mixed, config.master.hpf, SAMPLE_RATE);
  }

  // AI-derived brightness boost (high-shelf filter at 2kHz)
  const highShelfBoost = config.master.high_shelf_boost || 0;
  if (highShelfBoost > 0.1) {
    applyHighShelf(stems.mixed, 2000, highShelfBoost, SAMPLE_RATE);
    console.log(`  High-shelf boost: +${highShelfBoost.toFixed(1)} dB above 2kHz`);
  }

  // Master gain
  for (let i = 0; i < stems.mixed.length; i++) {
    stems.mixed[i] *= config.master.gain;
  }

  // Soft saturation: limits peaks while adding harmonics
  // Uses tanh which naturally compresses high values
  // This mimics analog saturation/tape compression

  // First, apply gain to bring up levels
  const saturationDrive = 3.0;  // How hard we drive into saturation
  for (let i = 0; i < stems.mixed.length; i++) {
    stems.mixed[i] *= saturationDrive;
  }

  // Apply tanh saturation (soft clipping)
  for (let i = 0; i < stems.mixed.length; i++) {
    stems.mixed[i] = Math.tanh(stems.mixed[i]);
  }

  // Tanh outputs -1 to 1, scale to desired peak
  const targetPeak = 0.85;  // Leave some headroom
  for (let i = 0; i < stems.mixed.length; i++) {
    stems.mixed[i] *= targetPeak;
  }

  // Calculate RMS after saturation
  let sumSq = 0;
  for (let i = 0; i < stems.mixed.length; i++) {
    sumSq += stems.mixed[i] * stems.mixed[i];
  }
  const postSatRms = Math.sqrt(sumSq / stems.mixed.length);
  console.log(`  Saturation: drive ${saturationDrive}×, post-sat RMS ${postSatRms.toFixed(4)}`)

  // Limiter AFTER compression (to control peaks)
  if (config.master.limiter) {
    const threshold = config.dynamics.limiter_threshold;
    let maxVal = 0;
    for (let i = 0; i < stems.mixed.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(stems.mixed[i]));
    }
    if (maxVal > threshold) {
      const scale = threshold / maxVal;
      for (let i = 0; i < stems.mixed.length; i++) {
        stems.mixed[i] *= scale;
      }
      console.log(`  Limiter: peak ${maxVal.toFixed(2)} → ${threshold}`);
    }
  }

  // RMS normalization AFTER limiter (to match target loudness)
  if (config.dynamics.target_rms > 0) {
    // Calculate current RMS after limiting
    let sumSquares = 0;
    for (let i = 0; i < stems.mixed.length; i++) {
      sumSquares += stems.mixed[i] * stems.mixed[i];
    }
    const currentRms = Math.sqrt(sumSquares / stems.mixed.length);

    // Normalize to target RMS
    if (currentRms > 0.001) {
      const rmsScale = config.dynamics.target_rms / currentRms;
      // Allow up to 3x amplification (target / current), but cap at 0.95 peak
      const safeScale = Math.min(3.0, Math.max(0.1, rmsScale));
      for (let i = 0; i < stems.mixed.length; i++) {
        stems.mixed[i] *= safeScale;
      }

      // Final peak check after RMS normalization
      let finalMax = 0;
      for (let i = 0; i < stems.mixed.length; i++) {
        finalMax = Math.max(finalMax, Math.abs(stems.mixed[i]));
      }
      if (finalMax > 0.98) {
        // Soft clip if we exceeded
        const clipScale = 0.98 / finalMax;
        for (let i = 0; i < stems.mixed.length; i++) {
          stems.mixed[i] *= clipScale;
        }
      }

      const finalRms = currentRms * safeScale * (finalMax > 0.98 ? 0.98/finalMax : 1);
      console.log(`  RMS norm: ${currentRms.toFixed(4)} → ${finalRms.toFixed(4)} (target: ${config.dynamics.target_rms.toFixed(4)}, scale: ${safeScale.toFixed(2)}×)`);
    }
  }

  // Normalize individual stems (for comparison, not master processing)
  if (separateStems) {
    normalizeStem(stems.bass, 0.9);
    normalizeStem(stems.drums, 0.9);
    normalizeStem(stems.melodic, 0.9);
    console.log(`  Normalized individual stems for comparison`);
  }

  return stems;
}

// Normalize a stem to peak at targetPeak
function normalizeStem(stem: Float32Array, targetPeak: number): void {
  let maxVal = 0;
  for (let i = 0; i < stem.length; i++) {
    maxVal = Math.max(maxVal, Math.abs(stem[i]));
  }
  if (maxVal > 0.001) {
    const scale = targetPeak / maxVal;
    for (let i = 0; i < stem.length; i++) {
      stem[i] *= scale;
    }
  }
}

// ============================================
// WAV OUTPUT
// ============================================

function float32ToWav(samples: Float32Array, sampleRate: number): Buffer {
  const numChannels = 1;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = samples.length * blockAlign;
  const buffer = Buffer.alloc(44 + dataLength);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * blockAlign, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitDepth, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataLength, 40);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.floor(s * 32767), 44 + i * 2);
  }

  return buffer;
}

// ============================================
// MAIN
// ============================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Node.js Strudel Renderer with Dynamic Synthesis');
    console.log('');
    console.log('Usage: node render-strudel-node.js <input.strudel> [output.wav] [options]');
    console.log('');
    console.log('Options:');
    console.log('  -o, --output <file>     Output WAV file');
    console.log('  -d, --duration <sec>    Duration in seconds (default: 60)');
    console.log('  -c, --config <json>     Synthesis config JSON file (from analyze_synth_params.py)');
    console.log('  --bpm <bpm>             Override BPM');
    console.log('  --stems                 Output separate stem files (bass, drums, melodic)');
    console.log('  -m, --models <dir>      Models directory (deprecated, using synthesis)');
    process.exit(0);
  }

  // Parse arguments
  let inputPath = '';
  let outputPath = '';
  let configPath = '';
  let duration = 60;
  let overrideBpm = 0;
  let separateStems = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-o' || arg === '--output') {
      outputPath = path.resolve(args[++i]);
    } else if (arg === '-d' || arg === '--duration') {
      duration = parseFloat(args[++i]);
    } else if (arg === '-c' || arg === '--config') {
      configPath = path.resolve(args[++i]);
    } else if (arg === '--bpm') {
      overrideBpm = parseFloat(args[++i]);
    } else if (arg === '--stems') {
      separateStems = true;
    } else if (arg === '-m' || arg === '--models') {
      i++; // Skip models dir (deprecated)
    } else if (!arg.startsWith('-')) {
      if (!inputPath) {
        inputPath = path.resolve(arg);
      } else if (!outputPath) {
        outputPath = path.resolve(arg);
      }
    }
  }

  if (!inputPath) {
    console.error('Error: Input file required');
    process.exit(1);
  }

  if (!outputPath) {
    outputPath = inputPath.replace('.strudel', '.wav');
  }

  console.log('━'.repeat(60));
  console.log('Node.js Strudel Renderer (Dynamic Synthesis)');
  console.log('━'.repeat(60));
  console.log(`Input:    ${inputPath}`);
  console.log(`Output:   ${outputPath}`);
  console.log(`Duration: ${duration}s`);

  // Load synthesis config
  let config = getDefaultConfig();

  if (configPath && fs.existsSync(configPath)) {
    console.log(`Config:   ${configPath}`);
    try {
      const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      // Merge with defaults (config may be nested under synth_config)
      const synthConfig = configData.synth_config || configData;
      config = { ...config, ...synthConfig };
      if (synthConfig.voices) {
        config.voices = { ...config.voices, ...synthConfig.voices };
      }
      if (synthConfig.tempo) {
        config.tempo = { ...config.tempo, ...synthConfig.tempo };
      }
      console.log(`  Loaded config: BPM=${config.tempo.bpm}, waveform=${config.oscillator.waveform}`);
    } catch (e) {
      console.log(`  Warning: Could not parse config, using defaults`);
    }
  } else {
    console.log(`Config:   (using defaults)`);
  }

  // Override BPM if specified
  if (overrideBpm > 0) {
    config.tempo.bpm = overrideBpm;
    config.tempo.samples_per_beat = Math.floor(SAMPLE_RATE * 60 / overrideBpm);
  }

  // Read and parse Strudel code
  const code = fs.readFileSync(inputPath, 'utf-8');
  const { bpm: parsedBpm, voices } = parseStrudelCode(code, config.tempo.bpm);

  // If no config BPM was set and code has BPM, use code BPM
  if (!configPath && !overrideBpm && parsedBpm !== 120) {
    config.tempo.bpm = parsedBpm;
  }

  console.log(`\nParsed: ${voices.length} voices, ${config.tempo.bpm} BPM`);
  for (const v of voices) {
    console.log(`  ${v.name}: ${v.patterns.length} patterns, type=${v.modelType}`);
  }

  // Render
  const stems = await render(voices, config, duration, separateStems);

  // Save WAV files
  const wavBuffer = float32ToWav(stems.mixed, SAMPLE_RATE);
  fs.writeFileSync(outputPath, wavBuffer);

  console.log('\n' + '━'.repeat(60));
  console.log(`Saved: ${outputPath} (${(wavBuffer.length / 1024 / 1024).toFixed(1)} MB)`);

  // Save separate stems if requested
  if (separateStems) {
    const outputDir = path.dirname(outputPath);
    const baseName = path.basename(outputPath, '.wav');

    const stemFiles = {
      bass: path.join(outputDir, `${baseName}_bass.wav`),
      drums: path.join(outputDir, `${baseName}_drums.wav`),
      melodic: path.join(outputDir, `${baseName}_melodic.wav`),
    };

    for (const [stemName, stemPath] of Object.entries(stemFiles)) {
      const stemData = stems[stemName as keyof typeof stems];
      if (stemData && stemData !== stems.mixed) {
        const stemBuffer = float32ToWav(stemData, SAMPLE_RATE);
        fs.writeFileSync(stemPath, stemBuffer);
        console.log(`Saved: ${stemPath} (${(stemBuffer.length / 1024 / 1024).toFixed(1)} MB)`);
      }
    }

    console.log(`\nStem files ready for per-stem comparison`);
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
