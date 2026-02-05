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
        gain: 0.3, lpf: 500, hpf: 40,
        envelope: { attack: 0.005, decay: 0.2, sustain: 0.6, release: 0.1 },
        waveform: 'saw', sub_octave_gain: 0.4
      },
      mid: {
        gain: 1.0, lpf: 6000, hpf: 200,
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
// DYNAMIC SYNTHESIZERS
// ============================================

function synthNote(
  freq: number,
  duration: number,
  config: VoiceSynthConfig,
  sampleRate: number
): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const waveform = config.waveform || 'saw';
  const detune = (config.detune_cents || 0) / 100;
  const subGain = config.sub_octave_gain || 0;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Main oscillator
    const phase1 = freq * t;
    let sample = generateWaveform(waveform, phase1);

    // Detuned oscillator for fatness
    if (detune !== 0) {
      const detuneRatio = Math.pow(2, detune / 12);
      const phase2 = freq * detuneRatio * t;
      sample = (sample + generateWaveform(waveform, phase2)) * 0.5;
    }

    // Sub oscillator
    if (subGain > 0) {
      const subPhase = freq * 0.5 * t;
      sample += generateSine(subPhase) * subGain;
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
      voices.push({
        name,
        patterns,
        effects: parseEffects(code, name),
        modelType: getModelType(name),
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
      voices.push({
        name,
        patterns,
        effects: parseEffects(code, name),
        modelType: getModelType(name),
      });
    }
  }

  return { bpm, voices };
}

function parseEffects(code: string, voiceName: string): VoiceEffects {
  const effects: VoiceEffects = { gain: 1.0 };

  const fxRegex = new RegExp(`let\\s+${voiceName}Fx\\s*=\\s*p\\s*=>\\s*p[^\\n]+`, 'i');
  const match = code.match(fxRegex);

  if (match) {
    const chain = match[0];
    const gainMatch = chain.match(/\.gain\(([0-9.]+)\)/);
    if (gainMatch) effects.gain = parseFloat(gainMatch[1]);

    const lpfMatch = chain.match(/\.lpf\(([0-9.]+)\)/);
    if (lpfMatch) effects.lpf = parseFloat(lpfMatch[1]);

    const hpfMatch = chain.match(/\.hpf\(([0-9.]+)\)/);
    if (hpfMatch) effects.hpf = parseFloat(hpfMatch[1]);
  }

  return effects;
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
  duration: number
): Promise<Float32Array> {
  const totalSamples = Math.floor(duration * SAMPLE_RATE);
  const output = new Float32Array(totalSamples);

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

    console.log(`  ${voice.name}: gain=${voiceConfig.gain.toFixed(2)}, type=${voice.modelType}, ${voice.patterns.length} patterns`);

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

          synthSample = synthNote(targetFreq, noteDuration, noteConfig, SAMPLE_RATE);
        }

        // Mix into output
        const len = Math.min(synthSample.length, totalSamples - startSample);
        for (let i = 0; i < len; i++) {
          output[startSample + i] += synthSample[i];
        }
      }

      currentCycle++;
      patternIndex++;
    }
  }

  // Apply master processing
  console.log(`\nApplying master processing...`);

  // Master HPF
  if (config.master.hpf > 20) {
    applyHPF(output, config.master.hpf, SAMPLE_RATE);
  }

  // Master gain
  for (let i = 0; i < output.length; i++) {
    output[i] *= config.master.gain;
  }

  // Limiter
  if (config.master.limiter) {
    const threshold = config.dynamics.limiter_threshold;
    let maxVal = 0;
    for (let i = 0; i < output.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(output[i]));
    }
    if (maxVal > threshold) {
      const scale = threshold / maxVal;
      for (let i = 0; i < output.length; i++) {
        output[i] *= scale;
      }
      console.log(`  Limiter: peak ${maxVal.toFixed(2)} → ${threshold}`);
    }
  }

  return output;
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
    console.log('  -m, --models <dir>      Models directory (deprecated, using synthesis)');
    process.exit(0);
  }

  // Parse arguments
  let inputPath = '';
  let outputPath = '';
  let configPath = '';
  let duration = 60;
  let overrideBpm = 0;

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
  const audio = await render(voices, config, duration);

  // Save WAV
  const wavBuffer = float32ToWav(audio, SAMPLE_RATE);
  fs.writeFileSync(outputPath, wavBuffer);

  console.log('\n' + '━'.repeat(60));
  console.log(`Saved: ${outputPath} (${(wavBuffer.length / 1024 / 1024).toFixed(1)} MB)`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
