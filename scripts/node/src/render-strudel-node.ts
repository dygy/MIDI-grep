#!/usr/bin/env node
/**
 * Node.js Strudel Renderer
 *
 * Uses Strudel's mini notation parser for accurate pattern parsing
 * Synthesizes audio using 808 drums, saw bass, and synth leads
 */

// @ts-ignore - Strudel types not available
import { mini } from '@strudel/mini';
import fs from 'fs';
import path from 'path';

const SAMPLE_RATE = 44100;

// Note name to MIDI mapping
const NOTE_TO_MIDI: { [key: string]: number } = {};
const NOTES = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'];
for (let octave = 0; octave <= 8; octave++) {
  NOTES.forEach((note, i) => {
    NOTE_TO_MIDI[`${note}${octave}`] = 12 + octave * 12 + i;
  });
}

// MIDI to frequency
function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

// Parse note name to MIDI
function noteToMidi(note: string): number | null {
  const normalized = note.toLowerCase().replace('#', 's');
  return NOTE_TO_MIDI[normalized] ?? null;
}

interface GranularModel {
  name: string;
  samples: Map<string, Float32Array>; // note name -> audio data
  drumSamples?: Map<string, Float32Array[]>; // bd/sd/hh -> array of samples
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

/**
 * Load a WAV file and return Float32Array of samples
 */
function loadWav(filePath: string): Float32Array | null {
  try {
    const buffer = fs.readFileSync(filePath);
    // Simple WAV parser - assumes 16-bit PCM
    const dataStart = buffer.indexOf('data') + 8;
    const samples = new Float32Array((buffer.length - dataStart) / 2);
    for (let i = 0; i < samples.length; i++) {
      const int16 = buffer.readInt16LE(dataStart + i * 2);
      samples[i] = int16 / 32768;
    }
    return samples;
  } catch {
    return null;
  }
}

/**
 * Load granular model from models directory
 */
function loadGranularModel(modelsDir: string, modelName: string): GranularModel | null {
  const modelPath = path.join(modelsDir, modelName);
  const pitchedDir = path.join(modelPath, 'pitched');
  const grainsDir = path.join(modelPath, 'grains');
  const metadataPath = path.join(modelPath, 'metadata.json');

  const samples = new Map<string, Float32Array>();
  let drumSamples: Map<string, Float32Array[]> | undefined;

  // Load pitched samples for melodic/bass
  if (fs.existsSync(pitchedDir)) {
    const files = fs.readdirSync(pitchedDir).filter(f => f.endsWith('.wav'));
    for (const file of files) {
      const noteName = file.replace('.wav', '');
      const audioData = loadWav(path.join(pitchedDir, file));
      if (audioData) {
        samples.set(noteName, audioData);
      }
    }
  }

  // For drums: load grains and classify by frequency
  if (modelName === 'drums' && fs.existsSync(grainsDir) && fs.existsSync(metadataPath)) {
    try {
      const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
      drumSamples = new Map<string, Float32Array[]>();
      drumSamples.set('bd', []);  // kick: low frequency
      drumSamples.set('sd', []);  // snare: mid frequency
      drumSamples.set('hh', []);  // hi-hat: high frequency
      drumSamples.set('oh', []);  // open hi-hat

      for (const grain of metadata.grains || []) {
        const grainPath = path.join(grainsDir, grain.file);
        const audioData = loadWav(grainPath);
        if (audioData) {
          const midi = grain.midi_note || 60;
          // Classify by pitch: kick < 50, snare 50-65, hi-hat > 65
          if (midi < 50) {
            drumSamples.get('bd')!.push(audioData);
          } else if (midi < 65) {
            drumSamples.get('sd')!.push(audioData);
          } else {
            drumSamples.get('hh')!.push(audioData);
            drumSamples.get('oh')!.push(audioData);
          }
        }
      }

      const bdCount = drumSamples.get('bd')!.length;
      const sdCount = drumSamples.get('sd')!.length;
      const hhCount = drumSamples.get('hh')!.length;
      console.log(`  Loaded ${modelName}: ${bdCount} kicks, ${sdCount} snares, ${hhCount} hi-hats`);
    } catch (e) {
      console.log(`  Warning: Could not load drum grains: ${e}`);
    }
  } else if (samples.size > 0) {
    console.log(`  Loaded ${modelName}: ${samples.size} pitched samples`);
  }

  if (samples.size === 0 && !drumSamples) {
    return null;
  }

  return { name: modelName, samples, drumSamples, sampleRate: SAMPLE_RATE };
}

/**
 * Parse effect function from Strudel code
 */
function parseEffects(code: string, voiceName: string): VoiceEffects {
  const effects: VoiceEffects = { gain: 1.0 };

  // Match: let bassFx = p => p.sound("...").gain(0.8).lpf(800)
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

/**
 * Parse Strudel code to extract voices and patterns
 */
function parseStrudelCode(code: string): { bpm: number; voices: ParsedVoice[] } {
  let bpm = 120;

  // Extract BPM from setcps
  const bpmMatch = code.match(/setcps\((\d+)\/60\/4\)/);
  if (bpmMatch) {
    bpm = parseInt(bpmMatch[1]);
  }

  const voices: ParsedVoice[] = [];

  // Pattern 1: Array format - let bass = ["c2 e2", "f2 g2"]
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
      const modelType = getModelType(name);
      voices.push({
        name,
        patterns,
        effects: parseEffects(code, name),
        modelType,
      });
    }
  }

  // Pattern 2: Single pattern format (Brazilian funk) - let kick1 = "bd ~ bd ~"
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
      const modelType = getModelType(name);
      voices.push({
        name,
        patterns,
        effects: parseEffects(code, name),
        modelType,
      });
    }
  }

  return { bpm, voices };
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

/**
 * Get events from a mini notation pattern using Strudel's parser
 */
function getPatternEvents(pattern: string, startCycle: number, endCycle: number): Array<{
  value: string;
  begin: number;
  end: number;
}> {
  try {
    // Use Strudel's mini notation parser
    const pat = mini(pattern);
    const haps = pat.queryArc(startCycle, endCycle);
    return haps.map((hap: any) => ({
      value: String(hap.value),
      begin: Number(hap.whole?.begin ?? hap.part.begin),
      end: Number(hap.whole?.end ?? hap.part.end),
    }));
  } catch (e) {
    // Fallback to simple parser if Strudel fails
    return parsePatternSimple(pattern, startCycle, endCycle);
  }
}

/**
 * Simple fallback pattern parser
 */
function parsePatternSimple(pattern: string, startCycle: number, endCycle: number): Array<{
  value: string;
  begin: number;
  end: number;
}> {
  const events: Array<{ value: string; begin: number; end: number }> = [];
  const tokens = pattern.split(/\s+/).filter(t => t);
  const totalSlots = tokens.reduce((sum, t) => {
    if (t.startsWith('~*')) return sum + (parseInt(t.slice(2)) || 1);
    return sum + 1;
  }, 0) || 1;

  let slot = 0;
  for (const token of tokens) {
    if (token === '~') {
      slot++;
    } else if (token.startsWith('~*')) {
      slot += parseInt(token.slice(2)) || 1;
    } else if (token.startsWith('[') && token.endsWith(']')) {
      // Chord
      const notes = token.slice(1, -1).split(',');
      for (const note of notes) {
        events.push({
          value: note.trim(),
          begin: startCycle + slot / totalSlots,
          end: startCycle + (slot + 1) / totalSlots,
        });
      }
      slot++;
    } else {
      events.push({
        value: token,
        begin: startCycle + slot / totalSlots,
        end: startCycle + (slot + 1) / totalSlots,
      });
      slot++;
    }
  }
  return events;
}

// ============================================
// SYNTHESIS ENGINE - 808 drums, bass, synths
// ============================================

/**
 * Synthesize 808-style kick drum
 */
function synthKick(duration: number, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const pitchStart = 150;  // Start frequency
  const pitchEnd = 40;     // End frequency
  const pitchDecay = 0.05; // Pitch envelope decay time

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Pitch envelope (exponential decay)
    const pitchEnv = Math.exp(-t / pitchDecay);
    const freq = pitchEnd + (pitchStart - pitchEnd) * pitchEnv;

    // Sine oscillator with pitch envelope
    const phase = 2 * Math.PI * freq * t;
    let sample = Math.sin(phase);

    // Add some harmonics for punch
    sample += 0.3 * Math.sin(phase * 2) * Math.exp(-t / 0.02);

    // Amplitude envelope
    const ampDecay = 0.15;
    const ampEnv = Math.exp(-t / ampDecay);

    // Click transient
    const click = Math.exp(-t / 0.003) * 0.5;

    output[i] = (sample * ampEnv + click) * 0.8;
  }

  return output;
}

/**
 * Synthesize snare drum
 */
function synthSnare(duration: number, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Body tone (two sine waves)
    const bodyFreq1 = 180;
    const bodyFreq2 = 330;
    const body = (Math.sin(2 * Math.PI * bodyFreq1 * t) * 0.5 +
                  Math.sin(2 * Math.PI * bodyFreq2 * t) * 0.3) *
                  Math.exp(-t / 0.08);

    // Noise (snare wires)
    const noise = (Math.random() * 2 - 1) * Math.exp(-t / 0.12);

    // High-pass the noise a bit (simple differencing)
    output[i] = body * 0.6 + noise * 0.7;
  }

  // Simple high-pass filter for noise crispness
  for (let i = len - 1; i > 0; i--) {
    output[i] = output[i] * 0.7 + (output[i] - output[i-1]) * 0.3;
  }

  return output;
}

/**
 * Synthesize hi-hat
 */
function synthHihat(duration: number, sampleRate: number, open: boolean = false): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const decay = open ? 0.15 : 0.03;

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Metallic noise (multiple square waves at inharmonic frequencies)
    let metallic = 0;
    const freqs = [800, 1500, 3000, 6000, 10000];
    for (const f of freqs) {
      metallic += Math.sin(2 * Math.PI * f * t * (1 + Math.random() * 0.01)) * 0.2;
    }

    // Noise component
    const noise = (Math.random() * 2 - 1);

    // Mix and envelope
    const env = Math.exp(-t / decay);
    output[i] = (metallic * 0.4 + noise * 0.6) * env * 0.5;
  }

  return output;
}

/**
 * Synthesize bass note (saw + sub)
 */
function synthBass(freq: number, duration: number, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;
    const phase = freq * t;

    // Sawtooth wave
    const saw = 2 * (phase % 1) - 1;

    // Sub oscillator (one octave down, sine)
    const sub = Math.sin(2 * Math.PI * freq * 0.5 * t);

    // Mix
    let sample = saw * 0.5 + sub * 0.5;

    // Simple low-pass filter (smoothing)
    if (i > 0) {
      const cutoff = Math.min(800 / freq, 0.8);
      sample = output[i-1] * (1 - cutoff) + sample * cutoff;
    }

    // Amplitude envelope
    const attack = 0.005;
    const release = 0.05;
    let env = 1;
    if (t < attack) env = t / attack;
    if (t > duration - release) env = (duration - t) / release;

    output[i] = sample * env * 0.7;
  }

  return output;
}

/**
 * Synthesize melodic note (detuned saws for synth sound)
 */
function synthLead(freq: number, duration: number, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  const detune = 1.005; // Slight detune for fatness

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Two detuned sawtooth waves
    const phase1 = freq * t;
    const phase2 = freq * detune * t;
    const saw1 = 2 * (phase1 % 1) - 1;
    const saw2 = 2 * (phase2 % 1) - 1;

    // Add a triangle for body
    const triPhase = freq * t * 2;
    const tri = Math.abs((triPhase % 1) * 4 - 2) - 1;

    let sample = (saw1 + saw2) * 0.3 + tri * 0.2;

    // Filter envelope (opens then closes)
    const filterAttack = 0.02;
    const filterDecay = 0.1;
    let filterEnv: number;
    if (t < filterAttack) {
      filterEnv = t / filterAttack;
    } else {
      filterEnv = Math.exp(-(t - filterAttack) / filterDecay) * 0.7 + 0.3;
    }

    // Simple low-pass (smoothing based on filter envelope)
    // Higher cutoff = more mid/high frequencies
    if (i > 0) {
      const cutoff = 0.4 + filterEnv * 0.5; // Brighter filter
      sample = output[i-1] * (1 - cutoff) + sample * cutoff;
    }

    // Amplitude envelope
    const attack = 0.01;
    const release = 0.08;
    let ampEnv = 1;
    if (t < attack) ampEnv = t / attack;
    if (t > duration - release) ampEnv = Math.max(0, (duration - t) / release);

    output[i] = sample * ampEnv * 0.6;
  }

  return output;
}

/**
 * Synthesize high melodic note (brighter, more harmonics)
 */
function synthHigh(freq: number, duration: number, sampleRate: number): Float32Array {
  const len = Math.floor(duration * sampleRate);
  const output = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    const t = i / sampleRate;

    // Square-ish wave (odd harmonics)
    let sample = 0;
    for (let h = 1; h <= 5; h += 2) {
      sample += Math.sin(2 * Math.PI * freq * h * t) / h;
    }
    sample *= 0.5;

    // Add brightness with a bit of saw
    const sawPhase = freq * t;
    const saw = 2 * (sawPhase % 1) - 1;
    sample = sample * 0.6 + saw * 0.2;

    // Amplitude envelope (plucky)
    const attack = 0.005;
    const decay = 0.1;
    const sustain = 0.4;
    const release = 0.1;

    let ampEnv: number;
    if (t < attack) {
      ampEnv = t / attack;
    } else if (t < attack + decay) {
      ampEnv = 1 - (1 - sustain) * (t - attack) / decay;
    } else if (t > duration - release) {
      ampEnv = sustain * (duration - t) / release;
    } else {
      ampEnv = sustain;
    }

    output[i] = sample * Math.max(0, ampEnv) * 0.5;
  }

  return output;
}

/**
 * Pitch shift a sample to target frequency
 */
function pitchShiftSample(
  sample: Float32Array,
  sourceFreq: number,
  targetFreq: number,
  duration: number,
  sampleRate: number
): Float32Array {
  const ratio = targetFreq / sourceFreq;
  const outputLength = Math.min(Math.floor(duration * sampleRate), Math.floor(sample.length / ratio));
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const sourceIndex = i * ratio;
    const idx = Math.floor(sourceIndex);
    const frac = sourceIndex - idx;

    if (idx + 1 < sample.length) {
      output[i] = sample[idx] * (1 - frac) + sample[idx + 1] * frac;
    } else if (idx < sample.length) {
      output[i] = sample[idx];
    }
  }

  // Apply envelope
  const attackSamples = Math.floor(0.005 * sampleRate);
  const releaseSamples = Math.floor(0.05 * sampleRate);
  for (let i = 0; i < attackSamples && i < output.length; i++) {
    output[i] *= i / attackSamples;
  }
  for (let i = 0; i < releaseSamples && i < output.length; i++) {
    const idx = output.length - 1 - i;
    output[idx] *= i / releaseSamples;
  }

  return output;
}

/**
 * Find closest sample in model for a given note
 * Handles samples named with or without octave (e.g., "c" or "c4")
 */
function findClosestSample(model: GranularModel, targetNote: string): { sample: Float32Array; freq: number } | null {
  const targetMidi = noteToMidi(targetNote);
  if (targetMidi === null) return null;

  let closest: { sample: Float32Array; freq: number; diff: number } | null = null;

  for (const [noteName, sample] of model.samples) {
    // Try direct lookup first (e.g., "c4")
    let sampleMidi = NOTE_TO_MIDI[noteName];

    // If not found, it's a pitch class without octave (e.g., "c")
    // Use octave 4 as reference
    if (sampleMidi === undefined) {
      sampleMidi = NOTE_TO_MIDI[noteName + '4'];
    }

    if (sampleMidi !== undefined) {
      // For pitch class samples, find the best octave match
      const targetPitchClass = targetMidi % 12;
      const samplePitchClass = sampleMidi % 12;

      // Calculate difference considering pitch class match is best
      let diff: number;
      if (targetPitchClass === samplePitchClass) {
        // Same pitch class - perfect match, will pitch shift octaves
        diff = 0;
      } else {
        // Different pitch class - use semitone distance
        diff = Math.min(
          Math.abs(targetPitchClass - samplePitchClass),
          12 - Math.abs(targetPitchClass - samplePitchClass)
        );
      }

      if (!closest || diff < closest.diff) {
        // Use the sample's reference pitch (at octave 4 if pitch class only)
        closest = { sample, freq: midiToFreq(sampleMidi), diff };
      }
    }
  }

  return closest ? { sample: closest.sample, freq: closest.freq } : null;
}

/**
 * Render all voices to audio buffer
 */
async function render(
  voices: ParsedVoice[],
  models: Map<string, GranularModel>,
  bpm: number,
  duration: number
): Promise<Float32Array> {
  const totalSamples = Math.floor(duration * SAMPLE_RATE);
  const output = new Float32Array(totalSamples);

  const cyclesPerSecond = bpm / 60 / 4;
  const totalCycles = duration * cyclesPerSecond;
  const samplesPerCycle = SAMPLE_RATE / cyclesPerSecond;

  console.log(`\nRendering ${voices.length} voices at ${bpm} BPM for ${duration}s (${totalCycles.toFixed(1)} cycles)`);

  for (const voice of voices) {
    const gain = voice.effects.gain;
    console.log(`  ${voice.name}: gain=${gain.toFixed(2)}, type=${voice.modelType}, ${voice.patterns.length} patterns`);

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

        // Handle drum triggers - USE SYNTHESIS
        if (voice.modelType === 'drums') {
          const drumType = event.value.toLowerCase();
          let drumSample: Float32Array;
          let drumGain = gain * 0.15; // Minimal drums

          if (drumType === 'bd' || drumType === 'kick') {
            drumSample = synthKick(0.3, SAMPLE_RATE);
            drumGain *= 0.2; // Very low kicks
          } else if (drumType === 'sd' || drumType === 'snare' || drumType === 'sn') {
            drumSample = synthSnare(0.2, SAMPLE_RATE);
          } else if (drumType === 'oh' || drumType === 'openhat') {
            drumSample = synthHihat(0.3, SAMPLE_RATE, true);
            drumGain *= 1.5; // Boost hi-hats for brightness
          } else if (drumType === 'hh' || drumType === 'hihat' || drumType === 'ch') {
            drumSample = synthHihat(0.1, SAMPLE_RATE, false);
            drumGain *= 1.5; // Boost hi-hats for brightness
          } else {
            // Default to kick for unknown drums
            drumSample = synthKick(0.3, SAMPLE_RATE);
            drumGain *= 0.5;
          }

          const len = Math.min(drumSample.length, totalSamples - startSample);
          for (let i = 0; i < len; i++) {
            output[startSample + i] += drumSample[i] * drumGain;
          }
          continue;
        }

        // Handle melodic notes - USE SYNTHESIS
        const targetMidi = noteToMidi(event.value);
        if (targetMidi === null) continue;

        const targetFreq = midiToFreq(targetMidi);
        let synthSample: Float32Array;
        let voiceGain = gain;

        // Choose synth based on voice type and register
        // D5=74 is ~587Hz (mid), C6=84 is ~1047Hz (high-mid)
        // Adjust gains to favor mids over bass (match typical melodic stem)
        if (voice.modelType === 'bass') {
          synthSample = synthBass(targetFreq, Math.min(eventDuration, 0.5), SAMPLE_RATE);
          voiceGain *= 0.08; // Minimal bass
        } else if (targetMidi >= 84) {
          // Only C6 and above use high synth (bright, upper harmonics)
          synthSample = synthHigh(targetFreq, Math.min(eventDuration, 0.4), SAMPLE_RATE);
          voiceGain *= 2.5;
        } else {
          // Everything else (including "high" voice D5-G5) uses lead synth for mids
          synthSample = synthLead(targetFreq, Math.min(eventDuration, 0.4), SAMPLE_RATE);
          voiceGain *= 3.0; // Dominant mids
        }

        const len = Math.min(synthSample.length, totalSamples - startSample);
        for (let i = 0; i < len; i++) {
          output[startSample + i] += synthSample[i] * voiceGain;
        }
      }

      currentCycle++;
      patternIndex++;
    }
  }

  // Apply simple high-pass filter to reduce mud (remove sub-bass below ~80Hz)
  const hpfAlpha = 0.995; // Cutoff ~80Hz at 44100 sample rate
  let hpfPrev = 0;
  let hpfPrevOut = 0;
  for (let i = 0; i < output.length; i++) {
    const filtered = hpfAlpha * (hpfPrevOut + output[i] - hpfPrev);
    hpfPrev = output[i];
    hpfPrevOut = filtered;
    output[i] = filtered;
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < output.length; i++) {
    maxVal = Math.max(maxVal, Math.abs(output[i]));
  }
  if (maxVal > 0.95) {
    const scale = 0.95 / maxVal;
    for (let i = 0; i < output.length; i++) {
      output[i] *= scale;
    }
    console.log(`  Normalized: peak ${maxVal.toFixed(2)} → 0.95`);
  }

  return output;
}

/**
 * Convert Float32Array to WAV buffer
 */
function float32ToWav(samples: Float32Array, sampleRate: number): Buffer {
  const numChannels = 1;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = samples.length * blockAlign;
  const buffer = Buffer.alloc(44 + dataLength);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * blockAlign, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitDepth, 34);

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataLength, 40);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.floor(s * 32767), 44 + i * 2);
  }

  return buffer;
}

/**
 * Main function
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Node.js Strudel Renderer (with Granular Samples)');
    console.log('');
    console.log('Usage: node render-strudel-node.js <input.strudel> -m <models_dir> -o <output.wav> [options]');
    console.log('');
    console.log('Options:');
    console.log('  -m, --models <dir>     Models directory (required for granular)');
    console.log('  -o, --output <file>    Output WAV file');
    console.log('  -d, --duration <sec>   Duration in seconds (default: 60)');
    console.log('  --bpm <bpm>            Override BPM');
    process.exit(0);
  }

  // Parse arguments
  let inputPath = '';
  let outputPath = '';
  let modelsDir = '';
  let duration = 60;
  let overrideBpm = 0;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-m' || arg === '--models') {
      modelsDir = path.resolve(args[++i]);
    } else if (arg === '-o' || arg === '--output') {
      outputPath = path.resolve(args[++i]);
    } else if (arg === '-d' || arg === '--duration') {
      duration = parseFloat(args[++i]);
    } else if (arg === '--bpm') {
      overrideBpm = parseFloat(args[++i]);
    } else if (!arg.startsWith('-')) {
      inputPath = path.resolve(arg);
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
  console.log('Node.js Strudel Renderer (Granular)');
  console.log('━'.repeat(60));
  console.log(`Input:    ${inputPath}`);
  console.log(`Output:   ${outputPath}`);
  console.log(`Models:   ${modelsDir || '(none - will use synth)'}`);
  console.log(`Duration: ${duration}s`);

  // Read and parse Strudel code
  const code = fs.readFileSync(inputPath, 'utf-8');
  const { bpm: parsedBpm, voices } = parseStrudelCode(code);
  const bpm = overrideBpm || parsedBpm;

  console.log(`\nParsed: ${voices.length} voices, ${bpm} BPM`);
  for (const v of voices) {
    console.log(`  ${v.name}: ${v.patterns.length} patterns, model=${v.modelType}, gain=${v.effects.gain.toFixed(2)}`);
  }

  // Load granular models
  const models = new Map<string, GranularModel>();
  if (modelsDir && fs.existsSync(modelsDir)) {
    console.log(`\nLoading granular models from ${modelsDir}:`);
    for (const modelName of ['melodic', 'bass', 'drums']) {
      const model = loadGranularModel(modelsDir, modelName);
      if (model) {
        models.set(modelName, model);
      }
    }
  }

  if (models.size === 0) {
    console.log('\nWarning: No granular models loaded, output may be silent');
  }

  // Render
  const audio = await render(voices, models, bpm, duration);

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
