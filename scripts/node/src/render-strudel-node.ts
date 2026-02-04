#!/usr/bin/env node
/**
 * Node.js Strudel Renderer
 *
 * Renders Strudel patterns using Node.js Web Audio API for offline rendering.
 * This avoids browser dependencies and allows direct WAV output.
 */

import { OfflineAudioContext, AudioBuffer } from 'node-web-audio-api';
import fs from 'fs';
import path from 'path';

// Types
interface NoteFrequencies {
  [key: string]: number;
}

interface ParsedPattern {
  events: PatternEvent[];
  totalBeats: number;
}

interface PatternEvent {
  note: string;
  time: number;
  duration: number;
}

interface ParsedCode {
  bpm: number;
  patterns: { [voice: string]: string[] };
}

interface VoiceConfig {
  type: OscillatorType;
  gain: number;
  voiceType: VoiceType;
}

interface DrumConfig {
  gain: number;
}

type VoiceType = 'bass' | 'mid' | 'high';

// Note frequencies (A4 = 440Hz)
const NOTE_FREQ: NoteFrequencies = {};
const NOTES = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'];

for (let octave = 0; octave <= 8; octave++) {
  NOTES.forEach((note, i) => {
    const midiNote = 12 + octave * 12 + i;
    NOTE_FREQ[`${note}${octave}`] = 440 * Math.pow(2, (midiNote - 69) / 12);
  });
}

/**
 * Parse Strudel code to extract patterns
 */
function parseStrudelCode(code: string): ParsedCode {
  const result: ParsedCode = {
    bpm: 120,
    patterns: {},
  };

  // Extract BPM
  const bpmMatch = code.match(/setcps\((\d+)\/60\/4\)/);
  if (bpmMatch) {
    result.bpm = parseInt(bpmMatch[1]);
  }

  // Extract pattern arrays
  const arrayRegex = /let\s+(\w+)\s*=\s*\[([\s\S]*?)\]/g;
  let match;
  while ((match = arrayRegex.exec(code)) !== null) {
    const name = match[1];
    const content = match[2];

    // Parse string patterns
    const patterns: string[] = [];
    const stringRegex = /"([^"]+)"/g;
    let strMatch;
    while ((strMatch = stringRegex.exec(content)) !== null) {
      patterns.push(strMatch[1]);
    }

    if (patterns.length > 0) {
      result.patterns[name] = patterns;
    }
  }

  return result;
}

/**
 * Parse mini notation pattern to events
 */
function parsePattern(pattern: string, bpm: number, startBeat: number = 0): ParsedPattern {
  const events: PatternEvent[] = [];
  const cps = bpm / 60 / 4;
  const beatDuration = 1 / cps / 4;

  const tokens = pattern.split(/\s+/).filter(t => t);
  let beat = startBeat;

  for (const token of tokens) {
    if (token === '~') {
      beat += 1;
    } else if (token.startsWith('~*')) {
      const count = parseInt(token.slice(2)) || 1;
      beat += count;
    } else if (token.startsWith('[') && token.endsWith(']')) {
      const notes = token.slice(1, -1).split(',');
      for (const note of notes) {
        events.push({
          note: note.trim(),
          time: beat * beatDuration,
          duration: beatDuration * 0.8,
        });
      }
      beat += 1;
    } else {
      events.push({
        note: token,
        time: beat * beatDuration,
        duration: beatDuration * 0.8,
      });
      beat += 1;
    }
  }

  return { events, totalBeats: beat - startBeat };
}

/**
 * Create oscillator for melodic notes - optimized for mid/high frequencies
 */
function createTone(
  ctx: OfflineAudioContext,
  freq: number,
  startTime: number,
  duration: number,
  type: OscillatorType = 'sine',
  gain: number = 0.3,
  voiceType: VoiceType = 'mid'
): void {
  const osc = ctx.createOscillator();
  const gainNode = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  osc.type = type;
  osc.frequency.value = freq;

  // Voice-specific filtering to match original frequency balance
  if (voiceType === 'bass') {
    filter.type = 'lowpass';
    filter.frequency.value = 800;
    gain *= 0.15;
  } else if (voiceType === 'mid') {
    filter.type = 'bandpass';
    filter.frequency.value = 2500;
    filter.Q.value = 0.3;
    gain *= 1.5;
  } else if (voiceType === 'high') {
    filter.type = 'highpass';
    filter.frequency.value = 400;
    gain *= 2.5;
  }

  // Brighter attack envelope
  gainNode.gain.setValueAtTime(0, startTime);
  gainNode.gain.linearRampToValueAtTime(gain, startTime + 0.005);
  gainNode.gain.linearRampToValueAtTime(gain * 0.8, startTime + 0.02);
  gainNode.gain.linearRampToValueAtTime(gain * 0.6, startTime + duration * 0.3);
  gainNode.gain.linearRampToValueAtTime(0, startTime + duration);

  osc.connect(filter);
  filter.connect(gainNode);
  gainNode.connect(ctx.destination);

  osc.start(startTime);
  osc.stop(startTime + duration + 0.1);

  // Add harmonics for ALL melodic voices to get high-frequency content
  const harmConfig: { [key in VoiceType]: { harmonics: number[]; gains: number[] } } = {
    bass: { harmonics: [2], gains: [0.05] },
    mid: { harmonics: [2, 3, 4, 5], gains: [0.15, 0.1, 0.06, 0.03] },
    high: { harmonics: [2, 3, 4, 5, 6, 7, 8], gains: [0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04] },
  };

  const config = harmConfig[voiceType];
  for (let i = 0; i < config.harmonics.length; i++) {
    const h = config.harmonics[i];
    const harmFreq = freq * h;
    if (harmFreq < 12000) {
      const oscH = ctx.createOscillator();
      const gainH = ctx.createGain();
      oscH.type = 'sine';
      oscH.frequency.value = harmFreq;

      const harmGain = gain * config.gains[i];
      gainH.gain.setValueAtTime(0, startTime);
      gainH.gain.linearRampToValueAtTime(harmGain, startTime + 0.003);
      gainH.gain.linearRampToValueAtTime(harmGain * 0.5, startTime + duration * 0.3);
      gainH.gain.linearRampToValueAtTime(0, startTime + duration * 0.8);

      oscH.connect(gainH);
      gainH.connect(ctx.destination);
      oscH.start(startTime);
      oscH.stop(startTime + duration);
    }
  }
}

/**
 * Create drum sound - reduced bass, more click/highs
 */
function createDrum(ctx: OfflineAudioContext, type: string, startTime: number, gain: number = 0.5): void {
  if (type === 'bd') {
    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();
    const hpf = ctx.createBiquadFilter();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(100, startTime);
    osc.frequency.exponentialRampToValueAtTime(50, startTime + 0.08);

    hpf.type = 'highpass';
    hpf.frequency.value = 40;

    gainNode.gain.setValueAtTime(gain * 0.3, startTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + 0.15);

    osc.connect(hpf);
    hpf.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc.start(startTime);
    osc.stop(startTime + 0.2);
  } else if (type === 'sd') {
    const osc = ctx.createOscillator();
    const noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 0.2, ctx.sampleRate);
    const noiseData = noiseBuffer.getChannelData(0);
    for (let i = 0; i < noiseData.length; i++) {
      noiseData[i] = Math.random() * 2 - 1;
    }
    const noise = ctx.createBufferSource();
    noise.buffer = noiseBuffer;

    const oscGain = ctx.createGain();
    const noiseGain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    osc.type = 'triangle';
    osc.frequency.value = 180;

    filter.type = 'highpass';
    filter.frequency.value = 1000;

    oscGain.gain.setValueAtTime(gain * 0.5, startTime);
    oscGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.1);

    noiseGain.gain.setValueAtTime(gain * 0.3, startTime);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.15);

    osc.connect(oscGain);
    oscGain.connect(ctx.destination);

    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(ctx.destination);

    osc.start(startTime);
    osc.stop(startTime + 0.1);
    noise.start(startTime);
    noise.stop(startTime + 0.2);
  } else if (type === 'hh' || type === 'oh') {
    const dur = type === 'oh' ? 0.3 : 0.08;
    const noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const noiseData = noiseBuffer.getChannelData(0);
    for (let i = 0; i < noiseData.length; i++) {
      noiseData[i] = Math.random() * 2 - 1;
    }
    const noise = ctx.createBufferSource();
    noise.buffer = noiseBuffer;

    const filter = ctx.createBiquadFilter();
    const gainNode = ctx.createGain();

    filter.type = 'highpass';
    filter.frequency.value = 5000;

    gainNode.gain.setValueAtTime(gain * 0.5, startTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + dur);

    noise.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(ctx.destination);

    noise.start(startTime);
    noise.stop(startTime + dur);
  }
}

/**
 * Render patterns to audio buffer
 */
async function renderPatterns(parsedCode: ParsedCode, duration: number): Promise<AudioBuffer> {
  const sampleRate = 44100;
  const totalSamples = Math.ceil(duration * sampleRate);
  const ctx = new OfflineAudioContext(2, totalSamples, sampleRate);

  const { bpm, patterns } = parsedCode;
  const cps = bpm / 60 / 4;

  console.log(`Rendering at ${bpm} BPM, ${duration}s duration`);

  const voiceConfig: { [key: string]: VoiceConfig | DrumConfig } = {
    bass: { type: 'sine', gain: 0.08, voiceType: 'bass' } as VoiceConfig,
    mid: { type: 'triangle', gain: 0.5, voiceType: 'mid' } as VoiceConfig,
    high: { type: 'sine', gain: 0.6, voiceType: 'high' } as VoiceConfig,
    drums: { gain: 0.2 } as DrumConfig,
  };

  for (const [voiceName, voicePatterns] of Object.entries(patterns)) {
    const config = voiceConfig[voiceName] || { type: 'sine' as OscillatorType, gain: 0.2, voiceType: 'mid' as VoiceType };
    let currentTime = 0;
    let patternIndex = 0;

    while (currentTime < duration) {
      const pattern = voicePatterns[patternIndex % voicePatterns.length];
      const { events, totalBeats } = parsePattern(pattern, bpm);
      const beatDuration = 1 / cps / 4;

      for (const event of events) {
        const eventTime = currentTime + event.time;
        if (eventTime >= duration) break;

        if (voiceName === 'drums') {
          createDrum(ctx, event.note, eventTime, (config as DrumConfig).gain);
        } else {
          const noteName = event.note.toLowerCase().replace('#', 's');
          const freq = NOTE_FREQ[noteName];
          if (freq) {
            const vc = config as VoiceConfig;
            createTone(ctx, freq, eventTime, event.duration, vc.type, vc.gain, vc.voiceType);
          }
        }
      }

      currentTime += totalBeats * beatDuration;
      patternIndex++;
    }
  }

  console.log('Rendering audio buffer...');
  const buffer = await ctx.startRendering();
  return buffer;
}

/**
 * Convert AudioBuffer to WAV
 */
function audioBufferToWav(buffer: AudioBuffer): Buffer {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1;
  const bitDepth = 16;

  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;

  const dataLength = buffer.length * blockAlign;
  const headerLength = 44;
  const totalLength = headerLength + dataLength;

  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string): void => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, totalLength - 8, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);

  const channels: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) {
    channels.push(buffer.getChannelData(c));
  }

  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let c = 0; c < numChannels; c++) {
      const sample = Math.max(-1, Math.min(1, channels[c][i]));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, intSample, true);
      offset += 2;
    }
  }

  return Buffer.from(arrayBuffer);
}

/**
 * Main function
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Node.js Strudel Renderer');
    console.log('');
    console.log('Usage: node render-strudel-node.js <input.strudel> [output.wav] [--duration <sec>]');
    console.log('');
    console.log('Renders Strudel patterns directly in Node.js without browser.');
    process.exit(0);
  }

  const inputPath = path.resolve(args[0]);
  let outputPath = args[1] && !args[1].startsWith('--') ? path.resolve(args[1]) : inputPath.replace('.strudel', '_node.wav');
  let duration = 30;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--duration' && args[i + 1]) {
      duration = parseInt(args[++i]);
    }
    if (args[i] === '-o' && args[i + 1]) {
      outputPath = path.resolve(args[++i]);
    }
  }

  if (!fs.existsSync(inputPath)) {
    console.error(`File not found: ${inputPath}`);
    process.exit(1);
  }

  console.log(`\nNode.js Strudel Renderer`);
  console.log('━'.repeat(40));
  console.log(`Input: ${inputPath}`);
  console.log(`Output: ${outputPath}`);
  console.log(`Duration: ${duration}s`);
  console.log('━'.repeat(40) + '\n');

  const code = fs.readFileSync(inputPath, 'utf-8');
  const parsedCode = parseStrudelCode(code);

  console.log(`BPM: ${parsedCode.bpm}`);
  console.log(`Patterns found: ${Object.keys(parsedCode.patterns).join(', ')}`);

  const buffer = await renderPatterns(parsedCode, duration);

  console.log('Converting to WAV...');
  const wavBuffer = audioBufferToWav(buffer);

  fs.writeFileSync(outputPath, wavBuffer);
  console.log(`\nSaved: ${outputPath} (${(wavBuffer.length / 1024 / 1024).toFixed(1)} MB)`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
