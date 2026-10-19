import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeWav, checkRender, sampleFailures, expectedFrames } from '../dist/offline-helpers.js';

function wav16(frames, sr = 44100, ch = 2, amp = 0.5) {
  const b = Buffer.alloc(44 + frames * ch * 2);
  b.write('RIFF', 0); b.writeUInt32LE(b.length - 8, 4); b.write('WAVE', 8); b.write('fmt ', 12);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(ch, 22); b.writeUInt32LE(sr, 24);
  b.writeUInt32LE(sr * ch * 2, 28); b.writeUInt16LE(ch * 2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(frames * ch * 2, 40);
  for (let i = 0; i < frames * ch; i++) b.writeInt16LE(Math.round(Math.sin(i / 7) * amp * 32767), 44 + i * 2);
  return new Uint8Array(b);
}
test('analyzeWav reads format, frames, rms', () => {
  const i = analyzeWav(wav16(44100));
  assert.equal(i.sampleRate, 44100); assert.equal(i.channels, 2); assert.equal(i.frames, 44100);
  assert.ok(Math.abs(i.rms - 0.5 / Math.SQRT2) < 0.01);
});
test('expectedFrames', () => assert.equal(expectedFrames(80.7495, 0.53833, 44100), Math.floor(80.7495 / 0.53833 * 44100)));
test('checkRender passes a good render', () => {
  const i = analyzeWav(wav16(44100));
  assert.ok(checkRender(i, { sampleRate: 44100, expectedFrames: 44100, failures: [] }).ok);
});
test('checkRender fails on silence, duration mismatch, failed samples', () => {
  const sil = analyzeWav(wav16(44100, 44100, 2, 0));
  const r = checkRender(sil, { sampleRate: 44100, expectedFrames: 50000, failures: ['[sampler] error: failed to load x'] });
  assert.equal(r.ok, false); assert.equal(r.problems.length, 3);
  assert.equal(checkRender(analyzeWav(wav16(44100)), { sampleRate: 44100, expectedFrames: 44100, failures: ['error: x'], allowFailures: true }).ok, true);
});
test('sampleFailures dedups and ignores benign lines', () => {
  const f = sampleFailures(['[sampler] load sound "hh" done', 'error: failed to load a', 'error:  failed to load a', '[webaudio] start rendering', 'sound xyz not found']);
  assert.equal(f.length, 2);
});

import { overlapAdd, floatToWav16, chunkWindows } from '../dist/offline-helpers.js';
test('overlapAdd superposes tails and clips at the end', () => {
  const L = new Float32Array(10), R = new Float32Array(10);
  overlapAdd(L, R, Float32Array.from([1, 1, 1, 1, 2, 2, 2, 2]), 0);   // 4 frames L=1 R=2
  const n = overlapAdd(L, R, Float32Array.from([1, 1, 1, 1, 2, 2, 2, 2]), 8); // only 2 frames fit
  assert.equal(n, 2); assert.equal(L[8], 1); assert.equal(R[9], 2); assert.equal(L[3], 1);
  overlapAdd(L, R, Float32Array.from([1, 1, 1, 1, 2, 2, 2, 2]), 2);   // overlaps frames 2,3
  assert.equal(L[2], 2); assert.equal(L[3], 2); assert.equal(R[2], 4);
});
test('chunk windows tile [0,cycles) exactly; wav roundtrip', () => {
  const w = chunkWindows(10.5, 4);
  assert.deepEqual(w, [[0, 4], [4, 8], [8, 10.5]]);
  const L = Float32Array.from([0, 0.5, -0.5]), R = Float32Array.from([1, -1, 0]);
  const i = analyzeWav(new Uint8Array(floatToWav16(L, R, 44100)));
  assert.equal(i.frames, 3); assert.equal(i.channels, 2); assert.ok(i.peak > 0.99);
});

import { planChunks } from '../dist/offline-helpers.js';
test('planChunks: time cap, onset cap, min one cycle, exact tiling', () => {
  const flat = Array(25).fill(10);
  assert.deepEqual(planChunks(flat, 24.5, 11, 1e9), [[0, 11], [11, 22], [22, 24.5]]);
  // onset cap: 10/cycle, cap 35 -> 3 cycles per window
  const w = planChunks(flat, 25, 11, 35);
  assert.ok(w.every(([b, e]) => (e - b) <= 3) && w[0][0] === 0 && w.at(-1)[1] === 25);
  for (let i = 1; i < w.length; i++) assert.equal(w[i][0], w[i - 1][1]);
  // one dense cycle still gets its own window (never empty / never split below a cycle)
  const dense = [5, 5, 5000, 5, 5];
  assert.deepEqual(planChunks(dense, 5, 11, 100), [[0, 2], [2, 3], [3, 5]]);
  // the onsets of every window respect the cap unless it is a single cycle
  const d2 = [300, 400, 600, 700, 100, 100, 100];
  for (const [b, e] of planChunks(d2, 7, 11, 1000)) assert.ok(e - b === 1 || d2.slice(b, e).reduce((x, y) => x + y, 0) <= 1000);
});

import { closeAllWithin } from '../dist/offline-helpers.js';
test('closeAllWithin: hung/rejecting close() is killed within the grace, good ones are not', async () => {
  const killed = [];
  const mk = (name, close) => ({ close, kill: () => killed.push(name) });
  const t0 = Date.now();
  const n = await closeAllWithin([
    mk('ok', async () => {}),
    mk('hang', () => new Promise(() => {})),
    mk('reject', () => Promise.reject(new Error('gone'))),
  ], 100);
  assert.equal(n, 2); assert.deepEqual(killed.sort(), ['hang', 'reject']); assert.ok(Date.now() - t0 < 1000);
});

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { triggerArgs, prerollPlan, trimChunk, removeStaleOutput, commitOutput } from '../dist/offline-helpers.js';

test('triggerArgs: cycle is the hap ABSOLUTE cycle in every chunk, t is chunk-relative', () => {
  const cps = 0.5, sr = 44100;
  // hap at cycle 10.25 in a chunk whose context starts at cycle 8 (= 16 s)
  const a = triggerArgs(10.25, cps, 8 / cps);
  assert.equal(a.cycle, 10.25);               // old code passed t (= 4.5 s) here
  assert.ok(Math.abs(a.t - 4.5) < 1e-12);
  assert.notEqual(a.cycle, a.t);
  assert.equal(triggerArgs(0.5, cps, 0).cycle, 0.5);
});
test('prerollPlan: none for the first chunk, whole-frame window before later chunks', () => {
  assert.deepEqual(prerollPlan(0, 0, 0.5, 44100, 4), { preCycle: 0, preFrames: 0 });
  const p = prerollPlan(10, 20 * 44100, 0.5, 44100, 4); // 4 s = 2 cycles
  assert.equal(p.preCycle, 8); assert.equal(p.preFrames, 4 * 44100);
  assert.equal(prerollPlan(1, 2 * 44100, 0.5, 44100, 4).preFrames, 2 * 44100); // clamped at song start
});
test('trimChunk drops pre-roll and tail, keeps planar layout', () => {
  const len = 6, f = Float32Array.from([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]);
  assert.deepEqual([...trimChunk(f, len, 2, 3)], [2, 3, 4, 12, 13, 14]);
  assert.deepEqual([...trimChunk(f, len, 4, 99)], [4, 5, 14, 15]);
});
test('commitOutput: failed checks leave NO file, passing checks rename atomically', () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'commit-')), t = path.join(d, 'out.wav');
  fs.writeFileSync(t, 'stale'); removeStaleOutput(t); assert.equal(fs.existsSync(t), false);
  assert.equal(commitOutput(t, new Uint8Array([1, 2, 3]), () => false), false);
  assert.deepEqual(fs.readdirSync(d), []);                       // no target, no temp
  assert.equal(commitOutput(t, new Uint8Array([1, 2, 3]), () => true), true);
  assert.deepEqual(fs.readdirSync(d), ['out.wav']);
  assert.deepEqual([...fs.readFileSync(t)], [1, 2, 3]);
  fs.rmSync(d, { recursive: true });
});
