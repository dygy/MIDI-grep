// Run-to-run determinism of the offline renderer (src/determinism.ts): two renders of the SAME file must be sample-identical.
// Measured on a real piece before the fix: max abs diff 6281 int16, SNR 37 dB between two renders (reverb impulse response
// from unseeded Math.random + orbit ducking scheduled from a main-thread onended callback during the render + voice
// release (disconnect) at a random render position). MIDIGREP_TEST_RENDERER=<render-strudel-offline.js> points the suite at
// another build (the proof that the tests fail on the old renderer); MIDIGREP_RENDER_DETERMINISTIC=0 switches the fix off.
// Skipped when no Chromium is available. One renderer at a time machine-wide.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { mixSeed, deterministicInitScript } from '../dist/determinism.js';

const RENDERER = process.env.MIDIGREP_TEST_RENDERER || fileURLToPath(new URL('../dist/render-strudel-offline.js', import.meta.url));
const haveRenderer = fs.existsSync(RENDERER);
let haveChromium = false;
try { const p = (await import('puppeteer')).default; haveChromium = !!p.executablePath() && fs.existsSync(p.executablePath()); } catch { /* no puppeteer */ }
const skip = !haveRenderer || !haveChromium ? 'renderer build or Chromium missing' : false;

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitForIdleRenderer() {
  for (let i = 0; i < 360; i++) {
    const r = spawnSync('pgrep', ['-f', '^(nice )?node .*dist/render-strudel-offline\\.js']);
    if (r.status !== 0) return;
    await sleep(5000);
  }
}
async function render(code, seconds, extra = []) {
  await waitForIdleRenderer();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'offline-det-'));
  const src = path.join(dir, 'p.strudel'), out = path.join(dir, 'o.wav');
  fs.writeFileSync(src, code);
  const p = spawn('nice', ['node', RENDERER, src, '-o', out, '-d', String(seconds), ...extra], { env: process.env });
  let log = ''; p.stdout.on('data', d => log += d); p.stderr.on('data', d => log += d);
  const status = await new Promise(r => p.on('close', r));
  assert.equal(status, 0, log);
  return fs.readFileSync(out);
}
/** First differing sample and the max abs int16 difference of two 16-bit WAVs (44-byte header). */
function diff(a, b) {
  assert.equal(a.length, b.length, 'render lengths differ');
  let max = 0, n = 0, first = -1;
  for (let o = 44; o + 1 < a.length; o += 2) {
    const d = Math.abs(a.readInt16LE(o) - b.readInt16LE(o));
    if (d) { n++; if (first < 0) first = (o - 44) / 4; if (d > max) max = d; }
  }
  return { max, n, firstFrame: first, samples: (a.length - 44) / 2 };
}
/** What is left once the engine is seeded: Chrome sums >= 3 inputs of one AudioNode in an order that depends on heap addresses
 *  (float addition is not associative), so a few samples flip by 1 LSB. Anything above that is real non-determinism. */
function assertSame(d, what) {
  console.log(`  ${what}: ${d.n} of ${d.samples} samples differ, max abs ${d.max} (int16)`);
  assert.ok(d.max <= 1 && d.n / d.samples < 1e-3, `renders differ: ${d.n} samples, max abs ${d.max}, first at frame ${d.firstFrame}`);
}
function rmsBetween(wav, sr, t0, t1) {
  let s = 0, n = 0;
  for (let i = Math.floor(t0 * sr); i < Math.floor(t1 * sr); i++) { const v = wav.readInt16LE(44 + i * 4) / 32768; s += v * v; n++; }
  return Math.sqrt(s / n);
}
const CHUNKED = ['--chunk-seconds', '2', '--preroll-seconds', '0.6', '--tail-seconds', '2']; // several pages, each with its own reverb

test('two renders of a piece with reverb, noise and ducking are the same render (at most 1-LSB float summation flips)', { skip }, async () => {
  // cps 0.5, 3 chunks of 2 s windows. Reverb on two layers (default size: one impulse response per orbit and page),
  // a noise oscillator, a saw voice, and a sine bass on orbit 2 ducked by a kick (duckorbit).
  const code = `setcps(0.5)
$: stack(
  note("c3 e3 g3 c4").s("sawtooth").lpf(1500).room(0.7).gain(0.3),
  s("white*4").hpf(3000).decay(0.12).sustain(0).gain(0.25).room(0.5),
  s("pink*2").lpf(900).decay(0.3).sustain(0).gain(0.25),
  note("c2").s("sine").orbit(2).gain(0.5),
  note("~ c1*2").s("sine").gain(0.3).duckorbit(2).duckdepth(0.8).duckattack(0.2).duckonset(0.02)
)`;
  const a = await render(code, 6, CHUNKED), b = await render(code, 6, CHUNKED);
  const d = diff(a, b);
  assertSame(d, 'two renders');
});

test('supersaw / wavetable worklet start phases are seeded: two renders are the same', { skip }, async () => {
  const code = `setcps(0.5)
$: note("c3 e3 g3 b3").s("supersaw").gain(0.2).lpf(2500)`;
  const a = await render(code, 4, CHUNKED), b = await render(code, 4, CHUNKED);
  const d = diff(a, b);
  assertSame(d, 'two renders');
});

test('ducking lands on the kick: the ducked orbit sits at its ducked level at kick + onset', { skip }, async () => {
  // constant tone on orbit 2; a (silent) kick at 1.0 s (second half of the 2 s cycle) ducks it to 1 - sqrt(0.75) = 0.134 over 0.05 s, back up over 0.5 s.
  // Three periods of c3 (22.9 ms) at the bottom (1.05 s) against the same window before the kick.
  const code = `setcps(0.5)
$: stack(
  note("c3").s("sine").orbit(2).gain(0.5).slow(2),
  note("~ c1").s("sine").gain(0).duckorbit(2).duckdepth(0.75).duckonset(0.05).duckattack(0.5)
)`;
  const w = await render(code, 4, CHUNKED);
  const before = rmsBetween(w, 44100, 0.9, 0.9229), bottom = rmsBetween(w, 44100, 1.045, 1.0679), late = rmsBetween(w, 44100, 1.9, 1.9229);
  const ratio = bottom / before;
  console.log(`  rms before ${before.toFixed(4)}, at the bottom ${bottom.toFixed(4)} (ratio ${ratio.toFixed(3)}, expected ~0.134), 0.9 s after the kick ${late.toFixed(4)}`);
  assert.ok(ratio > 0.09 && ratio < 0.2, `ducked level ${ratio.toFixed(3)} x the undipped level, expected ~0.134`);
  assert.ok(late / before > 0.9, 'the orbit gain recovers to 1');
});

test('the init script seeds Math.random identically every time and re-seeds per buffer shape', () => {
  assert.equal(mixSeed(1, 2, 88200, 44100), mixSeed(1, 2, 88200, 44100));
  assert.notEqual(mixSeed(1, 2, 88200, 44100), mixSeed(2, 2, 88200, 44100));
  assert.notEqual(mixSeed(1, 2, 88200, 44100), mixSeed(1, 1, 88200, 44100));
  // evaluate the script against a minimal fake DOM-audio surface, twice: same sequence, same buffer-shape reseed
  const run = () => {
    const proto = { createBuffer() { return {}; }, };
    const g = { BaseAudioContext: function () {}, AudioNode: function () {}, OfflineAudioContext: function () {}, window: {}, Math: Object.create(Math), setTimeout, fetch: undefined, URL, Blob };
    g.BaseAudioContext.prototype = proto; g.AudioNode.prototype = { disconnect() {} }; g.OfflineAudioContext.prototype = { startRendering() { return Promise.resolve(); } };
    new Function(...Object.keys(g), deterministicInitScript(7))(...Object.values(g));
    const seq = [g.Math.random(), g.Math.random()];
    proto.createBuffer(2, 100, 44100); const after = g.Math.random();
    proto.createBuffer(2, 100, 44100); const again = g.Math.random();
    return { seq, after, again };
  };
  const a = run(), b = run();
  assert.deepEqual(a, b);
  assert.equal(a.after, a.again, 'the same buffer shape re-seeds to the same stream');
  assert.notEqual(a.seq[0], a.seq[1]);
});
