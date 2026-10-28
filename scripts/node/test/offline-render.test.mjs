// End-to-end checks of the offline renderer on tiny synthetic pieces (real headless Chromium + the real superdough).
// Skipped when no Chromium is available. One renderer at a time machine-wide: each run waits for any other
// render-strudel-offline process to finish first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const RENDERER = fileURLToPath(new URL('../dist/render-strudel-offline.js', import.meta.url));
const haveRenderer = fs.existsSync(RENDERER);
let haveChromium = false;
try { haveChromium = !!(await import('puppeteer')).default.executablePath() && fs.existsSync((await import('puppeteer')).default.executablePath()); } catch { /* no puppeteer */ }
const skip = !haveRenderer || !haveChromium ? 'renderer build or Chromium missing' : false;

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitForIdleRenderer() {
  for (let i = 0; i < 360; i++) { // up to 30 min
    const r = spawnSync('pgrep', ['-f', '^(nice )?node .*dist/render-strudel-offline\\.js']) // anchored: a shell whose command line merely mentions the name is not a renderer;
    if (r.status !== 0) return;
    await sleep(5000);
  }
}
async function render(code, seconds, extra = [], env = {}) {
  await waitForIdleRenderer();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'offline-render-'));
  const src = path.join(dir, 'p.strudel'), out = path.join(dir, 'o.wav');
  fs.writeFileSync(src, code);
  const p = spawn('nice', ['node', RENDERER, src, '-o', out, '-d', String(seconds), ...extra], { env: { ...process.env, ...env } });
  let log = ''; p.stdout.on('data', d => log += d); p.stderr.on('data', d => log += d);
  const status = await new Promise(r => p.on('close', r));
  return { status, log, out: fs.existsSync(out) ? fs.readFileSync(out) : null };
}
/** RMS of a 16-bit stereo PCM WAV between two times (seconds). */
function rmsBetween(wav, sr, t0, t1) {
  const off = 44, a = Math.floor(t0 * sr), b = Math.floor(t1 * sr);
  let s = 0, n = 0;
  for (let i = a; i < b; i++) for (let c = 0; c < 2; c++) { const v = wav.readInt16LE(off + (i * 2 + c) * 2) / 32768; s += v * v; n++; }
  return Math.sqrt(s / n);
}

// cps 0.5: one cycle = 2 s. Chunks are one cycle long; the pre-roll is half a second, i.e. it starts MID-cycle.
const CHUNKED = ['--chunk-seconds', '2', '--preroll-seconds', '0.6', '--tail-seconds', '1'];

test('a duck target is not lost when a chunk starts between the keeper hap and the kick', { skip }, async () => {
  // keeper (silent, orbit 2) once per cycle at the cycle start; the kick ducking orbit 2 sits at the END of the cycle,
  // so every chunk's pre-roll window [b - 0.25 cycles, b) holds a kick and not the keeper
  const code = `setcps(0.5)
$: stack(
  note("c1 ~ ~ ~").s("sine").orbit(2).gain(0),
  note("~ ~ ~ c2").s("sine").duckorbit(2).duckdepth(0.8).gain(0.4)
)`;
  const r = await render(code, 8, CHUNKED);
  assert.doesNotMatch(r.log, /duck target orbit 2 does not exist/);
  assert.equal(r.status, 0, r.log);
});

test('a ducking hap at the very start of a render context (pre-roll edge, song start) does not throw', { skip }, async () => {
  // superdough's duck() schedules its gain ramp with stop(t - 0.01): a ducking hap at context time < 0.01 s threw
  // "stop time (-0.01) is less than the minimum bound". Pre-roll 0.5 s puts the kick (cycle position 0.75) exactly at frame 0
  // of every later chunk; the first test below puts one at t = 0 of the song.
  const code = `setcps(0.5)
$: stack(
  note("c1 ~ ~ ~").s("sine").orbit(2).gain(0),
  note("~ ~ ~ c2").s("sine").duckorbit(2).duckdepth(0.8).gain(0.4)
)`;
  const r = await render(code, 8, ['--chunk-seconds', '2', '--preroll-seconds', '0.5', '--tail-seconds', '1']);
  assert.doesNotMatch(r.log, /minimum bound/);
  assert.equal(r.status, 0, r.log);
  const first = await render(`setcps(0.5)
$: stack(note("c1 ~ ~ ~").s("sine").orbit(2).gain(0), note("c2 ~ ~ ~").s("sine").duckorbit(2).gain(0.4))`, 4, CHUNKED);
  assert.doesNotMatch(first.log, /minimum bound/);
  assert.equal(first.status, 0, first.log);
});

test('a duck target no earlier hap ever used still fails loudly', { skip }, async () => {
  const code = `setcps(0.5)
$: note("c2 ~ ~ ~").s("sine").duckorbit(3).gain(0.4)`;
  const r = await render(code, 4, CHUNKED);
  assert.notEqual(r.status, 0);
  assert.match(r.log, /duck target orbit 3 does not exist/);
});

/** A 6 s decay-free 220 Hz tone (22.05 kHz mono 16-bit WAV) served over http; `failFirst` requests answer 500 first. fn(url, hits). */
async function withTone(fn, failFirst = 0) {
  const sr = 22050, n = sr * 6, b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 220 / sr) * 0.4 * 32767), 44 + i * 2);
  const hits = { n: 0 };
  const srv = http.createServer((q, s) => {
    if (q.method === 'OPTIONS') { s.writeHead(204, { 'access-control-allow-origin': '*' }); return s.end(); }
    if (hits.n++ < failFirst) { s.writeHead(500, { 'access-control-allow-origin': '*' }); return s.end('no'); }
    s.writeHead(200, { 'content-type': 'audio/wav', 'access-control-allow-origin': '*' }); s.end(b);
  });
  await new Promise(r => srv.listen(0, '127.0.0.1', r));
  try { return await fn(`http://127.0.0.1:${srv.address().port}/long.wav`, hits); } finally { srv.close(); }
}
const ctxFrames = log => [...log.matchAll(/chunk (\d+)\/\d+ cycles .*?ctx (\d+) frames/g)].map(m => [Number(m[1]), Number(m[2])]);

test('a one-shot longer than the tail margin is reproduced whole, and a missing sound still fails', { skip }, async () => {
  // 6 s decay-free tone served over http: it rings through three 2 s chunks, far past the 1 s margin
  await withTone(async url => {
    const code = `setcps(0.5)
await samples({ longtone: ['${url}'] })
$: s("<longtone ~ ~ ~ ~ ~>")`;
    const r = await render(code, 10, CHUNKED);
    assert.equal(r.status, 0, r.log);
    // the tone starts at 0 s and rings to 6 s: the 3rd chunk (4..6 s) can only have it through the derived look-back
    const wav = r.out;
    assert.ok(rmsBetween(wav, 44100, 4.2, 5.8) > 0.1, `tone missing at 4.2-5.8 s: ${rmsBetween(wav, 44100, 4.2, 5.8)}`);
    assert.ok(rmsBetween(wav, 44100, 6.5, 9.5) < 0.01);
    // each context ends the fx margin after its window: the long voice's end is rendered by the chunk that holds it, not by every earlier one
    const first = ctxFrames(r.log).find(([i]) => i === 1);
    assert.ok(first && first[1] <= Math.ceil((2 + 1 + 0.1) * 44100), `chunk 1 context is ${first && first[1]} frames, it renders the whole 6 s voice`);
    const miss = await render(`setcps(0.5)\n$: s("nosuchsound ~ ~ ~")`, 4, CHUNKED);
    assert.notEqual(miss.status, 0);
    assert.match(miss.log, /not found|failed/i);
  });
});

test('a cut group chokes the long voice: it is not re-triggered (and heard) in later chunks', { skip }, async () => {
  // long tone at 0 s, cut(1); a 0.12 s hit of the same cut group at 1 s chokes it. Chunk 2 starts at 2 s with a 0.6 s pre-roll (from 1.4 s):
  // the choker is OLDER than the pre-roll window, only the long voice's (unchoked) ring made it a pre-roll hap.
  await withTone(async url => {
    const code = `setcps(0.5)
await samples({ longtone: ['${url}'] })
$: stack(s("<longtone ~ ~ ~ ~ ~>").cut(1), s("<[~ longtone] ~ ~ ~ ~ ~>").end(0.02).cut(1))`;
    const r = await render(code, 10, CHUNKED);
    assert.equal(r.status, 0, r.log);
    assert.ok(rmsBetween(r.out, 44100, 0.3, 0.9) > 0.1, 'the long voice must sound until it is choked at 1 s');
    assert.ok(rmsBetween(r.out, 44100, 1.5, 9.5) < 0.01, `choked voice is audible after the choke: ${rmsBetween(r.out, 44100, 1.5, 9.5)}`);
  });
});

test('a failing ring estimate fails the render loudly instead of truncating long one-shots in silence', { skip }, async () => {
  // the estimate runs in its own page: its first fetches of the sample fail (HTTP 500), the chunk pages fetch it fine
  await withTone(async url => {
    const code = `setcps(0.5)
await samples({ longtone: ['${url}'] })
$: s("<longtone ~ ~ ~ ~ ~>")`;
    const r = await render(code, 10, CHUNKED);
    assert.notEqual(r.status, 0, r.log);
    assert.match(r.log, /ring estimate failed/);
    assert.equal(r.out, null);
  }, 3);
});

test('a hiccup of the ring estimate is retried loudly and the long one-shot is not truncated', { skip }, async () => {
  await withTone(async url => {
    const r = await render(`setcps(0.5)\nawait samples({ longtone: ['${url}'] })\n$: s("<longtone ~ ~ ~ ~ ~>")`, 10, CHUNKED);
    assert.match(r.log, /\[CRASH\] ring estimate/);
    assert.equal(r.status, 0, r.log);
    assert.ok(rmsBetween(r.out, 44100, 4.2, 5.8) > 0.1, `tone truncated at 4.2-5.8 s: ${rmsBetween(r.out, 44100, 4.2, 5.8)}`);
  }, 1);
});

test('pattern values are not re-queried from 0 for every chunk (orbits come from the one whole-pattern pass)', { skip }, async () => {
  // a value probe counts evaluations; 16 one-hap cycles in 16 chunks. A per-chunk query of [0, b) alone adds 0+1+..+15 = 120.
  const code = `setcps(0.5)
$: note("c3 ~ ~ ~").s("sine").gain(0.2).withValue(v => { console.log('probe-eval'); return v })`;
  const r = await render(code, 32, CHUNKED, { OFFLINE_VERBOSE: '1' });
  assert.equal(r.status, 0, r.log);
  const probes = (r.log.match(/probe-eval/g) || []).length;
  assert.ok(probes <= 16 * 5, `${probes} pattern evaluations for 16 chunks of 1 hap`);
});

test('an echo tail that outlasts the derived margin still fails loudly', { skip }, async () => {
  const r = await render(`setcps(0.5)\n$: note("c3 ~ ~ ~").s("sine").gain(0.8).delay(1).delaytime(0.25).delayfeedback(0.9)`, 8, CHUNKED);
  assert.notEqual(r.status, 0, r.log);
  assert.match(r.log, /tail margin .* truncates ringing audio/);
});
