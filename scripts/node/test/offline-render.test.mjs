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
    const r = spawnSync('pgrep', ['-f', 'render-strudel-offline']);
    if (r.status !== 0) return;
    await sleep(5000);
  }
}
async function render(code, seconds, extra = []) {
  await waitForIdleRenderer();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'offline-render-'));
  const src = path.join(dir, 'p.strudel'), out = path.join(dir, 'o.wav');
  fs.writeFileSync(src, code);
  const p = spawn('nice', ['node', RENDERER, src, '-o', out, '-d', String(seconds), ...extra]);
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

test('a duck target no earlier hap ever used still fails loudly', { skip }, async () => {
  const code = `setcps(0.5)
$: note("c2 ~ ~ ~").s("sine").duckorbit(3).gain(0.4)`;
  const r = await render(code, 4, CHUNKED);
  assert.notEqual(r.status, 0);
  assert.match(r.log, /duck target orbit 3 does not exist/);
});

test('a one-shot longer than the tail margin is reproduced whole, and a missing sound still fails', { skip }, async () => {
  // 6 s decay-free tone served over http: it rings through three 2 s chunks, far past the 1 s margin
  const sr = 22050, n = sr * 6, b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 220 / sr) * 0.4 * 32767), 44 + i * 2);
  const srv = http.createServer((q, s) => { s.writeHead(200, { 'content-type': 'audio/wav', 'access-control-allow-origin': '*' }); s.end(b); });
  await new Promise(r => srv.listen(0, '127.0.0.1', r));
  try {
    const url = `http://127.0.0.1:${srv.address().port}/long.wav`;
    const code = `setcps(0.5)
await samples({ longtone: ['${url}'] })
$: s("<longtone ~ ~ ~ ~ ~>")`;
    const r = await render(code, 10, CHUNKED);
    assert.equal(r.status, 0, r.log);
    // the tone starts at 0 s and rings to 6 s: the 3rd chunk (4..6 s) can only have it through the derived look-back
    const wav = r.out;
    assert.ok(rmsBetween(wav, 44100, 4.2, 5.8) > 0.1, `tone missing at 4.2-5.8 s: ${rmsBetween(wav, 44100, 4.2, 5.8)}`);
    assert.ok(rmsBetween(wav, 44100, 6.5, 9.5) < 0.01);
    const miss = await render(`setcps(0.5)\n$: s("nosuchsound ~ ~ ~")`, 4, CHUNKED);
    assert.notEqual(miss.status, 0);
    assert.match(miss.log, /not found|failed/i);
  } finally { srv.close(); }
});

test('an echo tail that outlasts the derived margin still fails loudly', { skip }, async () => {
  const r = await render(`setcps(0.5)\n$: note("c3 ~ ~ ~").s("sine").gain(0.8).delay(1).delaytime(0.25).delayfeedback(0.9)`, 8, CHUNKED);
  assert.notEqual(r.status, 0, r.log);
  assert.match(r.log, /tail margin .* truncates ringing audio/);
});
