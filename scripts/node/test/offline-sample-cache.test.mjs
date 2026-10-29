// The on-disk sample cache (src/sample-cache.ts) and the hang watchdog (offline-helpers.ts), unit level and end to end.
// End-to-end cases need Chromium and the real superdough, and wait for any other render-strudel-offline process (one render at a time).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { SampleCache, SampleFetchError } from '../dist/sample-cache.js';
import { watched, StallError, stallLimitS, STALL_FLOOR_S } from '../dist/offline-helpers.js';

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'sample-cache-'));
const never = () => new Promise(() => {});

// ---- SampleCache (unit) --------------------------------------------------------------------------------------------
function fakeUpstream() {
  const up = { body: Buffer.from('v1'), etag: '"a"', status: 200, calls: [], down: false };
  up.fetchRaw = async (url, headers) => {
    up.calls.push({ url, headers });
    if (up.down) throw new SampleFetchError('down', 'internetdisconnected');
    if (up.status === 200 && headers['if-none-match'] === up.etag) return { status: 304, contentType: '', etag: up.etag, lastModified: '', body: Buffer.alloc(0) };
    return { status: up.status, contentType: 'audio/wav', etag: up.etag, lastModified: '', body: up.body };
  };
  return up;
}
// One cache instance = one render process: it validates a URL at most once (a render sees one consistent version of the pack).
// `later(s)` is the next process, s seconds on, over the same directory.
let clock = 1_000_000;
const mk = (up, o = {}) => { const c = new SampleCache({ dir: o.dir ?? tmp(), ttlS: o.ttlS ?? 600, stallMs: () => 1000, fetchRaw: up.fetchRaw, now: () => clock, ...o }); c.later = s => { clock += s * 1000; return mk(up, { ...o, dir: c.o.dir }); }; return c; };

test('cache: a miss is fetched once and written through; a fresh entry makes no request', async () => {
  const up = fakeUpstream(), c = mk(up);
  const a = await c.get('http://h/x.wav'), b = await c.get('http://h/x.wav');
  assert.equal(a.from, 'network'); assert.equal(b.from, 'cache'); assert.deepEqual(b.body, Buffer.from('v1'));
  assert.equal(up.calls.length, 1);
  // a second process (new instance, same dir) is served from disk
  const c2 = new SampleCache({ dir: c.o.dir, ttlS: 600, stallMs: () => 1000, fetchRaw: up.fetchRaw, now: () => 1_000_000 });
  assert.equal((await c2.get('http://h/x.wav')).from, 'cache'); assert.equal(up.calls.length, 1);
});
test('cache: after the TTL a conditional GET renews an unchanged entry (304) and replaces a changed one (200)', async () => {
  const up = fakeUpstream(), c = mk(up);
  await c.get('http://h/x.wav');
  const c1 = c.later(601);
  const r = await c1.get('http://h/x.wav');
  assert.equal(r.from, 'revalidated'); assert.equal(up.calls.at(-1).headers['if-none-match'], '"a"'); assert.deepEqual(r.body, Buffer.from('v1'));
  const c2 = c1.later(601); up.body = Buffer.from('v2 rebuilt pack'); up.etag = '"b"';   // the pack is re-uploaded under the same URL
  const r2 = await c2.get('http://h/x.wav');
  assert.equal(r2.from, 'network'); assert.deepEqual(r2.body, Buffer.from('v2 rebuilt pack'));
  assert.deepEqual((await c2.get('http://h/x.wav')).body, Buffer.from('v2 rebuilt pack'));
  assert.deepEqual((await c2.later(1).get('http://h/x.wav')).body, Buffer.from('v2 rebuilt pack'));   // fresh again, no request
});
test('cache: an unreachable upstream serves the cached copy; with no copy the failure is loud', async () => {
  const up = fakeUpstream(), c0 = mk(up, { ttlS: 0 });
  await c0.get('http://h/x.wav');
  up.down = true;
  const c = c0.later(1);
  const r = await c.get('http://h/x.wav'); assert.equal(r.from, 'stale'); assert.deepEqual(r.body, Buffer.from('v1'));
  const n = up.calls.length; await c.get('http://h/x.wav'); assert.equal(up.calls.length, n, 'no retry storm: the process already knows the upstream is down');
  await assert.rejects(c.get('http://h/never-fetched.wav'), e => e instanceof SampleFetchError && e.code === 'internetdisconnected');
});
test('cache: 404 and 5xx are passed through, never cached (a missing sound still fails)', async () => {
  const up = fakeUpstream(), c = mk(up);
  up.status = 404; assert.equal((await c.get('http://h/m.wav')).status, 404);
  up.status = 200; assert.equal((await c.get('http://h/m.wav')).status, 200);
});
test('cache: concurrent identical requests share one fetch; concurrent writers leave a valid entry', async () => {
  const up = fakeUpstream(), dir = tmp(), a = mk(up, { dir }), b = mk(up, { dir });
  const rs = await Promise.all([a.get('http://h/z.wav'), a.get('http://h/z.wav'), b.get('http://h/z.wav')]);
  assert.ok(rs.every(r => r.body.equals(Buffer.from('v1'))));
  assert.equal(up.calls.length, 2);                       // one per instance
  assert.ok(fs.readdirSync(dir).every(f => !f.includes('.tmp-')), 'no temp files left behind');
  assert.equal(mk(up, { dir }).read('http://h/z.wav').body.toString(), 'v1');
});

// ---- watchdog (unit) -----------------------------------------------------------------------------------------------
test('stall limit is derived from the audio length and the observed render speed', () => {
  assert.equal(stallLimitS({ slowestStepS: 0, slowestRate: 0 }, 20), STALL_FLOOR_S);
  assert.equal(stallLimitS({ slowestStepS: 5, slowestRate: 4 }, 20), 3 * 80);               // 20 s of audio at 4 s per audio-second
  assert.equal(stallLimitS({ slowestStepS: 500, slowestRate: 4 }, 20), 3 * 500);            // a slow step already seen
  assert.equal(stallLimitS({ slowestStepS: 5, slowestRate: 4 }, 20, 7), 7);                 // explicit override
});
test('a call that never returns is detected within the derived bound; one that keeps ticking is not killed', async () => {
  let last = Date.now(), killed = 0;
  const t0 = Date.now();
  await assert.rejects(watched(never, { label: 'chunk 2/9', limitS: 0.4, pollMs: 50, lastProgress: () => last, onStall: () => killed++ }), e => e instanceof StallError && /chunk 2\/9/.test(e.message));
  const took = Date.now() - t0;
  assert.ok(took >= 400 && took < 1200, `detected after ${took} ms, bound 400 ms`); assert.equal(killed, 1);
  // slow but progressing: 1.2 s of work, ticking every 100 ms, limit 0.4 s
  last = Date.now(); const tick = setInterval(() => { last = Date.now(); }, 100);
  assert.equal(await watched(() => new Promise(r => setTimeout(() => r('done'), 1200)), { label: 'slow', limitS: 0.4, pollMs: 50, lastProgress: () => last }), 'done');
  clearInterval(tick);
});

// ---- end to end ----------------------------------------------------------------------------------------------------
const RENDERER = fileURLToPath(new URL('../dist/render-strudel-offline.js', import.meta.url));
let haveChromium = false;
try { const p = (await import('puppeteer')).default; haveChromium = !!p.executablePath() && fs.existsSync(p.executablePath()); } catch { /* no puppeteer */ }
const skip = !fs.existsSync(RENDERER) || !haveChromium ? 'renderer build or Chromium missing' : false;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitForIdleRenderer() {
  for (let i = 0; i < 360; i++) { if (spawnSync('pgrep', ['-f', '^(nice )?node .*dist/render-strudel-offline\\.js']).status !== 0) return; await sleep(5000); }
}
async function render(code, seconds, extra = [], env = {}, killAfterMs = 0) {
  await waitForIdleRenderer();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'offline-speed-'));
  const src = path.join(dir, 'p.strudel'), out = path.join(dir, 'o.wav');
  fs.writeFileSync(src, code);
  const t0 = Date.now();
  const p = spawn('nice', ['node', RENDERER, src, '-o', out, '-d', String(seconds), ...extra], { env: { ...process.env, ...env } });
  let log = ''; p.stdout.on('data', d => log += d); p.stderr.on('data', d => log += d);
  const killer = killAfterMs ? setTimeout(() => { log += `\n[TEST] killed after ${killAfterMs} ms\n`; spawnSync('pkill', ['-KILL', '-P', String(p.pid)]); p.kill('SIGKILL'); }, killAfterMs) : null;
  const status = await new Promise(r => p.on('close', r));
  clearTimeout(killer);
  return { status, log, wall: Date.now() - t0, out: fs.existsSync(out) ? fs.readFileSync(out) : null };
}
/** A 2 s 220 Hz tone served over http. mode: 'ok' | 'fail' (500) | 'stall' (never answers) | 'etag' (200 with ETag, 304 on match). */
async function withTone(fn) {
  const sr = 22050, n = sr * 2, b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 220 / sr) * 0.4 * 32767), 44 + i * 2);
  const srv = { mode: 'ok', hits: 0, conditional: 0 };
  const s = http.createServer((q, r) => {
    srv.hits++;
    if (q.headers['if-none-match']) srv.conditional++;
    if (srv.mode === 'stall') return;
    if (srv.mode === 'fail') { r.writeHead(500, { 'access-control-allow-origin': '*' }); return r.end('no'); }
    if (q.headers['if-none-match'] === '"t1"') { r.writeHead(304, { 'access-control-allow-origin': '*' }); return r.end(); }
    r.writeHead(200, { 'content-type': 'audio/wav', etag: '"t1"', 'access-control-allow-origin': '*' }); r.end(b);
  });
  await new Promise(r => s.listen(0, '127.0.0.1', r));
  try { return await fn(`http://127.0.0.1:${s.address().port}/tone.wav`, srv); } finally { s.closeAllConnections?.(); s.close(); }
}
const CODE = url => `setcps(0.5)\nawait samples({ tone: ['${url}'] })\n$: s("<tone ~>")`;
const CHUNKED = ['--chunk-seconds', '2', '--preroll-seconds', '0.6', '--tail-seconds', '1'];

test('a render whose sample server fails or stalls after the first fetch completes from the cache; the second render makes 0 requests', { skip }, async () => {
  await withTone(async (url, srv) => {
    const env = { MIDIGREP_RENDER_SAMPLE_CACHE_DIR: tmp(), MIDIGREP_RENDER_SAMPLE_CACHE_TTL_S: '600' };
    const cold = await render(CODE(url), 8, CHUNKED, env);
    assert.equal(cold.status, 0, cold.log);
    const hits = srv.hits;
    assert.ok(hits >= 1);
    for (const mode of ['fail', 'stall']) {
      srv.mode = mode;
      const warm = await render(CODE(url), 8, CHUNKED, env);
      assert.equal(warm.status, 0, `${mode}: ${warm.log}`);
      assert.equal(srv.hits, hits, `${mode}: the warm render hit the sample server`);
      assert.ok(warm.out.equals(cold.out), `${mode}: warm audio differs from cold`);
    }
  });
});

test('an outdated entry is revalidated (304) and an unreachable upstream serves the cached copy; MIDIGREP_RENDER_SAMPLE_CACHE=0 bypasses', { skip }, async () => {
  await withTone(async (url, srv) => {
    const env = { MIDIGREP_RENDER_SAMPLE_CACHE_DIR: tmp(), MIDIGREP_RENDER_SAMPLE_CACHE_TTL_S: '0' };
    assert.equal((await render(CODE(url), 4, CHUNKED, env)).status, 0);
    const h0 = srv.hits;
    const re = await render(CODE(url), 4, CHUNKED, env);
    assert.equal(re.status, 0, re.log); assert.equal(srv.hits, h0 + 1); assert.equal(srv.conditional, 1);
    assert.match(re.log, /[1-9]\d* revalidated \(304\)/);
    srv.mode = 'fail';                                  // 500 on revalidation: stale-if-error
    const st = await render(CODE(url), 4, CHUNKED, env); assert.equal(st.status, 0, st.log); assert.match(st.log, /serving the cached copy/);
    const off = await render(CODE(url), 4, CHUNKED, { ...env, MIDIGREP_RENDER_SAMPLE_CACHE: '0' });
    assert.notEqual(off.status, 0, 'bypass must go to the (failing) network');
  });
});

test('a hung page is detected within the derived bound, the browser is replaced and the chunk is retried', { skip }, async () => {
  const code = `setcps(0.5)\n$: note("c3 ~ ~ ~").s("sine").gain(0.2)`;
  const r = await render(code, 8, CHUNKED, { MIDIGREP_RENDER_TEST_HANG: 'chunk:2', MIDIGREP_RENDER_STALL_S: '4', MIDIGREP_RENDER_SAMPLE_CACHE_DIR: tmp() });
  assert.match(r.log, /\[STALL\] chunk 2\//);
  assert.match(r.log, /\[CRASH\] chunk 2\/\d+ .*attempt 1\/3: page stalled: chunk 2\/\d+ made no progress for [\d.]+ s \(limit 4\.0 s/);
  assert.equal(r.status, 0, r.log);
  assert.ok(r.wall < 120_000, `took ${r.wall} ms`);
});

test('a frozen browser (page AND every CDP call hang) is detected at the derived limit, not at protocolTimeout', { skip }, async () => {
  const code = `setcps(0.5)\n$: note("c3 ~ ~ ~").s("sine").gain(0.2)`;
  const r = await render(code, 8, CHUNKED, { MIDIGREP_RENDER_TEST_HANG: 'chunk:2:freeze', MIDIGREP_RENDER_STALL_S: '4', MIDIGREP_RENDER_SAMPLE_CACHE_DIR: tmp() });
  assert.match(r.log, /\[STALL\] chunk 2\//);
  const m = /\[CRASH\] chunk 2\/\d+ .*attempt 1\/3: page stalled: chunk 2\/\d+ made no progress for ([\d.]+) s \(limit 4\.0 s/.exec(r.log);
  assert.ok(m, r.log);
  assert.ok(Number(m[1]) < 4 + 5, `stall noticed after ${m[1]} s, limit 4 s`);
  assert.equal(r.status, 0, r.log);
  assert.ok(r.wall < 120_000, `took ${r.wall} ms`);
});

test('a page spinning in a busy loop (browser CPU busy, main thread dead) is detected at the derived limit, not at protocolTimeout', { skip, timeout: 240_000 }, async () => {
  const code = `setcps(0.5)\n$: note("c3 ~ ~ ~").s("sine").gain(0.2)`;
  // killAfterMs: on the unfixed watchdog this render sits for protocolTimeout (900 s); the test must not wait for it
  const r = await render(code, 8, CHUNKED, { MIDIGREP_RENDER_TEST_HANG: 'chunk:2:spin', MIDIGREP_RENDER_STALL_S: '4', MIDIGREP_RENDER_SAMPLE_CACHE_DIR: tmp() }, 90_000);
  assert.doesNotMatch(r.log, /\[TEST\] killed/, `no stall detected within 90 s (limit 4 s):\n${r.log.slice(-600)}`);
  assert.match(r.log, /\[STALL\] chunk 2\//);
  const m = /page stalled: chunk 2\/\d+ made no progress for ([\d.]+) s \(limit 4\.0 s/.exec(r.log);
  assert.ok(m, r.log);
  assert.ok(Number(m[1]) < 4 + 5, `stall noticed after ${m[1]} s, limit 4 s`);
  assert.equal(r.status, 0, r.log);
  assert.ok(r.wall < 120_000, `took ${r.wall} ms`);
});
