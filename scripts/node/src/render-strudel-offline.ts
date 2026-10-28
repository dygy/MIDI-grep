#!/usr/bin/env node
/**
 * OFFLINE Strudel render — the real superdough engine, no BlackHole, no real-time wait.
 *
 * Runs the same .strudel code the BlackHole recorder plays (setcps, await samples(...), `$:` blocks,
 * mini-notation) in a headless Chromium page that loads the pinned @strudel/web 1.3.0 IIFE bundle
 * (core/mini/tonal/transpiler 1.2.6 + webaudio 1.3.0 + superdough 1.3.0 inside), evaluates the code
 * with the REPL's own evaluate (no scheduler start), then calls the bundle's renderPatternAudio on
 * an OfflineAudioContext. Bar 0 is t=0, so NO leading-silence trim is applied (the recorder trims
 * because ffmpeg starts before Strudel loads).
 *
 * Usage: node dist/render-strudel-offline.js input.strudel -o out.wav -d 150 [--sample-rate 44100] [--allow-failed-samples]
 */
import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import { createRequire } from 'module';
import { chunkMargins, Ring, analyzeWav, checkRender, closeAllWithin, planChunks, expectedFrames, floatToWav16, overlapAdd, sampleFailures, triggerArgs, prerollPlan, trimChunk, removeStaleOutput, commitOutput, isNetworkFailure, isOutageSampleFailure, isNetworkOutageFailure, netWaitBudgetS, nextNetWaitS } from './offline-helpers.js';

const SHIM = `const S = window.strudel;
export const { noteToMidi, freqToMidi, getSoundIndex, getAudioContext, registerSound, getParamADSR, getADSRValues, getPitchEnvelope, getVibratoOscillator } = S;`;
const SITE = 'https://strudel.dygy.app'; // same hosted sample maps the REPL prebake uses (BASE_URL "/")

const PAGE_HTML = String.raw`<!doctype html><meta charset="utf-8"><title>offline</title>
<script src="/strudel.js"></script>
<script>
window.__triggerArgs = ${triggerArgs.toString()};
window.__fontsP = import("/fontloader.mjs").then(m => m.registerSoundfonts);

window.__logs = [];
document.addEventListener('strudel.log', e => window.__logs.push(String(e.detail && e.detail.message)));
for (const k of ['error', 'warn']) { const o = console[k].bind(console); console[k] = (...a) => { window.__logs.push(k + ': ' + a.map(String).join(' ')); o(...a); }; }
window.addEventListener('unhandledrejection', e => window.__logs.push('error: unhandledrejection ' + (e.reason && e.reason.message || e.reason)));
window.__getSlice = (i, n) => {
  const u = window.__bytes.subarray(i, i + n); let s = '';
  for (let k = 0; k < u.length; k += 0x8000) s += String.fromCharCode.apply(null, u.subarray(k, k + 0x8000));
  return btoa(s);
};
// 1) init: engine + prebake + evaluate. Returns cps. The pattern stays in this page.
window.__count = (cycles) => window.__pattern.queryArc(0, cycles, { _cps: window.__cps }).filter(h => h.hasOnset()).length;
window.__density = (cycles) => { const n = Math.ceil(cycles - 1e-9), d = new Array(n).fill(0); for (const h of window.__pattern.queryArc(0, cycles, { _cps: window.__cps })) if (h.hasOnset()) { const c = Math.floor(h.whole.begin.valueOf() + 1e-9); if (c >= 0 && c < n) d[c]++; } return d; };
// Ring estimate per onset hap, in SECONDS: [onset, end of the audible voice]. A sample with no clip/loop/release plays its
// whole (speed-scaled) slice regardless of the hap length (superdough onTriggerSample: duration = sliceDuration), so a
// one-shot that is 14 s long rings 14 s; a synth or a held/looped sample ends with the hap + its release. Effect tails
// (reverb, delay) are NOT included: the caller adds its margin on top. A sample that cannot be fetched/decoded gets
// the hap length only - the chunk render then reports the load failure loudly.
window.__rings = async (cycles) => {
  const S = strudel, cps = window.__cps, out = [], dur = {};
  for (const h of window.__pattern.queryArc(0, cycles, { _cps: cps })) {
    if (!h.hasOnset()) continue;
    h.ensureObjectValue(); const v = h.value;
    const onset = h.whole.begin.valueOf() / cps, hapDur = h.duration / cps;
    let len = hapDur + (v.release ?? 0);
    const name = v.bank && v.s ? v.bank + '_' + v.s : v.s;
    const snd = typeof name === 'string' ? S.getSound(name) : undefined;
    const bank = snd && snd.data && snd.data.type === 'sample' ? snd.data.samples : null;
    if (bank && v.speed !== 0) {
      try {
        const info = S.getSampleInfo(v, bank);
        if (dur[info.url] === undefined) dur[info.url] = (await S.getSampleBuffer(v, bank)).buffer.duration;
        const rate = info.playbackRate * (v.unit === 'c' ? dur[info.url] : 1);
        const slice = ((v.end ?? 1) - (v.begin ?? 0)) * dur[info.url] / rate;
        const sustained = v.clip != null || v.loop != null || v.release != null;
        len = Math.max(hapDur, sustained ? 0 : slice) + (v.release ?? 0);
      } catch (_) { /* unloadable sample: the chunk render reports it */ }
    }
    out.push([onset, onset + len]);
  }
  return out;
};
window.__init = async (code, site) => {
  const S = strudel;
  const prebake = async () => {
    const o = { prebake: true };
    await Promise.all([
      S.registerZZFXSounds(),
      (async () => (await window.__fontsP)())(), // General-MIDI soundfonts (gm_*), as the site's prebake does
      S.samples(site + '/piano.json', undefined, o),
      S.samples(site + '/vcsl.json', undefined, o),
      S.samples(site + '/vsco2.json', undefined, { ...o, tag: 'orchestral' }),
      S.samples(site + '/tidal-drum-machines.json', 'github:ritchse/tidal-drum-machines/main/machines/', { ...o, tag: 'drum-machines' }),
      S.samples(site + '/uzu-drumkit.json', undefined, { ...o, tag: 'drum-machines' }),
      S.samples(site + '/mridangam.json', undefined, { ...o, tag: 'drum-machines' }),
    ]);
    const alias = await (await fetch(site + '/tidal-drum-machines-alias.json')).json();
    S.aliasBank(alias);
  };
  const repl = await S.initStrudel({ prebake });
  window.__logs.length = 0; // only count failures from evaluate/render, not init chatter
  // Compat shim: this bundled transpiler turns EVERY "double-quoted" string into mini(), including the
  // URL in samples("https://..."), which then fails to parse. Single-quoted strings are left alone, so
  // rewrite samples("<url>") -> samples('<url>') (same string, no other change to the code).
  let rewrites = 0;
  code = code.replace(/\bsamples\(\s*"([^"']*)"/g, (m, u) => { rewrites++; return "samples('" + u + "'"; });
  const pattern = await repl.evaluate(code, false);
  if (!pattern) throw new Error('evaluate returned no pattern (eval error): ' + window.__logs.join(' | '));
  window.__pattern = pattern; window.__cps = repl.scheduler.cps;
  return { cps: window.__cps, rewrites, countOnsets: undefined };
};
// 2a) single pass (reference): the bundle's own renderPatternAudio, download intercepted -> WAV bytes
window.__single = async (cycles, sampleRate, poly) => {
  const S = strudel; let blob = null;
  const create = URL.createObjectURL.bind(URL);
  URL.createObjectURL = b => { blob = b; return create(b); };
  HTMLAnchorElement.prototype.click = function () { /* swallow the download */ };
  const onsets = window.__pattern.queryArc(0, cycles, { _cps: window.__cps }).filter(h => h.hasOnset()).length;
  await S.renderPatternAudio(window.__pattern, window.__cps, 0, cycles, sampleRate, poly, false, 'offline');
  if (!blob) throw new Error('renderPatternAudio produced no blob');
  window.__bytes = new Uint8Array(await blob.arrayBuffer());
  return { bytes: window.__bytes.length, onsets, logs: window.__logs.slice(), errs: [] };
};
// 2b) one chunk: onsets from cycles [b,e) only, context extended by len frames (tail), same engine
// path as renderPatternAudio. The context's frame 0 is song frame startFrame-preFrames: haps with onsets in
// [preCycle,b) are PRE-ROLL — triggered so cut groups / orbit ducking see the previous chunk's ringing voices —
// and their audio (the first preFrames) is discarded by the caller. cycle is the hap's absolute cycle.
window.__chunk = async (b, e, sampleRate, startFrame, len, poly, preFrames, preCycle, nearCycle, ringingOnsets) => {
  const S = strudel, cps = window.__cps, pattern = window.__pattern;
  const Ctrl = S.getSuperdoughAudioController().constructor; // class is not exported; reuse the live one's constructor
  const hap2value = h => { h.ensureObjectValue(); return h.value; };
  const ctxStartSec = (startFrame - preFrames) / sampleRate;
  try { await S.getAudioContext().close(); } catch (_) {}
  const off = new OfflineAudioContext(2, len, sampleRate);
  S.setAudioContext(off); S.setSuperdoughAudioController(new Ctrl(off));
  await S.initAudio({ maxPolyphony: poly, multiChannelOrbits: false });
  const byBegin = (x, y) => x.whole.begin.valueOf() - y.whole.begin.valueOf();
  // pre-roll = every onset in [nearCycle, b) (cut groups, ducking) + the OLDER onsets whose voice still sounds at b (ringingOnsets, seconds)
  const ringing = new Set(ringingOnsets.map(x => Math.round(x * 1e6)));
  const isPre = h => h.hasOnset() && h.whole.begin.valueOf() < b && (h.whole.begin.valueOf() >= nearCycle || ringing.has(Math.round((h.whole.begin.valueOf() / cps) * 1e6)));
  const pre = preFrames > 0 ? pattern.queryArc(preCycle, b, { _cps: cps }).filter(isPre).sort(byBegin) : [];
  const haps = pattern.queryArc(b, e, { _cps: cps }).sort(byBegin);
  // An orbit is created by the first hap that plays on it and then lives on, so a duck target (.duckorbit(n)) exists in
  // continuous playback from that hap on. A chunk starts mid-song: orbits that earlier haps of the pattern used must exist
  // here too, or a kick at the start of the window finds no target before the first hap of its own on that orbit.
  // A silent '-' hap goes through superdough's own orbit creation (it returns right after it). An orbit that NO earlier hap
  // used is still missing, as in continuous playback, and still fails loudly.
  if (b > 0) {
    const used = new Set();
    for (const h of pattern.queryArc(0, b, { _cps: cps })) if (h.hasOnset() && h.whole.begin.valueOf() < b) { h.ensureObjectValue(); used.add(h.value.orbit ?? 1); }
    for (const orbit of used) await S.superdough({ s: '-', orbit }, 0, 0, cps, 0);
  }
  let onsets = 0; const errs = [];
  for (const hap of pre.concat(haps)) {
    if (!hap.hasOnset()) continue;
    const isMain = hap.whole.begin.valueOf() >= b;
    if (isMain) onsets++;
    const a = window.__triggerArgs(hap.whole.begin.valueOf(), cps, ctxStartSec);
    try { await S.superdough(hap2value(hap), a.t, hap.duration / cps, cps, a.cycle); }
    catch (err) { const m = 'error: ' + (err && err.message || err) + ' hap=' + JSON.stringify(hap.value).slice(0, 160); window.__logs.push(m); if (errs.length < 2) errs.push(m); }
  }
  const buf = await off.startRendering();
  const f = new Float32Array(len * 2); f.set(buf.getChannelData(0), 0); f.set(buf.getChannelData(1), len);
  window.__bytes = new Uint8Array(f.buffer);
  return { bytes: window.__bytes.length, onsets, logs: window.__logs.slice(), errs };
};
</script>`;

/** superdough duck(): webAudioTimeout(..., t - 0.01) - a hap must sit at least this far into the context (+ margin for rounding). */
const DUCK_GUARD_S = 0.02;

async function main() {
  const args = process.argv.slice(2);
  let input = '', outputPath = 'offline.wav', duration = 30, sampleRate = 44100, allowFailures = false;
  let chunkSeconds = 20, prerollSeconds = -1, maxChunkOnsets = 1200, tailSeconds = 4, maxPolyphony = 4096; // NOT the REPL's 128: offline, currentTime is 0 while every hap is scheduled, so the cap counts ALL voices and drops most of the music (measured rms 0.050 at 128 vs 0.099 at 4096 on 40 s of v004; BlackHole live = 0.100) // chunkSeconds 0 = single pass (reference)
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '-o') outputPath = args[++i];
    else if (args[i] === '-d') duration = parseFloat(args[++i]);
    else if (args[i] === '--sample-rate') sampleRate = parseInt(args[++i], 10);
    else if (args[i] === '--chunk-seconds') chunkSeconds = parseFloat(args[++i]);
    else if (args[i] === '--max-chunk-onsets') maxChunkOnsets = parseInt(args[++i], 10);
    else if (args[i] === '--max-polyphony') maxPolyphony = parseInt(args[++i], 10);
    else if (args[i] === '--tail-seconds') tailSeconds = parseFloat(args[++i]);
    else if (args[i] === '--preroll-seconds') prerollSeconds = parseFloat(args[++i]);
    else if (args[i] === '--allow-failed-samples') allowFailures = true;
    else if (!input) input = args[i];
  }
  if (!input || !fs.existsSync(input)) { console.error('Usage: render-strudel-offline <file.strudel> -o out.wav -d <seconds>'); process.exit(2); }
  if (!(duration > 0)) { console.error('ERROR: -d must be > 0'); process.exit(2); }
  if (prerollSeconds < 0) prerollSeconds = tailSeconds; // pre-roll covers the same window the tail margin does
  removeStaleOutput(outputPath); // a file from an earlier render must never survive a failed one
  const code = fs.readFileSync(input, 'utf8');
  const require = createRequire(import.meta.url);
  // MIDIGREP_STRUDEL_BUNDLE=<path to an @strudel/web-compatible dist/index.js> selects another engine build (e.g. the
  // fork with the master bus); unset = the pinned npm @strudel/web. The choice is logged ("Engine: ... [source]").
  const envBundle = (process.env.MIDIGREP_STRUDEL_BUNDLE || '').trim();
  const bundlePath = envBundle ? path.resolve(envBundle) : path.join(path.dirname(require.resolve('@strudel/web/package.json')), 'dist', 'index.js');
  if (envBundle && !fs.existsSync(bundlePath)) { console.error(`ERROR: MIDIGREP_STRUDEL_BUNDLE=${envBundle}: no such file`); process.exit(2); }
  const sfDir = path.dirname(require.resolve('@strudel/soundfonts/package.json'));
  const engineVersions = ['@strudel/web', '@strudel/core', '@strudel/webaudio', 'superdough', '@strudel/soundfonts']
    .map(p => { try { return `${p}@${JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve('@strudel/web/package.json')), '..', '..', p, 'package.json'), 'utf8')).version}`; } catch { return `${p}@?`; } });
  console.log(`Engine: bundle ${bundlePath} [${envBundle ? 'MIDIGREP_STRUDEL_BUNDLE' : 'npm @strudel/web'}] (${envBundle ? 'versions of the npm packages are NOT this bundle: ' : ''}${engineVersions.join(', ')}); site sample maps from ${SITE}; GM soundfonts from the pinned package via https://felixroos.github.io/webaudiofontdata/sound (the site's default)`);

  const server = http.createServer((req, res) => {
    const js = (body: string | Buffer) => { res.writeHead(200, { 'content-type': 'text/javascript' }); res.end(body); };
    if (req.url === '/strudel.js') { res.writeHead(200, { 'content-type': 'text/javascript' }); fs.createReadStream(bundlePath).pipe(res); }
    else if (req.url === '/gm.mjs') js(fs.readFileSync(path.join(sfDir, 'gm.mjs')));
    else if (req.url === '/strudel-shim.mjs') js(SHIM);
    // the pinned package's own fontloader, only the bare imports are pointed at the bundle's exports
    else if (req.url === '/fontloader.mjs') js(fs.readFileSync(path.join(sfDir, 'fontloader.mjs'), 'utf8').replace(/from '@strudel\/(core|webaudio)'/g, "from '/strudel-shim.mjs'"));
    else { res.writeHead(200, { 'content-type': 'text/html' }); res.end(PAGE_HTML); }
  });
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const port = (server.address() as any).port;

  const t0 = Date.now();
  // EVERY browser this run launches (initial + crash relaunches) is tracked and closed at the end.
  const launched: any[] = [];
  const launch = async () => {
    const b = await puppeteer.launch({ headless: true, protocolTimeout: 900000, args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--disable-web-security'] });
    launched.push(b); return b;
  };
  let browser = await launch();
  let exit = 0;
  // A render step that dies with the page/browser ("Target closed") is retried on a fresh page (and a fresh
  // browser if it is gone). Every attempt is reported with WHAT it held; after MAX_ATTEMPTS it fails loudly.
  const MAX_ATTEMPTS = 3;
  const quickClose = (pg: any) => Promise.race([pg.close().catch(() => {}), new Promise(r => setTimeout(r, 5000))]); // a dead page's close() must not hang the run
  // NETWORK outages (net::ERR_INTERNET_DISCONNECTED / TIMED_OUT / ... seen in the browser console, or in the error) are NOT
  // crashes: they are waited out with exponential backoff (5 s, 10 s, ... cap 60 s) for a total of MIDIGREP_RENDER_NET_WAIT_S
  // seconds (default 1800; 0 = fail at once), logged each time, then retried on a fresh page. They do not use up MAX_ATTEMPTS.
  // Anything else keeps the fail-loudly path; the WAV is still only written after every check passed.
  const netBudgetS = netWaitBudgetS(process.env.MIDIGREP_RENDER_NET_WAIT_S);
  let netEvents = 0, netWaitedS = 0;
  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
  const attempt = async <T>(label: string, held: string, fn: () => Promise<T>, sampleLogs?: (v: T) => string[]): Promise<T> => {
    for (let k = 1, net = 0; ; ) {
      const ev0 = netEvents;
      try {
        const v = await fn();
        // a sample that failed to load while the network was down renders silently wrong: treat as network, retry
        const bad = sampleLogs ? sampleFailures(sampleLogs(v)) : [];
        const loadBad = bad.filter(isOutageSampleFailure); // only a failed load/fetch is attributable to the outage, not any logged failure
        if (loadBad.length && netEvents > ev0) throw new Error(`sample load failed during network outage: ${loadBad[0]}`);
        return v;
      } catch (err: any) {
        const isNet = isNetworkOutageFailure(String(err.message), netEvents - ev0); // the attempt's OWN failure decides, not stray network noise
        if (isNet) {
          const w = nextNetWaitS(net, netWaitedS, netBudgetS);
          console.log(`[NETWORK] ${label} (${held}): ${err.message.slice(0, 160)}; browser ${browser.connected ? 'alive' : 'gone'}; ` +
            (w > 0 ? `waiting ${w} s (outage retry ${++net}, ${netWaitedS.toFixed(0)}/${netBudgetS} s of MIDIGREP_RENDER_NET_WAIT_S used)` : `wait budget ${netBudgetS} s (MIDIGREP_RENDER_NET_WAIT_S) exhausted`));
          if (w <= 0) throw new Error(`${label} failed: network outage lasted beyond ${netBudgetS} s (${held}): ${err.message}`);
          await sleep(w * 1000); netWaitedS += w;
          if (!browser.connected) browser = await launch();
          continue;
        }
        console.log(`[CRASH] ${label} (${held}) attempt ${k}/${MAX_ATTEMPTS}: ${err.message}; browser ${browser.connected ? 'alive' : 'gone'}`);
        if (k >= MAX_ATTEMPTS) throw new Error(`${label} failed ${MAX_ATTEMPTS}x (${held}): ${err.message}`);
        k++;
        if (!browser.connected) browser = await launch();
      }
    }
  };
  const openPage = async () => {
    const page = await browser.newPage();
    page.on('error', e => console.log(`[PAGE CRASH] ${e.message}`));
    page.on('pageerror', e => console.log(`[PAGE ERROR] ${e.message}`));
    page.on('console', m => { const t = m.text(); if (isNetworkFailure(t)) netEvents++; if (process.env.OFFLINE_VERBOSE || /error|fail/i.test(t)) console.log(`[BROWSER] ${t.slice(0, 200)}`); });
    await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });
    return page;
  };
  const fetchBytes = async (page: any, n: number): Promise<Buffer> => {
    const parts: Buffer[] = [];
    for (let o = 0; o < n; o += 3 * 1024 * 1024) parts.push(Buffer.from(await page.evaluate((i: number, k: number) => (window as any).__getSlice(i, k), o, 3 * 1024 * 1024), 'base64'));
    return Buffer.concat(parts);
  };
  try {
    // the first page is opened inside an attempt: its navigation can time out like any later one (a stalled
    // remote script/sample map holds 'load'), and that must be waited out, not fail the render outright
    let page = await attempt('open', 'first page', openPage);
    let initTry = 0;
    const init: any = await attempt('init', 'first page', async () => { if (initTry++ > 0) { await quickClose(page); page = await openPage(); } return page.evaluate((c, site) => (window as any).__init(c, site), code, SITE); });
    const cps: number = init.cps, cycles = duration * cps;
    const total = Math.floor((cycles / cps) * sampleRate);
    console.log(`samples("url") rewritten to single quotes: ${init.rewrites}`);
    console.log(`cps ${cps} (${(cps * 240).toFixed(2)} BPM @4/4), ${cycles.toFixed(2)} cycles`);
    const expectedOnsets: number = await page.evaluate((cy) => (window as any).__count(cy), cycles);
    const logs: string[] = []; const errs: string[] = [];
    let onsets = 0, chunks = 0, tailEndPeak = 0, wav: Buffer;
    if (!(chunkSeconds > 0)) {
      console.log(`Rendering ${duration} s offline at ${sampleRate} Hz (single pass) ...`);
      const r: any = await page.evaluate((cy, sr, po) => (window as any).__single(cy, sr, po), cycles, sampleRate, maxPolyphony);
      wav = await fetchBytes(page, r.bytes); onsets = r.onsets; logs.push(...r.logs); chunks = 1;
    } else {
      // Chunked: one FRESH PAGE per chunk. superdough pools audio nodes at module level and returns
      // them keyed without regard to context, so a second OfflineAudioContext in the same page
      // reuses nodes of the first ("cannot connect to an AudioNode belonging to a different audio
      // context"); a new page also frees all decoded buffers/graphs (page memory grew ~1.1 GB for 150 s).
      const density: number[] = await page.evaluate((cy) => (window as any).__density(cy), cycles);
      const rings: Ring[] = ((await page.evaluate((cy) => (window as any).__rings(cy), cycles)) as number[][]).map(([begin, end]) => ({ begin, end }));
      const wins = planChunks(density, cycles, Math.max(1, Math.round(chunkSeconds * cps)), maxChunkOnsets);
      console.log(`Rendering ${duration} s offline at ${sampleRate} Hz (chunked: ${wins.length} windows, <= ${Math.max(1, Math.round(chunkSeconds * cps))} cycles and <= ${maxChunkOnsets} onsets each, ${tailSeconds} s effect-tail margin + ring of the longest voice (longest ring ${Math.max(0, ...rings.map(r => r.end - r.begin)).toFixed(2)} s), ${prerollSeconds} s minimum pre-roll, fresh page each) ...`);
      const L = new Float32Array(total), R = new Float32Array(total);
      for (let ci = 0; ci < wins.length; ci++) {
        const [b, e] = wins[ci];
        const startFrame = Math.floor((b / cps) * sampleRate);
        const m = chunkMargins(rings, b / cps, e / cps, tailSeconds, prerollSeconds);
        const plan = prerollPlan(b, startFrame, cps, sampleRate, m.prerollSeconds);
        const { preCycle } = plan;
        // lead-in: superdough's duck() schedules its gain ramp with stop(t - 0.01), which throws for a ducking hap at context
        // time < 0.01 s (a pre-roll hap at the window start, or a kick at t = 0). The context starts a few ms earlier so no hap does.
        const preFrames = plan.preFrames + Math.ceil(DUCK_GUARD_S * sampleRate);
        const nextStart = ci + 1 < wins.length ? Math.floor((wins[ci + 1][0] / cps) * sampleRate) : total;
        const keep = nextStart - startFrame; // this chunk owns [startFrame, nextStart); later chunks re-trigger its ringing voices as pre-roll
        const len = preFrames + Math.max(1, Math.round(((e - b) / cps) * sampleRate + m.tailSeconds * sampleRate));
        const held = `${density.slice(b, Math.ceil(e)).reduce((x, y) => x + y, 0)} haps over cycles [${b.toFixed(2)}, ${e.toFixed(2)}), ctx ${len} frames`;
        let heapMb = 0;
        const { r, raw } = await attempt(`chunk ${ci + 1}/${wins.length}`, held, async () => {
          await quickClose(page); page = await openPage();
          await page.evaluate((c, site) => (window as any).__init(c, site), code, SITE);
          const rr: any = await page.evaluate((b0, e0, sr, sf, ln, po, pf, pc, nc, rg) => (window as any).__chunk(b0, e0, sr, sf, ln, po, pf, pc, nc, rg), b, e, sampleRate, startFrame, len, maxPolyphony, preFrames, preCycle, Math.max(0, b - prerollSeconds * cps), m.ringingOnsets);
          const bytes = await fetchBytes(page, rr.bytes);
          try { heapMb = (await page.metrics()).JSHeapUsedSize! / 1048576; } catch { /* page gone */ }
          return { r: rr, raw: bytes };
        }, v => [...(v.r.logs || []), ...(v.r.errs || [])]);
        const f = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
        // tail-truncation check: peak over the last 0.25 s of this chunk's tail margin (skip the last chunk: file ends there); measured on the context before trimming
        if (ci < wins.length - 1) for (let i = Math.max(0, len - Math.round(0.25 * sampleRate)); i < len; i++) tailEndPeak = Math.max(tailEndPeak, Math.abs(f[i]), Math.abs(f[len + i]));
        overlapAdd(L, R, trimChunk(f, len, preFrames, keep), startFrame);
        onsets += r.onsets; logs.push(...r.logs); errs.push(...r.errs); chunks++;
        console.log(`  chunk ${ci + 1}/${wins.length} cycles [${b.toFixed(2)}, ${e.toFixed(2)}) ${r.onsets} onsets, js heap ${heapMb.toFixed(0)} MB (${((Date.now() - t0) / 1000).toFixed(1)} s elapsed)`);
      }
      wav = floatToWav16(L, R, sampleRate);
    }
    const info = analyzeWav(new Uint8Array(wav));
    const failures = sampleFailures(logs);
    const exp = expectedFrames(cycles, cps, sampleRate);
    console.log(`${onsets} onsets, ${chunks} render window(s), peak in last 0.25 s of tail margins ${tailEndPeak.toExponential(2)}`);
    console.log(`Output: ${info.frames} frames = ${info.seconds.toFixed(3)} s @ ${info.sampleRate} Hz x${info.channels} (${info.bitsPerSample}-bit fmt ${info.format}), rms ${info.rms.toFixed(5)}, peak ${info.peak.toFixed(3)}; expected ${exp} frames`);
    if (errs.length) console.log(`first trigger errors:\n  ${errs.slice(0, 2).join('\n  ')}`);
    console.log(`Sample/trigger failures: ${failures.length}${failures.length ? '\n  ' + failures.join('\n  ') : ''}`);
    console.log('No leading-silence trim: bar 0 is t=0 (offline render has no capture start-up latency).');
    const chk = checkRender(info, { sampleRate, expectedFrames: exp, failures, allowFailures });
    if (tailEndPeak > 1e-3) { chk.problems.push(`tail margin ${tailSeconds} s truncates ringing audio (peak ${tailEndPeak.toExponential(2)} at its end) — raise --tail-seconds`); chk.ok = false; }
    if (onsets !== expectedOnsets) { chk.problems.push(`triggered ${onsets} onsets but the pattern has ${expectedOnsets} in range (chunk boundary loss?)`); chk.ok = false; }
    if (onsets === 0) { chk.problems.push('pattern has 0 onsets in range'); chk.ok = false; }
    console.log(`Wall time: ${((Date.now() - t0) / 1000).toFixed(1)} s for ${duration} s of audio`);
    // temp file beside the target, renamed only if every check passed; a failed render leaves NO file
    if (!commitOutput(outputPath, wav, () => chk.ok)) { console.error(`ERROR: offline render failed checks: ${chk.problems.join('; ')}; no output written`); exit = 1; }
    else console.log('OK');
  } catch (e: any) {
    console.error(`ERROR: ${e.message}`); exit = 1;
  } finally {
    // close all browsers (grace 5 s each, then SIGKILL the Chromium process: it is only ever OUR headless
    // Chromium, never an ffmpeg capture), stop the server, then exit explicitly so no handle can keep node alive.
    const killed = await closeAllWithin(launched.map(b => ({ close: () => b.close(), kill: () => b.process()?.kill('SIGKILL') })), 5000);
    if (killed) console.log(`[CLEANUP] ${killed} browser(s) did not close within 5 s and were killed`);
    server.close();
  }
  process.exit(exit);
}
main();
