#!/usr/bin/env node
/**
 * Headless check of a visuals.strudel (scripts/python/visuals): does its Hydra block run, does the scene stay
 * recognizable, and does the picture move with the patterns?
 *
 * The page loads the pinned @strudel/web bundle (patterns, mini-notation, transpiler; no sounds are loaded, nothing
 * plays) and the pinned hydra-synth, defines initHydra the way @strudel/hydra does (a Hydra on its own canvas,
 * detectAudio off), evaluates the WHOLE file with the REPL's evaluate, then drives Strudel's clock (setTime) by hand
 * and ticks Hydra at 30 fps through a window at the start of every section (every 16 bars without sections).
 *
 * Measured per frame: correlation of the frame's luminance with the scene image's (recognizable), the mean absolute
 * luminance change from the previous frame (motion), and every control's value. Verdict PASS when the block evaluated
 * without page errors, the mean scene correlation >= --min-scene-corr, every control the piece wires changed value,
 * and the quarter of frames with the strongest pulse move more than the quarter with the weakest by --min-motion-ratio.
 *
 * Usage: node dist/check-hydra.js visuals.strudel --image visual.jpg -o outdir [--wired kick,bassWave,...] [--fps 30] [--window 2]
 * Writes outdir/check.json and outdir/frame_<section>.png (one frame per section, at its strongest kick).
 */
import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const PAGE_HTML = String.raw`<!doctype html><meta charset="utf-8"><title>hydra check</title>
<body style="margin:0;background:#000">
<script src="/strudel.js"></script>
<script src="/hydra.js"></script>
<script>
window.__errors = [];
window.addEventListener('error', e => window.__errors.push(String(e.message)));
document.addEventListener('strudel.log', e => { const d = e.detail || {}; if (d.type === 'error' || /error/i.test(String(d.message))) window.__errors.push(String(d.message)); });
for (const k of ['error']) { const o = console[k].bind(console); console[k] = (...a) => { window.__errors.push(a.map(String).join(' ')); o(...a); }; }
window.addEventListener('unhandledrejection', e => window.__errors.push('unhandledrejection ' + (e.reason && e.reason.message || e.reason)));
let hydra = null;
// @strudel/hydra initHydra, minus the REPL draw context: a canvas of the scene's size, detectAudio off, no auto loop
window.initHydra = async (options = {}) => {
  if (hydra) return hydra;
  const canvas = document.createElement('canvas');
  canvas.width = window.__W; canvas.height = window.__H;
  document.body.appendChild(canvas);
  hydra = new Hydra({ detectAudio: false, ...options, canvas, autoLoop: false, makeGlobal: true, width: window.__W, height: window.__H });
  window.__canvas = canvas;
  return hydra;
};
window.__clock = 0;
window.__init = async (code, W, H) => {
  window.__W = W; window.__H = H;
  const repl = await strudel.initStrudel({});
  // expose the block's controls to the probe (const declarations are local to the evaluated code)
  const probe = "\nwindow.__vis = { kick, snare, hat, bassHit, leadHit, bassWave, leadHue, chordDepth, energy }\n";
  await repl.evaluate(code + probe, false);
  strudel.setTime(() => window.__clock); // after evaluate: the REPL points the time source at its scheduler there
  if (!window.__vis) throw new Error('evaluate did not reach the end of the block: ' + window.__errors.join(' | '));
  // hydra-synth sets s0.src only in the image's onload
  const loaded = () => globalThis.s0 && globalThis.s0.src && globalThis.s0.src.naturalWidth;
  for (let i = 0; i < 200 && !loaded(); i++) await new Promise(r => setTimeout(r, 50));
  const img = globalThis.s0 && globalThis.s0.src;
  if (!(img && img.naturalWidth)) throw new Error('s0 image did not load');
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0, W, H);
  window.__sceneLum = lum(g.getImageData(0, 0, W, H).data);
  window.__grab = document.createElement('canvas'); window.__grab.width = W; window.__grab.height = H;
  return { cps: repl.scheduler.cps };
};
function lum(d) { const n = d.length / 4, out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = 0.299 * d[4*i] + 0.587 * d[4*i+1] + 0.114 * d[4*i+2]; return out; }
function corr(a, b) { let ma = 0, mb = 0; for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; } ma /= a.length; mb /= b.length;
  let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0; }
// one window: cycles [c0, c0 + n/fps*cps), one Hydra tick per frame
window.__window = (c0, frames, fps, cps) => {
  const g = window.__grab.getContext('2d', { willReadFrequently: true }), out = []; let prev = null;
  for (let f = 0; f < frames; f++) {
    window.__clock = c0 + f / fps * cps;
    hydra.tick(1000 / fps);
    g.drawImage(window.__canvas, 0, 0);
    const L = lum(g.getImageData(0, 0, window.__W, window.__H).data);
    let motion = 0; if (prev) { for (let i = 0; i < L.length; i++) motion += Math.abs(L[i] - prev[i]); motion /= L.length; }
    const ctl = {}; for (const [k, fn] of Object.entries(window.__vis)) ctl[k] = Number(fn());
    out.push({ cycle: window.__clock, sceneCorr: corr(L, window.__sceneLum), motion: prev ? motion : null, ctl });
    prev = L;
  }
  return out;
};
window.__png = () => window.__grab.toDataURL('image/png').split(',')[1];
window.__frameAt = (cycle, fps) => { window.__clock = cycle; hydra.tick(1000 / fps); window.__grab.getContext('2d').drawImage(window.__canvas, 0, 0); return window.__png(); };
</script>`;

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : def;
}

async function main() {
  const input = process.argv[2];
  const image = arg('--image');
  const outDir = arg('-o');
  if (!input || !image || !outDir) { console.error('usage: check-hydra.js visuals.strudel --image visual.jpg -o outdir'); process.exit(2); }
  const fps = Number(arg('--fps', '30'));
  const windowS = Number(arg('--window', '2'));
  const minSceneCorr = Number(arg('--min-scene-corr', '0.5'));
  const minMotionRatio = Number(arg('--min-motion-ratio', '1.2'));
  fs.mkdirSync(outDir, { recursive: true });
  const code = fs.readFileSync(input, 'utf8');

  // the scene's size from the JPEG SOF0/SOF2 marker: the canvas matches the image, no rescale in the correlation
  const jpg = fs.readFileSync(image);
  let W = 640, H = 384;
  for (let i = 2; i < jpg.length - 9;) {
    if (jpg[i] !== 0xff) { i++; continue; }
    const m = jpg[i + 1], len = jpg.readUInt16BE(i + 2);
    if (m === 0xc0 || m === 0xc2) { H = jpg.readUInt16BE(i + 5); W = jpg.readUInt16BE(i + 7); break; }
    i += 2 + len;
  }
  const sectionsLine = code.split('\n').find(l => l.includes('sections:')) || '';
  const sections = [...sectionsLine.matchAll(/(\w+)\[(\d+)-(\d+)\]/g)].map(m => ({ name: m[1], bar: Number(m[2]) }));
  const bars = (code.match(/^let bass = \[\n([\s\S]*?)^\]/m)?.[1].split('\n').filter(l => l.trim()).length) || 64;
  const windows = sections.length ? sections : Array.from({ length: Math.ceil(bars / 16) }, (_, i) => ({ name: `bar${i * 16}`, bar: i * 16 }));

  const require = createRequire(import.meta.url);
  const bundle = path.join(path.dirname(require.resolve('@strudel/web/package.json')), 'dist', 'index.js');
  // hydra-synth's exports map hides package.json; its dist sits next to the bundle's node_modules
  const hydraJs = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'node_modules', 'hydra-synth', 'dist', 'hydra-synth.js');
  const server = http.createServer((req, res) => {
    if (req.url === '/strudel.js' || req.url === '/hydra.js') {
      res.writeHead(200, { 'content-type': 'text/javascript' });
      fs.createReadStream(req.url === '/strudel.js' ? bundle : hydraJs).pipe(res);
    } else { res.writeHead(200, { 'content-type': 'text/html' }); res.end(PAGE_HTML); }
  });
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const port = (server.address() as any).port;
  // no --use-angle=swiftshader: on macOS it leaves headless Chromium with no WebGL context at all
  const browser = await puppeteer.launch({ headless: true, protocolTimeout: 900000, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const result: any = { input: path.basename(input), image: path.basename(image), size: [W, H], fps, window_s: windowS,
    thresholds: { min_scene_corr: minSceneCorr, min_motion_ratio: minMotionRatio } };
  try {
    const page = await browser.newPage();
    const pageErrors: string[] = [];
    page.on('pageerror', e => pageErrors.push(String((e as any).message || e)));
    page.setDefaultTimeout(600000); // swiftshader WebGL + a busy machine: the per-section windows can take minutes
    await page.goto(`http://localhost:${port}/`, { waitUntil: 'load', timeout: 120000 });
    const { cps } = await page.evaluate((c, w, h) => (window as any).__init(c, w, h), code, W, H) as any;
    result.cps = cps;
    const frames = Math.round(windowS * fps);
    const perSection: any[] = [];
    for (const s of windows) {
      const fr: any[] = await page.evaluate((c0, n, f, cp) => (window as any).__window(c0, n, f, cp), s.bar, frames, fps, cps);
      const best = fr.reduce((a, b) => (b.ctl.kick > a.ctl.kick ? b : a), fr[0]);
      const png = await page.evaluate((cy, f) => (window as any).__frameAt(cy, f), best.cycle, fps) as string;
      fs.writeFileSync(path.join(outDir, `frame_${s.name}_${s.bar}.png`), Buffer.from(png, 'base64'));
      perSection.push({ section: s.name, bar: s.bar, frames: fr });
      console.log(`[check-hydra] ${s.name}@${s.bar}: ${fr.length} frames`);
    }
    const all = perSection.flatMap(s => s.frames);
    const mean = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
    const pulses = ['kick', 'snare', 'hat', 'bassHit', 'leadHit'];
    const moving = all.filter(f => f.motion !== null);
    // frames ranked by their strongest pulse: the top quarter (on the hits) must move more than the bottom quarter
    // (between them); a dense piece never has every pulse near 0, so the split is relative, not absolute
    const ranked = moving.map(f => ({ p: Math.max(...pulses.map(k => f.ctl[k])), m: f.motion })).sort((a, b) => a.p - b.p);
    const q = Math.floor(ranked.length / 4);
    const spread = ranked.length ? ranked[ranked.length - 1].p - ranked[0].p : 0;
    const high = spread > 0 ? ranked.slice(ranked.length - q).map(f => f.m) : [];
    const low = spread > 0 ? ranked.slice(0, q).map(f => f.m) : [];
    const ranges: Record<string, number[]> = {};
    for (const k of Object.keys(all[0].ctl)) { const v = all.map(f => f.ctl[k]); ranges[k] = [Math.min(...v), Math.max(...v)]; }
    const wired = (arg('--wired', '') || '').split(',').filter(Boolean);
    const dead = wired.filter(k => ranges[k] && ranges[k][0] === ranges[k][1]);
    const nonFinite = Object.entries(ranges).filter(([, r]) => !r.every(Number.isFinite)).map(([k]) => k);
    result.scene_corr = { mean: mean(all.map(f => f.sceneCorr)), min: Math.min(...all.map(f => f.sceneCorr)) };
    result.motion = { pulse_high: mean(high), pulse_low: mean(low), n_high: high.length, n_low: low.length,
      ratio: mean(low) > 0 ? mean(high) / mean(low) : (mean(high) > 0 ? Infinity : 0) };
    result.control_ranges = ranges;
    result.sections = perSection.map(s => ({ section: s.section, bar: s.bar, scene_corr: mean(s.frames.map((f: any) => f.sceneCorr)),
      motion: mean(s.frames.filter((f: any) => f.motion !== null).map((f: any) => f.motion)) }));
    result.page_errors = pageErrors;
    const fails: string[] = [];
    if (pageErrors.length) fails.push(`page errors: ${pageErrors.slice(0, 3).join(' | ')}`);
    if (result.scene_corr.mean < minSceneCorr) fails.push(`scene not recognizable: mean luminance correlation ${result.scene_corr.mean.toFixed(3)} < ${minSceneCorr}`);
    if (nonFinite.length) fails.push(`non-finite controls: ${nonFinite.join(', ')}`);
    if (dead.length) fails.push(`wired controls never changed: ${dead.join(', ')}`);
    if (!high.length || !low.length) fails.push('the pulses never change over the probed windows: cannot test motion');
    else if (result.motion.ratio < minMotionRatio) fails.push(`picture does not move with the hits: motion ratio ${result.motion.ratio.toFixed(2)} < ${minMotionRatio}`);
    result.verdict = fails.length ? 'FAIL' : 'PASS';
    result.failures = fails;
  } catch (e: any) {
    result.verdict = 'FAIL';
    result.failures = [String(e && e.message || e)];
  } finally {
    await browser.close();
    server.close();
  }
  fs.writeFileSync(path.join(outDir, 'check.json'), JSON.stringify(result, null, 1));
  console.log(`[check-hydra] ${result.verdict}` + (result.scene_corr ? ` scene_corr ${result.scene_corr.mean.toFixed(3)} motion_ratio ${Number(result.motion.ratio).toFixed(2)}` : '') + (result.failures.length ? ` — ${result.failures.join('; ')}` : ''));
  process.exit(result.verdict === 'PASS' ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(2); });
