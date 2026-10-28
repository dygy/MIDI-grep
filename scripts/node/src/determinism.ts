/**
 * Run-to-run determinism of the offline renderer (page side). Measured sources of non-determinism in superdough 1.3.0
 * on a real piece (reverb + ducking + looped samples): the reverb impulse response is filled with Math.random() per
 * page/orbit (+ its async low-pass render may finish after the main render started), and orbit ducking (`.duckorbit`)
 * is scheduled from a ConstantSourceNode `onended` callback on the MAIN thread DURING the offline render, so which
 * render quantum the gain ramp lands in depends on thread timing. Both are fixed here, from outside the bundle:
 *
 *  1. `deterministicInitScript(seed)` - evaluated before ANY page script (puppeteer evaluateOnNewDocument):
 *     Math.random is a seeded PRNG; every `createBuffer(ch, len, sr)` re-seeds it from (seed, ch, len, sr), so the
 *     noise fill that follows (reverb IR, noise oscillators) depends only on the buffer shape - identical in every page
 *     and chunk, independent of call order; AudioWorklet modules get the same seeded Math.random as a prelude
 *     (the worklet scope has its own Math.random: supersaw / wavetable start phases); the number of in-flight child
 *     OfflineAudioContext renders (the reverb low-pass) is tracked so the main render can wait for them.
 *     A third source is neutralised here too: voice release (disconnect) callbacks that run at a random render position.
 *  2. `PAGE_DETERMINISM_JS` - installed after the bundle is up: orbit ducking is scheduled on the AudioParam timeline
 *     at trigger time (the context has not started rendering), emulating the superdough callback exactly:
 *     cancelScheduledValues(t-0.01); setValueAtTime(value at t-0.01); ramps to the ducked value and back.
 *     `window.__settleChildRenders()` resolves when every reverb IR is in its convolver.
 */

/** FNV-style mix of integers into an unsigned 32-bit value. Same function in the page script and in tests. */
export function mixSeed(seed: number, ...xs: number[]): number {
  let h = (2166136261 ^ seed) >>> 0;
  for (const x of xs) { h = Math.imul(h ^ (x | 0), 16777619) >>> 0; h ^= h >>> 13; h = h >>> 0; }
  return h >>> 0;
}

/** mulberry32 as a source string; `s` is the state variable name it advances. */
const PRNG_SRC = `const mix=(seed,...xs)=>{let h=(2166136261^seed)>>>0;for(const x of xs){h=Math.imul(h^(x|0),16777619)>>>0;h^=h>>>13;h>>>=0}return h>>>0};
let st=0;const rnd=()=>{st=(st+0x6D2B79F5)|0;let t=Math.imul(st^(st>>>15),1|st);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296};`;

export function deterministicInitScript(seed: number): string {
  const s = (seed >>> 0);
  return `(() => {
  const SEED = ${s};
  ${PRNG_SRC}
  st = mix(SEED, 0);
  Math.random = rnd;
  const cb = BaseAudioContext.prototype.createBuffer;
  BaseAudioContext.prototype.createBuffer = function (ch, len, sr) { st = mix(SEED, ch, len, Math.round(sr)); return cb.apply(this, arguments); };
  // worklet scope: its own Math.random (supersaw / wavetable phases); the module text gets the same seeded PRNG in front
  const PRELUDE = ${JSON.stringify(`(() => { const SEED = ${s}; ${PRNG_SRC} st = mix(SEED, 1); Math.random = rnd; })();\n`)};
  if (typeof AudioWorklet !== 'undefined') {
    const am = AudioWorklet.prototype.addModule;
    AudioWorklet.prototype.addModule = async function (url, opts) {
      let src;
      try { src = await (await fetch(String(url))).text(); } catch (_) { return am.call(this, url, opts); }
      const u = URL.createObjectURL(new Blob([PRELUDE, src], { type: 'text/javascript' }));
      try { return await am.call(this, u, opts); } finally { URL.revokeObjectURL(u); }
    };
  }
  // superdough releases a voice's nodes (disconnect) from an onended callback that the main thread runs at an arbitrary render
  // position while the offline context renders: the tail of a filter / oscillator is cut at a random sample. Offline, nothing
  // reuses those nodes (every hap is scheduled before the render starts), so a disconnect during the render is ignored.
  const dc = AudioNode.prototype.disconnect;
  AudioNode.prototype.disconnect = function () {
    if (typeof OfflineAudioContext !== 'undefined' && this.context instanceof OfflineAudioContext && this.context.state === 'running') return;
    return dc.apply(this, arguments);
  };
  // in-flight child renders (reverb IR low-pass): the main render must not start before they are in their convolvers
  window.__childRenders = 0;
  const sr0 = OfflineAudioContext.prototype.startRendering;
  OfflineAudioContext.prototype.startRendering = function () {
    window.__childRenders++;
    const p = sr0.apply(this, arguments);
    const done = () => setTimeout(() => { window.__childRenders--; }, 0);
    p.then(done, done);
    return p;
  };
})();`;
}

/** Evaluated in the page after the bundle is loaded and initStrudel() has run (needs window.strudel). */
export const PAGE_DETERMINISM_JS = String.raw`
window.__installDeterminism = () => {
  if (window.__childRenders === undefined) return; // init script not installed (MIDIGREP_RENDER_DETERMINISTIC=0): stock engine
  const S = window.strudel;
  const proto = Object.getPrototypeOf(S.getSuperdoughAudioController());
  if (proto.__detDuck) return;
  proto.__detDuck = true;
  const timelines = new WeakMap(); // orbit gain param -> [{k:'set'|'exp', t, v}]
  const valueAt = (ev, x) => {
    let p = null, n = null;
    for (const e of ev) { if (e.t <= x) p = e; else { n = e; break; } }
    if (!p) return 1;
    if (n && n.k === 'exp' && n.t > p.t) return p.v * Math.pow(n.v / p.v, (x - p.t) / (n.t - p.t));
    return p.v;
  };
  const insert = (ev, e) => { let i = ev.length; while (i > 0 && ev[i - 1].t > e.t) i--; ev.splice(i, 0, e); };
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  // superdough Orbit.duck(), scheduled now instead of from an onended callback while the context renders
  const duckOrbit = (orbit, t, onset, attack, depth) => {
    const param = orbit.output.gain;
    let ev = timelines.get(param); if (!ev) { ev = []; timelines.set(param, ev); }
    const now = t - 0.01;
    const cur = valueAt(ev, now);
    for (let i = ev.length - 1; i >= 0; i--) if (ev[i].t >= now) ev.splice(i, 1);
    param.cancelScheduledValues(now);
    param.setValueAtTime(cur, now); insert(ev, { k: 'set', t: now, v: cur });
    const ducked = clamp(1 - Math.sqrt(depth), 0.01, cur);
    param.exponentialRampToValueAtTime(ducked, t + onset); insert(ev, { k: 'exp', t: t + onset, v: ducked });
    param.exponentialRampToValueAtTime(1, t + onset + attack); insert(ev, { k: 'exp', t: t + onset + attack, v: 1 });
  };
  proto.duck = function (targetOrbits, t, onsettime = 0, attacktime = 0.1, depth = 1) {
    const targetArr = [targetOrbits].flat(), onsetArr = [onsettime].flat(), attackArr = [attacktime].flat(), depthArr = [depth].flat();
    targetArr.forEach((target, idx) => {
      const orbit = this.nodes[target];
      if (orbit == null) { document.dispatchEvent(new CustomEvent('strudel.log', { detail: { message: '[superdough] error: duck target orbit ' + target + ' does not exist' } })); return; }
      duckOrbit(orbit, t, onsetArr[idx] ?? onsetArr[0], Math.max(attackArr[idx] ?? attackArr[0], 0.002), depthArr[idx] ?? depthArr[0]);
    });
  };
  // reverb: remember every convolver so "IR in place" can be asserted before the main render
  const bp = BaseAudioContext.prototype, cr = bp.createReverb;
  window.__reverbs = [];
  if (cr) bp.createReverb = function () { const c = cr.apply(this, arguments); window.__reverbs.push(c); return c; };
};
window.__settleChildRenders = async () => {
  if (window.__childRenders === undefined) return;
  for (let i = 0; i < 20000 && window.__childRenders > 0; i++) await new Promise(r => setTimeout(r, 1));
  await new Promise(r => setTimeout(r, 0)); await new Promise(r => setTimeout(r, 0));
  if (window.__childRenders > 0) throw new Error('reverb impulse response render did not finish');
  const missing = (window.__reverbs || []).filter(c => !c.buffer).length;
  if (missing) throw new Error(missing + ' reverb convolver(s) have no impulse response before the main render started');
};
`;
