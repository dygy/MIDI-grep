/** Pure helpers for the offline Strudel renderer (no side effects on import). */
import * as fsSync from 'fs';

export interface WavInfo {
  sampleRate: number; channels: number; bitsPerSample: number; format: number;
  frames: number; seconds: number; rms: number; peak: number;
}

/** Parse a RIFF/WAVE buffer (PCM 8-32 bit or float32) and measure RMS / peak over all samples. */
export function analyzeWav(buf: Uint8Array): WavInfo {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const tag = (o: number) => String.fromCharCode(buf[o], buf[o + 1], buf[o + 2], buf[o + 3]);
  if (buf.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('not a RIFF/WAVE file');
  let fmt: { format: number; channels: number; sampleRate: number; bits: number } | null = null;
  let dataOff = -1, dataLen = 0;
  for (let o = 12; o + 8 <= buf.length;) {
    const id = tag(o), len = dv.getUint32(o + 4, true);
    if (id === 'fmt ') fmt = { format: dv.getUint16(o + 8, true), channels: dv.getUint16(o + 10, true), sampleRate: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) };
    if (id === 'data') { dataOff = o + 8; dataLen = Math.min(len, buf.length - dataOff); break; }
    o += 8 + len + (len & 1);
  }
  if (!fmt || dataOff < 0) throw new Error('WAV missing fmt/data chunk');
  const bps = fmt.bits / 8, n = Math.floor(dataLen / bps);
  let sumSq = 0, peak = 0;
  for (let i = 0; i < n; i++) {
    const p = dataOff + i * bps;
    let v: number;
    if (fmt.format === 3 && fmt.bits === 32) v = dv.getFloat32(p, true);
    else if (fmt.bits === 16) v = dv.getInt16(p, true) / 32768;
    else if (fmt.bits === 32) v = dv.getInt32(p, true) / 2147483648;
    else if (fmt.bits === 8) v = (buf[p] - 128) / 128;
    else if (fmt.bits === 24) v = ((buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16)) << 8 >> 8) / 8388608;
    else throw new Error(`unsupported WAV bit depth ${fmt.bits}`);
    sumSq += v * v; const a = Math.abs(v); if (a > peak) peak = a;
  }
  const frames = Math.floor(n / fmt.channels);
  return { sampleRate: fmt.sampleRate, channels: fmt.channels, bitsPerSample: fmt.bits, format: fmt.format,
    frames, seconds: frames / fmt.sampleRate, rms: n ? Math.sqrt(sumSq / n) : 0, peak };
}

/** Distinct log lines that indicate a sample / sound failed to load or a hap failed to trigger. */
export function sampleFailures(lines: string[]): string[] {
  const re = /\b(error|failed|failure)\b|not found|could not|unable to|\b404\b/i;
  return [...new Set(lines.filter(l => re.test(l)).map(l => l.replace(/\s+/g, ' ').trim().slice(0, 200)))];
}

/** Frames an OfflineAudioContext of `cycles` cycles at `cps` has at `sampleRate` (browsers floor/round to int). */
export function expectedFrames(cycles: number, cps: number, sampleRate: number): number {
  return Math.floor((cycles / cps) * sampleRate);
}

export interface RenderChecks { ok: boolean; problems: string[]; }
/** Loud-failure rules: wrong rate/channels, duration mismatch (> 1 frame), silent output, failed loads. */
export function checkRender(info: WavInfo, o: { sampleRate: number; expectedFrames: number; failures: string[]; allowFailures?: boolean; silentRms?: number }): RenderChecks {
  const problems: string[] = [];
  if (info.sampleRate !== o.sampleRate) problems.push(`sample rate ${info.sampleRate} != ${o.sampleRate}`);
  if (info.channels !== 2) problems.push(`channels ${info.channels} != 2`);
  if (Math.abs(info.frames - o.expectedFrames) > 1) problems.push(`duration mismatch: ${info.frames} frames vs expected ${o.expectedFrames}`);
  if (info.rms < (o.silentRms ?? 1e-5)) problems.push(`silent output (rms ${info.rms.toExponential(2)})`);
  if (o.failures.length && !o.allowFailures) problems.push(`${o.failures.length} distinct load/trigger failure(s): ${o.failures.slice(0, 5).join(' | ')}`);
  return { ok: problems.length === 0, problems };
}

/** Overlap-add `chunk` (interleaved L,R float32) into out starting at `startFrame`, clipped to out length. */
export function overlapAdd(outL: Float32Array, outR: Float32Array, chunk: Float32Array, startFrame: number): number {
  const n = chunk.length / 2, lim = Math.max(0, Math.min(n, outL.length - startFrame));
  for (let i = 0; i < lim; i++) { outL[startFrame + i] += chunk[i]; outR[startFrame + i] += chunk[n + i]; }
  return lim;
}

/** 16-bit PCM stereo WAV from float channels (same quantiser as the old in-page encoder). */
export function floatToWav16(L: Float32Array, R: Float32Array, sampleRate: number): Buffer {
  const n = L.length, out = Buffer.alloc(44 + n * 4);
  out.write('RIFF', 0); out.writeUInt32LE(36 + n * 4, 4); out.write('WAVE', 8); out.write('fmt ', 12);
  out.writeUInt32LE(16, 16); out.writeUInt16LE(1, 20); out.writeUInt16LE(2, 22); out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * 4, 28); out.writeUInt16LE(4, 32); out.writeUInt16LE(16, 34); out.write('data', 36); out.writeUInt32LE(n * 4, 40);
  const q = (x: number) => { x = Math.max(-1, Math.min(1, x)); return x < 0 ? Math.round(x * 32768) : Math.round(x * 32767); };
  for (let i = 0; i < n; i++) { out.writeInt16LE(q(L[i]), 44 + i * 4); out.writeInt16LE(q(R[i]), 46 + i * 4); }
  return out;
}

/** Whole-cycle chunk windows [b,e) covering [0,cycles). */
export function chunkWindows(cycles: number, chunkCycles: number): Array<[number, number]> {
  const w: Array<[number, number]> = [];
  for (let b = 0; b < cycles - 1e-9; b += chunkCycles) w.push([b, Math.min(b + chunkCycles, cycles)]);
  return w;
}

/**
 * Chunk plan bounded in BOTH time and work: greedy windows of whole cycles, closing a window when it
 * reaches `maxCycles` cycles or adding the next cycle would push it past `maxOnsets` triggered haps
 * (a window always holds at least one cycle). `onsetsPerCycle[c]` = haps starting in cycle c.
 */
export function planChunks(onsetsPerCycle: number[], cycles: number, maxCycles: number, maxOnsets: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const n = Math.ceil(cycles - 1e-9);
  let b = 0, acc = 0;
  for (let c = 0; c < n; c++) {
    const k = onsetsPerCycle[c] ?? 0;
    if (c > b && (c - b >= maxCycles || acc + k > maxOnsets)) { out.push([b, c]); b = c; acc = 0; }
    acc += k;
  }
  if (n > b) out.push([b, Math.min(n, cycles)]);
  return out;
}

export interface Closable { close(): Promise<unknown>; kill(): void; }
/**
 * Close every resource within `graceMs`; anything whose close() hangs or rejects is killed. Resolves
 * (never hangs) with how many needed the kill. Used for ALL browsers a run launched (initial + relaunches).
 */
export async function closeAllWithin(items: Closable[], graceMs: number): Promise<number> {
  let killed = 0;
  await Promise.all(items.map(async it => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ok = await Promise.race([
      it.close().then(() => true, () => false),
      new Promise<boolean>(r => { timer = setTimeout(() => r(false), graceMs); }),
    ]);
    if (timer) clearTimeout(timer);
    if (!ok) { killed++; try { it.kill(); } catch { /* already gone */ } }
  }));
  return killed;
}

/**
 * superdough arguments for one hap. `t` is the trigger time inside the (chunk) context whose frame 0 is
 * `ctxStartSec` of the song; `cycle` is the hap's ABSOLUTE start cycle, exactly what the live REPL passes
 * (LFO / tremolo phase derives from it, so it must not restart per chunk). This function's source is also
 * injected into the render page, so it must stay self-contained.
 */
export function triggerArgs(beginCycle: number, cps: number, ctxStartSec: number): { t: number; cycle: number } {
  return { t: beginCycle / cps - ctxStartSec, cycle: beginCycle };
}

/**
 * Pre-roll geometry for chunk starting at cycle `b`: haps with onsets in [preCycle, b) are re-triggered
 * (audio discarded) so cut groups / orbit ducking see the previous chunk's still-ringing voices.
 * `preFrames` = frames between the context origin and the chunk's startFrame (integer, 0 for the first chunk).
 */
export function prerollPlan(b: number, startFrame: number, cps: number, sampleRate: number, prerollSeconds: number): { preCycle: number; preFrames: number } {
  const preCycle = Math.max(0, b - prerollSeconds * cps);
  const preFrames = Math.max(0, startFrame - Math.max(0, Math.floor((preCycle / cps) * sampleRate)));
  return { preCycle, preFrames };
}

/** From planar [L(len), R(len)] context audio keep `keep` frames after skipping `skip`, planar again. */
export function trimChunk(f: Float32Array, len: number, skip: number, keep: number): Float32Array {
  const n = Math.max(0, Math.min(keep, len - skip)), out = new Float32Array(n * 2);
  out.set(f.subarray(skip, skip + n), 0); out.set(f.subarray(len + skip, len + skip + n), n);
  return out;
}

/** Remove a leftover output from an earlier run so a stale file can never be mistaken for this render. */
export function removeStaleOutput(target: string): void {
  try { fsSync.rmSync(target, { force: true }); } catch { /* nothing to remove */ }
}
export function tempOutputPath(target: string): string { return `${target}.tmp-${process.pid}`; }
/** Publish `wav` at `target` only if `ok`: written to a temp file beside it, then atomically renamed; else nothing is left behind. */
export function commitOutput(target: string, wav: Uint8Array, ok: () => boolean): boolean {
  const tmp = tempOutputPath(target);
  fsSync.writeFileSync(tmp, wav);
  if (!ok()) { fsSync.rmSync(tmp, { force: true }); return false; }
  fsSync.renameSync(tmp, target);
  return true;
}

// ---- transient network outage handling -------------------------------------------------------------------------
/** Chromium net errors / fetch messages that mean "the network is down or flapping", not "the render is wrong". */
const NETWORK_RE = /net::ERR_(INTERNET_DISCONNECTED|TIMED_OUT|NAME_NOT_RESOLVED|NETWORK_CHANGED|NETWORK_ACCESS_DENIED|ADDRESS_UNREACHABLE|CONNECTION_\w+|QUIC_\w+|SOCKET_NOT_CONNECTED|TUNNEL_CONNECTION_FAILED|PROXY_CONNECTION_FAILED|NETWORK_IO_SUSPENDED)|ERR_(INTERNET_DISCONNECTED|NAME_NOT_RESOLVED)|\b(ENOTFOUND|EAI_AGAIN|ECONNRESET|ECONNREFUSED|ETIMEDOUT|ENETUNREACH)\b|failed to fetch|networkerror|load failed|navigation timeout of \d+ ms exceeded/i;
/** True when `text` (an error message or a browser console line) is a network failure. A page navigation timeout counts: the page waits
 * for 'load', i.e. for every remote script and sample map, so it times out when those stall (Regime 2026-10-26). */
export function isNetworkFailure(text: string): boolean { return NETWORK_RE.test(text || ''); }

/** A log/error line saying a sample map or sample file FAILED TO LOAD (as opposed to a sound that is simply unknown, or a render-check failure). */
const SAMPLE_LOAD_RE = /error loading|failed to load|could not load|unable to load|sample load failed|\bload(?:ing)?\b.{0,40}\b(?:fail\w*|error)\b/i;
export function isSampleLoadFailure(text: string): boolean { return SAMPLE_LOAD_RE.test(text || ''); }

/** A logged sample failure the outage explains: a failed load, or a failure that is itself a network error (a sample
 * fetch that died: superdough logs `error: Failed to fetch hap={...}`). A sound that is simply unknown is neither. */
export function isOutageSampleFailure(text: string): boolean { return isSampleLoadFailure(text) || isNetworkFailure(text); }

/**
 * Is an attempt's OWN failure a network outage (wait + retry) rather than a crash/real failure (MAX_ATTEMPTS, then fail)?
 * Yes when the error itself is a network failure, or when it is a sample-load failure AND network errors were logged during the
 * attempt (the host was down, so the load failed for that reason). Unrelated network console noise never reclassifies other errors
 * (eval error, onset mismatch, unloaded sound, closed page).
 */
export function isNetworkOutageFailure(message: string, netEventsDuringAttempt: number): boolean {
  return isNetworkFailure(message) || (netEventsDuringAttempt > 0 && isSampleLoadFailure(message));
}

/** Default total seconds a render waits out a network outage before failing; override with MIDIGREP_RENDER_NET_WAIT_S (0 = never wait). */
export const DEFAULT_NET_WAIT_S = 1800;
/** Backoff schedule (s): first wait, doubling up to the cap. */
export const NET_BACKOFF_FIRST_S = 5;
export const NET_BACKOFF_CAP_S = 60;

/** Parse MIDIGREP_RENDER_NET_WAIT_S: a non-negative number of seconds, else the default (invalid values are not silently used). */
export function netWaitBudgetS(env: string | undefined): number {
  if (env === undefined || env.trim() === '') return DEFAULT_NET_WAIT_S;
  const v = Number(env);
  return Number.isFinite(v) && v >= 0 ? v : DEFAULT_NET_WAIT_S;
}

/** Next backoff wait in seconds after `waitedS` already waited, for the `n`th (0-based) consecutive network retry; 0 when the budget is spent. The last wait is trimmed to fit the budget. */
export function nextNetWaitS(n: number, waitedS: number, budgetS: number): number {
  const left = budgetS - waitedS;
  if (!(left > 0)) return 0;
  return Math.min(NET_BACKOFF_FIRST_S * 2 ** n, NET_BACKOFF_CAP_S, left);
}

/** One onset's voice, in seconds: when it starts and when it stops sounding (hap + release, or the whole one-shot slice). */
export interface Ring { begin: number; end: number }

/** Width (s) of the window at the very end of a chunk context in which the renderer demands silence (the tail-truncation check). */
/**
 * Run `fn` up to `attempts` times, reporting each failure through `onFail(attempt, error)`; the last failure is rethrown.
 * Used for launching the headless browser, which can time out when the machine is loaded (2026-10-29: a 30 s
 * puppeteer launch timeout at load average ~20 failed a whole calibration).
 */
export async function retried<T>(fn: () => Promise<T>, attempts: number, onFail: (attempt: number, err: any) => void): Promise<T> {
  for (let k = 1; ; k++) {
    try { return await fn(); } catch (err) { onFail(k, err); if (k >= attempts) throw err; }
  }
}

export const TAIL_CHECK_S = 0.25;

/**
 * Per-chunk margins derived from what the piece actually plays (window [tb, te) in seconds). `fxSeconds` is the effect
 * tail (reverb/delay) added on top of a voice's end, `minPreroll` the least look-back kept for cut groups / ducking.
 *  - tailSeconds: how far past te the context is rendered = fxSeconds. A voice still sounding at te is NOT rendered to its end here:
 *    the chunk owns [tb, te) only, and the next chunk re-triggers that voice as pre-roll and renders its end (ring + fx > its tb);
 *  - tailCheck: whether the end of the context may be required to be silent. Not when a voice that starts before te still sounds
 *    inside the check window (te + fx - TAIL_CHECK_S, te + fx]: its audio there is legitimate and the chunk that holds its end checks it;
 *  - prerollSeconds: how far back from tb an onset must be re-triggered so a voice still sounding at tb is reproduced
 *    from its start, never below minPreroll;
 *  - ringingOnsets: the onsets (seconds) of those older, still-sounding voices, for the page to include in the pre-roll.
 */
export function chunkMargins(rings: Ring[], tb: number, te: number, fxSeconds: number, minPreroll: number): { tailSeconds: number; tailCheck: boolean; prerollSeconds: number; ringingOnsets: number[] } {
  let back = minPreroll, lastEnd = -Infinity;
  const ringingOnsets: number[] = [];
  const eps = 1e-9;
  for (const r of rings) {
    if (r.begin < te - eps) lastEnd = Math.max(lastEnd, r.end);
    if (r.begin < tb - eps && r.end + fxSeconds > tb) {
      back = Math.max(back, tb - r.begin);
      if (tb - r.begin > minPreroll) ringingOnsets.push(r.begin);
    }
  }
  return { tailSeconds: fxSeconds, tailCheck: lastEnd <= te + fxSeconds - TAIL_CHECK_S + eps, prerollSeconds: back, ringingOnsets };
}
