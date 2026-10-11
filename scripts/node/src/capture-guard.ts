/**
 * Process-hygiene helpers for the BlackHole recorder (no audio-path logic here).
 * Kept free of side effects on import so they can be unit-tested with fake children.
 */
import { EventEmitter } from 'events';

/** The slice of ChildProcess these helpers need (a fake implements it in tests). */
export interface GuardedChild extends EventEmitter {
  exitCode: number | null;
  signalCode: string | null;
  kill(signal?: NodeJS.Signals | number): boolean;
}

export function isAlive(child: GuardedChild): boolean {
  return child.exitCode === null && child.signalCode === null;
}

export type KillOutcome = 'already-dead' | 'exited-on-quit' | 'exited-on-sigterm' | 'killed' | 'unkillable';

export interface EscalateOptions {
  quit?: () => void;        // graceful request first (ffmpeg: write 'q' to stdin)
  quitGraceMs?: number;     // wait after quit
  termGraceMs?: number;     // wait after SIGTERM before the last-resort SIGKILL
  killWaitMs?: number;
  log?: (msg: string) => void;
}

/**
 * quit (optional) -> wait -> SIGTERM -> wait -> SIGKILL (last resort) -> wait. Logs which step worked.
 * Waits on 'exit' (not 'close'): a wedged child keeps its stdio pipes open, so 'close' may never fire.
 * NOTE: SIGKILL of an ffmpeg that is still opening the avfoundation device is known to wedge
 * BlackHole, so callers must pass a long termGraceMs.
 */
export async function killEscalate(child: GuardedChild, o: EscalateOptions = {}): Promise<KillOutcome> {
  const { quit, quitGraceMs = 0, termGraceMs = 2000, killWaitMs = 2000, log = () => {} } = o;
  if (!isAlive(child)) return 'already-dead';
  const waitExit = (ms: number) => new Promise<boolean>(resolve => {
    if (!isAlive(child)) return resolve(true);
    const onExit = () => { clearTimeout(t); resolve(true); };
    const t = setTimeout(() => { child.removeListener('exit', onExit); resolve(false); }, ms);
    child.once('exit', onExit);
  });
  if (quit) {
    try { quit(); } catch { /* stdin closed */ }
    if (await waitExit(quitGraceMs)) { log('ffmpeg exited on quit'); return 'exited-on-quit'; }
    log(`ffmpeg ignored quit for ${quitGraceMs} ms; sending SIGTERM`);
  }
  try { child.kill('SIGTERM'); } catch { /* already gone */ }
  if (await waitExit(termGraceMs)) { log('ffmpeg exited on SIGTERM'); return 'exited-on-sigterm'; }
  log(`ffmpeg ignored SIGTERM for ${termGraceMs} ms; sending SIGKILL (last resort, may wedge BlackHole)`);
  try { child.kill('SIGKILL'); } catch { /* already gone */ }
  const ok = await waitExit(killWaitMs);
  log(ok ? 'ffmpeg killed by SIGKILL' : 'ffmpeg survived SIGKILL');
  return ok ? 'killed' : 'unkillable';
}

/** Keeps the last `max` non-empty lines of a text stream (ffmpeg uses \r for progress lines too). */
export class TailBuffer {
  private lines: string[] = [];
  private partial = '';
  constructor(private readonly max = 20) {}
  push(chunk: Buffer | string): void {
    const parts = (this.partial + chunk.toString()).split(/[\r\n]+/);
    this.partial = parts.pop() ?? '';
    for (const p of parts) if (p.trim()) this.lines.push(p);
    if (this.lines.length > this.max) this.lines.splice(0, this.lines.length - this.max);
  }
  tail(): string {
    const all = this.partial.trim() ? [...this.lines, this.partial] : this.lines;
    return all.slice(-this.max).join('\n');
  }
}

export type StartOutcome = 'ok' | 'no-output' | 'ffmpeg-exited';

export interface WatchOptions {
  child: GuardedChild;
  sizeOf: () => number;       // current output size in bytes, 0 if missing
  deadlineMs: number;         // counted from arm(), NOT from spawn
  pollMs?: number;
  onStarted?: (latencyMs: number) => void;   // first time the output has bytes
}

/**
 * Before arm(): only watches for the process DYING (device open may legitimately be slow while the
 * browser competes for the machine — never time-box it). After arm() (call it when playback has
 * started, i.e. audio must flow): 'no-output' if still no bytes `deadlineMs` later.
 * Resolves 'ok' on first output; cancel() stops polling (resolves 'ok' so callers do not react).
 */
export function watchCaptureStart(opts: WatchOptions): { result: Promise<StartOutcome>; arm: () => void; cancel: () => void } {
  const { child, sizeOf, deadlineMs, pollMs = 250, onStarted } = opts;
  const t0 = Date.now();
  let armedAt: number | null = null;
  let cancel = () => {};
  const result = new Promise<StartOutcome>(resolve => {
    const finish = (o: StartOutcome) => { clearInterval(iv); resolve(o); };
    const iv = setInterval(() => {
      if (sizeOf() > 0) { onStarted?.(Date.now() - t0); return finish('ok'); }
      if (!isAlive(child)) return finish('ffmpeg-exited');
      if (armedAt !== null && Date.now() - armedAt >= deadlineMs) return finish('no-output');
    }, pollMs);
    cancel = () => finish('ok');
  });
  return { result, arm: () => { if (armedAt === null) armedAt = Date.now(); }, cancel: () => cancel() };
}

export interface FatalDeps {
  log: (msg: string) => void;
  tail: () => string;
  cleanup: () => Promise<void>;
  exit: (code: number) => void;
}

/**
 * Re-entrant shutdown: the FIRST call logs, runs cleanup once and exits with ITS code; any later
 * (concurrent) call just awaits that same cleanup and exits with the first call's code.
 */
export function createFatal(deps: FatalDeps): (message: string, code?: number) => Promise<void> {
  let inFlight: Promise<void> | null = null;
  let firstCode = 1;
  return (message: string, code = 1) => {
    if (!inFlight) {
      firstCode = code;
      inFlight = (async () => {
        deps.log(message);
        const tail = deps.tail();
        if (tail) deps.log(`--- ffmpeg stderr (last lines) ---\n${tail}\n--- end ffmpeg stderr ---`);
        try { await deps.cleanup(); } catch (e) { deps.log(`cleanup error: ${e}`); }
      })();
    }
    return inFlight.then(() => deps.exit(firstCode));
  };
}
