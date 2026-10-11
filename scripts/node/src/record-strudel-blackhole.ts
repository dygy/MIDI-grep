#!/usr/bin/env npx ts-node
/**
 * Record Strudel using BlackHole virtual audio device
 *
 * Prerequisites:
 * - BlackHole: brew install blackhole-2ch
 *
 * The script:
 * 1. Grants microphone permission (to enumerate audio devices)
 * 2. Opens strudel.dygy.app/embed (self-hosted Strudel REPL)
 * 3. Sets AudioContext.setSinkId to BlackHole (direct API, not settings UI)
 * 4. Inserts code and clicks play
 * 5. Records via ffmpeg from BlackHole
 *
 * Process hygiene: SIGKILL of an ffmpeg that is still opening the avfoundation device is KNOWN to
 * wedge BlackHole (spec 004 s3), and device open can be slow while the browser launch competes for
 * the machine. So ffmpeg is stopped quit -> SIGTERM -> (long grace) -> SIGKILL, and "no output" is
 * only judged after playback has started.
 *
 * Usage:
 *   node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30
 */

import puppeteer from 'puppeteer';
import { spawn, spawnSync } from 'child_process';
import * as fs from 'fs';
import { killEscalate, TailBuffer, isAlive, watchCaptureStart, createFatal } from './capture-guard.js';

// Robustness state (Oct 2026): a wedged ffmpeg (SIGTERM ignored, 0 bytes written) once outlived the
// recorder and kept BlackHole busy. Every exit path goes through fatal()/cleanupChildren().
let ffmpegProc: ReturnType<typeof spawn> | null = null;
let browserRef: { close(): Promise<void>; process(): { kill(s?: NodeJS.Signals): boolean } | null } | null = null;
const ffmpegStderr = new TailBuffer(20);
let startWatch: { cancel: () => void } | null = null;
let dying = false;

// "No output" is judged only once playback has started (audio must flow from then on). Deadline is
// derived from the recorder's own waits after that point: the 2 s sample-load settle below plus
// ffmpeg's default 5 s input probe (analyzeduration) = 7 s. Device open itself is never time-boxed.
const POST_PLAY_SETTLE_MS = 2000;
const FFMPEG_ANALYZE_MS = 5000;
const NO_OUTPUT_DEADLINE_MS = POST_PLAY_SETTLE_MS + FFMPEG_ANALYZE_MS;
const SPAWN_SETTLE_MS = 1000;
// Escalation graces. SIGKILL only after quit (5 s) + SIGTERM (25 s) = 30 s: device open has been seen
// to take several seconds under browser-launch load and must complete by the 60 s play-wait bound at
// worst; 30 s is far past the observed slow case, and SIGKILL mid-open wedges BlackHole.
const QUIT_GRACE_MS = 5000;
const TERM_GRACE_MS = 25000;
const escalation = { quitGraceMs: QUIT_GRACE_MS, termGraceMs: TERM_GRACE_MS, killWaitMs: 2000, log: (m: string) => console.error(m) };

async function cleanupChildren(): Promise<void> {
  dying = true;
  startWatch?.cancel();
  try { await browserRef?.close(); } catch { /* ignore */ }
  try { browserRef?.process()?.kill('SIGKILL'); } catch { /* ignore */ }
  if (ffmpegProc && isAlive(ffmpegProc)) {
    const proc = ffmpegProc;
    const outcome = await killEscalate(proc, { ...escalation, quit: () => { proc.stdin?.write('q'); } });
    console.error(`ffmpeg cleanup: ${outcome}`);
  }
}

const fatalImpl = createFatal({
  log: m => console.error(m),
  tail: () => ffmpegStderr.tail(),
  cleanup: cleanupChildren,
  exit: c => process.exit(c),
});
// Returns a promise that never resolves in practice (process exits); typed never for callers.
const fatal = (message: string, code = 1): Promise<never> =>
  fatalImpl(message, code) as unknown as Promise<never>;

// Puppeteer's own signal handlers are disabled at launch so fatal() owns shutdown (q-stop, escalation, stderr tail).
const SIGNAL_EXIT = { SIGINT: 130, SIGTERM: 143, SIGHUP: 129 } as const;
for (const sig of Object.keys(SIGNAL_EXIT) as (keyof typeof SIGNAL_EXIT)[]) {
  process.on(sig, () => { void fatal(`Recorder received ${sig}; shutting down children`, SIGNAL_EXIT[sig]); });
}
process.on('uncaughtException', err => { void fatal(`Uncaught exception: ${err?.stack || err}`); });
process.on('unhandledRejection', err => { void fatal(`Unhandled rejection: ${err}`); });
// Last-resort synchronous net: never leave ffmpeg behind even if the async path was skipped.
process.on('exit', () => { try { if (ffmpegProc && isAlive(ffmpegProc)) ffmpegProc.kill('SIGKILL'); } catch { /* ignore */ } });

interface RecordOptions {
  duration: number;
  outputPath: string;
  useLocal?: boolean;  // Use localhost:4321 for testing
}

async function recordStrudel(strudelCode: string, options: RecordOptions): Promise<void> {
  const { duration, outputPath, useLocal = false } = options;

  // URL configuration - use self-hosted Strudel embed endpoint (or localhost for testing)
  const STRUDEL_URL = useLocal ? 'http://localhost:4321/embed' : 'https://strudel.dygy.app/embed';
  const STRUDEL_ORIGIN = useLocal ? 'http://localhost:4321' : 'https://strudel.dygy.app';

  console.log('━'.repeat(60));
  console.log('Strudel BlackHole Recorder');
  console.log('━'.repeat(60));
  console.log(`Duration: ${duration}s | Output: ${outputPath}`);
  console.log(`Using: ${STRUDEL_URL}`);

  // Start ffmpeg recording from BlackHole.
  // CRITICAL TEMPO FIX: avfoundation hands ffmpeg BlackHole's samples with timestamps from the
  // device clock that don't track real time, so the captured audio came out ~25% FAST (a 136 BPM
  // track recorded as ~170 BPM / read as 103 by tempo trackers). `-use_wallclock_as_timestamps 1`
  // (before -i) restamps every incoming buffer by the host wall-clock, and `-af aresample=async=1`
  // then adds/drops samples to honour those timestamps — restoring real-time duration. Verified on
  // a setcps(0.25) click train: 48.0s expected -> 48.09s captured (was 38.5s). Without this, EVERY
  // render is sped up and all similarity/tempo numbers are computed against mis-timed audio.
  // Oct 2026 finding: the wall-clock + async path is right LONG-TERM (a 170 s click train fits
  // 135.993 BPM) but adds ±35 ms local timing jitter (15 s windows read 132–148 BPM), because
  // aresample=async=1 inserts/drops chunks whenever a buffer's wall-clock stamp disagrees with the
  // sample count. That jitter destroys beat-period coherence on syncopated music (renders read
  // ~123 BPM by every tracker while a sparse click survives). MIDIGREP_CAPTURE=raw captures the
  // device stream untouched (continuous samples, no restamping); its speed is then corrected ONCE,
  // uniformly, by the measured wall-clock/sample ratio in the trim step below.
  // Default is RAW since 2026-10-09 (measured on a 170 s click train: raw+uniform correction →
  // inter-click std 12 ms, 0 gaps > 50 ms; wall-clock+async → std 38 ms, 61 gaps > 50 ms).
  // MIDIGREP_CAPTURE=wallclock restores the Jun 2026 behaviour.
  const captureMode = process.env.MIDIGREP_CAPTURE === 'wallclock' ? 'wallclock' : 'raw';
  console.log(`Starting ffmpeg recording from BlackHole (${captureMode} capture)...`);
  // -thread_queue_size: avfoundation delivers buffers on a real-time thread; when ffmpeg's input
  // queue (default 8 packets) is full under load, buffers are DROPPED — measured as 10–20% of the
  // stream missing in raw mode and as ±35 ms gaps after async repair. A deep queue prevents that.
  const ffmpegArgs = captureMode === 'raw'
    ? ['-thread_queue_size', '16384', '-f', 'avfoundation', '-i', ':BlackHole 2ch',
       '-t', String(duration + 10), '-ac', '2', '-y', outputPath]
    : ['-thread_queue_size', '16384', '-use_wallclock_as_timestamps', '1',
       '-f', 'avfoundation', '-i', ':BlackHole 2ch',
       '-t', String(duration + 10),
       '-af', 'aresample=async=1',
       '-ar', '44100', '-ac', '2', '-y', outputPath];
  // A stale file from an earlier run would defeat the "ffmpeg created output" check below.
  try { fs.unlinkSync(outputPath); } catch { /* none */ }
  const captureStartMs = Date.now();
  const ffmpeg = spawn('ffmpeg', ffmpegArgs, { stdio: ['pipe', 'pipe', 'pipe'] });
  ffmpegProc = ffmpeg;
  // Drain both pipes: stderr was previously never read, and a full pipe blocks ffmpeg. Keep a tail.
  ffmpeg.stderr?.on('data', d => ffmpegStderr.push(d));
  ffmpeg.stdout?.on('data', () => {});
  ffmpeg.on('error', err => { void fatal(`Failed to run ffmpeg: ${err.message}`); });
  // Early detection: before play only a dead ffmpeg is an error; watch.arm() below starts the no-output clock.
  const watch = watchCaptureStart({
    child: ffmpeg,
    deadlineMs: NO_OUTPUT_DEADLINE_MS,
    onStarted: ms => console.log(`ffmpeg capture started after ${(ms / 1000).toFixed(2)} s`),
    sizeOf: () => { try { return fs.statSync(outputPath).size; } catch { return 0; } },
  });
  startWatch = watch;
  watch.result.then(o => {
    if (o === 'no-output') {
      void fatal(`ffmpeg produced no output ${NO_OUTPUT_DEADLINE_MS / 1000} s after playback started — BlackHole capture did not start; device busy or wedged`);
    } else if (o === 'ffmpeg-exited' && !dying) {
      void fatal(`ffmpeg exited (code ${ffmpeg.exitCode}, signal ${ffmpeg.signalCode}) before producing output — BlackHole capture did not start`);
    }
  });

  await new Promise(r => setTimeout(r, SPAWN_SETTLE_MS));

  // Launch browser (NOT headless - Web Audio is silent in headless mode)
  // Position offscreen with minimal size, disable background throttling
  console.log('Opening browser (background)...');
  const browser = await puppeteer.launch({
    headless: false,
    handleSIGINT: false,
    handleSIGTERM: false,
    handleSIGHUP: false,
    args: [
      '--no-sandbox',
      '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream',
      '--disable-features=MediaStreamSystemSettingsPrompt',
      // Minimal window far offscreen
      '--window-position=-32000,-32000',
      '--window-size=1,1',
      // Prevent throttling when window is in background/offscreen
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      // Additional background flags
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check'
    ]
  });
  browserRef = browser;

  // Use AppleScript to hide Chrome window (macOS only)
  try {
    const { execSync } = await import('child_process');
    execSync(`osascript -e 'tell application "System Events" to set visible of process "Chromium" to false'`, { stdio: 'ignore' });
  } catch (e) {
    // Ignore - not critical
  }

  // Use default context (NOT incognito) to preserve cached samples
  // Grant microphone permission so enumerateDevices() returns device labels (including audiooutput)
  const context = browser.defaultBrowserContext();
  await context.overridePermissions(STRUDEL_ORIGIN, ['microphone']);

  const page = await browser.newPage();

  // Track browser console messages
  page.on('console', msg => {
    const text = msg.text();
    // Filter to relevant messages
    if (text.includes('error') || text.includes('Error') ||
        text.includes('superdough') || text.includes('cyclist') ||
        text.includes('eval') || text.includes('Audio') ||
        text.includes('load') || text.includes('sample')) {
      console.log(`[BROWSER] ${msg.type()}: ${text.substring(0, 200)}`);
    }
  });
  page.on('pageerror', err => console.log(`[PAGE ERROR] ${err.message}`));

  // Track failed network requests
  page.on('requestfailed', request => {
    console.log(`[NET FAIL] ${request.url().substring(0, 100)} - ${request.failure()?.errorText}`);
  });

  await page.goto(STRUDEL_URL, { waitUntil: 'networkidle2', timeout: 60000 });
  await page.waitForSelector('.cm-content', { timeout: 30000 });
  console.log('Page loaded');

  // Step 1: Try enumerateDevices WITHOUT getUserMedia (modern Chrome supports this)
  console.log('Finding BlackHole device...');
  let blackholeId = await page.evaluate(async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioOutputs = devices.filter(d => d.kind === 'audiooutput');
    const blackhole = audioOutputs.find(d => d.label.includes('BlackHole'));
    return blackhole?.deviceId || null;
  });

  // Step 2: If labels are empty (privacy restriction), fall back to getUserMedia with fake device
  if (!blackholeId) {
    console.log('Device labels restricted, requesting permission with fake device...');
    await page.evaluate(async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
      } catch (e) { /* fake device via --use-fake-device-for-media-stream, ignore errors */ }
    });
    await new Promise(r => setTimeout(r, 500));

    blackholeId = await page.evaluate(async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioOutputs = devices.filter(d => d.kind === 'audiooutput');
      const blackhole = audioOutputs.find(d => d.label.includes('BlackHole'));
      return blackhole?.deviceId || null;
    });
  }

  if (!blackholeId) {
    throw new Error('ERROR: BlackHole not found. Install with: brew install blackhole-2ch');
  }
  console.log(`Found BlackHole: ${blackholeId.substring(0, 16)}...`);

  // Store blackholeId for later use (after click play)
  console.log('BlackHole device ready, will set sink after play starts...');

  // Set code using CodeMirror's proper API
  console.log('Setting code...');
  const codeSet = await page.evaluate((code) => {
    // Find the CodeMirror EditorView instance
    const cmContent = document.querySelector('.cm-content') as any;
    if (!cmContent) return { error: 'No cm-content' };

    // Get the EditorView from CodeMirror's DOM binding
    const view = (cmContent as any).cmView?.view;
    if (view && view.dispatch) {
      // Use CodeMirror's transaction API to replace all content
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: code }
      });
      return { success: true, method: 'dispatch', docLength: view.state.doc.length };
    }

    // Fallback: try selecting all and typing
    cmContent.focus();
    document.execCommand('selectAll', false, undefined);
    document.execCommand('insertText', false, code);
    return { success: true, method: 'execCommand' };
  }, strudelCode);
  console.log('Code set:', JSON.stringify(codeSet));

  // Wait a bit for CodeMirror to process the code
  await new Promise(r => setTimeout(r, 1000));

  // Click play button
  console.log('Clicking play button...');
  const buttons = await page.$$('button');
  let playClicked = false;
  for (const btn of buttons) {
    const text = await btn.evaluate(el => el.textContent?.toLowerCase() || '');
    if (text.includes('play')) {
      await btn.click();
      playClicked = true;
      break;
    }
  }
  if (!playClicked) {
    throw new Error('ERROR: Play button not found');
  }

  // Route the AudioContext to BlackHole ASAP. setSinkId only works once superdough has created
  // the context (right after the play click), so we POLL in a tight loop instead of waiting a
  // fixed 500ms. That fixed gap let ~0.5s of cycle-0 play to the DEFAULT device (audible on the
  // speakers — what looked like "plays before recording") and never reach BlackHole, so the
  // captured audio started mid-cycle-0 and the start point drifted run-to-run → inconsistent
  // stems. Confirming ctx.sinkId === BlackHole within ~50ms makes the routing tight & repeatable.
  console.log('Routing audio output to BlackHole (tight retry)...');
  let sinkOk = false;
  const sinkDeadline = Date.now() + 8000;
  while (Date.now() < sinkDeadline) {
    const res = await page.evaluate(async (deviceId: string) => {
      try {
        const ctx = (window as any).getAudioContext?.();
        if (!ctx) return { pending: true };
        // @ts-ignore - setSinkId exists on AudioContext
        if (ctx.sinkId !== deviceId) await ctx.setSinkId(deviceId);
        return { success: ctx.sinkId === deviceId, sinkId: ctx.sinkId, state: ctx.state };
      } catch (e: any) {
        return { error: e.message };
      }
    }, blackholeId);
    if ((res as any).success) { sinkOk = true; console.log('Routed to BlackHole:', JSON.stringify(res)); break; }
    await new Promise(r => setTimeout(r, 50));
  }
  if (!sinkOk) {
    throw new Error('ERROR: Failed to route audio sink to BlackHole within 8s.');
  }

  // Wait for playback to start (either "stop" button appears OR AudioContext is running)
  console.log('Waiting for playback to start...');
  const startTime = Date.now();
  let isPlaying = false;
  while (Date.now() - startTime < 60000) {
    isPlaying = await page.evaluate(() => {
      // Check for stop button (main site UI)
      const btns = document.querySelectorAll('button');
      for (const btn of btns) {
        if (btn.textContent?.toLowerCase().includes('stop')) return true;
      }
      // Check if AudioContext is running (embed mode)
      try {
        // @ts-ignore - getAudioContext is Strudel's global function
        const ctx = (window as any).getAudioContext?.();
        if (ctx && ctx.state === 'running') return true;
      } catch (e) {
        // Ignore - AudioContext might not be ready yet
      }
      return false;
    });
    if (isPlaying) {
      console.log('Playback started, recording...');
      watch.arm();
      break;
    }
    await new Promise(r => setTimeout(r, 200));
  }
  if (!isPlaying) {
    throw new Error('ERROR: Playback did not start within 60 seconds.');
  }

  // Wait a bit more for samples to fully load
  await new Promise(r => setTimeout(r, POST_PLAY_SETTLE_MS));

  // Wait for duration
  console.log(`Recording for ${duration} seconds...`);
  await new Promise(r => setTimeout(r, duration * 1000));

  // Stop playback
  console.log('Stopping...');
  await page.evaluate(() => {
    const btns = document.querySelectorAll('button');
    for (const btn of btns) {
      if (btn.textContent?.toLowerCase().includes('stop')) {
        btn.click();
        return;
      }
    }
  });

  await browser.close();
  const captureWallSeconds = (Date.now() - captureStartMs) / 1000;
  console.log(`Capture wall-clock span: ${captureWallSeconds.toFixed(3)} s (mode ${captureMode})`);
  watch.cancel();
  browserRef = null;
  try { ffmpeg.stdin?.write('q'); } catch { /* stdin already closed */ }
  // 3 s for a graceful 'q' finish, then SIGTERM (2 s grace), then SIGKILL — never leave ffmpeg behind.
  await new Promise<void>(resolve => {
    if (!isAlive(ffmpeg)) return resolve();
    const t = setTimeout(resolve, 3000);
    ffmpeg.once('exit', () => { clearTimeout(t); resolve(); });
  });
  const stopOutcome = await killEscalate(ffmpeg, { ...escalation, quitGraceMs: 0 });
  if (stopOutcome !== 'already-dead') console.log(`ffmpeg did not exit on 'q'; ${stopOutcome}`);

  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size === 0) {
    await fatal('Recording failed: ffmpeg left no output file');
  }

  // Trim leading silence (Strudel takes a few seconds to load before audio starts)
  console.log('Trimming leading silence...');
  const trimmedPath = outputPath.replace('.wav', '_trimmed.wav');
  // Raw mode: the device stream is labelled 48 kHz but BlackHole delivers it at a different, run-
  // dependent real rate (measured 11.9% fast on 2026-10-09; 25% in Jun 2026). Relabel the sample
  // rate ONCE by the measured samples/wall-clock ratio, then resample to 44.1 kHz — a uniform
  // correction, no per-chunk insert/drop, so beat timing inside the file stays intact.
  let speedFilter = '';
  if (captureMode === 'raw') {
    const probe = spawnSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration',
      '-of', 'csv=p=0', outputPath], { encoding: 'utf8' });
    const rawSeconds = parseFloat((probe.stdout || '').trim());
    if (Number.isFinite(rawSeconds) && rawSeconds > 1 && captureWallSeconds > 1) {
      const realRate = Math.round(48000 * rawSeconds / captureWallSeconds);
      console.log(`Raw capture: ${rawSeconds.toFixed(3)} s of audio over ${captureWallSeconds.toFixed(3)} s wall-clock ` +
        `-> real device rate ${realRate} Hz (ratio ${(rawSeconds / captureWallSeconds).toFixed(4)}); correcting uniformly`);
      speedFilter = `asetrate=${realRate},aresample=44100,`;
    } else {
      console.log('Raw capture: could not measure the speed ratio, leaving speed uncorrected');
    }
  }
  const trimProcess = spawn('ffmpeg', [
    '-i', outputPath,
    '-af', speedFilter + 'silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB',
    '-y', trimmedPath
  ], { stdio: ['pipe', 'pipe', 'pipe'] });

  await new Promise<void>((resolve, reject) => {
    trimProcess.on('close', (code) => {
      if (code === 0 && fs.existsSync(trimmedPath)) {
        // Replace original with trimmed
        fs.unlinkSync(outputPath);
        fs.renameSync(trimmedPath, outputPath);
        resolve();
      } else {
        console.log('Silence trim failed, keeping original');
        resolve();
      }
    });
    trimProcess.on('error', () => resolve());
  });

  const mb = (fs.statSync(outputPath).size / 1024 / 1024).toFixed(1);
  console.log(`━━━ Saved: ${outputPath} (${mb} MB) ━━━`);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.log('Usage: record-strudel-blackhole.js <input.strudel> [-o output.wav] [-d duration] [--local]');
    console.log('  --local  Use localhost:4321 instead of strudel.dygy.app (for testing)');
    process.exit(1);
  }

  const inputFile = args[0];
  let outputPath = '/tmp/strudel_recording.wav';
  let duration = 30;
  let useLocal = false;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '-o') outputPath = args[++i];
    else if (args[i] === '-d') duration = parseFloat(args[++i]);
    else if (args[i] === '--local') useLocal = true;
  }

  if (!fs.existsSync(inputFile)) {
    console.error(`File not found: ${inputFile}`);
    process.exit(1);
  }

  await recordStrudel(fs.readFileSync(inputFile, 'utf-8'), { duration, outputPath, useLocal });
}

main().catch(err => { void fatal(`Error: ${err?.stack || err}`); });
