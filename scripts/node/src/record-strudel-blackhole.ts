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
 * Usage:
 *   node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import * as fs from 'fs';

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

  // Start ffmpeg recording from BlackHole
  console.log('Starting ffmpeg recording from BlackHole...');
  const ffmpeg = spawn('ffmpeg', [
    '-f', 'avfoundation', '-i', ':BlackHole 2ch',
    '-t', String(duration + 10), '-ar', '44100', '-ac', '2', '-y', outputPath
  ], { stdio: ['pipe', 'pipe', 'pipe'] });

  await new Promise(r => setTimeout(r, 1000));

  // Launch browser (NOT headless - Web Audio is silent in headless mode)
  // Position offscreen with minimal size, disable background throttling
  console.log('Opening browser (background)...');
  const browser = await puppeteer.launch({
    headless: false,
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

  // Use AppleScript to hide Chrome window (macOS only)
  try {
    const { execSync } = await import('child_process');
    execSync(`osascript -e 'tell application "System Events" to set visible of process "Chromium" to false'`, { stdio: 'ignore' });
  } catch (e) {
    // Ignore - not critical
  }

  // Use default context (NOT incognito) to preserve cached samples
  const context = browser.defaultBrowserContext();
  await context.overridePermissions(STRUDEL_ORIGIN, ['microphone', 'camera']);

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

  // Request mic permission to enumerate devices
  console.log('Requesting audio permission...');
  await page.evaluate(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
    } catch (e) {
      console.log('getUserMedia failed:', e);
    }
  });
  await new Promise(r => setTimeout(r, 500));

  // Get audio output devices and find BlackHole
  console.log('Finding BlackHole device...');
  const blackholeId = await page.evaluate(async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioOutputs = devices.filter(d => d.kind === 'audiooutput');
    const blackhole = audioOutputs.find(d => d.label.includes('BlackHole'));
    return blackhole?.deviceId || null;
  });

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

  // Wait a moment for superdough to initialize
  await new Promise(r => setTimeout(r, 500));

  // Now set AudioContext sink to BlackHole (after superdough has initialized)
  console.log('Setting audio output to BlackHole...');
  const sinkResult = await page.evaluate(async (deviceId: string) => {
    try {
      // @ts-ignore - getAudioContext is Strudel's global function
      const ctx = (window as any).getAudioContext();
      if (!ctx) return { error: 'AudioContext not found' };
      // @ts-ignore - setSinkId exists on AudioContext
      await ctx.setSinkId(deviceId);
      return { success: true, sinkId: ctx.sinkId, state: ctx.state };
    } catch (e: any) {
      return { error: e.message };
    }
  }, blackholeId);

  if ((sinkResult as any).error) {
    throw new Error(`ERROR: Failed to set audio sink: ${(sinkResult as any).error}`);
  }
  console.log('Audio output set to BlackHole:', JSON.stringify(sinkResult));

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
      break;
    }
    await new Promise(r => setTimeout(r, 200));
  }
  if (!isPlaying) {
    throw new Error('ERROR: Playback did not start within 60 seconds.');
  }

  // Wait a bit more for samples to fully load
  await new Promise(r => setTimeout(r, 2000));

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
  ffmpeg.stdin?.write('q');
  await new Promise<void>(resolve => {
    ffmpeg.on('close', resolve);
    setTimeout(() => { ffmpeg.kill('SIGTERM'); resolve(); }, 3000);
  });

  if (fs.existsSync(outputPath)) {
    const mb = (fs.statSync(outputPath).size / 1024 / 1024).toFixed(1);
    console.log(`━━━ Saved: ${outputPath} (${mb} MB) ━━━`);
  } else {
    console.error('Recording failed');
    process.exit(1);
  }
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

main().catch(err => { console.error('Error:', err); process.exit(1); });
