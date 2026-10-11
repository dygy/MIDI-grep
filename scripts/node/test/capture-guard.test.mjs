// Run: npm run build && npm test   (node --test, no extra deps)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { killEscalate, TailBuffer, watchCaptureStart } from '../dist/capture-guard.js';

class FakeChild extends EventEmitter {
  constructor({ ignoreTerm = false, ignoreKill = false } = {}) {
    super(); this.exitCode = null; this.signalCode = null; this.sent = [];
    this.ignoreTerm = ignoreTerm; this.ignoreKill = ignoreKill;
  }
  kill(sig = 'SIGTERM') {
    this.sent.push(sig);
    if (sig === 'SIGTERM' && this.ignoreTerm) return true;
    if (sig === 'SIGKILL' && this.ignoreKill) return true;
    this.signalCode = sig; this.emit('exit', null, sig); return true;
  }
}

test('killEscalate: honours SIGTERM', async () => {
  const c = new FakeChild();
  assert.equal(await killEscalate(c, { termGraceMs: 50 }), 'exited-on-sigterm');
  assert.deepEqual(c.sent, ['SIGTERM']);
});
test('killEscalate: quit first; no signal sent if child exits on quit', async () => {
  const c = new FakeChild(); const logs = [];
  const out = await killEscalate(c, { quit: () => { c.exitCode = 0; c.emit('exit', 0, null); }, quitGraceMs: 50, termGraceMs: 50, log: m => logs.push(m) });
  assert.equal(out, 'exited-on-quit'); assert.deepEqual(c.sent, []); assert.ok(logs.some(l => l.includes('quit')));
});
test('killEscalate: quit ignored -> SIGTERM -> SIGKILL only after the full SIGTERM grace', async () => {
  const c = new FakeChild({ ignoreTerm: true }); const t0 = Date.now(); let killAt = 0;
  const orig = c.kill.bind(c); c.kill = s => { if (s === 'SIGKILL') killAt = Date.now() - t0; return orig(s); };
  const out = await killEscalate(c, { quit: () => {}, quitGraceMs: 30, termGraceMs: 120, killWaitMs: 50 });
  assert.equal(out, 'killed'); assert.deepEqual(c.sent, ['SIGTERM', 'SIGKILL']);
  assert.ok(killAt >= 140, 'SIGKILL must wait quit+term grace, got ' + killAt);
});
test('killEscalate: unkillable is reported, not hung', async () => {
  const c = new FakeChild({ ignoreTerm: true, ignoreKill: true });
  assert.equal(await killEscalate(c, { termGraceMs: 30, killWaitMs: 30 }), 'unkillable');
});
test('killEscalate: already dead is a no-op', async () => {
  const c = new FakeChild(); c.exitCode = 0;
  assert.equal(await killEscalate(c, {}), 'already-dead');
  assert.deepEqual(c.sent, []);
});

test('watchCaptureStart: slow device open before arm() is NEVER no-output; late output is ok + latency logged', async () => {
  let size = 0; let lat = null; setTimeout(() => { size = 44; }, 200);
  const w = watchCaptureStart({ child: new FakeChild(), sizeOf: () => size, deadlineMs: 30, pollMs: 10, onStarted: ms => { lat = ms; } });
  assert.equal(await w.result, 'ok');
  assert.ok(lat >= 190, 'latency ' + lat);
});
test('watchCaptureStart: no-output only after arm(), counted from arm()', async () => {
  const w = watchCaptureStart({ child: new FakeChild(), sizeOf: () => 0, deadlineMs: 100, pollMs: 10 });
  let settled = false; w.result.then(() => { settled = true; });
  await new Promise(r => setTimeout(r, 250));
  assert.equal(settled, false, 'must not fire while unarmed');
  const t0 = Date.now(); w.arm();
  assert.equal(await w.result, 'no-output');
  assert.ok(Date.now() - t0 >= 90);
});
test('watchCaptureStart: ffmpeg-exited is detected before arm()', async () => {
  const c = new FakeChild();
  setTimeout(() => { c.exitCode = 1; c.emit('exit', 1, null); }, 30);
  const w = watchCaptureStart({ child: c, sizeOf: () => 0, deadlineMs: 1000, pollMs: 10 });
  assert.equal(await w.result, 'ffmpeg-exited');
});
test('watchCaptureStart: output arriving after arm() but before deadline is ok', async () => {
  let size = 0; const w = watchCaptureStart({ child: new FakeChild(), sizeOf: () => size, deadlineMs: 200, pollMs: 10 });
  w.arm(); setTimeout(() => { size = 78; }, 50);
  assert.equal(await w.result, 'ok');
});
test('watchCaptureStart: cancel stops polling', async () => {
  const w = watchCaptureStart({ child: new FakeChild(), sizeOf: () => 0, deadlineMs: 50, pollMs: 10 });
  w.arm(); w.cancel();
  assert.equal(await w.result, 'ok');
});

test('TailBuffer keeps last N lines incl. \\r progress and partial line', () => {
  const b = new TailBuffer(3);
  b.push('a\nb\r'); b.push('c\nd\ne'); 
  assert.equal(b.tail(), 'c\nd\ne');
});

import { createFatal } from '../dist/capture-guard.js';

test('createFatal: concurrent calls -> one cleanup, first exit code wins, exit only after cleanup', async () => {
  const events = []; let release; let cleanups = 0;
  const gate = new Promise(r => { release = r; });
  const fatal = createFatal({
    log: m => events.push('log:' + m.split('\n')[0]),
    tail: () => 'ffmpeg: device busy',
    cleanup: async () => { cleanups++; events.push('cleanup-start'); await gate; events.push('cleanup-end'); },
    exit: c => events.push('exit:' + c),
  });
  const p1 = fatal('no output', 1);
  const p2 = fatal('page.goto rejected', 7);   // e.g. main().catch after browser.close()
  await new Promise(r => setTimeout(r, 20));
  assert.deepEqual(events.filter(e => e.startsWith('exit')), [], 'must not exit while cleanup is in flight');
  release(); await Promise.all([p1, p2]);
  assert.equal(cleanups, 1);
  assert.deepEqual(events.filter(e => e.startsWith('exit')), ['exit:1', 'exit:1']);
  assert.ok(events.indexOf('cleanup-end') < events.indexOf('exit:1'));
  assert.ok(!events.some(e => e.includes('page.goto')), 'later message is not logged');
});
test('createFatal: cleanup throwing still exits with the first code', async () => {
  const exits = [];
  const fatal = createFatal({ log() {}, tail: () => '', cleanup: async () => { throw new Error('x'); }, exit: c => exits.push(c) });
  await fatal('a', 130);
  assert.deepEqual(exits, [130]);
});
