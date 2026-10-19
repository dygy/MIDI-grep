import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureDeviceRate, legacyRealRate } from '../dist/capture-rate.js';

const BPF = 8; // f32 stereo
// Device really delivers 0.85*48000 fps. ffmpeg spawned at t=0, device opens at 3 s, capture
// runs to 38 s, then ~2 s stop/close.
function synth(realFps, openS, endS, stepMs = 250) {
  const out = [];
  for (let t = 0; t <= endS * 1000; t += stepMs) {
    const flowS = Math.max(0, t / 1000 - openS);
    out.push({ tMs: 1e6 + t, bytes: flowS > 0 ? 44 + Math.floor(flowS * realFps * BPF / 32768) * 32768 : 0 });
  }
  return out;
}
test('measures the steady-state rate independent of open latency', () => {
  const m = measureDeviceRate(synth(0.85 * 48000, 3, 38), BPF, 44);
  assert.ok(Math.abs(m.realRate / (0.85 * 48000) - 1) < 0.002, `got ${m.realRate}`);
  assert.ok(m.deltaSeconds > 30);
});
test('legacy formula (process-lifetime) is biased by latency; new one is not', () => {
  const real = 0.85 * 48000, wall = 40, rawSeconds = (wall - 3 - 2) * real / 48000; // 3 s open + 2 s close
  const legacy = legacyRealRate(rawSeconds, wall);
  assert.ok(Math.abs(legacy / real - 1) > 0.05, `legacy off by ${(legacy / real - 1) * 100}%`);
  const m = measureDeviceRate(synth(real, 3, 38), BPF, 44);
  assert.ok(Math.abs(m.realRate / real - 1) < 0.002);
});
test('too-short span -> null', () => {
  assert.equal(measureDeviceRate(synth(40000, 1, 4), BPF, 44), null);
  assert.equal(measureDeviceRate([], BPF, 44), null);
});
