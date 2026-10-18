/**
 * Steady-state device-rate measurement for the raw BlackHole capture (no side effects on import).
 *
 * The old formula (48000 * fileSeconds / (spawn -> browser.close wall seconds)) charged avfoundation
 * device-open latency and stop/close time to the device as if they had produced samples, so the
 * error was a ~fixed number of seconds: negligible on 170 s, 4-13% on 35 s. Here the rate comes only
 * from how fast the output file GROWS while audio is flowing.
 */
export interface SizeSample { tMs: number; bytes: number; }
export interface RateMeasurement { realRate: number; deltaFrames: number; deltaSeconds: number; points: number; }

export const MIN_SPAN_SECONDS = 5;
export const WARMUP_SECONDS = 1;

/**
 * Least-squares slope of bytes vs time over the steady-state window: from WARMUP_SECONDS after the
 * file first grew past `headerBytes` up to the last sample. Returns null if the span is too short
 * (or nothing flowed). `nominalRate` is only used to convert bytes -> frames via bytesPerFrame.
 */
export function measureDeviceRate(
  samples: SizeSample[], bytesPerFrame: number, headerBytes = 0, minSpanSeconds = MIN_SPAN_SECONDS,
): RateMeasurement | null {
  if (!(bytesPerFrame > 0)) return null;
  const flowing = samples.filter(s => s.bytes > headerBytes);
  if (flowing.length < 3) return null;
  const t0 = flowing[0].tMs + WARMUP_SECONDS * 1000;
  const win = flowing.filter(s => s.tMs >= t0);
  if (win.length < 3) return null;
  const spanS = (win[win.length - 1].tMs - win[0].tMs) / 1000;
  if (spanS < minSpanSeconds) return null;
  const n = win.length;
  const mt = win.reduce((a, s) => a + s.tMs, 0) / n;
  const mb = win.reduce((a, s) => a + s.bytes, 0) / n;
  let num = 0, den = 0;
  for (const s of win) { num += (s.tMs - mt) * (s.bytes - mb); den += (s.tMs - mt) ** 2; }
  if (den <= 0 || num <= 0) return null;
  const bytesPerSec = (num / den) * 1000;
  const framesPerSec = bytesPerSec / bytesPerFrame;
  return { realRate: Math.round(framesPerSec), deltaFrames: Math.round(framesPerSec * spanS), deltaSeconds: spanS, points: n };
}

/** The retired formula, kept only so the test can show it is latency-sensitive. */
export function legacyRealRate(rawSeconds: number, wallSeconds: number, nominal = 48000): number {
  return Math.round(nominal * rawSeconds / wallSeconds);
}
