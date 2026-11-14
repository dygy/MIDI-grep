#!/usr/bin/env python3
"""The drums stem's sustained low tone (a long tuned 808 kick, "boom") as a synth-mode voice.

Brazilian funk kicks are often a tuned 808 whose tone rings at one pitch through the beat. Demucs files it under drums,
and the synth palette's drum machine kick is short and higher (Sem Aliança: the stem's low tone at 45 Hz, still 0.74 of
its peak power at the end of the beat; the RolandTR808 bd at 54 Hz, 0.13). The mix then lacks that sub wherever the boom
plays (Sem Aliança bars 11-29: render sub_bass share 36 % vs 64 %).

Not a bass: its pitch stays put while the bass stem moves (Sem bars 9-24: 45 Hz under bass notes at 58 / 73 / 59 Hz),
so stage 3c (``stem_repair.py``), which moves a sustained BASS out of the drums, rightly leaves it. When 3c did move a
part into the bass view, the generator reads the view's drums, where that part is gone, so nothing is played twice.

Per bar, in the band stage 3c uses (``stem_repair.harmonic_low``: below twice the bass register's top, one-beat HPSS):
  * present: the sustained part carries more power than the percussive part AND is at least ``silence`` x the bass
    stem's median active level (what the generator's gate calls present - ``bar_env_pattern``'s rule);
  * pitch: the strongest bin of the sustained part between the analysis floor (``bass_register.SUB_FLOOR_HZ``) and that
    band's top (MIDI, fractional - the stem's own tuning);
  * level: its RMS over the bass stem's median active RMS in the same band. The generator multiplies it by the sub sine
    layer's base gain, which stands for the bass stem at that median level, so the calibrator's sub-gain lever moves the
    boom together with the bass sub (both are sines standing for a stem's low band).
The notes sit on the kick hits of the detected drum grid and hold to the next hit (the tone rings through the beat).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bass_register as BR  # noqa: E402
import stem_repair as SR  # noqa: E402

KICK = "bd"


@dataclass
class BoomBar:
    pitch: float | None          # MIDI (fractional); None = no boom in this bar
    level: float                 # sustained-part RMS / the bass stem's median active RMS in the same band


def _bar_rms(x: np.ndarray, bar: float, nbars: int) -> np.ndarray:
    return np.array([float(np.sqrt(np.mean(s ** 2))) if (s := x[int(round(k * bar)):int(round((k + 1) * bar))]).size else 0.0
                     for k in range(nbars)])


def measure(drums: Path, bass: Path, bass_f0: Path, bpm: float, nbars: int, silence: float) -> list[BoomBar]:
    import soundfile as sf

    window = BR.Register.load(bass_f0).window()
    d, sr = sf.read(str(drums), dtype="float32", always_2d=True)
    b, sr2 = sf.read(str(bass), dtype="float32", always_2d=True)
    if sr2 != sr:
        raise ValueError(f"drums is {sr} Hz but bass is {sr2} Hz")
    harm, perc, lo_sr = SR.harmonic_low(d.mean(1), sr, bpm, window)
    bh, bp, _ = SR.harmonic_low(b.mean(1), sr, bpm, window)
    bar = 60.0 / bpm * 4 * lo_sr
    eh, ep, eb = (_bar_rms(v, bar, nbars) for v in (harm, perc, bh + bp))
    active = eb[eb > eb.max() * 0.01] if eb.max() > 0 else eb      # bar_env_pattern's reference: the median active level
    ref = float(np.median(active)) if active.size else 0.0
    if ref <= 0.0:
        return [BoomBar(None, 0.0) for _ in range(nbars)]
    # floor: the analysis floor, not the bass's lowest note - a boom may sit under every note the bass plays
    floor_hz, top_hz = BR.SUB_FLOOR_HZ, 2.0 * SR._midi_hz(window[1])
    out = []
    for k in range(nbars):
        if not (eh[k] > ep[k] and eh[k] >= silence * ref):
            out.append(BoomBar(None, 0.0))
            continue
        seg = harm[int(round(k * bar)):int(round((k + 1) * bar))]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        f = np.fft.rfftfreq(seg.size, 1.0 / lo_sr)
        band = (f >= floor_hz) & (f <= top_hz)
        if not band.any() or spec[band].max() <= 0.0:
            out.append(BoomBar(None, 0.0))
            continue
        hz = float(f[band][int(np.argmax(spec[band]))])
        out.append(BoomBar(round(69.0 + 12.0 * np.log2(hz / 440.0), 2), round(float(eh[k] / ref), 4)))
    return out


def bar_strings(boom: list[BoomBar], kick_steps: set[int], quantize: int) -> list[str]:
    """One mini-notation string per bar: the bar's pitch on each kick hit, held to the next hit (or the bar end)."""
    bars = []
    for k, bb in enumerate(boom):
        hits = sorted(s - k * quantize for s in kick_steps if k * quantize <= s < (k + 1) * quantize)
        if bb.pitch is None or not hits:
            bars.append(" ".join(["~"] * quantize))
            continue
        toks = ["~"] * hits[0]
        for i, h in enumerate(hits):
            n = (hits[i + 1] if i + 1 < len(hits) else quantize) - h
            toks.append(f"{bb.pitch:g}" + (f"@{n}" if n > 1 else ""))
        bars.append(" ".join(toks))
    return bars
