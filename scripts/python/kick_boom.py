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
  * level: the RMS of the sustained part's FUNDAMENTAL (+-1 semitone around the pitch), absolute. A sine carries only the
    fundamental; a saturated 808's harmonics belong to the bass band (Sem Aliança bars 12-24: the fundamental holds 31 % of
    the sustained power, the whole band read 1.7x the level that scored best).
The generator plays gain = level x master gain x KB / SINE_UNIT: KB (calibrate_dynamic.py) is the fraction of its stem
the bass voice renders at, measured from the bass voice alone (render amplitude / (stem amplitude x master gain)), and
SINE_UNIT the RMS the engine renders a sine at gain 1 (``kick_boom.py probe`` -> data/kick_boom_unit.json). Before, the
level was relative to the bass stem's median and rode the sub sine's gain: a weak bass stem (Tatuagem's stage-3c view)
read the boom at 12-18x, and KB 0.40 put the render's sub_bass at 91.5 % vs 47 % (0.762 -> 0.548).
The notes sit on the kick hits of the detected drum grid and hold to the next hit (the tone rings through the beat).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bass_register as BR  # noqa: E402
import stem_repair as SR  # noqa: E402

KICK = "bd"
UNIT_PATH = Path(__file__).resolve().parent / "data" / "kick_boom_unit.json"
PROBE_NOTE = 33                 # A1 55 Hz: inside the boom's range (the measured booms sit at 26-61 Hz)


def sine_unit(path: Path = UNIT_PATH) -> float | None:
    """RMS of a sustained engine sine at gain 1 (``probe``), or None when it was never measured."""
    try:
        return float(json.loads(path.read_text())["rms"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass
class BoomBar:
    pitch: float | None          # MIDI (fractional); None = no boom in this bar
    level: float                 # RMS of the sustained part's fundamental (+-1 semitone), absolute (stem full scale)


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
        semi = 2.0 ** (1.0 / 12.0)
        power = spec ** 2
        fund = float(power[(f >= hz / semi) & (f <= hz * semi)].sum() / power[band].sum())
        out.append(BoomBar(round(69.0 + 12.0 * np.log2(hz / 440.0), 2), round(float(eh[k] * np.sqrt(fund)), 5)))
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


def probe(out: Path = UNIT_PATH) -> float:
    """Render a sustained sine at gain 1 through the selected renderer and record its RMS (the steady part)."""
    import subprocess
    import tempfile

    import soundfile as sf
    import renderer as RND

    seconds, settle = 8, 2
    with tempfile.TemporaryDirectory() as t:
        code, wav = Path(t) / "unit.strudel", Path(t) / "unit.wav"
        code.write_text(f'setcps(0.5)\n$: note("{PROBE_NOTE}").s("sine").gain(1)\n')
        subprocess.run(["node", str(RND.renderer_script()), str(code), "-o", str(wav), "-d", str(seconds)], check=True,
                       capture_output=True)
        y, sr = sf.read(str(wav), always_2d=True)
        seg = y[settle * sr:(seconds - settle) * sr].mean(1)
        rms = float(np.sqrt(np.mean(seg ** 2)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rms": round(rms, 6), "note": PROBE_NOTE, "renderer": RND.renderer_name(),
                               "code": f'note("{PROBE_NOTE}").s("sine").gain(1)'}, indent=1) + "\n")
    return rms


if __name__ == "__main__":
    if sys.argv[1:] == ["probe"]:
        print(f"sine unit rms {probe():.6f} -> {UNIT_PATH}")
    else:
        sys.exit("usage: kick_boom.py probe")
