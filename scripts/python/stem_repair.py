#!/usr/bin/env python3
"""Return the bass Demucs put in the drums stem to the bass stem, where the bass stem is silent.

Why: in a funk intro or break the 808 often plays without the kick pattern around it, and Demucs files the
sustained sub tone under DRUMS. "Automotivo Novinho Depressivo": bars 4-6 / 17-19 / 33-35 have the bass stem at
-59 dB while the drums stem holds a sustained 48 Hz tone (G1, 95% of the time sounding) at -11 dB. The generator
reads its per-bar voice envelopes from the stems, so it muted the bass exactly there and played the kick LOUDER
(the drums envelope carried the 808): sub_bass 1-19% of the render vs ~50% of the original in those windows.

What moves: the HARMONIC (sustained over one beat, HPSS) part of the drums stem below twice the top of the bass
register, in the generator's envelope steps where (a) the bass voice is silent by the generator's own gate
(``generate_dynamic_strudel.bar_env_pattern``), (b) that harmonic part carries more power than the percussive part
of the same band, and (c) it is loud enough that the same gate would call it present. Nothing else is touched;
bass + drums sums to the same signal, so the instrumental (the scored reference) is unchanged.

The repaired stems are a read-only VIEW for the generator and the bass-solo measurement
(``auto-calibrate.sh --gen-stems``): ``<track>/.pipeline/stems/`` with the new ``bass.wav`` / ``drums.wav``, symlinks
to ``melodic.wav`` / ``instrumental.wav`` and its own ``.pipeline/bass_f0.json``. The Demucs stems, MIDI, sample pack
and drum onsets stay as they were. When no step qualifies the view is removed and the track dir is used.

Every number below is taken from the generator, the bass register or the tempo - no per-track constant.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bass_register as BR  # noqa: E402  (register window + the f0 frame geometry)

VERSION = 1
VIEW_LINKS = ("melodic.wav", "instrumental.wav")   # read by the generator, never written through the view


def _gen():
    import generate_dynamic_strudel as G   # heavy import: only when building
    return G


def _midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _step_rms(x: np.ndarray, step: float, n: int) -> np.ndarray:
    return np.array([float(np.sqrt(np.mean(x[int(round(i * step)):int(round((i + 1) * step))] ** 2)))
                     if x[int(round(i * step)):int(round((i + 1) * step))].size else 0.0 for i in range(n)])


def harmonic_low(x: np.ndarray, sr: int, bpm: float, window: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """(harmonic, percussive) parts of ``x`` below twice the register's top fundamental, at ``bass_register.SR``
    (its f0 frame geometry), with ``x``'s length at that rate. Harmonic = sustained over one beat (HPSS kernel)."""
    import librosa

    lo_sr = BR.SR
    frame, hop = BR._frame_geometry(lo_sr, BR.SUB_FLOOR_HZ)
    beat = max(1, int(round(60.0 / bpm * lo_sr / hop)))
    beat += 1 - beat % 2                               # median kernels are odd
    d = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=lo_sr)
    S = librosa.stft(d, n_fft=frame, hop_length=hop)
    S[librosa.fft_frequencies(sr=lo_sr, n_fft=frame) > 2.0 * _midi_hz(window[1])] = 0.0
    Hs, Ps = librosa.decompose.hpss(S, kernel_size=beat)
    return (librosa.istft(Hs, hop_length=hop, length=len(d)), librosa.istft(Ps, hop_length=hop, length=len(d)), lo_sr)


def misrouted_steps(bass: np.ndarray, harm: np.ndarray, perc: np.ndarray, lo_sr: int, bpm: float, nsub: int,
                    steps: int, bass_env: list[list[float]] | None, silence: float) -> np.ndarray:
    """Boolean mask over the ``nsub`` envelope steps: the generator plays the bass silent there (a 0.0 step of
    ``bass_env``), the drums' low band is mostly sustained (harmonic power > percussive), and that sustained part is
    at least ``silence`` x the bass stem's median active step (what the generator's gate calls present). ``bass``,
    ``harm`` and ``perc`` are mono at ``lo_sr``."""
    step = 60.0 / bpm * 4 / steps * lo_sr
    eb, eh, ep = (_step_rms(v, step, nsub) for v in (bass, harm, perc))
    active = eb[eb > eb.max() * 0.01] if eb.max() > 0 else eb   # bar_env_pattern's reference: the median active step
    ref = float(np.median(active)) if active.size else 0.0
    silent = np.array([v == 0.0 for bar in bass_env for v in bar])[:nsub] if bass_env else np.ones(nsub, bool)
    return silent & (eh > ep) & (eh >= silence * ref)


def _hash(paths: list[Path]) -> str:
    hh = hashlib.sha1()
    for p in paths:
        hh.update(BR.stem_hash(p).encode())
    return hh.hexdigest()


def build(track: Path, out: Path, bpm: float, nbars: int, bass_f0: Path, steps: int | None = None) -> dict:
    """Write the repaired view into ``out`` (or remove it when nothing moves); returns the summary."""
    import librosa
    import soundfile as sf

    G = _gen()
    steps = steps or G.ENV_STEPS
    silence = inspect.signature(G.bar_env_pattern).parameters["silence"].default
    bass_p, drums_p = track / "bass.wav", track / "drums.wav"
    stamp = {"version": VERSION, "inputs": _hash([bass_p, drums_p, bass_f0]), "bpm": bpm, "bars": nbars,
             "steps": steps}
    rec = out / "repair.json"
    if rec.is_file():
        try:
            old = json.loads(rec.read_text())
            if old.get("stamp") == stamp:
                return old
        except (OSError, ValueError):
            pass
    window = BR.Register.load(bass_f0).window()
    b, sr = sf.read(str(bass_p), dtype="float32", always_2d=True)
    dr, sr2 = sf.read(str(drums_p), dtype="float32", always_2d=True)
    if sr2 != sr:
        raise ValueError(f"bass is {sr} Hz but drums is {sr2} Hz")
    n = min(len(b), len(dr))
    b, dr = b[:n], dr[:n]
    env = G.bar_env_pattern(bass_p, nbars, bpm, floor=0.15, steps=steps)   # the generator's own call
    harm, perc, lo_sr = harmonic_low(dr.mean(1), sr, bpm, window)
    b_lo = librosa.resample(b.mean(1), orig_sr=sr, target_sr=lo_sr)
    mask = misrouted_steps(b_lo, harm, perc, lo_sr, bpm, nbars * steps, steps, env, silence)
    moved = [int(i) for i in np.flatnonzero(mask)]
    summary = {"stamp": stamp, "moved_steps": moved, "moved_bars": sorted({i // steps for i in moved}),
               "window": list(window), "view": str(out) if moved else None}
    if not moved:
        if out.exists():
            shutil.rmtree(out)
        return summary
    # sample-level gain: 1 on moved steps, crossfaded over one period of the register's lowest note
    step = 60.0 / bpm * 4 / steps * sr
    g = np.zeros(n)
    for i in moved:
        g[int(round(i * step)):int(round((i + 1) * step))] = 1.0
    fade = max(1, int(round(sr / _midi_hz(window[0]))))
    g = np.convolve(g, np.hanning(fade) / np.hanning(fade).sum(), mode="same")
    hm = np.stack([librosa.resample(harmonic_low(dr[:, c], sr, bpm, window)[0], orig_sr=lo_sr, target_sr=sr)[:n]
                   for c in range(dr.shape[1])], axis=1)   # per channel: each keeps its own share of the tone
    hm = np.pad(hm, ((0, n - len(hm)), (0, 0))) * g[:, None]
    out.mkdir(parents=True, exist_ok=True)
    tmp_b, tmp_d = out / ".bass.tmp.wav", out / ".drums.tmp.wav"
    sf.write(str(tmp_b), (b + hm).astype(np.float32), sr, subtype="FLOAT")
    sf.write(str(tmp_d), (dr - hm).astype(np.float32), sr, subtype="FLOAT")
    os.replace(tmp_b, out / "bass.wav")
    os.replace(tmp_d, out / "drums.wav")
    for name in VIEW_LINKS:
        link, target = out / name, (track / name).resolve()
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    BR.measure(out / "bass.wav", out / ".pipeline" / "bass_f0.json")
    rec.write_text(json.dumps(summary))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build", help="write <out> = the stem view the generator reads (prints the stems dir to use)")
    bp.add_argument("--track", type=Path, required=True)
    bp.add_argument("--out", type=Path, required=True)
    bp.add_argument("--bpm", type=float, required=True)
    bp.add_argument("--bars", type=int, required=True)
    bp.add_argument("--bass-f0", type=Path, required=True, help="bass_register.py f0 track of <track>/bass.wav")
    a = ap.parse_args()
    s = build(a.track, a.out, a.bpm, a.bars, a.bass_f0)
    print(f"stem repair: {len(s['moved_steps'])} envelope steps in bars {s['moved_bars'] or '-'} carry bass from "
          f"the drums stem", file=sys.stderr)
    print(s["view"] or str(a.track))
    return 0


if __name__ == "__main__":
    sys.exit(main())
