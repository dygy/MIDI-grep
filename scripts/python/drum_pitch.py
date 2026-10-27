"""
drum_pitch.py — the pitch each kick of the drums stem settles on, as per-hit data the generator can write.

Why (measured on VAGABUNDO, `drums-carry-pitch`): the track plays ONE tuned 808 kick whose tail sits at midi ~30 for
90 of 139 hits but at ~34-35 for 33 more (and elsewhere for 16). The pattern wrote the same one-shot for all of them,
so the per-frame chroma under those 49 hits correlated -0.4 against the stem (0.985 under the other 90), and the
render's drums chroma fell behind the real drums' (swapping the stem drums in lifted temporal chroma 0.63 -> 0.81).
It is not bass bleed: the bass stem's peak at the same hits sits elsewhere and 5-10x lower.

So each hit's tail pitch is MEASURED on the stem and compared with the one-shot's own tail pitch measured the same way;
the ratio is what `generate_hybrid_strudel.build_detected_drums` writes as a parallel `.speed()` pattern. A hit whose
pitch cannot be read (no clear tonal peak, a tail cut short by the next hit, two reads that disagree) gets ratio 1
- the one-shot as it is - never a pitch the stem does not sound.

Every threshold is read from the stem: the tail window is the one-shot's own tail, the peak must stand above the
band's own median by the factor that the stem's clear hits show, and the two half-window reads must agree within
median + 3 MAD of the stem's own half-to-half disagreement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# band the tonal tail of a kick is looked for in; the lower edge is the stem's useful floor (Demucs high-passes
# below it), the upper edge the third harmonic of a deep 808 — conventions of the "kick" class, as drum_oneshots.CLASS_BANDS
BAND_HZ = (25.0, 200.0)
NFFT_PAD = 1 << 16
QUARTER_TONE = 0.5   # semitones


def _peak(seg: np.ndarray, sr: int, lo: float, hi: float) -> tuple[Optional[float], float]:
    """(frequency of the strongest spectral peak in [lo, hi] — parabolic-interpolated, prominence over the band median)."""
    if len(seg) < 64 or float(np.max(np.abs(seg))) < 1e-7:
        return None, 0.0
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), NFFT_PAD))
    fr = np.fft.rfftfreq(NFFT_PAD, 1.0 / sr)
    idx = np.where((fr >= lo) & (fr <= hi))[0]
    if len(idx) < 3:
        return None, 0.0
    k = idx[int(np.argmax(sp[idx]))]
    if k == idx[0] or k == idx[-1]:     # the strongest bin is the band's edge: the slope of something outside it, not a peak
        return None, 0.0
    if 0 < k < len(sp) - 1 and sp[k - 1] > 0 and sp[k + 1] > 0:
        a, b, c = np.log(sp[k - 1]), np.log(sp[k]), np.log(sp[k + 1])
        curv = a - 2 * b + c
        d = float(np.clip(0.5 * (a - c) / curv, -1.0, 1.0)) if curv < 0 else 0.0
        f = float(fr[k] + d * (fr[1] - fr[0]))
    else:
        f = float(fr[k])
    return f, float(sp[k] / (np.median(sp[idx]) + 1e-12))


def tail_span(n_samples: int) -> tuple[float, float]:
    """The part of a one-shot that is its tail (fractions of its length): after the pitch sweep, until its end."""
    return 1.0 / 3.0, 1.0


def oneshot_pitch(x: np.ndarray, sr: int) -> Optional[float]:
    """Tail pitch (Hz) of a one-shot file, read exactly as a stem hit's tail is."""
    a, b = tail_span(len(x))
    f, prom = _peak(x[int(a * len(x)): int(b * len(x))], sr, *BAND_HZ)
    return f if prom >= 3.0 else None


def hit_pitches(y: np.ndarray, sr: int, hits: list[dict], cls: str, shot_len_s: float) -> list[Optional[dict]]:
    """Per *cls* hit (time order): {'time', 'hz', 'prom', 'dis'} of its tail, or None where the tail is not readable.

    The tail window is the one-shot's own tail span laid on the hit; it ends early at the next hit of a class that
    sounds in this band (any non-hat hit), and a hit whose window is then under half the nominal one is unreadable."""
    a, b = tail_span(int(shot_len_s * sr))
    own = sorted(h["time"] for h in hits if h["type"] == cls)
    low = np.array(sorted(h["time"] for h in hits if h["type"] in ("bd", "sd")))
    out: list[Optional[dict]] = []
    for t in own:
        t0, t1 = t + a * shot_len_s, t + b * shot_len_s
        nxt = low[low > t + 1e-3]
        if nxt.size:
            t1 = min(t1, float(nxt[0]))
        if t1 - t0 < 0.5 * (b - a) * shot_len_s:
            out.append(None)
            continue
        seg = y[int(t0 * sr): int(t1 * sr)]
        f, prom = _peak(seg, sr, *BAND_HZ)
        h = len(seg) // 2
        f1, _ = _peak(seg[:h], sr, *BAND_HZ)
        f2, _ = _peak(seg[h:], sr, *BAND_HZ)
        if f is None or f1 is None or f2 is None:
            out.append(None)
            continue
        out.append({"time": t, "hz": f, "prom": prom, "dis": abs(12 * np.log2(f1 / f2))})
    return out


def hit_ratios(y: np.ndarray, sr: int, hits: list[dict], cls: str, shot: np.ndarray) -> dict[float, float]:
    """{hit time -> playback-rate ratio (stem tail pitch / one-shot tail pitch)} for the readable hits of *cls*.

    Readable = a clear tonal peak (prominence >= 3x the band median and not an outlier below the stem's own median prominence)
    whose two half-window reads agree to within median + 3 robust-sigma of the stem's own disagreement (a quarter-tone at least). Empty when the one-shot
    has no readable tail of its own (nothing to compare against)."""
    ref = oneshot_pitch(shot, sr)
    if ref is None:
        return {}
    rd = hit_pitches(y, sr, hits, cls, len(shot) / sr)
    ok = [r for r in rd if r is not None]
    if len(ok) < 4:
        return {}
    prom = np.array([r["prom"] for r in ok])
    dis = np.array([r["dis"] for r in ok])
    lp = np.log(prom)
    prom_min = max(3.0, float(np.exp(np.median(lp) - 3 * max(np.log(2.0), 1.4826 * np.median(np.abs(lp - np.median(lp)))))))
    # (prominence: a factor of 2 per sigma at least - a retuned kick is not less readable for being 2x weaker)
    # half-window reads of a steady tail agree to a fraction of a semitone; a quarter-tone is the floor so a stem whose
    # reads all agree exactly (MAD 0) does not discard the half of its hits above the median
    dis_max = max(QUARTER_TONE, float(np.median(dis) + 3 * 1.4826 * np.median(np.abs(dis - np.median(dis)))))
    st = {}
    for r in ok:
        if r["prom"] >= prom_min and r["dis"] <= dis_max:
            ratio = r["hz"] / ref
            if 0.5 <= ratio <= 2.0:      # beyond an octave either way is a mis-pick (a harmonic), not a retuned kick
                st[r["time"]] = 12 * np.log2(ratio)
    if not st:
        return {}
    # Hits that sit within the stem's own spread of its most common reading are ONE pitch read with noise: they are
    # written as that pitch (exactly 1 - the one-shot itself - when it is within the noise of the one-shot's).
    v = np.array(list(st.values()))
    med = float(np.median(v))
    noise = 3 * 1.4826 * float(np.median(np.abs(v - med)))
    centre = 0.0 if abs(med) <= noise else med
    return {t: float(2 ** ((centre if abs(s - med) <= noise else s) / 12)) for t, s in st.items()}
