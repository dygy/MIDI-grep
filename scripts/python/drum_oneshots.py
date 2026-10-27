"""
drum_oneshots.py — cut drum one-shots from a drums stem so they HIT where the pattern says.

Why (measured, `context` fix `drum-oneshots-late-and-wrong-class`): the stem one-shots were 250 ms windows
that started at librosa's BACKTRACKED onset (the preceding energy minimum). The audible attack sat a median
26-43 ms (p90 57-62 ms) inside the file, so every hit of the pattern landed that much late and the rendered
drums' onset F1 (±50 ms) fell to 0.54-0.81 against 0.74-0.90 for the library 808 (attack at sample 0).
Their classes also came from a different classifier than the one that places the hits
(`detect_drums_bands.py`): VAGABUNDO's "sd" one-shot was 82 % below 150 Hz and only 2 % of its class
coincided with a band-detected snare.

So here every one-shot is taken from the SAME hits that drive the pattern (`detect_drums_bands.detect`),
and each cut

  1. starts at the hit's true attack (rising edge of the class's own band, zero-phase),
  2. ends at the first of: its decay to the stem's noise floor, the next retrigger of its own class
     (25th-percentile own-class interval: most retriggers do not land on a predecessor's tail), or the next hit of ANOTHER class (so no foreign hit is baked in),
  3. is chosen among the cleanest candidates (least energy outside the class's band), then the medoid of
     their spectral profile — a representative hit, not an outlier.

Every threshold below is read from this stem; nothing is per-track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as sps

# band (Hz) in which each class is detected AND judged for bleed — the detector's own bands
CLASS_BANDS: dict[str, tuple[float, float]] = {"bd": (20, 160), "sd": (180, 1200), "hh": (6000, 11000)}

# conventions, not tuning: -60 dB is the RT60 definition of "decayed"; 1 ms smooths a rectified envelope
DECAY_RATIO = 1e-3
ENV_SMOOTH_S = 0.001
ATTACK_SEARCH_PRE_S = 0.080     # how far before the detector time the rising edge is looked for
ATTACK_SEARCH_POST_S = 0.040
EDGE_FADE_IN_S = 0.0015
EDGE_FADE_OUT_S = 0.010


@dataclass
class Cut:
    audio: np.ndarray
    attack: int          # sample index in the stem
    truncated_by_other: bool   # ended by a foreign hit
    bleed: float         # fraction of the cut's energy outside the class band
    profile: np.ndarray  # log band energies, for the medoid
    coincident: bool = False   # a foreign hit is detected within the attack itself (baked into the cut)
    band_peak: float = 0.0  # where in the cut the class band peaks (0..1)
    decayed: bool = True    # ended by its own decay (not by running into the length cap)
    isolation: float = 0.0  # level under the hit / its peak: how much sustained material (a bass note) it sits on


def band_envelope(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band-passed, rectified, 1 ms smoothed envelope (zero phase: no filter group delay)."""
    nyq = sr / 2.0
    sos = sps.butter(2, [max(1e-4, lo / nyq), min(0.999, hi / nyq)], btype="band", output="sos")
    return smooth_envelope(sps.sosfiltfilt(sos, y), sr)


def smooth_envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Analytic-signal magnitude, 1 ms smoothed. (A rectified low-frequency band dips to zero every half cycle —
    5 ms at 100 Hz — and the attack search would stop at the last dip instead of at the rise.)"""
    env = np.abs(sps.hilbert(x))
    k = max(1, int(ENV_SMOOTH_S * sr))
    return np.convolve(env, np.ones(k) / k, mode="same")


def attack_index(env: np.ndarray, sr: int, t: float) -> Optional[int]:
    """Sample where the rising edge of the hit detected at time *t* begins.

    Peak of the envelope near *t*; walk back to the last sample below the geometric mean of the pre-hit floor
    and that peak — halfway in dB between silence-before and the hit, so it adapts to however much bleed
    sits under the hit. None when the neighbourhood is silent."""
    c = int(round(t * sr))
    a = max(0, c - int(ATTACK_SEARCH_PRE_S * sr))
    b = min(len(env), c + int(ATTACK_SEARCH_POST_S * sr))
    if b - a < 4:
        return None
    seg = env[a:b]
    pk = int(np.argmax(seg))
    peak = float(seg[pk])
    if peak <= 1e-7:
        return None
    floor = max(float(seg[: pk + 1].min()), peak * DECAY_RATIO)
    thr = float(np.sqrt(floor * peak))
    below = np.where(seg[: pk + 1] < thr)[0]
    return a + (int(below[-1]) + 1 if below.size else 0)


def rise_ratio(env: np.ndarray, attack: int, width: int) -> float:
    """How far *env* jumps at *attack*: the peak of the next *width* samples over the mean of the previous *width*."""
    pre = float(env[max(0, attack - width): attack].mean()) if attack > 0 else 0.0
    post = float(env[attack: attack + width].max()) if attack < len(env) else 0.0
    return post / (pre + 1e-12)


def decay_end(env: np.ndarray, start: int, limit: int, floor: float, background: float = 0.0) -> int:
    """First sample after the hit's peak where the envelope has fallen halfway in dB from the peak to the level
    it rose from (the same midpoint that defines the attack), never below the stem's noise floor.

    The -60 dB (RT60) tail is NOT used: on a drums stem it runs into the next hits and bass bleed, and
    measured (proxy onset F1 0.87 -> 0.57) smears the following onsets."""
    seg = env[start:limit]
    if seg.size < 2:
        return limit
    pk = int(np.argmax(seg))
    peak = float(seg[pk])
    bg = max(background, peak * DECAY_RATIO)
    thr = max(floor, float(np.sqrt(bg * peak)))
    under = np.where(seg[pk:] < thr)[0]
    return limit if under.size == 0 else start + pk + int(under[0])


def _profile(x: np.ndarray, sr: int, n_bands: int = 10) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    edges = np.geomspace(40.0, min(sr / 2.0, 16000.0), n_bands + 1)
    e = np.array([spec[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]) + 1e-12
    return np.log10(e / e.sum())


def _bleed(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Fraction of the cut's energy BELOW the class band: the ring of a lower class (a kick under a hat). Energy
    above the band is the class's own (a kick's body and click, a snare's wires), so it is not bleed."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(spec[f < lo].sum() / (spec.sum() + 1e-12))


def cut_class(
    y: np.ndarray, sr: int, hits: list[dict], cls: str, floor: float, info: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Return the representative one-shot for *cls* from detector *hits* ({'time', 'type'}), or None."""
    lo, hi = CLASS_BANDS[cls]
    env = band_envelope(y, sr, lo, hi)
    # decay is read on the class band AND EVERYTHING ABOVE it: the ring of a lower class beneath (a kick under a
    # hat) must not hold the hat open (Regime: 337 ms hat, high band 7 % over)
    sos_hp = sps.butter(2, max(1e-4, lo / (sr / 2.0)), btype="high", output="sos")
    wide = smooth_envelope(sps.sosfiltfilt(sos_hp, y), sr)

    own = sorted(h["time"] for h in hits if h["type"] == cls)
    if len(own) < 2:
        return None
    # the longest a hit of this class may ring: most retriggers of the class come later than this
    max_len = int(float(np.percentile(np.diff(own), 25)) * sr)

    other_t = np.array(sorted(h["time"] for h in hits if h["type"] != cls and h["type"] in CLASS_BANDS))
    # the window a band jump is read over = this class's own measured detection lag (detector time - true attack)
    lags = [t_ - a_ / sr for t_ in own for a_ in [attack_index(env, sr, t_)] if a_ is not None and t_ * sr > a_]
    width = max(2, int(float(np.median(lags)) * sr)) if lags else max(2, int(0.02 * sr))
    own_jump = []
    for t_ in own:
        a_ = attack_index(env, sr, t_)
        if a_ is not None:
            own_jump.append(rise_ratio(env, a_, width))
    visible = float(np.sqrt(max(1.0, float(np.median(own_jump)) if own_jump else 1.0)))

    cuts: list[Cut] = []
    for t in own:
        a = attack_index(env, sr, t)
        if a is None:
            continue
        limit = min(len(y), a + max_len)
        # the detector reports a hit `lag` after its true attack; a foreign hit detected (minus that lag) inside
        # the window truncates the cut, and one detected within a lag of ours is baked into the attack itself
        lag = max(0.0, t - a / sr)
        nxt_own = [u for u in own if u > t + 1e-3]
        if nxt_own:   # the class's own next hit ends this one too (and must not be mistaken for its peak)
            limit = min(limit, max(a + 1, int((nxt_own[0] - lag) * sr)))
        # A foreign hit ends the cut only if it shows up IN THIS CLASS'S BAND: a hat or snare over a ringing 808 kick
        # adds nothing below 160 Hz, and ending the kick there dropped its sub tail (VAGABUNDO: sub energy 6x
        # below the old one-shot, sub_bass band 13 % off). Visible = the band envelope jumps there by more than the
        # geometric mean between "no jump" (1) and this class's own typical jump at its own attacks.
        later = [u for u in other_t[(other_t - lag > a / sr) & (other_t - lag < limit / sr)]
                 if rise_ratio(env, int((u - lag) * sr), width) >= visible]
        later = np.array(later)
        trunc = later.size > 0
        if trunc:
            limit = int((later.min() - lag) * sr)
        background = float(wide[max(0, a - int(ATTACK_SEARCH_PRE_S * sr)): a + 1].min())
        end = decay_end(wide, a, limit, floor, background)
        # baked in = a foreign hit inside the attack that is visible in this class's band (a hat over a kick is not:
        # it only adds out-of-band energy, which the bleed ranking below already penalises)
        coincident = bool(other_t.size and any(
            abs(u - t) <= lag and rise_ratio(env, max(1, int((u - lag) * sr)), width) >= visible for u in other_t))
        x = y[a:end]
        if len(x) < int(0.01 * sr) or float(np.max(np.abs(x))) < 1e-5:
            continue
        cuts.append(Cut(x.copy(), a, trunc or coincident, _bleed(x, sr, lo, hi), _profile(x, sr), coincident,
                        float(np.argmax(env[a:end])) / max(1, end - a),
                        background / (float(wide[a:end].max()) + 1e-12)))
        cuts[-1].decayed = end < limit

    if info is not None:
        info.update(max_len_ms=max_len / sr * 1000, n_cuts=len(cuts), n_decayed=sum(c.decayed for c in cuts),
                    n_clean=sum(not c.coincident for c in cuts))
    if not cuts:
        return None
    # Tiers, best first. A one-shot has a natural end: a "hit" that is really a sustained note (VAGABUNDO's 453 ms
    # "kick", overall 0.796 -> 0.687) or a hat that never dips below the kick under it (Regime's 337 ms hat
    # ran to the length cap and pushed the high band 7 % over) reaches the cap instead, so a cut that decayed
    # beats one that did not; a foreign hit baked into the attack only matters between cuts that decayed.
    # (Regime: 211 of 223 hats coincide with a kick and the 12 that do not are stubs cut short by the next hit, so
    # a decayed-but-coincident hat — ranked by bleed below — is the better one-shot.)
    clean = ([c for c in cuts if c.decayed and not c.coincident]
             or [c for c in cuts if c.decayed]
             or [c for c in cuts if not c.coincident]
             or cuts)
    # a cut shorter than the class's typical ring was ended early by a dip in the bleed beneath it; it would also
    # win the bleed ranking below just for being short
    # the class's reference spectrum = the median over ALL its candidates of the tier, taken before the filters below
    # thin the pool (the last filters leave a handful of cuts; their own median is not the class's sound — VAGABUNDO's
    # pick had sub/bass 0.10 against the stem kicks' 1.29)
    ref = np.median(np.array([c.profile for c in clean]), axis=0)
    typical = float(np.median([len(c.audio) for c in clean]))
    clean = [c for c in clean if len(c.audio) >= typical] or clean
    # The class's sound: the third of the candidates closest to the class's reference spectrum, where the reference
    # is the median of the cleaner half (least energy BELOW the band = the ring of a lower class under the hit).
    # Energy above the band is the class's own (a kick's body, a snare's wires), so it is not bleed. (Ranking by
    # "energy outside the band" threw away VAGABUNDO's sub kicks - their body sits above 160 Hz - and kept punchy
    # cuts with sub/bass 0.04 against the stem kicks' 1.29: sub_bass band 13 % off, bass 18 %.)
    half = [c for c in clean if c.bleed <= float(np.median([x.bleed for x in clean]))] or clean
    ref = np.median(np.array([c.profile for c in half]), axis=0)
    dist = np.array([np.linalg.norm(c.profile - ref) for c in clean])
    pool = [clean[i] for i in np.argsort(dist)[: max(1, len(clean) // 3)]]
    q = float(np.median([c.bleed for c in pool]))
    pool = [c for c in pool if c.bleed <= q] or pool
    # ... then, among those, the one hit cleanest: attack at the file start, loudest point in its own attack
    # (a bass note swelling under the hit would otherwise make the "kick" a bass sample)
    pk_med = float(np.median([c.band_peak for c in pool]))
    pool = [c for c in pool if c.band_peak <= pk_med] or pool
    best = min(pool, key=lambda c: onset_in_file(c.audio, sr))
    return best.audio


def start_at_attack(x: np.ndarray, sr: int, passes: int = 2) -> np.ndarray:
    """Drop the lead-in before the rising edge. Removing it changes the floor the edge is measured against, so
    repeat until the onset stops moving (or *passes*)."""
    for _ in range(passes):
        i = int(round(onset_in_file(x, sr) * sr))
        if i <= 0 or i >= len(x) // 2:
            break
        x = x[i:]
    return x


def trim_to_decay(x: np.ndarray, sr: int, floor: float) -> np.ndarray:
    """Cut a one-shot that starts at its attack where it has decayed (same midpoint rule as `cut_class`), for the
    classes cut from a fixed window (oh): the window would otherwise carry whatever hit follows the open hat."""
    env = smooth_envelope(x, sr)
    background = float(env[: max(2, int(ENV_SMOOTH_S * sr * 4))].min())
    return x[: max(int(0.01 * sr), decay_end(env, 0, len(x), floor, background))]


def stem_noise_floor(y: np.ndarray, sr: int) -> float:
    """Level of the stem's quiet gaps: the 10th percentile of its 10 ms frame peaks."""
    n = int(0.01 * sr)
    m = len(y) // n
    if m < 2:
        return 0.0
    return float(np.percentile(np.abs(y[: m * n]).reshape(m, n).max(axis=1), 10))


def onset_in_file(x: np.ndarray, sr: int) -> float:
    """Seconds from file start to where its (wideband) envelope first reaches the geometric mean of its own
    floor and peak — 0 for a one-shot cut at its attack. (First crossing: a one-shot has no earlier hit.)"""
    env = smooth_envelope(x, sr)
    peak = float(env.max())
    if peak <= 0:
        return 0.0
    floor = max(float(env.min()), peak * DECAY_RATIO)
    return int(np.argmax(env >= np.sqrt(floor * peak))) / sr
