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
  3. is chosen among the candidates that decayed, nearest the median spectral profile of ALL of them — a
     representative hit, not an outlier and not one mode of a bimodal class.

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
OUTLIER_SIGMAS = 3.0            # convention for "outside the candidates' own spread" (robust sigmas above their median)


@dataclass
class Cut:
    audio: np.ndarray
    attack: int          # sample index in the stem
    truncated_by_other: bool   # ended by a foreign hit
    bleed: float         # fraction of the cut's energy outside the class's own median profile (set once all cuts exist)
    profile: np.ndarray  # log band energies, for the medoid
    coincident: bool = False   # a foreign hit is detected within the attack AND adds energy outside the class's profile
    foreign_near: bool = False  # a foreign hit is detected within the attack itself (not yet judged by its energy)
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


def decay_end(env: np.ndarray, start: int, limit: int, floor: float, background: float = 0.0,
              peak_limit: Optional[int] = None) -> int:
    """First sample after the hit's peak where the envelope has fallen halfway in dB from the peak to the level
    it rose from (the same midpoint that defines the attack), never below the stem's noise floor.

    The -60 dB (RT60) tail is NOT used: on a drums stem it runs into the next hits and bass bleed, and
    measured (proxy onset F1 0.87 -> 0.57) smears the following onsets.

    The hit's peak is looked for before *peak_limit* (the next onset of any class): a LATER, louder hit inside a long
    window is not this hit's peak, and measured against it the decay threshold sat so high/low that the one-shot ran
    on through the music that follows (Automotivo: a 14.3 s snare)."""
    seg = env[start:limit]
    if seg.size < 2:
        return limit
    pk = int(np.argmax(seg[: max(2, peak_limit - start)] if peak_limit is not None else seg))
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


def profile_bleed(shares: np.ndarray, typical: np.ndarray) -> float:
    """Fraction of a cut's energy that sits outside the class's own spectral profile: half the L1 distance between
    its band shares and the class's median shares (0 = the typical hit, 1 = disjoint). Judged against what the class
    itself carries, not against a fixed band edge: a sub-heavy 808 kick has 3 % of a click's energy below 160 Hz and
    "energy below the band's lower edge" ranked the click as the cleaner kick."""
    return float(0.5 * np.abs(shares - typical).sum())


def robust_sigma(x: np.ndarray) -> float:
    """Normal-consistent spread (1.4826 x MAD): ignores the minority mode that a standard deviation would absorb."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


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
        # the hit's own peak lies before the next onset of ANY class (a hat or kick later in the window may be louder)
        nxt_any = [u for u in np.concatenate([other_t, np.array(own)]) if u - lag > a / sr + 1e-3]
        peak_limit = int((min(nxt_any) - lag) * sr) if nxt_any else None
        end = decay_end(wide, a, limit, floor, background, peak_limit)
        # A hit that has not decayed by the time a later hit of ANY class starts ends there: the length cap above comes
        # from this class's own gaps, which for a rare class (3 snares in 130 s) is tens of seconds of other music.
        by_onset = False
        if end >= limit and nxt_any:
            stop = int((min(nxt_any) - lag) * sr)
            if a < stop < limit:
                end, by_onset = stop, True
        # a foreign hit detected within the attack is only a CANDIDATE for "baked in": on a kick-driven track most
        # "hh" hits are the kick's own high-frequency attack (91 % of them sit within 40 ms of a kick), so whether it
        # counts is decided below by the energy it adds outside the class's own profile
        foreign_near = bool(other_t.size and np.any(np.abs(other_t - t) <= lag))
        x = y[a:end]
        if len(x) < int(0.01 * sr) or float(np.max(np.abs(x))) < 1e-5:
            continue
        cuts.append(Cut(x.copy(), a, trunc, 0.0, _profile(x, sr), False,
                        float(np.argmax(env[a:end])) / max(1, end - a),
                        background / (float(wide[a:end].max()) + 1e-12)))
        cuts[-1].decayed = end < limit and not by_onset
        cuts[-1].foreign_near = foreign_near

    if not cuts:
        return None
    # The class's own profile = the median over ALL its decayed candidates (a cut that ran to the length cap is a
    # sustained note, not a hit of the class). Bleed is the energy a cut has outside it, and a foreign hit near the
    # attack makes the cut "coincident" only if it left more than the candidates' own robust spread of it there.
    # (Dumb Blond Bitches, 808 kicks: 214 of 422 cuts had an "hh" on their attack; "bleed = energy below 20 Hz"
    # preferred cuts with less body, and the tiers narrowed to 6 click-only cuts - a 63 ms click with 3 % of its
    # energy below 160 Hz against a median of 88 %.)
    base = [c for c in cuts if c.decayed] or cuts
    typical = np.median(np.array([10.0 ** c.profile for c in base]), axis=0)
    for c in cuts:
        c.bleed = profile_bleed(10.0 ** c.profile, typical)
    b = np.array([c.bleed for c in base])
    gate = float(np.median(b)) + OUTLIER_SIGMAS * robust_sigma(b)
    for c in cuts:
        c.coincident = c.foreign_near and c.bleed > gate
        c.truncated_by_other = c.truncated_by_other or c.coincident
    if info is not None:
        info.update(max_len_ms=max_len / sr * 1000, n_cuts=len(cuts), n_decayed=sum(c.decayed for c in cuts),
                    n_clean=sum(not c.coincident for c in cuts))
    # A one-shot has a natural end: a "hit" that is really a sustained note (VAGABUNDO's 453 ms "kick", overall
    # 0.796 -> 0.687) or a hat that never dips below the kick under it (Regime's 337 ms hat ran to the length cap and
    # pushed the high band 7 % over) reaches the cap instead, so a cut that decayed beats one that did not.
    # Rank ALL of those by distance to the median profile; no filter ahead of it may thin the pool to one mode.
    ref = np.median(np.array([c.profile for c in base]), axis=0)
    dist = np.array([np.linalg.norm(c.profile - ref) for c in base])
    pool = [base[i] for i in np.argsort(dist)[: max(1, len(base) // 3)]]
    # among the nearest third: not baked-in, then the one hit cleanest: attack at the file start, loudest point in its
    # own attack (a bass note swelling under the hit would otherwise make the "kick" a bass sample)
    pool = [c for c in pool if not c.coincident] or pool
    pk_med = float(np.median([c.band_peak for c in pool]))
    pool = [c for c in pool if c.band_peak <= pk_med] or pool
    best = min(pool, key=lambda c: onset_in_file(c.audio, sr))
    # sanity: the pick must be a typical hit - if its band shares sit further from the candidates' median than the
    # candidates' own spread allows, fall back to the medoid
    spread = float(np.median(dist)) + OUTLIER_SIGMAS * robust_sigma(dist)
    if float(np.linalg.norm(best.profile - ref)) > spread:
        best = base[int(np.argmin(dist))]
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
