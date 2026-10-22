"""Drum one-shots must HIT at sample 0 and be the class the pattern plays (fix drum-oneshots-late-and-wrong-class).

Measured on the real tracks: the stem one-shots started at librosa's backtracked onset, so the audible
attack sat 12-50 ms inside the file and every hit of the pattern landed that late (rendered drums onset F1
0.54-0.81 vs 0.74-0.90 for the 808 bank, whose attack is at sample 0); VAGABUNDO's "sd" one-shot was 82 %
sub-150 Hz because its class came from a different classifier than the one that places the hits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_sample_pack as B  # noqa: E402
import drum_oneshots as O  # noqa: E402

SR = 44_100


def _kick(n=0.3):
    t = np.arange(int(n * SR)) / SR
    return np.sin(2 * np.pi * (45 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 12)


def _snare(n=0.2, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n * SR)) / SR
    x = rng.standard_normal(len(t))
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(4, [400 / (SR / 2), 2500 / (SR / 2)], "band", output="sos"), x)
    return 0.8 * x / np.abs(x).max() * np.exp(-t * 22)


def _hat(n=0.06, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n * SR)) / SR
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 7000 / (SR / 2), "high", output="sos"), rng.standard_normal(len(t)))
    return 0.6 * x / np.abs(x).max() * np.exp(-t * 60)


@pytest.fixture
def drum_stem(tmp_path):
    """8 bars of 4/4 at 120 BPM: a kick on every beat, a hat on every off-beat 8th; a sub-bass pad bleeds
    under everything (a Demucs drum stem is never silent between hits)."""
    beat = 0.5
    y = np.zeros(int(8 * 4 * beat * SR) + SR)
    rng = np.random.default_rng(0)
    for bar in range(8):
        for b in range(4):
            t0 = (bar * 4 + b) * beat
            y[int(t0 * SR): int(t0 * SR) + len(_kick())] += _kick()
            for off in (beat / 2,):
                h = _hat(seed=int(bar * 8 + b * 2 + off * 4))
                i = int((t0 + off) * SR)
                y[i: i + len(h)] += h
    y += 0.002 * rng.standard_normal(len(y))
    sf.write(tmp_path / "drums.wav", y.astype(np.float32), SR)
    return tmp_path


def test_every_one_shot_hits_at_sample_zero(drum_stem, tmp_path):
    out = tmp_path / "pack"
    m = B.build_drums(drum_stem, out)
    for cls in ("bd", "hh"):
        x, sr = sf.read(out / m[cls])
        lead_ms = O.onset_in_file(x, sr) * 1000
        # old code: 16 ms here, 12-50 ms on the real tracks (backtracked onset, 250 ms window); the 808 bank's is 0
        assert lead_ms < 10.0, f"{cls} attack sits {lead_ms:.1f} ms inside the file"


def test_one_shot_classes_match_the_pattern_classes(drum_stem, tmp_path):
    out = tmp_path / "pack"
    m = B.build_drums(drum_stem, out)

    def share(cls, lo, hi):
        x, sr = sf.read(out / m[cls])
        sp = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return sp[(f >= lo) & (f < hi)].sum() / sp.sum()

    assert share("bd", 0, 250) > 0.6       # kick: low
    assert share("hh", 0, 500) < 0.25      # hat: (almost) no kick/snare ring baked into it (the old cut: 0.99)
    assert share("hh", 5000, 22050) > 0.5  # hat: bright


def test_one_shot_ends_before_a_foreign_hit_is_baked_in(drum_stem, tmp_path):
    out = tmp_path / "pack"
    m = B.build_drums(drum_stem, out)
    x, sr = sf.read(out / m["hh"])
    # the hat rings ~60 ms; the old fixed 250 ms window carried the next kick/snare/hat inside the file
    assert len(x) / sr < 0.2


def _stem_with_hits(kick_on_bass, hats_with_kicks):
    """16 kicks. The first *kick_on_bass* are 1 s apart and each starts together with a loud sustained 50 Hz bass
    note (0.9 s, longer than the length cap) — what a Demucs drum stem carries under a bass-heavy kick, a "hit"
    that never decays to the level it rose from. The rest are clean and 0.5 s apart. The hat, if any, coincides
    with every kick."""
    times, t = [], 0.0
    for i in range(16):
        times.append(t)
        t += 1.0 if i < kick_on_bass else 0.5
    y = np.zeros(int((t + 1.5) * SR))
    hits = []
    for i, t0 in enumerate(times):
        a = int(t0 * SR)
        y[a: a + len(_kick())] += _kick()
        hits.append({"time": t0 + 0.012, "type": "bd"})
        if hats_with_kicks:
            h = _hat(seed=i)
            y[a: a + len(h)] += h
            hits.append({"time": t0 + 0.012, "type": "hh"})
        if i < kick_on_bass:
            n = int(0.9 * SR)
            y[a: a + n] += 1.5 * np.sin(2 * np.pi * 50 * np.arange(n) / SR)
    return y, sorted(hits, key=lambda h: h["time"])


def test_kick_on_a_sustained_bass_note_ends_at_its_decay_not_the_cap():
    y, hits = _stem_with_hits(kick_on_bass=10, hats_with_kicks=False)
    info: dict = {}
    x = O.cut_class(y, SR, hits, "bd", O.stem_noise_floor(y, SR), info)
    ms = len(x) / SR * 1000
    # 10 of 16 kicks sit on a bass note that never decays: their cuts run to the cap (VAGABUNDO's 453 ms "kick",
    # bass band 31 % off). The clean kicks decay in ~150 ms; the pick must be one of those.
    assert ms < 0.8 * info["max_len_ms"], f"{ms:.0f} ms is the length cap ({info['max_len_ms']:.0f} ms)"
    assert info["n_decayed"] < info["n_cuts"]          # the fixture does contain cap-bound cuts


def test_hat_coincident_with_every_kick_ends_at_its_decay_not_the_lone_clean_hat_s_cap():
    """Regime: 211 of 223 hats coincide with a kick; the one hat with no kick near it was a sustained shimmer that
    never decayed, so the old 'prefer clean' rule picked it and the one-shot ran to the 337 ms cap (high band
    7 % over, brightness similarity 0.82 -> 0.46)."""
    y, hits = _stem_with_hits(kick_on_bass=0, hats_with_kicks=True)
    t_lone = 8.2
    a = int(t_lone * SR)
    n = int(0.9 * SR)
    rng = np.random.default_rng(5)
    from scipy.signal import butter, sosfilt
    shimmer = sosfilt(butter(2, 7000 / (SR / 2), "high", output="sos"), rng.standard_normal(n))
    y[a: a + n] += 0.5 * shimmer / np.abs(shimmer).max()           # no kick within 0.2 s of it
    hits = sorted(hits + [{"time": t_lone + 0.012, "type": "hh"}], key=lambda h: h["time"])
    info: dict = {}
    x = O.cut_class(y, SR, hits, "hh", O.stem_noise_floor(y, SR), info)
    ms = len(x) / SR * 1000
    assert ms < 0.5 * info["max_len_ms"], f"{ms:.0f} ms is the length cap ({info['max_len_ms']:.0f} ms)"


def test_a_hat_over_a_ringing_kick_does_not_end_the_kick():
    """VAGABUNDO: the kick cut ended at the next hit of another class, dropping the 808's sub tail (sub energy 6x
    below the old one-shot; rendered sub_bass band 13 %, bass 18 % off). A hat adds nothing in the kick's band, so
    it must not end the kick; the kick ends at its own decay."""
    times = [i * 1.0 for i in range(12)]
    y = np.zeros(int(14 * SR))
    hits = []
    for i, t0 in enumerate(times):
        a = int(t0 * SR)
        k = _kick(0.6)
        y[a: a + len(k)] += k
        h = _hat(seed=i)
        y[a + int(0.1 * SR): a + int(0.1 * SR) + len(h)] += h          # a hat 100 ms into every kick's ring
        hits.append({"time": t0 + 0.012, "type": "bd"})
        hits.append({"time": t0 + 0.112, "type": "hh"})
    x = O.cut_class(y, SR, sorted(hits, key=lambda h: h["time"]), "bd", O.stem_noise_floor(y, SR))
    ms = len(x) / SR * 1000
    assert ms > 150, f"the kick was ended at {ms:.0f} ms, by the hat 100 ms in"
    sub = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    assert sub[(f < 160)].sum() / sub.sum() > 0.5
