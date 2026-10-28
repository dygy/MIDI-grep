"""Two drum defects found on a sub-heavy 808 track (Dumb Blond Bitches), both general.

1. Kick one-shot: "bleed" was the energy below the class band's lower edge (20 Hz for bd), so a 63 ms click with 3 %
   of its energy below 160 Hz out-ranked the sub-heavy body of the median kick (88 %), and an "hh" detected on a
   kick's own attack (91 % of them sit within 40 ms of a kick) marked 214 of 422 kicks as coincident, which narrowed
   the tiers to six click-only cuts.
2. Grid phase: `round(t / step)` assumes the beat grid starts at t = 0; hits sitting half a step late fell on the
   neighbouring slot for about half of them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import drum_oneshots as O  # noqa: E402
import generate_hybrid_strudel as H  # noqa: E402

SR = 44_100


def _sub_kick(n=0.25):
    t = np.arange(int(n * SR)) / SR
    return np.sin(2 * np.pi * (45 + 60 * np.exp(-t * 25)) * t) * np.exp(-t * 14)


def _click(n=0.06):
    rng = np.random.default_rng(7)
    t = np.arange(int(n * SR)) / SR
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 1500 / (SR / 2), "high", output="sos"), rng.standard_normal(len(t)))
    return 0.7 * x / np.abs(x).max() * np.exp(-t * 70)


def _low_share(x):
    sp = np.abs(np.fft.rfft(x)) ** 2
    return float(sp[np.fft.rfftfreq(len(x), 1 / SR) < 160].sum() / sp.sum())


def _kick_stem(n_sub, n_click):
    """*n_sub* sub-heavy 808 kicks, then *n_click* click-only kicks; each carries an "hh" detected on its attack (the
    kick's own high-frequency edge, as the detector reports it)."""
    t, hits, y = 0.0, [], np.zeros(int((n_sub + n_click) * 0.6 * SR) + SR)
    for i in range(n_sub + n_click):
        x = _sub_kick() if i < n_sub else _click()
        a = int(t * SR)
        y[a: a + len(x)] += x
        hits.append({"time": t + 0.008, "type": "bd"})
        hits.append({"time": t + 0.008, "type": "hh"})
        t += 0.6
    return y, hits


def test_kick_picker_follows_the_sub_heavy_majority_not_the_click_minority():
    y, hits = _kick_stem(n_sub=18, n_click=6)
    x = O.cut_class(y, SR, hits, "bd", O.stem_noise_floor(y, SR), {})
    # old code: the click (0.03 of its energy below 160 Hz) - "less energy below 20 Hz" ranked it the cleaner kick
    assert _low_share(x) > 0.6, f"picked a {len(x) / SR * 1000:.0f} ms cut with {_low_share(x):.2f} of its energy < 160 Hz"


def test_hh_on_a_kick_s_own_attack_does_not_exclude_the_kick():
    y, hits = _kick_stem(n_sub=20, n_click=0)
    info: dict = {}
    O.cut_class(y, SR, hits, "bd", O.stem_noise_floor(y, SR), info)
    # every kick has an "hh" on its attack but adds nothing outside the kick's own profile: they are not "baked in"
    # (old code: n_clean == 0 - all 20 coincident; only the one cut that really differs, a stub, may be flagged)
    assert info["n_clean"] >= info["n_cuts"] - 2, info


# ---- grid phase --------------------------------------------------------------------------------------------------

BPM = 120.0
STEP = 60.0 / BPM / 4


def _drums_json(tmp_path, offset_steps, cls="bd", n=16):
    hits = [{"time": (s * 2 + offset_steps) * STEP, "type": cls} for s in range(n)]    # every other 16th
    j = tmp_path / "d.json"
    j.write_text(json.dumps({"hits": hits}))
    return j, hits


def _bar0_slots(voice):
    return re.search(r's\("([^"]*)"\)', voice).group(1).split()


def test_hits_half_a_step_late_land_on_their_own_slots(tmp_path):
    # hits at 0.45 step past the grid, jittered either side of the half-step: round(t / step) flips them between
    # the neighbouring slots (old code: 8 hits on slots 0,2,4,6,8,... half of them on 1,3,5,... instead)
    j, hits = _drums_json(tmp_path, 0.45)
    for k, h in enumerate(hits):
        h["time"] += (0.08 if k % 2 else -0.08) * STEP
    j.write_text(json.dumps({"hits": hits}))
    voices = H.build_detected_drums(j, BPM, 2)
    slots = _bar0_slots(voices[0])
    on = [i for i, s in enumerate(slots) if s == "bd"]
    assert on == [0, 2, 4, 6, 8, 10, 12, 14], on


def test_phase_is_emitted_as_late_in_cycles_and_is_the_hits_offset(tmp_path):
    j, _ = _drums_json(tmp_path, 0.45)
    late = H.drums_late(j, BPM)
    assert abs(late - 0.45 * STEP / (STEP * 16)) < 1e-6          # seconds / bar duration
    assert H.late_suffix(late) == f".late({late:.6f})"
    j2, _ = _drums_json(tmp_path, -0.30)
    # a grid early of the lines is a LATE of the next line (a negative shift would send step 0 before the piece)
    assert abs(H.drums_late(j2, BPM) - (STEP - 0.30 * STEP) / (STEP * 16)) < 1e-6


def test_a_track_on_the_grid_gets_no_late(tmp_path):
    j, _ = _drums_json(tmp_path, 0.0)
    assert H.drums_late(j, BPM) == 0.0
    assert H.late_suffix(0.0) == ""


def test_velocity_and_speed_patterns_still_index_the_same_hits(tmp_path):
    j, hits = _drums_json(tmp_path, 0.45)
    for k, h in enumerate(hits):
        h["time"] += (0.08 if k % 2 else -0.08) * STEP
        h["strength"] = 1.0 + k / 10
    j.write_text(json.dumps({"hits": hits}))
    pitch = {"bd": {h["time"]: 1.0 + k / 20 for k, h in enumerate(hits)}}        # keyed by the ORIGINAL times
    voices = H.build_detected_drums(j, BPM, 2, velocity="global", pitch=pitch)
    line = voices[0]
    slots = _bar0_slots(line)
    vel = re.search(r'\.velocity\("([^"]*)"\)', line).group(1).split()
    spd = re.search(r'\.speed\("([^"]*)"\)', line).group(1).split()
    assert [i for i, s in enumerate(slots) if s != "~"] == [i for i, v in enumerate(vel) if v != "~"]
    assert [i for i, s in enumerate(slots) if s != "~"] == [i for i, v in enumerate(spd) if v != "~"]
    # the 8 hits of bar 0 keep their own (rising) pitch order on the slots they were moved to
    r = [float(v) for v in spd if v != "~"]
    assert r == sorted(r) and len(set(r)) == len(r)


def test_dynamic_generator_places_late_before_the_gain_envelope():
    src = (Path(__file__).resolve().parent.parent / "generate_dynamic_strudel.py").read_text()
    i = src.index("H.late_suffix(detected.late)")
    j = src.index("gain_pattern(drum_env", i)
    assert i < j       # `.late()` closes the stack; the per-bar gain is applied after it, so the envelope is not shifted


# ---- a rare class with long gaps --------------------------------------------------------------------------------

def _rare_snare_stem():
    """3 snares 20 s apart; 0.2 s after each, 12 s of LOUDER sustained high-passed noise (the music that follows a rare
    hit: it never dips) with a hat onset every 0.25 s. Silence precedes each snare."""
    rng = np.random.default_rng(3)
    from scipy.signal import butter, sosfilt
    y = np.zeros(int(62 * SR))
    hits = []
    sn = sosfilt(butter(4, [400 / (SR / 2), 2500 / (SR / 2)], "band", output="sos"), rng.standard_normal(int(0.2 * SR)))
    sn = 0.3 * sn / np.abs(sn).max() * np.exp(-np.arange(len(sn)) / SR * 25)
    for k in range(3):
        t0 = 2.0 + 20.0 * k
        a = int(t0 * SR)
        y[a: a + len(sn)] += sn
        hits.append({"time": t0 + 0.008, "type": "sd"})
        bed = sosfilt(butter(2, 7000 / (SR / 2), "high", output="sos"), rng.standard_normal(12 * SR))
        i = int((t0 + 0.2) * SR)
        y[i: i + len(bed)] += 0.8 * bed / np.abs(bed).max()
        hits += [{"time": t0 + 0.208 + 0.25 * j, "type": "hh"} for j in range(48)]
    return y + 0.001 * rng.standard_normal(len(y)), sorted(hits, key=lambda h: h["time"])


def test_a_rare_class_one_shot_ends_where_its_own_hit_decays():
    y, hits = _rare_snare_stem()
    x = O.cut_class(y, SR, hits, "sd", O.stem_noise_floor(y, SR), {})
    ms = len(x) / SR * 1000
    # old code: the peak was the loudest point of the 12 s that follow, the cap 20 s (25th percentile of the snare
    # gaps): a 12 s "snare" (Automotivo: 14.3 s). The snare itself is gone in ~200 ms and the next hat starts at 200 ms.
    assert ms <= 250, f"{ms:.0f} ms"
