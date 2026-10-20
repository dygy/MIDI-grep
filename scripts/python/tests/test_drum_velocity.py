"""Per-hit drum velocity: detect_drums_bands measures each hit's strength; the generator writes it as a parallel
.velocity() pattern (velocity multiplies the bar .gain() envelope in superdough, so it does not replace it)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detect_drums_bands as D  # noqa: E402
import generate_hybrid_strudel as H  # noqa: E402

SR = D.SR
BPM = 120.0                      # bar = 2 s, 16th = 0.125 s


def _kick_stem(path: Path, amps: list[float], period_s: float = 0.5):
    y = np.zeros(int(SR * (period_s * len(amps) + 1.0)), dtype=np.float32)
    n = int(0.12 * SR)
    tt = np.arange(n) / SR
    for k, a in enumerate(amps):
        i = int((0.2 + k * period_s) * SR)
        y[i:i + n] += (a * np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 25)).astype(np.float32)
    sf.write(str(path), y, SR)


def test_detector_measures_per_hit_strength(tmp_path):
    amps = [0.8, 0.4, 0.2, 0.8, 0.1, 0.4]
    f = tmp_path / "drums.wav"
    _kick_stem(f, amps)
    res = D.detect(f, BPM)
    bd = [h for h in res["hits"] if h["type"] == "bd"]
    assert bd and all("strength" in h for h in bd), "every hit must carry its measured strength"
    st = np.array([h["strength"] for h in bd])
    assert len(st) >= 4
    # linear strengths: an amplitude ratio of 8 between the loudest and quietest detected hit survives
    assert st.max() / st.min() > 3.0, st


def _hits(strengths: dict[int, float], t="bd", bar=2.0):
    step = bar / 16
    return [{"time": s * step, "type": t, "strength": v} for s, v in strengths.items()]


def _run(tmp_path, hits, velocity, nbars=2):
    j = tmp_path / "d.json"
    j.write_text(json.dumps({"hits": hits}))
    return H.build_detected_drums(j, BPM, nbars, velocity=velocity)


def test_velocity_pattern_parallels_hits_and_orders_by_strength(tmp_path):
    strengths = {k: 1.0 + k / 15 for k in range(16)}          # 16 hits, strength rising across the bar
    voices = _run(tmp_path, _hits(strengths), "global")
    bar0 = re.search(r's\("([^"]*)"\)\.velocity\("([^"]*)"\)', voices[0])
    steps, vel = bar0.group(1).split(), bar0.group(2).split()
    assert len(steps) == len(vel) == 16
    assert [i for i, x in enumerate(steps) if x != "~"] == [i for i, x in enumerate(vel) if x != "~"]
    v = [float(x) for x in vel]
    assert v == sorted(v) and v[-1] > v[0], v                  # stronger hit -> larger written velocity
    assert abs(np.sqrt(np.mean(np.square(v))) - 1.0) < 0.02     # class mean power stays 1 (calibrated drum gain)


def test_no_strength_or_off_leaves_plain_pattern(tmp_path):
    plain = [{k: v for k, v in h.items() if k != "strength"} for h in _hits({0: 1.0, 8: 0.5})]
    assert ".velocity" not in "".join(_run(tmp_path, plain, "bar"))
    assert ".velocity" not in "".join(_run(tmp_path, _hits({0: 1.0, 8: 0.5}), None))


def test_bar_mode_normalises_by_bar_median_global_by_piece_median():
    quiet_bar = {0: 0.1, 4: 0.1, 8: 0.1, 12: 0.2}
    loud_bar = {16: 1.0, 20: 1.0, 24: 1.0, 28: 2.0}
    hits = _hits({**quiet_bar, **loud_bar})
    vb = H.hit_velocities(hits, BPM, 2, mode="bar")
    vg = H.hit_velocities(hits, BPM, 2, mode="global")
    assert abs(vb[("bd", 28)] - vb[("bd", 12)]) < 1e-9                 # same accent relative to its own bar
    assert vg[("bd", 28)] > 3 * vg[("bd", 12)]                         # global keeps the bar-to-bar level too


def test_extremes_are_winsorised_and_power_preserved():
    st = {k: 1.0 for k in range(15)}
    st[15] = 500.0                                                     # one bleed-like outlier
    v = H.hit_velocities(_hits(st), BPM, 1, mode="global")
    assert max(v.values()) / np.median(list(v.values())) < 2.0, v
    assert abs(np.sqrt(np.mean(np.square(list(v.values())))) - 1.0) < 1e-6
