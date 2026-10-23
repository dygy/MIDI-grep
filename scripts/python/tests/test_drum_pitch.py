# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""A tuned kick is written at the pitch each stem hit sounds (fix `drums-carry-pitch`).

Measured on VAGABUNDO: the track plays one 808 kick whose tail sits at midi ~30 for 90 of 139 hits and at ~34-35 for
33 more; the pattern wrote the same one-shot for all of them, so the chroma under those hits correlated -0.4 (0.985
under the rest). The generator now measures each hit's tail pitch on the drums stem, compares it with the one-shot's
tail read the same way, and writes the ratio as a parallel per-hit `.speed()` pattern.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

import drum_pitch as DP  # noqa: E402

SR = 22_050
BPM = 120.0                     # bar 2 s, 16th 0.125 s
F_BASE, F_UP = 45.0, 45.0 * 2 ** (5 / 12)     # the one-shot's tail pitch, and a kick tuned 5 semitones above it


def _kick(f_tail: float, n=0.3, amp=0.8):
    t = np.arange(int(n * SR)) / SR
    f = f_tail + 90 * np.exp(-t * 40)          # 808 sweep down onto the tail pitch
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (amp * np.sin(ph) * np.exp(-t * 6)).astype(np.float32)


def _stem(tails: list[float]):
    """One kick per half bar; hit k at step 8k (a hit sits 1 s apart), each on its own tail pitch."""
    y = np.zeros(int(SR * (len(tails) + 1.0)), dtype=np.float32)
    hits = []
    for k, f in enumerate(tails):
        t0 = 0.2 + k * 1.0
        x = _kick(f)
        y[int(t0 * SR): int(t0 * SR) + len(x)] += x
        hits.append({"time": t0 + 0.0, "type": "bd", "strength": 0.8})
    return y, hits


TAILS = [F_BASE, F_BASE, F_UP, F_BASE, F_UP, F_BASE, F_BASE, F_UP, F_BASE, F_BASE]


def test_hit_ratios_follow_the_stem_pitch_per_hit():
    y, hits = _stem(TAILS)
    r = DP.hit_ratios(y, SR, hits, "bd", _kick(F_BASE, n=0.25))
    st = [12 * np.log2(r[h["time"]]) for h in hits]
    for f, s in zip(TAILS, st):
        assert abs(s - 12 * np.log2(f / F_BASE)) < 0.5, (TAILS, st)


def test_untuned_stem_writes_no_ratio_that_differs_from_one():
    y, hits = _stem([F_BASE] * 8)
    r = DP.hit_ratios(y, SR, hits, "bd", _kick(F_BASE, n=0.25))
    assert all(v == 1.0 for v in r.values()), r


def test_noise_has_no_pitch_to_write():
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(SR * 9)).astype(np.float32)
    hits = [{"time": 0.2 + k, "type": "bd"} for k in range(8)]
    assert DP.hit_ratios(y, SR, hits, "bd", _kick(F_BASE, n=0.25)) == {}


# ── end to end through the generator: the written pattern, not just the measurement ──────────────────────────
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
_REQ = [STEMS / "bass.wav", STEMS / "melodic.wav", PACK / "bass.mid", PACK / "melodic.mid",
        PACK / "strudel.json", PACK / "instruments" / "samples.json"]


@pytest.mark.skipif(not all(p.exists() for p in _REQ), reason="Regime CLT sample_pack not present in .cache")
def test_generator_writes_the_per_hit_kick_pitch(tmp_path):
    stems, pack = tmp_path / "stems", tmp_path / "pack"
    (pack / "instruments" / "kit").mkdir(parents=True)
    stems.mkdir()
    for n in ("bass.wav", "melodic.wav"):
        os.symlink(STEMS / n, stems / n)
    for n in os.listdir(PACK):
        if n not in ("instruments", "drums_bands.json", "drums"):
            os.symlink(PACK / n, pack / n)
    for n in os.listdir(PACK / "instruments"):
        if n not in ("kit", "samples.json"):
            os.symlink(PACK / "instruments" / n, pack / "instruments" / n)
    sm = json.loads((PACK / "instruments" / "samples.json").read_text())
    sm["bd"] = ["kit/bd.wav"]
    (pack / "instruments" / "samples.json").write_text(json.dumps(sm))
    sf.write(str(pack / "instruments" / "kit" / "bd.wav"), _kick(F_BASE, n=0.25), SR)

    y, hits = _stem(TAILS)
    sf.write(str(stems / "drums.wav"), y, SR)
    step = 60.0 / BPM * 4 / 16
    for h in hits:
        h["time"] = round(h["time"] / step) * step      # on the grid, as the detector's hits are
    # a 16th = 0.125 s: hit k lands on step 8 + 8k; shift the audio hits' reading by the same grid offset (0.2 -> 0.25)
    y2 = np.zeros_like(y)
    for k, f in enumerate(TAILS):
        x = _kick(f)
        i = int(hits[k]["time"] * SR)
        y2[i: i + len(x)] += x
    sf.write(str(stems / "drums.wav"), y2, SR)
    (pack / "drums_bands.json").write_text(json.dumps({"hits": hits, "bpm": BPM}))

    out = tmp_path / "o.strudel"
    cmd = [sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"),
           "--stems-dir", str(stems), "--pack-dir", str(pack),
           "--bass-midi", str(PACK / "bass.mid"), "--lead-midi", str(PACK / "melodic.mid"),
           "--drums-json", str(pack / "drums_bands.json"),
           "--base-url", "https://example.invalid/x", "--samples-url", "https://example.invalid/x/instruments/samples.json",
           "--bass-sound", "regime_bass", "--lead-sound", "regime_lead", "--mode", "sample-instrument",
           "--bpm", str(BPM), "--key", "C# minor", "--genre", "brazilian_funk", "--num-bars", "6",
           "--drum-mode", "extracted", "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-1500:]
    code = out.read_text(encoding="utf-8")
    rates = [float(v) for m in re.finditer(r'\.speed\("([^"]*)"\)', code) for v in m.group(1).split() if v != "~"]
    assert rates, "the retuned kicks must be written with a per-hit .speed()"
    up = [r for r in rates if abs(r - 1.0) > 0.03]
    assert len(up) == TAILS.count(F_UP), (rates, TAILS)
    assert up == pytest.approx([F_UP / F_BASE] * len(up), abs=0.03), up
    assert "bd" in code and ".speed(" not in "".join(l for l in code.splitlines() if 's("hh' in l or 's("sd' in l)
