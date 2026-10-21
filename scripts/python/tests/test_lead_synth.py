# @layer: integration
# @regression
"""Synth-mode lead as FITTED synthesis parameters (lead_synth.py + generate_dynamic_strudel.py --lead-synth).

The palette lead is a fixed soundfont preset. A fitted lead is a plain oscillator whose controls (unison/detune,
ADSR, low-pass + filter envelope, gain) are measured on the melodic stem and searched on the real mix metric.
Behaviour pinned here:
  * the measurement picks the oscillator family and the envelope from the AUDIO (a synthesized square tone with a
    plucked envelope reads back as a square with a short decay; a saw as a saw) — no typed constants;
  * the search minimises a black-box objective inside its box;
  * the generator emits the fitted controls as plain Strudel on BOTH the lead and the harmony line, drops the
    palette preset + --lead-lpf there, scales the lead gain by the spec's gain, and still passes the editability check.
"""
from __future__ import annotations

import json
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

import lead_synth as LS  # noqa: E402
from editability_check import check_editability  # noqa: E402

SR = 22050


def _tone(kind: str, tau: float, notes=(220.0, 261.63, 329.63, 293.66) * 3, step=0.5) -> np.ndarray:
    out = []
    for f in notes:
        t = np.arange(int(step * SR)) / SR
        w = np.zeros_like(t)
        for k in range(1, 40):
            if kind == "square" and k % 2 == 0:
                continue
            w += np.sin(2 * np.pi * k * f * t) / k
        env = np.exp(-t / tau)
        out.append(w * env * (1 - np.exp(-t / 0.004)))
    return np.concatenate(out) * 0.2


@pytest.mark.parametrize("kind", ["square", "sawtooth"])
def test_measurement_reads_oscillator_family_and_decay_from_audio(tmp_path, kind):
    y = _tone("square" if kind == "square" else "saw", tau=0.08)
    p = tmp_path / "m.wav"
    sf.write(p, y, SR)
    m = LS.measure_lead(p, bpm=120.0)
    assert m["family"] == kind, m["family_dist"]
    # a 80 ms exponential decay inside a 500 ms note: the measured decay is short (well under the note), the
    # measured sustain is low — not the defaults of a held pad
    assert m["decay_s"] < 0.25 and m["sustain"] < 0.5, m


def test_vector_roundtrip_and_chain_is_plain_strudel():
    m = {"family": "sawtooth", "centroid_hz": 1500.0, "rolloff85_hz": 3000.0, "width_cents": 12.0, "step_s": 0.11,
         "attack_s": 0.01, "decay_s": 0.05, "sustain": 0.6, "lpenv_oct": 1.0, "lpdecay_s": 0.05}
    p0 = LS.initial_params(m)
    b = LS.bounds(m, p0)
    p1 = LS.from_vec(LS.to_vec(p0, b), b)
    assert all(p1[k] == pytest.approx(p0[k], rel=1e-6) for k in LS.PARAM_NAMES)
    chain = LS.lead_chain(LS.spec_from(m, p0))
    assert chain.startswith('.s("supersaw").unison(') and ".lpenv(" in chain and ".lpf(" in chain
    assert "gm_" not in chain


def test_nelder_mead_minimises_inside_the_unit_box():
    target = np.array([0.7, 0.2, 0.55])
    x, v, hist = LS.nelder_mead(lambda x: float(np.sum((x - target) ** 2)), np.array([0.5, 0.5, 0.5]),
                                max_evals=120, log=lambda *_: None)
    assert v < 1e-3 and hist[0] > v
    assert np.all((x >= 0) & (x <= 1))


STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
_need = [STEMS / "bass.wav", STEMS / "melodic.wav", STEMS / "drums.wav", PACK / "bass.mid", PACK / "melodic.mid",
         PACK / "drums_bands.json"]


@pytest.mark.skipif(not all(p.exists() for p in _need), reason="Regime CLT pack not in .cache")
def test_generator_emits_the_fitted_lead_on_lead_and_chords(tmp_path):
    spec = {"sound": "supersaw", "unison": 5, "lpf": 3333.0, "lpenv": 1.25, "lpdecay": 0.07, "attack": 0.012,
            "decay": 0.09, "sustain": 0.4, "detune": 0.21, "gain": 1.5}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    base = ["--stems-dir", str(STEMS), "--pack-dir", str(PACK), "--bass-midi", str(PACK / "bass.mid"),
            "--lead-midi", str(PACK / "melodic.mid"), "--drums-json", str(PACK / "drums_bands.json"),
            "--base-url", "https://example.invalid/x", "--mode", "synth", "--bpm", "135.017", "--key", "C# minor",
            "--genre", "brazilian_funk", "--num-bars", "16", "--lead-lpf", "9000", "--lead-hpf", "95"]
    outs = {}
    for tag, extra in (("palette", []), ("fitted", ["--lead-synth", str(sp)])):
        o = tmp_path / f"{tag}.strudel"
        r = subprocess.run([sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"), *base, *extra, "--out", str(o)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        outs[tag] = o.read_text()

    def line(txt, name):
        return next(l for l in txt.splitlines() if l.startswith(f"$: note(cat(...{name}))"))

    def gains(l):
        return [float(v) for v in re.findall(r"\d+\.\d+", l.split(".gain(", 1)[1])]

    for name in ("lead", "chords"):
        f, p = line(outs["fitted"], name), line(outs["palette"], name)
        assert '.s("supersaw").unison(5).detune(0.21)' in f
        assert ".lpf(3333)" in f and ".lpenv(1.25)" in f and ".sustain(0.4)" in f
        assert "gm_lead" not in f and ".lpf(9000)" not in f          # preset and --lead-lpf are replaced, not stacked
        assert "gm_" in p and ".lpf(9000)" in p                       # the palette path is unchanged
        gf, gp = gains(f), gains(p)
        assert gf[:6] == pytest.approx([round(g * 1.5, 3) for g in gp[:6]], abs=2e-3)   # spec gain scales the voice
    assert check_editability(outs["fitted"]).passed
