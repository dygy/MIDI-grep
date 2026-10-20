"""polyphony.estimate: peak simultaneous superdough sources of a generated .strudel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polyphony as P  # noqa: E402


def _code(bars, chain, cps=0.5):
    arr = ",\n".join(f'  "{b}"' for b in bars)
    return f"const cps = {cps}\nsetcps(cps)\nlet lead = [\n{arr}\n]\n$: note(cat(...lead)){chain}\n"


def test_chord_with_long_clip_overlaps_neighbours():
    # bar = 2 s, 4 steps -> 0.5 s each; a 3-note chord per step, held 4x its step by clip
    held = P.estimate(_code(["[c3,e3,g3] [c3,e3,g3] [c3,e3,g3] [c3,e3,g3]"], '.s("sawtooth").clip(4)'))
    short = P.estimate(_code(["[c3,e3,g3] [c3,e3,g3] [c3,e3,g3] [c3,e3,g3]"], '.s("sawtooth")'))
    assert held["peak"] == 12 and held["burst"] == 3
    assert short["peak"] < held["peak"]


def test_rests_and_elongation():
    r = P.estimate(_code(["c3@3 ~"], '.s("sine")'))
    assert r["events"] == 1 and r["peak"] == 1


def test_sample_without_clip_plays_whole_slice(tmp_path):
    sf.write(str(tmp_path / "n.wav"), np.zeros(int(22050 * 3.0), dtype=np.float32), 22050)
    (tmp_path / "samples.json").write_text('{"_base": "x/", "inst": [["c3"]]}'.replace('[["c3"]]', '{"c3": ["n.wav"]}'))
    code = _code(["c3 ~ ~ ~", "c3 ~ ~ ~"], '.s("inst")')
    long = P.estimate(code, tmp_path / "samples.json")      # 3 s slice, steps 2 s apart -> 2 alive together
    clipped = P.estimate(code.replace('.s("inst")', '.s("inst").clip(1)'), tmp_path / "samples.json")
    assert long["peak"] == 2 and clipped["peak"] == 1
