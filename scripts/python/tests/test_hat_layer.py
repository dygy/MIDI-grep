"""The hat layer's level is real and its rhythm is the stem's (generate_dynamic_strudel.hat_layer, calibrate_dynamic).

Before 2026-11-15 the layer was `s("hh*8").gain(g)` inside the drum stack, whose own per-bar `.gain()` replaces an
inner one: every g > 0 rendered the same (BibiBabydoll "Melhor Sobrar": g 0.18 / 0.36 / 0.6 -> 0.7356 each) - an
on/off switch playing steady 8ths over a stem whose highs are claps on 2 and 4. On the stem's detected hh hits at 4x
the old energy: 0.804 / 0.764 -> 0.814 / 0.777 (both floors).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_dynamic_strudel as G  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / "calibrate_dynamic.py"
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate" / "tatuagem_synth_cal1.json"


def test_the_layer_plays_the_stems_hh_hits_at_the_velocity_that_carries_its_energy():
    grid = {"hh": {4, 12, 16 + 4}, "bd": {0, 4, 8, 12}}       # bar 0: 2 hits, bar 1: 1 hit, bar 2: none
    code = G.hat_layer(4.0, grid, 16, 3)
    bars = re.findall(r's\("([^"]*)"\)', code)
    assert [b.split().count("hh") for b in bars] == [2, 1, 0]
    assert bars[0].split()[4] == "hh" and bars[0].split()[12] == "hh"
    vel = float(re.search(r"velocity\(([\d.]+)\)", code).group(1))
    assert vel == pytest.approx((4.0 * 8 / 1.5) ** 0.5, rel=1e-2)   # 1.5 hits per active bar carry 4 x (8 hits at 1)
    assert ".gain(" not in code                                     # the stack's gain would replace it


def test_without_detected_hats_it_is_steady_8ths_and_level_still_scales():
    assert G.hat_layer(1.0, {"bd": {0}}, 16, 2) == 's("hh*8").velocity(1)'
    assert G.hat_layer(4.0, None, 16, 2) == 's("hh*8").velocity(2)'


def _cal(*extra: str) -> dict[str, str]:
    d = json.loads(FIX.read_text())
    b = d["rendered"]["bands"]
    b["high"], b["high_mid"] = 0.0004, 0.0008                      # a dark render: hats engage via the drums path
    d["rendered"]["spectral"]["centroid_mean"] = d["original"]["spectral"]["centroid_mean"] / 1.6
    p = FIX.parent / "_dark_tmp.json"
    p.write_text(json.dumps(d))
    try:
        out = subprocess.run([sys.executable, str(CAL), "--comparison", str(p), "--mode", "synth", "--lead-lpf", "7000",
                              "--cal-lead", "0.888", "--sub-gain", "0.474", "--bass-mult", "1.3", "--sub-up", "0.743",
                              "--master-gain", "0.78", *extra], capture_output=True, text=True, check=True).stdout.split()
    finally:
        p.unlink()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def test_hats_engage_at_one_unit_and_then_scale_energy_at_most_doubling():
    assert float(_cal("--hat-gain", "0")["--hat-gain"]) == pytest.approx(1.0)
    assert float(_cal("--hat-gain", "1")["--hat-gain"]) == pytest.approx(2.0)
    assert float(_cal("--hat-gain", "0.18")["--hat-gain"]) == pytest.approx(1.0)   # an old-scale value: at least 1 unit
