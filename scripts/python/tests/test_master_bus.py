# @layer: unit
# @regression
"""Loudness through the engine's master bus (limiter + compressor) instead of clipping.

Pins: the renderer can be pointed at another engine bundle and knows whether it has the bus; the bus settings are
MEASURED from the original and an unclipped probe render; the emitted Strudel lines are guarded so an engine without the
functions ignores them (behaviour checked by running the lines in node with and without the functions); the generated
code still passes the editability contract and the validator; a limited render must stay under its ceiling.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_master_bus.py -q
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_dynamic_strudel as G  # noqa: E402
import mix_dynamics as M  # noqa: E402
import renderer  # noqa: E402
from editability_check import check_editability  # noqa: E402

SR = 16000


def _bundle(tmp_path: Path, text: str) -> dict:
    f = tmp_path / "index.js"
    f.write_text(text)
    return {"MIDIGREP_STRUDEL_BUNDLE": str(f)}


def test_renderer_selects_the_engine_bundle_and_detects_the_master_bus(tmp_path):
    assert renderer.engine_bundle({}) == renderer.NPM_BUNDLE
    fork = _bundle(tmp_path, "function setMasterLimiter(){} function setMasterCompressor(){}")
    assert renderer.engine_bundle(fork) == (tmp_path / "index.js").resolve()
    assert renderer.supports_master_bus(fork)
    assert not renderer.supports_master_bus(_bundle(tmp_path, "function setMasterLimiter(){}"))     # half an API is none
    assert not renderer.supports_master_bus({**fork, "MIDIGREP_RENDERER": "blackhole"})            # the hosted site is unknowable


SPEC = {"ceiling": -0.64, "limiter_release": 0.0249,
        "compressor": {"threshold": 0.0, "knee": 0.0, "ratio": 1.0, "attack": 0.0249, "release": 0.1247, "makeup": 2.09}}


def test_master_bus_lines_are_guarded_and_do_nothing_on_an_engine_without_the_functions():
    lines = G.master_bus_lines(SPEC)
    prog = "\n".join(lines)
    assert "typeof setMasterLimiter === 'function'" in prog and "typeof setMasterCompressor === 'function'" in prog
    assert "ceiling: -0.64" in prog and "makeup: 2.09" in prog and "release: 0.0249" in prog
    node = shutil.which("node")
    if not node:
        pytest.skip("node missing")

    def run(defined: bool) -> list:
        defs = "const calls=[]; const setMasterLimiter=(o)=>calls.push(['lim',o]); const setMasterCompressor=(o)=>calls.push(['comp',o]);" if defined else "const calls=[];"
        src = f"{defs}\n{prog}\nconsole.log(JSON.stringify(calls))"
        return json.loads(subprocess.run([node, "-e", src], capture_output=True, text=True, check=True).stdout)

    assert run(False) == []                                                  # npm 1.3.0 / the live site: no ReferenceError, no effect
    calls = dict(run(True))
    assert calls["lim"] == {"ceiling": -0.64, "release": 0.0249}
    assert calls["comp"]["makeup"] == 2.09 and calls["comp"]["ratio"] == 1


def test_no_spec_emits_nothing_and_the_lines_pass_the_editability_contract_and_validator():
    assert G.master_bus_lines(None) == []
    code = "\n".join(["// generation_mode: synth", "let bass = [\"c2 e2\"]", *G.master_bus_lines(SPEC),
                      "$: note(cat(...bass)).s(\"sawtooth\").gain(\"<0.5>\")"]) + "\n"
    assert check_editability(code).passed
    from strudel_validation import validate_code
    _, err = validate_code(code, autocorrect=False)
    assert not err, err


def _music(seconds: float, spread_db: float, transient_db: float, seed: int = 1) -> np.ndarray:
    """A 110 Hz bed whose 3 s loudness swings by ``spread_db`` plus a click every 0.5 s ``transient_db`` above the bed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    swing = 10 ** ((spread_db / 2) * np.sign(np.sin(2 * np.pi * t / 12)) / 20)
    y = 0.1 * swing * np.sin(2 * np.pi * 110 * t)
    for k in range(int(seconds * 2)):
        i = int(k * 0.5 * SR)
        y[i:i + 40] += 0.1 * 10 ** (transient_db / 20) * rng.choice([-1, 1])
    return y


def test_derive_master_bus_ceiling_is_the_originals_true_peak_and_a_peak_only_excess_is_not_compressed():
    ref = _music(24, 12, 3)
    probe = _music(24, 12, 15) * 0.25                                        # same loudness spread, transients 12 dB hotter
    spec = M.derive_master_bus(ref, probe, SR, probe_offset_db=12.0412)
    assert spec["ceiling"] == pytest.approx(M.true_peak_db(ref, SR), abs=0.01)
    assert spec["compressor"]["ratio"] == 1.0                                # the limiter's job, not the compressor's
    assert spec["limiter_release"] == pytest.approx(spec["compressor"]["attack"])


def test_derive_master_bus_compresses_only_when_the_loudness_spread_is_wider_than_the_originals():
    ref = _music(24, 6, 3)
    probe = _music(24, 18, 3) * 0.25
    comp = M.derive_master_bus(ref, probe, SR, 12.0412)["compressor"]
    assert comp["ratio"] > 1.5
    assert -60 < comp["threshold"] < 0


def test_next_master_bus_closes_the_rms_error_with_makeup():
    ref = {"rms_db": -11.0, "crest_db": 10.0}
    spec = {"ceiling": -0.6, "compressor": {"threshold": 0.0, "ratio": 1.0, "makeup": 1.0}}
    out = M.next_master_bus(spec, ref, {"rms_db": -13.0, "crest_db": 10.0})
    assert out["compressor"]["makeup"] == pytest.approx(3.0)
    assert spec["compressor"]["makeup"] == 1.0                               # input untouched


def test_next_master_bus_can_raise_a_unity_ratio_for_a_peaky_render():
    ref = {"rms_db": -11.0, "crest_db": 10.0}
    spec = {"ceiling": -0.6, "compressor": {"threshold": -20.0, "ratio": 1.0, "makeup": 0.0}}
    peaky = M.next_master_bus(spec, ref, {"rms_db": -11.0, "crest_db": 14.0})["compressor"]["ratio"]
    assert peaky == pytest.approx(10 ** (4.0 / 40), abs=0.01) and peaky > 1.0
    assert M.next_master_bus(spec, ref, {"rms_db": -11.0, "crest_db": 8.0})["compressor"]["ratio"] == 1.0   # flatter: never below unity


def test_a_limited_render_must_stay_under_its_ceiling_and_unclipped():
    ref = {"rms_db": -11.0, "crest_db": 10.0}
    ok = {"rms_db": -11.2, "crest_db": 10.5, "peak_db": -0.64, "clip_frac": 0.0}
    assert M.level_verdict(ref, ok, ceiling_db=-0.64)["ok"]
    assert not M.level_verdict(ref, {**ok, "peak_db": 0.0, "clip_frac": 0.014}, ceiling_db=-0.64)["ok"]
    assert M.level_verdict(ref, {**ok, "peak_db": 0.0, "clip_frac": 0.014})["ok"]      # no ceiling given = today's rule


def test_limiter_release_follows_the_originals_loudness_rise():
    y = _music(24, 6, 3)
    attack, release = M.envelope_times(y, SR)
    assert 0.005 <= attack <= 0.5 and 0.005 <= release <= 1.0
