# @layer: integration
# @spec: 005-hydra-visuals
# @regression
"""Hydra visuals (scripts/python/visuals): the chain whitelist, the controls read from the piece, and the headless check.

Regressions pinned here:
- the whitelist stripped the digits out of ``s0``/``o0`` and then rejected the names ``s``/``o``: the fallback chain
  itself failed validation;
- the checker installed its clock before ``repl.evaluate``, which points Strudel's time source back at the stopped
  scheduler: every control read one constant value and the motion test was meaningless (now: wired controls must move);
- ``--use-angle=swiftshader`` left macOS headless Chromium with no WebGL context (Hydra died on ``null.clear``).
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

from visuals import hydra_code as HC  # noqa: E402
from visuals.__main__ import hydra_block  # noqa: E402

CHECKER = REPO / "scripts" / "node" / "dist" / "check-hydra.js"

# the generator's shape (generate_dynamic_strudel.py): header, bar arrays, DJ-flow box, $: channels, drum rows
PIECE = """// MIDI-grep dynamic — test piece
const cps = 0.5   // 120 BPM, 4 beats/bar
setcps(cps)
let bass = [
  "c2 ~ ~ ~ g1 ~ ~ ~ c2 ~ ~ ~ g1 ~ ~ ~",
  "f1@8 c2@8",
  "c2 ~ c2 ~ ~ ~ ~ ~ g1 ~ ~ ~ ~ ~ ~ ~",
  "c3@16",
]
let lead = [
  "e4 ~ g4 ~ ~ ~ ~ ~ c5 ~ ~ ~ ~ ~ ~ ~",
  "~ ~ ~ ~ a4@4 ~ ~ ~ ~ f4@4",
  "g4@16",
  "b4 ~ d5 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",
]
// │ sections: intro[0-1]  drop[2-3]
$: note(cat(...bass)).s("sawtooth")
$: note(cat(...lead)).s("triangle")
$: stack(
    cat(s("bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~"), s("bd ~ ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ ~ ~"), s("bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~"), s("bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~")),
    cat(s("~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~"), s("~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~"), s("~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~"), s("~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~")),
).bank("RolandTR808")
"""


def test_fallback_chain_passes_and_the_whitelist_rejects_what_breaks_the_scene_or_the_sandbox():
    assert HC.validate_chain(HC.fallback_chain()) == []
    assert HC.validate_chain("src(s0).modulate(osc(() => bassWave()), () => 0.01 * kick()).blend(src(o0), 0.2).out()") == []
    assert any("kaleid" in e for e in HC.validate_chain("src(s0).modulate(osc(() => bassWave()), () => kick()).kaleid(4).out()"))
    assert any("one expression" in e for e in HC.validate_chain("src(s0).scale(() => kick()); fetch('x'); osc(bassWave).out()"))
    assert any("must use the kick" in e for e in HC.validate_chain("src(s0).modulate(osc(() => bassWave()), 0.01).out()"))
    assert any("must start with src(s0)" in e for e in HC.validate_chain("osc(() => bassWave()).scale(() => kick()).out()"))
    assert any("unknown names: window" in e for e in HC.validate_chain("src(s0).scale(() => kick() + window.x + bassWave()).out()"))


def test_controls_are_wired_to_the_pieces_own_arrays_and_drum_rows():
    js, wiring = HC.controls_js(PIECE)
    assert HC.drum_rows(PIECE).keys() == {"bd", "hh"}
    assert wiring["kick"] == "drum row bd" and wiring["hat"] == "drum row hh" and wiring["snare"] is None
    assert wiring["bassWave"] == "note(cat(...bass))" and wiring["leadHue"] == "note(cat(...lead))"
    assert wiring["chordDepth"] is None  # no chords array: the control is a constant 0, not an error
    assert "const visGrid = 1 / 16" in js and "const snare = visNone" in js and "const chordDepth = visNone" in js


def _scene_jpeg(path: Path, w: int = 320, h: int = 192) -> None:
    from PIL import Image
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([(x * 255 // w), (y * 255 // h), ((x // 32 + y // 32) % 2) * 200 + 30], -1).astype(np.uint8)
    Image.fromarray(rgb).save(path, quality=90)


def _check(tmp_path: Path, chain: str) -> dict:
    node = shutil.which("node")
    if not node or not CHECKER.exists():
        pytest.skip("node or scripts/node/dist/check-hydra.js missing (cd scripts/node && npm run build)")
    img = tmp_path / "visual.jpg"
    _scene_jpeg(img)
    controls, wiring = HC.controls_js(PIECE)
    plan = {"image_prompt": "test", "chain": chain, "chain_source": "test",
            "image": {"model": "test", "seed": 0, "steps": 0, "size": [320, 192]}}
    vis = tmp_path / "visuals.strudel"
    vis.write_text(PIECE + hydra_block(controls, "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode(), plan, wiring))
    subprocess.run([node, str(CHECKER), str(vis), "--image", str(img), "-o", str(tmp_path / "check"), "--window", "1",
                    "--wired", ",".join(k for k, v in wiring.items() if v)], capture_output=True, text=True, timeout=600)
    return json.loads((tmp_path / "check" / "check.json").read_text())


def test_checker_passes_the_fallback_chain_and_every_wired_control_follows_the_patterns(tmp_path):
    r = _check(tmp_path, HC.fallback_chain())
    assert r["verdict"] == "PASS", r["failures"]
    rng = r["control_ranges"]
    assert rng["kick"][1] > 0.5 and rng["hat"][1] > 0.5        # onsets reach the pulse
    # bars 0 and 2 play c2 and g1 over a lowest bass note f1: 2^(2/12) .. 2^(7/12)
    assert rng["bassWave"] == pytest.approx([2 ** (2 / 12), 2 ** (7 / 12)])
    assert rng["snare"] == [0, 0] and rng["chordDepth"] == [0, 0]
    assert r["page_errors"] == []
    assert sorted(p.name for p in (tmp_path / "check").glob("frame_*.png")) == ["frame_drop_2.png", "frame_intro_0.png"]


def test_checker_fails_a_chain_that_washes_the_scene_out(tmp_path):
    chain = "src(s0).blend(osc(() => 40 * bassWave(), 0.2, 1), () => 0.95 + 0.05 * kick()).out()"
    assert HC.validate_chain(chain) == []                       # the whitelist alone lets it through
    r = _check(tmp_path, chain)
    assert r["verdict"] == "FAIL" and any("not recognizable" in f for f in r["failures"])
