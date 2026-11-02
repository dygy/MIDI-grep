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
from visuals.__main__ import checker_args, hydra_block  # noqa: E402

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


def _scene_jpeg(path: Path, w: int = 320, h: int = 192, texture: bool = False, top_band: bool = False,
                sky: bool = False) -> None:
    from PIL import Image
    y, x = np.mgrid[0:h, 0:w]
    if sky:  # blue sky over brown ground: blue in the bottom rows of a frame = the top wrapped in
        rgb = np.where((y < h // 2)[..., None], [30, 40, 160], [150, 100, 50])
        Image.fromarray(rgb.astype(np.uint8)).save(path, quality=95)
        return
    if top_band:  # dark grey scene, pure red along the top edge only: red at the BOTTOM of a frame = wrapped coordinates
        # neutral grey elsewhere: hue()/saturate() in a chain leave grey grey, so only wrapped red can reach the bottom
        rgb = np.stack([np.where(y < h // 25, 255, 40), np.full_like(y, 40), np.full_like(y, 40)], -1)
        Image.fromarray(rgb.astype(np.uint8)).save(path, quality=95)
        return
    rgb = np.stack([(x * 255 // w), (y * 255 // h), ((x // 32 + y // 32) % 2) * 200 + 30], -1).astype(np.float64)
    if texture:  # one-pixel grain over the same layout (cobblestones, sparkles): a 1-px warp decorrelates it
        rgb = rgb * 0.5 + np.random.default_rng(0).integers(0, 2, (h, w, 1)) * 120
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path, quality=95)


def _check(tmp_path: Path, chain: str, piece: str = PIECE, texture: bool = False, top_band: bool = False, sky: bool = False,
           scene_zoom: float | None = None) -> dict:
    node = shutil.which("node")
    if not node or not CHECKER.exists():
        pytest.skip("node or scripts/node/dist/check-hydra.js missing (cd scripts/node && npm run build)")
    tmp_path.mkdir(parents=True, exist_ok=True)
    img = tmp_path / "visual.jpg"
    _scene_jpeg(img, texture=texture, top_band=top_band, sky=sky)
    controls, wiring = HC.controls_js(piece)
    plan = {"image_prompt": "test", "chain": chain, "chain_source": "test",
            "image": {"model": "test", "seed": 0, "steps": 0, "size": [320, 192]}}
    vis = tmp_path / "visuals.strudel"
    vis.write_text(piece + hydra_block(controls, "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode(), plan, wiring))
    args = checker_args(vis, img, tmp_path / "check", wiring)   # the arguments the pipeline passes
    if scene_zoom is not None:
        args[args.index("--scene-zoom") + 1] = str(scene_zoom)
    subprocess.run([node, str(CHECKER), *args, "--window", "1"], capture_output=True, text=True, timeout=600)
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


def test_checker_runs_a_sample_instrument_piece_without_loading_its_pack(tmp_path):
    # sample-instrument pieces open with `await samples("<pack url>")`; the checker's bundle parsed the double-quoted
    # URL as mini-notation (VAGABUNDO v070: "[mini] parse error ... '/' found") and the pack host must not matter
    piece = PIECE.replace("setcps(cps)\n", 'setcps(cps)\nawait samples("http://127.0.0.1:9/no-such-pack/samples.json")\n')
    r = _check(tmp_path, HC.fallback_chain(), piece)
    assert r["verdict"] == "PASS", r["failures"]


def test_a_sparse_voice_outside_every_probe_window_is_still_live(tmp_path):
    # Automotivo XM v013 (2026-10-30): 3 snare hits in 81 bars, none inside the 2-s section windows -> "snare never
    # changed"; liveness is now swept over the whole piece. Here: one snare at cycle 0.75, after the first 1-s window
    # (cycles 0-0.5) and more than the pulse's one-cycle look-back before the next (from cycle 2)
    snare = '    cat(s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~"), s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"), s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"), s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")),\n'
    piece = PIECE.replace(").bank(\"RolandTR808\")", snare + ").bank(\"RolandTR808\")")
    assert HC.controls_js(piece)[1]["snare"] == "drum row sd"
    r = _check(tmp_path, HC.fallback_chain(), piece)
    assert r["control_ranges"]["snare"] == [0, 0]               # the probe windows never see it
    assert r["control_ranges_piece"]["snare"][1] > 0.5          # the whole-piece sweep does
    assert r["verdict"] == "PASS", r["failures"]


def test_the_scene_never_wraps_its_opposite_edge_into_view(tmp_path, monkeypatch):
    # live REPL 2026-10-30: Hydra's src() samples fract(uv), so a warp past an edge showed a stripe of the image's other
    # edge; every chain now zooms the scene by 1 + 2*MAX_WARP at its source
    from PIL import Image

    def bottom_red(d: Path) -> float:
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=float) for p in (d / "check").glob("frame_*.png")]
        return max(float((f[-f.shape[0] // 20:, :, 0] - f[-f.shape[0] // 20:, :, 1]).mean()) for f in frames)

    assert HC.EDGE_ZOOM == pytest.approx(1 + 2 * HC.MAX_WARP)
    _check(tmp_path / "margin", HC.fallback_chain(), top_band=True)
    monkeypatch.setattr(HC, "with_edge_margin", lambda chain: chain)    # the old assembly, for contrast
    _check(tmp_path / "none", HC.fallback_chain(), top_band=True)
    # measured 2026-10-30, fallback chain: 7.6 without the margin, 0.0 with it
    assert bottom_red(tmp_path / "none") > 3        # without the margin the top band wraps in at the bottom
    assert bottom_red(tmp_path / "margin") < 1


def test_the_scene_reference_carries_the_edge_zoom(tmp_path):
    # 2026-10-30: with_edge_margin zoomed every chain's scene by EDGE_ZOOM, but the check compared the frames with the
    # unzoomed image, so every mistral-small chain on 4 tracks failed the scene gate (Regime CLT v086: 0.406; the
    # fallback chain itself 0.532, 0.831 against the zoomed reference)
    rest = "src(s0).out()"                                      # no warp: the frame IS the reference
    r = _check(tmp_path / "zoomed", rest)
    assert r["scene_zoom"] == HC.EDGE_ZOOM
    assert r["scene_corr"]["mean"] > 0.999                      # measured 0.99993
    old = _check(tmp_path / "unzoomed", rest, scene_zoom=1)     # the old reference, for contrast
    # measured 0.979: this smooth test scene barely changes at layout scale under a 10 % zoom; real scenes lost 0.3
    assert old["scene_corr"]["mean"] < 0.99


def test_scene_gate_measures_the_layout_not_the_pixel_texture(tmp_path):
    # Regime CLT v084 (2026-10-30): a plainly recognizable street scored 0.479 at pixel scale and failed; the gate now
    # correlates block means (~1/40 of the width); the pixel number is still reported
    r = _check(tmp_path, HC.fallback_chain(), texture=True)
    assert r["pixel_corr"]["mean"] < 0.5                        # the old pixel-scale gate would have failed this
    assert r["scene_corr"]["mean"] >= 0.5 and r["verdict"] == "PASS", r["failures"]
    assert r["scene_corr"]["scale_px"] == 8                     # 320 px wide / 40 blocks


def test_checker_fails_a_chain_that_washes_the_scene_out(tmp_path):
    chain = "src(s0).blend(osc(() => 40 * bassWave(), 0.2, 1), () => 0.95 + 0.05 * kick()).out()"
    assert HC.validate_chain(chain) == []                       # the whitelist alone lets it through
    r = _check(tmp_path, chain)
    assert r["verdict"] == "FAIL" and any("not recognizable" in f for f in r["failures"])


def test_the_scene_prompt_drops_what_the_image_model_cannot_paint():
    # mistral-small wrote the key into the scene prompt (Enter, Regime CLT, Automotivo XM; 2026-10-30)
    enter = ("A photorealistic scene of a vast desert road under a starry night sky. The key of C# minor suggests deep "
             "blues and purples, while the euphoric mood calls for vibrant stars.")
    assert HC.scene_prompt(enter) == "A photorealistic scene of a vast desert road under a starry night sky."
    one = "A vibrant nightclub dance floor with neon lights, real lighting, colors in G minor key."
    assert HC.scene_prompt(one) == "A vibrant nightclub dance floor with neon lights, real lighting"
    clean = "A rainy street at night. Neon signs reflect in the puddles."
    assert HC.scene_prompt(clean) == clean
    assert HC.scene_prompt("Play it at 135 BPM in A minor.") == ""     # nothing left: the fallback prompt stays


def test_a_chain_copied_from_the_prompts_example_is_not_counted_as_the_llms():
    # mistral-small returned the example verbatim on 2 of 5 tracks (2026-11-01) and was reported as "(validated)"
    assert HC.copies_example(HC.fallback_chain())
    assert HC.copies_example("  " + HC.fallback_chain().replace(".", "\n  .", 2))
    assert not HC.copies_example(HC.fallback_chain().replace("0.08 * leadHue()", "0.1 * leadHue()"))


def test_the_prompt_shows_no_chain_to_copy_and_names_the_pieces_live_controls():
    controls, wiring = HC.controls_js(PIECE)
    prompt = HC.build_prompt({"title": "test"}, wiring)
    assert "".join(HC.fallback_chain().split()) not in "".join(prompt.split())
    assert "- kick: live, from" in prompt and "- snare: always 0" in prompt


@pytest.mark.parametrize("chain, bare", [
    # VAGABUNDO v073 (2026-11-01), the first answer without an example chain to copy
    ("src(s0).modulate(osc(bassWave() * Math.PI / 4, 1), bassHit()).rotate(Math.sin(kick()) * PI / 20, energy())"
     ".hue(leadHue() * 0.15).out()", "bassHit, bassWave, energy, kick, leadHue"),
    ("src(s0).modulate(osc(() => 2 * bassWave(), 0.1, 0), bassHit()).out()", "bassHit"),           # a sibling's arrow
    ("src(s0).modulate(osc(() => 2 * bassWave(), 0.1, 0), () => 0.02 * kick()).hue(leadHue()).out()", "leadHue"),
    ("src(s0).modulate(osc(() => 2 * bassWave(), 0.1, 0), () => 0.02 * kick()).hue(() => Math.min(0.1, leadHue())).out()",
     None),                                                                                            # nested in an arrow
])
def test_a_control_read_outside_an_arrow_function_is_rejected(chain, bare):
    errs = [e for e in HC.validate_chain(chain) if "arrow function" in e]
    assert errs == ([f"controls outside an arrow function (read once, never move): {bare}; write () => ... "
                     f"{bare.split(',')[0]}() ..."] if bare else [])


def test_a_speed_on_the_photo_itself_is_rejected_but_not_inside_a_source():
    # VAGABUNDO v073 (2026-11-01): scrollY speed on the photo, scene corr 0.89 -> 0.19 over the sections
    drift = ("src(s0).modulate(osc(() => bassWave() * Math.PI / 2), () => kick() * 0.04 + 0.01)"
             ".rotate(() => snare() * 0.03, () => energy() * 0.02).out()")
    assert any("speed argument on the photo itself (rotate)" in e for e in HC.validate_chain(drift))
    assert not any("speed" in e for e in HC.validate_chain(drift.replace("() => energy() * 0.02", "0")))
    # the fallback's .rotate(Math.PI / 2) and the blend's scale sit inside SOURCEs: not the photo's steps
    assert [n for n, _ in HC._photo_steps(HC.fallback_chain())] == [
        "modulate", "scale", "modulateScale", "hue", "saturate", "brightness", "blend", "out"]
    assert HC.validate_chain(HC.fallback_chain()) == []


def test_a_source_where_a_number_belongs_and_boolean_arguments_are_rejected():
    # Dumb Blond Bitches v020 / Enter v010 (2026-11-01): .scrollX(osc(...), 0) compiled to a black frame;
    # 4 of 10 answers wrote osc(f, true) for a "sync" they read as boolean
    black = "src(s0).modulate(osc(() => 2 * bassWave(), 0.1, 0), () => 0.02 * kick()).scale(osc(() => bassWave() * 0.5)).out()"
    assert any("a SOURCE where a number is expected (scale)" in e for e in HC.validate_chain(black))
    boolean = "src(s0).modulate(osc(() => 2 * bassWave(), false), () => 0.02 * kick()).out()"
    assert any("true/false" in e for e in HC.validate_chain(boolean))
    assert "osc(frequency, speed, offset)" in HC.SIGNATURES


def test_the_photo_turn_bound_follows_from_the_edge_margin():
    # a turn by MAX_PHOTO_TURN moves a corner (0.707 from the centre in uv) by MAX_WARP: the margin EDGE_ZOOM covers
    assert HC.MAX_PHOTO_TURN * np.hypot(0.5, 0.5) == pytest.approx(HC.MAX_WARP, abs=1e-3)
    assert f"at most {HC.MAX_PHOTO_TURN} rad" in HC.build_prompt({"title": "test"}, HC.controls_js(PIECE)[1])


def test_the_generated_block_has_no_exponent_literals():
    # the strudel.dygy.app formatter split `getTime() + 1e-4` into `1` / `e-4` (a syntax error; fixed in strudel-client
    # d404f49, 2026-11-02): the block writes plain decimals so it survives a formatter that is not deployed yet
    import re
    controls, _ = HC.controls_js(PIECE)
    assert not re.search(r"(?<![\w$.])\d[\d.]*[eE][+-]?\d", controls + HC.fallback_chain())


def test_scrolling_the_photo_is_rejected():
    # Enter v011 (2026-11-02): Hydra's scroll*/modulateScroll* end in `return fract(_st)`, so .modulateScrollX(osc, 0.05)
    # put a hard seam at x = 1 - shift (column jump 21-24 vs a median of 6, at x 607-616 of 640) whatever the margin
    chain = ("src(s0).modulate(osc(() => bassWave() * 2, () => kick()), 0.03)"
             ".modulateScrollX(osc(() => bassWave(), 0), () => Math.sin(bassWave()) * 0.05, 0).out()")
    errs = HC.validate_chain(chain)
    assert any(e.startswith("modulateScrollX: Hydra wraps a scrolled frame around") for e in errs)
    assert not any("unknown names" in e for e in errs)                    # one clear message, not two
    assert not ({"scroll", "scrollX", "scrollY", "modulateScrollX", "modulateScrollY"} & HC.HYDRA_FUNCS)
    assert "croll" not in HC.SIGNATURES


def test_the_prompt_gives_each_colour_steps_unchanged_value():
    # 2026-11-02: contrast/saturate/brightness read as amounts around 0 turned the photo flat grey (scene corr 0.000)
    prompt = HC.build_prompt({"title": "test"}, HC.controls_js(PIECE)[1])
    for step in (".saturate(factor: 1 = unchanged", ".contrast(factor: 1 = unchanged", ".brightness(offset: 0 = unchanged"):
        assert step in prompt


def test_math_names_must_come_from_math():
    # VAGABUNDO v073 (2026-11-02): .rotate(() => 0.071 * Math.sin(bassWave() * PI), 0) passed the whitelist, then the
    # check (scene corr -0.065, no page error); with Math.PI it scores 0.788
    bare = "src(s0).modulate(osc(() => 2 * bassWave(), 0.1, 0), () => 0.02 * kick()).rotate(() => 0.071 * Math.sin(bassWave() * PI), 0).out()"
    assert any(e.startswith("write Math.PI, not a bare PI") for e in HC.validate_chain(bare))
    assert HC.validate_chain(bare.replace("* PI)", "* Math.PI)")) == []


def test_warps_that_add_up_past_the_margin_show_the_same_edge_not_the_opposite_one(tmp_path, monkeypatch):
    # Enter v011 (2026-11-02): a 0.071 rad turn (corners 0.05) plus a 0.05 modulate put sky in the bottom-right corner
    # in the live REPL; the scene is read with mirrored edges
    from PIL import Image

    def bottom_sky(d: Path) -> float:
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=float)[-8:] for p in (d / "check").glob("frame_*.png")]
        return max(float(((f[..., 2] - f[..., 0]) > 15).mean()) for f in frames)

    chain = "src(s0).rotate(0.071, 0).modulate(osc(8, 0, 0), 0.05).modulate(osc(5, 0, 0).rotate(1.2), 0.05).out()"
    assert "scene(s0)" in HC.with_edge_margin(chain) and HC.SCENE_SOURCE.startswith("setFunction({ name: 'scene'")
    _check(tmp_path / "mirror", chain, sky=True)
    monkeypatch.setattr(HC, "with_edge_margin", lambda c: c.replace("src(s0)", f"src(s0).scale({HC.EDGE_ZOOM})", 1))
    _check(tmp_path / "wrap", chain, sky=True)                       # the margin alone, for contrast
    print("bottom sky share, wrap vs mirror:", bottom_sky(tmp_path / "wrap"), bottom_sky(tmp_path / "mirror"))
    assert bottom_sky(tmp_path / "wrap") > 0.01
    assert bottom_sky(tmp_path / "mirror") == 0
