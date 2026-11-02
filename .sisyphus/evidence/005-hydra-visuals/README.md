# Spec 005 — Hydra visuals: evidence

Final run: 2026-11-02, branch `feat/hydra-visuals`. Every number below is from the files in this folder
(`matrix.json`, `logs/`). Machine: 24 GB Apple Silicon, `mistral-small` (Ollama) + `stabilityai/sd-turbo` (MPS).

## Matrix — 5 tracks x both modes, `logs/matrix-final.log`

`run_matrix.sh` runs `python -m visuals <track> --version vNNN` one run at a time; each line of `logs/matrix-summary.log`
is one run. "layout" is the gate (scene correlation at block-mean scale, >= 0.5), "motion" the pulse ratio (>= 1.2).

| Track | Version | Mode | Chain | Layout | Motion |
|---|---|---|---|---|---|
| VAGABUNDO NÃO NAMORA | v073 | synth | mistral-small | 0.927 | 8.10 |
| VAGABUNDO NÃO NAMORA | v072 | sample-instrument | fallback (LLM chain failed the check) | 0.878 | 4.94 |
| Regime CLT | v088 | synth | fallback (LLM chain failed the check) | 0.840 | 4.07 |
| Regime CLT | v087 | sample-instrument | mistral-small | 0.837 | 5.06 |
| Dumb Blond Bitches | v020 | synth | mistral-small | 0.670 | 5.53 |
| Dumb Blond Bitches | v019 | sample-instrument | mistral-small | 0.768 | 3.76 |
| Automotivo XM | v018 | synth | mistral-small | 0.679 | 14.02 |
| Automotivo XM | v017 | sample-instrument | mistral-small | 0.908 | 26.53 |
| Enter - Nigdy Więcej Wojny | v011 | synth | mistral-small | 0.892 | 87.48 |
| Enter - Nigdy Więcej Wojny | v010 | sample-instrument | mistral-small | 0.973 | 2.21 |

- Re-run 2026-11-02 after the speed fix (below). 10/10 PASS; 8 of them on the LLM's own chain, 2 fell back after the
  LLM chain failed the check. Every LLM chain now gives its osc speed 0: the motion comes from the hits and the bass.
- Motion now follows the music, not the clock: per-frame motion by section (`matrix.json` → `section_motion`, with
  Hydra's time) is flat across the piece, ~0 in the intros and in breaks without drums (Regime v087, verse at bar 37:
  0.2) and steady in the drops (Dumb Blond v019: 21.2 at 7 s, 25.3 at 107 s). Before, the same check run at the piece's
  time put the old VAGABUNDO v073 chain at 9.0 in the intro and 31-36 in the drops at bars 26-67, rising with time.
- Motion ratios are higher than before (2.2-87): between hits the photo now holds still instead of drifting.
- Every run kept the validator's repair round available; 5 of 10 used it (`visuals.json` → `llm.first_errors`).
- The previous matrix (before the speed fix) is in git history (`a6994f0`).

## What changed on the way (each found in a run, fixed with a regression test in `tests/test_hydra_visuals.py`)

| Found | Fix | Measured |
|---|---|---|
| The check compared frames with the UNZOOMED scene after the edge-margin zoom | `--scene-zoom` (reference = the scene as the chain shows it at rest) | fallback on Regime v086: 0.532 → 0.831 |
| `keep_alive: 30s` held mistral-small (~14 GB) beside sd-turbo; matrix runs killed for memory | `keep_alive: 0` | runs 250-1587 s → 67-164 s |
| The LLM wrote the key into the scene prompt ("C# minor suggests deep blues") | prompt rule + `scene_prompt` drops music sentences | 0 music terms in 10 prompts |
| The LLM returned the prompt's example chain verbatim (2 of 5 tracks), labelled "validated" | no example chain in the prompt; `copies_example` | 0 copies |
| Controls read bare (`bassHit()`), `true` for osc's speed, a SOURCE in a number slot (black frame), JSON errors | validator rules + one repair round with the errors | — |
| A speed on the photo slid the scene away (VAGABUNDO v073: 0.89 → 0.19 over the sections) | speed on the photo must be 0 | same chain at speed 0: 0.861 |
| Photo turned by π/4, π/2 (scene corr ~0) | turn bound derived from the margin: MAX_WARP / 0.707 = 0.071 rad | no large turns in 10 chains |
| `contrast(() => 0.3 * snare())` = flat grey: the LLM read factors as amounts around 0 | each colour step's unchanged value in the signatures | 1/10 → 9/10 LLM chains pass |
| Bare `PI` passes the whitelist but is not π in the REPL | `Math.` required | VAGABUNDO v073: −0.065 → 0.788 |
| Hard seam at x = 1 − shift: Hydra's scroll*/modulateScroll* end in `fract(_st)` | scroll functions removed | column jump 21-24 (3.5-4x median) at x 607-616 → none |
| Turn + modulate together past the margin wrapped sky into a corner | scene read through `scene` (src, mirrored edges) | sky in bottom rows: 1.8 % → 0 % (Enter v011), 13.2 % → 0 % (test) |
| Strudel's formatter split `1e-4` into `1` / `e-4` | generator writes `0.0001`; strudel-client fixed and deployed (`d404f49`) | prod bundles carry the new rule |
| The background raced in the live REPL: 9 of 10 chains drove osc's speed or frequency with kick/bassWave, and Hydra multiplies both by the seconds since start, so each hit jumped the waves by time x change | the whitelist takes only constant speeds (osc speed, osc frequency unless speed 0, noise offset, gradient/rotate speed), bounded to one pattern period per bar of the piece | the 10 old chains: all rejected; new chains: motion flat over the piece |
| The check ran Hydra from 0 s (~18 s over all windows), not at the piece's time: it could not see that | `check-hydra` sets Hydra's time to the seconds played at each frame; `check.json` → `sections[].hydra_time` | old VAGABUNDO v073 chain at the piece's time: drops 31-36 per frame vs 10-15 in the old check |

## Live REPL (`img/`)

- `enter_repl_code.jpg`, `enter_repl_a.jpg` … `enter_repl_f.jpg`: Enter v011 with its LLM chain playing in
  strudel.dygy.app `/embed` (editor hidden by CSS for a-f). No page errors; in all 6 frames 0 % sky-coloured pixels in
  the bottom 20 rows and no seam (largest column jump 1.6-4.1x the median, at scene details, not at a fixed x).
- `vagabundo_repl_*_before_edge_fix.jpg`: the earlier state — a stripe of the opposite edge along the bottom and right.
- `<track>_scene.jpg` / `<track>_frame_drop_*.jpg`: each track's painted scene and one check frame (first version).

## Other logs

- `logs/failmode-no-llm.log`, `logs/failmode-ollama-unreachable.log` (2026-11-02, Dumb Blond Bitches v019 copy):
  both complete on the fallback chain, check PASS 0.886.
- `logs/pytest_full.log`: full Python suite. `logs/determinism.log`: same seed → same scene image (re-run 2026-11-02 on the current code).
  `logs/scene-metric-calibration.log`: why the gate measures layout, not pixels. `logs/go.log`: Go build/vet/test.
