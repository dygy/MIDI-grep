# Spec 005 — Hydra visuals: evidence

Final run: 2026-11-02, branch `feat/hydra-visuals`. Every number below is from the files in this folder
(`matrix.json`, `logs/`). Machine: 24 GB Apple Silicon, `mistral-small` (Ollama) + `stabilityai/sd-turbo` (MPS).

## Matrix — 5 tracks x both modes, `logs/matrix-final.log`

`run_matrix.sh` runs `python -m visuals <track> --version vNNN` one run at a time; each line of `logs/matrix-summary.log`
is one run. "layout" is the gate (scene correlation at block-mean scale, >= 0.5), "motion" the pulse ratio (>= 1.2).

| Track | Version | Mode | Chain | Layout | Motion |
|---|---|---|---|---|---|
| VAGABUNDO NÃO NAMORA | v073 | synth | mistral-small | 0.789 | 3.05 |
| VAGABUNDO NÃO NAMORA | v072 | sample-instrument | mistral-small | 0.757 | 4.81 |
| Regime CLT | v088 | synth | mistral-small | 0.691 | 3.71 |
| Regime CLT | v087 | sample-instrument | mistral-small | 0.689 | 3.73 |
| Dumb Blond Bitches | v020 | synth | mistral-small | 0.569 | 1.60 |
| Dumb Blond Bitches | v019 | sample-instrument | mistral-small | 0.568 | 1.55 |
| Automotivo XM | v018 | synth | mistral-small | 0.522 | 4.62 |
| Automotivo XM | v017 | sample-instrument | fallback (LLM chain 0.318) | 0.860 | 5.07 |
| Enter - Nigdy Więcej Wojny | v011 | synth | mistral-small | 0.734 | 4.18 |
| Enter - Nigdy Więcej Wojny | v010 | sample-instrument | mistral-small | 0.757 | 1.52 |

- 10/10 PASS; 9 of them on the LLM's own chain, none a copy of the fallback (`hydra_code.copies_example`).
- Thin margins: Dumb Blond Bitches (0.57) and Automotivo XM v018 (0.52) clear 0.5 by little.
- Every run needed the validator's repair round once (`visuals.json` → `llm.first_errors`).
- Memory: Ollama empty after every run (`keep_alive: 0`), runs 67-164 s, swap 10-13 GB, nothing killed.

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
