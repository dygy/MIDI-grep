# Functional Specification: Hydra Visuals Wired to the Piece

- **Topic:** Give every generated piece a visual: a realistic scene that fits the track, animated by Hydra so that it
  moves with the piece's own voices.
- **Status:** Completed — verified 2026-11-02 (`/awos:verify`): 10/10 matrix runs on 5 reference tracks PASS, image
  determinism and the no-Ollama path exercised on the current code
- **Author:** (with user) — prompt, 2026-10-29
- **Ticket:** prompt (no issue); roadmap Phase "Alternative output formats" → "Hydra: visual pattern suggestions based on audio"
- **Decisions (owner, 2026-10-29):** a self-hosted image model paints the scene and a self-hosted Ollama model writes
  the scene prompt and the Hydra chain; the visuals are driven by the Strudel patterns (not baked audio envelopes); a
  module in this repo (`scripts/python/visuals/`), not a git submodule.

---

## 1. Overview and Rationale (The "Why")

Strudel's REPL runs Hydra behind the code. A live coder who opens a MIDI-grep piece gets sound but an empty screen.
The owner wants a picture that **fits the track's vision**, **looks like something real** (a place, not an abstract
shader), and **waves with the stems** — the bass, drums and lead visibly moving it.

Because the controls are queried from the piece's bar arrays while it plays, the picture stays tied to the music the
coder is editing: change a bass bar and the wave changes with it.

## 2. Functional Requirements (The "What")

### 2.1 A scene that fits the track

- [x] The scene prompt is derived from measurements of the track (title, genre, key/mode, BPM, sections, CLAP scene and
      mood ranking of `instrumental.wav`, spectral centroid), not chosen per track. *(VAGABUNDO v063: CLAP top scenes
      snowy mountain 0.318 / nightclub 0.295 / rainforest 0.235, moods euphoric / menacing / dark; mistral-small chose a
      neon nightclub dance floor — `visuals.json`.)*
- [x] The image is painted locally (default `stabilityai/sd-turbo`; `--image-model` selects another diffusers model)
      with a recorded seed; the same inputs paint the same image. *(2026-11-02, current code: two pipeline runs and a
      repaint with seed 7 are byte-identical, sha256 c0af166d…; seed 8 differs — `logs/determinism.log` in the evidence
      folder; the seed and model are recorded in `visuals.json` → `image`.)*
- [x] Ollama unavailable → the run still completes with the CLAP-derived prompt and the fallback chain (recorded).
      *(2026-11-02, Dumb Blond Bitches v019 copy, Ollama unreachable: `chain_source` "fallback (Ollama failed:
      ConnectionError)", `image_prompt` built from the CLAP top scene + mood, check PASS 0.886; `--no-llm` gives the
      same — `logs/failmode-*.log`.)*

### 2.2 Visuals wired to the stems

- [x] Controls kick / snare / hat (drum rows), bassHit / bassWave (bass), leadHit / leadHue (lead), chordDepth (chords),
      energy (onsets per bar) are read live from the piece at Strudel's clock; a voice the piece lacks is a constant 0.
- [x] An LLM chain is used only if it passes the whitelist (src(s0) base, allowed Hydra functions, controls, Math; no
      strings or statements; must use kick and bassWave) AND the headless check; otherwise the fallback chain is used and
      the rejection is recorded in `visuals.json`. *(VAGABUNDO v063: the mistral-small chain passed the whitelist but
      failed the check — scene correlation 0.437 < 0.5 — and the fallback chain passed: 0.660, motion ratio 5.70; both at the
      pixel scale used before the 2026-10-30 amendment below.)*
      *(Amended 2026-11-02: the prompt shows no example chain — mistral-small returned the one it was given verbatim on 2
      of 5 tracks — only the shape, the piece's live controls and each colour step's unchanged value. The whitelist also
      rejects scroll*/modulateScroll* (Hydra wraps a scrolled frame: a seam at x = 1 - shift), controls outside an arrow
      function, bare PI/sin/... (not Math's in the REPL), a SOURCE where a number belongs, true/false, and a speed on the
      photo itself; a rejected chain goes back to the LLM once with the errors. Matrix of 10 runs, 5 tracks: 9 LLM chains
      passed the check, 1 fell back — `.sisyphus/evidence/005-hydra-visuals/README.md`.)*

### 2.3 Verified, and audio untouched

- [x] `check-hydra.js` runs the whole `visuals.strudel` in headless Chromium (pinned @strudel/web + hydra-synth) and
      PASSES only when: no page errors; mean luminance correlation with the scene at layout scale (block means ~1/40 of the
      width) >= 0.5 (recognizable — amended 2026-10-30 from pixel scale, which failed a plainly recognizable Regime CLT
      street at 0.479 because a one-pixel warp decorrelates fine texture; calibration in `.sisyphus/evidence/005-hydra-visuals`); every wired
      control changed value; the quarter of frames with the strongest pulse moves >= 1.2x the weakest quarter. The
      reference is the scene as the chain shows it at rest (zoomed by the edge margin; amended 2026-10-30 after the
      unzoomed reference failed every LLM chain on 4 tracks).
- [x] `output.strudel` is never modified; the visuals go to `vNNN/visuals.strudel` (+ `visual.jpg`, `visuals.json`,
      `visuals_check/`), so no similarity score or gate changes.
- [x] The block runs in the live REPL (strudel.dygy.app) with the piece playing. *(2026-10-29, `/embed`, VAGABUNDO v063 +
      the block with a stand-in image: evaluated with no page errors, hydra-canvas up, s0 loaded; `getTime`, `noteToMidi`,
      `initHydra` are REPL globals.)* *(2026-11-02, `/embed`, Enter v011 with its LLM chain: no page errors, no edge seam
      or wrapped sky in 6 frames — `img/enter_repl_*.jpg` in the evidence folder; the scene is read through `scene`, src
      with mirrored edges.)*
