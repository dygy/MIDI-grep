# Functional Specification: Hydra Visuals Wired to the Piece

- **Topic:** Give every generated piece a visual: a realistic scene that fits the track, animated by Hydra so that it
  moves with the piece's own voices.
- **Status:** In progress — built 2026-10-29; first reference track (VAGABUNDO v063) PASSES 2026-10-30; image
  determinism and the no-Ollama path not yet exercised
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
- [ ] The image is painted locally (default `stabilityai/sd-turbo`; `--image-model` selects another diffusers model)
      with a recorded seed; the same inputs paint the same image.
- [ ] Ollama unavailable → the run still completes with the CLAP-derived prompt and the fallback chain (recorded).

### 2.2 Visuals wired to the stems

- [x] Controls kick / snare / hat (drum rows), bassHit / bassWave (bass), leadHit / leadHue (lead), chordDepth (chords),
      energy (onsets per bar) are read live from the piece at Strudel's clock; a voice the piece lacks is a constant 0.
- [x] An LLM chain is used only if it passes the whitelist (src(s0) base, allowed Hydra functions, controls, Math; no
      strings or statements; must use kick and bassWave) AND the headless check; otherwise the fallback chain is used and
      the rejection is recorded in `visuals.json`. *(VAGABUNDO v063: the mistral-small chain passed the whitelist but
      failed the check — scene correlation 0.437 < 0.5 — and the fallback chain passed: 0.660, motion ratio 5.70.)*

### 2.3 Verified, and audio untouched

- [x] `check-hydra.js` runs the whole `visuals.strudel` in headless Chromium (pinned @strudel/web + hydra-synth) and
      PASSES only when: no page errors; mean luminance correlation with the scene >= 0.5 (recognizable); every wired
      control changed value; the quarter of frames with the strongest pulse moves >= 1.2x the weakest quarter.
- [x] `output.strudel` is never modified; the visuals go to `vNNN/visuals.strudel` (+ `visual.jpg`, `visuals.json`,
      `visuals_check/`), so no similarity score or gate changes.
- [x] The block runs in the live REPL (strudel.dygy.app) with the piece playing. *(2026-10-29, `/embed`, VAGABUNDO v063 +
      the block with a stand-in image: evaluated with no page errors, hydra-canvas up, s0 loaded; `getTime`, `noteToMidi`,
      `initHydra` are REPL globals.)*
