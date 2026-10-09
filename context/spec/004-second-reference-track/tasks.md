# Tasks: Second Reference Track for the Honest Eval Dataset

- **Functional Spec:** [functional-spec.md](./functional-spec.md)
- **Technical Spec:** [technical-considerations.md](./technical-considerations.md)
- `SKIP_TESTS = false`

> Track: https://youtu.be/SKjOR5EOR8Y — "VAGABUNDO NÃO NAMORA" (Christopher Luz). Slug `vagabundo-nao-namora`.
> Honesty rule: a task is `[x]` only with evidence appended (command output, path, number). Every
> similarity number traces to a real BlackHole `comparison.json`. No per-track constants anywhere.

---

- [ ] **Slice 1: One command reproduces the editable pipeline (driver + instrument assembler)**

  > After this slice `scripts/editable-pipeline.sh --url … --prefix …` can run stages 1–9 idempotently, and `build_instruments.py` turns granular models + the kit into a hosted-manifest `instruments/` dir.
  - [ ] Task: Create `scripts/python/build_instruments.py` — inputs: one or more granular model dirs (`models/<name>/pitched/*.wav` + `metadata.json`), the pack's `drums/` kit one-shots, `--out`, `--base-url`; outputs `instruments/<name>/<note>.wav`, `instruments/kit/{bd,sd,hh,oh}.wav`, `instruments/samples.json` keyed exactly like `models/regime_instruments/r2/samples.json` (note-name dict per instrument using `#`, kit entries as single-element arrays, absolute `_base` ending `/`); note names from the model's per-pitch-class `midi_note` medians (no fixed octave table). **[Agent: python-expert]**
  - [ ] Task: Create `scripts/python/pipeline_helpers.py` (pure functions, tested): `slug_from_title`, `num_bars(duration_s, bpm)`, `next_version_dir(track_dir)`, `promote_run(track_dir, mode, strudel, wav, comparison, meta)` writing `vNNN/{output.strudel,render.wav,comparison.json,metadata.json}` with the metadata keys the tech spec §2.1 stage 9 lists. **[Agent: python-expert]**
  - [ ] Task: Create `scripts/editable-pipeline.sh` implementing tech spec §2.1 stages 1–9 with per-stage skip-if-output-exists (`--force` to redo), `--mode sample-instrument|synth|both` (default both), `--iters` (default 3), `--dur` (default 170), `--r2|--local`, `--genre` override (logged), bpm/key/genre read from the track's `metadata.json`, a 3-second BlackHole capture probe before each render, and `pgrep record-strudel-blackhole` guard; prints a final JSON summary (paths, numbers, gate verdicts). **[Agent: python-expert]**
  - [ ] Task: Tests — `scripts/python/tests/test_build_instruments.py` (fixture model dir → manifest shape, note keys, `_base`, kit arrays) and `scripts/python/tests/test_pipeline_helpers.py` (slug, num_bars incl. rounding, next version dir, promote_run writes the four files + metadata keys); `@layer: unit`, `@spec: 004-second-reference-track`, `@regression`; RED-proven. **[Agent: python-expert]**
  - [ ] Verify: `bash -n scripts/editable-pipeline.sh`; `pytest -q` on the two new test files green; `build_instruments.py --models models/regime_bass … --out <scratch>` reproduces a manifest with the same keys as `models/regime_instruments/r2/samples.json` for `regime_bass`; delete the scratch dir. **[Agent: python-expert]**

- [ ] **Slice 2: The new track is extracted, transcribed, packed, trained and hosted**

  > After this slice `.cache/stems/<title>/` holds stems, `sample_pack/` (bass.mid, melodic.mid, vocals.mid, drums_bands.json, drum one-shots, vocals/, instruments/) and everything is on R2 under `midi-grep/vagabundo-nao-namora/`.
  - [ ] Task: Run stage 1 (`./bin/midi-grep extract --url https://youtu.be/SKjOR5EOR8Y --render none --iterate 0`); record bpm/key/genre from `metadata.json` and the exact title folder. Inline (long-running). **[Agent: python-expert]**
  - [ ] Task: Run stages 2–5 through the driver (`--mode none` or stage flags): MIDI per stem, drum bands, sample pack with `--prefix vagabundo-nao-namora`, granular `vagabundo-nao-namora_bass` / `_lead`, `build_instruments.py`. Any stage failure is a defect → fix in the stage's script with a reproducing test (Slice 5), then re-run. **[Agent: python-expert]**
  - [ ] Task: Run stage 6 (`upload_r2.py … --prefix midi-grep/vagabundo-nao-namora --bucket 4cast --backend wrangler`, `CLOUDFLARE_ACCOUNT_ID=503e92d7d95838d80c33802d3274f284`); verify the public `samples.json` and `instruments/samples.json` and one wav of each instrument return HTTP 200. **[Agent: python-expert]**
  - [ ] Verify: `ls` of the pack shows every stage output; `curl` of the public manifests lists `<slug>_bass`, `<slug>_lead`, `<slug>_vocal`, `vox0`, `bd/sd/hh/oh`. No scratch artifacts left outside `.cache/`. **[Agent: python-expert]**

- [ ] **Slice 3: Both modes generated, rendered, scored and gated (render gate)**

  > After this slice two `vNNN/` dirs exist for the track with stamped `comparison.json` files, both `editability: pass`.
  - [ ] Task: Stage 7 — `auto-calibrate.sh --iters 3 --dur 170 --mode sample-instrument` with the hosted manifests; keep the best; then one `--mode synth` generation with the best knobs. Both outputs through `editability_check.py` (must PASS). Inline — BlackHole is shared; 3-second probe before each render. **[Agent: audio-dsp-expert]**
  - [ ] Task: Stage 8–9 — `compare_audio.py original.wav <render> -d 135 -j --strudel <strudel> -o comparison.json` per mode; `eval/gate.py comparison.json --genre <genre>`; promote each to the next `vNNN/` with metadata (generator knobs, render mode raw, device ratio, compare args). **[Agent: ml-audio-expert]**
  - [ ] Verify: both `comparison.json` carry `editability: pass` and `generation_mode`; rendered tempo within 0.5% of the track's BPM (raw-capture check) else it is a defect; numbers + paths written to `flow-log.md`. Temp renders only under `.cache/`/scratchpad. **[Agent: ml-audio-expert]**

- [ ] **Slice 4: The track is a permanent reference and the docs say so honestly**

  > After this slice the dataset, thresholds (if a new genre) and CLAUDE.md carry the second track.
  - [ ] Task: `eval/datasets/reference_tracks.yaml` — add both entries (`cache_key`, `version`, `genre`, `mode`, measured numbers in a comment). **[Agent: ml-audio-expert]**
  - [ ] Task: `eval/thresholds.yaml` — if genre ≠ brazilian_funk add `modes.<mode>.genres/section_aware/measured.<genre>` = measured − 0.02 with `run`; if brazilian_funk leave floors and, on a shortfall, add a dated `# shortfall:` note under the mode + a follow-up task here. Never lower a floor. **[Agent: ml-audio-expert]**
  - [ ] Task: `CLAUDE.md` "Current achievement" — add the second track line (both modes, run ids, PASS/shortfall vs floors); `llms.txt` + `llms-full.txt` — document `scripts/editable-pipeline.sh` and `build_instruments.py`. **[Agent: python-expert]**
  - [ ] Verify: `pytest -q scripts/python/tests/test_similarity_gate.py` green with the new dataset entries exercised (not skipped); `grep` shows the second track in CLAUDE.md and the driver in llms.txt. **[Agent: python-expert]**

- [ ] **Slice 5: Defects fixed generally; Regime CLT does not regress**

  > After this slice every defect met on the new track has a general fix + test, and v026/v027 still clear their floors.
  - [ ] Task: For each defect logged in Slices 2–3: fix in the owning script (analysis/calibrator-derived, no per-track constant), add a reproducing test on a fixture, note it in `flow-log.md`. Owner by file: `scripts/python/*` → **[Agent: python-expert]**, `compare_audio.py`/`eval/`/`analyze*.py`/`detect_drums_bands.py` → **[Agent: ml-audio-expert]**, recorder/synthesis → **[Agent: audio-dsp-expert]**, `internal/`/`cmd/` → **[Agent: golang-expert]**. (No-op if no defects.)
  - [ ] Verify: `eval/gate.py` PASS on Regime CLT v026 and v027 after all fixes; full suite green (`go build/vet/test`, `pytest -q scripts/python/tests`, root `test_*.py`, `npm run build`). **[Agent: ml-audio-expert]**

- [ ] **Slice 6: Feature Testing & Regression**

  > Verifies the whole feature end-to-end against functional-spec.md, run after all implementation slices are complete.
  - [ ] Read functional-spec.md acceptance criteria in full. Generate acceptance-level tests that verify the entire feature as a whole — not individual slices. Cover applicable layers (unit for pure logic, integration for service interactions, e2e for user flows) based on the project's testing stack. Write tests with RED validation (must fail before implementation is confirmed done). Annotate each test with `@spec: 004-second-reference-track` and `@regression` if suitable for long-term regression. **[Agent: testing-expert]**
  - [ ] Run all generated tests. All must pass. Fix any failures before proceeding. **[Agent: testing-expert]**
