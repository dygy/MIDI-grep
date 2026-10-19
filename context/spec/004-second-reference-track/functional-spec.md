# Functional Specification: Second Reference Track for the Honest Eval Dataset

- **Topic:** Prove the editable-Strudel pipeline on a second real track and keep its honest
  measurements as a permanent reference, so the quality claims no longer rest on a single song.
- **Status:** Completed — all 15 acceptance criteria verified 2026-10-10 (evidence inline); shortfalls recorded as
  GitHub #10/#11, genre-detection hardening as #12
- **Author:** (with user) — `/implement-feature`, ticket `second-reference-track`
- **Ticket:** prompt (no issue); track https://youtu.be/SKjOR5EOR8Y — "VAGABUNDO NÃO NAMORA"
  (Christopher Luz, 2 min 31 s)
- **Parent spec:** `003-editable-strudel-generation` (its contract is exercised here on a new track)
- **Governing values:** `context/product/values.md`

---

## 1. Overview and Rationale (The "Why")

Every quality number MIDI-grep publishes today comes from one track (Regime CLT). A live coder who
tries the tool on a different song has no way to know whether the published similarity is typical or
a one-off. The October audit named this the biggest honesty gap: "1 track on disk".

This change runs the complete, unchanged pipeline on a second Brazilian-funk track — from the YouTube
link to two playable, editable Strudel pieces (one on sampled instruments built from the song, one on
pure synth sounds) — records how close each one gets to the original, and keeps those numbers as a
second permanent reference the project is measured against from now on.

**Success is measured by honesty, not by a score.** The deliverable is complete when both pieces are
editable and live-codeable by the project's own rules, their similarity is measured on real recorded
playback, the numbers are written down where the project keeps its reference results, and any place
the pipeline broke on the new song was fixed in the pipeline rather than patched for this song.

---

## 2. Functional Requirements (The "What")

### 2.1 Two editable pieces from the new track

- **As a** live coder, **I want** MIDI-grep to turn this second song into playable Strudel in both
  of its modes, **so that** I get the same kind of result the documentation shows for the first song.
  - **Acceptance Criteria:**
    - [x] When the pipeline is run on the track's YouTube link, then it produces a "sample-instrument"
          piece whose bass, lead and drum parts are each a separate, editable block of notes or
          patterns. The singing is not part of the generated music — no vocal part is produced, and the
          piece is compared against the song's instrumental. *(amended 2026-10-19 — see Change Log)*
          — verified 2026-10-10: v002 `output.strudel`: separate `bass`/`lead`/`vocal` bar arrays + drum pattern blocks; vocal is `note(cat(...vocal)).s("vagabundo_nao_namora_vocal")` (multisample built from the song's vocal stem, no `vocalsfull`) — `test_spec004_acceptance` §2.1
    - [x] When the pipeline is run in synth mode on the same track, then it produces a second piece that
          uses only built-in synthesizer and drum-machine sounds and loads no audio from the song.
          — verified 2026-10-10: v003 `output.strudel`: `generation_mode: synth`, no `samples(`, no song audio, palette sounds only — §2.1 tests
    - [x] When either piece is checked with the project's editability check, then the verdict is
          "pass" (no replayed audio, at least one editable voice per part, tempo set from the song).
          — verified 2026-10-10: `editability_check.py` PASS on both (driver stage 8 `editability_rc=0`); `test_spec004_acceptance` §2.1
    - [x] When either piece is opened in Strudel, then it plays without manual fixes.
          — verified 2026-10-10: both rendered through the real Strudel engine (BlackHole, raw capture, 170 s) without edits — v002/v003 `render.wav`; sample loads from R2 verified in the recorder log

### 2.2 Honest measurement of both pieces

- **As a** developer, **I want** both pieces scored against the original on a real recording of their
  playback, **so that** the second track's numbers are as trustworthy as the first track's.
  - **Acceptance Criteria:**
    - [x] When each piece is recorded through the real Strudel engine and compared with the original
          song, then the comparison states which mode produced it and that it is editable, and reports
          overall and section-by-section similarity.
          — verified 2026-10-10: v002/v003 `comparison.json` carry `generation_mode`, `editability: pass`, overall + section-aware similarity — §2.2 tests
    - [x] When the recorded playback is checked for speed, then its tempo matches the song's tempo
          (the recording chain must not slow down or speed up the music).
          — verified 2026-10-10: `tempo_diff_bpm` 0.0, `tempo_similarity` 1.000 on both (raw-capture recorder) — §2.2 tests (0.5% bound)
    - [x] When a piece fails the editability check, then no similarity is reported for it — the result
          says why it was rejected instead.
          — verified 2026-10-10: `compare_audio.py --strudel` exits 3 with `comparison: null` and no similarity on a replay fixture — §2.2 negative tests

### 2.3 The track becomes a permanent reference

- **As a** developer, **I want** this track's results kept alongside the first track's, **so that**
  future changes are checked against two songs, not one.
  - **Acceptance Criteria:**
    - [x] When the project's list of reference tracks is read, then it contains this track in both
          modes, each pointing at the recorded run it came from.
          — verified 2026-10-10: `eval/datasets/reference_tracks.yaml` entries for v002 (sample-instrument) and v003 (synth) — §2.3 tests
    - [x] Given the track's genre is Brazilian funk, when its scores are compared with the existing
          per-mode minimums for that genre, then the result is either "cleared" or an honestly recorded
          shortfall with a follow-up task — the minimums are never lowered to make it pass.
          — verified 2026-10-10: genre brazilian_funk: gate FAIL recorded honestly — v002 section-aware 0.8721 < 0.90 (overall 0.9282 clears 0.88); v003 0.7209/0.7520 < 0.77/0.81 + worst band 30.07%; floors unchanged (0.88/0.90, 0.77/0.81 from Regime CLT); `expected: shortfall` records equal the runs; follow-ups GitHub #10, #11 — §2.3 tests
    - [x] Given the track's genre turns out to be something else, when its scores are recorded, then new
          per-mode minimums for that genre are set from the measured scores minus the stated margin —
          never typed by hand.
          — verified 2026-10-10: not exercised on this track (genre brazilian_funk: CLAP 0.50 with bpm prior, override logged, floors already exist). The rule is implemented as `pipeline_helpers.py record-floor` (floor = round(measured − 0.02, 2) + measured block, no-op when a floor exists or the run is not detector-passing), called by driver stage 8; covered by `test_record_mode_floor_derives_from_measurement_and_never_touches_existing` — added after review finding #4 flagged the earlier claim as unbacked.
    - [x] When the project's quality claims are read (the "current achievement" section), then the
          second track's numbers appear next to the first track's, labelled with the run they come from.
          — verified 2026-10-10: CLAUDE.md "Current achievement" second-track paragraph (92.8/87.2 editable v002, 72.1/75.2 synth v003, shortfall recorded) — §2.3 test

### 2.4 The pieces play from the public host

- **As a** live coder, **I want** the sampled instruments of the new song hosted publicly, **so that**
  the sample-instrument piece plays from the link I am given, without any local server.
  - **Acceptance Criteria:**
    - [x] When the sample-instrument piece is loaded in Strudel on a machine with no local server
          running, then every instrument it uses (bass, lead, drums) loads and plays.
          *(amended 2026-10-19: no vocal instrument exists any more)*
          — verified 2026-10-10: render straight from R2 (no local host) loads `vagabundo_nao_namora_bass/lead/vocal`, kit and vox; pieces reference exactly the two public manifests under `midi-grep/vagabundo-nao-namora/` — §2.4 static tests + recorder log
    - [x] When the hosted sample set is inspected, then it lives under this track's own folder next
          to the first track's, on the project's existing public host.
          — verified 2026-10-10: public `…/midi-grep/vagabundo-nao-namora/samples.json` (138 keys) and `…/instruments/samples.json` next to `…/midi-grep/regime-clt/` on the same host — HTTP 200 checks in flow-log

### 2.5 Defects are fixed in the pipeline, not around it

- **As a** maintainer, **I want** anything that breaks on this song fixed for all songs, **so that**
  the tool does not accumulate one-song patches.
  - **Acceptance Criteria:**
    - [x] When a step fails or misbehaves on this track, then the fix is a change to the step's
          general behavior (derived from analysis of the audio, or from the calibrator), with a test,
          and no value in the pipeline is tuned by hand to this song.
          — verified 2026-10-10: eight defects fixed generally with tests: yt-dlp resolver, report skip, CLAP safetensors, train output anchoring, stamp step, pack naming, calibrator wait, driver itself; no per-track constant (§2.5 tests grep the driver/assembler/helpers for the track's bpm/key/slug/id); genre-detection hardening filed as #12
    - [x] When the first track is re-run through the fixed pipeline, then its existing reference
          results still clear their minimums (no regression).
          — verified 2026-10-10: Regime CLT v026 PASS 0.901/0.921, v027 PASS 0.788/0.832 after all fixes — §2.5 regression test

---

### 2.6 Every promoted version has a complete report (added 2026-10-10, fix `editable-reports-missing`)

- **As a** developer reviewing a run, **I want** every version the pipeline promotes to come with the
  same self-contained HTML report the `extract` path produces, **so that** I can listen to the original
  and rendered stems, read the charts and the per-stem comparison, and follow the calibration history
  without re-running anything.
  - **Acceptance Criteria:**
    - [ ] When the pipeline promotes a version (either mode), then `vNNN/report.html` exists and the run
          FAILS loudly if it cannot be produced — a report is never skipped silently.
    - [ ] When the report is opened, then it contains the original stems, the rendered mix and its
          Demucs-separated stems (labelled as re-separation per spec 003 §2.5), the overall charts, the
          per-stem comparison, the gate/editability headline, and the calibration iteration history with
          each iteration's render (and its stems unless `--no-iteration-stems`).
    - [ ] When the report stage runs, then the version's stamped `comparison.json` is byte-identical
          before and after (the detector verdict and metrics are never recomputed by the report).
    - [ ] When the report embeds audio, then it uses compressed (mp3) copies so the file stays
          practical to open (the first run embedded WAVs and reached 227 MB).

## 3. Scope and Boundaries

### In-Scope

- One complete pipeline run on the new track, both modes, with real recorded playback and scoring.
- Recording the results as permanent reference entries, with per-genre minimums handled as in 2.3.
- Hosting the new track's sample set publicly and pointing the piece at it.
- Fixing pipeline defects the track exposes, with tests.
- A complete HTML report for every promoted version (§2.6, stage 10 of the driver).
- Updating the project's quality claims and context documents.

### Out-of-Scope

- Improving the similarity scores themselves (calibration runs beyond the pipeline's standard pass).
- Any new generation mode, genre, or sound palette.
- Changes to the web interface.
- Hosting on any host other than the project's existing one.

---

## Change Log

- **2026-10-11 — fix #11 (follow-up; tech spec stage 7 amended, no functional criterion changed).** The §2.3 synth
  shortfall (`v003` 0.7209 / 0.7520, worst band 30.07%) came from the driver rendering synth mode ONCE with the
  sample-instrument knobs (hat layer 0, bass/sub tuned for a sampled bass) and from the calibrator having no lever for
  the band a sawtooth bass over-fills (low_mid 37.6% vs 7.5%). Stage 7 now runs `auto-calibrate.sh --mode synth`
  (own knobs + per-bar correction, started from the sample-instrument BEST); `calibrate_dynamic.py` gains a
  measurement-driven `bass_lpf` lever (bounded by the bass note range and the analysis cutoff) and a data-driven hat
  engagement rule. Floors unchanged. Result recorded in the fix log `context/fix-log-11.md` and the dataset.
- **2026-10-10 — fix `editable-reports-missing` (divergence: scope omission).** The driver was specified as
  stages 1–9 and `promote` as four artifacts; no version promoted by it had an HTML report (VAGABUNDO
  v001–v004, Regime v024–v027), while CLAUDE.md promises the report as an always-on analysis feature and
  spec 003 §2.5 assumes it exists. Added §2.6 (complete report per promoted version) and stage 10 in the
  technical spec: `pipeline_helpers.py report` = charts → Demucs on `render.wav` → per-stem comparison →
  `iterations.json` from the calibration workdir → `generate_report.py`, failure = error. Criteria stay
  unticked until `/awos:verify`; evidence lands in the fix log `context/fix-log-editable-reports-missing.md`.
- **2026-10-10 — fix #10 (follow-up, no criterion changed).** The §2.3 shortfall recorded for the
  sample-instrument run `v002` (0.9282 / 0.8721 < 0.90 section floor) is resolved: the per-window data
  (`compare_audio.py` now exports `comparison.section_windows`) showed the bass/lead balance flipping
  across sections under one global knob set; `calibrate_dynamic.py --env-correction-out` derives a damped
  per-bar bass/lead/master correction from it and `generate_dynamic_strudel.py --env-correction` applies
  it. Same global knobs, floors unchanged → `v004` **0.9320 / 0.9216 PASS** (`editability: pass`, tempo
  1.000). `eval/datasets/reference_tracks.yaml` now points the sample-instrument entry at `v004`
  (`expected: pass`); `v002` stays on record in its comment and in CLAUDE.md. The synth shortfall (`v003`,
  GitHub #11) is unchanged. Evidence lines above that cite `v002` as the dataset run are historical.

- **2026-10-19 — product decision (owner): vocals are never generated.** §2.1 and §2.4 criteria amended: the
  sample-instrument piece has bass, lead and drum blocks only, and no vocal instrument is built or hosted; pieces
  are scored against the instrumental (`instrumental.wav`), not the full mix. The 2026-10-10 verification notes
  that cite a vocal block / vocal instrument describe the behaviour at that time and are superseded. Same run:
  the recorder was found to pitch-shift renders by about −2.3 semitones (sample loss + rate relabel) and is
  replaced by an offline Strudel render as the default; the "tempo matches the song" criterion (§2.2) is
  re-verified with sample-exact timing. Re-measured on the instrumental target: sample-instrument VAGABUNDO v013 0.7170/0.7073, Regime v031 0.7794/0.7744 → floors 0.70/0.69; synth VAGABUNDO v014 0.7240/0.7007, Regime v032 0.7479/0.7361 → floors 0.70/0.68;
  the dataset now points at v013 / v014 (VAGABUNDO) and v031 / v032 (Regime), all `expected: pass`.
