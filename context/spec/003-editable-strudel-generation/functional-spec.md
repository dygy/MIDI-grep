# Functional Specification: Editable, Live-Codeable Strudel Generation

- **Roadmap Item:** Phase 10 (reframed) — raise the honest similarity of *editable*
  Strudel output (supersedes "raise similarity" framing).
- **Status:** Completed — all 21 acceptance criteria verified 2026-10-09 (evidence inline). Two hardening
  tasks stay open in `tasks.md` Slice 4: a take-deterministic live Editability Test and a robust tempo estimate.
- **Author:** (with user) — codifies the Jun 2026 "wtf is this / how do you edit
  stems" course-correction.
- **Governing values:** `context/product/values.md` (V1–V5, anti-patterns A1–A4).

---

## 1. Overview and Rationale (The "Why")

### Problem Statement

MIDI-grep's purpose is to hand a live coder **Strudel they can play and edit**. A prior
effort chased a single similarity number and produced a deliverable that scored ~95% by
**replaying the original recording** (`s("originalfull").slice(78, run(78)).slow(78)`).
That output is worthless for the product: you cannot live-play it, you cannot edit a
note, you cannot touch a stem — it is the master recording in a wrapper (Anti-Pattern
A1). The high score was a tautology: it measured "is this the original?", not "is this
good generated music?".

### Solution

Specify, as a hard contract, that **every MIDI-grep deliverable must be editable and
live-codeable** (Values V1–V2), that **resemblance must come from re-performance, not
reproduction** (V3), and that **similarity is only ever measured on genuinely generated
output** (V4, anti-pattern A3). Replay is disqualified *before* scoring.

### Success Criteria

- 100% of shipped deliverables pass the **Editability Test** (§2.1) and **Live-Coding
  Test** (§2.2).
- 0 deliverables are produced by audio replay (A1) or scored on replayed audio (A3).
- Reported similarity numbers are reproducible from an actual run on **generated**
  output and are honest (V4).
- The honest similarity of editable output **improves over time** without resorting to
  replay.

---

## 2. Functional Requirements (The "What")

### 2.1 Editability — the deliverable is editable source

**As a** live coder, **I want** the generated Strudel to be editable note/pattern
material, **so that** I can change notes, sounds, and structure — not just replay a file.

**Acceptance Criteria:**

- [x] The output exposes per-voice **editable data**: note material as
      `note("…")`/bar-arrays, drum patterns as `s("bd sd …")`/bar-arrays, and per-voice
      effect chains — one identifiable block per voice.
      — verified 2026-10-09: `test_21_every_voice_is_its_own_block_of_editable_data` on v024/v025 + fixtures
- [x] **Editability Test:** changing a single note (e.g. `c2`→`eb2`) in any voice
      produces audio that audibly reflects that change at that position.
      — verified 2026-10-09: static: `test_21_single_note_edit_changes_exactly_one_token_and_stays_valid`; live: `editability_test.py` on v024 lead bar 2 (b3→d4) — the edited step's dominant pitch class moved B→D in the BlackHole render (audible at that position); 'nothing else changed' is inconclusive on this non-deterministic recorder (tasks.md Slice 4)
- [x] **Sound-swap Test:** changing a voice's `.s("…")`/`.bank("…")` re-voices that part
      while keeping its pattern.
      — verified 2026-10-09: `test_21_sound_swap_revoices_but_keeps_the_pattern` (+ replay-sound swap rejected)
- [x] **Structure Test:** a user can reorder/duplicate bars (e.g. `bass.slice(0,4)`) and
      the arrangement changes accordingly.
      — verified 2026-10-09: `test_21_reordering_and_duplicating_bars_stays_a_valid_deliverable`, `test_21_user_bar_slicing_is_not_replay`
- [x] **FORBIDDEN:** the deliverable MUST NOT be a single baked audio file replayed back
      (full-mix or stem) via `slice(N,run(N)).slow(N)`, `loopAt(N)`, or equivalent
      reconstruction-by-playback (Anti-Pattern A1). Such output is rejected regardless of
      similarity score.
      — verified 2026-10-09: `editability_check.py` R1/R2/R5/R6; `test_21_forbidden_replay_forms_are_rejected_at_every_gate` (detector, validator, gate); v012/v023 FAIL

### 2.2 Live-codeability — the output performs

**As a** performer, **I want** to mute/solo/swap voices live, **so that** I can use the
output on stage.

**Acceptance Criteria:**

- [x] Output is structured as independent voices (e.g. multiple `$:` blocks or a
      `stack(...)` of named voices) that can be commented out individually.
      — verified 2026-10-09: `test_22_each_voice_can_be_commented_out_individually`
- [x] **Mute Test:** commenting out one voice leaves the others playing musically.
      — verified 2026-10-09: `test_22_each_voice_can_be_commented_out_individually` (remaining voices stay a valid deliverable); negative `test_22_muting_everything_or_half_a_block_is_detected`
- [x] The output plays in Strudel **without manual fixes** (valid syntax, valid sound
      names — see existing `ollama_agent.py` validation).
      — verified 2026-10-09: v024 and v025 rendered through the real Strudel engine (BlackHole) with no edits; `test_22_output_passes_the_generation_time_validator`
- [x] `setcps()` is set from the detected BPM so the patterns play at the track's tempo.
      — verified 2026-10-09: `test_22_setcps_matches_the_declared_bpm`, `test_22_generator_sets_cps_from_the_requested_bpm`; a 136-BPM click through the chain measured 136.0

### 2.3 Resemblance via re-performance (timbre + content, not the master)

**As a** user, **I want** the output to *sound like* the track, **so that** it's
recognizable — while staying fully mine to edit.

**Acceptance Criteria:**

- [x] Resemblance is achieved through extracted **musical content** (notes, groove,
      harmony, key, tempo) plus **timbre** from one of:
      (a) synth voices, or (b) **sample-instruments built from the track's stems** that
      are played by the **user's own** `note()`/drum patterns.
      — verified 2026-10-09: `--mode sample-instrument` (stem-derived multisamples incl. the vocal) and `--mode synth` (palette); `test_23_sample_instrument_realism_lives_in_declared_note_data`, `test_23_synth_mode_is_pure_synthesis`
- [x] If sample-instruments are used, the **notes/patterns/arrangement are editable
      data** (the realism is in the *sounds*, not in a baked performance) — this is the
      line between allowed (sample-instrument) and forbidden (replay).
      — verified 2026-10-09: v024 output: `let bass/lead/vocal` arrays + drum patterns on `regime_*` instruments, detector PASS with 8 editable voices
- [x] Per-bar real-stem **loops** are permitted only as *texture layered under* editable
      voices and only when user-arrangeable; a loop-only output is **not** a deliverable
      (see values §3).
      — verified 2026-10-09: R4 texture rule (`// texture` marker + ≥2 editable voices); `test_23_loops_are_texture_only_under_two_editable_voices`, `test_23_a_loop_only_output_is_not_a_deliverable`

### 2.4 Honest, valid similarity measurement

**As a** developer, **I want** similarity scored honestly and only on generated output,
**so that** the metric guides real improvement.

**Acceptance Criteria:**

- [x] Similarity (`compare_audio.py`, MAE-weighted: freq 40 / MFCC 20 / energy 15 /
      brightness 15 / tempo 5 / chroma 5) is computed **only** on output that passes
      §2.1 (the Editability Test).
      — verified 2026-10-09: `compare_audio.py --strudel` runs the detector first; `test_24_replay_is_refused_before_scoring_end_to_end`
- [x] Computing or reporting similarity on replayed/baked source audio is **INVALID**
      (Anti-Pattern A3) and MUST NOT be presented as a result.
      — verified 2026-10-09: exit 3 + `comparison: null`; gate FAIL `editability: fail`; `test_24_v023_is_disqualified`
- [x] Every reported number is reproducible from an actual run (no assumed/aspirational
      figures) — per CLAUDE.md Self-Review.
      — verified 2026-10-09: `eval/thresholds.yaml modes.*.measured` → v024/v025 runs; `test_24_mode_floors_are_reproducible_from_a_real_run` (+ tamper test); CLAUDE.md numbers cite the run
- [x] When a render scores high, the pipeline/report MUST be able to confirm the output
      is generated (passes §2.1), not replayed — a replayed render is flagged, not
      celebrated.
      — verified 2026-10-09: `editability`/`generation_mode` stamped in comparison.json & metadata; `ai_improver` fail-closed; report badge; `test_24_ai_improver_stamps_fail_closed`
- [x] **Two generation modes ship, both honest** (decided Jun 2026):
      `--mode sample-instrument` (**default** — stem-derived timbre, user patterns) and
      `--mode synth` (pure synthesized voices). Each is editable + live-codeable (§2.1–2.2).
      — verified 2026-10-09: `generate_dynamic_strudel.py --mode {sample-instrument,synth}`; `test_24_two_modes_ship_and_loops_is_not_one_of_them`
- [x] **Targets are measured per mode, not guessed.** For each mode: build the editable
      output, run `compare_audio.py` on that generated render, and set the mode's target
      from the measured honest score (the floor to then beat). Measured honest points on
      Regime CLT: `synth` ≈ 59%; **`sample-instrument` DJ-flow = 91.4%** (freq-balance
      95.2%, timbre 98%, harmony 99%, tempo 100%, energy 76%) — a fully editable,
      live-codeable render, NOT replay. This is the `sample-instrument` floor to hold/beat.
      No target is ever set by replay.
      — verified 2026-10-09: measured 2026-10-09 on BlackHole renders: sample-instrument v024 0.8743/0.9194 (floor 0.85/0.90), synth v025 0.7589/0.8406 (floor 0.74/0.82); the 91.4%/59% figures quoted in this bullet were pre-contract (vocal replay) and are superseded

### 2.5 Reporting reflects the values

**As a** user reviewing a run, **I want** the report to show editability and honest
scores, **so that** I'm never shown a misleading number again.

**Acceptance Criteria:**

- [x] The HTML report's similarity headline is taken from a **generated** render
      (§2.4), and the report states which generation mode produced it (synth /
      sample-instrument).
      — verified 2026-10-09: `generate_report.py` + Go report: `<pct>% — mode: … · editable: pass`; `test_25_headline_states_the_generating_mode`
- [x] The report MUST NOT present a replay-derived score as the deliverable's quality.
      — verified 2026-10-09: replay badge, no percentage; `test_25_replay_score_never_reaches_the_report`
- [x] Per-stem panels make clear whether numbers come from generated stems or from
      demucs re-separation (the Jun 2026 confusion: re-separating a mix is lossy and not
      a true stem-match view).
      — verified 2026-10-09: demucs re-separation caption on per-stem panels (Python + Go); `test_25_go_and_python_reports_agree`

---

## 3. Scope and Boundaries

### In-Scope

- The contract that all deliverables are editable + live-codeable Strudel (§2.1–2.2).
- Allowed resemblance techniques: synth note-material and stem-derived
  sample-instruments played by user patterns (§2.3).
- Honest, valid similarity measurement on generated output only (§2.4).
- Report honesty (§2.5).
- Reframing Phase 10 as "raise honest similarity of *editable* output".

### Out-of-Scope

- **Audio replay of any kind** (A1) — not deferred, **forbidden** as a deliverable.
- The *numeric* similarity targets — set per mode by measurement (§2.4), not up front.
- Net-new genres/modes, web-UI changes, distribution — other roadmap phases.
- Training new RAVE/granular models — Phase 9 / generative spec (002).
