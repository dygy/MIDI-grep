# Functional Specification: Second Reference Track for the Honest Eval Dataset

- **Topic:** Prove the editable-Strudel pipeline on a second real track and keep its honest
  measurements as a permanent reference, so the quality claims no longer rest on a single song.
- **Status:** Approved
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
    - [ ] When the pipeline is run on the track's YouTube link, then it produces a "sample-instrument"
          piece whose bass, lead, vocal and drum parts are each a separate, editable block of notes or
          patterns, with the vocal part played as notes on an instrument built from the song (not a
          replay of the singer's recording).
    - [ ] When the pipeline is run in synth mode on the same track, then it produces a second piece that
          uses only built-in synthesizer and drum-machine sounds and loads no audio from the song.
    - [ ] When either piece is checked with the project's editability check, then the verdict is
          "pass" (no replayed audio, at least one editable voice per part, tempo set from the song).
    - [ ] When either piece is opened in Strudel, then it plays without manual fixes.

### 2.2 Honest measurement of both pieces

- **As a** developer, **I want** both pieces scored against the original on a real recording of their
  playback, **so that** the second track's numbers are as trustworthy as the first track's.
  - **Acceptance Criteria:**
    - [ ] When each piece is recorded through the real Strudel engine and compared with the original
          song, then the comparison states which mode produced it and that it is editable, and reports
          overall and section-by-section similarity.
    - [ ] When the recorded playback is checked for speed, then its tempo matches the song's tempo
          (the recording chain must not slow down or speed up the music).
    - [ ] When a piece fails the editability check, then no similarity is reported for it — the result
          says why it was rejected instead.

### 2.3 The track becomes a permanent reference

- **As a** developer, **I want** this track's results kept alongside the first track's, **so that**
  future changes are checked against two songs, not one.
  - **Acceptance Criteria:**
    - [ ] When the project's list of reference tracks is read, then it contains this track in both
          modes, each pointing at the recorded run it came from.
    - [ ] Given the track's genre is Brazilian funk, when its scores are compared with the existing
          per-mode minimums for that genre, then the result is either "cleared" or an honestly recorded
          shortfall with a follow-up task — the minimums are never lowered to make it pass.
    - [ ] Given the track's genre turns out to be something else, when its scores are recorded, then new
          per-mode minimums for that genre are set from the measured scores minus the stated margin —
          never typed by hand.
    - [ ] When the project's quality claims are read (the "current achievement" section), then the
          second track's numbers appear next to the first track's, labelled with the run they come from.

### 2.4 The pieces play from the public host

- **As a** live coder, **I want** the sampled instruments of the new song hosted publicly, **so that**
  the sample-instrument piece plays from the link I am given, without any local server.
  - **Acceptance Criteria:**
    - [ ] When the sample-instrument piece is loaded in Strudel on a machine with no local server
          running, then every instrument it uses (bass, lead, vocal, drums) loads and plays.
    - [ ] When the hosted sample set is inspected, then it lives under this track's own folder next
          to the first track's, on the project's existing public host.

### 2.5 Defects are fixed in the pipeline, not around it

- **As a** maintainer, **I want** anything that breaks on this song fixed for all songs, **so that**
  the tool does not accumulate one-song patches.
  - **Acceptance Criteria:**
    - [ ] When a step fails or misbehaves on this track, then the fix is a change to the step's
          general behavior (derived from analysis of the audio, or from the calibrator), with a test,
          and no value in the pipeline is tuned by hand to this song.
    - [ ] When the first track is re-run through the fixed pipeline, then its existing reference
          results still clear their minimums (no regression).

---

## 3. Scope and Boundaries

### In-Scope

- One complete pipeline run on the new track, both modes, with real recorded playback and scoring.
- Recording the results as permanent reference entries, with per-genre minimums handled as in 2.3.
- Hosting the new track's sample set publicly and pointing the piece at it.
- Fixing pipeline defects the track exposes, with tests.
- Updating the project's quality claims and context documents.

### Out-of-Scope

- Improving the similarity scores themselves (calibration runs beyond the pipeline's standard pass).
- Any new generation mode, genre, or sound palette.
- Changes to the web interface.
- Hosting on any host other than the project's existing one.

---

## Change Log

- (none yet)
