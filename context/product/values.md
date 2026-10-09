# MIDI-grep — Core Values & Anti-Patterns

- **Version:** 1.0
- **Status:** Approved
- **Purpose:** The non-negotiable principles that define what a *good* MIDI-grep
  output is. Every spec, metric, and acceptance criterion must serve these. When
  a metric and a value conflict, **the value wins and the metric is wrong.**

---

## 0. The One-Sentence Value

> MIDI-grep turns a recording into **Strudel code a live coder can play and edit** —
> code that *resembles* the track because the musician can perform it, **not** because
> it secretly *is* the track.

---

## 1. Core Values (ranked)

### V1 — Editability is the product. (highest)
The deliverable is **editable Strudel source**: note material (`note("c2 e2 …")`),
pattern/bar arrays, drum patterns, and per-voice effect chains. A user must be able to
open the output and **change a single note, bar, sound, or effect** and hear a
correspondingly different, musically-coherent result.

> **Test:** Change one note in any voice. Does the audio change in a way that matches
> that edit? If yes → editable. If the audio is unaffected (or only a playback index
> moved) → it is **not** a MIDI-grep deliverable.

### V2 — It must be live-codeable.
The output is meant to be **performed live in Strudel**: voices can be muted/soloed,
patterns swapped, sounds changed, sections rearranged on the fly. Structure is voices
and patterns, not one opaque block.

> **Test:** Comment out one voice (`// $: …`). The rest keeps playing musically. Swap a
> `.s("…")` sound. The part re-voices. Both must work.

### V3 — Resemblance comes from *re-performance*, not reproduction.
The output should sound like the track because we extracted its **musical content**
(notes, groove, harmony, key, tempo) and its **timbre** (via stem-derived
sample-instruments or chosen synth voices) — and let the user perform it. Realism is
earned through faithful *content + timbre*, never by playing the master back.

### V4 — Honesty over flattering numbers.
We report the **real** ceiling and the **real** score of genuinely generated output,
even when it's lower than a number we could fake. A metric exists to guide
improvement, not to be won.

### V5 — Generalization over per-track tuning.
A technique must work for *any* input by learning/adapting (this is the existing
ZERO-HARDCODING principle in CLAUDE.md). Tuning to one test file is not progress.

---

## 2. Anti-Patterns (explicitly forbidden as deliverables)

### A1 — Audio replay / "tape with a wrapper". ❌ FORBIDDEN
Hosting the original recording (full mix **or** a stem) as a baked audio file and
playing it back — e.g. `s("originalfull").slice(N, run(N)).slow(N)`,
`loopAt(N)`, or any reconstruction-by-playback. It is **not** a deliverable because it
**fails V1 and V2**: nothing can be edited, nothing can be performed; changing a slice
index just seeks to a different second of the same recording.

> Replay will always score near-100% similarity **because it is the original**. This
> makes a high replay score **meaningless** — it measures "is this a copy of the
> master?", a tautology, not "did we generate something good?".

### A2 — Optimizing the similarity metric as the goal. ❌
Similarity is a *secondary* signal of resemblance (V3), not the objective. Chasing the
number leads straight to A1. The objective is V1–V3; similarity only ranks among
deliverables that already satisfy them.

### A3 — Measuring similarity on replayed/baked audio. ❌ INVALID
A similarity score is only valid when computed on **genuinely generated** output
(synth voices and/or user-pattern-driven sample-instruments). Scoring replayed source
audio is invalid and must never be reported as a result.

### A4 — Hardcoding values to a specific track. ❌ (see CLAUDE.md ZERO-HARDCODING)

---

## 3. What "resemblance" is allowed to use

| Technique | Editable? | Allowed as deliverable? | Why |
|---|---|---|---|
| Synth voices from note-material (`.s("sawtooth")` etc.) | ✅ | ✅ | Pure generation; fully editable. Honest ceiling ~lower. |
| **Sample-instruments** built from stems (one-shots/multisamples played by the user's *own* `note()`/drum patterns) | ✅ | ✅ | Timbre is real, **but the user controls the notes** → editable & live (V1–V3). |
| Per-bar **loops** of real stems arranged by the user | ⚠️ partial | ⚠️ only if user-arrangeable & layered with editable voices | Loops are coarse; acceptable as *texture* under editable parts, never as the whole output. |
| Full-mix or single baked stem replayed back (A1) | ❌ | ❌ **NEVER** | Not editable, not performable. It's the master. |

**Rule of thumb:** the realism may come from the **sounds** (stem-derived timbre); the
**notes, groove, and arrangement must be data the user can edit.**

---

## 4. The similarity metric — corrected role

- Similarity (`compare_audio.py`, MAE-weighted) is a **recognizability check on
  generated output**, reported honestly (V4).
- It is **only** computed on output that passes the Editability Test (V1) — never on
  replay (A3).
- A lower honest score on an editable deliverable **beats** a higher score on replay,
  every time. Replay is disqualified before scoring.
- **Two modes ship (decided Jun 2026):** `--mode sample-instrument` (default,
  stem-derived timbre + user patterns) and `--mode synth` (pure synthesis). Each mode's
  target is **measured** from its first honest generated render, not guessed. Known
  honest point: `synth` ~59% on Regime CLT; `sample-instrument` measured after build.
  These are *floors to raise honestly*, never bypassed by replay.

---

## 5. How this reframes the roadmap

Phase 10 ("Audio Similarity & Synthesis Quality 🔴 CRITICAL") is hereby scoped as
**"raise the honest similarity of *editable* output"** — not "raise similarity." Any
roadmap/spec item that would be satisfied by replay (A1) is out of spec by definition.
The functional contract lives in
`context/spec/003-editable-strudel-generation/functional-spec.md`.
