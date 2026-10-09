# Orchestration v2 — Audio-Driven Tasks (stop guessing by name)

## TL;DR

> **Problem**: the orchestrated codegen produces a short **looped** fragment with **static** params and
> sounds chosen by **genre name only** — it doesn't use the actual audio. The eval gate rewards
> whole-track average spectral match, so it reads ~90% while sounding nothing like the evolving
> original. Four interdependent fixes, all feeding the same `arrange()`.
>
> **Estimated Effort**: Large · **Sequential** (one model, 24GB)

## Root causes (confirmed)
- **Loops**: `arrange()` ≈ 19 cycles ≈ 37s, rendered over 170s → repeats ~4.6×. Structure job invents
  ~19 cycles, ignores real duration (Go hardcodes `--duration 30`). 170s needs ~87 cycles.
- **Frozen params**: assembler emits static `.gain(x).lpf(y)`; never uses Strudel signals
  (`sine`/`saw`/`perlin`/`range`/`.slow()`). No sweeps/swells/movement.
- **Sound = name guess**: selection is genre-RAG + LLM; no per-sound timbre data, no acoustic match to
  the original stem. `select_sounds_for_timbre()` exists but is a hardcoded heuristic and unused.
- **Metric lies**: gate scores whole-track *average* (freq 40% / MFCC 20% / energy 15% …). A static
  loop matching the average scores ~90%. Optimizing it ≠ sounding real; more dynamism may *lower* it.

## Tasks

- [x] **A — Full-song timeline.** Structure step derives total cycles from the REAL duration
  (total = duration · cps) and the detected sections (`smart_analyze` start/end when present, else
  proportional). `arrange()` spans the whole track → no perceptible loop. Stop hardcoding
  `--duration 30` in Go; pass the real duration. Each real section = a subtask. Annotate each
  section with its time range (cycles→seconds) as the renderer instruction.

- [x] **D — Sound resolver task** (retrieve→re-rank). One-time offline: profile each Strudel sound
  (render a note, extract centroid/brightness/harmonic-richness/attack — or a CLAP embedding) into a
  sound-timbre DB. New job: analyze the original stem's timbre per voice → take genre-RAG candidates
  → re-rank by timbre distance → pick the acoustically closest. Replaces name-only selection.

- [x] **B — Continuous modulation.** Assembler emits Strudel signals on the METHOD side (safe, no
  note()-parse risk): `.lpf(sine.range(lo,hi).slow(n))`, gain swells via `saw`/`perlin.range(...)`,
  filter opens through builds. Voice jobs request modulation as structured params; assembler renders
  valid signals + validation allows `sine|saw|perlin|rand`+`.range(`+`.slow(` on method args.

- [x] **C — Section-aware scoring.** Use `compare_by_sections` so each rendered section is scored vs
  the original's SAME time window. Gate rewards matching the song's evolution, not its average.
  Expect overall "average" score to drop when dynamism rises — that's correct; re-baseline floors.

## Sequencing / recommendation
A + B first (audible: full song + movement). Then D (right sounds). Then C (metric stops lying;
re-baseline `eval/thresholds.yaml`). A+B+D will likely LOWER the current average-spectral gate — that
is expected and fine; C is what makes a high score mean "sounds like the track over time".

## Guardrails
- Modulation goes on METHOD args only, never inside `note("...")` (caused the silent-render crash).
- Keep the deterministic assembler (structure/setcps/3-voice guaranteed by construction).
- Don't let "average gate %" be the goal until C lands.

## Re-baseline (done)
Gate re-centered on section-aware: the iteration loop now requires BOTH the overall floor AND the
section-aware floor to pass/early-stop. A static loop with high overall but low section-aware no
longer passes. section_aware floors in thresholds.yaml are conservative (ratchet up as data grows).
All four tasks (A/B/C/D) done + verified; v012 render: overall 0.779, section-aware 0.804.
