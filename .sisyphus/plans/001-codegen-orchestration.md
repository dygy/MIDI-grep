# Job-Based Codegen Orchestration

## TL;DR

> **Quick Summary**: Replace the single-giant-prompt Strudel codegen with a job/task mechanism —
> small, individually-validated, retryable LLM jobs (per voice / per concern) assembled
> deterministically into Strudel. The LLM produces only validated JSON content decisions; a pure
> assembler builds the structure (3 voices, arrange(), setcps from BPM) so whole error classes
> (hallucinated sounds, missing voices, broken tempo) become impossible.
>
> **Deliverables**:
> - `strudel_validation.py` — single source of truth for valid sounds/banks/methods + `validate_code()`
> - `job_runner.py` — JobSpec/JobResult + DAG runner with validate→retry→logged-fallback
> - `codegen_orchestrator.py` — builds the JobGraph, assembles, writes output.strudel + job_run.json
> - Go `--codegen orchestrated|single` flag (default single until proven)
> - Gate/report canonical-artifact fix
>
> **Estimated Effort**: Large
> **Parallel Execution**: NO — jobs run sequentially (one 13GB Ollama model on 24GB)
> **Critical Path**: Phase 0 → Phase 1 → A/B → flip default

## Context

### Problem
Go calls `ollama_codegen.py` with ONE prompt for all 3 voices × all sections. One bad token ruins
the run: `tr808` instead of `RolandTR808`, `sub_bass`, tempo 0.9%, huge run-to-run variance. The
iterate loop (`ai_improver` → `ollama_agent`) also uses one big rewrite prompt.

### Research findings
- `ai_orchestrator.py` already has decomposed prompts (`prompt_sections/prompt_voice/prompt_drums/
  prompt_mix/assemble_code`) — NOT wired in, NO validation/retry (fires once, falls back silent).
- Validation is fragmented across 3 files: `strudel_validation.py` (SOUND/BANK_CORRECTIONS incl.
  tr808), `ollama_agent._validate_code` (VALID_SOUNDS, INVALID_METHODS, INVALID_GM_PATTERNS), and
  `ollama_codegen.fix_strudel_syntax` (own regex). The fragmentation is why tr808 leaked.
- Gate/report mismatch: `ai_improver` picks the gate-approved render, but `main.go` post-iterate
  RE-RENDERS output.strudel and overwrites comparison.json → report shows a different render.

## Work Objectives

### Core Objective
LLM emits small validated JSON per concern; deterministic assembler builds correct Strudel. Mistakes
are caught + retried per job, never doom the whole run.

### Must NOT Have (Guardrails)
- **MUST NOT** let the assembler call the LLM (structure is deterministic).
- **MUST NOT** let the JobRunner emit/parse Strudel syntax (JSON only).
- **MUST NOT** explode into per-section×per-voice jobs (keep 4–5 jobs).
- **MUST NOT** flip the Go default to orchestrated until it beats single-shot on the gate.
- **MUST NOT** silently fall back — a job that exhausts retries logs status=fallback into job_run.json.

## Phases (each shippable; default unchanged until proven)

- [x] **Phase 0 — shared validation** (standalone value: fixes tr808 on all paths)
  - Consolidate VALID_* + INVALID_* + a standalone `validate_code(code)->(corrected, error)` into
    `strudel_validation.py`. Re-export from `ollama_agent` for back-compat. Route
    `ollama_codegen.fix_strudel_syntax` + `ollama_agent._validate_code` through it.
- [x] **Phase 1 — job runner** : `job_runner.py` (JobSpec/JobResult, topo order, validate→retry≤2→
  logged fallback, job_run.json) + `codegen_orchestrator.py` wrapping ai_orchestrator prompts.
  Go `--codegen` flag (default single).
- [x] **Phase 2 — deterministic assembler** with structure guarantees; A/B vs single-shot on the
  Caravan Palace track via the eval gate.
- [x] **Phase 3 — flip default** to orchestrated once it wins; fix gate/report canonical artifact
  (ai_improver writes canonical render+comparison; Go post-step separates only, no re-render).
- [ ] **Phase 4 — targeted iteration**: ai_improver re-runs only the job(s) implicated by the gap.
- [x] **Phase 5 — retire** single-shot.

## Verification
- Each phase: `go build`, `pytest`, and for codegen changes an A/B gate comparison on a known track.
- Evidence in `.sisyphus/evidence/` with the real similarity % from a BlackHole render.

## MVP
Phase 0 + 1 behind `--codegen orchestrated`, A/B on Caravan Palace, flip only when gate wins.
