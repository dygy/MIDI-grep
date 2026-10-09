# Phase 0 + 1 — Job-Based Codegen Orchestration — COMPLETED ✓

## Phase 0: shared validation (strudel_validation.py single source of truth)
- Consolidated VALID_SYNTHS/GM/DRUM_BANKS + INVALID_GM_PATTERNS + INVALID_METHODS + validate_code()
  into strudel_validation.py. ollama_agent re-exports them (219 sounds); ollama_codegen routes
  fix_strudel_syntax through shared fix_names; ollama_agent._validate_code delegates to shared validate_code.
- Verified: tr808 -> RolandTR808 auto-corrected; sub_bass rejected; .volume() rejected; valid passes.
- Fixes the tr808 leak on ALL paths (the v005 bug).

## Phase 1: job mechanism
- job_runner.py: JobSpec/JobResult, topological order, per-job validate -> retry(<=2, error fed back)
  -> logged deterministic fallback. Unit-tested: retry recovers, fallback triggers, deps ordered.
- codegen_orchestrator.py: 4 jobs (structure, voice.bass, voice.lead, drums), JSON validated against
  the shared sound library, assembled by a PURE deterministic assembler (setcps from BPM, exactly
  3 $: voices guaranteed by construction). Writes job_run.json for observability.
- Go: --codegen single|orchestrated flag (default single); orchestrator.go selects the script.

## Real end-to-end run (model: midi-grep-strudel-mistral, electro_swing)
job_run.json: all_ok=true, 0 fallbacks. RETRY RECOVERED REAL FAILURES:
  voice.bass ok on attempt 2 (attempt 1 non-JSON); drums ok on attempt 3 (2 non-JSON).
In single-shot, any of those would have corrupted the whole output. Here each was isolated + retried.
Output: valid 3-voice Strudel, real electro-swing instruments (gm_electric_bass_finger, gm_alto_sax,
RolandTR808), no hallucinations, assembled_valid=true.

## Next (Phase 2)
A/B `--codegen orchestrated` vs `single` end-to-end on the Caravan Palace track; compare eval-gate
scores. Flip default only when orchestrated wins.
