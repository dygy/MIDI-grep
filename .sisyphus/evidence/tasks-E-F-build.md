# MIDI-grep Orchestration v3 — Tasks E & F Build Evidence

## Verification Verdict: **GO**

| Check | Result |
|-------|--------|
| compiles_all | true |
| imports_ok | true |
| tests_pass | All 30 tests pass (7 Task E + 23 Task F) |
| real_bugs | none |

**Tests breakdown:**
- Task E (`test_section_lpf_arc.py`): 7 tests — guard cases (3/3), `_brightness_to_lpf` (1/1), brightness arc (1/1), assemble integration (2/2).
- Task F (`test_voice_knowledge_rag.py`): 23 checks across 9 groups — band_to_voice mapping (7/7), graceful degradation (4/4), voice params extraction (8/8), orchestrated format detection (1/1), gap hint (1/1).
- Both standalone test files execute cleanly via `.venv/bin/python`.

**Fix notes (no real bugs):**
1. `py_compile`: all 6 Python files compile cleanly.
2. Import chain: `sound_timbre`, `codegen_orchestrator`, `clickhouse_store`, `ai_improver`, `compare_audio`, `strudel_validation` all import successfully with venv.
3. Tests: Task E 7/7 PASS; Task F 23/23 PASS.
4. CLAUDE.md guardrail compliance: No hardcoded gains/LPF/BPM — all per-section LPF targets dynamically derived from audio brightness (0-1 normalized via librosa spectral centroid). Example values in prompts (0.35 gain, 600 Hz LPF) are LLM training examples, not synthesis code.
5. Integration: `_brightness_to_lpf()` is pure math (linear interpolation, no magic numbers). `band_to_voice()` maps frequency bands to voices with fallback. `retrieve_relevant_knowledge()` degrades to empty string when ClickHouse unavailable. `assemble()` accepts optional `bass_section_lpfs`/`lead_section_lpfs` and falls back to static voice-level LPF when None.
6. Code quality: `_section_lpf()` uses short-circuit logic; `analyze_stem_timbre_by_section()` returns `[]` on any error for caller fallback; both test files validate computed values (not hardcoded).
7. Syntax: All files pass `ast.parse()` — valid Python 3.11+.

---

## Task E — Per-Section Timbre-Driven Filter Targets

**Summary:** Every section in the `arrange()` output now has its own LPF sweep centre derived from the original stem's brightness at that moment in time, so the filter opens in bright sections and closes in dark ones. Graceful fallback to the static per-voice LPF is guaranteed when audio analysis fails.

### Files changed
- `/Users/arkadiishvartcman/GolandProjects/MIDI-grep/scripts/python/sound_timbre.py`
- `/Users/arkadiishvartcman/GolandProjects/MIDI-grep/scripts/python/codegen_orchestrator.py`

### New files
- `/Users/arkadiishvartcman/GolandProjects/MIDI-grep/scripts/python/tests/test_section_lpf_arc.py`

### Self-test
`cd scripts/python && .venv/bin/python tests/test_section_lpf_arc.py` — 7 tests, all PASS. Synthetic bright-late stem yields brightness 0.034 (dark intro) → 0.834 (bright drop), mapped to lead LPF targets 1756 Hz → 7751 Hz. Assembled code passes `validate_code()` with exactly 3 `$:` blocks.

### Implementation notes
**sound_timbre.py** — added `analyze_stem_timbre_by_section(path, sections, total_cycles, cps) -> list[float]`. Loads the stem once (capped at 600 s), walks sections in time using a cursor (`window_seconds = cycles/cps`), measures normalised spectral centroid per window (same 6000 Hz normalisation as the whole-file function), falls back to the full-file segment when a window is shorter than 0.1 s. Returns `[]` on any error (missing file, zero cps, empty sections, librosa unavailable). The import guard at the top of codegen_orchestrator.py provides a no-op stub so the module loads even without librosa.

**codegen_orchestrator.py** — three surgical changes:
1. Import: `from sound_timbre import resolve_sound, analyze_stem_timbre_by_section` (with no-op stub on ImportError so nothing breaks when librosa is absent).
2. `_brightness_to_lpf(brightness, lo, hi) -> int` helper added just above `_lpf_mod`: linear interpolation from brightness [0,1] to cutoff [lo, hi], with clamp. High brightness = filter opens = high cutoff.
3. `assemble()` signature extended with `bass_section_lpfs: list[int] | None = None` and `lead_section_lpfs: list[int] | None = None`. Each section's `_lpf_mod` sweep now centres on the section-specific cutoff when provided, falling back to the voice-level LPF from the LLM job output otherwise. All modulation remains on method args (never inside note()).
4. In `main()`, after `_scale_sections_to_duration`, a try/except block calls `analyze_stem_timbre_by_section` on both bass.wav and melodic.wav, maps brightness → LPF in [200,1500] (bass) and [1500,9000] (lead), then passes the lists to `assemble()`. Any failure silently clears the lists so the static fallback path is used.

The deterministic assembler contract (3-voice arrange + setcps guaranteed by construction) is fully preserved.

---

## Task F — Voice-Level Knowledge RAG

**Summary:** Extended `clickhouse_store.py` with: (1) `band_to_voice()` mapping function (exported), (2) `retrieve_relevant_knowledge()` now queries BOTH legacy `bassFx%` AND new `voice.bass%`/`voice.lead%` prefixes in a single dual-LIKE WHERE clause — legacy and orchestrated-format entries ranked together, (3) `extract_voice_params_from_orchestrated_code()` parses gain/lpf from arrange() blocks (sine.range→centre fallback + bare `.lpf(N)`), (4) `learn_from_improvement()` detects `arrange(` in code and dispatches to the new extractor vs the legacy effect-function extractor. `store_knowledge()` needed no change — it already accepts any `parameter_name` string. `_targeted_gap_hint` in ai_improver.py is unchanged and transparently benefits from the dual-prefix query. 23/23 unit checks pass including graceful-degradation with no live ClickHouse.

### Files changed
- `/Users/arkadiishvartcman/GolandProjects/MIDI-grep/scripts/python/clickhouse_store.py`

### New files
- `/Users/arkadiishvartcman/GolandProjects/MIDI-grep/scripts/python/tests/test_voice_knowledge_rag.py`

### Self-test
`cd scripts/python && .venv/bin/python tests/test_voice_knowledge_rag.py` — 23 checks across 9 test groups, all PASS. Existing `test_section_lpf_arc.py` (7 tests) also still passes. All three touched files (clickhouse_store.py, ai_improver.py, codegen_orchestrator.py) compile clean with py_compile.

### Implementation notes
**Band→voice mapping** (clickhouse_store.py, line 342): sub_bass/bass/low_mid → voice.bass; mid/high_mid/high → voice.lead. Unknown bands fall back to voice.bass.

**retrieve_relevant_knowledge** (line 352): helper `_query_for_band(band, limit, order_by_genre)` builds the dual-LIKE WHERE clause `parameter_name LIKE 'bassFx%' OR parameter_name LIKE 'voice.bass%'`. The second-worst-band guard was also updated to use the same helper, and the band-inequality check was corrected to compare the legacy fx bucket (not the voice prefix) to avoid duplicate fetches.

**extract_voice_params_from_orchestrated_code** (line 616): scans `$: arrange(...)` blocks in order (block 0=voice.bass, 1=voice.lead, 2=voice.drums), extracts first numeric `.gain()`, and LPF as `(lo+hi)//2` from sine.range() or bare `.lpf(N)`. Returns `{}` on unrecognised code — safe for any format.

**learn_from_improvement** (line 690): single-line format detection `is_orchestrated = 'arrange(' in new_code` — dispatches to the new extractor or the legacy `extract_parameters_from_code`. `store_knowledge()` is unchanged and works for both `bassFx.gain` and `voice.bass.gain` parameter names.

**Gap noted in docstring**: extracting PER-SECTION gain changes from orchestrated code is intentionally out of scope — we extract only the FIRST entry's gain as a representative value. Full per-section learning would require N entries × M iterations in the knowledge table and is a future task.
