# Evidence: Tasks D (Sound Resolver) & C (Section-Aware Scoring)

Plan: `002-orchestration-v2-audio-driven.md`
Date: 2026-06-19
Verdict: **GO** (both tasks marked complete)

---

## Task D — Sound Resolver (retrieve → re-rank by timbre)

### What it implements
A timbre-based re-ranking layer that replaces name-only sound selection. Genre-RAG
returns candidate sound names; Task D re-ranks those candidates against the original
stem's actual timbre and pins the acoustically closest sound in the LLM prompt.

- `sound_timbre.py`:
  - Static curated **TIMBRE_TABLE** (65 sounds drawn from `GENRE_PALETTES`).
  - `analyze_stem_timbre()` — librosa-based, extracts three axes:
    - **brightness** = normalised spectral centroid (0–1)
    - **warmth** = low/high energy ratio
    - **attack** = onset sharpness
  - `resolve_sound()` — picks the euclidean-nearest candidate to the stem timbre.
- Wired into `codegen_orchestrator.py`: before voice jobs run, bass and lead candidates
  from genre-RAG are re-ranked against the original stems (`bass.wav` → bass voice,
  `melodic.wav` → lead voice). The resolved sound is injected as a MUST-use rule in
  `_voice_prompt` only when available. The assembler, `setcps`, and 3-voice contract are
  unchanged. Validation still enforces `VALID_SOUNDS` via the existing
  `_voice_validate_factory`.
- Wiring is fully guarded with `try/except` so codegen degrades cleanly if librosa or
  `sound_timbre` are unavailable.

### Files
- New: `scripts/python/sound_timbre.py`
- New: `scripts/python/tests/test_sound_timbre.py`
- Changed: `scripts/python/codegen_orchestrator.py`

### Self-test
`python tests/test_sound_timbre.py` — **6/6 passed**: valid candidate returned;
bright vs warm stems pick different candidates (`gm_lead_5_charang` vs `sine`);
missing-file fallback returns default dict; all 65 palette sounds covered in
TIMBRE_TABLE (0 using default); single-candidate round-trip; empty-candidate fallback.

---

## Task C — Section-Aware Scoring

### What it implements
Adds `section_aware_similarity` to the comparison output and an optional eval-gate check
so the gate rewards matching the song's evolution rather than its average.

- `compare_audio.py`: `section_aware_similarity` computed via **10s sliding windows**
  (fixed windows, since musical sections aren't available in single-file mode). Same
  scoring formula as `compare_windowed`: **MFCC 40%, bands 35%, energy 25%**. Falls back
  to the overall score on short audio or errors. In per-stem mode, `compare_stems()`
  collects a weighted `section_aware_similarity` from the per-stem `compare_audio()`
  calls; true musical sections via `compare_by_sections()` are available in per-stem mode.
- `eval/gate.py`: optionally checks `section_aware_similarity` against per-genre or global
  floors. **Double-gated** — both the comparison dict AND `thresholds.yaml` must carry the
  section_aware fields; if either is absent the check is silently skipped (full backward
  compatibility).
- `eval/thresholds.yaml`: conservative section-aware floors, set 10–15% below overall
  floors (the simpler per-window formula scores lower than the full weighted
  `compare_audio()` formula — no tempo/chroma inflation).
- `test_similarity_gate.py`: 5 new section-aware tests.

### Files
- Changed: `scripts/python/compare_audio.py`
- Changed: `eval/gate.py`
- Changed: `eval/thresholds.yaml`
- Changed: `scripts/python/tests/test_similarity_gate.py`

### Self-test
`.venv/bin/python -m pytest scripts/python/tests/ -v` — **22 passed, 1 skipped**
(dataset gate skip expected: no `reference_tracks.yaml`). New gate tests cover:
pass-when-above-floor, fail-when-below-floor, absent-behaves-as-before (backward compat),
unknown-genre-uses-global-floor, aggregate-dict-pickup.

---

## Verification verdict

```json
{
  "compiles_all": true,
  "imports_ok": true,
  "tests_pass": "22 passed, 1 skipped (in test_similarity_gate.py - expected dataset skip)",
  "verdict": "GO"
}
```

### Real bugs found
- **Dead code** in `compare_audio.py` line 460: `_energy_w` assignment is immediately
  overwritten by line 461, making line 460 useless. Harmless (line 461 is correct) but
  indicates a copy-paste error. Can be removed in a follow-up cleanup commit. **Zero
  functional impact.**

### Reviewer notes
All 22 tests pass. The sound timbre resolver (bright vs warm stems pick different sounds
correctly) and the section-aware gate (field properly populated, backward-compatible when
absent) both work as designed. Python 3.11+ syntax is clean, all imports resolve, no
hardcoding violations (the timbre table is a curated acoustic knowledge base, not
track-specific gains), and the integrations are properly guarded with `try/except` for
graceful degradation.
