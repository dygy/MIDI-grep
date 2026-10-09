# Technical Specification: Editable, Live-Codeable Strudel Generation

- **Functional Specification:** [functional-spec.md](./functional-spec.md)
- **Status:** Approved
- **Author(s):** Claude (with user) — written against the repo state audited 2026-10-09 on
  branch `feat/data-driven-stem-matching` (HEAD `dc51d24`).

---

## 1. High-Level Technical Approach

The functional spec turns `context/product/values.md` into a contract: a deliverable is
Strudel whose **notes, patterns and arrangement are data the user edits**; resemblance may
come from stem-derived *sounds* but never from playing the recording back; similarity is
only ever computed on output that already satisfies that contract. Most of the generation
machinery for this already exists — `scripts/python/generate_dynamic_strudel.py` emits
per-voice bar arrays on R2-hosted pitched sample-instruments, and `calibrate_dynamic.py` +
`scripts/auto-calibrate.sh` tune the mix from measured `comparison.json` data. What is
missing is the **enforcement and the honesty plumbing** around it. Four pieces:

1. **Editability contract enforced in code — a static "replay detector".** A new
   `scripts/python/editability_check.py` parses generated Strudel (text/regex level, no
   Strudel runtime) and rejects: `slice(N, run(N)).slow(N)` reconstruction, `loopAt(N)` on a
   hosted stem, and any voice whose sound is a full-stem sample (`s("<stem>full")`,
   `originalfull`, the `samples_orig.json` manifest) — *unless* that voice is explicitly
   flagged as texture (`// texture` marker) **and** the output has at least two editable
   voices (`note(cat(...X))` bar arrays or drum mini-notation on one-shots). A loop-only
   output fails. The detector runs **before** rendering/comparison everywhere a score is
   produced: `ai_improver.py`, `compare_audio.py`, and the `loop` MCP `verify_strudel`.
2. **Two honest generation modes.** `generate_dynamic_strudel.py` gains
   `--mode sample-instrument|synth` (default `sample-instrument`). `sample-instrument` keeps
   today's path (pitched stem multisamples via `await samples(...)`). `synth` voices the same
   bar arrays on library sounds chosen from the genre palette (`sound_selector.py`,
   `synth_profiles.py`) and emits no `samples()` loads. Both write a
   `// generation_mode: <mode>` header the detector and report read. The current `vocalsfull`
   replay voice (the one §2.1 violation in the current best, v023) is replaced by an
   **editable vocal voice**: a pitched vocal sample-instrument driven by a transcribed `vocal`
   bar array, or onset-sliced vocal chops triggered by an editable `s("vox0 ~ vox3 …")`
   pattern; the loop survives only as an opt-in `--vocal-mode texture` layer that the
   detector allows under ≥2 editable voices.
3. **Honest measurement.** `compare_audio.py` and `ai_improver.py` stamp
   `generation_mode` and `editability: "pass"|"fail"` (plus the violation list) into
   `comparison.json`, `metadata.json` and the `gate` block of `iterations.json`. A failing
   detector short-circuits: no `overall_similarity` is written, the iteration is rejected and
   cannot become `best`. `eval/thresholds.yaml` gets **per-mode** floors, set from a measured
   honest render of each mode (not from the stale genre-wide 0.62/0.50), and `eval/gate.py`
   resolves `(genre, mode)` before falling back to `genre`.
4. **Reporting reflects the values.** The Python and Go report generators label the
   headline with the mode and the editability verdict, render a "REPLAY — not a deliverable"
   badge with no percentage when editability failed, and label the per-stem panels as
   "demucs re-separation of the render" (which is what `ai_improver.py` does today).

Systems touched: `scripts/python/` (generator, comparer, improver, report, new checker,
tests), `eval/` (gate + thresholds + dataset), `mcp_servers/loop/server.py`,
`internal/report/generator.go`, a first Go test in `internal/cache/`, and the docs
(`CLAUDE.md`, `README.md`, `llms.txt`, `llms-full.txt`). Nothing in the Go pipeline's
stem-separation/analysis path changes.

**Important honesty note on the current headline.** v023 of Regime CLT
(`.cache/stems/Regime CLT (Dj Brunin XM, Aurora Shukita)/v023/comparison.json`) measures
`overall_similarity 0.9383`, `section_aware_similarity 0.9585`,
`frequency_balance_similarity 0.9668`. Its `output.strudel` plays bass/lead as editable bar
arrays on `regime_bass`/`regime_lead` (lines 12, 93, 182–187) and drums as one-shot patterns
(189–193), **but** line 196 is `s("vocalsfull").slice(vocalBars, run(vocalBars)).slow(vocalBars)`
— a straight replay of the vocal stem. Under this spec v023 is `editability: fail` and its
score may not be presented as the deliverable's quality. The `sample-instrument` floor must
therefore be **re-measured** once the vocal voice is editable (Slice 3 in `tasks.md`); the
functional spec's 91.4% "DJ-flow" figure and CLAUDE.md's 93.6/95.6/96.2 (stale even as
numbers — v023 is 93.8/95.9/96.7) are both pre-contract measurements.

---

## 2. Proposed Solution & Implementation Plan (The "How")

### 2.1 What already exists (verified 2026-10-09, cite before re-building)

| Capability | Where | Status |
|---|---|---|
| Per-voice bar arrays `let bass = [...]`, `let lead = [...]`, one `$:` block per voice, `setcps(cps)` from BPM | `scripts/python/generate_dynamic_strudel.py:11-19` (docstring contract), `:451` (`setcps(cps)` emission); v023 `output.strudel:6-7, 12, 93, 182, 187, 189, 196` | exists |
| Stem-derived **pitched sample-instruments** (pyin multisamples for bass/melodic) + drum one-shots + `samples.json` manifest | `scripts/python/build_sample_pack.py:1-20` (docstring), pack dir `.cache/stems/Regime CLT …/sample_pack/{bass,melodic,drums,samples.json}`; played via `.s("regime_bass")`/`.s("regime_lead")` in v023 | exists |
| Data-driven mix calibration loop | `scripts/python/calibrate_dynamic.py`, `scripts/auto-calibrate.sh`, v023 `calibration_params.json` | exists |
| MAE-weighted similarity (freq .40 / mfcc .20 / energy .15 / brightness .15 / tempo .05 / chroma .05) + `section_aware_similarity` | `scripts/python/compare_audio.py:398-407`, `:461-464` | exists |
| Eval gate: per-genre floors, worst-band guardrail, optional section-aware floor; wired into the iteration loop (auto-reject after floor cleared, early-success stop, `gate` block in `iterations.json`) | `eval/gate.py` (`evaluate_comparison`, `floor_for_genre`, `section_aware_floor_for_genre`), `eval/thresholds.yaml`, `scripts/python/ai_improver.py:60-72, 417-428, 996-1001, 1029-1047`; `scripts/python/tests/test_similarity_gate.py` (12 tests) | exists |
| Loop MCP `verify_strudel` = render → compare → gate | `mcp_servers/loop/server.py:131-150`; registered in `.mcp.json:10-16` | exists |
| Sound-name / method validation for LLM output (`VALID_SOUNDS`, `INVALID_METHODS`, `validate_code`) | `scripts/python/strudel_validation.py:182, 195, 209`; used by `ollama_agent.py:878` (`_validate_code`) | exists for the LLM codegen path only — **not** run over `generate_dynamic_strudel.py` output |
| Hosting prerequisite documented (`npm install`) | only `scripts/sample-pipeline.sh:15` | partial — absent from CLAUDE.md "Environment Preflight" (`CLAUDE.md:15-21`) |

What does **not** exist (grep-verified): any `sample-instrument`/`sample_instrument` token
in code (only in `values.md` and the functional spec); any `--mode sample-instrument|synth`
flag (the only generator `--mode` is `generate_sample_strudel.py:190`,
`loops|instrument|hybrid`, default `loops` = loop-only replay); any `editab*`/`replay`
detection code (the words appear only in generator docstrings); `generation_mode`/
`editability` keys in any `comparison.json`/`metadata.json` (v023 has neither); vocal
transcription (no `vocals.mid`, no vocal chop slicing anywhere under `scripts/`); any
`*_test.go`; a reference-track entry (`eval/datasets/reference_tracks.yaml` → `tracks: []`).

### 2.2 Architecture changes

**A. `scripts/python/editability_check.py` (new).** Pure-Python, no audio deps, importable
from `ai_improver.py`, `compare_audio.py` and the MCP server (which already does
`sys.path.insert(0, REPO_ROOT)` for `eval.gate`).

- `@dataclass EditabilityResult`: `passed: bool`, `generation_mode: str | None`
  (`"sample-instrument" | "synth" | "loops" | None`), `editable_voices: list[VoiceInfo]`,
  `texture_voices: list[VoiceInfo]`, `violations: list[str]`, `summary: str`.
- `check_editability(code: str, *, mode_hint: str | None = None) -> EditabilityResult`.
- `to_json_fields(res) -> dict` → `{"editability": "pass"|"fail", "generation_mode": …,
  "editability_violations": [...], "editable_voice_count": n, "texture_voice_count": n}`.
- CLI: `editability_check.py <file.strudel> [--json]`, exit 0 pass / 1 fail / 2 parse error.
- **Rules** (regexes, comments stripped first; identifiers or literals for `N`):
  - R1 *reconstruction-by-playback*: `\.slice\(\s*(\w+)\s*,\s*run\(\s*\1\s*\)\s*\)\s*\.slow\(\s*\1\s*\)`
    and `\.loopAt\(` → the enclosing voice is a replay voice.
  - R2 *full-stem sound*: `s\("(originalfull|[a-z]+full|origseg\d+)"\)` or a `samples(...)`
    load of `samples_orig.json` / `samples_segs.json` / `samples_stems.json` → replay voice.
  - R3 *editable voice*: a `$:` block (or `stack(...)` member) that contains `note(` fed by
    a bar array (`cat(...name)` with `let name = [` present) or a mini-notation string, **or**
    `s("…")` whose tokens are one-shot names (not matching R2) → counts toward
    `editable_voices`. Every editable voice must be in its own `$:` block or a named member of
    a `stack(...)`, so it can be commented out (§2.2 Mute Test).
  - R4 *texture allowance*: a replay voice is tolerated only when its line carries a
    `// texture` marker **and** `len(editable_voices) >= 2`; otherwise it is a violation.
    `generate_dynamic_strudel.py --vocal-mode texture` is the only producer of that marker.
  - R5 *loop-only*: `len(editable_voices) == 0` → fail regardless of markers.
  - R6 *mode*: `// generation_mode:` header wins; else infer `sample-instrument` when a
    `samples(` load exists and `.s("<id>_bass"|"<id>_lead"|…)` reference it, else `synth`.
- Expected verdicts on existing artifacts (used as test fixtures, copied into
  `scripts/python/tests/fixtures/editability/`): v012 `output.strudel` → fail (R1+R2,
  loop-only); v023 `output.strudel` → fail (R1+R2 on line 196, no marker);
  `sample_pack/output_loops.strudel` → fail; a v023 copy with line 196 removed → pass,
  mode `sample-instrument`.

**B. `generate_dynamic_strudel.py` changes.**

- `--mode {sample-instrument,synth}` (default `sample-instrument`). In `synth` mode:
  skip both `await samples(...)` lines (currently emitted around `:452-458`), map
  `--bass-sound/--lead-sound` defaults to palette picks from
  `sound_selector.retrieve_genre_context(genre)` / `synth_profiles.py` (no hardcoded sound
  names — CLAUDE.md ZERO HARDCODING), keep bar arrays, drums on `.bank(<genre drum machine>)`
  (`--drum-mode bank` path already exists at `:291`).
- Emit `// generation_mode: <mode>` and `// editability: checked-by editability_check.py`
  in the header (after the existing `// genre=` line) so the detector/report never guess.
- Replace `--vocal-loop/--no-vocal-loop` (`:288-290`) with
  `--vocal-mode {instrument,chops,texture,none}` (default `instrument`; `texture`
  reproduces today's loop **with** the `// texture` marker; `none` drops the voice).
  - `instrument`: needs `vocals.mid` (new: run `transcribe.py` on `vocals.wav`, fold to the
    detected vocal range with the existing `fold_pitch`) and a pitched vocal multisample
    (`build_sample_pack.py` gains a `vocals/` pitched set, same pyin path as `bass/`,
    `melodic/`; manifest key `<prefix>_vocal`). Emits `let vocal = [...]` +
    `$: note(cat(...vocal)).s("regime_vocal")…`.
  - `chops`: onset-sliced one-shots `vox0..voxN` (new helper in `build_sample_pack.py`,
    edge-faded like `write_continuous_loop`) + an editable `let vox = ["vox0 ~ ~ vox3 …"]`
    pattern derived from vocal onsets quantised to the grid.
- The generated vocal gain envelope (`bar_env_pattern`, `:145`) is reused unchanged.

**C. `generate_sample_strudel.py` / `scripts/sample-pipeline.sh`.** Add `sample-instrument`
as an alias of `instrument` in `--mode` (`:190`) and make it the default; keep `loops`
only with a banner `// generation_mode: loops — texture/diagnostic, NOT a deliverable`
that the detector fails. `sample-pipeline.sh` `MODE="loops"` default → `instrument`.

**D. Honest-measurement plumbing.**

- `compare_audio.py`: new optional `--strudel PATH`. When given, run
  `check_editability` first; on fail write `{"editability": "fail", "generation_mode": …,
  "editability_violations": [...], "comparison": null}` to `-o`/stdout and exit 3 — no
  `overall_similarity` is produced. On pass, merge `to_json_fields()` into the top level of
  `results` next to `comparison`/`original`/`rendered` (`compare_audio.py:1042`, `:1781`
  are the write sites). Metric math is untouched.
- `ai_improver.py`: before the BlackHole render (`:579`), call the detector on the candidate
  code. On fail: log, append an iteration entry with `editability: "fail"`, skip render +
  compare, do not update `best_similarity`, and continue (mirrors the existing
  `last_validation_error` skip). Pass `--strudel` to both `compare_audio.py` invocations
  (`:626`, `:1110`). Stamp `generation_mode` + `editability` into the `metadata.json` dict
  (`:1156-1168`) and into `gate_summary` (`:1029-1047`). Read the mode from the Strudel
  header; the Go orchestrator does not need a new flag for this spec.
- `mcp_servers/loop/server.py`: `verify_strudel` runs the detector before `render_strudel`
  and returns `{"ok": False, "stage": "editability", …}` on fail; `compare_render` gains
  `strudel_path: str | None` and passes `--strudel`; `eval_gate`/`compare_render` accept
  `mode` and pass it to the gate. **Cleanup:** drop `NODE_SYNTH`
  (`server.py:28` → `scripts/node/dist/render-strudel-node.js`, which does not exist — the
  source was deleted in commit `f18f5cc`, `scripts/node/src/` holds only
  `record-strudel-blackhole.ts`) and the `recorder: Literal["blackhole","node"]` option;
  also fix `scripts/node/package.json` `"render"` script that points at the same file.
  Keep the "renderer not built" error, and add a "`scripts/node/node_modules` missing — run
  `npm install`" preflight error (it is missing on this machine, so the recorder cannot run).

**E. Per-mode floors.**

- `eval/thresholds.yaml`: add
  ```yaml
  modes:
    sample_instrument:
      genres: { brazilian_funk: <measured − margin> }
      section_aware: { brazilian_funk: <measured − margin> }
      measured: { brazilian_funk: { overall: <x>, section_aware: <y>, run: "<cache_key>/vNNN" } }
    synth:
      genres: { brazilian_funk: <measured − margin> }
      ...
  ```
  Floors are **regression floors** (the file header already says so): set each to the
  first honest, detector-passing render of that mode minus a small noise margin, and record
  the raw measured value + the run next to it so the number is reproducible (§2.4). The
  genre-wide `brazilian_funk: 0.62 / 0.50` stays as the fallback for modes without a
  measurement; it is never raised by guesswork.
- `eval/gate.py`: `floor_for_genre(genre, thresholds, mode=None)` and
  `section_aware_floor_for_genre(…, mode=None)` look up `modes.<mode>` first;
  `evaluate_comparison(..., mode=None)` reads `generation_mode` from the comparison JSON when
  `mode` is not passed, and fails with reason `editability: fail` when the JSON carries it.
  `GateResult` gains `mode: str | None` and `editability: str | None`.
- `eval/datasets/reference_tracks.yaml`: add the Regime CLT entry (`cache_key`, `version`,
  `genre`, `mode`) once a detector-passing render exists, so
  `test_similarity_gate.py`'s dataset layer finally runs on a real track.

**F. Reporting.**

- `scripts/python/generate_report.py:95-104` and `internal/report/generator.go:349-356`:
  headline becomes `"<pct>% — Overall Similarity · mode: <generation_mode> · editable: pass"`.
  When `editability == "fail"` (or the key is absent on a run produced after this change),
  render a red "REPLAY / UNVERIFIED — not a deliverable" badge instead of the percentage.
  Go side: add `GenerationMode string \`json:"generation_mode"\`` and
  `Editability string \`json:"editability"\`` to the struct that wraps `ComparisonData`
  (`generator.go:67-75`).
- Per-stem section (`stem_comparison.json`, produced by `ai_improver.py:1078/1139` calling
  `separate.py --mode full` on the *render*): add the caption "stems obtained by demucs
  re-separation of the rendered mix — lossy, not a true stem-match view" (§2.5).

**G. Docs / defaults / tests hygiene (same spec, small tasks).**

- `CLAUDE.md:73`: replace 93.6/95.6/96.2 with the real v023 values 93.8/95.9/96.7 **and**
  state that v023's vocal voice was a stem replay, so the number predates the editability
  contract; replace again after the re-measure. `CLAUDE.md:15-21` preflight: add
  "`scripts/node/node_modules` present (`cd scripts/node && npm install`)" and
  "`dist/record-strudel-blackhole.js` built".
- `README.md:391-394` vs `cmd/midi-grep/main.go:323-326`: README says `--iterate 5`,
  `--target-similarity 0.85`, `--ollama-model llama3:8b`; the binary defaults are `20`,
  `0.99`, `midi-grep-strudel-mistral`. Fix README to match the binary.
- First Go test: `internal/cache/cache_test.go` for `KeyForURL` (`cache.go:168`),
  `ExtractVideoID` (`:498`) and `KeyForFile` generic-name fallback (`:184`). Pure functions,
  stdlib `testing` only.
- `llms.txt` / `llms-full.txt`: document `editability_check.py`, the `--mode` flag,
  `--vocal-mode`, the new JSON keys and per-mode floors (CLAUDE.md "Context Document
  Maintenance").

### 2.3 Data shape changes

`comparison.json` (top level, beside `comparison`, `original`, `rendered`, `insights`,
`duration_analyzed`):

```json
"generation_mode": "sample-instrument",
"editability": "pass",
"editability_violations": [],
"editable_voice_count": 4,
"texture_voice_count": 0
```

`metadata.json` (per version): same `generation_mode` + `editability` keys added to the dict
written at `ai_improver.py:1156-1168`. `iterations.json`: each iteration entry gains
`editability`; `gate` gains `mode` and `floor_source: "modes.<mode>" | "genres"`. Strudel
header: `// generation_mode: <mode>` line. No ClickHouse schema change is required for this
spec (store `generation_mode` inside the existing `parameters` JSON column if the loop
writes runs).

---

## 3. Impact and Risk Analysis

- **System dependencies.** The detector sits in front of `compare_audio.py`,
  `ai_improver.py` and the `loop` MCP — any generator (`generate_dynamic_strudel.py`,
  `generate_hybrid_strudel.py`, `generate_sample_strudel.py`, the Ollama codegen path) that
  emits `loopAt`/`slice(N,run(N))`/`*full` sounds will now be rejected before scoring.
  `generate_hybrid_strudel.py:259` emits `loopAt(nbars)` for uploaded voices and will fail
  unless it is updated to mark those voices as texture (or dropped). `test_similarity_gate.py`
  must keep passing with the new `mode` parameter defaulting to `None`.
- **Risk: vocal voice fidelity.** Replacing the `vocalsfull` replay with a transcribed
  vocal on a pitched multisample (or onset chops) will almost certainly *lower* the
  measured score (the vocal was ~the whole recognizable hook). That is the honest outcome
  the spec demands (V4). Mitigations: ship `--vocal-mode instrument` and `chops` and pick
  per genre by measurement; keep `texture` as an explicit, detector-visible layer under ≥2
  editable voices; re-run `auto-calibrate.sh` after the change so `cal-vocal`/`cal-lead`
  re-balance. The R2-hosted `vocalsfull.wav` is a stale 16-bar loop (session handoff memo);
  a new vocal instrument pack needs an R2 upload (see next risk).
- **Risk: R2 dependency.** `sample-instrument` mode plays from
  `https://pub-….r2.dev/midi-grep/regime-clt/…`; the embed fetches it at render time, so
  no network / stale manifest = silent voice = a bogus low score. Uploads need R2 auth
  (`R2_BUCKET`/`R2_PUBLIC_BASE` env or `wrangler login` with R2 scope — currently not
  configured here per the handoff memo). Mitigations: `sample-pipeline.sh --local`
  (localhost `generative serve`/`:5555`) is the proof path for verification tasks;
  `editability_check` is network-free; the per-mode floor task records the hosting base
  used in `measured.run`.
- **Risk: floor ratchet discouraging iteration.** A floor set at the measured honest value
  turns every exploratory iteration into an auto-reject once cleared, and makes the
  early-success stop (`ai_improver.py:996-1001`) trigger only at a very high bar.
  Mitigations: floors are set at measured − margin (stated in the YAML), apply only to the
  *shipped best* and to post-clear regressions (existing semantics), `--ignore-gate-stop`
  stays available, and the per-mode block is only populated from a real run — a genre/mode
  without a measurement falls back to the conservative genre floor.
- **Risk: macOS-only recorder.** BlackHole + avfoundation + the `-use_wallclock_as_timestamps`
  tempo fix are macOS-specific; the recorder also needs `scripts/node/node_modules`
  (missing on this machine) and a Multi-Output device. Everything in this spec except the
  per-mode floor measurement and the final E2E slice is verifiable without a render
  (detector, JSON stamping, gate logic, report rendering on existing `comparison.json`,
  Go test). The render-dependent tasks name the preflight explicitly.
- **Risk: regex detector false positives/negatives.** A user-authored `slice(…)` on a bar
  array (e.g. `bass.slice(0,4)`, which §2.1 *encourages*) must not trip R1 — the rule
  requires the `run(N)…slow(N)` shape. Tests include that positive case. A determined
  generator could still hide replay behind indirection; the detector is a contract check,
  not a security boundary, and the `editable_voice_count` is surfaced in the report so a
  human can see "1 editable voice + 1 texture" at a glance.
- **Risk: stale documentation drift.** Three places already disagree (CLAUDE.md numbers,
  README defaults, loop MCP node path). Each is a one-line task with a grep-able check.

---

## 4. Testing Strategy

Declared stack: Python `pytest` in `scripts/python/tests/` (run with
`scripts/python/.venv/bin/python -m pytest scripts/python/tests -q`; `yaml` + `pytest` import
OK in that venv), Go `go test ./...` (stdlib `testing`), E2E = BlackHole render →
`compare_audio.py` → `eval/gate.py` through the `loop` MCP `verify_strudel`
(`recorder='blackhole'`).

- **Unit (pytest, no audio, no network).**
  - `test_editability_check.py`: fixtures copied from v012 (`fail`, loop-only), v023
    (`fail`, R1+R2 on the vocal line), v023-minus-vocal (`pass`, `sample-instrument`),
    `output_loops.strudel` (`fail`), a hand-written `synth` sample (`pass`, `synth`), a
    texture case with 2 editable voices + `// texture` (`pass`, `texture_voice_count 1`),
    the same with 1 editable voice (`fail`), and `bass.slice(0,4)` (must not trip R1).
    Positive + negative pair for every rule; `@spec: 003-editable-strudel-generation`.
  - `test_similarity_gate.py` extensions: `floor_for_genre(..., mode=…)` prefers
    `modes.<mode>`, falls back to `genres`; `evaluate_comparison` fails when the JSON says
    `editability: fail`; existing 12 tests unchanged.
  - `test_compare_audio_stamping.py`: `compare_audio.py --strudel` on a failing fixture
    exits 3 and writes no `overall_similarity`; on a passing fixture the five new keys are
    present (use tiny synthetic WAVs via `soundfile`, 2 s, so the metric runs quickly).
  - Generator tests: `generate_dynamic_strudel.py --mode synth` output contains no
    `samples(` and passes the detector; `--mode sample-instrument --vocal-mode none|chops|
    instrument` all pass; `--vocal-mode texture` carries the marker and passes only with ≥2
    editable voices. These run against the Regime CLT `sample_pack/` MIDI + stems when
    present and are `skip`ped otherwise (same pattern as the dataset gate).
  - Report tests: `generate_report.py` headline contains `mode:` and `editable:`; a
    comparison with `editability: fail` renders the badge and **no** percentage.
- **Go unit.** `internal/cache/cache_test.go` (`KeyForURL`, `ExtractVideoID`, `KeyForFile`
  generic-name fallback) — the repo's first `_test.go`; `internal/report` headline label test
  on a fixture `comparison.json` with and without the new keys.
- **Integration (python, no render).** Run the detector + stamping over the existing v023
  artifacts into a scratch copy: `eval/gate.py <scratch>/comparison.json --genre
  brazilian_funk --mode sample-instrument` must FAIL with `editability: fail` (vocal replay),
  proving the current headline is correctly disqualified; `loop` MCP `eval_gate` on the same
  file returns `gate_passed: false`.
- **E2E (BlackHole, macOS, needs `npm install` + Multi-Output device).** Generate both modes
  for Regime CLT with the editable vocal, render via `verify_strudel(recorder='blackhole')`,
  confirm `editability: pass`, record the two measured scores into
  `eval/thresholds.yaml modes.*.measured` and the reference dataset, then re-run the pytest
  gate. Also the §2.1 Editability Test: change one note in `bass[0]`, re-render 8 bars, and
  assert the two renders differ in the edited bar window (RMS of the difference signal above
  the noise floor) while matching elsewhere — this is the behavioural proof that the output
  is data, not tape.
- **Regression.** The final "Feature Testing & Regression" slice in `tasks.md` writes
  acceptance tests across these layers with RED validation and `@spec`/`@regression`
  annotations; verification artifacts of the earlier slices are deleted, the regression
  slice's are kept.
