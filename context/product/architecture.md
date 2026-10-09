# System Architecture Overview: MIDI-grep

- **Version:** 2.0
- **Status:** Approved
- **Last reviewed:** 2026-10-09 (post-AWOS 1.5.0 upgrade; supersedes the Feb 2026 revision)

Governing values: `context/product/values.md` (V1 editability, V2 live-codeable, V3 resemblance
by re-performance, not reproduction). Audio replay of the original is **not** a deliverable;
similarity is measured only on Strudel code the user can edit. This document describes *how* the
system is built; `values.md` decides what counts as a valid output.

---

## 1. Application & Technology Stack

- **Backend Language:** Go 1.25.5 (`go.mod`; single binary, orchestrates everything via subprocess)
- **CLI Framework:** Cobra v1.10.2 (`cmd/midi-grep/main.go` — `extract`, `serve`, `train`, `generative`, `report`)
- **HTTP Router:** Chi v5.2.4 (`internal/server/server.go`)
- **Template Engine:** Go `html/template`, templates embedded from `internal/server/templates/*.html`
- **Frontend Interactivity:** HTMX + SSE progress (no client-side JS framework)
- **CSS Framework:** PicoCSS
- **Audio / ML runtime:** Python 3.11 venv at `scripts/python/.venv`. **Pinned to 3.11** by
  `basic-pitch` → TensorFlow 2.15 + `keras<3` (`scripts/python/requirements.txt`); 3.12-only syntax is
  not allowed.
- **Strudel rendering:** TypeScript/Node — Puppeteer BlackHole recorder only
  (`scripts/node/src/record-strudel-blackhole.ts`, built to `scripts/node/dist/` with `npm run build`).
  The offline Node synthesizer (`render-strudel-node.ts`) has been **deleted**.
- **LLM:** Ollama (default model `midi-grep-strudel-mistral`, built from `Modelfile.mistral`); Claude API
  optional via `--ollama=false`.
- **Learning store:** ClickHouse local binary (`bin/clickhouse`, data in `.clickhouse/db`;
  `scripts/python/clickhouse_store.py`).
- **Agent tooling:** FastMCP `loop` server (`mcp_servers/loop/server.py`), Playwright MCP — both
  registered in `.mcp.json`. AWOS workflow under `.awos/` + `.claude/`.

---

## 2. Audio Processing Pipeline

*Go orchestrates external Python tools via subprocess (`internal/exec/runner.go`, auto-detects the
venv). Each tool runs in isolation with clear input/output contracts. **Go does NOT generate Strudel
code** — all codegen is Python/LLM.*

- **Stem Separation:** Demucs htdemucs (`separate.py` → melodic/other, drums, bass, vocals)
- **Caching:** Stems cached by URL/file hash in `.cache/stems/`, auto-invalidates when `separate.py` changes
- **Audio-to-MIDI Transcription:** Basic Pitch (`transcribe.py`), cleanup/quantization (`cleanup.py`)
- **Drum Detection:** librosa onset detection + spectral classification (`detect_drums.py`)
- **BPM / Key Detection:** librosa (`analyze.py`, candidates reported in output header);
  chords + sections via `smart_analyze.py`
- **Genre Detection:** heuristics in `internal/pipeline/orchestrator.go`
  (`shouldUseBrazilianFunkMode()` etc.), CLAP zero-shot (`detect_genre_dl.py`, default on),
  Essentia (`detect_genre_essentia.py`); `--genre` bypasses detection
- **Codegen (LLM-first):** `--codegen orchestrated` (**default**) runs
  `scripts/python/codegen_orchestrator.py` — small validated jobs (structure / voice.bass /
  voice.lead / drums) executed by `job_runner.py` (validate → retry ≤2 with error feedback → logged
  deterministic fallback), then a pure assembler guarantees the 3-voice `arrange()` + `setcps()`
  contract and writes `job_run.json`. `--codegen single` runs the legacy one-shot
  `ollama_codegen.py`. `strudel_validation.py` is the single source of truth for valid sounds,
  banks, methods and name corrections used by every codegen path.
- **Iteration loop:** `scripts/python/ai_improver.py` (render → compare → LLM feedback via
  `ollama_agent.py` → re-render), gated by `eval/` (see below), learning stored in ClickHouse.

### Pipeline Flow

```
[WAV/MP3/YouTube URL]
         │
         ▼
┌─────────────────┐
│ Cache Check     │ (URL/file hash → .cache/stems/<key>/)
│   ↓ miss        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stem Separation │ (Demucs htdemucs, separate.py)
│   → melodic.wav │
│   → drums.wav   │
│   → bass.wav    │
│   → vocals.wav  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌────────────┐
│ Melodic│  │ Drums      │
│ Path   │  │ Path       │
└───┬────┘  └─────┬──────┘
    │             │
    ▼             ▼
┌─────────────┐  ┌──────────────┐
│ BPM + Key   │  │ Drum Detect  │
│ + Genre     │  │ → bd/sd/hh   │
└──────┬──────┘  └──────┬───────┘
       │                │
       ▼                │
┌─────────────┐         │
│ Audio→MIDI  │         │
│ (Basic Pitch)         │
└──────┬──────┘         │
       │                │
       ▼                │
┌─────────────┐         │
│ MIDI Cleanup│         │
│ - quantize  │         │
└──────┬──────┘         │
       │                │
       └───────┬────────┘
               │
               ▼
      ┌──────────────────────────┐
      │ Python LLM Codegen       │ codegen_orchestrator.py (default)
      │ - job_runner.py jobs     │ or ollama_codegen.py (--codegen single)
      │ - strudel_validation.py  │
      │ - genre sound RAG        │ (sound_selector.py)
      │ → 3-voice arrange()      │
      └────────────┬─────────────┘
                   │
                   ▼
      ┌──────────────────────────┐
      │ BlackHole Recorder       │ record-strudel-blackhole.ts
      │ (real Strudel playback)  │ → render_vNNN.wav
      └────────────┬─────────────┘
                   │
                   ▼
      ┌──────────────────────────┐
      │ compare_audio.py         │ → comparison.json
      │ eval/gate.py floor check │ (per-genre, thresholds.yaml)
      └────────────┬─────────────┘
                   │
                   ▼ (--iterate N, ai_improver.py)
      ┌──────────────────────────┐
      │ LLM feedback → new code  │ ollama_agent.py; auto-reject below
      │ early-stop on gate pass  │ floor once cleared; ClickHouse log
      └──────────────────────────┘
```

### Audio Rendering (single path: BlackHole recorder)

There is exactly **one** render path. The former offline Node.js synthesizer was deleted; the
only script in `scripts/node/src/` is `record-strudel-blackhole.ts`. It records the **real Strudel
engine** playing the generated code — it is not an emulation — so what is scored is what a user
would hear in Strudel. (It is still a recording of a browser, so treat results as a faithful
capture, not a claim of bit-exact "100% accuracy".)

```
Strudel Code (.strudel)
     │
     ▼
┌─────────────────────────────────────┐
│ ffmpeg capture starts               │
│ -use_wallclock_as_timestamps 1      │ (BEFORE -i; Jun 2026 tempo fix)
│ -f avfoundation -i ":BlackHole 2ch" │
│ -af aresample=async=1               │ (restamps/resamples to real time)
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ Puppeteer (headless:false, window   │ record-strudel-blackhole.ts
│ parked at -32000,-32000, 1x1)       │
│ - opens strudel.dygy.app/embed      │ (--local → localhost:4321/embed)
│ - default browser context           │ (not incognito; keeps sample cache)
│ - inserts code via CodeMirror       │ cmView.view.dispatch({changes})
│ - clicks Play                       │
│ - AudioContext.setSinkId(BlackHole) │ AFTER play (superdough must exist)
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ BlackHole 2ch virtual device (macOS)│
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ ffmpeg post-pass                    │
│ silenceremove=start_periods=1:...   │ (trim load-time leading silence)
│ → output.wav                        │
└─────────────────────────────────────┘
```

**Why the ffmpeg flags matter:** avfoundation hands BlackHole's 48 kHz stream to ffmpeg with
device-clock timestamps that do not track real time, so recordings came out ~25% fast (136 BPM
read as ~103). `-use_wallclock_as_timestamps 1` + `-af aresample=async=1` fix this; output `-ar`
does not, and input `-ar` breaks avfoundation. All similarity numbers measured before Jun 2026 were
on sped-up audio and are invalid.

**Constraints:**
- **macOS only** (BlackHole + avfoundation). Setup: `brew install blackhole-2ch` (reboot), create a
  Multi-Output Device (BlackHole + speakers) in Audio MIDI Setup and select it as system output;
  otherwise the recorder silently produces empty audio.
- **Single pass.** The embed offers no programmatic stop/restart (`window.stop()` does nothing,
  `ctx.state` is always `running`), so there is no warm-up → restart; the recorder records once and
  trims leading silence.
- Build: `cd scripts/node && npm run build`. Usage:
  `node scripts/node/dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30`.

**Call sites:** `cmd/midi-grep/main.go` (`renderStrudelBlackHole`, `--blackhole`),
`scripts/python/ai_improver.py` (every `--iterate` round), `scripts/auto-calibrate.sh`,
`scripts/sample-pipeline.sh`, `scripts/python/stem_match.py`, `scripts/python/audition_drum_banks.py`,
`mcp_servers/loop/server.py`. Residual references to `render-strudel-node.js` remain in
`cmd/midi-grep/main.go` (`renderStrudelNodeJS`), `mcp_servers/loop/server.py` (`recorder='node'`),
`scripts/node/package.json` (`npm run render`), `synth_profiles.py` and `ai_learning_optimizer.py`;
they point at a file that no longer exists and are dead paths. **Only the BlackHole recorder may
gate accept/reject decisions.**

### Comparison & Eval Gate

- **`scripts/python/compare_audio.py`** — rendered-vs-original similarity. Uses **MAE** on
  frequency bands, not cosine (cosine hid 20%+ band errors). Weights: Frequency Balance **40%**,
  MFCC 20%, Energy 15%, Brightness 15%, Tempo 5%, Chroma 5%; penalty if any band is >15% off.
  Also emits `section_aware_similarity` (per-window MFCC 40% / bands 35% / energy 25%) and
  per-stem comparison. Writes `comparison.json` + charts.
- **`eval/`** — code-authoritative similarity gate:
  - `eval/thresholds.yaml` — per-genre regression floors (`default: 0.55`; e.g. `brazilian_funk: 0.62`,
    `electro_swing: 0.60`, `russian_hip_hop: 0.52`, `jazz: 0.50`), `target: 0.80` (informational),
    `max_worst_band_diff: 0.30` hard guardrail, and a stricter `section_aware` floor block.
    Floors only ratchet **up**.
  - `eval/gate.py` — `load_thresholds()`, `floor_for_genre()`, `section_aware_floor_for_genre()`,
    `evaluate_comparison()` → `GateResult(passed, genre, similarity, floor, …)`; CLI
    `python eval/gate.py comparison.json --genre <g>`.
  - `eval/datasets/reference_tracks.yaml` — pins known tracks to genres and a BlackHole
    `comparison.json`; currently `tracks: []`. Missing comparisons are skipped, present ones must
    clear the floor (`scripts/python/tests/test_similarity_gate.py`).
  - **Wired into the `--iterate` loop** (`ai_improver.py`): the gate floor for the genre is resolved
    up front; once any iteration clears both the overall and section-aware floors, later below-floor
    renders are **auto-rejected** (reverted to best) and the loop **early-success stops**
    (`--ignore-gate-stop` keeps climbing toward `--target-similarity`). The verdict is written as a
    `gate` block in `iterations.json`. A missing `eval/` or `pyyaml` disables the gate without
    breaking the run.
- **Data-driven mix calibration** — `scripts/python/calibrate_dynamic.py` maps a render's measured
  `comparison.json` to the next knobs of `generate_dynamic_strudel.py` (sub-gain, bass-mult,
  cal-lead, lead-lpf, hat-gain, master-gain), each a damped (sqrt) clamped proportional correction of
  an observed band/centroid ratio. `scripts/auto-calibrate.sh` closes the generate → render
  (BlackHole) → compare → calibrate loop and keeps the best render. No hardcoded per-track values.

### Loop MCP server (`mcp_servers/loop/server.py`)

FastMCP server registered in `.mcp.json` as `loop` (`scripts/python/.venv/bin/python -m
mcp_servers.loop.server`, `PYTHONPATH=.`). Tools:

| Tool | Does |
|------|------|
| `render_strudel(strudel_path, out, duration, recorder)` | Runs the recorder (`recorder='blackhole'` → `record-strudel-blackhole.js`; `'node'` is a dead option) |
| `compare_render(original, rendered, duration)` | Runs `compare_audio.py -j` and returns the comparison dict |
| `verify_strudel(strudel_path, original, …, recorder, genre)` | render → compare → gate; returns the accept/reject verdict |
| `eval_gate(comparison_json, genre)` | Applies `eval/gate.py` to an existing comparison |

Agents must use `recorder='blackhole'` for any accept/reject decision.

### Sample-Pack Mode (hosted sample-instruments)

```
Input (URL or .cache/stems/<key>) → build_sample_pack.py
    → drums/ one-shots (bd/sd/hh/oh), bass/ + melodic/ pitched multi-samples (pyin),
      loops/ raw per-bar slices, strudel.json manifest, pack.json metadata
    → host: localhost (--local) or Cloudflare R2 (--r2 via upload_r2.py)
    → generate_sample_strudel.py → samples.json (absolute _base ending in '/', array-valued
      entries) + output_<mode>.strudel (modes: loops / instrument / hybrid)
    → Strudel: await samples("<base>/samples.json")
```

One command: `scripts/sample-pipeline.sh --url <U>|--stems-dir <D> --prefix <id> [--mode …]
(--local|--r2)`. Thresholds inside `build_sample_pack.py` are percentile-based per track (no
absolute constants). Per `values.md`, replaying the original's per-bar loops is **not** a
deliverable; the sanctioned use is **pitched sample-instruments** (`note(...).s("trackbass")`)
driven by generated/edited `note()` bar arrays — the "dynamic Strudel" path in
`scripts/python/generate_dynamic_strudel.py`, which `auto-calibrate.sh` tunes. R2 hosting needs
`npx wrangler login` or R2 S3 keys (`R2_ACCOUNT_ID/R2_BUCKET/R2_PUBLIC_BASE/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY`).

### Generative Mode (`generative` / `gen` / `rave`)

Stems → timbre embedding (OpenL3/CLAP, `scripts/python/rave/timbre_embeddings.py`) → model search
(threshold 0.88) → reuse or train (granular = minutes, RAVE = hours; `rave/trainer.py`) → repository
+ GitHub sync (`rave/repository.py`) → Strudel `note()` control via `generative serve` on port 5555.
Go wrapper: `internal/generative/pipeline.go`.

### Default Analysis Features (All Enabled)

```
Strudel Code
     │
     ▼
┌─────────────────────────┐
│ BlackHole Recorder      │ (record-strudel-blackhole.ts)
│ → render_vNNN.wav       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Demucs on the render    │ (ai_improver.py batch stem separation)
│ → render_v*_melodic.mp3 │
│ → render_v*_drums.mp3   │
│ → render_v*_bass.mp3    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Comparison              │ (compare_audio.py)
│ → comparison.json       │
│ → chart_stem_*.png      │
│ → section-aware score   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Eval gate + AI loop     │ (--iterate 20 default, --target-similarity 0.99)
│ - gate floor per genre  │ eval/thresholds.yaml
│ - early-stop on pass    │
│ - revert on regression  │
│ → iterations.json       │ (incl. gate block)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ HTML Report             │ generate_report.py or internal/report (midi-grep report)
│ - DAW-style stem player │ (isolated Original / Rendered / Iteration groups)
│ - per-stem charts       │
│ - comparison tables     │
│ - Strudel code          │
└─────────────────────────┘
```

Supporting Python modules: `spectrogram_analyzer.py` (mel-spectrogram gap insights),
`sound_selector.py` (67 drum machines, 128 GM instruments, 17 genre palettes, genre sound RAG),
`synth_profiles.py` (per-genre synth character + sidechain depth applied via
`analyze_synth_params.py --genre`), `thin_patterns.py` (onset-density control),
`audio_to_strudel_params.py` (effect-parameter suggestion), `stem_match.py` + `cross_track_eval.py`
(data-driven stem matching / multi-track evaluation).

---

## 3. Data & Persistence

- **Stem Cache:** `.cache/stems/<key>/` in repository root, keyed by URL (`yt_<VIDEO_ID>`) or file hash
- **Cache Versioning:** `.version` holds the `separate.py` hash; stems regenerate when it changes
- **Output Versioning:** each run creates `v001/`, `v002/`, … with `metadata.json`
- **Learning Store:** ClickHouse local (`bin/clickhouse local --path .clickhouse/db`):
  `midi_grep.runs` (every render attempt with similarity_* and code) and `midi_grep.knowledge`
  (parameter deltas with confidence, keyed by genre/BPM/key). Never cleared — it is learning data.
- **Plan / evidence trail:** `.sisyphus/` (`boulder.json`, `plans/`, `evidence/`, `notepads/`,
  `drafts/`) maintained by the `sisyphus-plan` skill for multi-step runs
- **Eval data:** `eval/datasets/reference_tracks.yaml` (+ `eval/*.json` baselines)
- **Web jobs:** in-process background jobs (`internal/server/jobs.go`); no user accounts or history
- **Workspace:** per-job temp directory (`internal/workspace/workspace.go`), cleaned after use

### Cache Directory Structure

```
.cache/stems/yt_VIDEO_ID/
├── melodic.wav            # Separated melodic stem
├── drums.wav              # Separated drums stem
├── bass.wav               # Separated bass stem
├── vocals.wav             # Separated vocals stem
├── .version               # Cache version (separate.py hash)
├── sample_pack/           # (sample-pack mode) one-shots, pitched, loops, strudel.json
└── v001/                  # Output version
    ├── output.strudel     # Strudel code
    ├── metadata.json      # BPM, key, style, genre, notes, drum hits, timestamp
    ├── render.wav         # BlackHole render of the final code
    ├── render_v*.wav      # Per-iteration renders
    ├── render_v*_{melodic,drums,bass}.mp3  # Per-iteration Demucs stems
    ├── iterations.json    # Iteration manifest (+ gate block)
    ├── comparison.json    # compare_audio.py output
    ├── comparison.png, chart_*.png
    ├── ai_params.json     # AI-suggested mix parameters
    ├── synth_config.json  # Analysis-derived synthesis config (BPM, tempo tolerance)
    ├── job_run.json       # codegen_orchestrator job trace (which jobs retried / fell back)
    └── report.html        # Self-contained HTML report
```

### Output Metadata (`metadata.json`)

```json
{
  "bpm": 136,
  "key": "C# minor",
  "style": "brazilian_funk",
  "genre": "brazilian_funk",
  "notes": 497,
  "drum_hits": 287,
  "version": 1,
  "created_at": "2026-06-03T01:24:00Z"
}
```

---

## 4. Infrastructure & Deployment

**Current reality: MIDI-grep runs on a local macOS workstation. There is no viable container or
cloud deployment path today.**

- **Docker is NOT currently viable.** `Dockerfile` builds with `golang:1.21-alpine` while `go.mod`
  requires Go 1.25.5, and the runtime stage installs no `yt-dlp`, no Node/Puppeteer, and no Demucs
  models; `docker-compose.yml` does not exist (only `docker-compose.clickhouse.yml`). Treat the
  Dockerfile as stale until rewritten.
- **BlackHole rendering is macOS-only** (BlackHole virtual device + ffmpeg avfoundation). Since the
  recorder is the only render path, similarity scoring, the `--iterate` loop, the eval gate and the
  `loop` MCP all require macOS.
- **Local Development Requirements:**
  - Go 1.25.5 (`go build -o bin/midi-grep ./cmd/midi-grep`)
  - Python 3.11 venv at `scripts/python/.venv` (`scripts/install-deps.sh`, `make deps`)
  - Node + `cd scripts/node && npm install && npm run build`
  - `ffmpeg`, `yt-dlp`
  - BlackHole 2ch + a selected Multi-Output Device
  - Ollama running with `midi-grep-strudel-mistral` created from `Modelfile.mistral`
  - Optional: ClickHouse local binary (`bin/clickhouse`), Cloudflare R2 credentials or
    `wrangler login` for sample-pack hosting
- **Environment preflight** before long runs (see `CLAUDE.md` → Working Agreement): venv imports
  (librosa, demucs, basic-pitch), ML models present (~1 GB first download), BlackHole + Multi-Output
  selected, `ollama serve` up with the model pulled.
- **CI:** none. No `.github/workflows/`; tests run locally only.

### Resource Requirements

- **Memory:** reference machine is 24 GB — `mistral-small` (~13 GB) must coexist with Demucs and
  the recorder browser; the 70B `midi-grep-strudel` model needs ~48 GB and is unusable here
- **CPU:** stem separation is 1–2 min per track; multi-core recommended
- **Disk:** `.cache/stems/` grows with every track (stems + per-iteration renders and mp3 stems)

---

## 5. Project Structure

```
midi-grep/
├── cmd/
│   └── midi-grep/
│       └── main.go           # CLI: extract, serve, train, generative, report
│
├── internal/
│   ├── analysis/analysis.go  # BPM + key detection wrapper
│   ├── audio/
│   │   ├── input.go          # File validation, format detection
│   │   ├── stems.go          # Demucs orchestration
│   │   └── youtube.go        # yt-dlp integration
│   ├── cache/cache.go        # Stem + output caching, versioning
│   ├── drums/detector.go     # Drum hit detection/classification
│   ├── errors/errors.go      # Sentinel + ProcessError types
│   ├── exec/runner.go        # Python subprocess runner (venv auto-detect)
│   ├── generative/pipeline.go# Go wrapper for RAVE/granular pipeline
│   ├── midi/
│   │   ├── transcribe.go     # Basic Pitch wrapper
│   │   └── cleanup.go        # Quantization, filtering
│   ├── pipeline/orchestrator.go # End-to-end pipeline; calls Python codegen
│   ├── progress/progress.go  # CLI progress output
│   ├── report/generator.go   # Go HTML report generator (midi-grep report)
│   ├── server/
│   │   ├── server.go         # Chi router, embedded templates/static
│   │   ├── handlers.go       # Request handlers
│   │   ├── jobs.go           # Background job processing + SSE
│   │   ├── templates/        # index/processing/result/error.html (HTMX)
│   │   └── static/
│   └── workspace/workspace.go# Per-job temp directories
│
├── scripts/
│   ├── midi-grep.sh, extract-youtube.sh, extract-file.sh, quick-riff.sh, serve.sh, install-deps.sh
│   ├── sample-pipeline.sh    # URL/stems → sample pack → host (local|R2) → Strudel → render/score
│   ├── auto-calibrate.sh     # generate → render(BlackHole) → compare → calibrate loop
│   ├── node/
│   │   ├── src/record-strudel-blackhole.ts  # THE renderer (Puppeteer + BlackHole + ffmpeg)
│   │   ├── dist/             # Compiled JS (npm run build)
│   │   └── package.json
│   └── python/
│       ├── requirements.txt  # Python 3.11 pins (basic-pitch → TF 2.15, keras<3)
│       ├── separate.py, analyze.py, smart_analyze.py, transcribe.py, cleanup.py, detect_drums.py
│       ├── detect_genre_dl.py, detect_genre_essentia.py
│       ├── codegen_orchestrator.py  # DEFAULT codegen (job-based, validated)
│       ├── job_runner.py            # Job DAG runner (validate → retry → logged fallback)
│       ├── strudel_validation.py    # Single source of truth: valid sounds/banks/methods
│       ├── ollama_codegen.py        # Legacy single-shot codegen (--codegen single)
│       ├── ollama_agent.py          # Iteration-time agentic LLM with per-track memory
│       ├── ai_improver.py           # --iterate loop, eval gate wiring, ClickHouse learning
│       ├── clickhouse_store.py      # ClickHouse runs/knowledge tables
│       ├── compare_audio.py         # MAE similarity + section-aware + per-stem
│       ├── calibrate_dynamic.py     # Data-driven mix calibrator
│       ├── generate_dynamic_strudel.py # Editable note-material Strudel on sample-instruments
│       ├── build_sample_pack.py, generate_sample_strudel.py, upload_r2.py  # Sample-pack mode
│       ├── stem_match.py, cross_track_eval.py, analyze_synth_params.py, synth_profiles.py
│       ├── sound_selector.py, spectrogram_analyzer.py, thin_patterns.py, generate_report.py
│       ├── test_*.py                # Root-level pytest files
│       ├── tests/                   # pytest suite
│       └── rave/                    # RAVE/granular generative pipeline
│
├── eval/
│   ├── gate.py               # Similarity gate (GateResult, CLI)
│   ├── thresholds.yaml       # Per-genre floors, section-aware floors, guardrails
│   ├── datasets/reference_tracks.yaml
│   └── *.json                # Baselines (stem_match_baseline, drum_bank_timbre)
│
├── mcp_servers/
│   └── loop/server.py        # FastMCP: render_strudel, compare_render, verify_strudel, eval_gate
│
├── .sisyphus/                # Durable plans + evidence (boulder.json, plans/, evidence/)
├── .awos/                    # AWOS commands + templates
├── .claude/
│   ├── agents/               # Domain experts + testing-expert
│   ├── commands/awos/        # /awos:* slash commands
│   └── skills/               # sisyphus-plan, self-review, prompt-engineering, …
├── .mcp.json                 # playwright, loop, awos-recruitment MCP servers
│
├── context/
│   ├── product/
│   │   ├── values.md         # Core values: editability, live-codeable, re-performance
│   │   ├── product-definition.md, product-definition-lite.md, roadmap.md
│   │   └── architecture.md   # This document
│   └── spec/
│       ├── 001-core-pipeline/
│       ├── 002-ml-customization/
│       └── 003-editable-strudel-generation/
│
├── CLAUDE.md, llms.txt, llms-full.txt   # Canonical build/run + reference docs
├── Modelfile, Modelfile.mistral          # Ollama custom model definitions
├── Dockerfile                            # STALE — not currently viable (see §4)
├── docker-compose.clickhouse.yml
├── Makefile
└── go.mod                                # go 1.25.5
```

---

## 6. API Design

### CLI Interface

```bash
# Extract from YouTube (uses cache; default: orchestrated codegen, 20 AI iterations)
midi-grep extract --url "https://youtu.be/VIDEO_ID"

# Extract from file
midi-grep extract --input track.wav --output riff.strudel

# Chord mode (for electronic/funk)
midi-grep extract --url "..." --chords

# Force fresh extraction (skip cache)
midi-grep extract --url "..." --no-cache

# Drums only
midi-grep extract --url "..." --drums-only --drum-kit tr808

# Manual genre override / legacy single-prompt codegen / BlackHole render
midi-grep extract --url "..." --genre retro_wave --codegen single --blackhole

# Iteration control
midi-grep extract --url "..." --iterate 5 --target-similarity 0.80 --ignore-gate-stop
midi-grep extract --url "..." --iterate 5 --ollama-model llama3.1:8b   # fast smoke run

# Generative models
midi-grep generative train piano.wav --name my_piano --mode granular
midi-grep generative serve --port 5555

# Regenerate the HTML report for a cached version
midi-grep report .cache/stems/<key> --version 2 -o report.html

# Start web server
midi-grep serve --port 8080
```

### HTTP Endpoints (`internal/server/server.go`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Upload page (HTML) |
| GET | `/health` | Health check |
| POST | `/upload` | Accept audio file, return job ID |
| GET | `/status/{id}` | SSE stream of processing progress |
| GET | `/result/{id}` | Final result (HTML partial) |
| GET | `/download/{id}/midi` | Download cleaned MIDI file |
| GET | `/audio/{id}/{stem}` | Stream a separated stem |
| GET | `/static/*` | Embedded static assets |

### MCP Tools (`loop` server)

`render_strudel`, `compare_render`, `verify_strudel`, `eval_gate` — see §2 "Loop MCP server".

---

## 7. Error Handling Strategy

- **User Errors:** Clear HTML/CLI messages (unsupported format, file too large)
- **Processing Errors:** Python subprocess failures surface as `ProcessError`
  (`internal/errors/errors.go`) with captured stderr; sentinel errors `ErrFileNotFound`,
  `ErrUnsupportedFormat`, `ErrCorruptedFile`, `ErrFileTooLarge`, `ErrTimeout`, `ErrToolNotInstalled`
- **Codegen Errors:** each orchestrated job validates its JSON, retries ≤2 with the error fed back,
  then falls back to a logged deterministic default — never silently. Invalid Strudel (unknown
  sounds/methods) is rejected by `strudel_validation.py`; the iteration loop skips that round
  rather than crashing
- **Render Errors:** missing BlackHole / missing `dist/record-strudel-blackhole.js` are reported
  immediately (preflight), not discovered mid-run
- **Gate Resilience:** a missing `eval/` or `pyyaml` disables the gate and the run continues
- **No pre-existing issues:** any failure met during a run is fixed, not worked around (CLAUDE.md)
- **Timeout:** web jobs (`internal/server/jobs.go`) use `context.WithTimeout` per stage — 5 min
  download, 5 min stem separation, 3 min transcription — and job state is purged 10 min after
  completion; temp workspaces cleaned on exit

---

## 8. Security Considerations

- **File Validation:** Magic-byte format detection (RIFF / MP3 sync / ID3), not just extension
  (`internal/audio/input.go`)
- **Size Limits:** 100MB — enforced both by `http.MaxBytesReader` in `internal/server/handlers.go`
  (`maxUploadSize`) and by `MaxFileSize` in `internal/audio/input.go`
- **Temp Isolation:** Each job gets a unique temp directory
- **No Execution:** Audio files are never executed, only processed by trusted tools
- **Subprocess Hygiene:** Python/Node invoked with explicit argv (no shell interpolation of user input)
- **Secrets:** R2 and API keys come from environment variables only; nothing is committed
- **HTMX Security:** server-rendered templates with auto-escaping

---

## 9. Testing Stack

Declared per layer; the AWOS `testing-expert` agent treats this section as the single source of
truth and blocks if it is missing. **There is no CI** — every layer runs locally on the macOS
workstation.

### Unit

- **Go:** standard library `testing`, run with `go test ./...`. There are currently **zero**
  `*_test.go` files; new Go tests go next to the package under test as `<file>_test.go`.
- **Python:** pytest. Suite lives in `scripts/python/tests/` (plus a few root-level
  `scripts/python/test_*.py`). Run with
  `scripts/python/.venv/bin/python -m pytest -q scripts/python/tests scripts/python/test_*.py`
  (174 tests collected on 2026-10-09). Every new test file carries the annotation tokens
  `# @layer: unit | integration | e2e | contract` and `# @spec: <spec-dir>`; `# @regression` marks
  cases that belong to the permanent regression suite (existing tests predate this convention and
  are to be annotated as they are touched).

### Integration

- **Python against cached stems:** pytest tests that read real stems/comparisons from
  `.cache/stems/<key>/` — no network, no model downloads; skip when the cache key is absent
  (pattern: `eval/datasets/reference_tracks.yaml` + `tests/test_similarity_gate.py`).
- **Go ↔ Python subprocess contracts:** exercised through `internal/exec/runner.go`
  (`RunScript`) — argument/stdout/JSON contracts between the orchestrator and
  `scripts/python/*.py`.

### End-to-End

- **Audio E2E:** render via the BlackHole recorder → `scripts/python/compare_audio.py` →
  `eval/gate.py`, driven through the `loop` MCP (`verify_strudel`). **`recorder='blackhole'` only**;
  the `node` recorder option is dead and must never gate. macOS only.
- **Report E2E:** `midi-grep report <cache-dir>` (Go `internal/report`) or `generate_report.py`
  producing a self-contained `report.html`.

### Browser Automation

- Playwright MCP (`.mcp.json` → `playwright`) against the `serve` web UI (`midi-grep serve --port
  8080`). Screenshots are saved to `docs/screenshots/` (convention; the directory is created on
  first use).

### Contract

- None declared. (Candidates: `comparison.json` and `job_run.json` schemas — `automation_schema.py`
  exists but no contract tests are wired.)
