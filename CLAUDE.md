# CLAUDE.md - Project Context for Claude Code

This file provides context for Claude Code when working on this project.

## AWOS baseline

This repo runs the AWOS spec-driven workflow at **npm `@provectusinc/awos@1.5.0`** (migration
version 3, upgraded 2026-10-09). Update with `npx @provectusinc/awos@latest --overwrite` from the
repo root — the wrappers in `.claude/commands/awos/` are stock indirection, not customizations, so
overwriting is correct; never hand-edit `.awos/`. Layout:

- `.awos/` — framework internals (commands, templates, scripts). Overwritten on update.
- `.claude/commands/awos/` — `/awos:product|architecture|hire|spec|tech|tasks|implement|verify`
  wrappers. `/awos:roadmap` is retired upstream; `context/product/roadmap.md` is ours to keep by hand.
- `.claude/commands/{implement-feature,fix-bug,session-init}.md` — the project's delivery flow,
  generated from the (now upstream-removed) `/awos:flow` templates. Decisions live in
  `context/product/delivery-flow.md`; edit these files directly to change the flow.
- `.claude/agents/` — hired specialist roster (`context/product/hired-agents.md` is the coverage
  record); `.claude/skills/` — project skills they bind to.
- `.claude/hooks/` + `.claude/settings.json` — `branch-current.sh` (blocking: no PR/branch from a
  stale base), `docs-freshness.sh` (advisory), SessionStart pointer to `/session-init`.
- `.mcp.json` — `loop` (render/compare/eval gate), `playwright`, `awos-recruitment`.
- `context/product/` — product-definition, values, architecture, delivery-flow, hired-agents;
  `context/spec/NNN-<slug>/` — functional-spec, technical-considerations, tasks (+ flow-log).

Reference install at the same level: the Citation orchestrator repo
(`~/PycharmProjects/proj-citation-audit-context`).

## Working Agreement (governance)

These rules govern *how* to work, independent of the audio domain.

### Clarifying Questions Budget
Ask at most ONE round of clarifying questions before taking action. If the request is ambiguous, pick the most likely interpretation, state your assumption, and proceed. Do not use `AskUserQuestion` for routine git operations, stash decisions, or small implementation choices.

### Handling Local Changes
When git operations are blocked by local changes, do NOT prompt for each option. Default: stash with a descriptive name, perform the operation, then inform the user. Only ask if the changes look like intentional uncommitted work that may be lost.

### Environment Preflight
Before any long-running run (extraction, `--iterate`, rendering, comparison) verify the environment FIRST so it doesn't fail 15 minutes in:
- Python venv resolves (`scripts/python/.venv`) and key deps import (librosa, demucs, basic-pitch)
- Node recorder built: `scripts/node/node_modules` present (`cd scripts/node && npm install`) and
  `scripts/node/dist/record-strudel-blackhole.js` exists (`npm run build`) — without them every render fails
- ML models are present (first run downloads ~1GB)
- For BlackHole recording: the BlackHole device exists and a Multi-Output Device is selected (`node dist/record-strudel-blackhole.js` will silently produce empty audio otherwise)
- For Ollama runs: `ollama serve` is up and the model is pulled
Surface a missing dependency immediately rather than letting the run fail partway through.

### Self-Review After Edits
After code or doc changes, run a self-review pass before declaring done. Verify: (1) claims in commit messages/docs match the actual diff, (2) no over-engineering beyond the ask, (3) similarity/eval numbers cited are from an actual run, not assumed. For substantive diffs, the `/self-review` skill launches a 4-agent audit.

### No Pre-Existing Issues
NO ISSUES ARE PRE-EXISTING. If you encounter ANY issue during development/testing — broken script, failing test, wrong similarity metric — it must be fixed, not worked around.

### Delegate to Domain Experts
Specialist standards live in the project agents under `.claude/agents/` (AWOS 1.5 no longer bundles domain experts; `/awos:hire` manages the roster). Delegate, don't reinvent:
- Go implementation → `golang-expert`
- Python (analysis, codegen, comparison) → `python-expert`
- Audio synthesis / DSP → `audio-dsp-expert`
- librosa / spectral / stem ML → `ml-audio-expert`
- Strudel pattern generation → `strudel-expert`
- Ollama / Claude / prompt work → `llm-expert` (and the `/prompt-engineering` skill)
- Music theory (keys, chords, arrangement) → `music-theory-expert`

### Context Document Maintenance
After meaningful changes to the pipeline (`internal/`, `scripts/python/`, `scripts/node/`), update the docs so they stay accurate:
1. **`llms.txt`** — concise (~100-line) project overview. Update when pipelines, modes, or key directories change.
2. **`llms-full.txt`** — comprehensive reference. Update with detailed changes: new scripts, flags, synthesis params, file paths.
3. **`CLAUDE.md`** — this file, for build/run instructions and architecture-level guidance.
4. **`context/product/architecture.md`** — when the stack, a pipeline stage, the render path or the
   testing stack changes (the `testing-expert` agent reads its Testing Stack section).
5. **The owning spec** under `context/spec/` — tick acceptance criteria only via `/awos:verify`;
   a behavior change that contradicts a spec is a *divergence* and amends the spec (`/fix-bug`).
Update triggers: new modes/genres, synthesis-parameter changes, new scripts, renderer changes, or similarity-metric changes.
The `docs-freshness` hook reminds you once per session when a pipeline file changes.

## CRITICAL PRINCIPLES - ZERO HARDCODING

**NEVER hardcode values. The AI must learn and generate everything.**

1. **No hardcoded gains** - AI analyzes frequency bands and generates gain values
2. **No hardcoded filters** - AI determines hpf/lpf based on spectral analysis
3. **No hardcoded effects** - AI learns when to use crush, room, delay, etc.
4. **No magic numbers** - Every parameter must come from analysis or AI decision

**Why:** Hardcoding for one track doesn't help any other track. The system must work for ANY audio input by learning and adapting, not by being tuned to specific test files.

**How it works:**
1. **AI Audio Analysis** (`analyze_synth_params.py`):
   - Analyzes original melodic stem for transients, spectral envelope, harmonics
   - Extracts BPM, waveform suggestions, filter cutoffs, gain ratios
   - Generates JSON synthesis config with per-voice parameters
2. **Dynamic Synthesis** (Node.js renderer):
   - Reads AI-generated config for envelope, filters, waveform per voice
   - Uses saw waveform for mid-heavy content (harmonics fill spectrum)
   - Adjusts master HPF based on original's bass content
3. **Comparison & Iteration**:
   - Compare rendered audio to melodic stem
   - Measure frequency bands, energy, brightness
   - AI analyzes differences and generates new parameters
   - Store learnings in ClickHouse for future tracks

**Current achievement:** editable dynamic-Strudel (transcribed notes on trained instruments, all
voices present incl. the real vocal) at **93.8% overall / 95.9% section-aware / 96.7% freq balance,
tempo_sim 1.000** on Regime CLT (brazilian_funk; `v023/comparison.json`, 2026-06-30), driven by the data-driven `calibrate_dynamic.py`
loop (no hardcoded mix values). CAVEAT (Oct 2026 audit): v023's vocal voice is a full-stem replay
(`s("vocalsfull")…slow(N)`), which violates `values.md` A1 — so this is NOT yet a contract-passing
editable score; spec 003 Slice 3 makes the vocal editable and re-measures the floor. NOTE: numbers measured BEFORE the Jun-2026 recorder tempo fix (the
72% / 88.7% / 92.4% / 94.6% history) were on ~25%-sped-up audio and are invalid — see the recorder
fix below. Earlier honest baselines were ~60-70% (the old 90%+ was inflated by a cosine bug).
**Target:** 80%+ similarity across all genres through AI learning, not hardcoding

**Data-driven mix calibration (Jun 2026):** `scripts/python/calibrate_dynamic.py` +
`scripts/auto-calibrate.sh` close the generate→render→compare→calibrate loop that was previously
hand-tuned. The calibrator maps a render's measured `comparison.json` to the generator's tuning
knobs (sub-gain/bass-mult/cal-lead/lead-lpf/hat-gain/master-gain), each a damped (sqrt) clamped
proportional correction of an observed band/centroid ratio. Two non-obvious lessons baked in:
(1) drive the brightness lever off the spectral CENTROID, not the high bands — at ~1% magnitude
those bands are swamped by demucs bleed/noise and pinned lead-lpf while the mix was clearly dull;
(2) when lead-lpf maxes out and the mix is still dark, the missing brightness is in the DRUMS —
raise a steady TR808 hat layer (`--hat-gain`), which lifted brightness 70%→91% and overall
89.8%→92.6% in one step (the extracted lead sample is inherently darker than the original).

**CRITICAL: Similarity Calculation Fix (Feb 2026)**
The old cosine-based frequency balance was HIDING massive errors (25% sub_bass, 20% mid differences showed as 95%!).
Now uses MAE (Mean Absolute Error) with per-band penalty:
- Weights: Freq Balance 40%, MFCC 20%, Energy 15%, Brightness 15%, Tempo/Chroma 5% each
- Penalty if any band is off by >15%
- Real scores: Electro Swing 67%, Russian Hip-Hop 59%

**Key implementation details:**
- Original audio (full mix) is used for AI synthesis analysis (proper frequency balance)
- `analyze_synth_params.py` extracts transients, spectrum, harmonics, tempo
- `synth_config.json` stores AI-derived BPM and tempo tolerance for comparison
- Node.js renderer accepts `--config` flag for dynamic synthesis parameters
- Saw waveforms essential for mid-band content (sine only produces fundamental)
- Per-voice gain scaling from original frequency band analysis

## Project Overview

**MIDI-grep** is a Go CLI and web application that extracts musical content from audio files or YouTube videos and generates Strudel code for live coding.

### Core Pipelines

**Standard Mode** (note transcription - best for piano/melodic instruments):
```
Input (WAV/MP3/YouTube URL)
    ↓
Stem Separation (Demucs) → melodic/bass/drums/vocals stems
    ↓
Analysis (librosa) → BPM, Key detection
    ↓
Transcription (Basic Pitch) → MIDI notes
    ↓
Cleanup → Quantization, filtering
    ↓
Output → Strudel code (bar arrays + effect functions)
```

**Chord Mode** (chord detection - best for electronic/non-piano music):
```
Input (WAV/MP3/YouTube URL)
    ↓
Stem Separation (Demucs) → melodic stem
    ↓
Smart Analysis (librosa) → Tempo, Key, Chord progression, Sections
    ↓
Output → Strudel code (chord patterns + bass + drums)
```

**Brazilian Funk Mode** (auto-detected or `--brazilian-funk` - for funk carioca/phonk):
```
Input (WAV/MP3/YouTube URL)
    ↓
Stem Separation (Demucs) → for BPM/key detection only
    ↓
Analysis (librosa) → Tempo, Key
    ↓
Auto-Detection → BPM 125-155 + vocal-range notes + short durations + low bass
    ↓
Template Generation → Tamborzão drums + 808 bass + synth stabs
```

**Generative Mode** (RAVE neural synthesizers - for full creative control):
```
Input → Separated Stems
    ↓
Timbre Analysis (OpenL3/CLAP) → Embedding vector per stem
    ↓
Model Search → Find similar existing models (threshold: 88%)
    ↓
If no match: Train New Model
  - Granular (fast, minutes): Onset-based grain extraction
  - RAVE (quality, hours): Full neural network training
    ↓
GitHub Sync → Upload/download models for reuse
    ↓
Strudel Generation → note() control with trained models
```

This mode trains neural synthesizers that learn the "sound" of your track material,
enabling full note() control - edit any pitch, create new melodies, all sounding
like the original. Models are stored in a repository and reused across tracks.

**Sample-Pack Mode** (R2/localhost-hosted real-stem samples - HIGHEST similarity):
```
Input (URL) → Stems (Demucs)
    ↓
build_sample_pack.py → drum one-shots + pitched bass/melodic + RAW per-bar loops
                       (drums/bass/melodic/vocals) + strudel.json
    ↓
Host pack: localhost (proof) or Cloudflare R2 (upload_r2.py → public r2.dev/custom domain)
    ↓
generate_sample_strudel.py → samples.json (_base = host) + output_<mode>.strudel
    ↓
Strudel: await samples("<base>/samples.json"); plays the REAL audio → ~91% similar
```
Instead of synthesizing (≤72% ceiling), this reuses the original's actual audio as
Strudel samples, so the output is genuinely "quite similar". One command:
`scripts/sample-pipeline.sh --url <U> --prefix <id> --local` (or `--r2`).
**Two non-obvious rules:** (1) loops must be written RAW/un-normalized — per-bar
normalization flattens dynamics + inter-stem balance (reconstruction 99%→70%);
one-shots/pitched stay normalized. (2) Strudel's `samples(jsonUrl)` resolves
`_base` by raw `base+path` concat with NO trailing slash, and only accepts
array-valued entries — so the generated `samples.json` bakes an absolute `_base`
ending in `/` and emits every value (incl. one-shots) as single-element arrays.
R2 hosting needs `npx wrangler login` (or R2 S3 keys via env) — see
`scripts/python/upload_r2.py`.

### Caching

All outputs are cached in `.cache/stems/{key}/` by URL or file hash:

```
.cache/stems/yt_VIDEO_ID/
├── melodic.wav            # Separated melodic stem (instruments)
├── drums.wav              # Separated drums stem
├── bass.wav               # Separated bass stem
├── vocals.wav             # Separated vocals stem
├── .version               # Cache version (script hash)
└── v001/                  # Version directory
    ├── output.strudel     # Strudel code
    ├── metadata.json      # BPM, key, style, notes, etc.
    ├── render.wav         # Rendered audio preview
    ├── render_v*.wav      # Per-iteration render files
    ├── render_v*_melodic.mp3  # Per-iteration melodic stem
    ├── render_v*_drums.mp3    # Per-iteration drums stem
    ├── render_v*_bass.mp3     # Per-iteration bass stem
    ├── iterations.json    # Iteration manifest with stem paths
    ├── comparison.json    # Audio comparison data
    ├── comparison.png     # Combined comparison chart
    ├── chart_*.png        # Individual analysis charts
    ├── ai_params.json     # AI-suggested mix parameters
    ├── synth_config.json  # AI-derived synthesis config (BPM, tempo tolerance)
    └── report.html        # Self-contained HTML report
```

- **Stem cache**: Auto-invalidates when `separate.py` changes
- **Output versioning**: Each run creates new version (v001, v002, ...)
- **Metadata stored**: BPM, key, style, genre, notes, drum hits, timestamp

### Audio Rendering & AI Analysis

The `--render` flag synthesizes WAV audio from patterns:

```bash
./bin/midi-grep extract --url "..." --render auto  # Save to cache
./bin/midi-grep extract --url "..." --render out.wav  # Custom path
```

**Synthesis (`scripts/python/render_audio.py`):**
- Kick: Pitch envelope + distortion (808 style)
- Snare: Body tone + high-passed noise
- Hi-hat: Filtered noise with decay
- Bass: Sawtooth + sub-octave, LPF
- Vocal chops: Square wave with fast attack
- Chord stabs: Filtered sawtooth
- Lead: Triangle wave with vibrato

**Node.js Strudel Renderer — REMOVED (Jun 2026):** `render-strudel-node.ts` (offline synthesis
emulating Strudel sounds, ~16% similarity vs the real engine) was deleted. The BlackHole recorder
below is the only render path. Dead references to `render-strudel-node.js` still exist in
`cmd/midi-grep/main.go`, `mcp_servers/loop/server.py` (recorder='node'), `synth_profiles.py` and
`ai_learning_optimizer.py` — tracked in spec 003 tasks.

**Puppeteer BlackHole Recorder (`scripts/node/src/record-strudel-blackhole.ts`):** *(RECOMMENDED)*
- Records REAL Strudel playback using BlackHole virtual audio device
- Opens `https://strudel.dygy.app/embed` in browser, loads code, plays, records via ffmpeg
- **Runs invisibly** - browser hidden offscreen, doesn't steal focus
- **Setup required:**
  1. Install BlackHole: `brew install blackhole-2ch` (requires reboot)
  2. Create Multi-Output Device in Audio MIDI Setup (BlackHole + speakers)
  3. Set system audio output to Multi-Output Device
- **Usage:**
  ```bash
  node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30
  ```
- Records the real Strudel engine (not an emulation); timing is exact only with the Jun 2026 ffmpeg
  wallclock/aresample fix below
- Uses self-hosted Strudel at `strudel.dygy.app`

**Key implementation details:**
- `--autoplay-policy=no-user-gesture-required` bypasses gesture requirement
- Use default context (NOT incognito) - preserves sample cache
- **DO NOT use settings UI** - it doesn't actually route audio
- **DO use `setSinkId()` directly on AudioContext** - this works!
- `headless: false` required (Web Audio quirks in headless mode)
- **Code insertion:** Use `cmContent.cmView.view.dispatch({changes: {...}})` not textContent
- **setSinkId timing:** Must be AFTER clicking play (after superdough initializes)
- **TEMPO FIX (Jun 2026, CRITICAL):** avfoundation captures BlackHole's 48kHz stream with
  device-clock timestamps that don't track real time, so the recording came out ~25% FAST (136 BPM
  read as ~103). The ffmpeg capture MUST use `-use_wallclock_as_timestamps 1` (before `-i`) +
  `-af aresample=async=1` (before output) to restamp/resample to real time. Strudel itself is fine
  (AudioContext clock is real-time). All similarity numbers recorded before this fix were on
  sped-up audio. Output `-ar` does NOT fix it; input `-ar` before `-i` breaks avfoundation.
- **No programmatic stop/restart in the embed:** `window.stop()` doesn't stop, the play button
  loses its text after starting, and `ctx.state` is always 'running' — so a warm-up→restart pass
  is impossible. The recorder is single-pass (record, trim leading silence to anchor ~bar-0).

**Hidden window configuration:**
- Position: `--window-position=-32000,-32000` (far offscreen)
- Size: `--window-size=1,1` (minimal)
- AppleScript hides Chromium process visibility
- Background flags: `--disable-background-timer-throttling`, `--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`

**Key files:** `scripts/node/src/record-strudel-blackhole.ts`

**AI Parameter Suggestion (`scripts/python/audio_to_strudel_params.py`):**
- Analyzes original audio spectral/dynamic characteristics
- Suggests optimal Strudel effect parameters (filters, compression, reverb)
- Feeds back into renderer for AI-driven mix balance

**AI Code Generator (`scripts/python/ai_code_generator.py`):**
- Analyzes original audio for ALL characteristics (spectrum, dynamics, timing, timbre)
- Generates Strudel code with parameters inherently matched to target
- No hardcoded values - everything derived from analysis
- Works universally for any track/genre
- Outputs AudioProfile with spectral bands, dynamics, and timing info

**Pattern Thinner (`scripts/python/thin_patterns.py`):**
- AI-driven pattern density control
- Thins drum patterns to match original's onset density
- Prevents tempo detection errors from too many drum hits
- Parses and modifies Strudel bar arrays

**Model-based Renderer (`scripts/python/render_with_models.py`):** *(deprecated - Node.js is primary)*
- Audio rendering using trained granular models
- Loads pitched samples from model directories
- Fallback when Node.js renderer unavailable

**Editability / Replay Detector (`scripts/python/editability_check.py`, spec 003 Slice 1):**
- Static analysis of a `.strudel` file against `values.md`: R1 `slice(N,run(N)).slow(N)`/`loopAt`
  reconstruction, R2 full-stem or per-bar-loop sounds (`originalfull`, `vocalsfull`, `drumsloop`…),
  R3 ≥1 editable voice, R4 texture allowance (`// texture` marker + ≥2 editable voices), R5 loop-only,
  R6 `generation_mode` header/inference. Exit 0 pass / 1 fail / 2 parse error; `--json` for tooling.
- Current state: v012 and v023 FAIL (v023 only on the `vocalsfull` line); v023 minus that voice PASSES as
  `sample-instrument`. Slice 2 wires it ahead of compare/gate so replay is never scored.

**Audio Comparison (`scripts/python/compare_audio.py`):**
- Compares rendered output vs original stems
- **CRITICAL: Uses MAE for frequency balance, NOT cosine similarity!**
  - Cosine was hiding 20%+ band differences (showed 95% when sub_bass was -25% off!)
  - MAE properly penalizes per-band differences
  - Penalty if ANY band is >15% off
- **Similarity Weights:**
  - Frequency Balance: **40%** (most important - if bands are off, audio sounds wrong)
  - MFCC (timbre): 20%
  - Energy: 15%
  - Brightness: 15%
  - Tempo: 5% (usually matches)
  - Chroma: 5% (often inflated)
- Tracks per-band differences in `band_differences` and `worst_band_diff`
- Generates 6 individual chart images + combined comparison chart
- Saves comparison.json for HTML report data
- Accepts `--config` for AI-derived tempo tolerance from synth_config.json

**Mel Spectrogram Analyzer (`scripts/python/spectrogram_analyzer.py`):**
- Deep mel spectrogram analysis for AI learning
- Compares original vs rendered spectrograms to identify:
  - Which frequency bands differ at which times
  - Envelope/amplitude differences over time
  - Harmonic content differences
  - Transient/attack differences
- Generates actionable insights for LLM prompts (dB differences → gain multipliers)

**AI Code Improver (`scripts/python/ai_code_improver.py`):**
- Gap analysis between original and rendered audio
- Identifies specific frequency band deficiencies
- Modifies Strudel code to address gaps
- Conservative 50% correction per iteration to avoid over-correction

**AI Iterative Codegen (`scripts/python/ai_iterative_codegen.py`):**
- Iteration loop with automatic revert-on-regression
- Tracks best similarity across iterations
- Reverts to best code if similarity drops

**Sound Selector (`scripts/python/sound_selector.py`):**
- Complete Strudel sound catalog:
  - **67 drum machines** from tidal-drum-machines (Roland, Linn, Akai, Boss, Korg, etc.)
  - **128 General MIDI instruments** (gm_* prefix)
  - 5 basic waveforms + ZZFX synths
- **17 genre palettes** (brazilian_funk, electro_swing, house, jpop, trance, lofi, synthwave, etc.)
- Sound alternation patterns using `<sound1 sound2>` syntax
- Timbre-based selection (brightness, warmth, attack time, harmonic richness)
- **Genre-Aware Sound RAG** (`retrieve_genre_context(genre)`):
  - Returns ~40-token compact string of genre-appropriate sounds for LLM prompts
  - Replaces sending the full 800-token catalog to the LLM (760 tokens saved per call)
  - Falls back to "default" palette for unknown genres
  - Injected into `ollama_codegen.py` `build_prompt()` and `ollama_agent.py` `generate()`/`add_iteration_result()`
  - Example: `Available sounds for brazilian_funk (aggressive, punchy, 808-heavy) — Bass: sawtooth, gm_synth_bass_1 | Lead: gm_lead_2_sawtooth, supersaw | ...`

**HTML Report (`scripts/python/generate_report.py`):**
- Self-contained single-file HTML report with embedded audio and charts
- **DAW-Style Audio Studio Player** with ISOLATED stem groups:
  - **Original Stems Section**: melodic, drums, bass, vocals
    - Play/Stop button for entire section
    - Individual mute (M) buttons per stem
    - Waveform visualizations
  - **Rendered Stems Section**: render-melodic, render-drums, render-bass
    - Completely isolated from Original (NEVER play together)
    - Same controls: Play/Stop + per-stem mute
  - **Iteration History Section**: per-iteration stem tracks (melodic, drums, bass)
    - Sub-headers with version number and similarity score (color-coded)
    - Mute buttons per iteration stem for isolated A/B comparison
    - Falls back to single full-mix track when stems unavailable
    - Playback group isolation: each iteration plays independently
  - Web Audio API for synchronized playback
  - Volume faders per stem
  - Synchronized playback controls
  - A/B comparison mode (toggle between original and rendered)
  - **Shimmer skeleton loading**: animated placeholder while audio decodes, removed on waveform draw
- **Per-stem comparison charts** (bass, drums, melodic)
- Visual comparison charts (spectrograms, chromagrams, frequency bands, similarity)
- HTML-based data tables (copyable text, not images)
- Strudel code block with copy button
- Dark theme styled like Playwright/Jupyter reports

**Go Report Generator (`internal/report/generator.go`):**
- Type-safe Go implementation that can replace Python report generator
- Embeds audio files as base64 for self-contained HTML
- Parses comparison.json and ai_params.json for data tables
- Same feature set as Python version with better type safety

## Tech Stack

- **Language**: Go 1.25+
- **CLI Framework**: Cobra
- **Web Framework**: Chi + HTMX + Go templates
- **Audio Processing**: Python scripts (demucs, basic-pitch, librosa)
- **Audio Rendering**: TypeScript/Node.js (node-web-audio-api for offline synthesis)
- **YouTube Download**: yt-dlp

## Project Structure

```
midi-grep/
├── cmd/midi-grep/main.go       # CLI entrypoint (extract, serve commands)
├── internal/
│   ├── audio/
│   │   ├── input.go            # File validation, format detection
│   │   ├── stems.go            # Demucs stem separation wrapper
│   │   └── youtube.go          # yt-dlp integration
│   ├── analysis/analysis.go    # BPM & key detection via librosa
│   ├── midi/
│   │   ├── transcribe.go       # Basic Pitch wrapper
│   │   └── cleanup.go          # Quantization, velocity filtering
│   ├── pipeline/orchestrator.go # End-to-end CLI pipeline (calls Python AI for code gen)
│   ├── server/
│   │   ├── server.go           # HTTP server setup
│   │   ├── handlers.go         # Request handlers
│   │   ├── jobs.go             # Background job processing
│   │   └── templates/          # HTMX templates
│   ├── exec/runner.go          # Python subprocess execution
│   ├── progress/progress.go    # CLI progress output
│   ├── workspace/workspace.go  # Temp file management
│   ├── errors/errors.go        # Custom error types
│   ├── cache/cache.go          # Stem + output caching with versioning
│   ├── drums/detector.go       # Drum pattern detection
│   └── report/generator.go     # Go HTML report generation (replaces Python)
├── scripts/
│   ├── midi-grep.sh            # Main CLI wrapper script
│   ├── extract-youtube.sh      # Quick YouTube extraction
│   ├── node/                   # TypeScript audio rendering
│   │   ├── src/
│   │   │   └── record-strudel-blackhole.ts  # Puppeteer + BlackHole recorder (only render path)
│   │   ├── dist/               # Compiled JavaScript output
│   │   ├── package.json        # Node.js dependencies
│   │   └── tsconfig.json       # TypeScript configuration
│   └── python/
│       ├── separate.py         # Demucs stem separation (melodic/bass/drums/vocals)
│       ├── analyze.py          # BPM/key detection with candidates
│       ├── detect_drums.py     # Drum onset detection and classification
│       ├── render_audio.py     # WAV audio synthesis from patterns
│       ├── smart_analyze.py    # Advanced chord/section detection
│       ├── chord_to_strudel.py # Chord-based Strudel generation
│       ├── transcribe.py       # Basic Pitch transcription
│       ├── cleanup.py          # MIDI quantization
│       ├── detect_genre_dl.py  # CLAP-based deep learning genre detection
│       ├── detect_genre_essentia.py # Essentia-based genre detection
│       ├── audio_to_strudel_params.py # AI-driven effect parameter suggestion
│       ├── compare_audio.py    # Rendered vs original audio comparison
│       ├── generate_report.py  # HTML report generation
│       ├── ai_code_generator.py # AI-driven Strudel code generation
│       ├── ai_code_improver.py # Gap analysis and Strudel code modification
│       ├── ai_iterative_codegen.py # Iteration loop with revert-on-regression
│       ├── ollama_codegen.py   # Ollama LLM code gen with genre sound RAG
│       ├── ai_learning_optimizer.py # AI learning optimization
│       ├── spectrogram_analyzer.py # Mel spectrogram deep analysis for AI
│       ├── sound_selector.py   # Complete sound catalog (67 drums, 128 GM)
│       ├── synth_profiles.py   # Per-genre synthesis profiles + sidechain depth/instruction
│       ├── thin_patterns.py    # Pattern density control
│       ├── render_with_models.py # Render using trained granular models
│       ├── iterative_render.py # AI-driven iterative audio refinement
│       ├── learn_artist.py     # Artist-specific learning
│       ├── seed_knowledge.py   # Knowledge base seeding
│       ├── iterative_optimizer.py # Iterative optimization loop
│       ├── requirements.txt
│       └── rave/               # RAVE generative model system
│           ├── __init__.py
│           ├── cli.py          # CLI wrapper for pipeline
│           ├── pipeline.py     # End-to-end generative pipeline
│           ├── trainer.py      # RAVE + Granular model training
│           ├── repository.py   # Model storage + GitHub sync
│           └── timbre_embeddings.py # OpenL3/CLAP timbre analysis
├── internal/
│   ├── generative/pipeline.go  # Go wrapper for RAVE pipeline
│   └── cache/cache.go          # Stem caching (by URL/file hash)
├── context/                    # AWOS product documentation
│   ├── product/
│   │   ├── product-definition.md
│   │   ├── values.md           # editability contract (replay forbidden)
│   │   ├── roadmap.md          # hand-maintained (AWOS retired /awos:roadmap)
│   │   ├── architecture.md
│   │   ├── delivery-flow.md    # decisions behind /implement-feature and /fix-bug
│   │   └── hired-agents.md     # specialist roster + hooks + gaps
│   └── spec/
│       ├── 001-core-pipeline/
│       ├── 002-ml-customization/
│       └── 003-editable-strudel-generation/   # functional + technical + tasks
├── eval/                       # similarity gate: gate.py, thresholds.yaml, datasets/
├── mcp_servers/loop/           # FastMCP: render_strudel, compare_render, verify_strudel, eval_gate
├── .sisyphus/                  # plan + evidence trail for multi-iteration runs
├── Makefile
├── Dockerfile
└── go.mod
```

## Key Patterns

### Go/Python Subprocess Integration

Go orchestrates Python scripts via `internal/exec/runner.go`:

```go
runner := exec.NewRunner("", scriptsDir)
result, err := runner.RunScript(ctx, "separate.py", inputPath, outputDir)
```

The runner auto-detects the Python venv at `scripts/python/.venv`.

### Error Handling

Custom errors in `internal/errors/errors.go`:
- `ErrFileNotFound`, `ErrUnsupportedFormat`, `ErrTimeout`
- `ProcessError` for Python subprocess failures

### Workspace Management

Each job gets an isolated temp directory via `internal/workspace/workspace.go`:
```go
ws, _ := workspace.Create()
defer ws.Cleanup()
// ws.PianoStem(), ws.RawMIDI(), etc.
```

### Web Interface

- HTMX for reactivity (no JavaScript frameworks)
- SSE for real-time progress updates
- Go templates with PicoCSS styling

## Common Tasks

### Adding a new processing stage

1. Create Python script in `scripts/python/`
2. Add Go wrapper in `internal/<domain>/`
3. Update pipeline in `internal/pipeline/orchestrator.go`
4. Update progress stages in `internal/progress/progress.go`

### LLM-First Code Generation Architecture

**IMPORTANT: Go does NOT generate Strudel code.** All Strudel code is generated by Python AI (`ai_code_generator.py`).

The architecture is:
1. **Go Pipeline** (`internal/pipeline/orchestrator.go`): Stem separation, BPM/key analysis
2. **Python AI Generator** (`scripts/python/ai_code_generator.py`): Generates ALL Strudel code
3. **BlackHole Recorder** (`scripts/node/src/record-strudel-blackhole.ts`): Renders for verification

**Why LLM-First?**
- Hardcoded Go patterns don't generalize across genres
- LLM can adapt to any audio characteristics
- ClickHouse stores learning for incremental improvement
- BlackHole produces 100% accurate audio (not emulation)

**`scripts/python/ai_code_generator.py`** - Main AI code generator:
- Analyzes audio for spectral, dynamic, and timing characteristics
- Generates Strudel code with AI-derived parameters
- Queries ClickHouse for best previous runs
- Supports all genres via `sound_selector.py` integration
- Arguments: `--bpm`, `--key`, `--style`, `--genre`, `--drum-kit`, `--drums-only`

**`scripts/python/ollama_codegen.py`** - Ollama-powered Strudel code generator:
- Replaces rule-based `ai_code_generator.py` with LLM-based generation
- `build_prompt()` includes genre-aware sound RAG context (~40 tokens vs 800)
- Example sounds in the prompt use genre-appropriate palette (not hardcoded)
- Enforces 3-voice structure (bass, lead, drums) via `enforce_three_voices()`
- Comprehensive `fix_strudel_syntax()` auto-corrects LLM hallucinations

**Go Pipeline Role** (no code generation):
- Stem separation via Demucs
- BPM/key detection via librosa
- Genre auto-detection (Brazilian funk, phonk, synthwave, etc.)
- Style detection (passed as hints to AI generator)
- Calls Python AI generator for all Strudel code
- Manages caching and output versioning

### Detection Candidates

All detection algorithms show top candidates in the output header for transparency:

```javascript
// Key candidates: C# minor (80%), G# minor (67%), E major (61%), B major (58%), C# major (47%)
// BPM candidates: 136 (100%), 68 (70%)
// Time sig candidates: 4/4 (100%), 2/4 (100%), 6/8 (51%)
// Style candidates: trance (105%), electronic (95%), house (90%)
```

This helps users understand the analysis confidence and pick alternatives if needed.

### Output Format

The default output uses **bar arrays with effect functions**:

```javascript
// Bar arrays - one string per bar
let bass = ["c2 ~ e2 ~", "f2 ~ g2 ~", ...]
let mid = ["c4 e4 g4", "d4 f4 a4", ...]
let high = ["c5 ~ e5", "g5 ~ b5", ...]
let drums = ["bd ~ sd ~", "bd sd ~ hh", ...]

// Effect functions (applied at playback)
let bassFx = p => p.sound("supersaw").lpf(800).room(0.1)
let midFx = p => p.sound("gm_pad_poly").lpf(4000).room(0.2)
let highFx = p => p.sound("gm_lead_5_charang").delay(0.15)
let drumsFx = p => p.bank("RolandTR808").room(0.15)

// Play all
$: stack(
  bassFx(cat(...bass.map(b => note(b)))),
  midFx(cat(...mid.map(b => note(b)))),
  highFx(cat(...high.map(b => note(b)))),
  drumsFx(cat(...drums.map(b => s(b))))
)

// Mix & match bars:
// $: bassFx(note(bass[0]))
// $: cat(...bass.slice(0,4).map(b => note(b)))
```

This format allows users to:
- Pick individual bars: `bass[3]`
- Slice ranges: `bass.slice(0,4)`
- Mix voices freely in the stack
- Modify effects without touching patterns

### Style-specific effects

Each style has unique effect settings:
- **piano**: Minimal effects, natural envelope, clip=1.0
- **synth**: Phaser, vibrato, saw LFO, ADSR, FM synthesis, echo, superimpose, off, tremolo, filter envelope, jux (high voice)
- **orchestral**: Long attack envelope, vibrato, more reverb, clip=1.5 (sustained), tremolo, superimpose, off, sometimes
- **electronic**: Phaser, distort, saw LFO, ADSR, FM synthesis, echo, superimpose, off, tremolo, filter envelope, sidechain ducking, iter, ply (bass), clip=0.8 (punchy)
- **jazz**: Perlin LFO (organic), vibrato, swing, off (harmonic), sometimes/rarely
- **lofi**: Bitcrush, coarse, perlin LFO, degradeBy, swing, echo, superimpose, iter, sometimes/rarely, clip=1.1

### Adding CLI flags

Edit `cmd/midi-grep/main.go`:
- Add flag in `init()`
- Use in `runExtract()` or `runServe()`

## Build & Run

```bash
# Build
go build -o bin/midi-grep ./cmd/midi-grep

# Standard mode (note transcription - best for piano)
./bin/midi-grep extract --url "https://youtu.be/..."

# Chord mode (best for electronic/funk/EDM - detects chord progression)
./bin/midi-grep extract --url "https://youtu.be/..." --chords

# Force fresh extraction (skip cache)
./bin/midi-grep extract --url "https://youtu.be/..." --no-cache

# Drums only
./bin/midi-grep extract --url "https://youtu.be/..." --drums-only

# Custom style
./bin/midi-grep extract --url "https://youtu.be/..." --style house

# Web server
./bin/midi-grep serve --port 8080
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--chords` | Use chord-based generation (better for electronic/non-piano) |
| `--no-cache` | Skip stem cache, force fresh extraction |
| `--drums` | Include drum patterns (default: on) |
| `--drums-only` | Extract only drums |
| `--style` | Sound style (auto, piano, synth, electronic, house, etc.) |
| `--quantize` | Quantization (4, 8, 16) |
| `--simplify` | Simplify notes (default: on) |
| `--drum-kit` | Drum kit (tr808, tr909, linn, acoustic, lofi) |
| `--render` | Render audio to WAV (default: `auto`, use `none` to disable). **Always outputs stems** |
| `--brazilian-funk` | Force Brazilian funk mode (auto-detected normally) |
| `--genre` | Manual genre override (`brazilian_funk`, `brazilian_phonk`, `retro_wave`, `synthwave`, `trance`, `house`, `lofi`, `jazz`) |
| `--deep-genre` | Use deep learning (CLAP) for genre detection (default: enabled, skipped when `--genre` is specified) |
| `--iterate N` | AI-driven improvement iterations (default: 20) |
| `--target-similarity` | Target similarity for --iterate (0.0-1.0, default: 0.85) |
| `--ignore-gate-stop` | Keep iterating toward `--target-similarity` even after the eval-gate floor is cleared (disables the early-success stop; auto-reject + reporting stay on) |

### Default Analysis Features (Always Enabled)

The following analysis features are **always enabled by default**:

1. **Stem Rendering**: Renders 3 separate stems (`render_bass.wav`, `render_drums.wav`, `render_melodic.wav`)
2. **Per-Stem Comparison**: Generates per-stem comparison charts (`chart_stem_bass.png`, `chart_stem_drums.png`, `chart_stem_melodic.png`)
3. **Overall Comparison**: Generates combined comparison chart and `comparison.json`
4. **AI-Driven Improvement**: 20 iterations by default with 99% target (ensures ALL iterations run)
5. **Iteration Stem Separation**: Batch Demucs on each iteration render → per-iteration melodic/drums/bass stems
6. **HTML Report**: Self-contained DAW-style player with isolated Original/Rendered/Iteration stem groups and shimmer loading

### AI-Driven Iterative Improvement

The `--iterate` flag enables AI-driven code improvement using Claude:

```bash
# Run 5 iterations, target 70% similarity
./bin/midi-grep extract --url "..." --iterate 5

# Higher target similarity
./bin/midi-grep extract --url "..." --iterate 10 --target-similarity 0.80
```

**How it works:**
1. Extract and render initial Strudel code
2. Compare rendered audio with original (frequency bands, MFCC, chroma)
3. Send comparison results to LLM (Ollama local or Claude API)
4. LLM analyzes gaps and generates improved code
5. **Eval gate** (`eval/thresholds.yaml`): each render is scored against an absolute, genre-aware
   similarity floor. Once any iteration clears the floor, later below-floor renders are
   auto-rejected (reverted to best), and the loop **early-success stops** as soon as the floor is
   cleared (no further iterations spent climbing toward the higher `--target-similarity`; pass
   `--ignore-gate-stop` to keep climbing). The
   final result is reported PASSED/FAILED vs the floor and recorded in `iterations.json` (`gate`
   block) + the `improve_strudel` return dict. The gate is resilient — a missing `eval/`/pyyaml
   disables it without breaking the run.
6. Repeat until target similarity or max iterations reached
7. Batch stem separation: run Demucs on each iteration render to produce per-iteration stems
8. Store all runs in ClickHouse for incremental learning
9. Generate HTML report with per-iteration stem tracks (mute buttons, shimmer loading)

**LLM Options:**

| Flag | Description |
|------|-------------|
| `--ollama` | Use Ollama (local, free) - **default: enabled** |
| `--ollama-model` | Model to use (default: `midi-grep-strudel-mistral`) |

```bash
# Default: uses Ollama (free, local) with midi-grep-strudel-mistral
./bin/midi-grep extract --url "..." --iterate 5

# Use Claude API instead (requires ANTHROPIC_API_KEY)
./bin/midi-grep extract --url "..." --iterate 5 --ollama=false

# Use a specific Ollama model (e.g. fast smoke runs)
./bin/midi-grep extract --url "..." --iterate 5 --ollama-model llama3.1:8b
```

**Ollama Setup (one-time):**
```bash
# Install
brew install ollama

# Start service
ollama serve  # or: brew services start ollama

# Build the default custom model (constrains hallucinations via Modelfile system prompt)
ollama pull mistral-small
ollama create midi-grep-strudel-mistral -f Modelfile.mistral
```

**Tested Models (default chosen for 24GB RAM):**
| Model | Size | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| `midi-grep-strudel-mistral` | ~13GB | Medium | ⭐⭐⭐⭐ | **Default** — `mistral-small` base + Modelfile system prompt; middle ground, fits 24GB alongside Demucs/BlackHole |
| `midi-grep-strudel` | 42GB | Slow | ⭐⭐⭐⭐⭐ | `llama3.3:70b` base; best quality but needs ~48GB RAM (unusable on 24GB) |
| `llama3.1:8b` | 4.9GB | Fast | ⭐⭐ | Fast smoke runs only — hallucinates sound names, weak `arrange()` structure |

**Why a custom model?** The `Modelfile`/`Modelfile.mistral` SYSTEM prompt enforces the 3-voice `arrange()` structure and the sound-naming rules that stop the LLM inventing sounds like `sub_bass`. Applying it to a `mistral-small` base gives the best quality that fits 24GB RAM. The plain `llama3.1:8b` (no system prompt) hallucinates and produces sparse, low-energy arrangements.

**Strudel Code Validation & Genre RAG (`scripts/python/ollama_agent.py`):**

The agent uses **Genre-Aware Sound RAG** to provide only ~15 genre-relevant sounds per Ollama call, reducing hallucinated sound names. It also validates generated code and rejects invalid methods:

```python
INVALID_METHODS = [
    '.peak(',      # Doesn't exist - use .hpf() instead
    '.volume(',    # Should be .gain()
    '.eq(',        # Use .lpf/.hpf instead
    '.filter(',    # Too generic
    '.bass()', '.treble()', '.mid()', '.high()', '.low()',  # Not methods
]
```

- `_validate_code()` rejects code with invalid methods
- `last_validation_error` tracks rejection reason
- `ai_improver.py` skips iteration on validation failure (continues instead of crash)
- System prompt tells LLM what methods NOT to use

**ClickHouse for Learning Storage (`scripts/python/ai_improver.py`):**

ClickHouse stores all improvement runs for incremental learning across tracks.

**Tables:**
- `midi_grep.runs` - Every render attempt with similarity scores
  - `track_hash` - Unique identifier for the track
  - `version` - Run version number
  - `similarity_overall`, `similarity_mfcc`, `similarity_chroma`, etc.
  - `strudel_code` - The generated code for this run
  - `parameters` - JSON of effect parameters used
  - `genre`, `bpm`, `key_type` - Track metadata for context matching

- `midi_grep.knowledge` - Learned parameter improvements
  - `parameter_name` - Which parameter was changed (e.g., "bassFx.gain")
  - `parameter_old_value`, `parameter_new_value` - Before/after values
  - `similarity_improvement` - How much similarity increased
  - `confidence` - Statistical confidence in this learning
  - `genre`, `bpm_range_low`, `bpm_range_high`, `key_type` - Context for applying

**How learning works:**
1. Each render run is stored with full metadata
2. When parameters improve similarity, the delta is stored in `knowledge`
3. Future tracks query `knowledge` for similar context (genre, BPM, key)
4. System applies proven improvements automatically

**Setup:**
```bash
# Option 1: Local (development) - auto-used, no setup needed
./bin/clickhouse local --path .clickhouse/db --query "SELECT 1"

# Option 2: Docker (production)
docker-compose -f docker-compose.clickhouse.yml up -d

# Query stored runs
./bin/clickhouse local --path .clickhouse/db --query "SELECT track_hash, version, similarity_overall FROM midi_grep.runs ORDER BY created_at DESC LIMIT 10"
```

### Generative Mode Commands

The `generative` command (aliases: `gen`, `rave`) provides neural synthesizer training:

```bash
# List available generative models
./bin/midi-grep generative list

# Train a new granular model (fast, uses onset detection)
./bin/midi-grep generative train piano.wav --name my_piano --mode granular

# Train a RAVE neural network (quality, takes hours)
./bin/midi-grep generative train piano.wav --name my_synth --mode rave --epochs 500

# Search for similar models before training
./bin/midi-grep generative search piano.wav --threshold 0.85

# Process stems through full pipeline (auto-trains or reuses models)
./bin/midi-grep generative process ./stems --track-id mytrack

# Start local HTTP server for Strudel samples
./bin/midi-grep generative serve --port 5555

# After starting server, use in Strudel:
# await samples('http://localhost:5555/my_piano/')
# $: note("c3 e3 g3").sound("my_piano")
```

| Flag | Description |
|------|-------------|
| `--mode` | Training mode: `granular` (fast) or `rave` (quality) |
| `--models` | Models repository directory (default: `models`) |
| `--github` | GitHub repo for sync (e.g., `user/midi-grep-sounds`) |
| `--threshold` | Similarity threshold for reusing models (0.0-1.0, default: 0.88) |
| `--epochs` | Training epochs for RAVE mode (default: 500) |
| `--grain-ms` | Grain duration for granular mode (default: 100ms) |

## Genre Auto-Detection

The pipeline includes intelligent genre detection in `internal/pipeline/orchestrator.go`:

**Detection Functions:**
- `shouldUseBrazilianFunkMode()` - Detects Brazilian funk (BPM 130-145 or half-time 85-95, rejects long synth notes)
- `shouldUseBrazilianPhonkMode()` - Detects Brazilian phonk (BPM 80-100 or 145-180, darker sound)
- `shouldUseRetroWaveMode()` - Detects synthwave/retro wave (longer note durations, BPM 130-170)

**Manual Override:**
The `--genre` flag bypasses auto-detection and forces a specific genre:
```go
switch cfg.GenreOverride {
case "brazilian_funk":
    cfg.BrazilianFunk = true
case "retro_wave", "synthwave":
    cfg.SoundStyle = "synthwave"
    skipAutoDetection = true
// ...
}
```

**Deep Learning Detection (enabled by default):**
CLAP (Contrastive Language-Audio Pretraining) model for zero-shot classification:
- Script: `scripts/python/detect_genre_dl.py`
- Uses laion-clap or transformers CLAP implementation
- Compares audio embeddings against text descriptions of genres

## Dependencies

### Go
- `github.com/spf13/cobra` - CLI framework
- `github.com/go-chi/chi/v5` - HTTP router

### Python (in venv)
- `demucs` - Stem separation
- `basic-pitch` - Audio-to-MIDI
- `librosa` - Audio analysis
- `pretty_midi` - MIDI manipulation
- `openl3` - Timbre embeddings for RAVE pipeline
- `laion-clap` - Deep learning genre detection + timbre embeddings
- `acids-rave` - (Optional) Full RAVE neural network training

### Node.js/TypeScript (in scripts/node)
- `node-web-audio-api` - Web Audio API for Node.js (offline rendering)
- `typescript` - TypeScript compiler
- `@types/node` - Node.js type definitions

### System
- `yt-dlp` - YouTube downloads
- `ffmpeg` - Audio format conversion

## Testing

```bash
# Run Go tests
go test ./...

# Test extraction
./bin/midi-grep extract --input testdata/sample.wav
```

## Domain Experts

When working on specific areas, the golang-expert (`.claude/agents/golang-expert.md`) provides patterns for:
- Concurrency (errgroup, channels)
- Error handling (wrapping, sentinel errors)
- Interface design
- Subprocess execution

## Notes

- **Python 3.11 is the target** (venv is 3.11.x). The ML stack pins us here: `basic-pitch` needs
  TensorFlow 2.15 + `keras<3`, which do not support 3.12+. **Do NOT use 3.12-only syntax** — most
  notably the `type X = ...` alias statement (use plain `X = ...` aliases). `StrEnum`,
  `dataclass(slots=True)`, `Protocol`, and `X | None` are all fine on 3.11.
- First run downloads ~1GB of ML models
- Stem separation is CPU-intensive (1-2 min per track)
- HTMX used for web UI - no client-side JS frameworks
