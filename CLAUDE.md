# CLAUDE.md - Project Context for Claude Code

This file provides context for Claude Code when working on this project.

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

**Current achievement:** ~60-70% similarity with honest calculation (previous 90%+ was inflated by cosine bug)
**Target:** 80%+ similarity across all genres through AI learning, not hardcoding

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

**Node.js Strudel Renderer (`scripts/node/src/render-strudel-node.ts`):**
- TypeScript-based offline audio rendering with Strudel pattern parsing
- Uses `@strudel/mini` v1.1.0 for accurate mini-notation parsing
- **Synthesis engine with frequency-balanced mix:**
  - `synthKick()` - 808-style kick with pitch envelope (150→40Hz), amp decay, click transient
  - `synthSnare()` - Dual-sine body (180Hz + 330Hz) + high-passed noise for wires
  - `synthHihat()` - Metallic multi-frequency noise with envelope (open/closed variants)
  - `synthBass()` - Sawtooth + sub-octave sine, low-pass filtered for warmth
  - `synthLead()` - Detuned saws + triangle, filter envelope for movement
  - `synthHigh()` - Odd-harmonic square wave + saw for brightness
- **Mix levels tuned for melodic content:**
  - Bass: 0.08x gain (minimal to avoid mud)
  - Mids: 3.0x gain (dominant, matches typical melodic stems)
  - Highs: 2.5x gain (bright presence)
  - Drums: 0.15x gain, kicks extra low at 0.2x
- 80Hz high-pass filter on master to reduce sub-bass mud
- Achieves ~79% similarity against melodic stems, 99% frequency balance
- Outputs 16-bit 44.1kHz mono WAV files
- Build: `cd scripts/node && npm run build`
- Usage: `node dist/render-strudel-node.js input.strudel -o output.wav -d 30`

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
  - Web Audio API for synchronized playback
  - Volume faders per stem
  - Synchronized playback controls
  - A/B comparison mode (toggle between original and rendered)
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

- **Language**: Go 1.21+
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
│   ├── strudel/
│   │   ├── generator.go        # MIDI → Strudel conversion (bar arrays + effects)
│   │   ├── drums.go            # Drum pattern generation (s() patterns)
│   │   ├── effects.go          # Per-voice effect settings (filter, pan, reverb, delay)
│   │   ├── sections.go         # Section detection (intro, verse, chorus)
│   │   ├── brazilian.go        # Brazilian funk/phonk template generation
│   │   ├── chords.go           # Chord detection and voicings
│   │   └── arrangement.go      # Arrangement-based output (chord variables)
│   ├── pipeline/orchestrator.go # End-to-end CLI pipeline
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
│   │   │   └── render-strudel-node.ts  # Offline Strudel renderer
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
│       ├── ai_learning_optimizer.py # AI learning optimization
│       ├── spectrogram_analyzer.py # Mel spectrogram deep analysis for AI
│       ├── sound_selector.py   # Complete sound catalog (67 drums, 128 GM)
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
│   │   ├── roadmap.md
│   │   └── architecture.md
│   └── spec/
│       └── 001-core-pipeline/
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

### Modifying Strudel output

The Strudel generator is split across three files:

**`internal/strudel/generator.go`** - Main output generation:
- `Generate()` - main entry point
- `outputChunkedPatterns()` - bar arrays with effect functions (default format)
- `voiceToPattern()` - note conversion to mini-notation
- `midiToNoteName()` - pitch notation

**`internal/strudel/drums.go`** - Drum pattern generation:
- `GenerateDrumPattern()` - creates drum patterns from detection results
- `GenerateFullOutput()` - combines melodic + drums into final output
- `buildDrumBarArrays()` - creates drum bar arrays matching melodic format
- `DrumKit` types: tr808, tr909, linn, acoustic, lofi

**`internal/strudel/effects.go`** - Per-voice effect settings:
- `VoiceEffects` struct - filter, pan, reverb, delay, envelope, styleFX, patternFX, legato, echo, harmony, tremolo, filterEnv, duck
- `EnvelopeSettings` - ADSR envelope (attack, decay, sustain, release)
- `StyleFXSettings` - phaser, crush, coarse, vowel, distort, vibrato, FM synthesis (fm, fmh, fmdecay, fmsustain)
- `PatternFXSettings` - jux, swing, degradeBy, ply, iter, rev
- `LegatoSettings` - clip for note duration control
- `EchoSettings` - echo/stutter effect (times, time, feedback)
- `HarmonySettings` - superimpose (detune), off (harmonic layering)
- `TremoloSettings` - amplitude modulation (sync, depth, shape)
- `FilterEnvSettings` - filter envelope (attack, decay, sustain, release, amount)
- `DuckSettings` - sidechain ducking (orbit, attack, depth)
- `AccentSettings` - beat emphasis (pattern, amount)
- `CompressorSettings` - dynamics compression (threshold, ratio, knee, attack, release)
- `DynamicsSettings` - velocity processing (range expansion, velocity curve)
- `LFOShape` - sine, cosine, saw, tri, square, perlin, rand
- `GetVoiceEffects()` - returns effects for voice type + style
- `BuildEffectChain()` - generates Strudel effect method chain
- `BuildPatternTransforms()` - generates pattern-level transforms (.swing, .degradeBy)
- `BuildHarmonyEffects()` - generates layering effects (.superimpose, .off)
- `BuildScaleEffect()` - generates scale quantization effect

**`internal/strudel/sections.go`** - Section detection:
- `DetectSections()` - analyzes note density, velocity, register per bar
- `Section` struct - start/end beats, type, energy level
- `GenerateSectionHeader()` - creates time-stamped section comments

**`internal/strudel/brazilian.go`** - Brazilian funk/phonk generation:
- `GenerateBrazilianFunk()` - template-based output for funk carioca
- Tamborzão drum patterns (syncopated 808 kicks)
- Multiple pattern variations with block comments (`/* */`) for easy live coding
- 808 bass with slide and distortion
- Synth stabs using detected key (C# minor chord tones, etc.)
- Helper functions: `getThird()`, `getSeventh()`, `getMinorScale()`, `getMajorScale()`

**`internal/strudel/chords.go`** - Chord detection:
- `DetectChords()` - identifies chord types from MIDI notes
- Chord types: major, minor, 7th, 9th, dim, aug, sus
- `ChordProgression` struct for tracking changes over time

**`internal/strudel/arrangement.go`** - Arrangement-based output:
- `GenerateArrangement()` - creates chord-variable based output
- Uses `arrange([bars, "<chord>"])` syntax
- Chord voicings with `.dict('ireal-ext').voicing()`
- Root note bass, arpeggiated parts, pad layers

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
| `--iterate N` | AI-driven improvement iterations (default: 5) |
| `--target-similarity` | Target similarity for --iterate (0.0-1.0, default: 0.85) |

### Default Analysis Features (Always Enabled)

The following analysis features are **always enabled by default**:

1. **Stem Rendering**: Renders 3 separate stems (`render_bass.wav`, `render_drums.wav`, `render_melodic.wav`)
2. **Per-Stem Comparison**: Generates per-stem comparison charts (`chart_stem_bass.png`, `chart_stem_drums.png`, `chart_stem_melodic.png`)
3. **Overall Comparison**: Generates combined comparison chart and `comparison.json`
4. **AI-Driven Improvement**: 5 iterations by default with 99% target (ensures ALL iterations run)
5. **HTML Report**: Self-contained DAW-style player with isolated Original/Rendered stem groups

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
5. Repeat until target similarity or max iterations reached
6. Store all runs in ClickHouse for incremental learning

**LLM Options:**

| Flag | Description |
|------|-------------|
| `--ollama` | Use Ollama (local, free) - **default: enabled** |
| `--ollama-model` | Model to use (default: `llama3:8b`) |

```bash
# Default: uses Ollama (free, local) with llama3:8b
./bin/midi-grep extract --url "..." --iterate 5

# Use Claude API instead (requires ANTHROPIC_API_KEY)
./bin/midi-grep extract --url "..." --iterate 5 --ollama=false

# Use specific Ollama model
./bin/midi-grep extract --url "..." --iterate 5 --ollama-model llama3:8b
```

**Ollama Setup (one-time):**
```bash
# Install
brew install ollama

# Start service
ollama serve  # or: brew services start ollama

# Pull recommended model (understands music concepts)
ollama pull llama3:8b
```

**Tested Models:**
| Model | Size | Speed | Music Understanding | Notes |
|-------|------|-------|---------------------|-------|
| `llama3:8b` | 4.7GB | Medium | ⭐⭐⭐⭐⭐ | **Recommended** - best for music + audio concepts |
| `deepseek-coder:6.7b` | 3.8GB | Fast | ⭐⭐ | Good at JSON but code-focused |
| `codellama:7b` | 3.8GB | Fast | ⭐⭐ | Code-focused, less musical knowledge |
| `mistral:7b` | 4.1GB | Fast | ⭐⭐⭐⭐ | Good general model |

**Why `llama3:8b`?** The LLM needs to understand audio/music concepts ("bass sounds muddy", "mids are harsh", "drums lack punch") not just generate code. General-purpose models with broad knowledge outperform code-only models for this task.

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

When working on specific areas, the golang-expert (`.awos/subagents/golang-expert.md`) provides patterns for:
- Concurrency (errgroup, channels)
- Error handling (wrapping, sentinel errors)
- Interface design
- Subprocess execution

## Notes

- Python 3.11+ required for ML dependencies
- First run downloads ~1GB of ML models
- Stem separation is CPU-intensive (1-2 min per track)
- HTMX used for web UI - no client-side JS frameworks
