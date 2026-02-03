# CLAUDE.md - Project Context for Claude Code

This file provides context for Claude Code when working on this project.

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

### Caching

Stems are cached in `.cache/stems/` by URL or file hash. Subsequent runs skip download and separation.

Cache version is auto-computed from script hashes (`separate.py`). When scripts change, cache is automatically invalidated.

## Tech Stack

- **Language**: Go 1.21+
- **CLI Framework**: Cobra
- **Web Framework**: Chi + HTMX + Go templates
- **Audio Processing**: Python scripts (demucs, basic-pitch, librosa)
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
│   │   ├── generator.go        # MIDI → Strudel conversion
│   │   ├── effects.go          # Per-voice effect settings (filter, pan, reverb, delay)
│   │   └── sections.go         # Section detection (intro, verse, chorus)
│   ├── pipeline/orchestrator.go # End-to-end CLI pipeline
│   ├── server/
│   │   ├── server.go           # HTTP server setup
│   │   ├── handlers.go         # Request handlers
│   │   ├── jobs.go             # Background job processing
│   │   └── templates/          # HTMX templates
│   ├── exec/runner.go          # Python subprocess execution
│   ├── progress/progress.go    # CLI progress output
│   ├── workspace/workspace.go  # Temp file management
│   └── errors/errors.go        # Custom error types
├── scripts/
│   ├── midi-grep.sh            # Main CLI wrapper script
│   ├── extract-youtube.sh      # Quick YouTube extraction
│   └── python/
│       ├── separate.py         # Demucs stem separation (melodic/bass/drums/vocals)
│       ├── analyze.py          # BPM/key detection
│       ├── smart_analyze.py    # Advanced chord/section detection
│       ├── chord_to_strudel.py # Chord-based Strudel generation
│       ├── transcribe.py       # Basic Pitch transcription
│       ├── cleanup.py          # MIDI quantization
│       ├── detect_drums.py     # Drum pattern detection
│       └── requirements.txt
├── internal/cache/cache.go     # Stem caching (by URL/file hash)
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

## Dependencies

### Go
- `github.com/spf13/cobra` - CLI framework
- `github.com/go-chi/chi/v5` - HTTP router

### Python (in venv)
- `demucs` - Stem separation
- `basic-pitch` - Audio-to-MIDI
- `librosa` - Audio analysis
- `pretty_midi` - MIDI manipulation

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
