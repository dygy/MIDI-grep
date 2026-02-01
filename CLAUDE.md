# CLAUDE.md - Project Context for Claude Code

This file provides context for Claude Code when working on this project.

## Project Overview

**MIDI-grep** is a Go CLI and web application that extracts piano riffs from audio files or YouTube videos and generates Strudel code for live coding.

### Core Pipeline

```
Input (WAV/MP3/YouTube URL)
    ↓
Stem Separation (Demucs) → piano/instrumental stem
    ↓
Analysis (librosa) → BPM, Key detection
    ↓
Transcription (Basic Pitch) → MIDI notes
    ↓
Cleanup → Quantization, filtering
    ↓
Output → Strudel code
```

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
│   ├── strudel/generator.go    # MIDI → Strudel conversion
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
│   └── python/
│       ├── separate.py         # Demucs stem separation
│       ├── analyze.py          # BPM/key detection
│       ├── transcribe.py       # Basic Pitch transcription
│       ├── cleanup.py          # MIDI quantization
│       └── requirements.txt
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

Edit `internal/strudel/generator.go`:
- `Generate()` - main output format
- `notesToPattern()` - note conversion
- `midiToNoteName()` - pitch notation

### Adding CLI flags

Edit `cmd/midi-grep/main.go`:
- Add flag in `init()`
- Use in `runExtract()` or `runServe()`

## Build & Run

```bash
# Build
go build -o bin/midi-grep ./cmd/midi-grep

# Run CLI
./bin/midi-grep extract --url "https://youtu.be/..."
./bin/midi-grep serve --port 8080

# Or use wrapper script
./scripts/midi-grep.sh extract --url "..."
./scripts/midi-grep.sh serve
```

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
