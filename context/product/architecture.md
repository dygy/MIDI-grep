# System Architecture Overview: MIDI-grep

---

## 1. Application & Technology Stack

- **Backend Language:** Go 1.21+ (performance, single binary, excellent concurrency)
- **HTTP Router:** Chi (lightweight, idiomatic Go, composable middleware)
- **Template Engine:** Go `html/template` (standard library, auto-escaping, secure)
- **Frontend Interactivity:** HTMX 1.9+ (reactive UI via HTML attributes, zero custom JS)
- **CSS Framework:** PicoCSS (classless semantic styling, minimal footprint)
- **CLI Framework:** Cobra (standard for Go CLIs, flag parsing, subcommands)

---

## 2. Audio Processing Pipeline

*Go orchestrates external Python tools via subprocess (`os/exec`). Each tool runs in isolation with clear input/output contracts.*

- **Stem Separation:** Demucs htdemucs (outputs: melodic/other, drums, bass, vocals)
- **Caching:** Stems cached by URL/file hash in `.cache/stems/`, auto-invalidates when scripts change
- **Audio-to-MIDI Transcription:** Basic Pitch (Spotify's ML model, high accuracy)
- **Drum Detection:** librosa onset detection + spectral analysis for kick/snare/hi-hat classification
- **MIDI Manipulation:** pretty_midi (Python library for cleanup, quantization)
- **BPM Detection:** librosa.beat.beat_track (robust tempo estimation)
- **Key Detection:** librosa + Krumhansl-Schmuckler algorithm (key profile matching)

### Pipeline Flow

```
[WAV/MP3/YouTube URL]
         │
         ▼
┌─────────────────┐
│ Cache Check     │ (URL/file hash → .cache/stems/)
│   ↓ miss        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stem Separation │ (Demucs htdemucs)
│   → melodic.wav │
│   → drums.wav   │
│   → bass.wav    │
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
│ Detection   │  │ → bd/sd/hh   │
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
│ - loop det  │         │
└──────┬──────┘         │
       │                │
       └───────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │ Strudel Gen     │ (Go)
      │ → bar arrays    │
      │ → effect funcs  │
      │ → drums + notes │
      └─────────────────┘
```

---

## 3. Data & Persistence

- **Stem Cache:** `.cache/stems/` in repository root, keyed by URL/file hash
- **Cache Versioning:** Auto-computed from script hashes, invalidates on code changes
- **File Storage:** Local filesystem (OS temp directory for processing artifacts)
- **Session State:** None — fully stateless request/response cycle
- **Cleanup Strategy:** Delete temp files after response sent (or after timeout)
- **No Database:** V1 requires no persistence (no users, no history)

---

## 4. Infrastructure & Deployment

- **Containerization:** Docker with multi-stage build
  - Stage 1: Build Go binary (`golang:1.21-alpine`)
  - Stage 2: Runtime (`python:3.11-slim` + Go binary + Python deps)
- **Python Environment:** pip install spleeter demucs basic-pitch librosa pretty_midi
- **Local Development:**
  - Go: standard `go run`
  - Python: venv in `scripts/python/.venv`
- **Deployment Targets:**
  - fly.io (simple, scales down to zero)
  - Railway (git push deploy)
  - Self-hosted Docker (any VPS)

### Resource Requirements

- **Memory:** 4GB minimum (Spleeter/Demucs are memory-intensive)
- **CPU:** 2+ cores recommended for parallel analysis
- **Disk:** Temp space for audio files (~100MB per request max)

---

## 5. Project Structure

```
midi-grep/
├── cmd/
│   └── midi-grep/
│       └── main.go           # Entrypoint: CLI + server modes
│
├── internal/
│   ├── audio/
│   │   ├── input.go          # File validation, format detection
│   │   ├── stems.go          # Demucs orchestration
│   │   └── youtube.go        # yt-dlp integration
│   │
│   ├── cache/
│   │   └── cache.go          # Stem caching by URL/file hash
│   │
│   ├── drums/
│   │   └── detector.go       # Drum hit detection/classification
│   │
│   ├── analysis/
│   │   └── analysis.go       # BPM + key detection wrapper
│   │
│   ├── midi/
│   │   ├── transcribe.go     # Basic Pitch wrapper
│   │   ├── cleanup.go        # Quantization, filtering
│   │   └── loop.go           # Loop detection
│   │
│   ├── strudel/
│   │   ├── generator.go      # MIDI → Strudel bar arrays
│   │   ├── drums.go          # Drum pattern generation
│   │   ├── effects.go        # Per-voice effect settings
│   │   ├── sections.go       # Section detection
│   │   ├── chords.go         # Chord detection/voicings
│   │   └── arrangement.go    # Arrangement-based output
│   │
│   ├── pipeline/
│   │   └── orchestrator.go   # End-to-end pipeline coordination
│   │
│   └── server/
│       ├── server.go         # HTTP server setup
│       ├── handlers.go       # Request handlers
│       └── jobs.go           # Background job processing
│
├── web/
│   ├── templates/
│   │   ├── base.html         # Layout with HTMX
│   │   ├── index.html        # Upload form
│   │   ├── progress.html     # Processing status partial
│   │   └── result.html       # Output display partial
│   │
│   └── static/
│       ├── htmx.min.js       # HTMX (vendored, ~14KB)
│       └── pico.min.css      # PicoCSS (vendored)
│
├── scripts/
│   └── python/
│       ├── requirements.txt  # Python dependencies
│       ├── separate.py       # Stem separation (melodic/bass/drums/vocals)
│       ├── transcribe.py     # Basic Pitch wrapper
│       ├── analyze.py        # BPM + key detection
│       ├── cleanup.py        # MIDI quantization
│       ├── detect_drums.py   # Drum hit detection
│       ├── smart_analyze.py  # Chord/section detection
│       └── chord_to_strudel.py # Chord-based generation
│
├── Dockerfile
├── docker-compose.yml        # Local dev with volume mounts
├── Makefile                  # build, run, test, docker commands
└── go.mod
```

---

## 6. API Design

### CLI Interface

```bash
# Extract from YouTube (uses cache)
midi-grep extract --url "https://youtu.be/VIDEO_ID"

# Extract from file
midi-grep extract --input track.wav --output riff.strudel

# Chord mode (for electronic/funk)
midi-grep extract --url "..." --chords

# Force fresh extraction (skip cache)
midi-grep extract --url "..." --no-cache

# Drums only
midi-grep extract --url "..." --drums-only --drum-kit tr808

# Custom style
midi-grep extract --url "..." --style house

# Start web server
midi-grep serve --port 8080
```

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Upload page (HTML) |
| POST | `/upload` | Accept audio file, return job ID |
| GET | `/status/{id}` | SSE stream of processing progress |
| GET | `/result/{id}` | Final result (HTML partial or JSON) |
| GET | `/download/{id}/midi` | Download cleaned MIDI file |
| GET | `/health` | Health check |

---

## 7. Error Handling Strategy

- **User Errors:** Clear HTML messages (unsupported format, file too large)
- **Processing Errors:** Graceful degradation (if Spleeter fails, try Demucs)
- **System Errors:** Logged server-side, generic message to user
- **Timeout:** 5-minute max processing time, cleanup on timeout

---

## 8. Security Considerations

- **File Validation:** Check magic bytes, not just extension
- **Size Limits:** Max 50MB upload
- **Temp Isolation:** Each request gets unique temp directory
- **No Execution:** Audio files never executed, only processed by trusted tools
- **HTMX Security:** CSP headers, no inline scripts
