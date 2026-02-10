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
- **Genre Detection:** Multi-method approach:
  - Heuristic: BPM ranges, note durations, spectral characteristics
  - Deep Learning: CLAP (Contrastive Language-Audio Pretraining) zero-shot classification
  - Essentia: Pre-trained ML models for genre/style classification

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
               │
               ▼ (optional)
      ┌─────────────────┐
      │ Audio Render    │ (Python)
      │ → kick/snare/hh │
      │ → bass synth    │
      │ → pad synth     │
      │ → stereo WAV    │
      └─────────────────┘
```

### Audio Rendering & AI Analysis

**Two approaches for audio rendering:**

1. **BlackHole Recording (RECOMMENDED - 100% accuracy):** Records real Strudel playback via virtual audio device
2. **Node.js Synthesis (Fallback - ~72% accuracy):** Offline synthesis emulating Strudel sounds

#### BlackHole Recording (Best Approach)

Records actual Strudel browser playback for perfect audio reproduction.
**Runs fully headless** - no browser window opens.

```
Strudel Code
     │
     ▼
┌─────────────────────────┐
│ ffmpeg Recording Start  │
│ -f avfoundation         │
│ -i :BlackHole 2ch       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Headless Puppeteer      │ (record-strudel-blackhole.ts)
│ - Opens strudel.cc      │
│ - Incognito context     │
│ - Opens Settings panel  │
│ - Selects BlackHole     │
│   as audio output       │
│ - Inserts code via      │
│   textContent (simple!) │
│ - Clicks Play button    │
│ - Waits for samples     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ BlackHole Virtual Audio │ (macOS)
│ - Routes browser audio  │
│ - No Multi-Output needed│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ ffmpeg Recording        │
│ → output.wav (44.1kHz)  │
└─────────────────────────┘
```

**Setup:** `brew install blackhole-2ch` (requires reboot)
**Usage:** `node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30`

**Key implementation:**
- `--autoplay-policy=no-user-gesture-required` for audio context
- Incognito context avoids localStorage issues
- CodeMirror code insertion: `cmContent.textContent = code`
- Play button found by iterating buttons & checking textContent

#### Node.js Synthesis (Fallback)

The `--render` flag synthesizes WAV audio preview from Strudel patterns with AI-driven parameter optimization:

```
Original Audio                    Strudel Code
     │                                 │
     ▼                                 ▼
┌─────────────────┐          ┌─────────────────┐
│ AI Analysis     │          │ Parse Patterns  │
│ (audio_to_      │          │ - let name = `..`
│  strudel_params)│          │ - BPM from setcps
│ - Spectral      │          └────────┬────────┘
│ - Dynamics      │                   │
│ - Timbre        │                   │
└────────┬────────┘                   │
         │          ┌─────────────────┘
         ▼          ▼
┌─────────────────────────┐
│ Synthesize (render_audio)│
│ - Kick: pitch env + distort
│ - Snare: tone + noise
│ - HH: filtered noise
│ - Bass: saw + sub + LPF
│ - Vox: square + fast attack
│ - Stabs: filtered sawtooth
│ - Lead: triangle + vibrato
│ + AI-suggested mix levels
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Mix to Stereo           │ (44.1kHz, 16-bit)
│ - Pan per voice
│ - AI-driven gain staging
│ - Normalize
└────────────┬────────────┘
             │
             ▼
      render_vXXX.wav
             │
             ▼
┌─────────────────────────┐
│ Compare (compare_audio) │
│ - Frequency balance (MAE)
│ - MFCC timbral match
│ - Energy/loudness
│ - Brightness (centroid)
│ - Per-band difference tracking
│ - Overall score (0-100%)
│                         │
│ WEIGHTS:                │
│ - Freq Balance: 40%     │
│ - MFCC: 20%             │
│ - Energy: 15%           │
│ - Brightness: 15%       │
│ - Tempo: 5%             │
│ - Chroma: 5%            │
└─────────────────────────┘
```

**AI Scripts:**
- `audio_to_strudel_params.py` - Analyzes original audio to suggest Strudel effect parameters
- `compare_audio.py` - Compares rendered vs original for quality feedback
  - **CRITICAL:** Uses MAE-based frequency balance, not cosine (cosine hides 20%+ band errors!)
  - Weights: Frequency Balance 40%, MFCC 20%, Energy 15%, Brightness 15%, Tempo/Chroma 5% each
  - Penalty if any band off by >15%
- `ai_improver.py` - AI-driven iterative code improvement (Ollama/Claude)
- `ai_code_improver.py` - Gap analysis and Strudel code modification
- `ai_iterative_codegen.py` - Iteration loop with automatic revert-on-regression
- `spectrogram_analyzer.py` - Deep mel spectrogram analysis for AI learning
- `sound_selector.py` - Complete sound catalog (67 drum machines, 128 GM instruments, 17 genre palettes)
- `generate_report.py` - Self-contained HTML report with DAW-style audio studio

### Default Analysis Features (All Enabled)

All analysis features run by default:

```
Strudel Code
     │
     ▼
┌─────────────────────────┐
│ Node.js Renderer        │ (render-strudel-node.ts)
│ --stems flag ALWAYS on  │
│ → render_bass.wav       │
│ → render_drums.wav      │
│ → render_melodic.wav    │
│ → render.wav (mix)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Per-Stem Comparison     │ (compare_audio.py --stems)
│ → chart_stem_bass.png   │
│ → chart_stem_drums.png  │
│ → chart_stem_melodic.png│
│ → stem_comparison.json  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ AI Improvement          │ (5 iterations, 85% target)
│ - Analyze gaps          │
│ - LLM suggests changes  │
│ - Apply & re-render     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ HTML Report             │ (generate_report.py)
│ - Audio Studio Player   │
│   - Original stems      │
│   - Rendered stems      │
│   - Solo/Mute controls  │
│   - A/B comparison      │
│ - Waveform visualizations
│ - Per-stem charts       │
│ - Comparison data       │
│ - Strudel code          │
└─────────────────────────┘
```

---

## 3. Data & Persistence

- **Stem Cache:** `.cache/stems/` in repository root, keyed by URL/file hash
- **Cache Versioning:** Auto-computed from script hashes, invalidates on code changes
- **Output Versioning:** Each generation creates v001, v002, etc. with metadata
- **File Storage:** Local filesystem (OS temp directory for processing artifacts)
- **Session State:** None — fully stateless request/response cycle
- **Cleanup Strategy:** Delete temp files after response sent (or after timeout)
- **No Database:** V1 requires no persistence (no users, no history)

### Cache Directory Structure

```
.cache/stems/yt_VIDEO_ID/
├── piano.wav              # Separated melodic stem
├── drums.wav              # Separated drums stem
├── bass.wav               # Separated bass stem (full mode)
├── .version               # Cache version (script hash)
├── output_v001.strudel    # Version 1 Strudel code
├── output_v001.json       # Version 1 metadata
├── output_v002.strudel    # Version 2...
├── output_latest.strudel  # Symlink to latest
├── render_v001.wav        # Rendered audio for v1
└── render_v002.wav        # Rendered audio for v2
```

### Output Metadata (`output_vXXX.json`)

```json
{
  "code": "// MIDI-grep output...",
  "bpm": 136,
  "key": "C# minor",
  "style": "brazilian_funk",
  "genre": "brazilian_funk",
  "notes": 497,
  "drum_hits": 287,
  "version": 1,
  "created_at": "2025-02-03T01:24:00Z"
}
```

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
│   │   ├── arrangement.go    # Arrangement-based output
│   │   └── brazilian.go      # Brazilian funk/phonk template generation
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
│   ├── python/
│   │   ├── requirements.txt  # Python dependencies
│   │   ├── separate.py       # Stem separation (melodic/bass/drums/vocals)
│   │   ├── transcribe.py     # Basic Pitch wrapper
│   │   ├── analyze.py        # BPM + key detection
│   │   ├── cleanup.py        # MIDI quantization
│   │   ├── detect_drums.py   # Drum hit detection
│   │   ├── smart_analyze.py  # Chord/section detection
│   │   ├── chord_to_strudel.py # Chord-based generation
│   │   ├── render_audio.py   # Basic WAV synthesis (preview only)
│   │   ├── compare_audio.py  # Audio similarity comparison (spectral/rhythmic/timbral)
│   │   ├── audio_to_strudel_params.py # AI-driven Strudel effect parameter suggestion
│   │   ├── aggregate_genre_analysis.py # Batch genre analysis aggregation
│   │   ├── detect_genre_dl.py    # CLAP deep learning genre detection (zero-shot)
│   │   └── detect_genre_essentia.py # Essentia ML model genre classification
│   │
│   └── node/                 # Future: Strudel-native rendering
│       └── package.json      # @strudel/core, @strudel/webaudio deps
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

# Manual genre override (when auto-detection fails)
midi-grep extract --url "..." --genre retro_wave

# CLAP genre detection runs automatically (use --genre to override)

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
