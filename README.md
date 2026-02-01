# MIDI-grep

Extract piano riffs from audio files or YouTube videos and generate [Strudel](https://strudel.dygy.app/) code for live coding.

```
Audio/YouTube → Stem Separation → MIDI Transcription → Strudel Code
```

## Features

- **YouTube Support**: Paste a URL, get playable code
- **AI-Powered Separation**: Demucs isolates piano/instruments from any mix
- **Accurate Transcription**: Spotify's Basic Pitch for audio-to-MIDI
- **BPM & Key Detection**: Automatic tempo and musical key analysis
- **Strudel Output**: Ready-to-play `note()` patterns
- **Web Interface**: HTMX-powered UI, no JavaScript frameworks
- **CLI Tool**: Full-featured command-line interface

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MIDI-grep                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INPUT     │    │  SEPARATE   │    │  TRANSCRIBE │    │   OUTPUT    │  │
│  │             │    │             │    │             │    │             │  │
│  │ • YouTube   │───▶│ • Demucs    │───▶│ • Basic     │───▶│ • Strudel   │  │
│  │   (yt-dlp)  │    │   (PyTorch) │    │   Pitch     │    │   code      │  │
│  │ • WAV/MP3   │    │ • Stems:    │    │   (TF)      │    │ • MIDI file │  │
│  │             │    │   piano,    │    │ • librosa   │    │ • JSON      │  │
│  │             │    │   bass,     │    │   (BPM/Key) │    │             │  │
│  │             │    │   drums     │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           TECHNOLOGY STACK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐│
│  │         GO (1.21+)          │    │           PYTHON (3.11+)            ││
│  │                             │    │                                     ││
│  │  • CLI (Cobra)              │    │  • demucs      - Stem separation    ││
│  │  • HTTP Server (Chi)        │    │  • basic-pitch - Audio → MIDI       ││
│  │  • Pipeline orchestration   │    │  • librosa     - Audio analysis     ││
│  │  • Strudel code generation  │    │  • pretty_midi - MIDI processing    ││
│  │                             │    │  • tensorflow  - ML inference       ││
│  └─────────────────────────────┘    └─────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐│
│  │        WEB FRONTEND         │    │            EXTERNAL                 ││
│  │                             │    │                                     ││
│  │  • HTMX (no JS frameworks)  │    │  • yt-dlp     - YouTube download    ││
│  │  • PicoCSS (styling)        │    │  • ffmpeg     - Audio conversion    ││
│  │  • SSE (real-time updates)  │    │                                     ││
│  │  • Go templates             │    │                                     ││
│  └─────────────────────────────┘    └─────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
midi-grep/
├── cmd/midi-grep/           # CLI entrypoint (Go)
│   └── main.go              # Cobra commands: extract, serve, train
│
├── internal/                # Go packages
│   ├── audio/               # Input validation, YouTube download, stem separation
│   ├── analysis/            # BPM & key detection (calls Python)
│   ├── midi/                # Transcription & cleanup (calls Python)
│   ├── strudel/             # MIDI → Strudel code generation
│   ├── pipeline/            # Orchestrates the full extraction flow
│   ├── server/              # HTTP server, HTMX templates, SSE
│   └── exec/                # Python subprocess runner
│
├── scripts/
│   ├── midi-grep.sh         # Main CLI wrapper (Bash)
│   └── python/              # Python ML scripts
│       ├── separate.py      # Demucs stem separation
│       ├── transcribe.py    # Basic Pitch audio → MIDI
│       ├── analyze.py       # librosa BPM/key detection
│       ├── cleanup.py       # MIDI quantization & filtering
│       └── training/        # Model fine-tuning (Phase 9)
│
└── context/                 # AWOS documentation
    ├── product/             # Product definition, roadmap
    └── spec/                # Feature specifications
```

### Data Flow

```
1. INPUT
   YouTube URL ──▶ yt-dlp ──▶ audio.wav
   Local file ─────────────▶ audio.wav

2. STEM SEPARATION
   audio.wav ──▶ Demucs ──▶ piano.mp3 (isolated instrument)

3. ANALYSIS
   piano.mp3 ──▶ librosa ──▶ { bpm: 120, key: "A minor" }

4. TRANSCRIPTION
   piano.mp3 ──▶ Basic Pitch ──▶ raw.mid ──▶ cleanup ──▶ notes.json

5. GENERATION
   notes.json + analysis ──▶ Strudel Generator ──▶ code.strudel
```

## Quick Start

```bash
# Install dependencies
./scripts/midi-grep.sh install

# Extract from YouTube
./scripts/midi-grep.sh extract --url "https://youtu.be/Q4801HzWZfg"

# Extract from local file
./scripts/midi-grep.sh extract --file track.wav --output riff.strudel

# Start web interface
./scripts/midi-grep.sh serve --port 8080
```

## Installation

### Prerequisites

- **Go 1.21+**: `brew install go`
- **Python 3.11+**: `brew install python@3.11`
- **yt-dlp** (for YouTube): `brew install yt-dlp`
- **ffmpeg** (for audio processing): `brew install ffmpeg`

### Install

```bash
# Clone the repository
git clone https://github.com/arkadiishvartcman/midi-grep.git
cd midi-grep

# Install all dependencies (Go + Python)
./scripts/midi-grep.sh install
```

This installs:
- Python packages: `demucs`, `basic-pitch`, `librosa`, `pretty_midi`
- Builds the Go binary

## Usage

### Command Line

```bash
# Basic extraction from YouTube
./scripts/midi-grep.sh extract --url "https://youtu.be/VIDEO_ID"

# With options
./scripts/midi-grep.sh extract \
  --url "https://youtu.be/VIDEO_ID" \
  --quantize 8 \
  --output riff.strudel \
  --midi piano.mid

# From local file
./scripts/midi-grep.sh extract --file track.wav

# Copy to clipboard (macOS)
./scripts/midi-grep.sh extract --url "..." --copy
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--url` | `-u` | YouTube URL to extract from |
| `--file` | `-f` | Local audio file (WAV/MP3) |
| `--output` | `-o` | Output file for Strudel code |
| `--quantize` | `-q` | Quantization: 4, 8, or 16 (default: 16) |
| `--midi` | `-m` | Also save cleaned MIDI file |
| `--copy` | `-c` | Copy result to clipboard |
| `--verbose` | `-v` | Show verbose output |

### Web Interface

```bash
./scripts/midi-grep.sh serve --port 8080
```

Open http://localhost:8080 in your browser:
- Drag & drop audio files
- Paste YouTube URLs
- Real-time progress updates
- Copy Strudel code with one click

### Direct Binary Usage

```bash
# Build
make build

# CLI
./bin/midi-grep extract --input track.wav
./bin/midi-grep extract --url "https://youtu.be/..."
./bin/midi-grep serve --port 8080

# Help
./bin/midi-grep --help
./bin/midi-grep extract --help
```

## Example Output

```javascript
// MIDI-grep output
// BPM: 89, Key: E major
setcps(89/60/4)

$: note("[e2,e3,b3,gs4,ds5] [b3,ds4,ds5] gs5 e5 [ds4,cs5]...")
  .sound("piano")
  .room(0.3).size(0.6)
```

Paste this into [Strudel](https://strudel.dygy.app/) and press Ctrl+Enter to play!

## How It Works

1. **Input**: Audio file or YouTube URL (downloaded via yt-dlp)
2. **Stem Separation**: Demucs AI model isolates instruments
3. **Analysis**: librosa detects BPM and musical key
4. **Transcription**: Basic Pitch converts audio to MIDI
5. **Cleanup**: Quantization, velocity filtering, noise removal
6. **Generation**: MIDI notes converted to Strudel mini-notation

## Project Structure

```
midi-grep/
├── cmd/midi-grep/          # CLI entrypoint
├── internal/
│   ├── audio/              # File validation, stems, YouTube
│   ├── analysis/           # BPM & key detection
│   ├── midi/               # Transcription & cleanup
│   ├── strudel/            # Code generation
│   ├── pipeline/           # Orchestration
│   └── server/             # Web interface (HTMX)
├── scripts/
│   ├── midi-grep.sh        # Main CLI wrapper
│   └── python/             # Python processing scripts
├── context/                # AWOS product docs
├── Makefile
├── Dockerfile
└── README.md
```

## Configuration

### Quantization

Controls note timing precision:
- `4` = Quarter notes (simplified, loose timing)
- `8` = Eighth notes (moderate detail)
- `16` = Sixteenth notes (full detail, default)

### Audio Requirements

- **Formats**: WAV, MP3
- **Max size**: 100MB
- **Best results**: Clear piano recordings, minimal background noise

## Docker

```bash
# Build image
docker build -t midi-grep .

# Run extraction
docker run -v $(pwd):/data midi-grep extract --input /data/track.wav

# Run server
docker run -p 8080:8080 midi-grep serve
```

## Development

```bash
# Build
make build

# Run tests
make test

# Install deps
make deps

# Start dev server
make serve
```

## Tech Stack

- **Backend**: Go 1.21+, Chi router
- **Frontend**: HTMX, PicoCSS (no JavaScript frameworks)
- **Audio Processing**:
  - Demucs (stem separation)
  - Basic Pitch (audio-to-MIDI)
  - librosa (analysis)
- **CLI**: Cobra

## Troubleshooting

### "Demucs failed" or "Spleeter not installed"
```bash
./scripts/midi-grep.sh install
```

### "yt-dlp not found"
```bash
brew install yt-dlp
# or
pip install yt-dlp
```

### Python version issues
```bash
brew install python@3.11
```

### Slow processing
- Stem separation takes 1-2 minutes for a 3-minute track
- First run downloads ML models (~1GB)

## License

MIT

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) - Meta's audio source separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) - Spotify's audio-to-MIDI
- [librosa](https://librosa.org/) - Audio analysis
- [Strudel](https://strudel.dygy.app/) - Live coding environment
- [HTMX](https://htmx.org) - HTML-driven interactivity
