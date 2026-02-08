# Product Definition: MIDI-grep

- **Version:** 1.1
- **Status:** Implemented

---

## 1. The Big Picture (The "Why")

### 1.1. Project Vision & Purpose

Enable musicians and live coders to instantly extract piano riffs from any audio source (including YouTube) and transform them into playable Strudel patterns, bridging the gap between recorded music and algorithmic composition. Delivered as a Go-powered CLI and web application with a reactive HTMX interface - no JavaScript frameworks required.

### 1.2. Target Audience

- **Live coders** using Strudel/TidalCycles who want to sample melodic ideas from existing tracks
- **Music producers** learning songs by ear or seeking inspiration
- **Hobbyist musicians** extracting riffs for practice and study
- **Educators** demonstrating music theory concepts through real-world examples

### 1.3. User Personas

- **Persona 1: "Alex the Live Coder"**
  - **Role:** Performs algorithmic music at events using Strudel
  - **Goal:** Quickly grab a piano hook from a track and remix it live on stage
  - **Frustration:** Manually transcribing riffs is tedious and breaks creative flow

- **Persona 2: "Maya the Music Student"**
  - **Role:** Self-taught pianist learning popular songs
  - **Goal:** Get accurate note transcriptions to practice piano parts
  - **Frustration:** Existing transcription tools are expensive or inaccurate

### 1.4. Success Metrics

- Users can go from YouTube URL to Strudel code in under 2 minutes
- Piano transcription accuracy of 80%+ on clear recordings
- Web UI feels responsive despite heavy backend processing
- Zero JavaScript frameworks in the frontend codebase

---

## 2. The Product Experience (The "What")

### 2.1. Core Features

- **Audio Input Handler** - Accept WAV/MP3 uploads via web UI or CLI arguments
- **YouTube Integration** - Accept YouTube URLs, auto-download via yt-dlp
- **Stem Separation Engine** - Isolate piano/instrumental track using Demucs AI
- **Audio-to-MIDI Transcription** - Convert isolated audio to MIDI via Basic Pitch
- **MIDI Cleanup & Quantization** - Remove noise, snap to grid (1/4, 1/8, 1/16)
- **BPM Detection** - Analyze tempo with confidence score
- **Key Detection** - Identify musical key/scale (e.g., A minor, E major)
- **Strudel Code Generator** - Output playable `note()` patterns
- **Go Web Server** - HTTP server using Chi router
- **HTMX Frontend** - Reactive UI with file upload, progress indicators
- **CLI with Bash Scripts** - Full-featured command-line interface

### 2.2. User Journey

**Web Flow:**
1. User opens MIDI-grep web app → sees clean upload interface
2. Drags audio file OR pastes YouTube URL
3. HTMX triggers backend processing → live status updates via SSE
4. Results render: BPM, Key, confidence scores, Strudel code
5. User clicks "Copy" → code copied to clipboard
6. Paste into Strudel → instant playback

**CLI Flow:**
```bash
# From YouTube
./scripts/midi-grep.sh extract --url "https://youtu.be/VIDEO_ID"

# From file
./scripts/midi-grep.sh extract --file track.wav --output riff.strudel

# With options
./scripts/midi-grep.sh extract -u "..." -q 8 --copy
```

---

## 3. Project Boundaries

### 3.1. What's Implemented

- CLI interface with intuitive flags and bash wrapper scripts
- Web server (Go + Chi + HTMX + Go templates)
- File upload UI with drag-and-drop support
- YouTube URL support via yt-dlp
- Real-time progress updates via SSE
- Instrumental/piano stem extraction (Demucs)
- MIDI transcription via Basic Pitch
- MIDI cleanup: velocity threshold, duration filtering, quantization
- BPM detection with confidence percentage
- Key/scale detection with confidence percentage
- Strudel code generation with proper formatting
- Copy-to-clipboard functionality
- MIDI file download
- Clean styling with PicoCSS
- Docker deployment option
- **Drum pattern extraction** with onset detection and classification (kick, snare, hi-hat)
- **Bass stem extraction** and pattern generation
- **Audio rendering** with AI-driven synthesis (Node.js Strudel renderer)
- **Self-contained HTML reports** with audio studio player:
  - Two-section stem mixer (Original + Rendered stems)
  - Solo/Mute controls per stem
  - Waveform visualizations
  - A/B comparison mode
  - Per-stem comparison charts
- **AI-driven iterative improvement** with Ollama/Claude (default: 5 iterations, 85% target)
- **Per-stem comparison** charts and analysis (bass, drums, melodic)

### 3.2. What's Out-of-Scope (Non-Goals)

- User accounts or authentication
- Persistent storage or history (stateless processing only)
- Multiple output formats (TidalCycles, Sonic Pi, ABC notation)
- Real-time/streaming audio processing
- Mobile-specific UI optimization
- Complex JavaScript frameworks
