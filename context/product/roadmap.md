# Product Roadmap: MIDI-grep

_This roadmap outlines our strategic direction based on customer needs and business goals. It focuses on the "what" and "why," not the technical "how."_

---

### Phase 1: Core Pipeline (Foundation) ✅ COMPLETE

_The highest priority features that form the core audio-to-code pipeline._

- [x] **Audio Processing Core**
  - [x] **Audio Input Handler:** Accept WAV and MP3 files via CLI arguments, validating format and providing clear error messages for unsupported files.
  - [x] **YouTube Support:** Accept YouTube URLs and automatically download audio via yt-dlp.
  - [x] **Stem Separation Engine:** Isolate piano/instrumental tracks from mixed audio using Demucs, outputting a clean audio file.
  - [x] **Audio-to-MIDI Transcription:** Convert the isolated stem to MIDI using Basic Pitch, capturing notes, velocities, and timing.

- [x] **Analysis & Detection**
  - [x] **BPM Detection:** Analyze audio tempo automatically with a confidence score, enabling accurate rhythm in generated code.
  - [x] **Key Detection:** Identify the musical key/scale (e.g., A minor, C major) to inform pattern generation.

- [x] **Output Generation**
  - [x] **MIDI Cleanup & Quantization:** Remove low-velocity noise, filter extremely short notes, and snap timing to grid (1/16 or 1/8 notes).
  - [x] **Strudel Code Generator:** Transform cleaned MIDI into playable Strudel.cc `note()` patterns with proper BPM and formatting.

- [x] **CLI Interface**
  - [x] **Command-Line Tool:** Provide a `midi-grep extract` command with flags for input file, URL, output path, and quantization options.
  - [x] **Progress Output:** Display terminal progress messages during each processing stage.
  - [x] **Bash Scripts:** Flexible wrapper scripts with options for common workflows.

---

### Phase 2: Web Experience ✅ COMPLETE

_Deliver the same power through a reactive web interface._

- [x] **Go Web Server**
  - [x] **HTTP Server Setup:** Go web server using Chi to serve the application and handle API requests.
  - [x] **File Upload Endpoint:** Accept audio file uploads with size validation and format checking.
  - [x] **YouTube URL Support:** Accept YouTube URLs in the web interface.
  - [x] **Processing Pipeline Integration:** Connect uploaded files to the existing audio processing pipeline.

- [x] **HTMX Frontend**
  - [x] **Upload Interface:** Clean, drag-and-drop file upload UI using Go templates and HTMX.
  - [x] **Real-Time Progress:** Live processing status updates via SSE.
  - [x] **Results Display:** Render BPM, key, confidence scores, and generated Strudel code on the page.

- [x] **User Experience Polish**
  - [x] **Copy-to-Clipboard:** One-click copying of generated Strudel code.
  - [x] **MIDI Download:** Download button for the cleaned MIDI file.
  - [x] **Responsive Styling:** Clean styling with PicoCSS.

---

### Phase 3: Enhancement & Deployment 🔄 IN PROGRESS

_Features planned for future consideration to improve accuracy and enable easy deployment._

- [ ] **Advanced Processing**
  - [ ] **Loop Detection:** Automatically identify repeating 1-4 bar patterns and extract the core riff.
  - [x] **Quantization Options:** Users can choose quantization level (4, 8, 16) via CLI flags.
  - [x] **Demucs Integration:** Using Demucs as primary stem separator (more modern than Spleeter).

- [x] **Deployment & Distribution**
  - [x] **Docker Container:** Dockerfile for packaging with all dependencies.
  - [ ] **Single Binary + Assets:** Embed templates/static files in Go binary for easy distribution.
  - [ ] **Pre-built Releases:** GitHub releases with binaries for macOS/Linux.

---

### Future Ideas (Backlog)

- [ ] **Multiple Output Formats:** TidalCycles, Sonic Pi, ABC notation
- [ ] **Audio Playback:** Preview extracted audio in browser
- [ ] **Batch Processing:** Process multiple files at once
- [ ] **API Endpoint:** REST API for programmatic access
- [ ] **Browser Extension:** Extract from YouTube directly in browser
