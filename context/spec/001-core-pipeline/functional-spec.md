# Functional Specification: Core Pipeline (Phase 1)

- **Roadmap Item:** Phase 1: Core Pipeline — Audio Processing, Analysis, Output Generation, CLI Interface
- **Status:** Draft
- **Author:** Poe (AI Product Analyst)

---

## 1. Overview and Rationale (The "Why")

### Problem Statement

Musicians and live coders who want to sample piano riffs from existing tracks face a tedious, manual process:
1. They must use expensive DAW software to isolate instruments
2. They transcribe notes by ear (slow, error-prone)
3. They manually convert to their live coding syntax (Strudel.cc)

This workflow breaks creative flow and requires specialized skills that hobbyists may not have.

### Solution

MIDI-grep provides a single command that automates the entire pipeline: audio in → piano isolation → MIDI transcription → Strudel code out. Users get playable code in under 2 minutes.

### Success Criteria

- Users can process an audio file and receive Strudel code with a single CLI command
- Processing completes in under 2 minutes for a typical 3-minute track
- Piano transcription achieves 80%+ accuracy on clear recordings
- Output code plays correctly in Strudel.cc without manual edits

---

## 2. Functional Requirements (The "What")

### 2.1 Audio Input Handler

**As a** user, **I want to** provide an audio file to the tool, **so that** I can extract the piano part from it.

**Acceptance Criteria:**

- [ ] The tool accepts WAV files (16-bit and 24-bit, 44.1kHz and 48kHz sample rates)
- [ ] The tool accepts MP3 files (128kbps to 320kbps)
- [ ] The tool validates the file exists before processing
- [ ] The tool validates the file format by checking magic bytes (not just extension)
- [ ] If the file does not exist, the tool displays: `Error: File not found: [path]`
- [ ] If the file format is unsupported, the tool displays: `Error: Unsupported format. Please provide a WAV or MP3 file.`
- [ ] If the file is corrupted or unreadable, the tool displays: `Error: Could not read audio file. The file may be corrupted.`
- [ ] Maximum supported file size: 100MB
- [ ] If file exceeds size limit: `Error: File too large. Maximum size is 100MB.`

---

### 2.2 Stem Separation Engine

**As a** user, **I want** the piano track isolated from the full mix, **so that** the transcription focuses only on piano notes.

**Acceptance Criteria:**

- [ ] The tool separates audio into stems using Spleeter 5-stem model (vocals, drums, bass, piano, other)
- [ ] The tool extracts and saves the piano stem as a temporary WAV file
- [ ] If Spleeter is not installed or fails, the tool displays: `Error: Stem separation failed. Please ensure Spleeter is installed.`
- [ ] During separation, the tool displays progress: `Separating stems... (this may take a moment)`
- [ ] The separation process has a timeout of 5 minutes; if exceeded: `Error: Stem separation timed out.`

---

### 2.3 BPM Detection

**As a** user, **I want** the tempo detected automatically, **so that** the generated Strudel code plays at the correct speed.

**Acceptance Criteria:**

- [ ] The tool analyzes the audio and detects BPM (beats per minute)
- [ ] BPM detection returns a value between 40 and 240 BPM
- [ ] The tool provides a confidence score (0-100%) for the detected BPM
- [ ] During detection, the tool displays: `Detecting tempo...`
- [ ] Upon completion, the tool displays: `Detected BPM: [value] (confidence: [X]%)`
- [ ] If BPM cannot be determined, the tool defaults to 120 BPM and displays: `Warning: Could not determine BPM. Using default: 120 BPM`

---

### 2.4 Key Detection

**As a** user, **I want** the musical key detected automatically, **so that** I can use scale-based patterns in Strudel.

**Acceptance Criteria:**

- [ ] The tool analyzes the audio and detects the musical key (e.g., "A minor", "C major")
- [ ] The tool supports detection of all 12 major and 12 minor keys
- [ ] The tool provides a confidence score (0-100%) for the detected key
- [ ] During detection, the tool displays: `Detecting key...`
- [ ] Upon completion, the tool displays: `Detected key: [key] (confidence: [X]%)`
- [ ] If key cannot be determined, the tool displays: `Warning: Could not determine key.` and proceeds without key information

---

### 2.5 Audio-to-MIDI Transcription

**As a** user, **I want** the piano audio converted to MIDI notes, **so that** I have a structured representation of the melody.

**Acceptance Criteria:**

- [ ] The tool converts the isolated piano stem to MIDI using Basic Pitch
- [ ] The resulting MIDI contains note pitches (MIDI note numbers), velocities (0-127), start times, and durations
- [ ] If Basic Pitch is not installed or fails, the tool displays: `Error: Transcription failed. Please ensure Basic Pitch is installed.`
- [ ] During transcription, the tool displays: `Transcribing piano to MIDI...`
- [ ] The transcription process has a timeout of 3 minutes; if exceeded: `Error: Transcription timed out.`

---

### 2.6 MIDI Cleanup & Quantization

**As a** user, **I want** the MIDI cleaned up and quantized, **so that** the output is musically coherent and not cluttered with noise.

**Acceptance Criteria:**

- [ ] The tool removes notes with velocity below 20 (noise threshold)
- [ ] The tool removes notes shorter than 30ms (artifact threshold)
- [ ] The tool quantizes note start times to the nearest grid division
- [ ] Default quantization: 1/16 notes
- [ ] User can specify quantization via `--quantize` flag: `4` (quarter), `8` (eighth), `16` (sixteenth)
- [ ] The tool preserves the relative dynamics (velocity differences) between notes
- [ ] During cleanup, the tool displays: `Cleaning and quantizing MIDI...`
- [ ] Upon completion, the tool displays: `Cleanup complete: [X] notes retained, [Y] notes removed`

---

### 2.7 Strudel Code Generator

**As a** user, **I want** the MIDI converted to Strudel.cc syntax, **so that** I can immediately use it in my live coding session.

**Acceptance Criteria:**

- [ ] The tool generates valid Strudel.cc code using the `note()` function
- [ ] Notes are represented in scientific pitch notation (e.g., "C4", "A#3", "Gb5")
- [ ] The generated code includes `setcps()` with the correct tempo derived from BPM
- [ ] The generated code uses `.sound("piano")` as the default instrument
- [ ] The code includes basic effects: `.room(0.3).size(0.6)` for subtle reverb
- [ ] If detected key is available, include a comment: `// Key: [detected key]`
- [ ] Output is formatted with proper indentation for readability
- [ ] Example output format:
  ```javascript
  // MIDI-grep output
  // BPM: 120, Key: A minor
  setcps(120/60/4)

  $: note("a3 c4 e4 a3 c4 e4 g3 b3 d4")
    .sound("piano")
    .room(0.3).size(0.6)
  ```

---

### 2.8 CLI Interface

**As a** user, **I want** a simple command-line interface, **so that** I can run the tool from my terminal.

**Acceptance Criteria:**

- [ ] The tool provides a main command: `midi-grep`
- [ ] The tool provides a subcommand: `midi-grep extract`
- [ ] Required flag: `--input` or `-i` (path to audio file)
- [ ] Optional flag: `--output` or `-o` (path for Strudel output file; default: stdout)
- [ ] Optional flag: `--midi-out` (path to save cleaned MIDI file)
- [ ] Optional flag: `--quantize` or `-q` (quantization level: 4, 8, 16; default: 16)
- [ ] Optional flag: `--help` or `-h` (display usage information)
- [ ] Optional flag: `--version` or `-v` (display version number)
- [ ] If `--input` is not provided: `Error: --input flag is required. Usage: midi-grep extract --input <file>`
- [ ] Example usage displayed in help:
  ```
  Usage: midi-grep extract [flags]

  Flags:
    -i, --input string      Input audio file (WAV or MP3) [required]
    -o, --output string     Output file for Strudel code (default: stdout)
        --midi-out string   Save cleaned MIDI to file
    -q, --quantize int      Quantization (4, 8, or 16) (default: 16)
    -h, --help              Help for extract
  ```

---

### 2.9 Progress Output

**As a** user, **I want** to see progress during processing, **so that** I know the tool is working and how far along it is.

**Acceptance Criteria:**

- [ ] The tool displays the current processing stage in real-time
- [ ] Progress stages displayed in order:
  1. `[1/5] Validating input file...`
  2. `[2/5] Separating stems... (this may take a moment)`
  3. `[3/5] Analyzing audio (BPM, key)...`
  4. `[4/5] Transcribing piano to MIDI...`
  5. `[5/5] Generating Strudel code...`
- [ ] Upon successful completion: `Done! Strudel code generated successfully.`
- [ ] If output file specified: `Output saved to: [path]`
- [ ] Total processing time displayed: `Completed in [X.X] seconds`

---

## 3. Scope and Boundaries

### In-Scope

- WAV and MP3 file input via CLI
- Piano stem separation using Spleeter
- BPM detection with confidence score
- Key detection with confidence score
- Audio-to-MIDI transcription using Basic Pitch
- MIDI cleanup (velocity/duration filtering)
- MIDI quantization (1/4, 1/8, 1/16 grid)
- Strudel.cc code generation with `note()` patterns
- CLI with flags for input, output, quantization
- Terminal progress output
- Error messages for common failure modes

### Out-of-Scope (Separate Roadmap Items)

- **Phase 2: Web Experience**
  - Go web server
  - HTMX frontend
  - File upload via browser
  - Real-time progress via SSE
  - Copy-to-clipboard UI
  - MIDI download button
  - Responsive styling

- **Phase 3: Enhancement & Deployment**
  - Loop detection
  - User-selectable quantization options in UI
  - Fallback to Demucs when Spleeter fails
  - Docker container packaging
  - Single binary distribution

- **Explicitly Not Included in V1**
  - YouTube URL input (legal complexity)
  - Other instruments (drums, bass, vocals)
  - Multiple output formats (TidalCycles, Sonic Pi)
  - Audio playback
  - User accounts or history
