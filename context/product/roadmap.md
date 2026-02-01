# Product Roadmap: MIDI-grep

_This roadmap outlines our strategic direction based on customer needs and business goals. It focuses on the "what" and "why," not the technical "how."_

---

## Phase 1: Core Pipeline (Foundation) ✅ COMPLETE

_The highest priority features that form the core audio-to-code pipeline._

- [x] **Audio Processing Core**
  - [x] **Audio Input Handler:** Accept WAV and MP3 files via CLI arguments
  - [x] **YouTube Support:** Accept YouTube URLs and automatically download audio via yt-dlp
  - [x] **Stem Separation Engine:** Isolate piano/instrumental tracks using Demucs
  - [x] **Audio-to-MIDI Transcription:** Convert isolated stem to MIDI using Basic Pitch

- [x] **Analysis & Detection**
  - [x] **BPM Detection:** Analyze audio tempo with confidence score
  - [x] **Key Detection:** Identify musical key/scale (e.g., A minor, C major)

- [x] **Output Generation**
  - [x] **MIDI Cleanup & Quantization:** Remove noise, filter short notes, snap to grid
  - [x] **Strudel Code Generator:** Transform MIDI into playable `note()` patterns
  - [x] **Voice Separation:** Split into bass/mid/high voices with `stack()`

- [x] **CLI & Web Interface**
  - [x] **Command-Line Tool:** `midi-grep extract` with flags for input, URL, quantization
  - [x] **Web Server:** Go + Chi + HTMX with real-time SSE progress
  - [x] **Bash Scripts:** Flexible wrapper scripts with interactive prompts

---

## Phase 2: Smart Sound Selection 🔄 IN PROGRESS

_Intelligent instrument detection and appropriate Strudel sound mapping._

- [ ] **Instrument-Aware Output**
  - [ ] **Voice-Appropriate Sounds:** Map bass to `gm_acoustic_bass`/`gm_synth_bass_1`, mid to `gm_epiano1`, high to `gm_lead_2_sawtooth`
  - [ ] **Sound Style Presets:** `--style piano|synth|orchestral|electronic|auto`
  - [ ] **GM Soundfont Integration:** Use full range of 128 General MIDI instruments
  - [ ] **Dynamic Sound Assignment:** Analyze timbre characteristics to choose appropriate sounds

- [ ] **User Sound Control**
  - [ ] **Custom Sound Mapping:** `--bass-sound gm_synth_bass_1 --mid-sound gm_epiano1`
  - [ ] **Sound Palette Presets:** Pre-configured sound combinations (jazz, electronic, cinematic)
  - [ ] **Per-Voice Gain Control:** Balance volumes between bass/mid/high

- [ ] **Stem Selection**
  - [ ] **Target Instrument Flag:** `--instrument piano|vocals|bass|drums|other`
  - [ ] **Multi-Stem Output:** Extract and generate code for multiple stems separately
  - [ ] **Demucs Model Selection:** Use `htdemucs_6s` for 6-stem separation when needed

---

## Phase 3: Intelligent Extraction

_Moving from "dump all notes" to "find the actual riff."_

- [ ] **Loop Detection**
  - [ ] **Pattern Recognition:** Automatically identify repeating 1-4 bar patterns
  - [ ] **Loop Confidence Score:** Indicate how certain we are about detected loops
  - [ ] **Loop-Only Output:** Option to output just the core repeating pattern

- [ ] **Motif Extraction**
  - [ ] **Melody vs Accompaniment:** Separate lead melodic line from harmonic backing
  - [ ] **Theme Identification:** Find recurring melodic themes throughout the track
  - [ ] **Hook Detection:** Identify the most memorable/distinctive musical phrase

- [ ] **Region Selection**
  - [ ] **Time Range:** `--start 0:30 --end 1:00` to extract specific section
  - [ ] **Bar Range:** `--bars 1-8` to extract specific measures
  - [ ] **Auto-Section Detection:** Identify intro/verse/chorus boundaries

- [ ] **Smart Simplification**
  - [ ] **Note Density Control:** `--density sparse|normal|detailed`
  - [ ] **Chord Reduction:** Option to simplify complex voicings to triads
  - [ ] **Octave Consolidation:** Merge octave-doubled notes for cleaner output

---

## Phase 4: Enhanced Analysis

_Deeper musical understanding for better output._

- [ ] **Advanced Rhythm Analysis**
  - [ ] **Time Signature Detection:** Identify 3/4, 6/8, 5/4, not just 4/4
  - [ ] **Swing Detection:** Recognize swing feel and adjust quantization
  - [ ] **Groove Templates:** Apply detected groove to output patterns

- [ ] **Harmonic Analysis**
  - [ ] **Chord Recognition:** Output chord symbols (Cmaj7, Dm, G7)
  - [ ] **Chord Progression Detection:** Identify common progressions (ii-V-I, I-IV-V)
  - [ ] **Harmonic Rhythm:** Detect chord change points

- [ ] **Structural Analysis**
  - [ ] **Section Detection:** Auto-identify intro, verse, chorus, bridge, outro
  - [ ] **Section Labels:** Add comments marking sections in output
  - [ ] **Form Analysis:** Recognize AABA, verse-chorus, 12-bar blues structures

- [ ] **Confidence & Quality Metrics**
  - [ ] **Note Confidence Scores:** Per-note certainty from transcription
  - [ ] **Overall Quality Score:** Combined metric for transcription reliability
  - [ ] **Ambiguity Warnings:** Flag uncertain passages for manual review

---

## Phase 5: Rich Strudel Output

_Leveraging full Strudel capabilities for expressive output._

- [ ] **Dynamic Expression**
  - [ ] **Velocity Mapping:** Preserve note velocities with `.velocity()` or `.gain()`
  - [ ] **Dynamic Contours:** Represent crescendo/diminuendo patterns
  - [ ] **Accent Patterns:** Highlight emphasized notes

- [ ] **Articulation & Duration**
  - [ ] **Legato/Staccato:** Use `.legato()` for note lengths
  - [ ] **Sustain Pedal:** Detect and represent pedal markings
  - [ ] **Note Overlap:** Handle overlapping notes properly

- [ ] **Effects & Processing**
  - [ ] **Reverb Suggestions:** Appropriate room size based on detected style
  - [ ] **Filter Sweeps:** Suggest filter patterns for electronic styles
  - [ ] **Delay/Echo:** Rhythmic delay suggestions

- [ ] **Pattern Variations**
  - [ ] **Multiple Takes:** Generate slight variations of the same pattern
  - [ ] **Humanization Options:** Add subtle timing/velocity variations
  - [ ] **Simplified Versions:** Output both full and simplified patterns

---

## Phase 6: Alternative Outputs

_Support for other live coding environments and formats._

- [ ] **Live Coding Formats**
  - [ ] **TidalCycles:** Haskell syntax output
  - [ ] **Sonic Pi:** Ruby syntax output
  - [ ] **SuperCollider:** SC patterns output
  - [ ] **Hydra:** Visual pattern suggestions based on audio

- [ ] **Standard Music Formats**
  - [ ] **ABC Notation:** For sheet music generation
  - [ ] **MusicXML:** Universal notation exchange
  - [ ] **LilyPond:** High-quality engraving format

- [ ] **Data Formats**
  - [ ] **JSON Export:** Raw note data for custom processing
  - [ ] **CSV Export:** Spreadsheet-compatible note list
  - [ ] **OSC Messages:** Real-time streaming to other apps

---

## Phase 7: User Experience

_Making the tool more interactive and user-friendly._

- [ ] **Interactive Mode**
  - [ ] **Audio Preview:** Listen to extracted stem before generating code
  - [ ] **MIDI Preview:** Play back transcribed notes
  - [ ] **A/B Comparison:** Compare original and transcribed audio

- [ ] **Web UI Enhancements**
  - [ ] **Waveform Display:** Visual audio representation
  - [ ] **Piano Roll View:** See transcribed notes visually
  - [ ] **Real-time Editing:** Adjust notes in browser before export
  - [ ] **Direct Strudel Integration:** Send code directly to strudel.dygy.app

- [ ] **Batch Processing**
  - [ ] **Multiple Files:** Process entire folders
  - [ ] **Playlist Mode:** Process YouTube playlists
  - [ ] **Queue Management:** Background processing with notifications

---

## Phase 8: Platform & Distribution

_Making MIDI-grep accessible to more users._

- [ ] **Easy Installation**
  - [ ] **Single Binary:** Embed Python + models for zero-dependency install
  - [ ] **Homebrew Formula:** `brew install midi-grep`
  - [ ] **Docker Hub:** Pre-built images with all dependencies

- [ ] **Cloud/API**
  - [ ] **REST API:** Programmatic access for third-party apps
  - [ ] **WebSocket API:** Real-time streaming processing
  - [ ] **Rate Limiting & Auth:** Production-ready API infrastructure

- [ ] **Integrations**
  - [ ] **Browser Extension:** Extract from YouTube directly in browser
  - [ ] **VS Code Extension:** Process audio from editor
  - [ ] **Strudel Plugin:** Native integration with Strudel editor

---

## Future Ideas (Backlog)

_Ideas for future consideration, not yet prioritized._

- [ ] Real-time/streaming audio processing (live input)
- [ ] Hardware MIDI controller support
- [ ] Collaborative features (share extractions)
- [ ] Machine learning model fine-tuning for specific genres
- [ ] Custom model training on user's own samples
- [ ] Integration with DAWs (Ableton Link, VST plugin)
- [ ] Chord chart generation with lyrics sync
- [ ] Practice mode with tempo adjustment
- [ ] Comparison with existing transcriptions for accuracy benchmarking

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-02-01 | Initial roadmap with Phase 1 complete |
| 1.1 | 2024-02-01 | Added Phases 2-8 based on Strudel sound system analysis |
