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

## Phase 2: Smart Sound Selection ✅ COMPLETE

_Intelligent instrument detection and appropriate Strudel sound mapping._

- [x] **Instrument-Aware Output**
  - [x] **Voice-Appropriate Sounds:** Map bass to `gm_acoustic_bass`/`gm_synth_bass_1`, mid to `gm_epiano1`, high to `gm_lead_2_sawtooth`
  - [x] **Sound Style Presets:** `--style piano|synth|orchestral|electronic|jazz|lofi`
  - [x] **GM Soundfont Integration:** Use full range of 128 General MIDI instruments
  - [ ] **Dynamic Sound Assignment:** Analyze timbre characteristics to choose appropriate sounds (future)

- [x] **User Sound Control**
  - [ ] **Custom Sound Mapping:** `--bass-sound gm_synth_bass_1 --mid-sound gm_epiano1` (future)
  - [x] **Sound Palette Presets:** Pre-configured sound combinations (piano, synth, orchestral, electronic, jazz, lofi)
  - [x] **Per-Voice Gain Control:** Balance volumes between bass/mid/high

- [x] **Stem Selection** ✅ PARTIAL
  - [ ] **Target Instrument Flag:** `--instrument piano|vocals|bass|drums|other`
  - [x] **Multi-Stem Output:** Extract piano + drums with `--drums` flag, drums-only with `--drums-only`
  - [x] **Separation Modes:** `separate.py --mode piano|drums|full` for different extraction modes
  - [ ] **Demucs Model Selection:** Use `htdemucs_6s` for 6-stem separation when needed

---

## Phase 3: Intelligent Extraction ✅ PARTIAL

_Moving from "dump all notes" to "find the actual riff."_

- [x] **Loop Detection** ✅
  - [x] **Pattern Recognition:** Automatically identify repeating 1-8 bar patterns
  - [x] **Loop Confidence Score:** Indicate how certain we are about detected loops (0.45 threshold)
  - [x] **Loop-Only Output:** Option to output just the core repeating pattern (`--loop-only` flag)

- [x] **Genre Auto-Detection** ✅ NEW
  - [x] **Brazilian Funk/Phonk Detection:** Scoring system analyzes BPM (125-155), vocal-range note clustering, short note durations, low bass content
  - [x] **Template-Based Generation:** When funk detected, use authentic tamborzão drum patterns + 808 bass instead of transcription
  - [x] **Style Auto-Detection:** Analyze BPM, key (minor/major), note density to auto-select jazz/soul/funk/electronic/house/trance
  - [x] **`--brazilian-funk` Override:** Manual flag to force Brazilian funk mode when auto-detection misses

- [ ] **Motif Extraction**
  - [ ] **Melody vs Accompaniment:** Separate lead melodic line from harmonic backing
  - [ ] **Theme Identification:** Find recurring melodic themes throughout the track
  - [ ] **Hook Detection:** Identify the most memorable/distinctive musical phrase

- [ ] **Region Selection**
  - [ ] **Time Range:** `--start 0:30 --end 1:00` to extract specific section
  - [ ] **Bar Range:** `--bars 1-8` to extract specific measures
  - [ ] **Auto-Section Detection:** Identify intro/verse/chorus boundaries

- [x] **Smart Simplification** ✅
  - [x] **Note Density Control:** MaxNotesPerBeat=1 for clear patterns (configurable)
  - [x] **Chord Reduction:** MaxChordSize=2 to simplify complex voicings
  - [x] **Octave Consolidation:** Merge octave-doubled notes for cleaner output
  - [x] **Close Note Merging:** MergeThreshold=0.1 to merge notes within 100ms

---

## Phase 4: Enhanced Analysis ✅ MOSTLY COMPLETE

_Deeper musical understanding for better output._

- [x] **Advanced Rhythm Analysis** ✅
  - [x] **Time Signature Detection:** Identify 3/4, 6/8, 5/4, 2/4, 7/8 (accent-based detection)
  - [x] **Swing Detection:** Recognize swing feel (1.0=straight to 2.0=triplet), apply `.swing()`
  - [ ] **Groove Templates:** Apply detected groove to output patterns (future)

- [x] **Harmonic Analysis** ✅
  - [x] **Chord Recognition:** Detect chord types (major, minor, 7th, 9th, dim, aug, sus)
  - [x] **Chord Progression Detection:** Track chord changes over time with `arrange()`
  - [x] **Harmonic Rhythm:** Detect chord change points and durations
  - [x] **Key Detection from Chords:** Infer key from chord progression analysis

- [x] **Structural Analysis** ✅ COMPLETE
  - [x] **Section Detection:** Auto-detect intro, verse, chorus based on energy
  - [x] **Section Labels:** Time-stamped section markers in output header
  - [x] **Form Analysis:** Recognize AABA, ABA, AABB, verse-chorus, 12-bar blues structures

- [ ] **Confidence & Quality Metrics**
  - [ ] **Note Confidence Scores:** Per-note certainty from transcription
  - [ ] **Overall Quality Score:** Combined metric for transcription reliability
  - [ ] **Ambiguity Warnings:** Flag uncertain passages for manual review

- [x] **Drum Extraction** ✅ NEW
  - [x] **Drum Stem Separation:** Extract drum track using Demucs full mode
  - [x] **Drum Hit Detection:** Onset detection with librosa
  - [x] **Drum Classification:** Spectral analysis to identify bd/sd/hh/oh/cp
  - [x] **Drum Pattern Generation:** Generate `s()` patterns with `.bank()`
  - [x] **Drum Kit Selection:** TR-808, TR-909, LinnDrum, Acoustic, Lo-fi presets
  - [x] **Combined Output:** Merge melodic + drum patterns in `stack()`

---

## Phase 5: Rich Strudel Output ✅ COMPLETE

_Leveraging full Strudel capabilities for expressive output._

- [x] **Dynamic Expression**
  - [x] **Velocity Mapping:** Proper `.velocity()` patterns (0-1 range) per voice
  - [x] **Per-Voice Gain:** Base gain for voice level balance
  - [x] **Dynamic Range Expansion:** Style-specific velocity curve expansion
  - [x] **Accent Patterns:** Beat emphasis (downbeat, backbeat, offbeat) per style
  - [x] **Compressor:** `.compressor()` for dynamics control (electronic)

- [x] **ADSR Envelopes**
  - [x] **Per-Voice Envelopes:** Attack/decay/sustain/release per voice type
  - [x] **Style-Specific:** Longer attacks for orchestral, punchy for electronic
  - [x] **Voice-Appropriate:** Quick attack for bass, longer release for highs

- [ ] **Articulation & Duration** (future)
  - [ ] **Legato/Staccato:** Use `.legato()` for note lengths
  - [ ] **Sustain Pedal:** Detect and represent pedal markings
  - [ ] **Note Overlap:** Handle overlapping notes properly

- [x] **Effects & Processing**
  - [x] **Per-Voice Effects:** Voice-specific effect chains (filter, pan, reverb, delay, envelope)
  - [x] **Reverb Per-Voice:** Register-appropriate room size (less on bass, more on highs)
  - [x] **Filter Per-Voice:** HPF/LPF based on voice register
  - [x] **Delay on Highs:** Rhythmic delay on high voice for space
  - [x] **Stereo Panning:** Bass centered, mid/high with LFO stereo movement

- [x] **LFO & Modulation**
  - [x] **Multiple LFO Shapes:** sine, saw, tri, square, perlin (smooth random), rand
  - [x] **Style-Specific LFOs:** perlin for jazz/lofi (organic), saw for electronic (rhythmic)
  - [x] **Per-Voice LFO Speed:** Different speeds per voice for movement

- [x] **Style-Specific FX**
  - [x] **Synth:** Phaser, vibrato, FM synthesis
  - [x] **Electronic:** Phaser, distort, FM synthesis
  - [x] **Orchestral:** Vibrato with longer depth
  - [x] **Jazz:** Subtle vibrato
  - [x] **Lofi:** Bitcrush (crush), sample rate reduction (coarse)

- [x] **FM Synthesis**
  - [x] **FM Index:** `.fm()` for modulation brightness
  - [x] **FM Harmonicity:** `.fmh()` for timbre control
  - [x] **FM Envelope:** `.fmdecay()`, `.fmsustain()` for FM dynamics

- [x] **Tremolo & Filter Modulation**
  - [x] **Tremolo:** `.tremolo()`, `.tremolodepth()` for amplitude modulation
  - [x] **Tremolo Shape:** `.tremoloshape()` for waveform selection
  - [x] **Filter Envelope:** `.lpenv()`, `.lpattack()`, `.lpdecay()` for dynamic sweeps

- [x] **Sidechain/Ducking**
  - [x] **Duck Effect:** `.duck()` for sidechain-style pumping
  - [x] **Duck Parameters:** `.duckattack()`, `.duckdepth()` for timing/intensity

- [x] **Pattern Transforms**
  - [x] **Swing:** `.swing()` for jazz style shuffle feel
  - [x] **DegradeBy:** `.degradeBy()` for lofi random note removal
  - [x] **Iter:** `.iter()` for cyclic pattern variation (electronic/lofi)
  - [x] **Jux:** `.jux(rev)` for stereo width on synth style high voice
  - [x] **Ply:** `.ply()` for rhythmic density on electronic bass
  - [x] **Sometimes/Rarely:** `.sometimes()`, `.rarely()` for occasional effect variations (lofi)

- [x] **Articulation & Duration**
  - [x] **Clip/Legato:** `.clip()` for note duration (staccato/legato/sustained)
  - [x] **Style-Specific Legato:** Shorter for electronic, longer for orchestral

- [x] **Accumulation Effects**
  - [x] **Echo/Stutter:** `.echo(times, time, feedback)` for rhythmic repeats
  - [x] **Superimpose:** `.superimpose(add(0.03))` for detuned layering
  - [x] **Off:** `.off(time, add(interval))` for harmonic layering with offset
  - [x] **Scale:** `.scale("E:minor")` helper for key-aware quantization
  - [x] **Layer:** `.layer()` for parallel transformations (orchestral)
  - [x] **EchoWith:** `.echoWith()` for pitch-shifted echoes (electronic)

- [x] **Section Analysis**
  - [x] **Section Detection:** Auto-detect intro, verse, chorus based on energy
  - [x] **Section Comments:** Time-stamped section markers in output header
  - [x] **Energy Curve:** Calculate density + velocity + register energy per bar

- [ ] **Pattern Variations** (future)
  - [ ] **Multiple Takes:** Generate slight variations of the same pattern
  - [ ] **Humanization Options:** Add subtle timing/velocity variations
  - [ ] **Simplified Versions:** Output both full and simplified patterns

- [x] **Arrangement-Based Generation** ✅ NEW
  - [x] **Chord Variable:** `let chords = arrange([16, "<Cm7!4 Bb7>"], ...).chord()`
  - [x] **Voicings:** `.dict('ireal-ext').voicing()` for proper chord voicings
  - [x] **Chord-Derived Parts:** Pad, bass, arpeggio, melody all from same chord source
  - [x] **Root Notes Bass:** `chords.rootNotes(1)` for bass following chord roots
  - [x] **Arpeggiated Parts:** `n(run(8)).set(chords).voicing()` for arpeggios
  - [x] **Musical Variations:** `.sometimesBy()`, `.lastOf()`, `.mask()` patterns
  - [x] **Perlin Modulation:** `perlin.range()` for organic filter/parameter movement
  - [x] **Build Masks:** `.mask("<1!32 0!16>")` for arrangement drops/builds
  - [x] **CLI Flag:** `--arrange` to enable arrangement-based output

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

## Phase 7: User Experience ✅ PARTIAL

_Making the tool more interactive and user-friendly._

- [x] **Output Caching & Versioning** ✅ NEW
  - [x] **Versioned Outputs:** Each run creates v001, v002, etc. in `.cache/stems/{key}/`
  - [x] **Metadata Storage:** BPM, key, style, genre, notes, drum hits, timestamp saved in JSON
  - [x] **Latest Symlink:** `output_latest.strudel` always points to newest version
  - [x] **Iteration Support:** Compare previous outputs to track improvements

- [x] **Audio Rendering** ✅ NEW
  - [x] **WAV Synthesis:** `--render` flag generates audio preview from patterns
  - [x] **Drum Synthesis:** Kick (808 pitch envelope), snare, hi-hat with proper envelopes
  - [x] **Bass Synthesis:** Sawtooth + sub-octave, low-pass filtered
  - [x] **Synth Voices:** Square/saw with ADSR envelopes
  - [x] **Cache Integration:** `--render auto` saves to cache directory with versioning

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

## Phase 9: ML Model Customization

_Fine-tune transcription for specific genres and user samples._

- [ ] **Genre-Specific Models**
  - [ ] **Jazz Piano Model:** Train on jazz voicings, swing timing, blues notes
  - [ ] **Electronic/Synth Model:** Better detection of synth leads, arps, bass
  - [ ] **Classical Piano Model:** Handle complex classical dynamics and pedaling
  - [ ] **Lo-Fi/Hip-Hop Model:** Recognize sampled/chopped piano patterns

- [ ] **Custom Training Pipeline**
  - [ ] **User Sample Collection:** CLI to gather training pairs (audio + correct MIDI)
  - [ ] **Transfer Learning:** Fine-tune Basic Pitch on user's own recordings
  - [ ] **Model Export:** Save custom models for reuse
  - [ ] **A/B Testing:** Compare stock vs fine-tuned model accuracy

- [ ] **Training Data Tools**
  - [ ] **MIDI-Audio Alignment:** Tool to align existing MIDI with audio
  - [ ] **Annotation Helper:** Simple UI to correct transcription errors for training
  - [ ] **Dataset Builder:** Collect and organize training pairs

- [ ] **Model Management**
  - [ ] **Model Registry:** Store and switch between trained models
  - [ ] **`--model` Flag:** Select which model to use for extraction
  - [ ] **Model Sharing:** Export/import trained models

---

## Future Ideas (Backlog)

_Ideas for future consideration, not yet prioritized._

- [ ] Real-time/streaming audio processing (live input)
- [ ] Collaborative features (share extractions)
- [ ] Chord chart generation with lyrics sync
- [ ] Comparison with existing transcriptions for accuracy benchmarking

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-02-01 | Initial roadmap with Phase 1 complete |
| 1.1 | 2024-02-01 | Added Phases 2-8 based on Strudel sound system analysis |
| 1.2 | 2026-02-01 | Phase 2 complete: Sound style presets with GM soundfonts |
| 1.3 | 2026-02-01 | Phase 5 complete: Dynamic Strudel output with per-voice effects, velocity dynamics, section detection |
| 1.4 | 2026-02-01 | Phase 5 enhanced: ADSR envelopes, multiple LFO shapes (perlin/saw/tri), style-specific FX (phaser/crush/coarse/vibrato/distort), pattern transforms (swing/degradeBy), proper .velocity() |
| 1.5 | 2026-02-01 | Phase 5 complete: Articulation (.clip), accumulation (.echo, .superimpose, .off), scale quantization helper |
| 1.6 | 2026-02-01 | FM synthesis (.fm, .fmh, .fmdecay, .fmsustain) for synth/electronic styles |
| 1.7 | 2026-02-01 | Tremolo and filter envelope for amplitude/filter modulation |
| 1.8 | 2026-02-01 | Sidechain/ducking effect for electronic style pumping |
| 1.9 | 2026-02-01 | Iter pattern transform for cyclic variation |
| 1.10 | 2026-02-01 | Jux pattern transform for synth stereo width |
| 1.11 | 2026-02-01 | Ply pattern transform for electronic bass density |
| 1.12 | 2026-02-01 | Sometimes/rarely modifiers for lofi variations |
| 1.13 | 2026-02-01 | Enhanced jazz style with off, sometimes/rarely |
| 1.14 | 2026-02-01 | Enhanced orchestral style with superimpose, off, sometimes |
| 1.15 | 2026-02-01 | Enhanced dynamics: accent patterns, range expansion, compressor |
| 1.16 | 2026-02-02 | Added layer(), echoWith(), cosine LFO from Strudel utilities |
| 1.17 | 2026-02-02 | Prettified code output: each method on its own line |
| 1.18 | 2026-02-02 | Raw synthesizer styles: raw, chiptune, ambient, drone using Strudel oscillators (sawtooth, square, triangle, sine) |
| 1.19 | 2026-02-02 | Sample-based styles: mallets (vibraphone/marimba), plucked (harp), keys (Salamander piano), pad (warm/choir), percussive (timpani) |
| 1.20 | 2026-02-02 | Genre-specific styles: synthwave (80s retro), darkwave (atmospheric), minimal (sparse), industrial (harsh), newage (ethereal) |
| 1.21 | 2026-02-02 | Advanced effects: slide/portamento, ftype (ladder/24db filter), orbit routing, begin/end sample position, speed control |
| 1.22 | 2026-02-02 | Style auto-detection: analyze BPM, key (minor/major), note density to auto-select jazz/soul/funk/electronic/house/trance |
| 1.23 | 2026-02-02 | Extended sound palette: supersaw, ZZFX synths (z_sawtooth, z_square, z_triangle), wavetables (wt_digital, wt_vgame), noise (white, pink, brown, crackle) |
| 1.24 | 2026-02-02 | Enhanced style effects: ring modulation, chorus, leslie, shape/saturation, pitch envelope for jazz/soul/funk/electronic |
| 1.25 | 2026-02-02 | Improved loop detection: 8-bar loops, lower confidence threshold (0.45), better pattern matching |
| 1.26 | 2026-02-02 | Aggressive note simplification: MaxChordSize=2, MaxNotesPerBeat=1, MergeThreshold=0.1 for clearer patterns |
| 1.27 | 2026-02-02 | Drum extraction: separate.py --mode, detect_drums.py, drum kit presets (TR-808/909, Linn, acoustic, lofi) |
| 1.28 | 2026-02-02 | Drum pattern generation: s() patterns with .bank(), .cut(1) for hi-hats, .room() for ambience |
| 1.29 | 2026-02-02 | Chord detection: identify major/minor/7th/9th/dim/aug/sus chords from MIDI notes |
| 1.30 | 2026-02-02 | Arrangement-based generation: arrange(), .voicings(), .set(chords), chord-derived parts |
| 1.31 | 2026-02-02 | Musical variations: .sometimesBy(), .lastOf(), .mask(), perlin.range() modulation |
| 1.32 | 2026-02-02 | Time signature detection: analyze beat accent patterns for 4/4, 3/4, 6/8, 2/4, 5/4, 7/8 |
| 1.33 | 2026-02-02 | Swing detection: eighth-note timing deviation analysis, swing ratio (1.0-2.0), .swing() effect |
| 1.34 | 2026-02-02 | Form analysis: detect AABA, ABA, AABB, verse-chorus, 12-bar blues from section patterns |
| 1.35 | 2026-02-02 | Separate output modes: --output-mode=stack/separate/named for hushable voices and let bindings |
| 1.36 | 2026-02-03 | Stem caching: cache by URL/file hash in .cache/stems/, auto-invalidate when scripts change |
| 1.37 | 2026-02-03 | Bar array output: let bass/mid/high/drums = [...] with effect functions, cat(...map()) for playback |
| 1.38 | 2026-02-03 | Drums in default output: drumsFx with bank(), s() patterns integrated in stack |
| 1.39 | 2026-02-03 | Chord mode: --chords flag for electronic/funk music using chord detection instead of note transcription |
| 1.40 | 2026-02-03 | Brazilian Funk auto-detection: scoring system for BPM 125-155, vocal-range notes, short durations, low bass → template generation |
| 1.41 | 2026-02-03 | Output caching with versioning: save Strudel code + metadata (v001, v002, ...) to .cache/stems/{key}/ |
| 1.42 | 2026-02-03 | Audio rendering: --render flag to synthesize WAV from patterns (kick, snare, hh, bass, synth voices) |
| 1.43 | 2026-02-03 | Block comment variations (/* */) for easier live coding uncommenting |
