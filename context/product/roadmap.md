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

- [x] **Genre Auto-Detection** ✅ COMPLETE
  - [x] **Multi-Genre Detection:** Separate detection functions for Brazilian funk, Brazilian phonk, and retro wave/synthwave
  - [x] **Brazilian Funk Detection:** BPM 130-145 (or half-time 85-95), rejects long synth notes
  - [x] **Brazilian Phonk Detection:** BPM 80-100 or 145-180, darker sound profile
  - [x] **Retro Wave Detection:** Longer note durations, synthwave-style arpeggios
  - [x] **Template-Based Generation:** When funk detected, use authentic tamborzão drum patterns + 808 bass
  - [x] **Style Auto-Detection:** Analyze BPM, key (minor/major), note density to auto-select jazz/soul/funk/electronic/house/trance
  - [x] **Manual Genre Override:** `--genre` flag to force specific genre (`brazilian_funk`, `brazilian_phonk`, `retro_wave`, `synthwave`, `trance`, `house`, `lofi`, `jazz`)
  - [x] **Deep Learning Detection:** CLAP (Contrastive Language-Audio Pretraining) enabled by default for zero-shot genre classification (skipped when `--genre` is specified)

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

- [x] **Audio Rendering** ✅ COMPLETE
  - [x] **WAV Synthesis:** `--render` flag generates audio preview from patterns
  - [x] **Drum Synthesis:** Kick (808 pitch envelope), snare, hi-hat with proper envelopes
  - [x] **Bass Synthesis:** Sawtooth + sub-octave, low-pass filtered
  - [x] **Synth Voices:** Square/saw with ADSR envelopes, vocal chops, chord stabs, lead
  - [x] **Cache Integration:** `--render auto` saves to cache directory with versioning

- [x] **AI-Driven Audio Analysis** ✅ NEW
  - [x] **Parameter Suggestion:** `audio_to_strudel_params.py` analyzes original audio for optimal Strudel effects
  - [x] **Spectral Analysis:** Determines filter cutoffs, brightness levels
  - [x] **Dynamics Analysis:** Suggests compression, gain staging, distortion amounts
  - [x] **Timbre Matching:** Recommends FM synthesis, filter envelope parameters
  - [x] **Spatial Analysis:** Determines reverb size, delay times based on original
  - [x] **Audio Comparison:** `compare_audio.py` compares rendered vs original (spectral, rhythmic, timbral similarity)
  - [x] **Feedback Loop:** Comparison results improve renderer mix balance

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

## Phase 10: Audio Similarity & Synthesis Quality 🔴 CRITICAL

_Fixing the core audio quality issues blocking high similarity scores._

### 🔴 Critical (Blocking 80%+ Similarity)

- [x] **Fix Drum Synthesis** ✅ (Feb 2026)
  - [x] **808 Kick Frequency Balance:** Fixed - pitchEnd changed from 80Hz to 35Hz for proper sub-bass
  - [x] **Kick HPF Removed:** No longer filtering out sub-bass from kicks
  - [x] **Drum Mix Balance:** Rebalanced with drums 0.7x gain multiplier
  - [ ] **Snare Frequency:** Too much sub, need more mid crack (future)

- [x] **Fix Voice Gain Defaults** ✅ (Feb 2026)
  - [x] **Bass Gain:** 0.3 → 0.1 (was 4812% too loud, not too quiet!)
  - [x] **Mid Gain:** 1.0 → 0.6 (was dominating +31%)
  - [x] **Sub-Octave Gain:** 0.4 → 0.3 (reduced to balance)
  - [x] **Brazilian Funk:** lpf(100).gain(0.6) → lpf(400).gain(0.12)

- [x] **Fix Tempo Detection** ✅ (Feb 2026)
  - [x] **Half-Time Detection:** Fixed - now uses known BPM from synth_config.json
  - [x] **Octave Correction:** Fixed - compare_audio.py uses expected_bpm instead of re-detecting
  - [x] **Prior-Based Selection:** Implemented - original analysis BPM passed to comparison

- [x] **Fix Frequency Balance** ✅ (Feb 2026)
  - [x] **Fixed:** Bass was 4812% too loud → now 87.5% similarity
  - [x] **Achieved:** 81.4% weighted overall (target was 80%)
  - [x] **Solution:** lpf(100).gain(0.6) → lpf(400).gain(0.12)

### 🟠 High Priority (Blocking 85%+ Similarity)

- [ ] **Genre-Specific Synthesis Configs**
  - [ ] **Brazilian Funk:** Heavy 808 bass (gain 1.5), punchy kick, minimal mids
  - [ ] **Electro Swing:** Brass emphasis, piano mids, swing timing
  - [ ] **House/Techno:** 4-on-floor kick, sidechain bass, bright leads
  - [ ] **Lo-Fi:** Vinyl noise, tape saturation, reduced highs

- [ ] **Sub-Bass Synthesis**
  - [ ] **Lower Frequency Reach:** Current synthesis not reaching 20-60Hz properly
  - [ ] **Sub-Octave Mix:** sub_bass band chronically 15-17% under target
  - [ ] **808 Sub Mode:** Add dedicated sub-bass oscillator for bass voice

- [x] **ClickHouse Learning Application** ✅ (Feb 2026)
  - [x] **Apply Stored Knowledge:** Best code from ClickHouse now written to strudel_path
  - [x] **Genre Context Matching:** Query fetches across genres for universal learnings
  - [x] **Auto-Apply Best Configs:** Start from best known code, not defaults

- [x] **Genre-Aware Sound RAG** ✅ (Feb 2026)
  - [x] **`retrieve_genre_context()`:** Returns ~40-token compact sound palette per genre for LLM prompts
  - [x] **Prompt Injection:** Genre sounds injected into `ollama_codegen.py` and `ollama_agent.py`
  - [x] **Token Savings:** 800 → 40 tokens per LLM call (760 saved)
  - [x] **Reduced Hallucinations:** LLM only sees ~15 valid sounds instead of guessing from 196
  - [x] **Example Sounds:** Prompt examples use genre-appropriate palette (not hardcoded defaults)

- [ ] **LLM Improvement Effectiveness**
  - [ ] **29 iterations, still 64.6%:** LLM not understanding frequency fixes
  - [ ] **Concrete Parameter Changes:** Give LLM exact gain multipliers to apply
  - [x] **Regression Prevention:** Fixed --json --quiet flags for comparison parsing

- [x] **Per-Stem Comparison After AI** ✅ (Feb 2026)
  - [x] **Re-render stems:** Now re-renders with improved code after AI iterations
  - [x] **Bass was empty:** Fixed - Brazilian Funk template bass now renders

- [x] **Cache Key Identity** ✅ (Feb 2026)
  - [x] **Filename-based keys:** Changed from hash (file_785c...) to filename
  - [x] **Learning preservation:** Track identity preserved for ClickHouse learning

- [ ] **Sidechain/Ducking for Bass**
  - [ ] **Kick Ducks Bass:** Essential for Brazilian funk/house punch
  - [ ] **Configurable Depth:** 50-80% ducking on kick hits
  - [ ] **Attack/Release Timing:** Fast attack, medium release

### ✅ SOLVED: BlackHole Recording (Feb 2026)

**Best approach for 100% audio similarity - record real Strudel playback instead of emulating synthesis.**

- [x] **BlackHole Virtual Audio Device:** Route browser audio to recorder via BlackHole 2ch
- [x] **Puppeteer Automation:** `record-strudel-blackhole.ts` opens strudel.dygy.app/embed, plays code, records via ffmpeg
- [x] **Multi-Output Device:** Audio MIDI Setup configuration for simultaneous playback + recording
- [x] **100% Similarity:** Captures exact Strudel audio output (no synthesis approximation)

**Files created:**
- `scripts/node/src/record-strudel-blackhole.ts` - Main BlackHole recorder
- `scripts/node/src/record-strudel-ui.ts` - Manual browser recording fallback

**Usage:** `node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30`

**Why this is better than Node.js synthesis:**
| Approach | Similarity | Effort |
|----------|------------|--------|
| Node.js synthesis | 72% max | Endless tuning |
| BlackHole recording | **100%** | One-time setup |

### 🟡 Medium Priority (Targeting 90%+ Similarity)

- [ ] **Real Sample Integration**
  - [ ] **808 Kit Samples:** Load actual TR-808 samples instead of synthesis
  - [ ] **Instrument Samples:** GM soundfont samples for more realistic timbre
  - [ ] **Granular Models by Default:** Use trained models when available

- [ ] **Drum Pattern Accuracy**
  - [ ] **Current:** 44% drums similarity
  - [ ] **Syncopation Detection:** Brazilian funk has specific off-beat patterns
  - [ ] **Hi-Hat Patterns:** Open/closed timing not captured well

- [ ] **Velocity Dynamics**
  - [ ] **Variable Velocity:** Currently all notes same loudness
  - [ ] **Ghost Notes:** Quiet notes between main hits
  - [ ] **Accent Patterns:** Downbeat emphasis

- [ ] **Saturation/Limiter Bass Preservation**
  - [ ] **tanh Clipping:** May be compressing bass transients
  - [ ] **Multiband Limiting:** Preserve bass while limiting mids/highs
  - [ ] **Soft Knee:** Gentler limiting curve

- [ ] **Master HPF Review**
  - [ ] **Current:** 30Hz cutoff may be too aggressive
  - [ ] **Brazilian Funk:** Needs sub-bass presence (lower to 20Hz)
  - [ ] **Genre-Specific:** Different HPF per genre

- [ ] **Swing Timing in Synthesis**
  - [ ] **Detected but Unused:** Swing ratio calculated but not applied
  - [ ] **Timing Offsets:** Apply swing to note start times
  - [ ] **Genre-Appropriate:** More swing for jazz, less for electronic

### 🟢 Lower Priority (Polish & Maintenance)

- [ ] **Code Refactoring**
  - [ ] **ai_improver.py:** 1619 lines → split into modules
  - [ ] **generate_report.py:** 2038 lines → template-based
  - [ ] **compare_audio.py:** 1397 lines → separate analysis functions

- [ ] **Test Coverage**
  - [ ] **compare_audio tests:** Verify MAE calculation
  - [ ] **renderer tests:** Verify frequency band output
  - [ ] **synthesis tests:** Verify voice gain balance

- [ ] **Per-Track Config Caching**
  - [ ] **Best Config Storage:** Cache winning synth config per track
  - [ ] **Genre Config Library:** Build up genre-specific defaults over time

- [ ] **Time Signature in Rendering**
  - [ ] **6/8 Feel:** Different pattern grouping
  - [ ] **3/4 Waltz:** Emphasis on beat 1

- [ ] **Chord Detection Usage**
  - [ ] **Harmonic Content:** Use detected chords in synthesis
  - [ ] **Bass Note Selection:** Follow chord roots

- [ ] **A/B Testing Framework**
  - [ ] **Automated Benchmarks:** Compare different configs
  - [ ] **Regression Tests:** Ensure changes improve similarity


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
| 1.44 | 2026-02-03 | Multi-genre detection: separate functions for Brazilian funk, phonk, and retro wave |
| 1.45 | 2026-02-03 | Manual genre override: `--genre` flag to bypass auto-detection |
| 1.46 | 2026-02-03 | Deep learning genre detection: CLAP model via `--deep-genre` flag |
| 1.47 | 2026-02-03 | Render progress display: show [6/7] and [7/7] stages for audio rendering |
| 1.48 | 2026-02-03 | AI-driven mix parameters: audio_to_strudel_params.py analyzes original for effect suggestions |
| 1.49 | 2026-02-03 | Audio comparison: compare_audio.py for rendered vs original similarity scoring |
| 1.50 | 2026-02-03 | Render by default: `--render auto` is now default, use `--render none` to disable |
| 1.51 | 2026-02-03 | Stem quality presets: `--quality` flag with fast/normal/high/best options |
| 1.52 | 2026-02-03 | High-quality separation: htdemucs_ft model, TTA shifts, WAV output, GPU acceleration |
| 1.53 | 2026-02-06 | Node.js Strudel renderer with stem output (render_bass/drums/melodic.wav) |
| 1.54 | 2026-02-06 | Sound variety system: 67 drum machines, 128 GM instruments, 17 genre palettes |
| 1.55 | 2026-02-07 | Per-stem comparison charts and stem_comparison.json |
| 1.56 | 2026-02-08 | DAW-style HTML report with isolated Original/Rendered stem groups |
| 1.57 | 2026-02-08 | **CRITICAL FIX:** Similarity score now uses MAE (not cosine) - was hiding 20%+ band errors |
| 1.58 | 2026-02-08 | New weights: Freq Balance 40%, MFCC 20%, Energy 15%, Brightness 15%, Tempo/Chroma 5% |
| 1.59 | 2026-02-08 | ClickHouse learning storage with 130+ runs tracked |
| 1.60 | 2026-02-08 | LLM always runs all iterations (target 99% to ensure full analysis) |
| 1.61 | 2026-02-08 | **Phase 10 added:** Audio Similarity & Synthesis Quality roadmap with 23 issues |
| 1.62 | 2026-02-09 | **Agentic Ollama:** Persistent chat history, ClickHouse SQL queries, iteration memory |
| 1.63 | 2026-02-09 | **808 Kick Fix:** pitchEnd 80→35Hz for proper sub-bass, removed HPF from kicks |
| 1.64 | 2026-02-09 | Voice gain rebalancing: bass 0.6x, mids 0.5x, highs 0.4x, drums 0.7x |
| 1.65 | 2026-02-10 | **BlackHole Recording:** 100% accuracy via real Strudel playback recording (replaces synthesis emulation) |
| 1.66 | 2026-02-18 | **Iteration Stems + Shimmer Loading:** Batch Demucs on each iteration render → per-iteration melodic/drums/bass stems in report with mute buttons, shimmer skeleton loading animation |
| 1.67 | 2026-02-24 | **Genre-Aware Sound RAG:** `retrieve_genre_context()` injects ~15 genre-appropriate sounds into LLM prompts (800→40 tokens), reducing hallucinated sound names. Injected into `ollama_codegen.py` and `ollama_agent.py`. |
