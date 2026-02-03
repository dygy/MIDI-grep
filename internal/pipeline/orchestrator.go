package pipeline

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/dygy/midi-grep/internal/analysis"
	"github.com/dygy/midi-grep/internal/audio"
	"github.com/dygy/midi-grep/internal/cache"
	"github.com/dygy/midi-grep/internal/drums"
	"github.com/dygy/midi-grep/internal/exec"
	"github.com/dygy/midi-grep/internal/midi"
	"github.com/dygy/midi-grep/internal/progress"
	"github.com/dygy/midi-grep/internal/strudel"
	"github.com/dygy/midi-grep/internal/workspace"
)

// Config holds pipeline configuration
type Config struct {
	InputPath         string
	InputURL          string // Original URL (for cache key generation)
	CachedStemsDir    string // Pre-cached stems directory (skip download + separation)
	OutputPath        string
	MIDIOutputPath    string
	Quantize          int
	SoundStyle        string
	Simplify          bool   // Enable note simplification for cleaner output
	LoopOnly          bool   // Output only the detected loop pattern
	EnableDrums       bool   // Extract and include drum patterns
	DrumsOnly         bool   // Extract only drums (skip melodic processing)
	DrumKit           string // Drum kit to use (tr808, tr909, linn, acoustic, lofi)
	Arrange           bool   // Use arrangement-based generation with chord detection
	ChordMode         bool   // Use chord-based generation (better for electronic music)
	BrazilianFunk     bool   // Use Brazilian funk/phonk mode (tamborzão, 808 bass)
	UseCache          bool   // Use stem cache (skip separation if cached)
	StemTimeout       time.Duration
	TranscribeTimeout time.Duration
}

// DefaultConfig returns default pipeline configuration
func DefaultConfig() Config {
	return Config{
		Quantize:          16,
		SoundStyle:        "auto", // Auto-detect style from audio analysis
		Simplify:          true,   // Enable by default for cleaner output
		LoopOnly:          false,
		EnableDrums:       true,   // Extract drums by default
		DrumsOnly:         false,
		DrumKit:           "tr808",
		Arrange:           false,
		UseCache:          true, // Use stem cache by default
		StemTimeout:       5 * time.Minute,
		TranscribeTimeout: 3 * time.Minute,
	}
}

// Result contains all pipeline outputs
type Result struct {
	StrudelCode     string
	BPM             float64
	BPMConfidence   float64
	Key             string
	KeyConfidence   float64
	TimeSignature   string
	SwingRatio      float64
	NotesRetained   int
	NotesRemoved    int
	LoopDetected    bool
	LoopBars        int
	LoopConfidence  float64
	DrumHits        int            // Number of drum hits detected
	DrumTypes       map[string]int // Hits by drum type
	CacheKey        string         // Cache key for this extraction
	CacheDir        string         // Cache directory path
	OutputVersion   int            // Version number of this output
	PreviousOutputs int            // Number of previous outputs in cache
	Style           string         // Detected or specified style
	Genre           string         // Detected genre (e.g., "brazilian_funk")
}

// Orchestrator coordinates the full processing pipeline
type Orchestrator struct {
	runner       *exec.Runner
	separator    *audio.StemSeparator
	analyzer     *analysis.Analyzer
	transcriber  *midi.Transcriber
	cleaner      *midi.Cleaner
	drumDetector *drums.Detector
	progress     *progress.Reporter
}

// NewOrchestrator creates a new pipeline orchestrator
func NewOrchestrator(scriptsDir string, out io.Writer, verbose bool) *Orchestrator {
	runner := exec.NewRunner("", scriptsDir)
	return &Orchestrator{
		runner:       runner,
		separator:    audio.NewStemSeparator(runner),
		analyzer:     analysis.NewAnalyzer(runner),
		transcriber:  midi.NewTranscriber(runner),
		cleaner:      midi.NewCleaner(runner),
		drumDetector: drums.NewDetector(runner),
		progress:     progress.NewReporter(out, verbose),
	}
}

// Execute runs the full pipeline
func (o *Orchestrator) Execute(ctx context.Context, cfg Config) (*Result, error) {
	// Create workspace
	ws, err := workspace.Create()
	if err != nil {
		return nil, fmt.Errorf("create workspace: %w", err)
	}
	defer ws.Cleanup()

	// Variables for caching output
	var cacheKey string
	var stemCache *cache.StemCache

	// Check if we have pre-cached stems (from main.go cache check)
	var stemResult *audio.StemResult
	if cfg.CachedStemsDir != "" {
		// Extract cache key from the directory name
		cacheKey = filepath.Base(cfg.CachedStemsDir)
		// Initialize stem cache for output saving
		stemCache, _ = cache.NewStemCache()

		pianoPath := filepath.Join(cfg.CachedStemsDir, "piano.wav")
		drumsPath := filepath.Join(cfg.CachedStemsDir, "drums.wav")
		stemResult = &audio.StemResult{}
		if fileExists(pianoPath) {
			stemResult.PianoPath = pianoPath
		}
		if fileExists(drumsPath) {
			stemResult.DrumsPath = drumsPath
		}
		// Skip validation and separation stages
		o.progress.StartStage(progress.StageValidate)
		o.progress.StageComplete("Using cached stems")
		o.progress.StartStage(progress.StageSeparate)
		o.progress.StageComplete("Skipped (cached)")
	} else {
		// Stage 1: Validate input
		o.progress.StartStage(progress.StageValidate)
		format, err := audio.ValidateInput(cfg.InputPath)
		if err != nil {
			return nil, err
		}
		o.progress.StageComplete("Valid %s file", format)

		// Determine stem mode based on config
		stemMode := audio.StemModePiano
		if cfg.DrumsOnly {
			stemMode = audio.StemModeDrums
		} else if cfg.EnableDrums {
			stemMode = audio.StemModeFull
		}

		// Try to use cached stems
		usedCache := false

		if cfg.UseCache {
			stemCache, err = cache.NewStemCache()
			if err != nil {
				o.progress.Warning("Cache init failed: %v", err)
			} else {
				// Generate cache key from URL or file
				if cfg.InputURL != "" {
					cacheKey = cache.KeyForURL(cfg.InputURL)
				} else {
					cacheKey, err = cache.KeyForFile(cfg.InputPath)
					if err != nil {
						o.progress.Warning("Cache key failed: %v", err)
						cacheKey = ""
					}
				}

				// Check cache
				if cacheKey != "" {
					if cached, ok := stemCache.Get(cacheKey); ok {
						o.progress.StartStage(progress.StageSeparate)
						stemResult = &audio.StemResult{
							PianoPath: cached.PianoPath,
							DrumsPath: cached.DrumsPath,
						}
						usedCache = true
						o.progress.StageComplete("Using cached stems (key: %s)", cacheKey[:8])
					}
				}
			}
		}

		// Stage 2: Stem separation (if not cached)
		if !usedCache {
			o.progress.StartStage(progress.StageSeparate)
			stemCtx, stemCancel := context.WithTimeout(ctx, cfg.StemTimeout)
			defer stemCancel()

			stemResult, err = o.separator.SeparateWithMode(stemCtx, cfg.InputPath, ws.Dir, stemMode)
			if err != nil {
				return nil, fmt.Errorf("stem separation: %w", err)
			}

			if stemResult.PianoPath != "" {
				o.progress.StageComplete("Piano stem extracted")
			}
			if stemResult.DrumsPath != "" {
				o.progress.StageComplete("Drums stem extracted")
			}

			// Save to cache
			if stemCache != nil && cacheKey != "" {
				if _, err := stemCache.Put(cacheKey, stemResult.PianoPath, stemResult.DrumsPath); err != nil {
					o.progress.Warning("Cache save failed: %v", err)
				} else {
					o.progress.StageComplete("Cached stems (key: %s)", cacheKey[:8])
				}
			}
		}
	}

	// Initialize result
	result := &Result{
		DrumTypes: make(map[string]int),
	}

	var analysisResult *analysis.Result
	var cleanResult *midi.CleanupResult
	var strudelCode string

	// Process melodic content (unless drums-only mode)
	if !cfg.DrumsOnly && stemResult.PianoPath != "" {
		// Stage 3: Analysis (BPM + Key)
		o.progress.StartStage(progress.StageAnalyze)

		// Use chord-based analysis if chord mode enabled
		if cfg.ChordMode {
			// Run smart analysis for chord detection
			smartAnalysisPath := filepath.Join(ws.Dir, "smart_analysis.json")
			_, err := o.runner.RunScript(ctx, "smart_analyze.py", stemResult.PianoPath, smartAnalysisPath)
			if err != nil {
				o.progress.Warning("Smart analysis failed: %v", err)
			} else {
				// Generate chord-based Strudel code
				chordResult, err := o.runner.RunScript(ctx, "chord_to_strudel.py", smartAnalysisPath)
				if err == nil && chordResult.Stdout != "" {
					strudelCode = chordResult.Stdout
					// Parse basic info from smart analysis
					analysisResult = analysis.DefaultResult()
					// The chord_to_strudel.py output is the final code
					result.StrudelCode = strudelCode
					o.progress.StageComplete("Chord-based generation complete")

					// Skip to drum processing
					goto processDrums
				}
				o.progress.Warning("Chord generation failed, falling back to note transcription")
			}
		}


		analysisResult, err = o.analyzer.Analyze(ctx, stemResult.PianoPath, ws.AnalysisJSON())
		if err != nil {
			o.progress.Warning("Analysis failed, using defaults: %v", err)
			analysisResult = analysis.DefaultResult()
		}
		o.progress.StageComplete("Detected BPM: %.0f, Key: %s, Time: %s",
			analysisResult.BPM, analysisResult.Key, analysisResult.TimeSignature)
		if analysisResult.HasSwing() {
			o.progress.StageComplete("Swing detected: %.2f ratio (%.0f%% confidence)",
				analysisResult.SwingRatio, analysisResult.SwingConfidence*100)
		}

		// Stage 4: Transcription
		o.progress.StartStage(progress.StageTranscribe)
		transcribeCtx, transcribeCancel := context.WithTimeout(ctx, cfg.TranscribeTimeout)
		defer transcribeCancel()

		if err := o.transcriber.Transcribe(transcribeCtx, stemResult.PianoPath, ws.RawMIDI()); err != nil {
			return nil, fmt.Errorf("transcription: %w", err)
		}

		// MIDI Cleanup with optional simplification
		cleanOpts := midi.DefaultCleanupOptions()
		cleanOpts.Quantize = cfg.Quantize
		cleanOpts.Simplify = cfg.Simplify

		cleanResult, err = o.cleaner.CleanWithOptions(ctx, ws.RawMIDI(), ws.NotesJSON(), cleanOpts)
		if err != nil {
			return nil, fmt.Errorf("midi cleanup: %w", err)
		}

		// Report cleanup results
		if cfg.Simplify {
			o.progress.StageComplete("Cleanup complete: %d notes (simplified from raw transcription)",
				cleanResult.Retained)
		} else {
			o.progress.StageComplete("Cleanup complete: %d notes retained, %d notes removed",
				cleanResult.Retained, cleanResult.Removed)
		}

		// Report loop detection
		if cleanResult.Loop != nil && cleanResult.Loop.Detected {
			o.progress.StageComplete("Loop detected: %d bar(s), %.0f%% confidence",
				cleanResult.Loop.Bars, cleanResult.Loop.Confidence*100)
		}

		// Stage 5: Strudel generation for melodic content
		o.progress.StartStage(progress.StageGenerate)

		// Auto-detect Brazilian funk/phonk based on characteristics
		// Criteria: BPM 125-150 + sparse transcription + vocal-range notes (not real melody)
		if shouldUseBrazilianFunkMode(analysisResult, cleanResult) || cfg.BrazilianFunk {
			o.progress.StageComplete("Auto-detected Brazilian Funk/Phonk - using tamborzão templates")

			bfGen := strudel.NewBrazilianFunkGenerator(analysisResult.BPM, analysisResult.Key)
			result.StrudelCode = bfGen.Generate(analysisResult)
			result.BPM = analysisResult.BPM
			result.Key = analysisResult.Key
			result.NotesRetained = cleanResult.Retained
			result.Style = "brazilian_funk"
			result.Genre = "brazilian_funk"

			// Save output to cache
			if cacheKey != "" {
				if stemCache == nil {
					stemCache, _ = cache.NewStemCache()
				}
				if stemCache != nil {
					previousOutputs, _ := stemCache.GetOutputHistory(cacheKey)
					result.PreviousOutputs = len(previousOutputs)
					result.CacheKey = cacheKey
					result.CacheDir = stemCache.GetCacheDir(cacheKey)

					cachedOutput := &cache.CachedOutput{
						Code:     result.StrudelCode,
						BPM:      result.BPM,
						Key:      result.Key,
						Style:    result.Style,
						Genre:    result.Genre,
						Notes:    result.NotesRetained,
						DrumHits: result.DrumHits,
					}
					if err := stemCache.SaveOutput(cacheKey, cachedOutput); err != nil {
						o.progress.Warning("Failed to cache output: %v", err)
					} else {
						result.OutputVersion = cachedOutput.Version
						o.progress.StageComplete("Output saved (v%d) to %s", cachedOutput.Version, result.CacheDir)
					}
				}
			}

			return result, nil
		}

		// Auto-detect style if set to "auto" or empty
		style := strudel.SoundStyle(cfg.SoundStyle)
		var styleCandidates []StyleCandidate
		if style == strudel.StyleAuto || cfg.SoundStyle == "" {
			styleCandidates = detectStyleCandidates(analysisResult, cleanResult)
			if len(styleCandidates) > 0 {
				style = styleCandidates[0].Style
			}
			result.Style = string(style)
			o.progress.StageComplete("Auto-detected style: %s", style)
		}

		// Determine which notes to use
		var notesToUse []midi.Note
		useLoopNotes := cfg.LoopOnly && cleanResult.Loop != nil && cleanResult.Loop.Detected

		if useLoopNotes {
			notesToUse = cleanResult.Loop.Notes
			o.progress.StageComplete("Using loop pattern: %d notes (%d bar(s))",
				len(notesToUse), cleanResult.Loop.Bars)
		} else {
			notesToUse = cleanResult.Notes
		}

		// Use arrangement-based generator if enabled
		if cfg.Arrange {
			o.progress.StageComplete("Using arrangement-based generation with chord detection")
			arrGenerator := strudel.NewArrangementGenerator(cfg.Quantize, style)
			// Note: drumResult will be nil here, drums added later if enabled
			strudelCode = arrGenerator.GenerateArrangement(notesToUse, analysisResult, nil, strudel.ParseDrumKit(cfg.DrumKit))
		} else {
			// Standard generation
			generator := strudel.NewGeneratorWithStyle(cfg.Quantize, style)

			// Pass style candidates for header output
			if len(styleCandidates) > 0 {
				strudelCandidates := make([]strudel.StyleCandidate, len(styleCandidates))
				for i, c := range styleCandidates {
					strudelCandidates[i] = strudel.StyleCandidate{Style: c.Style, Score: c.Score}
				}
				generator.SetStyleCandidates(strudelCandidates)
			}

			// Try to use the detailed JSON output for richer generation
			if !useLoopNotes {
				// Can use JSON file which has voice separation
				strudelCode, err = generator.GenerateFromJSON(ws.NotesJSON(), analysisResult)
				if err != nil {
					// Fallback to legacy method
					o.progress.Warning("Using legacy generator: %v", err)
					strudelCode = generator.Generate(notesToUse, analysisResult)
				}
			} else {
				// Loop notes - use direct generation
				strudelCode = generator.Generate(notesToUse, analysisResult)
			}
		}

		// Build result with loop info if detected
		notesRetained := cleanResult.Retained
		if useLoopNotes {
			notesRetained = len(notesToUse)
		}

		result.BPM = analysisResult.BPM
		result.BPMConfidence = analysisResult.BPMConfidence
		result.Key = analysisResult.Key
		result.KeyConfidence = analysisResult.KeyConfidence
		result.TimeSignature = analysisResult.TimeSignature
		result.SwingRatio = analysisResult.SwingRatio
		result.NotesRetained = notesRetained
		result.NotesRemoved = cleanResult.Removed

		if cleanResult.Loop != nil && cleanResult.Loop.Detected {
			result.LoopDetected = true
			result.LoopBars = cleanResult.Loop.Bars
			result.LoopConfidence = cleanResult.Loop.Confidence
		}
	} else if cfg.DrumsOnly {
		// Drums-only mode: need to analyze drums audio for BPM
		o.progress.StartStage(progress.StageAnalyze)
		analysisResult, err = o.analyzer.Analyze(ctx, stemResult.DrumsPath, ws.AnalysisJSON())
		if err != nil {
			o.progress.Warning("Analysis failed, using defaults: %v", err)
			analysisResult = analysis.DefaultResult()
		}
		o.progress.StageComplete("Detected BPM: %.0f", analysisResult.BPM)
		result.BPM = analysisResult.BPM
		result.BPMConfidence = analysisResult.BPMConfidence
	}

processDrums:
	// Process drums if enabled or drums-only mode
	var drumResult *drums.DetectionResult
	if (cfg.EnableDrums || cfg.DrumsOnly) && stemResult.DrumsPath != "" {
		o.progress.StageComplete("Detecting drum patterns...")

		// Use BPM from analysis if available
		bpm := 0.0
		if analysisResult != nil {
			bpm = analysisResult.BPM
		}

		drumResult, err = o.drumDetector.Detect(ctx, stemResult.DrumsPath, ws.DrumsJSON(), cfg.Quantize, bpm)
		if err != nil {
			o.progress.Warning("Drum detection failed: %v", err)
		} else {
			o.progress.StageComplete("Detected %d drum hits (bd: %d, sd: %d, hh: %d)",
				drumResult.Stats.Total,
				drumResult.Stats.ByType["bd"],
				drumResult.Stats.ByType["sd"],
				drumResult.Stats.ByType["hh"])

			result.DrumHits = drumResult.Stats.Total
			result.DrumTypes = drumResult.Stats.ByType

			// If drums-only, use drum tempo
			if cfg.DrumsOnly && drumResult.Tempo > 0 {
				result.BPM = drumResult.Tempo
			}
		}
	}

	// Generate final Strudel output
	kit := strudel.ParseDrumKit(cfg.DrumKit)
	style := strudel.SoundStyle(cfg.SoundStyle)
	if style == strudel.StyleAuto || cfg.SoundStyle == "" {
		style = strudel.StylePiano // Default for final output
	}

	if cfg.DrumsOnly && drumResult != nil {
		// Drums-only output
		generator := strudel.NewGeneratorWithStyle(cfg.Quantize, style)
		var sb strings.Builder
		sb.WriteString("// MIDI-grep output (drums only)\n")
		sb.WriteString(fmt.Sprintf("// BPM: %.0f\n", result.BPM))
		sb.WriteString(strudel.GenerateDrumHeader(drumResult, kit))
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", result.BPM))
		sb.WriteString("$: ")
		drumPattern := generator.GenerateDrumPattern(drumResult, kit)
		if drumPattern != "" {
			// Remove the leading indentation and comment from drum pattern
			drumPattern = strings.TrimPrefix(drumPattern, "  // drums (")
			drumPattern = strings.TrimPrefix(drumPattern, string(kit)+")\n  ")
			sb.WriteString(drumPattern)
		} else {
			sb.WriteString("s(\"bd\")")
		}
		sb.WriteString("\n")
		result.StrudelCode = sb.String()
	} else if cfg.Arrange && drumResult != nil && cleanResult != nil && analysisResult != nil {
		// Arrangement mode with drums - regenerate with full arrangement
		arrGenerator := strudel.NewArrangementGenerator(cfg.Quantize, style)
		notesToUse := cleanResult.Notes
		if cfg.LoopOnly && cleanResult.Loop != nil && cleanResult.Loop.Detected {
			notesToUse = cleanResult.Loop.Notes
		}
		result.StrudelCode = arrGenerator.GenerateArrangement(notesToUse, analysisResult, drumResult, kit)
	} else if drumResult != nil && strudelCode != "" {
		// Combined melodic + drums output (standard mode)
		generator := strudel.NewGeneratorWithStyle(cfg.Quantize, style)
		result.StrudelCode = generator.GenerateFullOutput(strudelCode, drumResult, kit)
	} else {
		// Melodic only
		result.StrudelCode = strudelCode
	}

	// Save output to cache for iteration
	if cacheKey != "" && result.StrudelCode != "" {
		if stemCache == nil {
			stemCache, _ = cache.NewStemCache()
		}
		if stemCache != nil {
			// Get previous output count
			previousOutputs, _ := stemCache.GetOutputHistory(cacheKey)
			result.PreviousOutputs = len(previousOutputs)
			result.CacheKey = cacheKey
			result.CacheDir = stemCache.GetCacheDir(cacheKey)

			// Determine genre
			genre := ""
			if cfg.BrazilianFunk || (analysisResult != nil && cleanResult != nil && shouldUseBrazilianFunkMode(analysisResult, cleanResult)) {
				genre = "brazilian_funk"
			}

			// Save the output
			cachedOutput := &cache.CachedOutput{
				Code:     result.StrudelCode,
				BPM:      result.BPM,
				Key:      result.Key,
				Style:    result.Style,
				Genre:    genre,
				Notes:    result.NotesRetained,
				DrumHits: result.DrumHits,
			}
			if err := stemCache.SaveOutput(cacheKey, cachedOutput); err != nil {
				o.progress.Warning("Failed to cache output: %v", err)
			} else {
				result.OutputVersion = cachedOutput.Version
				o.progress.StageComplete("Output saved (v%d) to %s", cachedOutput.Version, result.CacheDir)
			}
		}
	}

	return result, nil
}

// StyleCandidate represents a style option with its score
type StyleCandidate struct {
	Style strudel.SoundStyle
	Score float64
}

// detectStyle auto-detects the best sound style based on analysis results
func detectStyle(analysis *analysis.Result, cleanup *midi.CleanupResult) strudel.SoundStyle {
	candidates := detectStyleCandidates(analysis, cleanup)
	if len(candidates) > 0 {
		return candidates[0].Style
	}
	return strudel.StyleSynth
}

// detectStyleCandidates returns ranked style candidates with scores
func detectStyleCandidates(analysis *analysis.Result, cleanup *midi.CleanupResult) []StyleCandidate {
	bpm := analysis.BPM
	key := analysis.Key
	isMinor := containsMinor(key)

	// Calculate note density (notes per second)
	var totalNotes int
	var maxEnd float64
	if cleanup != nil {
		totalNotes = cleanup.Retained
		for _, n := range cleanup.Notes {
			if end := n.Start + n.Duration; end > maxEnd {
				maxEnd = end
			}
		}
	}
	noteDensity := 0.0
	if maxEnd > 0 {
		noteDensity = float64(totalNotes) / maxEnd
	}

	// Score all styles based on criteria
	type styleScore struct {
		style strudel.SoundStyle
		score float64
	}
	var scores []styleScore

	// BPM-based scoring
	addScore := func(s strudel.SoundStyle, score float64) {
		scores = append(scores, styleScore{s, score})
	}

	// Fast (>125 BPM)
	if bpm >= 125 {
		addScore(strudel.StyleHouse, 0.9)
		addScore(strudel.StyleTrance, 0.8)
		addScore(strudel.StyleElectronic, 0.7)
		if noteDensity > 4 {
			// Boost trance for very dense
			for i := range scores {
				if scores[i].style == strudel.StyleTrance {
					scores[i].score += 0.2
				}
			}
		}
	} else if bpm >= 110 {
		// Medium-fast
		addScore(strudel.StyleElectronic, 0.8)
		addScore(strudel.StyleFunk, 0.75)
		addScore(strudel.StyleSynthwave, 0.7)
		addScore(strudel.StyleHouse, 0.6)
	} else if bpm >= 90 {
		// Medium
		addScore(strudel.StyleSynth, 0.8)
		addScore(strudel.StyleElectronic, 0.7)
		addScore(strudel.StyleDarkwave, 0.65)
		addScore(strudel.StyleFunk, 0.6)
	} else if bpm >= 70 {
		// Medium-slow
		addScore(strudel.StyleJazz, 0.8)
		addScore(strudel.StyleLofi, 0.75)
		addScore(strudel.StyleSoul, 0.7)
		addScore(strudel.StyleCinematic, 0.65)
	} else if bpm >= 50 {
		// Slow
		addScore(strudel.StyleAmbient, 0.8)
		addScore(strudel.StyleCinematic, 0.75)
		addScore(strudel.StyleNewAge, 0.7)
		addScore(strudel.StyleLofi, 0.6)
	} else {
		// Very slow
		addScore(strudel.StyleDrone, 0.85)
		addScore(strudel.StyleAmbient, 0.8)
	}

	// Adjust scores based on key (minor/major)
	for i := range scores {
		switch scores[i].style {
		case strudel.StyleDarkwave, strudel.StyleCinematic, strudel.StyleLofi:
			if isMinor {
				scores[i].score += 0.15
			} else {
				scores[i].score -= 0.1
			}
		case strudel.StyleSoul, strudel.StyleFunk, strudel.StyleJazz:
			if !isMinor {
				scores[i].score += 0.1
			}
		case strudel.StyleElectronic, strudel.StyleTrance:
			if isMinor {
				scores[i].score += 0.1
			}
		}
	}

	// Adjust based on density
	for i := range scores {
		switch scores[i].style {
		case strudel.StyleAmbient, strudel.StyleDrone:
			if noteDensity < 0.5 {
				scores[i].score += 0.2
			} else if noteDensity > 2 {
				scores[i].score -= 0.3
			}
		case strudel.StyleTrance, strudel.StyleElectronic:
			if noteDensity > 3 {
				scores[i].score += 0.15
			}
		}
	}

	// Sort by score
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Convert to candidates (top 5)
	var candidates []StyleCandidate
	for i := 0; i < len(scores) && i < 5; i++ {
		candidates = append(candidates, StyleCandidate{
			Style: scores[i].style,
			Score: scores[i].score,
		})
	}

	return candidates
}

// containsMinor checks if the key signature contains "minor" or "m"
func containsMinor(key string) bool {
	key = strings.ToLower(key)
	return strings.Contains(key, "minor") || strings.HasSuffix(key, "m")
}

// shouldUseBrazilianFunkMode auto-detects if Brazilian funk/phonk mode is appropriate
// Criteria:
// 1. BPM in 125-155 range (typical for Brazilian funk/phonk)
// 2. Notes are fragmented (short durations, characteristic of vocal chop transcription)
// 3. Notes clustered in mid-range (vocal frequencies, not bass or high melody)
// 4. Low bass content (real melodies have bass, vocal chops don't)
func shouldUseBrazilianFunkMode(analysis *analysis.Result, cleanup *midi.CleanupResult) bool {
	if analysis == nil || cleanup == nil {
		return false
	}

	bpm := analysis.BPM

	// Check BPM range (Brazilian funk is typically 125-155 BPM)
	if bpm < 125 || bpm > 155 {
		return false
	}

	if cleanup.Retained == 0 || len(cleanup.Notes) == 0 {
		return false
	}

	// Calculate note statistics
	var maxEnd float64
	var totalDuration float64
	var midRangeCount int  // Vocal range (MIDI 55-90, ~G3 to F#6)
	var bassCount int      // Bass range (MIDI < 50)
	var shortNoteCount int // Notes shorter than 0.15 seconds

	for _, n := range cleanup.Notes {
		if end := n.Start + n.Duration; end > maxEnd {
			maxEnd = end
		}
		totalDuration += n.Duration

		// Count by pitch range
		if n.Pitch >= 55 && n.Pitch <= 90 {
			midRangeCount++
		}
		if n.Pitch < 50 {
			bassCount++
		}

		// Count short/fragmented notes (characteristic of vocal chop transcription)
		if n.Duration < 0.15 {
			shortNoteCount++
		}
	}

	// Calculate ratios
	totalNotes := float64(cleanup.Retained)
	midRatio := float64(midRangeCount) / totalNotes
	bassRatio := float64(bassCount) / totalNotes
	shortNoteRatio := float64(shortNoteCount) / totalNotes
	avgDuration := totalDuration / totalNotes

	// Calculate notes per beat
	notesPerBeat := 0.0
	if maxEnd > 0 {
		trackBeats := maxEnd * bpm / 60
		notesPerBeat = totalNotes / trackBeats
	}

	// Brazilian funk detection scoring
	score := 0.0

	// High percentage of notes in vocal range (>60%) - vocal chops
	if midRatio > 0.7 {
		score += 3.0
	} else if midRatio > 0.6 {
		score += 2.0
	} else if midRatio > 0.5 {
		score += 1.0
	}

	// Low bass content (<10%) - real melodies have bass, vocal chops don't
	if bassRatio < 0.05 {
		score += 2.0
	} else if bassRatio < 0.10 {
		score += 1.0
	}

	// Fragmented notes (short durations) - vocal chops produce short transcribed notes
	if shortNoteRatio > 0.7 {
		score += 2.0
	} else if shortNoteRatio > 0.5 {
		score += 1.0
	}

	// Very short average duration (<0.2s) - fragmented transcription
	if avgDuration < 0.15 {
		score += 2.0
	} else if avgDuration < 0.25 {
		score += 1.0
	}

	// Moderate note density (0.3-2.0 notes/beat) - too sparse = silence, too dense = real melody
	// Vocal chops produce scattered notes, not coherent melodies
	if notesPerBeat >= 0.3 && notesPerBeat <= 2.0 {
		score += 1.5
	} else if notesPerBeat > 2.0 && notesPerBeat <= 3.0 {
		score += 0.5 // Could still be Brazilian funk
	}

	// BPM sweet spot for Brazilian funk (130-145)
	if bpm >= 130 && bpm <= 145 {
		score += 1.0
	}

	// Threshold: score >= 5.0 suggests Brazilian funk
	// High mid-range (3) + low bass (2) + fragmented (1) = 6 typical case
	return score >= 5.0
}

// fileExists checks if a file exists
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

