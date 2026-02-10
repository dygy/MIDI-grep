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
	TrackTitle        string // Track title (from YouTube or filename)
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
	GenreOverride     string // Manual genre override (brazilian_funk, brazilian_phonk, retro_wave, etc.)
	UseCache          bool   // Use stem cache (skip separation if cached)
	StemQuality       string // Stem separation quality (fast, normal, high, best)
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
		UseCache:          true,        // Use stem cache by default
		StemQuality:       "normal",    // Normal quality by default
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
	OriginalPath    string         // Path to original input audio (for comparison)
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

		// Restore original audio path from cache (for AI comparison)
		originalPath := filepath.Join(cfg.CachedStemsDir, "original.wav")
		if fileExists(originalPath) {
			cfg.InputPath = originalPath
		}

		// Try melodic.wav first, fall back to piano.wav
		melodicPath := filepath.Join(cfg.CachedStemsDir, "melodic.wav")
		if !fileExists(melodicPath) {
			melodicPath = filepath.Join(cfg.CachedStemsDir, "piano.wav")
		}
		drumsPath := filepath.Join(cfg.CachedStemsDir, "drums.wav")
		stemResult = &audio.StemResult{}
		if fileExists(melodicPath) {
			stemResult.PianoPath = melodicPath
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
							PianoPath: cached.MelodicPath,
							DrumsPath: cached.DrumsPath,
						}
						// Restore original audio path from cache (for AI comparison)
						if cached.OriginalPath != "" {
							cfg.InputPath = cached.OriginalPath
						}
						usedCache = true
						keyDisplay := cacheKey
						if len(keyDisplay) > 8 {
							keyDisplay = keyDisplay[:8]
						}
						o.progress.StageComplete("Using cached stems (key: %s)", keyDisplay)
					}
				}
			}
		}

		// Stage 2: Stem separation (if not cached)
		if !usedCache {
			o.progress.StartStage(progress.StageSeparate)
			stemCtx, stemCancel := context.WithTimeout(ctx, cfg.StemTimeout)
			defer stemCancel()

			// Use quality setting (default to normal if not set)
			quality := cfg.StemQuality
			if quality == "" {
				quality = "normal"
			}
			if quality == "high" || quality == "best" {
				o.progress.StageComplete("Using %s quality (this may take longer)", quality)
			}

			stemResult, err = o.separator.SeparateWithModeAndQuality(stemCtx, cfg.InputPath, ws.Dir, stemMode, quality)
			if err != nil {
				return nil, fmt.Errorf("stem separation: %w", err)
			}

			if stemResult.PianoPath != "" {
				o.progress.StageComplete("Piano stem extracted")
			}
			if stemResult.DrumsPath != "" {
				o.progress.StageComplete("Drums stem extracted")
			}

			// Save to cache with track metadata (all 4 stems + original)
			if stemCache != nil && cacheKey != "" {
				stems := &cache.StemPaths{
					OriginalPath: cfg.InputPath, // Store original for proper comparison
					MelodicPath:  stemResult.PianoPath,
					DrumsPath:    stemResult.DrumsPath,
					VocalsPath:   stemResult.VocalsPath,
					BassPath:     stemResult.BassPath,
				}
				cached, err := stemCache.PutWithMetadata(cacheKey, stems, cfg.TrackTitle, cfg.InputURL)
				if err != nil {
					o.progress.Warning("Cache save failed: %v", err)
				} else {
					// Update cacheKey to the actual folder name used (may be track name)
					if cached.CacheKey != "" {
						cacheKey = cached.CacheKey
					}
					displayKey := cacheKey
					if cached.TrackName != "" {
						displayKey = cached.TrackName
					}
					o.progress.StageComplete("Cached stems (%s)", displayKey)
				}
			}
		}
	}

	// Initialize result
	result := &Result{
		DrumTypes:    make(map[string]int),
		OriginalPath: cfg.InputPath, // Original audio for comparison
	}

	var analysisResult *analysis.Result
	var cleanResult *midi.CleanupResult
	var strudelCode string
	var synthConfigPath string // Track synth config path for copying to output

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

		// Pass time signature and swing info from analysis to cleanup for better loop detection
		if analysisResult != nil {
			cleanOpts.TimeSignature = analysisResult.TimeSignature
			cleanOpts.SwingRatio = analysisResult.SwingRatio
			cleanOpts.SwingConfidence = analysisResult.SwingConfidence
		}

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

		// Check for manual genre override first
		skipAutoDetection := false
		if cfg.GenreOverride != "" {
			switch cfg.GenreOverride {
			case "brazilian_funk":
				o.progress.StageComplete("Using manual genre: Brazilian Funk")
				cfg.BrazilianFunk = true
			case "brazilian_phonk":
				o.progress.StageComplete("Using manual genre: Brazilian Phonk")
				cfg.SoundStyle = "electronic" // Use electronic style for phonk
				result.Genre = "brazilian_phonk"
				skipAutoDetection = true
			case "retro_wave", "synthwave":
				o.progress.StageComplete("Using manual genre: Retro Wave / Synthwave")
				cfg.SoundStyle = "synthwave"
				skipAutoDetection = true // Skip Brazilian funk/phonk auto-detection
			case "trance":
				o.progress.StageComplete("Using manual genre: Trance")
				cfg.SoundStyle = "trance"
				skipAutoDetection = true
			case "house":
				o.progress.StageComplete("Using manual genre: House")
				cfg.SoundStyle = "house"
				skipAutoDetection = true
			case "lofi":
				o.progress.StageComplete("Using manual genre: Lo-fi")
				cfg.SoundStyle = "lofi"
				skipAutoDetection = true
			}
		}

		// Auto-detect genre based on audio characteristics (unless skipped by manual override)
		// No templates - always use transcription + AI generation
		// Genre detection just sets style hints for the Strudel generator

		// 1. Check for Brazilian Funk (funk carioca) - BPM 130-145 or half-time 85-95
		if !skipAutoDetection && (shouldUseBrazilianFunkMode(analysisResult, cleanResult) || cfg.BrazilianFunk) {
			o.progress.StageComplete("Auto-detected Brazilian Funk - using electronic style with AI generation")
			cfg.SoundStyle = "electronic"
			result.Genre = "brazilian_funk"
		}

		// 2. Check for Brazilian Phonk - BPM 80-100 or 145-180
		if !skipAutoDetection && shouldUseBrazilianPhonkMode(analysisResult, cleanResult) {
			o.progress.StageComplete("Auto-detected Brazilian Phonk - using electronic style with AI generation")
			cfg.SoundStyle = "electronic"
			result.Genre = "brazilian_phonk"
		}

		// 3. Check for Retro Wave (Polish, Russian, etc.) - BPM 130-170, longer synth notes
		if !skipAutoDetection && shouldUseRetroWaveMode(analysisResult, cleanResult) {
			o.progress.StageComplete("Auto-detected Retro Wave/Synthwave")
			cfg.SoundStyle = "synthwave"
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

			// AI synthesis analysis: analyze original audio to derive gains/effects
			// NO HARDCODING - all parameters come from AI analysis
			synthConfigPath = filepath.Join(ws.Dir, "synth_config.json")
			audioToAnalyze := cfg.InputPath
			if audioToAnalyze == "" && stemResult != nil {
				audioToAnalyze = stemResult.PianoPath // Use melodic stem
			}
			if audioToAnalyze != "" {
				_, err := o.runner.RunScript(ctx, "analyze_synth_params.py", audioToAnalyze, "-o", synthConfigPath, "-d", "60")
				if err != nil {
					o.progress.Warning("AI synthesis analysis failed: %v", err)
				} else {
					if loadErr := generator.LoadAIParamsFromJSON(synthConfigPath); loadErr != nil {
						o.progress.Warning("Failed to load AI params: %v", loadErr)
					} else {
						o.progress.StageComplete("Loaded AI-derived synthesis parameters")
					}
				}
			}

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
		// Load AI params if available (for drum gain etc.)
		if synthConfigPath != "" && fileExists(synthConfigPath) {
			_ = generator.LoadAIParamsFromJSON(synthConfigPath)
		}
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

				// Copy synth_config.json to version output directory
				// This allows the renderer to access AI-derived parameters
				if synthConfigPath != "" && fileExists(synthConfigPath) {
					versionDir := filepath.Join(result.CacheDir, fmt.Sprintf("v%03d", cachedOutput.Version))
					destConfig := filepath.Join(versionDir, "synth_config.json")
					if data, readErr := os.ReadFile(synthConfigPath); readErr == nil {
						if writeErr := os.WriteFile(destConfig, data, 0644); writeErr == nil {
							o.progress.StageComplete("Copied AI synth config to output")
						}
					}
				}
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

// shouldUseBrazilianFunkMode auto-detects if Brazilian funk (funk carioca) mode is appropriate
// Based on analysis of Brazilian funk tracks (NOT phonk or retro wave):
// - Brazilian Funk: 130-145 BPM (136 typical), tamborzão drums, vocal chops, mid-heavy
// - Half-time: 85-95 BPM (half of 170-190)
// Key differentiators from retro wave: vocal chop fragmentation (SHORT notes), specific BPM range
func shouldUseBrazilianFunkMode(analysis *analysis.Result, cleanup *midi.CleanupResult) bool {
	if analysis == nil || cleanup == nil {
		return false
	}

	bpm := analysis.BPM

	// Brazilian funk has specific BPM ranges:
	// Primary: 130-145 BPM (funk carioca, 136 most common)
	// Half-time: 85-95 BPM (half of ~170-190)
	inFunkRange := bpm >= 130 && bpm <= 145
	inHalfTimeRange := bpm >= 85 && bpm <= 95

	// Reject if outside Brazilian funk BPM ranges
	if !inFunkRange && !inHalfTimeRange {
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
	var longNoteCount int  // Notes longer than 0.3 seconds (synth pads - retro wave indicator)

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
		// Count long notes (characteristic of synth pads - retro wave)
		if n.Duration > 0.3 {
			longNoteCount++
		}
	}

	// Calculate ratios
	totalNotes := float64(cleanup.Retained)
	midRatio := float64(midRangeCount) / totalNotes
	bassRatio := float64(bassCount) / totalNotes
	shortNoteRatio := float64(shortNoteCount) / totalNotes
	longNoteRatio := float64(longNoteCount) / totalNotes
	avgDuration := totalDuration / totalNotes

	// REJECTION: If track has many long notes or high avg duration, it's likely retro wave/synthwave
	// Retro wave has sustained synth pads, Brazilian funk has fragmented vocal chops
	if longNoteRatio > 0.25 {
		return false // Too many long notes - likely retro wave
	}
	if avgDuration > 0.3 {
		return false // Average duration too long - likely retro wave
	}

	// Brazilian funk detection scoring
	score := 0.0

	// BPM scoring
	if bpm >= 134 && bpm <= 138 {
		score += 3.0 // Perfect funk carioca range
	} else if inFunkRange {
		score += 2.0
	} else if inHalfTimeRange {
		score += 2.0 // Half-time funk
	}

	// Mid-range dominance (vocal chops are mid-heavy)
	if midRatio > 0.6 {
		score += 2.5
	} else if midRatio > 0.5 {
		score += 2.0
	} else if midRatio > 0.4 {
		score += 1.0
	}

	// Low bass content (Brazilian funk is mid-heavy, not bass-heavy)
	if bassRatio < 0.05 {
		score += 1.5
	} else if bassRatio < 0.10 {
		score += 1.0
	}

	// Fragmented notes - REQUIRED for Brazilian funk (vocal chop transcription)
	if shortNoteRatio > 0.5 {
		score += 2.5
	} else if shortNoteRatio > 0.3 {
		score += 1.5
	} else {
		score -= 1.0 // Not fragmented enough - penalty
	}

	// Very short average duration - fragmented vocal chop transcription
	if avgDuration < 0.15 {
		score += 1.5
	} else if avgDuration < 0.2 {
		score += 1.0
	}

	// No swing (Brazilian funk has straight rhythms)
	if !analysis.HasSwing() {
		score += 0.5
	}

	// Threshold: score >= 6.0 for Brazilian funk
	return score >= 6.0
}

// shouldUseBrazilianPhonkMode detects Brazilian phonk (distinct from funk carioca)
// Brazilian phonk: wider BPM range, darker sound, phonk-style drums, KORDHELL/6YNTHMANE style
func shouldUseBrazilianPhonkMode(analysis *analysis.Result, cleanup *midi.CleanupResult) bool {
	if analysis == nil || cleanup == nil {
		return false
	}

	bpm := analysis.BPM

	// Brazilian phonk has wider BPM range than funk:
	// Slow phonk: 80-100 BPM
	// Fast phonk: 145-180 BPM
	// Excludes 130-145 which is funk carioca territory
	inSlowPhonkRange := bpm >= 80 && bpm <= 100
	inFastPhonkRange := bpm >= 145 && bpm <= 180

	if !inSlowPhonkRange && !inFastPhonkRange {
		return false
	}

	if cleanup.Retained == 0 || len(cleanup.Notes) == 0 {
		return false
	}

	// Calculate note statistics
	var totalDuration float64
	var midRangeCount int
	var bassCount int
	var shortNoteCount int
	var longNoteCount int // Retro wave indicator

	for _, n := range cleanup.Notes {
		totalDuration += n.Duration

		if n.Pitch >= 55 && n.Pitch <= 90 {
			midRangeCount++
		}
		if n.Pitch < 50 {
			bassCount++
		}
		if n.Duration < 0.15 {
			shortNoteCount++
		}
		if n.Duration > 0.3 {
			longNoteCount++
		}
	}

	totalNotes := float64(cleanup.Retained)
	midRatio := float64(midRangeCount) / totalNotes
	bassRatio := float64(bassCount) / totalNotes
	shortNoteRatio := float64(shortNoteCount) / totalNotes
	longNoteRatio := float64(longNoteCount) / totalNotes
	avgDuration := totalDuration / totalNotes

	// REJECTION: If track has many long notes, it's likely retro wave/synthwave, NOT phonk
	if longNoteRatio > 0.3 {
		return false // Too many long notes - likely retro wave
	}
	if avgDuration > 0.35 {
		return false // Average duration too long - likely retro wave
	}

	// Brazilian phonk scoring
	score := 0.0

	// BPM in phonk ranges
	if inSlowPhonkRange || inFastPhonkRange {
		score += 2.0
	}

	// Mid-range presence (still has vocal elements)
	if midRatio > 0.4 {
		score += 1.5
	} else if midRatio > 0.3 {
		score += 1.0
	}

	// Phonk can have more bass than funk
	if bassRatio < 0.15 {
		score += 1.0
	}

	// Fragmented notes (vocal chops still present) - REQUIRED
	if shortNoteRatio > 0.4 {
		score += 2.0
	} else if shortNoteRatio > 0.25 {
		score += 1.0
	} else {
		score -= 1.0 // Not fragmented - penalty
	}

	// No swing
	if !analysis.HasSwing() {
		score += 0.5
	}

	// Threshold for phonk
	return score >= 5.0
}

// shouldUseRetroWaveMode detects retro wave/synthwave (Polish, Russian, etc.)
// Characteristics: 150-170 BPM, longer synth notes, more melodic structure
func shouldUseRetroWaveMode(analysis *analysis.Result, cleanup *midi.CleanupResult) bool {
	if analysis == nil || cleanup == nil {
		return false
	}

	bpm := analysis.BPM

	// Retro wave typically 130-170 BPM
	if bpm < 130 || bpm > 170 {
		return false
	}

	if cleanup.Retained == 0 || len(cleanup.Notes) == 0 {
		return false
	}

	// Calculate note statistics
	var totalDuration float64
	var midRangeCount int
	var bassCount int
	var longNoteCount int // Notes longer than 0.3 seconds (synth pads/leads)

	for _, n := range cleanup.Notes {
		totalDuration += n.Duration

		if n.Pitch >= 55 && n.Pitch <= 90 {
			midRangeCount++
		}
		if n.Pitch < 50 {
			bassCount++
		}
		if n.Duration > 0.3 {
			longNoteCount++
		}
	}

	totalNotes := float64(cleanup.Retained)
	midRatio := float64(midRangeCount) / totalNotes
	bassRatio := float64(bassCount) / totalNotes
	longNoteRatio := float64(longNoteCount) / totalNotes
	avgDuration := totalDuration / totalNotes

	// Retro wave scoring
	score := 0.0

	// BPM in retro wave range (but not funk carioca sweet spot)
	if bpm >= 155 && bpm <= 165 {
		score += 2.0 // Classic retro wave tempo
	} else if bpm >= 130 && bpm <= 145 {
		score -= 1.0 // More likely funk carioca
	}

	// Longer notes (synth pads/leads, not vocal chops)
	if longNoteRatio > 0.3 {
		score += 2.5
	} else if longNoteRatio > 0.2 {
		score += 1.5
	}

	// Average duration longer than vocal chops
	if avgDuration > 0.35 {
		score += 2.0
	} else if avgDuration > 0.25 {
		score += 1.0
	}

	// Bass presence (retro wave often has prominent bass lines)
	if bassRatio > 0.15 {
		score += 1.5
	} else if bassRatio > 0.10 {
		score += 1.0
	}

	// Mid-range (synth leads)
	if midRatio > 0.3 && midRatio < 0.6 {
		score += 1.0
	}

	// Threshold for retro wave
	return score >= 5.0
}

// fileExists checks if a file exists
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

