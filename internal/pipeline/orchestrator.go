package pipeline

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/arkadiishvartcman/midi-grep/internal/analysis"
	"github.com/arkadiishvartcman/midi-grep/internal/audio"
	"github.com/arkadiishvartcman/midi-grep/internal/exec"
	"github.com/arkadiishvartcman/midi-grep/internal/midi"
	"github.com/arkadiishvartcman/midi-grep/internal/progress"
	"github.com/arkadiishvartcman/midi-grep/internal/strudel"
	"github.com/arkadiishvartcman/midi-grep/internal/workspace"
)

// Config holds pipeline configuration
type Config struct {
	InputPath         string
	OutputPath        string
	MIDIOutputPath    string
	Quantize          int
	SoundStyle        string
	Simplify          bool // Enable note simplification for cleaner output
	LoopOnly          bool // Output only the detected loop pattern
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
		StemTimeout:       5 * time.Minute,
		TranscribeTimeout: 3 * time.Minute,
	}
}

// Result contains all pipeline outputs
type Result struct {
	StrudelCode   string
	BPM           float64
	BPMConfidence float64
	Key           string
	KeyConfidence float64
	NotesRetained int
	NotesRemoved  int
	LoopDetected  bool
	LoopBars      int
	LoopConfidence float64
}

// Orchestrator coordinates the full processing pipeline
type Orchestrator struct {
	runner     *exec.Runner
	separator  *audio.StemSeparator
	analyzer   *analysis.Analyzer
	transcriber *midi.Transcriber
	cleaner    *midi.Cleaner
	progress   *progress.Reporter
}

// NewOrchestrator creates a new pipeline orchestrator
func NewOrchestrator(scriptsDir string, out io.Writer, verbose bool) *Orchestrator {
	runner := exec.NewRunner("", scriptsDir)
	return &Orchestrator{
		runner:     runner,
		separator:  audio.NewStemSeparator(runner),
		analyzer:   analysis.NewAnalyzer(runner),
		transcriber: midi.NewTranscriber(runner),
		cleaner:    midi.NewCleaner(runner),
		progress:   progress.NewReporter(out, verbose),
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

	// Stage 1: Validate input
	o.progress.StartStage(progress.StageValidate)
	format, err := audio.ValidateInput(cfg.InputPath)
	if err != nil {
		return nil, err
	}
	o.progress.StageComplete("Valid %s file", format)

	// Stage 2: Stem separation
	o.progress.StartStage(progress.StageSeparate)
	stemCtx, stemCancel := context.WithTimeout(ctx, cfg.StemTimeout)
	defer stemCancel()

	pianoPath, err := o.separator.Separate(stemCtx, cfg.InputPath, ws.Dir)
	if err != nil {
		return nil, fmt.Errorf("stem separation: %w", err)
	}
	o.progress.StageComplete("Piano stem extracted")

	// Stage 3: Analysis (BPM + Key)
	o.progress.StartStage(progress.StageAnalyze)
	analysisResult, err := o.analyzer.Analyze(ctx, pianoPath, ws.AnalysisJSON())
	if err != nil {
		o.progress.Warning("Analysis failed, using defaults: %v", err)
		analysisResult = analysis.DefaultResult()
	}
	o.progress.StageComplete("Detected BPM: %.0f (confidence: %.0f%%), Key: %s",
		analysisResult.BPM, analysisResult.BPMConfidence*100, analysisResult.Key)

	// Stage 4: Transcription
	o.progress.StartStage(progress.StageTranscribe)
	transcribeCtx, transcribeCancel := context.WithTimeout(ctx, cfg.TranscribeTimeout)
	defer transcribeCancel()

	if err := o.transcriber.Transcribe(transcribeCtx, pianoPath, ws.RawMIDI()); err != nil {
		return nil, fmt.Errorf("transcription: %w", err)
	}

	// MIDI Cleanup with optional simplification
	cleanOpts := midi.DefaultCleanupOptions()
	cleanOpts.Quantize = cfg.Quantize
	cleanOpts.Simplify = cfg.Simplify

	cleanResult, err := o.cleaner.CleanWithOptions(ctx, ws.RawMIDI(), ws.NotesJSON(), cleanOpts)
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

	// Stage 5: Strudel generation
	o.progress.StartStage(progress.StageGenerate)

	// Auto-detect style if set to "auto" or empty
	style := strudel.SoundStyle(cfg.SoundStyle)
	if style == strudel.StyleAuto || cfg.SoundStyle == "" {
		style = detectStyle(analysisResult, cleanResult)
		o.progress.StageComplete("Auto-detected style: %s", style)
	}

	generator := strudel.NewGeneratorWithStyle(cfg.Quantize, style)

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

	// Try to use the detailed JSON output for richer generation
	var strudelCode string
	if !useLoopNotes {
		// Can use JSON file which has voice separation
		var err error
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

	// Build result with loop info if detected
	notesRetained := cleanResult.Retained
	if useLoopNotes {
		notesRetained = len(notesToUse)
	}

	result := &Result{
		StrudelCode:   strudelCode,
		BPM:           analysisResult.BPM,
		BPMConfidence: analysisResult.BPMConfidence,
		Key:           analysisResult.Key,
		KeyConfidence: analysisResult.KeyConfidence,
		NotesRetained: notesRetained,
		NotesRemoved:  cleanResult.Removed,
	}

	if cleanResult.Loop != nil && cleanResult.Loop.Detected {
		result.LoopDetected = true
		result.LoopBars = cleanResult.Loop.Bars
		result.LoopConfidence = cleanResult.Loop.Confidence
	}

	return result, nil
}

// detectStyle auto-detects the best sound style based on analysis results
func detectStyle(analysis *analysis.Result, cleanup *midi.CleanupResult) strudel.SoundStyle {
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

	// Style detection based on BPM, key, and density
	switch {
	// Fast electronic (>125 BPM)
	case bpm >= 130:
		if noteDensity > 4 {
			return strudel.StyleTrance // Very fast + dense = trance
		}
		return strudel.StyleHouse // Fast + moderate = house

	// Medium-fast (110-130 BPM)
	case bpm >= 110:
		if isMinor {
			return strudel.StyleElectronic // Minor + upbeat = electronic
		}
		if noteDensity > 3 {
			return strudel.StyleSynthwave // 80s electronic feel
		}
		return strudel.StyleFunk // Funky groove

	// Medium (90-110 BPM)
	case bpm >= 90:
		if isMinor && noteDensity < 2 {
			return strudel.StyleDarkwave // Dark, moderate tempo
		}
		if noteDensity > 3 {
			return strudel.StyleElectronic
		}
		return strudel.StyleSynth // General synth

	// Medium-slow (70-90 BPM)
	case bpm >= 70:
		if isMinor {
			if noteDensity < 1 {
				return strudel.StyleCinematic // Slow, sparse, minor
			}
			return strudel.StyleLofi // Chill minor
		}
		if noteDensity < 1 {
			return strudel.StyleSoul // Slow, sparse, major
		}
		return strudel.StyleJazz // Medium jazz

	// Slow (50-70 BPM)
	case bpm >= 50:
		if noteDensity < 0.5 {
			return strudel.StyleAmbient // Very slow, very sparse
		}
		if isMinor {
			return strudel.StyleCinematic
		}
		return strudel.StyleNewAge

	// Very slow (<50 BPM)
	default:
		if noteDensity < 0.3 {
			return strudel.StyleDrone // Extremely slow and sparse
		}
		return strudel.StyleAmbient
	}
}

// containsMinor checks if the key signature contains "minor" or "m"
func containsMinor(key string) bool {
	key = strings.ToLower(key)
	return strings.Contains(key, "minor") || strings.HasSuffix(key, "m")
}

