package pipeline

import (
	"context"
	"fmt"
	"io"
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
	InputPath        string
	OutputPath       string
	MIDIOutputPath   string
	Quantize         int
	StemTimeout      time.Duration
	TranscribeTimeout time.Duration
}

// DefaultConfig returns default pipeline configuration
func DefaultConfig() Config {
	return Config{
		Quantize:         16,
		StemTimeout:      5 * time.Minute,
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

	// MIDI Cleanup
	cleanResult, err := o.cleaner.Clean(ctx, ws.RawMIDI(), ws.NotesJSON(), cfg.Quantize)
	if err != nil {
		return nil, fmt.Errorf("midi cleanup: %w", err)
	}
	o.progress.StageComplete("Cleanup complete: %d notes retained, %d notes removed",
		cleanResult.Retained, cleanResult.Removed)

	// Stage 5: Strudel generation
	o.progress.StartStage(progress.StageGenerate)
	generator := strudel.NewGenerator(cfg.Quantize)
	strudelCode := generator.Generate(cleanResult.Notes, analysisResult)

	return &Result{
		StrudelCode:   strudelCode,
		BPM:           analysisResult.BPM,
		BPMConfidence: analysisResult.BPMConfidence,
		Key:           analysisResult.Key,
		KeyConfidence: analysisResult.KeyConfidence,
		NotesRetained: cleanResult.Retained,
		NotesRemoved:  cleanResult.Removed,
	}, nil
}
