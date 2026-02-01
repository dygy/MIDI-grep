package analysis

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/arkadiishvartcman/midi-grep/internal/exec"
)

// Result contains audio analysis results
type Result struct {
	BPM           float64 `json:"bpm"`
	BPMConfidence float64 `json:"bpm_confidence"`
	Key           string  `json:"key"`
	KeyConfidence float64 `json:"key_confidence"`
}

// Analyzer performs audio analysis (BPM and key detection)
type Analyzer struct {
	runner *exec.Runner
}

// NewAnalyzer creates a new audio analyzer
func NewAnalyzer(runner *exec.Runner) *Analyzer {
	return &Analyzer{runner: runner}
}

// Analyze detects BPM and musical key from an audio file
func (a *Analyzer) Analyze(ctx context.Context, audioPath, outputPath string) (*Result, error) {
	// Run analysis script
	result, err := a.runner.RunScript(ctx, "analyze.py", audioPath, outputPath)
	if err != nil {
		return nil, fmt.Errorf("analysis failed: %w (stderr: %s)", err, result.Stderr)
	}

	// Read analysis results
	data, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read analysis results: %w", err)
	}

	var analysis Result
	if err := json.Unmarshal(data, &analysis); err != nil {
		return nil, fmt.Errorf("parse analysis results: %w", err)
	}

	return &analysis, nil
}

// DefaultResult returns default values when analysis fails
func DefaultResult() *Result {
	return &Result{
		BPM:           120,
		BPMConfidence: 0,
		Key:           "",
		KeyConfidence: 0,
	}
}
