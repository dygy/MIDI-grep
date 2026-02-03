package analysis

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/dygy/midi-grep/internal/exec"
)

// Candidate represents an alternative detection result
type KeyCandidate struct {
	Key        string  `json:"key"`
	Confidence float64 `json:"confidence"`
}

type BPMCandidate struct {
	BPM        float64 `json:"bpm"`
	Confidence float64 `json:"confidence"`
}

type TimeSigCandidate struct {
	TimeSignature string  `json:"time_signature"`
	Confidence    float64 `json:"confidence"`
}

// Result contains audio analysis results
type Result struct {
	BPM                     float64            `json:"bpm"`
	BPMConfidence           float64            `json:"bpm_confidence"`
	BPMCandidates           []BPMCandidate     `json:"bpm_candidates,omitempty"`
	Key                     string             `json:"key"`
	KeyConfidence           float64            `json:"key_confidence"`
	KeyCandidates           []KeyCandidate     `json:"key_candidates,omitempty"`
	TimeSignature           string             `json:"time_signature"`
	TimeSignatureConfidence float64            `json:"time_signature_confidence"`
	TimeSignatureCandidates []TimeSigCandidate `json:"time_signature_candidates,omitempty"`
	SwingRatio              float64            `json:"swing_ratio"`
	SwingConfidence         float64            `json:"swing_confidence"`
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
		BPM:                     120,
		BPMConfidence:           0,
		Key:                     "",
		KeyConfidence:           0,
		TimeSignature:           "4/4",
		TimeSignatureConfidence: 0,
		SwingRatio:              1.0,
		SwingConfidence:         0,
	}
}

// HasSwing returns true if swing was detected with reasonable confidence
func (r *Result) HasSwing() bool {
	return r.SwingRatio > 1.1 && r.SwingConfidence > 0.5
}

// BeatsPerBar returns the number of beats per bar based on time signature
func (r *Result) BeatsPerBar() int {
	switch r.TimeSignature {
	case "3/4":
		return 3
	case "6/8":
		return 6
	case "2/4":
		return 2
	case "5/4":
		return 5
	case "7/8":
		return 7
	default:
		return 4
	}
}
