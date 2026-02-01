package midi

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/arkadiishvartcman/midi-grep/internal/exec"
)

// Note represents a single MIDI note
type Note struct {
	Pitch    int     `json:"pitch"`
	Start    float64 `json:"start"`
	Duration float64 `json:"duration"`
	Velocity int     `json:"velocity"`
}

// CleanupResult contains the cleaned notes and statistics
type CleanupResult struct {
	Notes    []Note `json:"notes"`
	Retained int    `json:"retained"`
	Removed  int    `json:"removed"`
}

// Cleaner handles MIDI cleanup and quantization
type Cleaner struct {
	runner *exec.Runner
}

// NewCleaner creates a new MIDI cleaner
func NewCleaner(runner *exec.Runner) *Cleaner {
	return &Cleaner{runner: runner}
}

// Clean removes noise and quantizes a MIDI file
func (c *Cleaner) Clean(ctx context.Context, inputMIDI, outputJSON string, quantize int) (*CleanupResult, error) {
	// Run cleanup script
	result, err := c.runner.RunScript(ctx, "cleanup.py",
		inputMIDI,
		outputJSON,
		fmt.Sprintf("--quantize=%d", quantize),
	)
	if err != nil {
		return nil, fmt.Errorf("cleanup failed: %w (stderr: %s)", err, result.Stderr)
	}

	// Read cleanup results
	data, err := os.ReadFile(outputJSON)
	if err != nil {
		return nil, fmt.Errorf("read cleanup results: %w", err)
	}

	var cleanup CleanupResult
	if err := json.Unmarshal(data, &cleanup); err != nil {
		return nil, fmt.Errorf("parse cleanup results: %w", err)
	}

	return &cleanup, nil
}
