package midi

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/dygy/midi-grep/internal/exec"
)

// Note represents a single MIDI note
type Note struct {
	Pitch    int     `json:"pitch"`
	Start    float64 `json:"start"`
	Duration float64 `json:"duration"`
	Velocity int     `json:"velocity"`
}

// LoopInfo contains detected loop information
type LoopInfo struct {
	Detected           bool    `json:"detected"`
	Bars               int     `json:"bars"`
	Confidence         float64 `json:"confidence"`
	StartBeat          float64 `json:"start_beat"`
	EndBeat            float64 `json:"end_beat"`
	Notes              []Note  `json:"notes"`
	Repetitions        int     `json:"repetitions"`
	ReferenceIteration int     `json:"reference_iteration,omitempty"`
	PatternType        string  `json:"pattern_type,omitempty"` // "single", "alternating", or "none"
	VariationA         []Note  `json:"variation_a,omitempty"`
	VariationB         []Note  `json:"variation_b,omitempty"`
}

// VoiceLoopInfo contains loop info for a single voice
type VoiceLoopInfo struct {
	Bars        int     `json:"bars"`
	Confidence  float64 `json:"confidence"`
	PatternType string  `json:"pattern_type,omitempty"`
	Repetitions int     `json:"repetitions"`
}

// CleanupResult contains the cleaned notes and statistics
type CleanupResult struct {
	Notes      []Note                    `json:"notes"`
	Retained   int                       `json:"retained"`
	Removed    int                       `json:"removed"`
	Loop       *LoopInfo                 `json:"loop,omitempty"`
	VoiceLoops map[string]*VoiceLoopInfo `json:"voice_loops,omitempty"` // Per-voice loop detection
}

// CleanupOptions configures the cleanup process
type CleanupOptions struct {
	Quantize        int
	Simplify        bool
	MaxChordSize    int
	MaxNotesPerBeat int
	PreferredOctave int
	MergeThreshold  float64
	// Loop detection options
	TimeSignature   string  // Time signature for loop detection (e.g., "4/4", "3/4", "6/8")
	SwingRatio      float64 // Swing timing ratio (1.0=straight, 1.5-2.0=swing)
	SwingConfidence float64 // Confidence of swing detection (0.0-1.0)
	MultiVoiceLoops bool    // Detect loops separately for each voice
}

// DefaultCleanupOptions returns sensible defaults
func DefaultCleanupOptions() CleanupOptions {
	return CleanupOptions{
		Quantize:        16,
		Simplify:        false,
		MaxChordSize:    2,              // Keep chords simple (2 notes max)
		MaxNotesPerBeat: 1,              // Only 1 note per beat for clear patterns
		PreferredOctave: 4,
		MergeThreshold:  0.1,            // Merge notes within 100ms
		TimeSignature:   "4/4",          // Default 4/4 time
		SwingRatio:      1.0,            // Straight timing by default
		SwingConfidence: 0.0,
		MultiVoiceLoops: false,
	}
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
	opts := DefaultCleanupOptions()
	opts.Quantize = quantize
	return c.CleanWithOptions(ctx, inputMIDI, outputJSON, opts)
}

// CleanWithOptions removes noise and quantizes with full options
func (c *Cleaner) CleanWithOptions(ctx context.Context, inputMIDI, outputJSON string, opts CleanupOptions) (*CleanupResult, error) {
	// Build arguments
	args := []string{
		inputMIDI,
		outputJSON,
		fmt.Sprintf("--quantize=%d", opts.Quantize),
	}

	if opts.Simplify {
		args = append(args, "--simplify")
		args = append(args, fmt.Sprintf("--max-chord-size=%d", opts.MaxChordSize))
		args = append(args, fmt.Sprintf("--max-notes-per-beat=%d", opts.MaxNotesPerBeat))
		args = append(args, fmt.Sprintf("--preferred-octave=%d", opts.PreferredOctave))
		args = append(args, fmt.Sprintf("--merge-threshold=%.3f", opts.MergeThreshold))
	}

	// Loop detection options
	if opts.TimeSignature != "" {
		args = append(args, fmt.Sprintf("--time-signature=%s", opts.TimeSignature))
	}
	if opts.SwingRatio > 0 {
		args = append(args, fmt.Sprintf("--swing-ratio=%.2f", opts.SwingRatio))
	}
	if opts.SwingConfidence > 0 {
		args = append(args, fmt.Sprintf("--swing-confidence=%.2f", opts.SwingConfidence))
	}
	if opts.MultiVoiceLoops {
		args = append(args, "--multi-voice-loops")
	}

	// Run cleanup script
	result, err := c.runner.RunScript(ctx, "cleanup.py", args...)
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
