package midi

import (
	"context"
	"fmt"

	"github.com/arkadiishvartcman/midi-grep/internal/exec"
)

// Transcriber converts audio to MIDI using Basic Pitch
type Transcriber struct {
	runner *exec.Runner
}

// NewTranscriber creates a new MIDI transcriber
func NewTranscriber(runner *exec.Runner) *Transcriber {
	return &Transcriber{runner: runner}
}

// Transcribe converts an audio file to MIDI
func (t *Transcriber) Transcribe(ctx context.Context, audioPath, midiPath string) error {
	result, err := t.runner.RunScript(ctx, "transcribe.py", audioPath, midiPath)
	if err != nil {
		return fmt.Errorf("transcription failed: %w (stderr: %s)", err, result.Stderr)
	}
	return nil
}
