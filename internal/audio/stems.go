package audio

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	apperrors "github.com/arkadiishvartcman/midi-grep/internal/errors"
	"github.com/arkadiishvartcman/midi-grep/internal/exec"
)

// StemSeparator handles audio stem separation using Spleeter
type StemSeparator struct {
	runner *exec.Runner
}

// NewStemSeparator creates a new stem separator
func NewStemSeparator(runner *exec.Runner) *StemSeparator {
	return &StemSeparator{runner: runner}
}

// Separate extracts the piano stem from an audio file
func (s *StemSeparator) Separate(ctx context.Context, inputPath, outputDir string) (string, error) {
	// Run separation script (uses demucs)
	result, err := s.runner.RunScript(ctx, "separate.py", inputPath, outputDir)
	if err != nil {
		// Check if it's a tool installation issue
		if result != nil && result.ExitCode == 1 {
			return "", apperrors.NewProcessError("spleeter", "stem_separation", result.ExitCode, result.Stderr, err)
		}
		return "", fmt.Errorf("stem separation: %w", err)
	}

	// Find the piano stem in output (could be .wav or .mp3)
	candidates := []string{
		filepath.Join(outputDir, "piano.wav"),
		filepath.Join(outputDir, "piano.mp3"),
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("piano stem not found in %s", outputDir)
}
