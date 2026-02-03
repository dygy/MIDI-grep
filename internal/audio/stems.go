package audio

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	apperrors "github.com/dygy/midi-grep/internal/errors"
	"github.com/dygy/midi-grep/internal/exec"
)

// StemMode defines the stem separation mode
type StemMode string

const (
	StemModePiano StemMode = "piano" // Extract only piano/instrumental stem
	StemModeDrums StemMode = "drums" // Extract only drums stem
	StemModeFull  StemMode = "full"  // Extract both piano and drums stems
)

// StemResult contains the paths to extracted stems
type StemResult struct {
	PianoPath  string // Path to melodic/other stem (if extracted)
	BassPath   string // Path to bass stem (if extracted)
	DrumsPath  string // Path to drums stem (if extracted)
	VocalsPath string // Path to vocals stem (if extracted)
}

// StemSeparator handles audio stem separation using Demucs
type StemSeparator struct {
	runner *exec.Runner
}

// NewStemSeparator creates a new stem separator
func NewStemSeparator(runner *exec.Runner) *StemSeparator {
	return &StemSeparator{runner: runner}
}

// Separate extracts the piano stem from an audio file (legacy method)
func (s *StemSeparator) Separate(ctx context.Context, inputPath, outputDir string) (string, error) {
	result, err := s.SeparateWithMode(ctx, inputPath, outputDir, StemModePiano)
	if err != nil {
		return "", err
	}
	return result.PianoPath, nil
}

// SeparateWithMode extracts stems based on the specified mode
func (s *StemSeparator) SeparateWithMode(ctx context.Context, inputPath, outputDir string, mode StemMode) (*StemResult, error) {
	// Run separation script with mode argument
	result, err := s.runner.RunScript(ctx, "separate.py", inputPath, outputDir, "--mode", string(mode))
	if err != nil {
		if result != nil && result.ExitCode == 1 {
			return nil, apperrors.NewProcessError("demucs", "stem_separation", result.ExitCode, result.Stderr, err)
		}
		return nil, fmt.Errorf("stem separation: %w", err)
	}

	stemResult := &StemResult{}

	// Look for piano stem
	if mode == StemModePiano || mode == StemModeFull {
		pianoCandidates := []string{
			filepath.Join(outputDir, "piano.wav"),
			filepath.Join(outputDir, "piano.mp3"),
		}
		for _, path := range pianoCandidates {
			if _, err := os.Stat(path); err == nil {
				stemResult.PianoPath = path
				break
			}
		}
		if mode == StemModePiano && stemResult.PianoPath == "" {
			return nil, fmt.Errorf("piano stem not found in %s", outputDir)
		}
	}

	// Look for drums stem
	if mode == StemModeDrums || mode == StemModeFull {
		drumsCandidates := []string{
			filepath.Join(outputDir, "drums.wav"),
			filepath.Join(outputDir, "drums.mp3"),
		}
		for _, path := range drumsCandidates {
			if _, err := os.Stat(path); err == nil {
				stemResult.DrumsPath = path
				break
			}
		}
		if mode == StemModeDrums && stemResult.DrumsPath == "" {
			return nil, fmt.Errorf("drums stem not found in %s", outputDir)
		}
	}

	// Look for bass stem (full mode only)
	if mode == StemModeFull {
		bassCandidates := []string{
			filepath.Join(outputDir, "bass.wav"),
			filepath.Join(outputDir, "bass.mp3"),
		}
		for _, path := range bassCandidates {
			if _, err := os.Stat(path); err == nil {
				stemResult.BassPath = path
				break
			}
		}
	}

	// Look for vocals stem (full mode only)
	if mode == StemModeFull {
		vocalsCandidates := []string{
			filepath.Join(outputDir, "vocals.wav"),
			filepath.Join(outputDir, "vocals.mp3"),
		}
		for _, path := range vocalsCandidates {
			if _, err := os.Stat(path); err == nil {
				stemResult.VocalsPath = path
				break
			}
		}
	}

	// Ensure at least one stem was found
	if stemResult.PianoPath == "" && stemResult.DrumsPath == "" && stemResult.BassPath == "" {
		return nil, fmt.Errorf("no stems found in %s", outputDir)
	}

	return stemResult, nil
}
