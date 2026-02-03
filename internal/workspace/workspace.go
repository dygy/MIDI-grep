package workspace

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Workspace manages temporary files for a single processing job
type Workspace struct {
	Dir       string
	CreatedAt time.Time
}

// Create creates a new isolated workspace in the system temp directory
func Create() (*Workspace, error) {
	dir, err := os.MkdirTemp("", "midi-grep-*")
	if err != nil {
		return nil, fmt.Errorf("create workspace: %w", err)
	}

	return &Workspace{
		Dir:       dir,
		CreatedAt: time.Now(),
	}, nil
}

// Path helpers for workspace files
func (w *Workspace) InputCopy() string    { return filepath.Join(w.Dir, "input.wav") }
func (w *Workspace) PianoStem() string    { return filepath.Join(w.Dir, "piano.wav") }
func (w *Workspace) DrumsStem() string    { return filepath.Join(w.Dir, "drums.wav") }
func (w *Workspace) RawMIDI() string      { return filepath.Join(w.Dir, "raw.mid") }
func (w *Workspace) CleanMIDI() string    { return filepath.Join(w.Dir, "clean.mid") }
func (w *Workspace) NotesJSON() string    { return filepath.Join(w.Dir, "notes.json") }
func (w *Workspace) DrumsJSON() string    { return filepath.Join(w.Dir, "drums.json") }
func (w *Workspace) AnalysisJSON() string { return filepath.Join(w.Dir, "analysis.json") }
func (w *Workspace) SpleeterOut() string  { return filepath.Join(w.Dir, "spleeter_out") }

// Cleanup removes the workspace directory and all contents
func (w *Workspace) Cleanup() error {
	return os.RemoveAll(w.Dir)
}

// CopyFile copies a file into the workspace
func (w *Workspace) CopyFile(src, dstName string) (string, error) {
	dst := filepath.Join(w.Dir, dstName)
	input, err := os.ReadFile(src)
	if err != nil {
		return "", fmt.Errorf("read source: %w", err)
	}
	if err := os.WriteFile(dst, input, 0644); err != nil {
		return "", fmt.Errorf("write destination: %w", err)
	}
	return dst, nil
}
