package errors

import (
	"errors"
	"fmt"
)

// Sentinel errors for expected failure modes
var (
	ErrFileNotFound      = errors.New("file not found")
	ErrUnsupportedFormat = errors.New("unsupported format")
	ErrCorruptedFile     = errors.New("file corrupted or unreadable")
	ErrFileTooLarge      = errors.New("file exceeds size limit")
	ErrTimeout           = errors.New("operation timed out")
	ErrToolNotInstalled  = errors.New("required tool not installed")
)

// ProcessError represents a failure in an external process
type ProcessError struct {
	Tool     string // "spleeter", "basic-pitch", "librosa"
	Stage    string // "stem_separation", "transcription", "analysis"
	ExitCode int
	Stderr   string
	Cause    error
}

func (e *ProcessError) Error() string {
	if e.Stderr != "" {
		return fmt.Sprintf("%s failed at %s (exit %d): %s", e.Tool, e.Stage, e.ExitCode, e.Stderr)
	}
	return fmt.Sprintf("%s failed at %s (exit %d)", e.Tool, e.Stage, e.ExitCode)
}

func (e *ProcessError) Unwrap() error {
	return e.Cause
}

// IsRecoverable returns true if fallback strategy exists
func (e *ProcessError) IsRecoverable() bool {
	return e.Tool == "spleeter" && e.Stage == "stem_separation"
}

// NewProcessError creates a ProcessError
func NewProcessError(tool, stage string, exitCode int, stderr string, cause error) *ProcessError {
	return &ProcessError{
		Tool:     tool,
		Stage:    stage,
		ExitCode: exitCode,
		Stderr:   stderr,
		Cause:    cause,
	}
}
