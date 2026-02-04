package exec

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Result holds command execution output
type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
	Duration time.Duration
}

// Runner executes external commands with context support
type Runner struct {
	PythonPath string
	ScriptsDir string
}

// NewRunner creates a new command runner
func NewRunner(pythonPath, scriptsDir string) *Runner {
	if pythonPath == "" {
		// Try to find Python in virtual environment first
		venvPython := filepath.Join(scriptsDir, ".venv", "bin", "python")
		if _, err := os.Stat(venvPython); err == nil {
			pythonPath = venvPython
		} else {
			pythonPath = "python3"
		}
	}
	return &Runner{
		PythonPath: pythonPath,
		ScriptsDir: scriptsDir,
	}
}

// RunScript executes a Python script with arguments
func (r *Runner) RunScript(ctx context.Context, script string, args ...string) (*Result, error) {
	scriptPath := filepath.Join(r.ScriptsDir, script)
	fullArgs := append([]string{scriptPath}, args...)
	return r.execute(ctx, r.PythonPath, fullArgs...)
}

// RunModule executes a Python module with -m flag
func (r *Runner) RunModule(ctx context.Context, module string, args ...string) (*Result, error) {
	// Set PYTHONPATH to include scripts directory for local modules
	cmd := exec.CommandContext(ctx, r.PythonPath, append([]string{"-m", module}, args...)...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Dir = r.ScriptsDir
	cmd.Env = append(os.Environ(), fmt.Sprintf("PYTHONPATH=%s", r.ScriptsDir))

	start := time.Now()
	err := cmd.Run()

	result := &Result{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}

	if exitErr, ok := err.(*exec.ExitError); ok {
		result.ExitCode = exitErr.ExitCode()
	}

	if err != nil {
		return result, fmt.Errorf("module %s failed: %w\nstderr: %s", module, err, result.Stderr)
	}

	return result, nil
}

// execute runs a command and captures output
func (r *Runner) execute(ctx context.Context, name string, args ...string) (*Result, error) {
	start := time.Now()

	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	result := &Result{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}

	if exitErr, ok := err.(*exec.ExitError); ok {
		result.ExitCode = exitErr.ExitCode()
	}

	if err != nil {
		return result, fmt.Errorf("command failed: %w", err)
	}

	return result, nil
}

// CheckPythonDependency verifies a Python package is installed
func (r *Runner) CheckPythonDependency(ctx context.Context, packageName string) error {
	result, err := r.execute(ctx, r.PythonPath, "-c", fmt.Sprintf("import %s", packageName))
	if err != nil {
		return fmt.Errorf("%s not installed: %s", packageName, result.Stderr)
	}
	return nil
}
