// Package codegen runs the Python Strudel code generators. Go does not generate Strudel code: the CLI pipeline
// and the web server both call one of these scripts and read its output here, so the two cannot drift apart
// (the server kept calling ai_code_generator.py after it was deleted and served a one-note placeholder).
package codegen

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/dygy/midi-grep/internal/exec"
)

// The generator scripts, relative to the runner's scripts dir.
const (
	ScriptSingle       = "ollama_codegen.py"
	ScriptOrchestrated = "codegen_orchestrator.py"
)

// BestPrevious is the best stored earlier run a generator reports alongside new code.
type BestPrevious struct {
	Code       string  `json:"code"`
	Similarity float64 `json:"similarity"`
	Version    int     `json:"version"`
}

// Output is what a generator produced.
type Output struct {
	Code         string
	BestPrevious *BestPrevious
}

// Script returns the generator for a codegen mode: "orchestrated" or the single-call generator.
func Script(mode string) string {
	if mode == "orchestrated" {
		return ScriptOrchestrated
	}
	return ScriptSingle
}

// Parse reads a generator's stdout: a JSON object {"code": ..., "best_previous": ...}, or raw Strudel code.
func Parse(stdout string) Output {
	var v struct {
		Code         string        `json:"code"`
		BestPrevious *BestPrevious `json:"best_previous"`
	}
	if json.Unmarshal([]byte(stdout), &v) == nil && v.Code != "" {
		return Output{Code: v.Code, BestPrevious: v.BestPrevious}
	}
	return Output{Code: stdout}
}

// Generate runs script with args and returns its parsed output. A failed run or empty code is an error.
func Generate(ctx context.Context, r *exec.Runner, script string, args ...string) (Output, error) {
	res, err := r.RunScript(ctx, script, args...)
	if err != nil {
		return Output{}, fmt.Errorf("%s: %w", script, err)
	}
	out := Parse(res.Stdout)
	if strings.TrimSpace(out.Code) == "" {
		return Output{}, fmt.Errorf("%s produced no code", script)
	}
	return out, nil
}
