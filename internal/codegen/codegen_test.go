package codegen

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name, stdout, code string
		best               *BestPrevious
	}{
		{"json with best", `{"code":"$: s(\"bd\")","best_previous":{"code":"x","similarity":0.8,"version":3}}`,
			`$: s("bd")`, &BestPrevious{Code: "x", Similarity: 0.8, Version: 3}},
		{"json without best", `{"code":"$: s(\"bd\")"}`, `$: s("bd")`, nil},
		{"raw code", "$: s(\"bd\")\n", "$: s(\"bd\")\n", nil},
		{"json without code is raw", `{"error":"x"}`, `{"error":"x"}`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Parse(tt.stdout)
			if got.Code != tt.code {
				t.Errorf("code = %q, want %q", got.Code, tt.code)
			}
			if (got.BestPrevious == nil) != (tt.best == nil) || (tt.best != nil && *got.BestPrevious != *tt.best) {
				t.Errorf("best = %+v, want %+v", got.BestPrevious, tt.best)
			}
		})
	}
}

func TestScript(t *testing.T) {
	if Script("orchestrated") != ScriptOrchestrated || Script("single") != ScriptSingle || Script("") != ScriptSingle {
		t.Fatal("Script does not map the codegen modes")
	}
}

// Regression (2026-11-01): internal/server/jobs.go kept calling ai_code_generator.py after it was deleted, so every
// web-UI job "succeeded" with a one-note placeholder; cmd/midi-grep kept a dead path to render-strudel-node.js.
// Every Python / Node script named in the Go code must exist (Node by its TypeScript source: dist/ is build output).
func TestScriptsNamedInGoExist(t *testing.T) {
	root := filepath.Join("..", "..")
	lit := regexp.MustCompile(`"([A-Za-z0-9_./-]+\.(?:py|js))"`)
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (strings.HasPrefix(d.Name(), ".") && path != root || d.Name() == "node_modules") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range lit.FindAllStringSubmatch(string(src), -1) {
			name := filepath.Base(m[1])
			var cands []string
			if strings.HasSuffix(name, ".js") {
				cands = []string{filepath.Join(root, "scripts", "node", "src", strings.TrimSuffix(name, ".js")+".ts")}
			} else {
				cands = []string{filepath.Join(root, m[1]), filepath.Join(root, "scripts", "python", m[1])}
				// a bare name may live in a scripts/python subdirectory joined at the call site (training/train.py)
				sub, _ := filepath.Glob(filepath.Join(root, "scripts", "python", "*", name))
				cands = append(cands, sub...)
			}
			found := false
			for _, c := range cands {
				if _, err := os.Stat(c); err == nil {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s names %s, which does not exist (looked in %v)", path, m[1], cands)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
