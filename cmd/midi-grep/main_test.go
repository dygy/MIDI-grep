package main

// @layer: unit
// @spec: 004-second-reference-track
// @regression

import (
	"os"
	"path/filepath"
	"testing"
)

func TestShouldGenerateReport(t *testing.T) {
	cases := []struct {
		mode, rendered string
		want           bool
	}{
		{"none", "", false},
		{"none", "/x/render.wav", false}, // --render none wins even if a stale render exists
		{"auto", "", false},              // nothing rendered → nothing to report
		{"auto", "/x/render.wav", true},
		{"/tmp/out.wav", "/tmp/out.wav", true},
	}
	for _, c := range cases {
		if got := shouldGenerateReport(c.mode, c.rendered); got != c.want {
			t.Errorf("shouldGenerateReport(%q, %q) = %v, want %v", c.mode, c.rendered, got, c.want)
		}
	}
}

func TestAbsOutputDir(t *testing.T) {
	wd, _ := os.Getwd()
	if got := absOutputDir("models"); got != filepath.Join(wd, "models") {
		t.Fatalf("relative dir not anchored to cwd: %q", got)
	}
	if got := absOutputDir(""); got != filepath.Join(wd, "models") {
		t.Fatalf("empty dir should default to <cwd>/models: %q", got)
	}
	if got := absOutputDir("/abs/models"); got != "/abs/models" {
		t.Fatalf("absolute dir must be kept: %q", got)
	}
}
