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

func TestGenTrainArgs(t *testing.T) {
	cases := []struct {
		name     string
		mode     string
		midi     string
		grainMS  int
		grainSet bool
		want     []string // flag/value pairs that must appear ("--x" alone = flag must appear)
		absent   []string
	}{
		{"midi only: no grain-ms", "granular", "/s/bass.mid", 0, false, []string{"--midi", "/s/bass.mid"}, []string{"--grain-ms"}},
		{"neither: no grain-ms", "granular", "", 0, false, nil, []string{"--grain-ms", "--midi"}},
		{"explicit grain-ms passed", "granular", "", 250, true, []string{"--grain-ms", "250"}, []string{"--midi"}},
		{"midi and explicit grain-ms", "granular", "/s/l.mid", 80, true, []string{"--midi", "/s/l.mid", "--grain-ms", "80"}, nil},
		{"rave ignores grain/midi", "rave", "/s/l.mid", 80, true, []string{"--epochs"}, []string{"--grain-ms", "--midi"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			genTrainingMode, genMidiPath, genGrainMS = tc.mode, tc.midi, tc.grainMS
			got := genTrainArgs("/a.wav", "/out", tc.grainSet)
			idx := func(f string) int {
				for i, a := range got {
					if a == f {
						return i
					}
				}
				return -1
			}
			for i := 0; i < len(tc.want); i++ {
				j := idx(tc.want[i])
				if j < 0 {
					t.Fatalf("missing %q in %v", tc.want[i], got)
				}
				if tc.want[i] != "--epochs" && i+1 < len(tc.want) && tc.want[i+1][0] != '-' {
					if j+1 >= len(got) || got[j+1] != tc.want[i+1] {
						t.Fatalf("%s value: want %q in %v", tc.want[i], tc.want[i+1], got)
					}
					i++
				}
			}
			for _, a := range tc.absent {
				if idx(a) >= 0 {
					t.Errorf("unexpected %q in %v", a, got)
				}
			}
		})
	}
}
