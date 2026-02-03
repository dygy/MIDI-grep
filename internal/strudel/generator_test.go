package strudel

import (
	"strings"
	"testing"

	"github.com/arkadiishvartcman/midi-grep/internal/analysis"
	"github.com/arkadiishvartcman/midi-grep/internal/midi"
)

func TestAutoOutputMode(t *testing.T) {
	analysisResult := &analysis.Result{
		BPM:           120,
		Key:           "C major",
		TimeSignature: "4/4",
	}

	t.Run("SmallOutput_UsesSeparate", func(t *testing.T) {
		// Small output (1 bar) should use separate patterns
		notes := []midi.Note{
			{Pitch: 36, Start: 0, Duration: 0.5, Velocity: 100},
			{Pitch: 60, Start: 0, Duration: 0.5, Velocity: 80},
		}

		gen := NewGeneratorWithStyle(16, StylePiano)
		output := gen.Generate(notes, analysisResult)

		// Should have $bass: and $mid: patterns
		if !strings.Contains(output, "$bass:") {
			t.Error("Small output should use separate mode with '$bass:'")
		}
		// Should NOT have array syntax
		if strings.Contains(output, "let bass = [") {
			t.Error("Small output should NOT use chunked arrays")
		}
	})

	t.Run("LargeOutput_UsesChunked", func(t *testing.T) {
		// Large output (8 bars) should use chunked arrays
		var notes []midi.Note
		for bar := 0; bar < 8; bar++ {
			notes = append(notes, midi.Note{
				Pitch:    36 + bar,
				Start:    float64(bar) * 2.0, // 2 seconds per bar at 120 BPM
				Duration: 0.5,
				Velocity: 100,
			})
			notes = append(notes, midi.Note{
				Pitch:    60 + bar,
				Start:    float64(bar) * 2.0,
				Duration: 0.5,
				Velocity: 80,
			})
		}

		gen := NewGeneratorWithStyle(16, StylePiano)
		output := gen.Generate(notes, analysisResult)

		// Should have array syntax
		if !strings.Contains(output, "let bass = [") {
			t.Error("Large output should use chunked mode with 'let bass = ['")
		}
		if !strings.Contains(output, "cat(...bass)") {
			t.Error("Large output should have cat() helper")
		}

		t.Logf("\n=== Large output (auto-chunked) ===\n%s", output)
	})
}

func TestOutputFormat(t *testing.T) {
	notes := []midi.Note{
		{Pitch: 36, Start: 0, Duration: 0.5, Velocity: 100},
		{Pitch: 60, Start: 0, Duration: 0.5, Velocity: 80},
	}

	analysisResult := &analysis.Result{
		BPM:           120,
		Key:           "C major",
		TimeSignature: "4/4",
	}

	gen := NewGeneratorWithStyle(16, StylePiano)
	output := gen.Generate(notes, analysisResult)

	// Check basic structure
	if !strings.Contains(output, "setcps(") {
		t.Error("Output should contain setcps()")
	}
	if !strings.Contains(output, "note(") {
		t.Error("Output should contain note()")
	}
	if !strings.Contains(output, ".sound(") {
		t.Error("Output should contain .sound()")
	}
}
