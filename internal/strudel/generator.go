package strudel

import (
	"fmt"
	"strings"

	"github.com/arkadiishvartcman/midi-grep/internal/analysis"
	"github.com/arkadiishvartcman/midi-grep/internal/midi"
)

// Generator converts MIDI notes to Strudel.cc code
type Generator struct {
	quantize int
}

// NewGenerator creates a new Strudel code generator
func NewGenerator(quantize int) *Generator {
	return &Generator{quantize: quantize}
}

// Generate creates Strudel.cc code from notes and analysis
func (g *Generator) Generate(notes []midi.Note, analysis *analysis.Result) string {
	var sb strings.Builder

	// Header comments
	sb.WriteString("// MIDI-grep output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f", analysis.BPM))
	if analysis.Key != "" {
		sb.WriteString(fmt.Sprintf(", Key: %s", analysis.Key))
	}
	sb.WriteString("\n")

	// Tempo setting
	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", analysis.BPM))

	// Convert notes to pattern
	pattern := g.notesToPattern(notes, analysis.BPM)

	// Main pattern
	sb.WriteString(fmt.Sprintf("$: note(\"%s\")\n", pattern))
	sb.WriteString("  .sound(\"piano\")\n")
	sb.WriteString("  .room(0.3).size(0.6)\n")

	return sb.String()
}

// notesToPattern converts MIDI notes to Strudel mini-notation
func (g *Generator) notesToPattern(notes []midi.Note, bpm float64) string {
	if len(notes) == 0 {
		return "~"
	}

	// Calculate beat duration and grid size
	beatDuration := 60.0 / bpm
	gridSize := beatDuration / float64(g.quantize/4)

	// Find time span
	maxEnd := 0.0
	for _, n := range notes {
		if end := n.Start + n.Duration; end > maxEnd {
			maxEnd = end
		}
	}

	// Quantize to grid
	numSlots := int(maxEnd/gridSize) + 1
	if numSlots > 256 {
		numSlots = 256 // Limit pattern length
	}

	slots := make([][]string, numSlots)
	for i := range slots {
		slots[i] = []string{}
	}

	for _, n := range notes {
		slot := int(n.Start / gridSize)
		if slot < numSlots {
			slots[slot] = append(slots[slot], midiToNoteName(n.Pitch))
		}
	}

	// Build pattern string
	var parts []string
	for _, slot := range slots {
		switch len(slot) {
		case 0:
			parts = append(parts, "~")
		case 1:
			parts = append(parts, slot[0])
		default:
			// Chord notation
			parts = append(parts, "["+strings.Join(slot, ",")+"]")
		}
	}

	// Simplify consecutive rests
	return simplifyPattern(parts)
}

// midiToNoteName converts MIDI note number to Strudel notation
func midiToNoteName(midiNote int) string {
	noteNames := []string{"c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"}
	octave := (midiNote / 12) - 1
	note := midiNote % 12
	return fmt.Sprintf("%s%d", noteNames[note], octave)
}

// simplifyPattern reduces consecutive rests
func simplifyPattern(parts []string) string {
	if len(parts) == 0 {
		return "~"
	}

	var result []string
	restCount := 0

	for _, p := range parts {
		if p == "~" {
			restCount++
		} else {
			if restCount > 0 {
				if restCount == 1 {
					result = append(result, "~")
				} else {
					result = append(result, fmt.Sprintf("~*%d", restCount))
				}
				restCount = 0
			}
			result = append(result, p)
		}
	}

	// Handle trailing rests
	if restCount > 0 {
		if restCount == 1 {
			result = append(result, "~")
		} else {
			result = append(result, fmt.Sprintf("~*%d", restCount))
		}
	}

	if len(result) == 0 {
		return "~"
	}

	return strings.Join(result, " ")
}
