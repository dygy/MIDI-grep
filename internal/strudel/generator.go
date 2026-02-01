package strudel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/arkadiishvartcman/midi-grep/internal/analysis"
	"github.com/arkadiishvartcman/midi-grep/internal/midi"
)

// Generator converts MIDI notes to Strudel code
type Generator struct {
	quantize int
}

// VoiceNotes holds notes for a specific voice range
type VoiceNotes struct {
	Name  string
	Notes []midi.Note
}

// CleanupResult represents the JSON output from cleanup.py
type CleanupResult struct {
	Notes  []NoteJSON `json:"notes"`
	Voices struct {
		Bass []NoteJSON `json:"bass"`
		Mid  []NoteJSON `json:"mid"`
		High []NoteJSON `json:"high"`
	} `json:"voices"`
	Stats struct {
		Total     int `json:"total"`
		BassCount int `json:"bass_count"`
		MidCount  int `json:"mid_count"`
		HighCount int `json:"high_count"`
	} `json:"stats"`
	Tempo      float64 `json:"tempo"`
	TotalBeats float64 `json:"total_beats"`
	Quantize   int     `json:"quantize"`
}

// NoteJSON represents a note from the cleanup JSON
type NoteJSON struct {
	Pitch              int     `json:"pitch"`
	Start              float64 `json:"start"`
	Duration           float64 `json:"duration"`
	DurationBeats      float64 `json:"duration_beats"`
	Velocity           int     `json:"velocity"`
	VelocityNormalized float64 `json:"velocity_normalized"`
}

// NewGenerator creates a new Strudel code generator
func NewGenerator(quantize int) *Generator {
	return &Generator{quantize: quantize}
}

// GenerateFromJSON creates Strudel code from cleanup JSON file
func (g *Generator) GenerateFromJSON(jsonPath string, analysisResult *analysis.Result) (string, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return "", fmt.Errorf("failed to read JSON: %w", err)
	}

	var result CleanupResult
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("failed to parse JSON: %w", err)
	}

	return g.generateFromCleanup(&result, analysisResult), nil
}

// Generate creates Strudel code from notes and analysis (legacy method)
func (g *Generator) Generate(notes []midi.Note, analysisResult *analysis.Result) string {
	var sb strings.Builder

	// Header
	sb.WriteString("// MIDI-grep output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f", analysisResult.BPM))
	if analysisResult.Key != "" {
		sb.WriteString(fmt.Sprintf(", Key: %s", analysisResult.Key))
	}
	sb.WriteString(fmt.Sprintf("\n// Notes: %d\n", len(notes)))

	// Tempo
	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", analysisResult.BPM))

	// Separate voices
	var bass, mid, high []midi.Note
	for _, n := range notes {
		switch {
		case n.Pitch < 48:
			bass = append(bass, n)
		case n.Pitch < 72:
			mid = append(mid, n)
		default:
			high = append(high, n)
		}
	}

	// Generate stacked pattern
	g.generateStackedPattern(&sb, bass, mid, high, analysisResult.BPM)

	return sb.String()
}

// generateFromCleanup creates rich Strudel output from cleanup result
func (g *Generator) generateFromCleanup(result *CleanupResult, analysisResult *analysis.Result) string {
	var sb strings.Builder

	// Header with stats
	sb.WriteString("// MIDI-grep output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f", analysisResult.BPM))
	if analysisResult.Key != "" {
		sb.WriteString(fmt.Sprintf(", Key: %s", analysisResult.Key))
	}
	sb.WriteString(fmt.Sprintf("\n// Notes: %d (bass: %d, mid: %d, high: %d)\n",
		result.Stats.Total, result.Stats.BassCount, result.Stats.MidCount, result.Stats.HighCount))
	sb.WriteString(fmt.Sprintf("// Duration: %.1f beats\n\n", result.TotalBeats))

	// Tempo
	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", analysisResult.BPM))

	// Convert JSON notes to midi.Note
	bass := jsonToNotes(result.Voices.Bass)
	mid := jsonToNotes(result.Voices.Mid)
	high := jsonToNotes(result.Voices.High)

	// Generate the stacked pattern
	g.generateStackedPattern(&sb, bass, mid, high, analysisResult.BPM)

	return sb.String()
}

func jsonToNotes(jsonNotes []NoteJSON) []midi.Note {
	notes := make([]midi.Note, len(jsonNotes))
	for i, n := range jsonNotes {
		notes[i] = midi.Note{
			Pitch:    n.Pitch,
			Start:    n.Start,
			Duration: n.Duration,
			Velocity: n.Velocity,
		}
	}
	return notes
}

// generateStackedPattern creates a Strudel stack() with separate voices
func (g *Generator) generateStackedPattern(sb *strings.Builder, bass, mid, high []midi.Note, bpm float64) {
	// Calculate timing
	beatDuration := 60.0 / bpm
	gridSize := beatDuration / float64(g.quantize/4)

	// Find total duration across all voices
	maxEnd := 0.0
	for _, notes := range [][]midi.Note{bass, mid, high} {
		for _, n := range notes {
			if end := n.Start + n.Duration; end > maxEnd {
				maxEnd = end
			}
		}
	}

	numBars := int(math.Ceil(maxEnd / (beatDuration * 4)))
	if numBars < 1 {
		numBars = 1
	}
	if numBars > 16 {
		numBars = 16 // Limit to 16 bars for readability
	}

	// Count non-empty voices
	voices := []struct {
		name   string
		notes  []midi.Note
		sound  string
		octave int
	}{
		{"bass", bass, "piano", -1},
		{"mid", mid, "piano", 0},
		{"high", high, "piano", 0},
	}

	var activeVoices []string
	for _, v := range voices {
		if len(v.notes) > 0 {
			pattern := g.voiceToPattern(v.notes, gridSize, numBars, beatDuration)
			if pattern != "" && pattern != "~" {
				voiceCode := fmt.Sprintf("  // %s (%d notes)\n", v.name, len(v.notes))
				voiceCode += fmt.Sprintf("  note(\"%s\")\n", pattern)
				voiceCode += fmt.Sprintf("    .sound(\"%s\")", v.sound)
				if v.name == "bass" {
					voiceCode += "\n    .gain(1.2)"
				}
				activeVoices = append(activeVoices, voiceCode)
			}
		}
	}

	if len(activeVoices) == 0 {
		sb.WriteString("$: note(\"c4\").sound(\"piano\")\n")
		return
	}

	if len(activeVoices) == 1 {
		// Single voice - no stack needed
		sb.WriteString("$: ")
		sb.WriteString(strings.TrimPrefix(activeVoices[0], "  "))
		sb.WriteString("\n  .room(0.3).size(0.6)\n")
	} else {
		// Multiple voices - use stack
		sb.WriteString("$: stack(\n")
		for i, voice := range activeVoices {
			sb.WriteString(voice)
			if i < len(activeVoices)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n")
		}
		sb.WriteString(")\n")
		sb.WriteString("  .room(0.3).size(0.6)\n")
	}
}

// voiceToPattern converts notes to Strudel mini-notation with proper bar structure
func (g *Generator) voiceToPattern(notes []midi.Note, gridSize float64, numBars int, beatDuration float64) string {
	if len(notes) == 0 {
		return "~"
	}

	// Grid slots per bar (e.g., 16 for 16th notes in 4/4)
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays
	slots := make([][]string, totalSlots)
	for i := range slots {
		slots[i] = []string{}
	}

	// Place notes in slots
	for _, n := range notes {
		slot := int(n.Start / gridSize)
		if slot >= 0 && slot < totalSlots {
			noteName := midiToNoteName(n.Pitch)
			slots[slot] = append(slots[slot], noteName)
		}
	}

	// Build pattern with bar structure
	var bars []string
	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			slot := slots[i]
			switch len(slot) {
			case 0:
				barParts = append(barParts, "~")
			case 1:
				barParts = append(barParts, slot[0])
			default:
				// Sort chord notes low to high
				sort.Strings(slot)
				barParts = append(barParts, "["+strings.Join(slot, ",")+"]")
			}
		}

		// Simplify consecutive rests within bar
		simplified := simplifyPattern(barParts)
		if simplified != "" && !isAllRests(simplified) {
			bars = append(bars, simplified)
		}
	}

	if len(bars) == 0 {
		return "~"
	}

	// Join bars with | for visual separation (Strudel treats space and | same)
	return strings.Join(bars, " | ")
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

	// Handle trailing rests - skip them for cleaner output
	// (Strudel will naturally loop)

	if len(result) == 0 {
		return "~"
	}

	return strings.Join(result, " ")
}

// isAllRests checks if pattern is only rests
func isAllRests(pattern string) bool {
	parts := strings.Fields(pattern)
	for _, p := range parts {
		if p != "~" && !strings.HasPrefix(p, "~*") {
			return false
		}
	}
	return true
}
