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

// SoundStyle defines preset sound combinations for different genres
type SoundStyle string

const (
	StylePiano      SoundStyle = "piano"
	StyleSynth      SoundStyle = "synth"
	StyleOrchestral SoundStyle = "orchestral"
	StyleElectronic SoundStyle = "electronic"
	StyleJazz       SoundStyle = "jazz"
	StyleLofi       SoundStyle = "lofi"
	StyleAuto       SoundStyle = "auto"
)

// SoundPalette defines sounds for each voice (bass, mid, high)
type SoundPalette struct {
	Bass     string
	BassGain float64
	Mid      string
	MidGain  float64
	High     string
	HighGain float64
}

// Predefined sound palettes for each style
var soundPalettes = map[SoundStyle]SoundPalette{
	StylePiano: {
		Bass:     "gm_acoustic_grand_piano",
		BassGain: 1.2,
		Mid:      "gm_acoustic_grand_piano",
		MidGain:  1.0,
		High:     "gm_acoustic_grand_piano",
		HighGain: 0.9,
	},
	StyleSynth: {
		Bass:     "gm_synth_bass_1",
		BassGain: 1.3,
		Mid:      "gm_pad_warm",
		MidGain:  0.8,
		High:     "gm_lead_2_sawtooth",
		HighGain: 0.7,
	},
	StyleOrchestral: {
		Bass:     "gm_contrabass",
		BassGain: 1.2,
		Mid:      "gm_string_ensemble_1",
		MidGain:  1.0,
		High:     "gm_violin",
		HighGain: 0.9,
	},
	StyleElectronic: {
		Bass:     "gm_synth_bass_2",
		BassGain: 1.4,
		Mid:      "gm_pad_poly",
		MidGain:  0.7,
		High:     "gm_lead_1_square",
		HighGain: 0.6,
	},
	StyleJazz: {
		Bass:     "gm_acoustic_bass",
		BassGain: 1.2,
		Mid:      "gm_electric_piano_1",
		MidGain:  1.0,
		High:     "gm_vibraphone",
		HighGain: 0.8,
	},
	StyleLofi: {
		Bass:     "gm_electric_bass_finger",
		BassGain: 1.1,
		Mid:      "gm_electric_piano_2",
		MidGain:  0.9,
		High:     "gm_music_box",
		HighGain: 0.7,
	},
}

// Generator converts MIDI notes to Strudel code
type Generator struct {
	quantize int
	style    SoundStyle
	palette  SoundPalette
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

// NewGenerator creates a new Strudel code generator with default piano style
func NewGenerator(quantize int) *Generator {
	return NewGeneratorWithStyle(quantize, StylePiano)
}

// NewGeneratorWithStyle creates a generator with specified sound style
func NewGeneratorWithStyle(quantize int, style SoundStyle) *Generator {
	palette, ok := soundPalettes[style]
	if !ok {
		palette = soundPalettes[StylePiano]
	}
	return &Generator{
		quantize: quantize,
		style:    style,
		palette:  palette,
	}
}

// SetStyle changes the sound style
func (g *Generator) SetStyle(style SoundStyle) {
	g.style = style
	if palette, ok := soundPalettes[style]; ok {
		g.palette = palette
	}
}

// SetCustomPalette allows setting custom sounds for each voice
func (g *Generator) SetCustomPalette(bass, mid, high string) {
	g.palette.Bass = bass
	g.palette.Mid = mid
	g.palette.High = high
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
	sb.WriteString(fmt.Sprintf("\n// Notes: %d, Style: %s\n", len(notes), g.style))

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
	sb.WriteString(fmt.Sprintf("// Style: %s\n", g.style))
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

// generateStackedPattern creates a Strudel stack() with separate voices and GM sounds
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

	// Define voices with their sounds from palette
	voices := []struct {
		name  string
		notes []midi.Note
		sound string
		gain  float64
	}{
		{"bass", bass, g.palette.Bass, g.palette.BassGain},
		{"mid", mid, g.palette.Mid, g.palette.MidGain},
		{"high", high, g.palette.High, g.palette.HighGain},
	}

	var activeVoices []string
	for _, v := range voices {
		if len(v.notes) > 0 {
			pattern := g.voiceToPattern(v.notes, gridSize, numBars, beatDuration)
			if pattern != "" && pattern != "~" {
				voiceCode := fmt.Sprintf("  // %s (%d notes)\n", v.name, len(v.notes))
				voiceCode += fmt.Sprintf("  note(\"%s\")\n", pattern)
				voiceCode += fmt.Sprintf("    .sound(\"%s\")", v.sound)
				if v.gain != 1.0 {
					voiceCode += fmt.Sprintf("\n    .gain(%.1f)", v.gain)
				}
				activeVoices = append(activeVoices, voiceCode)
			}
		}
	}

	if len(activeVoices) == 0 {
		sb.WriteString(fmt.Sprintf("$: note(\"c4\").sound(\"%s\")\n", g.palette.Mid))
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

// AvailableStyles returns list of available sound styles
func AvailableStyles() []SoundStyle {
	return []SoundStyle{
		StylePiano,
		StyleSynth,
		StyleOrchestral,
		StyleElectronic,
		StyleJazz,
		StyleLofi,
	}
}

// StyleDescription returns a description for each style
func StyleDescription(style SoundStyle) string {
	descriptions := map[SoundStyle]string{
		StylePiano:      "Acoustic grand piano for all voices",
		StyleSynth:      "Synth bass, warm pad, sawtooth lead",
		StyleOrchestral: "Contrabass, strings, violin",
		StyleElectronic: "Synth bass, poly pad, square lead",
		StyleJazz:       "Acoustic bass, electric piano, vibraphone",
		StyleLofi:       "Finger bass, electric piano 2, music box",
	}
	return descriptions[style]
}
