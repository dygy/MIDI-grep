package strudel

import (
	"fmt"
	"strings"

	"github.com/dygy/midi-grep/internal/analysis"
	"github.com/dygy/midi-grep/internal/drums"
	"github.com/dygy/midi-grep/internal/midi"
)

// ArrangementGenerator creates musical Strudel code with chord progressions
type ArrangementGenerator struct {
	quantize      int
	style         SoundStyle
	chordDetector *ChordDetector
}

// NewArrangementGenerator creates a new arrangement-based generator
func NewArrangementGenerator(quantize int, style SoundStyle) *ArrangementGenerator {
	return &ArrangementGenerator{
		quantize:      quantize,
		style:         style,
		chordDetector: NewChordDetector(quantize),
	}
}

// GenerateArrangement creates full arrangement-based Strudel code
func (g *ArrangementGenerator) GenerateArrangement(notes []midi.Note, analysisResult *analysis.Result, drumResult *drums.DetectionResult, drumKit DrumKit) string {
	if len(notes) == 0 {
		return ""
	}

	var sb strings.Builder

	// Detect chords from notes
	progression := g.chordDetector.DetectChords(notes, analysisResult.BPM)

	// Detect key if not from analysis
	key := analysisResult.Key
	if key == "" && progression != nil {
		key = progression.DetectKey()
	}

	// Header
	sb.WriteString("// MIDI-grep arrangement output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f, Key: %s\n", analysisResult.BPM, key))
	sb.WriteString(fmt.Sprintf("// Style: %s\n", g.style))
	if progression != nil && len(progression.Chords) > 0 {
		sb.WriteString(fmt.Sprintf("// Chords: %d detected\n", len(progression.Chords)))
	}
	sb.WriteString("\n")

	// Generate chord variable if we have chords
	hasChords := progression != nil && len(progression.Chords) >= 2
	if hasChords {
		chordDef := progression.ToStrudelArrange()
		sb.WriteString(chordDef)
		sb.WriteString("\n\n")
	}

	// Tempo
	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", analysisResult.BPM))

	// Main stack
	sb.WriteString("stack(\n")

	// Generate melodic parts
	var parts []string

	if hasChords {
		// Chord-based generation
		parts = append(parts, g.generatePad(progression))
		parts = append(parts, g.generateBass(progression))
		parts = append(parts, g.generateArpeggio(progression))
		parts = append(parts, g.generateMelody(notes, progression, analysisResult.BPM))
	} else {
		// Fallback to note-based generation
		parts = append(parts, g.generateNoteParts(notes, analysisResult.BPM))
	}

	// Add drums if present
	if drumResult != nil && len(drumResult.Hits) > 0 {
		drumPart := g.generateDrumSection(drumResult, drumKit)
		if drumPart != "" {
			parts = append(parts, drumPart)
		}
	}

	// Filter empty parts and write
	var activeParts []string
	for _, p := range parts {
		if strings.TrimSpace(p) != "" {
			activeParts = append(activeParts, p)
		}
	}

	for i, part := range activeParts {
		sb.WriteString(part)
		if i < len(activeParts)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}

	sb.WriteString(").late(0.02)\n")

	return sb.String()
}

// generatePad creates a pad/chord voicing part
func (g *ArrangementGenerator) generatePad(prog *ChordProgression) string {
	if prog == nil || len(prog.Chords) == 0 {
		return ""
	}

	sound := g.getPadSound()

	var sb strings.Builder
	sb.WriteString("  // pad\n")
	sb.WriteString("  chords.dict('ireal-ext')\n")
	sb.WriteString("    .set.mix(offset(\"<0!3 [1 2 1 4]>\"))\n")
	sb.WriteString("    .voicing()\n")
	sb.WriteString(fmt.Sprintf("    .s(\"%s\")\n", sound))
	sb.WriteString("    .clip(0.9).attack(0.3)\n")
	sb.WriteString("    .lpf(sine.range(100, 400).slow(8)).lpq(4)\n")
	sb.WriteString("    .room(0.5).gain(0.4).pan(0.4)\n")
	sb.WriteString("    .mask(\"<1!32 0!16>\")")

	return sb.String()
}

// generateBass creates a bass part from chord roots
func (g *ArrangementGenerator) generateBass(prog *ChordProgression) string {
	if prog == nil || len(prog.Chords) == 0 {
		return ""
	}

	sound := g.getBassSound()

	var sb strings.Builder
	sb.WriteString("  // bass\n")
	sb.WriteString("  chords.rootNotes(1)\n")
	sb.WriteString(fmt.Sprintf("    .s(\"%s\")\n", sound))
	sb.WriteString("    .lpf(perlin.range(300, 600)).lpq(2)\n")
	sb.WriteString("    .ply(\"<2 4>\")\n")
	sb.WriteString("    .attack(0.02).sustain(0.2).clip(0.5)\n")
	sb.WriteString("    .sometimesBy(\"0 .5@3\", x => x.add(note(12)))\n")
	sb.WriteString("    .gain(0.5)\n")
	sb.WriteString("    .mask(\"<0!16 1!32>\")")

	return sb.String()
}

// generateArpeggio creates an arpeggiated part
func (g *ArrangementGenerator) generateArpeggio(prog *ChordProgression) string {
	if prog == nil || len(prog.Chords) == 0 {
		return ""
	}

	sound := g.getArpSound()

	var sb strings.Builder
	sb.WriteString("  // arpeggio\n")
	sb.WriteString("  n(run(8).slow(4).sub(\"<0 1 2 3>\"))\n")
	sb.WriteString("    .set(chords).voicing()\n")
	sb.WriteString("    .clip(sine.range(1, 3).slow(16))\n")
	sb.WriteString("    .add.squeeze(note(\"0 -12\"))\n")
	sb.WriteString("    .room(0.8).jux(rev)\n")
	sb.WriteString(fmt.Sprintf("    .s(\"%s\")\n", sound))
	sb.WriteString("    .hpf(perlin.range(600, 2000)).hpq(4)\n")
	sb.WriteString("    .gain(0.4).pan(0.6)\n")
	sb.WriteString("    .mask(\"<0!8 1!32 0!8>\")")

	return sb.String()
}

// generateMelody creates a melodic part based on detected notes and chords
func (g *ArrangementGenerator) generateMelody(notes []midi.Note, prog *ChordProgression, bpm float64) string {
	if len(notes) == 0 {
		return ""
	}

	sound := g.getMelodySound()

	// Extract a melodic pattern from the highest notes
	pattern := extractMelodicPattern(notes, bpm, 16)

	var sb strings.Builder
	sb.WriteString("  // melody\n")
	if prog != nil && len(prog.Chords) > 0 {
		sb.WriteString(fmt.Sprintf("  n(\"%s\").set(chords).voicing()\n", pattern))
	} else {
		sb.WriteString(fmt.Sprintf("  note(\"%s\")\n", pattern))
	}
	sb.WriteString("    .clip(perlin.range(0.4, 0.8))\n")
	sb.WriteString("    .juxBy(0.5, rev)\n")
	sb.WriteString(fmt.Sprintf("    .s(\"%s\")\n", sound))
	sb.WriteString("    .delay(0.4)\n")
	sb.WriteString("    .lpf(perlin.range(300, 4000).slow(4)).lpq(6)\n")
	sb.WriteString("    .room(0.7).gain(rand.range(0.4, 0.6))\n")
	sb.WriteString("    .mask(\"<0!32 1!32>\")")

	return sb.String()
}

// generateNoteParts creates parts from raw notes when no chords detected
func (g *ArrangementGenerator) generateNoteParts(notes []midi.Note, bpm float64) string {
	// Separate into bass, mid, high
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

	var parts []string

	if len(mid) > 0 {
		pattern := extractNotePattern(mid, bpm, 16)
		part := fmt.Sprintf("  // mid\n  note(\"%s\")\n    .s(\"%s\")\n    .lpf(perlin.range(200, 2000)).lpq(4)\n    .room(0.5).gain(0.5)",
			pattern, g.getMelodySound())
		parts = append(parts, part)
	}

	if len(bass) > 0 {
		pattern := extractNotePattern(bass, bpm, 8)
		part := fmt.Sprintf("  // bass\n  note(\"%s\")\n    .s(\"%s\")\n    .lpf(400).lpq(2)\n    .gain(0.6)",
			pattern, g.getBassSound())
		parts = append(parts, part)
	}

	if len(high) > 0 {
		pattern := extractNotePattern(high, bpm, 16)
		part := fmt.Sprintf("  // high\n  note(\"%s\")\n    .s(\"%s\")\n    .hpf(800).room(0.7)\n    .gain(0.4)",
			pattern, g.getArpSound())
		parts = append(parts, part)
	}

	return strings.Join(parts, ",\n")
}

// generateDrumSection creates the drum stack
func (g *ArrangementGenerator) generateDrumSection(result *drums.DetectionResult, kit DrumKit) string {
	if result == nil || len(result.Hits) == 0 {
		return ""
	}

	kitInfo := DrumKitSoundsInfo[kit]
	if kitInfo.Sounds == nil {
		kitInfo = DrumKitSoundsInfo[DrumKitDefault]
	}

	bpm := result.Tempo
	if bpm <= 0 {
		bpm = 120
	}

	hitsByType := result.HitsByType()

	var drumParts []string

	// Kick
	if bdHits, ok := hitsByType["bd"]; ok && len(bdHits) > 0 {
		pattern := buildSimpleDrumPattern(bdHits, bpm, g.quantize)
		part := fmt.Sprintf("    s(\"%s\").bank(\"%s\").speed(0.9).gain(0.5)", pattern, kitInfo.Bank)
		drumParts = append(drumParts, part)
	}

	// Snare/Rim
	if sdHits, ok := hitsByType["sd"]; ok && len(sdHits) > 0 {
		pattern := buildSimpleDrumPattern(sdHits, bpm, g.quantize)
		part := fmt.Sprintf("    s(\"%s\").bank(\"%s\")\n      .lastOf(8, ply(\"~ [2 4]\"))", pattern, kitInfo.Bank)
		drumParts = append(drumParts, part)
	}

	// Hi-hats
	hhHits := hitsByType["hh"]
	ohHits := hitsByType["oh"]
	if len(hhHits) > 0 || len(ohHits) > 0 {
		allHH := append(hhHits, ohHits...)
		pattern := buildSimpleDrumPattern(allHH, bpm, g.quantize)
		part := fmt.Sprintf("    s(\"%s\").bank(\"%s\")\n      .gain(saw.range(0.2, 0.6))\n      .cut(1).release(0.02)\n      .lpf(saw.range(1000, 2000).slow(4))", pattern, kitInfo.Bank)
		drumParts = append(drumParts, part)
	}

	if len(drumParts) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("  // drums\n")
	sb.WriteString("  stack(\n")
	for i, p := range drumParts {
		sb.WriteString(p)
		if i < len(drumParts)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("  ).room(0.2)\n")
	sb.WriteString("    .off(1/8, x => x.mul(speed(0.5)).gain(0.25).degrade().hpf(300))\n")
	sb.WriteString("    .mask(\"<1!32 0!8 1>\")")

	return sb.String()
}

// Sound selection helpers based on style
func (g *ArrangementGenerator) getPadSound() string {
	switch g.style {
	case StyleOrchestral, StyleCinematic:
		return "gm_string_ensemble_1"
	case StyleJazz:
		return "gm_epiano1"
	case StyleLofi:
		return "gm_epiano2"
	case StyleElectronic, StyleHouse, StyleTrance:
		return "gm_pad_poly"
	default:
		return "gm_church_organ"
	}
}

func (g *ArrangementGenerator) getBassSound() string {
	switch g.style {
	case StyleOrchestral, StyleCinematic:
		return "gm_contrabass"
	case StyleJazz:
		return "gm_acoustic_bass"
	case StyleElectronic, StyleHouse, StyleTrance:
		return "supersaw"
	case StyleLofi:
		return "gm_electric_bass_finger"
	default:
		return "gm_synth_bass_1"
	}
}

func (g *ArrangementGenerator) getArpSound() string {
	switch g.style {
	case StyleOrchestral, StyleCinematic:
		return "gm_orchestral_harp"
	case StyleJazz:
		return "gm_vibraphone"
	case StyleElectronic, StyleHouse, StyleTrance:
		return "gm_lead_2_sawtooth"
	case StyleLofi:
		return "gm_music_box"
	default:
		return "gm_church_organ"
	}
}

func (g *ArrangementGenerator) getMelodySound() string {
	switch g.style {
	case StyleOrchestral, StyleCinematic:
		return "gm_violin"
	case StyleJazz:
		return "gm_alto_sax"
	case StyleElectronic, StyleHouse, StyleTrance:
		return "gm_lead_5_charang"
	case StyleLofi:
		return "gm_epiano2"
	default:
		return "gm_bagpipe"
	}
}

// extractMelodicPattern creates a scale-degree pattern from notes
func extractMelodicPattern(notes []midi.Note, bpm float64, maxNotes int) string {
	if len(notes) == 0 {
		return "0"
	}

	// Get high notes (likely melody)
	var melodicNotes []midi.Note
	for _, n := range notes {
		if n.Pitch >= 60 { // Middle C and above
			melodicNotes = append(melodicNotes, n)
		}
	}

	if len(melodicNotes) == 0 {
		melodicNotes = notes
	}

	// Limit to maxNotes
	if len(melodicNotes) > maxNotes {
		melodicNotes = melodicNotes[:maxNotes]
	}

	// Convert to scale degrees (simplified: use pitch class)
	var degrees []string
	for _, n := range melodicNotes {
		degree := (n.Pitch - 60) % 12 // Relative to middle C
		degrees = append(degrees, fmt.Sprintf("%d", degree))
	}

	// Add rhythm variation
	pattern := "<" + strings.Join(degrees, " ") + ">*<3 4>"
	return pattern
}

// extractNotePattern creates a note pattern string
func extractNotePattern(notes []midi.Note, bpm float64, maxNotes int) string {
	if len(notes) == 0 {
		return "c4"
	}

	if len(notes) > maxNotes {
		notes = notes[:maxNotes]
	}

	var noteNames []string
	for _, n := range notes {
		noteNames = append(noteNames, midiToNoteName(n.Pitch))
	}

	return strings.Join(noteNames, " ")
}

// buildSimpleDrumPattern creates a simple drum pattern string
func buildSimpleDrumPattern(hits []drums.Hit, bpm float64, quantize int) string {
	if len(hits) == 0 {
		return "~"
	}

	// Count hits per beat to determine pattern density
	hitsPerBeat := float64(len(hits)) * bpm / 60.0 / float64(quantize)

	// For dense patterns, use multiplier notation
	if hitsPerBeat > 0.5 {
		count := len(hits)
		if count >= 8 {
			return fmt.Sprintf("bd*%d", count/2)
		}
	}

	// For sparse patterns, use explicit timing
	// Simplified: just use the drum type with count
	return fmt.Sprintf("bd*<2!3 [2 4]>")
}
