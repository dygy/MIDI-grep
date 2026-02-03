package strudel

import (
	"fmt"
	"strings"

	"github.com/dygy/midi-grep/internal/analysis"
)

// BrazilianFunkGenerator creates Strudel code for Brazilian funk/phonk tracks
type BrazilianFunkGenerator struct {
	bpm float64
	key string
}

// NewBrazilianFunkGenerator creates a generator for Brazilian funk
func NewBrazilianFunkGenerator(bpm float64, key string) *BrazilianFunkGenerator {
	return &BrazilianFunkGenerator{bpm: bpm, key: key}
}

// Generate creates Brazilian funk Strudel code with full energy
func (g *BrazilianFunkGenerator) Generate(analysisResult *analysis.Result) string {
	var sb strings.Builder

	bpm := g.bpm
	if bpm == 0 {
		bpm = analysisResult.BPM
	}
	if bpm == 0 {
		bpm = 130 // Default for Brazilian funk
	}

	key := g.key
	if key == "" {
		key = analysisResult.Key
	}

	// Extract root note from key
	root := extractRoot(key)
	isMinor := strings.Contains(strings.ToLower(key), "minor") || strings.HasSuffix(key, "m")

	// Header
	sb.WriteString("// MIDI-grep output (Brazilian Funk mode)\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f, Key: %s\n", bpm, key))
	sb.WriteString("// Genre: Brazilian Funk / Phonk\n")
	sb.WriteString("// Style: Tamborzão + 808 + Vocal Chops\n\n")

	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", bpm))

	// Root and chord notes
	r := strings.ToLower(root)   // e.g., "c#"
	third := strings.ToLower(getThird(root, isMinor)) // e.g., "e"
	fifth := strings.ToLower(getFifth(root))          // e.g., "g#"

	// ========== DRUMS ==========
	// Original: sparse feel, ~2 hits/second, irregular timing
	sb.WriteString("// ===== DRUMS (Sparse Tamborzão) =====\n")

	// Sparser kick - matches original's loose feel
	sb.WriteString("// Kick - syncopated but sparse\n")
	sb.WriteString("let kick1 = \"bd ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ bd ~ ~ ~\"\n")
	sb.WriteString("let kick2 = \"~ ~ bd ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ bd ~\"\n")
	sb.WriteString("let kick3 = \"bd ~ ~ ~ bd ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~\"\n\n")

	// Snare on 2 and 4 only
	sb.WriteString("// Snare - simple backbeat\n")
	sb.WriteString("let snare1 = \"~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~\"\n")
	sb.WriteString("let snare2 = \"~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~\"\n")
	sb.WriteString("let snare3 = \"~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~\"\n\n")

	// Sparse hi-hats - not constant 16ths
	sb.WriteString("// Hi-hats - sparse, accent on offbeats\n")
	sb.WriteString("let hh1 = \"~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~\"\n")
	sb.WriteString("let hh2 = \"~ ~ hh ~ ~ ~ oh ~ ~ ~ hh ~ ~ ~ oh ~\"\n")
	sb.WriteString("let hh3 = \"hh ~ ~ ~ hh ~ ~ ~ oh ~ ~ ~ hh ~ ~ ~\"\n\n")

	// ========== 808 BASS ==========
	// Sparse bass following the sparse kick
	sb.WriteString("// ===== 808 BASS (sparse, follows kick) =====\n")
	sb.WriteString(fmt.Sprintf("let bass1 = \"%s1 ~ ~ ~ ~ ~ %s1 ~ ~ ~ ~ ~ %s1 ~ ~ ~\"\n", r, r, fifth))
	sb.WriteString(fmt.Sprintf("let bass2 = \"~ ~ %s1 ~ ~ ~ ~ ~ %s1 ~ ~ ~ ~ ~ %s1 ~\"\n", r, fifth, r))
	sb.WriteString(fmt.Sprintf("let bass3 = \"%s1 ~ ~ ~ %s1 ~ ~ ~ ~ ~ %s1 ~ ~ ~ ~ ~\"\n\n", r, fifth, r))

	// ========== VOCAL CHOPS ==========
	// Original analysis: sparse (2.1 notes/sec), irregular timing, high register (C5, G#4, D#5)
	sb.WriteString("// ===== VOCAL CHOPS (sparse, high register like original) =====\n")
	sb.WriteString(fmt.Sprintf("let vox1 = \"%s5 ~ ~ ~ %s4 ~ ~ ~ ~ ~ %s5 ~ ~ ~ ~ ~\"\n", r, fifth, third))
	sb.WriteString(fmt.Sprintf("let vox2 = \"~ ~ %s5 ~ ~ ~ ~ ~ %s4 ~ ~ ~ %s5 ~ ~ ~\"\n", fifth, r, r))
	sb.WriteString(fmt.Sprintf("let vox3 = \"~ %s5 ~ ~ ~ ~ %s4 ~ ~ ~ ~ ~ ~ %s5 ~ ~\"\n", r, fifth, third))
	sb.WriteString(fmt.Sprintf("let vox4 = \"%s5 ~ ~ %s4 ~ ~ ~ ~ ~ ~ %s5 ~ ~ ~ %s4 ~\"\n\n", r, fifth, r, third))

	// ========== STABS ==========
	sb.WriteString("// ===== CHORD STABS =====\n")
	chordNotes := fmt.Sprintf("%s4,%s4,%s4", r, third, fifth)
	chord7 := fmt.Sprintf("%s4,%s4,%s4,%s4", r, third, fifth, strings.ToLower(getSeventh(root)))

	sb.WriteString(fmt.Sprintf("let stab1 = \"[%s] ~ ~ ~ [%s] ~ ~ ~ [%s] ~ ~ ~ [%s] ~ ~ ~\"\n", chordNotes, chordNotes, chordNotes, chordNotes))
	sb.WriteString(fmt.Sprintf("let stab2 = \"~ ~ [%s] ~ ~ ~ [%s] ~ ~ ~ [%s] ~ ~ ~ [%s] ~\"\n", chordNotes, chordNotes, chordNotes, chordNotes))
	sb.WriteString(fmt.Sprintf("let stab3 = \"[%s] ~ [%s] ~ [%s] ~ [%s] ~ ~ ~ ~ ~ ~ ~ ~ ~\"\n\n", chordNotes, chord7, chordNotes, chord7))

	// ========== LEAD ==========
	sb.WriteString("// ===== LEAD (high register) =====\n")
	sb.WriteString(fmt.Sprintf("let lead1 = \"%s5 ~ %s5 ~ %s5 ~ %s5 %s5 ~ ~ %s5 ~ %s5 ~ ~ ~\"\n",
		r, third, fifth, r, third, fifth, r))
	sb.WriteString(fmt.Sprintf("let lead2 = \"~ %s5 ~ %s5 ~ %s5 ~ ~ %s5 %s5 ~ %s5 ~ ~ %s5 ~\"\n",
		fifth, r, third, r, fifth, r, third))
	sb.WriteString(fmt.Sprintf("let lead3 = \"%s5 %s5 %s5 ~ %s5 %s5 %s5 ~ ~ ~ ~ ~ %s5 %s5 %s5 %s5\"\n\n",
		r, third, fifth, fifth, r, third, r, third, fifth, r))

	// ========== EFFECTS ==========
	// Tuned for better frequency balance (based on audio analysis)
	sb.WriteString("// ===== EFFECTS =====\n")

	// Drums - balanced for mix
	sb.WriteString("let kickFx = p => p.bank(\"RolandTR808\").gain(0.9).lpf(120).distort(0.4)\n")
	sb.WriteString("let snareFx = p => p.bank(\"RolandTR808\").gain(0.85).room(0.08).hpf(200)\n")
	sb.WriteString("let hhFx = p => p.bank(\"RolandTR808\").gain(0.9).hpf(6000)\n\n")

	// 808 bass - reduced to balance with mids
	sb.WriteString("let bassFx = p => p.sound(\"sawtooth\")\n")
	sb.WriteString("  .lpf(100).gain(0.6).distort(0.4)\n")
	sb.WriteString("  .attack(0.001).decay(0.2).sustain(0.4).release(0.15)\n")
	sb.WriteString("  .slide(0.1)\n\n")

	// Vocal chop - high register, spacious
	sb.WriteString("let voxFx = p => p.sound(\"gm_lead_2_sawtooth\")\n")
	sb.WriteString("  .lpf(4000).gain(0.9).room(0.3)\n")
	sb.WriteString("  .attack(0.01).decay(0.1).sustain(0.6).release(0.15)\n")
	sb.WriteString("  .delay(0.2).delaytime(0.375).delayfeedback(0.3)\n\n")

	// Stab - increased for mid presence
	sb.WriteString("let stabFx = p => p.sound(\"sawtooth\")\n")
	sb.WriteString("  .lpf(4000).gain(0.55).distort(0.2).room(0.15)\n")
	sb.WriteString("  .attack(0.003).decay(0.08).sustain(0.3).release(0.06)\n\n")

	// Lead - boosted highs
	sb.WriteString("let leadFx = p => p.sound(\"square\")\n")
	sb.WriteString("  .lpf(6000).gain(0.45).room(0.2)\n")
	sb.WriteString("  .attack(0.008).decay(0.1).sustain(0.4).release(0.1)\n")
	sb.WriteString("  .delay(0.25).delaytime(0.375).delayfeedback(0.35)\n\n")

	// ========== MAIN PATTERN ==========
	sb.WriteString("// ===== PLAY (Full Energy) =====\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  // Drums\n")
	sb.WriteString("  kickFx(s(cat(kick1, kick2, kick1, kick3))),\n")
	sb.WriteString("  snareFx(s(cat(snare1, snare2, snare1, snare3))),\n")
	sb.WriteString("  hhFx(s(cat(hh1, hh2, hh1, hh3))),\n")
	sb.WriteString("  // Bass\n")
	sb.WriteString("  bassFx(note(cat(bass1, bass2, bass1, bass3))),\n")
	sb.WriteString("  // Melodic\n")
	sb.WriteString("  voxFx(note(cat(vox1, vox2, vox3, vox4))),\n")
	sb.WriteString("  stabFx(note(cat(stab1, stab2, stab1, stab3))),\n")
	sb.WriteString("  leadFx(note(cat(lead1, lead2, lead1, lead3)))\n")
	sb.WriteString(")\n\n")

	// ========== VARIATIONS ==========
	sb.WriteString("// ===== VARIATIONS (delete /* */ to activate) =====\n\n")

	sb.WriteString("/* Intro (just drums + bass)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  kickFx(s(kick1)),\n")
	sb.WriteString("  snareFx(s(snare1)),\n")
	sb.WriteString("  hhFx(s(hh1)),\n")
	sb.WriteString("  bassFx(note(bass1))\n")
	sb.WriteString(")\n*/\n\n")

	sb.WriteString("/* Build (add vox)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  kickFx(s(cat(kick1, kick2))),\n")
	sb.WriteString("  snareFx(s(cat(snare1, snare2))),\n")
	sb.WriteString("  hhFx(s(cat(hh1, hh2))),\n")
	sb.WriteString("  bassFx(note(cat(bass1, bass2))),\n")
	sb.WriteString("  voxFx(note(cat(vox2, vox3)))\n")
	sb.WriteString(")\n*/\n\n")

	sb.WriteString("/* Drop (full energy with filter sweep)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  kickFx(s(cat(kick1, kick2, kick3, kick1))),\n")
	sb.WriteString("  snareFx(s(cat(snare1, snare2, snare3, snare1))),\n")
	sb.WriteString("  hhFx(s(cat(hh2, hh3, hh2, hh1))),\n")
	sb.WriteString("  bassFx(note(cat(bass1, bass2, bass3, bass1))),\n")
	sb.WriteString("  voxFx(note(cat(vox1, vox2, vox3, vox4))).lpf(sine.range(1500, 4000).slow(4)),\n")
	sb.WriteString("  stabFx(note(cat(stab1, stab2, stab3, stab1))),\n")
	sb.WriteString("  leadFx(note(cat(lead1, lead2, lead3, lead1)))\n")
	sb.WriteString(")\n*/\n\n")

	sb.WriteString("/* Breakdown (atmospheric, half-time feel)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString(fmt.Sprintf("  note(\"%s2 ~ ~ ~ ~ ~ ~ ~ %s2 ~ ~ ~ ~ ~ ~ ~\").sound(\"sine\").lpf(200).gain(1.0).room(0.4).attack(0.05).release(0.3),\n", r, fifth))
	sb.WriteString(fmt.Sprintf("  note(\"[%s] ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\").sound(\"triangle\").lpf(1500).gain(0.25).room(0.5).attack(0.1)\n", chordNotes))
	sb.WriteString(")\n*/\n\n")

	sb.WriteString("/* Bass solo (heavy 808)\n")
	sb.WriteString(fmt.Sprintf("$: bassFx(note(\"%s1 %s1 ~ %s1 ~ %s1 %s1 ~ %s1 ~ ~ %s1 ~ %s1 %s1 %s1\")).lpf(sine.range(60, 150).slow(2))\n", r, r, r, fifth, r, r, r, r, fifth, r))
	sb.WriteString("*/\n\n")

	sb.WriteString("/* Vox solo (with stutter)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  kickFx(s(kick1)),\n")
	sb.WriteString("  voxFx(note(vox1).ply(2).gain(0.5)).lpf(sine.range(2000, 5000).slow(2))\n")
	sb.WriteString(")\n*/\n\n")

	sb.WriteString("/* Climax (everything + distortion)\n")
	sb.WriteString("$: stack(\n")
	sb.WriteString("  kickFx(s(cat(kick1, kick2, kick3, kick1)).ply(\"1 1 2 1\")),\n")
	sb.WriteString("  snareFx(s(cat(snare2, snare3, snare2, snare3))),\n")
	sb.WriteString("  hhFx(s(hh3)),\n")
	sb.WriteString("  bassFx(note(cat(bass1, bass2, bass3, bass1))).distort(0.8),\n")
	sb.WriteString("  voxFx(note(vox1)).crush(8),\n")
	sb.WriteString("  stabFx(note(stab3)).distort(0.4),\n")
	sb.WriteString("  leadFx(note(lead3)).delay(0.4)\n")
	sb.WriteString(")\n*/\n")

	return sb.String()
}

// extractRoot gets the root note from a key string like "C# minor"
func extractRoot(key string) string {
	key = strings.TrimSpace(key)
	if len(key) == 0 {
		return "C"
	}

	// Handle flats and sharps
	if len(key) >= 2 && (key[1] == '#' || key[1] == 'b') {
		return key[:2]
	}
	return string(key[0])
}

// getFifth returns the fifth note above the root
func getFifth(root string) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		return "G" // Default
	}
	fifthIdx := (rootIdx + 7) % 12
	return notes[fifthIdx]
}

// getThird returns the third note (major or minor)
func getThird(root string, isMinor bool) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		return "E"
	}
	var thirdIdx int
	if isMinor {
		thirdIdx = (rootIdx + 3) % 12
	} else {
		thirdIdx = (rootIdx + 4) % 12
	}
	return notes[thirdIdx]
}

// getSeventh returns the minor 7th note
func getSeventh(root string) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		return "Bb"
	}
	seventhIdx := (rootIdx + 10) % 12
	return notes[seventhIdx]
}

// getMinorScale returns the natural minor scale notes
func getMinorScale(root string) []string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		rootIdx = 0
	}
	// Natural minor intervals: 0, 2, 3, 5, 7, 8, 10
	intervals := []int{0, 2, 3, 5, 7, 8, 10}
	scale := make([]string, len(intervals))
	for i, interval := range intervals {
		scale[i] = strings.ToLower(notes[(rootIdx+interval)%12])
	}
	return scale
}

// getMajorScale returns the major scale notes
func getMajorScale(root string) []string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		rootIdx = 0
	}
	// Major intervals: 0, 2, 4, 5, 7, 9, 11
	intervals := []int{0, 2, 4, 5, 7, 9, 11}
	scale := make([]string, len(intervals))
	for i, interval := range intervals {
		scale[i] = strings.ToLower(notes[(rootIdx+interval)%12])
	}
	return scale
}

// findNoteIndex finds the index of a note in the chromatic scale
func findNoteIndex(notes []string, note string) int {
	for i, n := range notes {
		if strings.EqualFold(n, note) {
			return i
		}
	}
	return -1
}

// getChordNotes returns chord notes as a comma-separated string
func getChordNotes(root string, isMinor bool) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		rootIdx = 0
	}

	// Build chord in octave 4
	rootNote := strings.ToLower(notes[rootIdx]) + "4"

	var thirdIdx int
	if isMinor {
		thirdIdx = (rootIdx + 3) % 12 // Minor third
	} else {
		thirdIdx = (rootIdx + 4) % 12 // Major third
	}
	thirdNote := strings.ToLower(notes[thirdIdx]) + "4"

	fifthIdx := (rootIdx + 7) % 12
	fifthNote := strings.ToLower(notes[fifthIdx]) + "4"

	return fmt.Sprintf("%s,%s,%s", rootNote, thirdNote, fifthNote)
}

// getChord7Notes returns 7th chord notes
func getChord7Notes(root string, isMinor bool) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	rootIdx := findNoteIndex(notes, root)
	if rootIdx == -1 {
		rootIdx = 0
	}

	rootNote := strings.ToLower(notes[rootIdx]) + "4"

	var thirdIdx int
	if isMinor {
		thirdIdx = (rootIdx + 3) % 12
	} else {
		thirdIdx = (rootIdx + 4) % 12
	}
	thirdNote := strings.ToLower(notes[thirdIdx]) + "4"

	fifthIdx := (rootIdx + 7) % 12
	fifthNote := strings.ToLower(notes[fifthIdx]) + "4"

	// Minor 7th
	seventhIdx := (rootIdx + 10) % 12
	seventhNote := strings.ToLower(notes[seventhIdx]) + "4"

	return fmt.Sprintf("%s,%s,%s,%s", rootNote, thirdNote, fifthNote, seventhNote)
}
