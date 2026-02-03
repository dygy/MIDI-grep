package strudel

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/arkadiishvartcman/midi-grep/internal/midi"
)

// ChordType represents the quality of a chord
type ChordType string

const (
	ChordMajor     ChordType = ""      // C (no suffix for major triad)
	ChordMinor     ChordType = "m"     // Cm
	ChordDim       ChordType = "dim"   // Cdim
	ChordAug       ChordType = "aug"   // Caug
	ChordMaj7      ChordType = "^7"    // C^7 (major 7th)
	ChordMin7      ChordType = "m7"    // Cm7
	ChordDom7      ChordType = "7"     // C7 (dominant 7th)
	ChordMin7b5    ChordType = "m7b5"  // Cm7b5 (half-diminished)
	ChordDim7      ChordType = "dim7"  // Cdim7
	ChordMaj9      ChordType = "^9"    // C^9
	ChordMin9      ChordType = "m9"    // Cm9
	ChordDom9      ChordType = "9"     // C9
	ChordDom7b9    ChordType = "7b9"   // C7b9
	ChordDom7Sharp9 ChordType = "7#9"  // C7#9
	ChordAdd9      ChordType = "add9"  // Cadd9
	ChordSus2      ChordType = "sus2"  // Csus2
	ChordSus4      ChordType = "sus4"  // Csus4
	ChordUnknown   ChordType = "?"     // Unknown chord
)

// Chord represents a detected chord
type Chord struct {
	Root      string    // Root note name (C, D, E, etc.)
	Type      ChordType // Chord quality
	Bass      string    // Bass note if different from root (for slash chords)
	StartBeat float64   // Start time in beats
	Duration  float64   // Duration in beats
	Notes     []int     // MIDI pitches in the chord
}

// Name returns the full chord name (e.g., "Cm7", "G7b9")
func (c *Chord) Name() string {
	name := c.Root + string(c.Type)
	if c.Bass != "" && c.Bass != c.Root {
		name += "/" + c.Bass
	}
	return name
}

// ChordProgression represents a sequence of chords
type ChordProgression struct {
	Chords      []Chord
	Key         string  // Detected key
	TotalBeats  float64 // Total duration in beats
	BeatsPerBar int     // Time signature (usually 4)
}

// ChordDetector analyzes notes to find chord progressions
type ChordDetector struct {
	quantize    int     // Quantization grid
	minDuration float64 // Minimum chord duration in beats
}

// NewChordDetector creates a new chord detector
func NewChordDetector(quantize int) *ChordDetector {
	return &ChordDetector{
		quantize:    quantize,
		minDuration: 0.5, // Minimum half beat for a chord
	}
}

// DetectChords analyzes notes and returns chord progression
func (d *ChordDetector) DetectChords(notes []midi.Note, bpm float64) *ChordProgression {
	if len(notes) == 0 {
		return nil
	}

	beatDuration := 60.0 / bpm
	gridSize := beatDuration / float64(d.quantize/4)

	// Group notes by time slots
	notesBySlot := make(map[int][]midi.Note)
	maxSlot := 0

	for _, n := range notes {
		slot := int(n.Start / gridSize)
		notesBySlot[slot] = append(notesBySlot[slot], n)
		if slot > maxSlot {
			maxSlot = slot
		}
	}

	// Find chord changes by detecting when the set of pitches changes significantly
	var chords []Chord
	var currentPitches []int
	var currentSlot int

	for slot := 0; slot <= maxSlot; slot++ {
		slotNotes := notesBySlot[slot]
		if len(slotNotes) == 0 {
			continue
		}

		// Get pitches at this slot (reduce to pitch classes for comparison)
		pitches := make([]int, 0, len(slotNotes))
		for _, n := range slotNotes {
			pitches = append(pitches, n.Pitch)
		}
		sort.Ints(pitches)

		// Check if this is a chord change
		if !samePitchClasses(currentPitches, pitches) {
			// Save previous chord if it exists
			if len(currentPitches) >= 2 {
				startBeat := float64(currentSlot) * gridSize / beatDuration
				endBeat := float64(slot) * gridSize / beatDuration
				duration := endBeat - startBeat

				if duration >= d.minDuration {
					chord := d.identifyChord(currentPitches)
					chord.StartBeat = startBeat
					chord.Duration = duration
					chords = append(chords, chord)
				}
			}

			currentPitches = pitches
			currentSlot = slot
		}
	}

	// Don't forget the last chord
	if len(currentPitches) >= 2 {
		startBeat := float64(currentSlot) * gridSize / beatDuration
		endBeat := float64(maxSlot+1) * gridSize / beatDuration
		duration := endBeat - startBeat

		if duration >= d.minDuration {
			chord := d.identifyChord(currentPitches)
			chord.StartBeat = startBeat
			chord.Duration = duration
			chords = append(chords, chord)
		}
	}

	// Merge consecutive identical chords
	chords = mergeConsecutiveChords(chords)

	totalBeats := float64(maxSlot+1) * gridSize / beatDuration

	return &ChordProgression{
		Chords:      chords,
		TotalBeats:  totalBeats,
		BeatsPerBar: 4,
	}
}

// identifyChord analyzes pitches to determine chord type
func (d *ChordDetector) identifyChord(pitches []int) Chord {
	if len(pitches) < 2 {
		return Chord{Root: "C", Type: ChordUnknown, Notes: pitches}
	}

	// Get pitch classes (0-11) and find intervals
	pitchClasses := make([]int, len(pitches))
	for i, p := range pitches {
		pitchClasses[i] = p % 12
	}

	// Remove duplicates and sort
	unique := make(map[int]bool)
	for _, pc := range pitchClasses {
		unique[pc] = true
	}
	classes := make([]int, 0, len(unique))
	for pc := range unique {
		classes = append(classes, pc)
	}
	sort.Ints(classes)

	// Try each pitch class as potential root
	bestRoot := 0
	bestType := ChordUnknown
	bestScore := 0

	for _, root := range classes {
		intervals := getIntervalsFromRoot(classes, root)
		chordType, score := matchChordType(intervals)
		if score > bestScore {
			bestScore = score
			bestRoot = root
			bestType = chordType
		}
	}

	// Get bass note (lowest pitch)
	bass := pitches[0] % 12

	rootName := pitchClassName(bestRoot)
	bassName := pitchClassName(bass)

	return Chord{
		Root:  rootName,
		Type:  bestType,
		Bass:  bassName,
		Notes: pitches,
	}
}

// getIntervalsFromRoot calculates semitone intervals from a root
func getIntervalsFromRoot(pitchClasses []int, root int) []int {
	intervals := make([]int, len(pitchClasses))
	for i, pc := range pitchClasses {
		interval := (pc - root + 12) % 12
		intervals[i] = interval
	}
	sort.Ints(intervals)
	return intervals
}

// matchChordType matches intervals to known chord types
func matchChordType(intervals []int) (ChordType, int) {
	// Convert to set for easier matching
	hasInterval := make(map[int]bool)
	for _, i := range intervals {
		hasInterval[i] = true
	}

	// Check for each chord type (ordered by specificity)
	// Score indicates how well it matches

	// 7th chords with extensions
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[11] && hasInterval[2] {
		return ChordMaj9, 5 // Major 9
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[7] && hasInterval[10] && hasInterval[2] {
		return ChordMin9, 5 // Minor 9
	}
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[10] && hasInterval[2] {
		return ChordDom9, 5 // Dominant 9
	}
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[10] && hasInterval[1] {
		return ChordDom7b9, 5 // Dominant 7b9
	}
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[10] && hasInterval[3] {
		return ChordDom7Sharp9, 5 // Dominant 7#9
	}

	// 7th chords
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[11] {
		return ChordMaj7, 4 // Major 7
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[7] && hasInterval[10] {
		return ChordMin7, 4 // Minor 7
	}
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[10] {
		return ChordDom7, 4 // Dominant 7
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[6] && hasInterval[10] {
		return ChordMin7b5, 4 // Half-diminished
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[6] && hasInterval[9] {
		return ChordDim7, 4 // Diminished 7
	}

	// Triads
	if hasInterval[0] && hasInterval[4] && hasInterval[7] {
		return ChordMajor, 3 // Major
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[7] {
		return ChordMinor, 3 // Minor
	}
	if hasInterval[0] && hasInterval[3] && hasInterval[6] {
		return ChordDim, 3 // Diminished
	}
	if hasInterval[0] && hasInterval[4] && hasInterval[8] {
		return ChordAug, 3 // Augmented
	}

	// Sus chords
	if hasInterval[0] && hasInterval[2] && hasInterval[7] {
		return ChordSus2, 3 // Sus2
	}
	if hasInterval[0] && hasInterval[5] && hasInterval[7] {
		return ChordSus4, 3 // Sus4
	}

	// Add9
	if hasInterval[0] && hasInterval[4] && hasInterval[7] && hasInterval[2] {
		return ChordAdd9, 3
	}

	// Default based on intervals present
	if hasInterval[3] {
		return ChordMinor, 1
	}
	if hasInterval[4] {
		return ChordMajor, 1
	}

	return ChordUnknown, 0
}

// pitchClassName returns the note name for a pitch class (0-11)
func pitchClassName(pc int) string {
	names := []string{"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"}
	return names[pc%12]
}

// samePitchClasses checks if two pitch sets have the same pitch classes
func samePitchClasses(a, b []int) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	if len(a) == 0 || len(b) == 0 {
		return false
	}

	// Convert to pitch class sets
	setA := make(map[int]bool)
	setB := make(map[int]bool)
	for _, p := range a {
		setA[p%12] = true
	}
	for _, p := range b {
		setB[p%12] = true
	}

	if len(setA) != len(setB) {
		return false
	}
	for k := range setA {
		if !setB[k] {
			return false
		}
	}
	return true
}

// mergeConsecutiveChords combines adjacent identical chords
func mergeConsecutiveChords(chords []Chord) []Chord {
	if len(chords) <= 1 {
		return chords
	}

	var merged []Chord
	current := chords[0]

	for i := 1; i < len(chords); i++ {
		if chords[i].Root == current.Root && chords[i].Type == current.Type {
			// Extend current chord
			current.Duration = chords[i].StartBeat + chords[i].Duration - current.StartBeat
		} else {
			merged = append(merged, current)
			current = chords[i]
		}
	}
	merged = append(merged, current)

	return merged
}

// ToStrudelArrange converts chord progression to Strudel arrange() format
func (p *ChordProgression) ToStrudelArrange() string {
	if p == nil || len(p.Chords) == 0 {
		return ""
	}

	// Group chords by section (4 or 8 bars)
	sectionLength := float64(p.BeatsPerBar * 4) // 4 bars per section
	sections := make(map[int][]Chord)

	for _, c := range p.Chords {
		section := int(c.StartBeat / sectionLength)
		sections[section] = append(sections[section], c)
	}

	// Build arrange() structure
	var sb strings.Builder
	sb.WriteString("let chords = arrange(\n")

	// Get sorted section keys
	sectionKeys := make([]int, 0, len(sections))
	for k := range sections {
		sectionKeys = append(sectionKeys, k)
	}
	sort.Ints(sectionKeys)

	for i, secIdx := range sectionKeys {
		sectionChords := sections[secIdx]
		beats := int(sectionLength)

		// Build chord pattern for this section
		pattern := buildChordPattern(sectionChords, p.BeatsPerBar)

		sb.WriteString(fmt.Sprintf("  [%d, \"%s\"]", beats, pattern))
		if i < len(sectionKeys)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}

	sb.WriteString(").chord()")
	return sb.String()
}

// buildChordPattern creates a mini-notation pattern from chords
func buildChordPattern(chords []Chord, beatsPerBar int) string {
	if len(chords) == 0 {
		return "~"
	}

	// Simple case: one chord for the section
	if len(chords) == 1 {
		return chords[0].Name()
	}

	// Multiple chords: build pattern with timing
	var parts []string
	for _, c := range chords {
		// Determine repetition based on duration
		bars := int(math.Round(c.Duration / float64(beatsPerBar)))
		if bars <= 0 {
			bars = 1
		}

		name := c.Name()
		if bars > 1 {
			parts = append(parts, fmt.Sprintf("%s!%d", name, bars))
		} else {
			parts = append(parts, name)
		}
	}

	// Wrap in angle brackets for sequence
	return "<" + strings.Join(parts, " ") + ">"
}

// SimplifyProgression reduces chord complexity for cleaner output
func (p *ChordProgression) SimplifyProgression() {
	for i := range p.Chords {
		// Simplify complex extensions to basic 7th chords
		switch p.Chords[i].Type {
		case ChordMaj9:
			p.Chords[i].Type = ChordMaj7
		case ChordMin9:
			p.Chords[i].Type = ChordMin7
		case ChordDom9:
			p.Chords[i].Type = ChordDom7
		case ChordDom7Sharp9:
			p.Chords[i].Type = ChordDom7b9
		}
	}
}

// DetectKey attempts to detect the key from the chord progression
func (p *ChordProgression) DetectKey() string {
	if len(p.Chords) == 0 {
		return "C"
	}

	// Count root notes weighted by duration
	rootCounts := make(map[string]float64)
	for _, c := range p.Chords {
		rootCounts[c.Root] += c.Duration
	}

	// Find most common root
	var maxRoot string
	var maxCount float64
	for root, count := range rootCounts {
		if count > maxCount {
			maxCount = count
			maxRoot = root
		}
	}

	// Determine major/minor based on chord types
	minorCount := 0
	majorCount := 0
	for _, c := range p.Chords {
		if c.Root == maxRoot {
			if c.Type == ChordMinor || c.Type == ChordMin7 || c.Type == ChordMin9 || c.Type == ChordMin7b5 {
				minorCount++
			} else {
				majorCount++
			}
		}
	}

	if minorCount > majorCount {
		return maxRoot + "m"
	}
	return maxRoot
}
