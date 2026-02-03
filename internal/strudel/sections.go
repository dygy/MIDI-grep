package strudel

import (
	"fmt"
	"math"
	"strings"
)

// SectionType represents the type of musical section
type SectionType string

const (
	SectionIntro   SectionType = "intro"
	SectionVerse   SectionType = "verse"
	SectionChorus  SectionType = "chorus"
	SectionBridge  SectionType = "bridge"
	SectionOutro   SectionType = "outro"
	SectionSparse  SectionType = "sparse"
	SectionDense   SectionType = "dense"
	SectionBuildup SectionType = "buildup"
)

// FormType represents detected musical form
type FormType string

const (
	FormUnknown     FormType = "unknown"
	FormAABA        FormType = "AABA"        // Jazz standard (32-bar)
	FormABA         FormType = "ABA"         // Ternary form
	FormVerseChorus FormType = "verse-chorus"
	Form12BarBlues  FormType = "12-bar blues"
	FormAABB        FormType = "AABB"        // Binary with repeats
	FormRondo       FormType = "rondo"       // ABACA pattern
	FormThroughComp FormType = "through-composed"
)

// FormAnalysis contains the detected form and section labels
type FormAnalysis struct {
	Form       FormType
	Confidence float64
	Labels     []string // A, B, A, A or verse, chorus, verse, chorus
	NumBars    int
}

// Section represents a detected musical section
type Section struct {
	StartBeat   float64
	EndBeat     float64
	Type        SectionType
	Energy      float64 // 0.0-1.0 energy level
	NoteDensity float64 // Notes per beat
	AvgVelocity float64 // Average velocity in section
	AvgPitch    float64 // Average pitch (register)
	Description string  // Human-readable description
}

// SectionAnalysis holds analysis data for section detection
type SectionAnalysis struct {
	BeatDensities []float64 // Note density per bar
	BeatVelocities []float64 // Average velocity per bar
	BeatRegisters  []float64 // Average pitch per bar
	TotalBeats     float64
	NumBars        int
}

// DetectSections analyzes cleanup result and returns detected sections
func DetectSections(result *CleanupResult) []Section {
	if result == nil || result.TotalBeats == 0 {
		return nil
	}

	// Combine all notes for analysis
	allNotes := make([]NoteJSON, 0, len(result.Voices.Bass)+len(result.Voices.Mid)+len(result.Voices.High))
	allNotes = append(allNotes, result.Voices.Bass...)
	allNotes = append(allNotes, result.Voices.Mid...)
	allNotes = append(allNotes, result.Voices.High...)

	if len(allNotes) == 0 {
		return nil
	}

	// Analyze per-bar metrics
	analysis := analyzeByBar(allNotes, result.TotalBeats)
	if analysis.NumBars == 0 {
		return nil
	}

	// Detect section boundaries based on energy changes
	sections := detectSectionBoundaries(analysis)

	// Classify sections based on their characteristics
	classifySections(sections, analysis)

	return sections
}

// analyzeByBar computes metrics for each bar
func analyzeByBar(notes []NoteJSON, totalBeats float64) *SectionAnalysis {
	numBars := int(math.Ceil(totalBeats / 4)) // Assume 4 beats per bar
	if numBars < 1 {
		numBars = 1
	}

	analysis := &SectionAnalysis{
		BeatDensities:  make([]float64, numBars),
		BeatVelocities: make([]float64, numBars),
		BeatRegisters:  make([]float64, numBars),
		TotalBeats:     totalBeats,
		NumBars:        numBars,
	}

	// Count notes per bar and accumulate velocity/pitch
	noteCounts := make([]int, numBars)
	velocitySums := make([]float64, numBars)
	pitchSums := make([]float64, numBars)

	for _, n := range notes {
		bar := int(n.Start / 4) // Start is in beats, 4 beats per bar
		if bar >= numBars {
			bar = numBars - 1
		}
		if bar < 0 {
			bar = 0
		}

		noteCounts[bar]++
		velocitySums[bar] += n.VelocityNormalized
		pitchSums[bar] += float64(n.Pitch)
	}

	// Calculate averages
	for i := 0; i < numBars; i++ {
		if noteCounts[i] > 0 {
			analysis.BeatDensities[i] = float64(noteCounts[i]) / 4.0 // Notes per beat
			analysis.BeatVelocities[i] = velocitySums[i] / float64(noteCounts[i])
			analysis.BeatRegisters[i] = pitchSums[i] / float64(noteCounts[i])
		}
	}

	return analysis
}

// detectSectionBoundaries finds where sections change based on energy
func detectSectionBoundaries(analysis *SectionAnalysis) []Section {
	if analysis.NumBars <= 1 {
		// Single bar - one section
		return []Section{{
			StartBeat: 0,
			EndBeat:   analysis.TotalBeats,
			Energy:    calculateEnergy(analysis, 0, 0),
		}}
	}

	var sections []Section
	sectionStart := 0

	// Calculate energy for each bar
	energies := make([]float64, analysis.NumBars)
	for i := 0; i < analysis.NumBars; i++ {
		energies[i] = calculateEnergy(analysis, i, i)
	}

	// Detect significant changes (> 30% change in energy)
	threshold := 0.3
	for i := 1; i < analysis.NumBars; i++ {
		energyChange := math.Abs(energies[i] - energies[i-1])
		if energyChange > threshold || i == analysis.NumBars-1 {
			// End previous section
			endBar := i
			if i == analysis.NumBars-1 {
				endBar = analysis.NumBars
			}

			section := Section{
				StartBeat:   float64(sectionStart * 4),
				EndBeat:     float64(endBar * 4),
				Energy:      calculateEnergy(analysis, sectionStart, endBar-1),
				NoteDensity: avgDensity(analysis.BeatDensities, sectionStart, endBar-1),
				AvgVelocity: avgValue(analysis.BeatVelocities, sectionStart, endBar-1),
				AvgPitch:    avgValue(analysis.BeatRegisters, sectionStart, endBar-1),
			}

			// Only add if section has meaningful length
			if endBar > sectionStart {
				sections = append(sections, section)
			}

			sectionStart = i
		}
	}

	// Merge very short sections (< 2 bars) with adjacent
	sections = mergeShortsections(sections)

	return sections
}

// calculateEnergy computes energy level for a bar range
func calculateEnergy(analysis *SectionAnalysis, startBar, endBar int) float64 {
	if startBar > endBar {
		return 0
	}

	density := avgDensity(analysis.BeatDensities, startBar, endBar)
	velocity := avgValue(analysis.BeatVelocities, startBar, endBar)

	// Normalize density (assume max 4 notes per beat is high)
	normalizedDensity := math.Min(density/4.0, 1.0)

	// Energy is combination of density and velocity
	return (normalizedDensity*0.6 + velocity*0.4)
}

// avgDensity calculates average density over bar range
func avgDensity(densities []float64, start, end int) float64 {
	if start > end || start < 0 || end >= len(densities) {
		return 0
	}
	sum := 0.0
	for i := start; i <= end; i++ {
		sum += densities[i]
	}
	return sum / float64(end-start+1)
}

// avgValue calculates average of values over bar range
func avgValue(values []float64, start, end int) float64 {
	if start > end || start < 0 || end >= len(values) {
		return 0
	}
	sum := 0.0
	count := 0
	for i := start; i <= end; i++ {
		if values[i] > 0 {
			sum += values[i]
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// mergeShortsections combines very short sections with neighbors
func mergeShortsections(sections []Section) []Section {
	if len(sections) <= 1 {
		return sections
	}

	var merged []Section
	for i := 0; i < len(sections); i++ {
		section := sections[i]
		bars := int((section.EndBeat - section.StartBeat) / 4)

		// If section is too short and not first/last, merge with previous
		if bars < 2 && i > 0 && len(merged) > 0 {
			// Merge with previous section
			prev := &merged[len(merged)-1]
			prev.EndBeat = section.EndBeat
			prev.Energy = (prev.Energy + section.Energy) / 2
			prev.NoteDensity = (prev.NoteDensity + section.NoteDensity) / 2
			prev.AvgVelocity = (prev.AvgVelocity + section.AvgVelocity) / 2
		} else {
			merged = append(merged, section)
		}
	}

	return merged
}

// classifySections assigns types to sections based on characteristics
func classifySections(sections []Section, analysis *SectionAnalysis) {
	if len(sections) == 0 {
		return
	}

	// Find global max/min energy for relative comparison
	maxEnergy := 0.0
	minEnergy := 1.0
	for _, s := range sections {
		if s.Energy > maxEnergy {
			maxEnergy = s.Energy
		}
		if s.Energy < minEnergy {
			minEnergy = s.Energy
		}
	}

	energyRange := maxEnergy - minEnergy
	if energyRange < 0.1 {
		energyRange = 0.1 // Prevent division issues
	}

	for i := range sections {
		section := &sections[i]
		relativeEnergy := (section.Energy - minEnergy) / energyRange

		// Classify based on position and energy
		isFirst := i == 0
		isLast := i == len(sections)-1

		switch {
		case isFirst && section.Energy < 0.4:
			section.Type = SectionIntro
			section.Description = "intro"
		case isLast && section.Energy < 0.4:
			section.Type = SectionOutro
			section.Description = "outro"
		case relativeEnergy > 0.7:
			section.Type = SectionChorus
			section.Description = "high energy"
		case relativeEnergy < 0.3:
			section.Type = SectionSparse
			section.Description = "sparse"
		case i > 0 && sections[i-1].Energy < section.Energy*0.7:
			section.Type = SectionBuildup
			section.Description = "building"
		default:
			section.Type = SectionVerse
			section.Description = "main"
		}
	}
}

// GenerateSectionHeader creates a comment string with section info
func GenerateSectionHeader(sections []Section, bpm float64) string {
	if len(sections) == 0 {
		return ""
	}

	var parts []string
	for _, s := range sections {
		timestamp := beatsToTimestamp(s.StartBeat, bpm)
		parts = append(parts, fmt.Sprintf("%s %s", timestamp, s.Description))
	}

	return fmt.Sprintf("// Sections: %s\n", strings.Join(parts, " | "))
}

// beatsToTimestamp converts beat position to MM:SS format
func beatsToTimestamp(beats float64, bpm float64) string {
	if bpm <= 0 {
		bpm = 120 // Default
	}
	seconds := beats * 60.0 / bpm
	minutes := int(seconds) / 60
	secs := int(seconds) % 60
	return fmt.Sprintf("%d:%02d", minutes, secs)
}

// GetSectionEffect returns suggested effect modifications for a section
func GetSectionEffect(section Section) map[string]float64 {
	effects := make(map[string]float64)

	switch section.Type {
	case SectionIntro, SectionSparse:
		// More reverb, less filter in sparse sections
		effects["reverbMult"] = 1.5
		effects["filterMult"] = 0.8
	case SectionChorus, SectionDense:
		// Tighter reverb, open filter in dense sections
		effects["reverbMult"] = 0.7
		effects["filterMult"] = 1.2
	case SectionBuildup:
		// Building energy
		effects["reverbMult"] = 1.0
		effects["filterMult"] = 1.1
	default:
		effects["reverbMult"] = 1.0
		effects["filterMult"] = 1.0
	}

	return effects
}

// SectionFingerprint represents a section's characteristics for comparison
type SectionFingerprint struct {
	Energy      float64
	Density     float64
	Register    float64
	Duration    float64 // In bars
	BarIndex    int     // Position in song
	SectionIdx  int     // Original section index
}

// AnalyzeForm detects the musical form from sections and analysis
func AnalyzeForm(sections []Section, analysis *SectionAnalysis) *FormAnalysis {
	if len(sections) == 0 || analysis == nil {
		return &FormAnalysis{Form: FormUnknown, Confidence: 0}
	}

	// Create fingerprints for each section
	fingerprints := make([]SectionFingerprint, len(sections))
	for i, s := range sections {
		fingerprints[i] = SectionFingerprint{
			Energy:     s.Energy,
			Density:    s.NoteDensity,
			Register:   s.AvgPitch,
			Duration:   (s.EndBeat - s.StartBeat) / 4, // In bars
			BarIndex:   int(s.StartBeat / 4),
			SectionIdx: i,
		}
	}

	// Check for specific forms in order of specificity
	if form := detect12BarBlues(fingerprints, analysis); form != nil {
		return form
	}

	if form := detectAABA(fingerprints); form != nil {
		return form
	}

	if form := detectVerseChorus(fingerprints, sections); form != nil {
		return form
	}

	if form := detectABA(fingerprints); form != nil {
		return form
	}

	if form := detectAABB(fingerprints); form != nil {
		return form
	}

	// Default to through-composed if no pattern detected
	labels := make([]string, len(sections))
	for i := range labels {
		labels[i] = string(rune('A' + i))
		if i > 25 {
			labels[i] = fmt.Sprintf("X%d", i-25)
		}
	}

	return &FormAnalysis{
		Form:       FormThroughComp,
		Confidence: 0.3,
		Labels:     labels,
		NumBars:    analysis.NumBars,
	}
}

// detect12BarBlues checks for 12-bar blues structure
func detect12BarBlues(fps []SectionFingerprint, analysis *SectionAnalysis) *FormAnalysis {
	// 12-bar blues: 4 + 4 + 4 structure
	// Total should be 12 bars (or multiple of 12)
	totalBars := analysis.NumBars

	if totalBars < 12 {
		return nil
	}

	// Check if total is multiple of 12
	if totalBars%12 != 0 {
		// Allow for intro/outro (check if core is 12 bars)
		coreBars := totalBars
		if totalBars > 12 && totalBars < 24 {
			coreBars = 12 // Might have intro/outro
		} else if totalBars%12 > 4 {
			return nil
		}
		_ = coreBars
	}

	// Look for characteristic 4-bar phrases
	// In 12-bar blues: bars 1-4 (I), 5-8 (IV-I), 9-12 (V-IV-I)
	numPhrases := totalBars / 4

	if numPhrases < 3 {
		return nil
	}

	// Analyze energy pattern - blues typically has tension build in last 4 bars
	var phraseEnergies []float64
	for i := 0; i < numPhrases && i < 3; i++ {
		startBar := i * 4
		endBar := startBar + 3
		if endBar >= len(analysis.BeatDensities) {
			endBar = len(analysis.BeatDensities) - 1
		}
		phraseEnergies = append(phraseEnergies, avgDensity(analysis.BeatDensities, startBar, endBar))
	}

	// Check for blues turnaround pattern (last phrase has more movement)
	if len(phraseEnergies) >= 3 {
		// Blues typically has higher activity in the turnaround
		avgFirst := (phraseEnergies[0] + phraseEnergies[1]) / 2
		if phraseEnergies[2] >= avgFirst*0.8 { // Third phrase at least as active
			numRepeats := totalBars / 12
			labels := make([]string, 0)
			for r := 0; r < numRepeats; r++ {
				labels = append(labels, "I (bars 1-4)", "IV-I (bars 5-8)", "V-IV-I (bars 9-12)")
			}
			return &FormAnalysis{
				Form:       Form12BarBlues,
				Confidence: 0.7,
				Labels:     labels,
				NumBars:    totalBars,
			}
		}
	}

	return nil
}

// detectAABA checks for AABA form (common in jazz standards)
func detectAABA(fps []SectionFingerprint) *FormAnalysis {
	// AABA: 4 sections where sections 0, 1, 3 are similar, section 2 is different
	// Typically 8 bars each = 32 bars total

	if len(fps) < 4 {
		return nil
	}

	// Check if we have 4 roughly equal sections
	totalDuration := 0.0
	for _, fp := range fps {
		totalDuration += fp.Duration
	}
	avgDuration := totalDuration / float64(len(fps))

	// All sections should be similar duration (within 50%)
	for _, fp := range fps[:4] {
		if fp.Duration < avgDuration*0.5 || fp.Duration > avgDuration*1.5 {
			return nil
		}
	}

	// Check similarity: A sections (0, 1, 3) should be similar
	simA0A1 := sectionSimilarity(fps[0], fps[1])
	simA1A3 := sectionSimilarity(fps[1], fps[3])
	simA0B := sectionSimilarity(fps[0], fps[2])

	// A sections similar to each other, different from B
	if simA0A1 > 0.6 && simA1A3 > 0.6 && simA0B < 0.5 {
		labels := []string{"A", "A", "B", "A"}
		// Add more if there are additional sections
		for i := 4; i < len(fps); i++ {
			if sectionSimilarity(fps[i], fps[0]) > 0.6 {
				labels = append(labels, "A")
			} else if sectionSimilarity(fps[i], fps[2]) > 0.6 {
				labels = append(labels, "B")
			} else {
				labels = append(labels, "C")
			}
		}
		return &FormAnalysis{
			Form:       FormAABA,
			Confidence: (simA0A1 + simA1A3 + (1 - simA0B)) / 3,
			Labels:     labels,
			NumBars:    int(totalDuration),
		}
	}

	return nil
}

// detectVerseChorus checks for verse-chorus alternation
func detectVerseChorus(fps []SectionFingerprint, sections []Section) *FormAnalysis {
	if len(fps) < 2 {
		return nil
	}

	// Look for alternating energy pattern (verse=low, chorus=high)
	// Or find two distinct recurring section types

	// Group sections by similarity
	groups := groupSimilarSections(fps)

	if len(groups) < 2 {
		return nil
	}

	// Check for alternating pattern
	var pattern []int
	for _, fp := range fps {
		for groupIdx, group := range groups {
			for _, member := range group {
				if member == fp.SectionIdx {
					pattern = append(pattern, groupIdx)
					break
				}
			}
		}
	}

	// Check if pattern alternates (0,1,0,1 or similar)
	if isAlternating(pattern) {
		// Determine which is verse (lower energy) and which is chorus (higher energy)
		group0Energy := 0.0
		group1Energy := 0.0
		for _, idx := range groups[0] {
			group0Energy += fps[idx].Energy
		}
		for _, idx := range groups[1] {
			group1Energy += fps[idx].Energy
		}
		group0Energy /= float64(len(groups[0]))
		group1Energy /= float64(len(groups[1]))

		labels := make([]string, len(fps))
		for i, p := range pattern {
			if (p == 0 && group0Energy < group1Energy) || (p == 1 && group1Energy < group0Energy) {
				labels[i] = "verse"
			} else {
				labels[i] = "chorus"
			}
		}

		totalDuration := 0.0
		for _, fp := range fps {
			totalDuration += fp.Duration
		}

		return &FormAnalysis{
			Form:       FormVerseChorus,
			Confidence: 0.7,
			Labels:     labels,
			NumBars:    int(totalDuration),
		}
	}

	return nil
}

// detectABA checks for ternary form
func detectABA(fps []SectionFingerprint) *FormAnalysis {
	if len(fps) < 3 {
		return nil
	}

	// Check if first and last sections are similar, middle is different
	simFirstLast := sectionSimilarity(fps[0], fps[len(fps)-1])
	simFirstMiddle := 0.0
	for i := 1; i < len(fps)-1; i++ {
		simFirstMiddle = math.Max(simFirstMiddle, sectionSimilarity(fps[0], fps[i]))
	}

	if simFirstLast > 0.6 && simFirstMiddle < 0.5 {
		labels := []string{"A"}
		for i := 1; i < len(fps)-1; i++ {
			labels = append(labels, "B")
		}
		labels = append(labels, "A")

		totalDuration := 0.0
		for _, fp := range fps {
			totalDuration += fp.Duration
		}

		return &FormAnalysis{
			Form:       FormABA,
			Confidence: simFirstLast,
			Labels:     labels,
			NumBars:    int(totalDuration),
		}
	}

	return nil
}

// detectAABB checks for binary form with repeats
func detectAABB(fps []SectionFingerprint) *FormAnalysis {
	if len(fps) < 4 {
		return nil
	}

	// Check for AA followed by BB pattern
	simAA := sectionSimilarity(fps[0], fps[1])
	simBB := sectionSimilarity(fps[2], fps[3])
	simAB := sectionSimilarity(fps[0], fps[2])

	if simAA > 0.6 && simBB > 0.6 && simAB < 0.5 {
		labels := []string{"A", "A", "B", "B"}
		for i := 4; i < len(fps); i++ {
			if sectionSimilarity(fps[i], fps[0]) > 0.6 {
				labels = append(labels, "A")
			} else {
				labels = append(labels, "B")
			}
		}

		totalDuration := 0.0
		for _, fp := range fps {
			totalDuration += fp.Duration
		}

		return &FormAnalysis{
			Form:       FormAABB,
			Confidence: (simAA + simBB + (1 - simAB)) / 3,
			Labels:     labels,
			NumBars:    int(totalDuration),
		}
	}

	return nil
}

// sectionSimilarity computes similarity between two section fingerprints (0-1)
func sectionSimilarity(a, b SectionFingerprint) float64 {
	// Compare energy, density, and register
	energyDiff := math.Abs(a.Energy - b.Energy)
	densityDiff := math.Abs(a.Density-b.Density) / math.Max(a.Density, b.Density+0.001)
	registerDiff := math.Abs(a.Register-b.Register) / 12.0 // Normalize by octave

	// Weight the differences
	similarity := 1.0 - (energyDiff*0.4 + densityDiff*0.3 + registerDiff*0.3)
	return math.Max(0, similarity)
}

// groupSimilarSections clusters sections by similarity
func groupSimilarSections(fps []SectionFingerprint) [][]int {
	if len(fps) == 0 {
		return nil
	}

	threshold := 0.6
	assigned := make([]bool, len(fps))
	var groups [][]int

	for i := range fps {
		if assigned[i] {
			continue
		}

		group := []int{i}
		assigned[i] = true

		for j := i + 1; j < len(fps); j++ {
			if !assigned[j] && sectionSimilarity(fps[i], fps[j]) > threshold {
				group = append(group, j)
				assigned[j] = true
			}
		}

		groups = append(groups, group)
	}

	return groups
}

// isAlternating checks if a pattern alternates between values
func isAlternating(pattern []int) bool {
	if len(pattern) < 3 {
		return false
	}

	// Check for A-B-A-B or similar alternating pattern
	alternations := 0
	for i := 1; i < len(pattern); i++ {
		if pattern[i] != pattern[i-1] {
			alternations++
		}
	}

	// At least 50% of transitions should be alternations
	return float64(alternations)/float64(len(pattern)-1) >= 0.5
}

// GenerateFormHeader creates a comment string with form info
func GenerateFormHeader(form *FormAnalysis) string {
	if form == nil || form.Form == FormUnknown {
		return ""
	}

	confidence := int(form.Confidence * 100)
	labelStr := strings.Join(form.Labels, " ")

	return fmt.Sprintf("// Form: %s (%d%% confidence) - %s\n", form.Form, confidence, labelStr)
}
