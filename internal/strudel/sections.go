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
