package strudel

import (
	"fmt"
	"math"
	"strings"

	"github.com/dygy/midi-grep/internal/drums"
)

// DrumKit defines preset drum sound mappings
type DrumKit string

const (
	DrumKitTR808    DrumKit = "tr808"
	DrumKitTR909    DrumKit = "tr909"
	DrumKitLinn     DrumKit = "linn"
	DrumKitAcoustic DrumKit = "acoustic"
	DrumKitLofi     DrumKit = "lofi"
	DrumKitDefault  DrumKit = "default"
)

// DrumKitInfo contains the bank name and sound mappings for a drum kit
type DrumKitInfo struct {
	Bank   string            // Bank name for .bank() method
	Sounds map[string]string // Sound mappings (bd, sd, hh, oh, cp)
}

// DrumKitSoundsInfo maps drum kits to their bank and sound info
var DrumKitSoundsInfo = map[DrumKit]DrumKitInfo{
	DrumKitTR808: {
		Bank: "RolandTR808",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
	DrumKitTR909: {
		Bank: "RolandTR909",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
	DrumKitLinn: {
		Bank: "LinnDrum",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
	DrumKitAcoustic: {
		Bank: "AlesisSR16",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
	DrumKitLofi: {
		Bank: "CasioRZ1",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
	DrumKitDefault: {
		Bank: "",
		Sounds: map[string]string{
			"bd": "bd",
			"sd": "sd",
			"hh": "hh",
			"oh": "oh",
			"cp": "cp",
		},
	},
}

// DrumKitSounds maps drum kits to their sample names (legacy, for compatibility)
var DrumKitSounds = map[DrumKit]map[string]string{
	DrumKitTR808: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
	DrumKitTR909: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
	DrumKitLinn: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
	DrumKitAcoustic: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
	DrumKitLofi: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
	DrumKitDefault: {
		"bd": "bd",
		"sd": "sd",
		"hh": "hh",
		"oh": "oh",
		"cp": "cp",
	},
}

// AvailableDrumKits returns list of available drum kits
func AvailableDrumKits() []DrumKit {
	return []DrumKit{
		DrumKitTR808,
		DrumKitTR909,
		DrumKitLinn,
		DrumKitAcoustic,
		DrumKitLofi,
	}
}

// DrumKitDescription returns a description for each drum kit
func DrumKitDescription(kit DrumKit) string {
	descriptions := map[DrumKit]string{
		DrumKitTR808:    "Roland TR-808 drum machine - classic hip-hop/electronic",
		DrumKitTR909:    "Roland TR-909 drum machine - house/techno",
		DrumKitLinn:     "LinnDrum - 80s pop/R&B",
		DrumKitAcoustic: "Acoustic drum kit samples",
		DrumKitLofi:     "Lo-fi/vintage drum machine",
	}
	return descriptions[kit]
}

// ParseDrumKit converts a string to DrumKit
func ParseDrumKit(s string) DrumKit {
	switch strings.ToLower(s) {
	case "tr808", "808":
		return DrumKitTR808
	case "tr909", "909":
		return DrumKitTR909
	case "linn", "linndrum":
		return DrumKitLinn
	case "acoustic":
		return DrumKitAcoustic
	case "lofi", "lo-fi":
		return DrumKitLofi
	default:
		return DrumKitTR808
	}
}

// GenerateDrumPattern creates Strudel drum patterns from detection results
// Uses modern Strudel idioms like .bank() for cleaner, more musical output
func (g *Generator) GenerateDrumPattern(result *drums.DetectionResult, kit DrumKit) string {
	if result == nil || len(result.Hits) == 0 {
		return ""
	}

	kitInfo := DrumKitSoundsInfo[kit]
	if kitInfo.Sounds == nil {
		kitInfo = DrumKitSoundsInfo[DrumKitDefault]
	}

	// Get BPM and calculate timing
	bpm := result.Tempo
	if bpm <= 0 {
		bpm = 120
	}

	beatDuration := 60.0 / bpm
	gridSize := beatDuration / float64(g.quantize/4)

	// Find total duration and number of bars
	maxTime := 0.0
	for _, hit := range result.Hits {
		if hit.Time > maxTime {
			maxTime = hit.Time
		}
	}

	numBars := int(math.Ceil(maxTime / (beatDuration * 4)))
	if numBars < 1 {
		numBars = 1
	}
	if numBars > 16 {
		numBars = 16
	}

	// Group hits by type
	hitsByType := result.HitsByType()

	// Generate pattern for each drum type with bank-style output
	var patterns []string

	// Order: bd, sd, hh/oh (combined), cp
	// Combine hh and oh into one pattern for more musical results

	// Kick (bd)
	if bdHits, ok := hitsByType["bd"]; ok && len(bdHits) > 0 {
		pattern := g.buildDrumTypePatternWithBank(bdHits, gridSize, numBars, "bd", kitInfo.Bank)
		if pattern != "" {
			patterns = append(patterns, pattern)
		}
	}

	// Snare/Clap (sd or cp) - snare takes priority
	if sdHits, ok := hitsByType["sd"]; ok && len(sdHits) > 0 {
		pattern := g.buildDrumTypePatternWithBank(sdHits, gridSize, numBars, "sd", kitInfo.Bank)
		if pattern != "" {
			patterns = append(patterns, pattern)
		}
	} else if cpHits, ok := hitsByType["cp"]; ok && len(cpHits) > 0 {
		pattern := g.buildDrumTypePatternWithBank(cpHits, gridSize, numBars, "cp", kitInfo.Bank)
		if pattern != "" {
			patterns = append(patterns, pattern)
		}
	}

	// Hi-hats (combine hh and oh for more musical results)
	hhHits := hitsByType["hh"]
	ohHits := hitsByType["oh"]
	if len(hhHits) > 0 || len(ohHits) > 0 {
		pattern := g.buildHiHatPattern(hhHits, ohHits, gridSize, numBars, kitInfo.Bank)
		if pattern != "" {
			patterns = append(patterns, pattern)
		}
	}

	if len(patterns) == 0 {
		return ""
	}

	// Build output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("  // drums (%s)\n", kit))

	if len(patterns) == 1 {
		sb.WriteString(fmt.Sprintf("  %s", patterns[0]))
	} else {
		// Stack multiple drum patterns
		sb.WriteString("  stack(\n")
		for i, p := range patterns {
			sb.WriteString(fmt.Sprintf("    %s", p))
			if i < len(patterns)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n")
		}
		sb.WriteString("  ).room(0.2)")
	}

	return sb.String()
}

// buildDrumTypePattern creates a Strudel s() pattern for one drum type (legacy)
func (g *Generator) buildDrumTypePattern(hits []drums.Hit, gridSize float64, numBars int, soundName string) string {
	if len(hits) == 0 {
		return ""
	}

	// Grid slots per bar
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays
	slots := make([]bool, totalSlots)
	velocities := make([]float64, totalSlots)

	// Place hits in slots
	for _, hit := range hits {
		slot := int(hit.Time / gridSize)
		if slot >= 0 && slot < totalSlots {
			slots[slot] = true
			velocities[slot] = hit.VelocityNormalized
		}
	}

	// Build pattern with bar structure
	var bars []string
	hasContent := false

	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasHits := false

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if slots[i] {
				barParts = append(barParts, soundName)
				barHasHits = true
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasHits {
			hasContent = true
			simplified := simplifyDrumPattern(barParts)
			bars = append(bars, simplified)
		}
	}

	if !hasContent || len(bars) == 0 {
		return ""
	}

	pattern := strings.Join(bars, " | ")
	return fmt.Sprintf("s(\"%s\")", pattern)
}

// buildDrumTypePatternWithBank creates a Strudel s() pattern with .bank() for cleaner output
func (g *Generator) buildDrumTypePatternWithBank(hits []drums.Hit, gridSize float64, numBars int, soundType string, bank string) string {
	if len(hits) == 0 {
		return ""
	}

	// Grid slots per bar
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays
	slots := make([]bool, totalSlots)
	velocities := make([]float64, totalSlots)

	// Place hits in slots
	for _, hit := range hits {
		slot := int(hit.Time / gridSize)
		if slot >= 0 && slot < totalSlots {
			slots[slot] = true
			velocities[slot] = hit.VelocityNormalized
		}
	}

	// Analyze pattern to create a more compact representation
	// Count hits per bar to detect repetition patterns
	barsWithHits := 0
	hitCounts := make([]int, numBars)

	for bar := 0; bar < numBars; bar++ {
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if slots[i] {
				hitCounts[bar]++
			}
		}
		if hitCounts[bar] > 0 {
			barsWithHits++
		}
	}

	if barsWithHits == 0 {
		return ""
	}

	// Build pattern with bar structure
	var bars []string

	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasHits := false

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if slots[i] {
				barParts = append(barParts, soundType)
				barHasHits = true
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasHits {
			simplified := simplifyDrumPattern(barParts)
			bars = append(bars, simplified)
		}
	}

	if len(bars) == 0 {
		return ""
	}

	pattern := strings.Join(bars, " | ")

	// Build the output with bank if specified
	if bank != "" {
		return fmt.Sprintf("s(\"%s\").bank(\"%s\")", pattern, bank)
	}
	return fmt.Sprintf("s(\"%s\")", pattern)
}

// buildHiHatPattern creates a combined hi-hat pattern with closed/open hat alternation
func (g *Generator) buildHiHatPattern(hhHits, ohHits []drums.Hit, gridSize float64, numBars int, bank string) string {
	// Grid slots per bar
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays for both hi-hat types
	hhSlots := make([]bool, totalSlots)
	ohSlots := make([]bool, totalSlots)

	// Place hits in slots
	for _, hit := range hhHits {
		slot := int(hit.Time / gridSize)
		if slot >= 0 && slot < totalSlots {
			hhSlots[slot] = true
		}
	}
	for _, hit := range ohHits {
		slot := int(hit.Time / gridSize)
		if slot >= 0 && slot < totalSlots {
			ohSlots[slot] = true
		}
	}

	// Analyze hi-hat density to decide on pattern style
	totalHH := len(hhHits)
	totalOH := len(ohHits)

	if totalHH == 0 && totalOH == 0 {
		return ""
	}

	// Check if hi-hats are regular (every beat or every 8th/16th)
	// This helps create more compact patterns like "hh*8" instead of "hh ~ hh ~ hh ~ hh ~"
	hitsPerBar := (totalHH + totalOH) / numBars
	isRegular := false
	regularInterval := 0

	// Check for regular patterns
	if hitsPerBar >= slotsPerBar/2 {
		// Very dense - likely every 8th or 16th note
		isRegular = true
		if hitsPerBar >= slotsPerBar*3/4 {
			regularInterval = slotsPerBar
		} else {
			regularInterval = slotsPerBar / 2
		}
	}

	// Build pattern with bar structure
	var bars []string

	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasHits := false
		hhCount := 0
		ohCount := 0

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if ohSlots[i] {
				barParts = append(barParts, "oh")
				barHasHits = true
				ohCount++
			} else if hhSlots[i] {
				barParts = append(barParts, "hh")
				barHasHits = true
				hhCount++
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasHits {
			// Check if this bar has a regular pattern that can be simplified
			if isRegular && hhCount >= regularInterval-2 && ohCount <= 2 {
				// Can use multiplier notation
				if ohCount == 0 {
					bars = append(bars, fmt.Sprintf("hh*%d", hhCount))
				} else {
					// Mix of hh and oh - use the full pattern
					simplified := simplifyDrumPattern(barParts)
					bars = append(bars, simplified)
				}
			} else {
				simplified := simplifyDrumPattern(barParts)
				bars = append(bars, simplified)
			}
		}
	}

	if len(bars) == 0 {
		return ""
	}

	pattern := strings.Join(bars, " | ")

	// Build output with bank and cut(1) for hi-hats (open hat cuts closed)
	if bank != "" {
		return fmt.Sprintf("s(\"%s\").bank(\"%s\").cut(1)", pattern, bank)
	}
	return fmt.Sprintf("s(\"%s\").cut(1)", pattern)
}

// simplifyDrumPattern reduces consecutive rests in drum patterns
func simplifyDrumPattern(parts []string) string {
	if len(parts) == 0 {
		return "~"
	}

	var result []string
	restCount := 0
	var lastSound string

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
			lastSound = p
		}
	}

	// Handle trailing rests - skip them for cleaner output
	_ = lastSound

	if len(result) == 0 {
		return "~"
	}

	return strings.Join(result, " ")
}

// GenerateDrumHeader creates a comment header for drum patterns
func GenerateDrumHeader(result *drums.DetectionResult, kit DrumKit) string {
	if result == nil || result.Stats.Total == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("// Drums: ")

	// Build stats string
	var parts []string
	for drumType, count := range result.Stats.ByType {
		parts = append(parts, fmt.Sprintf("%s: %d", drumType, count))
	}
	sb.WriteString(fmt.Sprintf("%d hits (%s)\n", result.Stats.Total, strings.Join(parts, ", ")))
	sb.WriteString(fmt.Sprintf("// Kit: %s\n", kit))

	return sb.String()
}

// GenerateFullOutput creates complete Strudel output with melodic and drum patterns
func (g *Generator) GenerateFullOutput(melodicCode string, drumResult *drums.DetectionResult, kit DrumKit) string {
	if drumResult == nil || len(drumResult.Hits) == 0 {
		return melodicCode
	}

	// Generate drum pattern
	drumPattern := g.GenerateDrumPattern(drumResult, kit)
	if drumPattern == "" {
		return melodicCode
	}

	// Add drum header to the comment section
	drumHeader := GenerateDrumHeader(drumResult, kit)

	// Find where to insert drum info in header (after Notes line)
	lines := strings.Split(melodicCode, "\n")
	var newLines []string
	insertedHeader := false

	for _, line := range lines {
		newLines = append(newLines, line)
		if strings.HasPrefix(line, "// Notes:") && !insertedHeader {
			newLines = append(newLines, strings.TrimSuffix(drumHeader, "\n"))
			insertedHeader = true
		}
	}

	if !insertedHeader {
		// Insert after header comments
		newLines = append([]string{strings.TrimSuffix(drumHeader, "\n")}, newLines...)
	}

	result := strings.Join(newLines, "\n")

	// Detect output format and add drums appropriately
	hasSeparatePatterns := strings.Contains(result, "$bass:") || strings.Contains(result, "$mid:") || strings.Contains(result, "$high:")
	hasChunkedArrays := strings.Contains(result, "let bass = [") || strings.Contains(result, "let mid = [")
	hasStack := strings.Contains(result, "$: stack(")

	if hasChunkedArrays {
		// Chunked array format - add drums as bar arrays with effect function
		kitInfo := DrumKitSoundsInfo[kit]

		// Find where to insert (before "// Effects (applied at playback)")
		insertIdx := strings.Index(result, "// Effects (applied at playback)")
		if insertIdx > 0 {
			// Group drum hits by bar and type
			drumsCode := g.buildDrumBarArrays(drumResult, kit)
			result = result[:insertIdx] + drumsCode + "\n" + result[insertIdx:]
		}

		// Add drum effect function after other effect functions
		effectsEnd := strings.Index(result, "\n// Play all")
		if effectsEnd > 0 {
			// Build drum effect function
			bankStr := ""
			if kitInfo.Bank != "" {
				bankStr = fmt.Sprintf(".bank(\"%s\")", kitInfo.Bank)
			}
			drumFx := fmt.Sprintf("let drumsFx = p => p%s.room(0.15).gain(0.9)\n", bankStr)
			result = result[:effectsEnd] + "\n" + drumFx + result[effectsEnd:]
		}

		// Update the stack to include drums
		// The new format uses: bassFx(cat(...bass.map(b => note(b))))
		// For drums we use: drumsFx(cat(...drums.map(b => s(b))))
		// Find the last voice line ending with ))))\n before the stack close
		stackClose := strings.Index(result, "))))\n)\n\n// Play specific bars:")
		if stackClose > 0 {
			// Insert after the )))) but before \n)
			insertPos := stackClose + 4 // After ))))
			drumsLine := ",\n  drumsFx(cat(...drums.map(b => s(b))))"
			result = result[:insertPos] + drumsLine + result[insertPos:]
		}

	} else if hasSeparatePatterns {
		// Separate $voice: format - add $drums: pattern
		// Find the "// To play all together:" line and insert drums before it
		insertIdx := strings.Index(result, "// To play all together:")
		if insertIdx > 0 {
			drumsCode := "$drums: " + strings.TrimSpace(drumPattern) + "\n\n"
			result = result[:insertIdx] + drumsCode + result[insertIdx:]
		} else {
			// Just append drums
			result += "\n$drums: " + strings.TrimSpace(drumPattern) + "\n"
		}

		// Update the all() comment to include drums
		result = strings.Replace(result, "// all($bass, $mid, $high)", "// all($bass, $mid, $high, $drums)", 1)

	} else if hasStack {
		// Old stack format - add drums inside stack
		lastStackClose := strings.LastIndex(result, ")\n")
		if lastStackClose > 0 {
			result = result[:lastStackClose] + ",\n" + drumPattern + "\n" + result[lastStackClose:]
		}
	} else {
		// Unknown format - append drums as separate pattern
		result += "\n$drums: " + strings.TrimSpace(drumPattern) + "\n"
	}

	return result
}

// splitDrumsByBar splits drum hits into per-bar patterns
func splitDrumsByBar(drumResult *drums.DetectionResult, kit DrumKit, quantize int) []string {
	if drumResult == nil || len(drumResult.Hits) == 0 {
		return nil
	}

	// Group hits by bar (4 beats per bar)
	barHits := make(map[int][]drums.Hit)
	maxBar := 0
	for _, hit := range drumResult.Hits {
		bar := int(hit.Time / 4) // 4 beats per bar
		barHits[bar] = append(barHits[bar], hit)
		if bar > maxBar {
			maxBar = bar
		}
	}

	// Generate pattern for each bar
	var bars []string
	sounds := DrumKitSoundsInfo[kit]

	for bar := 0; bar <= maxBar; bar++ {
		hits := barHits[bar]
		if len(hits) == 0 {
			bars = append(bars, "s(\"~\")")
			continue
		}

		// Create a simple pattern for this bar
		// Group by type
		typeHits := make(map[string]bool)
		for _, h := range hits {
			typeHits[h.Type] = true
		}

		var patterns []string
		for dtype := range typeHits {
			sound := sounds.Sounds[dtype]
			if sound == "" {
				continue
			}
			// Simple pattern: just the sound name with bank
			patterns = append(patterns, fmt.Sprintf("s(\"%s\").bank(\"%s\")", sound, sounds.Bank))
		}

		if len(patterns) == 0 {
			bars = append(bars, "s(\"~\")")
		} else if len(patterns) == 1 {
			bars = append(bars, patterns[0])
		} else {
			bars = append(bars, "stack("+strings.Join(patterns, ", ")+")")
		}
	}

	return bars
}

// buildDrumBarArrays creates drum bar arrays in the same format as melodic voices
// Output: let drums = ["bd ~ sd ~", "bd sd ~ hh", ...]
func (g *Generator) buildDrumBarArrays(drumResult *drums.DetectionResult, kit DrumKit) string {
	if drumResult == nil || len(drumResult.Hits) == 0 {
		return ""
	}

	bpm := drumResult.Tempo
	if bpm <= 0 {
		bpm = 120
	}

	beatDuration := 60.0 / bpm
	gridSize := beatDuration / float64(g.quantize/4)

	// Find total duration and number of bars
	maxTime := 0.0
	for _, hit := range drumResult.Hits {
		if hit.Time > maxTime {
			maxTime = hit.Time
		}
	}

	numBars := int(math.Ceil(maxTime / (beatDuration * 4)))
	if numBars < 1 {
		numBars = 1
	}
	if numBars > 64 {
		numBars = 64
	}

	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays for each drum type
	type slotData struct {
		drumType string
		velocity float64
	}
	slots := make([]slotData, totalSlots)

	// Place hits in slots (later hits overwrite earlier ones at same slot)
	for _, hit := range drumResult.Hits {
		slot := int(hit.Time / gridSize)
		if slot >= 0 && slot < totalSlots {
			slots[slot] = slotData{hit.Type, hit.VelocityNormalized}
		}
	}

	// Build bar patterns
	var bars []string
	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasHits := false

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if slots[i].drumType != "" {
				barParts = append(barParts, slots[i].drumType)
				barHasHits = true
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasHits {
			simplified := simplifyDrumPattern(barParts)
			bars = append(bars, simplified)
		} else {
			bars = append(bars, "~")
		}
	}

	// Build output
	var sb strings.Builder
	sb.WriteString("let drums = [\n")
	for i, bar := range bars {
		sb.WriteString(fmt.Sprintf("  \"%s\"", bar))
		if i < len(bars)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("]\n")

	return sb.String()
}
