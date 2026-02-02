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
	// Raw synthesizer styles using basic waveforms
	StyleRaw      SoundStyle = "raw"      // Pure waveforms (sawtooth, square, triangle)
	StyleChiptune SoundStyle = "chiptune" // 8-bit style with square/triangle waves
	StyleAmbient  SoundStyle = "ambient"  // Sine waves with heavy reverb/delay
	StyleDrone    SoundStyle = "drone"    // Sustained tones with slow modulation
	// Sample-based styles using Strudel sample libraries
	StyleMallets    SoundStyle = "mallets"    // Vibraphone, marimba, xylophone samples
	StylePlucked    SoundStyle = "plucked"    // Harp, kalimba, music box samples
	StyleKeys       SoundStyle = "keys"       // Salamander piano, rhodes, wurlitzer
	StylePad        SoundStyle = "pad"        // Soft pad sounds with long tails
	StylePercussive SoundStyle = "percussive" // Timpani, mallet percussion
	// Genre-specific styles
	StyleSynthwave  SoundStyle = "synthwave"  // 80s retro synth aesthetic
	StyleDarkwave   SoundStyle = "darkwave"   // Dark, atmospheric synths
	StyleMinimal    SoundStyle = "minimal"    // Clean, sparse electronic
	StyleIndustrial SoundStyle = "industrial" // Harsh, distorted sounds
	StyleNewAge     SoundStyle = "newage"     // Soft, ethereal, meditative
	// Noise and texture styles
	StyleNoise   SoundStyle = "noise"   // Noise-based ambient/texture
	StyleGlitch  SoundStyle = "glitch"  // Glitchy, broken sounds
	StyleTexture SoundStyle = "texture" // Textural background layers
	StyleRetro   SoundStyle = "retro"   // ZZFX retro game sounds
	// Dance music styles
	StyleHouse    SoundStyle = "house"    // House music sounds
	StyleTrance   SoundStyle = "trance"   // Trance/EDM sounds
	StyleDub      SoundStyle = "dub"      // Reggae/dub sounds
	StyleFunk     SoundStyle = "funk"     // Funk/disco sounds
	StyleSoul     SoundStyle = "soul"     // Soul/R&B sounds
	StyleCinematic SoundStyle = "cinematic" // Cinematic/film score
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
// Sound names must match exactly what's in strudel-client/packages/soundfonts/gm.mjs
var soundPalettes = map[SoundStyle]SoundPalette{
	StylePiano: {
		Bass:     "gm_piano",
		BassGain: 1.2,
		Mid:      "gm_piano",
		MidGain:  1.0,
		High:     "gm_piano",
		HighGain: 0.9,
	},
	StyleSynth: {
		Bass:     "supersaw",           // Fat supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_poly",        // Poly pad
		MidGain:  0.8,
		High:     "gm_lead_2_sawtooth", // Saw lead
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
		Bass:     "supersaw",           // Supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_sweep",       // Sweeping pad
		MidGain:  0.7,
		High:     "gm_lead_5_charang",  // Charang lead
		HighGain: 0.6,
	},
	StyleJazz: {
		Bass:     "gm_acoustic_bass",
		BassGain: 1.2,
		Mid:      "gm_epiano1",
		MidGain:  1.0,
		High:     "gm_vibraphone",
		HighGain: 0.8,
	},
	StyleLofi: {
		Bass:     "gm_electric_bass_finger",
		BassGain: 1.1,
		Mid:      "gm_epiano2",
		MidGain:  0.9,
		High:     "gm_music_box",
		HighGain: 0.7,
	},
	// Raw synthesizer styles using Strudel's built-in oscillators
	StyleRaw: {
		Bass:     "sawtooth", // Raw sawtooth oscillator
		BassGain: 0.8,
		Mid:      "square", // Raw square wave
		MidGain:  0.6,
		High:     "triangle", // Softer triangle wave
		HighGain: 0.5,
	},
	StyleChiptune: {
		Bass:     "z_square",   // ZZFX 8-bit bass
		BassGain: 0.7,
		Mid:      "z_square",   // ZZFX chiptune
		MidGain:  0.5,
		High:     "z_triangle", // ZZFX NES-style lead
		HighGain: 0.4,
	},
	StyleAmbient: {
		Bass:     "sine",    // Pure sine bass
		BassGain: 1.0,
		Mid:      "triangle", // Soft mid tones
		MidGain:  0.7,
		High:     "sine",    // Pure sine highs
		HighGain: 0.6,
	},
	StyleDrone: {
		Bass:     "sawtooth", // Rich drone bass
		BassGain: 0.6,
		Mid:      "sine",     // Clean mid drone
		MidGain:  0.5,
		High:     "triangle", // Soft high harmonics
		HighGain: 0.4,
	},
	// Sample-based styles using Strudel sample libraries
	StyleMallets: {
		Bass:     "gm_marimba",     // Warm marimba bass
		BassGain: 1.1,
		Mid:      "gm_vibraphone",  // Classic vibes
		MidGain:  0.9,
		High:     "gm_xylophone",   // Bright xylophone
		HighGain: 0.7,
	},
	StylePlucked: {
		Bass:     "gm_acoustic_bass", // Upright bass
		BassGain: 1.2,
		Mid:      "harp",             // VCSL harp samples
		MidGain:  0.9,
		High:     "gm_music_box",     // Delicate music box
		HighGain: 0.6,
	},
	StyleKeys: {
		Bass:     "piano",           // Salamander grand piano
		BassGain: 1.1,
		Mid:      "piano",           // Salamander grand piano
		MidGain:  1.0,
		High:     "piano",           // Salamander grand piano
		HighGain: 0.9,
	},
	StylePad: {
		Bass:     "gm_pad_warm",     // Warm analog pad
		BassGain: 0.8,
		Mid:      "gm_pad_choir",    // Choir pad
		MidGain:  0.7,
		High:     "gm_pad_halo",     // Ethereal pad
		HighGain: 0.6,
	},
	StylePercussive: {
		Bass:     "timpani",         // VCSL timpani
		BassGain: 1.3,
		Mid:      "gm_marimba",      // Marimba
		MidGain:  1.0,
		High:     "gm_tubular_bells", // Tubular bells
		HighGain: 0.8,
	},
	// Genre-specific styles
	StyleSynthwave: {
		Bass:     "sawtooth",           // Fat analog bass
		BassGain: 0.9,
		Mid:      "gm_pad_poly",        // Poly synth pad
		MidGain:  0.7,
		High:     "gm_lead_2_sawtooth", // Bright lead
		HighGain: 0.6,
	},
	StyleDarkwave: {
		Bass:     "sawtooth",        // Dark bass
		BassGain: 0.8,
		Mid:      "gm_pad_metallic", // Metallic pad
		MidGain:  0.6,
		High:     "triangle",        // Softer high
		HighGain: 0.4,
	},
	StyleMinimal: {
		Bass:     "sine",    // Pure sine bass
		BassGain: 0.9,
		Mid:      "triangle", // Clean mid
		MidGain:  0.5,
		High:     "sine",    // Pure high
		HighGain: 0.4,
	},
	StyleIndustrial: {
		Bass:     "sawtooth",       // Aggressive bass
		BassGain: 1.0,
		Mid:      "square",         // Harsh mid
		MidGain:  0.7,
		High:     "sawtooth",       // Harsh high
		HighGain: 0.5,
	},
	StyleNewAge: {
		Bass:     "sine",           // Pure bass
		BassGain: 0.8,
		Mid:      "gm_pad_warm",    // Warm pad
		MidGain:  0.7,
		High:     "gm_pad_halo",    // Ethereal
		HighGain: 0.5,
	},
	// Noise-based styles using Strudel's noise oscillators
	StyleNoise: {
		Bass:     "brown",    // Soft rumble bass
		BassGain: 0.6,
		Mid:      "pink",     // Mid-frequency noise
		MidGain:  0.4,
		High:     "white",    // High noise texture
		HighGain: 0.3,
	},
	StyleGlitch: {
		Bass:     "sawtooth", // Digital bass
		BassGain: 0.8,
		Mid:      "square",   // Harsh digital
		MidGain:  0.5,
		High:     "crackle",  // Glitchy texture
		HighGain: 0.4,
	},
	// Texture styles for layering
	StyleTexture: {
		Bass:     "brown",          // Noise bed
		BassGain: 0.4,
		Mid:      "gm_pad_sweep",   // Moving pad
		MidGain:  0.5,
		High:     "pink",           // Bright texture
		HighGain: 0.3,
	},
	// ZZFX synth styles (more aggressive/retro)
	StyleRetro: {
		Bass:     "z_sawtooth", // ZZFX sawtooth
		BassGain: 0.8,
		Mid:      "z_square",   // ZZFX square
		MidGain:  0.6,
		High:     "z_sine",     // ZZFX sine
		HighGain: 0.5,
	},
	// House/Techno style
	StyleHouse: {
		Bass:     "supersaw",         // Fat supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_poly",      // Poly stab
		MidGain:  0.8,
		High:     "gm_lead_5_charang", // Charang lead
		HighGain: 0.6,
	},
	// Trance style
	StyleTrance: {
		Bass:     "supersaw",          // Supersaw bass
		BassGain: 0.9,
		Mid:      "gm_pad_sweep",      // Sweeping pad
		MidGain:  0.7,
		High:     "gm_lead_2_sawtooth", // Saw lead
		HighGain: 0.6,
	},
	// Reggae/Dub style
	StyleDub: {
		Bass:     "gm_synth_bass_2", // Deep dub bass
		BassGain: 1.4,
		Mid:      "gm_epiano1",      // Skank keys
		MidGain:  0.8,
		High:     "gm_harmonica",    // Melodica/harmonica
		HighGain: 0.6,
	},
	// Funk style
	StyleFunk: {
		Bass:     "gm_slap_bass_1",  // Slap bass
		BassGain: 1.2,
		Mid:      "gm_clavinet",     // Clavinet
		MidGain:  1.0,
		High:     "gm_brass_section", // Brass stabs
		HighGain: 0.7,
	},
	// Soul/R&B style
	StyleSoul: {
		Bass:     "gm_electric_bass_finger", // Finger bass
		BassGain: 1.1,
		Mid:      "gm_epiano2",              // Wurlitzer
		MidGain:  1.0,
		High:     "gm_flute",                // Soft flute
		HighGain: 0.7,
	},
	// Cinematic style
	StyleCinematic: {
		Bass:     "gm_contrabass",       // Deep strings
		BassGain: 1.2,
		Mid:      "gm_string_ensemble_1", // Full strings
		MidGain:  1.0,
		High:     "gm_choir_aahs",        // Choir
		HighGain: 0.8,
	},
}

// AdditionalSounds lists all available sounds from strudel-client
// These can be used with .sound() in Strudel patterns
var AdditionalSounds = map[string][]string{
	// Built-in oscillators
	"oscillators": {
		"sine", "triangle", "square", "sawtooth",
		"supersaw", "pulse",
	},
	// Noise generators
	"noise": {
		"white", "pink", "brown", "crackle",
	},
	// ZZFX synths (retro game sounds)
	"zzfx": {
		"z_sine", "z_sawtooth", "z_triangle", "z_square", "z_tan", "z_noise",
	},
	// Wavetables
	"wavetables": {
		"wt_digital", "wt_digital_bad_day", "wt_digital_basique",
		"wt_digital_crickets", "wt_digital_curses", "wt_digital_echoes", "wt_vgame",
	},
	// Classic drum machines
	"drum_machines": {
		"RolandTR808", "RolandTR909", "RolandTR707", "RolandTR606", "RolandTR505",
		"LinnDrum", "LinnLM1", "AkaiMPC60", "EmuSP12", "OberheimDMX",
		"KorgKR55", "KorgMinipops", "AlesisHR16", "BossDR110",
	},
	// VCSL percussion
	"vcsl_percussion": {
		"Timpani", "Cajon", "Conga", "Bongos", "Cowbells", "Claves",
		"Tambourine", "Shaker", "Agogo Bells", "Frame Drum", "Gong",
		"Bass Drum", "Snare Drum", "Hi Hat Cymbal", "Clash Cymbals",
	},
	// Piano samples
	"piano": {"piano"}, // Salamander Grand Piano
	// Exotic/World
	"world": {
		"gm_sitar", "gm_banjo", "gm_shamisen", "gm_koto", "gm_kalimba",
		"gm_bagpipe", "gm_fiddle", "gm_shanai",
	},
	// Sound effects
	"fx": {
		"gm_fx_rain", "gm_fx_soundtrack", "gm_fx_crystal", "gm_fx_atmosphere",
		"gm_fx_brightness", "gm_fx_goblins", "gm_fx_echoes", "gm_fx_sci_fi",
		"gm_seashore", "gm_bird_tweet", "gm_helicopter", "gm_applause",
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

	// Detect sections for comments
	sections := DetectSections(result)

	// Header with stats
	sb.WriteString("// MIDI-grep output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f", analysisResult.BPM))
	if analysisResult.Key != "" {
		sb.WriteString(fmt.Sprintf(", Key: %s", analysisResult.Key))
	}
	sb.WriteString(fmt.Sprintf("\n// Notes: %d (bass: %d, mid: %d, high: %d)\n",
		result.Stats.Total, result.Stats.BassCount, result.Stats.MidCount, result.Stats.HighCount))
	sb.WriteString(fmt.Sprintf("// Style: %s\n", g.style))
	sb.WriteString(fmt.Sprintf("// Duration: %.1f beats\n", result.TotalBeats))

	// Add section markers
	if len(sections) > 0 {
		sectionInfo := GenerateSectionHeader(sections, analysisResult.BPM)
		sb.WriteString(sectionInfo)
	}
	sb.WriteString("\n")

	// Tempo
	sb.WriteString(fmt.Sprintf("setcps(%.0f/60/4)\n\n", analysisResult.BPM))

	// Convert JSON notes to midi.Note (for pattern generation)
	bass := jsonToNotes(result.Voices.Bass)
	mid := jsonToNotes(result.Voices.Mid)
	high := jsonToNotes(result.Voices.High)

	// Generate the stacked pattern with velocity data
	g.generateStackedPatternWithVelocity(&sb, bass, mid, high,
		result.Voices.Bass, result.Voices.Mid, result.Voices.High,
		analysisResult.BPM)

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
	g.generateStackedPatternWithVelocity(sb, bass, mid, high, nil, nil, nil, bpm)
}

// generateStackedPatternWithVelocity creates a Strudel stack() with per-voice effects and velocity dynamics
func (g *Generator) generateStackedPatternWithVelocity(sb *strings.Builder, bass, mid, high []midi.Note, bassJSON, midJSON, highJSON []NoteJSON, bpm float64) {
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

	// Define voices with their sounds, effects, and optional velocity data
	voices := []struct {
		name     string
		notes    []midi.Note
		jsonData []NoteJSON
		sound    string
		gain     float64
	}{
		{"bass", bass, bassJSON, g.palette.Bass, g.palette.BassGain},
		{"mid", mid, midJSON, g.palette.Mid, g.palette.MidGain},
		{"high", high, highJSON, g.palette.High, g.palette.HighGain},
	}

	var activeVoices []string
	for _, v := range voices {
		if len(v.notes) > 0 {
			pattern := g.voiceToPattern(v.notes, gridSize, numBars, beatDuration)
			if pattern != "" && pattern != "~" {
				// Get per-voice effects based on voice type and style
				effects := GetVoiceEffects(v.name, g.style)

				// Build voice code with per-voice effects
				voiceCode := fmt.Sprintf("  // %s (%d notes)\n", v.name, len(v.notes))
				voiceCode += fmt.Sprintf("  note(\"%s\")\n", pattern)
				voiceCode += fmt.Sprintf("    .sound(\"%s\")", v.sound)

				// Add velocity pattern for dynamics (proper Strudel way)
				// Also add base gain for voice level balance
				if len(v.jsonData) > 0 {
					// Build velocity pattern with dynamic range expansion
					velocityPattern := g.buildVelocityPatternWithDynamics(v.jsonData, gridSize, numBars, effects.Dynamics)
					if velocityPattern != "" {
						voiceCode += fmt.Sprintf("\n    .velocity(\"%s\")", velocityPattern)
					}
				}
				// Always add base gain for voice balance
				if v.gain != 1.0 {
					voiceCode += fmt.Sprintf("\n    .gain(%.2f)", v.gain)
				}

				// Add per-voice effect chain (filter, pan, reverb, delay, envelope, style FX, legato, echo)
				effectChain := BuildEffectChain(effects, true)
				if effectChain != "" {
					voiceCode += "\n    " + effectChain
				}

				// Add harmony effects (superimpose, off) for layering
				harmonyFX := BuildHarmonyEffects(effects)
				if harmonyFX != "" {
					voiceCode += "\n    " + harmonyFX
				}

				// Add pattern-level transforms (swing, degradeBy, etc.)
				patternTransforms := BuildPatternTransforms(effects)
				if patternTransforms != "" {
					voiceCode += "\n    " + patternTransforms
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
		// Single voice - no stack needed, effects already included per-voice
		sb.WriteString("$: ")
		sb.WriteString(strings.TrimPrefix(activeVoices[0], "  "))
		sb.WriteString("\n")
	} else {
		// Multiple voices - use stack, effects already per-voice
		sb.WriteString("$: stack(\n")
		for i, voice := range activeVoices {
			sb.WriteString(voice)
			if i < len(activeVoices)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n")
		}
		sb.WriteString(")\n")
	}
}

// buildVelocityGainPattern creates a gain pattern matching the note pattern from velocity data
func (g *Generator) buildVelocityGainPattern(notes []NoteJSON, gridSize float64, numBars int) string {
	if len(notes) == 0 {
		return ""
	}

	// Grid slots per bar (e.g., 16 for 16th notes in 4/4)
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays for velocities
	slots := make([]float64, totalSlots)
	hasNote := make([]bool, totalSlots)

	// Place velocity values in slots
	for _, n := range notes {
		slot := int(n.Start / gridSize)
		if slot >= 0 && slot < totalSlots {
			// For chords, use average velocity
			if hasNote[slot] {
				slots[slot] = (slots[slot] + VelocityToGain(n.VelocityNormalized)) / 2
			} else {
				slots[slot] = VelocityToGain(n.VelocityNormalized)
			}
			hasNote[slot] = true
		}
	}

	// Build pattern with bar structure matching note pattern
	var bars []string
	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasNotes := false

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if hasNote[i] {
				barParts = append(barParts, fmt.Sprintf("%.2f", slots[i]))
				barHasNotes = true
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasNotes {
			// Simplify the gain pattern same as note pattern
			simplified := simplifyPattern(barParts)
			if simplified != "" && !isAllRests(simplified) {
				bars = append(bars, simplified)
			}
		}
	}

	if len(bars) == 0 {
		return ""
	}

	return strings.Join(bars, " | ")
}

// buildVelocityPattern creates a velocity pattern (0-1 range) matching the note pattern
func (g *Generator) buildVelocityPattern(notes []NoteJSON, gridSize float64, numBars int) string {
	// Use default dynamics (no expansion)
	return g.buildVelocityPatternWithDynamics(notes, gridSize, numBars, DynamicsSettings{RangeExpansion: 1.0})
}

// buildVelocityPatternWithDynamics creates a velocity pattern with dynamic range processing
func (g *Generator) buildVelocityPatternWithDynamics(notes []NoteJSON, gridSize float64, numBars int, dynamics DynamicsSettings) string {
	if len(notes) == 0 {
		return ""
	}

	// Grid slots per bar (e.g., 16 for 16th notes in 4/4)
	slotsPerBar := g.quantize
	totalSlots := slotsPerBar * numBars

	// Create slot arrays for velocities
	slots := make([]float64, totalSlots)
	hasNote := make([]bool, totalSlots)

	// Place velocity values in slots (normalized 0-1 range)
	for _, n := range notes {
		slot := int(n.Start / gridSize)
		if slot >= 0 && slot < totalSlots {
			velocity := n.VelocityNormalized

			// Apply dynamic range expansion
			if dynamics.RangeExpansion > 0 && dynamics.RangeExpansion != 1.0 {
				velocity = ApplyDynamicRange(velocity, dynamics.RangeExpansion)
			}

			// Apply velocity curve for more expression
			if dynamics.VelocityCurve != "" && dynamics.VelocityCurve != "linear" {
				velocity = ApplyVelocityCurve(velocity, dynamics.VelocityCurve)
			}

			// For chords, use average velocity
			if hasNote[slot] {
				slots[slot] = (slots[slot] + velocity) / 2
			} else {
				slots[slot] = velocity
			}
			hasNote[slot] = true
		}
	}

	// Build pattern with bar structure matching note pattern
	var bars []string
	for bar := 0; bar < numBars; bar++ {
		var barParts []string
		startSlot := bar * slotsPerBar
		endSlot := startSlot + slotsPerBar
		barHasNotes := false

		for i := startSlot; i < endSlot && i < totalSlots; i++ {
			if hasNote[i] {
				barParts = append(barParts, fmt.Sprintf("%.2f", slots[i]))
				barHasNotes = true
			} else {
				barParts = append(barParts, "~")
			}
		}

		if barHasNotes {
			// Simplify the velocity pattern same as note pattern
			simplified := simplifyPattern(barParts)
			if simplified != "" && !isAllRests(simplified) {
				bars = append(bars, simplified)
			}
		}
	}

	if len(bars) == 0 {
		return ""
	}

	return strings.Join(bars, " | ")
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

// ValidGMSounds contains all valid GM soundfont names from strudel-client
// Source: strudel-client/packages/soundfonts/gm.mjs
var ValidGMSounds = map[string]bool{
	// Piano
	"gm_piano":       true,
	"gm_epiano1":     true,
	"gm_epiano2":     true,
	"gm_harpsichord": true,
	"gm_clavinet":    true,
	// Chromatic Percussion
	"gm_celesta":      true,
	"gm_glockenspiel": true,
	"gm_music_box":    true,
	"gm_vibraphone":   true,
	"gm_marimba":      true,
	"gm_xylophone":    true,
	"gm_tubular_bells": true,
	"gm_dulcimer":     true,
	// Organ
	"gm_drawbar_organ":    true,
	"gm_percussive_organ": true,
	"gm_rock_organ":       true,
	"gm_church_organ":     true,
	"gm_reed_organ":       true,
	"gm_accordion":        true,
	"gm_harmonica":        true,
	"gm_bandoneon":        true,
	// Guitar
	"gm_acoustic_guitar_nylon": true,
	"gm_acoustic_guitar_steel": true,
	"gm_electric_guitar_jazz":  true,
	"gm_electric_guitar_clean": true,
	"gm_electric_guitar_muted": true,
	"gm_overdriven_guitar":     true,
	"gm_distortion_guitar":     true,
	"gm_guitar_harmonics":      true,
	// Bass
	"gm_acoustic_bass":        true,
	"gm_electric_bass_finger": true,
	"gm_electric_bass_pick":   true,
	"gm_fretless_bass":        true,
	"gm_slap_bass_1":          true,
	"gm_slap_bass_2":          true,
	"gm_synth_bass_1":         true,
	"gm_synth_bass_2":         true,
	// Strings
	"gm_violin":            true,
	"gm_viola":             true,
	"gm_cello":             true,
	"gm_contrabass":        true,
	"gm_tremolo_strings":   true,
	"gm_pizzicato_strings": true,
	"gm_orchestral_harp":   true,
	"gm_timpani":           true,
	"gm_string_ensemble_1": true,
	"gm_string_ensemble_2": true,
	"gm_synth_strings_1":   true,
	"gm_synth_strings_2":   true,
	// Choir
	"gm_choir_aahs":     true,
	"gm_voice_oohs":     true,
	"gm_synth_choir":    true,
	"gm_orchestra_hit":  true,
	// Brass
	"gm_trumpet":       true,
	"gm_trombone":      true,
	"gm_tuba":          true,
	"gm_muted_trumpet": true,
	"gm_french_horn":   true,
	"gm_brass_section": true,
	"gm_synth_brass_1": true,
	"gm_synth_brass_2": true,
	// Reed
	"gm_soprano_sax":  true,
	"gm_alto_sax":     true,
	"gm_tenor_sax":    true,
	"gm_baritone_sax": true,
	"gm_oboe":         true,
	"gm_english_horn": true,
	"gm_bassoon":      true,
	"gm_clarinet":     true,
	// Pipe
	"gm_piccolo":       true,
	"gm_flute":         true,
	"gm_recorder":      true,
	"gm_pan_flute":     true,
	"gm_blown_bottle":  true,
	"gm_shakuhachi":    true,
	"gm_whistle":       true,
	"gm_ocarina":       true,
	// Synth Lead
	"gm_lead_1_square":    true,
	"gm_lead_2_sawtooth":  true,
	"gm_lead_3_calliope":  true,
	"gm_lead_4_chiff":     true,
	"gm_lead_5_charang":   true,
	"gm_lead_6_voice":     true,
	"gm_lead_7_fifths":    true,
	"gm_lead_8_bass_lead": true,
	// Synth Pad
	"gm_pad_new_age":  true,
	"gm_pad_warm":     true,
	"gm_pad_poly":     true,
	"gm_pad_choir":    true,
	"gm_pad_bowed":    true,
	"gm_pad_metallic": true,
	"gm_pad_halo":     true,
	"gm_pad_sweep":    true,
	// Synth Effects
	"gm_fx_rain":       true,
	"gm_fx_soundtrack": true,
	"gm_fx_crystal":    true,
	"gm_fx_atmosphere": true,
	"gm_fx_brightness": true,
	"gm_fx_goblins":    true,
	"gm_fx_echoes":     true,
	"gm_fx_sci_fi":     true,
	// Ethnic
	"gm_sitar":    true,
	"gm_banjo":    true,
	"gm_shamisen": true,
	"gm_koto":     true,
	"gm_kalimba":  true,
	"gm_bagpipe":  true,
	"gm_fiddle":   true,
	"gm_shanai":   true,
	// Percussive
	"gm_tinkle_bell":    true,
	"gm_agogo":          true,
	"gm_steel_drums":    true,
	"gm_woodblock":      true,
	"gm_taiko_drum":     true,
	"gm_melodic_tom":    true,
	"gm_synth_drum":     true,
	"gm_reverse_cymbal": true,
	// Sound Effects
	"gm_guitar_fret_noise": true,
	"gm_breath_noise":      true,
	"gm_seashore":          true,
	"gm_bird_tweet":        true,
	"gm_telephone":         true,
	"gm_helicopter":        true,
	"gm_applause":          true,
	"gm_gunshot":           true,
	// Built-in Oscillators
	"sine":      true,
	"triangle":  true,
	"square":    true,
	"sawtooth":  true,
	"supersaw":  true,
	"pulse":     true,
	// Noise Generators
	"white":   true,
	"pink":    true,
	"brown":   true,
	"crackle": true,
	// ZZFX Synths (retro game sounds)
	"z_sine":     true,
	"z_sawtooth": true,
	"z_triangle": true,
	"z_square":   true,
	"z_tan":      true,
	"z_noise":    true,
	// Wavetables
	"wt_digital":           true,
	"wt_digital_bad_day":   true,
	"wt_digital_basique":   true,
	"wt_digital_crickets":  true,
	"wt_digital_curses":    true,
	"wt_digital_echoes":    true,
	"wt_vgame":             true,
	// Salamander Piano
	"piano": true,
	// VCSL samples (use with .s())
	"harp":    true,
	"timpani": true,
}

// IsValidSound checks if a sound name is valid in Strudel
func IsValidSound(sound string) bool {
	return ValidGMSounds[sound]
}

// ValidateSound returns the sound if valid, otherwise returns a fallback
func ValidateSound(sound, fallback string) string {
	if IsValidSound(sound) {
		return sound
	}
	return fallback
}

// ListValidSounds returns all valid GM sound names
func ListValidSounds() []string {
	sounds := make([]string, 0, len(ValidGMSounds))
	for sound := range ValidGMSounds {
		sounds = append(sounds, sound)
	}
	sort.Strings(sounds)
	return sounds
}
