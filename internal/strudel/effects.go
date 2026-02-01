package strudel

import (
	"fmt"
	"strings"
)

// LFOShape defines different LFO waveform types
type LFOShape string

const (
	LFOSine   LFOShape = "sine"
	LFOSaw    LFOShape = "saw"
	LFOTri    LFOShape = "tri"
	LFOSquare LFOShape = "square"
	LFOPerlin LFOShape = "perlin" // Smooth random
	LFORand   LFOShape = "rand"   // Random per-event
)

// FilterSettings defines high-pass and low-pass filter parameters
type FilterSettings struct {
	HPF       int     // High-pass filter cutoff frequency (Hz)
	LPF       int     // Low-pass filter cutoff frequency (Hz)
	Resonance float64 // Filter resonance (0.0-1.0)
}

// PanSettings defines stereo positioning
type PanSettings struct {
	Position string   // Static value "0.5" or LFO expression
	IsLFO    bool     // Whether Position is an LFO expression
	Width    float64  // How wide the stereo movement (0.0-1.0)
	Shape    LFOShape // LFO waveform shape
	Speed    float64  // LFO speed (cycles)
}

// ReverbSettings defines room reverb parameters
type ReverbSettings struct {
	Room float64 // Reverb amount (0.0-1.0)
	Size float64 // Room size (0.0-1.0)
}

// DelaySettings defines delay effect parameters
type DelaySettings struct {
	Mix      float64 // Delay mix (0.0-1.0)
	Time     float64 // Delay time in seconds
	Feedback float64 // Delay feedback (0.0-1.0)
}

// EnvelopeSettings defines ADSR envelope parameters
type EnvelopeSettings struct {
	Attack  float64 // Attack time in seconds
	Decay   float64 // Decay time in seconds
	Sustain float64 // Sustain level (0.0-1.0)
	Release float64 // Release time in seconds
}

// StyleFXSettings defines style-specific special effects
type StyleFXSettings struct {
	Phaser       float64 // Phaser speed (0 = off)
	PhaserDepth  float64 // Phaser depth (0-1)
	Crush        int     // Bit crush depth (0 = off, 1-16)
	Coarse       int     // Sample rate reduction (0 = off)
	Vowel        string  // Formant filter vowel (empty = off)
	Distort      float64 // Distortion amount (0 = off)
	Vibrato      float64 // Vibrato speed (0 = off)
	VibratoDepth float64 // Vibrato depth in semitones
}

// PatternFXSettings defines pattern-level transformations
type PatternFXSettings struct {
	Jux       bool    // Apply jux(rev) for stereo width
	Swing     float64 // Swing amount (0 = off, 0.1 = subtle)
	DegradeBy float64 // Random note removal (0 = off, 0.1 = subtle)
	Ply       int     // Repeat each note n times (0 = off)
}

// LegatoSettings defines note duration/articulation
type LegatoSettings struct {
	Clip float64 // Note duration multiplier (0.5=staccato, 1.0=legato, 2.0=sustained)
}

// EchoSettings defines echo/stutter effect
type EchoSettings struct {
	Times    int     // Number of echoes (0 = off)
	Time     float64 // Time between echoes (in cycles, e.g., 1/8)
	Feedback float64 // Volume reduction per echo (0.0-1.0)
}

// HarmonySettings defines layering/harmony effects
type HarmonySettings struct {
	Off            float64 // Time offset for .off() (0 = off)
	OffInterval    int     // Semitone interval for .off() (e.g., 7 = 5th)
	Superimpose    float64 // Detune amount for .superimpose() (0 = off)
	SuperimposeOct int     // Octave for superimpose (0 = same, 12 = octave up)
}

// VoiceEffects contains all effect settings for a voice
type VoiceEffects struct {
	Filter    FilterSettings
	Pan       PanSettings
	Reverb    ReverbSettings
	Delay     DelaySettings
	Envelope  EnvelopeSettings
	StyleFX   StyleFXSettings
	PatternFX PatternFXSettings
	Legato    LegatoSettings
	Echo      EchoSettings
	Harmony   HarmonySettings
}

// StyleEffects defines effect variations per style
type StyleEffects struct {
	ModulationAmount float64  // How much LFO modulation to apply
	FilterDynamic    bool     // Whether filter responds to energy
	DelayEnabled     bool     // Whether to use delay
	ReverbAmount     float64  // Reverb multiplier
	LFOShape         LFOShape // Preferred LFO shape for this style
	UseEnvelope      bool     // Whether to use ADSR envelope
	UseStyleFX       bool     // Whether to use style-specific effects
	SwingAmount      float64  // Amount of swing (0 = none)
	LegatoAmount     float64  // Note duration multiplier (1.0 = default)
	UseEcho          bool     // Whether to use echo effect
	UseSuperimpose   bool     // Whether to use detuned superimpose
	UseOff           bool     // Whether to use .off() for harmony
}

// Voice effect presets by voice type
var voiceEffectPresets = map[string]VoiceEffects{
	"bass": {
		Filter: FilterSettings{
			HPF:       50,  // Clean sub-bass mud
			LPF:       800, // Roll off highs
			Resonance: 0.3,
		},
		Pan: PanSettings{
			Position: "0.5", // Center
			IsLFO:    false,
			Width:    0,
			Shape:    LFOSine,
			Speed:    4,
		},
		Reverb: ReverbSettings{
			Room: 0.2, // Less reverb on bass
			Size: 0.3,
		},
		Delay: DelaySettings{
			Mix:      0,
			Time:     0,
			Feedback: 0,
		},
		Envelope: EnvelopeSettings{
			Attack:  0.005, // Quick attack for punch
			Decay:   0.1,
			Sustain: 0.8,
			Release: 0.1,
		},
		Legato: LegatoSettings{
			Clip: 0.9, // Slightly shorter for punch
		},
		Echo: EchoSettings{},
		Harmony: HarmonySettings{},
	},
	"mid": {
		Filter: FilterSettings{
			HPF:       200,
			LPF:       4000,
			Resonance: 0.4,
		},
		Pan: PanSettings{
			Position: "sine.range(0.35,0.65).slow(4)", // Gentle stereo movement
			IsLFO:    true,
			Width:    0.3,
			Shape:    LFOSine,
			Speed:    4,
		},
		Reverb: ReverbSettings{
			Room: 0.35,
			Size: 0.5,
		},
		Delay: DelaySettings{
			Mix:      0,
			Time:     0,
			Feedback: 0,
		},
		Envelope: EnvelopeSettings{
			Attack:  0.01,
			Decay:   0.2,
			Sustain: 0.7,
			Release: 0.3,
		},
		Legato: LegatoSettings{
			Clip: 1.0, // Normal legato
		},
		Echo: EchoSettings{},
		Harmony: HarmonySettings{},
	},
	"high": {
		Filter: FilterSettings{
			HPF:       400,
			LPF:       10000, // Keep bright
			Resonance: 0.35,
		},
		Pan: PanSettings{
			Position: "sine.range(0.2,0.8).slow(3)", // Wider stereo
			IsLFO:    true,
			Width:    0.6,
			Shape:    LFOSine,
			Speed:    3,
		},
		Reverb: ReverbSettings{
			Room: 0.4,
			Size: 0.6,
		},
		Delay: DelaySettings{
			Mix:      0.15,
			Time:     0.375, // Dotted eighth
			Feedback: 0.3,
		},
		Envelope: EnvelopeSettings{
			Attack:  0.02,
			Decay:   0.3,
			Sustain: 0.6,
			Release: 0.5,
		},
		Legato: LegatoSettings{
			Clip: 1.2, // Slightly longer for sustain
		},
		Echo: EchoSettings{
			Times:    2,
			Time:     0.125, // 1/8 note
			Feedback: 0.5,
		},
		Harmony: HarmonySettings{},
	},
}

// Style-specific effect modifications
var styleEffectMods = map[SoundStyle]StyleEffects{
	StylePiano: {
		ModulationAmount: 0.3, // Subtle
		FilterDynamic:    false,
		DelayEnabled:     false,
		ReverbAmount:     1.0,
		LFOShape:         LFOSine,
		UseEnvelope:      false, // Piano has natural envelope
		UseStyleFX:       false,
		SwingAmount:      0,
		LegatoAmount:     1.0,
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
	},
	StyleSynth: {
		ModulationAmount: 0.7,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.8,
		LFOShape:         LFOSaw, // Sawtooth for rhythmic feel
		UseEnvelope:      true,
		UseStyleFX:       true,
		SwingAmount:      0,
		LegatoAmount:     1.0,
		UseEcho:          true, // Echo for synth pads
		UseSuperimpose:   true, // Detuned layers for richness
		UseOff:           true, // Harmonic layering
	},
	StyleOrchestral: {
		ModulationAmount: 0.2, // Very subtle
		FilterDynamic:    false,
		DelayEnabled:     false,
		ReverbAmount:     1.3, // More reverb
		LFOShape:         LFOSine,
		UseEnvelope:      true, // Long attacks for strings
		UseStyleFX:       false,
		SwingAmount:      0,
		LegatoAmount:     1.5, // Long sustained notes
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
	},
	StyleElectronic: {
		ModulationAmount: 1.0, // Full modulation
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.7,
		LFOShape:         LFOSaw, // Rhythmic filter sweeps
		UseEnvelope:      true,
		UseStyleFX:       true,
		SwingAmount:      0,
		LegatoAmount:     0.8, // Tighter, punchier
		UseEcho:          true,
		UseSuperimpose:   true,
		UseOff:           true,
	},
	StyleJazz: {
		ModulationAmount: 0.4,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.0,
		LFOShape:         LFOPerlin, // Organic movement
		UseEnvelope:      false,
		UseStyleFX:       false,
		SwingAmount:      0.1, // Subtle swing
		LegatoAmount:     1.0,
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
	},
	StyleLofi: {
		ModulationAmount: 0.5,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.9,
		LFOShape:         LFOPerlin, // Organic wobble
		UseEnvelope:      false,
		UseStyleFX:       true, // Bitcrush, coarse
		SwingAmount:      0.05,
		LegatoAmount:     1.1, // Slightly sustained for dreamy feel
		UseEcho:          true,
		UseSuperimpose:   true, // Subtle detune
		UseOff:           false,
	},
}

// Style-specific FX presets
var styleFXPresets = map[SoundStyle]StyleFXSettings{
	StylePiano: {},
	StyleSynth: {
		Phaser:       0.5,
		PhaserDepth:  0.3,
		Vibrato:      4,
		VibratoDepth: 0.1,
	},
	StyleOrchestral: {
		Vibrato:      5,
		VibratoDepth: 0.15,
	},
	StyleElectronic: {
		Phaser:      0.8,
		PhaserDepth: 0.5,
		Distort:     0.1,
	},
	StyleJazz: {
		Vibrato:      3,
		VibratoDepth: 0.08,
	},
	StyleLofi: {
		Crush:  10, // Subtle bit reduction
		Coarse: 4,  // Sample rate reduction
	},
}

// GetVoiceEffects returns effect settings for a voice, adjusted for style
func GetVoiceEffects(voice string, style SoundStyle) VoiceEffects {
	// Get base preset
	effects, ok := voiceEffectPresets[voice]
	if !ok {
		effects = voiceEffectPresets["mid"] // Default to mid settings
	}

	// Get style modifications
	styleMod, ok := styleEffectMods[style]
	if !ok {
		styleMod = styleEffectMods[StylePiano]
	}

	// Apply style modifications
	effects.Reverb.Room *= styleMod.ReverbAmount
	effects.Reverb.Size *= styleMod.ReverbAmount

	// Clamp reverb values
	if effects.Reverb.Room > 0.8 {
		effects.Reverb.Room = 0.8
	}
	if effects.Reverb.Size > 0.9 {
		effects.Reverb.Size = 0.9
	}

	// Apply LFO shape based on style
	if effects.Pan.IsLFO {
		effects.Pan.Shape = styleMod.LFOShape
		effects.Pan = buildPanLFO(effects.Pan, styleMod.ModulationAmount)
	}

	// Disable delay if style doesn't use it
	if !styleMod.DelayEnabled {
		effects.Delay = DelaySettings{}
	}

	// Clear envelope if style doesn't use it
	if !styleMod.UseEnvelope {
		effects.Envelope = EnvelopeSettings{}
	} else {
		// Adjust envelope based on voice type
		effects.Envelope = adjustEnvelopeForVoice(effects.Envelope, voice, style)
	}

	// Apply style-specific FX
	if styleMod.UseStyleFX {
		if fx, ok := styleFXPresets[style]; ok {
			effects.StyleFX = fx
		}
	}

	// Apply pattern FX
	effects.PatternFX.Swing = styleMod.SwingAmount
	if style == StyleLofi {
		effects.PatternFX.DegradeBy = 0.05 // Subtle random note removal
	}

	// Apply legato amount based on style
	if styleMod.LegatoAmount > 0 {
		effects.Legato.Clip *= styleMod.LegatoAmount
	}

	// Apply echo for styles that use it
	if !styleMod.UseEcho {
		effects.Echo = EchoSettings{}
	} else if effects.Echo.Times == 0 {
		// Set default echo if style wants it but voice doesn't have it
		effects.Echo = EchoSettings{
			Times:    2,
			Time:     0.125,
			Feedback: 0.4,
		}
	}

	// Apply superimpose/harmony for styles that use it
	if styleMod.UseSuperimpose && effects.Harmony.Superimpose == 0 {
		effects.Harmony.Superimpose = 0.03 // Subtle detune
	}
	if !styleMod.UseSuperimpose {
		effects.Harmony.Superimpose = 0
	}

	// Apply .off() for styles that use it
	if styleMod.UseOff && effects.Harmony.Off == 0 {
		effects.Harmony.Off = 0.125       // 1/8 note offset
		effects.Harmony.OffInterval = 12  // Octave up
	}
	if !styleMod.UseOff {
		effects.Harmony.Off = 0
	}

	return effects
}

// buildPanLFO creates an LFO pan expression with the specified shape
func buildPanLFO(pan PanSettings, modAmount float64) PanSettings {
	if !pan.IsLFO {
		return pan
	}

	// Calculate range based on modulation amount
	center := 0.5
	newWidth := pan.Width * modAmount
	minPan := center - newWidth/2
	maxPan := center + newWidth/2

	// Build LFO expression based on shape
	var position string
	switch pan.Shape {
	case LFOPerlin:
		position = fmt.Sprintf("perlin.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
	case LFOSaw:
		position = fmt.Sprintf("saw.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
	case LFOTri:
		position = fmt.Sprintf("tri.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
	case LFOSquare:
		position = fmt.Sprintf("square.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
	case LFORand:
		position = fmt.Sprintf("rand.range(%.2f,%.2f)", minPan, maxPan)
	default: // LFOSine
		position = fmt.Sprintf("sine.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
	}

	return PanSettings{
		Position: position,
		IsLFO:    true,
		Width:    newWidth,
		Shape:    pan.Shape,
		Speed:    pan.Speed,
	}
}

// adjustEnvelopeForVoice adjusts envelope based on voice and style
func adjustEnvelopeForVoice(env EnvelopeSettings, voice string, style SoundStyle) EnvelopeSettings {
	switch style {
	case StyleOrchestral:
		// Longer attacks for strings
		env.Attack *= 3
		env.Release *= 2
	case StyleSynth, StyleElectronic:
		// Punchy attacks
		env.Attack *= 0.5
		env.Decay *= 0.8
	}

	// Voice-specific adjustments
	switch voice {
	case "bass":
		env.Attack *= 0.5 // Quick attack
		env.Sustain = 0.9  // High sustain
	case "high":
		env.Release *= 1.5 // Longer release for sparkle
	}

	return env
}

// BuildEffectChain generates Strudel effect method chain for a voice
func BuildEffectChain(effects VoiceEffects, includeFilter bool) string {
	var parts []string

	// Pan - always include
	parts = append(parts, fmt.Sprintf(".pan(%s)", effects.Pan.Position))

	// Filters - optional based on style/preference
	if includeFilter {
		if effects.Filter.HPF > 0 {
			parts = append(parts, fmt.Sprintf(".hpf(%d)", effects.Filter.HPF))
		}
		if effects.Filter.LPF > 0 && effects.Filter.LPF < 20000 {
			parts = append(parts, fmt.Sprintf(".lpf(%d)", effects.Filter.LPF))
		}
	}

	// ADSR Envelope
	if effects.Envelope.Attack > 0 || effects.Envelope.Release > 0 {
		parts = append(parts, fmt.Sprintf(".attack(%.3f).decay(%.2f).sustain(%.2f).release(%.2f)",
			effects.Envelope.Attack, effects.Envelope.Decay,
			effects.Envelope.Sustain, effects.Envelope.Release))
	}

	// Style-specific effects
	if effects.StyleFX.Vibrato > 0 {
		parts = append(parts, fmt.Sprintf(".vib(%.1f).vibmod(%.2f)",
			effects.StyleFX.Vibrato, effects.StyleFX.VibratoDepth))
	}
	if effects.StyleFX.Phaser > 0 {
		parts = append(parts, fmt.Sprintf(".phaser(%.2f).phaserdepth(%.2f)",
			effects.StyleFX.Phaser, effects.StyleFX.PhaserDepth))
	}
	if effects.StyleFX.Crush > 0 {
		parts = append(parts, fmt.Sprintf(".crush(%d)", effects.StyleFX.Crush))
	}
	if effects.StyleFX.Coarse > 0 {
		parts = append(parts, fmt.Sprintf(".coarse(%d)", effects.StyleFX.Coarse))
	}
	if effects.StyleFX.Distort > 0 {
		parts = append(parts, fmt.Sprintf(".distort(%.2f)", effects.StyleFX.Distort))
	}
	if effects.StyleFX.Vowel != "" {
		parts = append(parts, fmt.Sprintf(".vowel(\"%s\")", effects.StyleFX.Vowel))
	}

	// Legato/Clip for note duration
	if effects.Legato.Clip > 0 && effects.Legato.Clip != 1.0 {
		parts = append(parts, fmt.Sprintf(".clip(%.2f)", effects.Legato.Clip))
	}

	// Reverb
	if effects.Reverb.Room > 0 {
		parts = append(parts, fmt.Sprintf(".room(%.2f).size(%.2f)", effects.Reverb.Room, effects.Reverb.Size))
	}

	// Delay
	if effects.Delay.Mix > 0 {
		parts = append(parts, fmt.Sprintf(".delay(%.2f).delaytime(%.3f).delayfeedback(%.2f)",
			effects.Delay.Mix, effects.Delay.Time, effects.Delay.Feedback))
	}

	// Echo/Stutter effect
	if effects.Echo.Times > 0 {
		parts = append(parts, fmt.Sprintf(".echo(%d,%.3f,%.2f)",
			effects.Echo.Times, effects.Echo.Time, effects.Echo.Feedback))
	}

	return strings.Join(parts, "")
}

// BuildPatternTransforms generates pattern-level transformation chain
func BuildPatternTransforms(effects VoiceEffects) string {
	var parts []string

	if effects.PatternFX.Jux {
		parts = append(parts, ".jux(rev)")
	}
	if effects.PatternFX.Swing > 0 {
		parts = append(parts, fmt.Sprintf(".swing(%.2f)", effects.PatternFX.Swing))
	}
	if effects.PatternFX.DegradeBy > 0 {
		parts = append(parts, fmt.Sprintf(".degradeBy(%.2f)", effects.PatternFX.DegradeBy))
	}
	if effects.PatternFX.Ply > 1 {
		parts = append(parts, fmt.Sprintf(".ply(%d)", effects.PatternFX.Ply))
	}

	return strings.Join(parts, "")
}

// FormatGain formats a gain value for Strudel output
func FormatGain(gain float64) string {
	if gain == 1.0 {
		return ""
	}
	return fmt.Sprintf(".gain(%.2f)", gain)
}

// VelocityToGain converts normalized velocity (0-1) to gain value
func VelocityToGain(velocityNormalized float64) float64 {
	// Map 0-1 velocity to 0.5-1.2 gain range
	// This gives dynamic range without being too extreme
	return 0.5 + (velocityNormalized * 0.7)
}

// BuildGainPattern creates a gain pattern string from velocity values
func BuildGainPattern(velocities []float64) string {
	if len(velocities) == 0 {
		return ""
	}

	var gains []string
	for _, v := range velocities {
		gain := VelocityToGain(v)
		gains = append(gains, fmt.Sprintf("%.2f", gain))
	}

	return strings.Join(gains, " ")
}

// BuildVelocityPattern creates a velocity pattern string (0-1 range)
func BuildVelocityPattern(velocities []float64) string {
	if len(velocities) == 0 {
		return ""
	}

	var vels []string
	for _, v := range velocities {
		vels = append(vels, fmt.Sprintf("%.2f", v))
	}

	return strings.Join(vels, " ")
}

// BuildHarmonyEffects generates harmony/layering effects (.off, .superimpose)
func BuildHarmonyEffects(effects VoiceEffects) string {
	var parts []string

	// Superimpose for detuned voices (creates fuller sound)
	if effects.Harmony.Superimpose > 0 {
		if effects.Harmony.SuperimposeOct != 0 {
			parts = append(parts, fmt.Sprintf(".superimpose(add(%.2f).add(note(%d)))",
				effects.Harmony.Superimpose, effects.Harmony.SuperimposeOct))
		} else {
			parts = append(parts, fmt.Sprintf(".superimpose(add(%.2f))",
				effects.Harmony.Superimpose))
		}
	}

	// Off for harmonic layering
	if effects.Harmony.Off > 0 && effects.Harmony.OffInterval != 0 {
		parts = append(parts, fmt.Sprintf(".off(%.3f, add(%d))",
			effects.Harmony.Off, effects.Harmony.OffInterval))
	}

	return strings.Join(parts, "")
}

// BuildScaleEffect generates scale quantization effect
// key should be in format "C:minor", "E:major", etc.
func BuildScaleEffect(key string) string {
	if key == "" {
		return ""
	}

	// Convert key format from "E minor" to "E:minor"
	key = strings.ReplaceAll(key, " ", ":")

	// Capitalize first letter
	if len(key) > 0 {
		key = strings.ToUpper(key[:1]) + key[1:]
	}

	return fmt.Sprintf(".scale(\"%s\")", key)
}
