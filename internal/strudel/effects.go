package strudel

import (
	"fmt"
	"math"
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
	// FM Synthesis parameters
	FM        float64 // FM modulation index / brightness (0 = off)
	FMH       float64 // FM harmonicity ratio (0 = off, use default)
	FMDecay   float64 // FM envelope decay in seconds (0 = off)
	FMSustain float64 // FM envelope sustain level (0-1)
}

// PatternFXSettings defines pattern-level transformations
type PatternFXSettings struct {
	Jux        bool    // Apply jux(rev) for stereo width
	Swing      float64 // Swing amount (0 = off, 0.1 = subtle)
	DegradeBy  float64 // Random note removal (0 = off, 0.1 = subtle)
	Ply        int     // Repeat each note n times (0 = off)
	Iter       int     // Cycle through pattern subdivisions (0 = off)
	Rev        bool    // Reverse pattern every other cycle
	Sometimes  string  // Effect to apply sometimes (e.g., "crush(8)")
	Rarely     string  // Effect to apply rarely (e.g., "rev")
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

// TremoloSettings defines amplitude modulation parameters
type TremoloSettings struct {
	Sync  float64 // Tremolo sync rate (cycles, 0 = off)
	Depth float64 // Tremolo depth (0-1)
	Shape string  // Tremolo waveform: "sine", "tri", "saw", "square"
}

// FilterEnvSettings defines filter envelope parameters
type FilterEnvSettings struct {
	Attack  float64 // Filter attack time (0 = off)
	Decay   float64 // Filter decay time
	Sustain float64 // Filter sustain level (0-1)
	Release float64 // Filter release time
	Amount  float64 // Filter envelope amount (semitones, 0 = off)
}

// DuckSettings defines sidechain/ducking effect parameters
type DuckSettings struct {
	Orbit  int     // Orbit to duck against (0 = off)
	Attack float64 // Time for ducked signal to return (seconds)
	Depth  float64 // Amount of ducking (0-1)
}

// AccentSettings defines beat emphasis patterns
type AccentSettings struct {
	Pattern     string  // Accent pattern: "downbeat", "backbeat", "all-fours", "offbeat"
	Amount      float64 // Gain boost on accented beats (0.1-0.3 typical)
	Enabled     bool    // Whether accents are enabled
}

// CompressorSettings defines dynamics compression
type CompressorSettings struct {
	Threshold float64 // Threshold in dB (e.g., -20)
	Ratio     float64 // Compression ratio (e.g., 4 for 4:1)
	Knee      float64 // Knee softness
	Attack    float64 // Attack time in seconds
	Release   float64 // Release time in seconds
	Enabled   bool    // Whether compressor is enabled
}

// DynamicsSettings defines overall dynamics processing
type DynamicsSettings struct {
	RangeExpansion float64 // Expand velocity range (1.0 = none, 1.5 = more dynamic)
	VelocityCurve  string  // Curve type: "linear", "exponential", "logarithmic"
}

// VoiceEffects contains all effect settings for a voice
type VoiceEffects struct {
	Filter     FilterSettings
	Pan        PanSettings
	Reverb     ReverbSettings
	Delay      DelaySettings
	Envelope   EnvelopeSettings
	StyleFX    StyleFXSettings
	PatternFX  PatternFXSettings
	Legato     LegatoSettings
	Echo       EchoSettings
	Harmony    HarmonySettings
	Tremolo    TremoloSettings
	FilterEnv  FilterEnvSettings
	Duck       DuckSettings
	Accent     AccentSettings
	Compressor CompressorSettings
	Dynamics   DynamicsSettings
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
	UseTremolo       bool     // Whether to use tremolo/amplitude modulation
	UseFilterEnv     bool     // Whether to use filter envelope
	UseDuck          bool     // Whether to use ducking/sidechain effect
	IterAmount       int      // Iter subdivisions (0 = off)
	UseJux           bool     // Whether to use jux(rev) for stereo width
	PlyAmount        int      // Ply repetitions (0 = off)
	SometimesFX      string   // Effect to apply sometimes (50%)
	RarelyFX         string   // Effect to apply rarely (25%)
	AccentPattern    string   // Accent pattern: "downbeat", "backbeat", "offbeat"
	AccentAmount     float64  // Gain boost for accents (0 = off)
	UseCompressor    bool     // Whether to use compressor
	DynamicRange     float64  // Velocity range expansion (1.0 = none)
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
		Tremolo: TremoloSettings{
			Sync:  4,      // 4 cycles
			Depth: 0.3,    // Subtle modulation
			Shape: "sine", // Smooth
		},
		FilterEnv: FilterEnvSettings{},
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
		DynamicRange:     1.2,       // Slight expansion for expression
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
		UseEcho:          true,        // Echo for synth pads
		UseSuperimpose:   true,        // Detuned layers for richness
		UseOff:           true,        // Harmonic layering
		UseTremolo:       true,        // Amplitude modulation for movement
		UseFilterEnv:     true,        // Dynamic filter sweeps
		UseJux:           true,        // Stereo width via reversed right channel
		DynamicRange:     1.3,         // More dynamic range for expression
	},
	StyleOrchestral: {
		ModulationAmount: 0.2, // Very subtle
		FilterDynamic:    false,
		DelayEnabled:     false,
		ReverbAmount:     1.3, // More reverb
		LFOShape:         LFOSine,
		UseEnvelope:      true,      // Long attacks for strings
		UseStyleFX:       true,      // Enable vibrato
		SwingAmount:      0,
		LegatoAmount:     1.5,       // Long sustained notes
		UseEcho:          false,
		UseSuperimpose:   true,      // Slight detune for string section width
		UseOff:           true,      // Octave doubling for fullness
		UseTremolo:       true,      // Subtle tremolo for strings effect
		UseFilterEnv:     false,
		SometimesFX:      "add(12)", // Sometimes double an octave up
		AccentPattern:    "downbeat", // Accent on beats 1 and 3
		AccentAmount:     0.12,      // Moderate accent for orchestral dynamics
		DynamicRange:     1.5,       // Wide dynamics for orchestral expression
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
		LegatoAmount:     0.8,      // Tighter, punchier
		UseEcho:          true,
		UseSuperimpose:   true,
		UseOff:           true,
		UseTremolo:       true,     // Amplitude modulation for movement
		UseFilterEnv:     true,     // Dynamic filter sweeps
		UseDuck:          true,     // Sidechain pumping effect
		IterAmount:       4,        // Cycle through 4 subdivisions
		PlyAmount:        2,        // Double each bass note for drive
		AccentPattern:    "downbeat", // Accent 1 and 3
		AccentAmount:     0.15,     // Moderate accent
		UseCompressor:    true,     // Tight compression for punch
		DynamicRange:     0.8,      // Tighter dynamics for consistency
	},
	StyleJazz: {
		ModulationAmount: 0.4,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.0,
		LFOShape:         LFOPerlin, // Organic movement
		UseEnvelope:      false,
		UseStyleFX:       true, // Enable vibrato
		SwingAmount:      0.1,  // Subtle swing
		LegatoAmount:     1.0,
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           true,       // Harmonic layering for rich chords
		SometimesFX:      "add(7)",   // Sometimes add a fifth
		RarelyFX:         "room(0.6)", // Rarely more reverb
		AccentPattern:    "backbeat", // Accent 2 and 4 (jazz feel)
		AccentAmount:     0.1,        // Subtle accent
		DynamicRange:     1.4,        // Wide dynamics for expression
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
		LegatoAmount:     1.1,       // Slightly sustained for dreamy feel
		UseEcho:          true,
		UseSuperimpose:   true,      // Subtle detune
		UseOff:           false,
		IterAmount:       4,         // Cycle through 4 subdivisions for variation
		SometimesFX:      "lpf(800)", // Sometimes muffle the sound
		RarelyFX:         "rev",     // Rarely reverse for tape-like effect
		AccentPattern:    "offbeat", // Subtle offbeat accents for groove
		AccentAmount:     0.08,      // Very subtle
		DynamicRange:     1.1,       // Slightly compressed lofi feel
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
		FM:           1.5,  // Subtle FM for warmth
		FMH:          1,    // Unison harmonicity
		FMDecay:      0.3,
		FMSustain:    0.5,
	},
	StyleOrchestral: {
		Vibrato:      5,
		VibratoDepth: 0.15,
	},
	StyleElectronic: {
		Phaser:      0.8,
		PhaserDepth: 0.5,
		Distort:     0.1,
		FM:          2,    // Moderate FM for brightness
		FMH:         2,    // Octave harmonicity
		FMDecay:     0.2,
		FMSustain:   0.3,
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

	// Apply iter for styles that use it
	if styleMod.IterAmount > 0 {
		effects.PatternFX.Iter = styleMod.IterAmount
	}

	// Apply jux for styles that use it (only on high voice to avoid muddiness)
	if styleMod.UseJux && voice == "high" {
		effects.PatternFX.Jux = true
	}

	// Apply ply for styles that use it (only on bass for rhythmic drive)
	if styleMod.PlyAmount > 0 && voice == "bass" {
		effects.PatternFX.Ply = styleMod.PlyAmount
	}

	// Apply sometimes/rarely effects
	if styleMod.SometimesFX != "" {
		effects.PatternFX.Sometimes = styleMod.SometimesFX
	}
	if styleMod.RarelyFX != "" {
		effects.PatternFX.Rarely = styleMod.RarelyFX
	}

	// Apply accent settings
	if styleMod.AccentAmount > 0 && styleMod.AccentPattern != "" {
		effects.Accent = AccentSettings{
			Pattern: styleMod.AccentPattern,
			Amount:  styleMod.AccentAmount,
			Enabled: true,
		}
	}

	// Apply compressor for styles that use it
	if styleMod.UseCompressor {
		effects.Compressor = getCompressorForStyle(style)
	}

	// Apply dynamic range settings
	if styleMod.DynamicRange > 0 && styleMod.DynamicRange != 1.0 {
		effects.Dynamics = DynamicsSettings{
			RangeExpansion: styleMod.DynamicRange,
			VelocityCurve:  "exponential",
		}
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

	// Apply tremolo for styles that use it (mainly on mid/high voices)
	if styleMod.UseTremolo && voice != "bass" {
		if effects.Tremolo.Sync == 0 {
			effects.Tremolo = getTremoloForStyle(style)
		}
	} else if !styleMod.UseTremolo {
		effects.Tremolo = TremoloSettings{}
	}

	// Apply filter envelope for styles that use it (mainly on bass/mid)
	if styleMod.UseFilterEnv && voice != "high" {
		effects.FilterEnv = getFilterEnvForStyle(style, voice)
	} else if !styleMod.UseFilterEnv {
		effects.FilterEnv = FilterEnvSettings{}
	}

	// Apply ducking for styles that use it (mainly on mid/high voices)
	if styleMod.UseDuck && voice != "bass" {
		effects.Duck = getDuckForStyle(style, voice)
	} else {
		effects.Duck = DuckSettings{}
	}

	return effects
}

// getTremoloForStyle returns tremolo settings for a style
func getTremoloForStyle(style SoundStyle) TremoloSettings {
	switch style {
	case StyleSynth:
		return TremoloSettings{
			Sync:  8,      // 8 cycles - slow
			Depth: 0.2,    // Subtle
			Shape: "sine", // Smooth
		}
	case StyleElectronic:
		return TremoloSettings{
			Sync:  4,     // Faster
			Depth: 0.4,   // More pronounced
			Shape: "tri", // Slightly edgier
		}
	case StyleOrchestral:
		return TremoloSettings{
			Sync:  6,      // Moderate
			Depth: 0.15,   // Very subtle
			Shape: "sine", // Smooth like strings
		}
	default:
		return TremoloSettings{}
	}
}

// getFilterEnvForStyle returns filter envelope settings for a style and voice
func getFilterEnvForStyle(style SoundStyle, voice string) FilterEnvSettings {
	switch style {
	case StyleSynth:
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.01,
				Decay:   0.3,
				Sustain: 0.4,
				Release: 0.2,
				Amount:  2000, // 2000 Hz sweep
			}
		}
		return FilterEnvSettings{
			Attack:  0.05,
			Decay:   0.4,
			Sustain: 0.5,
			Release: 0.3,
			Amount:  3000, // 3000 Hz sweep
		}
	case StyleElectronic:
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.005,
				Decay:   0.2,
				Sustain: 0.3,
				Release: 0.1,
				Amount:  2500, // Punchy bass sweep
			}
		}
		return FilterEnvSettings{
			Attack:  0.02,
			Decay:   0.3,
			Sustain: 0.4,
			Release: 0.2,
			Amount:  4000, // Wide sweep
		}
	default:
		return FilterEnvSettings{}
	}
}

// getCompressorForStyle returns compressor settings for a style
func getCompressorForStyle(style SoundStyle) CompressorSettings {
	switch style {
	case StyleElectronic:
		return CompressorSettings{
			Threshold: -20,
			Ratio:     4,
			Knee:      10,
			Attack:    0.003,
			Release:   0.1,
			Enabled:   true,
		}
	case StyleLofi:
		return CompressorSettings{
			Threshold: -15,
			Ratio:     3,
			Knee:      15,
			Attack:    0.01,
			Release:   0.2,
			Enabled:   true,
		}
	default:
		return CompressorSettings{}
	}
}

// getDuckForStyle returns ducking settings for a style and voice
func getDuckForStyle(style SoundStyle, voice string) DuckSettings {
	switch style {
	case StyleElectronic:
		if voice == "mid" {
			return DuckSettings{
				Orbit:  1,    // Duck against orbit 1 (bass)
				Attack: 0.1,  // Quick return
				Depth:  0.4,  // Moderate pumping
			}
		}
		// High voice - more subtle ducking
		return DuckSettings{
			Orbit:  1,
			Attack: 0.15,
			Depth:  0.3,
		}
	default:
		return DuckSettings{}
	}
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

	// Filter envelope (for synth/electronic styles)
	if effects.FilterEnv.Amount > 0 {
		parts = append(parts, fmt.Sprintf(".lpattack(%.3f).lpdecay(%.2f).lpsustain(%.2f).lprelease(%.2f).lpenv(%.0f)",
			effects.FilterEnv.Attack, effects.FilterEnv.Decay,
			effects.FilterEnv.Sustain, effects.FilterEnv.Release, effects.FilterEnv.Amount))
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

	// FM Synthesis
	if effects.StyleFX.FM > 0 {
		parts = append(parts, fmt.Sprintf(".fm(%.1f)", effects.StyleFX.FM))
		if effects.StyleFX.FMH > 0 {
			parts = append(parts, fmt.Sprintf(".fmh(%.1f)", effects.StyleFX.FMH))
		}
		if effects.StyleFX.FMDecay > 0 {
			parts = append(parts, fmt.Sprintf(".fmdecay(%.2f)", effects.StyleFX.FMDecay))
		}
		if effects.StyleFX.FMSustain > 0 {
			parts = append(parts, fmt.Sprintf(".fmsustain(%.2f)", effects.StyleFX.FMSustain))
		}
	}

	// Legato/Clip for note duration
	if effects.Legato.Clip > 0 && effects.Legato.Clip != 1.0 {
		parts = append(parts, fmt.Sprintf(".clip(%.2f)", effects.Legato.Clip))
	}

	// Tremolo (amplitude modulation)
	if effects.Tremolo.Sync > 0 {
		parts = append(parts, fmt.Sprintf(".tremolo(%.1f).tremolodepth(%.2f)",
			effects.Tremolo.Sync, effects.Tremolo.Depth))
		if effects.Tremolo.Shape != "" && effects.Tremolo.Shape != "sine" {
			parts = append(parts, fmt.Sprintf(".tremoloshape(\"%s\")", effects.Tremolo.Shape))
		}
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

	// Ducking/Sidechain effect
	if effects.Duck.Orbit > 0 {
		parts = append(parts, fmt.Sprintf(".duck(%d).duckattack(%.2f).duckdepth(%.2f)",
			effects.Duck.Orbit, effects.Duck.Attack, effects.Duck.Depth))
	}

	// Compressor for dynamics control
	if effects.Compressor.Enabled {
		parts = append(parts, fmt.Sprintf(".compressor(\"%.0f:%.0f:%.0f:%.3f:%.2f\")",
			effects.Compressor.Threshold, effects.Compressor.Ratio, effects.Compressor.Knee,
			effects.Compressor.Attack, effects.Compressor.Release))
	}

	return strings.Join(parts, "")
}

// BuildPatternTransforms generates pattern-level transformation chain
func BuildPatternTransforms(effects VoiceEffects) string {
	var parts []string

	if effects.PatternFX.Jux {
		parts = append(parts, ".jux(rev)")
	}
	if effects.PatternFX.Iter > 0 {
		parts = append(parts, fmt.Sprintf(".iter(%d)", effects.PatternFX.Iter))
	}
	if effects.PatternFX.Rev {
		parts = append(parts, ".lastOf(2, rev)")
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
	if effects.PatternFX.Sometimes != "" {
		parts = append(parts, fmt.Sprintf(".sometimes(x => x.%s)", effects.PatternFX.Sometimes))
	}
	if effects.PatternFX.Rarely != "" {
		parts = append(parts, fmt.Sprintf(".rarely(x => x.%s)", effects.PatternFX.Rarely))
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

// BuildAccentPattern generates gain pattern for beat accents
// Returns a gain pattern string like "1.1 1 1.1 1" for downbeat accents
func BuildAccentPattern(pattern string, amount float64, beatsPerBar int) string {
	if pattern == "" || amount <= 0 {
		return ""
	}

	baseGain := 1.0
	accentGain := baseGain + amount

	var gains []string
	for i := 0; i < beatsPerBar; i++ {
		beat := i + 1 // 1-indexed beat
		isAccented := false

		switch pattern {
		case "downbeat":
			// Accent beats 1 and 3 (in 4/4)
			isAccented = beat == 1 || beat == 3
		case "backbeat":
			// Accent beats 2 and 4 (jazz/rock)
			isAccented = beat == 2 || beat == 4
		case "offbeat":
			// Accent the "and" of each beat (requires 8th note resolution)
			isAccented = beat%2 == 0
		case "all-fours":
			// Accent all four beats equally
			isAccented = true
		}

		if isAccented {
			gains = append(gains, fmt.Sprintf("%.2f", accentGain))
		} else {
			gains = append(gains, fmt.Sprintf("%.2f", baseGain))
		}
	}

	return strings.Join(gains, " ")
}

// ApplyDynamicRange expands or compresses velocity values
func ApplyDynamicRange(velocity float64, expansion float64) float64 {
	if expansion == 1.0 {
		return velocity
	}

	// Center point for expansion (0.5)
	center := 0.5
	deviation := velocity - center

	// Expand deviation from center
	newVelocity := center + (deviation * expansion)

	// Clamp to valid range
	if newVelocity < 0.1 {
		newVelocity = 0.1
	}
	if newVelocity > 1.0 {
		newVelocity = 1.0
	}

	return newVelocity
}

// ApplyVelocityCurve applies a curve to velocity for more expression
func ApplyVelocityCurve(velocity float64, curveType string) float64 {
	switch curveType {
	case "exponential":
		// More dramatic - soft notes softer, loud notes louder
		return velocity * velocity
	case "logarithmic":
		// Less dramatic - brings up quieter notes
		if velocity <= 0 {
			return 0
		}
		return 0.5 + (0.5 * (1 + math.Log10(velocity*10)/2))
	default: // linear
		return velocity
	}
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
