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
	LFOCosine LFOShape = "cosine" // Phase-shifted sine for smoother transitions
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
	PhaserCenter float64 // Phaser center frequency (Hz)
	PhaserSweep  float64 // Phaser sweep range (Hz)
	Crush        int     // Bit crush depth (0 = off, 1-16)
	Coarse       int     // Sample rate reduction (0 = off)
	Vowel        string  // Formant filter vowel (empty = off): a, e, i, o, u
	Distort      float64 // Distortion amount (0 = off)
	Shape        float64 // Waveshaping/saturation (0 = off, 0-1)
	Vibrato      float64 // Vibrato speed (0 = off)
	VibratoDepth float64 // Vibrato depth in semitones
	// FM Synthesis parameters
	FM        float64 // FM modulation index / brightness (0 = off)
	FMH       float64 // FM harmonicity ratio (0 = off, use default)
	FMDecay   float64 // FM envelope decay in seconds (0 = off)
	FMSustain float64 // FM envelope sustain level (0-1)
	// Additional modulation effects
	Ring       float64 // Ring modulation frequency in Hz (0 = off)
	Chorus     float64 // Chorus depth (0 = off, 0.5 = moderate)
	Leslie     float64 // Leslie speaker simulation (0 = off, 1 = full)
	LeslieRate float64 // Leslie rotation speed
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
	Layer          string  // Layer transformation function (e.g., "add(12).lpf(2000)")
	EchoWith       string  // Custom echo function (e.g., "add(7)")
	EchoWithTimes  int     // Number of echoWith iterations
	EchoWithTime   float64 // Time between echoWith iterations
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

// SlideSettings defines pitch slide/portamento parameters
type SlideSettings struct {
	Slide      float64 // Pitch slide amount (+/- semitones, 0 = off)
	DeltaSlide float64 // Additional pitch slide
}

// FilterTypeSettings defines filter type selection
type FilterTypeSettings struct {
	Type string // Filter type: "12db", "ladder", "24db"
}

// OrbitSettings defines effect bus routing
type OrbitSettings struct {
	Orbit int // Effect bus number (0-11, each has separate effects)
}

// ShapeSettings defines waveshaping/saturation parameters
type ShapeSettings struct {
	Shape    float64 // Waveshaping amount (0-1, subtle saturation to hard clipping)
	Postgain float64 // Gain after shaping to compensate for level changes
}

// RingSettings defines ring modulation parameters
type RingSettings struct {
	Ring     float64 // Ring modulation frequency in Hz (0 = off)
	RingMix  float64 // Wet/dry mix for ring mod (0-1)
}

// ChorusSettings defines chorus effect parameters
type ChorusSettings struct {
	Chorus      float64 // Chorus depth/amount (0 = off, 0.5 = moderate)
	ChorusRate  float64 // Chorus LFO rate in Hz
	ChorusDelay float64 // Chorus delay time in ms
}

// LeslieSettings defines Leslie speaker simulation
type LeslieSettings struct {
	Leslie float64 // Leslie effect amount (0 = off, 1 = full)
	LRate  float64 // Leslie rotation speed (slow=0.5, fast=5)
	LSize  float64 // Leslie cabinet size/depth
}

// BandPassSettings defines band-pass filter parameters
type BandPassSettings struct {
	BPF int     // Band-pass center frequency in Hz (0 = off)
	BPQ float64 // Band-pass Q/resonance (0.1-50)
}

// PitchEnvSettings defines pitch envelope parameters
type PitchEnvSettings struct {
	PAttack  float64 // Pitch envelope attack time
	PDecay   float64 // Pitch envelope decay time
	PRelease float64 // Pitch envelope release time
	PEnv     float64 // Pitch envelope depth in semitones (can be negative)
	PCurve   string  // Envelope curve type: "lin" or "exp"
	PAnchor  float64 // Envelope anchor point (0, 0.5, or 1)
}

// GranularSettings defines granular synthesis parameters
type GranularSettings struct {
	Striate int     // Cut sample into n parts, trigger progressively (0 = off)
	Chop    int     // Cut sample into n parts for granular exploration (0 = off)
	Slice   int     // Number of slices for slice pattern (0 = off)
	Speed   float64 // Playback speed multiplier (1.0 = normal, -1 = reverse)
	Begin   float64 // Sample start point (0-1)
	End     float64 // Sample end point (0-1)
	Loop    bool    // Enable sample looping
	Cut     int     // Cutgroup number (same group samples cut each other)
}

// ZZFXSettings defines ZZFX synth engine parameters
type ZZFXSettings struct {
	Curve         float64 // Envelope curve shape
	ZMod          float64 // Modulation amount
	ZCrush        float64 // ZZFX-specific crush
	ZDelay        float64 // ZZFX delay
	PitchJump     float64 // Pitch jump amount in semitones
	PitchJumpTime float64 // When pitch jump occurs
	LFO           float64 // LFO rate for ZZFX
	Noise         float64 // Noise amount in ZZFX synth
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
	BandPass   BandPassSettings
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
	PitchEnv   PitchEnvSettings
	Duck       DuckSettings
	Accent     AccentSettings
	Compressor CompressorSettings
	Dynamics   DynamicsSettings
	Slide      SlideSettings
	FilterType FilterTypeSettings
	Orbit      OrbitSettings
	Shape      ShapeSettings
	Ring       RingSettings
	Chorus     ChorusSettings
	Leslie     LeslieSettings
	Granular   GranularSettings
	ZZFX       ZZFXSettings
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
	LayerFX          string   // Layer transformation (e.g., "add(12).lpf(1000)")
	EchoWithFX       string   // EchoWith custom function (e.g., "add(7)")
	EchoWithTimes    int      // EchoWith iterations (0 = off)
	UseRangex        bool     // Use exponential range for filter LFOs
	UseSlide         bool     // Whether to use pitch slide
	SlideAmount      float64  // Pitch slide amount
	FilterType       string   // Filter type: "12db", "ladder", "24db"
	UseOrbit         bool     // Whether to use separate orbit
	OrbitNumber      int      // Orbit bus number
	// New modulation effects
	UseShape       bool    // Whether to use waveshaping/saturation
	ShapeAmount    float64 // Waveshaping amount (0-1)
	UseRing        bool    // Whether to use ring modulation
	RingFreq       float64 // Ring mod frequency in Hz
	UseChorus      bool    // Whether to use chorus
	ChorusAmount   float64 // Chorus depth
	UseLeslie      bool    // Whether to use Leslie speaker sim
	LeslieSpeed    float64 // Leslie rotation speed
	UsePitchEnv    bool    // Whether to use pitch envelope
	PitchEnvAmount float64 // Pitch envelope depth in semitones
	UseBandPass    bool    // Whether to use band-pass filter
	BandPassFreq   int     // Band-pass center frequency
	BandPassQ      float64 // Band-pass resonance
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
		LayerFX:          "add(12).gain(0.5)", // Octave layer for fullness
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
		EchoWithFX:       "add(12).gain(0.6)", // Octave echo for space
		EchoWithTimes:    3,        // 3 echo iterations
		UseRangex:        true,     // Exponential filter sweeps
		UseOrbit:         true,     // Separate orbits for sidechain
		FilterType:       "ladder", // Analog ladder filter
		UseRing:          true,     // Ring modulation for metallic tones
		RingFreq:         150,      // Ring mod frequency
		UseChorus:        true,     // Chorus for width
		ChorusAmount:     0.4,
		UseShape:         true,     // Saturation for punch
		ShapeAmount:      0.25,
		UsePitchEnv:      true,     // Pitch envelope for bass punch
		PitchEnvAmount:   -12,      // Drop pitch at start
	},
	StyleJazz: {
		ModulationAmount: 0.6,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.1,
		LFOShape:         LFOPerlin, // Organic movement
		UseEnvelope:      true,      // Smooth envelope for jazz
		UseStyleFX:       true,      // Enable vibrato, leslie
		SwingAmount:      0.12,      // Classic jazz swing
		LegatoAmount:     1.1,       // Slightly sustained
		UseEcho:          true,      // Jazz echo for space
		UseSuperimpose:   true,      // Slight detune for warmth
		UseOff:           true,      // Harmonic layering for rich chords
		UseTremolo:       true,      // Subtle tremolo for vibes
		UseFilterEnv:     true,      // Dynamic filter for expression
		UseJux:           true,      // Stereo width
		SometimesFX:      "add(7)",  // Sometimes add a fifth
		RarelyFX:         "speed(0.5).room(0.7)", // Rarely slow + reverb
		AccentPattern:    "backbeat", // Accent 2 and 4 (jazz feel)
		AccentAmount:     0.12,      // Moderate accent
		DynamicRange:     1.5,       // Wide dynamics for expression
		UseLeslie:        true,      // Leslie speaker for organ feel
		LeslieSpeed:      2.5,       // Moderate rotation
		UseChorus:        true,      // Chorus for warmth
		ChorusAmount:     0.35,      // Moderate chorus
		UseShape:         true,      // Subtle warmth
		ShapeAmount:      0.1,       // Very subtle saturation
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
	// Raw synthesizer styles using basic waveforms
	StyleRaw: {
		ModulationAmount: 0.8,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.5,        // Less reverb to hear raw tone
		LFOShape:         LFOSaw,     // Rhythmic filter sweeps
		UseEnvelope:      true,       // ADSR essential for raw oscillators
		UseStyleFX:       true,       // Enable phaser, etc.
		SwingAmount:      0,
		LegatoAmount:     0.9,        // Slightly punchy
		UseEcho:          true,
		UseSuperimpose:   true,       // Detune for thickness
		UseOff:           true,       // Octave layering
		UseTremolo:       false,
		UseFilterEnv:     true,       // Filter envelope for movement
		AccentPattern:    "downbeat",
		AccentAmount:     0.15,
		DynamicRange:     1.2,
	},
	StyleChiptune: {
		ModulationAmount: 0.3,        // Subtle modulation
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     0.3,        // Minimal reverb for clarity
		LFOShape:         LFOSquare,  // Square LFO for chiptune feel
		UseEnvelope:      true,       // Quick attack/decay
		UseStyleFX:       true,       // Enable crush for 8-bit feel
		SwingAmount:      0,
		LegatoAmount:     0.7,        // Short, punchy notes
		UseEcho:          true,       // Arpeggio-style echo
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       false,
		UseFilterEnv:     false,
		IterAmount:       2,          // Pattern cycling for variation
		AccentPattern:    "all-fours",
		AccentAmount:     0.1,
		DynamicRange:     0.9,        // Tight dynamics like 8-bit
	},
	StyleAmbient: {
		ModulationAmount: 0.6,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.5,        // Heavy reverb for space
		LFOShape:         LFOSine,    // Smooth modulation
		UseEnvelope:      true,       // Long attack/release
		UseStyleFX:       true,       // Subtle vibrato
		SwingAmount:      0,
		LegatoAmount:     2.0,        // Long sustained notes
		UseEcho:          true,       // Long echo trails
		UseSuperimpose:   true,       // Layered harmonics
		UseOff:           true,       // Fifth/octave layering
		UseTremolo:       true,       // Slow amplitude modulation
		UseFilterEnv:     false,
		SometimesFX:      "room(0.8)", // Sometimes more reverb
		RarelyFX:         "delay(0.5)", // Rarely extra delay
		AccentPattern:    "",         // No accents for smooth feel
		AccentAmount:     0,
		DynamicRange:     1.3,        // Wide dynamics for expression
	},
	StyleDrone: {
		ModulationAmount: 0.4,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.3,        // Lots of reverb
		LFOShape:         LFOPerlin,  // Organic slow movement
		UseEnvelope:      true,       // Very long attack/release
		UseStyleFX:       true,       // Subtle vibrato
		SwingAmount:      0,
		LegatoAmount:     3.0,        // Very long sustained notes
		UseEcho:          true,
		UseSuperimpose:   true,       // Thick layered sound
		UseOff:           true,       // Harmonic layering
		UseTremolo:       true,       // Slow tremolo
		UseFilterEnv:     true,       // Slow filter sweeps
		SometimesFX:      "lpf(500)", // Sometimes darker
		RarelyFX:         "hpf(200)", // Rarely thinner
		AccentPattern:    "",
		AccentAmount:     0,
		DynamicRange:     1.0,        // Steady dynamics
		UseSlide:         true,       // Slow pitch glide
		SlideAmount:      0.2,        // Very slow slide for drones
	},
	// Sample-based styles
	StyleMallets: {
		ModulationAmount: 0.3,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.0,        // Natural room reverb
		LFOShape:         LFOSine,    // Subtle movement
		UseEnvelope:      false,      // Samples have natural envelope
		UseStyleFX:       true,       // Subtle vibrato
		SwingAmount:      0,
		LegatoAmount:     0.8,        // Slightly shorter for mallet attack
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       true,       // Subtle tremolo for vibes effect
		UseFilterEnv:     false,
		AccentPattern:    "downbeat",
		AccentAmount:     0.12,
		DynamicRange:     1.3,        // Wide dynamics for expression
	},
	StylePlucked: {
		ModulationAmount: 0.2,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.1,        // Soft reverb
		LFOShape:         LFOSine,    // Gentle
		UseEnvelope:      false,      // Natural pluck envelope
		UseStyleFX:       false,
		SwingAmount:      0,
		LegatoAmount:     0.7,        // Short plucked notes
		UseEcho:          true,       // Gentle echo
		UseSuperimpose:   false,
		UseOff:           true,       // Octave doubling for harp
		UseTremolo:       false,
		UseFilterEnv:     false,
		AccentPattern:    "",
		AccentAmount:     0,
		DynamicRange:     1.4,        // Very expressive
	},
	StyleKeys: {
		ModulationAmount: 0.1,
		FilterDynamic:    false,
		DelayEnabled:     false,
		ReverbAmount:     1.0,        // Concert hall reverb
		LFOShape:         LFOSine,
		UseEnvelope:      false,      // Piano has natural envelope
		UseStyleFX:       false,      // Pure piano sound
		SwingAmount:      0,
		LegatoAmount:     1.0,        // Natural sustain
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       false,
		UseFilterEnv:     false,
		AccentPattern:    "downbeat",
		AccentAmount:     0.1,
		DynamicRange:     1.5,        // Wide dynamics like real piano
	},
	StylePad: {
		ModulationAmount: 0.5,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.4,        // Lush reverb
		LFOShape:         LFOSine,    // Smooth
		UseEnvelope:      true,       // Long envelopes
		UseStyleFX:       true,       // Subtle vibrato
		SwingAmount:      0,
		LegatoAmount:     2.0,        // Long sustained
		UseEcho:          true,
		UseSuperimpose:   true,       // Layered pads
		UseOff:           true,       // Harmonic thickness
		UseTremolo:       true,       // Slow movement
		UseFilterEnv:     false,
		SometimesFX:      "room(0.7)",
		AccentPattern:    "",
		AccentAmount:     0,
		DynamicRange:     1.1,        // Gentle dynamics
	},
	StylePercussive: {
		ModulationAmount: 0.2,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.2,        // Concert hall
		LFOShape:         LFOSine,
		UseEnvelope:      false,      // Natural percussion envelope
		UseStyleFX:       false,
		SwingAmount:      0,
		LegatoAmount:     0.6,        // Short, punchy
		UseEcho:          true,       // Timpani rolls
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       false,
		UseFilterEnv:     false,
		AccentPattern:    "downbeat",
		AccentAmount:     0.2,        // Strong accents
		DynamicRange:     1.6,        // Very wide dynamics
	},
	// Genre-specific styles
	StyleSynthwave: {
		ModulationAmount: 0.8,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.9,        // Gated reverb feel
		LFOShape:         LFOSaw,     // Rhythmic sweeps
		UseEnvelope:      true,       // Punchy envelopes
		UseStyleFX:       true,       // Chorus, phaser
		SwingAmount:      0,
		LegatoAmount:     0.9,
		UseEcho:          true,       // 80s delay
		UseSuperimpose:   true,       // Thick layering
		UseOff:           true,       // Octave stacking
		UseTremolo:       false,
		UseFilterEnv:     true,       // Classic filter sweeps
		UseDuck:          true,       // Sidechain pumping
		AccentPattern:    "downbeat",
		AccentAmount:     0.15,
		DynamicRange:     1.2,
		UseCompressor:    true,       // Punchy compression
		FilterType:       "ladder",   // Analog ladder filter
		UseSlide:         true,       // Portamento for smooth bass
		SlideAmount:      0.1,        // Subtle slide
	},
	StyleDarkwave: {
		ModulationAmount: 0.6,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.4,        // Dark, deep reverb
		LFOShape:         LFOPerlin,  // Organic movement
		UseEnvelope:      true,
		UseStyleFX:       true,       // Phaser, distortion
		SwingAmount:      0,
		LegatoAmount:     1.2,        // Sustained, moody
		UseEcho:          true,
		UseSuperimpose:   true,
		UseOff:           true,
		UseTremolo:       true,       // Slow tremolo
		UseFilterEnv:     true,
		SometimesFX:      "lpf(600)", // Sometimes darker
		RarelyFX:         "distort(0.2)",
		AccentPattern:    "",
		AccentAmount:     0,
		DynamicRange:     1.1,
		UseSlide:         true,       // Moody portamento
		SlideAmount:      0.15,       // Slower slide for atmosphere
		FilterType:       "ladder",   // Dark analog filter
	},
	StyleMinimal: {
		ModulationAmount: 0.2,        // Very subtle
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     0.6,        // Clean, sparse reverb
		LFOShape:         LFOSine,    // Simple
		UseEnvelope:      true,
		UseStyleFX:       false,      // No extra effects
		SwingAmount:      0,
		LegatoAmount:     0.8,
		UseEcho:          false,
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       false,
		UseFilterEnv:     false,
		AccentPattern:    "all-fours",
		AccentAmount:     0.08,
		DynamicRange:     0.9,        // Tight dynamics
	},
	StyleIndustrial: {
		ModulationAmount: 1.0,        // Full modulation
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.5,        // Tight, metallic
		LFOShape:         LFOSquare,  // Harsh stepping
		UseEnvelope:      true,       // Punchy
		UseStyleFX:       true,       // Heavy distortion
		SwingAmount:      0,
		LegatoAmount:     0.6,        // Short, aggressive
		UseEcho:          true,
		UseSuperimpose:   false,
		UseOff:           false,
		UseTremolo:       false,
		UseFilterEnv:     true,       // Aggressive sweeps
		UseCompressor:    true,       // Crushed dynamics
		AccentPattern:    "all-fours",
		AccentAmount:     0.2,
		DynamicRange:     0.7,        // Compressed
		FilterType:       "ladder",   // Aggressive ladder filter
	},
	StyleNewAge: {
		ModulationAmount: 0.3,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.6,        // Huge reverb
		LFOShape:         LFOSine,    // Smooth
		UseEnvelope:      true,       // Long, soft envelopes
		UseStyleFX:       true,       // Subtle vibrato
		SwingAmount:      0,
		LegatoAmount:     2.5,        // Very sustained
		UseEcho:          true,       // Long echo trails
		UseSuperimpose:   true,       // Layered
		UseOff:           true,       // Harmonic richness
		UseTremolo:       true,       // Slow breathing
		UseFilterEnv:     false,
		SometimesFX:      "room(0.8)",
		AccentPattern:    "",
		AccentAmount:     0,
		DynamicRange:     1.2,
	},
	// Noise and texture styles
	StyleNoise: {
		ModulationAmount: 0.2,
		FilterDynamic:    true,       // Filter noise dynamically
		DelayEnabled:     true,
		ReverbAmount:     1.5,        // Lots of reverb
		LFOShape:         LFOPerlin,  // Organic noise movement
		UseEnvelope:      true,
		UseStyleFX:       false,
		LegatoAmount:     2.0,
		DynamicRange:     0.8,        // Keep consistent
	},
	StyleGlitch: {
		ModulationAmount: 1.0,        // Heavy modulation
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.5,
		LFOShape:         LFORand,    // Random modulation
		UseEnvelope:      true,
		UseStyleFX:       true,       // Crush, coarse
		LegatoAmount:     0.5,        // Short, choppy
		IterAmount:       4,          // Pattern variation
		SometimesFX:      "crush(4)",
		RarelyFX:         "speed(-1)",
		DynamicRange:     0.7,
	},
	StyleTexture: {
		ModulationAmount: 0.4,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.8,        // Deep reverb for texture
		LFOShape:         LFOPerlin,
		UseEnvelope:      true,
		UseStyleFX:       false,
		LegatoAmount:     3.0,        // Very long sustain
		UseSuperimpose:   true,
		DynamicRange:     1.0,
	},
	StyleRetro: {
		ModulationAmount: 0.6,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     0.4,        // Tight reverb
		LFOShape:         LFOSquare,  // 8-bit style
		UseEnvelope:      true,
		UseStyleFX:       true,       // Crush for retro
		LegatoAmount:     0.7,
		AccentPattern:    "downbeat",
		AccentAmount:     0.15,
		DynamicRange:     0.9,
	},
	// Dance music styles
	StyleHouse: {
		ModulationAmount: 0.9,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.8,
		LFOShape:         LFOSaw,     // Rhythmic sweeps
		UseEnvelope:      true,
		UseStyleFX:       true,
		LegatoAmount:     0.8,
		UseEcho:          true,       // House echo
		UseSuperimpose:   true,       // Detune for width
		UseOff:           true,       // Octave layers
		UseTremolo:       true,       // Pumping tremolo
		UseDuck:          true,       // Sidechain pumping
		UseFilterEnv:     true,
		UseJux:           true,       // Stereo width
		AccentPattern:    "all-fours",
		AccentAmount:     0.15,
		UseCompressor:    true,
		DynamicRange:     0.8,
		FilterType:       "ladder",
		UseOrbit:         true,
		SometimesFX:      "speed(2).lpf(2000)", // Sometimes octave up filtered
		RarelyFX:         "crush(10)",          // Rarely bitcrush
		UseShape:         true,       // Punchy saturation
		ShapeAmount:      0.2,
		UseChorus:        true,
		ChorusAmount:     0.35,
		PlyAmount:        2,          // Double bass hits
	},
	StyleTrance: {
		ModulationAmount: 1.0,        // Heavy modulation
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.1,
		LFOShape:         LFOSaw,
		UseEnvelope:      true,
		UseStyleFX:       true,
		LegatoAmount:     1.0,
		UseEcho:          true,       // Trance echo
		UseFilterEnv:     true,       // Classic trance filter
		UseSuperimpose:   true,       // Supersaw-style
		UseOff:           true,       // Octave layers for supersaws
		UseTremolo:       true,       // Gated tremolo
		UseJux:           true,       // Wide stereo
		AccentPattern:    "all-fours",
		AccentAmount:     0.12,
		UseCompressor:    true,
		DynamicRange:     1.0,
		FilterType:       "ladder",
		UseChorus:        true,
		ChorusAmount:     0.5,        // Heavy chorus for supersaws
		SometimesFX:      "add(12).lpf(4000)", // Sometimes octave up
		RarelyFX:         "rev.room(0.8)",     // Rarely reverse + reverb
		UseShape:         true,
		ShapeAmount:      0.15,
		UsePitchEnv:      true,       // Pitch drop on bass
		PitchEnvAmount:   -24,        // 2 octave drop
	},
	StyleDub: {
		ModulationAmount: 0.5,
		FilterDynamic:    true,
		DelayEnabled:     true,       // Heavy delay for dub
		ReverbAmount:     1.2,        // Spring reverb feel
		LFOShape:         LFOSine,
		UseEnvelope:      false,
		UseStyleFX:       true,
		SwingAmount:      0.08,       // Reggae swing
		LegatoAmount:     1.2,
		UseEcho:          true,       // Dub delay
		AccentPattern:    "offbeat",  // Reggae offbeat
		AccentAmount:     0.15,
		DynamicRange:     1.3,
	},
	StyleFunk: {
		ModulationAmount: 0.6,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     0.7,
		LFOShape:         LFOSaw,     // Funky filter sweeps
		UseEnvelope:      true,       // Punchy envelope
		UseStyleFX:       true,       // Phaser, wah-like
		SwingAmount:      0.15,       // Heavy funky swing
		LegatoAmount:     0.7,        // Tight, punchy
		UseEcho:          true,       // Funky echo
		UseFilterEnv:     true,       // Wah-like filter
		UseJux:           true,       // Stereo funk
		AccentPattern:    "backbeat",
		AccentAmount:     0.25,       // Strong accents
		DynamicRange:     1.5,        // Wide dynamics
		SometimesFX:      "speed(2)",  // Sometimes octave up
		RarelyFX:         "crush(12).coarse(2)", // Rarely lo-fi
		UseShape:         true,       // Funky saturation
		ShapeAmount:      0.2,
		PlyAmount:        2,          // Double bass notes
	},
	StyleSoul: {
		ModulationAmount: 0.5,
		FilterDynamic:    true,
		DelayEnabled:     true,
		ReverbAmount:     1.1,        // Warm reverb
		LFOShape:         LFOSine,
		UseEnvelope:      true,       // Smooth soul envelope
		UseStyleFX:       true,       // Vibrato, phaser, chorus
		SwingAmount:      0.08,       // Subtle groove
		LegatoAmount:     1.1,        // Slightly sustained
		UseEcho:          true,       // Echo for depth
		UseSuperimpose:   true,       // Slight detune for warmth
		UseOff:           true,       // Harmonic layering
		UseTremolo:       true,       // Soul tremolo
		UseFilterEnv:     true,       // Dynamic wah-like filter
		UseJux:           true,       // Stereo width
		AccentPattern:    "backbeat",
		AccentAmount:     0.15,       // Soul backbeat
		DynamicRange:     1.4,        // Wide dynamics
		SometimesFX:      "add(7).gain(0.4)", // Sometimes add a fifth
		RarelyFX:         "speed(2).lpf(1500)", // Rarely octave up filtered
		UseChorus:        true,       // Warm chorus
		ChorusAmount:     0.4,
		UseShape:         true,       // Warm saturation
		ShapeAmount:      0.12,
		UseLeslie:        true,       // Leslie for organ sounds
		LeslieSpeed:      2,
	},
	StyleCinematic: {
		ModulationAmount: 0.2,
		FilterDynamic:    false,
		DelayEnabled:     true,
		ReverbAmount:     1.5,        // Big hall reverb
		LFOShape:         LFOSine,
		UseEnvelope:      true,       // Long cinematic envelopes
		UseStyleFX:       true,
		LegatoAmount:     2.0,        // Long sustained notes
		UseSuperimpose:   true,       // Layered strings
		UseOff:           true,       // Octave doubling
		UseTremolo:       true,       // Subtle movement
		AccentPattern:    "downbeat",
		AccentAmount:     0.15,
		DynamicRange:     1.6,        // Very wide dynamics
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
		Chorus:       0.3,  // Subtle chorus for width
	},
	StyleOrchestral: {
		Vibrato:      5,
		VibratoDepth: 0.15,
		Chorus:       0.2,  // Subtle ensemble chorus
	},
	StyleElectronic: {
		Phaser:       0.8,
		PhaserDepth:  0.5,
		Distort:      0.15,
		Shape:        0.2,   // Punchy saturation
		FM:           2.5,   // Strong FM for brightness
		FMH:          2,     // Octave harmonicity
		FMDecay:      0.15,
		FMSustain:    0.2,
		Ring:         150,   // Ring mod for metallic edge
		Chorus:       0.4,   // Wide chorus
		Vibrato:      5,
		VibratoDepth: 0.15,
	},
	StyleJazz: {
		Vibrato:      4,
		VibratoDepth: 0.1,
		Leslie:       0.5,      // Noticeable Leslie on organ sounds
		LeslieRate:   3,        // Medium rotation
		Phaser:       0.3,      // Subtle phaser for richness
		PhaserDepth:  0.25,
		Chorus:       0.3,      // Warm chorus
		FM:           0.8,      // Subtle FM for harmonic richness
		FMH:          1,        // Unison
		FMDecay:      0.5,
		FMSustain:    0.7,
		Shape:        0.08,     // Very subtle warmth
	},
	StyleLofi: {
		Crush:  10, // Subtle bit reduction
		Coarse: 4,  // Sample rate reduction
		Shape:  0.1, // Gentle saturation for warmth
	},
	// Raw synthesizer styles
	StyleRaw: {
		Phaser:       0.6,
		PhaserDepth:  0.4,
		FM:           2.0,  // FM for harmonic richness
		FMH:          1.5,
		FMDecay:      0.4,
		FMSustain:    0.6,
		Shape:        0.2,  // Subtle waveshaping
	},
	StyleChiptune: {
		Crush:  8,    // 8-bit reduction
		Coarse: 8,    // Lower sample rate for retro feel
	},
	StyleAmbient: {
		Vibrato:      2,    // Slow subtle vibrato
		VibratoDepth: 0.05,
		Chorus:       0.4,  // Lush chorus
	},
	StyleDrone: {
		Vibrato:      1,    // Very slow vibrato
		VibratoDepth: 0.08,
		Phaser:       0.2,  // Slow phaser
		PhaserDepth:  0.3,
		Chorus:       0.5,  // Thick chorus for drones
	},
	// Sample-based styles
	StyleMallets: {
		Vibrato:      4,    // Motor-driven vibraphone effect
		VibratoDepth: 0.1,
		Leslie:       0.4,  // Leslie for vibraphone motor effect
		LeslieRate:   4,    // Moderate rotation
	},
	StylePlucked: {},       // Pure plucked sound
	StyleKeys: {
		Leslie:     0.5,    // Classic organ Leslie
		LeslieRate: 3,
	},
	StylePad: {
		Vibrato:      2,    // Slow subtle vibrato
		VibratoDepth: 0.05,
		Chorus:       0.5,  // Lush pad chorus
	},
	StylePercussive: {},    // Pure percussion
	// Genre-specific styles
	StyleSynthwave: {
		Phaser:       0.6,
		PhaserDepth:  0.4,
		Vibrato:      5,
		VibratoDepth: 0.12,
		FM:           1.8,
		FMH:          2,
		FMDecay:      0.3,
		FMSustain:    0.5,
		Chorus:       0.4,  // 80s chorus sound
		Shape:        0.15, // Subtle saturation
	},
	StyleDarkwave: {
		Phaser:       0.4,
		PhaserDepth:  0.5,
		Distort:      0.15,
		Vibrato:      3,
		VibratoDepth: 0.1,
		Chorus:       0.3,  // Dark chorus
		Ring:         150,  // Subtle ring mod for dissonance
	},
	StyleMinimal: {},       // Clean, no effects
	StyleIndustrial: {
		Distort:     0.3,   // Heavy distortion
		Crush:       6,     // Bit crushing
		Coarse:      4,     // Sample reduction
		Shape:       0.4,   // Heavy waveshaping
		Ring:        300,   // Harsh ring mod
	},
	StyleNewAge: {
		Vibrato:      2,
		VibratoDepth: 0.05,
		Chorus:       0.4,  // Ethereal chorus
	},
	// Noise and texture styles
	StyleNoise:   {},       // Pure noise - no extra FX
	StyleGlitch: {
		Crush:  4,          // Extreme bit crush
		Coarse: 8,          // Heavy sample reduction
		Shape:  0.3,        // Distortion
	},
	StyleTexture: {
		Chorus: 0.5,        // Wide chorus for texture
	},
	StyleRetro: {
		Crush:  8,          // 8-bit
		Coarse: 4,
	},
	// Dance music styles
	StyleHouse: {
		Phaser:      0.3,
		PhaserDepth: 0.3,
		Shape:       0.1,   // Subtle saturation
	},
	StyleTrance: {
		Phaser:       0.5,
		PhaserDepth:  0.4,
		FM:           1.5,  // FM for supersaw-like timbres
		FMH:          1,
		Chorus:       0.5,  // Wide supersaw chorus
	},
	StyleDub: {
		Vibrato:      2,
		VibratoDepth: 0.05,
	},
	StyleFunk: {
		Phaser:       0.4,      // Funky phaser
		PhaserDepth:  0.35,
		Shape:        0.15,     // Punchy saturation
		Vibrato:      3,
		VibratoDepth: 0.08,
	},
	StyleSoul: {
		Vibrato:      4,
		VibratoDepth: 0.1,
		Phaser:       0.3,      // Subtle soul phaser
		PhaserDepth:  0.25,
		Chorus:       0.35,     // Warm chorus
		Leslie:       0.4,      // Leslie for organ
		LeslieRate:   2.5,
		Shape:        0.1,      // Warm saturation
		FM:           0.6,      // Subtle FM warmth
		FMH:          1,
		FMDecay:      0.4,
		FMSustain:    0.6,
	},
	StyleCinematic: {
		Vibrato:      4,
		VibratoDepth: 0.1,
		Chorus:       0.3,  // Ensemble width
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
	// Don't use jux if we're also using pan LFO modulation - they conflict
	// (jux creates stereo width by reversing right channel, pan LFO already provides movement)
	if styleMod.UseJux && voice == "high" && effects.Pan.Width == 0 {
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

	// Apply layer for styles that use it
	if styleMod.LayerFX != "" {
		effects.Harmony.Layer = styleMod.LayerFX
	}

	// Apply echoWith for styles that use it
	if styleMod.EchoWithFX != "" && styleMod.EchoWithTimes > 0 {
		effects.Harmony.EchoWith = styleMod.EchoWithFX
		effects.Harmony.EchoWithTimes = styleMod.EchoWithTimes
		effects.Harmony.EchoWithTime = 0.125 // Default 1/8 note
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

	// Apply filter type for styles that specify it
	if styleMod.FilterType != "" {
		effects.FilterType = FilterTypeSettings{
			Type: styleMod.FilterType,
		}
	}

	// Apply slide/portamento for styles that use it (mainly on bass)
	if styleMod.UseSlide && voice == "bass" {
		effects.Slide = SlideSettings{
			Slide: styleMod.SlideAmount,
		}
	}

	// Apply orbit routing for styles that use it (separates voices for sidechain)
	if styleMod.UseOrbit {
		// Assign different orbits per voice for independent effects
		orbitNum := 0
		switch voice {
		case "bass":
			orbitNum = 1 // Bass on orbit 1 (sidechain trigger)
		case "mid":
			orbitNum = 2 // Mid on orbit 2 (ducks against bass)
		case "high":
			orbitNum = 3 // High on orbit 3 (ducks against bass)
		}
		effects.Orbit = OrbitSettings{
			Orbit: orbitNum,
		}
	}

	// Apply waveshaping for styles that use it
	if styleMod.UseShape && styleMod.ShapeAmount > 0 {
		effects.Shape = ShapeSettings{
			Shape:    styleMod.ShapeAmount,
			Postgain: -styleMod.ShapeAmount * 3, // Compensate for volume increase
		}
	}

	// Apply ring modulation for styles that use it (mainly on high voice for leads)
	if styleMod.UseRing && styleMod.RingFreq > 0 && voice == "high" {
		effects.Ring = RingSettings{
			Ring: styleMod.RingFreq,
		}
	}

	// Apply chorus for styles that use it (mainly on mid/high voices)
	if styleMod.UseChorus && styleMod.ChorusAmount > 0 && voice != "bass" {
		effects.Chorus = ChorusSettings{
			Chorus: styleMod.ChorusAmount,
		}
	}

	// Apply Leslie speaker for styles that use it
	if styleMod.UseLeslie && styleMod.LeslieSpeed > 0 {
		effects.Leslie = LeslieSettings{
			Leslie: 1.0, // Full wet
			LRate:  styleMod.LeslieSpeed,
		}
	}

	// Apply pitch envelope for styles that use it (mainly on bass for punch)
	if styleMod.UsePitchEnv && voice == "bass" {
		effects.PitchEnv = PitchEnvSettings{
			PAttack:  0.001,
			PDecay:   0.1,
			PRelease: 0.05,
			PEnv:     styleMod.PitchEnvAmount,
		}
	}

	// Apply band-pass filter for styles that use it
	if styleMod.UseBandPass && styleMod.BandPassFreq > 0 {
		effects.BandPass = BandPassSettings{
			BPF: styleMod.BandPassFreq,
			BPQ: styleMod.BandPassQ,
		}
	}

	// Copy new StyleFX settings to effect structs
	if fx, ok := styleFXPresets[style]; ok {
		if fx.Shape > 0 && effects.Shape.Shape == 0 {
			effects.Shape.Shape = fx.Shape
		}
		if fx.Ring > 0 && effects.Ring.Ring == 0 && voice == "high" {
			effects.Ring.Ring = fx.Ring
		}
		if fx.Chorus > 0 && effects.Chorus.Chorus == 0 && voice != "bass" {
			effects.Chorus.Chorus = fx.Chorus
		}
		if fx.Leslie > 0 && effects.Leslie.Leslie == 0 {
			effects.Leslie.Leslie = fx.Leslie
			effects.Leslie.LRate = fx.LeslieRate
		}
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
	case StyleAmbient:
		return TremoloSettings{
			Sync:  16,     // Very slow
			Depth: 0.25,   // Gentle pulsing
			Shape: "sine", // Smooth
		}
	case StyleDrone:
		return TremoloSettings{
			Sync:  32,     // Extremely slow
			Depth: 0.2,    // Subtle movement
			Shape: "sine", // Smooth
		}
	case StyleMallets:
		return TremoloSettings{
			Sync:  6,      // Vibraphone motor speed
			Depth: 0.3,    // Noticeable vibrato
			Shape: "sine", // Smooth like real vibes
		}
	case StylePad:
		return TremoloSettings{
			Sync:  12,     // Slow
			Depth: 0.15,   // Subtle
			Shape: "sine", // Smooth
		}
	case StyleDarkwave:
		return TremoloSettings{
			Sync:  8,      // Moody
			Depth: 0.25,   // Noticeable
			Shape: "sine", // Smooth
		}
	case StyleNewAge:
		return TremoloSettings{
			Sync:  16,     // Very slow
			Depth: 0.2,    // Gentle breathing
			Shape: "sine", // Smooth
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
	case StyleRaw:
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.01,
				Decay:   0.2,
				Sustain: 0.5,
				Release: 0.15,
				Amount:  3000, // Wide sweep for raw bass
			}
		}
		return FilterEnvSettings{
			Attack:  0.02,
			Decay:   0.3,
			Sustain: 0.6,
			Release: 0.25,
			Amount:  4000, // Bright sweep
		}
	case StyleDrone:
		// Very slow filter movements for drones
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  1.0,
				Decay:   3.0,
				Sustain: 0.6,
				Release: 2.0,
				Amount:  1500, // Subtle bass sweep
			}
		}
		return FilterEnvSettings{
			Attack:  2.0,
			Decay:   4.0,
			Sustain: 0.7,
			Release: 3.0,
			Amount:  2000, // Slow wide sweep
		}
	case StyleSynthwave:
		// Classic 80s filter sweep
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.01,
				Decay:   0.25,
				Sustain: 0.4,
				Release: 0.2,
				Amount:  3000, // Big bass sweep
			}
		}
		return FilterEnvSettings{
			Attack:  0.02,
			Decay:   0.3,
			Sustain: 0.5,
			Release: 0.25,
			Amount:  4000, // Bright sweep
		}
	case StyleDarkwave:
		// Dark, slow filter movement
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.1,
				Decay:   0.6,
				Sustain: 0.3,
				Release: 0.5,
				Amount:  1500, // Subtle dark sweep
			}
		}
		return FilterEnvSettings{
			Attack:  0.2,
			Decay:   0.8,
			Sustain: 0.4,
			Release: 0.6,
			Amount:  2000,
		}
	case StyleIndustrial:
		// Aggressive, harsh filter
		if voice == "bass" {
			return FilterEnvSettings{
				Attack:  0.001,
				Decay:   0.1,
				Sustain: 0.2,
				Release: 0.05,
				Amount:  4000, // Aggressive sweep
			}
		}
		return FilterEnvSettings{
			Attack:  0.005,
			Decay:   0.15,
			Sustain: 0.3,
			Release: 0.1,
			Amount:  5000, // Very aggressive
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
	case LFOCosine:
		position = fmt.Sprintf("cosine.range(%.2f,%.2f).slow(%.0f)", minPan, maxPan, pan.Speed)
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
	case StyleRaw:
		// Clean punchy envelopes for raw oscillators
		env.Attack = 0.005
		env.Decay = 0.1
		env.Sustain = 0.7
		env.Release = 0.2
	case StyleChiptune:
		// Very short, punchy 8-bit style
		env.Attack = 0.001
		env.Decay = 0.05
		env.Sustain = 0.6
		env.Release = 0.1
	case StyleAmbient:
		// Long, slow envelopes for pads
		env.Attack = 0.5
		env.Decay = 1.0
		env.Sustain = 0.8
		env.Release = 2.0
	case StyleDrone:
		// Very long, sustained envelopes
		env.Attack = 1.0
		env.Decay = 2.0
		env.Sustain = 0.9
		env.Release = 3.0
	case StylePad:
		// Long pad envelopes
		env.Attack = 0.3
		env.Decay = 0.8
		env.Sustain = 0.85
		env.Release = 1.5
	case StyleSynthwave:
		// Punchy 80s envelopes
		env.Attack = 0.01
		env.Decay = 0.15
		env.Sustain = 0.7
		env.Release = 0.3
	case StyleDarkwave:
		// Moody, sustained
		env.Attack = 0.1
		env.Decay = 0.5
		env.Sustain = 0.75
		env.Release = 0.8
	case StyleMinimal:
		// Clean, simple
		env.Attack = 0.01
		env.Decay = 0.1
		env.Sustain = 0.6
		env.Release = 0.2
	case StyleIndustrial:
		// Aggressive, punchy
		env.Attack = 0.001
		env.Decay = 0.05
		env.Sustain = 0.5
		env.Release = 0.1
	case StyleNewAge:
		// Very soft, long
		env.Attack = 0.8
		env.Decay = 1.5
		env.Sustain = 0.9
		env.Release = 2.5
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
// Each method is placed on its own line for readability
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

	// Filter envelope (for synth/electronic styles) - each param on own line
	if effects.FilterEnv.Amount > 0 {
		parts = append(parts, fmt.Sprintf(".lpattack(%.3f)", effects.FilterEnv.Attack))
		parts = append(parts, fmt.Sprintf(".lpdecay(%.2f)", effects.FilterEnv.Decay))
		parts = append(parts, fmt.Sprintf(".lpsustain(%.2f)", effects.FilterEnv.Sustain))
		parts = append(parts, fmt.Sprintf(".lprelease(%.2f)", effects.FilterEnv.Release))
		parts = append(parts, fmt.Sprintf(".lpenv(%.0f)", effects.FilterEnv.Amount))
	}

	// ADSR Envelope - each param on own line
	if effects.Envelope.Attack > 0 || effects.Envelope.Release > 0 {
		parts = append(parts, fmt.Sprintf(".attack(%.3f)", effects.Envelope.Attack))
		parts = append(parts, fmt.Sprintf(".decay(%.2f)", effects.Envelope.Decay))
		parts = append(parts, fmt.Sprintf(".sustain(%.2f)", effects.Envelope.Sustain))
		parts = append(parts, fmt.Sprintf(".release(%.2f)", effects.Envelope.Release))
	}

	// Style-specific effects
	if effects.StyleFX.Vibrato > 0 {
		parts = append(parts, fmt.Sprintf(".vib(%.1f)", effects.StyleFX.Vibrato))
		parts = append(parts, fmt.Sprintf(".vibmod(%.2f)", effects.StyleFX.VibratoDepth))
	}
	if effects.StyleFX.Phaser > 0 {
		parts = append(parts, fmt.Sprintf(".phaser(%.2f)", effects.StyleFX.Phaser))
		parts = append(parts, fmt.Sprintf(".phaserdepth(%.2f)", effects.StyleFX.PhaserDepth))
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

	// FM Synthesis - each param on own line
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
		parts = append(parts, fmt.Sprintf(".tremolo(%.1f)", effects.Tremolo.Sync))
		parts = append(parts, fmt.Sprintf(".tremolodepth(%.2f)", effects.Tremolo.Depth))
		if effects.Tremolo.Shape != "" && effects.Tremolo.Shape != "sine" {
			parts = append(parts, fmt.Sprintf(".tremoloshape(\"%s\")", effects.Tremolo.Shape))
		}
	}

	// Reverb
	if effects.Reverb.Room > 0 {
		parts = append(parts, fmt.Sprintf(".room(%.2f)", effects.Reverb.Room))
		parts = append(parts, fmt.Sprintf(".size(%.2f)", effects.Reverb.Size))
	}

	// Delay
	if effects.Delay.Mix > 0 {
		parts = append(parts, fmt.Sprintf(".delay(%.2f)", effects.Delay.Mix))
		parts = append(parts, fmt.Sprintf(".delaytime(%.3f)", effects.Delay.Time))
		parts = append(parts, fmt.Sprintf(".delayfeedback(%.2f)", effects.Delay.Feedback))
	}

	// Echo/Stutter effect
	if effects.Echo.Times > 0 {
		parts = append(parts, fmt.Sprintf(".echo(%d, %.3f, %.2f)",
			effects.Echo.Times, effects.Echo.Time, effects.Echo.Feedback))
	}

	// Ducking/Sidechain effect
	if effects.Duck.Orbit > 0 {
		parts = append(parts, fmt.Sprintf(".duck(%d)", effects.Duck.Orbit))
		parts = append(parts, fmt.Sprintf(".duckattack(%.2f)", effects.Duck.Attack))
		parts = append(parts, fmt.Sprintf(".duckdepth(%.2f)", effects.Duck.Depth))
	}

	// Compressor for dynamics control
	if effects.Compressor.Enabled {
		parts = append(parts, fmt.Sprintf(".compressor(\"%.0f:%.0f:%.0f:%.3f:%.2f\")",
			effects.Compressor.Threshold, effects.Compressor.Ratio, effects.Compressor.Knee,
			effects.Compressor.Attack, effects.Compressor.Release))
	}

	// Pitch slide/portamento
	if effects.Slide.Slide != 0 {
		parts = append(parts, fmt.Sprintf(".slide(%.2f)", effects.Slide.Slide))
	}
	if effects.Slide.DeltaSlide != 0 {
		parts = append(parts, fmt.Sprintf(".deltaSlide(%.2f)", effects.Slide.DeltaSlide))
	}

	// Filter type selection (ladder filter for analog style)
	if effects.FilterType.Type != "" && effects.FilterType.Type != "12db" {
		parts = append(parts, fmt.Sprintf(".ftype(\"%s\")", effects.FilterType.Type))
	}

	// Band-pass filter
	if effects.BandPass.BPF > 0 {
		parts = append(parts, fmt.Sprintf(".bpf(%d)", effects.BandPass.BPF))
		if effects.BandPass.BPQ > 0 {
			parts = append(parts, fmt.Sprintf(".bpq(%.2f)", effects.BandPass.BPQ))
		}
	}

	// Waveshaping/saturation
	if effects.Shape.Shape > 0 {
		parts = append(parts, fmt.Sprintf(".shape(%.2f)", effects.Shape.Shape))
		if effects.Shape.Postgain != 0 {
			parts = append(parts, fmt.Sprintf(".postgain(%.2f)", effects.Shape.Postgain))
		}
	}

	// Ring modulation
	if effects.Ring.Ring > 0 {
		parts = append(parts, fmt.Sprintf(".ring(%.1f)", effects.Ring.Ring))
	}

	// Chorus effect
	if effects.Chorus.Chorus > 0 {
		parts = append(parts, fmt.Sprintf(".chorus(%.2f)", effects.Chorus.Chorus))
	}

	// Leslie speaker simulation
	if effects.Leslie.Leslie > 0 {
		parts = append(parts, fmt.Sprintf(".leslie(%.2f)", effects.Leslie.Leslie))
		if effects.Leslie.LRate > 0 {
			parts = append(parts, fmt.Sprintf(".lrate(%.1f)", effects.Leslie.LRate))
		}
		if effects.Leslie.LSize > 0 {
			parts = append(parts, fmt.Sprintf(".lsize(%.2f)", effects.Leslie.LSize))
		}
	}

	// Pitch envelope
	if effects.PitchEnv.PEnv != 0 {
		parts = append(parts, fmt.Sprintf(".penv(%.1f)", effects.PitchEnv.PEnv))
		if effects.PitchEnv.PAttack > 0 {
			parts = append(parts, fmt.Sprintf(".pattack(%.3f)", effects.PitchEnv.PAttack))
		}
		if effects.PitchEnv.PDecay > 0 {
			parts = append(parts, fmt.Sprintf(".pdecay(%.2f)", effects.PitchEnv.PDecay))
		}
		if effects.PitchEnv.PRelease > 0 {
			parts = append(parts, fmt.Sprintf(".prelease(%.2f)", effects.PitchEnv.PRelease))
		}
	}

	// Granular effects
	if effects.Granular.Striate > 0 {
		parts = append(parts, fmt.Sprintf(".striate(%d)", effects.Granular.Striate))
	}
	if effects.Granular.Chop > 0 {
		parts = append(parts, fmt.Sprintf(".chop(%d)", effects.Granular.Chop))
	}
	if effects.Granular.Speed != 0 && effects.Granular.Speed != 1.0 {
		parts = append(parts, fmt.Sprintf(".speed(%.2f)", effects.Granular.Speed))
	}
	if effects.Granular.Begin > 0 {
		parts = append(parts, fmt.Sprintf(".begin(%.2f)", effects.Granular.Begin))
	}
	if effects.Granular.End > 0 && effects.Granular.End < 1.0 {
		parts = append(parts, fmt.Sprintf(".end(%.2f)", effects.Granular.End))
	}
	if effects.Granular.Loop {
		parts = append(parts, ".loop(1)")
	}
	if effects.Granular.Cut > 0 {
		parts = append(parts, fmt.Sprintf(".cut(%d)", effects.Granular.Cut))
	}

	// Orbit routing (for separate effect buses per voice)
	if effects.Orbit.Orbit > 0 {
		parts = append(parts, fmt.Sprintf(".orbit(%d)", effects.Orbit.Orbit))
	}

	// Join with newline and indentation for pretty output
	return strings.Join(parts, "\n    ")
}

// BuildPatternTransforms generates pattern-level transformation chain
// Each transform is placed on its own line for readability
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

	// Join with newline and indentation for pretty output
	return strings.Join(parts, "\n    ")
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
// Each effect is placed on its own line for readability
// NOTE: Strudel requires arrow function syntax: x => x.add(n)
func BuildHarmonyEffects(effects VoiceEffects) string {
	var parts []string

	// Superimpose for detuned voices (creates fuller sound)
	// Syntax: .superimpose(x => x.add(0.03)) for slight detuning
	if effects.Harmony.Superimpose > 0 {
		if effects.Harmony.SuperimposeOct != 0 {
			// Detune + octave shift
			parts = append(parts, fmt.Sprintf(".superimpose(x => x.add(%.2f).add(%d))",
				effects.Harmony.Superimpose, effects.Harmony.SuperimposeOct))
		} else {
			// Just slight detuning for chorus/width effect
			parts = append(parts, fmt.Sprintf(".superimpose(x => x.add(%.2f))",
				effects.Harmony.Superimpose))
		}
	}

	// Off for harmonic layering with time offset
	// Syntax: .off(0.125, x => x.add(12)) for octave up with 1/8 note delay
	if effects.Harmony.Off > 0 && effects.Harmony.OffInterval != 0 {
		parts = append(parts, fmt.Sprintf(".off(%.3f, x => x.add(%d))",
			effects.Harmony.Off, effects.Harmony.OffInterval))
	}

	// Layer for parallel transformations (without original)
	if effects.Harmony.Layer != "" {
		parts = append(parts, fmt.Sprintf(".layer(x => x.%s)",
			effects.Harmony.Layer))
	}

	// EchoWith for sophisticated echo with custom function per iteration
	if effects.Harmony.EchoWith != "" && effects.Harmony.EchoWithTimes > 0 {
		parts = append(parts, fmt.Sprintf(".echoWith(%d, %.3f, (p,i) => p.%s.gain(1/(i+1)))",
			effects.Harmony.EchoWithTimes, effects.Harmony.EchoWithTime, effects.Harmony.EchoWith))
	}

	// Join with newline and indentation for pretty output
	return strings.Join(parts, "\n    ")
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
