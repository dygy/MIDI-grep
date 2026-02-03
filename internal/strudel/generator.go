package strudel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/dygy/midi-grep/internal/analysis"
	"github.com/dygy/midi-grep/internal/midi"
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
	StyleHouse     SoundStyle = "house"     // House music sounds
	StyleTrance    SoundStyle = "trance"    // Trance/EDM sounds
	StyleDub       SoundStyle = "dub"       // Reggae/dub sounds
	StyleFunk      SoundStyle = "funk"      // Funk/disco sounds
	StyleSoul      SoundStyle = "soul"      // Soul/R&B sounds
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
		Bass:     "supersaw", // Fat supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_poly", // Poly pad
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
		Bass:     "supersaw", // Supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_sweep", // Sweeping pad
		MidGain:  0.7,
		High:     "gm_lead_5_charang", // Charang lead
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
		Bass:     "z_square", // ZZFX 8-bit bass
		BassGain: 0.7,
		Mid:      "z_square", // ZZFX chiptune
		MidGain:  0.5,
		High:     "z_triangle", // ZZFX NES-style lead
		HighGain: 0.4,
	},
	StyleAmbient: {
		Bass:     "sine", // Pure sine bass
		BassGain: 1.0,
		Mid:      "triangle", // Soft mid tones
		MidGain:  0.7,
		High:     "sine", // Pure sine highs
		HighGain: 0.6,
	},
	StyleDrone: {
		Bass:     "sawtooth", // Rich drone bass
		BassGain: 0.6,
		Mid:      "sine", // Clean mid drone
		MidGain:  0.5,
		High:     "triangle", // Soft high harmonics
		HighGain: 0.4,
	},
	// Sample-based styles using Strudel sample libraries
	StyleMallets: {
		Bass:     "gm_marimba", // Warm marimba bass
		BassGain: 1.1,
		Mid:      "gm_vibraphone", // Classic vibes
		MidGain:  0.9,
		High:     "gm_xylophone", // Bright xylophone
		HighGain: 0.7,
	},
	StylePlucked: {
		Bass:     "gm_acoustic_bass", // Upright bass
		BassGain: 1.2,
		Mid:      "harp", // VCSL harp samples
		MidGain:  0.9,
		High:     "gm_music_box", // Delicate music box
		HighGain: 0.6,
	},
	StyleKeys: {
		Bass:     "piano", // Salamander grand piano
		BassGain: 1.1,
		Mid:      "piano", // Salamander grand piano
		MidGain:  1.0,
		High:     "piano", // Salamander grand piano
		HighGain: 0.9,
	},
	StylePad: {
		Bass:     "gm_pad_warm", // Warm analog pad
		BassGain: 0.8,
		Mid:      "gm_pad_choir", // Choir pad
		MidGain:  0.7,
		High:     "gm_pad_halo", // Ethereal pad
		HighGain: 0.6,
	},
	StylePercussive: {
		Bass:     "timpani", // VCSL timpani
		BassGain: 1.3,
		Mid:      "gm_marimba", // Marimba
		MidGain:  1.0,
		High:     "gm_tubular_bells", // Tubular bells
		HighGain: 0.8,
	},
	// Genre-specific styles
	StyleSynthwave: {
		Bass:     "sawtooth", // Fat analog bass
		BassGain: 0.9,
		Mid:      "gm_pad_poly", // Poly synth pad
		MidGain:  0.7,
		High:     "gm_lead_2_sawtooth", // Bright lead
		HighGain: 0.6,
	},
	StyleDarkwave: {
		Bass:     "sawtooth", // Dark bass
		BassGain: 0.8,
		Mid:      "gm_pad_metallic", // Metallic pad
		MidGain:  0.6,
		High:     "triangle", // Softer high
		HighGain: 0.4,
	},
	StyleMinimal: {
		Bass:     "sine", // Pure sine bass
		BassGain: 0.9,
		Mid:      "triangle", // Clean mid
		MidGain:  0.5,
		High:     "sine", // Pure high
		HighGain: 0.4,
	},
	StyleIndustrial: {
		Bass:     "sawtooth", // Aggressive bass
		BassGain: 1.0,
		Mid:      "square", // Harsh mid
		MidGain:  0.7,
		High:     "sawtooth", // Harsh high
		HighGain: 0.5,
	},
	StyleNewAge: {
		Bass:     "sine", // Pure bass
		BassGain: 0.8,
		Mid:      "gm_pad_warm", // Warm pad
		MidGain:  0.7,
		High:     "gm_pad_halo", // Ethereal
		HighGain: 0.5,
	},
	// Noise-based styles using Strudel's noise oscillators
	StyleNoise: {
		Bass:     "brown", // Soft rumble bass
		BassGain: 0.6,
		Mid:      "pink", // Mid-frequency noise
		MidGain:  0.4,
		High:     "white", // High noise texture
		HighGain: 0.3,
	},
	StyleGlitch: {
		Bass:     "sawtooth", // Digital bass
		BassGain: 0.8,
		Mid:      "square", // Harsh digital
		MidGain:  0.5,
		High:     "crackle", // Glitchy texture
		HighGain: 0.4,
	},
	// Texture styles for layering
	StyleTexture: {
		Bass:     "brown", // Noise bed
		BassGain: 0.4,
		Mid:      "gm_pad_sweep", // Moving pad
		MidGain:  0.5,
		High:     "pink", // Bright texture
		HighGain: 0.3,
	},
	// ZZFX synth styles (more aggressive/retro)
	StyleRetro: {
		Bass:     "z_sawtooth", // ZZFX sawtooth
		BassGain: 0.8,
		Mid:      "z_square", // ZZFX square
		MidGain:  0.6,
		High:     "z_sine", // ZZFX sine
		HighGain: 0.5,
	},
	// House/Techno style
	StyleHouse: {
		Bass:     "supersaw", // Fat supersaw bass
		BassGain: 1.0,
		Mid:      "gm_pad_poly", // Poly stab
		MidGain:  0.8,
		High:     "gm_lead_5_charang", // Charang lead
		HighGain: 0.6,
	},
	// Trance style
	StyleTrance: {
		Bass:     "supersaw", // Supersaw bass
		BassGain: 0.9,
		Mid:      "gm_pad_sweep", // Sweeping pad
		MidGain:  0.7,
		High:     "gm_lead_2_sawtooth", // Saw lead
		HighGain: 0.6,
	},
	// Reggae/Dub style
	StyleDub: {
		Bass:     "gm_synth_bass_2", // Deep dub bass
		BassGain: 1.4,
		Mid:      "gm_epiano1", // Skank keys
		MidGain:  0.8,
		High:     "gm_harmonica", // Melodica/harmonica
		HighGain: 0.6,
	},
	// Funk style
	StyleFunk: {
		Bass:     "gm_slap_bass_1", // Slap bass
		BassGain: 1.2,
		Mid:      "gm_clavinet", // Clavinet
		MidGain:  1.0,
		High:     "gm_brass_section", // Brass stabs
		HighGain: 0.7,
	},
	// Soul/R&B style
	StyleSoul: {
		Bass:     "gm_electric_bass_finger", // Finger bass
		BassGain: 1.1,
		Mid:      "gm_epiano2", // Wurlitzer
		MidGain:  1.0,
		High:     "gm_flute", // Soft flute
		HighGain: 0.7,
	},
	// Cinematic style
	StyleCinematic: {
		Bass:     "gm_contrabass", // Deep strings
		BassGain: 1.2,
		Mid:      "gm_string_ensemble_1", // Full strings
		MidGain:  1.0,
		High:     "gm_choir_aahs", // Choir
		HighGain: 0.8,
	},
}

// AdditionalSounds lists all available sounds from strudel-client
// These can be used with .sound() in Strudel patterns
var AdditionalSounds = map[string][]string{
	// Built-in oscillators (from superdough/synth.mjs)
	"oscillators": {
		"sine", "triangle", "square", "sawtooth",
		"supersaw", "pulse", "sbd", "bytebeat",
	},
	// Noise generators (from superdough/noise.mjs)
	"noise": {
		"white", "pink", "brown", "crackle",
	},
	// ZZFX synths - retro game sounds (from superdough/zzfx.mjs)
	"zzfx": {
		"zzfx", "z_sine", "z_sawtooth", "z_triangle", "z_square", "z_tan", "z_noise",
	},
	// Wavetables (from uzu-wavetables.json)
	"wavetables": {
		"wt_digital", "wt_digital_bad_day", "wt_digital_basique",
		"wt_digital_crickets", "wt_digital_curses", "wt_digital_echoes", "wt_vgame",
	},
	// Salamander Piano samples
	"piano": {"piano"},
	// Dirt samples (from tidalcycles/dirt-samples)
	"dirt": {
		"casio", "crow", "insect", "wind", "jazz", "metal", "east", "space", "numbers", "num",
	},
	// Mridangam - Indian percussion (from mridangam.json)
	"mridangam": {
		"mridangam_ardha", "mridangam_chaapu", "mridangam_dhi", "mridangam_dhin",
		"mridangam_dhum", "mridangam_gumki", "mridangam_ka", "mridangam_ki",
		"mridangam_na", "mridangam_nam", "mridangam_ta", "mridangam_tha", "mridangam_thom",
	},
	// Uzu drumkit (from uzu-drumkit.json)
	"uzu_drums": {
		"bd", "brk", "cb", "cp", "cr", "hh", "ht", "lt", "misc", "mt", "oh", "rd", "rim", "sd", "sh", "tb",
	},
	// VCSL percussion - VSCO Community Sample Library (from vcsl.json)
	"vcsl_percussion": {
		"Agogo Bells", "Anvil", "Ball Whistle", "Bass Drum 1", "Bass Drum 2", "Bongos",
		"Brake Drum", "Cabasa", "Cajon", "Claps", "Clash Cymbals 1", "Clash Cymbals 2",
		"Claves", "Conga", "Cowbells", "Darbuka", "Didgeridoo", "Finger Cymbals",
		"Flexatone", "Frame Drum", "Gong 1", "Gong 2", "Guiro", "Hand Bells 2c Nepalese",
		"Hi Hat Cymbal", "Mark Trees Legacy", "Ocean Drum", "Ratchet", "Shaker 2c Large",
		"Shaker 2c Small", "Siren", "Slapstick", "Sleigh Bells", "Slit Drum",
		"Snare Drum 2c Modern 1", "Snare Drum 2c Modern 2", "Snare Drum 2c Modern 3",
		"Snare Drum 2c Rope Tension", "Snare Drum 2c Rope Tension Hi", "Snare Drum 2c Rope Tension Low",
		"Suspended Cymbal 1", "Suspended Cymbal 2", "Tambourine 1", "Tambourine 2",
		"Timpani 1 Hit", "Timpani 1 Roll", "Timpani 2 Hit", "Tom 1", "Tom 1 Stick",
		"Tom 2", "Tom 2 Mallet", "Tom 2 Stick", "Train Whistle 2c Toy", "Triangles",
		"Vibraslap Legacy", "Woodblock",
	},
	// Classic drum machines (from tidal-drum-machines.json) - Akai
	"drum_machines_akai": {
		"AkaiLinn_bd", "AkaiLinn_cb", "AkaiLinn_cp", "AkaiLinn_cr", "AkaiLinn_hh",
		"AkaiLinn_ht", "AkaiLinn_lt", "AkaiLinn_mt", "AkaiLinn_oh", "AkaiLinn_rd",
		"AkaiLinn_sd", "AkaiLinn_sh", "AkaiLinn_tb",
		"AkaiMPC60_bd", "AkaiMPC60_cp", "AkaiMPC60_cr", "AkaiMPC60_hh", "AkaiMPC60_ht",
		"AkaiMPC60_lt", "AkaiMPC60_misc", "AkaiMPC60_mt", "AkaiMPC60_oh", "AkaiMPC60_perc",
		"AkaiMPC60_rd", "AkaiMPC60_rim", "AkaiMPC60_sd",
		"AkaiXR10_bd", "AkaiXR10_cb", "AkaiXR10_cp", "AkaiXR10_cr", "AkaiXR10_hh",
		"AkaiXR10_ht", "AkaiXR10_lt", "AkaiXR10_misc", "AkaiXR10_mt", "AkaiXR10_oh",
		"AkaiXR10_perc", "AkaiXR10_rd", "AkaiXR10_rim", "AkaiXR10_sd", "AkaiXR10_sh", "AkaiXR10_tb",
	},
	// Roland drum machines (TR-808, TR-909, etc.)
	"drum_machines_roland": {
		"RolandTR505_bd", "RolandTR505_cb", "RolandTR505_cp", "RolandTR505_cr", "RolandTR505_hh",
		"RolandTR505_ht", "RolandTR505_lt", "RolandTR505_mt", "RolandTR505_oh", "RolandTR505_rim",
		"RolandTR505_sd", "RolandTR505_tb",
		"RolandTR606_bd", "RolandTR606_hh", "RolandTR606_ht", "RolandTR606_lt", "RolandTR606_mt",
		"RolandTR606_oh", "RolandTR606_sd",
		"RolandTR626_bd", "RolandTR626_cb", "RolandTR626_cp", "RolandTR626_cr", "RolandTR626_hh",
		"RolandTR626_ht", "RolandTR626_lt", "RolandTR626_mt", "RolandTR626_oh", "RolandTR626_perc",
		"RolandTR626_rd", "RolandTR626_rim", "RolandTR626_sd", "RolandTR626_sh", "RolandTR626_tb",
		"RolandTR707_bd", "RolandTR707_cb", "RolandTR707_cp", "RolandTR707_cr", "RolandTR707_hh",
		"RolandTR707_ht", "RolandTR707_lt", "RolandTR707_mt", "RolandTR707_oh", "RolandTR707_rim",
		"RolandTR707_sd", "RolandTR707_tb", "RolandTR727_perc", "RolandTR727_sh",
		"RolandTR808_bd", "RolandTR808_cb", "RolandTR808_cp", "RolandTR808_cr", "RolandTR808_hh",
		"RolandTR808_ht", "RolandTR808_lt", "RolandTR808_mt", "RolandTR808_oh", "RolandTR808_perc",
		"RolandTR808_rim", "RolandTR808_sd", "RolandTR808_sh",
		"RolandTR909_bd", "RolandTR909_cp", "RolandTR909_cr", "RolandTR909_hh", "RolandTR909_ht",
		"RolandTR909_lt", "RolandTR909_mt", "RolandTR909_oh", "RolandTR909_rd", "RolandTR909_rim", "RolandTR909_sd",
	},
	// Linn drum machines
	"drum_machines_linn": {
		"Linn9000_bd", "Linn9000_cb", "Linn9000_cr", "Linn9000_hh", "Linn9000_ht",
		"Linn9000_lt", "Linn9000_mt", "Linn9000_oh", "Linn9000_perc", "Linn9000_rd",
		"Linn9000_rim", "Linn9000_sd", "Linn9000_tb",
		"LinnDrum_bd", "LinnDrum_cb", "LinnDrum_cp", "LinnDrum_cr", "LinnDrum_hh",
		"LinnDrum_ht", "LinnDrum_lt", "LinnDrum_mt", "LinnDrum_oh", "LinnDrum_perc",
		"LinnDrum_rd", "LinnDrum_rim", "LinnDrum_sd", "LinnDrum_sh", "LinnDrum_tb",
		"LinnLM1_bd", "LinnLM1_cb", "LinnLM1_cp", "LinnLM1_hh", "LinnLM1_ht",
		"LinnLM1_lt", "LinnLM1_oh", "LinnLM1_perc", "LinnLM1_rim", "LinnLM1_sd",
		"LinnLM1_sh", "LinnLM1_tb",
		"LinnLM2_bd", "LinnLM2_cb", "LinnLM2_cp", "LinnLM2_cr", "LinnLM2_hh",
		"LinnLM2_ht", "LinnLM2_lt", "LinnLM2_mt", "LinnLM2_oh", "LinnLM2_rd",
		"LinnLM2_rim", "LinnLM2_sd", "LinnLM2_sh", "LinnLM2_tb",
	},
	// Korg drum machines
	"drum_machines_korg": {
		"KorgDDM110_bd", "KorgDDM110_cp", "KorgDDM110_cr", "KorgDDM110_hh", "KorgDDM110_ht",
		"KorgDDM110_lt", "KorgDDM110_oh", "KorgDDM110_rim", "KorgDDM110_sd",
		"KorgKPR77_bd", "KorgKPR77_cp", "KorgKPR77_hh", "KorgKPR77_oh", "KorgKPR77_sd",
		"KorgKR55_bd", "KorgKR55_cb", "KorgKR55_cr", "KorgKR55_hh", "KorgKR55_ht",
		"KorgKR55_oh", "KorgKR55_perc", "KorgKR55_rim", "KorgKR55_sd",
		"KorgKRZ_bd", "KorgKRZ_cr", "KorgKRZ_fx", "KorgKRZ_hh", "KorgKRZ_ht",
		"KorgKRZ_lt", "KorgKRZ_misc", "KorgKRZ_oh", "KorgKRZ_rd", "KorgKRZ_sd",
		"KorgM1_bd", "KorgM1_cb", "KorgM1_cp", "KorgM1_cr", "KorgM1_hh", "KorgM1_ht",
		"KorgM1_misc", "KorgM1_mt", "KorgM1_oh", "KorgM1_perc", "KorgM1_rd",
		"KorgM1_rim", "KorgM1_sd", "KorgM1_sh", "KorgM1_tb",
		"KorgMinipops_bd", "KorgMinipops_hh", "KorgMinipops_misc", "KorgMinipops_oh", "KorgMinipops_sd",
		"KorgPoly800_bd",
		"KorgT3_bd", "KorgT3_cp", "KorgT3_hh", "KorgT3_misc", "KorgT3_oh",
		"KorgT3_perc", "KorgT3_rim", "KorgT3_sd", "KorgT3_sh",
	},
	// Yamaha drum machines
	"drum_machines_yamaha": {
		"YamahaRM50_bd", "YamahaRM50_cb", "YamahaRM50_cp", "YamahaRM50_cr", "YamahaRM50_hh",
		"YamahaRM50_ht", "YamahaRM50_lt", "YamahaRM50_misc", "YamahaRM50_mt", "YamahaRM50_oh",
		"YamahaRM50_perc", "YamahaRM50_rd", "YamahaRM50_sd", "YamahaRM50_sh", "YamahaRM50_tb",
		"YamahaRX21_bd", "YamahaRX21_cp", "YamahaRX21_cr", "YamahaRX21_hh", "YamahaRX21_ht",
		"YamahaRX21_lt", "YamahaRX21_mt", "YamahaRX21_oh", "YamahaRX21_sd",
		"YamahaRX5_bd", "YamahaRX5_cb", "YamahaRX5_fx", "YamahaRX5_hh", "YamahaRX5_lt",
		"YamahaRX5_oh", "YamahaRX5_rim", "YamahaRX5_sd", "YamahaRX5_sh", "YamahaRX5_tb",
		"YamahaRY30_bd", "YamahaRY30_cb", "YamahaRY30_cp", "YamahaRY30_cr", "YamahaRY30_hh",
		"YamahaRY30_ht", "YamahaRY30_lt", "YamahaRY30_misc", "YamahaRY30_mt", "YamahaRY30_oh",
		"YamahaRY30_perc", "YamahaRY30_rd", "YamahaRY30_rim", "YamahaRY30_sd", "YamahaRY30_sh", "YamahaRY30_tb",
		"YamahaTG33_bd", "YamahaTG33_cb", "YamahaTG33_cp", "YamahaTG33_cr", "YamahaTG33_fx",
		"YamahaTG33_ht", "YamahaTG33_lt", "YamahaTG33_misc", "YamahaTG33_mt", "YamahaTG33_oh",
		"YamahaTG33_perc", "YamahaTG33_rd", "YamahaTG33_rim", "YamahaTG33_sd", "YamahaTG33_sh", "YamahaTG33_tb",
	},
	// Emu drum machines
	"drum_machines_emu": {
		"EmuDrumulator_bd", "EmuDrumulator_cb", "EmuDrumulator_cp", "EmuDrumulator_cr",
		"EmuDrumulator_hh", "EmuDrumulator_ht", "EmuDrumulator_lt", "EmuDrumulator_mt",
		"EmuDrumulator_oh", "EmuDrumulator_perc", "EmuDrumulator_rim", "EmuDrumulator_sd",
		"EmuModular_bd", "EmuModular_misc", "EmuModular_perc",
		"EmuSP12_bd", "EmuSP12_cb", "EmuSP12_cp", "EmuSP12_cr", "EmuSP12_hh", "EmuSP12_ht",
		"EmuSP12_lt", "EmuSP12_misc", "EmuSP12_mt", "EmuSP12_oh", "EmuSP12_perc",
		"EmuSP12_rd", "EmuSP12_rim", "EmuSP12_sd",
	},
	// Alesis drum machines
	"drum_machines_alesis": {
		"AlesisHR16_bd", "AlesisHR16_cp", "AlesisHR16_hh", "AlesisHR16_ht", "AlesisHR16_lt",
		"AlesisHR16_oh", "AlesisHR16_perc", "AlesisHR16_rim", "AlesisHR16_sd", "AlesisHR16_sh",
		"AlesisSR16_bd", "AlesisSR16_cb", "AlesisSR16_cp", "AlesisSR16_cr", "AlesisSR16_hh",
		"AlesisSR16_misc", "AlesisSR16_oh", "AlesisSR16_perc", "AlesisSR16_rd", "AlesisSR16_rim",
		"AlesisSR16_sd", "AlesisSR16_sh", "AlesisSR16_tb",
	},
	// Boss drum machines
	"drum_machines_boss": {
		"BossDR110_bd", "BossDR110_cp", "BossDR110_cr", "BossDR110_hh", "BossDR110_oh",
		"BossDR110_rd", "BossDR110_sd",
		"BossDR220_bd", "BossDR220_cp", "BossDR220_cr", "BossDR220_hh", "BossDR220_ht",
		"BossDR220_lt", "BossDR220_mt", "BossDR220_oh", "BossDR220_perc", "BossDR220_rd", "BossDR220_sd",
		"BossDR550_bd", "BossDR550_cb", "BossDR550_cp", "BossDR550_cr", "BossDR550_hh",
		"BossDR550_ht", "BossDR550_lt", "BossDR550_misc", "BossDR550_mt", "BossDR550_oh",
		"BossDR550_perc", "BossDR550_rd", "BossDR550_rim", "BossDR550_sd", "BossDR550_sh", "BossDR550_tb",
		"BossDR55_bd", "BossDR55_hh", "BossDR55_rim", "BossDR55_sd",
	},
	// Casio drum machines
	"drum_machines_casio": {
		"CasioRZ1_bd", "CasioRZ1_cb", "CasioRZ1_cp", "CasioRZ1_cr", "CasioRZ1_hh",
		"CasioRZ1_ht", "CasioRZ1_lt", "CasioRZ1_mt", "CasioRZ1_rd", "CasioRZ1_rim", "CasioRZ1_sd",
		"CasioSK1_bd", "CasioSK1_hh", "CasioSK1_ht", "CasioSK1_mt", "CasioSK1_oh", "CasioSK1_sd",
		"CasioVL1_bd", "CasioVL1_hh", "CasioVL1_sd",
	},
	// Simmons/Sequential/Oberheim drum machines
	"drum_machines_other": {
		"OberheimDMX_bd", "OberheimDMX_cp",
		"SimmonsSDS400_ht", "SimmonsSDS400_lt", "SimmonsSDS400_mt", "SimmonsSDS400_sd",
		"SimmonsSDS5_bd", "SimmonsSDS5_hh", "SimmonsSDS5_ht", "SimmonsSDS5_lt",
		"SimmonsSDS5_mt", "SimmonsSDS5_oh", "SimmonsSDS5_rim", "SimmonsSDS5_sd",
		"SequentialCircuitsDrumtracks_bd", "SequentialCircuitsDrumtracks_cb", "SequentialCircuitsDrumtracks_cp",
		"SequentialCircuitsDrumtracks_cr", "SequentialCircuitsDrumtracks_hh", "SequentialCircuitsDrumtracks_ht",
		"SequentialCircuitsDrumtracks_oh", "SequentialCircuitsDrumtracks_rd", "SequentialCircuitsDrumtracks_rim",
		"SequentialCircuitsDrumtracks_sd", "SequentialCircuitsDrumtracks_sh", "SequentialCircuitsDrumtracks_tb",
		"SequentialCircuitsTom_bd", "SequentialCircuitsTom_cp", "SequentialCircuitsTom_cr",
		"SequentialCircuitsTom_hh", "SequentialCircuitsTom_ht", "SequentialCircuitsTom_oh", "SequentialCircuitsTom_sd",
		"SergeModular_bd", "SergeModular_misc", "SergeModular_perc",
		"MFB512_bd", "MFB512_cp", "MFB512_cr", "MFB512_hh", "MFB512_ht",
		"MFB512_lt", "MFB512_mt", "MFB512_oh", "MFB512_sd",
		"MPC1000_bd", "MPC1000_cp", "MPC1000_hh", "MPC1000_oh", "MPC1000_perc", "MPC1000_sd", "MPC1000_sh",
		"MoogConcertMateMG1_bd", "MoogConcertMateMG1_sd",
		"ViscoSpaceDrum_bd", "ViscoSpaceDrum_cb", "ViscoSpaceDrum_hh", "ViscoSpaceDrum_ht",
		"ViscoSpaceDrum_lt", "ViscoSpaceDrum_misc", "ViscoSpaceDrum_mt", "ViscoSpaceDrum_oh",
		"ViscoSpaceDrum_perc", "ViscoSpaceDrum_rim", "ViscoSpaceDrum_sd",
		"XdrumLM8953_bd", "XdrumLM8953_cr", "XdrumLM8953_hh", "XdrumLM8953_ht",
		"XdrumLM8953_lt", "XdrumLM8953_mt", "XdrumLM8953_oh", "XdrumLM8953_rd",
		"XdrumLM8953_rim", "XdrumLM8953_sd", "XdrumLM8953_tb",
	},
	// Exotic/World GM sounds
	"world": {
		"gm_sitar", "gm_banjo", "gm_shamisen", "gm_koto", "gm_kalimba",
		"gm_bagpipe", "gm_fiddle", "gm_shanai",
	},
	// Sound effects GM sounds
	"fx": {
		"gm_fx_rain", "gm_fx_soundtrack", "gm_fx_crystal", "gm_fx_atmosphere",
		"gm_fx_brightness", "gm_fx_goblins", "gm_fx_echoes", "gm_fx_sci_fi",
		"gm_seashore", "gm_bird_tweet", "gm_helicopter", "gm_applause",
	},
}

// StyleCandidate represents a style option with score
type StyleCandidate struct {
	Style SoundStyle
	Score float64
}

// Generator converts MIDI notes to Strudel code
type Generator struct {
	quantize        int
	style           SoundStyle
	palette         SoundPalette
	styleCandidates []StyleCandidate
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

// SetStyleCandidates sets the alternative style candidates for the header
func (g *Generator) SetStyleCandidates(candidates []StyleCandidate) {
	g.styleCandidates = candidates
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
	if analysisResult.TimeSignature != "" && analysisResult.TimeSignature != "4/4" {
		sb.WriteString(fmt.Sprintf(", Time: %s", analysisResult.TimeSignature))
	}
	if analysisResult.SwingRatio > 1.1 {
		sb.WriteString(fmt.Sprintf(", Swing: %.2f", analysisResult.SwingRatio))
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
	g.generateStackedPattern(&sb, bass, mid, high, analysisResult)

	return sb.String()
}

// generateFromCleanup creates rich Strudel output from cleanup result
func (g *Generator) generateFromCleanup(result *CleanupResult, analysisResult *analysis.Result) string {
	var sb strings.Builder

	// Detect sections and form for comments
	sections := DetectSections(result)
	sectionAnalysis := getSectionAnalysis(result)
	formAnalysis := AnalyzeForm(sections, sectionAnalysis)

	// Header with stats
	sb.WriteString("// MIDI-grep output\n")
	sb.WriteString(fmt.Sprintf("// BPM: %.0f", analysisResult.BPM))
	if analysisResult.Key != "" {
		sb.WriteString(fmt.Sprintf(", Key: %s", analysisResult.Key))
	}
	if analysisResult.TimeSignature != "" && analysisResult.TimeSignature != "4/4" {
		sb.WriteString(fmt.Sprintf(", Time: %s", analysisResult.TimeSignature))
	}
	if analysisResult.SwingRatio > 1.1 {
		sb.WriteString(fmt.Sprintf(", Swing: %.2f", analysisResult.SwingRatio))
	}
	sb.WriteString(fmt.Sprintf("\n// Notes: %d (bass: %d, mid: %d, high: %d)\n",
		result.Stats.Total, result.Stats.BassCount, result.Stats.MidCount, result.Stats.HighCount))
	sb.WriteString(fmt.Sprintf("// Style: %s\n", g.style))
	sb.WriteString(fmt.Sprintf("// Duration: %.1f beats\n", result.TotalBeats))

	// Add candidates as comments for transparency
	if len(analysisResult.KeyCandidates) > 1 {
		sb.WriteString("// Key candidates: ")
		for i, c := range analysisResult.KeyCandidates {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s (%.0f%%)", c.Key, c.Confidence*100))
		}
		sb.WriteString("\n")
	}
	if len(analysisResult.BPMCandidates) > 1 {
		sb.WriteString("// BPM candidates: ")
		for i, c := range analysisResult.BPMCandidates {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%.0f (%.0f%%)", c.BPM, c.Confidence*100))
		}
		sb.WriteString("\n")
	}
	if len(analysisResult.TimeSignatureCandidates) > 1 {
		sb.WriteString("// Time sig candidates: ")
		for i, c := range analysisResult.TimeSignatureCandidates {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s (%.0f%%)", c.TimeSignature, c.Confidence*100))
		}
		sb.WriteString("\n")
	}
	if len(g.styleCandidates) > 1 {
		sb.WriteString("// Style candidates: ")
		for i, c := range g.styleCandidates {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s (%.0f%%)", c.Style, c.Score*100))
		}
		sb.WriteString("\n")
	}

	// Add form analysis
	if formAnalysis != nil && formAnalysis.Form != FormUnknown && formAnalysis.Form != FormThroughComp {
		sb.WriteString(GenerateFormHeader(formAnalysis))
	}

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
		analysisResult)

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

// getSectionAnalysis creates a SectionAnalysis from CleanupResult for form detection
func getSectionAnalysis(result *CleanupResult) *SectionAnalysis {
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

	numBars := int(math.Ceil(result.TotalBeats / 4))
	if numBars < 1 {
		numBars = 1
	}

	analysis := &SectionAnalysis{
		BeatDensities:  make([]float64, numBars),
		BeatVelocities: make([]float64, numBars),
		BeatRegisters:  make([]float64, numBars),
		TotalBeats:     result.TotalBeats,
		NumBars:        numBars,
	}

	// Count notes per bar and accumulate velocity/pitch
	noteCounts := make([]int, numBars)
	velocitySums := make([]float64, numBars)
	pitchSums := make([]float64, numBars)

	for _, n := range allNotes {
		bar := int(n.Start / 4)
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
			analysis.BeatDensities[i] = float64(noteCounts[i]) / 4.0
			analysis.BeatVelocities[i] = velocitySums[i] / float64(noteCounts[i])
			analysis.BeatRegisters[i] = pitchSums[i] / float64(noteCounts[i])
		}
	}

	return analysis
}

// generateStackedPattern creates a Strudel stack() with separate voices and GM sounds
func (g *Generator) generateStackedPattern(sb *strings.Builder, bass, mid, high []midi.Note, analysisResult *analysis.Result) {
	g.generateStackedPatternWithVelocity(sb, bass, mid, high, nil, nil, nil, analysisResult)
}

// generateStackedPatternWithVelocity creates a Strudel stack() with per-voice effects and velocity dynamics
func (g *Generator) generateStackedPatternWithVelocity(sb *strings.Builder, bass, mid, high []midi.Note, bassJSON, midJSON, highJSON []NoteJSON, analysisResult *analysis.Result) {
	bpm := analysisResult.BPM

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
	// Don't cap numBars - we need accurate count for auto-chunking

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
	var voiceNames []string
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

				// For short tracks (<= 8 bars), use detailed velocity
				// For long tracks, use LFO dynamics instead (cleaner output)
				if len(v.jsonData) > 0 && numBars <= 8 {
					velocityPattern := g.buildVelocityPatternWithDynamics(v.jsonData, gridSize, numBars, effects.Dynamics)
					if velocityPattern != "" {
						voiceCode += fmt.Sprintf("\n    .velocity(\"%s\")", velocityPattern)
					}
				}
				// Use gain with optional LFO for dynamics
				if v.gain != 1.0 {
					if numBars > 8 {
						// Long tracks: use perlin for organic dynamics
						voiceCode += fmt.Sprintf("\n    .gain(perlin.range(%.2f, %.2f).slow(8))", v.gain*0.8, v.gain)
					} else {
						voiceCode += fmt.Sprintf("\n    .gain(%.2f)", v.gain)
					}
				} else if numBars > 8 {
					voiceCode += "\n    .gain(perlin.range(0.8, 1.0).slow(8))"
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

				voiceNames = append(voiceNames, v.name)

				activeVoices = append(activeVoices, voiceCode)
			}
		}
	}

	if len(activeVoices) == 0 {
		sb.WriteString(fmt.Sprintf("$: note(\"c4\").sound(\"%s\")\n", g.palette.Mid))
		return
	}

	// Build global effects string for detected swing
	globalEffects := ""
	if analysisResult.SwingRatio > 1.1 && analysisResult.SwingConfidence > 0.5 {
		// Convert swing ratio to Strudel swing amount (0-1)
		// Swing ratio 1.0 = 0, 2.0 (triplet) = 0.33
		swingAmount := (analysisResult.SwingRatio - 1.0) / 3.0
		if swingAmount > 0.33 {
			swingAmount = 0.33
		}
		globalEffects = fmt.Sprintf(".swing(%.2f)", swingAmount)
	}

	// Use chunked output with effects as functions (not repeated per bar)
	g.outputChunkedPatterns(sb, activeVoices, voiceNames, globalEffects, numBars)
}

// outputStackPattern outputs all voices in a single $: stack()
func (g *Generator) outputStackPattern(sb *strings.Builder, activeVoices []string, globalEffects string) {
	if len(activeVoices) == 1 {
		// Single voice - no stack needed
		sb.WriteString("$: ")
		sb.WriteString(strings.TrimPrefix(activeVoices[0], "  "))
		if globalEffects != "" {
			sb.WriteString("\n    " + globalEffects)
		}
		sb.WriteString("\n")
	} else {
		// Multiple voices - use stack
		sb.WriteString("$: stack(\n")
		for i, voice := range activeVoices {
			sb.WriteString(voice)
			if i < len(activeVoices)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n")
		}
		sb.WriteString(")")
		if globalEffects != "" {
			sb.WriteString(globalEffects)
		}
		sb.WriteString("\n")
	}
}

// outputSeparatePatterns outputs each voice as a separate $: pattern that can be hushed individually
func (g *Generator) outputSeparatePatterns(sb *strings.Builder, activeVoices []string, voiceNames []string, globalEffects string) {
	sb.WriteString("// Separate patterns - hush with $bass.hush(), $mid.hush(), $high.hush()\n")
	sb.WriteString("// Or use all() to play together, silence() to stop all\n\n")

	for i, voice := range activeVoices {
		name := "pattern"
		if i < len(voiceNames) {
			name = voiceNames[i]
		}
		sb.WriteString(fmt.Sprintf("$%s: ", name))

		// Extract just the note() and effects, skip the comment line
		voiceClean := voice
		if idx := strings.Index(voiceClean, "note("); idx >= 0 {
			voiceClean = voiceClean[idx:]
		}
		sb.WriteString(strings.TrimSpace(voiceClean))
		if globalEffects != "" {
			sb.WriteString("\n    " + globalEffects)
		}
		sb.WriteString("\n\n")
	}

	// Add helper comment for combining
	sb.WriteString("// To play all together:\n")
	sb.WriteString("// all(")
	for i := 0; i < len(activeVoices) && i < len(voiceNames); i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("$%s", voiceNames[i]))
	}
	sb.WriteString(")\n")
}

// outputNamedPatterns outputs let-bound named patterns with an all() combiner
func (g *Generator) outputNamedPatterns(sb *strings.Builder, activeVoices []string, voiceNames []string, globalEffects string) {
	sb.WriteString("// Named patterns - toggle by commenting/uncommenting in all()\n")
	sb.WriteString("// Or call individually: bass.play(), mid.hush()\n\n")

	// Define each pattern with let
	for i, voice := range activeVoices {
		name := "pattern"
		if i < len(voiceNames) {
			name = voiceNames[i]
		}
		sb.WriteString(fmt.Sprintf("let %s = ", name))
		// Extract just the note() and effects, skip the comment
		voiceClean := voice
		if idx := strings.Index(voiceClean, "note("); idx >= 0 {
			voiceClean = voiceClean[idx:]
		}
		sb.WriteString(strings.TrimSpace(voiceClean))
		if globalEffects != "" {
			sb.WriteString("\n  " + globalEffects)
		}
		sb.WriteString("\n\n")
	}

	// Combine with all()
	sb.WriteString("// Play all together:\n")
	sb.WriteString("$: all(\n")
	for i := range activeVoices {
		name := "pattern"
		if i < len(voiceNames) {
			name = voiceNames[i]
		}
		sb.WriteString(fmt.Sprintf("  %s", name))
		if i < len(activeVoices)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString(")\n\n")

	// Add individual pattern triggers as comments
	sb.WriteString("// Solo patterns:\n")
	for i := range activeVoices {
		if i < len(voiceNames) {
			sb.WriteString(fmt.Sprintf("// $: %s\n", voiceNames[i]))
		}
	}
}

// outputChunkedPatterns outputs per-bar arrays for mix & match composition
func (g *Generator) outputChunkedPatterns(sb *strings.Builder, activeVoices []string, voiceNames []string, globalEffects string, numBars int) {
	sb.WriteString("// Bar arrays - mix & match freely\n")
	sb.WriteString("// Pick bars: bass[0], bass[3], or slice: bass.slice(0,4)\n\n")

	// Collect voice data for output
	type voiceData struct {
		name    string
		bars    []string
		sound   string
		effects string
	}
	var voices []voiceData

	// For each voice, extract the note pattern and split by bars
	for i, voice := range activeVoices {
		name := "pattern"
		if i < len(voiceNames) {
			name = voiceNames[i]
		}

		// Extract the note pattern from voice code
		notePattern := extractNotePatternFromVoice(voice)
		if notePattern == "" {
			continue
		}

		// Split by bar separator "|"
		bars := splitBars(notePattern)

		// Extract sound name
		sound := extractSoundName(voice)

		// Extract effects (everything after .sound(...) line)
		effects := extractEffectsOnly(voice)

		voices = append(voices, voiceData{name, bars, sound, effects})

		// Output bars as simple string array
		sb.WriteString(fmt.Sprintf("let %s = [\n", name))
		for j, bar := range bars {
			bar = strings.TrimSpace(bar)
			if bar == "" {
				bar = "~"
			}
			sb.WriteString(fmt.Sprintf("  \"%s\"", bar))
			if j < len(bars)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n")
		}
		sb.WriteString("]\n\n")
	}

	// Output effect functions for each voice
	sb.WriteString("// Effects (applied at playback)\n")
	for _, v := range voices {
		sb.WriteString(fmt.Sprintf("let %sFx = p => p.sound(\"%s\")%s\n", v.name, v.sound, v.effects))
	}
	sb.WriteString("\n")

	// Play all - use cat() to concatenate bars properly
	sb.WriteString("// Play all\n")
	sb.WriteString("$: stack(\n")
	for i, v := range voices {
		sb.WriteString(fmt.Sprintf("  %sFx(cat(...%s.map(b => note(b))))", v.name, v.name))
		if i < len(voices)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString(")\n\n")

	// Examples
	sb.WriteString("// Play specific bars:\n")
	sb.WriteString("// $: bassFx(note(bass[0]))\n")
	sb.WriteString("// $: stack(bassFx(note(bass[0])), midFx(note(mid[0])))\n")
	sb.WriteString("// Loop first 4 bars:\n")
	sb.WriteString("// $: bassFx(cat(...bass.slice(0,4).map(b => note(b))))\n")
}

// extractSoundName extracts just the sound name from voice code
func extractSoundName(voice string) string {
	start := strings.Index(voice, ".sound(\"")
	if start == -1 {
		return "piano"
	}
	start += 8 // len(".sound(\"")
	end := strings.Index(voice[start:], "\"")
	if end == -1 {
		return "piano"
	}
	return voice[start : start+end]
}

// extractEffectsOnly extracts effects chain without the sound
func extractEffectsOnly(voice string) string {
	// Find .sound("...") and get everything after it
	soundStart := strings.Index(voice, ".sound(\"")
	if soundStart == -1 {
		return ""
	}

	// Find the end of .sound("...")
	afterSound := voice[soundStart+8:]
	endQuote := strings.Index(afterSound, "\")")
	if endQuote == -1 {
		return ""
	}

	// Get everything after .sound("...")
	rest := afterSound[endQuote+2:]

	// Clean up - get just the effect chain
	var effects []string
	lines := strings.Split(rest, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		// Remove trailing comma if present
		line = strings.TrimSuffix(line, ",")
		if line != "" {
			effects = append(effects, line)
		}
	}

	if len(effects) == 0 {
		return ""
	}
	return "\n    " + strings.Join(effects, "\n    ")
}

// extractNotePatternFromVoice extracts the note pattern string from voice code
func extractNotePatternFromVoice(voice string) string {
	start := strings.Index(voice, "note(\"")
	if start == -1 {
		return ""
	}
	start += 6 // len("note(\"")

	// Find closing quote - handle escaped quotes
	end := start
	for end < len(voice) {
		if voice[end] == '"' && (end == 0 || voice[end-1] != '\\') {
			break
		}
		end++
	}

	if end >= len(voice) {
		return ""
	}

	return voice[start:end]
}

// splitBars splits a pattern string by bar separator "|"
func splitBars(pattern string) []string {
	// Split by |
	bars := strings.Split(pattern, "|")

	// Clean up each bar
	result := make([]string, 0, len(bars))
	for _, bar := range bars {
		bar = strings.TrimSpace(bar)
		if bar != "" {
			result = append(result, bar)
		}
	}

	return result
}

// extractSoundAndEffects extracts .sound() and all following effect chains
func extractSoundAndEffects(voice string) string {
	start := strings.Index(voice, ".sound(")
	if start == -1 {
		return ""
	}

	// Find where effects end (before any comment or newline that's not followed by a dot)
	result := voice[start:]

	// Remove trailing whitespace and comments
	lines := strings.Split(result, "\n")
	var effectLines []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		effectLines = append(effectLines, line)
	}

	return "\n    " + strings.Join(effectLines, "\n    ")
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

// ValidGMSounds contains all valid sound names from strudel-client
// Sources: gm.mjs, synth.mjs, zzfx.mjs, noise.mjs, prebake.ts, vcsl.json, tidal-drum-machines.json
var ValidGMSounds = map[string]bool{
	// ========== GM SOUNDFONTS (128 instruments) ==========
	// Piano (0-7)
	"gm_piano":       true,
	"gm_epiano1":     true,
	"gm_epiano2":     true,
	"gm_harpsichord": true,
	"gm_clavinet":    true,
	// Chromatic Percussion (8-15)
	"gm_celesta":       true,
	"gm_glockenspiel":  true,
	"gm_music_box":     true,
	"gm_vibraphone":    true,
	"gm_marimba":       true,
	"gm_xylophone":     true,
	"gm_tubular_bells": true,
	"gm_dulcimer":      true,
	// Organ (16-23)
	"gm_drawbar_organ":    true,
	"gm_percussive_organ": true,
	"gm_rock_organ":       true,
	"gm_church_organ":     true,
	"gm_reed_organ":       true,
	"gm_accordion":        true,
	"gm_harmonica":        true,
	"gm_bandoneon":        true,
	// Guitar (24-31)
	"gm_acoustic_guitar_nylon": true,
	"gm_acoustic_guitar_steel": true,
	"gm_electric_guitar_jazz":  true,
	"gm_electric_guitar_clean": true,
	"gm_electric_guitar_muted": true,
	"gm_overdriven_guitar":     true,
	"gm_distortion_guitar":     true,
	"gm_guitar_harmonics":      true,
	// Bass (32-39)
	"gm_acoustic_bass":        true,
	"gm_electric_bass_finger": true,
	"gm_electric_bass_pick":   true,
	"gm_fretless_bass":        true,
	"gm_slap_bass_1":          true,
	"gm_slap_bass_2":          true,
	"gm_synth_bass_1":         true,
	"gm_synth_bass_2":         true,
	// Strings (40-47)
	"gm_violin":            true,
	"gm_viola":             true,
	"gm_cello":             true,
	"gm_contrabass":        true,
	"gm_tremolo_strings":   true,
	"gm_pizzicato_strings": true,
	"gm_orchestral_harp":   true,
	"gm_timpani":           true,
	// Ensemble (48-55)
	"gm_string_ensemble_1": true,
	"gm_string_ensemble_2": true,
	"gm_synth_strings_1":   true,
	"gm_synth_strings_2":   true,
	"gm_choir_aahs":        true,
	"gm_voice_oohs":        true,
	"gm_synth_choir":       true,
	"gm_orchestra_hit":     true,
	// Brass (56-63)
	"gm_trumpet":       true,
	"gm_trombone":      true,
	"gm_tuba":          true,
	"gm_muted_trumpet": true,
	"gm_french_horn":   true,
	"gm_brass_section": true,
	"gm_synth_brass_1": true,
	"gm_synth_brass_2": true,
	// Reed (64-71)
	"gm_soprano_sax":  true,
	"gm_alto_sax":     true,
	"gm_tenor_sax":    true,
	"gm_baritone_sax": true,
	"gm_oboe":         true,
	"gm_english_horn": true,
	"gm_bassoon":      true,
	"gm_clarinet":     true,
	// Pipe (72-79)
	"gm_piccolo":      true,
	"gm_flute":        true,
	"gm_recorder":     true,
	"gm_pan_flute":    true,
	"gm_blown_bottle": true,
	"gm_shakuhachi":   true,
	"gm_whistle":      true,
	"gm_ocarina":      true,
	// Synth Lead (80-87)
	"gm_lead_1_square":    true,
	"gm_lead_2_sawtooth":  true,
	"gm_lead_3_calliope":  true,
	"gm_lead_4_chiff":     true,
	"gm_lead_5_charang":   true,
	"gm_lead_6_voice":     true,
	"gm_lead_7_fifths":    true,
	"gm_lead_8_bass_lead": true,
	// Synth Pad (88-95)
	"gm_pad_new_age":  true,
	"gm_pad_warm":     true,
	"gm_pad_poly":     true,
	"gm_pad_choir":    true,
	"gm_pad_bowed":    true,
	"gm_pad_metallic": true,
	"gm_pad_halo":     true,
	"gm_pad_sweep":    true,
	// Synth Effects (96-103)
	"gm_fx_rain":       true,
	"gm_fx_soundtrack": true,
	"gm_fx_crystal":    true,
	"gm_fx_atmosphere": true,
	"gm_fx_brightness": true,
	"gm_fx_goblins":    true,
	"gm_fx_echoes":     true,
	"gm_fx_sci_fi":     true,
	// Ethnic (104-111)
	"gm_sitar":    true,
	"gm_banjo":    true,
	"gm_shamisen": true,
	"gm_koto":     true,
	"gm_kalimba":  true,
	"gm_bagpipe":  true,
	"gm_fiddle":   true,
	"gm_shanai":   true,
	// Percussive (112-119)
	"gm_tinkle_bell":    true,
	"gm_agogo":          true,
	"gm_steel_drums":    true,
	"gm_woodblock":      true,
	"gm_taiko_drum":     true,
	"gm_melodic_tom":    true,
	"gm_synth_drum":     true,
	"gm_reverse_cymbal": true,
	// Sound Effects (120-127)
	"gm_guitar_fret_noise": true,
	"gm_breath_noise":      true,
	"gm_seashore":          true,
	"gm_bird_tweet":        true,
	"gm_telephone":         true,
	"gm_helicopter":        true,
	"gm_applause":          true,
	"gm_gunshot":           true,

	// ========== BUILT-IN OSCILLATORS (synth.mjs) ==========
	"sine":     true,
	"triangle": true,
	"square":   true,
	"sawtooth": true,
	"supersaw": true,
	"pulse":    true,
	"sbd":      true, // Synth bass drum
	"bytebeat": true,

	// ========== NOISE GENERATORS (noise.mjs) ==========
	"white":   true,
	"pink":    true,
	"brown":   true,
	"crackle": true,

	// ========== ZZFX SYNTHS (zzfx.mjs) ==========
	"zzfx":       true,
	"z_sine":     true,
	"z_sawtooth": true,
	"z_triangle": true,
	"z_square":   true,
	"z_tan":      true,
	"z_noise":    true,

	// ========== WAVETABLES (uzu-wavetables.json) ==========
	"wt_digital":          true,
	"wt_digital_bad_day":  true,
	"wt_digital_basique":  true,
	"wt_digital_crickets": true,
	"wt_digital_curses":   true,
	"wt_digital_echoes":   true,
	"wt_vgame":            true,

	// ========== SALAMANDER PIANO ==========
	"piano": true,

	// ========== DIRT SAMPLES (tidalcycles/dirt-samples) ==========
	"casio":   true,
	"crow":    true,
	"insect":  true,
	"wind":    true,
	"jazz":    true,
	"metal":   true,
	"east":    true,
	"space":   true,
	"numbers": true,
	"num":     true,

	// ========== MRIDANGAM - Indian Percussion ==========
	"mridangam_ardha":  true,
	"mridangam_chaapu": true,
	"mridangam_dhi":    true,
	"mridangam_dhin":   true,
	"mridangam_dhum":   true,
	"mridangam_gumki":  true,
	"mridangam_ka":     true,
	"mridangam_ki":     true,
	"mridangam_na":     true,
	"mridangam_nam":    true,
	"mridangam_ta":     true,
	"mridangam_tha":    true,
	"mridangam_thom":   true,

	// ========== UZU DRUMKIT ==========
	"bd":   true, // Bass drum
	"brk":  true, // Break
	"cb":   true, // Cowbell
	"cp":   true, // Clap
	"cr":   true, // Crash
	"hh":   true, // Hi-hat
	"ht":   true, // High tom
	"lt":   true, // Low tom
	"misc": true, // Miscellaneous
	"mt":   true, // Mid tom
	"oh":   true, // Open hi-hat
	"rd":   true, // Ride
	"rim":  true, // Rim shot
	"sd":   true, // Snare drum
	"sh":   true, // Shaker
	"tb":   true, // Tambourine

	// ========== VCSL PERCUSSION (VSCO Community Sample Library) ==========
	"Agogo Bells":                    true,
	"Anvil":                          true,
	"Ball Whistle":                   true,
	"Bass Drum 1":                    true,
	"Bass Drum 2":                    true,
	"Bongos":                         true,
	"Brake Drum":                     true,
	"Cabasa":                         true,
	"Cajon":                          true,
	"Claps":                          true,
	"Clash Cymbals 1":                true,
	"Clash Cymbals 2":                true,
	"Claves":                         true,
	"Conga":                          true,
	"Cowbells":                       true,
	"Darbuka":                        true,
	"Didgeridoo":                     true,
	"Finger Cymbals":                 true,
	"Flexatone":                      true,
	"Frame Drum":                     true,
	"Gong 1":                         true,
	"Gong 2":                         true,
	"Guiro":                          true,
	"Hand Bells 2c Nepalese":         true,
	"Hi Hat Cymbal":                  true,
	"Mark Trees Legacy":              true,
	"Ocean Drum":                     true,
	"Ratchet":                        true,
	"Shaker 2c Large":                true,
	"Shaker 2c Small":                true,
	"Siren":                          true,
	"Slapstick":                      true,
	"Sleigh Bells":                   true,
	"Slit Drum":                      true,
	"Snare Drum 2c Modern 1":         true,
	"Snare Drum 2c Modern 2":         true,
	"Snare Drum 2c Modern 3":         true,
	"Snare Drum 2c Rope Tension":     true,
	"Snare Drum 2c Rope Tension Hi":  true,
	"Snare Drum 2c Rope Tension Low": true,
	"Suspended Cymbal 1":             true,
	"Suspended Cymbal 2":             true,
	"Tambourine 1":                   true,
	"Tambourine 2":                   true,
	"Timpani 1 Hit":                  true,
	"Timpani 1 Roll":                 true,
	"Timpani 2 Hit":                  true,
	"Tom 1":                          true,
	"Tom 1 Stick":                    true,
	"Tom 2":                          true,
	"Tom 2 Mallet":                   true,
	"Tom 2 Stick":                    true,
	"Train Whistle 2c Toy":           true,
	"Triangles":                      true,
	"Vibraslap Legacy":               true,
	"Woodblock":                      true,
	// Also add simplified VCSL names
	"harp":    true,
	"timpani": true,

	// ========== DRUM MACHINES - Roland ==========
	"RolandTR505_bd": true, "RolandTR505_cb": true, "RolandTR505_cp": true, "RolandTR505_cr": true,
	"RolandTR505_hh": true, "RolandTR505_ht": true, "RolandTR505_lt": true, "RolandTR505_mt": true,
	"RolandTR505_oh": true, "RolandTR505_rim": true, "RolandTR505_sd": true, "RolandTR505_tb": true,
	"RolandTR606_bd": true, "RolandTR606_hh": true, "RolandTR606_ht": true, "RolandTR606_lt": true,
	"RolandTR606_mt": true, "RolandTR606_oh": true, "RolandTR606_sd": true,
	"RolandTR626_bd": true, "RolandTR626_cb": true, "RolandTR626_cp": true, "RolandTR626_cr": true,
	"RolandTR626_hh": true, "RolandTR626_ht": true, "RolandTR626_lt": true, "RolandTR626_mt": true,
	"RolandTR626_oh": true, "RolandTR626_perc": true, "RolandTR626_rd": true, "RolandTR626_rim": true,
	"RolandTR626_sd": true, "RolandTR626_sh": true, "RolandTR626_tb": true,
	"RolandTR707_bd": true, "RolandTR707_cb": true, "RolandTR707_cp": true, "RolandTR707_cr": true,
	"RolandTR707_hh": true, "RolandTR707_ht": true, "RolandTR707_lt": true, "RolandTR707_mt": true,
	"RolandTR707_oh": true, "RolandTR707_rim": true, "RolandTR707_sd": true, "RolandTR707_tb": true,
	"RolandTR727_perc": true, "RolandTR727_sh": true,
	"RolandTR808_bd": true, "RolandTR808_cb": true, "RolandTR808_cp": true, "RolandTR808_cr": true,
	"RolandTR808_hh": true, "RolandTR808_ht": true, "RolandTR808_lt": true, "RolandTR808_mt": true,
	"RolandTR808_oh": true, "RolandTR808_perc": true, "RolandTR808_rim": true, "RolandTR808_sd": true, "RolandTR808_sh": true,
	"RolandTR909_bd": true, "RolandTR909_cp": true, "RolandTR909_cr": true, "RolandTR909_hh": true,
	"RolandTR909_ht": true, "RolandTR909_lt": true, "RolandTR909_mt": true, "RolandTR909_oh": true,
	"RolandTR909_rd": true, "RolandTR909_rim": true, "RolandTR909_sd": true,

	// ========== DRUM MACHINES - Linn ==========
	"Linn9000_bd": true, "Linn9000_cb": true, "Linn9000_cr": true, "Linn9000_hh": true,
	"Linn9000_ht": true, "Linn9000_lt": true, "Linn9000_mt": true, "Linn9000_oh": true,
	"Linn9000_perc": true, "Linn9000_rd": true, "Linn9000_rim": true, "Linn9000_sd": true, "Linn9000_tb": true,
	"LinnDrum_bd": true, "LinnDrum_cb": true, "LinnDrum_cp": true, "LinnDrum_cr": true,
	"LinnDrum_hh": true, "LinnDrum_ht": true, "LinnDrum_lt": true, "LinnDrum_mt": true,
	"LinnDrum_oh": true, "LinnDrum_perc": true, "LinnDrum_rd": true, "LinnDrum_rim": true,
	"LinnDrum_sd": true, "LinnDrum_sh": true, "LinnDrum_tb": true,
	"LinnLM1_bd": true, "LinnLM1_cb": true, "LinnLM1_cp": true, "LinnLM1_hh": true,
	"LinnLM1_ht": true, "LinnLM1_lt": true, "LinnLM1_oh": true, "LinnLM1_perc": true,
	"LinnLM1_rim": true, "LinnLM1_sd": true, "LinnLM1_sh": true, "LinnLM1_tb": true,
	"LinnLM2_bd": true, "LinnLM2_cb": true, "LinnLM2_cp": true, "LinnLM2_cr": true,
	"LinnLM2_hh": true, "LinnLM2_ht": true, "LinnLM2_lt": true, "LinnLM2_mt": true,
	"LinnLM2_oh": true, "LinnLM2_rd": true, "LinnLM2_rim": true, "LinnLM2_sd": true,
	"LinnLM2_sh": true, "LinnLM2_tb": true,

	// ========== DRUM MACHINES - Akai ==========
	"AkaiLinn_bd": true, "AkaiLinn_cb": true, "AkaiLinn_cp": true, "AkaiLinn_cr": true,
	"AkaiLinn_hh": true, "AkaiLinn_ht": true, "AkaiLinn_lt": true, "AkaiLinn_mt": true,
	"AkaiLinn_oh": true, "AkaiLinn_rd": true, "AkaiLinn_sd": true, "AkaiLinn_sh": true, "AkaiLinn_tb": true,
	"AkaiMPC60_bd": true, "AkaiMPC60_cp": true, "AkaiMPC60_cr": true, "AkaiMPC60_hh": true,
	"AkaiMPC60_ht": true, "AkaiMPC60_lt": true, "AkaiMPC60_misc": true, "AkaiMPC60_mt": true,
	"AkaiMPC60_oh": true, "AkaiMPC60_perc": true, "AkaiMPC60_rd": true, "AkaiMPC60_rim": true, "AkaiMPC60_sd": true,
	"AkaiXR10_bd": true, "AkaiXR10_cb": true, "AkaiXR10_cp": true, "AkaiXR10_cr": true,
	"AkaiXR10_hh": true, "AkaiXR10_ht": true, "AkaiXR10_lt": true, "AkaiXR10_misc": true,
	"AkaiXR10_mt": true, "AkaiXR10_oh": true, "AkaiXR10_perc": true, "AkaiXR10_rd": true,
	"AkaiXR10_rim": true, "AkaiXR10_sd": true, "AkaiXR10_sh": true, "AkaiXR10_tb": true,

	// ========== DRUM MACHINES - Korg ==========
	"KorgDDM110_bd": true, "KorgDDM110_cp": true, "KorgDDM110_cr": true, "KorgDDM110_hh": true,
	"KorgDDM110_ht": true, "KorgDDM110_lt": true, "KorgDDM110_oh": true, "KorgDDM110_rim": true, "KorgDDM110_sd": true,
	"KorgKPR77_bd": true, "KorgKPR77_cp": true, "KorgKPR77_hh": true, "KorgKPR77_oh": true, "KorgKPR77_sd": true,
	"KorgKR55_bd": true, "KorgKR55_cb": true, "KorgKR55_cr": true, "KorgKR55_hh": true,
	"KorgKR55_ht": true, "KorgKR55_oh": true, "KorgKR55_perc": true, "KorgKR55_rim": true, "KorgKR55_sd": true,
	"KorgKRZ_bd": true, "KorgKRZ_cr": true, "KorgKRZ_fx": true, "KorgKRZ_hh": true,
	"KorgKRZ_ht": true, "KorgKRZ_lt": true, "KorgKRZ_misc": true, "KorgKRZ_oh": true, "KorgKRZ_rd": true, "KorgKRZ_sd": true,
	"KorgM1_bd": true, "KorgM1_cb": true, "KorgM1_cp": true, "KorgM1_cr": true, "KorgM1_hh": true,
	"KorgM1_ht": true, "KorgM1_misc": true, "KorgM1_mt": true, "KorgM1_oh": true, "KorgM1_perc": true,
	"KorgM1_rd": true, "KorgM1_rim": true, "KorgM1_sd": true, "KorgM1_sh": true, "KorgM1_tb": true,
	"KorgMinipops_bd": true, "KorgMinipops_hh": true, "KorgMinipops_misc": true, "KorgMinipops_oh": true, "KorgMinipops_sd": true,
	"KorgPoly800_bd": true,
	"KorgT3_bd":      true, "KorgT3_cp": true, "KorgT3_hh": true, "KorgT3_misc": true, "KorgT3_oh": true,
	"KorgT3_perc": true, "KorgT3_rim": true, "KorgT3_sd": true, "KorgT3_sh": true,

	// ========== DRUM MACHINES - Yamaha ==========
	"YamahaRM50_bd": true, "YamahaRM50_cb": true, "YamahaRM50_cp": true, "YamahaRM50_cr": true, "YamahaRM50_hh": true,
	"YamahaRM50_ht": true, "YamahaRM50_lt": true, "YamahaRM50_misc": true, "YamahaRM50_mt": true, "YamahaRM50_oh": true,
	"YamahaRM50_perc": true, "YamahaRM50_rd": true, "YamahaRM50_sd": true, "YamahaRM50_sh": true, "YamahaRM50_tb": true,
	"YamahaRX21_bd": true, "YamahaRX21_cp": true, "YamahaRX21_cr": true, "YamahaRX21_hh": true,
	"YamahaRX21_ht": true, "YamahaRX21_lt": true, "YamahaRX21_mt": true, "YamahaRX21_oh": true, "YamahaRX21_sd": true,
	"YamahaRX5_bd": true, "YamahaRX5_cb": true, "YamahaRX5_fx": true, "YamahaRX5_hh": true, "YamahaRX5_lt": true,
	"YamahaRX5_oh": true, "YamahaRX5_rim": true, "YamahaRX5_sd": true, "YamahaRX5_sh": true, "YamahaRX5_tb": true,
	"YamahaRY30_bd": true, "YamahaRY30_cb": true, "YamahaRY30_cp": true, "YamahaRY30_cr": true, "YamahaRY30_hh": true,
	"YamahaRY30_ht": true, "YamahaRY30_lt": true, "YamahaRY30_misc": true, "YamahaRY30_mt": true, "YamahaRY30_oh": true,
	"YamahaRY30_perc": true, "YamahaRY30_rd": true, "YamahaRY30_rim": true, "YamahaRY30_sd": true, "YamahaRY30_sh": true, "YamahaRY30_tb": true,
	"YamahaTG33_bd": true, "YamahaTG33_cb": true, "YamahaTG33_cp": true, "YamahaTG33_cr": true, "YamahaTG33_fx": true,
	"YamahaTG33_ht": true, "YamahaTG33_lt": true, "YamahaTG33_misc": true, "YamahaTG33_mt": true, "YamahaTG33_oh": true,
	"YamahaTG33_perc": true, "YamahaTG33_rd": true, "YamahaTG33_rim": true, "YamahaTG33_sd": true, "YamahaTG33_sh": true, "YamahaTG33_tb": true,

	// ========== DRUM MACHINES - Emu ==========
	"EmuDrumulator_bd": true, "EmuDrumulator_cb": true, "EmuDrumulator_cp": true, "EmuDrumulator_cr": true,
	"EmuDrumulator_hh": true, "EmuDrumulator_ht": true, "EmuDrumulator_lt": true, "EmuDrumulator_mt": true,
	"EmuDrumulator_oh": true, "EmuDrumulator_perc": true, "EmuDrumulator_rim": true, "EmuDrumulator_sd": true,
	"EmuModular_bd": true, "EmuModular_misc": true, "EmuModular_perc": true,
	"EmuSP12_bd": true, "EmuSP12_cb": true, "EmuSP12_cp": true, "EmuSP12_cr": true, "EmuSP12_hh": true,
	"EmuSP12_ht": true, "EmuSP12_lt": true, "EmuSP12_misc": true, "EmuSP12_mt": true, "EmuSP12_oh": true,
	"EmuSP12_perc": true, "EmuSP12_rd": true, "EmuSP12_rim": true, "EmuSP12_sd": true,

	// ========== DRUM MACHINES - Alesis ==========
	"AlesisHR16_bd": true, "AlesisHR16_cp": true, "AlesisHR16_hh": true, "AlesisHR16_ht": true,
	"AlesisHR16_lt": true, "AlesisHR16_oh": true, "AlesisHR16_perc": true, "AlesisHR16_rim": true,
	"AlesisHR16_sd": true, "AlesisHR16_sh": true,
	"AlesisSR16_bd": true, "AlesisSR16_cb": true, "AlesisSR16_cp": true, "AlesisSR16_cr": true,
	"AlesisSR16_hh": true, "AlesisSR16_misc": true, "AlesisSR16_oh": true, "AlesisSR16_perc": true,
	"AlesisSR16_rd": true, "AlesisSR16_rim": true, "AlesisSR16_sd": true, "AlesisSR16_sh": true, "AlesisSR16_tb": true,

	// ========== DRUM MACHINES - Boss ==========
	"BossDR110_bd": true, "BossDR110_cp": true, "BossDR110_cr": true, "BossDR110_hh": true,
	"BossDR110_oh": true, "BossDR110_rd": true, "BossDR110_sd": true,
	"BossDR220_bd": true, "BossDR220_cp": true, "BossDR220_cr": true, "BossDR220_hh": true, "BossDR220_ht": true,
	"BossDR220_lt": true, "BossDR220_mt": true, "BossDR220_oh": true, "BossDR220_perc": true, "BossDR220_rd": true, "BossDR220_sd": true,
	"BossDR550_bd": true, "BossDR550_cb": true, "BossDR550_cp": true, "BossDR550_cr": true, "BossDR550_hh": true,
	"BossDR550_ht": true, "BossDR550_lt": true, "BossDR550_misc": true, "BossDR550_mt": true, "BossDR550_oh": true,
	"BossDR550_perc": true, "BossDR550_rd": true, "BossDR550_rim": true, "BossDR550_sd": true, "BossDR550_sh": true, "BossDR550_tb": true,
	"BossDR55_bd": true, "BossDR55_hh": true, "BossDR55_rim": true, "BossDR55_sd": true,

	// ========== DRUM MACHINES - Casio ==========
	"CasioRZ1_bd": true, "CasioRZ1_cb": true, "CasioRZ1_cp": true, "CasioRZ1_cr": true, "CasioRZ1_hh": true,
	"CasioRZ1_ht": true, "CasioRZ1_lt": true, "CasioRZ1_mt": true, "CasioRZ1_rd": true, "CasioRZ1_rim": true, "CasioRZ1_sd": true,
	"CasioSK1_bd": true, "CasioSK1_hh": true, "CasioSK1_ht": true, "CasioSK1_mt": true, "CasioSK1_oh": true, "CasioSK1_sd": true,
	"CasioVL1_bd": true, "CasioVL1_hh": true, "CasioVL1_sd": true,

	// ========== DRUM MACHINES - Other (Oberheim, Simmons, Sequential, etc.) ==========
	"OberheimDMX_bd": true, "OberheimDMX_cp": true,
	"SimmonsSDS400_ht": true, "SimmonsSDS400_lt": true, "SimmonsSDS400_mt": true, "SimmonsSDS400_sd": true,
	"SimmonsSDS5_bd": true, "SimmonsSDS5_hh": true, "SimmonsSDS5_ht": true, "SimmonsSDS5_lt": true,
	"SimmonsSDS5_mt": true, "SimmonsSDS5_oh": true, "SimmonsSDS5_rim": true, "SimmonsSDS5_sd": true,
	"SequentialCircuitsDrumtracks_bd": true, "SequentialCircuitsDrumtracks_cb": true, "SequentialCircuitsDrumtracks_cp": true,
	"SequentialCircuitsDrumtracks_cr": true, "SequentialCircuitsDrumtracks_hh": true, "SequentialCircuitsDrumtracks_ht": true,
	"SequentialCircuitsDrumtracks_oh": true, "SequentialCircuitsDrumtracks_rd": true, "SequentialCircuitsDrumtracks_rim": true,
	"SequentialCircuitsDrumtracks_sd": true, "SequentialCircuitsDrumtracks_sh": true, "SequentialCircuitsDrumtracks_tb": true,
	"SequentialCircuitsTom_bd": true, "SequentialCircuitsTom_cp": true, "SequentialCircuitsTom_cr": true,
	"SequentialCircuitsTom_hh": true, "SequentialCircuitsTom_ht": true, "SequentialCircuitsTom_oh": true, "SequentialCircuitsTom_sd": true,
	"SergeModular_bd": true, "SergeModular_misc": true, "SergeModular_perc": true,
	"MFB512_bd": true, "MFB512_cp": true, "MFB512_cr": true, "MFB512_hh": true, "MFB512_ht": true,
	"MFB512_lt": true, "MFB512_mt": true, "MFB512_oh": true, "MFB512_sd": true,
	"MPC1000_bd": true, "MPC1000_cp": true, "MPC1000_hh": true, "MPC1000_oh": true, "MPC1000_perc": true, "MPC1000_sd": true, "MPC1000_sh": true,
	"MoogConcertMateMG1_bd": true, "MoogConcertMateMG1_sd": true,
	"ViscoSpaceDrum_bd": true, "ViscoSpaceDrum_cb": true, "ViscoSpaceDrum_hh": true, "ViscoSpaceDrum_ht": true,
	"ViscoSpaceDrum_lt": true, "ViscoSpaceDrum_misc": true, "ViscoSpaceDrum_mt": true, "ViscoSpaceDrum_oh": true,
	"ViscoSpaceDrum_perc": true, "ViscoSpaceDrum_rim": true, "ViscoSpaceDrum_sd": true,
	"XdrumLM8953_bd": true, "XdrumLM8953_cr": true, "XdrumLM8953_hh": true, "XdrumLM8953_ht": true,
	"XdrumLM8953_lt": true, "XdrumLM8953_mt": true, "XdrumLM8953_oh": true, "XdrumLM8953_rd": true,
	"XdrumLM8953_rim": true, "XdrumLM8953_sd": true, "XdrumLM8953_tb": true,
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
