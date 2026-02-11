#!/usr/bin/env python3
"""
Strudel Live Coding Reference for AI Code Generation

This module provides comprehensive documentation about Strudel's interactive
features, controls, and patterns for use in LLM prompts.
"""

STRUDEL_INTERACTIVE_CONTROLS = """
## STRUDEL INTERACTIVE CONTROLS

### Inline Sliders - Create draggable UI controls in code
```javascript
// slider(default, min, max, step)
.lpf(slider(500, 100, 2000, 1))     // Draggable filter cutoff
.gain(slider(0.5, 0, 1, 0.01))       // Draggable volume
.room(slider(0.2, 0, 0.8, 0.05))     // Draggable reverb
.attack(slider(0.01, 0.001, 0.2))    // Draggable envelope
```

### Tempo Control
```javascript
// Set tempo in cycles per second (default 0.5 = 120 BPM at 4 beats/cycle)
setcps(0.5)                          // 120 BPM standard
setcps(140/60/4)                     // 140 BPM with 4 beats/cycle
setcpm(140/4)                        // Same as above, using CPM

// Dynamic tempo with slider
setcps(slider(0.5, 0.3, 1.0, 0.01))  // Tempo control slider
```

### Mute/Solo/Hush Controls
```javascript
// Stop everything
hush()

// Layer control (define layers with names)
let drums = s("bd sd hh")
let bass = note("c2 g2").sound("bass")

// Solo one layer (mutes everything else)
solo(drums)

// Keyboard shortcuts in REPL:
// Alt-n: Toggle mute/unmute pattern
// Shift-Alt-n: Toggle solo/unsolo pattern
```

### Pattern Modifiers for Live Control
```javascript
// Probability-based - great for live variation
.sometimes(x => x.speed(2))          // 50% chance of double speed
.rarely(x => x.rev())                // 25% chance of reverse
.almostAlways(x => x.crush(4))       // 75% chance of bitcrush
.degradeBy(slider(0.3, 0, 1))        // Slider-controlled note dropping

// Conditional patterns
.when(tglA, x => x.lpf(400))         // Gamepad A button toggles filter
.mask(gp.a)                          // Mask pattern with button

// Swing for groove
.swing(slider(0.1, 0, 0.3))          // Adjustable swing amount
```

### Signal-Based Control (Continuous Values)
```javascript
// LFO-style modulation
sine.range(400, 4000).slow(8)        // Sine LFO for filter sweep
perlin.range(0.3, 0.7).slow(16)      // Perlin noise for organic variation
saw.range(0, 1).fast(2)              // Sawtooth LFO
tri.range(-1, 1)                     // Triangle LFO
square.range(0, 1)                   // Square LFO
rand.range(0.5, 1.5)                 // Random values

// Use with effects
.lpf(sine.range(400, 4000).slow(8))  // Automated filter sweep
.gain(perlin.range(0.3, 0.7).slow(8)) // Natural-feeling volume variation
```

### Gamepad/MIDI Input
```javascript
// Initialize gamepad
const gp = gamepad(0)

// Button masks (play/stop notes)
note("c3 e3 g3").mask(gp.a)          // Only plays when A is held

// Analog stick continuous control
.lpf(gp.x1.range(100, 4000))         // Left stick X controls filter
.room(gp.y2.range(0, 0.5))           // Right stick Y controls reverb

// Toggle buttons
.gain(gp.tglA.range(0, 1))           // A button toggles between 0 and 1
```
"""

STRUDEL_PATTERN_SYNTAX = """
## STRUDEL PATTERN SYNTAX

### Mini-Notation Basics
```javascript
"c3 d3 e3 f3"                        // 4 notes per cycle
"c3*4"                               // c3 repeated 4 times
"c3!4"                               // c3 held for 4 beats
"c3@2 d3"                            // c3 takes 2/3, d3 takes 1/3
"c3 ~ d3 ~"                          // ~ is rest/silence
"[c3 e3] g3"                         // [brackets] subdivide
"<c3 e3 g3>"                         // <angles> alternate each cycle
"c3?"                                // 50% chance to play
"c3(3,8)"                            // Euclidean rhythm (3 hits in 8 slots)
```

### Pattern Combinators
```javascript
cat("c3 d3", "e3 f3")                // Concatenate patterns (A then B)
stack(drums, bass, lead)             // Play all simultaneously
sequence("c3", "d3", "e3")           // Same as cat
layer(pattern1, pattern2)             // Same as stack

// Useful for bar-based composition:
let bars = ["c3 e3 g3", "d3 f3 a3", "e3 g3 b3"]
cat(...bars.map(b => note(b)))       // Play all bars in sequence
```

### Value Modifiers
```javascript
.add("<0 3 7>")                      // Add to note values
.sub(2)                              // Subtract from values
.mul(1.5)                            // Multiply values
.range(0.5, 1.5)                     // Scale 0-1 to 0.5-1.5
.rangex(100, 4000)                   // Exponential scaling (good for freq)
```
"""

STRUDEL_SOUND_LIBRARY = """
## STRUDEL SOUND LIBRARY

### Synthesizer Waveforms
- sine, sawtooth, square, triangle, supersaw (detuned saw stack)

### General MIDI Instruments (gm_* prefix)
**Bass**: gm_acoustic_bass, gm_electric_bass_finger, gm_electric_bass_pick, gm_fretless_bass, gm_slap_bass_1, gm_synth_bass_1, gm_synth_bass_2
**Piano/Keys**: gm_piano, gm_bright_acoustic_piano, gm_epiano1, gm_epiano2, gm_harpsichord, gm_clavinet, gm_celesta
**Leads**: gm_lead_1_square, gm_lead_2_sawtooth, gm_lead_3_calliope, gm_lead_5_charang, gm_lead_6_voice, gm_lead_7_fifths
**Pads**: gm_pad_new_age, gm_pad_warm, gm_pad_poly, gm_pad_choir, gm_string_ensemble_1, gm_synth_strings_1
**Brass**: gm_trumpet, gm_trombone, gm_alto_sax, gm_tenor_sax, gm_brass_section, gm_synth_brass_1
**Strings**: gm_violin, gm_viola, gm_cello, gm_contrabass, gm_orchestral_harp
**Percussion**: gm_glockenspiel, gm_music_box, gm_vibraphone, gm_marimba, gm_xylophone

### Drum Banks (use with .bank())
**Roland**: RolandTR808, RolandTR909, RolandTR707, RolandTR606, RolandCR78
**LinnDrum**: LinnDrum (classic 80s)
**Oberheim**: OberheimDMX (funk/disco)
**Alesis**: AlesisHR16 (80s/90s)
**Boss**: BossDR110 (lo-fi)
**Korg**: KorgKR55, KorgDDD1
**Sequential**: SequentialCircuits

### Sound Alternation (variety)
```javascript
.sound("<supersaw gm_synth_bass_1>")  // Alternates sounds each cycle
.bank("<RolandTR808 RolandTR909>")    // Alternates drum kits
```
"""

STRUDEL_EFFECTS = """
## STRUDEL EFFECTS

### Filter Effects
```javascript
.lpf(1000)                           // Low-pass filter (removes highs)
.hpf(200)                            // High-pass filter (removes lows)
.lpq(4)                              // Filter resonance (higher = more peak)
.lpenv(4)                            // Filter envelope depth
.vowel("a e i o u")                  // Vowel formant filter
```

### Envelope (ADSR)
```javascript
.attack(0.01)                        // Attack time (0.001-0.5s)
.decay(0.1)                          // Decay time (0.01-1.0s)
.sustain(0.5)                        // Sustain level (0-1)
.release(0.2)                        // Release time (0.01-2.0s)
```

### Space Effects
```javascript
.room(0.3)                           // Reverb amount (0-1)
.size(0.5)                           // Reverb size
.delay(0.25)                         // Delay wet mix (0-1)
.delaytime(0.125)                    // Delay time
.delayfeedback(0.5)                  // Delay feedback
.orbit(1)                            // Effect bus routing
```

### Distortion/Lo-Fi
```javascript
.crush(8)                            // Bit depth (1-16, lower = grittier)
.coarse(4)                           // Sample rate reduction
.distort(0.3)                        // Soft distortion
.shape(0.4)                          // Wave shaping
```

### Modulation
```javascript
.phaser(0.3)                         // Phaser effect
.vibrato(4, 0.5)                     // Vibrato (rate, depth)
.tremolo(8, 0.3)                     // Tremolo (rate, depth)
.pan(sine.slow(4))                   // Auto-panning
```

### Gain/Mix
```javascript
.gain(0.8)                           // Volume (0-2, >1 = boost)
.velocity(0.7)                       // Same as gain, MIDI-style
.pan(-0.5)                           // Stereo position (-1 to 1)
```
"""

STRUDEL_LIVE_CODING_TIPS = """
## LIVE CODING PATTERNS

### Interactive Template
```javascript
// Tempo control at top
setcps(slider(0.5, 0.3, 0.8, 0.01))

// Layer definitions with individual controls
let drums = s("bd*2 sd:3 [~ hh]*4")
  .bank("RolandTR808")
  .gain(slider(0.8, 0, 1))

let bass = note("c2 ~ e2 f2")
  .sound("gm_synth_bass_1")
  .lpf(slider(500, 100, 2000))
  .gain(slider(0.6, 0, 1))

let lead = note("c4 e4 g4 b4")
  .sound("supersaw")
  .lpf(sine.range(500, 4000).slow(8))
  .gain(slider(0.7, 0, 1))

// Main stack - comment/uncomment layers to toggle
$: stack(
  drums,
  bass,
  lead
)
```

### Variation Techniques
```javascript
// Probability-based variation
.sometimes(x => x.speed(2))
.rarely(x => x.rev())

// Pattern-based variation
.every(4, x => x.fast(2))            // Every 4 cycles, double speed
.every(8, x => x.add(7))             // Every 8 cycles, transpose up 5th

// Continuous variation with LFOs
.lpf(sine.range(400, 4000).slow(8))
.gain(perlin.range(0.3, 0.8).slow(16))
```

### Quick Transitions
```javascript
// Fade patterns in/out
.gain(slider(0, 0, 1, 0.01))         // Manual fade slider

// Breakdown/build with degradeBy
.degradeBy(slider(0.5, 0, 1))        // Drop random notes

// Filter sweep for builds
.lpf(slider(4000, 100, 8000))        // Sweep filter for tension/release
```
"""

def get_voice_prompt(voice_type: str, comparison_data: dict, current_fx: str) -> str:
    """
    Generate a focused prompt for a specific voice.

    Args:
        voice_type: "bass", "mid", "high", or "drums"
        comparison_data: Dict with band differences and similarity scores
        current_fx: Current effect function code for this voice

    Returns:
        Focused prompt string for this voice
    """

    # Extract voice-specific comparison data
    if voice_type == "bass":
        band_diff = comparison_data.get("band_bass", 0) + comparison_data.get("band_sub_bass", 0)
        freq_range = "20-250Hz"
        filter_advice = "Use .lpf(300-500) to stay below mid. Use .hpf(30-60) to remove sub rumble."
        sound_options = "gm_acoustic_bass, gm_electric_bass_finger, gm_synth_bass_1, gm_synth_bass_2, sawtooth"
        gain_range = "0.1-0.5 (bass should be felt, not dominate)"
    elif voice_type == "mid":
        band_diff = comparison_data.get("band_mid", 0) + comparison_data.get("band_low_mid", 0)
        freq_range = "250Hz-2kHz"
        filter_advice = "Use .hpf(200-400) to leave room for bass. Use .lpf(4000-6000) for warmth."
        sound_options = "gm_epiano1, gm_lead_2_sawtooth, gm_pad_warm, supersaw"
        gain_range = "0.5-1.5 (main melodic content)"
    elif voice_type == "high":
        band_diff = comparison_data.get("band_high", 0) + comparison_data.get("band_high_mid", 0)
        freq_range = "2kHz-10kHz"
        filter_advice = "Use .hpf(400-800) to stay above mids. Use .lpf(10000-15000) for air."
        sound_options = "gm_music_box, gm_vibraphone, gm_glockenspiel, triangle"
        gain_range = "0.3-0.8 (sparkle, not harsh)"
    else:  # drums
        band_diff = comparison_data.get("band_bass", 0)  # Drums mainly in bass
        freq_range = "varies (kick 40-100Hz, snare 150-500Hz, hats 4k-16kHz)"
        filter_advice = "Usually no filtering needed. Use .room(0.1-0.3) for glue."
        sound_options = "RolandTR808, RolandTR909, LinnDrum, OberheimDMX, AlesisHR16"
        gain_range = "0.6-1.0 (punchy but not overpowering)"

    status = "too loud" if band_diff > 0.05 else "too quiet" if band_diff < -0.05 else "balanced"

    return f"""You are a STRUDEL AUDIO ENGINEER specializing in {voice_type.upper()}.

## TASK: Fix the {voice_type} voice

### Current Status
The {voice_type} is **{status}** by {abs(band_diff)*100:.0f}%
Frequency range: {freq_range}

### Current Code
```javascript
{current_fx}
```

### Fix Guidelines
{filter_advice}
Gain range: {gain_range}
Recommended sounds: {sound_options}

### Interactive Controls (USE THESE!)
```javascript
// Add sliders for live adjustment:
.gain(slider({0.5 if voice_type != 'bass' else 0.3}, 0, 1, 0.01))
.lpf(slider(2000, 100, 8000, 10))
.room(slider(0.2, 0, 0.5, 0.01))

// Add variation with LFOs:
.lpf(sine.range(400, 4000).slow(8))
.gain(perlin.range(0.3, 0.7).slow(16))

// Add probability variation:
.sometimes(x => x.speed(2))
```

### Output Format
Return ONLY the fixed effect function line:
```javascript
let {voice_type}Fx = p => p.sound("...").gain(slider(...)).lpf(...)...
```
"""


def get_full_reference() -> str:
    """Get the complete Strudel reference documentation."""
    return (
        STRUDEL_INTERACTIVE_CONTROLS +
        STRUDEL_PATTERN_SYNTAX +
        STRUDEL_SOUND_LIBRARY +
        STRUDEL_EFFECTS +
        STRUDEL_LIVE_CODING_TIPS
    )


def build_per_voice_prompts(comparison_data: dict, current_code: str) -> dict:
    """
    Build separate prompts for each voice.

    Returns:
        Dict with keys: "bass", "mid", "high", "drums", "combine"
    """
    import re

    # Extract current effect functions
    fx_pattern = r'let\s+(\w+)Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
    fx_dict = {}
    for match in re.finditer(fx_pattern, current_code, re.MULTILINE):
        name = match.group(1)  # bass, mid, high, drums
        fx_dict[name] = match.group(0)

    prompts = {}

    for voice in ["bass", "mid", "high", "drums"]:
        current_fx = fx_dict.get(voice, f"let {voice}Fx = p => p")
        prompts[voice] = get_voice_prompt(voice, comparison_data, current_fx)

    # Combine prompt
    prompts["combine"] = f"""You are a STRUDEL MIX ENGINEER.

## TASK: Combine all voices into a cohesive live-coding friendly output

You have fixed each voice separately. Now combine them with:

1. **Tempo control at the top**:
```javascript
setcps(slider(0.5, 0.3, 0.8, 0.01))
```

2. **Layer definitions with sliders for each voice**

3. **Main stack with easy solo/mute structure**:
```javascript
$: stack(
  drums,
  bass,
  mid,
  high
)
```

4. **Comment hints for live performance**:
```javascript
// To solo drums: comment out bass, mid, high
// To filter sweep: adjust the lpf slider
// To breakdown: increase degradeBy slider
```

Output the COMPLETE Strudel code with all interactive controls.
"""

    return prompts


if __name__ == "__main__":
    # Print reference for testing
    print(get_full_reference())
