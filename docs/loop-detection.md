# Loop Detection in MIDI-grep

## Overview

The loop detection feature automatically identifies repeating musical patterns in transcribed MIDI files. This is useful for live coding and looping, as it extracts the core repeating riff from a track.

## How It Works

### 1. Pattern Analysis

The system tests for loops of different lengths:
- **1 bar** (4 beats)
- **2 bars** (8 beats)
- **4 bars** (16 beats)

For each length, it divides the MIDI track into chunks and compares them for similarity.

### 2. Similarity Calculation

To determine if chunks are similar, the algorithm uses a weighted combination of:

- **Sequence similarity (60%)**: Time-binned note matching
  - Groups notes into 50ms time bins
  - Compares which pitches occur at similar times
  - Most important factor for loop detection

- **Pitch set similarity (30%)**: Jaccard index of pitches
  - Measures overlap of note pitches used
  - Ensures musical content is similar

- **Note count ratio (10%)**: Relative number of notes
  - Ensures patterns have similar density
  - Less important than actual content

### 3. Confidence Scoring

- Each loop candidate gets a confidence score (0.0 to 1.0)
- Minimum threshold: **0.65** (65% similarity)
- Higher scores indicate more consistent repetition
- The pattern with the highest confidence is selected

### 4. Output Format

When a loop is detected, the JSON output includes:

```json
{
  "loop": {
    "detected": true,
    "bars": 2,
    "confidence": 0.85,
    "start_beat": 0,
    "end_beat": 8,
    "notes": [...],
    "repetitions": 4
  }
}
```

If no loop is detected:

```json
{
  "loop": {
    "detected": false,
    "bars": 0,
    "confidence": 0.0,
    "start_beat": 0,
    "end_beat": 0,
    "notes": []
  }
}
```

## Parameters

The loop detection runs automatically during MIDI cleanup. It uses:

- **Tempo**: From MIDI analysis (BPM)
- **Beat duration**: Calculated from tempo
- **Time signature**: Assumes 4/4 (can be extended)
- **Minimum repetitions**: 2 (needs at least 2 loops to detect)

## Use Cases

1. **Live Coding**: Extract the core loop for Strudel patterns
2. **Sampling**: Identify the cleanest repeat for sampling
3. **Analysis**: Understand song structure and repetition
4. **Auto-looping**: Set loop points automatically

## Limitations

- Assumes 4/4 time signature
- Requires at least 2 full repetitions
- May miss loops with significant variation
- Works best with quantized, clean MIDI data

## Future Enhancements

- Support for different time signatures (3/4, 6/8, etc.)
- Detection of loops with variations (A-B-A-B patterns)
- Multiple simultaneous loops in different voices
- Swing and groove quantization awareness
