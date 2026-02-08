# Loop Detection in MIDI-grep

## Overview

The loop detection feature automatically identifies repeating musical patterns in transcribed MIDI files. This is useful for live coding and looping, as it extracts the core repeating riff from a track.

## How It Works

### 1. Pattern Analysis

The system tests for loops of different lengths:
- **1 bar** (4 beats)
- **2 bars** (8 beats)
- **4 bars** (16 beats)
- **8 bars** (32 beats)

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
- Minimum threshold: **0.45** (45% similarity)
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

## Implemented Features (Feb 2026)

All planned enhancements are now implemented:

| Feature | Status | Implementation |
|---------|--------|----------------|
| **Time Signature Support** | ✅ Done | `get_beats_per_bar()` handles 3/4, 6/8, 5/4, 7/8 |
| **Consensus-Based Reference** | ✅ Done | `_find_consensus_pattern()` builds NxN similarity matrix |
| **A-B-A-B Detection** | ✅ Done | `_detect_alternating_pattern()` for verse-chorus patterns |
| **Swing Awareness** | ✅ Done | `_calculate_similarity()` adjusts bin size for swing |
| **Multi-Voice Loops** | ✅ Done | `detect_loops_per_voice()` for bass/mid/high separately |

### Time Signature Support

```python
def get_beats_per_bar(time_sig: str) -> int:
    return {"4/4": 4, "3/4": 3, "2/4": 2, "6/8": 2, "5/4": 5, "7/8": 7}.get(time_sig, 4)
```

### Consensus Reference Selection

Instead of always using first iteration (fails on intro variations), we build NxN similarity matrix:

```python
def _find_consensus_pattern(iterations, swing_ratio):
    # Build similarity matrix
    sim_matrix = [[_calculate_similarity(i, j) for j in iterations] for i in iterations]
    # Pick iteration with highest total similarity
    totals = [sum(row) for row in sim_matrix]
    best_idx = max(range(len(totals)), key=lambda i: totals[i])
    return iterations[best_idx], best_idx
```

### Swing Awareness

Time bins expand for swung rhythms:
```python
if swing_ratio > 1.1:
    bin_size = base_bin_size * (1 + (swing_ratio - 1) * 0.5)
    # Light swing (1.2): 62.5ms, Heavy swing (2.0): 100ms
```

### Go Integration

Time signature and swing are passed from analysis to cleanup:
```go
cleanOpts.TimeSignature = analysisResult.TimeSignature
cleanOpts.SwingRatio = analysisResult.SwingRatio
```
