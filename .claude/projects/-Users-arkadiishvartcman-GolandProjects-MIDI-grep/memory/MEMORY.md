# MIDI-grep AI Synthesis Learnings

## Key Breakthrough: Soft Saturation

**Using tanh soft clipping improved similarity from 71% to 85.6%**

The saturation does three things:
1. **Limits peaks naturally** - No hard limiter needed
2. **Adds harmonics** - Improves MFCC match with recorded audio
3. **Raises average RMS** - Better energy/loudness match

### Implementation
```typescript
// Drive into saturation
for (let i = 0; i < output.length; i++) {
  output[i] *= saturationDrive;  // 3.0x
}
// Soft clip with tanh
for (let i = 0; i < output.length; i++) {
  output[i] = Math.tanh(output[i]);
}
```

## Results Comparison

| Metric | Before Saturation | After Saturation |
|--------|-------------------|------------------|
| Overall | 71% | **85.6%** |
| MFCC (Timbre) | 42% | **72.2%** |
| Energy | 37% | **94.8%** |
| Brightness | 72% | **76.9%** |
| Freq Balance | 99% | 94.9% |

## Key Insights

### What Works
1. **Saturation** - Most important for energy and MFCC
2. **AI-derived gains** from original audio frequency analysis
3. **FM synthesis** for richer harmonics
4. **High-shelf boost** for brightness matching
5. **RMS normalization** after saturation

### Code Generation Improvements
- Filter sparse patterns (bars with mostly rests)
- Deduplicate consecutive similar bars
- Require 2+ notes per bar for meaningful patterns

## Files Modified
- `scripts/node/src/render-strudel-node.ts` - Added saturation, FM synthesis
- `scripts/python/analyze_synth_params.py` - AI-derived parameters
- `internal/strudel/generator.go` - Pattern filtering and deduplication
