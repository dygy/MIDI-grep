# MIDI-grep AI Synthesis Learnings

## Key Insights

### Synthesis Quality Limits
- **MFCC ceiling**: ~42-44% with pure synthesis (fundamental limit)
- Pure waveforms (saw, square, sine) cannot match recorded audio MFCC
- FM synthesis helps (+1-2% MFCC) by producing richer harmonics
- High-shelf brightness boost trades off against MFCC quality

### What Works
1. **Analyze ORIGINAL audio** (not stems) for frequency balance/gains
2. **Always use saw waveform** for melodic voices (harmonics needed)
3. **AI-derived tempo tolerance** based on beat regularity
4. **FM synthesis** for richer timbre (enables always)
5. **Gentle high-shelf boost** (~0.7dB at 2kHz) for brightness

### What Doesn't Help
- Formant filters: Hurt frequency balance more than help timbre
- Additive synthesis with harmonic profile: No better than saw wave
- Aggressive high-shelf boost: Improves brightness but hurts MFCC

### Best Results Achieved
- **72.2% overall similarity** (v022)
- 99.5% frequency balance
- 72% brightness match
- 42% MFCC (synthesis limitation)

## To Reach 85%+
Need sample-based or neural synthesis:
1. Granular synthesis using actual stem audio grains
2. RAVE neural network models
3. Wavetable synthesis from original's spectral content

## Key Files
- `analyze_synth_params.py`: AI-driven parameter extraction
- `render-strudel-node.ts`: TypeScript renderer with FM synthesis
- `compare_audio.py`: Similarity comparison with AI-derived tolerance
