# MIDI-grep - Product Summary

## Vision

Enable musicians and live coders to instantly extract musical content from audio and transform it into playable Strudel patterns. Go-powered CLI + HTMX web app, no JavaScript frameworks.

## Target Audience

- Live coders (Strudel/TidalCycles performers)
- Music producers learning songs
- Hobbyist musicians practicing riffs
- Music educators

## Core Features

- **Audio Input** - WAV/MP3/YouTube URL via CLI or web upload
- **Stem Separation** - Melodic/bass/drums/vocals via Demucs
- **MIDI Transcription** - Audio-to-MIDI via Basic Pitch
- **MIDI Cleanup** - Quantization, noise removal, loop detection
- **BPM & Key Detection** - Tempo, scale, time signature analysis
- **Strudel Generator** - Bar arrays + effect functions for bass/mid/high/drums
- **Audio Rendering** - Node.js renderer with stem output
- **AI Improvement** - LLM-driven iterative code optimization (Ollama/Claude)
- **Audio Comparison** - MAE-based similarity scoring (frequency bands, MFCC, energy)
- **DAW-Style Reports** - HTML reports with isolated Original/Rendered stem players
- **HTMX Web UI** - Reactive, zero-JS frontend with Go templates
- **CLI Interface** - Full-featured terminal workflow

## Current State (Feb 2026)

**Implemented:**
- All stem separation and transcription
- 67 drum machines + 128 GM instruments + 17 genre palettes
- Per-stem comparison charts
- AI-driven improvement (5 iterations, ClickHouse learning storage)
- Accurate similarity scoring (MAE-based, not cosine)
- Self-contained HTML reports with audio studio

**In Progress:**
- Loop detection improvements (time signature support, consensus-based reference)
- Higher similarity scores through synthesis improvements

## Key Metrics

- Similarity: ~60-70% with honest calculation (freq balance weighted 40%)
- Processing: YouTube URL to Strudel code in ~2 minutes
- Rendering: Full track audio in ~30 seconds
