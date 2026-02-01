# MIDI-grep - Product Summary

## Vision

Enable musicians and live coders to instantly extract piano riffs from audio and transform them into playable Strudel.cc patterns. Go-powered CLI + HTMX web app, no JavaScript.

## Target Audience

- Live coders (Strudel.cc/TidalCycles performers)
- Music producers learning songs
- Hobbyist musicians practicing riffs
- Music educators

## Core Features

- **Audio Input** - WAV/MP3 upload (web) or CLI argument
- **Stem Separation** - Piano isolation via Spleeter/Demucs
- **MIDI Transcription** - Audio-to-MIDI via Basic Pitch
- **MIDI Cleanup** - Quantization, noise removal, loop detection
- **BPM & Key Detection** - Tempo and scale analysis
- **Strudel Generator** - Output playable `note()` patterns
- **HTMX Web UI** - Reactive, zero-JS frontend with Go templates
- **CLI Interface** - Terminal-based workflow

## V1 Boundaries

**In:** CLI, Web UI, piano extraction, MIDI cleanup, Strudel output, Docker deploy

**Out:** YouTube download, other instruments, user accounts, multiple output formats
