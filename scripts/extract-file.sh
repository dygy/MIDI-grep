#!/bin/bash
# Extract piano riff from audio file and generate Strudel.cc code
#
# Usage: ./scripts/extract-file.sh <audio-file> [output-file]
# Example: ./scripts/extract-file.sh track.wav riff.strudel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/midi-grep"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Building midi-grep..."
    cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep
fi

# Check arguments
if [[ -z "$1" ]]; then
    echo "Usage: $0 <audio-file> [output-file]"
    echo ""
    echo "Examples:"
    echo "  $0 track.wav"
    echo "  $0 song.mp3 riff.strudel"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

# Check if input file exists
if [[ ! -f "$INPUT" ]]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

# Run extraction
if [[ -n "$OUTPUT" ]]; then
    "$BINARY" extract --input "$INPUT" --output "$OUTPUT" --quantize 16
    echo ""
    echo "Saved to: $OUTPUT"
else
    "$BINARY" extract --input "$INPUT" --quantize 16
fi
