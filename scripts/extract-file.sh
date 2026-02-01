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

# Get input file (from argument or prompt)
INPUT="$1"
if [[ -z "$INPUT" ]]; then
    echo "MIDI-grep: Extract piano from audio file"
    echo ""
    read -p "Enter audio file path: " INPUT
    if [[ -z "$INPUT" ]]; then
        echo "Error: File path is required"
        exit 1
    fi
fi

# Check if input file exists
if [[ ! -f "$INPUT" ]]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

# Get output file (from argument or prompt)
OUTPUT="$2"
if [[ -z "$OUTPUT" ]]; then
    read -p "Output file (press Enter for stdout): " OUTPUT
fi

# Run extraction
if [[ -n "$OUTPUT" ]]; then
    "$BINARY" extract --input "$INPUT" --output "$OUTPUT" --quantize 16
    echo ""
    echo "Saved to: $OUTPUT"
else
    "$BINARY" extract --input "$INPUT" --quantize 16
fi
