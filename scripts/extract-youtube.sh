#!/bin/bash
# Extract piano riff from YouTube URL and generate Strudel code
#
# Usage: ./scripts/extract-youtube.sh <youtube-url> [output-file]
# Example: ./scripts/extract-youtube.sh "https://youtu.be/Q4801HzWZfg" riff.strudel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/midi-grep"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Building midi-grep..."
    cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep
fi

# Get URL (from argument or prompt)
URL="$1"
if [[ -z "$URL" ]]; then
    echo "MIDI-grep: Extract piano from YouTube"
    echo ""
    read -p "Enter YouTube URL: " URL
    if [[ -z "$URL" ]]; then
        echo "Error: URL is required"
        exit 1
    fi
fi

# Get output file (from argument or prompt)
OUTPUT="$2"
if [[ -z "$OUTPUT" ]]; then
    read -p "Output file (press Enter for stdout): " OUTPUT
fi

# Run extraction
if [[ -n "$OUTPUT" ]]; then
    "$BINARY" extract --url "$URL" --output "$OUTPUT" --quantize 16
    echo ""
    echo "Saved to: $OUTPUT"
else
    "$BINARY" extract --url "$URL" --quantize 16
fi
