#!/bin/bash
# Extract piano riff from YouTube URL and generate Strudel.cc code
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

# Check arguments
if [[ -z "$1" ]]; then
    echo "Usage: $0 <youtube-url> [output-file]"
    echo ""
    echo "Examples:"
    echo "  $0 'https://youtu.be/Q4801HzWZfg'"
    echo "  $0 'https://youtube.com/watch?v=dQw4w9WgXcQ' riff.strudel"
    exit 1
fi

URL="$1"
OUTPUT="$2"

# Run extraction
if [[ -n "$OUTPUT" ]]; then
    "$BINARY" extract --url "$URL" --output "$OUTPUT" --quantize 16
    echo ""
    echo "Saved to: $OUTPUT"
else
    "$BINARY" extract --url "$URL" --quantize 16
fi
