#!/bin/bash
# Quick extraction with simplified output - just the Strudel code
#
# Usage: ./scripts/quick-riff.sh <youtube-url-or-file>
# Example: ./scripts/quick-riff.sh "https://youtu.be/Q4801HzWZfg"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/midi-grep"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Building midi-grep..." >&2
    cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep
fi

if [[ -z "$1" ]]; then
    echo "Usage: $0 <youtube-url-or-file>" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 'https://youtu.be/Q4801HzWZfg'" >&2
    echo "  $0 track.wav" >&2
    echo "" >&2
    echo "Pipe to clipboard:" >&2
    echo "  $0 'https://youtu.be/...' | pbcopy" >&2
    exit 1
fi

INPUT="$1"

# Detect if URL or file
if [[ "$INPUT" =~ ^https?:// ]]; then
    # YouTube URL - extract and filter to just the code
    "$BINARY" extract --url "$INPUT" --quantize 8 2>&1 | \
        sed -n '/^\/\/ MIDI-grep/,/^Done!/p' | \
        grep -v "^Done!"
else
    # Local file
    if [[ ! -f "$INPUT" ]]; then
        echo "Error: File not found: $INPUT" >&2
        exit 1
    fi
    "$BINARY" extract --input "$INPUT" --quantize 8 2>&1 | \
        sed -n '/^\/\/ MIDI-grep/,/^Done!/p' | \
        grep -v "^Done!"
fi
