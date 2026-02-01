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

INPUT="$1"
if [[ -z "$INPUT" ]]; then
    echo "Quick Riff - Extract Strudel code" >&2
    echo "" >&2
    read -p "Enter YouTube URL or file path: " INPUT
    if [[ -z "$INPUT" ]]; then
        echo "Error: Input is required" >&2
        exit 1
    fi
fi

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
