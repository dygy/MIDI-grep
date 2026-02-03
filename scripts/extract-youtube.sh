#!/bin/bash
# Extract music from YouTube URL and generate Strudel code with full report
#
# Usage: ./scripts/extract-youtube.sh [youtube-url]
# Example: ./scripts/extract-youtube.sh "https://youtu.be/Q4801HzWZfg"

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
    echo "MIDI-grep: YouTube to Strudel"
    echo ""
    read -p "YouTube URL: " URL
    if [[ -z "$URL" ]]; then
        echo "Error: URL is required"
        exit 1
    fi
fi

# Run extraction with everything enabled
# - render auto: generates audio preview
# - comparison chart: auto-generated with render
# - HTML report: auto-generated
# - cache: stores everything in versioned directories
"$BINARY" extract --url "$URL" --render auto
