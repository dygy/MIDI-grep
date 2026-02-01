#!/bin/bash
# Start the MIDI-grep web server
#
# Usage: ./scripts/serve.sh [port]
# Example: ./scripts/serve.sh 8080

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/midi-grep"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Building midi-grep..."
    cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep
fi

PORT="${1:-8080}"

echo "Starting MIDI-grep web server..."
echo "Open http://localhost:$PORT in your browser"
echo ""

"$BINARY" serve --port "$PORT"
