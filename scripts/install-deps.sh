#!/bin/bash
# Install all dependencies for MIDI-grep
#
# Usage: ./scripts/install-deps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$SCRIPT_DIR/python"

echo "=== MIDI-grep Dependency Installer ==="
echo ""

# Check for Python 3.11+
echo "Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON="python3.11"
elif command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    PYTHON="/opt/homebrew/bin/python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    VERSION=$($PYTHON -c 'import sys; print(sys.version_info.minor)')
    if [[ "$VERSION" -lt 10 ]]; then
        echo "Warning: Python 3.10+ recommended. You have Python 3.$VERSION"
        echo "Install Python 3.11 with: brew install python@3.11"
    fi
else
    echo "Error: Python 3 not found. Install with: brew install python@3.11"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
rm -rf "$PYTHON_DIR/.venv"
$PYTHON -m venv "$PYTHON_DIR/.venv"

# Activate and install
echo "Installing Python dependencies..."
source "$PYTHON_DIR/.venv/bin/activate"
pip install --upgrade pip wheel setuptools > /dev/null

echo "  - Installing numpy..."
pip install numpy==1.26.4 > /dev/null

echo "  - Installing librosa, pretty_midi..."
pip install librosa pretty_midi soundfile > /dev/null

echo "  - Installing basic-pitch..."
pip install basic-pitch > /dev/null

echo "  - Installing demucs..."
pip install demucs > /dev/null

deactivate

# Check for yt-dlp
echo ""
echo "Checking yt-dlp..."
if command -v yt-dlp &> /dev/null; then
    echo "  yt-dlp: $(yt-dlp --version)"
else
    echo "  yt-dlp not found. Install with: brew install yt-dlp"
    echo "  (Required for YouTube URL support)"
fi

# Build Go binary
echo ""
echo "Building Go binary..."
cd "$PROJECT_DIR"
go build -o bin/midi-grep ./cmd/midi-grep

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Usage:"
echo "  ./scripts/extract-youtube.sh 'https://youtu.be/VIDEO_ID'"
echo "  ./scripts/extract-file.sh track.wav"
echo "  ./scripts/serve.sh 8080"
