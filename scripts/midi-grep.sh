#!/bin/bash
#
# MIDI-grep - Extract piano riffs from audio and generate Strudel.cc code
#
# Usage: ./scripts/midi-grep.sh [command] [options]
#
# Commands:
#   extract     Extract piano from audio/YouTube and generate Strudel code
#   serve       Start the web interface
#   install     Install Python dependencies
#   help        Show this help message
#
# Examples:
#   ./scripts/midi-grep.sh extract --url "https://youtu.be/VIDEO_ID"
#   ./scripts/midi-grep.sh extract --file track.wav --output riff.strudel
#   ./scripts/midi-grep.sh serve --port 3000
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/midi-grep"
PYTHON_VENV="$SCRIPT_DIR/python/.venv"

# Default values
QUANTIZE=16
PORT=8080
OUTPUT=""
VERBOSE=false
COPY_CLIPBOARD=false

# Print colored output
info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# Show help
show_help() {
    cat << 'EOF'
MIDI-grep - Extract piano riffs from audio and generate Strudel.cc code

USAGE:
    midi-grep.sh <command> [options]

COMMANDS:
    extract     Extract piano from audio/YouTube URL
    serve       Start the web interface
    install     Install Python dependencies
    help        Show this help message

EXTRACT OPTIONS:
    -u, --url <url>         YouTube URL to extract from
    -f, --file <path>       Local audio file (WAV/MP3)
    -o, --output <path>     Output file for Strudel code (default: stdout)
    -q, --quantize <4|8|16> Quantization level (default: 16)
    -m, --midi <path>       Also save cleaned MIDI file
    -c, --copy              Copy result to clipboard (macOS)
    -v, --verbose           Show verbose output

SERVE OPTIONS:
    -p, --port <port>       Port to listen on (default: 8080)

EXAMPLES:
    # Extract from YouTube
    midi-grep.sh extract --url "https://youtu.be/Q4801HzWZfg"

    # Extract with options
    midi-grep.sh extract -u "https://youtu.be/VIDEO" -q 8 -o riff.strudel

    # Extract from local file and copy to clipboard
    midi-grep.sh extract --file track.wav --copy

    # Start web server on port 3000
    midi-grep.sh serve --port 3000

EOF
}

# Ensure binary is built
ensure_binary() {
    if [[ ! -f "$BINARY" ]]; then
        info "Building midi-grep..."
        cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep
        success "Binary built"
    fi
}

# Check dependencies
check_deps() {
    local missing=()

    if ! command -v go &> /dev/null; then
        missing+=("go")
    fi

    if [[ ! -d "$PYTHON_VENV" ]]; then
        missing+=("python-deps (run: $0 install)")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing dependencies: ${missing[*]}"
    fi
}

# Install Python dependencies
cmd_install() {
    info "Installing MIDI-grep dependencies..."

    # Find Python 3.11+
    local PYTHON=""
    for py in python3.11 /opt/homebrew/bin/python3.11 python3.12 python3; do
        if command -v "$py" &> /dev/null; then
            PYTHON="$py"
            break
        fi
    done

    if [[ -z "$PYTHON" ]]; then
        error "Python 3.10+ not found. Install with: brew install python@3.11"
    fi

    info "Using Python: $($PYTHON --version)"

    # Create venv
    info "Creating virtual environment..."
    rm -rf "$PYTHON_VENV"
    $PYTHON -m venv "$PYTHON_VENV"

    # Install packages
    source "$PYTHON_VENV/bin/activate"
    pip install --upgrade pip wheel setuptools > /dev/null

    info "Installing numpy..."
    pip install numpy==1.26.4 > /dev/null

    info "Installing librosa, pretty_midi..."
    pip install librosa pretty_midi soundfile > /dev/null

    info "Installing basic-pitch..."
    pip install basic-pitch > /dev/null

    info "Installing demucs..."
    pip install demucs > /dev/null

    deactivate

    # Check yt-dlp
    if ! command -v yt-dlp &> /dev/null; then
        warn "yt-dlp not found. Install for YouTube support: brew install yt-dlp"
    else
        success "yt-dlp found: $(yt-dlp --version)"
    fi

    # Build binary
    info "Building Go binary..."
    cd "$PROJECT_DIR" && go build -o bin/midi-grep ./cmd/midi-grep

    success "Installation complete!"
    echo ""
    echo "Usage:"
    echo "  $0 extract --url 'https://youtu.be/VIDEO_ID'"
    echo "  $0 extract --file track.wav"
    echo "  $0 serve"
}

# Extract command
cmd_extract() {
    local URL=""
    local FILE=""
    local MIDI_OUT=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--url)
                URL="$2"
                shift 2
                ;;
            -f|--file)
                FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT="$2"
                shift 2
                ;;
            -q|--quantize)
                QUANTIZE="$2"
                shift 2
                ;;
            -m|--midi)
                MIDI_OUT="$2"
                shift 2
                ;;
            -c|--copy)
                COPY_CLIPBOARD=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done

    # Validate
    if [[ -z "$URL" && -z "$FILE" ]]; then
        error "Either --url or --file is required"
    fi

    if [[ "$QUANTIZE" != "4" && "$QUANTIZE" != "8" && "$QUANTIZE" != "16" ]]; then
        error "Quantize must be 4, 8, or 16"
    fi

    check_deps
    ensure_binary

    # Build command
    local CMD=("$BINARY" "extract" "--quantize" "$QUANTIZE")

    if [[ -n "$URL" ]]; then
        CMD+=("--url" "$URL")
    else
        if [[ ! -f "$FILE" ]]; then
            error "File not found: $FILE"
        fi
        CMD+=("--input" "$FILE")
    fi

    if [[ -n "$OUTPUT" ]]; then
        CMD+=("--output" "$OUTPUT")
    fi

    if [[ -n "$MIDI_OUT" ]]; then
        CMD+=("--midi-out" "$MIDI_OUT")
    fi

    if [[ "$VERBOSE" == true ]]; then
        CMD+=("--verbose")
    fi

    # Run extraction
    if [[ "$COPY_CLIPBOARD" == true && -z "$OUTPUT" ]]; then
        # Capture output and copy to clipboard
        local RESULT
        RESULT=$("${CMD[@]}" 2>&1)
        echo "$RESULT"

        # Extract just the Strudel code
        echo "$RESULT" | sed -n '/^\/\/ MIDI-grep/,/\.size/p' | pbcopy
        success "Strudel code copied to clipboard!"
    else
        "${CMD[@]}"
    fi
}

# Serve command
cmd_serve() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done

    check_deps
    ensure_binary

    info "Starting MIDI-grep web server on port $PORT..."
    echo ""
    "$BINARY" serve --port "$PORT"
}

# Main
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    local COMMAND="$1"
    shift

    case "$COMMAND" in
        extract)
            cmd_extract "$@"
            ;;
        serve)
            cmd_serve "$@"
            ;;
        install)
            cmd_install "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $COMMAND (try: $0 help)"
            ;;
    esac
}

main "$@"
