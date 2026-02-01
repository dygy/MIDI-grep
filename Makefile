.PHONY: build run test clean install-python-deps help

BINARY_NAME=midi-grep
BUILD_DIR=bin
PYTHON_VENV=scripts/python/.venv

# Build the Go binary
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/midi-grep

# Run the CLI with arguments
run: build
	./$(BUILD_DIR)/$(BINARY_NAME) $(ARGS)

# Run tests
test:
	go test -v ./...

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PYTHON_VENV)

# Install Python dependencies in a virtual environment
install-python-deps:
	@echo "Creating Python virtual environment..."
	python3 -m venv $(PYTHON_VENV)
	@echo "Installing Python dependencies..."
	$(PYTHON_VENV)/bin/pip install -r scripts/python/requirements.txt
	@echo "Done! Activate with: source $(PYTHON_VENV)/bin/activate"

# Install all dependencies (Go + Python)
deps: install-python-deps
	go mod download

# Build and install to $GOPATH/bin
install: build
	go install ./cmd/midi-grep

# Start web server
serve: build
	./$(BUILD_DIR)/$(BINARY_NAME) serve --port 8080

# Run example extraction
example: build
	@echo "Running example (provide ARGS='--input your-file.wav')..."
	./$(BUILD_DIR)/$(BINARY_NAME) extract $(ARGS)

# Extract from YouTube URL
youtube: build
	@echo "Extracting from YouTube (provide URL='https://...')..."
	./$(BUILD_DIR)/$(BINARY_NAME) extract --url "$(URL)"

# Show help
help:
	@echo "MIDI-grep Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build                Build the Go binary"
	@echo "  make serve                Start web server on port 8080"
	@echo "  make run ARGS='...'       Build and run with arguments"
	@echo "  make youtube URL='...'    Extract from YouTube URL"
	@echo "  make test                 Run tests"
	@echo "  make clean                Remove build artifacts"
	@echo "  make install-python-deps  Install Python dependencies in venv"
	@echo "  make deps                 Install all dependencies"
	@echo "  make install              Install to GOPATH/bin"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make serve"
	@echo "  make run ARGS='extract --input track.wav'"
	@echo "  make youtube URL='https://youtube.com/watch?v=...'"
