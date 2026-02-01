# Multi-stage Dockerfile for MIDI-grep
# Stage 1: Build Go binary
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build -o midi-grep ./cmd/midi-grep

# Stage 2: Runtime with Python
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python scripts and install dependencies
COPY scripts/python/requirements.txt ./scripts/python/
RUN pip install --no-cache-dir -r scripts/python/requirements.txt

COPY scripts/python/*.py ./scripts/python/
RUN chmod +x ./scripts/python/*.py

# Copy Go binary from builder
COPY --from=builder /app/midi-grep /usr/local/bin/midi-grep

# Create working directory for file processing
RUN mkdir -p /data
WORKDIR /data

ENTRYPOINT ["midi-grep"]
CMD ["--help"]
