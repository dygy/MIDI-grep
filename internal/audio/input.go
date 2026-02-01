package audio

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	apperrors "github.com/arkadiishvartcman/midi-grep/internal/errors"
)

const (
	MaxFileSize = 100 * 1024 * 1024 // 100MB
)

// Magic bytes for audio format detection
var (
	wavMagic  = []byte("RIFF")
	mp3Magic1 = []byte{0xFF, 0xFB} // MPEG Audio Layer 3
	mp3Magic2 = []byte{0xFF, 0xFA}
	mp3Magic3 = []byte{0xFF, 0xF3}
	mp3Magic4 = []byte{0xFF, 0xF2}
	id3Magic  = []byte("ID3") // MP3 with ID3 tag
)

// Format represents an audio file format
type Format string

const (
	FormatWAV     Format = "wav"
	FormatMP3     Format = "mp3"
	FormatUnknown Format = "unknown"
)

// ValidateInput checks if the input file is valid for processing
func ValidateInput(path string) (Format, error) {
	// Check file exists
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return FormatUnknown, fmt.Errorf("%w: %s", apperrors.ErrFileNotFound, path)
	}
	if err != nil {
		return FormatUnknown, fmt.Errorf("stat file: %w", err)
	}

	// Check file size
	if info.Size() > MaxFileSize {
		return FormatUnknown, fmt.Errorf("%w: maximum size is 100MB", apperrors.ErrFileTooLarge)
	}

	// Check format by magic bytes
	format, err := detectFormat(path)
	if err != nil {
		return FormatUnknown, err
	}

	if format == FormatUnknown {
		return FormatUnknown, fmt.Errorf("%w: please provide a WAV or MP3 file", apperrors.ErrUnsupportedFormat)
	}

	return format, nil
}

// detectFormat checks file magic bytes to determine audio format
func detectFormat(path string) (Format, error) {
	f, err := os.Open(path)
	if err != nil {
		return FormatUnknown, fmt.Errorf("%w: %v", apperrors.ErrCorruptedFile, err)
	}
	defer f.Close()

	// Read first 12 bytes for magic detection
	header := make([]byte, 12)
	n, err := f.Read(header)
	if err != nil || n < 4 {
		return FormatUnknown, fmt.Errorf("%w: could not read file header", apperrors.ErrCorruptedFile)
	}

	// Check WAV (RIFF....WAVE)
	if string(header[:4]) == "RIFF" && n >= 12 && string(header[8:12]) == "WAVE" {
		return FormatWAV, nil
	}

	// Check MP3 with ID3 tag
	if string(header[:3]) == "ID3" {
		return FormatMP3, nil
	}

	// Check MP3 frame sync
	if len(header) >= 2 {
		if (header[0] == 0xFF && (header[1]&0xE0) == 0xE0) {
			return FormatMP3, nil
		}
	}

	// Fallback: check extension
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".wav":
		return FormatWAV, nil
	case ".mp3":
		return FormatMP3, nil
	}

	return FormatUnknown, nil
}
