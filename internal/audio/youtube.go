package audio

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// YouTubeDownloader handles downloading audio from YouTube
type YouTubeDownloader struct{}

// NewYouTubeDownloader creates a new YouTube downloader
func NewYouTubeDownloader() *YouTubeDownloader {
	return &YouTubeDownloader{}
}

// IsYouTubeURL checks if the given string is a YouTube URL
func IsYouTubeURL(url string) bool {
	patterns := []string{
		`^https?://(www\.)?youtube\.com/watch\?v=[\w-]+`,
		`^https?://(www\.)?youtube\.com/shorts/[\w-]+`,
		`^https?://youtu\.be/[\w-]+`,
		`^https?://music\.youtube\.com/watch\?v=[\w-]+`,
	}

	for _, pattern := range patterns {
		if matched, _ := regexp.MatchString(pattern, url); matched {
			return true
		}
	}
	return false
}

// Download downloads audio from a YouTube URL using yt-dlp
func (d *YouTubeDownloader) Download(ctx context.Context, url, outputDir string) (string, error) {
	// Check if yt-dlp is installed
	if err := d.checkYtDlp(); err != nil {
		return "", err
	}

	outputPath := filepath.Join(outputDir, "input.%(ext)s")

	// Download best audio and convert to wav
	cmd := exec.CommandContext(ctx, "yt-dlp",
		"--no-playlist",           // Only download single video
		"--extract-audio",         // Extract audio only
		"--audio-format", "wav",   // Convert to WAV
		"--audio-quality", "0",    // Best quality
		"--output", outputPath,    // Output path template
		"--no-warnings",           // Suppress warnings
		"--quiet",                 // Quiet mode
		"--progress",              // But show progress
		url,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		// Try with mp3 if wav fails
		return d.downloadAsMp3(ctx, url, outputDir)
	}

	// Find the output file
	wavPath := filepath.Join(outputDir, "input.wav")
	return wavPath, nil
}

// downloadAsMp3 fallback to mp3 download
func (d *YouTubeDownloader) downloadAsMp3(ctx context.Context, url, outputDir string) (string, error) {
	outputPath := filepath.Join(outputDir, "input.%(ext)s")

	cmd := exec.CommandContext(ctx, "yt-dlp",
		"--no-playlist",
		"--extract-audio",
		"--audio-format", "mp3",
		"--audio-quality", "0",
		"--output", outputPath,
		"--no-warnings",
		url,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("yt-dlp failed: %w (stderr: %s)", err, stderr.String())
	}

	mp3Path := filepath.Join(outputDir, "input.mp3")
	return mp3Path, nil
}

// checkYtDlp verifies yt-dlp is installed
func (d *YouTubeDownloader) checkYtDlp() error {
	cmd := exec.Command("yt-dlp", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("yt-dlp not installed. Install with: brew install yt-dlp (macOS) or pip install yt-dlp")
	}
	return nil
}

// GetVideoTitle fetches the video title for display
func (d *YouTubeDownloader) GetVideoTitle(ctx context.Context, url string) (string, error) {
	cmd := exec.CommandContext(ctx, "yt-dlp",
		"--get-title",
		"--no-warnings",
		url,
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		return "", err
	}

	title := strings.TrimSpace(stdout.String())
	if title == "" {
		return "YouTube Video", nil
	}

	// Truncate if too long
	if len(title) > 50 {
		title = title[:47] + "..."
	}

	return title, nil
}
