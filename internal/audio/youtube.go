package audio

import (
	"bytes"
	"context"
	"fmt"
	"os"
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

// ytDlpBinary resolves which yt-dlp to run. YouTube breaks old yt-dlp builds every few months
// (HTTP 403 on every format), and the Homebrew binary on PATH can lag the venv's pip-installed one
// by half a year — the first run on a new track on 2026-10-10 failed exactly that way. Order:
//  1. MIDIGREP_YTDLP (explicit path),
//  2. the project venv's yt-dlp (scripts/python/.venv/bin/yt-dlp, resolved from the working
//     directory, then from the directory the binary lives in, i.e. bin/../),
//  3. "yt-dlp" on PATH.
func ytDlpBinary() string {
	if p := os.Getenv("MIDIGREP_YTDLP"); p != "" {
		return p
	}
	candidates := []string{}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, "scripts", "python", ".venv", "bin", "yt-dlp"))
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "..", "scripts", "python", ".venv", "bin", "yt-dlp"))
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return "yt-dlp"
}

// ytDlpAuthArgs returns the cookie flags for every yt-dlp call. YouTube answers some videos (and
// some networks) with "Sign in to confirm you're not a bot" unless the request carries a signed-in
// browser's cookies — on 2026-10-30 four of five new tracks failed that way while a fifth downloaded.
// MIDIGREP_YTDLP_COOKIES_FROM_BROWSER names a browser (chrome, firefox, safari, ...; yt-dlp's
// --cookies-from-browser syntax), MIDIGREP_YTDLP_COOKIES a Netscape cookies.txt; both may be set.
func ytDlpAuthArgs() []string {
	var args []string
	if b := os.Getenv("MIDIGREP_YTDLP_COOKIES_FROM_BROWSER"); b != "" {
		args = append(args, "--cookies-from-browser", b)
	}
	if f := os.Getenv("MIDIGREP_YTDLP_COOKIES"); f != "" {
		args = append(args, "--cookies", f)
	}
	return args
}

// ytDlpCommand builds a yt-dlp command with the auth flags ahead of the call's own arguments.
func ytDlpCommand(ctx context.Context, args ...string) *exec.Cmd {
	return exec.CommandContext(ctx, ytDlpBinary(), append(ytDlpAuthArgs(), args...)...)
}

// botCheckError turns YouTube's sign-in wall into an error that says how to get past it, or
// returns nil when stderr is not that wall. Retrying in another format cannot help here.
func botCheckError(stderr string) error {
	if !strings.Contains(stderr, "confirm you") || !strings.Contains(stderr, "not a bot") {
		return nil
	}
	return fmt.Errorf("YouTube wants a signed-in session for this video: set MIDIGREP_YTDLP_COOKIES_FROM_BROWSER=chrome "+
		"(or another browser you are signed in with) or MIDIGREP_YTDLP_COOKIES=<cookies.txt> (yt-dlp: %s)", strings.TrimSpace(stderr))
}

// Download downloads audio from a YouTube URL using yt-dlp
func (d *YouTubeDownloader) Download(ctx context.Context, url, outputDir string) (string, error) {
	// Check if yt-dlp is installed
	if err := d.checkYtDlp(); err != nil {
		return "", err
	}

	outputPath := filepath.Join(outputDir, "input.%(ext)s")

	// Download best audio and convert to wav
	cmd := ytDlpCommand(ctx,
		"--no-playlist",         // Only download single video
		"--extract-audio",       // Extract audio only
		"--audio-format", "wav", // Convert to WAV
		"--audio-quality", "0", // Best quality
		"--output", outputPath, // Output path template
		"--no-warnings", // Suppress warnings
		"--quiet",       // Quiet mode
		"--progress",    // But show progress
		url,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if botErr := botCheckError(stderr.String()); botErr != nil {
			return "", botErr
		}
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

	cmd := ytDlpCommand(ctx,
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
		if botErr := botCheckError(stderr.String()); botErr != nil {
			return "", botErr
		}
		return "", fmt.Errorf("yt-dlp failed: %w (stderr: %s)", err, stderr.String())
	}

	mp3Path := filepath.Join(outputDir, "input.mp3")
	return mp3Path, nil
}

// checkYtDlp verifies yt-dlp is installed
func (d *YouTubeDownloader) checkYtDlp() error {
	cmd := exec.Command(ytDlpBinary(), "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("yt-dlp not runnable (%s). Install/upgrade: scripts/python/.venv/bin/pip install -U yt-dlp (preferred — a stale yt-dlp fails with HTTP 403), or brew install yt-dlp; or set MIDIGREP_YTDLP", ytDlpBinary())
	}
	return nil
}

// GetVideoTitle fetches the video title for display
func (d *YouTubeDownloader) GetVideoTitle(ctx context.Context, url string) (string, error) {
	cmd := ytDlpCommand(ctx,
		"--no-playlist", // Only get title for single video, not entire playlist
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
