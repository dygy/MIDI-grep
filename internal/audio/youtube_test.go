package audio

// @layer: unit
// @spec: 004-second-reference-track
// @regression

import (
	"os"
	"path/filepath"
	"testing"
)

func TestYtDlpBinaryPrefersEnvOverride(t *testing.T) {
	t.Setenv("MIDIGREP_YTDLP", "/opt/custom/yt-dlp")
	if got := ytDlpBinary(); got != "/opt/custom/yt-dlp" {
		t.Fatalf("env override ignored: %q", got)
	}
}

func TestYtDlpBinaryPrefersProjectVenv(t *testing.T) {
	t.Setenv("MIDIGREP_YTDLP", "")
	dir := t.TempDir()
	venvBin := filepath.Join(dir, "scripts", "python", ".venv", "bin")
	if err := os.MkdirAll(venvBin, 0o755); err != nil {
		t.Fatal(err)
	}
	fake := filepath.Join(venvBin, "yt-dlp")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(wd) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	got, _ := filepath.EvalSymlinks(ytDlpBinary()) // macOS: /var -> /private/var
	want, _ := filepath.EvalSymlinks(fake)
	if got != want {
		t.Fatalf("venv yt-dlp not preferred: got %q want %q", got, want)
	}
}

func TestYtDlpBinaryFallsBackToPath(t *testing.T) {
	t.Setenv("MIDIGREP_YTDLP", "")
	dir := t.TempDir() // no venv here; the executable's bin/../ has none in a test binary either
	wd, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(wd) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	if got := ytDlpBinary(); got != "yt-dlp" {
		t.Fatalf("expected PATH fallback, got %q", got)
	}
}
