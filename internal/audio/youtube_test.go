package audio

// @layer: unit
// @spec: 004-second-reference-track
// @regression

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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

func TestYtDlpCommandCarriesCookies(t *testing.T) {
	t.Setenv("MIDIGREP_YTDLP", "/opt/custom/yt-dlp")
	t.Setenv("MIDIGREP_YTDLP_COOKIES_FROM_BROWSER", "chrome")
	t.Setenv("MIDIGREP_YTDLP_COOKIES", "/tmp/cookies.txt")
	cmd := ytDlpCommand(context.Background(), "--no-playlist", "URL")
	want := []string{"/opt/custom/yt-dlp", "--cookies-from-browser", "chrome", "--cookies", "/tmp/cookies.txt", "--no-playlist", "URL"}
	if !reflect.DeepEqual(cmd.Args, want) {
		t.Fatalf("args = %q, want %q", cmd.Args, want)
	}
}

func TestYtDlpCommandWithoutCookiesIsUnchanged(t *testing.T) {
	t.Setenv("MIDIGREP_YTDLP", "/opt/custom/yt-dlp")
	t.Setenv("MIDIGREP_YTDLP_COOKIES_FROM_BROWSER", "")
	t.Setenv("MIDIGREP_YTDLP_COOKIES", "")
	cmd := ytDlpCommand(context.Background(), "URL")
	if want := []string{"/opt/custom/yt-dlp", "URL"}; !reflect.DeepEqual(cmd.Args, want) {
		t.Fatalf("args = %q, want %q", cmd.Args, want)
	}
}

func TestBotCheckErrorNamesTheFix(t *testing.T) {
	wall := "ERROR: [youtube] ZO7yYRIM0-g: Sign in to confirm you’re not a bot. Use --cookies-from-browser"
	err := botCheckError(wall)
	if err == nil || !strings.Contains(err.Error(), "MIDIGREP_YTDLP_COOKIES_FROM_BROWSER") || !strings.Contains(err.Error(), "ZO7yYRIM0-g") {
		t.Fatalf("bot wall not reported with its fix: %v", err)
	}
	if err := botCheckError("ERROR: HTTP Error 403: Forbidden"); err != nil {
		t.Fatalf("a 403 is not the bot wall: %v", err)
	}
}

// A bot wall on the wav download must fail with the fix, not fall through to an mp3 retry that
// hits the same wall and reports a bare "exit status 1".
func TestDownloadStopsAtTheBotWall(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "yt-dlp")
	script := "#!/bin/sh\n[ \"$1\" = --version ] && exit 0\necho 'ERROR: [youtube] x: Sign in to confirm you’re not a bot.' >&2\necho called >> " + filepath.Join(dir, "calls") + "\nexit 1\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MIDIGREP_YTDLP", fake)
	_, err := NewYouTubeDownloader().Download(context.Background(), "https://youtu.be/x", dir)
	if err == nil || !strings.Contains(err.Error(), "MIDIGREP_YTDLP_COOKIES_FROM_BROWSER") {
		t.Fatalf("expected the bot-wall error, got %v", err)
	}
	calls, _ := os.ReadFile(filepath.Join(dir, "calls"))
	if n := strings.Count(string(calls), "called"); n != 1 {
		t.Fatalf("yt-dlp download ran %d times, want 1 (no mp3 retry)", n)
	}
}
