package cache

// @layer: unit
// @spec: 003-editable-strudel-generation
// @regression

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractVideoID(t *testing.T) {
	cases := []struct {
		name, url, want string
	}{
		{"watch", "https://www.youtube.com/watch?v=Q4801HzWZfg", "Q4801HzWZfg"},
		{"watch with extra params", "https://www.youtube.com/watch?v=Q4801HzWZfg&list=PL123&t=42", "Q4801HzWZfg"},
		{"short link", "https://youtu.be/Q4801HzWZfg", "Q4801HzWZfg"},
		{"short link with query", "https://youtu.be/Q4801HzWZfg?si=abc", "Q4801HzWZfg"},
		{"shorts", "https://youtube.com/shorts/abc_DEF-123", "abc_DEF-123"},
		{"music", "https://music.youtube.com/watch?v=Q4801HzWZfg", "Q4801HzWZfg"},
		{"not youtube", "https://soundcloud.com/artist/track", ""},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := ExtractVideoID(c.url); got != c.want {
				t.Fatalf("ExtractVideoID(%q) = %q, want %q", c.url, got, c.want)
			}
		})
	}
}

func TestKeyForURL(t *testing.T) {
	if got := KeyForURL("https://youtu.be/Q4801HzWZfg"); got != "yt_Q4801HzWZfg" {
		t.Fatalf("YouTube key = %q, want yt_Q4801HzWZfg", got)
	}
	// Non-YouTube URLs fall back to a hash: stable, non-empty, never yt_-prefixed.
	a := KeyForURL("https://example.com/a.mp3")
	b := KeyForURL("https://example.com/a.mp3")
	other := KeyForURL("https://example.com/b.mp3")
	if a == "" || a != b {
		t.Fatalf("hash fallback must be stable and non-empty, got %q / %q", a, b)
	}
	if a == other {
		t.Fatalf("different URLs must not share a key: %q", a)
	}
	if strings.HasPrefix(a, "yt_") {
		t.Fatalf("non-YouTube key must not look like a video key: %q", a)
	}
}

func TestKeyForFile(t *testing.T) {
	cases := []struct {
		name, path, want string
	}{
		{"plain name", filepath.Join("tmp", "Regime CLT.wav"), "Regime CLT"},
		{"generic name uses parent dir", filepath.Join("tracks", "Automotivo XM", "original.wav"), "Automotivo XM"},
		{"generic name and generic parent keeps name", filepath.Join("audio", "input.wav"), "input"},
		{"generic name at relative root keeps name", "original.wav", "original"},
		{"unsafe characters sanitized", filepath.Join("tmp", "a:b*c?d.wav"), "a-bcd"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := KeyForFile(c.path)
			if err != nil {
				t.Fatalf("KeyForFile(%q) error: %v", c.path, err)
			}
			if got != c.want {
				t.Fatalf("KeyForFile(%q) = %q, want %q", c.path, got, c.want)
			}
		})
	}
}
