package cache

// @layer: unit
// @regression

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The pipeline stamps bpm / bpm_decided / genre into the same metadata.json the cache writes. Re-titling a cached
// track (UpdateTrackMetadata) or re-separating it must not wipe that stamp: before 2026-11-18 it did, and the next run
// went back to the extraction's tempo (BibiBabydoll: 150 decided, 112.5 again).
func TestUpdateTrackMetadataKeepsThePipelineStamp(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "trk")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	stamp := `{"title": "old", "bpm": 150.012, "bpm_decided": {"bpm_coarse": 150.012, "by": "test"}, "genre": "brazilian_funk"}`
	metaPath := filepath.Join(dir, "metadata.json")
	if err := os.WriteFile(metaPath, []byte(stamp), 0o644); err != nil {
		t.Fatal(err)
	}
	c := &StemCache{dir: root}
	if _, err := c.UpdateTrackMetadata("trk", "New Title", "https://youtu.be/abcdefghijk"); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	data, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got["title"] != "New Title" || got["video_id"] != "abcdefghijk" {
		t.Errorf("cache fields not written: %v", got)
	}
	if got["bpm"] != 150.012 || got["genre"] != "brazilian_funk" {
		t.Errorf("pipeline stamp lost: %v", got)
	}
	if d, ok := got["bpm_decided"].(map[string]any); !ok || d["bpm_coarse"] != 150.012 {
		t.Errorf("bpm_decided lost: %v", got["bpm_decided"])
	}
}

func TestWriteTrackMetadataCreatesTheFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "metadata.json")
	if err := writeTrackMetadata(p, TrackMetadata{Title: "T", URL: "u"}); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	data, _ := os.ReadFile(p)
	if err := json.Unmarshal(data, &got); err != nil || got["title"] != "T" {
		t.Fatalf("got %v (%v)", got, err)
	}
}
