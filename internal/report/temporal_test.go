// @layer: unit
// @spec: 003-editable-strudel-generation
// @regression
//
// The report must show the time-aware similarity rows when comparison.json carries them,
// and render exactly as before (no such rows, no error) for legacy files without them.
package report

import (
	"encoding/json"
	"strings"
	"testing"
)

func comparisonJSON(extra string) string {
	return `{"comparison": {"overall_similarity": 0.8, "mfcc_similarity": 0.7,
	"chroma_similarity": 0.7, "brightness_similarity": 0.7, "tempo_similarity": 1.0,
	"frequency_balance_similarity": 0.9, "energy_similarity": 0.9` + extra + `}}`
}

func TestTemporalRows(t *testing.T) {
	full := `, "static_similarity": 0.9,
	"temporal": {"lag_seconds": 0.1, "chroma_corr": 0.5, "mel_corr": 0.6, "onset_f1": 0.4,
		"envelope_corr": 0.7, "temporal_similarity": 0.55},
	"chance": {"path": "x.wav", "overall": 0.42, "section_aware": 0.4, "temporal": 0.1},
	"margin_over_chance": 0.38`
	noChance := `, "margin_over_chance": -0.05`
	labels := []string{
		"Static (time-averaged)", "Temporal (time-aligned)", "pitch-in-time (chroma)",
		"spectrum-in-time (mel)", "onset F1 (±50 ms)", "loudness envelope",
		"Margin over a different song",
	}
	tests := []struct {
		name     string
		extra    string
		want     []string
		wantNone []string
	}{
		{"full", full, append(append([]string{}, labels...), "control scored 42% overall", "+38.0 pts"), nil},
		{"negative margin without control", noChance, []string{"Margin over a different song", "-5.0 pts"}, labels[:6]},
		{"legacy", "", nil, labels},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var comp ComparisonResult
			if err := json.Unmarshal([]byte(comparisonJSON(tt.extra)), &comp); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			out := generateChartsHTML(&comp)
			if !strings.Contains(out, "Timbre (MFCC)") {
				t.Fatalf("base metric rows missing")
			}
			for _, w := range tt.want {
				if !strings.Contains(out, w) {
					t.Errorf("missing %q", w)
				}
			}
			for _, w := range tt.wantNone {
				if strings.Contains(out, w) {
					t.Errorf("unexpected %q", w)
				}
			}
		})
	}
}

// Unscorable alignment scores are written as null (never -Infinity/NaN); the report must parse and render them.
func TestLagCandidatesAcceptNull(t *testing.T) {
	extra := `, "static_similarity": 0.9,
	"temporal": {"lag_seconds": 0.0, "lag_candidates": [
		{"lag_seconds": 0.0, "onset_score": 0.4, "harmony_score": null, "onset_tied": true, "chosen": true, "decided_by": "onset"},
		{"lag_seconds": 1.7, "onset_score": null, "harmony_score": null, "onset_tied": false, "chosen": false}],
		"chroma_corr": 0.5, "mel_corr": 0.6, "onset_f1": 0.4, "envelope_corr": 0.7, "temporal_similarity": 0.55},
	"chance": {"path": "x.wav", "overall": 0.42, "section_aware": 0.4, "temporal": 0.1, "lag_seconds": null, "lag_candidates": null},
	"margin_over_chance": 0.38`
	var comp ComparisonResult
	if err := json.Unmarshal([]byte(comparisonJSON(extra)), &comp); err != nil {
		t.Fatalf("unmarshal with null scores: %v", err)
	}
	if got := comp.Comparison.Temporal.LagCandidates; len(got) != 2 || got[0].HarmonyScore != nil || !got[0].Chosen {
		t.Fatalf("candidates not parsed as expected: %+v", got)
	}
	if out := generateChartsHTML(&comp); !strings.Contains(out, "Temporal (time-aligned)") {
		t.Fatalf("temporal row missing")
	}
}
