// @layer: unit
// @spec: 003-editable-strudel-generation
// @regression
//
// Report editability verdict tests (spec 003 §2.2-F, Slice 5). The Go report must show the
// generation mode and editability verdict in the headline, render the REPLAY badge with no
// percentage when editability != "pass", keep the legacy headline (plus a "not checked" note)
// when the keys are absent, and caption the per-stem section as a demucs re-separation.
//
// Fixtures are shared with the Python tests: scripts/python/tests/fixtures/report/.
package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const (
	fixtureDir  = "../../scripts/python/tests/fixtures/report"
	badgeText   = "REPLAY / UNVERIFIED — not a deliverable"
	legacyNote  = "editability: not checked (pre-spec-003 run)"
	stemCaption = "stems obtained by demucs re-separation of the rendered mix (lossy, not a true stem-match view)"
)

var (
	headlineRe = regexp.MustCompile(`(?s)<div class="overall-headline"[^>]*>(.*?)</div>\s*</div>`)
	percentRe  = regexp.MustCompile(`\d+%`)
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir, name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return b
}

func loadComparison(t *testing.T, name string) *ComparisonResult {
	t.Helper()
	var comp ComparisonResult
	if err := json.Unmarshal(readFixture(t, name), &comp); err != nil {
		t.Fatalf("unmarshal %s: %v", name, err)
	}
	return &comp
}

func headline(html string) (string, bool) {
	m := headlineRe.FindStringSubmatch(html)
	if m == nil {
		return "", false
	}
	return m[1], true
}

func TestComparisonResultDecodesEditabilityKeys(t *testing.T) {
	comp := loadComparison(t, "comparison_pass_sample_instrument.json")
	if comp.GenerationMode != "sample-instrument" {
		t.Errorf("GenerationMode = %q, want sample-instrument", comp.GenerationMode)
	}
	if comp.Editability != "pass" {
		t.Errorf("Editability = %q, want pass", comp.Editability)
	}
	if len(comp.EditabilityViolations) != 0 {
		t.Errorf("EditabilityViolations = %v, want empty", comp.EditabilityViolations)
	}

	fail := loadComparison(t, "comparison_fail_replay.json")
	if fail.Editability != "fail" || fail.GenerationMode != "loops" {
		t.Errorf("fail fixture decoded as editability=%q mode=%q", fail.Editability, fail.GenerationMode)
	}
	if len(fail.EditabilityViolations) != 2 {
		t.Errorf("fail fixture violations = %d, want 2", len(fail.EditabilityViolations))
	}

	legacy := loadComparison(t, "comparison_legacy_no_keys.json")
	if legacy.Editability != "" || legacy.GenerationMode != "" {
		t.Errorf("legacy fixture must leave the new keys empty, got editability=%q mode=%q", legacy.Editability, legacy.GenerationMode)
	}
}

func TestChartsHeadlinePassShowsModeAndEditable(t *testing.T) {
	html := generateChartsHTML(loadComparison(t, "comparison_pass_sample_instrument.json"))
	h, ok := headline(html)
	if !ok {
		t.Fatalf("pass run must render the overall-headline block:\n%s", html)
	}
	if !strings.Contains(h, "94% — mode: sample-instrument · editable: pass") {
		t.Errorf("headline missing mode/editable label: %q", h)
	}
	if strings.Contains(html, badgeText) {
		t.Errorf("pass run must not render the replay badge")
	}
	if strings.Contains(html, legacyNote) {
		t.Errorf("pass run must not render the legacy note")
	}
}

func TestChartsFailRendersBadgeAndNoPercentage(t *testing.T) {
	html := generateChartsHTML(loadComparison(t, "comparison_fail_replay.json"))
	if !strings.Contains(html, badgeText) {
		t.Fatalf("fail run must render the replay badge:\n%s", html)
	}
	if !strings.Contains(html, `class="editability-badge"`) {
		t.Errorf("badge element missing")
	}
	if _, ok := headline(html); ok {
		t.Errorf("fail run must not render a percentage headline")
	}
	for _, forbidden := range []string{"Overall Similarity", "Similarity Scores"} {
		if strings.Contains(html, forbidden) {
			t.Errorf("fail run must not contain %q", forbidden)
		}
	}
	if percentRe.MatchString(html) {
		t.Errorf("no percentage may appear for a replay run:\n%s", html)
	}
	if !strings.Contains(html, "R2: fewer than 2 editable note() voices") {
		t.Errorf("violations must be listed under the badge")
	}
}

func TestChartsFailWithNumbersStillHidesScores(t *testing.T) {
	comp := loadComparison(t, "comparison_pass_sample_instrument.json")
	comp.Editability = "fail"
	comp.EditabilityViolations = []string{"R1: vocal stem replay"}
	html := generateChartsHTML(comp)
	if !strings.Contains(html, badgeText) {
		t.Fatalf("stamped fail must render the badge")
	}
	if _, ok := headline(html); ok {
		t.Errorf("stamped fail must not render the headline")
	}
	if strings.Contains(html, "Similarity Scores") || strings.Contains(html, "94%") {
		t.Errorf("stamped fail must not show similarity scores")
	}
}

func TestChartsLegacyKeepsHeadlineWithNote(t *testing.T) {
	html := generateChartsHTML(loadComparison(t, "comparison_legacy_no_keys.json"))
	h, ok := headline(html)
	if !ok {
		t.Fatalf("legacy run must keep the headline")
	}
	if !strings.Contains(h, "94%") {
		t.Errorf("legacy headline must show the percentage: %q", h)
	}
	if strings.Contains(h, "mode:") || strings.Contains(h, "editable:") {
		t.Errorf("legacy headline must not claim a mode/verdict: %q", h)
	}
	if !strings.Contains(html, legacyNote) {
		t.Errorf("legacy run must carry the not-checked note")
	}
	if strings.Contains(html, badgeText) {
		t.Errorf("legacy run must not render the badge")
	}
}

func TestStemSectionCaptionAndReplaySuppression(t *testing.T) {
	var stems StemComparisonResult
	if err := json.Unmarshal(readFixture(t, "stem_comparison_min.json"), &stems); err != nil {
		t.Fatalf("unmarshal stem fixture: %v", err)
	}

	html := generateStemComparisonHTML(&stems, "pass")
	if !strings.Contains(html, stemCaption) {
		t.Errorf("per-stem section must carry the demucs caption")
	}
	if !strings.Contains(html, "Weighted Per-Stem Similarity") {
		t.Errorf("pass run should show per-stem scores")
	}

	replay := generateStemComparisonHTML(&stems, "fail")
	if !strings.Contains(replay, stemCaption) {
		t.Errorf("replay per-stem section must still carry the caption")
	}
	if percentRe.MatchString(replay) {
		t.Errorf("replay run must not show per-stem percentages:\n%s", replay)
	}

	if generateStemComparisonHTML(nil, "pass") != "" {
		t.Errorf("missing stem_comparison.json must render nothing")
	}
}

func TestLoadDataAndGenerateEndToEnd(t *testing.T) {
	cases := []struct {
		fixture string
		expect  []string
		forbid  []string
	}{
		{
			"comparison_pass_sample_instrument.json",
			[]string{"mode: sample-instrument · editable: pass", stemCaption, `alt="Similarity Scores"`},
			[]string{badgeText, legacyNote},
		},
		{
			"comparison_fail_replay.json",
			[]string{badgeText, stemCaption, "R2: fewer than 2 editable note() voices"},
			[]string{"Overall Similarity", "Similarity Scores", legacyNote},
		},
		{
			"comparison_legacy_no_keys.json",
			[]string{"Overall Similarity", legacyNote, stemCaption},
			[]string{badgeText, "editable:"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			cache := filepath.Join(t.TempDir(), "yt_fixture")
			version := filepath.Join(cache, "v001")
			if err := os.MkdirAll(version, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(version, "comparison.json"), readFixture(t, tc.fixture), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(version, "stem_comparison.json"), readFixture(t, "stem_comparison_min.json"), 0o644); err != nil {
				t.Fatal(err)
			}
			// the legacy similarity gauge image is a replay score by another route
			if err := os.WriteFile(filepath.Join(version, "chart_similarity.png"), []byte("\x89PNG stub"), 0o644); err != nil {
				t.Fatal(err)
			}

			data, err := NewGenerator(cache, version).LoadData()
			if err != nil {
				t.Fatalf("LoadData: %v", err)
			}
			if data.Comparison == nil {
				t.Fatalf("comparison.json not loaded")
			}
			html := GenerateFromData(data)
			for _, s := range tc.expect {
				if !strings.Contains(html, s) {
					t.Errorf("missing %q", s)
				}
			}
			for _, s := range tc.forbid {
				if strings.Contains(html, s) {
					t.Errorf("unexpected %q", s)
				}
			}
		})
	}
}
