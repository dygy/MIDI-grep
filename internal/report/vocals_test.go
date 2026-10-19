// @layer: unit
// @spec: 003-editable-strudel-generation
// @regression
package report

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVocalsNotGenerated(t *testing.T) {
	dir := t.TempDir()
	wav := filepath.Join(dir, "vocals.wav")
	if err := os.WriteFile(wav, []byte("RIFFfake"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name      string
		path      string
		wantAudio bool
	}{
		{"with vocals.wav", wav, true},
		{"without vocals.wav", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := GenerateFromData(&ReportData{VocalsPath: tc.path})
			has := strings.Contains(h, `id="audio-vocals"`)
			if has != tc.wantAudio {
				t.Fatalf("vocal player present=%v, want %v", has, tc.wantAudio)
			}
			if strings.Contains(h, "<label>Vocals</label>") {
				t.Error("bare 'Vocals' label must not appear")
			}
			if strings.Contains(h, "'audio-vocals'") {
				t.Error("audio-vocals must not be in any playback group")
			}
			if tc.wantAudio {
				if !strings.Contains(h, "Vocals (not generated)") {
					t.Error("missing 'not generated' label")
				}
				if !strings.Contains(h, `<audio controls muted src="data:`) {
					t.Error("vocal player must start muted")
				}
			} else if strings.Contains(h, "not generated") {
				t.Error("no vocal label expected without a stem")
			}
		})
	}
}
