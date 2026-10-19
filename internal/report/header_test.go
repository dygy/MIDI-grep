// @layer: unit
// @spec: 003-editable-strudel-generation
// @regression
//
// Strudel header parsing: decimal BPM, Genre fallback, Style card falls back to genre.
package report

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHeaderBPMGenreStyle(t *testing.T) {
	const header = "// Genre: brazilian_funk\n// BPM: 129.989\n// Key: C# minor\n// Notes: 1628\n// Drums: 233 hits\n"
	cases := []struct {
		name     string
		meta     string
		wantStat []string
		forbid   []string
	}{
		{"header only: decimal BPM, genre as style", `{}`,
			[]string{statCard("129.989", "BPM"), statCard("brazilian_funk", "Style")},
			[]string{statCard("129", "BPM"), statCard("N/A", "Style")}},
		{"style auto falls back to genre", `{"style":"auto"}`,
			[]string{statCard("brazilian_funk", "Style")}, []string{statCard("auto", "Style")}},
		{"style N/A falls back to genre", `{"style":"N/A"}`,
			[]string{statCard("brazilian_funk", "Style")}, nil},
		{"metadata genre wins over header", `{"genre":"phonk","style":""}`,
			[]string{statCard("phonk", "Style")}, []string{statCard("brazilian_funk", "Style")}},
		{"explicit style kept", `{"style":"synth","genre":"phonk"}`,
			[]string{statCard("synth", "Style")}, nil},
		{"metadata decimal BPM kept", `{"bpm":135.017}`,
			[]string{statCard("135.017", "BPM")}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			write := func(name, body string) {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			write("output.strudel", header+"$: s(\"bd\")\n")
			write("metadata.json", tc.meta)
			data, err := NewGenerator(filepath.Dir(dir), dir).LoadData()
			if err != nil {
				t.Fatal(err)
			}
			out := GenerateFromData(data)
			for _, w := range tc.wantStat {
				if !strings.Contains(out, w) {
					t.Errorf("missing %s", w)
				}
			}
			for _, f := range tc.forbid {
				if strings.Contains(out, f) {
					t.Errorf("unexpected %s", f)
				}
			}
		})
	}
}

func statCard(value, label string) string {
	return `<div class="stat-value">` + value + `</div><div class="stat-label">` + label + `</div>`
}
