// Package report provides HTML report generation for MIDI-grep results.
// This is a type-safe Go implementation that replaces the Python report generator.
package report

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"html"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ReportData holds all data needed to generate a report
type ReportData struct {
	TrackName   string
	Version     int
	BPM         int
	Key         string
	Style       string
	Genre       string
	Notes       int
	DrumHits    int
	CreatedAt   time.Time
	StrudelCode string

	// Audio file paths (will be encoded as base64)
	MelodicPath string
	DrumsPath   string
	VocalsPath  string
	BassPath    string
	RenderPath  string

	// Chart image paths
	ChartFrequencyPath     string
	ChartSimilarityPath    string
	ChartSpecOrigPath      string
	ChartSpecRendPath      string
	ChartChromaOrigPath    string
	ChartChromaRendPath    string

	// Comparison results (from JSON)
	Comparison *ComparisonResult

	// AI params (from JSON)
	AIParams map[string]interface{}
}

// ComparisonResult holds audio comparison metrics
type ComparisonResult struct {
	Original   AudioMetrics   `json:"original"`
	Rendered   AudioMetrics   `json:"rendered"`
	Comparison ComparisonData `json:"comparison"`
}

// AudioMetrics holds metrics for a single audio file
type AudioMetrics struct {
	Bands    map[string]float64 `json:"bands"`
	Spectral map[string]float64 `json:"spectral"`
	Rhythm   map[string]float64 `json:"rhythm"`
}

// ComparisonData holds similarity scores
type ComparisonData struct {
	OverallSimilarity          float64 `json:"overall_similarity"`
	MFCCSimilarity             float64 `json:"mfcc_similarity"`
	ChromaSimilarity           float64 `json:"chroma_similarity"`
	BrightnessSimilarity       float64 `json:"brightness_similarity"`
	TempoSimilarity            float64 `json:"tempo_similarity"`
	FrequencyBalanceSimilarity float64 `json:"frequency_balance_similarity"`
	EnergySimilarity           float64 `json:"energy_similarity"`
}

// Generator creates HTML reports
type Generator struct {
	cacheDir   string
	versionDir string
}

// NewGenerator creates a new report generator
func NewGenerator(cacheDir, versionDir string) *Generator {
	return &Generator{
		cacheDir:   cacheDir,
		versionDir: versionDir,
	}
}

// LoadData loads all data needed for the report from disk
func (g *Generator) LoadData() (*ReportData, error) {
	data := &ReportData{
		CreatedAt: time.Now(),
		Version:   1,
	}

	// Find audio files in cache dir
	data.MelodicPath = findFile(g.cacheDir, "melodic.wav", "piano.wav")
	data.DrumsPath = findFile(g.cacheDir, "drums.wav")
	data.VocalsPath = findFile(g.cacheDir, "vocals.wav")
	data.BassPath = findFile(g.cacheDir, "bass.wav")

	// Find rendered audio in version dir
	data.RenderPath = findFile(g.versionDir, "render.wav")

	// Find Strudel code
	strudelPath := findFile(g.versionDir, "output.strudel", "output_latest.strudel")
	if strudelPath != "" {
		code, err := os.ReadFile(strudelPath)
		if err == nil {
			data.StrudelCode = string(code)
			// Extract info from strudel comments
			extractStrudelInfo(data)
		}
	}

	// Find chart images
	data.ChartFrequencyPath = findFile(g.versionDir, "chart_frequency.png")
	data.ChartSimilarityPath = findFile(g.versionDir, "chart_similarity.png")
	data.ChartSpecOrigPath = findFile(g.versionDir, "chart_spectrogram_original.png")
	data.ChartSpecRendPath = findFile(g.versionDir, "chart_spectrogram_rendered.png")
	data.ChartChromaOrigPath = findFile(g.versionDir, "chart_chromagram_original.png")
	data.ChartChromaRendPath = findFile(g.versionDir, "chart_chromagram_rendered.png")

	// Load comparison JSON
	compPath := filepath.Join(g.versionDir, "comparison.json")
	if compData, err := os.ReadFile(compPath); err == nil {
		var comp ComparisonResult
		if json.Unmarshal(compData, &comp) == nil {
			data.Comparison = &comp
		}
	}

	// Load metadata
	metaPath := filepath.Join(g.versionDir, "metadata.json")
	if metaData, err := os.ReadFile(metaPath); err == nil {
		var meta map[string]interface{}
		if json.Unmarshal(metaData, &meta) == nil {
			if v, ok := meta["version"].(float64); ok {
				data.Version = int(v)
			}
			if v, ok := meta["bpm"].(float64); ok {
				data.BPM = int(v)
			}
			if v, ok := meta["key"].(string); ok {
				data.Key = v
			}
			if v, ok := meta["style"].(string); ok {
				data.Style = v
			}
			if v, ok := meta["genre"].(string); ok {
				data.Genre = v
			}
			if v, ok := meta["notes"].(float64); ok {
				data.Notes = int(v)
			}
			if v, ok := meta["drum_hits"].(float64); ok {
				data.DrumHits = int(v)
			}
		}
	}

	// Load track metadata for name
	trackMetaPath := filepath.Join(g.cacheDir, "metadata.json")
	if trackData, err := os.ReadFile(trackMetaPath); err == nil {
		var trackMeta map[string]interface{}
		if json.Unmarshal(trackData, &trackMeta) == nil {
			if v, ok := trackMeta["title"].(string); ok {
				data.TrackName = v
			}
		}
	}

	// Fallback track name from directory
	if data.TrackName == "" {
		data.TrackName = cleanTrackName(filepath.Base(g.cacheDir))
	}

	// Load AI params
	aiPath := findFile(g.versionDir, "ai_params.json")
	if aiPath == "" {
		aiPath = findFile(g.cacheDir, "ai_params.json")
	}
	if aiPath != "" {
		if aiData, err := os.ReadFile(aiPath); err == nil {
			json.Unmarshal(aiData, &data.AIParams)
		}
	}

	return data, nil
}

// Generate creates the HTML report
func (g *Generator) Generate(outputPath string) (string, error) {
	data, err := g.LoadData()
	if err != nil {
		return "", fmt.Errorf("failed to load data: %w", err)
	}

	if outputPath == "" {
		outputPath = filepath.Join(g.versionDir, "report.html")
	}

	htmlContent := generateHTML(data)

	if err := os.WriteFile(outputPath, []byte(htmlContent), 0644); err != nil {
		return "", fmt.Errorf("failed to write report: %w", err)
	}

	return outputPath, nil
}

// GenerateFromData creates HTML from pre-loaded data
func GenerateFromData(data *ReportData) string {
	return generateHTML(data)
}

// Helper functions

func findFile(dir string, names ...string) string {
	for _, name := range names {
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func cleanTrackName(name string) string {
	if strings.HasPrefix(name, "yt_") {
		return fmt.Sprintf("YouTube Track (%s)", name[3:])
	}
	// Remove trailing [xxx] format
	re := regexp.MustCompile(`\s*\[[^\]]+\]$`)
	return re.ReplaceAllString(name, "")
}

func extractStrudelInfo(data *ReportData) {
	code := data.StrudelCode

	// BPM
	if re := regexp.MustCompile(`BPM:\s*(\d+)`); re.MatchString(code) {
		if m := re.FindStringSubmatch(code); len(m) > 1 {
			if v, _ := strconv.Atoi(m[1]); v > 0 && data.BPM == 0 {
				data.BPM = v
			}
		}
	}

	// Key
	if re := regexp.MustCompile(`Key:\s*([A-G][#b]?\s*(?:major|minor)?)`); re.MatchString(code) {
		if m := re.FindStringSubmatch(code); len(m) > 1 && data.Key == "" {
			data.Key = m[1]
		}
	}

	// Notes
	if re := regexp.MustCompile(`Notes:\s*(\d+)`); re.MatchString(code) {
		if m := re.FindStringSubmatch(code); len(m) > 1 {
			if v, _ := strconv.Atoi(m[1]); v > 0 && data.Notes == 0 {
				data.Notes = v
			}
		}
	}

	// Drums
	if re := regexp.MustCompile(`Drums:\s*(\d+)\s*hits`); re.MatchString(code) {
		if m := re.FindStringSubmatch(code); len(m) > 1 {
			if v, _ := strconv.Atoi(m[1]); v > 0 && data.DrumHits == 0 {
				data.DrumHits = v
			}
		}
	}

	// Style
	if re := regexp.MustCompile(`Style:\s*(\w+)`); re.MatchString(code) {
		if m := re.FindStringSubmatch(code); len(m) > 1 && data.Style == "" {
			data.Style = m[1]
		}
	}
}

func encodeAudioBase64(path string) string {
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	ext := strings.ToLower(filepath.Ext(path))
	mime := "audio/wav"
	if ext == ".mp3" {
		mime = "audio/mp3"
	}
	return fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(data))
}

func encodeImageBase64(path string) string {
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	ext := strings.ToLower(filepath.Ext(path))
	mime := "image/png"
	if ext == ".jpg" || ext == ".jpeg" {
		mime = "image/jpeg"
	}
	return fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(data))
}

func formatValue(v int, fallback string) string {
	if v > 0 {
		return strconv.Itoa(v)
	}
	return fallback
}

func formatStringValue(v, fallback string) string {
	if v != "" {
		return v
	}
	return fallback
}

func similarityColor(pct float64) string {
	if pct >= 70 {
		return "#3fb950"
	}
	if pct >= 50 {
		return "#d29922"
	}
	return "#f85149"
}

func generateChartsHTML(comp *ComparisonResult) string {
	if comp == nil {
		return `<div class="no-data" style="margin-top: 1rem;">No comparison data available (render audio first)</div>`
	}

	var sb strings.Builder

	// Overall similarity score
	overall := comp.Comparison.OverallSimilarity * 100
	color := similarityColor(overall)
	sb.WriteString(fmt.Sprintf(`
		<div style="text-align: center; margin-bottom: 1.5rem;">
			<div style="font-size: 3rem; font-weight: bold; color: %s;">%.0f%%</div>
			<div style="color: var(--text-secondary);">Overall Similarity</div>
		</div>
	`, color, overall))

	// Similarity scores table
	metrics := []struct {
		Label string
		Value float64
	}{
		{"Timbre (MFCC)", comp.Comparison.MFCCSimilarity},
		{"Harmony (Chroma)", comp.Comparison.ChromaSimilarity},
		{"Brightness", comp.Comparison.BrightnessSimilarity},
		{"Tempo", comp.Comparison.TempoSimilarity},
		{"Frequency Balance", comp.Comparison.FrequencyBalanceSimilarity},
		{"Energy", comp.Comparison.EnergySimilarity},
	}

	var rows strings.Builder
	for _, m := range metrics {
		pct := m.Value * 100
		c := similarityColor(pct)
		barWidth := pct
		if barWidth > 100 {
			barWidth = 100
		}
		rows.WriteString(fmt.Sprintf(`
			<tr>
				<td style="padding: 0.5rem; color: var(--text-secondary);">%s</td>
				<td style="padding: 0.5rem; width: 60%%;">
					<div style="background: var(--bg-primary); border-radius: 4px; height: 20px; overflow: hidden;">
						<div style="width: %.0f%%; height: 100%%; background: %s;"></div>
					</div>
				</td>
				<td style="padding: 0.5rem; text-align: right; font-weight: 600; color: %s;">%.0f%%</td>
			</tr>
		`, m.Label, barWidth, c, c, pct))
	}

	sb.WriteString(fmt.Sprintf(`
		<div class="chart-item" style="padding: 1rem;">
			<h4 style="margin-bottom: 1rem; color: var(--text-primary);">Similarity Scores</h4>
			<table style="width: 100%%; border-collapse: collapse;">%s</table>
		</div>
	`, rows.String()))

	// Frequency bands comparison
	bands := []struct {
		Label string
		Key   string
	}{
		{"Sub Bass", "sub_bass"},
		{"Bass", "bass"},
		{"Low Mid", "low_mid"},
		{"Mid", "mid"},
		{"High Mid", "high_mid"},
		{"High", "high"},
	}

	var bandRows strings.Builder
	for _, b := range bands {
		origVal := comp.Original.Bands[b.Key] * 100
		rendVal := comp.Rendered.Bands[b.Key] * 100
		diff := rendVal - origVal
		diffStr := fmt.Sprintf("%.1f", diff)
		if diff > 0 {
			diffStr = "+" + diffStr
		}
		diffColor := similarityColor(100 - abs(diff)*2) // Smaller diff = better

		bandRows.WriteString(fmt.Sprintf(`
			<tr>
				<td style="padding: 0.4rem; color: var(--text-secondary);">%s</td>
				<td style="padding: 0.4rem; text-align: right;">%.1f%%</td>
				<td style="padding: 0.4rem; text-align: right;">%.1f%%</td>
				<td style="padding: 0.4rem; text-align: right; color: %s;">%s%%</td>
			</tr>
		`, b.Label, origVal, rendVal, diffColor, diffStr))
	}

	sb.WriteString(fmt.Sprintf(`
		<div class="chart-item" style="padding: 1rem;">
			<h4 style="margin-bottom: 1rem; color: var(--text-primary);">Frequency Bands</h4>
			<table style="width: 100%%; border-collapse: collapse;">
				<tr style="border-bottom: 1px solid var(--border);">
					<th style="padding: 0.4rem; text-align: left; color: var(--text-secondary);">Band</th>
					<th style="padding: 0.4rem; text-align: right; color: #58a6ff;">Original</th>
					<th style="padding: 0.4rem; text-align: right; color: #3fb950;">Rendered</th>
					<th style="padding: 0.4rem; text-align: right; color: var(--text-secondary);">Diff</th>
				</tr>
				%s
			</table>
		</div>
	`, bandRows.String()))

	return sb.String()
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func generateAIAnalysisCard(aiParams map[string]interface{}) string {
	if aiParams == nil || len(aiParams) == 0 {
		return ""
	}

	suggestions, ok := aiParams["suggestions"].(map[string]interface{})
	if !ok {
		return ""
	}

	global, ok := suggestions["global"].(map[string]interface{})
	if !ok {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(`
		<div class="card">
			<div class="card-title">
				<svg viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"/></svg>
				AI Analysis
			</div>
			<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
	`)

	// Brightness
	if brightness, ok := global["brightness"].(string); ok {
		sb.WriteString(fmt.Sprintf(`
			<div>
				<h5 style="color: var(--accent); margin-bottom: 0.5rem;">Brightness</h5>
				<p style="font-size: 0.85rem; color: var(--text-primary);">%s</p>
			</div>
		`, html.EscapeString(brightness)))
	}

	// Dynamics
	if dynamics, ok := global["dynamics"].(string); ok {
		sb.WriteString(fmt.Sprintf(`
			<div>
				<h5 style="color: var(--accent); margin-bottom: 0.5rem;">Dynamics</h5>
				<p style="font-size: 0.85rem; color: var(--text-primary);">%s</p>
			</div>
		`, html.EscapeString(dynamics)))
	}

	sb.WriteString(`
			</div>
		</div>
	`)

	return sb.String()
}

func generateHTML(data *ReportData) string {
	// Encode audio files as base64
	melodicData := encodeAudioBase64(data.MelodicPath)
	drumsData := encodeAudioBase64(data.DrumsPath)
	vocalsData := encodeAudioBase64(data.VocalsPath)
	bassData := encodeAudioBase64(data.BassPath)
	renderData := encodeAudioBase64(data.RenderPath)

	// Encode chart images
	chartFrequencyData := encodeImageBase64(data.ChartFrequencyPath)
	chartSimilarityData := encodeImageBase64(data.ChartSimilarityPath)
	chartSpecOrigData := encodeImageBase64(data.ChartSpecOrigPath)
	chartSpecRendData := encodeImageBase64(data.ChartSpecRendPath)
	chartChromaOrigData := encodeImageBase64(data.ChartChromaOrigPath)
	chartChromaRendData := encodeImageBase64(data.ChartChromaRendPath)
	hasIndividualCharts := chartFrequencyData != "" || chartSimilarityData != ""

	// Format values
	trackName := html.EscapeString(data.TrackName)
	bpm := formatValue(data.BPM, "N/A")
	key := formatStringValue(data.Key, "N/A")
	style := formatStringValue(data.Style, formatStringValue(data.Genre, "N/A"))
	notes := formatValue(data.Notes, "N/A")
	drumHits := formatValue(data.DrumHits, "N/A")

	// Generate audio player HTML
	audioPlayer := func(label, dataURI, id string) string {
		if dataURI == "" {
			return fmt.Sprintf(`<div class="audio-player"><label>%s</label><div class="no-data">Not available</div></div>`, label)
		}
		return fmt.Sprintf(`<div class="audio-player"><label>%s</label><audio controls src="%s" id="%s"></audio></div>`, label, dataURI, id)
	}

	// Generate chart image HTML
	chartImage := func(dataURI, alt string) string {
		if dataURI == "" {
			return ""
		}
		return fmt.Sprintf(`<div class="chart-item"><img src="%s" alt="%s"/></div>`, dataURI, alt)
	}

	// Build visual charts section
	chartsSection := ""
	if hasIndividualCharts {
		chartsSection = fmt.Sprintf(`
		<div class="card">
			<div class="card-title">
				<svg viewBox="0 0 16 16"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Z"/></svg>
				Visual Comparison Charts
			</div>
			<div class="charts-grid">
				%s%s%s%s%s%s
			</div>
		</div>`,
			chartImage(chartFrequencyData, "Frequency Bands"),
			chartImage(chartSimilarityData, "Similarity Scores"),
			chartImage(chartSpecOrigData, "Original Spectrogram"),
			chartImage(chartSpecRendData, "Rendered Spectrogram"),
			chartImage(chartChromaOrigData, "Original Chromagram"),
			chartImage(chartChromaRendData, "Rendered Chromagram"),
		)
	}

	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIDI-grep Report: %s</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --border: #30363d;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
        h1 { font-size: 1.75rem; font-weight: 600; margin-bottom: 0.5rem; }
        .subtitle { color: var(--text-secondary); font-size: 0.9rem; }
        .badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; margin-right: 0.5rem; }
        .badge-blue { background: rgba(88, 166, 255, 0.2); color: var(--accent); }
        .badge-green { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .badge-orange { background: rgba(210, 153, 34, 0.2); color: var(--accent-orange); }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
        .card-title { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        .card-title svg { width: 18px; height: 18px; fill: var(--text-secondary); }
        .audio-section { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .audio-player { background: var(--bg-tertiary); border-radius: 6px; padding: 1rem; }
        .audio-player label { display: block; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
        audio { width: 100%%; height: 40px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; }
        .stat { text-align: center; padding: 1rem; background: var(--bg-tertiary); border-radius: 6px; }
        .stat-value { font-size: 1.5rem; font-weight: 600; color: var(--accent); }
        .stat-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        .code-block { background: var(--bg-tertiary); border-radius: 6px; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', Monaco, 'Courier New', monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto; }
        .code-block pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
        .no-data { color: var(--text-secondary); font-style: italic; padding: 2rem; text-align: center; }
        footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.8rem; text-align: center; }
        .copy-btn { background: var(--accent); color: var(--bg-primary); border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.85rem; float: right; margin-bottom: 0.5rem; }
        .copy-btn:hover { opacity: 0.9; }
        .playback-controls { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
        .play-btn { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; transition: all 0.2s; }
        .play-btn:hover { background: var(--bg-primary); border-color: var(--accent); }
        .play-btn.playing { background: var(--accent); color: var(--bg-primary); border-color: var(--accent); }
        .play-btn svg { width: 14px; height: 14px; fill: currentColor; }
        .charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem; }
        .chart-item { background: var(--bg-tertiary); border-radius: 6px; overflow: hidden; }
        .chart-item img { width: 100%%; display: block; }
        @media (max-width: 800px) { .charts-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>%s</h1>
            <div class="subtitle">
                <span class="badge badge-blue">v%d</span>
                <span class="badge badge-green">%s BPM</span>
                <span class="badge badge-orange">%s</span>
                <span class="badge badge-blue">%s</span>
            </div>
        </header>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/></svg>
                Audio Comparison
            </div>
            <div class="playback-controls">
                <button class="play-btn" onclick="toggleSync('drums-melodic')" id="btn-drums-melodic">
                    <svg viewBox="0 0 16 16" class="play-icon"><path d="M6.79 5.093A.5.5 0 0 0 6 5.5v5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5z"/></svg>
                    <svg viewBox="0 0 16 16" class="stop-icon" style="display:none"><path d="M5 3.5h1.5v9H5v-9zm4.5 0H11v9H9.5v-9z"/></svg>
                    Drums + Melodic
                </button>
                <button class="play-btn" onclick="toggleSync('all-stems')" id="btn-all-stems">
                    <svg viewBox="0 0 16 16" class="play-icon"><path d="M6.79 5.093A.5.5 0 0 0 6 5.5v5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5z"/></svg>
                    <svg viewBox="0 0 16 16" class="stop-icon" style="display:none"><path d="M5 3.5h1.5v9H5v-9zm4.5 0H11v9H9.5v-9z"/></svg>
                    All Stems
                </button>
                <button class="play-btn" onclick="stopAll()">
                    <svg viewBox="0 0 16 16"><path d="M5 3.5A1.5 1.5 0 0 1 6.5 2h3A1.5 1.5 0 0 1 11 3.5v9A1.5 1.5 0 0 1 9.5 14h-3A1.5 1.5 0 0 1 5 12.5v-9z"/></svg>
                    Stop All
                </button>
            </div>
            <div class="audio-section">
                %s
                %s
                %s
                %s
                %s
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"/></svg>
                Analysis
            </div>
            <div class="stats-grid">
                <div class="stat"><div class="stat-value">%s</div><div class="stat-label">BPM</div></div>
                <div class="stat"><div class="stat-value">%s</div><div class="stat-label">Key</div></div>
                <div class="stat"><div class="stat-value">%s</div><div class="stat-label">Notes</div></div>
                <div class="stat"><div class="stat-value">%s</div><div class="stat-label">Drum Hits</div></div>
                <div class="stat"><div class="stat-value">%s</div><div class="stat-label">Style</div></div>
            </div>
            %s
        </div>

        %s

        %s

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Z"/></svg>
                Strudel Code
                <button class="copy-btn" onclick="copyCode()">Copy Code</button>
            </div>
            <div class="code-block">
                <pre id="strudel-code">%s</pre>
            </div>
        </div>

        <footer>
            Generated by MIDI-grep &bull; %s
        </footer>
    </div>

    <script>
        function copyCode() {
            const code = document.getElementById('strudel-code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                const btn = document.querySelector('.copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy Code', 2000);
            });
        }

        const audioGroups = {
            'drums-melodic': ['audio-drums', 'audio-melodic'],
            'all-stems': ['audio-drums', 'audio-melodic', 'audio-vocals', 'audio-bass']
        };
        let activeGroup = null;

        function stopAll() {
            document.querySelectorAll('audio').forEach(a => { a.pause(); a.currentTime = 0; });
            document.querySelectorAll('.play-btn').forEach(btn => {
                btn.classList.remove('playing');
                const playIcon = btn.querySelector('.play-icon');
                const stopIcon = btn.querySelector('.stop-icon');
                if (playIcon) playIcon.style.display = '';
                if (stopIcon) stopIcon.style.display = 'none';
            });
            activeGroup = null;
        }

        function toggleSync(group) {
            const btn = document.getElementById('btn-' + group);
            const isPlaying = btn.classList.contains('playing');
            stopAll();
            if (!isPlaying) {
                const ids = audioGroups[group];
                const audios = ids.map(id => document.getElementById(id)).filter(a => a);
                if (audios.length > 0) {
                    audios.forEach(a => { a.currentTime = 0; });
                    Promise.all(audios.map(a => a.play().catch(() => {}))).then(() => {
                        btn.classList.add('playing');
                        const playIcon = btn.querySelector('.play-icon');
                        const stopIcon = btn.querySelector('.stop-icon');
                        if (playIcon) playIcon.style.display = 'none';
                        if (stopIcon) stopIcon.style.display = '';
                        activeGroup = group;
                    });
                    audios.forEach(a => { a.onended = () => { if (activeGroup === group) stopAll(); }; });
                }
            }
        }

        document.querySelectorAll('audio').forEach(audio => {
            audio.addEventListener('seeked', () => {
                if (activeGroup) {
                    const ids = audioGroups[activeGroup];
                    const time = audio.currentTime;
                    ids.forEach(id => {
                        const a = document.getElementById(id);
                        if (a && a !== audio) { a.currentTime = time; }
                    });
                }
            });
        });
    </script>
</body>
</html>`,
		trackName,                                    // title
		trackName,                                    // h1
		data.Version,                                 // version badge
		bpm,                                          // bpm badge
		html.EscapeString(key),                       // key badge
		html.EscapeString(style),                     // style badge
		audioPlayer("Melodic", melodicData, "audio-melodic"),
		audioPlayer("Drums", drumsData, "audio-drums"),
		audioPlayer("Vocals", vocalsData, "audio-vocals"),
		audioPlayer("Bass", bassData, "audio-bass"),
		audioPlayer("Strudel Render", renderData, "audio-render"),
		bpm,                                          // stat
		html.EscapeString(key),                       // stat
		notes,                                        // stat
		drumHits,                                     // stat
		html.EscapeString(style),                     // stat
		generateChartsHTML(data.Comparison),          // charts HTML
		generateAIAnalysisCard(data.AIParams),        // AI analysis card
		chartsSection,                                // visual charts
		html.EscapeString(data.StrudelCode),          // code block
		time.Now().Format("2006-01-02 15:04"),        // footer time
	)
}
