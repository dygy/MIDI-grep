package drums

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/arkadiishvartcman/midi-grep/internal/exec"
)

// Hit represents a single detected drum hit
type Hit struct {
	Type               string  `json:"type"`
	Time               float64 `json:"time"`
	Velocity           int     `json:"velocity"`
	VelocityNormalized float64 `json:"velocity_normalized"`
	Confidence         float64 `json:"confidence"`
	OriginalTime       float64 `json:"original_time"`
}

// Stats contains hit count statistics
type Stats struct {
	Total  int            `json:"total"`
	ByType map[string]int `json:"by_type"`
}

// DetectionResult contains the full drum detection output
type DetectionResult struct {
	Hits     []Hit   `json:"hits"`
	Stats    Stats   `json:"stats"`
	Tempo    float64 `json:"tempo"`
	Quantize int     `json:"quantize"`
	Duration float64 `json:"duration"`
}

// Detector wraps the Python drum detection script
type Detector struct {
	runner *exec.Runner
}

// NewDetector creates a new drum detector
func NewDetector(runner *exec.Runner) *Detector {
	return &Detector{runner: runner}
}

// Detect runs drum detection on an audio file
func (d *Detector) Detect(ctx context.Context, audioPath, outputJSON string, quantize int, bpm float64) (*DetectionResult, error) {
	// Build arguments
	args := []string{audioPath, outputJSON, "--quantize", strconv.Itoa(quantize)}

	// Add BPM if specified (> 0)
	if bpm > 0 {
		args = append(args, "--bpm", fmt.Sprintf("%.1f", bpm))
	}

	// Run detection script
	result, err := d.runner.RunScript(ctx, "detect_drums.py", args...)
	if err != nil {
		if result != nil && result.ExitCode != 0 {
			return nil, fmt.Errorf("drum detection failed: %s", result.Stderr)
		}
		return nil, fmt.Errorf("drum detection: %w", err)
	}

	// Read and parse output JSON
	data, err := os.ReadFile(outputJSON)
	if err != nil {
		return nil, fmt.Errorf("read drum detection output: %w", err)
	}

	var detection DetectionResult
	if err := json.Unmarshal(data, &detection); err != nil {
		return nil, fmt.Errorf("parse drum detection output: %w", err)
	}

	return &detection, nil
}

// HitsByType groups hits by their drum type
func (r *DetectionResult) HitsByType() map[string][]Hit {
	grouped := make(map[string][]Hit)
	for _, hit := range r.Hits {
		grouped[hit.Type] = append(grouped[hit.Type], hit)
	}
	return grouped
}

// HasDrumType checks if a specific drum type was detected
func (r *DetectionResult) HasDrumType(drumType string) bool {
	count, ok := r.Stats.ByType[drumType]
	return ok && count > 0
}

// GetHitsForType returns all hits of a specific type
func (r *DetectionResult) GetHitsForType(drumType string) []Hit {
	var hits []Hit
	for _, hit := range r.Hits {
		if hit.Type == drumType {
			hits = append(hits, hit)
		}
	}
	return hits
}
