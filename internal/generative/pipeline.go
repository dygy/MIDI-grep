// Package generative provides RAVE-based generative sound model integration.
// This enables training neural synthesizers on track material for full note control.
package generative

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dygy/midi-grep/internal/exec"
)

// PipelineConfig configures the generative pipeline.
type PipelineConfig struct {
	ModelsPath          string  // Path to model repository
	GitHubRepo          string  // GitHub repo for sync (optional)
	TrainingMode        string  // "granular" (fast) or "rave" (quality)
	SimilarityThreshold float64 // Min similarity for model reuse (0.0-1.0)
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() PipelineConfig {
	return PipelineConfig{
		ModelsPath:          "models",
		TrainingMode:        "granular",
		SimilarityThreshold: 0.88,
	}
}

// ProcessResult contains the pipeline output.
type ProcessResult struct {
	StrudelPath string                 `json:"strudel_path"`
	Models      map[string]ModelInfo   `json:"models"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ModelInfo describes a trained model.
type ModelInfo struct {
	ModelID string `json:"model_id"`
	IsNew   bool   `json:"is_new"`
	URL     string `json:"url"`
}

// Pipeline wraps the Python generative pipeline.
type Pipeline struct {
	config PipelineConfig
	runner *exec.Runner
}

// NewPipeline creates a new generative pipeline.
func NewPipeline(scriptsDir string, config PipelineConfig) *Pipeline {
	return &Pipeline{
		config: config,
		runner: exec.NewRunner("", scriptsDir),
	}
}

// ProcessTrack runs the full generative pipeline on separated stems.
func (p *Pipeline) ProcessTrack(ctx context.Context, stemsDir, trackID, outputDir string) (*ProcessResult, error) {
	args := []string{
		"process",
		stemsDir,
		"--track-id", trackID,
		"--output", outputDir,
		"--models", p.config.ModelsPath,
		"--mode", p.config.TrainingMode,
		"--threshold", fmt.Sprintf("%.2f", p.config.SimilarityThreshold),
	}

	if p.config.GitHubRepo != "" {
		args = append(args, "--github", p.config.GitHubRepo)
	}

	result, err := p.runner.RunModule(ctx, "rave.cli", args...)
	if err != nil {
		return nil, fmt.Errorf("generative pipeline failed: %w", err)
	}

	var processResult ProcessResult
	if err := json.Unmarshal([]byte(result.Stdout), &processResult); err != nil {
		// Pipeline outputs progress to stdout too, look for JSON at the end
		return nil, fmt.Errorf("failed to parse pipeline output: %w\nOutput: %s", err, result.Stdout)
	}

	return &processResult, nil
}

// TrainModel trains a new model from audio.
func (p *Pipeline) TrainModel(ctx context.Context, audioPath, modelName string) (*ModelMetadata, error) {
	args := []string{
		"train",
		audioPath,
		"--name", modelName,
		"--output", p.config.ModelsPath,
		"--mode", p.config.TrainingMode,
		"--add-to-repo",
	}

	if p.config.GitHubRepo != "" {
		args = append(args, "--github", p.config.GitHubRepo, "--sync")
	}

	result, err := p.runner.RunModule(ctx, "rave.cli", args...)
	if err != nil {
		return nil, fmt.Errorf("model training failed: %w", err)
	}

	var metadata ModelMetadata
	if err := json.Unmarshal([]byte(result.Stdout), &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse training output: %w", err)
	}

	return &metadata, nil
}

// ModelMetadata contains trained model info.
type ModelMetadata struct {
	Name        string  `json:"name"`
	Type        string  `json:"type"`
	SourceAudio string  `json:"source_audio"`
	SampleRate  int     `json:"sample_rate"`
	NumGrains   int     `json:"num_grains,omitempty"`
	GrainDurMS  int     `json:"grain_duration_ms,omitempty"`
	LatentDim   int     `json:"latent_dim,omitempty"`
	Created     string  `json:"created"`
}

// SearchResult contains model similarity search results.
type SearchResult struct {
	ModelID    string  `json:"model_id"`
	Similarity float64 `json:"similarity"`
	URL        string  `json:"url"`
}

// SearchSimilar finds models with similar timbre.
func (p *Pipeline) SearchSimilar(ctx context.Context, audioPath string, threshold float64) ([]SearchResult, error) {
	args := []string{
		"search",
		audioPath,
		"--models", p.config.ModelsPath,
		"--threshold", fmt.Sprintf("%.2f", threshold),
	}

	if p.config.GitHubRepo != "" {
		args = append(args, "--github", p.config.GitHubRepo)
	}

	_, err := p.runner.RunModule(ctx, "rave.cli", args...)
	if err != nil {
		return nil, fmt.Errorf("similarity search failed: %w", err)
	}

	// Parse output (text format, not JSON)
	// For now return empty - the CLI outputs human-readable text
	return nil, nil
}

// ListModels returns all available models.
func (p *Pipeline) ListModels(ctx context.Context) ([]string, error) {
	args := []string{
		"list",
		"--models", p.config.ModelsPath,
	}

	if p.config.GitHubRepo != "" {
		args = append(args, "--github", p.config.GitHubRepo)
	}

	_, err := p.runner.RunModule(ctx, "rave.cli", args...)
	if err != nil {
		return nil, fmt.Errorf("list models failed: %w", err)
	}

	// Read from index.json directly for structured data
	indexPath := filepath.Join(p.config.ModelsPath, "index.json")
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, nil // No models yet
	}

	var index struct {
		Models map[string]interface{} `json:"models"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, err
	}

	models := make([]string, 0, len(index.Models))
	for id := range index.Models {
		models = append(models, id)
	}

	return models, nil
}

// StartServer starts the local model server (blocking).
func (p *Pipeline) StartServer(ctx context.Context, port int) error {
	args := []string{
		"serve",
		"--models", p.config.ModelsPath,
		"--port", fmt.Sprintf("%d", port),
	}

	_, err := p.runner.RunModule(ctx, "rave.cli", args...)
	return err
}

// SyncToGitHub pushes models to GitHub.
func (p *Pipeline) SyncToGitHub(ctx context.Context) error {
	if p.config.GitHubRepo == "" {
		return fmt.Errorf("no GitHub repo configured")
	}

	args := []string{
		"sync",
		"--models", p.config.ModelsPath,
		"--github", p.config.GitHubRepo,
		"--push",
	}

	_, err := p.runner.RunModule(ctx, "rave.cli", args...)
	return err
}

// SyncFromGitHub pulls models from GitHub.
func (p *Pipeline) SyncFromGitHub(ctx context.Context) error {
	if p.config.GitHubRepo == "" {
		return fmt.Errorf("no GitHub repo configured")
	}

	args := []string{
		"sync",
		"--models", p.config.ModelsPath,
		"--github", p.config.GitHubRepo,
		"--pull",
	}

	_, err := p.runner.RunModule(ctx, "rave.cli", args...)
	return err
}
