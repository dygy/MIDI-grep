package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// scriptsToHash - files that affect stem extraction (changing these invalidates cache)
var scriptsToHash = []string{
	"separate.py",
}

// computedVersion is calculated once at startup from script hashes
var computedVersion string

// StemCache manages cached stem separation results
type StemCache struct {
	dir string
}

// CachedStems represents cached stem file paths
type CachedStems struct {
	PianoPath string
	DrumsPath string
	CacheKey  string
	CachedAt  time.Time
}

// CachedOutput represents a cached generation result
type CachedOutput struct {
	Code      string            `json:"code"`
	BPM       float64           `json:"bpm"`
	Key       string            `json:"key"`
	Style     string            `json:"style"`
	Genre     string            `json:"genre,omitempty"`
	Notes     int               `json:"notes"`
	DrumHits  int               `json:"drum_hits,omitempty"`
	Version   int               `json:"version"`
	CreatedAt time.Time         `json:"created_at"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// NewStemCache creates a new stem cache in the repository's .cache directory
func NewStemCache() (*StemCache, error) {
	// Find repository root by looking for go.mod
	cacheDir, err := findRepoCacheDir()
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir: %w", err)
	}

	// Compute version from script hashes if not already done
	if computedVersion == "" {
		computedVersion = computeScriptVersion(cacheDir)
	}

	return &StemCache{dir: cacheDir}, nil
}

// computeScriptVersion creates a hash from all scripts that affect stem extraction
func computeScriptVersion(cacheDir string) string {
	// Find scripts directory (go up from .cache/stems to repo root, then scripts/python)
	repoRoot := filepath.Dir(filepath.Dir(cacheDir))
	scriptsDir := filepath.Join(repoRoot, "scripts", "python")

	hasher := sha256.New()

	for _, script := range scriptsToHash {
		scriptPath := filepath.Join(scriptsDir, script)
		data, err := os.ReadFile(scriptPath)
		if err != nil {
			// Script not found - use filename as fallback
			hasher.Write([]byte(script))
			continue
		}
		hasher.Write(data)
	}

	hash := hex.EncodeToString(hasher.Sum(nil))
	return hash[:12] // First 12 chars is enough
}

// GetVersion returns the current cache version (based on script hashes)
func GetVersion() string {
	if computedVersion == "" {
		return "unknown"
	}
	return computedVersion
}

// findRepoCacheDir finds or creates .cache/stems in the repository root
func findRepoCacheDir() (string, error) {
	// Start from current working directory
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working dir: %w", err)
	}

	// Walk up looking for go.mod (repo root marker)
	for {
		if fileExists(filepath.Join(dir, "go.mod")) {
			return filepath.Join(dir, ".cache", "stems"), nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			// Reached filesystem root, use current directory
			cwd, _ := os.Getwd()
			return filepath.Join(cwd, ".cache", "stems"), nil
		}
		dir = parent
	}
}

// KeyForURL generates a cache key from a YouTube URL
func KeyForURL(url string) string {
	videoID := extractYouTubeID(url)
	if videoID == "" {
		// Fallback to URL hash
		return hashString(url)
	}
	return "yt_" + videoID
}

// KeyForFile generates a cache key from a file's content hash
func KeyForFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", fmt.Errorf("hash file: %w", err)
	}

	return "file_" + hex.EncodeToString(hash.Sum(nil))[:16], nil
}

// Get retrieves cached stems for the given key
func (c *StemCache) Get(key string) (*CachedStems, bool) {
	cacheSubdir := filepath.Join(c.dir, key)

	// Check if cache dir exists
	info, err := os.Stat(cacheSubdir)
	if err != nil || !info.IsDir() {
		return nil, false
	}

	// Check cache version (computed from script hashes)
	versionPath := filepath.Join(cacheSubdir, ".version")
	versionData, err := os.ReadFile(versionPath)
	if err != nil || strings.TrimSpace(string(versionData)) != computedVersion {
		// Version mismatch or missing - invalidate cache
		return nil, false
	}

	pianoPath := filepath.Join(cacheSubdir, "piano.wav")
	drumsPath := filepath.Join(cacheSubdir, "drums.wav")

	// At least one stem must exist
	pianoExists := fileExists(pianoPath)
	drumsExists := fileExists(drumsPath)

	if !pianoExists && !drumsExists {
		return nil, false
	}

	result := &CachedStems{
		CacheKey: key,
		CachedAt: info.ModTime(),
	}

	if pianoExists {
		result.PianoPath = pianoPath
	}
	if drumsExists {
		result.DrumsPath = drumsPath
	}

	return result, true
}

// Put stores stems in the cache
func (c *StemCache) Put(key string, pianoPath, drumsPath string) (*CachedStems, error) {
	cacheSubdir := filepath.Join(c.dir, key)

	// Create cache subdirectory
	if err := os.MkdirAll(cacheSubdir, 0755); err != nil {
		return nil, fmt.Errorf("create cache subdir: %w", err)
	}

	result := &CachedStems{
		CacheKey: key,
		CachedAt: time.Now(),
	}

	// Copy piano stem if exists
	if pianoPath != "" && fileExists(pianoPath) {
		dst := filepath.Join(cacheSubdir, "piano.wav")
		if err := copyFile(pianoPath, dst); err != nil {
			return nil, fmt.Errorf("cache piano stem: %w", err)
		}
		result.PianoPath = dst
	}

	// Copy drums stem if exists
	if drumsPath != "" && fileExists(drumsPath) {
		dst := filepath.Join(cacheSubdir, "drums.wav")
		if err := copyFile(drumsPath, dst); err != nil {
			return nil, fmt.Errorf("cache drums stem: %w", err)
		}
		result.DrumsPath = dst
	}

	// Write cache version (computed from script hashes)
	versionPath := filepath.Join(cacheSubdir, ".version")
	if err := os.WriteFile(versionPath, []byte(computedVersion), 0644); err != nil {
		return nil, fmt.Errorf("write cache version: %w", err)
	}

	return result, nil
}

// Clear removes all cached stems
func (c *StemCache) Clear() error {
	return os.RemoveAll(c.dir)
}

// Size returns the total size of cached stems in bytes
func (c *StemCache) Size() (int64, int, error) {
	var totalSize int64
	var count int

	entries, err := os.ReadDir(c.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, 0, nil
		}
		return 0, 0, err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		count++

		subdir := filepath.Join(c.dir, entry.Name())
		files, _ := os.ReadDir(subdir)
		for _, f := range files {
			info, err := f.Info()
			if err == nil {
				totalSize += info.Size()
			}
		}
	}

	return totalSize, count, nil
}

// extractYouTubeID extracts video ID from various YouTube URL formats
func extractYouTubeID(url string) string {
	patterns := []string{
		`youtube\.com/watch\?v=([\w-]+)`,
		`youtube\.com/shorts/([\w-]+)`,
		`youtu\.be/([\w-]+)`,
		`music\.youtube\.com/watch\?v=([\w-]+)`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(url)
		if len(matches) > 1 {
			// Remove any trailing query params
			id := matches[1]
			if idx := strings.Index(id, "&"); idx != -1 {
				id = id[:idx]
			}
			if idx := strings.Index(id, "?"); idx != -1 {
				id = id[:idx]
			}
			return id
		}
	}
	return ""
}

func hashString(s string) string {
	hash := sha256.Sum256([]byte(s))
	return hex.EncodeToString(hash[:])[:16]
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func copyFile(src, dst string) error {
	input, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, input, 0644)
}

// SaveOutput saves a generated Strudel output to the cache
func (c *StemCache) SaveOutput(key string, output *CachedOutput) error {
	cacheSubdir := filepath.Join(c.dir, key)

	// Create cache subdirectory if it doesn't exist
	if err := os.MkdirAll(cacheSubdir, 0755); err != nil {
		return fmt.Errorf("create cache subdir: %w", err)
	}

	// Determine next version number
	outputs, _ := c.GetOutputHistory(key)
	output.Version = len(outputs) + 1
	output.CreatedAt = time.Now()

	// Save as versioned JSON file
	filename := fmt.Sprintf("output_v%03d.json", output.Version)
	outputPath := filepath.Join(cacheSubdir, filename)

	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}

	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("write output: %w", err)
	}

	// Also save the code as a standalone .strudel file for easy access
	strudelPath := filepath.Join(cacheSubdir, fmt.Sprintf("output_v%03d.strudel", output.Version))
	if err := os.WriteFile(strudelPath, []byte(output.Code), 0644); err != nil {
		return fmt.Errorf("write strudel file: %w", err)
	}

	// Update "latest" symlink/copy
	latestPath := filepath.Join(cacheSubdir, "output_latest.strudel")
	_ = os.Remove(latestPath) // Ignore error if doesn't exist
	if err := os.WriteFile(latestPath, []byte(output.Code), 0644); err != nil {
		return fmt.Errorf("write latest: %w", err)
	}

	return nil
}

// GetLatestOutput retrieves the most recent generated output for a cache key
func (c *StemCache) GetLatestOutput(key string) (*CachedOutput, error) {
	outputs, err := c.GetOutputHistory(key)
	if err != nil {
		return nil, err
	}
	if len(outputs) == 0 {
		return nil, nil
	}
	return outputs[len(outputs)-1], nil
}

// GetOutputHistory retrieves all generated outputs for a cache key, sorted by version
func (c *StemCache) GetOutputHistory(key string) ([]*CachedOutput, error) {
	cacheSubdir := filepath.Join(c.dir, key)

	entries, err := os.ReadDir(cacheSubdir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read cache dir: %w", err)
	}

	var outputs []*CachedOutput

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, "output_v") || !strings.HasSuffix(name, ".json") {
			continue
		}

		outputPath := filepath.Join(cacheSubdir, name)
		data, err := os.ReadFile(outputPath)
		if err != nil {
			continue
		}

		var output CachedOutput
		if err := json.Unmarshal(data, &output); err != nil {
			continue
		}

		outputs = append(outputs, &output)
	}

	// Sort by version
	sort.Slice(outputs, func(i, j int) bool {
		return outputs[i].Version < outputs[j].Version
	})

	return outputs, nil
}

// GetCacheDir returns the cache directory for a key (for external access)
func (c *StemCache) GetCacheDir(key string) string {
	return filepath.Join(c.dir, key)
}
