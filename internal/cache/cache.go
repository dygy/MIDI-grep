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
	MelodicPath string // Renamed from PianoPath
	DrumsPath   string
	VocalsPath  string // Vocals stem
	BassPath    string // Bass stem
	CacheKey    string
	TrackName   string // Human-readable track name
	CachedAt    time.Time
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

// TrackMetadata stores track info for folder naming
type TrackMetadata struct {
	Title     string    `json:"title"`
	URL       string    `json:"url"`
	VideoID   string    `json:"video_id"`
	CachedAt  time.Time `json:"cached_at"`
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

// sanitizeFolderName creates a safe folder name from track title
func sanitizeFolderName(title string) string {
	// Remove or replace problematic characters for filesystem
	replacer := strings.NewReplacer(
		"/", "-",
		"\\", "-",
		":", "-",
		"*", "",
		"?", "",
		"\"", "",
		"<", "",
		">", "",
		"|", "-",
		"\n", " ",
		"\r", "",
	)
	name := replacer.Replace(title)

	// Trim spaces and dots from ends
	name = strings.Trim(name, " .")

	// If empty after sanitization, use fallback
	if name == "" {
		name = "Unknown Track"
	}

	return name
}

// KeyForURL generates a cache key from a YouTube URL
func KeyForURL(url string) string {
	videoID := ExtractVideoID(url)
	if videoID == "" {
		// Fallback to URL hash
		return hashString(url)
	}
	return "yt_" + videoID
}

// FolderNameForTrack creates folder name from track title
func FolderNameForTrack(title, videoID string) string {
	return sanitizeFolderName(title)
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

// GetByKey retrieves cached stems for the given key (legacy support)
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

	// Try new naming first (melodic.wav), fall back to old (piano.wav)
	melodicPath := filepath.Join(cacheSubdir, "melodic.wav")
	if !fileExists(melodicPath) {
		melodicPath = filepath.Join(cacheSubdir, "piano.wav")
	}
	drumsPath := filepath.Join(cacheSubdir, "drums.wav")
	vocalsPath := filepath.Join(cacheSubdir, "vocals.wav")
	bassPath := filepath.Join(cacheSubdir, "bass.wav")

	// At least one stem must exist
	melodicExists := fileExists(melodicPath)
	drumsExists := fileExists(drumsPath)
	vocalsExists := fileExists(vocalsPath)
	bassExists := fileExists(bassPath)

	if !melodicExists && !drumsExists {
		return nil, false
	}

	// Load track metadata if exists
	var trackName string
	metaPath := filepath.Join(cacheSubdir, "metadata.json")
	if metaData, err := os.ReadFile(metaPath); err == nil {
		var meta TrackMetadata
		if json.Unmarshal(metaData, &meta) == nil {
			trackName = meta.Title
		}
	}

	result := &CachedStems{
		CacheKey:  key,
		TrackName: trackName,
		CachedAt:  info.ModTime(),
	}

	if melodicExists {
		result.MelodicPath = melodicPath
	}
	if drumsExists {
		result.DrumsPath = drumsPath
	}
	if vocalsExists {
		result.VocalsPath = vocalsPath
	}
	if bassExists {
		result.BassPath = bassPath
	}

	return result, true
}

// UpdateTrackMetadata updates the track metadata and renames folder if needed
func (c *StemCache) UpdateTrackMetadata(key, title, url string) (string, error) {
	cacheSubdir := filepath.Join(c.dir, key)
	info, err := os.Stat(cacheSubdir)
	if err != nil || !info.IsDir() {
		return key, fmt.Errorf("cache entry not found: %s", key)
	}

	// Save metadata
	meta := TrackMetadata{
		Title:    title,
		URL:      url,
		VideoID:  ExtractVideoID(url),
		CachedAt: time.Now(),
	}
	metaData, _ := json.MarshalIndent(meta, "", "  ")
	metaPath := filepath.Join(cacheSubdir, "metadata.json")
	if err := os.WriteFile(metaPath, metaData, 0644); err != nil {
		return key, err
	}

	// Rename folder if it's using old yt_xxx format
	if strings.HasPrefix(key, "yt_") && title != "" {
		videoID := ExtractVideoID(url)
		newFolderName := FolderNameForTrack(title, videoID)
		newPath := filepath.Join(c.dir, newFolderName)

		// Only rename if new path doesn't exist (check for directory, not file)
		if _, err := os.Stat(newPath); os.IsNotExist(err) {
			if err := os.Rename(cacheSubdir, newPath); err == nil {
				return newFolderName, nil
			}
			// Rename failed, continue with old key
		}
		// If newPath already exists (same track cached before), return it as the key
		// since the stems are the same
		if info, err := os.Stat(newPath); err == nil && info.IsDir() {
			// Check if the existing folder has valid stems
			melodicPath := filepath.Join(newPath, "melodic.wav")
			pianoPath := filepath.Join(newPath, "piano.wav")
			if fileExists(melodicPath) || fileExists(pianoPath) {
				// Return the track-named folder as key (it has valid stems)
				return newFolderName, nil
			}
		}
	}

	return key, nil
}

// GetByTrackName finds cache by track name folder
func (c *StemCache) GetByTrackName(trackName string) (*CachedStems, bool) {
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		return nil, false
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if strings.Contains(entry.Name(), trackName) {
			return c.Get(entry.Name())
		}
	}
	return nil, false
}

// StemPaths contains all stem file paths for caching
type StemPaths struct {
	MelodicPath string
	DrumsPath   string
	VocalsPath  string
	BassPath    string
}

// Put stores stems in the cache with track name
func (c *StemCache) Put(key string, melodicPath, drumsPath string) (*CachedStems, error) {
	return c.PutWithMetadata(key, &StemPaths{MelodicPath: melodicPath, DrumsPath: drumsPath}, "", "")
}

// PutAllStems stores all 4 stems in the cache
func (c *StemCache) PutAllStems(key string, stems *StemPaths) (*CachedStems, error) {
	return c.PutWithMetadata(key, stems, "", "")
}

// PutWithMetadata stores stems with track metadata
func (c *StemCache) PutWithMetadata(key string, stems *StemPaths, trackTitle, url string) (*CachedStems, error) {
	// Determine folder name
	folderName := key
	if trackTitle != "" {
		videoID := ExtractVideoID(url)
		folderName = FolderNameForTrack(trackTitle, videoID)
	}

	cacheSubdir := filepath.Join(c.dir, folderName)

	// Create cache subdirectory
	if err := os.MkdirAll(cacheSubdir, 0755); err != nil {
		return nil, fmt.Errorf("create cache subdir: %w", err)
	}

	result := &CachedStems{
		CacheKey:  folderName,
		TrackName: trackTitle,
		CachedAt:  time.Now(),
	}

	// Copy melodic stem if exists
	if stems.MelodicPath != "" && fileExists(stems.MelodicPath) {
		dst := filepath.Join(cacheSubdir, "melodic.wav")
		if err := copyFile(stems.MelodicPath, dst); err != nil {
			return nil, fmt.Errorf("cache melodic stem: %w", err)
		}
		result.MelodicPath = dst
	}

	// Copy drums stem if exists
	if stems.DrumsPath != "" && fileExists(stems.DrumsPath) {
		dst := filepath.Join(cacheSubdir, "drums.wav")
		if err := copyFile(stems.DrumsPath, dst); err != nil {
			return nil, fmt.Errorf("cache drums stem: %w", err)
		}
		result.DrumsPath = dst
	}

	// Copy vocals stem if exists
	if stems.VocalsPath != "" && fileExists(stems.VocalsPath) {
		dst := filepath.Join(cacheSubdir, "vocals.wav")
		if err := copyFile(stems.VocalsPath, dst); err != nil {
			return nil, fmt.Errorf("cache vocals stem: %w", err)
		}
		result.VocalsPath = dst
	}

	// Copy bass stem if exists
	if stems.BassPath != "" && fileExists(stems.BassPath) {
		dst := filepath.Join(cacheSubdir, "bass.wav")
		if err := copyFile(stems.BassPath, dst); err != nil {
			return nil, fmt.Errorf("cache bass stem: %w", err)
		}
		result.BassPath = dst
	}

	// Write cache version
	versionPath := filepath.Join(cacheSubdir, ".version")
	if err := os.WriteFile(versionPath, []byte(computedVersion), 0644); err != nil {
		return nil, fmt.Errorf("write cache version: %w", err)
	}

	// Write track metadata
	if trackTitle != "" {
		meta := TrackMetadata{
			Title:    trackTitle,
			URL:      url,
			VideoID:  ExtractVideoID(url),
			CachedAt: time.Now(),
		}
		metaData, _ := json.MarshalIndent(meta, "", "  ")
		metaPath := filepath.Join(cacheSubdir, "metadata.json")
		os.WriteFile(metaPath, metaData, 0644)
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

// ExtractVideoID extracts video ID from various YouTube URL formats
func ExtractVideoID(url string) string {
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

// GetVersionDir returns the path to a version directory
func (c *StemCache) GetVersionDir(key string, version int) string {
	return filepath.Join(c.dir, key, fmt.Sprintf("v%03d", version))
}

// SaveOutput saves a generated Strudel output to a version directory
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

	// Create version directory
	versionDir := filepath.Join(cacheSubdir, fmt.Sprintf("v%03d", output.Version))
	if err := os.MkdirAll(versionDir, 0755); err != nil {
		return fmt.Errorf("create version dir: %w", err)
	}

	// Save JSON metadata
	metaPath := filepath.Join(versionDir, "metadata.json")
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}
	if err := os.WriteFile(metaPath, data, 0644); err != nil {
		return fmt.Errorf("write metadata: %w", err)
	}

	// Save Strudel code (only in version directory, no legacy files)
	strudelPath := filepath.Join(versionDir, "output.strudel")
	if err := os.WriteFile(strudelPath, []byte(output.Code), 0644); err != nil {
		return fmt.Errorf("write strudel file: %w", err)
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
		// Check for version directories (v001, v002, etc.)
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "v") && len(entry.Name()) == 4 {
			metaPath := filepath.Join(cacheSubdir, entry.Name(), "metadata.json")
			data, err := os.ReadFile(metaPath)
			if err != nil {
				continue
			}

			var output CachedOutput
			if err := json.Unmarshal(data, &output); err != nil {
				continue
			}
			outputs = append(outputs, &output)
			continue
		}

		// Legacy: check for output_v*.json files
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

// GetLatestVersion returns the latest version number for a cache key
func (c *StemCache) GetLatestVersion(key string) int {
	outputs, _ := c.GetOutputHistory(key)
	if len(outputs) == 0 {
		return 0
	}
	return outputs[len(outputs)-1].Version
}

// PianoPath returns the melodic path (backward compatibility)
func (cs *CachedStems) PianoPath() string {
	return cs.MelodicPath
}
