package server

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/dygy/midi-grep/internal/analysis"
	"github.com/dygy/midi-grep/internal/audio"
	"github.com/dygy/midi-grep/internal/exec"
	"github.com/dygy/midi-grep/internal/midi"
	"github.com/dygy/midi-grep/internal/strudel"
)

// Job status constants
type JobStatus string

const (
	StatusPending    JobStatus = "pending"
	StatusProcessing JobStatus = "processing"
	StatusComplete   JobStatus = "complete"
	StatusFailed     JobStatus = "failed"
)

// JobResult holds processing results
type JobResult struct {
	StrudelCode   string
	BPM           float64
	BPMConfidence float64
	Key           string
	KeyConfidence float64
	NotesRetained int
	NotesRemoved  int
	// Stem paths for audio playback
	MelodicPath string
	DrumsPath   string
	BassPath    string
	VocalsPath  string
	RenderPath  string
}

// Job represents a processing job
type Job struct {
	ID         string
	Status     JobStatus
	Stage      string
	Filename   string
	InputPath  string
	YouTubeURL string
	WorkDir    string
	Result     *JobResult
	Error      string
	Updates    chan string
	CreatedAt  time.Time
}

// JobManager manages processing jobs
type JobManager struct {
	jobs       map[string]*Job
	mu         sync.RWMutex
	scriptsDir string
}

// NewJobManager creates a new job manager
func NewJobManager(scriptsDir string) *JobManager {
	return &JobManager{
		jobs:       make(map[string]*Job),
		scriptsDir: scriptsDir,
	}
}

// Create creates a new job
func (m *JobManager) Create() *Job {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := fmt.Sprintf("%d", time.Now().UnixNano())

	// Create work directory
	workDir, _ := os.MkdirTemp("", "midi-grep-job-*")

	job := &Job{
		ID:        id,
		Status:    StatusPending,
		Stage:     "Uploading...",
		WorkDir:   workDir,
		Updates:   make(chan string, 10),
		CreatedAt: time.Now(),
	}

	m.jobs[id] = job
	return job
}

// Get retrieves a job by ID
func (m *JobManager) Get(id string) *Job {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.jobs[id]
}

// Process runs the extraction pipeline for a job
func (m *JobManager) Process(job *Job) {
	defer close(job.Updates)
	defer func() {
		// Cleanup after 10 minutes
		time.AfterFunc(10*time.Minute, func() {
			os.RemoveAll(job.WorkDir)
			m.mu.Lock()
			delete(m.jobs, job.ID)
			m.mu.Unlock()
		})
	}()

	job.Status = StatusProcessing
	ctx := context.Background()

	runner := exec.NewRunner("", m.scriptsDir)

	// Stage 1: Validate
	job.Stage = "Validating input file..."
	job.Updates <- job.Stage

	format, err := audio.ValidateInput(job.InputPath)
	if err != nil {
		job.Status = StatusFailed
		job.Error = err.Error()
		job.Updates <- fmt.Sprintf("Error: %s", err)
		return
	}
	job.Updates <- fmt.Sprintf("Valid %s file", format)

	// Stage 2: Stem separation
	job.Stage = "Separating stems..."
	job.Updates <- job.Stage

	separator := audio.NewStemSeparator(runner)
	stemCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	pianoPath, err := separator.Separate(stemCtx, job.InputPath, job.WorkDir)
	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Stem separation failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- "Piano stem extracted"

	// Capture all stem paths for audio playback
	melodicPath := job.WorkDir + "/melodic.wav"
	drumsPath := job.WorkDir + "/drums.wav"
	bassPath := job.WorkDir + "/bass.wav"
	vocalsPath := job.WorkDir + "/vocals.wav"

	// Stage 3: Analysis
	job.Stage = "Analyzing audio..."
	job.Updates <- job.Stage

	analyzer := analysis.NewAnalyzer(runner)
	analysisPath := job.WorkDir + "/analysis.json"
	analysisResult, err := analyzer.Analyze(ctx, pianoPath, analysisPath)
	if err != nil {
		// Non-fatal, use defaults
		analysisResult = analysis.DefaultResult()
		job.Updates <- "Using default BPM (120)"
	} else {
		job.Updates <- fmt.Sprintf("BPM: %.0f, Key: %s", analysisResult.BPM, analysisResult.Key)
	}

	// Stage 4: Transcription
	job.Stage = "Transcribing to MIDI..."
	job.Updates <- job.Stage

	transcriber := midi.NewTranscriber(runner)
	rawMIDI := job.WorkDir + "/raw.mid"
	transcribeCtx, cancel2 := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel2()

	if err := transcriber.Transcribe(transcribeCtx, pianoPath, rawMIDI); err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Transcription failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- "MIDI transcription complete"

	// Stage 5: Cleanup
	job.Stage = "Cleaning up MIDI..."
	job.Updates <- job.Stage

	cleaner := midi.NewCleaner(runner)
	notesJSON := job.WorkDir + "/notes.json"
	cleanResult, err := cleaner.Clean(ctx, rawMIDI, notesJSON, 16)
	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("MIDI cleanup failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- fmt.Sprintf("%d notes retained", cleanResult.Retained)

	// Stage 6: Generate Strudel
	job.Stage = "Generating Strudel code..."
	job.Updates <- job.Stage

	generator := strudel.NewGenerator(16)
	strudelCode := generator.Generate(cleanResult.Notes, analysisResult)

	// Complete
	job.Result = &JobResult{
		StrudelCode:   strudelCode,
		BPM:           analysisResult.BPM,
		BPMConfidence: analysisResult.BPMConfidence,
		Key:           analysisResult.Key,
		KeyConfidence: analysisResult.KeyConfidence,
		NotesRetained: cleanResult.Retained,
		NotesRemoved:  cleanResult.Removed,
		MelodicPath:   melodicPath,
		DrumsPath:     drumsPath,
		BassPath:      bassPath,
		VocalsPath:    vocalsPath,
	}
	job.Status = StatusComplete
	job.Stage = "Complete!"
	job.Updates <- job.Stage
}

// ProcessYouTube downloads from YouTube and then processes
func (m *JobManager) ProcessYouTube(job *Job, url string) {
	defer close(job.Updates)
	defer func() {
		// Cleanup after 10 minutes
		time.AfterFunc(10*time.Minute, func() {
			os.RemoveAll(job.WorkDir)
			m.mu.Lock()
			delete(m.jobs, job.ID)
			m.mu.Unlock()
		})
	}()

	job.Status = StatusProcessing
	ctx := context.Background()

	// Stage 0: Download from YouTube
	job.Stage = "Downloading from YouTube..."
	job.Updates <- job.Stage

	downloader := audio.NewYouTubeDownloader()
	downloadCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	audioPath, err := downloader.Download(downloadCtx, url, job.WorkDir)
	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Download failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.InputPath = audioPath
	job.Updates <- "Download complete"

	// Now run the normal pipeline
	runner := exec.NewRunner("", m.scriptsDir)

	// Stage 1: Validate
	job.Stage = "Validating audio..."
	job.Updates <- job.Stage

	format, err := audio.ValidateInput(job.InputPath)
	if err != nil {
		job.Status = StatusFailed
		job.Error = err.Error()
		job.Updates <- fmt.Sprintf("Error: %s", err)
		return
	}
	job.Updates <- fmt.Sprintf("Valid %s file", format)

	// Stage 2: Stem separation
	job.Stage = "Separating stems..."
	job.Updates <- job.Stage

	separator := audio.NewStemSeparator(runner)
	stemCtx, stemCancel := context.WithTimeout(ctx, 5*time.Minute)
	defer stemCancel()

	pianoPath, err := separator.Separate(stemCtx, job.InputPath, job.WorkDir)
	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Stem separation failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- "Piano stem extracted"

	// Capture all stem paths for audio playback
	melodicPath := job.WorkDir + "/melodic.wav"
	drumsPath := job.WorkDir + "/drums.wav"
	bassPath := job.WorkDir + "/bass.wav"
	vocalsPath := job.WorkDir + "/vocals.wav"

	// Stage 3: Analysis
	job.Stage = "Analyzing audio..."
	job.Updates <- job.Stage

	analyzer := analysis.NewAnalyzer(runner)
	analysisPath := job.WorkDir + "/analysis.json"
	analysisResult, err := analyzer.Analyze(ctx, pianoPath, analysisPath)
	if err != nil {
		analysisResult = analysis.DefaultResult()
		job.Updates <- "Using default BPM (120)"
	} else {
		job.Updates <- fmt.Sprintf("BPM: %.0f, Key: %s", analysisResult.BPM, analysisResult.Key)
	}

	// Stage 4: Transcription
	job.Stage = "Transcribing to MIDI..."
	job.Updates <- job.Stage

	transcriber := midi.NewTranscriber(runner)
	rawMIDI := job.WorkDir + "/raw.mid"
	transcribeCtx, transcribeCancel := context.WithTimeout(ctx, 3*time.Minute)
	defer transcribeCancel()

	if err := transcriber.Transcribe(transcribeCtx, pianoPath, rawMIDI); err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Transcription failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- "MIDI transcription complete"

	// Stage 5: Cleanup
	job.Stage = "Cleaning up MIDI..."
	job.Updates <- job.Stage

	cleaner := midi.NewCleaner(runner)
	notesJSON := job.WorkDir + "/notes.json"
	cleanResult, err := cleaner.Clean(ctx, rawMIDI, notesJSON, 16)
	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("MIDI cleanup failed: %v", err)
		job.Updates <- job.Error
		return
	}
	job.Updates <- fmt.Sprintf("%d notes retained", cleanResult.Retained)

	// Stage 6: Generate Strudel
	job.Stage = "Generating Strudel code..."
	job.Updates <- job.Stage

	generator := strudel.NewGenerator(16)
	strudelCode := generator.Generate(cleanResult.Notes, analysisResult)

	// Complete
	job.Result = &JobResult{
		StrudelCode:   strudelCode,
		BPM:           analysisResult.BPM,
		BPMConfidence: analysisResult.BPMConfidence,
		Key:           analysisResult.Key,
		KeyConfidence: analysisResult.KeyConfidence,
		NotesRetained: cleanResult.Retained,
		NotesRemoved:  cleanResult.Removed,
		MelodicPath:   melodicPath,
		DrumsPath:     drumsPath,
		BassPath:      bassPath,
		VocalsPath:    vocalsPath,
	}
	job.Status = StatusComplete
	job.Stage = "Complete!"
	job.Updates <- job.Stage
}
