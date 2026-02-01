package server

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/arkadiishvartcman/midi-grep/internal/audio"
	"github.com/go-chi/chi/v5"
)

const maxUploadSize = 100 * 1024 * 1024 // 100MB

// handleIndex serves the main upload page
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	s.render(w, "index.html", nil)
}

// handleHealth returns server health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

// handleUpload processes file uploads or YouTube URLs
func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	// Limit upload size
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	if err := r.ParseMultipartForm(maxUploadSize); err != nil && !strings.Contains(err.Error(), "no multipart") {
		s.renderError(w, "File too large. Maximum size is 100MB.", http.StatusBadRequest)
		return
	}

	// Check for YouTube URL first
	youtubeURL := r.FormValue("url")
	if youtubeURL != "" && audio.IsYouTubeURL(youtubeURL) {
		s.handleYouTubeURL(w, r, youtubeURL)
		return
	}

	// Handle file upload
	file, header, err := r.FormFile("audio")
	if err != nil {
		// If no file and no URL, show error
		if youtubeURL != "" {
			s.renderError(w, "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be link.", http.StatusBadRequest)
		} else {
			s.renderError(w, "Please upload an audio file or paste a YouTube URL.", http.StatusBadRequest)
		}
		return
	}
	defer file.Close()

	// Validate file extension
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext != ".wav" && ext != ".mp3" {
		s.renderError(w, "Unsupported format. Please upload a WAV or MP3 file.", http.StatusBadRequest)
		return
	}

	// Create job and save file
	job := s.jobs.Create()

	// Save uploaded file
	inputPath := filepath.Join(job.WorkDir, "input"+ext)
	dst, err := os.Create(inputPath)
	if err != nil {
		s.renderError(w, "Failed to save file.", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	if _, err := io.Copy(dst, file); err != nil {
		s.renderError(w, "Failed to save file.", http.StatusInternalServerError)
		return
	}

	job.InputPath = inputPath
	job.Filename = header.Filename

	// Start processing in background
	go s.jobs.Process(job)

	// Return processing partial with job ID for polling
	s.render(w, "processing.html", map[string]any{
		"JobID":    job.ID,
		"Filename": header.Filename,
	})
}

// handleYouTubeURL processes a YouTube URL using yt-dlp
func (s *Server) handleYouTubeURL(w http.ResponseWriter, r *http.Request, url string) {
	// Create job
	job := s.jobs.Create()

	// Get video title for display
	downloader := audio.NewYouTubeDownloader()
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	title, err := downloader.GetVideoTitle(ctx, url)
	cancel()

	if err != nil || title == "" {
		title = "YouTube Video"
	}

	job.Filename = title
	job.YouTubeURL = url

	// Start processing in background (will download first)
	go s.jobs.ProcessYouTube(job, url)

	// Return processing partial
	s.render(w, "processing.html", map[string]any{
		"JobID":    job.ID,
		"Filename": title,
	})
}

// handleStatus returns current job status via SSE
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "id")
	job := s.jobs.Get(jobID)

	if job == nil {
		s.renderError(w, "Job not found.", http.StatusNotFound)
		return
	}

	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	// Send updates until job completes
	for {
		select {
		case <-r.Context().Done():
			return
		case update := <-job.Updates:
			// Send SSE event
			fmt.Fprintf(w, "event: progress\n")
			fmt.Fprintf(w, "data: %s\n\n", update)
			flusher.Flush()

			// Check if complete
			if job.Status == StatusComplete || job.Status == StatusFailed {
				fmt.Fprintf(w, "event: done\n")
				fmt.Fprintf(w, "data: %s\n\n", job.Status)
				flusher.Flush()
				return
			}
		}
	}
}

// handleResult returns the final result
func (s *Server) handleResult(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "id")
	job := s.jobs.Get(jobID)

	if job == nil {
		s.renderError(w, "Job not found.", http.StatusNotFound)
		return
	}

	if job.Status == StatusFailed {
		s.render(w, "error.html", map[string]any{
			"Error": job.Error,
		})
		return
	}

	if job.Status != StatusComplete {
		s.render(w, "processing.html", map[string]any{
			"JobID":    job.ID,
			"Filename": job.Filename,
			"Stage":    job.Stage,
		})
		return
	}

	s.render(w, "result.html", map[string]any{
		"JobID":       job.ID,
		"Filename":    job.Filename,
		"BPM":         fmt.Sprintf("%.0f", job.Result.BPM),
		"BPMConf":     fmt.Sprintf("%.0f%%", job.Result.BPMConfidence*100),
		"Key":         job.Result.Key,
		"KeyConf":     fmt.Sprintf("%.0f%%", job.Result.KeyConfidence*100),
		"NotesCount":  job.Result.NotesRetained,
		"StrudelCode": job.Result.StrudelCode,
	})
}

// handleDownloadMIDI serves the cleaned MIDI file
func (s *Server) handleDownloadMIDI(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "id")
	job := s.jobs.Get(jobID)

	if job == nil || job.Status != StatusComplete {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}

	midiPath := filepath.Join(job.WorkDir, "clean.mid")
	if _, err := os.Stat(midiPath); os.IsNotExist(err) {
		http.Error(w, "MIDI file not available", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "audio/midi")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.mid\"", job.Filename))
	http.ServeFile(w, r, midiPath)
}

// render renders a template
func (s *Server) render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := s.templates.ExecuteTemplate(w, name, data); err != nil {
		s.logger.Error("template error", "template", name, "error", err)
		http.Error(w, "Internal error", http.StatusInternalServerError)
	}
}

// renderError renders an error message
func (s *Server) renderError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	s.templates.ExecuteTemplate(w, "error.html", map[string]any{
		"Error": message,
	})
}
