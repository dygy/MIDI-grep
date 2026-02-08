package server

import (
	"context"
	"embed"
	"fmt"
	"html/template"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

//go:embed templates/*.html
var templatesFS embed.FS

//go:embed static/*
var staticFS embed.FS

// Config holds server configuration
type Config struct {
	Port       int
	ScriptsDir string
	DevMode    bool
}

// Server is the HTTP server
type Server struct {
	config    Config
	router    *chi.Mux
	templates *template.Template
	logger    *slog.Logger
	jobs      *JobManager
}

// New creates a new server
func New(cfg Config) (*Server, error) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	// Parse templates
	tmpl, err := template.ParseFS(templatesFS, "templates/*.html")
	if err != nil {
		return nil, fmt.Errorf("parse templates: %w", err)
	}

	s := &Server{
		config:    cfg,
		router:    chi.NewRouter(),
		templates: tmpl,
		logger:    logger,
		jobs:      NewJobManager(cfg.ScriptsDir),
	}

	s.setupRoutes()
	return s, nil
}

// setupRoutes configures all HTTP routes
func (s *Server) setupRoutes() {
	r := s.router

	// Middleware
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Compress(5))

	// Static files
	staticSub, _ := fs.Sub(staticFS, "static")
	r.Handle("/static/*", http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))

	// Pages
	r.Get("/", s.handleIndex)
	r.Get("/health", s.handleHealth)

	// API
	r.Post("/upload", s.handleUpload)
	r.Get("/status/{id}", s.handleStatus)
	r.Get("/result/{id}", s.handleResult)
	r.Get("/download/{id}/midi", s.handleDownloadMIDI)
	r.Get("/audio/{id}/{stem}", s.handleAudioStem)
}

// Run starts the server
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)

	srv := &http.Server{
		Addr:         addr,
		Handler:      s.router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 5 * time.Minute, // Long for SSE
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh

		s.logger.Info("shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			s.logger.Error("shutdown error", slog.Any("error", err))
		}
		close(done)
	}()

	s.logger.Info("server starting", slog.Int("port", s.config.Port))
	fmt.Printf("\n  MIDI-grep web interface running at: http://localhost:%d\n\n", s.config.Port)

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}

	<-done
	return nil
}
