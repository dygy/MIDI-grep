package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/dygy/midi-grep/internal/audio"
	"github.com/dygy/midi-grep/internal/cache"
	"github.com/dygy/midi-grep/internal/pipeline"
	"github.com/dygy/midi-grep/internal/server"
	"github.com/spf13/cobra"
)

var (
	version = "0.1.0"
)

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Use:   "midi-grep",
	Short: "Extract piano riffs from audio and generate Strudel code",
	Long: `MIDI-grep extracts piano parts from audio files and converts them
to playable Strudel patterns for live coding.

Pipeline: audio → stem separation → MIDI transcription → Strudel code`,
	Version: version,
}

var extractCmd = &cobra.Command{
	Use:   "extract",
	Short: "Extract piano riff from audio file or YouTube URL",
	Long: `Extract the piano part from an audio file or YouTube video
and generate Strudel code.

Examples:
  midi-grep extract --input track.wav
  midi-grep extract -i track.mp3 -o riff.strudel --quantize 8
  midi-grep extract --url "https://youtube.com/watch?v=..."`,
	RunE: runExtract,
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the web interface",
	Long: `Start the HTMX-powered web interface for uploading audio
files or pasting YouTube URLs.

Example:
  midi-grep serve --port 8080`,
	RunE: runServe,
}

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train or fine-tune transcription model",
	Long: `Train a custom transcription model on your own audio/MIDI pairs
or fine-tune on genre-specific datasets.

Subcommands:
  prepare   Prepare dataset from audio/MIDI pairs
  run       Run training on prepared dataset
  list      List available models`,
}

var trainPrepareCmd = &cobra.Command{
	Use:   "prepare",
	Short: "Prepare training dataset",
	Long: `Prepare audio/MIDI pairs for training.

Examples:
  midi-grep train prepare --audio-dir ./audio --midi-dir ./midi --output ./dataset
  midi-grep train prepare --maestro --output ./dataset`,
	RunE: runTrainPrepare,
}

var trainRunCmd = &cobra.Command{
	Use:   "run",
	Short: "Run model training",
	Long: `Fine-tune Basic Pitch model on prepared dataset.

Examples:
  midi-grep train run --dataset ./dataset --output ./models/my-model
  midi-grep train run -d ./dataset -o ./models/jazz --epochs 200`,
	RunE: runTrainRun,
}

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Manage transcription models",
	Long: `List, download, or manage transcription models.

Subcommands:
  list      List available models
  info      Show model details`,
}

var modelsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List available models",
	RunE:  runModelsList,
}

var (
	// extract flags
	inputPath   string
	inputURL    string
	outputPath  string
	midiOutput  string
	quantize    int
	soundStyle  string
	verbose     bool
	simplify    bool
	loopOnly    bool
	enableDrums bool
	drumsOnly   bool
	drumKit     string
	arrange       bool
	noCache       bool
	chordMode     bool
	brazilianFunk bool
	genreOverride string // Manual genre override (brazilian_funk, brazilian_phonk, retro_wave, etc.)
	useDeepGenre  bool   // Use deep learning for genre detection
	renderAudio   string // Output path for rendered WAV
	compareAudio  bool   // Compare rendered with original
	stemQuality   string // Stem separation quality (fast, normal, high, best)

	// serve flags
	port int

	// train prepare flags
	audioDir   string
	midiDir    string
	datasetOut string
	useMaestro bool

	// train run flags
	datasetPath  string
	modelOutput  string
	epochs       int
	batchSize    int
	learningRate float64
	baseModel    string
)

func init() {
	rootCmd.AddCommand(extractCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(trainCmd)
	rootCmd.AddCommand(modelsCmd)

	// Train subcommands
	trainCmd.AddCommand(trainPrepareCmd)
	trainCmd.AddCommand(trainRunCmd)

	// Models subcommands
	modelsCmd.AddCommand(modelsListCmd)

	// Extract command flags
	extractCmd.Flags().StringVarP(&inputPath, "input", "i", "", "Input audio file (WAV or MP3)")
	extractCmd.Flags().StringVarP(&inputURL, "url", "u", "", "YouTube URL to extract from")
	extractCmd.Flags().StringVarP(&outputPath, "output", "o", "", "Output file for Strudel code (default: stdout)")
	extractCmd.Flags().StringVar(&midiOutput, "midi-out", "", "Save cleaned MIDI to file")
	extractCmd.Flags().IntVarP(&quantize, "quantize", "q", 16, "Quantization (4, 8, or 16)")
	extractCmd.Flags().StringVarP(&soundStyle, "style", "s", "auto", "Sound style (auto, piano, synth, electronic, jazz, lofi, funk, soul, house, etc.)")
	extractCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
	extractCmd.Flags().BoolVar(&simplify, "simplify", true, "Simplify notes (reduce chords, merge close notes)")
	extractCmd.Flags().BoolVar(&loopOnly, "loop-only", false, "Output only the detected loop pattern")
	extractCmd.Flags().BoolVar(&enableDrums, "drums", true, "Extract and include drum patterns (default: on)")
	extractCmd.Flags().BoolVar(&drumsOnly, "drums-only", false, "Extract only drums (skip melodic processing)")
	extractCmd.Flags().StringVar(&drumKit, "drum-kit", "tr808", "Drum kit to use (tr808, tr909, linn, acoustic, lofi)")
	extractCmd.Flags().BoolVar(&arrange, "arrange", false, "Use arrangement-based generation with chord detection and voicings")
	extractCmd.Flags().BoolVar(&noCache, "no-cache", false, "Skip stem cache (force fresh extraction)")
	extractCmd.Flags().BoolVar(&chordMode, "chords", false, "Use chord-based generation (better for electronic/non-piano music)")
	extractCmd.Flags().BoolVar(&brazilianFunk, "brazilian-funk", false, "Use Brazilian funk/phonk mode (tamborzão drums, 808 bass)")
	extractCmd.Flags().StringVar(&genreOverride, "genre", "", "Override genre detection (brazilian_funk, brazilian_phonk, retro_wave, trance, house, lofi)")
	extractCmd.Flags().BoolVar(&useDeepGenre, "deep-genre", true, "Use deep learning (CLAP) for genre detection")
	extractCmd.Flags().StringVar(&renderAudio, "render", "auto", "Render output to WAV ('auto' saves in cache dir, 'none' to disable)")
	extractCmd.Flags().BoolVar(&compareAudio, "compare", false, "Compare rendered audio with original stems (requires --render)")
	extractCmd.Flags().StringVar(&stemQuality, "quality", "normal", "Stem separation quality: fast, normal, high (better, slower), best (highest, slowest)")

	// Serve command flags
	serveCmd.Flags().IntVarP(&port, "port", "p", 8080, "Port to listen on")

	// Train prepare flags
	trainPrepareCmd.Flags().StringVar(&audioDir, "audio-dir", "", "Directory containing audio files")
	trainPrepareCmd.Flags().StringVar(&midiDir, "midi-dir", "", "Directory containing MIDI files")
	trainPrepareCmd.Flags().StringVarP(&datasetOut, "output", "o", "", "Output directory for dataset")
	trainPrepareCmd.Flags().BoolVar(&useMaestro, "maestro", false, "Download and use MAESTRO dataset")
	trainPrepareCmd.MarkFlagRequired("output")

	// Train run flags
	trainRunCmd.Flags().StringVarP(&datasetPath, "dataset", "d", "", "Path to prepared dataset")
	trainRunCmd.Flags().StringVarP(&modelOutput, "output", "o", "", "Output directory for model")
	trainRunCmd.Flags().IntVarP(&epochs, "epochs", "e", 100, "Number of training epochs")
	trainRunCmd.Flags().IntVarP(&batchSize, "batch-size", "b", 16, "Batch size")
	trainRunCmd.Flags().Float64VarP(&learningRate, "learning-rate", "l", 0.001, "Learning rate")
	trainRunCmd.Flags().StringVar(&baseModel, "base-model", "basic-pitch", "Base model to fine-tune from")
	trainRunCmd.MarkFlagRequired("dataset")
	trainRunCmd.MarkFlagRequired("output")
}

func runExtract(cmd *cobra.Command, args []string) error {
	// Validate inputs
	if inputPath == "" && inputURL == "" {
		return fmt.Errorf("either --input or --url is required")
	}

	if quantize != 4 && quantize != 8 && quantize != 16 {
		return fmt.Errorf("invalid quantize value: %d (must be 4, 8, or 16)", quantize)
	}

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupt
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Fprintln(os.Stderr, "\nInterrupted, cleaning up...")
		cancel()
	}()

	// If URL provided, check cache first then download if needed
	actualInput := inputPath
	var tempDir string
	var cachedStemsPath string
	var trackTitle string

	if inputURL != "" {
		if !audio.IsYouTubeURL(inputURL) {
			return fmt.Errorf("invalid YouTube URL: %s", inputURL)
		}

		// Get track title from YouTube with spinner
		downloader := audio.NewYouTubeDownloader()
		fmt.Print("[0/5] Fetching track info...")

		// Start spinner in background
		spinnerDone := make(chan bool)
		go func() {
			spinner := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
			i := 0
			for {
				select {
				case <-spinnerDone:
					return
				default:
					fmt.Printf("\r[0/5] Fetching track info... %s", spinner[i%len(spinner)])
					i++
					time.Sleep(100 * time.Millisecond)
				}
			}
		}()

		titleCtx, titleCancel := context.WithTimeout(ctx, 30*time.Second)
		title, err := downloader.GetVideoTitle(titleCtx, inputURL)
		titleCancel()
		spinnerDone <- true

		if err == nil && title != "" {
			trackTitle = title
			fmt.Printf("\r[0/5] Track: %s\n", trackTitle)
		} else {
			fmt.Print("\r[0/5] Fetching track info... done\n")
		}

		// Check cache before downloading
		if !noCache {
			stemCache, err := cache.NewStemCache()
			if err == nil {
				cacheKey := cache.KeyForURL(inputURL)
				cached, ok := stemCache.Get(cacheKey)

				// If not found by URL key but we have track title, try by track name
				// (folder may have been renamed from yt_xxx to track name)
				if !ok && trackTitle != "" {
					trackFolderName := cache.FolderNameForTrack(trackTitle, cache.ExtractVideoID(inputURL))
					if cachedByName, okByName := stemCache.Get(trackFolderName); okByName {
						cached = cachedByName
						cacheKey = trackFolderName
						ok = true
					}
				}

				if ok {
					displayName := cacheKey[:8]
					if cached.TrackName != "" {
						displayName = cached.TrackName
					} else if trackTitle != "" {
						// Update metadata and rename folder for existing cache entry
						displayName = trackTitle
						newKey, _ := stemCache.UpdateTrackMetadata(cacheKey, trackTitle, inputURL)
						if newKey != cacheKey {
							// Folder was renamed, update paths
							cacheKey = newKey
							cached, _ = stemCache.Get(cacheKey)
						}
					}
					fmt.Printf("\r[0/5] Using cached stems (%s)\n", displayName)
					if cached != nil {
						cachedStemsPath = filepath.Dir(cached.MelodicPath)
						if cachedStemsPath == "" && cached.DrumsPath != "" {
							cachedStemsPath = filepath.Dir(cached.DrumsPath)
						}
					}
				}
			}
		}

		// Only download if not cached
		if cachedStemsPath == "" {
			fmt.Print("[0/5] Downloading from YouTube...")

			var err error
			tempDir, err = os.MkdirTemp("", "midi-grep-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tempDir)

			// Start download spinner
			downloadSpinnerDone := make(chan bool)
			go func() {
				spinner := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
				i := 0
				for {
					select {
					case <-downloadSpinnerDone:
						return
					default:
						fmt.Printf("\r[0/5] Downloading from YouTube... %s", spinner[i%len(spinner)])
						i++
						time.Sleep(100 * time.Millisecond)
					}
				}
			}()

			downloadCtx, downloadCancel := context.WithTimeout(ctx, 5*time.Minute)
			defer downloadCancel()

			actualInput, err = downloader.Download(downloadCtx, inputURL, tempDir)
			downloadSpinnerDone <- true

			if err != nil {
				fmt.Println()
				return fmt.Errorf("download failed: %w", err)
			}
			fmt.Println("\r[0/5] Downloading from YouTube... done")
		}
	}

	// Find scripts directory
	scriptsDir := findScriptsDir()

	// Create and run pipeline
	orch := pipeline.NewOrchestrator(scriptsDir, os.Stdout, verbose)

	cfg := pipeline.DefaultConfig()
	cfg.InputPath = actualInput
	cfg.InputURL = inputURL // For cache key generation
	cfg.TrackTitle = trackTitle
	cfg.CachedStemsDir = cachedStemsPath
	cfg.OutputPath = outputPath
	cfg.MIDIOutputPath = midiOutput
	cfg.Quantize = quantize
	cfg.SoundStyle = soundStyle
	cfg.Simplify = simplify
	cfg.LoopOnly = loopOnly
	cfg.EnableDrums = enableDrums
	cfg.DrumsOnly = drumsOnly
	cfg.DrumKit = drumKit
	cfg.Arrange = arrange
	cfg.ChordMode = chordMode
	cfg.BrazilianFunk = brazilianFunk
	cfg.GenreOverride = genreOverride
	cfg.UseCache = !noCache
	cfg.StemQuality = stemQuality

	result, err := orch.Execute(ctx, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return err
	}

	// Get version directory from orchestrator result (it already saved the output)
	var versionDir string
	if result.CacheKey != "" && result.OutputVersion > 0 && !noCache {
		stemCache, err := cache.NewStemCache()
		if err == nil {
			versionDir = stemCache.GetVersionDir(result.CacheKey, result.OutputVersion)
		}
	}

	// Output Strudel code (only to file if specified, skip console when report is generated)
	if outputPath != "" {
		if err := os.WriteFile(outputPath, []byte(result.StrudelCode), 0644); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
	} else if versionDir == "" {
		// Only print code to console when no report is being generated
		fmt.Println("\n" + result.StrudelCode)
	}

	// Show brief summary
	fmt.Printf("  %d notes, %.0f BPM, %s", result.NotesRetained, result.BPM, result.Key)
	if result.DrumHits > 0 {
		fmt.Printf(", %d drum hits", result.DrumHits)
	}
	fmt.Println()

	// Render audio if requested (default: auto, use "none" to disable)
	var renderedPath string
	if renderAudio != "" && renderAudio != "none" {
		fmt.Println("[6/7] Rendering audio preview...")

		// Determine output path - save to version directory if available
		audioPath := renderAudio
		if renderAudio == "auto" {
			if versionDir != "" {
				audioPath = filepath.Join(versionDir, "render.wav")
			} else if result.CacheDir != "" {
				audioPath = filepath.Join(result.CacheDir, fmt.Sprintf("render_v%03d.wav", result.OutputVersion))
			} else {
				audioPath = "output.wav"
			}
		}

		// AI Analysis: Analyze original audio to get optimal parameters
		var feedbackPath string
		if result.CacheDir != "" {
			melodicStem := filepath.Join(result.CacheDir, "melodic.wav")
			// Also check for legacy piano.wav
			if _, err := os.Stat(melodicStem); os.IsNotExist(err) {
				melodicStem = filepath.Join(result.CacheDir, "piano.wav")
			}
			if _, err := os.Stat(melodicStem); err == nil {
				// Save AI params in version directory
				aiParamsDir := versionDir
				if aiParamsDir == "" {
					aiParamsDir = result.CacheDir
				}
				feedbackPath = filepath.Join(aiParamsDir, "ai_params.json")
				fmt.Println("       Analyzing audio for AI-driven mix parameters...")
				if err := analyzeAudioForParams(melodicStem, feedbackPath, findScriptsDir()); err != nil {
					fmt.Printf("       Warning: AI analysis failed: %v\n", err)
					feedbackPath = ""
				} else {
					fmt.Println("       AI parameters extracted")
				}
			}
		}

		fmt.Printf("       Rendering to %s...\n", audioPath)

		// Get actual duration from the melodic stem for accurate render length
		duration := 0.0
		melodicForDuration := filepath.Join(result.CacheDir, "melodic.wav")
		if _, err := os.Stat(melodicForDuration); os.IsNotExist(err) {
			melodicForDuration = filepath.Join(result.CacheDir, "piano.wav")
		}
		if audioDur, err := getAudioDuration(melodicForDuration); err == nil {
			duration = audioDur
			fmt.Printf("       Source audio duration: %.1fs\n", duration)
		}

		if err := renderStrudelToWavWithFeedback(result.StrudelCode, audioPath, duration, feedbackPath); err != nil {
			fmt.Printf("       Warning: Audio render failed: %v\n", err)
		} else {
			fmt.Printf("       Render complete: %s\n", audioPath)
			renderedPath = audioPath
		}
	}

	// Compare rendered audio with original stems and generate chart (automatic when render exists)
	var comparisonChartPath string
	if renderedPath != "" && result.CacheDir != "" {
		fmt.Println("[+] Generating comparison chart...")

		// Get melodic stem path
		melodicPath := filepath.Join(result.CacheDir, "melodic.wav")
		if _, err := os.Stat(melodicPath); os.IsNotExist(err) {
			melodicPath = filepath.Join(result.CacheDir, "piano.wav")
		}

		if _, err := os.Stat(melodicPath); err == nil {
			// Generate comparison chart
			chartPath := filepath.Join(versionDir, "comparison.png")
			if versionDir == "" {
				chartPath = filepath.Join(result.CacheDir, "comparison.png")
			}
			if err := generateComparisonChart(melodicPath, renderedPath, chartPath, findScriptsDir()); err != nil {
				fmt.Printf("       Warning: Chart generation failed: %v\n", err)
			} else {
				fmt.Printf("       Comparison chart: %s\n", chartPath)
				comparisonChartPath = chartPath
			}
		}
	}

	// Generate HTML report
	var reportPath string
	if versionDir != "" || result.CacheDir != "" {
		reportDir := versionDir
		if reportDir == "" {
			reportDir = result.CacheDir
		}
		reportPath = filepath.Join(reportDir, "report.html")
		if err := generateHTMLReport(result.CacheDir, reportDir, reportPath, findScriptsDir()); err != nil {
			fmt.Printf("Warning: Report generation failed: %v\n", err)
			reportPath = ""
		}
	}

	// Final output - show report link as primary result
	fmt.Println("")
	fmt.Println("========================================")
	if reportPath != "" {
		fmt.Println("Done! Report generated:")
		fmt.Printf("  file://%s\n", reportPath)
		fmt.Println("")
		fmt.Println("Open the report to:")
		fmt.Println("  - Listen to original vs rendered audio")
		fmt.Println("  - View frequency comparison chart")
		fmt.Println("  - Copy Strudel code to clipboard")
	} else {
		fmt.Println("Done! Strudel code generated.")
		if outputPath != "" {
			fmt.Printf("Output: %s\n", outputPath)
		}
	}
	fmt.Println("========================================")

	_ = comparisonChartPath // silence unused warning

	return nil
}

// analyzeAudioForParams calls the AI analyzer to get optimal Strudel parameters
func analyzeAudioForParams(audioPath, outputJSON, scriptsDir string) error {
	analyzeScript := filepath.Join(scriptsDir, "audio_to_strudel_params.py")
	if _, err := os.Stat(analyzeScript); os.IsNotExist(err) {
		return fmt.Errorf("audio_to_strudel_params.py not found")
	}

	python := findPython(scriptsDir)
	cmd := exec.Command(python, analyzeScript, audioPath, "-o", outputJSON)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// renderStrudelToWavWithFeedback renders with AI-derived parameters
func renderStrudelToWavWithFeedback(code, outputPath string, duration float64, feedbackPath string) error {
	// Find scripts directory
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("find executable: %w", err)
	}
	exeDir := filepath.Dir(exePath)

	// Try different script locations
	scriptPaths := []string{
		filepath.Join(exeDir, "..", "scripts", "python", "render_audio.py"),
		filepath.Join(exeDir, "scripts", "python", "render_audio.py"),
		"scripts/python/render_audio.py",
	}

	var scriptPath string
	for _, p := range scriptPaths {
		if _, err := os.Stat(p); err == nil {
			scriptPath = p
			break
		}
	}

	if scriptPath == "" {
		return fmt.Errorf("render_audio.py not found")
	}

	// Write code to temp file
	tmpFile, err := os.CreateTemp("", "strudel-*.txt")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(code); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}
	tmpFile.Close()

	// Find Python
	pythonPaths := []string{
		filepath.Join(exeDir, "..", "scripts", "python", ".venv", "bin", "python3"),
		filepath.Join(exeDir, "scripts", "python", ".venv", "bin", "python3"),
		"scripts/python/.venv/bin/python3",
		"python3",
	}

	var pythonPath string
	for _, p := range pythonPaths {
		if _, err := os.Stat(p); err == nil {
			pythonPath = p
			break
		}
	}

	if pythonPath == "" {
		pythonPath = "python3"
	}

	// Run render script with feedback
	args := []string{scriptPath, tmpFile.Name(), "-o", outputPath}
	if duration > 0 {
		args = append(args, "-d", fmt.Sprintf("%.1f", duration))
	}
	if feedbackPath != "" {
		args = append(args, "--feedback", feedbackPath)
	}

	cmd := exec.Command(pythonPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}

// renderStrudelToWav calls the Python script to render Strudel code to WAV
func renderStrudelToWav(code, outputPath string, duration float64) error {
	// Find scripts directory
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("find executable: %w", err)
	}
	exeDir := filepath.Dir(exePath)

	// Try different script locations
	scriptPaths := []string{
		filepath.Join(exeDir, "..", "scripts", "python", "render_audio.py"),
		filepath.Join(exeDir, "scripts", "python", "render_audio.py"),
		"scripts/python/render_audio.py",
	}

	var scriptPath string
	for _, p := range scriptPaths {
		if _, err := os.Stat(p); err == nil {
			scriptPath = p
			break
		}
	}

	if scriptPath == "" {
		return fmt.Errorf("render_audio.py not found")
	}

	// Write code to temp file
	tmpFile, err := os.CreateTemp("", "strudel-*.txt")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(code); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}
	tmpFile.Close()

	// Find Python
	pythonPaths := []string{
		filepath.Join(exeDir, "..", "scripts", "python", ".venv", "bin", "python3"),
		filepath.Join(exeDir, "scripts", "python", ".venv", "bin", "python3"),
		"scripts/python/.venv/bin/python3",
		"python3",
	}

	var pythonPath string
	for _, p := range pythonPaths {
		if _, err := os.Stat(p); err == nil {
			pythonPath = p
			break
		}
	}

	if pythonPath == "" {
		pythonPath = "python3"
	}

	// Run render script
	args := []string{scriptPath, tmpFile.Name(), "-o", outputPath}
	if duration > 0 {
		args = append(args, "-d", fmt.Sprintf("%.1f", duration))
	}

	cmd := exec.Command(pythonPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}

// generateComparisonChart generates a frequency comparison chart
func generateComparisonChart(originalPath, renderedPath, outputPath, scriptsDir string) error {
	script := filepath.Join(scriptsDir, "compare_audio.py")
	if _, err := os.Stat(script); os.IsNotExist(err) {
		return fmt.Errorf("compare_audio.py not found")
	}

	python := findPython(scriptsDir)
	cmd := exec.Command(python, script, originalPath, renderedPath, "--chart", outputPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// generateHTMLReport generates an HTML report for the extraction
func generateHTMLReport(cacheDir, versionDir, outputPath, scriptsDir string) error {
	script := filepath.Join(scriptsDir, "generate_report.py")
	if _, err := os.Stat(script); os.IsNotExist(err) {
		return fmt.Errorf("generate_report.py not found")
	}

	python := findPython(scriptsDir)
	args := []string{script, cacheDir}
	if versionDir != "" && versionDir != cacheDir {
		args = append(args, "-v", filepath.Base(versionDir)[1:]) // strip 'v' prefix
	}
	if outputPath != "" {
		args = append(args, "-o", outputPath)
	}
	cmd := exec.Command(python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func runServe(cmd *cobra.Command, args []string) error {
	scriptsDir := findScriptsDir()

	cfg := server.Config{
		Port:       port,
		ScriptsDir: scriptsDir,
	}

	srv, err := server.New(cfg)
	if err != nil {
		return fmt.Errorf("create server: %w", err)
	}

	return srv.Run()
}

// findScriptsDir locates the Python scripts directory
func findScriptsDir() string {
	// Check relative to executable
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Join(filepath.Dir(exe), "scripts", "python")
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
	}

	// Check relative to working directory
	if dir := filepath.Join("scripts", "python"); dirExists(dir) {
		return dir
	}

	// Check common development locations
	candidates := []string{
		"./scripts/python",
		"../scripts/python",
		"../../scripts/python",
	}

	for _, c := range candidates {
		if dirExists(c) {
			return c
		}
	}

	return "scripts/python"
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// getAudioDuration gets the duration of an audio file using ffprobe
func getAudioDuration(audioPath string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries",
		"format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audioPath)
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe failed: %w", err)
	}

	duration := 0.0
	_, err = fmt.Sscanf(string(output), "%f", &duration)
	if err != nil {
		return 0, fmt.Errorf("parse duration: %w", err)
	}

	return duration, nil
}

func runTrainPrepare(cmd *cobra.Command, args []string) error {
	scriptsDir := findScriptsDir()
	prepareScript := filepath.Join(scriptsDir, "training", "prepare_dataset.py")

	// Build command
	cmdArgs := []string{prepareScript, "--output", datasetOut}

	if useMaestro {
		cmdArgs = append(cmdArgs, "--maestro")
	} else {
		if audioDir == "" || midiDir == "" {
			return fmt.Errorf("either --maestro or both --audio-dir and --midi-dir required")
		}
		cmdArgs = append(cmdArgs, "--audio-dir", audioDir, "--midi-dir", midiDir)
	}

	// Find Python
	python := findPython(scriptsDir)

	fmt.Println("Preparing training dataset...")
	fmt.Printf("  Python: %s\n", python)
	fmt.Printf("  Output: %s\n", datasetOut)

	return runPythonScript(python, cmdArgs)
}

func runTrainRun(cmd *cobra.Command, args []string) error {
	scriptsDir := findScriptsDir()
	trainScript := filepath.Join(scriptsDir, "training", "train.py")

	// Build command
	cmdArgs := []string{
		trainScript,
		"--dataset", datasetPath,
		"--output", modelOutput,
		"--epochs", fmt.Sprintf("%d", epochs),
		"--batch-size", fmt.Sprintf("%d", batchSize),
		"--learning-rate", fmt.Sprintf("%f", learningRate),
		"--base-model", baseModel,
	}

	// Find Python
	python := findPython(scriptsDir)

	fmt.Println("Starting model training...")
	fmt.Printf("  Dataset: %s\n", datasetPath)
	fmt.Printf("  Output: %s\n", modelOutput)
	fmt.Printf("  Epochs: %d\n", epochs)

	return runPythonScript(python, cmdArgs)
}

func runModelsList(cmd *cobra.Command, args []string) error {
	fmt.Println("Available models:")
	fmt.Println()
	fmt.Println("  Built-in:")
	fmt.Println("    basic-pitch    Stock model (default)")
	fmt.Println()
	fmt.Println("  Pre-trained (download with: midi-grep models pull <name>):")
	fmt.Println("    jazz           Jazz piano, swing timing")
	fmt.Println("    classical      MAESTRO-trained, dynamics")
	fmt.Println("    electronic     Synths, arps, bass")
	fmt.Println("    lofi           Chopped/sampled piano")
	fmt.Println()

	// Check for local models
	homeDir, _ := os.UserHomeDir()
	modelsDir := filepath.Join(homeDir, ".midi-grep", "models")

	if dirExists(modelsDir) {
		entries, err := os.ReadDir(modelsDir)
		if err == nil && len(entries) > 0 {
			fmt.Println("  Local models:")
			for _, e := range entries {
				if e.IsDir() {
					fmt.Printf("    %s\n", e.Name())
				}
			}
			fmt.Println()
		}
	}

	fmt.Println("Use --model flag with extract command:")
	fmt.Println("  midi-grep extract --model jazz --url '...'")

	return nil
}

func findPython(scriptsDir string) string {
	// Check venv first
	venvPython := filepath.Join(scriptsDir, ".venv", "bin", "python")
	if _, err := os.Stat(venvPython); err == nil {
		return venvPython
	}

	// Fallback to system python
	return "python3"
}

func runPythonScript(python string, args []string) error {
	cmd := exec.Command(python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// runAudioComparison runs the compare_audio.py script
func runAudioComparison(originalPath, renderedPath, scriptsDir string) error {
	compareScript := filepath.Join(scriptsDir, "compare_audio.py")
	if _, err := os.Stat(compareScript); os.IsNotExist(err) {
		return fmt.Errorf("compare_audio.py not found at %s", compareScript)
	}

	python := findPython(scriptsDir)

	cmd := exec.Command(python, compareScript, originalPath, renderedPath, "-d", "30")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}
