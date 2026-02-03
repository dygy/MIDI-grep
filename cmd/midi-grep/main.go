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
	renderAudio   string // Output path for rendered WAV

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
	extractCmd.Flags().StringVar(&renderAudio, "render", "", "Render output to WAV file (e.g., --render output.wav)")

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

	if inputURL != "" {
		if !audio.IsYouTubeURL(inputURL) {
			return fmt.Errorf("invalid YouTube URL: %s", inputURL)
		}

		// Check cache before downloading
		if !noCache {
			stemCache, err := cache.NewStemCache()
			if err == nil {
				cacheKey := cache.KeyForURL(inputURL)
				if cached, ok := stemCache.Get(cacheKey); ok {
					fmt.Printf("[0/5] Using cached stems (key: %s)\n", cacheKey[:8])
					cachedStemsPath = filepath.Dir(cached.PianoPath)
					if cachedStemsPath == "" && cached.DrumsPath != "" {
						cachedStemsPath = filepath.Dir(cached.DrumsPath)
					}
				}
			}
		}

		// Only download if not cached
		if cachedStemsPath == "" {
			fmt.Println("[0/5] Downloading from YouTube...")

			var err error
			tempDir, err = os.MkdirTemp("", "midi-grep-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tempDir)

			downloader := audio.NewYouTubeDownloader()
			downloadCtx, downloadCancel := context.WithTimeout(ctx, 5*time.Minute)
			defer downloadCancel()

			actualInput, err = downloader.Download(downloadCtx, inputURL, tempDir)
			if err != nil {
				return fmt.Errorf("download failed: %w", err)
			}
			fmt.Println("       Download complete")
		}
	}

	// Find scripts directory
	scriptsDir := findScriptsDir()

	// Create and run pipeline
	orch := pipeline.NewOrchestrator(scriptsDir, os.Stdout, verbose)

	cfg := pipeline.DefaultConfig()
	cfg.InputPath = actualInput
	cfg.InputURL = inputURL // For cache key generation
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
	cfg.UseCache = !noCache

	result, err := orch.Execute(ctx, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return err
	}

	// Output Strudel code
	if outputPath != "" {
		if err := os.WriteFile(outputPath, []byte(result.StrudelCode), 0644); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
		fmt.Printf("Output saved to: %s\n", outputPath)
	} else {
		fmt.Println("\n" + result.StrudelCode)
	}

	// Show summary
	fmt.Println("\nSummary:")
	fmt.Printf("  BPM: %.0f", result.BPM)
	if result.Key != "" {
		fmt.Printf(", Key: %s", result.Key)
	}
	if result.TimeSignature != "" && result.TimeSignature != "4/4" {
		fmt.Printf(", Time: %s", result.TimeSignature)
	}
	if result.SwingRatio > 1.1 {
		fmt.Printf(", Swing: %.2f", result.SwingRatio)
	}
	fmt.Println()
	if result.NotesRetained > 0 {
		fmt.Printf("  Notes: %d retained", result.NotesRetained)
		if result.NotesRemoved > 0 {
			fmt.Printf(", %d simplified", result.NotesRemoved)
		}
		fmt.Println()
	}
	if result.LoopDetected {
		fmt.Printf("  Loop: %d bar(s) detected (%.0f%% confidence)\n", result.LoopBars, result.LoopConfidence*100)
	}
	if result.DrumHits > 0 {
		fmt.Printf("  Drums: %d hits", result.DrumHits)
		if len(result.DrumTypes) > 0 {
			fmt.Printf(" (bd: %d, sd: %d, hh: %d)",
				result.DrumTypes["bd"], result.DrumTypes["sd"], result.DrumTypes["hh"])
		}
		fmt.Println()
	}

	// Render audio if requested
	if renderAudio != "" {
		fmt.Printf("\nRendering audio to %s...\n", renderAudio)
		// Use 16 bars by default, or calculate from original if available
		duration := 0.0 // 0 means use default (16 bars)
		if err := renderStrudelToWav(result.StrudelCode, renderAudio, duration); err != nil {
			fmt.Printf("Warning: Audio render failed: %v\n", err)
		} else {
			fmt.Printf("Audio rendered: %s\n", renderAudio)
		}
	}

	fmt.Println("\nDone! Strudel code generated successfully.")

	return nil
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
