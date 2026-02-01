package progress

import (
	"fmt"
	"io"
	"time"
)

// Stage represents a processing stage
type Stage struct {
	Number      int
	Total       int
	Name        string
	Description string
}

// Predefined stages matching the spec
var (
	StageValidate   = Stage{1, 5, "validate", "Validating input file..."}
	StageSeparate   = Stage{2, 5, "separate", "Separating stems... (this may take a moment)"}
	StageAnalyze    = Stage{3, 5, "analyze", "Analyzing audio (BPM, key)..."}
	StageTranscribe = Stage{4, 5, "transcribe", "Transcribing piano to MIDI..."}
	StageGenerate   = Stage{5, 5, "generate", "Generating Strudel code..."}
)

// Reporter handles CLI progress output
type Reporter struct {
	out       io.Writer
	startTime time.Time
	verbose   bool
}

// NewReporter creates a new progress reporter
func NewReporter(out io.Writer, verbose bool) *Reporter {
	return &Reporter{
		out:       out,
		startTime: time.Now(),
		verbose:   verbose,
	}
}

// StartStage announces the beginning of a processing stage
func (r *Reporter) StartStage(stage Stage) {
	fmt.Fprintf(r.out, "[%d/%d] %s\n", stage.Number, stage.Total, stage.Description)
}

// Update shows a sub-progress message within a stage
func (r *Reporter) Update(format string, args ...any) {
	if r.verbose {
		fmt.Fprintf(r.out, "       %s\n", fmt.Sprintf(format, args...))
	}
}

// StageComplete shows completion message for a stage
func (r *Reporter) StageComplete(format string, args ...any) {
	fmt.Fprintf(r.out, "       %s\n", fmt.Sprintf(format, args...))
}

// Done announces successful completion
func (r *Reporter) Done(outputPath string) {
	elapsed := time.Since(r.startTime)
	fmt.Fprintln(r.out, "Done! Strudel code generated successfully.")
	if outputPath != "" {
		fmt.Fprintf(r.out, "Output saved to: %s\n", outputPath)
	}
	fmt.Fprintf(r.out, "Completed in %.1f seconds\n", elapsed.Seconds())
}

// Error announces an error
func (r *Reporter) Error(err error) {
	fmt.Fprintf(r.out, "Error: %s\n", err)
}

// Warning announces a non-fatal warning
func (r *Reporter) Warning(format string, args ...any) {
	fmt.Fprintf(r.out, "Warning: %s\n", fmt.Sprintf(format, args...))
}
