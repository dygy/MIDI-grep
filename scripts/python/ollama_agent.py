#!/usr/bin/env python3
"""
Agentic Ollama with ClickHouse Integration

One agent per track with:
- Persistent conversation history
- SQL queries to ClickHouse for data-driven decisions
- Context compression when approaching limits
- Memory of what was tried and what failed
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ClickHouse connection
CLICKHOUSE_BIN = Path(__file__).parent.parent.parent / "bin" / "clickhouse"
CLICKHOUSE_DB = Path(__file__).parent.parent.parent / ".clickhouse" / "db"
AGENTS_DIR = Path(__file__).parent.parent.parent / ".cache" / "agents"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")


class OllamaAgent:
    """
    Agentic Ollama wrapper with:
    - Persistent chat history per track
    - ClickHouse SQL tool via ReAct pattern
    - Iteration memory (what was tried, what failed)
    - Context compression
    """

    def __init__(self, track_hash: str, model: str = None):
        self.track_hash = track_hash
        self.model = model or DEFAULT_MODEL
        self.messages: List[Dict] = []
        self.iteration_history: List[Dict] = []
        self.best_similarity = 0.0
        self.best_code = ""
        self.tried_values: Dict[str, List[str]] = {}  # param -> [values tried]
        self.max_context_tokens = 6000  # Conservative limit
        self.last_validation_error: Optional[str] = None  # Track validation failures

        # Ensure agents directory exists
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.history_file = AGENTS_DIR / f"{track_hash}.json"

        # Load existing history if available
        self.load_history()

        # Initialize with system prompt if new session
        if not self.messages:
            self.messages = [{"role": "system", "content": self._system_prompt()}]

    def _system_prompt(self) -> str:
        return f"""You are an expert Strudel live coding AI with access to a ClickHouse database of previous runs.

## THIS TRACK
Track hash: {self.track_hash}
Use this track_hash in ALL your SQL queries to get data specific to THIS track.

## YOUR GOAL
Generate Strudel effect functions that make rendered audio match the original.
Each iteration, you receive feedback on what worked and what didn't.

## DATABASE ACCESS (ReAct Pattern)

To query the database, output SQL in this exact format:
<sql>YOUR SQL QUERY HERE</sql>

I will execute it and show you results. Then continue your analysis.

### Available Tables

**midi_grep.runs** - All previous rendering attempts
```
track_hash String, version UInt32, similarity_overall Float64,
similarity_mfcc Float64, similarity_chroma Float64,
strudel_code String, genre String, bpm Float64,
band_bass Float64, band_mid Float64, band_high Float64
```

**midi_grep.knowledge** - Proven parameter improvements per track
```
track_hash String, parameter_name String, parameter_old_value String, parameter_new_value String,
similarity_improvement Float64, genre String, bpm_range_low Float64, bpm_range_high Float64
```

### Example Queries

Find the BEST previous run for THIS track:
<sql>SELECT strudel_code, similarity_overall, version FROM midi_grep.runs
WHERE track_hash = '{track_hash}'
ORDER BY similarity_overall DESC LIMIT 1</sql>

Find what parameter changes improved THIS track:
<sql>SELECT parameter_name, parameter_old_value, parameter_new_value, similarity_improvement
FROM midi_grep.knowledge WHERE track_hash = '{track_hash}'
ORDER BY similarity_improvement DESC LIMIT 5</sql>

Compare iterations for THIS track:
<sql>SELECT version, similarity_overall, band_bass, band_mid, band_high
FROM midi_grep.runs WHERE track_hash = '{track_hash}'
ORDER BY version DESC LIMIT 10</sql>

## CRITICAL RULES

1. **NEVER REPEAT FAILED VALUES** - I will tell you what you already tried
2. **WIDE GAIN RANGES** - Use 0.2 to 0.8, not 0.5 to 0.6
3. **BEAT-SYNCED PATTERNS** - Use `"<v1 v2 v3 v4>".slow(16)` for dynamics
4. **NO PERLIN FOR GAIN** - perlin ranges are too subtle
5. **ONLY USE VALID STRUDEL METHODS** - These methods DO NOT exist and will be rejected:
   - `.peak()` - use `.hpf()` for high-pass filter instead
   - `.volume()` - use `.gain()` instead
   - `.eq()`, `.filter()` - use `.lpf()` and `.hpf()` instead
   - `.bass()`, `.treble()`, `.mid()`, `.high()`, `.low()` - not methods
   - `.compress()` - use `.compressor()` if available, or skip

## OUTPUT FORMAT

After analysis, output improved effect functions in a code block:
```javascript
let bassFx = p => p.sound("sawtooth")
    .gain("<0.2 0.4 0.7 0.5>".slow(16))
    .lpf(600)
```

Think step by step. Query the database first to see what worked for similar tracks."""

    def load_history(self):
        """Load conversation history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.messages = data.get("messages", [])
                self.iteration_history = data.get("iteration_history", [])
                self.best_similarity = data.get("best_similarity", 0.0)
                self.best_code = data.get("best_code", "")
                self.tried_values = data.get("tried_values", {})
                print(f"  [Agent] Loaded history: {len(self.messages)} messages, best={self.best_similarity*100:.1f}%")
            except Exception as e:
                print(f"  [Agent] Could not load history: {e}")

    def save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "iteration_history": self.iteration_history,
                    "best_similarity": self.best_similarity,
                    "best_code": self.best_code,
                    "tried_values": self.tried_values,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"  [Agent] Could not save history: {e}")

    def estimate_tokens(self) -> int:
        """Rough token count estimation (4 chars per token)."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        return total_chars // 4

    def compress_context(self):
        """Compress old messages when context too long."""
        if self.estimate_tokens() < self.max_context_tokens * 0.8:
            return

        print(f"  [Agent] Compressing context ({self.estimate_tokens()} tokens)...")

        # Summarize iteration history
        history_summary = self._summarize_history()

        # Keep system prompt + summary + last 4 messages
        self.messages = [
            self.messages[0],  # System prompt
            {"role": "user", "content": f"## SESSION HISTORY\n{history_summary}"},
            {"role": "assistant", "content": "I understand the history. I will try different approaches."},
            *self.messages[-4:]  # Last 2 turns
        ]

        print(f"  [Agent] Compressed to {self.estimate_tokens()} tokens")

    def _summarize_history(self) -> str:
        """Create a summary of what was tried."""
        lines = [
            f"Track: {self.track_hash}",
            f"Best similarity achieved: {self.best_similarity*100:.1f}%",
            f"Total iterations: {len(self.iteration_history)}",
            "",
            "## What Was Tried (DO NOT REPEAT):"
        ]

        for param, values in self.tried_values.items():
            lines.append(f"- {param}: {', '.join(values[:5])}{'...' if len(values) > 5 else ''}")

        lines.append("")
        lines.append("## Recent Iterations:")
        for h in self.iteration_history[-5:]:
            status = "✓" if h.get("improved") else "✗"
            lines.append(f"- {status} v{h.get('version', '?')}: {h.get('similarity', 0)*100:.1f}% - {h.get('notes', '')}")

        return "\n".join(lines)

    def execute_sql(self, sql: str) -> str:
        """Execute SQL query on ClickHouse."""
        sql = sql.strip()
        if not sql:
            return "ERROR: Empty query"

        # Security: only allow SELECT
        if not sql.upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries allowed"

        cmd = [
            str(CLICKHOUSE_BIN), "local",
            "--path", str(CLICKHOUSE_DB),
            "--query", sql,
            "--format", "PrettyCompact"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return f"ERROR: {result.stderr[:500]}"
            output = result.stdout.strip()
            if not output:
                return "(no results)"
            # Limit output size
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Query timeout"
        except Exception as e:
            return f"ERROR: {str(e)[:200]}"

    def add_iteration_result(
        self,
        iteration: int,
        version: int,
        similarity: float,
        band_diffs: Dict[str, float],
        code_generated: str,
        improved: bool
    ):
        """Add result of an iteration for learning."""
        # Track what was tried
        self._extract_tried_values(code_generated)

        # Update best
        if similarity > self.best_similarity:
            self.best_similarity = similarity
            self.best_code = code_generated

        # Add to history
        notes = []
        if band_diffs.get("bass", 0) > 0.1:
            notes.append("bass too loud")
        elif band_diffs.get("bass", 0) < -0.1:
            notes.append("bass too quiet")
        if band_diffs.get("mid", 0) > 0.1:
            notes.append("mid too loud")
        elif band_diffs.get("mid", 0) < -0.1:
            notes.append("mid too quiet")

        self.iteration_history.append({
            "iteration": iteration,
            "version": version,
            "similarity": similarity,
            "band_diffs": band_diffs,
            "improved": improved,
            "notes": ", ".join(notes) if notes else "balanced",
            "timestamp": datetime.now().isoformat()
        })

        # Add feedback message
        status = "IMPROVEMENT!" if improved else "NO IMPROVEMENT"
        feedback = f"""## Iteration {iteration} Result: {status}

Similarity: {similarity*100:.1f}% (best: {self.best_similarity*100:.1f}%)
Bass: {band_diffs.get('bass', 0)*100:+.1f}%
Mid: {band_diffs.get('mid', 0)*100:+.1f}%
High: {band_diffs.get('high', 0)*100:+.1f}%

{'The code you generated worked! Build on this.' if improved else 'Try something DIFFERENT. Do not repeat these values.'}

Your previous code:
```javascript
{code_generated[:1000]}{'...' if len(code_generated) > 1000 else ''}
```

Generate improved effect functions. Query the database if needed."""

        self.messages.append({"role": "user", "content": feedback})

        # Compress if needed
        self.compress_context()

        # Save
        self.save_history()

    def _extract_tried_values(self, code: str):
        """Extract parameter values from code to track what was tried."""
        # Extract gain values
        gain_matches = re.findall(r'\.gain\(([^)]+)\)', code)
        for g in gain_matches:
            self.tried_values.setdefault("gain", []).append(g[:50])

        # Extract lpf values
        lpf_matches = re.findall(r'\.lpf\((\d+)\)', code)
        for l in lpf_matches:
            self.tried_values.setdefault("lpf", []).append(l)

        # Extract sounds
        sound_matches = re.findall(r'\.sound\("([^"]+)"\)', code)
        for s in sound_matches:
            self.tried_values.setdefault("sound", []).append(s)

        # Keep only last 20 values per param
        for param in self.tried_values:
            self.tried_values[param] = self.tried_values[param][-20:]

    def generate(self, context: Dict = None) -> str:
        """
        Generate improved Strudel code using agentic loop.

        The agent can execute multiple SQL queries before generating code.
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests package not installed")

        # Add context if provided
        if context:
            context_msg = f"""## Current Track Context

Genre: {context.get('genre', 'unknown')}
BPM: {context.get('bpm', 120)}
Current similarity: {context.get('similarity', 0)*100:.1f}%

Frequency issues:
- Bass: {context.get('band_bass', 0)*100:+.1f}%
- Mid: {context.get('band_mid', 0)*100:+.1f}%
- High: {context.get('band_high', 0)*100:+.1f}%

Query the database using track_hash='{self.track_hash}' to find what worked for THIS track, then generate improved code."""

            self.messages.append({"role": "user", "content": context_msg})

        # Agentic loop - allow multiple SQL queries
        max_turns = 5
        for turn in range(max_turns):
            # Call Ollama
            response = self._call_ollama()

            if not response:
                print(f"  [Agent] No response from Ollama")
                break

            # Check for SQL queries
            sql_matches = re.findall(r'<sql>(.*?)</sql>', response, re.DOTALL | re.IGNORECASE)

            if sql_matches:
                # Execute queries and continue
                results = []
                for sql in sql_matches:
                    print(f"  [Agent] Executing SQL: {sql[:80]}...")
                    result = self.execute_sql(sql)
                    results.append(f"Query:\n{sql}\n\nResult:\n{result}")

                # Add results and continue
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": "SQL Results:\n\n" + "\n\n---\n\n".join(results) + "\n\nNow generate the improved code based on these results."
                })
            else:
                # No SQL, this is the final response
                self.messages.append({"role": "assistant", "content": response})
                self.save_history()
                return response

        self.save_history()
        return self.messages[-1].get("content", "") if self.messages else ""

    def _call_ollama(self) -> str:
        """Make a single call to Ollama."""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4096,
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            print(f"  [Agent] Ollama not running at {OLLAMA_URL}")
            return ""
        except Exception as e:
            print(f"  [Agent] Ollama error: {e}")
            return ""

    def extract_code(self, response: str) -> str:
        """Extract Strudel code from agent response."""
        self.last_validation_error = None  # Reset validation error

        # Look for code blocks
        code_match = re.search(r'```(?:javascript|js)?\n?([\s\S]*?)```', response)
        if code_match:
            code = code_match.group(1).strip()
            return self._validate_code(code)

        # Look for effect function patterns
        fx_pattern = r'let\s+\w+Fx\s*=\s*p\s*=>\s*p[^\n]*(?:\n\s+\.[^\n]*)*'
        fx_matches = re.findall(fx_pattern, response)
        if fx_matches:
            code = '\n\n'.join(fx_matches)
            return self._validate_code(code)

        return ""

    def _validate_code(self, code: str) -> str:
        """
        Validate Strudel code and reject if it contains invalid methods.

        Returns empty string if code is invalid, otherwise returns the code.
        """
        # Known invalid methods that LLMs sometimes hallucinate
        INVALID_METHODS = [
            '.peak(',      # Doesn't exist - maybe confused with .hpf or EQ peak
            '.eq(',        # Not a Strudel method (use .lpf/.hpf)
            '.volume(',    # Should be .gain()
            '.filter(',    # Too generic, use specific filters
            '.bass(',      # Not a method
            '.treble(',    # Not a method
            '.mid(',       # Not a method
            '.high(',      # Not a method (voice selector, not effect)
            '.low(',       # Not a method (voice selector, not effect)
            '.boost(',     # Not a method
            '.cut(',       # Not a method (use .lpf/.hpf)
            '.compress(',  # Not a method (use .compressor)
            '.limit(',     # Not a method
            '.normalize(', # Not a method
        ]

        for invalid in INVALID_METHODS:
            if invalid in code:
                msg = f"REJECTED: Code contains invalid method {invalid} - LLM hallucinated a non-existent Strudel method"
                print(f"  [Agent] {msg}")
                self.last_validation_error = msg
                return ""

        # Additional check: ensure at least one valid effect pattern
        valid_patterns = ['.sound(', '.gain(', '.lpf(', '.hpf(', '.room(', '.delay(', '.bank(']
        has_valid = any(p in code for p in valid_patterns)

        if not has_valid:
            msg = "REJECTED: Code has no recognizable effect methods"
            print(f"  [Agent] {msg}")
            self.last_validation_error = msg
            return ""

        return code

    def reset(self):
        """Reset agent state for fresh start."""
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self.iteration_history = []
        self.tried_values = {}
        # Keep best_similarity and best_code
        self.save_history()


def test_agent():
    """Quick test of agent functionality."""
    print("Testing OllamaAgent...")

    agent = OllamaAgent("test_track_123")

    # Test SQL execution
    print("\n1. Testing SQL execution:")
    result = agent.execute_sql("SELECT count() FROM midi_grep.runs")
    print(f"   Result: {result}")

    # Test generation with context
    print("\n2. Testing code generation:")
    response = agent.generate({
        "genre": "brazilian_funk",
        "bpm": 136,
        "similarity": 0.75,
        "band_bass": 0.15,
        "band_mid": -0.10,
        "band_high": 0.05
    })
    print(f"   Response length: {len(response)} chars")

    # Extract code
    code = agent.extract_code(response)
    print(f"   Extracted code length: {len(code)} chars")
    if code:
        print(f"   First 200 chars:\n{code[:200]}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_agent()
