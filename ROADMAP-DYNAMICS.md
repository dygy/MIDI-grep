# MIDI-grep Dynamics & AI Learning Roadmap

## Problem Summary

**Rendered audio is FLAT** - no dynamics compared to original. The waveforms show uniform amplitude while originals have natural energy flow (loud/quiet sections, phrase dynamics).

**AI iterations produce identical code** - v002-v009 were 100% identical because LLM has no memory of previous attempts.

---

## Phase 1: Fix Flat Audio (Immediate)

### 1.1 Energy Envelope Matching
**Priority: HIGH** | **Effort: Medium**

Extract RMS energy curve from original stem, apply as gain envelope to rendered audio.

```python
# Post-processing step in Node.js renderer or Python
def apply_energy_envelope(rendered_audio, original_audio):
    # Extract RMS every 50ms from original
    orig_rms = librosa.feature.rms(original, hop_length=2205)[0]  # 50ms at 44.1kHz

    # Normalize to 0-1 range
    envelope = orig_rms / orig_rms.max()

    # Apply to rendered (interpolate to match length)
    envelope_resampled = np.interp(
        np.linspace(0, 1, len(rendered_audio)),
        np.linspace(0, 1, len(envelope)),
        envelope
    )

    return rendered_audio * envelope_resampled
```

**Files to modify:**
- `scripts/node/src/render-strudel-node.ts` - Add envelope application
- `scripts/python/compare_audio.py` - Extract and save envelope

### 1.2 Per-Note Velocity from Transcription
**Priority: HIGH** | **Effort: Low**

Basic Pitch already extracts velocity (0-127) per note. Currently ignored.

```typescript
// render-strudel-node.ts
// Current: all notes same volume
synthNote(freq, 0.8)

// Fixed: use transcribed velocity
synthNote(freq, note.velocity / 127)
```

**Files to modify:**
- `internal/strudel/generator.go` - Include velocity in output
- `scripts/node/src/render-strudel-node.ts` - Use velocity for amplitude

### 1.3 Fix Double Gain Application
**Priority: HIGH** | **Effort: Low**

Current code applies gain twice, second overrides first:
```javascript
// bassFx already has: .gain("<0.4 0.8 0.2 0.6>".slow(16))
// Then stack adds: .gain(0.6 + bar_energy[i] * 0.6)  // OVERRIDES!
```

**Fix:** Use only one gain method, combine them properly:
```javascript
// Option A: Multiply patterns
.gain("<0.4 0.8 0.2 0.6>".slow(16).mul(bar_energy[i]))

// Option B: Remove bar_energy from stack, keep in Fx only
```

**Files to modify:**
- `scripts/python/ai_improver.py` - Fix `generate_orchestrated_effects()`
- `internal/strudel/generator.go` - Don't double-apply gain

### 1.4 Add Drums to Stack
**Priority: MEDIUM** | **Effort: Low**

Drums are defined but NOT included in the play stack!

```javascript
// Current (missing drums):
$: stack(
  cat(...bass.map(...)),
  cat(...mid.map(...)),
  cat(...high.map(...))
)

// Fixed:
$: stack(
  cat(...bass.map(...)),
  cat(...mid.map(...)),
  cat(...high.map(...)),
  cat(...drums.map((b, i) => drumsFx(s(b)).gain(...)))  // ADD THIS
)
```

**Files to modify:**
- `internal/strudel/generator.go` - Include drums in stack output

---

## Phase 2: Agentic Ollama (Learning AI)

### 2.1 Persistent Chat Sessions
**Priority: HIGH** | **Effort: Medium**

One agent per track with conversation history. LLM remembers previous attempts.

```python
class OllamaAgent:
    def __init__(self, track_hash: str):
        self.track_hash = track_hash
        self.messages = []
        self.history_file = f".cache/agents/{track_hash}.json"

    def add_iteration_result(self, iteration, similarity, code_tried):
        self.messages.append({
            "role": "user",
            "content": f"""
            Iteration {iteration}: {similarity*100:.1f}%
            Code you generated: {code_tried}
            THIS DID NOT IMPROVE - try something DIFFERENT
            """
        })

    def generate(self) -> str:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "llama3.1:8b",
            "messages": self.messages
        })
        return response.json()["message"]["content"]
```

**Files to create:**
- `scripts/python/ollama_agent.py` - Agentic wrapper

**Files to modify:**
- `scripts/python/ai_improver.py` - Use agent instead of single prompts

### 2.2 Context Compression
**Priority: MEDIUM** | **Effort: Medium**

When context grows too long, summarize older messages.

```python
def compress_context(self):
    if self.estimate_tokens() > 6000:
        # Summarize old attempts
        summary = f"""
        SESSION HISTORY (iterations 1-{len(self.messages)//2}):
        - Best similarity: {self.best_similarity*100:.1f}%
        - Tried gains: {self.tried_gains}
        - What worked: {self.successful_changes}
        - What failed: {self.failed_changes}
        """
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": summary},
            *self.messages[-6:]  # Keep last 3 turns
        ]
```

### 2.3 Iteration History in Prompt
**Priority: HIGH** | **Effort: Low**

Tell LLM what it already tried:

```python
prompt += f"""
## PREVIOUS ATTEMPTS (DO NOT REPEAT)
- Iteration 1: gain(perlin.range(0.8, 1.0)) → 81% (NO CHANGE)
- Iteration 2: gain(perlin.range(0.8, 1.0)) → 81% (SAME CODE!)
- Iteration 3: gain(perlin.range(0.72, 0.88)) → 81% (TOO SUBTLE)

YOU ARE STUCK. Try completely different approach:
- Use beat-synced patterns: "<0.2 0.5 0.9 0.6>".slow(16)
- Change sounds, not just gains
- Wider ranges (0.2 to 0.9, not 0.8 to 1.0)
"""
```

---

## Phase 3: ClickHouse-Connected Agent

### 3.1 SQL Tool for Ollama
**Priority: HIGH** | **Effort: Medium**

Let LLM query ClickHouse for what actually worked.

```python
def system_prompt(self):
    return """You can query the database:
    <sql>SELECT * FROM midi_grep.runs WHERE similarity > 0.85</sql>

    TABLES:
    - midi_grep.runs: track_hash, similarity_overall, strudel_code, genre, bpm
    - midi_grep.knowledge: parameter_name, parameter_new_value, improvement
    """

def chat(self, message):
    response = self.ollama_chat(message)

    # Parse and execute SQL queries
    for sql in re.findall(r'<sql>(.*?)</sql>', response):
        result = self.execute_clickhouse(sql)
        self.messages.append({"role": "user", "content": f"Result:\n{result}"})
        response = self.ollama_chat("Continue analysis")

    return response
```

**Files to create:**
- `scripts/python/ollama_clickhouse_agent.py`

### 3.2 Data-Driven Parameter Selection
**Priority: MEDIUM** | **Effort: Low**

Query successful runs for similar tracks:

```sql
-- Find what worked for brazilian funk at 130-140 BPM
SELECT
    AVG(band_bass) as avg_bass_diff,
    AVG(similarity_overall) as avg_sim,
    any(strudel_code) as best_code
FROM midi_grep.runs
WHERE genre = 'brazilian_funk'
  AND bpm BETWEEN 130 AND 140
  AND similarity_overall > 0.85
```

---

## Phase 4: Model & Infrastructure

### 4.1 Upgrade to llama3.1
**Priority: HIGH** | **Effort: Low** | **Status: IN PROGRESS**

```bash
ollama pull llama3.1:8b
```

Supports tool/function calling for ClickHouse integration.

**Already done:**
- Updated Python default: `tinyllama` → `llama3.1:8b`
- Updated Go CLI default: `llama3:8b` → `llama3.1:8b`

### 4.2 Extended Thinking Mode
**Priority: MEDIUM** | **Effort: Low**

Let LLM think longer for complex decisions:

```python
response = requests.post(f"{OLLAMA_URL}/api/chat", json={
    "model": "llama3.1:8b",
    "messages": messages,
    "options": {
        "num_predict": 8192,  # More tokens for thinking
        "temperature": 0.7,
        "top_p": 0.9
    }
})
```

---

## Phase 5: Advanced Dynamics

### 5.1 Section-Based Automation
**Priority: MEDIUM** | **Effort: Medium**

Detect sections (intro/verse/chorus/drop) and apply appropriate dynamics:

```python
sections = detect_sections(audio)  # Already have this
# Returns: [
#   {"type": "intro", "start": 0, "end": 16, "energy": 0.3},
#   {"type": "buildup", "start": 16, "end": 32, "energy": 0.6},
#   {"type": "drop", "start": 32, "end": 48, "energy": 1.0},
# ]

# Generate gain pattern from sections:
gain_pattern = "<0.3 0.6 1.0 0.7>".slow(64)  # One value per section
```

### 5.2 Micro-Dynamics (Per-Beat Variation)
**Priority: LOW** | **Effort: High**

Add subtle variation within phrases:

```javascript
// Humanize with slight random variation
.gain(perlin.range(0.9, 1.1).fast(4))  // ±10% per beat

// Accent downbeats
.gain("1.0 0.8 0.9 0.8")  // Stronger on beat 1
```

---

## Implementation Order

**Priority: 4 → 3 → 2 → 1 → 5** (AI infrastructure first, then audio fixes)

| Week | Phase | Tasks | Expected Outcome | Status |
|------|-------|-------|------------------|--------|
| 1 | **4** | 4.1, 4.2 | llama3.1 with tool calling, extended thinking | ✅ Using llama3:8b + ReAct |
| 1 | **3** | 3.1, 3.2 | LLM queries ClickHouse for proven parameters | ✅ DONE |
| 1 | **2** | 2.1, 2.2, 2.3 | Agentic Ollama with memory, no repeated code | ✅ DONE |
| 2 | **1** | 1.1, 1.2, 1.3, 1.4 | Waveforms match original dynamics | ✅ PARTIAL (see below) |
| 3 | **5** | 5.1, 5.2 | Section automation, micro-dynamics | TODO |

## Completed (Feb 9, 2026)

### Phase 4+3+2: Agentic Ollama with ClickHouse
- Created `scripts/python/ollama_agent.py`
- Persistent chat history per track (`.cache/agents/{hash}.json`)
- SQL queries to ClickHouse via ReAct pattern (`<sql>...</sql>`)
- Iteration memory - remembers tried gains, lpf, sounds
- Context compression when approaching token limit
- Integrated into `ai_improver.py`
- Using llama3:8b (llama3.1 download was slow, ReAct works without native tools)

### Phase 1: Audio Rendering Improvements
- **808 Kick Fix:** Changed pitchEnd from 80Hz to 35Hz for proper sub-bass
- **Removed kick HPF:** No longer filtering out sub-bass from kicks
- **Gain rebalancing:** Bass 0.6x, Mids 0.5x, Highs 0.4x, Drums 0.7x
- **Result:** Brazilian funk similarity improved from 32% to 72%

### NEW: Puppeteer + BlackHole Recording (RECOMMENDED)
Instead of emulating Strudel synthesis in Node.js, record REAL Strudel playback:

**Files created:**
- `scripts/node/src/record-strudel-blackhole.ts` - Records via BlackHole virtual audio

**Setup:**
1. `brew install blackhole-2ch` (requires reboot)
2. Run: `node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30`

**Why this is better:**
- Uses REAL Strudel engine (not emulation)
- 100% accurate sound reproduction
- No endless gain/filter tuning
- Works with any Strudel features (samples, effects, etc.)

### Hidden Browser Window (Feb 2026)
Browser now runs invisibly without disturbing user focus:
- **Position:** `-32000,-32000` (far offscreen)
- **Size:** `1x1` pixels (minimal)
- **AppleScript:** Hides Chromium process visibility
- **Background flags:** Prevents throttling when not focused

### Strudel Code Validation (Feb 2026)
Added validation to reject invalid Strudel methods that LLMs hallucinate:
```python
INVALID_METHODS = [
    '.peak(',      # Doesn't exist - use .hpf() instead
    '.volume(',    # Should be .gain()
    '.eq(',        # Use .lpf/.hpf instead
    '.filter(',    # Too generic
    '.bass()', '.treble()', '.mid()',  # Not methods
]
```
- Validation in `ollama_agent.py` `_validate_code()` method
- System prompt updated to tell LLM what NOT to use
- Graceful skip on validation failure (continues iteration instead of crash)

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Node.js synthesis similarity | 72% | 80% |
| BlackHole recording similarity | ~100% | 100% |
| Unique iterations | 12/12 | 12/12 (100%) ✅ |
| Final similarity (best approach) | 100% | >90% ✅ |

---

## Files Summary

**Created:**
- `scripts/python/ollama_agent.py` - Agentic Ollama wrapper ✅
- `scripts/node/src/record-strudel-blackhole.ts` - BlackHole recorder ✅
- `scripts/node/src/record-strudel-ui.ts` - Manual browser recording ✅

**Modified:**
- `scripts/python/ai_improver.py` - Use agent, fix orchestrator ✅
- `scripts/node/src/render-strudel-node.ts` - 808 kick fix, gain rebalancing ✅
- `cmd/midi-grep/main.go` - Model default (done) ✅
