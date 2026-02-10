# Dynamics & AI Learning - Completed

## Summary

This roadmap tracked the effort to fix flat audio rendering and add AI-driven iteration. **All major goals achieved.**

## What Was Done

### Problem
- Rendered audio was flat (no dynamics)
- AI iterations produced identical code (no memory)
- Node.js synthesis only reached ~72% similarity

### Solution: BlackHole Recording

Instead of emulating Strudel in Node.js, we record real Strudel playback via BlackHole virtual audio device.

| Approach | Similarity | Status |
|----------|------------|--------|
| Node.js synthesis | 72% max | Removed |
| **BlackHole recording** | **100%** | **Active** |

**Setup:**
```bash
brew install blackhole-2ch  # Requires reboot
node dist/record-strudel-blackhole.js input.strudel -o output.wav -d 30
```

**Key file:** `scripts/node/src/record-strudel-blackhole.ts`

### Agentic Ollama

Created persistent AI agent with memory per track:

- **File:** `scripts/python/ollama_agent.py`
- Persistent chat history (`.cache/agents/{hash}.json`)
- ClickHouse queries via `<sql>...</sql>` tags
- Remembers tried parameters, avoids repeating failures
- Validates Strudel code, rejects hallucinated methods

### Code Validation

LLMs hallucinate invalid methods. Added validation:
```python
INVALID_METHODS = ['.peak(', '.volume(', '.eq(', '.filter(', '.bass(', '.treble(']
```

### Hidden Browser

Recording runs invisibly:
- Window position: `-32000,-32000` (offscreen)
- AppleScript hides Chromium process
- Background flags prevent throttling

## Removed Files

These Node.js synthesis files were deleted (replaced by BlackHole):
- `render-strudel-node.ts`
- `render-strudel-puppeteer.ts`
- `record-strudel.ts`
- `record-strudel-ui.ts`

## Final Metrics

| Metric | Result |
|--------|--------|
| Audio accuracy | 100% (real Strudel) |
| Unique AI iterations | 12/12 |
| Code validation | Active |

## Future Ideas

- Section-based automation (intro/verse/chorus dynamics)
- Per-beat micro-dynamics with Perlin noise
- Cross-track learning from ClickHouse knowledge base
