---
description: Orient a Claude Code session in MIDI-grep — checks git, cache, venv, Node build, BlackHole and Ollama state in one call and routes you to the right flow (feature / bug / similarity run / docs-only). Skip it when you already know what to work on.
---

# Session Initialization

You are in **MIDI-grep** — a Go CLI (`cmd/`, `internal/`) that orchestrates Python ML (`scripts/python/`, venv at `scripts/python/.venv`) and a Node/Puppeteer Strudel recorder (`scripts/node/`) to turn audio into editable Strudel code, scored by `compare_audio.py` against `eval/thresholds.yaml`. Go never generates Strudel; Python LLM codegen does. Specs live in `context/spec/NNN-<slug>/`.

**Keep this cheap.** One shell call, no MCP probe calls, no doc reads, at most two questions, no subagents. If the user's message already says what to work on — an issue number, a slash command, a file, a concrete task — **skip Steps 2–3**: give the one-line status and start.

## Step 1: State (one Bash call)

```bash
git branch --show-current; git status --short | head -5; echo "stems: $(ls .cache/stems 2>/dev/null | wc -l | tr -d ' ')"; test -x scripts/python/.venv/bin/python && echo "venv ok" || echo "venv MISSING"; test -d scripts/node/node_modules && echo "node_modules ok" || echo "node_modules MISSING (cd scripts/node && npm install && npm run build)"; echo "blackhole: $(system_profiler SPAudioDataType 2>/dev/null | grep -c BlackHole)"; echo "ollama: $(curl -s -m 2 localhost:11434/api/tags | head -c 80)"; gh auth status 2>&1 | grep -m1 -E 'Logged in|not logged'
```

Do not probe MCP servers (`loop`, `playwright`) with test calls — connection failures are reported by the harness; say so only when a later step needs one.

One status line, e.g. **Branch:** `main` (clean) · **stems** 12 · venv ok · node_modules ok · BlackHole 1 · Ollama up (`midi-grep-strudel-mistral`) · gh ✓. Flag the three that bite long runs: `venv MISSING` → `make install-python-deps`; `node_modules MISSING` → the command shown; `blackhole: 0` → no render verification on this machine (macOS + `brew install blackhole-2ch` + reboot + a Multi-Output Device as system output). `.cache/stems` empty is fine — the first extraction fills it (~1 GB of models on first run).

## Step 2: Ask the bucket (only if intent is unknown)

`AskUserQuestion`, four options: **Build a feature** · **Fix a bug** · **Similarity run** (improve a track's score, multi-iteration) · **Docs / spec only**. One drill-down question at most:

- Build a feature → **Continue current branch** (only when not on `main`) / **New feature** (issue #, file, or text)
- Fix a bug → **GitHub issue** / **Describe it** / **A failing render** (`comparison.json` path)
- Similarity run → which track (URL or `.cache/stems/<key>`) and genre

## Step 3: Route

- **New feature** → `/implement-feature <issue# | path | text>`. **Fix a bug** → `/fix-bug <issue# | description | comparison.json>`. Both branch from `origin/main` (`feat/<slug>`, `fix/<slug>`), run the local gate (`go build/vet`, pytest, `tsc`), `/code-review` at medium effort, and open a PR on `dygy/MIDI-grep` for the owner to merge; any Strudel-affecting change renders through BlackHole via the `loop` MCP.
- **Similarity run** → `sisyphus-plan` skill for the plan + evidence trail (`.sisyphus/plans/NNN-<slug>.md`, evidence with genre + % + render path), driven by the **`loop` MCP** (`verify_strudel` with `recorder='blackhole'`) or `scripts/auto-calibrate.sh`. One render at a time; the Node recorder never gates; numbers come only from a real `comparison.json`. Preflight Ollama and BlackHole before starting (Step 1 already did).
- **Docs / spec only** → `/awos:spec` → `/awos:tech` → `/awos:tasks` in `context/spec/NNN-<slug>/` (next index after the highest existing), on a `docs/<slug>` branch; `/awos:verify` closes a spec whose tasks are all `[x]`. Update `llms.txt` / `llms-full.txt` / `CLAUDE.md` per "Context Document Maintenance" when the pipeline changed.
- **Continue current branch** → match the branch slug to its `context/spec/*-<slug>/` dir, read its `flow-log.md` (or the scratchpad `fix-log-*.md`) if present, and name the next step; with an open PR, `gh pr view` is the resume state.

End with one line: where the work lands and which command, skill or domain expert (`.claude/agents/`) applies. Specialists: `golang-expert` (Go), `python-expert`, `ml-audio-expert` (librosa/demucs/compare/eval), `audio-dsp-expert`, `strudel-expert`, `llm-expert` (ollama/prompts), `music-theory-expert`, `testing-expert`.
