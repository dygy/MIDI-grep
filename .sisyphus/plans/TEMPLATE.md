# <Feature / Run Name>

> Copy this file to `NNN-short-slug.md` (zero-padded, next number) and fill it in.
> Plan format for MIDI-grep's audio pipeline.

## TL;DR

> **Quick Summary**: One paragraph — what this run/feature delivers and how.
>
> **Deliverables**:
> - `path/to/file.go` — what it does
> - `scripts/python/foo.py` — what it does
>
> **Estimated Effort**: Small | Medium | Large
> **Parallel Execution**: YES/NO — N waves
> **Critical Path**: Task 2 → Task 5 → Task 8 → Final

---

## Context

### Original Request
What the user actually asked for (link the spec under `context/spec/` if one exists).

### Key Decisions
- Decision → rationale (e.g. "use chord mode — track is electronic, not piano")
- Genre / BPM / key assumptions and where they came from
- Which renderer (BlackHole vs Node.js) and why

### Research Findings
- File-path facts discovered while reading the codebase (so subagents don't rediscover them)
- e.g. "Strudel codegen lives in `scripts/python/ollama_codegen.py`, NOT Go"

---

## Work Objectives

### Core Objective
One sentence: the outcome that makes this run a success.

### Concrete Deliverables
- Bullet list of exact files created/changed.

### Definition of Done
- [ ] `go build -o bin/midi-grep ./cmd/midi-grep` → builds
- [ ] `go test ./...` → PASS
- [ ] `cd scripts/python && .venv/bin/python -m pytest tests/ -v` → PASS
- [ ] Eval gate passes: `pytest scripts/python/tests/test_similarity_gate.py` (see `eval/thresholds.yaml`)
- [ ] Similarity on the target track meets/exceeds the gate for its genre

### Must Have
- The non-negotiable behaviors.

### Must NOT Have (Guardrails)
- **MUST NOT** hardcode gains/filters/effects — everything from analysis or AI (see CLAUDE.md ZERO HARDCODING)
- **MUST NOT** clear ClickHouse `runs`/`knowledge` or agent history — that's learning data
- **MUST NOT** trust Node.js-renderer similarity for accept/reject decisions — use BlackHole for the gate
- **MUST NOT** commit `.cache/` artifacts or large render WAVs

---

## Verification Strategy

> ALL verification is agent-executed — no human in the loop.

- **Go / Python units**: `go test ./...`, `pytest`
- **Audio quality**: render via BlackHole, then `compare_audio.py` → similarity %
- **Eval gate**: assert per-genre similarity against `eval/thresholds.yaml`
- Evidence saved to `.sisyphus/evidence/task-{N}-{slug}.{txt,json,png}` — include the similarity number and the render path.

---

## Execution Strategy

### Manual Prerequisites (pre-wave)
- Env preflight (see CLAUDE.md): venv + models present, BlackHole device selected, `ollama serve` up if iterating.

### Parallel Execution Waves
```
Wave 1 (foundation, independent):
├── Task 01: ... [golang-expert]
└── Task 02: ... [python-expert]

Wave 2 (core, depends on Wave 1):
├── Task 03: ... [strudel-expert]
└── Task 04: ... [audio-dsp-expert / ml-audio-expert]

Wave FINAL (after all tasks — parallel review):
├── Task F1: /self-review 4-agent audit
└── Task F2: eval-gate + similarity check on target track
```
**Critical Path**: Task 01 → Task 03 → F2

### Agent Dispatch Summary
Map each task to a domain-expert subagent (`golang-expert`, `python-expert`, `audio-dsp-expert`, `ml-audio-expert`, `strudel-expert`, `llm-expert`, `music-theory-expert`).

---

## TODOs

- [ ] 1. **<Task title>**

  **What to do**:
  - Concrete, file-level steps.

  **Must NOT do**:
  - Scope guardrails for this specific task.

  **Recommended Agent**: `<domain-expert>`

  **Evidence**: `.sisyphus/evidence/task-01-<slug>.txt`

---

## Success Criteria

### Verification Commands
```bash
go build -o bin/midi-grep ./cmd/midi-grep
go test ./...
cd scripts/python && .venv/bin/python -m pytest tests/ -v
```

### Final Checklist
- [ ] All tasks complete with evidence captured
- [ ] Eval gate green
- [ ] Docs updated (`llms.txt`, `llms-full.txt`, `CLAUDE.md`) per Context Document Maintenance
