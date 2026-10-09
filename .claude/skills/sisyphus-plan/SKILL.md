---
name: sisyphus-plan
description: Plan and track multi-step / long-running MIDI-grep work (features, big refactors, multi-iteration similarity-improvement runs) using a durable plan + evidence trail under .sisyphus/. Use when a task needs 3+ coordinated steps, parallel subagents, or an auditable record of what ran and what similarity it achieved. Triggers on "plan this", "make a plan", "track this run", "boulder", "sisyphus".
---

# Sisyphus Plan & Evidence Workflow

A lightweight orchestration convention for work too big to hold in one context: write a durable plan, dispatch parallel subagents, and capture **evidence** of every task so the run is auditable. In MIDI-grep, "evidence" centrally means the **similarity score + render artifact** a change actually produced — not a claim that it improved.

## Directory layout

```
.sisyphus/
├── boulder.json          # active-run state (current plan, sessions)
├── plans/
│   ├── TEMPLATE.md        # copy this to start a plan
│   └── NNN-slug.md        # one file per feature/run, zero-padded number
├── drafts/                # rough notes before a plan is finalized
├── evidence/              # task-NN-<slug>.{txt,json,png} — proof each task ran
└── notepads/<plan>/       # decisions.md, issues.md, learnings.md, problems.md
```

## When to use

- A feature or refactor needing 3+ coordinated steps across Go / Python / Node.
- A multi-iteration similarity run where you want a record of what each change did.
- Anything you'll dispatch to multiple domain-expert subagents in parallel.

For a one-file change, skip this — just do it.

## Workflow

1. **Draft** (optional): jot the idea in `.sisyphus/drafts/NNN-slug.md`.
2. **Plan**: `cp .sisyphus/plans/TEMPLATE.md .sisyphus/plans/NNN-slug.md` (next number) and fill in:
   TL;DR, Deliverables, Definition of Done, **Must NOT Have** guardrails, parallel waves, and a TODO list mapping each task to a domain-expert agent.
3. **Activate**: set `boulder.json` → `active_plan` to the plan path, `plan_name`, and `started_at` (use a real timestamp — get it via `date` since it's a manual field).
4. **Execute in waves**: dispatch each wave's tasks to their `Recommended Agent` (the `.claude/agents/` domain experts) — independent tasks in parallel, dependent waves after. Mirror the plan's TODOs into the harness task list (`TaskCreate`) so progress is visible.
5. **Capture evidence**: after each task, write `.sisyphus/evidence/task-NN-<slug>.txt` (or `.json`/`.png`). For audio changes this MUST include the achieved **similarity %**, the genre, and the render path. Numbers must come from an actual `compare_audio.py` run via the BlackHole render — never the Node.js renderer for accept/reject (see MEMORY: Node.js gives ~16% vs BlackHole ~65%).
6. **Notepad** (as you go): record cross-task `decisions.md`, `issues.md`, `learnings.md` under `.sisyphus/notepads/<plan>/`.
7. **Final wave**: run `/self-review` (4-agent audit) and the eval gate (`eval/thresholds.yaml`) before declaring done. Update `llms.txt` / `llms-full.txt` per CLAUDE.md Context Document Maintenance.
8. **Close**: reset `boulder.json` to the empty schema.

## Rules

- The **Must NOT Have** section is mandatory — it's what keeps subagents in scope.
- Evidence is the contract: a task is not done until its evidence file exists and shows the real result.
- Don't delete ClickHouse `runs`/`knowledge` or agent history during a run — that's learning data, not garbage.
