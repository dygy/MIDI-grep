# Delivery Flow

Generated **by hand** on 2026-10-09 from the AWOS 2.4.5 `/awos:flow` templates (`delivery-flow-template.md`, `implement-feature-template.md`, `fix-bug-template.md`) — the `/awos:flow` generator itself no longer ships with the plugin, so this file and the commands derived from it were filled in manually, following the same interview dimensions. This file is the single source of truth for the project's delivery decisions — `.claude/commands/implement-feature.md`, `.claude/commands/fix-bug.md` and `.claude/commands/session-init.md` are derived from it. **To change a decision, edit this file and then the generated command(s) directly** — there is no regeneration step; keep the two in step by hand and record the change in the Generation Log. **The generated commands are self-contained and do not read this file during a run** — it is the record for humans, not per-run context.

- **Status:** Approved
- **Team Docs Consulted:**
  - `CLAUDE.md` — "Working Agreement" (clarifying-questions budget, environment preflight, self-review, no pre-existing issues, delegate to domain experts, context-document maintenance), "Build & Run", "Testing"
  - `llms.txt` / `llms-full.txt` — project overview and script/flag reference
  - `Makefile`, `scripts/node/package.json`, `.mcp.json`, `eval/thresholds.yaml`, `mcp_servers/loop/README.md`
  - `.claude/skills/{wt,self-review,sisyphus-plan}/` — the project's own stage automation
  - `.awos/commands/{spec,tech,tasks,implement,verify}.md` — the vendored AWOS core the flow chains
  - `context/product/{product-definition,architecture,roadmap,values}.md`, `context/spec/001–003`
  - The sibling project's hand-condensed flow (Citation, 2026-09-17) for shape and the agent-budget lesson

---

## Project Setup

The canonical project config every generated flow command checks against. Flow-agnostic — `implement-feature`, `fix-bug` and `session-init` reuse these facts.

- **Atlassian/Jira base URL:** n/a — no tracker. Work arrives as prompt text, a local file, or a GitHub issue on the code host (see §1).
- **Slack channel & handles:** n/a — no Slack; interactive runs only (§9).
- **Code-host org/repo:** GitHub **`dygy/MIDI-grep`** (`origin` = `git@github.com:dygy/MIDI-grep.git`); the `gh` CLI is authenticated as the repo owner.
- **Format/lint gate scope:** none repo-wide. The only static gate is `go build ./... && go vet ./...` (Go), `pytest` over `scripts/python/tests`, and `tsc` via `npm run build` in `scripts/node` — none of them sweeps markdown, so AWOS working files need no ignore entry. No `gofmt`/`ruff`/`prettier`/`eslint` gate exists; no pre-commit hooks.
- **Reused-skill overrides:** none — `wt` branches from `origin/main` (matches §2); `sisyphus-plan` already requires BlackHole evidence and the eval gate (matches §4 gate 6); `self-review` diffs against `main` by default (matches §2).

## Generated Commands

- **Feature command:** `/implement-feature` → `.claude/commands/implement-feature.md`
- **Bug-fix command:** `/fix-bug` → `.claude/commands/fix-bug.md`
- **Session router (extra, not an AWOS template):** `/session-init` → `.claude/commands/session-init.md` — one cheap state call plus routing to the two commands above, a similarity run, or doc-only spec work.

## 1. Feature Description Source

- **Source:** prompt text, a local file path, or a **GitHub issue** (URL or `#number`) on `dygy/MIDI-grep`. Pre-generated specs under `context/spec/` are supported (entry-point detection below). With no input at all the command resumes from the first unchecked `- [ ]` item in `context/product/roadmap.md`, as `/awos:spec` does; a missing or fully-checked roadmap means ask the user.
- **Fetch transport:** `gh issue view <n> --repo dygy/MIDI-grep --json number,title,body,labels,comments,url` (§7 — repo-provisioned via PATH; prerequisite `gh auth login`). Fallback when `gh` is unauthenticated: ask the user to paste the issue text and proceed as prompt text. A local file is read directly; prompt text is normalized in place.
- **Normalization:** `TICKET_ID` = the issue number (`#42`) or, for prompt/file sources, a kebab-case slug of the title (≤ 5 words); title; description; acceptance hints (bullet lists, "should"/"must" sentences, cited similarity targets); link (issue URL or file path). Spec directory = `context/spec/NNN-<slug>/` with `NNN` the next zero-padded sequential index (existing: `001-core-pipeline`, `002-ml-customization`, `003-editable-strudel-generation`) — created by `.awos/scripts/create-spec-directory.sh <slug>` when `/awos:spec` does not create it itself; the GitHub issue number, when there is one, goes in the spec header as the source link, never in the directory name.
- **Done-state names for resume-detection:** GitHub issue state `CLOSED`; owning spec `Status: Completed`; all `tasks.md` items `[x]`; a merged PR whose branch name carries the slug (`gh pr list --state merged --search "<slug>"`).
- **Surrounding context to pre-seed the spec:** for a GitHub issue — its comments and any linked issues/PRs the body references (`gh issue view`, `gh pr view`), read inline. Links to external docs are fetched with `WebFetch` best-effort and listed as unreachable when they fail. For prompt/file sources: none. The spec interview also opens on `CLAUDE.md`'s "CRITICAL PRINCIPLES — ZERO HARDCODING" and `context/product/values.md`, which every MIDI-grep feature must honor.
- **Pre-generated specs:** yes — `context/spec/NNN-<slug>/` may pre-exist with any subset of the three artifacts (today all three existing dirs hold `functional-spec.md` only). Entry-point rule: resume from the first missing artifact — skip `/awos:spec` if `functional-spec.md` exists, skip `/awos:tech` if `technical-considerations.md` exists, skip `/awos:tasks` if `tasks.md` exists without the `<!-- not-user-reviewed -->` marker. Matching is by slug or by the `TICKET_ID` recorded in the spec header / flow log.

## 2. Git Flow

- **Base branch:** `origin/main` (fetch first, always).
- **Branch naming:** `feat/<slug>` for features, `fix/<slug>` for bugs, `docs/<slug>` for spec-or-doc-only work. Example: `feat/section-aware-hat-gain`, `fix/recorder-tempo-drift`.
- **Submodules:** none.
- **Target sync & conflicts:** while the branch is private, **rebase onto `origin/main`** and push with `--force-with-lease`; once a PR is open, **merge `origin/main` in** instead (never rewrite a branch under review). Checked at branch creation, before `gh pr create`, and again before merge. Conflict detection is in-memory: `git fetch origin main && git merge-tree --write-tree origin/main HEAD` — a non-zero exit or `CONFLICT` lines mean conflicts; nothing touches the working tree. Trivial conflicts are resolved inline; a non-trivial resolution may use one subagent and is confirmed with the user; the §4 static gate re-runs after any sync. **Wanted guard (not yet present — probed 2026-10-09):** a `PreToolUse` hook `.claude/hooks/branch-current.sh` that runs the same `merge-tree` check and refuses `gh pr create` / `gh pr merge` / new-branch creation when the branch is behind or conflicting, bypass `MIDIGREP_SKIP_BRANCH_CHECK=1`. Until it lands, the generated commands run the check themselves at the three points above.
- **Commit message convention:** `feat(<area>): <summary>` / `fix(<area>): <summary>` / `docs(spec): <summary>` with `<area>` one of `go`, `python`, `node`, `strudel`, `eval`, `docs`, and the spec id (`spec 004`) or issue (`#42`) in the body. Repo history is free-form imperative; this convention starts with the generated commands. End every commit with the attribution trailer the running harness prescribes.
- **Worktrees:** viable for code-only work via the project's **`wt` skill** (`/wt <slug>` → `.claude/worktrees/<slug>` on a fresh branch from `origin/main`; the session `cd`s into it). Bring-up a fresh worktree lacks (all git-ignored): `go build -o bin/midi-grep ./cmd/midi-grep`; `cd scripts/node && npm install && npm run build` (`node_modules/` and `dist/` are ignored); the Python venv is **not re-created per worktree** — `scripts/python/.venv` is ignored and a `wt` worktree (`.claude/worktrees/<name>`, inside the repo) has none of its own, so Python runs there via the main repo's venv by absolute path (`$(git rev-parse --show-toplevel | sed 's#/.claude/worktrees/.*##')/scripts/python/.venv/bin/python`; rebuild with `make install-python-deps` only if truly missing — ~1 GB of ML models download on first use). The `.mcp.json` `loop` server's relative venv path does **not** resolve from a worktree session — one more reason renders run from the main repo; `.cache/stems/` is ignored and per-checkout — a worktree starts with an empty stem cache unless the user copies or symlinks the main repo's `.cache/` in. **Render verification is main-repo-only:** BlackHole + the Multi-Output audio device + the ffmpeg capture are one shared machine resource — two recorders at once produce two broken recordings. A worktree run therefore does its code work in the worktree and its BlackHole render from the main repo (or serially, after confirming no other recorder is running).
- **Sanctioned verification path:** one render at a time, from the main repo, via either the **`loop` MCP** — `verify_strudel(strudel_path, original, genre, duration, recorder='blackhole')` (render + compare + gate in one call) or `eval_gate(comparison_json, genre)` on an existing `comparison.json` — or the CLI `node scripts/node/dist/record-strudel-blackhole.js <in.strudel> -o <out.wav> -d 30` followed by `scripts/python/.venv/bin/python scripts/python/compare_audio.py`. Before starting a render: `pgrep -fl record-strudel-blackhole` must be empty and the Multi-Output device must be the system output (`system_profiler SPAudioDataType | grep -c BlackHole` ≥ 1). The Node synth (`recorder='node'`, `render-strudel-node.js`) is a smoke render only and never gates. The web UI (`./bin/midi-grep serve`) is verified on an **alternate port** `--port 8089` with the Playwright MCP so a user's running `:8080` server is never disturbed.

## 3. Repository Topology

- **Layout:** single repo (Go module at the root, Python under `scripts/python/`, TypeScript under `scripts/node/`, MCP server under `mcp_servers/loop/`).
- **`context/` location & sharing:** in-repo at `context/` (`product/`, `spec/`); nothing is symlinked.
- **Spec commits go to:** this repo, on the feature branch (`feat/<slug>`), committed before implementation starts (Step 5 of the feature command).
- **`context/` reachability check:** `git rev-parse --show-toplevel` is the repo root and `context/product/product-definition.md` exists there; `ls context/spec/` lists the existing `NNN-*` dirs. In a worktree the same check runs against the worktree root.

## 4. Review

Gates in order:

1. **Static checks (local, mirrored by CI):** `go build ./... && go vet ./... && go test ./...` (first Go tests: `internal/cache/cache_test.go`); `scripts/python/.venv/bin/python -m pytest -q scripts/python/tests` (14 test files incl. `test_similarity_gate.py`, which asserts `eval/thresholds.yaml` floors); `cd scripts/node && npm run build` (`tsc`; needs `npm install` once per checkout). Bar: all three green.
2. **Local AI review:** Claude Code's built-in **`/code-review` skill at medium effort** on the branch diff against `origin/main`, invoked from the main context (the skill dispatches its own reviewers), pointed at the spec dir as ground truth, with **no author-added focus areas**. Findings are written to `review/<slug>.md` — **git-ignored working evidence**; the `.gitignore` entry `review/` does not exist yet and adding it is a one-time project task for the user (the commands check `git check-ignore review/` and say so rather than editing `.gitignore`). The user keeps/drops each finding via `AskUserQuestion`; accepted findings go back to the specialist that wrote the code; the static gate re-runs. The project's `self-review` skill (4-agent audit) is the heavier alternative for substantive diffs — compose: run `/code-review` always, `/self-review` when the diff touches more than one of Go/Python/Node or the user asks. For fixes the file is `review/fix-<slug>.md`.
3. **Automatic reviewer on the code host:** none (probed 2026-10-09: no `.github/`, no CodeRabbit/Copilot config). The local review is the only automated review; the PR description says so.
4. **Remote PR review:** PR on GitHub `dygy/MIDI-grep` → `main`; human reviewer = the repo owner (who is also the operator). The flow does not poll for approval — it reports ready-to-merge and the owner reviews in the session or on GitHub.
5. **CI on the change request:** `.github/workflows/ci.yml` (added 2026-10-09) runs four jobs on every PR and on pushes to `main`: `go` (build, vet, gofmt check, test), `node` (`npm ci` with `PUPPETEER_SKIP_DOWNLOAD=1`, `tsc`, recorder artifact present), `python` (pytest over `tests/` + root `test_*.py` on the lightweight analysis subset — no TensorFlow/Demucs/CLAP), `hooks` (`bash -n` on the hook scripts, JSON validity of `.claude/settings.json` and `.mcp.json`). Typical duration: unknown until the first runs (estimate 3–6 min; the python job dominates). Policy: **wait + fix failures in a loop** via the `gha-diagnosis` skill (`gh run view`), re-push, repeat. BlackHole renders can never run in CI — the eval gate stays a local, macOS-only step (§2).
6. **Eval gate (render verification):** **every change that can alter Strudel output** — `scripts/python/*codegen*`, `ollama_*`, `synth_profiles.py`, `calibrate_dynamic.py`, `generate_*_strudel.py`, prompt/Modelfile text, the recorder, `compare_audio.py`, `eval/` — must be rendered through **BlackHole** and pass `eval/thresholds.yaml` via the `loop` MCP (`verify_strudel` with `recorder='blackhole'`, or `eval_gate` on the run's `comparison.json`). The Node recorder **never gates**. Numbers quoted anywhere (spec, PR, docs) come from a real `comparison.json` of that run. A multi-iteration similarity run records its evidence under `.sisyphus/evidence/` via the `sisyphus-plan` skill, with the plan linking the spec dir. Changes provably outside the Strudel path (Go CLI plumbing, report HTML, docs) skip this gate and say so.

- **Approval gates:** **two** — after `/awos:spec` and after `/awos:tech` (human, `AskUserQuestion` approve / revise). **`tasks.md` has no human gate:** after `/awos:tasks` the orchestrator sanity-checks the plan (every task traces to the tech spec; a Feature Testing & Regression slice exists unless `<!-- skip-tests: true -->`; each task carries an `**[Agent: name]**` from the hired roster), removes the `<!-- not-user-reviewed -->` marker itself and logs "tasks auto-approved per delivery-flow §4" — `/awos:implement` refuses to run while the marker is present, so this step is load-bearing.
- **Change-request timing:** after local review.
- **Max-wait & escalation:** poll `gh pr checks BRANCH --watch` (or a `Monitor` sized to ~10 min); after one automatic fix-and-repush loop that still fails, stop and show the user the failing job's log instead of iterating further. The human review is never polled.

## 5. Delivery

- **Mode:** deploy-when-ready — there is no deployment; "delivered" = merged into `main`.
- **Merge policy:** a human merges. The flow may run `gh pr merge <n> --squash --delete-branch` only behind a per-run `AskUserQuestion` (merge / don't merge) after showing the gate evidence; a skipped or unanswered prompt means **do not merge**. An earlier "run it end to end" is never merge approval.
- **Post-merge CI:** the same workflow runs on `main` after the merge; policy **wait + report** — a red `main` run is reported immediately (and fixed forward with `/fix-bug`), never ignored.
- **Approvals:** the repo owner, recorded by the PR approval/merge on GitHub.
- **Versioning:** none yet — no tags, no version bumps (`scripts/node/package.json` stays `1.0.0`).
- **Deployment:** none — local CLI tool; the binary is rebuilt by whoever pulls (`go build -o bin/midi-grep ./cmd/midi-grep`).
- **Ticket state transitions:** n/a — ticketless source. A GitHub issue, when one exists, is closed by the PR body's `Closes #n` on merge; no intermediate states.
- **Definition of Done:** PR merged into `main` **and** the spec's `Status: Completed` set by `/awos:verify` **and** `llms.txt` / `llms-full.txt` / `CLAUDE.md` updated per `CLAUDE.md` "Context Document Maintenance" whenever the change touched `internal/`, `scripts/python/`, `scripts/node/`, modes, flags, synthesis params or the similarity metric **and** every similarity number cited in the spec, PR or docs traces to a real `comparison.json` (BlackHole render) of that run. Review evidence (`review/<slug>.md`) is reported, not committed.

## 6. Trigger

- **Supported:** manual — `/implement-feature <issue# | URL | file path | feature text>` and `/fix-bug <issue# | URL | description | path to a failing comparison.json>`.
- **Wanted (setup notes only):** none — the owner runs the flow interactively.

## 7. Tooling Inventory

| Service | CLI | MCP | Plugin/Skill | Chosen Transport | Provenance | Operator prerequisites |
| ------- | --- | --- | ------------ | ---------------- | ---------- | ---------------------- |
| GitHub `dygy/MIDI-grep` (issues, PRs, merge) | ✓ `gh` | — (the `github` plugin MCP fails to connect on this machine) | — | **`gh` CLI** | repo-provisioned via PATH | `gh auth login` with `repo` scope (verified: account `dygy`) |
| Render → compare → gate | ✓ `node scripts/node/dist/record-strudel-blackhole.js`, `compare_audio.py`, `eval/gate.py` | ✓ **`loop`** (`verify_strudel`, `render_strudel`, `compare_render`, `eval_gate`) | — | **`loop` MCP** (CLI fallback) | repo-provisioned — `.mcp.json` (stdio, runs from `scripts/python/.venv`) | venv built (`make install-python-deps` + `pip install fastmcp pyyaml`); `cd scripts/node && npm install && npm run build`; **macOS + BlackHole 2ch + a Multi-Output Device set as system output** |
| Web UI verification (`serve`) | — | ✓ `playwright` | — | **Playwright MCP** on `--port 8089` | repo-provisioned — `.mcp.json` (`npx @playwright/mcp`) | Node on PATH |
| Agent hiring | — | ✓ `awos-recruitment` | `/awos:hire` | **MCP** | repo-provisioned — `.mcp.json` (http) | network |
| LLM codegen (Ollama) | ✓ `ollama` | — | — | **`ollama serve`** on `localhost:11434` | project setup (`CLAUDE.md` "Ollama Setup") | `ollama create midi-grep-strudel-mistral -f Modelfile.mistral`; ~13 GB RAM |
| Audio tooling | ✓ `ffmpeg`, `yt-dlp` | — | — | **CLI** | project setup (brew) | on PATH |
| Build/verify — Go | ✓ `go` | — | — | `go build/vet/test` | `go.mod` | Go 1.21+ |
| Build/verify — Python | ✓ `scripts/python/.venv/bin/python -m pytest` | — | `pytest-best-practices`, `python-testing-patterns` skills | **venv pytest** | `scripts/python/requirements.txt` | Python 3.11 venv (TensorFlow 2.15 pin) |
| Build/verify — Node | ✓ `npm run build` | — | `typescript-development` skill | **`tsc`** | `scripts/node/package.json` | Node 20+ |
| Local review | — | — | built-in **`/code-review`** skill; project `self-review` skill | **`/code-review` medium** | Claude Code built-in; `.claude/skills/self-review` | — |

**Machine-personal transports:** **BlackHole + Multi-Output Device** — macOS-only, installed per machine (`brew install blackhole-2ch` + reboot + Audio MIDI Setup). Kept as the **required** gate-6 transport, not an accelerator: there is no other honest render, so on a machine without it the eval gate cannot run and the flow stops at verify with "render not possible here" rather than gating on the Node synth. The `github` plugin MCP and IDE MCPs (`goland`, `pycharm`, `webstorm`) are user-level and unused by the flow.

**Long-lead operator prerequisites:** BlackHole install needs a **reboot** and a manual Multi-Output Device; the first ML run downloads ~1 GB of models (Demucs, Basic Pitch, CLAP); the Ollama model build pulls ~13 GB. All self-service, but each costs tens of minutes — do them before the first run, not inside it.

**Stage automation (reuse / replace / compose):**

- **workspace:** reuse the **`wt` skill** when a worktree is wanted; main repo by default (§2).
- **specs → implement → verify chain:** reuse the vendored AWOS core (`.awos/commands/*.md` via `.claude/commands/awos/*`) unmodified, except that the feature command drives `tasks.md` in per-domain batches instead of `/awos:implement`'s one-agent-per-task loop (§8).
- **local review:** **compose** — built-in `/code-review` (medium) always; the project's `self-review` skill for cross-language diffs or on request.
- **similarity runs / evidence trail:** reuse **`sisyphus-plan`** for any multi-iteration render loop inside implement — the plan under `.sisyphus/plans/NNN-<slug>.md` must link `context/spec/NNN-<slug>/`, evidence files carry genre + similarity % + render path.
- **verification:** reuse **`/awos:verify`** (main context) plus the **`loop` MCP** for the eval gate.
- **conventions:** `CLAUDE.md` "Working Agreement" is folded into every command's Run Discipline (one clarifying round, preflight before long runs, self-review after edits, no pre-existing issues, delegate to domain experts).

## 8. Context Strategy

- **Mode:** single session, **inline by default**, with a small fixed set of subagent stages. One feature = one branch = one session.
- **Agent budget (from `CLAUDE.md` "Working Agreement" and the Citation 2026-09-17 lesson):** at most **one round of clarifying questions**; work inline by default; **one specialist per domain** (Go / Python / Strudel-LLM / ML-audio), never one per task; **3–5 agents per feature normally, hard stop at 8** (1–3 per bug, hard stop 6) — past the stop, pause and tell the user why. Follow-ups (blocked tasks, accepted review findings, test retargeting) go back to the **same** agent via `SendMessage`. `/code-review`'s internal reviewers and `SendMessage` follow-ups are not new dispatches.
- **Stages isolated in subagents:** (1) **fetch** — `gh issue view` + normalization, fast tier (only when the source is a GitHub issue with many comments; a plain prompt is normalized inline); (2) **static checks** — gate 1 run and summarized, fast tier; (3) **implementation** — one hired specialist per affected domain (§ roster), strongest tier; (4) one `testing-expert` dispatch for the whole testing slice; (5) a non-trivial merge conflict; (6) deep investigation across files not yet opened (`Explore`), at most one per domain in one parallel batch. (7) **CI wait** — `gh pr checks --watch` and, on red, the `gha-diagnosis` triage, fast tier; the `gh pr create` call itself runs inline.
- **Stages kept in the main context:** `/awos:spec`, `/awos:tech`, `/awos:tasks` (all interview the user — 6/3/6 `AskUserQuestion` sites); implement orchestration (dispatches subagents; agents do not nest); `/awos:verify` (5 `AskUserQuestion` sites, and it drives the shared BlackHole device, which must be serialized from one place); `/code-review` invocation (dispatches its own reviewers); merge (asks); close.
- **Flow log:** `context/spec/NNN-<slug>/flow-log.md` for the feature command — committed, since the spec dir rides the feature branch. For the bug-fix command the log is a **scratchpad working file** `fix-log-<id>.md`, moved to `context/fix-log-<id>.md` and committed **only when the fix amends a spec** (then `context/` already changes). Both are finalized at commit-push and frozen once the PR exists; later progress is reported in the session; remote stages resume from `gh pr view`.
- **Model tiers:** fast tier for fetch and the static-check runner; strongest tier for every specialist, reviewer, `Explore` investigation and conflict resolution. Tiers, not model names.

## 9. Notifications

- **Channel:** none (interactive runs only).
- **Announce on:** nothing — the operator watches the session; the PR on GitHub is the only shared-state artifact.

## Bug-fix Flow

- **Generated:** yes — `/fix-bug` → `.claude/commands/fix-bug.md`.
- **Bug source:** a plain description; a GitHub issue (`gh issue view`); or a **failing render** — a path to a `comparison.json` (under `.cache/stems/<key>/vNNN/`) whose gate failed, in which case the symptom is the band/metric that breached `eval/thresholds.yaml` and the reproduction is `eval_gate(comparison_json, genre)` via the `loop` MCP. No crash-reporting tool exists.
- **Classification & amendment policy:** every fix is classified **conformance** (code violated a correct spec → fix + regression test, no spec change) vs **divergence** (spec wrong/incomplete or behavior intentionally changed → fix + regression test + amend the owning `context/spec/NNN-*/functional-spec.md` via `/awos:spec` update mode, user-confirmed). A bug mapping to no spec (most of today's pipeline predates specs 001–003's acceptance criteria) records "no owning spec" and proceeds without amendment; bugs never create a spec.
- **Regression-test expectation:** one failing→passing **pytest** (under `scripts/python/tests/`) or Go test, demonstrated by set-aside → fail → restore → pass with both outputs quoted (RED proof), honoring `<!-- skip-tests: true -->` in the owning spec's `tasks.md`. For a similarity regression the test is a threshold assertion in `test_similarity_gate.py` style **plus** the BlackHole re-render clearing the gate.

## 10. Local Customizations

- **2026-10-09 — hand-generated.** No `/awos:flow` exists to regenerate from; every later change is a direct edit here and in the commands, logged below.
- **2026-10-09 — hired roster for Step 6 dispatch:** `golang-expert` (`internal/`, `cmd/`), `python-expert` (`scripts/python/` plumbing), `ml-audio-expert` (librosa / demucs / `compare_audio.py` / `eval/`), `audio-dsp-expert` (synthesis, recorder DSP), `strudel-expert` (Strudel output and the Strudel content of prompts), `llm-expert` (`ollama_*`, prompts, Modelfiles), `music-theory-expert` (keys, chords, arrangement), `testing-expert` (acceptance tests, the Feature Testing & Regression slice). The first seven live in `.claude/agents/`; `testing-expert` is provided by the `awos` plugin (no file in `.claude/agents/` — probed 2026-10-09).

---

## Generation Log

- 2026-10-09 — **initial generation, by hand from the 2.4.5 templates.** Decisions settled from `CLAUDE.md`'s Working Agreement and the repo probes below: ticketless source (prompt / file / GitHub issue via `gh`); `origin/main` base with `feat|fix|docs/<slug>` branches, rebase-while-private / merge-in-once-open, `git merge-tree` conflict check; worktrees via `wt` for code, BlackHole render main-repo-only; gates = local static suite → `/code-review` medium → PR to `main` (owner reviews) → no CI (wanted) → BlackHole eval gate through the `loop` MCP, Node recorder never gates; two document gates, `tasks.md` auto-approved; human merges (squash) behind `AskUserQuestion`; DoD = merged + `Status: Completed` + context docs updated + cited numbers from a real `comparison.json`; agent budget 3–5/8 (features), 1–3/6 (bugs); no notifications; bug-fix flow with conformance/divergence classification and a RED-proven regression test.
- **Probe records — 2026-10-09:**
  - `git remote -v` → `origin git@github.com:dygy/MIDI-grep.git`; `gh auth status` → logged in as `dygy`, scopes `gist, read:org, repo, workflow` — **gh auth ok**.
  - `ls .github` → did not exist at generation → **no CI, no bot reviewer config**. Later the same day `.github/workflows/ci.yml` was added (go / node / python / hooks jobs); the python job's dependency subset was reproduced locally in a fresh 3.11 venv (76 + 97 tests green). No bot reviewer is configured.
  - `ls .pre-commit-config.yaml .git/hooks/pre-commit` → neither exists → **no pre-commit hooks**. `.claude/settings.json` `hooks` (added later the same day) → SessionStart `/session-init` pointer; PreToolUse `Bash` → `.claude/hooks/branch-current.sh` (**blocks** `git checkout -b`/`git switch -c`/`gh pr create` when behind `origin/main`, `gh pr merge` on conflict; override `MIDIGREP_SKIP_BRANCH_CHECK=1`); PostToolUse `Edit|Write|MultiEdit` → `.claude/hooks/docs-freshness.sh` (advisory). The commands' own `merge-tree` checks stay as defense in depth.
  - `.gitignore` → `review/` and `docs/screenshots/` ignored (added 2026-10-09); `*.wav`/`*.mp3` ignored (renders are never committed); `.cache/`, `scripts/python/.venv/`, `/scripts/node/node_modules/` ignored.
  - `.mcp.json` → `playwright` (npx), `loop` (stdio from the venv, `PYTHONPATH=.`), `awos-recruitment` (http). `.claude/settings.json` → `awos@awos-marketplace` plugin enabled.
  - `.claude/agents/` → 8 agent files (audio-dsp, golang, llm, ml-audio, music-theory, python, strudel, **testing-expert**), each with `skills:` bindings; roster recorded in `context/product/hired-agents.md`.
  - `.claude/skills/` → `wt`, `self-review`, `sisyphus-plan` (+ language skills); `.sisyphus/` exists with plans 001–002.
  - `context/spec/` → `001-core-pipeline` (Status: Implemented), `002-ml-customization` (Research Complete), `003-editable-strudel-generation` (Draft) — **each holds `functional-spec.md` only**; next index is `004`.
  - `scripts/python/tests/` → 14 `test_*.py` files, no `pytest.ini`/`conftest.py`; `find . -name '*_test.go'` → **no Go tests**.
  - `go build ./... && go vet ./...` → **green** at generation time.
  - `scripts/python/.venv/bin/python` present; `scripts/node/node_modules` **absent** (`dist/` is present) → `npm install` is a bring-up step before `npm run build`.
  - `system_profiler SPAudioDataType | grep -c BlackHole` → `1` (device present); `curl localhost:11434/api/tags` → Ollama up with `midi-grep-strudel-mistral:latest`.
  - `.awos/commands/*.md` interaction profile: `spec` 6 `AskUserQuestion` / 0 dispatch; `tech` 3 / 2; `tasks` 6 / 8; `implement` 1 / 11; `verify` 5 / 0 → all five stay in the main context (§8).
- 2026-10-09 — reconciliation after generation: hooks installed (`.claude/hooks/`, `.claude/settings.json`), `testing-expert.md` added, `review/` + `docs/screenshots/` git-ignored; probe records above updated to match.
- 2026-10-09 — CI added: `.github/workflows/ci.yml`; §4 gate 5, max-wait, post-merge policy and the `.github` probe record updated; `gha-diagnosis` installed as the CI-triage skill. Registry probed for a Go skill and hooks: none exist (see `hired-agents.md`).
