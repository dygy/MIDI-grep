---
description: Implements one MIDI-grep feature end-to-end — reads it from a prompt, a file or a GitHub issue, runs the AWOS chain (spec → tech → tasks → implement → verify), renders the result through BlackHole against the eval gate, reviews locally, and opens the PR on dygy/MIDI-grep for the owner to merge.
argument-hint: '[feature — GitHub issue # or URL, a file path, or the feature text]'
---

# Implement a Feature End-to-End

Takes one feature — from prompt text, a local file, or a GitHub issue on `dygy/MIDI-grep` — and drives it through spec, implementation, verification (including the BlackHole render gate when Strudel output can change), local review, and a pull request to `main` until it is Done. Run it from a session anchored at the repo root (or a `wt` worktree of it).

This command is self-contained and derived from `context/product/delivery-flow.md`, the decision record. **Do not read that file during a run.** There is no `/awos:flow` to regenerate this command: to change a decision, edit the record and then this file directly, and log the change in the record's Generation Log.

## Arguments

`$ARGUMENTS` — a GitHub issue number (`42`, `#42`) or URL on `dygy/MIDI-grep`, a path to a local file describing the feature, or the feature text itself. A pre-written spec dir under `context/spec/` is also accepted. If empty, take the first unchecked `- [ ]` item in `context/product/roadmap.md`; if the roadmap is missing or fully checked, ask the user.

## Run Discipline

**Agent budget (CLAUDE.md Working Agreement).** Subagents start cold and return claims you must re-check; one agent per stage or per task is slow and no more correct. Rules:

- **Work inline by default.** `gh` calls, `git`, static checks, file reads a diagnosis points at, `loop` MCP calls, flow-log writes — all in the main context. Never dispatch an agent for one tool call.
- **Subagents are allowed for exactly these:** (1) implementation — **one specialist per affected domain** (Step 6), never one per task; (2) **one** `testing-expert` dispatch for the whole testing slice; (3) deep investigation across files you have not opened (`Explore`), at most one per domain, independent ones in one parallel batch; (4) a non-trivial merge conflict; (5) a GitHub issue with a long comment thread may be fetched and normalized by a fast-tier agent.
- **Follow-ups reuse the agent.** Blocked tasks, accepted review findings and a test that must be retargeted go back to the **same** specialist via `SendMessage`; dispatch a new one only if it is gone.
- **Counting:** `/code-review`'s internal reviewers, `/self-review`'s four auditors and `SendMessage` follow-ups are not new dispatches. **Expected: 3–5 agents per feature. Past 8, stop and tell the user why before continuing.**
- Every brief says: read `.awos/subagents/<name>.md` (or `.claude/agents/<name>.md`) first; tools are functional — no exploratory calls; report tersely (paths, verdicts, counts) and **quote command output as evidence**. A report is a claim — re-read the named lines or re-run the named test before acting on it.
- Never launch `claude -p` from this command.

**Questions.** At most **one round** of clarifying questions before acting (CLAUDE.md); otherwise state the assumption and proceed. Every fixed-choice question goes through `AskUserQuestion` with the default marked. An unanswered prompt takes the safe default once and says so — never re-ask in a loop — and **never** authorizes an irreversible step: the merge confirmation treats silence as no.

**No pre-existing issues.** A failing test, a broken script or a wrong similarity metric met along the way is fixed in this run, not worked around (CLAUDE.md). **Zero hardcoding:** no gain, filter, effect or threshold is hand-tuned to the test track — it comes from analysis, the calibrator, or the LLM.

**Flow log.** From Step 4 on, append a short entry per completed stage to `context/spec/{SPEC_NAME}/flow-log.md` — stage, what was produced (paths, branch, commit, similarity numbers with their `comparison.json` path), decisions, next stage; the first entry records `TICKET_ID`, the title and the source. It rides the commit in Step 9 and is **frozen from then on**; afterwards report progress in the session and resume from remote state (`gh pr view`).

**Flow defects.** If a fact in this file turns out wrong (a path moved, a command renamed, a gate that no longer exists), follow reality, note the defect, and list it in the Step 13 report. **Do not edit this command, `delivery-flow.md` or any skill during the run** — flow fixes go in their own `docs/<slug>` branch afterwards.

<!-- awos:flow:stage=fetch-ticket -->

### Step 1: Fetch & Normalize (inline)

Pre-flight in one Bash call: `gh auth status` (must show `Logged in` with `repo` scope — a missing login blocks the PR stage, say so now); `scripts/python/.venv/bin/python -c "import librosa, yaml"`; `ls scripts/node/node_modules scripts/node/dist >/dev/null`; `system_profiler SPAudioDataType | grep -c BlackHole`; `curl -s -m 2 localhost:11434/api/tags | head -c 60`. Report what is missing. BlackHole absent → the eval gate (Step 7) cannot run on this machine; say so up front rather than discovering it after implementation. Ollama down → only blocks if the feature exercises LLM codegen.

Then normalize the source:

- **GitHub issue** (`42`, `#42`, or a `github.com/dygy/MIDI-grep/issues/N` URL): `gh issue view N --repo dygy/MIDI-grep --json number,title,body,labels,comments,url,state`. Keep `TICKET_ID = #N`, title, body, acceptance hints, labels, URL, state. Read linked issues/PRs the body references (`gh issue view` / `gh pr view`); fetch external links with `WebFetch` best-effort and **list the unreachable ones** instead of skipping them.
- **File path:** read it; `TICKET_ID` = kebab slug of its title or first heading (≤ 5 words).
- **Prompt text:** normalize in place; `TICKET_ID` = kebab slug of the feature (≤ 5 words).

Carry the bundle (id, title, description, acceptance hints, source link, unreachable list) into Step 4 — it pre-seeds `/awos:spec` so its interview opens warm.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=resume-detection -->

### Step 2: Detect the Entry Point

Stop only on a **delivered** signal: the GitHub issue is `CLOSED`, or a merged PR carries this slug (`gh pr list --repo dygy/MIDI-grep --state merged --search "<slug>"`). Everything else is a **progress** signal, never a stop: a spec `Status: Completed`, all `tasks.md` items `[x]`, an open PR — resume at the stage after the one the signal names.

Then find the spec: glob `context/spec/*/flow-log.md` and match `TICKET_ID`/title in each first entry; else match `context/spec/*-<slug>/`. If one matches, read its flow log first — it names the last completed stage, the branch and the PR. For the spec stages the on-disk artifacts win over the log: skip `/awos:spec` if `functional-spec.md` exists (and its gate), skip `/awos:tech` if `technical-considerations.md` exists (and its gate), skip `/awos:tasks` if `tasks.md` exists without `<!-- not-user-reviewed -->`. All three existing specs (`001`–`003`) hold `functional-spec.md` only, so a run against one of them resumes at `/awos:tech`. Past the spec stages the log is the only resume signal until a PR exists; then `gh pr view` is. Completed stages are skipped, never repeated, but Step 4 is always entered — the log starts there.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=workspace -->

### Step 3: Prepare the Workspace

`git rev-parse --show-toplevel` is the root; `context/product/product-definition.md` must exist there. `git status --short` — warn on a dirty tree; uncommitted `context/product/delivery-flow.md`, `.claude/commands/*.md` or `.awos/` migration files are an expected cause, not a blocker; anything else stays unstaged throughout the run. `git check-ignore review/` — if it is not ignored, tell the user to add `review/` to `.gitignore` (a one-time project task; do not edit `.gitignore` yourself).

**Main repo vs worktree.** Default is the main repo. Offer a worktree (`AskUserQuestion`, default main repo) only when the main repo is already on another feature's branch with uncommitted work. Worktree recipe: invoke the project's **`wt` skill** (`/wt <slug>`) — it creates `.claude/worktrees/<slug>` on a fresh branch from `origin/main` and `cd`s into it — then bring-up: `go build -o bin/midi-grep ./cmd/midi-grep`; `cd scripts/node && npm install && npm run build`; the Python venv is reused from the main repo's `scripts/python/.venv` (never re-create it — ~1 GB of models); `.cache/stems/` starts empty, so a render-verified feature needs its reference stems copied or the extraction re-run. **BlackHole renders always run from the main repo, one at a time** — a worktree does code work only.

Main repo: `git fetch origin main && git switch -c feat/<slug> origin/main`. Store `BRANCH`. Before creating it, confirm nothing else is in flight on BlackHole (`pgrep -fl record-strudel-blackhole` empty).

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=specs -->

### Step 4: Specs and Tasks (main context)

Honor Step 2: skip any artifact that exists, and its gate. All three commands interview the user, so all three run in the main context. Investigate inline first; dispatch an `Explore` agent only for a domain you have not opened, under the budget rule.

1. **`/awos:spec`** with the Step 1 bundle as its prompt. The spec dir is `context/spec/NNN-<slug>/` (next zero-padded index after the highest existing — `004` today; `.awos/scripts/create-spec-directory.sh <slug>` creates it). Remind the spec of CLAUDE.md's two non-negotiables: **zero hardcoding** (every parameter from analysis, calibration or the LLM) and **similarity numbers only from a real BlackHole `comparison.json`**. Acceptance criteria are written to be checkable by a test or a render. **Gate:** `AskUserQuestion` approve / revise.
2. **`/awos:tech`** against the same dir → `technical-considerations.md` naming concrete files under `internal/`, `scripts/python/`, `scripts/node/src/`, `eval/`, and stating whether the change **can alter Strudel output** (this decides Step 7's render gate). **Gate:** `AskUserQuestion` approve / revise.
3. **`/awos:tasks`** → `tasks.md`, every task carrying `**[Agent: name]**` from the roster: `golang-expert`, `python-expert`, `ml-audio-expert`, `audio-dsp-expert`, `strudel-expert`, `llm-expert`, `music-theory-expert`, `testing-expert`. Tell it **not to ask its Step 5 review question** — there is no human gate here: sanity-check the plan yourself (tasks trace to the tech spec; a Feature Testing & Regression slice exists unless `<!-- skip-tests: true -->`; each task names its files and agent), then **remove `<!-- not-user-reviewed -->`** and log "tasks auto-approved per delivery-flow §4". `/awos:implement` refuses to run while the marker is present.

Set `SPEC_NAME`; write the flow log's first entry (ticket id/title/source, the Step 1 unreachable list). If the feature is a multi-iteration similarity run, also start a `sisyphus-plan` (`.sisyphus/plans/NNN-<slug>.md`) that links the spec dir — evidence goes under `.sisyphus/evidence/`.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=commit-specs -->

### Step 5: Commit Specs

On `BRANCH`: `git add context/spec/{SPEC_NAME}/` and commit as `docs(spec): <slug> functional spec, tech spec + tasks (spec NNN)`. Nothing else is staged.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=implement -->

### Step 6: Implement — one specialist per domain

**Do not run `/awos:implement`'s per-task loop** — it dispatches one agent per task. Drive the same `tasks.md` in batches:

1. Group the open tasks by domain, keeping document order. Domain comes from the task's paths: `internal/`, `cmd/` → `golang-expert`; `scripts/python/*codegen*`, `ollama_*`, prompts, `Modelfile*` → `llm-expert` (Strudel content of a prompt or a `.strudel` template → `strudel-expert`); `compare_audio.py`, `eval/`, `separate.py`, `analyze*.py`, `calibrate_dynamic.py` → `ml-audio-expert`; renderers, `record-strudel-blackhole.ts`, synthesis → `audio-dsp-expert`; other `scripts/python/` → `python-expert`; key/chord/arrangement logic → `music-theory-expert`. All `testing-expert` tasks, in any slice, go to **one** `testing-expert` dispatch after the implementation groups finish. Cross-domain `Verify:` tasks run **inline** once their slice's domains are done. Tasks that need a BlackHole render run inline in Step 7, not inside a specialist (the device is shared).
2. Dispatch **one** agent per domain group. The brief carries: `functional-spec.md` and `technical-considerations.md` paths (it reads them — do not paste), its task list verbatim, `BRANCH`, "commit nothing", the zero-hardcoding rule, and the definition of done: every task implemented; **every test it writes proven red→green** (fails with the change set aside, passes with it, both outputs quoted); the domain's static check run (`go build ./... && go vet ./...` / `scripts/python/.venv/bin/python -m pytest -q scripts/python/tests` / `cd scripts/node && npm run build`); a terse report quoting command output.
3. **Order:** producers before consumers — Python analysis/codegen before the Go orchestrator that calls it; a renderer change before the comparison that scores it. Independent domains go in **one parallel batch**.
4. On return: spot-check (read a few named changes, re-run that domain's check), then tick that group's tasks `[x]` in `tasks.md` yourself, and tick a slice header once all its tasks are `[x]` — `/awos:verify` stops on anything unticked. A blocked task goes back to the **same** agent via `SendMessage`.
5. RED spot-check one test guarding the new behavior: set the implicated hunks aside (`git stash push -- <files>` is fine here — single-user repo), run that one test, watch it fail, `git stash pop`, watch it pass. Green-with-feature-removed → send back for retargeting. Skip under `<!-- skip-tests: true -->`.
6. Update `llms.txt`, `llms-full.txt` and `CLAUDE.md` per "Context Document Maintenance" whenever the change touched `internal/`, `scripts/python/`, `scripts/node/`, a mode, a flag, a synthesis parameter or the similarity metric — assign it to the specialist that made the change.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=verify -->

### Step 7: Verify

Run `/awos:verify` (main context) against the spec's acceptance criteria. **Mandatory, never skipped.** Then run the static gate in full: `go build ./... && go vet ./... && go test ./...`; `scripts/python/.venv/bin/python -m pytest -q scripts/python/tests`; `cd scripts/node && npm run build`. Bar: all green.

**Render gate — mandatory whenever the tech spec says Strudel output can change** (codegen, prompts, `synth_profiles.py`, calibrator, recorder, `compare_audio.py`, `eval/`): verifying by rendering is the flow's job, not the user's.

- Preflight: `pgrep -fl record-strudel-blackhole` empty; BlackHole present; Multi-Output device selected (a silent WAV means it is not — stop and say so, do not score silence). One render at a time, from the main repo.
- Produce the Strudel for the spec's reference track(s) (`./bin/midi-grep extract --url … --render auto`, or the script the spec names) and call the **`loop` MCP** `verify_strudel(strudel_path, original, genre, duration, recorder='blackhole')`; or, on an existing run, `eval_gate(comparison_json, genre)`. Pass = above the `eval/thresholds.yaml` floor for the genre with `worst_band_diff ≤ 0.30`. **The Node recorder never gates** — use it only for a quick smoke render.
- Record in the flow log: genre, overall and section-aware similarity, the `comparison.json` path, the render WAV path (renders are git-ignored, never committed). For a multi-iteration run, each iteration's numbers also go to `.sisyphus/evidence/task-NN-<slug>.txt`.
- A below-floor render is a gap: send the specialist the `comparison.json` band diffs and loop; never lower a threshold to pass.

**Web UI criteria** (only when `internal/server/` or its templates changed): start `./bin/midi-grep serve --port 8089` yourself (never touch a user's `:8080`), drive it with the Playwright MCP, screenshot to `docs/screenshots/{SPEC_NAME}-<state>.png` (untracked evidence — do not edit `.gitignore`), self-certify only objective states and ask the user about look-and-feel via `AskUserQuestion`. Stop what you started by PID.

Address gaps before proceeding — a criterion with no evidence is not verified, and the close-out says so.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=local-review -->

### Step 8: Local Review

Invoke Claude Code's built-in **`/code-review` skill at medium effort** from the main context on `BRANCH`'s diff against `origin/main`, pointing it at `context/spec/{SPEC_NAME}/` as ground truth. **Add no focus areas of your own** — the author framing the review is the bias. When the diff spans more than one of Go / Python / Node, or the user asks, also run the project's **`/self-review`** skill (4-agent audit). If `/code-review` is unavailable, dispatch one general-purpose reviewer with the diff range, the spec paths and the review-file contract.

Write the findings to `review/<slug>.md` (git-ignored working evidence). **Lead the presentation with that path on its own line** — `Review file: review/<slug>.md` — then the verdict and the counts by severity. Collect keep/drop via `AskUserQuestion`, send the accepted findings (the file path plus the user's decisions, not your summary) to the owning specialist via `SendMessage`, re-run the static gate, and re-run the render gate if the fix touched the Strudel path. Record verdict, counts and path in the flow log.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=commit-push -->

### Step 9: Commit & Push

Write the final flow-log entry (branch, commits, diffstat, similarity evidence paths, review path) — then freeze the log. Stage only what this flow produced: the specialists' code, tests, `context/spec/{SPEC_NAME}/`, `llms*.txt`/`CLAUDE.md` updates, `.sisyphus/plans|evidence|notepads` for this spec. Never `git add -A`; never `.env`, `.cache/`, `*.wav`, `review/`, `docs/screenshots/`; surface any unexpected changed file instead of staging it. Message: `feat(<area>): <summary>` with `spec NNN` (and `Closes #N` for an issue) in the body, ending with the attribution trailer the harness prescribes. No pre-commit hooks exist. `git push -u origin BRANCH`.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=remote-gates -->

### Step 10: Open the Pull Request

Before opening: `git fetch origin main && git merge-tree --write-tree origin/main HEAD`. Behind or conflicting → the branch is still private: `git rebase origin/main`, resolve (trivial inline; non-trivial may use one agent and is confirmed with the user), re-run the static gate, `git push --force-with-lease`. Should a `PreToolUse` branch-currency hook exist on this machine, it enforces the same check; `MIDIGREP_SKIP_BRANCH_CHECK=1` bypasses it only when the user says so.

`gh pr create --repo dygy/MIDI-grep --base main --head BRANCH --title "<type>(<area>): <summary>" --body …`. The body carries: the spec path; a **Testing coverage** section (static suite results; the render gate's genre, similarity numbers and `comparison.json` path, or "Strudel output unaffected — render gate not applicable"; web-UI states rendered, if any); the review verdict and that the local `/code-review` pass was the only automated review (CI is the remote gate — `.github/workflows/ci.yml`; no bot reviewer on this repo); `Closes #N` for an issue source.

**CI is the remote gate.** `.github/workflows/ci.yml` runs `go`, `node`, `python` and `hooks` jobs on the PR. Wait on it from an isolated subagent: `gh pr checks BRANCH --watch --fail-fast` (size the wait to ~10 min). Green → continue. Red → run the `gha-diagnosis` skill (it pulls the failing job's log via `gh run view`), fix locally, re-run the static gate, push, and wait once more. If the second run is still red, stop and show the user the failing job and log excerpt — do not iterate a third time. **Human review** is the repo owner's — present the PR URL and stop; do not poll a person. The flow log is frozen; progress from here is reported in the session.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=merge -->

### Step 11: Merge

Re-check mergeability (`gh pr view <n> --json mergeable,mergeStateStatus` plus the `merge-tree` check). If `main` moved and the branch no longer merges cleanly, **merge `origin/main` in** (never rebase an open PR), re-run the static gate, push, and update the PR.

**A human merges.** Before asking, show: the static gate green, the render gate verdict with its numbers (or "n/a" with the reason), the review file closed out (every finding kept or dropped, accepted ones fixed and pushed), and that the owner has looked at the PR. Then **one** `AskUserQuestion`: merge now (`gh pr merge <n> --repo dygy/MIDI-grep --squash --delete-branch`) / leave open for the owner to merge on GitHub. Unanswered = do not merge. An earlier "run it end to end" or "work autonomously" covers the work up to here — **it is never merge approval**. No post-merge CI exists.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=delivery -->

### Step 12: Deliver

Nothing deploys — MIDI-grep is a local CLI. Delivery is the merge into `main`; whoever pulls rebuilds with `go build -o bin/midi-grep ./cmd/midi-grep` (and `cd scripts/node && npm run build` if the renderer changed). No version bump, no tag. If the change altered a mode, flag, synthesis parameter or the metric, confirm `llms.txt` / `llms-full.txt` / `CLAUDE.md` landed in the merge — a context-doc update missing here is an outstanding step, not a nice-to-have.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=close-ticket -->

### Step 13: Close the Loop

Definition of Done: PR merged (or ready-to-merge, stated as such) **and** `functional-spec.md` + `technical-considerations.md` carry `Status: Completed` from `/awos:verify` **and** the context docs are updated where required **and** every similarity number in the spec, PR and docs traces to a real `comparison.json`. When `TICKET_ID` is a GitHub issue: on merge GitHub closes it through the PR body's `Closes #N`; if the PR is left open for the owner, comment the PR URL and the DoD status on the issue (`gh issue comment N --body …`) so the tracker shows where the work stands. Otherwise the report is the close.

Report to the user: PR link and merge commit (or "open, awaiting owner"); the render gate evidence (genre, overall / section-aware %, `comparison.json` path) or why it was not applicable; the review **verdict**, **finding counts by severity**, the **review file path** (`review/<slug>.md`, git-ignored — gone with a worktree teardown) and that a keep/drop gate ran; the unreachable sources from Step 1, read from the flow log; **agents dispatched this run (count)**; any flow defects found. Leave a clean tree: no closing flow-log entry (the log is frozen); surface any uncommitted flow artifact rather than leaving it. After a merge, `git switch main && git pull && git branch -d BRANCH`; offer `git worktree remove` for a `wt` worktree.

<!-- /awos:flow:stage -->

---

<!-- awos:flow:generated date=2026-10-09 version=2.4.5-hand source=context/product/delivery-flow.md -->
<!-- Hand-generated from the 2.4.5 templates (no /awos:flow generator ships). Edit directly; log changes in delivery-flow.md Generation Log. -->
