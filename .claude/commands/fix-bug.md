---
description: Fixes one MIDI-grep bug end-to-end — reproduces it (a description, a GitHub issue, or a failing comparison.json), finds the root cause, applies a scoped fix with a RED-proven regression test, re-verifies the touched criteria (re-rendering through BlackHole when Strudel output is involved), amends the owning spec on divergence, reviews locally, and opens the PR on dygy/MIDI-grep.
argument-hint: '[bug — GitHub issue # or URL, a description, or a path to a failing comparison.json]'
---

# Fix a Bug End-to-End

Takes one bug — a plain description, a GitHub issue on `dygy/MIDI-grep`, or a **failing render** (a `comparison.json` whose eval gate breached) — and drives it through diagnosis → classification → scoped fix + regression test → scoped re-verification → spec amendment on divergence → local review → pull request to `main` until it is closed. **A bug never creates a new spec.** Run from a session anchored at the repo root.

This command is self-contained and derived from `context/product/delivery-flow.md`, the decision record. **Do not read that file during a run.** There is no `/awos:flow` to regenerate this command: to change a decision, edit the record and then this file directly, and log the change in the record's Generation Log.

## Arguments

`$ARGUMENTS` — a GitHub issue number or URL, a free-text description, or a path to a `comparison.json` (typically `.cache/stems/<key>/vNNN/comparison.json`) whose gate failed. If empty, ask.

## Run Discipline

**Agent budget (CLAUDE.md Working Agreement).** Subagents start cold and return claims you must re-check. Rules:

- **Work inline by default** — fetch, preflight, reproduction, classification, `gh`/`git`, `loop` MCP calls, the static gate, and reading the few files a diagnosis points at.
- **Subagents only for:** (1) the code change + regression test — **one specialist per affected domain, one dispatch** carrying both; (2) diagnosis that needs deep reading across files you have not opened (`Explore`), at most one per domain, independent ones in one parallel batch; (3) a non-trivial merge conflict.
- **Follow-ups reuse the agent:** retargeting a vacuous test and accepted review findings go back to the same specialist via `SendMessage`.
- **Counting:** `/code-review`'s internal reviewers and `SendMessage` follow-ups are not new dispatches. **Expected: 1–3 agents per bug. Past 6, stop and tell the user why.**
- Briefs say: read `.awos/subagents/<name>.md` first; tools work, no exploratory calls; terse reports **quoting command output**. Reports are claims — re-read named lines, re-run named tests.
- The orchestrator does not edit product code itself. Never launch `claude -p`.

**Questions.** At most **one round** of clarifying questions (CLAUDE.md); otherwise state the assumption and proceed. Fixed choices go through `AskUserQuestion` with the default marked. Unanswered → safe default once, said aloud; never re-ask in a loop; silence is never consent to merge or to amend a spec.

**No pre-existing issues; zero hardcoding.** Anything broken that the fix trips over is fixed in this run. The fix never pins a gain, filter or threshold to the failing track — if the root cause is a bad parameter, the fix is in how it is *derived* (analysis, calibrator, prompt), and a threshold in `eval/thresholds.yaml` is never lowered to make a gate pass.

**Flow log.** One short entry per completed stage (stage, outputs, verdicts, evidence paths, next) written to **this session's scratchpad** as `fix-log-{BUG_ID}.md` — a working file. It moves to `context/fix-log-{BUG_ID}.md` and is committed **only when the run already changes `context/`** (a spec amendment, Step 9); a fix that amends no spec carries its record in the PR description instead. Frozen once the PR exists; resume then relies on `gh pr view`.

**Flow defects.** A wrong fact here → follow reality, note it, list it in the Step 15 report. **Do not edit this command, `delivery-flow.md` or skills during the run** — flow fixes go in their own `docs/<slug>` branch afterwards.

<!-- awos:flow:stage=fetch-bug -->

### Step 1: Fetch & Normalize the Bug (inline)

Pre-flight in one Bash call: `gh auth status`; `scripts/python/.venv/bin/python -c "import librosa, yaml"`; `ls scripts/node/node_modules scripts/node/dist >/dev/null`; `system_profiler SPAudioDataType | grep -c BlackHole`; `curl -s -m 2 localhost:11434/api/tags | head -c 60`. Report gaps now: no `gh` login blocks the PR stage; no BlackHole blocks the re-render in Step 8 for any Strudel-path bug.

Normalize by source:

- **GitHub issue:** `gh issue view N --repo dygy/MIDI-grep --json number,title,body,labels,comments,url,state`. `BUG_ID = N`; keep title, symptom, repro steps (often in comments), affected area, URL, state. Read linked issues/PRs; fetch external links with `WebFetch` best-effort and list the unreachable ones in the normalized report.
- **Description:** normalize; `BUG_ID` = short kebab slug (≤ 5 words).
- **Failing `comparison.json`:** read it — `overall_similarity`, `section_aware_similarity`, `band_differences`, `worst_band_diff`, tempo — and the sibling `metadata.json` / `synth_config.json` for genre, BPM and the source URL. Run `eval_gate(comparison_json, genre)` via the **`loop` MCP** to confirm the breach; the gate's message is the problem statement, the breached metric is the symptom. `BUG_ID` = `<cache-key>-vNNN-<metric>`. Keep the `.strudel` and render paths in the same dir — they are the reproduction.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=resume-detection -->

### Step 2: Detect the Entry Point

Stop only if the GitHub issue is `CLOSED` or a merged PR carries this `BUG_ID`/slug (`gh pr list --repo dygy/MIDI-grep --state merged --search "<slug>"`). An open PR is a resume signal — continue at merge. If the scratchpad holds `fix-log-{BUG_ID}.md` (or `context/fix-log-{BUG_ID}.md` from an amendment run), read it and resume after its last completed stage. Never repeat a completed stage.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=workspace -->

### Step 3: Prepare the Workspace

`git rev-parse --show-toplevel` is the root; `context/product/product-definition.md` exists there. `git status --short` — warn on a dirty tree (uncommitted `context/product/delivery-flow.md` or `.claude/commands/*.md` are expected, not blockers; anything else stays unstaged). `git check-ignore review/` — not ignored → tell the user to add `review/` to `.gitignore` (one-time project task; do not edit it yourself).

Default is the main repo. Offer a worktree (`AskUserQuestion`, default main repo) only when the main repo is mid-way through other uncommitted work; then invoke the **`wt` skill** (`/wt fix-<slug>`), bring up `go build -o bin/midi-grep ./cmd/midi-grep` and `cd scripts/node && npm install && npm run build`, reuse the main repo's `scripts/python/.venv`, and remember `.cache/stems/` is empty there — a render-reproduced bug is re-rendered **from the main repo**, where the stems are, one recorder at a time.

Main repo: `git fetch origin main && git switch -c fix/<slug> origin/main`. Store `BRANCH`.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=diagnose -->

### Step 4: Diagnose

Reproduce first, inline: a CLI bug → run the named `./bin/midi-grep …` command or the Python script with the same inputs; a render bug → `eval_gate` on the `comparison.json`, then read the `.strudel` and the band diffs; a test bug → run that one test. **Not reproducible → report and stop**; do not guess at a fix.

Follow the symptom to the code inline; dispatch an `Explore` agent only under the budget rule, for a domain you have not opened.

- **Enumerate every surface** that produces or consumes the broken data — the same metric computed in `compare_audio.py` *and* read by `eval/gate.py` *and* summarized by `generate_report.py` / `internal/report/generator.go`; a parameter written by `analyze_synth_params.py` *and* consumed by the renderer *and* by `calibrate_dynamic.py`; a prompt rule stated in `ollama_codegen.py` *and* in `Modelfile.mistral` *and* validated by `ollama_agent.py`. Return a verdict per surface, affected or clean — fixing only the named one comes back as "reopened".
- **Check the Go↔Python seam.** A wrong value in Go output often originates in a Python script's JSON (`metadata.json`, `synth_config.json`, `comparison.json`); read the producer before concluding the fix is Go-side, and vice-versa.
- **Recorder bugs are tempo-first.** A similarity regression with `tempo_sim < 1.0` or a BPM read ~25% off is the known avfoundation timestamp failure (`-use_wallclock_as_timestamps 1` + `aresample=async=1`) — check the ffmpeg invocation before touching codegen.
- Label every claim **verified** (line read / repro executed) or **hypothesis**. Re-read cited lines before accepting a fix shape and trim it to what the code shows.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=classify -->

### Step 5: Classify (inline)

Find the owning spec under `context/spec/NNN-*/functional-spec.md` (001 core pipeline, 002 ML customization, 003 editable Strudel generation, plus any later ones) and read the criteria the bug touches.

- **Conformance** — code violates a correct criterion → fix + regression test, no amendment.
- **Divergence** — the criterion was wrong or incomplete, or the fix intentionally changes documented behavior → fix + test + amend (Step 9).
- **No owning spec** — much of today's pipeline predates the specs' acceptance criteria → record "none"; never fabricate or create one.

Log the verdict and the spec dir (or "none").

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=fix -->

### Step 6: Fix + Regression Test — one dispatch per domain

Pick the specialist from the affected files: `internal/`, `cmd/` → `golang-expert`; `ollama_*`, prompts, `Modelfile*` → `llm-expert` (Strudel content → `strudel-expert`); `compare_audio.py`, `eval/`, `separate.py`, `analyze*.py`, `calibrate_dynamic.py` → `ml-audio-expert`; renderers, `record-strudel-blackhole.ts`, synthesis → `audio-dsp-expert`; other `scripts/python/` → `python-expert`; harmony/arrangement logic → `music-theory-expert`. Dispatch **once**, covering both the fix and the regression test. Brief: root cause and surfaces from Step 4; the classification; `BRANCH`; "commit nothing"; scope = the root cause plus every confirmed surface, no refactors; the zero-hardcoding rule; one test that fails on the old code and passes on the fix — pytest under `scripts/python/tests/` or a Go `_test.go` beside the package (skip under `<!-- skip-tests: true -->` in the owning spec's `tasks.md`); run the domain's static check; terse report **quoting command outputs**. Cross-domain: producer first (Python before the Go that reads its JSON), unrelated domains in one parallel batch.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=regression-test -->

### Step 7: Prove the Regression Test (inline)

Skip under `<!-- skip-tests: true -->` — the evidence is then Step 8's real render. Otherwise demonstrate fail→pass yourself: `git stash push -- <fixed files>` (single-user repo; or `git diff > /tmp/fix.patch && git checkout -- <files>`), run exactly that test (`scripts/python/.venv/bin/python -m pytest -q scripts/python/tests/test_x.py::test_y` or `go test ./internal/<pkg> -run TestY`), watch it fail, restore, watch it pass; log both outputs. Green on old code = vacuous → `SendMessage` the **same** specialist to retarget the changed lines. For a similarity regression the test asserts the metric/threshold logic (style of `test_similarity_gate.py`); the render itself is Step 8.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=verify-criteria -->

### Step 8: Verify the Touched Criteria

Re-check **only** the acceptance criteria the bug touched, with `/awos:verify`'s evidence discipline — do not flip the spec's Status. No owning spec → the evidence is the fail→pass test plus a real run of the fixed behavior.

Run the full static gate: `go build ./... && go vet ./... && go test ./...`; `scripts/python/.venv/bin/python -m pytest -q scripts/python/tests`; `cd scripts/node && npm run build`. Bar: all green.

**Render re-verification — mandatory when the fix touches anything that can alter Strudel output** (codegen, prompts, `synth_profiles.py`, calibrator, recorder, `compare_audio.py`, `eval/`): verifying by rendering is the flow's job, not the user's.

- Preflight: `pgrep -fl record-strudel-blackhole` empty; BlackHole present; Multi-Output device selected (a silent WAV means it is not — stop, do not score silence). One render at a time, from the main repo.
- Re-generate the Strudel for the failing track (or the spec's reference track) and call the **`loop` MCP** `verify_strudel(strudel_path, original, genre, duration, recorder='blackhole')`. Pass = above the `eval/thresholds.yaml` floor with `worst_band_diff ≤ 0.30`. **The Node recorder never gates.** For a bug reported as a failing `comparison.json`, the new render must clear the gate that the old one breached — quote both numbers.
- Log genre, overall / section-aware %, `comparison.json` and render paths (renders are git-ignored).

Scale the evidence to what changed: a fix provably outside the Strudel path (Go CLI plumbing, report HTML, cache keys, docs) needs the fail→pass test plus a real run of the fixed command, not a render — say so in the log. A fix in `internal/server/` is driven for real on `./bin/midi-grep serve --port 8089` with the Playwright MCP, screenshots to `docs/screenshots/fix-<slug>-<state>.png` (untracked evidence), look-and-feel confirmed by the user via `AskUserQuestion`; stop what you started by PID.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=amend-spec -->

### Step 9: Amend the Spec (divergence only)

Conformance or no owning spec → skip. Divergence → `AskUserQuestion` amend / leave as pending divergence (unanswered = don't amend). Then `/awos:spec` in update mode for the owning spec — `/awos:spec amend spec NNN: <what changed and why>` — which edits the affected criteria in place, appends a dated `## Change Log` entry, allocates no new index and leaves a `Completed` Status untouched. Move the flow log to `context/fix-log-{BUG_ID}.md` now, since `context/` is changing anyway. If the fix revealed drift in `context/product/product-definition.md` or `architecture.md`, suggest `/awos:product` / `/awos:architecture` — never auto-edit.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=local-review -->

### Step 10: Local Review

Invoke Claude Code's built-in **`/code-review` skill at medium effort** from the main context on `BRANCH`'s diff against `origin/main`, with the owning spec (if any) as ground truth and **no author-added focus areas**. If unavailable, dispatch one general-purpose reviewer with the diff range and the review-file contract. Write the findings to `review/fix-<slug>.md` (git-ignored). Lead with `Review file: review/fix-<slug>.md` on its own line, then verdict and counts by severity; collect keep/drop via `AskUserQuestion`; send accepted findings (path + decisions, not your summary) to the same specialist via `SendMessage`; re-run the static gate and, if the Strudel path moved, the render. Log verdict, counts, path.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=commit-push -->

### Step 11: Commit & Push

Write the final flow-log entry, then freeze it. Stage only flow output — the fix, the regression test, `context/spec/NNN-*/functional-spec.md` and `context/fix-log-{BUG_ID}.md` when Step 9 ran, `llms*.txt`/`CLAUDE.md` if the fix changed a flag, parameter or metric (CLAUDE.md "Context Document Maintenance"). Never `git add -A`; never `.env`, `.cache/`, `*.wav`, `review/`, `docs/screenshots/`. Message: `fix(<area>): <summary>` with the root cause in one line and `#N` for an issue in the body, ending with the attribution trailer the harness prescribes. No pre-commit hooks exist. `git push -u origin BRANCH`.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=remote-gates -->

### Step 12: Open the Pull Request

Before opening: `git fetch origin main && git merge-tree --write-tree origin/main HEAD`. Behind or conflicting → the branch is still private: `git rebase origin/main`, resolve (trivial inline; non-trivial may use one agent, confirmed with the user), re-run the static gate, `git push --force-with-lease`. A `PreToolUse` branch-currency hook, if present, enforces the same; `MIDIGREP_SKIP_BRANCH_CHECK=1` bypasses it only on the user's say-so.

`gh pr create --repo dygy/MIDI-grep --base main --head BRANCH --title "fix(<area>): <summary>" --body …`. Body: symptom → root cause → change, every surface touched; a **Testing coverage** section — the regression test and its fail→pass outputs, static suite results, the render re-verification numbers with `comparison.json` path (or "Strudel output unaffected"); the classification verdict and, on divergence, the amended spec; the review verdict and that the local `/code-review` pass was the only automated review (CI is the remote gate; no bot reviewer on this repo); `Closes #N` for an issue source.

**CI is the remote gate** (`.github/workflows/ci.yml`: go / node / python / hooks). From an isolated subagent run `gh pr checks BRANCH --watch --fail-fast` (~10 min budget). Red → `gha-diagnosis` skill, fix, re-run the static gate, push, wait once more; a second red stops the flow and shows the user the failing job. **Human review** is the owner's — present the URL and stop; do not poll a person.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=merge -->

### Step 13: Merge

Re-check mergeability (`gh pr view <n> --json mergeable,mergeStateStatus` + the `merge-tree` check); if `main` moved, **merge `origin/main` in** (never rebase an open PR), re-run the static gate, push.

**A human merges.** Show: static gate green, the fail→pass evidence, the render verdict or its "n/a" reason, the review file closed out, and that the owner has looked at the PR. Then **one** `AskUserQuestion`: merge now (`gh pr merge <n> --repo dygy/MIDI-grep --squash --delete-branch`) / leave open for the owner. Unanswered = do not merge; "work autonomously" is never merge approval. No post-merge CI exists.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=delivery -->

### Step 14: Deliver

Nothing deploys — the fix is delivered by the merge into `main`; whoever pulls rebuilds (`go build -o bin/midi-grep ./cmd/midi-grep`, `cd scripts/node && npm run build` if the renderer changed). No version bump, no tag. A fix that invalidated cached renders (recorder or metric change) says so in the PR: numbers measured before it are stale, as the Jun-2026 recorder tempo fix already taught.

<!-- /awos:flow:stage -->

<!-- awos:flow:stage=close-ticket -->

### Step 15: Close the Bug

A GitHub issue closes via `Closes #N` on merge; for a description or `comparison.json` source the report is the close.

Report: PR link and merge commit (or "open, awaiting owner"); classification verdict and, on divergence, the amended criteria and Change Log entry; the regression evidence (what was set aside, the failure output, the restored pass); the render re-verification numbers and paths, or why not applicable; review **verdict**, **finding counts by severity**, **review file path** (`review/fix-<slug>.md`, git-ignored) and that a keep/drop gate ran; the flow-log location (scratchpad unless an amendment moved it into `context/` — say so, it does not survive the session otherwise); unreachable sources from Step 1; **agents dispatched this run (count)**; flow defects found. Leave a clean tree — no closing flow-log entry; surface any uncommitted flow artifact. After a merge: `git switch main && git pull && git branch -d BRANCH`; offer `git worktree remove` for a `wt` worktree.

<!-- /awos:flow:stage -->

---

<!-- awos:flow:generated date=2026-10-09 version=2.4.5-hand source=context/product/delivery-flow.md -->
<!-- Hand-generated from the 2.4.5 templates (no /awos:flow generator ships). Edit directly; log changes in delivery-flow.md Generation Log. -->
