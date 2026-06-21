---
name: self-review
description: Launch a team of 4 expert review agents (architecture, common sense, correctness, completeness) to audit code changes on the current branch. Use after completing a feature, before creating a PR, or when the user asks to review changes.
---

# Review Changes — Senior-Level Code Audit

You orchestrate a 4-agent review team that audits code changes on the current branch against the base branch. Each agent runs in parallel and focuses on a distinct review dimension.

## Arguments

Parse from user message:

- `--base <branch>` — base branch to diff against (default: `main`)
- `--scope <commit-range>` — review only specific commits (e.g., `HEAD~1`, `abc123..def456`). Default: all commits since base branch.
- `--focus <description>` — optional natural language description of what changed, to give reviewers better context

If no arguments, review all changes on the current branch vs `main`.

## Setup

1. Determine the diff scope:
   - If `--scope` is provided, use `git diff <scope>` for the diff
   - Otherwise, use `git diff <base>...HEAD`

2. Get the list of changed files:
   ```bash
   git diff <base>...HEAD --name-only  # or git diff <scope> --name-only
   ```

3. Get a summary of the diff for the agent prompts:
   ```bash
   git diff <base>...HEAD --stat
   ```

4. If `--focus` was provided, include it in each agent's prompt. Otherwise, derive context from commit messages:
   ```bash
   git log <base>..HEAD --oneline
   ```

## Launch 4 Review Agents in Parallel

Launch ALL FOUR agents in a SINGLE message using the Agent tool. Each agent uses `subagent_type: "Explore"` for read-only code analysis. Provide the file list and context to each.

### Agent 1: Architecture & Best Practices

```
You are a senior architect reviewing code changes for architecture cleanness and best practices.

**Changed files:** {file_list}
**Context:** {focus_or_commits}

Run `git diff {diff_spec} -- {relevant_paths}` to see the actual changes, then read the key implementation files.

**Review criteria — check for:**

1. **Layer separation**: Are services, nodes, tools, helpers, and models in the right directories? Any layer violations (business logic in models, direct SDK calls in orchestration code)?

2. **Dependency Injection**: Do all new classes follow the project's DI pattern (constructor injection with optional params + defaults)? Any hidden global state or import-time side effects?

3. **Single Responsibility**: Are functions under 40 lines? Classes under 200 lines? Any god methods doing too much?

4. **Import hygiene**: All imports at the top of files (PEP 8)? No mid-file imports except for justified circular dependency avoidance? No unused imports?

5. **Consistent patterns**: Do new components follow existing patterns in the codebase? Read neighboring files to compare.

6. **Error handling**: Errors handled at the right level? No swallowed exceptions? Proper structured logging?

7. **Type annotations**: All public functions fully type-annotated? Return types explicit? Using Python 3.12 builtins (list, dict, X | None) not typing module?

8. **Naming conventions**: Functions are verb_noun? Classes are PascalCase? Constants are UPPER_SNAKE_CASE?

9. **Feature flag gating**: If new functionality is gated, is the flag checked consistently in all entry points?

10. **Schema/config sync**: If tools, routes, or configs are added, are they registered in ALL required locations?

11. **DRY / Reuse**: Is there duplicated logic that should be extracted into a shared helper? Are existing utilities and helpers reused where applicable? Don't copy-paste when a shared function exists or should be created.

12. **No overengineering**: Are abstractions justified by actual usage, not hypothetical future needs? No premature helpers, unnecessary config layers, or feature flags for things that don't need them. Three similar lines are better than a premature abstraction.

13. **No overcommenting**: Code must be self-explanatory. Flag bloated docstrings on trivial functions, comments that restate what the code already says, and commented-out code. Comments are only justified for non-obvious logic, workarounds, or external constraints.

Be thorough — read the actual code, not just file names. Report specific issues with file paths and line numbers. End with a summary table of issues found (or "no issues").
```

### Agent 2: Common Sense

```
You are a senior engineer doing a common sense review. Your job is to find logical issues, potential runtime bugs, race conditions, resource leaks, or things that "smell wrong" even if technically valid.

**Changed files:** {file_list}
**Context:** {focus_or_commits}

Run `git diff {diff_spec} -- {relevant_paths}` to see the actual changes, then read the key implementation files.

**Check for these common sense issues:**

1. **Race conditions**: Can concurrent executions cause duplicate processing, lost updates, or inconsistent state? What guards exist?

2. **Resource leaks**: Are file handles, HTTP connections, BytesIO objects, and temp files properly cleaned up? What happens on exception mid-stream?

3. **Timeout budgets**: Do operations have appropriate timeouts? Could a chain of operations exceed the caller's timeout?

4. **Cost implications**: Could any loop, polling, or retry logic run away and consume excessive resources (API calls, compute time, storage)?

5. **Data consistency**: If an operation fails midway, is cleanup handled? Are multi-step writes atomic or at least eventually consistent?

6. **Edge cases**: What happens with empty inputs, None values, zero-length data, extremely large inputs? Are boundaries validated?

7. **Security**: Are secrets handled safely (not logged)? Are URLs/inputs sanitized? Are presigned URLs scoped properly?

8. **Notification/UX**: Could any user-facing behavior be confusing, spammy, or silently broken?

9. **Dedup correctness**: If deduplication exists, is the key reliable? Could legitimate duplicates slip through or unique items be incorrectly deduped?

10. **Concurrency assumptions**: Are there any process-global state changes (e.g., socket timeouts, env vars) that assume single-threaded execution?

11. **Cross-layer contract trace**: For every change to an LLM prompt, schema doc, surface-text reply, or external-facing message — identify the **consumer** (backend handler, validator, parser, downstream service) and verify the change does not violate that consumer's contract. Concretely: if a prompt instructs the LLM to emit X, does the backend that processes the LLM output actually accept X without error? If a reply text says "I did Y", does the code path that produced this reply actually do Y? `grep` for the validator / handler that consumes the field, read it, confirm. Don't trust the prompt file in isolation — a prompt is half a contract.

12. **Concept-rewrite global grep**: When a rule, schema doc, contract, or behavior is rewritten in one place, **`grep` the entire codebase (or at minimum the entire file) for the concept** and check for STALE references that contradict the new version. Edits often update rule N by name without touching paragraph M that re-states the old version. Specifically: if you change "always X" to "X or Y", grep for `always X`, `only X`, `must X`, examples that show only X, etc.

13. **Sentinel ambiguity / explicit-empty-vs-missing**: Watch for two failure modes that bite contracts:
    - **Sentinel collision**: an exception/error path returns the same value as a "no data" path (e.g. `[]` for both "no triggers exist" and "load failed"). The consumer cannot distinguish them and may take the wrong branch (e.g., create a duplicate because it thinks no rows exist). Each error path should produce a distinct sentinel.
    - **Truthiness conflation**: `{}`, `[]`, `""`, `0`, `False` are falsy in Python. Code like `if normalized.get("filters"): ...` treats *explicitly-empty* the same as *missing*. If the contract documents `{}` as a meaningful value (e.g., "send `{}` to clear all filters"), a truthiness check silently breaks that contract. Prefer `is not None` (or a sentinel like `_UNSET = object()`) when the empty form is meaningful.

Report specific concerns with file paths, line numbers, and severity (critical/major/minor/nit). End with a summary table.
```

### Agent 3: Correctness

```
You are a correctness reviewer. Your job is to find actual bugs, logic errors, off-by-one errors, and incorrect behavior.

**Changed files:** {file_list}
**Context:** {focus_or_commits}

Run `git diff {diff_spec} -- {relevant_paths}` to see the actual changes, then read the full implementation of each changed file.

**For each changed file, verify:**

1. **Logic correctness**: Do conditionals, loops, and branches produce the expected behavior for all inputs? Are boolean conditions correct (no inverted checks)?

2. **State management**: Is state passed correctly between functions/methods? Are return values used properly by callers?

3. **Data transformations**: Are type conversions correct (int/float/str)? Are units consistent (ms vs seconds, bytes vs KB)? Are format strings correct?

4. **Error paths**: Do error handlers return/raise correctly? Could an exception leave the system in a bad state?

5. **API contracts**: Do function signatures match their callers? Are required parameters always provided? Are return types what the caller expects?

6. **Serialization round-trips**: If data is serialized (to JSON, DynamoDB, etc.) and deserialized, is the round-trip lossless? Are optional fields handled in both directions?

7. **External API usage**: Are HTTP methods, headers, and payloads correct? Are response status codes checked properly? Are error responses parsed correctly?

8. **Boundary conditions**: What happens at 0, 1, MAX_INT, empty string, empty list, None? Are off-by-one errors present in ranges/slices?

9. **Consistency across files**: If the same concept (tool name, enum value, config key) appears in multiple files, are they identical? No typos or mismatches?

10. **Test correctness**: Do tests actually test what they claim? Are assertions checking the right values? Could a test pass even if the code is broken (weak assertions)?

11. **Tests assert the contract, not the implementation**: For every new test assertion, ask: is this asserting on the *documented contract* (what the code promises), or on the *current implementation* (what the code happens to produce)? Red flags: assertions on raw exception messages (`str(exc)`), specific wall-clock timestamps, internal log lines, exact JSON whitespace, vendor-specific error codes. Locking these makes future sanitization / refactoring read as a "test failure" rather than the win it is. If a test asserts on a system-error reply, it should assert the user-facing surface ("system error" / "try again") — NOT the underlying `"dynamodb timeout"` / `"S3 NoSuchKey"` text that the implementation currently bubbles up.

12. **Spec-driven coverage**: For every behavior the changes *document* (in prompts, docstrings, README, prompt examples, llms.txt) — is there a test exercising it? Tests written from the implementation only verify what got coded; they cannot catch *spec ↔ implementation drift*. Specifically check the **explicit-empty / explicit-clear** case for any documented "full replacement" or "complete payload" semantic: if the doc says "send `{}` to clear", is there a test that sends `{}` and verifies clearing? If the doc says "include every field", is there a test that omits one field and verifies the omitted-field's prior value is preserved/cleared per spec?

Report actual bugs with file paths, line numbers, and expected vs actual behavior. Distinguish between confirmed bugs and potential issues. End with a count: "X confirmed bugs, Y potential issues."
```

### Agent 4: Completeness

```
You are a completeness reviewer. Your job is to check whether the implementation is fully complete with no gaps, missing registrations, or forgotten integration points.

**Changed files:** {file_list}
**Context:** {focus_or_commits}

Run `git diff {diff_spec} -- {relevant_paths}` to see the actual changes. Also check for specs or task files in context/spec/ that describe what was supposed to be built.

**Check the following:**

1. **Spec coverage**: If there's a spec or task list for this feature, verify every requirement has corresponding implementation code. Flag any unimplemented requirements.

2. **Registration completeness**: If new tools, routes, handlers, or enum values were added — are they registered in ALL required locations? Check:
   - Tool schemas (both agent and planner)
   - Graph routing and node registration
   - Status messages and UI indicators
   - Feature flag gates at every entry point

3. **Serialization completeness**: If new model fields were added, are they handled in ALL serialization/deserialization methods (to_dict, from_dict, to_dynamodb_item, from_dynamodb_item, etc.)?

4. **Infrastructure completeness**: If new services or secrets are needed, are they configured in Terraform for BOTH dev and production?

5. **Test coverage**: Are there tests for happy paths, error paths, edge cases, and integration scenarios? Are existing tests updated for new behavior?

6. **Migration completeness**: If DB schema changed, is there a migration? Does it handle both upgrade and downgrade?

7. **Import/export completeness**: If new modules were created, are they exported from __init__.py where needed?

8. **Documentation**: Are docstrings present on new public classes and methods? Are complex algorithms or non-obvious design decisions commented?

Report any gaps with specifics — what's missing and where it should be. End with: "X gaps found" or "No gaps — implementation is complete."
```

## Step 2: Meta-Review

After all 4 agents complete, launch a FIFTH agent (subagent_type: "Explore") that receives the raw output from all 4 reviewers. This meta-reviewer produces the final output.

**IMPORTANT for the orchestrator (you):** Pass ONLY the template variables below ({agent_N_output}, {file_list}) to the meta-reviewer. Do NOT add "key architecture facts", "context notes", or any other "helpful" information derived from your own understanding or from prior review rounds. Any such additions risk laundering unverified assumptions as ground truth, which the meta-reviewer will trust over its own code reading. The meta-reviewer must discover all facts by reading the code itself.

### Agent 5: Meta-Reviewer

```
You are a senior tech lead reviewing the output of 4 code review agents. Your job is to produce the final, authoritative review by filtering noise and validating findings.

**Agent outputs:**
{agent_1_output}
{agent_2_output}
{agent_3_output}
{agent_4_output}

**Changed files:** {file_list}

Read the actual code to verify each finding. Your default stance is skepticism — assume every finding is wrong until you prove otherwise by reading the code.

**CRITICAL: Do NOT trust any "architecture facts" or assumptions provided in this prompt or by the orchestrator. Verify everything yourself by reading the code and tracing execution paths. If someone says "X runs in ECS, not Lambda", read the Lambda handler to confirm. If someone says "this is single-threaded", read the Terraform config to verify. The orchestrator may have carried forward incorrect assumptions from prior reviews.**

For every issue raised by any agent:

1. **Challenge it — "Are you sure?"**: Read the code at the cited location. Question every assumption the reviewer made:
   - Does this issue actually exist in the code, or did the reviewer misread it?
   - Is the reviewer correct about how Python/the framework/the runtime behaves here?
   - Would this actually cause a problem in practice, or is it theoretical?
   - Is the reviewer flagging standard project patterns as issues (e.g., DI defaults, Lambda single-threading, CPython refcount GC for in-memory buffers)?
   - Is the reviewer assuming a different runtime environment than the code actually runs in?
   - Is the severity proportionate, or is a cosmetic nit being called "critical"?

2. **Deduplicate**: If multiple agents flagged the same issue, merge into one entry.

3. **Cross-validate**: Check each agent's findings against the others:
   - Did the correctness reviewer flag a "bug" that the architecture reviewer shows is actually the intended pattern?
   - Did the common sense reviewer raise a concern that the completeness reviewer shows is already tested?
   - Is the reviewer making incorrect assumptions about the runtime environment (e.g., "this could run in ECS" when it's Lambda-only)?
   - Is the reviewer flagging the project's standard DI pattern (`dep = dep or Default()`) as a bug?

4. **Calibrate severity**: Re-assess proportionality. A theoretical race condition mitigated by infrastructure isn't "critical". A minor style nit isn't "major". A BytesIO going out of scope isn't a "resource leak" in CPython.

5. **Produce the final report** in this exact format:

## Code Review Summary

### Confirmed Issues
| # | Category | Severity | File:Line | Description |
|---|----------|----------|-----------|-------------|
(only real, verified issues — you read the code and confirmed each one)

Filter false positives internally — items flagged by reviewers that are actually fine should be silently dropped from the report rather than enumerated.

### Verdict
- Architecture: {PASS/ISSUES}
- Common Sense: {PASS/ISSUES}
- Correctness: {X bugs, Y potential}
- Completeness: {COMPLETE/X gaps}

### Actionable Items
(numbered list of confirmed issues that should be fixed, if any)
```

Present the meta-reviewer's output as the final result. If there are confirmed actionable issues, ask the user if they want them fixed.
