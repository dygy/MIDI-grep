"""Job/task runner for decomposed LLM generation.

Instead of one giant prompt, generation is broken into small jobs. Each job:
  - builds a tight, single-concern prompt (gets the context + already-completed deps),
  - calls the LLM, parses JSON,
  - validates the parsed output against a contract,
  - retries (bounded) with the validation error fed back into the prompt,
  - falls back to a deterministic default (LOGGED, never silent) if retries are exhausted.

Jobs run sequentially in dependency (topological) order — there is one ~13GB Ollama model on a
24GB machine, so parallelism isn't available anyway. The runner is LLM-agnostic: pass any
`llm_call(prompt) -> str`. It never touches Strudel syntax — jobs produce/validate JSON only;
assembly into Strudel is a separate deterministic step.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

# Typed contracts for the LLM job I/O (documents the AI boundary).
# NOTE: project runs on Python 3.11, so plain aliases (not the 3.12 `type` statement).
Context = dict[str, object]      # task context passed to every job
JobOutput = dict[str, object]    # the validated JSON a job produces


class LLMCall(Protocol):
    """Anything that turns a prompt into a completion string (Ollama, Claude, a mock)."""
    def __call__(self, prompt: str) -> str: ...


class JobStatus(StrEnum):
    OK = "ok"              # validated output on some attempt
    FALLBACK = "fallback"  # retries exhausted → deterministic default (logged)
    FAILED = "failed"      # reserved for unrecoverable errors


@dataclass(slots=True)
class JobSpec:
    id: str
    build_prompt: Callable[[Context, dict[str, JobOutput]], str]  # (context, dep_outputs) -> prompt
    validate: Callable[[JobOutput, Context], list[str]]           # (parsed, context) -> [errors]
    fallback: Callable[[Context], JobOutput]                      # (context) -> default output
    deps: list[str] = field(default_factory=list)
    max_retries: int = 2


@dataclass(slots=True)
class JobResult:
    id: str
    status: JobStatus
    output: JobOutput
    attempts: int
    errors: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} object out of an LLM response."""
    if not text:
        return None
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _topo_order(jobs: list[JobSpec]) -> list[JobSpec]:
    """Return jobs in dependency order (raises on cycle / missing dep)."""
    by_id = {j.id: j for j in jobs}
    ordered, visiting, done = [], set(), set()

    def visit(jid: str):
        if jid in done:
            return
        if jid in visiting:
            raise ValueError(f"dependency cycle at job '{jid}'")
        if jid not in by_id:
            raise ValueError(f"unknown dependency '{jid}'")
        visiting.add(jid)
        for dep in by_id[jid].deps:
            visit(dep)
        visiting.discard(jid)
        done.add(jid)
        ordered.append(by_id[jid])

    for j in jobs:
        visit(j.id)
    return ordered


class JobRunner:
    def __init__(self, llm_call: LLMCall, log: Callable[[str], None] = print) -> None:
        self.llm_call = llm_call
        self.log = log
        self.results: dict[str, JobResult] = {}

    def run(self, jobs: list[JobSpec], context: Context) -> dict[str, JobResult]:
        """Execute jobs in dependency order. Returns {job_id: JobResult}."""
        self.results = {}
        for job in _topo_order(jobs):
            dep_outputs = {d: self.results[d].output for d in job.deps if d in self.results}
            self.results[job.id] = self._run_one(job, context, dep_outputs)
        return self.results

    def _run_one(self, job: JobSpec, context: Context, dep_outputs: dict[str, JobOutput]) -> JobResult:
        errors: list[str] = []
        last_error = ""
        for attempt in range(1, job.max_retries + 2):  # initial try + retries
            prompt = job.build_prompt(context, dep_outputs)
            if last_error:
                prompt += (
                    f"\n\nYour previous output was REJECTED: {last_error}. "
                    f"Fix this and return ONLY valid JSON."
                )
            raw = self.llm_call(prompt) or ""
            parsed = _extract_json(raw)
            if parsed is None:
                last_error = "output was not valid JSON"
                errors.append(f"attempt {attempt}: {last_error}")
                self.log(f"  [job:{job.id}] attempt {attempt} — no JSON")
                continue
            verrs = job.validate(parsed, context)
            if verrs:
                last_error = "; ".join(verrs)
                errors.append(f"attempt {attempt}: {last_error}")
                self.log(f"  [job:{job.id}] attempt {attempt} — invalid: {last_error}")
                continue
            self.log(f"  [job:{job.id}] ok (attempt {attempt})")
            return JobResult(job.id, JobStatus.OK, parsed, attempt, errors)

        # Retries exhausted — deterministic, LOGGED fallback (never silent).
        fb = job.fallback(context)
        self.log(f"  [job:{job.id}] FALLBACK after {job.max_retries + 1} attempts ({last_error})")
        return JobResult(job.id, JobStatus.FALLBACK, fb, job.max_retries + 1, errors)

    def manifest(self) -> dict:
        """Serializable summary for job_run.json (observability)."""
        return {
            "jobs": [
                {"id": r.id, "status": r.status.value, "attempts": r.attempts, "errors": r.errors}
                for r in self.results.values()
            ],
            "fallbacks": [r.id for r in self.results.values() if r.status is JobStatus.FALLBACK],
            "all_ok": all(r.status is JobStatus.OK for r in self.results.values()),
        }
