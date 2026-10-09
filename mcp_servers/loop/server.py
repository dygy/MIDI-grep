"""MIDI-grep verification loop MCP server.

Closes the check→render→compare→gate loop so an agent can verify a Strudel change end-to-end
without the human running the CLI by hand, specialised for MIDI-grep's chain:

    Strudel code --editability--> render (BlackHole) --compare--> similarity --gate--> pass/fail

Tools:
  - verify_strudel    : editability check + render + compare + gate in one call (the loop-closer)
  - render_strudel    : render a .strudel file to WAV via the BlackHole recorder
  - compare_render    : compare a rendered WAV to an original, return similarity + gate verdict
  - eval_gate         : run the eval gate on an existing comparison.json

Spec 003 (context/spec/003-editable-strudel-generation): similarity is only computed on output
that passes the replay detector (`scripts/python/editability_check.py`). `verify_strudel` runs
it FIRST and returns `{"ok": false, "stage": "editability", ...}` with the violations on a fail;
`compare_render` forwards `strudel_path` to `compare_audio.py --strudel` so the comparison.json
carries `generation_mode` + `editability`; `compare_render` / `eval_gate` accept `mode` for the
per-mode floors in `eval/thresholds.yaml`.

The only recorder is BlackHole (`scripts/node/dist/record-strudel-blackhole.js`) — the old Node
offline synth emulation was deleted (commit f18f5cc) and read ~16% vs ~65% for the same code, so
it must never gate; its `recorder='node'` option is gone.

Run (stdio): PYTHONPATH=. scripts/python/.venv/bin/python -m mcp_servers.loop.server
Requires: fastmcp (`scripts/python/.venv/bin/pip install fastmcp`). Registered in .mcp.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "scripts" / "python" / ".venv" / "bin" / "python"
COMPARE = REPO_ROOT / "scripts" / "python" / "compare_audio.py"
NODE_DIR = REPO_ROOT / "scripts" / "node"
NODE_MODULES = NODE_DIR / "node_modules"
NODE_BLACKHOLE = NODE_DIR / "dist" / "record-strudel-blackhole.js"

# compare_audio.py exit code when the editability detector refuses to score (spec 003)
EXIT_EDITABILITY_FAIL = 3

sys.path.insert(0, str(REPO_ROOT))
# editability_check.py lives next to compare_audio.py (scripts/python has no __init__.py, so it
# is imported as a top-level module from that directory, like the pytest suite does).
sys.path.insert(1, str(REPO_ROOT / "scripts" / "python"))
from eval.gate import evaluate_comparison, load_thresholds  # noqa: E402
from editability_check import (  # noqa: E402
    ParseError,
    check_editability,
    to_json_fields,
)

mcp = FastMCP(
    "midi-grep-loop",
    instructions=(
        "Verification loop for MIDI-grep. Use verify_strudel to check a Strudel file against the "
        "editability contract (spec 003 replay detector), render it through BlackHole, and gate "
        "its similarity to the original against eval/thresholds.yaml. Replay output (loopAt, "
        "slice(N,run(N)).slow(N), *full stem samples) is refused before any render — fix the "
        "code, do not try to score it."
    ),
)


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))


def _editability(strudel_path: str) -> dict:
    """Detector verdict for a Strudel file as the five comparison.json keys (fail-closed)."""
    p = Path(strudel_path)
    try:
        code = p.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "editability": "fail",
            "generation_mode": None,
            "editability_violations": [f"cannot read {p}: {exc}"],
            "editable_voice_count": 0,
            "texture_voice_count": 0,
        }
    try:
        return to_json_fields(check_editability(code))
    except ParseError as exc:
        return {
            "editability": "fail",
            "generation_mode": None,
            "editability_violations": [f"parse error: {exc}"],
            "editable_voice_count": 0,
            "texture_voice_count": 0,
        }


def _recorder_preflight() -> str | None:
    """Return an error message when the BlackHole recorder cannot run, else None."""
    if not NODE_MODULES.exists():
        return (
            f"scripts/node/node_modules missing ({NODE_MODULES}) — run "
            "`cd scripts/node && npm install && npm run build`."
        )
    if not NODE_BLACKHOLE.exists():
        return f"renderer not built: {NODE_BLACKHOLE}. Run `cd scripts/node && npm run build`."
    return None


def _gate_fields(verdict) -> dict:
    return {
        "genre": verdict.genre,
        "mode": verdict.mode,
        "editability": verdict.editability,
        "similarity": round(verdict.similarity, 4),
        "floor": verdict.floor,
        "floor_source": verdict.floor_source,
        "worst_band_diff": verdict.worst_band_diff,
        "gate_passed": verdict.passed,
        "message": verdict.message,
    }


@mcp.tool()
def render_strudel(
    strudel_path: str,
    output_wav: str,
    duration: int = 30,
) -> dict:
    """Render a .strudel file to a WAV file with the BlackHole recorder (real Strudel playback).

    Args:
        strudel_path: path to the .strudel code file.
        output_wav: destination WAV path.
        duration: seconds to render.

    Needs `scripts/node/node_modules` (npm install), the built recorder (npm run build), the
    BlackHole device and a Multi-Output Device selected as system output (see CLAUDE.md).
    """
    err = _recorder_preflight()
    if err:
        return {"ok": False, "error": err}
    cmd = ["node", str(NODE_BLACKHOLE), strudel_path, "-o", output_wav, "-d", str(duration)]
    try:
        proc = _run(cmd, timeout=duration + 120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"render timed out after {duration + 120}s"}
    ok = proc.returncode == 0 and Path(output_wav).exists()
    return {
        "ok": ok,
        "recorder": "blackhole",
        "output_wav": output_wav,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
    }


@mcp.tool()
def compare_render(
    original: str,
    rendered: str,
    genre: str | None = None,
    duration: int = 60,
    strudel_path: str | None = None,
    mode: str | None = None,
) -> dict:
    """Compare a rendered WAV to the original audio and return similarity + gate verdict.

    Runs compare_audio.py (MAE frequency-balance similarity) and evaluates the result against
    eval/thresholds.yaml for the given genre (and generation mode, when a per-mode floor exists).

    Args:
        strudel_path: the Strudel source of the render. Passed as `--strudel` so the detector
            runs first and the comparison.json is stamped with generation_mode + editability.
            A replay deliverable returns {"ok": false, "stage": "editability", ...} unscored.
        mode: generation mode for the per-mode floor ('sample-instrument' | 'synth'); defaults
            to the comparison's own `generation_mode` (from the Strudel header).
    """
    if not COMPARE.exists():
        return {"ok": False, "error": f"compare_audio.py not found at {COMPARE}"}
    cmd = [_python(), str(COMPARE), original, rendered, "-d", str(duration), "-j"]
    if strudel_path:
        cmd += ["--strudel", strudel_path]
    try:
        proc = _run(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "compare_audio timed out after 300s"}
    if proc.returncode == EXIT_EDITABILITY_FAIL:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"editability": "fail", "editability_violations": [proc.stderr[-1500:]]}
        return {
            "ok": False,
            "stage": "editability",
            "editability": "fail",
            "generation_mode": payload.get("generation_mode"),
            "editability_violations": payload.get("editability_violations", []),
            "gate_passed": False,
            "message": "editability: fail — replay / non-editable output is not scored",
        }
    if proc.returncode != 0:
        return {"ok": False, "error": "compare_audio failed", "stderr_tail": proc.stderr[-1500:]}
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "could not parse compare_audio JSON output", "stdout_tail": proc.stdout[-1500:]}

    tmp = Path(rendered).with_suffix(".comparison.json")
    tmp.write_text(json.dumps(results))
    try:
        verdict = evaluate_comparison(tmp, genre=genre, thresholds=load_thresholds(), mode=mode)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"gate evaluation failed: {exc}"}
    return {
        "ok": True,
        **_gate_fields(verdict),
        "comparison_json": str(tmp),
    }


@mcp.tool()
def verify_strudel(
    strudel_path: str,
    original: str,
    genre: str | None = None,
    duration: int = 30,
    mode: str | None = None,
) -> dict:
    """Check, render and verify a Strudel file against the original in one call (loop-closer).

    Order: editability detector (spec 003) → BlackHole render → compare_audio.py --strudel →
    eval gate. A replay / non-editable file stops at stage "editability" with its violations and
    is never rendered or scored. Returns the gate verdict so the agent can decide accept/reject.
    """
    fields = _editability(strudel_path)
    if fields["editability"] != "pass":
        return {"ok": False, "stage": "editability", "gate_passed": False, **fields}
    rendered = str(Path(strudel_path).with_suffix(".render.wav"))
    r = render_strudel(strudel_path, rendered, duration=duration)
    if not r.get("ok"):
        return {"ok": False, "stage": "render", **r}
    c = compare_render(
        original, rendered, genre=genre, duration=max(duration, 30),
        strudel_path=strudel_path, mode=mode or fields.get("generation_mode"),
    )
    if not c.get("ok"):
        return {"ok": False, "stage": c.get("stage", "compare"), **{k: v for k, v in c.items() if k != "stage"}}
    return {"ok": True, "stage": "verified", **c}


@mcp.tool()
def eval_gate(comparison_json: str, genre: str | None = None, mode: str | None = None) -> dict:
    """Run the similarity eval gate on an existing comparison.json (no rendering).

    `mode` selects the per-mode floor ('sample-instrument' | 'synth'); defaults to the file's
    `generation_mode`. A comparison.json stamped `editability: "fail"` (or `comparison: null`)
    fails the gate with reason `editability: fail`.
    """
    try:
        verdict = evaluate_comparison(comparison_json, genre=genre, thresholds=load_thresholds(), mode=mode)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **_gate_fields(verdict)}


if __name__ == "__main__":
    mcp.run()
