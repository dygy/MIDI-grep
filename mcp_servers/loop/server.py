"""MIDI-grep verification loop MCP server.

Closes the render→compare→gate loop so an agent can verify a Strudel change end-to-end
without the human running the CLI by hand, specialised for MIDI-grep's chain:

    Strudel code  --render-->  WAV  --compare-->  similarity  --gate-->  pass/fail

Tools:
  - render_strudel    : render a .strudel file to WAV (BlackHole = accurate, node = fast emulation)
  - compare_render    : compare a rendered WAV to an original, return similarity + gate verdict
  - verify_strudel    : render + compare + gate in one call (the loop-closer)
  - eval_gate         : run the eval gate on an existing comparison.json

Run (stdio): poetry/venv python -m mcp_servers.loop.server
Requires: fastmcp (`scripts/python/.venv/bin/pip install fastmcp`). Registered in .mcp.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "scripts" / "python" / ".venv" / "bin" / "python"
COMPARE = REPO_ROOT / "scripts" / "python" / "compare_audio.py"
NODE_BLACKHOLE = REPO_ROOT / "scripts" / "node" / "dist" / "record-strudel-blackhole.js"
NODE_SYNTH = REPO_ROOT / "scripts" / "node" / "dist" / "render-strudel-node.js"

sys.path.insert(0, str(REPO_ROOT))
from eval.gate import evaluate_comparison, load_thresholds  # noqa: E402

mcp = FastMCP(
    "midi-grep-loop",
    instructions=(
        "Verification loop for MIDI-grep. Use verify_strudel to render a Strudel file and check "
        "its similarity to the original against the eval gate. Prefer recorder='blackhole' for "
        "accept/reject decisions — the node synth reads far lower (~16% vs ~65%) and must not gate."
    ),
)


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))


@mcp.tool()
def render_strudel(
    strudel_path: str,
    output_wav: str,
    duration: int = 30,
    recorder: Literal["blackhole", "node"] = "blackhole",
) -> dict:
    """Render a .strudel file to a WAV file.

    Args:
        strudel_path: path to the .strudel code file.
        output_wav: destination WAV path.
        duration: seconds to render.
        recorder: 'blackhole' (records real Strudel, accurate, needs BlackHole device) or
                  'node' (offline synth emulation, fast, low fidelity — never gate on it).
    """
    script = NODE_BLACKHOLE if recorder == "blackhole" else NODE_SYNTH
    if not script.exists():
        return {"ok": False, "error": f"renderer not built: {script}. Run `cd scripts/node && npm run build`."}
    cmd = ["node", str(script), strudel_path, "-o", output_wav, "-d", str(duration)]
    try:
        proc = _run(cmd, timeout=duration + 120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"render timed out after {duration + 120}s"}
    ok = proc.returncode == 0 and Path(output_wav).exists()
    return {
        "ok": ok,
        "recorder": recorder,
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
) -> dict:
    """Compare a rendered WAV to the original audio and return similarity + gate verdict.

    Runs compare_audio.py (MAE frequency-balance similarity) and evaluates the result against
    eval/thresholds.yaml for the given genre.
    """
    if not COMPARE.exists():
        return {"ok": False, "error": f"compare_audio.py not found at {COMPARE}"}
    cmd = [_python(), str(COMPARE), original, rendered, "-d", str(duration), "-j"]
    try:
        proc = _run(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "compare_audio timed out after 300s"}
    if proc.returncode != 0:
        return {"ok": False, "error": "compare_audio failed", "stderr_tail": proc.stderr[-1500:]}
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "could not parse compare_audio JSON output", "stdout_tail": proc.stdout[-1500:]}

    tmp = Path(rendered).with_suffix(".comparison.json")
    tmp.write_text(json.dumps(results))
    try:
        verdict = evaluate_comparison(tmp, genre=genre, thresholds=load_thresholds())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"gate evaluation failed: {exc}"}
    return {
        "ok": True,
        "genre": verdict.genre,
        "similarity": round(verdict.similarity, 4),
        "floor": verdict.floor,
        "worst_band_diff": verdict.worst_band_diff,
        "gate_passed": verdict.passed,
        "message": verdict.message,
        "comparison_json": str(tmp),
    }


@mcp.tool()
def verify_strudel(
    strudel_path: str,
    original: str,
    genre: str | None = None,
    duration: int = 30,
    recorder: Literal["blackhole", "node"] = "blackhole",
) -> dict:
    """Render a Strudel file and verify it against the original in one call (loop-closer).

    Returns the gate verdict so the agent can decide accept/reject. Use recorder='blackhole'
    for real decisions.
    """
    rendered = str(Path(strudel_path).with_suffix(".render.wav"))
    r = render_strudel(strudel_path, rendered, duration=duration, recorder=recorder)
    if not r.get("ok"):
        return {"ok": False, "stage": "render", **r}
    c = compare_render(original, rendered, genre=genre, duration=max(duration, 30))
    if not c.get("ok"):
        return {"ok": False, "stage": "compare", **c}
    return {"ok": True, "stage": "verified", **c}


@mcp.tool()
def eval_gate(comparison_json: str, genre: str | None = None) -> dict:
    """Run the similarity eval gate on an existing comparison.json (no rendering)."""
    try:
        verdict = evaluate_comparison(comparison_json, genre=genre, thresholds=load_thresholds())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "genre": verdict.genre,
        "similarity": round(verdict.similarity, 4),
        "floor": verdict.floor,
        "gate_passed": verdict.passed,
        "message": verdict.message,
    }


if __name__ == "__main__":
    mcp.run()
