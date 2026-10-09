# MIDI-grep Verification Loop MCP

An MCP server that lets Claude Code verify Strudel changes end-to-end without you running the CLI
by hand. It closes the loop:

```
Strudel code --editability--> render (BlackHole) --compare--> similarity --gate--> pass/fail
```

Specialised for MIDI-grep's check→render→compare→gate chain. Similarity is **only computed on
output that passes the editability contract** (spec 003, `scripts/python/editability_check.py`):
replay output (`loopAt(…)`, `slice(N, run(N)).slow(N)`, `s("<stem>full")`, loop-only files) is
refused before any render and never gets a score.

## Tools

| Tool | What it does |
|------|--------------|
| `verify_strudel(strudel_path, original, genre, duration, mode)` | **The loop-closer** — editability check + BlackHole render + compare + gate in one call. Returns `{"ok": false, "stage": "editability", "editability_violations": [...]}` on a replay file (not rendered, not scored), otherwise the gate verdict. |
| `render_strudel(strudel_path, output_wav, duration)` | Render a `.strudel` file to WAV with the BlackHole recorder (real Strudel playback). |
| `compare_render(original, rendered, genre, duration, strudel_path, mode)` | Run `compare_audio.py` + the eval gate. Pass `strudel_path` so the detector runs first (`--strudel`) and the `comparison.json` is stamped with `generation_mode` / `editability`. |
| `eval_gate(comparison_json, genre, mode)` | Run the gate on an existing `comparison.json` (no render). A file stamped `editability: "fail"` fails with reason `editability: fail`. |

Every gate result carries `genre`, `mode`, `editability`, `similarity`, `floor`, `floor_source`
(`modes.<mode>` when a measured per-mode floor applied, else `genres`), `gate_passed`, `message`.

## Important

- **BlackHole is the only recorder.** The old Node.js offline synth emulation (and the
  `recorder='node'` option) was deleted — it read ~16% vs ~65% for the same code and must never gate.
- BlackHole rendering needs the BlackHole device + a Multi-Output Device selected as system output
  (see CLAUDE.md). Preflight: `cd scripts/node && npm install && npm run build` — `render_strudel`
  returns an error naming the missing `node_modules` or the unbuilt `dist/record-strudel-blackhole.js`.
- Gate floors live in `eval/thresholds.yaml` (per-genre, plus per-mode under `modes:` once a
  detector-passing render has been measured); gate logic in `eval/gate.py`.
- `mode` is `sample-instrument` or `synth`; when omitted it is read from the Strudel header
  (`// generation_mode:`) / the comparison's `generation_mode` key.

## Setup

```bash
scripts/python/.venv/bin/pip install fastmcp pyyaml   # or: pip install -r scripts/python/requirements.txt
```

Registered in `.mcp.json` as the `loop` server (stdio). Restart Claude Code to load it.

## Manual smoke test

```bash
PYTHONPATH=. scripts/python/.venv/bin/python -m mcp_servers.loop.server   # starts stdio server
PYTHONPATH=. scripts/python/.venv/bin/python -c "import mcp_servers.loop.server"   # import check
```
