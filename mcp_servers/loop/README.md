# MIDI-grep Verification Loop MCP

An MCP server that lets Claude Code verify Strudel changes end-to-end without you running the CLI
by hand. It closes the loop:

```
Strudel code  --render-->  WAV  --compare-->  similarity  --gate-->  pass/fail
```

Specialised for MIDI-grep's render→compare→gate chain.

## Tools

| Tool | What it does |
|------|--------------|
| `verify_strudel(strudel_path, original, genre, duration, recorder)` | **The loop-closer** — render + compare + gate in one call. Returns the gate verdict. |
| `render_strudel(strudel_path, output_wav, duration, recorder)` | Render a `.strudel` file to WAV. `recorder='blackhole'` (accurate) or `'node'` (fast emulation). |
| `compare_render(original, rendered, genre, duration)` | Run `compare_audio.py` + the eval gate; returns similarity + pass/fail. |
| `eval_gate(comparison_json, genre)` | Run the gate on an existing `comparison.json` (no render). |

## Important

- Use `recorder='blackhole'` for accept/reject decisions. The Node.js synth reads far lower
  (~16% vs ~65% for the same code) and **must not gate** — it's only for a quick smoke render.
- BlackHole rendering needs the BlackHole device + a Multi-Output Device selected as system output
  (see CLAUDE.md). Build the renderer first: `cd scripts/node && npm run build`.
- Gate floors live in `eval/thresholds.yaml`; gate logic in `eval/gate.py`.

## Setup

```bash
scripts/python/.venv/bin/pip install fastmcp pyyaml   # or: pip install -r scripts/python/requirements.txt
```

Registered in `.mcp.json` as the `loop` server (stdio). Restart Claude Code to load it.

## Manual smoke test

```bash
PYTHONPATH=. scripts/python/.venv/bin/python -m mcp_servers.loop.server   # starts stdio server
```
