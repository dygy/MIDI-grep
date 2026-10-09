#!/usr/bin/env bash
#
# MIDI-grep docs-freshness reminder — a PostToolUse hook on Edit|Write|MultiEdit.
#
# ADVISORY ONLY (the analogue of Citation's citation-ticket-flow.sh): when a
# pipeline file is edited it reminds the agent, once per session, of the
# CLAUDE.md "Context Document Maintenance" rule (update llms.txt, llms-full.txt,
# CLAUDE.md) and of the eval gate for Strudel-output-affecting changes. It
# never blocks, always exits 0, and fails silently.
set -u
payload=$(cat 2>/dev/null) || exit 0
read -r file sid < <(printf '%s' "$payload" | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""), d.get("session_id","nosession"))
except Exception: print("", "nosession")' 2>/dev/null) || exit 0
[ -z "$file" ] && exit 0
case "$file" in
  */internal/*|*/cmd/*|*/scripts/python/*|*/scripts/node/src/*|*/eval/*|*/mcp_servers/*) ;;
  *) exit 0 ;;
esac
case "$file" in */scripts/python/tests/*|*_test.go) exit 0 ;; esac
marker="${TMPDIR:-/tmp}/midigrep-docs-freshness-${sid}"
[ -e "$marker" ] && exit 0
touch "$marker" 2>/dev/null
python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":sys.argv[1]}}))' \
  "docs-freshness: a pipeline file changed ($file). Per CLAUDE.md 'Context Document Maintenance', update llms.txt / llms-full.txt / CLAUDE.md before finishing if a mode, flag, script, synthesis parameter, renderer or similarity metric changed. If the change can alter generated Strudel or its render, verify through the loop MCP (verify_strudel, recorder=blackhole) against eval/thresholds.yaml — the node recorder never gates. Any similarity number you cite must come from an actual comparison.json. (Shown once per session.)"
exit 0
