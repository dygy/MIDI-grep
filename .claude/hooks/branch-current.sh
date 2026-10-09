#!/usr/bin/env bash
#
# MIDI-grep branch-freshness gate — a PreToolUse hook on Bash.
#
# Modeled on Citation's citation-branch-current.sh, reduced to one repo and one
# target (origin/main). It BLOCKS three moments on a branch that is behind
# origin/main or no longer merges into it:
#
#   branch   creating a new branch from a stale base   (git checkout -b / git switch -c)
#   pr       opening a PR                              (gh pr create)
#   merge    merging a PR                              (gh pr merge)
#
# Policy: branch and pr refuse when HEAD is behind origin/main at all (rebase a
# private branch first); merge refuses only when a real conflict exists (a PR
# under review is merged-into, never rebased). Undetermined (offline, no
# remote, old git) is NEVER a silent pass: the call is allowed and a
# systemMessage says the check did not run.
#
# Override one command:  MIDIGREP_SKIP_BRANCH_CHECK=1 <cmd>
#
# Payload: .tool_input.command (Bash), .cwd. Output: PreToolUse deny JSON or a
# top-level systemMessage. Everything else exits 0 silently.
set -u
payload=$(cat 2>/dev/null) || exit 0
cmd=$(printf '%s' "$payload" | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception: pass' 2>/dev/null) || exit 0
[ -z "$cmd" ] && exit 0

# Blank quoted strings first so flag-shaped text inside --title/--body can never
# be read as a command, then classify.
stripped=$(printf '%s' "$cmd" | sed -E "s/\"[^\"]*\"//g; s/'[^']*'//g")
case "$stripped" in *MIDIGREP_SKIP_BRANCH_CHECK=1*) exit 0 ;; esac

moment=""
printf '%s' "$stripped" | grep -Eq '(^|[;&|[:space:]])git[[:space:]]+(checkout[[:space:]]+-b|switch[[:space:]]+-c)' && moment=branch
printf '%s' "$stripped" | grep -Eq '(^|[;&|[:space:]])gh[[:space:]]+pr[[:space:]]+create' && moment=pr
printf '%s' "$stripped" | grep -Eq '(^|[;&|[:space:]])gh[[:space:]]+pr[[:space:]]+merge' && moment=merge
[ -z "$moment" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0

emit_deny() {
  python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":sys.argv[1]}}))' "$1"
  exit 0
}
emit_warn() {
  python3 -c 'import json,sys; print(json.dumps({"systemMessage":sys.argv[1]}))' "$1"
  exit 0
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || emit_warn "branch-current gate: not a git work tree — '$moment' allowed UNCHECKED."
git fetch -q origin main 2>/dev/null || emit_warn "branch-current gate: could not fetch origin/main (offline? no remote?) — '$moment' allowed UNCHECKED. Re-check: git fetch origin main && git rev-list --count HEAD..origin/main"
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null) || emit_warn "branch-current gate: cannot compare with origin/main — '$moment' allowed UNCHECKED."
branch=$(git branch --show-current 2>/dev/null); [ -z "$branch" ] && branch="detached HEAD"

conflict=unknown
if git merge-tree --write-tree HEAD origin/main >/dev/null 2>&1; then conflict=no
elif [ $? -eq 1 ]; then conflict=yes
fi

case "$moment" in
  branch|pr)
    if [ "$behind" != "0" ]; then
      emit_deny "branch-current gate: '$branch' is $behind commit(s) behind origin/main (conflicts: $conflict). Rebase first — git pull --rebase origin main — then re-run. One-off override: MIDIGREP_SKIP_BRANCH_CHECK=1 <cmd>."
    fi ;;
  merge)
    if [ "$conflict" = yes ]; then
      emit_deny "branch-current gate: '$branch' conflicts with origin/main — merge origin/main into the branch, resolve, push, then retry gh pr merge. Override: MIDIGREP_SKIP_BRANCH_CHECK=1 <cmd>."
    fi ;;
esac
[ "$conflict" = unknown ] && emit_warn "branch-current gate: '$moment' allowed, but conflict check was undetermined (git merge-tree --write-tree needs git >= 2.38)."
exit 0
