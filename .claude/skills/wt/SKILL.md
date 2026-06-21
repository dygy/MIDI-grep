---
name: wt
description: Create a fresh git worktree branched off origin/main and switch into it so the developer can start coding immediately. Use when the developer wants an isolated worktree for a new task — with or without a name. Triggers on "wt", "new worktree", "worktree from main", "spin up a worktree".
argument-hint: "[name]  (optional — branch/dir slug; auto-generated if omitted)"
allowed-tools: Bash
---

Create a new git worktree based on `origin/main`, on a **real branch** (never a detached HEAD), **switch the session into it**, and leave the developer ready to code right away.

The argument (`$ARGUMENTS`) is an **optional** name used for both the branch and the directory. When omitted, a timestamped name is generated.

Worktrees are created **inside the repo** under `.claude/worktrees/<name>` — this is what lets the final `cd` persist across commands (a worktree outside the repo root gets the shell cwd reset back to the project root by the sandbox, so the switch would not stick).

## Step 1 — create the worktree and switch into it

Run this single block:

```bash
set -euo pipefail

# Resolve name from the optional argument; sanitise to a safe slug.
# Allow only alnum, dot, underscore, dash; collapse separators; forbid
# path traversal so a name like "../../tmp/x" can't escape the worktree root.
RAW="${ARGUMENTS:-}"
NAME="$(printf '%s' "$RAW" \
  | tr '[:space:]' '-' \
  | tr -cd '[:alnum:]._-' \
  | sed -E 's/-+/-/g; s/^[.-]+//; s/[.-]+$//')"
[ -z "$NAME" ] && NAME="dev-$(date +%Y%m%d-%H%M%S)"
case "$NAME" in
  ""|"."|".."|*..*) echo "Invalid worktree name: $NAME" >&2; exit 1 ;;
esac

# Always branch from the latest main.
git fetch origin main

ROOT="$(git rev-parse --show-toplevel)"
WTROOT="$ROOT/.claude/worktrees"
mkdir -p "$WTROOT"

# Pick a name that collides with neither an existing worktree dir nor an
# existing local branch, so re-runs never hard-fail.
BASE="$NAME"; i=2
while [ -e "$WTROOT/$NAME" ] || git show-ref --verify --quiet "refs/heads/$NAME"; do
  NAME="${BASE}-${i}"; i=$((i+1))
done
DIR="$WTROOT/$NAME"

# Create the worktree on a fresh branch tracking origin/main.
git worktree add -b "$NAME" "$DIR" origin/main

echo "------------------------------------------------------------"
echo "Worktree ready"
echo "  branch: $NAME   (from origin/main)"
echo "  path:   $DIR"
git -C "$DIR" status -sb | head -1
echo "------------------------------------------------------------"

# Switch the session into the worktree. Because DIR is INSIDE the repo
# root, this cwd persists across subsequent commands.
cd "$DIR"
```

## Step 2 — confirm and continue

After the block runs, the session is already in the new worktree on the new branch. Do NOT ask the developer to `cd` anywhere — the skill has done the switch.

1. **Notify the developer**, briefly: which branch they're on, which worktree path, that it's freshly branched from `main`, and that they can start working now. Example: "✅ You're now on branch `<name>` in worktree `.claude/worktrees/<name>` (branched from main). Ready to continue."
2. **Keep working in this worktree.** All subsequent file edits and commands operate here — the cwd persists, so plain relative paths and bare `git`/`poetry` commands all run against the worktree. Do not reach back into the original repo root.
