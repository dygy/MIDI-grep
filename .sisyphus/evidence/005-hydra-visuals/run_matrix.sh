#!/bin/bash
# Spec 005 evidence: run `python -m visuals` on one track's pinned versions, one log per run.
# usage: run_matrix.sh "<.cache/stems track dir name>" vNNN [vNNN ...]
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
L="$REPO/.sisyphus/evidence/005-hydra-visuals/logs"
cd "$REPO/scripts/python" || exit 2
t="$1"; shift
for v in "$@"; do
  slug=$(echo "$t" | tr -cs 'A-Za-z0-9' '_' | cut -c1-24)
  log="$L/matrix_${slug}_${v}.log"
  s=$(date +%s)
  .venv/bin/python -m visuals "$REPO/.cache/stems/$t" --version "$v" > "$log" 2>&1
  rc=$?
  echo "$t $v rc=$rc $(( $(date +%s) - s ))s | $(grep -E '^\[visuals\] chain:|^\[check-hydra\] (PASS|FAIL)|^\[visuals\] check:' "$log" | tr '\n' ' ' | cut -c1-400)"
done
