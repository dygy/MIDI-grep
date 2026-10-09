#!/usr/bin/env bash
#
# sample-pipeline.sh — URL → real stem sample-pack → Strudel "samples()" → similar music
#
# Builds a Strudel sample pack from extracted stems, generates code that loads it
# from a base URL, and either (a) serves it locally + renders + scores similarity,
# or (b) uploads it to Cloudflare R2 and points the code at the public bucket.
#
# The ONLY thing that differs between local and R2 is the base URL, so the same
# pack and code work in both.
#
# Usage:
#   scripts/sample-pipeline.sh --url "https://youtu.be/ID" --prefix myid [--mode instrument|loops|hybrid] --local
#   scripts/sample-pipeline.sh --stems-dir ".cache/stems/<name>" --prefix myid --r2
#
# Local mode needs: BlackHole 2ch + scripts/node deps (npm install) for rendering.
# R2 mode needs:  R2_BUCKET + R2_PUBLIC_BASE and either R2 S3 keys
#                 (R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY) or `wrangler login`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/scripts/python/.venv/bin/python"
URL=""; STEMS_DIR=""; PREFIX=""; MODE="instrument"; TARGET="local"; PORT="5555"; BPM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2;;
    --stems-dir) STEMS_DIR="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --bpm) BPM="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --local) TARGET="local"; shift;;
    --r2) TARGET="r2"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$PREFIX" ]] && { echo "ERROR: --prefix is required" >&2; exit 2; }

# 1. Get stems (extract from URL if no stems-dir given)
if [[ -z "$STEMS_DIR" ]]; then
  [[ -z "$URL" ]] && { echo "ERROR: pass --url or --stems-dir" >&2; exit 2; }
  echo "==> Extracting stems from $URL"
  "$ROOT/bin/midi-grep" extract --url "$URL" --render none --iterate 0
  STEMS_DIR="$(ls -dt "$ROOT"/.cache/stems/*/ | head -1)"
  echo "    stems: $STEMS_DIR"
fi
[[ -f "$STEMS_DIR/melodic.wav" ]] || { echo "ERROR: no melodic.wav in $STEMS_DIR" >&2; exit 1; }
STEMS_DIR="$(cd "$STEMS_DIR" && pwd)"   # absolute — needed for the serve symlink

PACK="$STEMS_DIR/sample_pack"

# 2. Build the sample pack
echo "==> Building sample pack -> $PACK"
"$PY" "$ROOT/scripts/python/build_sample_pack.py" \
  --stems-dir "$STEMS_DIR" --out "$PACK" ${BPM:+--bpm "$BPM"}

if [[ "$TARGET" == "r2" ]]; then
  # 3r. Upload to R2 and resolve the public base URL
  echo "==> Uploading pack to R2 (prefix=$PREFIX)"
  UP_JSON="$("$PY" "$ROOT/scripts/python/upload_r2.py" --pack-dir "$PACK" --prefix "$PREFIX")"
  echo "$UP_JSON"
  BASE_URL="$(echo "$UP_JSON" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["public_base"])')"
  [[ -z "$BASE_URL" ]] && { echo "ERROR: no public_base — set R2_PUBLIC_BASE" >&2; exit 1; }
  echo "==> Generating Strudel ($MODE) for R2 base $BASE_URL"
  "$PY" "$ROOT/scripts/python/generate_sample_strudel.py" \
    --pack-dir "$PACK" --base-url "$BASE_URL" --mode "$MODE"
  # samples.json now carries the R2 _base; re-upload it so the hosted copy matches.
  "$PY" "$ROOT/scripts/python/upload_r2.py" --pack-dir "$PACK" --prefix "$PREFIX" >/dev/null
  echo "==> Done. Open output_${MODE}.strudel in Strudel; samples load from R2."
  exit 0
fi

# 3l. Local: serve, generate, render, score
BASE_URL="http://localhost:$PORT/$PREFIX"
SERVE_ROOT="$(mktemp -d)"; ln -sfn "$PACK" "$SERVE_ROOT/$PREFIX"
SRV_PY="$SERVE_ROOT/_serve.py"; SRV_LOG="$SERVE_ROOT/_serve.log"
cat > "$SRV_PY" <<'PYSRV'
import http.server, socketserver, sys, os
os.chdir(sys.argv[2])
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*'); super().end_headers()
    def log_message(self, *a): pass
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
S(("", int(sys.argv[1])), H).serve_forever()
PYSRV
echo "==> Serving $PACK at $BASE_URL (threaded)"
"$PY" "$SRV_PY" "$PORT" "$SERVE_ROOT" >"$SRV_LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; rm -rf "$SERVE_ROOT"' EXIT

echo "==> Generating Strudel ($MODE) for $BASE_URL"
"$PY" "$ROOT/scripts/python/generate_sample_strudel.py" \
  --pack-dir "$PACK" --base-url "$BASE_URL" --mode "$MODE"

# Wait until the server actually serves samples.json before rendering.
for i in $(seq 1 20); do
  curl -fs -o /dev/null "$BASE_URL/samples.json" && break
  sleep 0.5
done
curl -fs -o /dev/null "$BASE_URL/samples.json" || {
  echo "ERROR: local server not reachable at $BASE_URL" >&2
  echo "--- server log ---" >&2; cat "$SRV_LOG" >&2 || true; exit 1; }

OUT="$PACK/output_${MODE}.strudel"
REND="$PACK/render_${MODE}.wav"
if [[ -f "$ROOT/scripts/node/dist/record-strudel-blackhole.js" && -d "$ROOT/scripts/node/node_modules/puppeteer" ]]; then
  echo "==> Rendering via BlackHole"
  node "$ROOT/scripts/node/dist/record-strudel-blackhole.js" "$OUT" -o "$REND" -d 30 2>&1 | grep -iE "saved|NET FAIL" | head -3 || true
  echo "==> Similarity vs original:"
  "$PY" "$ROOT/scripts/python/compare_audio.py" "$STEMS_DIR/original.wav" "$REND" -d 25 -j 2>/dev/null \
    | "$PY" -c 'import sys,json;c=json.load(sys.stdin)["comparison"];print("   overall %.1f%%  (mfcc %.2f, chroma %.2f, freq %.2f)"%(c["overall_similarity"]*100,c.get("mfcc_similarity",0),c.get("chroma_similarity",0),c.get("frequency_balance_similarity",0)))'
else
  echo "==> Skipping render (need: cd scripts/node && npm install, + BlackHole 2ch)"
  echo "    Strudel code: $OUT"
fi
echo "==> Done."
