#!/usr/bin/env bash
# editable-pipeline.sh — one command from a YouTube link to editable Strudel, scored on real playback.
#
# Implements spec 004 technical-considerations §2.1 stages 1-9. Each stage skips when its output
# already exists (use --force to redo). Every number that varies per track (bpm, key, genre, bars,
# generator knobs) is READ from the track's metadata.json / produced by calibrate_dynamic.py via
# scripts/auto-calibrate.sh — this driver sets no per-track constant.
#
# Stages:
#   1 stems+analysis   ./bin/midi-grep extract --url U --render none --iterate 0
#   2 MIDI per stem    transcribe.py {bass,melodic,vocals}.wav -> sample_pack/*.mid
#   3 drum onsets      detect_drums_bands.py drums.wav --bpm B -> sample_pack/drums_bands.json
#   4 sample pack      build_sample_pack.py
#   5 instruments      midi-grep generative train (bass, lead) + build_instruments.py
#   6 host             upload_r2.py (--r2) | local CORS server on :5555 (--local)
#   7 generate+calib.  auto-calibrate.sh (sample-instrument); one generate_dynamic_strudel.py
#                      --mode synth with the BEST knobs + a BlackHole render
#   8 score + gate     compare_audio.py -d 135 --strudel ; eval/gate.py ; editability_check.py
#   9 promote          pipeline_helpers.py promote -> vNNN/{output.strudel,render.wav,comparison.json,metadata.json}
#
# Usage:
#   scripts/editable-pipeline.sh --url <youtube> [--title "<cache folder title>"] [--prefix <slug>]
#       [--genre <g>] [--mode sample-instrument|synth|both] [--iters 3] [--dur 170]
#       [--r2 | --local] [--force]
#
#   --url      YouTube link (required unless --title names an already-extracted track dir)
#   --title    locate an already-extracted track dir .cache/stems/<title> (skips stage 1 when complete)
#   --prefix   slug for hosting/model names (default: kebab-case of the track title)
#   --genre    override the genre in metadata.json (the override is logged)
#   --mode     which pieces to build (default both)
#   --iters    sample-instrument calibration iterations (default 3; best of N is kept)
#   --dur      BlackHole capture seconds per render (default 170)
#   --r2       host on the project's public R2 bucket (default); --local serves on localhost:5555
#   --force    redo every stage even if its output exists
#
# Requires (checked per stage, not up front): BlackHole + Multi-Output selected, node recorder built
# (scripts/node/dist), wrangler OAuth login for --r2. stdout carries ONLY the final JSON summary;
# progress goes to stderr.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/scripts/python/.venv/bin/python"
SP="$ROOT/scripts/python"
PH=("$PY" "$SP/pipeline_helpers.py")
REC="$ROOT/scripts/node/dist/record-strudel-blackhole.js"
GATE="$ROOT/eval/gate.py"
CACHE="$ROOT/.cache/stems"
R2_BUCKET_NAME="4cast"
R2_PUBLIC="https://pub-56831423fee34641805da07cfdaf6812.r2.dev"
R2_ACCOUNT="503e92d7d95838d80c33802d3274f284"
LOCAL_PORT="5555"
COMPARE_SECONDS="135"
RENDER_NOTE="record-strudel-blackhole.js RAW capture + uniform speed correction (ffmpeg wallclock restamp)"

URL="" TITLE="" PREFIX="" GENRE_OVERRIDE="" MODE="both" ITERS="3" DUR="170" HOSTING="r2" FORCE=""

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
log() { echo "[editable-pipeline] $*" >&2; }
die() { echo "[editable-pipeline] ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do case "$1" in
  --url) URL="${2:?--url needs a value}"; shift 2;;
  --title) TITLE="${2:?--title needs a value}"; shift 2;;
  --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2;;
  --genre) GENRE_OVERRIDE="${2:?--genre needs a value}"; shift 2;;
  --mode) MODE="${2:?--mode needs a value}"; shift 2;;
  --iters) ITERS="${2:?--iters needs a value}"; shift 2;;
  --dur) DUR="${2:?--dur needs a value}"; shift 2;;
  --r2) HOSTING="r2"; shift;;
  --local) HOSTING="local"; shift;;
  --force) FORCE="1"; shift;;
  -h|--help) usage; exit 0;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 64;;
esac; done

case "$MODE" in sample-instrument|synth|both) ;; *) die "--mode must be sample-instrument, synth or both";; esac
[ -n "$URL" ] || [ -n "$TITLE" ] || die "need --url (or --title of an already-extracted track)"
[ -x "$PY" ] || die "python venv missing: $PY"

modes() { if [ "$MODE" = both ]; then echo "sample-instrument synth"; else echo "$MODE"; fi; }
# stage_done <file...>: true when every file exists and --force was not given
stage_done() { [ -z "$FORCE" ] || return 1; local f; for f in "$@"; do [ -s "$f" ] || return 1; done; }
jget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));v=d.get(sys.argv[2]);print('' if v is None else v)" "$1" "$2"; }
# base_ok <samples.json> <expected-base>: the baked _base matches the hosting we are about to use
base_ok() { [ -s "$1" ] && [ "$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('_base',''))" "$1")" = "$2" ]; }

# ---------------------------------------------------------------- locate / stage 1
find_track_dir() {  # prints the cache dir whose metadata.json matches --title or the URL's video id
  "$PY" - "$CACHE" "$TITLE" "$URL" <<'PYF'
import json, sys
from pathlib import Path
cache, title, url = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if title:
    d = cache / title
    if (d / "metadata.json").is_file():
        print(d)
    sys.exit(0)
for meta in sorted(cache.glob("*/metadata.json")):
    try:
        vid = json.loads(meta.read_text()).get("video_id") or ""
    except (OSError, ValueError):
        continue
    if vid and vid in url:
        print(meta.parent)
        break
PYF
}

stems_complete() { local d="$1" f; for f in original.wav bass.wav drums.wav melodic.wav vocals.wav metadata.json; do [ -s "$d/$f" ] || return 1; done; }

TRACK="$(find_track_dir)"
if [ -n "$TRACK" ] && stems_complete "$TRACK" && [ -z "$FORCE" ]; then
  log "stage 1 skip: stems + metadata exist in $TRACK"
else
  [ -n "$URL" ] || die "stage 1 needs --url (track dir '${TRACK:-$TITLE}' is missing stems)"
  log "stage 1: extract stems + analysis"
  (cd "$ROOT" && ./bin/midi-grep extract --url "$URL" --render none --iterate 0) >&2 || die "stage 1 extract failed"
  TRACK="$(find_track_dir)"
  [ -n "$TRACK" ] && stems_complete "$TRACK" || die "stage 1 produced no complete track dir under $CACHE"
fi

META="$TRACK/metadata.json"
# Stage 1b — stamp: the Go cache writes only title/url/video_id here; bpm/key/style live in the
# latest vNNN/metadata.json (spec 004 defect #6). Run the deep genre detector (CLAP, with the
# analysed bpm as prior) and consolidate everything into the track metadata — once.
if [ -z "$(jget "$META" bpm)" ] || [ -n "$FORCE" ]; then
  mkdir -p "$TRACK/.pipeline"
  VBPM="$(for v in "$TRACK"/v[0-9][0-9][0-9]; do jget "$v/metadata.json" bpm; done | tail -1)"
  GJ="$TRACK/.pipeline/genre.json"
  if [ ! -s "$GJ" ]; then
    log "stage 1b: deep genre detection (CLAP)"
    "$PY" "$SP/detect_genre_dl.py" "$TRACK/original.wav" ${VBPM:+--bpm "$VBPM"} -o "$GJ" >/dev/null 2>"$TRACK/.pipeline/genre.err" \
      || log "stage 1b: detector failed (see .pipeline/genre.err) — falling back to the heuristic style"
  fi
  "${PH[@]}" stamp "$TRACK" ${GENRE_OVERRIDE:+--genre "$GENRE_OVERRIDE"} ${GJ:+--detector-json "$GJ"} >&2 \
    || die "stage 1b stamp failed"
fi
BPM="$(jget "$META" bpm)"; KEY="$(jget "$META" key)"; GENRE="$(jget "$META" genre)"
DURATION="$(jget "$META" duration)"; TRACK_TITLE="$(jget "$META" title)"
[ -n "$BPM" ] && [ -n "$DURATION" ] || die "metadata.json lacks bpm/duration: $META"
if [ -n "$GENRE_OVERRIDE" ]; then
  log "GENRE OVERRIDE: detector said '${GENRE:-none}', using '$GENRE_OVERRIDE' (--genre)"
  GENRE="$GENRE_OVERRIDE"
fi
[ -n "$GENRE" ] || die "no genre in metadata.json and no --genre given"
SLUG="${PREFIX:-$("${PH[@]}" slug "${TRACK_TITLE:-$(basename "$TRACK")}")}" || die "cannot derive a slug"
SND="${SLUG//-/_}"      # model / Strudel sound names avoid '-' (mini-notation); hosting prefix keeps kebab
BARS="$("${PH[@]}" num-bars "$DURATION" "$BPM")" || die "cannot derive num-bars"
log "track='$TRACK' slug=$SLUG bpm=$BPM key='$KEY' genre=$GENRE duration=${DURATION}s bars=$BARS mode=$MODE host=$HOSTING"

PACK="$TRACK/sample_pack"; PW="$TRACK/.pipeline"; mkdir -p "$PACK" "$PW"
if [ "$HOSTING" = r2 ]; then HOST_BASE="$R2_PUBLIC/midi-grep"; else HOST_BASE="http://localhost:$LOCAL_PORT"; fi
BASE="$HOST_BASE/$SLUG"; INST_BASE="$BASE/instruments/"

# ---------------------------------------------------------------- stage 2: MIDI per stem
for pair in "bass.wav:bass.mid" "melodic.wav:melodic.mid" "vocals.wav:vocals.mid"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if stage_done "$PACK/$dst"; then log "stage 2 skip: $dst exists"; continue; fi
  log "stage 2: transcribe $src"
  "$PY" "$SP/transcribe.py" "$TRACK/$src" "$PACK/$dst" >&2 || die "stage 2 transcribe $src failed"
  [ -s "$PACK/$dst" ] || die "stage 2 produced no $dst"
done

# ---------------------------------------------------------------- stage 3: drum onsets
if stage_done "$PACK/drums_bands.json"; then log "stage 3 skip: drums_bands.json exists"; else
  log "stage 3: drum band onsets"
  "$PY" "$SP/detect_drums_bands.py" "$TRACK/drums.wav" --bpm "$BPM" --out "$PACK/drums_bands.json" >&2 \
    || die "stage 3 detect_drums_bands failed"
fi

# ---------------------------------------------------------------- stage 4: sample pack
if stage_done "$PACK/pack.json" "$PACK/samples.json" "$PACK/strudel.json" && base_ok "$PACK/samples.json" "$BASE/"; then
  log "stage 4 skip: sample pack exists for $BASE/"
else
  log "stage 4: build sample pack"
  key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  "$PY" "$SP/build_sample_pack.py" --stems-dir "$TRACK" --out "$PACK" --bpm "$BPM" "${key_arg[@]}" \
    --prefix "$SND" --base-url "$BASE/" >&2 || die "stage 4 build_sample_pack failed"   # SND: sound names never carry '-' 
fi
[ -s "$PACK/drums/bd.wav" ] || die "stage 4 pack has no drums/ one-shots in $PACK/drums"

# ---------------------------------------------------------------- stage 5: instruments
for pair in "bass:bass.wav" "lead:melodic.wav"; do
  role="${pair%%:*}"; src="${pair##*:}"
  if stage_done "$ROOT/models/${SND}_$role/metadata.json"; then log "stage 5 skip: models/${SND}_$role trained"; continue; fi
  log "stage 5: train granular ${SND}_$role from $src"
  (cd "$ROOT" && ./bin/midi-grep generative train "$TRACK/$src" --name "${SND}_$role" --mode granular --output "$ROOT/models") >&2 \
    || die "stage 5 train ${SND}_$role failed"
done
if stage_done "$PACK/instruments/samples.json" && base_ok "$PACK/instruments/samples.json" "$INST_BASE"; then
  log "stage 5 skip: instruments/samples.json exists for $INST_BASE"
else
  log "stage 5: assemble instruments manifest"
  "$PY" "$SP/build_instruments.py" --models "$ROOT/models/${SND}_bass" "$ROOT/models/${SND}_lead" \
    --kit "$PACK/drums" --out "$PACK/instruments" --base-url "$INST_BASE" >&2 \
    || die "stage 5 build_instruments failed"
fi

# ---------------------------------------------------------------- stage 6: host
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; [ -n "${SERVE_ROOT:-}" ] && rm -rf "$SERVE_ROOT"; }
trap cleanup EXIT
if [ "$HOSTING" = r2 ]; then
  if stage_done "$PACK/.uploaded" && [ "$(cat "$PACK/.uploaded")" = "$BASE" ] \
     && curl -fsI -o /dev/null "$INST_BASE""samples.json"; then
    log "stage 6 skip: already hosted at $BASE"
  else
    log "stage 6: upload pack to R2 ($R2_BUCKET_NAME/midi-grep/$SLUG)"
    CLOUDFLARE_ACCOUNT_ID="$R2_ACCOUNT" "$PY" "$SP/upload_r2.py" --pack-dir "$PACK" --prefix "midi-grep/$SLUG" \
      --bucket "$R2_BUCKET_NAME" --public-base "$R2_PUBLIC" --backend wrangler >&2 || die "stage 6 upload_r2 failed"
    curl -fsI -o /dev/null "$INST_BASE""samples.json" || die "stage 6: $INST_BASE""samples.json not publicly reachable after upload"
    echo "$BASE" > "$PACK/.uploaded"
  fi
else
  if curl -fsI -o /dev/null "$INST_BASE""samples.json"; then log "stage 6 skip: local server already serving $BASE"; else
    SERVE_ROOT="$(mktemp -d)"; ln -sfn "$PACK" "$SERVE_ROOT/$SLUG"
    cat > "$SERVE_ROOT/_serve.py" <<'PYSRV'
import http.server, socketserver, sys, os
os.chdir(sys.argv[2])
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*'); super().end_headers()
    def log_message(self, *a): pass
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
S(("", int(sys.argv[1])), H).serve_forever()
PYSRV
    "$PY" "$SERVE_ROOT/_serve.py" "$LOCAL_PORT" "$SERVE_ROOT" >"$SERVE_ROOT/_serve.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 20); do curl -fsI -o /dev/null "$INST_BASE""samples.json" && break; sleep 0.5; done
    curl -fsI -o /dev/null "$INST_BASE""samples.json" || die "stage 6: local server not reachable at $BASE"
    log "stage 6: serving $PACK at $BASE (stopped when this script exits)"
  fi
fi

# ---------------------------------------------------------------- render preflight helpers
probe_blackhole() {  # 3 s capture, 20 s hard watchdog (no pkill -9), abort with a clear message
  local out="$PW/probe.wav"; rm -f "$out"
  ffmpeg -nostdin -loglevel error -f avfoundation -i ":BlackHole 2ch" -t 3 -y "$out" >/dev/null 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    die "BlackHole capture probe hung for 20 s (wedged avfoundation/BlackHole). Re-select the Multi-Output device or restart coreaudiod, then re-run."; fi
  wait "$pid" 2>/dev/null
  [ -s "$out" ] || die "BlackHole capture probe produced no audio file. Is 'BlackHole 2ch' installed and a Multi-Output Device selected as system output?"
  log "BlackHole probe OK ($(wc -c < "$out" | tr -d ' ') bytes)"
}
guard_no_recorder() {
  local p; p="$(pgrep -f "node .*record-strudel-blackhole\.js|ffmpeg .* -f avfoundation" || true)"
  [ -z "$p" ] || die "another record-strudel-blackhole process is running (pid $p); one render at a time"
}
preflight_render() {
  [ -f "$REC" ] || die "node recorder not built: (cd scripts/node && npm run build)"
  guard_no_recorder; probe_blackhole
}

# ---------------------------------------------------------------- stage 7: generate + calibrate
run_sample_instrument() {
  local W="$PW/sample-instrument"; mkdir -p "$W"
  if stage_done "$W/best.strudel" "$W/best.wav" "$W/best.cmp.json" "$W/best.params"; then
    log "stage 7 skip (sample-instrument): best render exists in $W"; return 0; fi
  preflight_render
  log "stage 7: auto-calibrate sample-instrument (iters=$ITERS)"
  key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  "$ROOT/scripts/auto-calibrate.sh" --stems "$TRACK" --base "$BASE" --inst "$INST_BASE""samples.json" \
    --bpm "$BPM" "${key_arg[@]}" --genre "$GENRE" --bars "$BARS" --iters "$ITERS" --dur "$DUR" \
    --bass-sound "${SND}_bass" --lead-sound "${SND}_lead" --vocal-mode instrument \
    --mode sample-instrument --workdir "$W" 2>&1 | tee "$W/calibrate.log" >&2
  local best; best="$(grep '^BEST:' "$W/calibrate.log" | tail -1)"
  [ -n "$best" ] && [ -s "$W/best.wav" ] || die "auto-calibrate produced no BEST render (see $W/calibrate.log)"
  # "BEST: calN overall=X  params: BM SG CL LLPF HG MG CV" (7 values; CV = cal-vocal)
  echo "${best##*params: }" > "$W/best.params"
  sleep 3
}

run_synth() {
  local W="$PW/synth"; mkdir -p "$W"
  if stage_done "$W/best.strudel" "$W/best.wav" "$W/best.cmp.json" "$W/best.params"; then
    log "stage 7 skip (synth): best render exists in $W"; return 0; fi
  local knobs="" knob_args=()
  if [ -s "$PW/sample-instrument/best.params" ]; then
    knobs="$(cat "$PW/sample-instrument/best.params")"
    # shellcheck disable=SC2086
    set -- $knobs
    knob_args=(--bass-mult "$1" --sub-gain "$2" --cal-lead "$3" --lead-lpf "$4" --hat-gain "$5" --master-gain "$6" --cal-vocal "${7:-1.0}")
    log "stage 7: synth uses the calibrated knobs: $knobs"
  else
    log "stage 7: no sample-instrument calibration found; synth uses the generator's cold-start defaults"
  fi
  local key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  # NOTE: no --env-correction here: the per-bar correction was measured on sample-instrument renders, not
  # synth (follow-up: measure a synth-mode correction in its own calibration pass).
  "$PY" "$SP/generate_dynamic_strudel.py" --stems-dir "$TRACK" --pack-dir "$PACK" \
    --bass-midi "$PACK/bass.mid" --lead-midi "$PACK/melodic.mid" --drums-json "$PACK/drums_bands.json" \
    --base-url "$BASE" --mode synth --bpm "$BPM" "${key_arg[@]}" --genre "$GENRE" --num-bars "$BARS" \
    --drum-mode bank --vocal-mode instrument "${knob_args[@]}" \
    --out "$W/best.strudel" >&2 || die "stage 7 synth generation failed"
  preflight_render
  log "stage 7: render synth through BlackHole (-d $DUR)"
  node "$REC" "$W/best.strudel" -o "$W/best.wav" -d "$DUR" >"$W/render.log" 2>&1
  [ -s "$W/best.wav" ] || die "synth render produced no wav (see $W/render.log)"
  "$PY" "$SP/compare_audio.py" "$TRACK/original.wav" "$W/best.wav" -d "$COMPARE_SECONDS" -j > "$W/best.cmp.json" 2>/dev/null \
    || die "stage 7 synth compare failed"
  echo "${knobs:-generator-defaults}" > "$W/best.params"
  sleep 3
}

# ---------------------------------------------------------------- stages 8-9: score, gate, promote
score_and_promote() {
  local mode="$1"; local W="$PW/$mode"; local STR="$W/best.strudel" WAV="$W/best.wav" CJ="$W/comparison.json"
  if stage_done "$W/result.json"; then log "stage 8-9 skip ($mode): result.json exists"; return 0; fi
  log "stage 8: score $mode (compare_audio -d $COMPARE_SECONDS --strudel)"
  local cmp_rc=0
  "$PY" "$SP/compare_audio.py" "$TRACK/original.wav" "$WAV" -d "$COMPARE_SECONDS" -j --strudel "$STR" -o "$CJ" \
    >"$W/compare.stdout" 2>"$W/compare.stderr" || cmp_rc=$?
  [ -s "$CJ" ] || die "stage 8 compare_audio wrote no comparison.json for $mode (rc=$cmp_rc, see $W/compare.stderr)"
  local edit_rc=0 gate_rc=0
  "$PY" "$SP/editability_check.py" "$STR" --json >"$W/editability.json" 2>&1 || edit_rc=$?
  # Spec 004 §2.3: a genre with no measured floor yet gets one from THIS run (measured − margin),
  # never typed by hand; brazilian_funk already has floors, so this is a no-op for it.
  local vdir_planned; vdir_planned="$("${PH[@]}" next-version "$TRACK")"   # the dir stage 9 will promote into
  "${PH[@]}" record-floor --thresholds "$ROOT/eval/thresholds.yaml" --mode "$mode" --genre "$GENRE" \
    --comparison "$CJ" --run "$(basename "$TRACK")/$(basename "$vdir_planned")" >&2 || log "stage 8: record-floor skipped/failed (see above)"
  "$PY" "$GATE" "$CJ" --genre "$GENRE" --mode "$mode" >"$W/gate.txt" 2>&1 || gate_rc=$?
  log "stage 8: $mode editability_rc=$edit_rc gate_rc=$gate_rc :: $(tr '\n' ' ' < "$W/gate.txt")"

  log "stage 9: promote $mode"
  local vdir env_arg=()
  [ -s "$W/best.env.json" ] && env_arg=(--env-correction "$W/best.env.json")
  vdir="$("${PH[@]}" promote --track-dir "$TRACK" --mode "$mode" --strudel "$STR" --wav "$WAV" --comparison "$CJ" \
    --generator "generate_dynamic_strudel.py --mode $mode (knobs from calibrate_dynamic.py: $(cat "$W/best.params"); genre $GENRE; bars $BARS)" \
    --render "$RENDER_NOTE; -d $DUR; pack served from $BASE/" \
    --compare "compare_audio.py -d $COMPARE_SECONDS --strudel (detector-stamped)" --vocal-mode instrument ${env_arg[@]+"${env_arg[@]}"})" \
    || die "stage 9 promote failed for $mode"
  "$PY" - "$W/result.json" "$mode" "$vdir" "$STR" "$WAV" "$CJ" "$edit_rc" "$gate_rc" "$W/gate.txt" <<'PYR'
import json, sys
out, mode, vdir, strudel, wav, cj, edit_rc, gate_rc, gate_txt = sys.argv[1:10]
meta = json.load(open(f"{vdir}/metadata.json"))
json.dump({
    "mode": mode, "version_dir": vdir, "strudel": f"{vdir}/output.strudel", "render": f"{vdir}/render.wav",
    "comparison": f"{vdir}/comparison.json", "editability": meta["editability"],
    "editability_rc": int(edit_rc), "gate_rc": int(gate_rc), "gate_verdict": "PASS" if int(gate_rc) == 0 else "FAIL",
    "gate_message": open(gate_txt).read().strip(),
    "similarity_overall": meta["similarity_overall"], "similarity_section_aware": meta["similarity_section_aware"],
    "frequency_balance": meta["frequency_balance"], "tempo_similarity": meta["tempo_similarity"],
}, open(out, "w"), indent=2)
PYR
}

for m in $(modes); do
  if [ -n "$FORCE" ]; then rm -f "$PW/$m/result.json" "$PW/$m/best.params"; fi
  case "$m" in sample-instrument) run_sample_instrument;; synth) run_synth;; esac
  score_and_promote "$m"
done

# ---------------------------------------------------------------- final JSON summary (stdout)
RESULTS=(); for m in $(modes); do RESULTS+=("$PW/$m/result.json"); done
"$PY" - "$TRACK" "$SLUG" "$BPM" "$KEY" "$GENRE" "$BARS" "$BASE" "$HOSTING" "${RESULTS[@]}" <<'PYS'
import json, sys
track, slug, bpm, key, genre, bars, base, hosting, *results = sys.argv[1:]
runs = [json.load(open(p)) for p in results]
print(json.dumps({
    "track_dir": track, "slug": slug, "bpm": float(bpm), "key": key, "genre": genre, "num_bars": int(bars),
    "hosting": hosting, "samples_base": base, "runs": runs,
    "all_gates_passed": all(r["gate_verdict"] == "PASS" for r in runs),
    "all_editable": all(r["editability"] == "pass" for r in runs),
}, indent=2))
PYS
