#!/usr/bin/env bash
# editable-pipeline.sh — one command from a YouTube link to editable Strudel, scored on real playback.
#
# Implements spec 004 technical-considerations §2.1 stages 1-10. Each stage skips when its output
# already exists (use --force to redo). Every number that varies per track (bpm, key, genre, bars,
# generator knobs) is READ from the track's metadata.json / produced by calibrate_dynamic.py via
# scripts/auto-calibrate.sh — this driver sets no per-track constant.
#
# Stages:
#   1 stems+analysis   ./bin/midi-grep extract --url U --render none --iterate 0
#   2 MIDI per stem    transcribe.py {bass,melodic}.wav -> sample_pack/*.mid (vocals are never transcribed or generated)
#   3 drum onsets      detect_drums_bands.py drums.wav --bpm B -> sample_pack/drums_bands.json
#   4 sample pack      build_sample_pack.py
#   5 instruments      midi-grep generative train (bass, lead) + build_instruments.py
#   6 host             upload_r2.py (--r2) | local CORS server on :5555 (--local)
#   7 generate+calib.  auto-calibrate.sh (sample-instrument), then auto-calibrate.sh --mode synth
#                      (start = the sample-instrument BEST knobs when present; own env correction)
#   8 score + gate     compare_audio.py -d <whole-bar seconds> --strudel ; eval/gate.py ; editability_check.py
#   9 promote          pipeline_helpers.py promote -> vNNN/{output.strudel,render.wav,comparison.json,metadata.json}
#  10 report           pipeline_helpers.py report -> vNNN/{report.html,comparison.png,chart_*.png,render_*.wav,
#                      stem_comparison.json,iterations.json,iterations/}; Demucs on the render + each calibration
#                      iteration render (~2 min each). A missing/failed report is an ERROR, never a skip.
#
# Usage:
#   scripts/editable-pipeline.sh --url <youtube> [--title "<cache folder title>"] [--prefix <slug>]
#       [--genre <g>] [--mode sample-instrument|synth|both] [--iters N] [--synth-iters N] [--dur <seconds>]
#       [--r2 | --local] [--no-iteration-stems] [--force]
#
#   --url      YouTube link (required unless --title names an already-extracted track dir)
#   --title    locate an already-extracted track dir .cache/stems/<title> (skips stage 1 when complete)
#   --prefix   slug for hosting/model names (default: kebab-case of the track title)
#   --genre    override the genre in metadata.json (the override is logged)
#   --mode     which pieces to build (default both)
#   --iters    cap on sample-instrument calibration iterations (default: calibrate_loop.py MAX_ITERS; the loop
#              stops earlier after PATIENCE steps without a beyond-noise improvement; best iteration is kept)
#   --synth-iters  synth calibration iterations (default = --iters)
#   --dur      render seconds per render (default: the whole bars of the track, bars*240/bpm - never past the
#              song end, so the pattern cannot loop back to bar 0)
#   --r2       host on the project's public R2 bucket (default); --local serves on localhost:5555
#   --no-iteration-stems  stage 10: skip Demucs on each calibration iteration (report keeps the mixes only)
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
RENDERER_PY="$SP/renderer.py"   # the one place that reads MIDIGREP_RENDERER (default offline | blackhole); bad value -> exit 2
RENDERER="$("$PY" "$RENDERER_PY" name)" || exit 2
REC="$("$PY" "$RENDERER_PY" script)" || exit 2
BUSY_PATTERN="$("$PY" "$RENDERER_PY" busy-pattern)" || exit 2
GATE="$ROOT/eval/gate.py"
CACHE="$ROOT/.cache/stems"
R2_BUCKET_NAME="4cast"
R2_PUBLIC="https://pub-56831423fee34641805da07cfdaf6812.r2.dev"
R2_ACCOUNT="503e92d7d95838d80c33802d3274f284"
LOCAL_PORT="5555"
COMPARE_SECONDS=""   # derived below from the track: the seconds its generated bars cover (pipeline_helpers.py render-seconds)
if [ "$RENDERER" = offline ]; then
  RENDER_NOTE="render-strudel-offline.js (@strudel/web 1.3.0 OfflineAudioContext, chunked, polyphony 4096)"
else
  RENDER_NOTE="record-strudel-blackhole.js RAW capture + uniform speed correction (ffmpeg wallclock restamp)"
fi

URL="" TITLE="" PREFIX="" GENRE_OVERRIDE="" MODE="both" ITERS="" SYNTH_ITERS="" DUR="" HOSTING="r2" FORCE="" ITER_STEMS_ARG=()

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
  --synth-iters) SYNTH_ITERS="${2:?--synth-iters needs a value}"; shift 2;;
  --dur) DUR="${2:?--dur needs a value}"; shift 2;;
  --r2) HOSTING="r2"; shift;;
  --local) HOSTING="local"; shift;;
  --no-iteration-stems) ITER_STEMS_ARG=(--no-iteration-stems); shift;;
  --force) FORCE="1"; shift;;
  -h|--help) usage; exit 0;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 64;;
esac; done

SYNTH_ITERS="${SYNTH_ITERS:-$ITERS}"
case "$MODE" in sample-instrument|synth|both) ;; *) die "--mode must be sample-instrument, synth or both";; esac
[ -n "$URL" ] || [ -n "$TITLE" ] || die "need --url (or --title of an already-extracted track)"
[ -x "$PY" ] || die "python venv missing: $PY"

modes() { if [ "$MODE" = both ]; then echo "sample-instrument synth"; else echo "$MODE"; fi; }
# stage_done <file...>: true when every file exists and --force was not given
stage_done() { [ -z "$FORCE" ] || return 1; local f; for f in "$@"; do [ -s "$f" ] || return 1; done; }
jget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));v=d.get(sys.argv[2]);print('' if v is None else v)" "$1" "$2"; }
# built_at <stamp>: the bpm the stage output was built with (recorded in $PW/<stage>.bpm) equals the track's CURRENT bpm.
# A forced re-stamp (stage 1b) can change the bpm; outputs built at another one - or with no record - are rebuilt.
built_at() { [ "$(cat "$1" 2>/dev/null)" = "$BPM" ]; }
# fresh <stamp> <path...>: nothing under the paths is newer than the stamp file (pipeline_helpers.py newer)
fresh() { [ "$("${PH[@]}" newer "$@" 2>/dev/null)" = fresh ]; }
# params_ok <best.params>: in the CURRENT calibration format (6-8 fields). A pre-"no vocals" best.params has 9 (a removed
# CV field) and makes `auto-calibrate.sh --start` exit 64; it is STALE, not migrated: the best.wav/comparison beside it
# came from the old recorder and the old score, so dropping the CV field would only keep a result nobody can trust.
params_ok() { [ "$("${PH[@]}" params-valid "$1" 2>/dev/null)" = ok ]; }
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
# Also re-stamp a track whose bpm predates the phase-coherence refinement (no bpm_method).
if [ -z "$(jget "$META" bpm)" ] || [ -z "$(jget "$META" bpm_method)" ] || [ -n "$FORCE" ]; then
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
# The scoring REFERENCE is the INSTRUMENTAL (bass+drums+melodic): vocals are never generated, so the singer is not
# part of the target. Plain sample-aligned sum, no normalisation except clipping protection (pipeline_helpers.py).
INSTRUMENTAL="$("${PH[@]}" instrumental "$TRACK")" || die "cannot build instrumental.wav from the bass/drums/melodic stems"
if [ -n "$GENRE_OVERRIDE" ]; then
  log "GENRE OVERRIDE: detector said '${GENRE:-none}', using '$GENRE_OVERRIDE' (--genre)"
  GENRE="$GENRE_OVERRIDE"
fi
[ -n "$GENRE" ] || die "no genre in metadata.json and no --genre given"
SLUG="${PREFIX:-$("${PH[@]}" slug "${TRACK_TITLE:-$(basename "$TRACK")}")}" || die "cannot derive a slug"
SND="${SLUG//-/_}"      # model / Strudel sound names avoid '-' (mini-notation); hosting prefix keeps kebab
BARS="$("${PH[@]}" num-bars "$DURATION" "$BPM")" || die "cannot derive num-bars"
SONG_SECONDS="$("${PH[@]}" render-seconds "$DURATION" "$BPM")" || die "cannot derive render seconds"
[ -n "$DUR" ] || DUR="$SONG_SECONDS"
COMPARE_SECONDS="$SONG_SECONDS"
log "track='$TRACK' slug=$SLUG bpm=$BPM key='$KEY' genre=$GENRE duration=${DURATION}s bars=$BARS render=${DUR}s compare=${COMPARE_SECONDS}s mode=$MODE host=$HOSTING"

PACK="$TRACK/sample_pack"; PW="$TRACK/.pipeline"; mkdir -p "$PACK" "$PW"
if [ "$HOSTING" = r2 ]; then HOST_BASE="$R2_PUBLIC/midi-grep"; else HOST_BASE="http://localhost:$LOCAL_PORT"; fi
BASE="$HOST_BASE/$SLUG"; INST_BASE="$BASE/instruments/"

# ---------------------------------------------------------------- stage 2: MIDI per stem
for pair in "bass.wav:bass.mid" "melodic.wav:melodic.mid"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if stage_done "$PACK/$dst"; then log "stage 2 skip: $dst exists"; continue; fi
  log "stage 2: transcribe $src"
  "$PY" "$SP/transcribe.py" "$TRACK/$src" "$PACK/$dst" >&2 || die "stage 2 transcribe $src failed"
  [ -s "$PACK/$dst" ] || die "stage 2 produced no $dst"
done

# ---------------------------------------------------------------- stage 3: drum onsets
if stage_done "$PACK/drums_bands.json" && built_at "$PW/drums_bands.bpm"; then log "stage 3 skip: drums_bands.json exists (built at bpm $BPM)"; else
  log "stage 3: drum band onsets (bpm $BPM)"
  rm -f "$PW/drums_bands.bpm"
  "$PY" "$SP/detect_drums_bands.py" "$TRACK/drums.wav" --bpm "$BPM" --out "$PACK/drums_bands.json" >&2 \
    || die "stage 3 detect_drums_bands failed"
  echo "$BPM" > "$PW/drums_bands.bpm"
fi

# ---------------------------------------------------------------- stage 3b: bass register (f0 of the bass stem)
# bass_register.py measure is a no-op when $PW/bass_f0.json already matches the stem's content hash; the generator
# reads it for the bass window + per-note octave and falls back LOUDLY to the legacy typed window without it.
log "stage 3b: bass f0 / register"
"$PY" "$SP/bass_register.py" measure --stem "$TRACK/bass.wav" --out "$PW/bass_f0.json" >&2 \
  || die "stage 3b bass_register measure failed"

# ---------------------------------------------------------------- stage 4: sample pack
if stage_done "$PACK/pack.json" "$PACK/samples.json" "$PACK/strudel.json" && base_ok "$PACK/samples.json" "$BASE/" \
   && built_at "$PW/pack.bpm"; then
  log "stage 4 skip: sample pack exists for $BASE/ (built at bpm $BPM)"
else
  log "stage 4: build sample pack (bpm $BPM)"
  rm -f "$PW/pack.bpm"
  key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  "$PY" "$SP/build_sample_pack.py" --stems-dir "$TRACK" --out "$PACK" --bpm "$BPM" ${key_arg[@]+"${key_arg[@]}"} \
    --prefix "$SND" --base-url "$BASE/" >&2 || die "stage 4 build_sample_pack failed"   # SND: sound names never carry '-' 
  echo "$BPM" > "$PW/pack.bpm"
fi
[ -s "$PACK/drums/bd.wav" ] || die "stage 4 pack has no drums/ one-shots in $PACK/drums"

# ---------------------------------------------------------------- stage 5: instruments
for pair in "bass:bass.wav" "lead:melodic.wav"; do
  role="${pair%%:*}"; src="${pair##*:}"
  # a model trained before note-aligned grains (no grain_mode) is retrained
  if stage_done "$ROOT/models/${SND}_$role/metadata.json" && [ "$(jget "$ROOT/models/${SND}_$role/metadata.json" grain_mode)" = notes ]; then
    log "stage 5 skip: models/${SND}_$role trained"; continue; fi
  midi="$PACK/${src%.wav}.mid"   # the stem's transcribed notes (stage 2): one grain per note, length from the music
  log "stage 5: train granular ${SND}_$role from $src (note-aligned, $midi)"
  (cd "$ROOT" && ./bin/midi-grep generative train "$TRACK/$src" --name "${SND}_$role" --mode granular \
      --midi "$midi" --output "$ROOT/models") >&2 \
    || die "stage 5 train ${SND}_$role failed"
done
# the manifest is rebuilt when its inputs (the drum kit from stage 4, the models trained above) are newer than it
if stage_done "$PACK/instruments/samples.json" "$PACK/instruments/verification.json" && base_ok "$PACK/instruments/samples.json" "$INST_BASE" \
   && fresh "$PACK/instruments/samples.json" "$PACK/drums" "$ROOT/models/${SND}_bass" "$ROOT/models/${SND}_lead"; then
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
  # upload_r2.py is incremental: it hashes only what Strudel fetches (the manifests + the files they reference; analysis
  # files such as drums_bands.json are never served) and uploads only files whose hash differs from .uploaded.json.
  # No .uploaded.json (e.g. a legacy .uploaded base string), another base, or --force -> one full upload.
  R2_ARGS=(--pack-dir "$PACK" --prefix "midi-grep/$SLUG" --bucket "$R2_BUCKET_NAME" --public-base "$R2_PUBLIC" --backend wrangler)
  [ -n "$FORCE" ] && R2_ARGS+=(--full)
  if [ -z "$FORCE" ] && "$PY" "$SP/upload_r2.py" ${R2_ARGS[@]+"${R2_ARGS[@]}"} --check >/dev/null 2>&1 \
     && curl -fsI -o /dev/null "$INST_BASE""samples.json"; then
    log "stage 6 skip: served files unchanged since the last upload to $BASE"
  else
    log "stage 6: upload changed pack files to R2 ($R2_BUCKET_NAME/midi-grep/$SLUG)"
    CLOUDFLARE_ACCOUNT_ID="$R2_ACCOUNT" "$PY" "$SP/upload_r2.py" ${R2_ARGS[@]+"${R2_ARGS[@]}"} >&2 || die "stage 6 upload_r2 failed"
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
  local p; p="$(pgrep -f "$BUSY_PATTERN" || true)"
  [ -z "$p" ] || die "another record-strudel-blackhole process is running (pid $p); one render at a time"
}
preflight_render() {
  [ -f "$REC" ] || die "node renderer not built: (cd scripts/node && npm run build)"
  [ "$RENDERER" = blackhole ] || return 0   # offline: no device to guard or probe
  guard_no_recorder; probe_blackhole
}

# ---------------------------------------------------------------- stage 7: generate + calibrate
run_sample_instrument() {
  local W="$PW/sample-instrument"; mkdir -p "$W"
  if stage_done "$W/best.strudel" "$W/best.wav" "$W/best.cmp.json" "$W/best.params" && params_ok "$W/best.params"; then
    log "stage 7 skip (sample-instrument): best render exists in $W"; return 0; fi
  [ ! -s "$W/best.params" ] || params_ok "$W/best.params" || log "stage 7: $W/best.params is in the old (CV) format - stale, rerunning"
  rm -f "$W/result.json"   # a rerun invalidates what stages 8-9 derived from the old best render
  preflight_render
  log "stage 7: auto-calibrate sample-instrument (iters cap=${ITERS:-default})"
  key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  "$ROOT/scripts/auto-calibrate.sh" --stems "$TRACK" --base "$BASE" --inst "$INST_BASE""samples.json" \
    --bpm "$BPM" ${key_arg[@]+"${key_arg[@]}"} --genre "$GENRE" --bars "$BARS" ${ITERS:+--iters "$ITERS"} --dur "$DUR" \
    --bass-sound "${SND}_bass" --lead-sound "${SND}_lead" \
    --mode sample-instrument --workdir "$W" 2>&1 | tee "$W/calibrate.log" >&2
  local cal_rc="${PIPESTATUS[0]}"
  [ "$cal_rc" -eq 0 ] || die "sample-instrument auto-calibrate exited $cal_rc (see $W/calibrate.log)"
  local best; best="$(grep '^BEST:' "$W/calibrate.log" | tail -1)"
  [ -n "$best" ] && [ -s "$W/best.wav" ] || die "auto-calibrate produced no BEST render (see $W/calibrate.log)"
  # "BEST: calN overall=X  params: BM SG CL LLPF HG MG BL LH SU stem|off" (BL = bass-lpf, LH = lead high-pass Hz; 0 = generator default;
  # SU = sub octave up; the last token = the calibrator-chosen --bass-shape, carried into the stage-9 generator note below)
  echo "${best##*params: }" > "$W/best.params"
  sleep 3
}

run_synth() {
  local W="$PW/synth"; mkdir -p "$W"
  if stage_done "$W/best.strudel" "$W/best.wav" "$W/best.cmp.json" "$W/best.params" && params_ok "$W/best.params"; then
    log "stage 7 skip (synth): best render exists in $W"; return 0; fi
  [ ! -s "$W/best.params" ] || params_ok "$W/best.params" || log "stage 7: $W/best.params is in the old (CV) format - stale, rerunning"
  rm -f "$W/result.json"
  preflight_render
  # Start from the sample-instrument BEST knobs when present (6, 7 or 8 fields, passed to --start UNSPLIT);
  # otherwise the calibrator's cold-start defaults. The synth pass then runs its own calibration loop
  # (knobs + per-bar env correction measured on SYNTH renders), writing $W/best.env.json for stage 9.
  local start_arg=() knobs=""
  if [ -s "$PW/sample-instrument/best.params" ] && params_ok "$PW/sample-instrument/best.params"; then
    # Seed the magnitudes (fields 1-8) but NOT the sub-sine octave (9th): whether the sine layer sounds better an
    # octave up depends on the main layer (a recorded sample vs a sawtooth) - Regime CLT synth seeded with the sample's
    # su=1 scored 0.682 and FAILED the gate, while its own octave trial from su=0 reached 0.763. Synth re-trials it.
    knobs="$(cut -d' ' -f1-8 "$PW/sample-instrument/best.params")"
    start_arg=(--seed "$knobs")   # a FALLBACK: ClickHouse (this track's synth best, else the nearest track's) wins when it has one
    log "stage 7: synth calibration seed (used only when ClickHouse has no synth history for it): sample-instrument knobs $knobs"
  else
    log "stage 7: no sample-instrument calibration found; synth calibration starts from cold-start defaults"
  fi
  local key_arg=(); [ -n "$KEY" ] && key_arg=(--key "$KEY")
  log "stage 7: auto-calibrate synth (iters cap=${SYNTH_ITERS:-default})"
  "$ROOT/scripts/auto-calibrate.sh" --stems "$TRACK" --base "$BASE" \
    --bpm "$BPM" ${key_arg[@]+"${key_arg[@]}"} --genre "$GENRE" --bars "$BARS" ${SYNTH_ITERS:+--iters "$SYNTH_ITERS"} --dur "$DUR" \
    --mode synth --workdir "$W" ${start_arg[@]+"${start_arg[@]}"} 2>&1 | tee "$W/calibrate.log" >&2
  local cal_rc="${PIPESTATUS[0]}"
  [ "$cal_rc" -eq 0 ] || die "synth auto-calibrate exited $cal_rc (see $W/calibrate.log)"
  local best; best="$(grep '^BEST:' "$W/calibrate.log" | tail -1)"
  [ -n "$best" ] && [ -s "$W/best.wav" ] || die "synth auto-calibrate produced no BEST render (see $W/calibrate.log)"
  # "BEST: calN overall=X  params: BM SG CL LLPF HG MG BL LH SU stem|off" (the chosen bass shape is the last token)
  echo "${best##*params: }" > "$W/best.params"
  sleep 3
}

# ---------------------------------------------------------------- stage 10: report
# report_stage <mode> <vNNN dir>: charts + render stems + per-stem comparison + iteration history +
# report.html (idempotent per artifact). The mode's calibration workdir supplies the iteration history.
report_stage() {
  local mode="$1" vdir="$2" W="$PW/$1" gate_arg=()
  [ -n "$vdir" ] && [ -d "$vdir" ] || die "stage 10 ($mode): promoted version dir missing: '$vdir'"
  [ -s "$W/gate.txt" ] && gate_arg=(--gate-verdict "$(jget "$W/result.json" gate_verdict)" \
    --gate-message "$(tr '\n' ' ' < "$W/gate.txt")")
  log "stage 10: report $mode -> $vdir"
  "${PH[@]}" report --track-dir "$TRACK" --version "$(basename "$vdir")" --seconds "$COMPARE_SECONDS" \
    --calib-workdir "$W" ${ITER_STEMS_ARG[@]+"${ITER_STEMS_ARG[@]}"} ${FORCE:+--force} ${gate_arg[@]+"${gate_arg[@]}"} >&2 \
    || die "stage 10 report failed for $mode ($vdir)"
  [ -s "$vdir/report.html" ] || die "stage 10 produced no $vdir/report.html"
}

# ---------------------------------------------------------------- stages 8-9: score, gate, promote
score_and_promote() {
  local mode="$1"; local W="$PW/$mode"; local STR="$W/best.strudel" WAV="$W/best.wav" CJ="$W/comparison.json"
  if stage_done "$W/result.json"; then
    log "stage 8-9 skip ($mode): result.json exists"
    report_stage "$mode" "$(jget "$W/result.json" version_dir)"   # a run promoted before stage 10 existed (or whose report failed) still gets one
    return 0
  fi
  log "stage 8: score $mode (compare_audio -d $COMPARE_SECONDS --strudel)"
  # Chance control: a DIFFERENT same-genre reference track scored against this INSTRUMENTAL reference by the same function.
  local control control_arg=() control_rc=0
  control="$("${PH[@]}" control --genre "$GENRE" --exclude-key "$(basename "$TRACK")")" || control_rc=$?
  case "$control_rc" in
    0) control_arg=(--control "$control"); log "stage 8: chance control = $control";;
    3) log "stage 8: no control track (no other $GENRE reference track in the cache)";;
    *) die "stage 8: control lookup FAILED (rc=$control_rc) - fix eval/datasets/reference_tracks.yaml; not scoring without the chance check";;
  esac
  local cmp_rc=0
  "$PY" "$SP/compare_audio.py" "$INSTRUMENTAL" "$WAV" -d "$COMPARE_SECONDS" -j --strudel "$STR" -o "$CJ" \
    ${control_arg[@]+"${control_arg[@]}"} \
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
    --generator "generate_dynamic_strudel.py --mode $mode (knobs from calibrate_dynamic.py: $(cat "$W/best.params"); final --master-gain $([ -s "$W/best.master" ] && cat "$W/best.master" || echo "= the 6th knob"); genre $GENRE; bars $BARS)" \
    --render "$RENDER_NOTE; -d $DUR; pack served from $BASE/" \
    --compare "compare_audio.py -d $COMPARE_SECONDS --strudel vs instrumental.wav (detector-stamped)" ${env_arg[@]+"${env_arg[@]}"})" \
    || die "stage 9 promote failed for $mode"
  # calibration memory: the promoted version (knobs + scores + env) becomes a recorded row; a failure here never stops the run
  "$PY" "$SP/calibration_memory.py" record-promoted --track-dir "$TRACK" --version "$(basename "$vdir")" 2>&1 | grep -v '^  \[ClickHouse\]' >&2 \
    || log "stage 9: WARNING - could not record $(basename "$vdir") in ClickHouse"
  "$PY" - "$W/result.json" "$mode" "$vdir" "$STR" "$WAV" "$CJ" "$edit_rc" "$gate_rc" "$W/gate.txt" <<'PYR'
import json, sys
out, mode, vdir, strudel, wav, cj, edit_rc, gate_rc, gate_txt = sys.argv[1:10]
meta = json.load(open(f"{vdir}/metadata.json"))
json.dump({
    "mode": mode, "version_dir": vdir, "strudel": f"{vdir}/output.strudel", "render": f"{vdir}/render.wav",
    "comparison": f"{vdir}/comparison.json", "editability": meta["editability"],
    "editability_rc": int(edit_rc), "gate_rc": int(gate_rc), "gate_verdict": "PASS" if int(gate_rc) == 0 else "FAIL",
    "gate_message": open(gate_txt).read().strip(),
    "report": f"{vdir}/report.html",
    "similarity_overall": meta["similarity_overall"], "similarity_section_aware": meta["similarity_section_aware"],
    "frequency_balance": meta["frequency_balance"], "tempo_similarity": meta["tempo_similarity"],
}, open(out, "w"), indent=2)
PYR
  report_stage "$mode" "$vdir"
}

for m in $(modes); do
  if [ -n "$FORCE" ]; then rm -f "$PW/$m/result.json" "$PW/$m/best.params"; fi
  case "$m" in sample-instrument) run_sample_instrument;; synth) run_synth;; esac
  score_and_promote "$m"
done

# ---------------------------------------------------------------- final JSON summary (stdout)
RESULTS=(); for m in $(modes); do RESULTS+=("$PW/$m/result.json"); done
"$PY" - "$TRACK" "$SLUG" "$BPM" "$KEY" "$GENRE" "$BARS" "$BASE" "$HOSTING" ${RESULTS[@]+"${RESULTS[@]}"} <<'PYS'
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
