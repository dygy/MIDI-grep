#!/usr/bin/env bash
# Auto-calibration loop for the editable dynamic-Strudel generator.
#
# Closes the mix-tuning loop that used to be done by hand: generate -> render (BlackHole) ->
# compare -> calibrate -> repeat, where each round's generator knobs come from
# `calibrate_dynamic.py` reacting to the LAST render's measured `comparison.json`. Every knob
# delta is a damped proportional correction of an observed band/centroid ratio — no hardcoded
# per-track values (CLAUDE.md "ZERO HARDCODING"). Keeps the best render across iterations.
#
# Proven on "Regime CLT" (brazilian_funk): overall 88.7% -> 92.4%, freq balance 86% -> 95%,
# section-aware 82% -> 89% over ~6 iterations from a cold start.
#
# Usage:
#   scripts/auto-calibrate.sh \
#     --stems  ".cache/stems/<track>" \
#     --base   "https://<r2>/midi-grep/<id>" \
#     --inst   "https://<r2>/midi-grep/<id>/instruments/samples.json" \
#     --bpm 136 --key "C# minor" --genre brazilian_funk --bars 78 \
#     [--iters 6] [--mode sample-instrument|synth] \
#     [--bass-sound regime_bass --lead-sound regime_lead] \
#     [--vocal-mode instrument|chops|texture|none] [--dur 170] [--workdir <tmp>]
#
# --mode (spec 003 Slice 4) is passed straight through to the generator. `sample-instrument`
# (default) plays the pack's pitched instruments (bass/lead sounds default to regime_bass /
# regime_lead, drums = the extracted kit); `synth` emits no samples() at all — bass/lead/vocal
# sounds and the drum machine come from the sound_selector genre palette unless --bass-sound /
# --lead-sound are given explicitly, and drums take the --drum-mode bank path.
#
# The vocal voice (spec 003 Slice 3) is editable data by default (--vocal-mode instrument) and is
# balanced by the `cal-vocal` lever: calibrate_dynamic.py reads the vocal stem RMS ratio from a
# stem_comparison.json next to the comparison when one exists, else the high_mid band proxy.
#
# Requires: BlackHole device + Multi-Output selected (see CLAUDE.md preflight), node recorder
# built (scripts/node/dist), the sample pack + bass/lead MIDI present in <stems>/sample_pack,
# and the instruments samples.json already hosted (R2) so Strudel can fetch it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/scripts/python/.venv/bin/python"
GEN="$ROOT/scripts/python/generate_dynamic_strudel.py"
CAL="$ROOT/scripts/python/calibrate_dynamic.py"
CMP="$ROOT/scripts/python/compare_audio.py"
REC="$ROOT/scripts/node/dist/record-strudel-blackhole.js"

STEMS="" BASE="" INST="" BPM="136" KEY="" GENRE="brazilian_funk" BARS="78"
ENV_START=""; ITERS="6" DUR="170" BASS_SOUND="" LEAD_SOUND="" VOCAL_MODE="instrument" MODE="sample-instrument"
WORK="${TMPDIR:-/tmp}/auto-calibrate.$$"
# starting knobs (generator defaults are sane cold-start values)
BM="0.40" SG="0.7" CL="1.0" LLPF="5000" HG="0.0" MG="0.6" CV="1.0"

while [ $# -gt 0 ]; do case "$1" in
  --stems) STEMS="$2"; shift 2;; --base) BASE="$2"; shift 2;; --inst) INST="$2"; shift 2;;
  --bpm) BPM="$2"; shift 2;; --key) KEY="$2"; shift 2;; --genre) GENRE="$2"; shift 2;;
  --bars) BARS="$2"; shift 2;; --iters) ITERS="$2"; shift 2;; --dur) DUR="$2"; shift 2;;
  --bass-sound) BASS_SOUND="$2"; shift 2;; --lead-sound) LEAD_SOUND="$2"; shift 2;;
  --vocal-mode) VOCAL_MODE="$2"; shift 2;;
  --mode) MODE="$2"; shift 2;;
  --workdir) WORK="$2"; shift 2;;
  --start) read -r BM SG CL LLPF HG MG CV <<<"$2"; shift 2;;   # resume from a previous BEST: "BM SG CL LLPF HG MG CV"
  --env-start) ENV_START="$2"; shift 2;;                        # per-bar env-correction JSON to apply from iteration 1
  *) echo "unknown arg: $1" >&2; exit 64;;
esac; done
[ -n "$STEMS" ] && [ -n "$BASE" ] || { echo "need --stems and --base" >&2; exit 64; }
case "$MODE" in sample-instrument|synth) ;; *) echo "--mode must be sample-instrument or synth" >&2; exit 64;; esac
# sample-instrument: the hosted pitched instruments + extracted kit (v023 setup). synth: leave
# the sounds unset so the generator derives them from the genre palette; drums ride a bank.
if [ "$MODE" = "sample-instrument" ]; then
  [ -n "$BASS_SOUND" ] || BASS_SOUND="regime_bass"; [ -n "$LEAD_SOUND" ] || LEAD_SOUND="regime_lead"
  DRUM_MODE="extracted"
else
  DRUM_MODE="bank"
fi
PACK="$STEMS/sample_pack"
ORIG="$STEMS/original.wav"
[ -f "$ORIG" ] || ORIG="$PACK/originalfull.wav"
mkdir -p "$WORK"
echo "workdir: $WORK"

best=-1; best_params=""; best_tag=""
rm -f "$WORK/best.env.json"   # never report a stale env from a previous run
PREV_ENV="${ENV_START:-}"   # env-correction JSON measured from the previous iteration (fed to the next generation); --env-start seeds it
for i in $(seq 1 "$ITERS"); do
  TAG="cal$i"; STRU="$WORK/$TAG.strudel"; WAV="$WORK/$TAG.wav"; CJ="$WORK/$TAG.cmp.json"
  inst_arg=(); [ -n "$INST" ] && inst_arg=(--samples-url "$INST")
  key_arg=();  [ -n "$KEY" ]  && key_arg=(--key "$KEY")
  env_arg=();  [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && env_arg=(--env-correction "$PREV_ENV")
  snd_arg=();  [ -n "$BASS_SOUND" ] && snd_arg+=(--bass-sound "$BASS_SOUND")
  [ -n "$LEAD_SOUND" ] && snd_arg+=(--lead-sound "$LEAD_SOUND")
  "$PY" "$GEN" --stems-dir "$STEMS" --pack-dir "$PACK" \
    --bass-midi "$PACK/bass.mid" --lead-midi "$PACK/melodic.mid" \
    --drums-json "$PACK/drums_bands.json" --base-url "$BASE" "${inst_arg[@]}" "${snd_arg[@]}" \
    --mode "$MODE" --bpm "$BPM" "${key_arg[@]}" --genre "$GENRE" --num-bars "$BARS" \
    --drum-mode "$DRUM_MODE" --vocal-mode "$VOCAL_MODE" \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" --cal-vocal "$CV" "${env_arg[@]}" --out "$STRU" >/dev/null 2>&1 \
    || { echo "[$TAG] GEN FAILED"; break; }

  # Never pkill -9 the capture right before a render: on 2026-10-09 that left BlackHole/avfoundation
  # wedged (every following capture produced no file). Wait for any live recorder to finish instead.
  # Bounded + anchored (review #2): wait at most 120 s for a LIVE recorder/capture, matched on the
  # actual invocations (not any process whose command line mentions the file name).
  waited=0
  while pgrep -f "node .*record-strudel-blackhole\.js|ffmpeg .* -f avfoundation" >/dev/null; do
    [ "$waited" -ge 120 ] && { echo "[$TAG] a recorder/capture has been running for 120 s — stale? (pgrep -fl 'record-strudel|avfoundation'); aborting"; exit 75; }
    sleep 2; waited=$((waited+2))
  done; sleep 3
  node "$REC" "$STRU" -o "$WAV" -d "$DUR" >"$WORK/$TAG.render.log" 2>&1
  [ -f "$WAV" ] || { echo "[$TAG] RENDER FAILED (see $WORK/$TAG.render.log)"; break; }

  "$PY" "$CMP" "$ORIG" "$WAV" -d 135 -j > "$CJ" 2>/dev/null || { echo "[$TAG] CMP FAILED"; break; }
  read -r OVER SECT < <("$PY" -c "import json,sys;d=json.load(open('$CJ'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
  echo "[$TAG] mode=$MODE overall=$OVER section_aware=$SECT  (bm=$BM sg=$SG cl=$CL llpf=$LLPF hg=$HG mg=$MG cv=$CV)"

  # keep best by overall similarity
  better=$("$PY" -c "print(1 if $OVER>$best else 0)")
  if [ "$better" = "1" ]; then best="$OVER"; best_params="$BM $SG $CL $LLPF $HG $MG $CV"; best_tag="$TAG"
    rm -f "$WORK/best.env.json"; [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && cp "$PREV_ENV" "$WORK/best.env.json"
    cp "$STRU" "$WORK/best.strudel"; cp "$WAV" "$WORK/best.wav"; cp "$CJ" "$WORK/best.cmp.json"; fi

  # calibrate -> next knobs
  ENV_OUT="$WORK/$TAG.env.json"
  envin_arg=(); [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && envin_arg=(--env-correction-in "$PREV_ENV")
  CLI=$("$PY" "$CAL" --comparison "$CJ" --env-correction-out "$ENV_OUT" --bpm "$BPM" --bars "$BARS" "${envin_arg[@]}" \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" --cal-vocal "$CV" 2>"$WORK/$TAG.cal.log")
  # parse "--bass-mult X --sub-gain Y ..." into the loop vars
  set -- $CLI
  while [ $# -gt 0 ]; do case "$1" in
    --bass-mult) BM="$2";; --sub-gain) SG="$2";; --cal-lead) CL="$2";;
    --lead-lpf) LLPF="$2";; --hat-gain) HG="$2";; --master-gain) MG="$2";; --cal-vocal) CV="$2";;
  esac; shift 2; done
  [ -s "$ENV_OUT" ] && PREV_ENV="$ENV_OUT"
done

echo "BEST: $best_tag overall=$best  params: $best_params"
echo "  strudel: $WORK/best.strudel"
[ -s "$WORK/best.env.json" ] && echo "  env:     $WORK/best.env.json  (per-bar bass/lead/master correction used by the best render; pass --env-correction)"
echo "  wav:     $WORK/best.wav"
echo "  promote into the cache as a new vNNN (render/output.strudel/comparison + separate.py stems + generate_report.py)."
