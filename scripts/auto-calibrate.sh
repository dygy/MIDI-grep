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
#     [--iters 6] [--bass-sound regime_bass --lead-sound regime_lead] \
#     [--dur 170] [--workdir <tmp>]
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
ITERS="6" DUR="170" BASS_SOUND="regime_bass" LEAD_SOUND="regime_lead"
WORK="${TMPDIR:-/tmp}/auto-calibrate.$$"
# starting knobs (generator defaults are sane cold-start values)
BM="0.40" SG="0.7" CL="1.0" LLPF="5000" HG="0.0" MG="0.6"

while [ $# -gt 0 ]; do case "$1" in
  --stems) STEMS="$2"; shift 2;; --base) BASE="$2"; shift 2;; --inst) INST="$2"; shift 2;;
  --bpm) BPM="$2"; shift 2;; --key) KEY="$2"; shift 2;; --genre) GENRE="$2"; shift 2;;
  --bars) BARS="$2"; shift 2;; --iters) ITERS="$2"; shift 2;; --dur) DUR="$2"; shift 2;;
  --bass-sound) BASS_SOUND="$2"; shift 2;; --lead-sound) LEAD_SOUND="$2"; shift 2;;
  --workdir) WORK="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 64;;
esac; done
[ -n "$STEMS" ] && [ -n "$BASE" ] || { echo "need --stems and --base" >&2; exit 64; }
PACK="$STEMS/sample_pack"
ORIG="$STEMS/original.wav"
[ -f "$ORIG" ] || ORIG="$PACK/originalfull.wav"
mkdir -p "$WORK"
echo "workdir: $WORK"

best=-1; best_params=""; best_tag=""
for i in $(seq 1 "$ITERS"); do
  TAG="cal$i"; STRU="$WORK/$TAG.strudel"; WAV="$WORK/$TAG.wav"; CJ="$WORK/$TAG.cmp.json"
  inst_arg=(); [ -n "$INST" ] && inst_arg=(--samples-url "$INST")
  key_arg=();  [ -n "$KEY" ]  && key_arg=(--key "$KEY")
  "$PY" "$GEN" --stems-dir "$STEMS" --pack-dir "$PACK" \
    --bass-midi "$PACK/bass.mid" --lead-midi "$PACK/melodic.mid" \
    --drums-json "$PACK/drums_bands.json" --base-url "$BASE" "${inst_arg[@]}" \
    --bass-sound "$BASS_SOUND" --lead-sound "$LEAD_SOUND" \
    --bpm "$BPM" "${key_arg[@]}" --genre "$GENRE" --num-bars "$BARS" \
    --sub-octave 1 --lead-hpf 95 --drum-mode extracted \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" --out "$STRU" >/dev/null 2>&1 \
    || { echo "[$TAG] GEN FAILED"; break; }

  pkill -9 -f "ffmpeg.*avfoundation" 2>/dev/null; sleep 1
  node "$REC" "$STRU" -o "$WAV" -d "$DUR" >"$WORK/$TAG.render.log" 2>&1
  [ -f "$WAV" ] || { echo "[$TAG] RENDER FAILED (see $WORK/$TAG.render.log)"; break; }

  "$PY" "$CMP" "$ORIG" "$WAV" -d 135 -j > "$CJ" 2>/dev/null || { echo "[$TAG] CMP FAILED"; break; }
  read -r OVER SECT < <("$PY" -c "import json,sys;d=json.load(open('$CJ'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
  echo "[$TAG] overall=$OVER section_aware=$SECT  (bm=$BM sg=$SG cl=$CL llpf=$LLPF hg=$HG mg=$MG)"

  # keep best by overall similarity
  better=$("$PY" -c "print(1 if $OVER>$best else 0)")
  if [ "$better" = "1" ]; then best="$OVER"; best_params="$BM $SG $CL $LLPF $HG $MG"; best_tag="$TAG";
    cp "$STRU" "$WORK/best.strudel"; cp "$WAV" "$WORK/best.wav"; cp "$CJ" "$WORK/best.cmp.json"; fi

  # calibrate -> next knobs
  CLI=$("$PY" "$CAL" --comparison "$CJ" \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" 2>/dev/null)
  # parse "--bass-mult X --sub-gain Y ..." into the loop vars
  set -- $CLI
  while [ $# -gt 0 ]; do case "$1" in
    --bass-mult) BM="$2";; --sub-gain) SG="$2";; --cal-lead) CL="$2";;
    --lead-lpf) LLPF="$2";; --hat-gain) HG="$2";; --master-gain) MG="$2";;
  esac; shift 2; done
done

echo "BEST: $best_tag overall=$best  params: $best_params"
echo "  strudel: $WORK/best.strudel"
echo "  wav:     $WORK/best.wav"
echo "  promote into the cache as a new vNNN (render/output.strudel/comparison + separate.py stems + generate_report.py)."
