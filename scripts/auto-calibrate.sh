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
#     [--dur 170] [--workdir <tmp>]
#
# --mode (spec 003 Slice 4) is passed straight through to the generator. `sample-instrument`
# (default) plays the pack's pitched instruments (bass/lead sounds default to regime_bass /
# regime_lead, drums = the extracted kit); `synth` emits no samples() at all — bass/lead
# sounds and the drum machine come from the sound_selector genre palette unless --bass-sound /
# --lead-sound are given explicitly, and drums take the --drum-mode bank path.
#
# Vocals are never generated. Every iteration is scored against <stems>/instrumental.wav
# (bass+drums+melodic, built by pipeline_helpers.py instrumental), not the full mix.
#
# Requires: BlackHole device + Multi-Output selected (see CLAUDE.md preflight), node recorder
# built (scripts/node/dist), the sample pack + bass/lead MIDI present in <stems>/sample_pack,
# and the instruments samples.json already hosted (R2) so Strudel can fetch it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tool paths are overridable via env (tests point them at stubs); defaults are the real ones.
PY="${AC_PY:-$ROOT/scripts/python/.venv/bin/python}"
GEN="${AC_GEN:-$ROOT/scripts/python/generate_dynamic_strudel.py}"
CAL="${AC_CAL:-$ROOT/scripts/python/calibrate_dynamic.py}"
CMP="${AC_CMP:-$ROOT/scripts/python/compare_audio.py}"
GATE="${AC_GATE:-$ROOT/scripts/python/gate_margin.py}"
RENDERER="${MIDIGREP_RENDERER:-offline}"   # offline (default): exact Strudel OfflineAudioContext render; blackhole: real-time capture
case "$RENDERER" in
  offline)   REC="${AC_REC:-$ROOT/scripts/node/dist/render-strudel-offline.js}" ;;
  blackhole) REC="${AC_REC:-$ROOT/scripts/node/dist/record-strudel-blackhole.js}" ;;
  *) echo "MIDIGREP_RENDERER=$RENDERER: expected offline|blackhole" >&2; exit 2 ;;
esac
NODE_BIN="${AC_NODE:-node}"; SETTLE="${AC_SETTLE_S:-3}"
PGREP="${AC_PGREP:-pgrep}"; WAIT_MAX="${AC_RECORDER_WAIT_S:-120}"   # tests: AC_PGREP=false, AC_RECORDER_WAIT_S=0
RC=0   # exit status: 70 when any iteration failed (GEN/RENDER/CMP FAILED, recorder wait abort); BEST is still printed

STEMS="" BASE="" INST="" BPM="136" KEY="" GENRE="brazilian_funk" BARS="78"
ENV_START=""; ITERS="6" DUR="170" BASS_SOUND="" LEAD_SOUND="" MODE="sample-instrument"
WORK="${TMPDIR:-/tmp}/auto-calibrate.$$"
# starting knobs (generator defaults are sane cold-start values)
BM="0.40" SG="0.7" CL="1.0" LLPF="5000" HG="0.0" MG="0.6" BL="0" LH="0"   # BL = bass low-pass override Hz, LH = lead high-pass Hz; 0 = generator/analysis default

while [ $# -gt 0 ]; do case "$1" in
  --stems) STEMS="$2"; shift 2;; --base) BASE="$2"; shift 2;; --inst) INST="$2"; shift 2;;
  --bpm) BPM="$2"; shift 2;; --key) KEY="$2"; shift 2;; --genre) GENRE="$2"; shift 2;;
  --bars) BARS="$2"; shift 2;; --iters) ITERS="$2"; shift 2;; --dur) DUR="$2"; shift 2;;
  --bass-sound) BASS_SOUND="$2"; shift 2;; --lead-sound) LEAD_SOUND="$2"; shift 2;;
  --vocal-mode) echo "--vocal-mode was removed: vocals are never generated" >&2; exit 64;;
  --mode) MODE="$2"; shift 2;;
  --workdir) WORK="$2"; shift 2;;
  --start) BL=0; LH=0; read -r BM SG CL LLPF HG MG BL LH EXTRA <<<"$2"; BL="${BL:-0}"; LH="${LH:-0}"   # resume from a previous BEST: "BM SG CL LLPF HG MG [BL [LH]]" (6, 7 or 8 fields)
    # The pre-"no vocals" format had a CV field after MG ("... MG CV [BL [LH]]"). A 9-field start is that old format;
    # 7/8 fields would silently shift CV into BL, but a real bass/lead cutoff is 0 or >= 20 Hz, a cal-vocal is ~1.
    [ -z "${EXTRA:-}" ] || { echo "--start: more than 8 fields - the old CV (cal-vocal) field was removed; drop the 7th value" >&2; exit 64; }
    for _v in "$BL" "$LH"; do awk -v v="$_v" 'BEGIN{exit !(v==0 || v>=20)}' || { echo "--start: BL/LH must be 0 or >= 20 Hz (got $_v) - old format with the removed CV field? drop the 7th value" >&2; exit 64; }; done
    shift 2;;
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
# reference = the INSTRUMENTAL (bass+drums+melodic), never the full mix with the singer
ORIG="${AC_REF:-$("$PY" "$ROOT/scripts/python/pipeline_helpers.py" instrumental "$STEMS")}" || { echo "cannot build $STEMS/instrumental.wav" >&2; exit 65; }   # AC_REF: tests point at a stub
# Chance control (same lookup as editable-pipeline.sh stage 8): a DIFFERENT same-genre reference track scored against
# this original by every iteration, so the gate margin / BEST selection include the chance constraint.
# pipeline_helpers.py control exits 0 = found, 3 = legitimately none (logged), anything else = the lookup failed.
CONTROL_ARG=()
CTRL_PATH="$("$PY" "$ROOT/scripts/python/pipeline_helpers.py" control --genre "$GENRE" --exclude-key "$(basename "$STEMS")" 2>/dev/null)" && CTRL_RC=0 || CTRL_RC=$?
case "$CTRL_RC" in
  0) CONTROL_ARG=(--control "$CTRL_PATH"); echo "chance control: $CTRL_PATH";;
  3) echo "chance control: none (no other $GENRE reference track)";;
  *) echo "control lookup FAILED (rc=$CTRL_RC) - fix eval/datasets/reference_tracks.yaml" >&2; exit 65;;
esac
mkdir -p "$WORK"
echo "workdir: $WORK"

best=-1; best_params=""; best_tag=""; best_gate="n/a"; : >"$WORK/iters.tsv"
rm -f "$WORK/best.env.json"   # never report a stale env from a previous run
PREV_ENV="${ENV_START:-}"   # env-correction JSON measured from the previous iteration (fed to the next generation); --env-start seeds it
for i in $(seq 1 "$ITERS"); do
  TAG="cal$i"; STRU="$WORK/$TAG.strudel"; WAV="$WORK/$TAG.wav"; CJ="$WORK/$TAG.cmp.json"
  inst_arg=(); [ -n "$INST" ] && inst_arg=(--samples-url "$INST")
  key_arg=();  [ -n "$KEY" ]  && key_arg=(--key "$KEY")
  env_arg=();  [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && env_arg=(--env-correction "$PREV_ENV")
  lh_arg=();   [ "$LH" -gt 0 ] 2>/dev/null && lh_arg=(--lead-hpf "$LH")   # 0 = generator default
  snd_arg=();  [ -n "$BASS_SOUND" ] && snd_arg+=(--bass-sound "$BASS_SOUND")
  [ -n "$LEAD_SOUND" ] && snd_arg+=(--lead-sound "$LEAD_SOUND")
  "$PY" "$GEN" --stems-dir "$STEMS" --pack-dir "$PACK" \
    --bass-midi "$PACK/bass.mid" --lead-midi "$PACK/melodic.mid" \
    --drums-json "$PACK/drums_bands.json" --base-url "$BASE" ${inst_arg[@]+"${inst_arg[@]}"} ${snd_arg[@]+"${snd_arg[@]}"} \
    --mode "$MODE" --bpm "$BPM" ${key_arg[@]+"${key_arg[@]}"} --genre "$GENRE" --num-bars "$BARS" \
    --drum-mode "$DRUM_MODE" \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" --bass-lpf "$BL" ${lh_arg[@]+"${lh_arg[@]}"} ${env_arg[@]+"${env_arg[@]}"} --out "$STRU" \
    >"$WORK/$TAG.gen.json" 2>"$WORK/$TAG.gen.log" \
    || { echo "[$TAG] GEN FAILED"; RC=70; break; }
  # the generator reports the EFFECTIVE bass cutoff + its bounds (analysis value, top bass note);
  # the calibrator needs them to move the cutoff by measurement (0 = unreported -> lever off)
  if ! read -r EBL EBLMAX EBHZ ELH ELHZ < <("$PY" -c "import json;d=json.load(open('$WORK/$TAG.gen.json'));print(d['bass_lpf'],d['bass_lpf_analysis'],d['bass_note_max_hz'],d['lead_hpf'],d['lead_note_min_hz'])" 2>/dev/null); then
    echo "[$TAG] WARNING: cannot parse the generator summary $WORK/$TAG.gen.json - bass-lpf/lead-hpf levers are OFF this step"
    EBL=0 EBLMAX=0 EBHZ=0 ELH=0 ELHZ=0
  fi

  # Never pkill -9 the capture right before a render: on 2026-10-09 that left BlackHole/avfoundation
  # wedged (every following capture produced no file). Wait for any live recorder to finish instead.
  # Bounded + anchored (review #2): wait at most 120 s for a LIVE recorder/capture, matched on the
  # actual invocations (not any process whose command line mentions the file name).
  if [ "$RENDERER" = blackhole ]; then   # the offline renderer has no capture device to wait for
    waited=0
    while "$PGREP" -f "node .*record-strudel-blackhole\.js|ffmpeg .* -f avfoundation" >/dev/null; do
      [ "$waited" -ge "$WAIT_MAX" ] && { echo "[$TAG] a recorder/capture has been running for ${WAIT_MAX} s — stale? (pgrep -fl 'record-strudel|avfoundation'); aborting"; RC=70; break 2; }
      sleep 2; waited=$((waited+2))
    done; sleep "$SETTLE"
  fi
  "$NODE_BIN" "$REC" "$STRU" -o "$WAV" -d "$DUR" >"$WORK/$TAG.render.log" 2>&1
  [ -f "$WAV" ] || { echo "[$TAG] RENDER FAILED (see $WORK/$TAG.render.log)"; RC=70; break; }

  "$PY" "$CMP" "$ORIG" "$WAV" -d 135 -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$CJ" 2>/dev/null || { echo "[$TAG] CMP FAILED"; RC=70; break; }
  read -r OVER SECT < <("$PY" -c "import json,sys;d=json.load(open('$CJ'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
  echo "[$TAG] mode=$MODE overall=$OVER section_aware=$SECT  (bm=$BM sg=$SG cl=$CL llpf=$LLPF hg=$HG mg=$MG bl=$BL eff_bl=$EBL lh=$LH eff_lh=$ELH)"

  # Record this iteration; BEST is chosen AFTER the loop (see below) so the ordering is one consistent rule.
  # Gate margin = smallest margin to the eval-gate floors (eval/thresholds.yaml via eval/gate.py, per
  # --mode/--genre); "n/a" when the gate cannot be read for this iteration.
  if GATE_OUT=$("$PY" "$GATE" --comparison "$CJ" --genre "$GENRE" --mode "$MODE" 2>"$WORK/$TAG.gate.log"); then
    read -r MARG _ <<<"$GATE_OUT"
  else
    echo "[$TAG] WARNING: gate margin unavailable ($(tr '\n' ' ' <"$WORK/$TAG.gate.log"))"
    MARG="n/a"; GATE_OUT="n/a"
  fi
  echo "[$TAG] gate: $GATE_OUT"
  ENVUSED="-"; if [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ]; then cp "$PREV_ENV" "$WORK/$TAG.envused.json"; ENVUSED="$WORK/$TAG.envused.json"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$OVER" "$MARG" "$GATE_OUT" "$BM $SG $CL $LLPF $HG $MG $BL $LH" "$ENVUSED" >>"$WORK/iters.tsv"

  # calibrate -> next knobs
  ENV_OUT="$WORK/$TAG.env.json"
  envin_arg=(); [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && envin_arg=(--env-correction-in "$PREV_ENV")
  CLI=$("$PY" "$CAL" --comparison "$CJ" --env-correction-out "$ENV_OUT" --bpm "$BPM" --bars "$BARS" ${envin_arg[@]+"${envin_arg[@]}"} \
    --bass-mult "$BM" --sub-gain "$SG" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" \
    --mode "$MODE" --bass-lpf "$EBL" --bass-lpf-max "$EBLMAX" --bass-note-max-hz "$EBHZ" \
    --lead-hpf "$ELH" --lead-note-min-hz "$ELHZ" 2>"$WORK/$TAG.cal.log")
  # parse "--bass-mult X --sub-gain Y ..." into the loop vars
  set -- $CLI
  while [ $# -gt 0 ]; do case "$1" in
    --bass-mult) BM="$2";; --sub-gain) SG="$2";; --cal-lead) CL="$2";;
    --lead-lpf) LLPF="$2";; --hat-gain) HG="$2";; --master-gain) MG="$2";; --bass-lpf) BL="$2";; --lead-hpf) LH="$2";;
  esac; shift 2; done
  [ -s "$ENV_OUT" ] && PREV_ENV="$ENV_OUT"
done

# BEST = the iteration with the largest minimum margin to the gate floors (a render that passes both beats one
# that does not; among failures the one nearest to passing). If the gate was unreadable for ANY iteration the
# margins are not comparable, so ALL iterations are ranked by overall instead (one consistent ordering).
SEL=$("$PY" - "$WORK/iters.tsv" <<'PYEOF'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
if rows:
    gated = all(r[2] != "n/a" for r in rows)
    key = (lambda r: (float(r[2]), float(r[1]))) if gated else (lambda r: (float(r[1]),))
    b = max(rows, key=key)
    print("\t".join([b[0], b[1], b[3], b[4], b[5], "gate" if gated else "overall"]))
PYEOF
)
if [ -n "$SEL" ]; then
  IFS=$'\t' read -r best_tag best best_gate best_params best_envused SEL_BY <<<"$SEL"
  if [ "$SEL_BY" = "overall" ]; then
    echo "WARNING: gate margin was unreadable for at least one iteration - BEST selected by overall similarity across ALL iterations (floors not applied)"
    best_gate="n/a (gate unreadable for >=1 iteration)"
  fi
  rm -f "$WORK/best.env.json"; [ "$best_envused" != "-" ] && [ -s "$best_envused" ] && cp "$best_envused" "$WORK/best.env.json"
  cp "$WORK/$best_tag.strudel" "$WORK/best.strudel"; cp "$WORK/$best_tag.wav" "$WORK/best.wav"; cp "$WORK/$best_tag.cmp.json" "$WORK/best.cmp.json"
fi
echo "BEST: $best_tag overall=$best  params: $best_params"
echo "  gate:    $best_gate  (margin PASS/FAIL overall section_aware floor sa_floor; BEST = largest minimum margin to the floors)"
echo "  strudel: $WORK/best.strudel"
[ -s "$WORK/best.env.json" ] && echo "  env:     $WORK/best.env.json  (per-bar bass/lead/master correction used by the best render; pass --env-correction)"
echo "  wav:     $WORK/best.wav"
echo "  promote + report: scripts/editable-pipeline.sh --title <track> (stages 9-10), or pipeline_helpers.py promote then 'pipeline_helpers.py report --track-dir <track> --version vNNN --seconds 135 --calib-workdir $WORK'."
[ "$RC" -eq 0 ] || echo "FAILED: at least one iteration did not complete (exit $RC); BEST above covers only the iterations that ran" >&2
exit "$RC"
