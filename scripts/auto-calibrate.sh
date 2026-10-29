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
#     [--iters N] [--mode sample-instrument|synth] \
#     [--bass-sound regime_bass --lead-sound regime_lead] \
#     [--dur <seconds>] [--workdir <tmp>] [--no-mix-dynamics] [--bass-shape stem|off] [--no-shape-trial] [--no-lead-fit] [--lead-fit-evals N]
#
# After BEST, a loudness stage (mix_dynamics.py) moves --master-gain onto the instrumental's RMS (the score is level-
# invariant, the loop cannot see loudness); --no-mix-dynamics skips it. The final gain is written to <workdir>/best.master.
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
# Loop control (scripts/python/calibrate_loop.py, noise = eval/thresholds.yaml calibration_noise): a step whose
# result is worse than the best so far beyond noise is reverted - the next step starts from the BEST knobs
# (and env correction) along the proposed direction with a shorter step derived from the observed loss; BEST is the
# largest minimum margin to the gate floors, margins within noise of each other counting as tied (higher overall
# wins, then section-aware); the loop stops after PATIENCE steps without a beyond-noise improvement, or at the cap
# (--iters N; default MAX_ITERS from calibrate_loop.py).
#
# Bass shape (generate_dynamic_strudel.py --bass-shape stem|off) is a scored choice, not a constant: a fixed default won on three of
# four pieces and lost on the fourth. After the loop, when the memory is on and this track+mode has not answered the question for the
# CURRENT code version, the BEST knobs (and env) are rendered ONCE with the other shape; the default (stem) is kept unless the other
# shape improves on it beyond calibration noise (calibrate_loop.py shape). The trial is recorded (kind 'shape') and a winner also as
# the run BEST, so the next run warm-starts with the chosen shape and does not re-test until the generator/calibrator code changes.
# --bass-shape X pins the shape (no trial); --no-shape-trial keeps the warm-started/default shape untested; AC_SHAPE_TRIAL=0 (tests).
# The chosen shape is the 10th token of the BEST params line ("... SU stem|off") and of --start.
#
# Synth lead fit (synth mode only; scripts/python/lead_synth.py): the palette lead is a fixed soundfont preset. After BEST (and the bass-shape
# trial) the lead is rendered as a plain oscillator whose lpf / filter-envelope depth / detune / gain are searched (Nelder-Mead, the REAL
# full-song mix score = mean(overall, section-aware), start = values measured on the melodic stem) for at most --lead-fit-evals renders
# (default 10). The fit is kept only if its overall beats the palette lead's by more than calibration noise AND its section-aware is no worse
# than the palette's - 0.005 (calibrate_loop.py leadfit); else the palette lead stays. The decision + vector are recorded (kind 'leadfit',
# lead_fit column, migration 4). Warm start: this track's vector for the current code version = ONE verify render; a vector from older code or
# from the nearest same-genre track = 3 renders; a recorded 'palette' answer for the current code version = no renders.
# --no-lead-fit / AC_LEAD_FIT=0 skip it. A kept fit is written to <workdir>/best.leadsynth.json and baked into best.strudel.
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
LOOP="${AC_LOOP:-$ROOT/scripts/python/calibrate_loop.py}"
MIXD="${AC_MIXD:-$ROOT/scripts/python/mix_dynamics.py}"
# Bass voice level (scripts/python/bass_solo.py): after each render the BASS channel is rendered alone and measured against the
# original bass stem, so the bass layers are set by the bass and not by band shares the kick also fills. AC_BSOLO=off skips it.
BSOLO="${AC_BSOLO:-$ROOT/scripts/python/bass_solo.py}"
# Melodic voice level + brightness (scripts/python/melodic_solo.py): same idea for LEAD+CHORDS against melodic.wav, setting
# cal-lead (absolute body level) and lead-lpf (the voices' own centroid) instead of whole-mix shares. AC_MSOLO=off skips it.
MSOLO="${AC_MSOLO:-$ROOT/scripts/python/melodic_solo.py}"
# Calibration memory (ClickHouse, scripts/python/calibration_memory.py): warm start, per-step records, learned step sizes.
# AC_MEM=off (or --no-memory) disables it; every failure inside it is logged loudly and the run continues cold/unrecorded.
# When the compare tool is stubbed (AC_CMP, the tests) and AC_MEM is not given, the memory is OFF: a stub run must never
# write its rows into the real database. Tests of the memory itself pass AC_MEM=<script> and MIDIGREP_CLICKHOUSE_DB=<tmp>.
[ -z "${AC_MEM:-}" ] && [ -n "${AC_CMP:-}" ] && AC_MEM=off
MEM="${AC_MEM:-$ROOT/scripts/python/calibration_memory.py}"; MEM_ON=1; [ "$MEM" = off ] && MEM_ON=0
# Renderer selection lives in ONE place (scripts/python/renderer.py: MIDIGREP_RENDERER, default offline = exact
# Strudel OfflineAudioContext render; blackhole = real-time capture). A bad value exits 2 there.
RENDERER_PY="${AC_RENDERER_PY:-$ROOT/scripts/python/renderer.py}"
RENDERER="$("$PY" "$RENDERER_PY" name)" || exit 2
REC="${AC_REC:-$("$PY" "$RENDERER_PY" script)}" || exit 2
BUSY_PATTERN="$("$PY" "$RENDERER_PY" busy-pattern)" || exit 2
NODE_BIN="${AC_NODE:-node}"; SETTLE="${AC_SETTLE_S:-3}"
PGREP="${AC_PGREP:-pgrep}"; WAIT_MAX="${AC_RECORDER_WAIT_S:-120}"   # tests: AC_PGREP=false, AC_RECORDER_WAIT_S=0
RC=0   # exit status: 70 when any iteration failed (GEN/RENDER/CMP FAILED, recorder wait abort); BEST is still printed

# ---- concurrency ------------------------------------------------------------------------------------------------------
# The renders of one calibration iteration (mix, bass solo, melodic solo) are independent: all three strudels come from the
# same generated $STRU. They run concurrently, as many at once as scripts/python/render_jobs.py derives from this machine
# (cores, free memory against one render's measured footprint; MIDIGREP_RENDER_JOBS overrides; 1 = the old sequential order).
# Every background child is tracked so that a failure, SIGINT or SIGTERM leaves nothing behind (the whole process tree of
# each child - node driver and its Chromium - is terminated and waited for).
RJPY="${AC_RENDER_JOBS_PY:-$ROOT/scripts/python/render_jobs.py}"
CHILD_PIDS=()
descendants() {   # descendants PID: every pid below PID in the process tree (one per line)
  ps -A -o pid= -o ppid= 2>/dev/null | awk -v root="$1" '{ n++; id[n]=$1; par[$1]=$2 }
    END { seen[root]=1; ch=1; while (ch) { ch=0; for (i=1;i<=n;i++) if (!(id[i] in seen) && (par[id[i]] in seen)) { seen[id[i]]=1; print id[i]; ch=1 } } }'
}
reap_children() {   # terminate + wait for every tracked background child and its descendants; idempotent
  local p d all=""
  for p in ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"}; do all="$all $p $(descendants "$p" | tr '\n' ' ')"; done
  [ -z "${all// /}" ] && { CHILD_PIDS=(); return 0; }
  for p in ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"}; do kill -TERM "$p" 2>/dev/null; done   # lanes first: they must not start a next job
  kill -TERM $all 2>/dev/null
  for p in ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"}; do wait "$p" 2>/dev/null; done
  for d in $all; do kill -0 "$d" 2>/dev/null && { sleep 0.3; kill -KILL $all 2>/dev/null; break; }; done   # a straggler that ignored TERM
  CHILD_PIDS=(); return 0
}
trap reap_children EXIT
trap 'reap_children; exit 130' INT
trap 'reap_children; exit 143' TERM HUP
RJ_LAST=""
render_jobs_for() {   # render_jobs_for N: the concurrency for N independent renders (logged when it changes); sets RJ
  local n="$1" out rc err
  if [ "$RENDERER" = blackhole ]; then RJ=1; return 0; fi   # the capture device records one render at a time
  err="$WORK/render_jobs.err"
  out="$("$PY" "$RJPY" jobs --independent "$n" 2>"$err")"; rc=$?
  if [ "$rc" -eq 2 ]; then cat "$err" >&2; exit 64; fi    # an invalid MIDIGREP_RENDER_JOBS is an error, never a silent default
  case "$out" in ''|*[!0-9]*) echo "render jobs: WARNING cannot derive the concurrency ($(tr '\n' ' ' <"$err")) - sequential" >&2; RJ=1; return 0;; esac
  RJ="$out"
  if [ "$RJ_LAST" != "$RJ/$n" ]; then echo "render jobs: $(cat "$err")"; RJ_LAST="$RJ/$n"; fi
}
render_one() {   # render_one STRU WAV LOG: the renderer's EXIT STATUS decides; on failure no WAV is left and LOG.rc holds the status
  local rrc=0; rm -f "$2" "$3.rc"
  "$NODE_BIN" "$REC" "$1" -o "$2" -d "$DUR" >"$3" 2>&1 || rrc=$?
  if [ "$rrc" -ne 0 ] || [ ! -f "$2" ]; then rm -f "$2"; echo "$rrc" >"$3.rc"; return 1; fi
  return 0
}

STEMS="" BASE="" INST="" BPM="136" KEY="" GENRE="brazilian_funk" BARS="78"
LEAD_FIT="${AC_LEAD_FIT:-1}"; LEAD_EVALS=10; LEAD_ARG=(); LSYN="${AC_LEAD_SYNTH:-$ROOT/scripts/python/lead_synth.py}"   # synth lead fit; see header
BSHAPE="stem"; SHAPE_PIN=0; SHAPE_TESTED=0; SHAPE_TRIAL="${AC_SHAPE_TRIAL:-1}"   # generator --bass-shape; see header
MIXDYN=1; ENV_START=""; ITERS=""; START_GIVEN=0; SEED=""; WARM_SOURCE="cold" DUR="" BASS_SOUND="" LEAD_SOUND="" MODE="sample-instrument"
WORK="${TMPDIR:-/tmp}/auto-calibrate.$$"
# starting knobs (generator defaults are sane cold-start values)
BM="0.40" SG="0.7" CL="1.0" LLPF="5000" HG="0.0" MG="0.6" BL="0" LH="0" SU="0"   # BL = bass low-pass override Hz, LH = lead high-pass Hz; 0 = generator/analysis default

while [ $# -gt 0 ]; do case "$1" in
  --stems) STEMS="$2"; shift 2;; --base) BASE="$2"; shift 2;; --inst) INST="$2"; shift 2;;
  --bpm) BPM="$2"; shift 2;; --key) KEY="$2"; shift 2;; --genre) GENRE="$2"; shift 2;;
  --bars) BARS="$2"; shift 2;; --iters) ITERS="$2"; shift 2;; --dur) DUR="$2"; shift 2;;
  --bass-sound) BASS_SOUND="$2"; shift 2;; --lead-sound) LEAD_SOUND="$2"; shift 2;;
  --vocal-mode) echo "--vocal-mode was removed: vocals are never generated" >&2; exit 64;;
  --mode) MODE="$2"; shift 2;;
  --no-mix-dynamics) MIXDYN=0; shift;;
  --workdir) WORK="$2"; shift 2;;
  --no-memory) MEM_ON=0; shift;;
  --bass-shape) case "$2" in stem|off) BSHAPE="$2"; SHAPE_PIN=1;; *) echo "--bass-shape must be stem or off" >&2; exit 64;; esac; shift 2;;
  --no-shape-trial) SHAPE_TRIAL=0; shift;;
  --no-lead-fit) LEAD_FIT=0; shift;;
  --lead-fit-evals) LEAD_EVALS="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;     # fallback knobs ("BM SG CL LLPF HG MG [BL [LH]]"): used only when the memory has neither this track nor a neighbour
  --start) START_GIVEN=1; BL=0; LH=0; SU=0; read -r BM SG CL LLPF HG MG BL LH SU SHP EXTRA <<<"$2"; BL="${BL:-0}"; LH="${LH:-0}"; SU="${SU:-0}"   # resume from a previous BEST: "BM SG CL LLPF HG MG [BL [LH [SU [stem|off]]]]" (6 to 10 fields; SU = sub-sine octave up 0/1; the 10th = bass shape)
    # The pre-"no vocals" format had a CV field after MG ("... MG CV [BL [LH]]"). A 9-field start is that old format;
    # 7/8 fields would silently shift CV into BL, but a real bass/lead cutoff is 0 or >= 20 Hz, a cal-vocal is ~1.
    [ -z "${EXTRA:-}" ] || { echo "--start: more than 10 fields - the old CV (cal-vocal) field was removed; drop the 7th value" >&2; exit 64; }
    if [ -n "${SHP:-}" ]; then case "$SHP" in stem|off) [ "$SHAPE_PIN" = 1 ] || BSHAPE="$SHP";; *) echo "--start: the 10th field is the bass shape stem|off (got $SHP)" >&2; exit 64;; esac; fi
    case "$SU" in 0|1) ;; *) echo "--start: SU (9th field) must be 0 or 1 (got $SU) - old format with the removed CV field?" >&2; exit 64;; esac
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
# Song seconds = what the BARS whole bars cover (bars*240/bpm, floored to the ms). Render length defaults to it (the
# caller's --dur wins) - rendering longer makes the looped pattern wrap back to bar 0 past the song end - and it is
# the compare window (compare_audio reads min(window, both files)).
SONG_S="$("$PY" -c 'import math, sys; print(f"{math.floor(int(sys.argv[1]) * 240 / float(sys.argv[2]) * 1000) / 1000:g}")' "$BARS" "$BPM")" || exit 2
[ -n "$DUR" ] || DUR="$SONG_S"
ITERS_GIVEN="$ITERS"
mkdir -p "$WORK"
echo "workdir: $WORK"

# memcall SUBCOMMAND ...: run a memory subcommand; its stderr (progress + WARNINGs) is shown, its failure never stops the run.
memcall() {
  [ "$MEM_ON" = 1 ] || return 0
  local out rc; out="$("$PY" "$MEM" "$@" 2>&1)"; rc=$?
  [ -z "$out" ] || printf '%s\n' "$out" | grep -v '^  \[ClickHouse\]' >&2
  [ "$rc" -eq 0 ] || echo "[memory] WARNING: '$1' exited $rc - continuing without it" >&2
  return 0
}
TRACK_KEY="$(basename "$STEMS")"; RUN_ID="$(date +%Y%m%dT%H%M%S).$$"
MEM_ARGS=(--stems "$STEMS" --mode "$MODE" --genre "$GENRE" --bpm "$BPM" --bars "$BARS")
[ -n "$KEY" ] && MEM_ARGS+=(--key "$KEY")
RESP_ARG=()
# Warm start: knobs (+ per-bar env correction) from ClickHouse - this track+mode's best, else the nearest recorded track's,
# else --seed, else the cold defaults above. An explicit --start / --env-start from the caller wins over the memory.
if [ "$START_GIVEN" = 0 ] && [ "$MEM_ON" = 1 ]; then
  WARM="$("$PY" "$MEM" warm-start "${MEM_ARGS[@]}" --track-key "$TRACK_KEY" --env-out "$WORK/warm.env.json" 2> >(grep -v '^  \[ClickHouse\]' >&2))" || WARM=""
  if [ -n "$WARM" ]; then
    IFS=$'\t' read -r WARM_SOURCE WARM_CAP WARM_KNOBS WARM_ENV WARM_DETAIL WARM_SHAPE WARM_SHAPE_TESTED <<<"$WARM"
    if [ "$WARM_SOURCE" != cold ]; then
      read -r BM SG CL LLPF HG MG BL LH SU <<<"$WARM_KNOBS"; BL="${BL:-0}"; LH="${LH:-0}"; SU="${SU:-0}"
      [ -z "$ENV_START" ] && [ "$WARM_ENV" != "-" ] && ENV_START="$WARM_ENV"
      if [ "$SHAPE_PIN" = 0 ]; then   # the recorded shape decision of this track+mode (a neighbour's is never copied: stem, untested)
        case "${WARM_SHAPE:-stem}" in stem|off) BSHAPE="${WARM_SHAPE:-stem}";; esac; SHAPE_TESTED="${WARM_SHAPE_TESTED:-0}"
      fi
    fi
    echo "warm start: $WARM_SOURCE - $WARM_DETAIL"
  else
    echo "[memory] WARNING: warm-start produced no answer - cold start" >&2
  fi
fi
if [ "$WARM_SOURCE" = cold ] && [ -n "$SEED" ]; then
  WARM_SOURCE=seed; read -r BM SG CL LLPF HG MG BL LH SU <<<"$SEED"; BL="${BL:-0}"; LH="${LH:-0}"; SU="${SU:-0}"
  echo "warm start: seed - caller's fallback knobs: $SEED"
fi
[ "$START_GIVEN" = 1 ] && WARM_SOURCE=explicit
# Budget from history: a warm start earns calibrate_loop.WARM_ITERS, anything else the full MAX_ITERS; --iters N always wins.
if [ -z "$ITERS_GIVEN" ]; then ITERS="$("$PY" "$LOOP" cap "$([ "$WARM_SOURCE" = explicit ] && echo cold || echo "$WARM_SOURCE")")" || exit 2; fi
echo "start: $WARM_SOURCE; knobs: $BM $SG $CL $LLPF $HG $MG $BL $LH $SU; bass shape: $BSHAPE (tested for this code version: $SHAPE_TESTED, pinned: $SHAPE_PIN); iteration cap: $ITERS"
MEM_ARGS+=(--warm-source "$WARM_SOURCE")
if [ "$MEM_ON" = 1 ]; then   # learned per-knob damping exponents (empty file / no history = the fixed sqrt damping)
  memcall responses --mode "$MODE" --out "$WORK/responses.json"
  [ -s "$WORK/responses.json" ] && RESP_ARG=(--responses "$WORK/responses.json")
fi

# gen_piece OUT SUMMARY_JSON LOG [extra generator args]: one generation with the CURRENT knob variables
# (BM SG CL LLPF HG MG BL LH SU) - the loop and the final mix-dynamics stage share it, so they cannot drift apart.
gen_piece() {
  local out="$1" sum="$2" log="$3"; shift 3
  local inst_arg=() key_arg=() lh_arg=() snd_arg=()
  [ -n "$INST" ] && inst_arg=(--samples-url "$INST")
  [ -n "$KEY" ]  && key_arg=(--key "$KEY")
  [ "$LH" -gt 0 ] 2>/dev/null && lh_arg=(--lead-hpf "$LH")   # 0 = generator default
  [ -n "$BASS_SOUND" ] && snd_arg+=(--bass-sound "$BASS_SOUND")
  [ -n "$LEAD_SOUND" ] && snd_arg+=(--lead-sound "$LEAD_SOUND")
  "$PY" "$GEN" --stems-dir "$STEMS" --pack-dir "$PACK" \
    --bass-midi "$PACK/bass.mid" --lead-midi "$PACK/melodic.mid" \
    --drums-json "$PACK/drums_bands.json" --base-url "$BASE" ${inst_arg[@]+"${inst_arg[@]}"} ${snd_arg[@]+"${snd_arg[@]}"} \
    --mode "$MODE" --bpm "$BPM" ${key_arg[@]+"${key_arg[@]}"} --genre "$GENRE" --num-bars "$BARS" \
    --drum-mode "$DRUM_MODE" \
    --bass-mult "$BM" --sub-gain "$SG" --sub-up "$SU" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" --bass-lpf "$BL" --bass-shape "$BSHAPE" ${lh_arg[@]+"${lh_arg[@]}"} ${LEAD_ARG[@]+"${LEAD_ARG[@]}"} "$@" --out "$out" \
    >"$sum" 2>"$log"
}

# The gate line of the piece in best.cmp.json (the loudness/bus stages replace it after the loop picked BEST).
regate() {
  local g; if g=$("$PY" "$GATE" --comparison "$WORK/best.cmp.json" --genre "$GENRE" --mode "$MODE" 2>/dev/null); then best_gate="$g"; else best_gate="n/a"; fi
}

# solo_job bass|mel TAG: render the voice alone (its strudel is already written) and measure it against the original stem.
# Leaves $WORK/TAG.<kind>.json (only on success) and one line for the log in $WORK/TAG.<kind>.msg; never exits non-zero.
solo_job() {
  local kind="$1" tag="$2"
  local strudel="$WORK/$tag.$kind.strudel" wav="$WORK/$tag.$kind.wav" rlog="$WORK/$tag.$kind.render.log" mlog="$WORK/$tag.$kind.log"
  local json="$WORK/$tag.$kind.json" msg="$WORK/$tag.$kind.msg"
  if [ "$kind" = bass ]; then
    render_one "$strudel" "$wav" "$rlog" \
      && "$PY" "$BSOLO" measure --orig-bass "$STEMS/bass.wav" --solo "$wav" --master-gain "$MG" --seconds "$SONG_S" --out "$json" 2>"$mlog" \
      && { echo "[$tag] $(tail -1 "$mlog")" >"$msg"; return 0; }
    rm -f "$json"; echo "[$tag] WARNING: bass-alone measurement failed (see $rlog, $mlog) - bass layers follow the mix band shares this step" >"$msg"
  else
    render_one "$strudel" "$wav" "$rlog" \
      && "$PY" "$MSOLO" measure --orig-melodic "$STEMS/melodic.wav" --solo "$wav" --master-gain "$MG" --seconds "$SONG_S" --out "$json" 2>"$mlog" \
      && { echo "[$tag] $(tail -1 "$mlog")" >"$msg"; return 0; }
    rm -f "$json"; echo "[$tag] WARNING: melodic-alone measurement failed (see $rlog, $mlog) - cal-lead/lead-lpf follow the mix this step" >"$msg"
  fi
  return 0
}

SULOCK=0   # 1 once an octave-up step lost to its parent: the calibrator must not propose it again in this run
best=-1; best_params=""; best_tag=""; best_gate="n/a"; : >"$WORK/iters.tsv"
rm -f "$WORK/best.env.json" "$WORK/best.master" "$WORK/best.leadsynth.json"   # never report a stale env / level from a previous run
PREV_ENV="${ENV_START:-}"   # env-correction JSON measured from the previous iteration (fed to the next generation); --env-start seeds it
PARENT=""; ALPHA="1"   # the iteration these knobs were stepped from, and the fraction of its proposed step taken
i=0
while [ "$i" -lt "$ITERS" ]; do
  i=$((i+1))
  TAG="cal$i"; STRU="$WORK/$TAG.strudel"; WAV="$WORK/$TAG.wav"; CJ="$WORK/$TAG.cmp.json"
  env_arg=();  [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && env_arg=(--env-correction "$PREV_ENV")
  gen_piece "$STRU" "$WORK/$TAG.gen.json" "$WORK/$TAG.gen.log" ${env_arg[@]+"${env_arg[@]}"} \
    || { echo "[$TAG] GEN FAILED"; RC=70; break; }
  # the generator reports the EFFECTIVE bass cutoff + its bounds (analysis value, top bass note);
  # the calibrator needs them to move the cutoff by measurement (0 = unreported -> lever off)
  if ! read -r EBL EBLMAX EBHZ ELH ELHZ ESU < <("$PY" -c "import json;d=json.load(open('$WORK/$TAG.gen.json'));print(d['bass_lpf'],d['bass_lpf_analysis'],d['bass_note_max_hz'],d['lead_hpf'],d['lead_note_min_hz'],d.get('sub_up',0))" 2>/dev/null); then
    echo "[$TAG] WARNING: cannot parse the generator summary $WORK/$TAG.gen.json - bass-lpf/lead-hpf levers are OFF this step"
    EBL=0 EBLMAX=0 EBHZ=0 ELH=0 ELHZ=0 ESU=0
  elif [ "$ESU" != "$SU" ]; then
    # the generator refused the requested octave (a flip it would filter away): that hypothesis is answered too. Without the
    # lock the calibrator proposes it again every step, and each proposal holds sub-gain/bass-mult.
    SULOCK=1
  fi
  echo "$ESU" >"$WORK/$TAG.su"   # the octave the generator ACTUALLY used (it refuses an octave-up its low-pass would filter away)

  # Never pkill -9 the capture right before a render: on 2026-10-09 that left BlackHole/avfoundation
  # wedged (every following capture produced no file). Wait for any live recorder to finish instead.
  # Bounded + anchored (review #2): wait at most 120 s for a LIVE recorder/capture, matched on the
  # actual invocations (not any process whose command line mentions the file name).
  if [ "$RENDERER" = blackhole ]; then   # the offline renderer has no capture device to wait for
    waited=0
    while "$PGREP" -f "$BUSY_PATTERN" >/dev/null; do
      [ "$waited" -ge "$WAIT_MAX" ] && { echo "[$TAG] a recorder/capture has been running for ${WAIT_MAX} s — stale? (pgrep -fl 'record-strudel|avfoundation'); aborting"; RC=70; break 2; }
      sleep 2; waited=$((waited+2))
    done; sleep "$SETTLE"
  fi

  # The three renders of this iteration are independent (all derive from $STRU): the mix, the bass voice alone (bass_solo.py)
  # and the melodic voices alone (melodic_solo.py). Their strudels are written first (cheap); the renders then run
  # concurrently up to $RJ at a time, and the mix comparison overlaps the solos that are still rendering/measuring.
  # A solo that cannot be set up or measured only keeps the mix-share levers for this step (WARNING); a failed MIX render ends
  # the run (RENDER FAILED, exit 70) and the solos in flight are terminated.
  HAVE_BASS=0; HAVE_MEL=0
  [ "$BSOLO" != off ] && [ -f "$STEMS/bass.wav" ] && "$PY" "$BSOLO" solo "$STRU" "$WORK/$TAG.bass.strudel" 2>/dev/null && HAVE_BASS=1
  # melodic: measured on sample-instrument only
  [ "$MSOLO" != off ] && [ "$MODE" = sample-instrument ] && [ -f "$STEMS/melodic.wav" ] && "$PY" "$MSOLO" solo "$STRU" "$WORK/$TAG.mel.strudel" 2>/dev/null && HAVE_MEL=1
  render_jobs_for $((1 + HAVE_BASS + HAVE_MEL))
  rm -f "$WORK/$TAG.bass.json" "$WORK/$TAG.mel.json" "$WORK/$TAG.bass.msg" "$WORK/$TAG.mel.msg"
  CHILD_PIDS=(); MIX_PID=""; CMP_PID=""; LANE_PIDS=()
  if [ "$RJ" -gt 1 ]; then
    render_one "$STRU" "$WAV" "$WORK/$TAG.render.log" & MIX_PID=$!; CHILD_PIDS+=("$MIX_PID")
    if [ "$RJ" -ge $((1 + HAVE_BASS + HAVE_MEL)) ]; then       # a lane per solo
      [ "$HAVE_BASS" = 1 ] && { solo_job bass "$TAG" & LANE_PIDS+=("$!"); CHILD_PIDS+=("$!"); }
      [ "$HAVE_MEL" = 1 ]  && { solo_job mel  "$TAG" & LANE_PIDS+=("$!"); CHILD_PIDS+=("$!"); }
    elif [ "$HAVE_BASS" = 1 ] || [ "$HAVE_MEL" = 1 ]; then    # two lanes for three renders: the mix, and both solos one after the other
      ( [ "$HAVE_BASS" = 1 ] && solo_job bass "$TAG"; [ "$HAVE_MEL" = 1 ] && solo_job mel "$TAG"; exit 0 ) & LANE_PIDS+=("$!"); CHILD_PIDS+=("$!")
    fi
    wait "$MIX_PID"; mixrc=$?
  else
    render_one "$STRU" "$WAV" "$WORK/$TAG.render.log"; mixrc=$?
  fi
  if [ "$mixrc" -ne 0 ] || [ ! -f "$WAV" ]; then
    reap_children; rm -f "$WAV"
    echo "[$TAG] RENDER FAILED (rc=$(cat "$WORK/$TAG.render.log.rc" 2>/dev/null || echo "$mixrc"), see $WORK/$TAG.render.log)"; RC=70; break
  fi
  rm -f "$WORK/$TAG.render.log.rc"

  if [ "$RJ" -gt 1 ]; then
    "$PY" "$CMP" "$ORIG" "$WAV" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$CJ" 2>/dev/null & CMP_PID=$!; CHILD_PIDS+=("$CMP_PID")
    for lp in ${LANE_PIDS[@]+"${LANE_PIDS[@]}"}; do wait "$lp"; done
    wait "$CMP_PID"; cmprc=$?
    CHILD_PIDS=()
  else
    "$PY" "$CMP" "$ORIG" "$WAV" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$CJ" 2>/dev/null; cmprc=$?
  fi
  [ "$cmprc" -eq 0 ] || { reap_children; echo "[$TAG] CMP FAILED"; RC=70; break; }
  if [ "$RJ" -le 1 ]; then
    [ "$HAVE_BASS" = 1 ] && solo_job bass "$TAG"
    [ "$HAVE_MEL" = 1 ]  && solo_job mel  "$TAG"
  fi
  read -r OVER SECT < <("$PY" -c "import json,sys;d=json.load(open('$CJ'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
  echo "[$TAG] mode=$MODE overall=$OVER section_aware=$SECT  (bm=$BM sg=$SG cl=$CL llpf=$LLPF hg=$HG mg=$MG su=$ESU bl=$BL eff_bl=$EBL lh=$LH eff_lh=$ELH)"

  # Bass / melodic voices alone vs the original stems (see bass_solo.py, melodic_solo.py): the job left its one-line message and
  # the measurement JSON; any failure only keeps the mix-share levers for this step.
  BSOLO_ARG=(); MSOLO_ARG=()
  [ -s "$WORK/$TAG.bass.msg" ] && cat "$WORK/$TAG.bass.msg"
  [ -s "$WORK/$TAG.bass.json" ] && BSOLO_ARG=(--bass-solo "$WORK/$TAG.bass.json")
  [ -s "$WORK/$TAG.mel.msg" ] && cat "$WORK/$TAG.mel.msg"
  [ -s "$WORK/$TAG.mel.json" ] && MSOLO_ARG=(--melodic-solo "$WORK/$TAG.mel.json")

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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$OVER" "$MARG" "$GATE_OUT" "$BM $SG $CL $LLPF $HG $MG $BL $LH $ESU" "$ENVUSED" "$SECT" "${PARENT:--}" "$ALPHA" "$BSHAPE" >>"$WORK/iters.tsv"
  echo "$EBL $ELH" >"$WORK/$TAG.eff"
  memcall record-step "${MEM_ARGS[@]}" --workdir "$WORK" --run-id "$RUN_ID" --tag "$TAG"

  # calibrate -> next knobs
  ENV_OUT="$WORK/$TAG.env.json"
  envin_arg=(); [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && envin_arg=(--env-correction-in "$PREV_ENV")
  parent_arg=(); if [ -n "$PARENT" ] && [ -f "$WORK/$PARENT.cmp.json" ]; then
    read -r PEBL PELH <"$WORK/$PARENT.eff"; parent_arg=(--parent-comparison "$WORK/$PARENT.cmp.json" --parent-bass-lpf "$PEBL" --parent-lead-hpf "$PELH"); fi
  CLI=$("$PY" "$CAL" --comparison "$CJ" ${BSOLO_ARG[@]+"${BSOLO_ARG[@]}"} ${MSOLO_ARG[@]+"${MSOLO_ARG[@]}"} ${parent_arg[@]+"${parent_arg[@]}"} ${RESP_ARG[@]+"${RESP_ARG[@]}"} --env-correction-out "$ENV_OUT" --bpm "$BPM" --bars "$BARS" ${envin_arg[@]+"${envin_arg[@]}"} \
    --bass-mult "$BM" --sub-gain "$SG" --sub-up "$ESU" --sub-up-locked "$SULOCK" --cal-lead "$CL" --lead-lpf "$LLPF" \
    --hat-gain "$HG" --master-gain "$MG" \
    --mode "$MODE" --bass-lpf "$EBL" --bass-lpf-max "$EBLMAX" --bass-note-max-hz "$EBHZ" \
    --lead-hpf "$ELH" --lead-note-min-hz "$ELHZ" 2>"$WORK/$TAG.cal.log")
  # parse "--bass-mult X --sub-gain Y ..." into the loop vars
  set -- $CLI
  while [ $# -gt 0 ]; do case "$1" in
    --bass-mult) BM="$2";; --sub-gain) SG="$2";; --cal-lead) CL="$2";;
    --lead-lpf) LLPF="$2";; --hat-gain) HG="$2";; --master-gain) MG="$2";; --bass-lpf) BL="$2";; --lead-hpf) LH="$2";; --sub-up) SU="$2";;
  esac; shift 2; done
  echo "$BM $SG $CL $LLPF $HG $MG $BL $LH $SU" >"$WORK/$TAG.prop"

  # next step: continue from this iteration, or revert to the best and shorten the step; stop when nothing improves
  NEXT="$("$PY" "$LOOP" next "$WORK/iters.tsv" --cap "$ITERS")" || { echo "[$TAG] loop control FAILED"; RC=70; break; }
  read -r DECISION NPARENT NALPHA <<<"$NEXT"
  if [ "$DECISION" != "CONTINUE" ]; then echo "[$TAG] stop: ${NEXT#STOP }"; break; fi
  if [ "$NPARENT" != "$TAG" ]; then
    echo "[$TAG] regressed vs best $NPARENT beyond noise - next step restarts from $NPARENT at ${NALPHA} of its proposed step"
    # the lost step flipped the sub octave -> that hypothesis is answered (rejected): do not propose it again
    [ "$(cat "$WORK/$TAG.su")" != "$(cat "$WORK/$NPARENT.su")" ] && SULOCK=1
  fi
  PARENT="$NPARENT"; ALPHA="$NALPHA"
  prow=$(awk -F'\t' -v t="$PARENT" '$1==t{print $5"\t"$6}' "$WORK/iters.tsv")
  pknobs="${prow%%$'\t'*}"; penv="${prow##*$'\t'}"
  read -r PEBL PELH <"$WORK/$PARENT.eff"
  read -r BM SG CL LLPF HG MG BL LH SU < <("$PY" "$LOOP" blend --parent "$pknobs" --prop "$(cat "$WORK/$PARENT.prop")" --eff "$PEBL $PELH" --alpha "$ALPHA")
  PREV_ENV=""
  if [ -s "$WORK/$PARENT.env.json" ]; then
    "$PY" "$LOOP" blend-env --used "$penv" --new "$WORK/$PARENT.env.json" --alpha "$ALPHA" --out "$WORK/cal$((i+1)).envin.json" && PREV_ENV="$WORK/cal$((i+1)).envin.json"
  elif [ "$penv" != "-" ]; then PREV_ENV="$penv"; fi
done

# BEST = the iteration with the largest minimum margin to the gate floors (a render that passes both beats one
# that does not; among failures the one nearest to passing; margins within calibration_noise of the top are tied
# and the higher overall, then section-aware, wins). If the gate was unreadable for ANY iteration the
# margins are not comparable, so ALL iterations are ranked by overall instead (one consistent ordering).
SEL=""
if [ -s "$WORK/iters.tsv" ]; then
  SELTAG="$("$PY" "$LOOP" select "$WORK/iters.tsv")" || { echo "BEST selection failed" >&2; exit 70; }
  SEL=$(awk -F'\t' -v t="$SELTAG" '{if($3=="n/a")na=1} $1==t{row=$1"\t"$2"\t"$4"\t"$5"\t"$6} END{print row"\t"(na?"overall":"gate")}' "$WORK/iters.tsv")
fi
if [ -n "$SEL" ]; then
  IFS=$'\t' read -r best_tag best best_gate best_params best_envused SEL_BY <<<"$SEL"
  if [ "$SEL_BY" = "overall" ]; then
    echo "WARNING: gate margin was unreadable for at least one iteration - BEST selected by overall similarity across ALL iterations (floors not applied)"
    best_gate="n/a (gate unreadable for >=1 iteration)"
  fi
  rm -f "$WORK/best.env.json"; [ "$best_envused" != "-" ] && [ -s "$best_envused" ] && cp "$best_envused" "$WORK/best.env.json"
  cp "$WORK/$best_tag.strudel" "$WORK/best.strudel"; cp "$WORK/$best_tag.wav" "$WORK/best.wav"; cp "$WORK/$best_tag.cmp.json" "$WORK/best.cmp.json"
  # ---- bass-shape trial: the BEST knobs once more with the OTHER --bass-shape (header); decided by measurement ----
  BEST_TSV="iters.tsv"; BEST_REC_TAG="$best_tag"
  if [ "$MEM_ON" = 1 ] && [ "$SHAPE_PIN" = 0 ] && [ "$SHAPE_TRIAL" != 0 ] && [ "$SHAPE_TESTED" = 0 ] && [ "$RC" -eq 0 ]; then
    [ "$BSHAPE" = stem ] && ALT=off || ALT=stem; STAG="shape_$ALT"; KEPT_SHAPE="$BSHAPE"
    echo "[$STAG] bass-shape trial: BEST $best_tag knobs ($best_params) rendered once with --bass-shape $ALT (the loop ran $BSHAPE)"
    read -r BM SG CL LLPF HG MG BL LH SU <<<"$best_params"
    s_env=(); [ -s "$WORK/best.env.json" ] && s_env=(--env-correction "$WORK/best.env.json")
    SW="$WORK/$STAG.wav"; SC="$WORK/$STAG.cmp.json"; BSHAPE="$ALT"; srcok=1
    gen_piece "$WORK/$STAG.strudel" "$WORK/$STAG.gen.json" "$WORK/$STAG.gen.log" ${s_env[@]+"${s_env[@]}"} || { echo "[$STAG] GEN FAILED"; srcok=0; }
    if [ "$srcok" = 1 ]; then
      rm -f "$SW"; rrc=0; "$NODE_BIN" "$REC" "$WORK/$STAG.strudel" -o "$SW" -d "$DUR" >"$WORK/$STAG.render.log" 2>&1 || rrc=$?
      if [ "$rrc" -ne 0 ] || [ ! -f "$SW" ]; then rm -f "$SW"; echo "[$STAG] RENDER FAILED (rc=$rrc, see $WORK/$STAG.render.log)"; srcok=0; fi
    fi
    [ "$srcok" = 1 ] && { "$PY" "$CMP" "$ORIG" "$SW" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$SC" 2>/dev/null || { echo "[$STAG] CMP FAILED"; srcok=0; }; }
    BSHAPE="$KEPT_SHAPE"
    if [ "$srcok" = 1 ]; then
      read -r SOVER SSECT < <("$PY" -c "import json,sys;d=json.load(open('$SC'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
      if SGATE=$("$PY" "$GATE" --comparison "$SC" --genre "$GENRE" --mode "$MODE" 2>"$WORK/$STAG.gate.log"); then read -r SMARG _ <<<"$SGATE"
      else SMARG="n/a"; SGATE="n/a"; echo "[$STAG] WARNING: gate margin unavailable"; fi
      echo "[$STAG] overall=$SOVER section_aware=$SSECT gate: $SGATE  (BEST $best_tag: overall=$best gate: $best_gate)"
      senv="-"; [ -s "$WORK/best.env.json" ] && senv="$WORK/best.env.json"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$STAG" "$SOVER" "$SMARG" "$SGATE" "$best_params" "$senv" "$SSECT" "$best_tag" 1 "$ALT" >"$WORK/shape.tsv"
      memcall record-step "${MEM_ARGS[@]}" --workdir "$WORK" --run-id "$RUN_ID" --tag "$STAG" --tsv shape.tsv --kind shape
      if SPICK="$("$PY" "$LOOP" shape "$WORK/iters.tsv" "$best_tag" "$WORK/shape.tsv")"; then
        read -r WIN_SHAPE WIN_TAG <<<"$SPICK"
        if [ "$WIN_TAG" = "$STAG" ]; then
          echo "[$STAG] bass shape $ALT beats $KEPT_SHAPE beyond noise ($SOVER vs $best) -> --bass-shape $ALT"
          cp "$WORK/$STAG.strudel" "$WORK/best.strudel"; cp "$SW" "$WORK/best.wav"; cp "$SC" "$WORK/best.cmp.json"
          best="$SOVER"; best_gate="$SGATE"; BSHAPE="$ALT"; BEST_TSV="shape.tsv"; BEST_REC_TAG="$STAG"
        else echo "[$STAG] bass shape $ALT does not beat $KEPT_SHAPE beyond noise ($SOVER vs $best) -> keeping --bass-shape $KEPT_SHAPE"; fi
      else echo "[$STAG] shape selection FAILED - keeping --bass-shape $KEPT_SHAPE"; RC=70; fi
    else RC=70; echo "[$STAG] trial failed - keeping --bass-shape $KEPT_SHAPE (not recorded as tested)"; fi
  elif [ "$SHAPE_TESTED" = 1 ]; then echo "bass shape: $BSHAPE restored from the memory (the other shape was already measured for this code version)"
  fi
  # ---- synth lead fit (header): fitted oscillator lead vs the palette lead of the BEST render, decided by measurement ----
  if [ "$MODE" = synth ] && [ "$LEAD_FIT" != 0 ] && [ "$RC" -eq 0 ] && [ -s "$STEMS/melodic.wav" ]; then
    LF_SRC=none; LF_WARM=""; LF_DETAIL=""
    if [ "$MEM_ON" = 1 ]; then
      LW="$("$PY" "$MEM" lead-warm "${MEM_ARGS[@]}" --track-key "$TRACK_KEY" --spec-out "$WORK/leadfit.warm.json" 2> >(grep -v '^  \[ClickHouse\]' >&2))" || LW=""
      [ -z "$LW" ] || IFS=$'\t' read -r LF_SRC LF_WARM LF_DETAIL <<<"$LW"
    fi
    case "$LF_SRC" in
      track-keep) LF_CAP=1;; track|neighbour) LF_CAP=3;; track-palette) LF_CAP=0;; *) LF_CAP="$LEAD_EVALS"; LF_SRC=none;;
    esac
    [ "$LF_CAP" -le "$LEAD_EVALS" ] || LF_CAP="$LEAD_EVALS"
    echo "lead fit: start $LF_SRC${LF_DETAIL:+ ($LF_DETAIL)}; render budget $LF_CAP"
    if [ "$LF_CAP" -gt 0 ]; then
      read -r BM SG CL LLPF HG MG BL LH SU <<<"$best_params"
      lf_env=(); [ -s "$WORK/best.env.json" ] && lf_env=(--env-correction "$WORK/best.env.json")
      lf_warm=(); [ "$LF_WARM" != "-" ] && [ -n "$LF_WARM" ] && [ -s "$LF_WARM" ] && lf_warm=(--warm "$LF_WARM")
      LHIST="$WORK/leadfit.hist.json"; rm -f "$LHIST" "$WORK/leadfit.tsv"; lf_best=-1; lf_tag=""; lf_n=0; lf_ok=1
      read -r PSA < <("$PY" -c "import json;print(round(json.load(open('$WORK/best.cmp.json'))['comparison'].get('section_aware_similarity',0),4))")
      read -r PMARG _ <<<"$best_gate"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' palette "$best" "$PMARG" "$best_gate" "$best_params" "-" "$PSA" "-" 1 "$BSHAPE" >"$WORK/leadfit.palette.tsv"
      while [ "$lf_n" -lt "$LF_CAP" ]; do
        ASK="$("$PY" "$LSYN" ask --history "$LHIST" --melodic "$STEMS/melodic.wav" --bpm "$BPM" --cap "$LF_CAP" ${lf_warm[@]+"${lf_warm[@]}"} --out "$WORK/leadfit.cand.json")" \
          || { echo "[leadfit] ask FAILED - keeping the palette lead"; lf_ok=0; break; }
        read -r LST _ <<<"$ASK"; [ "$LST" = EVAL ] || break
        lf_n=$((lf_n+1)); LTAG="leadfit_$lf_n"; cp "$WORK/leadfit.cand.json" "$WORK/$LTAG.leadsynth.json"
        LW_="$WORK/$LTAG.wav"; LC_="$WORK/$LTAG.cmp.json"; srcok=1
        gen_piece "$WORK/$LTAG.strudel" "$WORK/$LTAG.gen.json" "$WORK/$LTAG.gen.log" ${lf_env[@]+"${lf_env[@]}"} --lead-synth "$WORK/$LTAG.leadsynth.json" \
          || { echo "[$LTAG] GEN FAILED"; srcok=0; }
        if [ "$srcok" = 1 ]; then
          rm -f "$LW_"; rrc=0; "$NODE_BIN" "$REC" "$WORK/$LTAG.strudel" -o "$LW_" -d "$DUR" >"$WORK/$LTAG.render.log" 2>&1 || rrc=$?
          if [ "$rrc" -ne 0 ] || [ ! -f "$LW_" ]; then rm -f "$LW_"; echo "[$LTAG] RENDER FAILED (rc=$rrc)"; srcok=0; fi
        fi
        [ "$srcok" = 1 ] && { "$PY" "$CMP" "$ORIG" "$LW_" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$LC_" 2>/dev/null || { echo "[$LTAG] CMP FAILED"; srcok=0; }; }
        if [ "$srcok" != 1 ]; then lf_ok=0; "$PY" "$LSYN" tell --history "$LHIST" --score 0; continue; fi
        read -r LOVER LSECT < <("$PY" -c "import json;d=json.load(open('$LC_'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
        if LGATE=$("$PY" "$GATE" --comparison "$LC_" --genre "$GENRE" --mode "$MODE" 2>/dev/null); then read -r LMARG _ <<<"$LGATE"; else LMARG="n/a"; LGATE="n/a"; fi
        LSCORE="$("$PY" -c "print(round(($LOVER+$LSECT)/2,5))")"
        echo "[$LTAG] overall=$LOVER section_aware=$LSECT score=$LSCORE (palette: overall=$best section_aware=$PSA)"
        "$PY" "$LSYN" tell --history "$LHIST" --score "$LSCORE"
        if awk -v a="$LSCORE" -v b="$lf_best" 'BEGIN{exit !(a>b)}'; then
          lf_best="$LSCORE"; lf_tag="$LTAG"; lf_gate="$LGATE"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LTAG" "$LOVER" "$LMARG" "$LGATE" "$best_params" "-" "$LSECT" "-" 1 "$BSHAPE" >"$WORK/leadfit.tsv"
        fi
      done
      if [ -n "$lf_tag" ]; then
        LDEC=palette
        if LPICK="$("$PY" "$LOOP" leadfit "$WORK/leadfit.palette.tsv" "$WORK/leadfit.tsv")"; then read -r LDEC _ <<<"$LPICK"
        else echo "[leadfit] decision FAILED - keeping the palette lead"; RC=70; fi
        read -r FOVER FSECT < <("$PY" -c "import json;d=json.load(open('$WORK/$lf_tag.cmp.json'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
        "$PY" - "$WORK/leadfit.json" "$LDEC" "$WORK/$lf_tag.leadsynth.json" "$lf_n" "$LF_SRC" "$best" "$PSA" "$FOVER" "$FSECT" <<'PYL'
import json, sys
out, dec, spec, n, src, po, ps, fo, fs = sys.argv[1:10]
json.dump({"decision": dec, "spec": json.load(open(spec)), "evals": int(n), "source": src,
           "palette": {"overall": float(po), "section_aware": float(ps)}, "fit": {"overall": float(fo), "section_aware": float(fs)}},
          open(out, "w"), indent=1)
PYL
        memcall record-step "${MEM_ARGS[@]}" --workdir "$WORK" --run-id "$RUN_ID" --tag "$lf_tag" --tsv leadfit.tsv --kind leadfit --lead-fit "$WORK/leadfit.json"
        if [ "$LDEC" = keep ]; then
          echo "[leadfit] fitted lead beats the palette lead beyond noise (overall $FOVER vs $best, section-aware $FSECT vs $PSA) -> --lead-synth"
          cp "$WORK/$lf_tag.strudel" "$WORK/best.strudel"; cp "$WORK/$lf_tag.wav" "$WORK/best.wav"; cp "$WORK/$lf_tag.cmp.json" "$WORK/best.cmp.json"
          cp "$WORK/$lf_tag.leadsynth.json" "$WORK/best.leadsynth.json"; LEAD_ARG=(--lead-synth "$WORK/best.leadsynth.json")
          best="$FOVER"; best_gate="$lf_gate"; BEST_TSV="leadfit.tsv"; BEST_REC_TAG="$lf_tag"
        else echo "[leadfit] fitted lead does not beat the palette lead (overall $FOVER vs $best, section-aware $FSECT vs $PSA; need > noise, section-aware >= palette - noise, and no PASS -> FAIL) -> palette lead kept"; fi
        LEAD_FIT_JSON="$WORK/leadfit.json"
      else echo "[leadfit] no candidate scored - palette lead kept (not recorded)"; [ "$lf_ok" = 1 ] || RC=70; fi
    fi
  elif [ "$MODE" = synth ] && [ "$LEAD_FIT" = 0 ]; then echo "lead fit: skipped (--no-lead-fit / AC_LEAD_FIT=0)"
  fi
  memcall record-best "${MEM_ARGS[@]}" --workdir "$WORK" --run-id "$RUN_ID" --tag "$BEST_REC_TAG" --tsv "$BEST_TSV" --best-env "$WORK/best.env.json" ${LEAD_FIT_JSON:+--lead-fit "$LEAD_FIT_JSON"}
fi

# ---- loudness stage (scripts/python/mix_dynamics.py) -----------------------------------------------------------
# The score is level-invariant, so the loop above never sees how loud a render is. This stage takes the BEST piece (same
# knobs, same env) and moves ONE lever, --master-gain, onto the instrumental's RMS: up to MIX_ROUNDS render -> measure ->
# next gain rounds (full-scale clipping makes the gain->RMS map sub-linear). A round is kept only if level (RMS) and crest
# match the reference AND neither score dropped more than the calibration noise (the clipped share is printed); otherwise the
# unmixed BEST stays and the log says why.
MIX_ROUNDS=3
if [ "$MIXDYN" = 1 ] && [ -n "$best_tag" ] && [ -s "$WORK/best.wav" ]; then
  read -r BM SG CL LLPF HG MG BL LH SU <<<"$best_params"     # the BEST piece's knobs, for gen_piece
  mix_env=(); [ -s "$WORK/best.env.json" ] && mix_env=(--env-correction "$WORK/best.env.json")
  round_args=(); mr=0; MIX_PREV_WAV="$WORK/best.wav"; MIX_MG="$MG"
  while [ "$mr" -lt "$MIX_ROUNDS" ]; do
    mr=$((mr+1)); MS="$WORK/mix$mr.strudel"; MW="$WORK/mix$mr.wav"; MC="$WORK/mix$mr.cmp.json"
    MG="$("$PY" "$MIXD" next-master --reference "$ORIG" --render "$MIX_PREV_WAV" --seconds "$SONG_S" --master "$MIX_MG")" \
      || { echo "[mix] round $mr: measuring $MIX_PREV_WAV FAILED"; break; }
    echo "[mix] round $mr: --master-gain $MIX_MG -> $MG"
    gen_piece "$MS" "$WORK/mix$mr.gen.json" "$WORK/mix$mr.gen.log" ${mix_env[@]+"${mix_env[@]}"} \
      || { echo "[mix] round $mr: GEN FAILED"; break; }
    rm -f "$MW"; rrc=0; "$NODE_BIN" "$REC" "$MS" -o "$MW" -d "$DUR" >"$WORK/mix$mr.render.log" 2>&1 || rrc=$?
    if [ "$rrc" -ne 0 ] || [ ! -f "$MW" ]; then rm -f "$MW"; echo "[mix] round $mr: RENDER FAILED (rc=$rrc, see $WORK/mix$mr.render.log)"; break; fi
    "$PY" "$CMP" "$ORIG" "$MW" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$MC" 2>/dev/null || { echo "[mix] round $mr: CMP FAILED"; break; }
    round_args+=(--round "$MW" "$MC"); echo "$MG" >"$WORK/mix$mr.master"; MIX_PREV_WAV="$MW"; MIX_MG="$MG"
  done
  if [ "${#round_args[@]}" -gt 0 ]; then
    PICK="$("$PY" "$MIXD" choose --reference "$ORIG" --seconds "$SONG_S" --best-cmp "$WORK/best.cmp.json" \
      --tolerance "$("$PY" "$LOOP" noise)" "${round_args[@]}" 2>"$WORK/mix.choose.log")" || PICK="none"
    sed 's/^/[mix] /' "$WORK/mix.choose.log"
    case "$PICK" in
      [0-9]*)
        cp "$WORK/mix$PICK.strudel" "$WORK/best.strudel"; cp "$WORK/mix$PICK.wav" "$WORK/best.wav"
        cp "$WORK/mix$PICK.cmp.json" "$WORK/best.cmp.json"
        best="$("$PY" -c "import json;print(round(json.load(open('$WORK/best.cmp.json'))['comparison']['overall_similarity'],4))")"
        regate
        cp "$WORK/mix$PICK.master" "$WORK/best.master"   # the BEST params keep the loop's knobs (a seed for the next mode); this is the final level
        echo "[mix] kept round $PICK (--master-gain $(cat "$WORK/mix$PICK.master")) -> best.strudel/best.wav/best.cmp.json";;
      *) echo "[mix] no round matched the reference level within the score tolerance - the unmixed BEST stays";;
    esac
  fi
fi
# ---- master-bus stage (engines that define setMasterLimiter/setMasterCompressor, e.g. the fork; renderer.py master-bus) ----
# The voices keep the clip-limited gain above (so the same file on an engine WITHOUT the bus plays as before); on a bus engine
# the measured limiter (ceiling = the original's true peak) + a unity-ratio-unless-the-loudness-spread-is-wider compressor
# take over the level and the clipping goes away. Probe render 12 dB down -> mix_dynamics.py derive-bus -> up to MIX_ROUNDS
# rounds (makeup closes the RMS error). Kept only if level, crest AND ceiling match (0 clipped samples) and the score did not
# drop beyond the calibration noise vs the clip result; otherwise the clip result stays.
if [ "$MIXDYN" = 1 ] && [ -n "$best_tag" ] && [ -s "$WORK/best.wav" ] && "$PY" "$RENDERER_PY" master-bus; then
  read -r BM SG CL LLPF HG MG BL LH SU <<<"$best_params"
  [ -s "$WORK/best.master" ] && MG="$(cat "$WORK/best.master")"      # the clip stage's final gain
  bus_env=(); [ -s "$WORK/best.env.json" ] && bus_env=(--env-correction "$WORK/best.env.json")
  CLIP_MG="$MG"; PROBE_MG="$("$PY" -c "print(round($CLIP_MG * 0.25, 4))")"      # 0.25 = 12.0412 dB below: peaks fit the 16-bit file
  MG="$PROBE_MG"
  if gen_piece "$WORK/bus.probe.strudel" "$WORK/bus.probe.gen.json" "$WORK/bus.probe.gen.log" ${bus_env[@]+"${bus_env[@]}"} \
     && rm -f "$WORK/bus.probe.wav" && "$NODE_BIN" "$REC" "$WORK/bus.probe.strudel" -o "$WORK/bus.probe.wav" -d "$DUR" >"$WORK/bus.probe.render.log" 2>&1 \
     && "$PY" "$MIXD" derive-bus --reference "$ORIG" --probe "$WORK/bus.probe.wav" --probe-offset-db 12.0412 --seconds "$SONG_S" --out "$WORK/bus.spec.json" >/dev/null; then
    MG="$CLIP_MG"; bus_args=(); br=0
    CEIL="$("$PY" -c "import json;print(json.load(open('$WORK/bus.spec.json'))['ceiling'])")"
    while [ "$br" -lt "$MIX_ROUNDS" ]; do
      br=$((br+1)); BS="$WORK/bus$br.strudel"; BW="$WORK/bus$br.wav"; BC="$WORK/bus$br.cmp.json"
      cp "$WORK/bus.spec.json" "$WORK/bus$br.spec.json"
      gen_piece "$BS" "$WORK/bus$br.gen.json" "$WORK/bus$br.gen.log" ${bus_env[@]+"${bus_env[@]}"} --master-bus "$WORK/bus$br.spec.json" \
        || { echo "[bus] round $br: GEN FAILED"; break; }
      rm -f "$BW"; rrc=0; "$NODE_BIN" "$REC" "$BS" -o "$BW" -d "$DUR" >"$WORK/bus$br.render.log" 2>&1 || rrc=$?
      if [ "$rrc" -ne 0 ] || [ ! -f "$BW" ]; then rm -f "$BW"; echo "[bus] round $br: RENDER FAILED (rc=$rrc)"; break; fi
      "$PY" "$CMP" "$ORIG" "$BW" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$BC" 2>/dev/null || { echo "[bus] round $br: CMP FAILED"; break; }
      bus_args+=(--round "$BW" "$BC")
      "$PY" "$MIXD" next-bus --reference "$ORIG" --render "$BW" --seconds "$SONG_S" --spec "$WORK/bus$br.spec.json" --out "$WORK/bus.spec.json" >/dev/null || break
    done
    if [ "${#bus_args[@]}" -gt 0 ]; then
      BPICK="$("$PY" "$MIXD" choose --reference "$ORIG" --seconds "$SONG_S" --best-cmp "$WORK/best.cmp.json" --ceiling-db "$CEIL" \
        --tolerance "$("$PY" "$LOOP" noise)" "${bus_args[@]}" 2>"$WORK/bus.choose.log")" || BPICK="none"
      sed 's/^/[bus] /' "$WORK/bus.choose.log"
      case "$BPICK" in
        [0-9]*)
          cp "$WORK/bus$BPICK.strudel" "$WORK/best.strudel"; cp "$WORK/bus$BPICK.wav" "$WORK/best.wav"; cp "$WORK/bus$BPICK.cmp.json" "$WORK/best.cmp.json"
          best="$("$PY" -c "import json;print(round(json.load(open('$WORK/best.cmp.json'))['comparison']['overall_similarity'],4))")"
          regate
          echo "[bus] kept round $BPICK (master-bus lines in best.strudel; voices keep --master-gain $CLIP_MG)";;
        *) echo "[bus] no bus round matched level+crest+ceiling within the score tolerance - the clip-limited result stays";;
      esac
    fi
  else
    echo "[bus] probe/derive FAILED (see $WORK/bus.probe.*.log) - the clip-limited result stays"
  fi
fi
echo "BEST: $best_tag overall=$best  params: $best_params $BSHAPE"
echo "  gate:    $best_gate  (margin PASS/FAIL overall section_aware floor sa_floor; BEST = largest minimum margin to the floors)"
[ -s "$WORK/best.leadsynth.json" ] && echo "  leadsynth: $WORK/best.leadsynth.json  (fitted synth lead kept over the palette lead; baked into best.strudel)"
echo "  strudel: $WORK/best.strudel"
[ -s "$WORK/best.master" ] && echo "  master:  $(cat "$WORK/best.master")  (final --master-gain, set by the loudness stage; the params above are the loop's)"
[ -s "$WORK/best.env.json" ] && echo "  env:     $WORK/best.env.json  (per-bar bass/lead/master correction used by the best render; pass --env-correction)"
echo "  wav:     $WORK/best.wav"
echo "  promote + report: scripts/editable-pipeline.sh --title <track> (stages 9-10), or pipeline_helpers.py promote then 'pipeline_helpers.py report --track-dir <track> --version vNNN --seconds $SONG_S --calib-workdir $WORK'."
[ "$RC" -eq 0 ] || echo "FAILED: at least one iteration did not complete (exit $RC); BEST above covers only the iterations that ran" >&2
exit "$RC"
