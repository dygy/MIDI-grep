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
#     [--dur <seconds>] [--workdir <tmp>] [--no-mix-dynamics]
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
# Renderer selection lives in ONE place (scripts/python/renderer.py: MIDIGREP_RENDERER, default offline = exact
# Strudel OfflineAudioContext render; blackhole = real-time capture). A bad value exits 2 there.
RENDERER_PY="${AC_RENDERER_PY:-$ROOT/scripts/python/renderer.py}"
RENDERER="$("$PY" "$RENDERER_PY" name)" || exit 2
REC="${AC_REC:-$("$PY" "$RENDERER_PY" script)}" || exit 2
BUSY_PATTERN="$("$PY" "$RENDERER_PY" busy-pattern)" || exit 2
NODE_BIN="${AC_NODE:-node}"; SETTLE="${AC_SETTLE_S:-3}"
PGREP="${AC_PGREP:-pgrep}"; WAIT_MAX="${AC_RECORDER_WAIT_S:-120}"   # tests: AC_PGREP=false, AC_RECORDER_WAIT_S=0
RC=0   # exit status: 70 when any iteration failed (GEN/RENDER/CMP FAILED, recorder wait abort); BEST is still printed

STEMS="" BASE="" INST="" BPM="136" KEY="" GENRE="brazilian_funk" BARS="78"
MIXDYN=1; ENV_START=""; ITERS="" DUR="" BASS_SOUND="" LEAD_SOUND="" MODE="sample-instrument"
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
  --start) BL=0; LH=0; SU=0; read -r BM SG CL LLPF HG MG BL LH SU EXTRA <<<"$2"; BL="${BL:-0}"; LH="${LH:-0}"; SU="${SU:-0}"   # resume from a previous BEST: "BM SG CL LLPF HG MG [BL [LH [SU]]]" (6 to 9 fields; SU = sub-sine octave up 0/1)
    # The pre-"no vocals" format had a CV field after MG ("... MG CV [BL [LH]]"). A 9-field start is that old format;
    # 7/8 fields would silently shift CV into BL, but a real bass/lead cutoff is 0 or >= 20 Hz, a cal-vocal is ~1.
    [ -z "${EXTRA:-}" ] || { echo "--start: more than 9 fields - the old CV (cal-vocal) field was removed; drop the 7th value" >&2; exit 64; }
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
[ -n "$ITERS" ] || ITERS="$("$PY" "$LOOP" max-iters)" || exit 2
mkdir -p "$WORK"
echo "workdir: $WORK"

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
    --hat-gain "$HG" --master-gain "$MG" --bass-lpf "$BL" ${lh_arg[@]+"${lh_arg[@]}"} "$@" --out "$out" \
    >"$sum" 2>"$log"
}

SULOCK=0   # 1 once an octave-up step lost to its parent: the calibrator must not propose it again in this run
best=-1; best_params=""; best_tag=""; best_gate="n/a"; : >"$WORK/iters.tsv"
rm -f "$WORK/best.env.json" "$WORK/best.master"   # never report a stale env / level from a previous run
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
  # The renderer's EXIT STATUS decides, not the file's existence: a non-zero exit means its own checks failed
  # (silence, missing sounds, truncated tail ...), so whatever is at $WAV must never be scored.
  rm -f "$WAV"
  rrc=0; "$NODE_BIN" "$REC" "$STRU" -o "$WAV" -d "$DUR" >"$WORK/$TAG.render.log" 2>&1 || rrc=$?
  if [ "$rrc" -ne 0 ] || [ ! -f "$WAV" ]; then
    rm -f "$WAV"; echo "[$TAG] RENDER FAILED (rc=$rrc, see $WORK/$TAG.render.log)"; RC=70; break
  fi

  "$PY" "$CMP" "$ORIG" "$WAV" -d "$SONG_S" -j ${CONTROL_ARG[@]+"${CONTROL_ARG[@]}"} > "$CJ" 2>/dev/null || { echo "[$TAG] CMP FAILED"; RC=70; break; }
  read -r OVER SECT < <("$PY" -c "import json,sys;d=json.load(open('$CJ'))['comparison'];print(round(d['overall_similarity'],4),round(d.get('section_aware_similarity',0),4))")
  echo "[$TAG] mode=$MODE overall=$OVER section_aware=$SECT  (bm=$BM sg=$SG cl=$CL llpf=$LLPF hg=$HG mg=$MG su=$ESU bl=$BL eff_bl=$EBL lh=$LH eff_lh=$ELH)"

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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$OVER" "$MARG" "$GATE_OUT" "$BM $SG $CL $LLPF $HG $MG $BL $LH $ESU" "$ENVUSED" "$SECT" "${PARENT:--}" "$ALPHA" >>"$WORK/iters.tsv"
  echo "$EBL $ELH" >"$WORK/$TAG.eff"

  # calibrate -> next knobs
  ENV_OUT="$WORK/$TAG.env.json"
  envin_arg=(); [ -n "$PREV_ENV" ] && [ -s "$PREV_ENV" ] && envin_arg=(--env-correction-in "$PREV_ENV")
  parent_arg=(); if [ -n "$PARENT" ] && [ -f "$WORK/$PARENT.cmp.json" ]; then
    read -r PEBL PELH <"$WORK/$PARENT.eff"; parent_arg=(--parent-comparison "$WORK/$PARENT.cmp.json" --parent-bass-lpf "$PEBL" --parent-lead-hpf "$PELH"); fi
  CLI=$("$PY" "$CAL" --comparison "$CJ" ${parent_arg[@]+"${parent_arg[@]}"} --env-correction-out "$ENV_OUT" --bpm "$BPM" --bars "$BARS" ${envin_arg[@]+"${envin_arg[@]}"} \
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
          echo "[bus] kept round $BPICK (master-bus lines in best.strudel; voices keep --master-gain $CLIP_MG)";;
        *) echo "[bus] no bus round matched level+crest+ceiling within the score tolerance - the clip-limited result stays";;
      esac
    fi
  else
    echo "[bus] probe/derive FAILED (see $WORK/bus.probe.*.log) - the clip-limited result stays"
  fi
fi
echo "BEST: $best_tag overall=$best  params: $best_params"
echo "  gate:    $best_gate  (margin PASS/FAIL overall section_aware floor sa_floor; BEST = largest minimum margin to the floors)"
echo "  strudel: $WORK/best.strudel"
[ -s "$WORK/best.master" ] && echo "  master:  $(cat "$WORK/best.master")  (final --master-gain, set by the loudness stage; the params above are the loop's)"
[ -s "$WORK/best.env.json" ] && echo "  env:     $WORK/best.env.json  (per-bar bass/lead/master correction used by the best render; pass --env-correction)"
echo "  wav:     $WORK/best.wav"
echo "  promote + report: scripts/editable-pipeline.sh --title <track> (stages 9-10), or pipeline_helpers.py promote then 'pipeline_helpers.py report --track-dir <track> --version vNNN --seconds 135 --calib-workdir $WORK'."
[ "$RC" -eq 0 ] || echo "FAILED: at least one iteration did not complete (exit $RC); BEST above covers only the iterations that ran" >&2
exit "$RC"
