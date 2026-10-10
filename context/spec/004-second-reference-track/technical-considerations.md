# Technical Specification: Second Reference Track for the Honest Eval Dataset

- **Functional Specification:** [functional-spec.md](./functional-spec.md) (Approved)
- **Status:** Completed (2026-10-10 — acceptance evidence in functional-spec.md; follow-ups #10 #11 #12)
- **Author(s):** `/implement-feature` run, 2026-10-10
- **Can this change alter Strudel output?** **Yes — only through pipeline-defect fixes.** Every step is
  the existing pipeline; a fix made because the new track exposes a defect changes what the generator
  emits for every track, so the render gate (§4) applies, and Regime CLT's reference runs are the
  regression check.

---

## 1. High-Level Technical Approach

The editable-Strudel path that produced Regime CLT v026/v027 exists as a set of scripts that were
chained **by hand** in June (memory: `session-handoff-jun2026`): Demucs stems → Basic Pitch MIDI per
stem → band-split drum onsets → granular instruments trained per stem → sample pack (drum one-shots,
vocal multisample/chops) → hosted manifests on R2 → `generate_dynamic_strudel.py` → BlackHole render
→ `compare_audio.py --strudel` → `eval/gate.py`. **Nothing drives that chain end to end today**;
`scripts/sample-pipeline.sh` drives only the older loops/instrument generator and
`scripts/auto-calibrate.sh` assumes the inputs already exist. That is the first pipeline defect this
feature fixes: a second track must be reproducible from one command.

So the plan is (a) write the missing driver, (b) run it on the new track in both modes with the
standard calibration pass, (c) record the results as reference entries, (d) fix — generally, with
tests — whatever the track breaks, (e) re-verify Regime CLT.

---

## 2. Proposed Solution & Implementation Plan (The "How")

### 2.1 New driver: `scripts/editable-pipeline.sh`

One command, idempotent per stage (each stage skips when its output exists unless `--force`):

```
scripts/editable-pipeline.sh --url <youtube> --prefix <slug> [--genre <g>] [--iters 3] [--dur 170]
                             [--mode sample-instrument|synth|both] [--r2|--local]
```

| Stage | Command (all existing unless marked NEW) | Output under `.cache/stems/<title>/` |
|---|---|---|
| 1 stems + analysis | `./bin/midi-grep extract --url <url> --render none --iterate 0` | `original.wav`, `bass/drums/melodic/vocals.wav`, `metadata.json` (bpm, key, genre) |
| 2 MIDI per stem | `scripts/python/transcribe.py bass.wav sample_pack/bass.mid`; same for `melodic.wav` → `melodic.mid`; `vocals.wav` → `vocals.mid` | `sample_pack/*.mid` |
| 3 drum onsets | `scripts/python/detect_drums_bands.py drums.wav --bpm <bpm> --out sample_pack/drums_bands.json` | `sample_pack/drums_bands.json` |
| 4 sample pack | `scripts/python/build_sample_pack.py --stems-dir . --out sample_pack --bpm <bpm> --key <key> --prefix <slug> --base-url <host>/<slug>/` | drum one-shots, `vocals/` multisample + chops, `samples.json`, `strudel.json`, `pack.json` |
| 5 instruments | `./bin/midi-grep generative train bass.wav --name <slug>_bass --mode granular`; same for `melodic.wav` → `<slug>_lead`; **NEW** `scripts/python/build_instruments.py --models models/<slug>_bass models/<slug>_lead --kit sample_pack/drums --out sample_pack/instruments --base-url <host>/<slug>/instruments/` assembles the note-keyed manifest (`<slug>_bass`, `<slug>_lead`, `bd`, `sd`, `hh`, `oh`) that today exists only as the hand-built `models/regime_instruments/r2/samples.json` | `sample_pack/instruments/{<slug>_bass,<slug>_lead,kit}/*.wav`, `instruments/samples.json` |
| 6 host | `scripts/python/upload_r2.py --pack-dir sample_pack --prefix midi-grep/<slug> --bucket 4cast --public-base https://pub-56831423fee34641805da07cfdaf6812.r2.dev --backend wrangler` with `CLOUDFLARE_ACCOUNT_ID=503e92d7d95838d80c33802d3274f284` (wrangler reads/writes the real bucket only with `--remote`, which the script already passes); `--local` serves `sample_pack/` on `localhost:5555/<slug>/` with CORS instead | public `…/midi-grep/<slug>/samples.json` and `…/instruments/samples.json` |
| 7 generate + calibrate | `scripts/auto-calibrate.sh --stems . --base <host>/<slug> --inst <host>/<slug>/instruments/samples.json --bpm <bpm> --key <key> --genre <genre> --bars <n> --iters <iters> --dur <dur> --bass-sound <slug>_bass --lead-sound <slug>_lead --mode sample-instrument`; then one `generate_dynamic_strudel.py --mode synth` with the best knobs | best `.strudel`, `.wav`, `.cmp.json` per mode |
| 8 score + gate | `compare_audio.py original.wav <render> -d 135 -j --strudel <strudel> -o comparison.json`; `eval/gate.py comparison.json --genre <genre>` | stamped `comparison.json` (editability, generation_mode, tempo candidates) |
| 9 promote | **NEW** small helper in the driver: next `vNNN/` with `output.strudel`, `render.wav`, `comparison.json`, `metadata.json` (version, generation_mode, editability, vocal_mode, generator knobs, render mode "raw capture", device ratio, compare args) | `vNNN/` per mode |

Standard pass = `--iters 3` for sample-instrument (cold-start knobs from the calibrator, keep the best
of three), one synth render with the best knobs. All knob values come from `calibrate_dynamic.py`;
the driver never sets a per-track constant. `--num-bars` comes from the track duration and BPM
(`floor(duration · bpm / 240)`), genre from `metadata.json` (`--genre` overrides only when the
detector is wrong, and that is logged).

### 2.2 New assembler: `scripts/python/build_instruments.py`

Reads a granular model dir (`models/<name>/pitched/<pc>.wav` + `metadata.json`) and emits a Strudel
multisample entry keyed by note name (same shape as `regime_vocal` in the pack manifest and as
`models/regime_instruments/r2/samples.json`: `{"<name>": {"c3": "<name>/c3.wav", …}}`), copies the
wavs into `instruments/<name>/`, adds the kit one-shots (`bd`, `sd`, `hh`, `oh` from the pack's
`drums/` classification), writes `instruments/samples.json` with an absolute `_base` ending in `/`
and every value array-valued or note-dict-valued as `samples()` requires. Pitch-class → note-name
mapping uses the model's `midi_note` medians per pitch class (no fixed octave table).

### 2.3 Reference data

- `eval/datasets/reference_tracks.yaml`: two entries (`mode: sample-instrument`, `mode: synth`) with
  `cache_key: "<title>"`, `version: vNNN`, `genre`.
- `eval/thresholds.yaml`: unchanged if the genre is `brazilian_funk` (its floors already exist). The
  new track's result is then PASS or an **honestly recorded shortfall** — `measured.<genre>` keeps
  Regime CLT as the floor source, and a shortfall is written as a note under `modes.<mode>` plus a
  follow-up task in `tasks.md`. If the detector says another genre, add
  `modes.<mode>.genres.<genre> = round(measured − 0.02, 2)` with its `measured` block, exactly as
  for brazilian_funk (the shape test enforces floor == measured − margin).
- `CLAUDE.md` "Current achievement": a second line for this track with its run ids.
- `llms.txt` / `llms-full.txt`: the driver and the assembler (new scripts), per "Context Document
  Maintenance".

### 2.4 Cache and hosting layout

`.cache/stems/<sanitized title>/` as the Go cache already creates it (`internal/cache` `KeyForURL` →
title folder); `sample_pack/` inside it; `vNNN/` version dirs continue the existing numbering (v001
for a fresh track). R2 prefix `midi-grep/<slug>` where `<slug>` is the kebab-case title
(`vagabundo-nao-namora`), public base `https://pub-56831423fee34641805da07cfdaf6812.r2.dev`.

### 2.5 Defect policy

A defect = a stage that fails, produces an empty/invalid artifact, or an output that fails
`editability_check.py`. Fixes go into the stage's script with a unit/integration test reproducing the
failure on a fixture (never only on the cached track). Explicitly forbidden: per-track thresholds,
sound names, gains or filter values in any script. If a defect cannot be fixed within the run, it is
recorded in `tasks.md` with the evidence and the feature stops at the honest state.

---

## 3. Impact and Risk Analysis

- **System dependencies:** yt-dlp (network), Demucs (~2 min), Basic Pitch (3 stems, ~1 min each),
  granular training (seconds), BlackHole + ffmpeg raw capture (one render at a time; `pgrep
  record-strudel-blackhole` must be empty), R2 via wrangler OAuth with `CLOUDFLARE_ACCOUNT_ID`.
- **Risk — BPM/key misdetection** (the June pack build mis-read Regime CLT at 90.67 BPM): the driver
  takes bpm/key from the Go pipeline's `metadata.json` and passes them explicitly to every stage;
  a half/double-time reading surfaces as a tempo mismatch in `compare_audio` and is a defect to fix in
  `analyze.py`, not a value to override.
- **Risk — granular pitch noise** (known: raw per-grain pyin octave errors): the assembler keeps the
  model's pitch-class medians; if the instrument is unusable the honest result is a lower score, not
  a hand-picked sample.
- **Risk — recorder state:** a wedged BlackHole capture (seen 2026-10-09 after hard kills) must be
  detected (3-second probe before each render) rather than scored as silence.
- **Risk — floors not cleared:** Regime CLT's floors (0.88/0.90 sample-instrument, 0.77/0.81 synth)
  were set from one track; a second track may legitimately land below. The spec says: record the
  shortfall, never lower the floor in this change.
- **Regression:** after any pipeline fix, `eval/gate.py` on Regime CLT v026 and v027 must still PASS
  and the full test suite must stay green.

---

## 4. Testing Strategy

- **Unit:** `build_instruments.py` on a fixture model dir (manifest shape, note keys, `_base`);
  driver helpers (`num_bars` from duration/bpm, slug from title, version-dir promotion) in a small
  Python helper module the shell driver calls, tested with pytest; any defect fix gets a reproducing
  test.
- **Integration:** `editability_check.py` on both generated outputs; `eval/gate.py` on both stamped
  `comparison.json` files; dataset layer of `test_similarity_gate.py` runs against the new entries.
- **E2E (macOS, the render gate):** two 170 s raw-capture BlackHole renders, compared with
  `-d 135 --strudel`; a 3 s capture probe before each; numbers recorded in the flow log with their
  `comparison.json` paths. Regime CLT v026/v027 re-gated after the run as the regression check.
- **Static gate:** `go build/vet/test`, `pytest -q scripts/python/tests`, `npm run build`.
