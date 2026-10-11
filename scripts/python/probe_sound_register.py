#!/usr/bin/env python3
"""Measure a palette sound's own dominant-partial register with a chromatic sweep (track-independent).

``partial_register.py`` compares the ORIGINAL stem's register with the register of the SYNTH sound that
will play the transcription. That sound reference must not come from a track's render (it would bake in
the track's note distribution and the lead-hpf lever's filter). This probe plays the sound alone, with no
filter and no gain pattern, over a chromatic sweep (one note per step) and measures the same statistic per
note, then merges the result into ``data/partial_reference.json``:

  {sound: {"log2": mean over the sweep, "per_note": {midi: log2}, "range": [lo, hi],
           "source": "chromatic sweep probe <date>"}}

  python probe_sound_register.py --sound gm_voice_oohs            # render (BlackHole) + analyse + write
  python probe_sound_register.py --sound gm_voice_oohs --dry-run  # just write the probe .strudel
  python probe_sound_register.py --sound X --wav probe.wav        # analyse an existing capture

The sweep layout is fixed here (PROBE_NOTE_S per note, notes lo..hi); nothing depends on any track.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partial_register as PR  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RECORDER = REPO / "scripts" / "node" / "dist" / "record-strudel-blackhole.js"
PROBE_NOTE_S = 0.5            # seconds per sweep note (one note per Strudel step)
DEFAULT_LO, DEFAULT_HI = 36, 84   # c2..c6 in semitones
LEAD_IN_TAIL_S = 6.0          # recorder start-up/trim slack added to the capture duration
ANALYSIS_WINDOW = (0.25, 0.9)     # fraction of each note analysed: skip the attack, stop before the release
RECORDER_BUSY = "node .*record-strudel-blackhole\\.js|ffmpeg .* -f avfoundation"
_NAMES = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]


def note_name(m: int) -> str:
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def probe_strudel(sound: str, lo: int = DEFAULT_LO, hi: int = DEFAULT_HI) -> str:
    n = hi - lo + 1
    notes = " ".join(note_name(m) for m in range(lo, hi + 1))
    return (f"// register probe: chromatic sweep {note_name(lo)}..{note_name(hi)}, {PROBE_NOTE_S}s/note, "
            f"no filters, no gain pattern\n"
            f"setcps(1 / {n * PROBE_NOTE_S})   // one cycle = the whole sweep\n"
            f'$: note("{notes}").s("{sound}")\n')


def capture_seconds(lo: int, hi: int) -> int:
    return int(math.ceil((hi - lo + 1) * PROBE_NOTE_S + LEAD_IN_TAIL_S))


def wait_for_idle_recorder(max_s: float = 120.0) -> None:
    """One render at a time (same pattern/bound as auto-calibrate.sh): wait for a live recorder."""
    waited = 0.0
    while subprocess.run(["pgrep", "-f", RECORDER_BUSY], capture_output=True).returncode == 0:
        if waited >= max_s:
            raise SystemExit(f"a recorder/capture has been running for {max_s:.0f} s - stale? aborting")
        time.sleep(2)
        waited += 2


def analyse_sweep(wav: Path, lo: int = DEFAULT_LO, hi: int = DEFAULT_HI) -> dict[int, float]:
    """Per-note log2(strongest partial / fundamental) from a sweep capture. The capture's own onset (first
    frame above 10% of peak RMS) is the timeline origin, so the recorder's leading-silence trim is irrelevant."""
    import librosa
    import numpy as np

    y, _ = librosa.load(str(wav), sr=PR.SR, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=PR.HOP)[0]
    if rms.size == 0 or rms.max() <= 0:
        raise SystemExit(f"{wav}: silent capture")
    t0 = float(np.argmax(rms > 0.1 * rms.max())) * PR.HOP / PR.SR
    spans = []
    for i, m in enumerate(range(lo, hi + 1)):
        s = t0 + i * PROBE_NOTE_S
        spans.append((s + ANALYSIS_WINDOW[0] * PROBE_NOTE_S, s + ANALYSIS_WINDOW[1] * PROBE_NOTE_S, m))
    rows = PR._spans_log2(y, [sp for sp in spans if sp[1] * PR.SR < y.size])
    return {pitch: round(l, 4) for pitch, l, _ in rows}


def merge_reference(sound: str, per_note: dict[int, float], lo: int, hi: int, path: Path = PR.REFERENCE_PATH) -> dict:
    if not per_note:
        raise SystemExit("no note could be measured from the capture")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = {"_doc": "", "sounds": {}}
    doc.setdefault("sounds", {})[sound] = {
        "log2": round(sum(per_note.values()) / len(per_note), 4),
        "per_note": {str(k): v for k, v in sorted(per_note.items())},
        "range": [min(per_note), max(per_note)],
        "source": f"chromatic sweep probe {datetime.date.today().isoformat()}",
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc["sounds"][sound]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sound", required=True)
    ap.add_argument("--lo", type=int, default=DEFAULT_LO)
    ap.add_argument("--hi", type=int, default=DEFAULT_HI)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/register-probe"))
    ap.add_argument("--wav", type=Path, default=None, help="analyse this existing capture instead of rendering")
    ap.add_argument("--dry-run", action="store_true", help="write the probe .strudel and print the commands only")
    ap.add_argument("--reference", type=Path, default=PR.REFERENCE_PATH)
    a = ap.parse_args(argv)
    a.workdir.mkdir(parents=True, exist_ok=True)
    strudel = a.workdir / f"probe_{a.sound}.strudel"
    strudel.write_text(probe_strudel(a.sound, a.lo, a.hi))
    wav = a.wav or a.workdir / f"probe_{a.sound}.wav"
    cmd = ["node", str(RECORDER), str(strudel), "-o", str(wav), "-d", str(capture_seconds(a.lo, a.hi))]
    if a.dry_run:
        print(json.dumps({"strudel": str(strudel), "render": cmd, "wav": str(wav)}))
        return 0
    if a.wav is None:
        wait_for_idle_recorder()
        if subprocess.run(cmd).returncode != 0 or not wav.exists():
            print("probe render failed", file=sys.stderr)
            return 70
    entry = merge_reference(a.sound, analyse_sweep(wav, a.lo, a.hi), a.lo, a.hi, a.reference)
    print(json.dumps({a.sound: {k: entry[k] for k in ("log2", "range", "source")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
