#!/usr/bin/env python3
"""Assemble a hosted, note-keyed Strudel instrument manifest from granular model dirs.

``midi-grep generative train <stem.wav> --name <n> --mode granular`` leaves
``models/<n>/pitched/*.wav`` + ``metadata.json``. Strudel's ``samples()`` needs those wavs laid out
under one hosted base with a ``samples.json`` that maps NOTE NAMES to files. This script does that
step, so a new track does not need the hand-built ``models/regime_instruments/r2/samples.json``.

Output layout (``--out``)::

    <out>/<name>/<note>.wav      pitched notes ('#' written as '_sharp_': d#1 -> d_sharp_1.wav)
    <out>/kit/{bd,sd,hh,oh}.wav  drum one-shots copied from the sample pack's ``drums/``
    <out>/samples.json           {"_base": "<base-url>/", "<name>": {"d#1": "<name>/d_sharp_1.wav"},
                                  "bd": ["kit/bd.wav"], ...}

Note names come from the model itself, never from a fixed octave table:

* current trainer output has ``metadata.pitched_map`` (``{"e1": "pitched/e1.wav"}``) -> used as is;
* legacy output has ``pitched/<pitchclass>.wav`` (``c``, ``cs``, ... ``as``, ``b``) and per-grain
  ``midi_note``: the note of a pitch-class file is the MEDIAN ``midi_note`` of the grains in that
  pitch class.

``samples()`` only accepts array-valued entries or note->file dicts, and resolves ``_base`` by raw
concatenation, so ``_base`` is absolute and ends in ``/`` and the kit entries are one-element lists.

Usage::

    build_instruments.py --models models/<slug>_bass models/<slug>_lead \\
        --kit <pack>/drums --out <pack>/instruments --base-url https://host/<slug>/instruments/
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Chromatic names; the 'c#' spelling is what Strudel and models/regime_instruments/r2/samples.json use.
NOTE_NAMES: tuple[str, ...] = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
# Pitch-class file stems written by the legacy trainer (rave/trainer.py ``NN``): 's' means sharp.
PITCH_CLASS_FILES: dict[str, int] = {n.replace("#", "s"): i for i, n in enumerate(NOTE_NAMES)}
# Strudel drum names the generator addresses; the sample pack's drums/ dir holds one wav for each.
KIT_SOUNDS: tuple[str, ...] = ("bd", "sd", "hh", "oh")


class Grain(BaseModel):
    """One grain row of a model's metadata.json (only the field this script needs)."""

    midi_note: int


class ModelMetadata(BaseModel):
    """Subset of a granular model's metadata.json."""

    name: str | None = None
    pitched_map: dict[str, str] | None = None
    grains: list[Grain] = Field(default_factory=list)


class InstrumentManifest(BaseModel):
    """The in-memory form of instruments/samples.json."""

    base: str
    instruments: dict[str, dict[str, str]]
    kit: dict[str, list[str]]

    def to_json_dict(self) -> dict[str, object]:
        """Serialise in the exact shape Strudel's ``samples(jsonUrl)`` expects."""
        out: dict[str, object] = {"_base": self.base}
        out.update(self.instruments)
        out.update(self.kit)
        return out


def midi_to_note(midi: int) -> str:
    """MIDI number -> Strudel note name (60 -> 'c4', 39 -> 'd#2')."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def note_filename(note: str) -> str:
    """Note name -> wav filename ('d#1' -> 'd_sharp_1.wav')."""
    return note.replace("#", "_sharp_") + ".wav"


def normalize_base_url(base_url: str) -> str:
    """Require an absolute URL and make it end in '/' (samples() concatenates raw)."""
    if "://" not in base_url:
        raise ValueError(f"--base-url must be absolute (http[s]://...), got {base_url!r}")
    return base_url if base_url.endswith("/") else base_url + "/"


def load_metadata(model_dir: Path) -> ModelMetadata:
    """Read and validate ``<model_dir>/metadata.json``."""
    meta_path = model_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"model {model_dir} has no metadata.json")
    return ModelMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))


def resolve_notes(model_dir: Path, meta: ModelMetadata) -> dict[str, Path]:
    """Return ``{note name: source wav}`` for one model, ordered by pitch."""
    notes: dict[str, Path] = {}
    if meta.pitched_map:
        for note, rel in meta.pitched_map.items():
            src = model_dir / rel
            if not src.is_file():
                raise FileNotFoundError(f"model {model_dir}: pitched_map names {note!r} -> {rel} but the file is missing")
            notes[note] = src
    else:
        pitched = model_dir / "pitched"
        wavs = sorted(pitched.glob("*.wav")) if pitched.is_dir() else []
        if not wavs:
            raise FileNotFoundError(f"model {model_dir} has no pitched/*.wav (train it with --mode granular)")
        by_pc: dict[int, list[int]] = {}
        for grain in meta.grains:
            by_pc.setdefault(grain.midi_note % 12, []).append(grain.midi_note)
        for wav in wavs:
            pc = PITCH_CLASS_FILES.get(wav.stem)
            if pc is None:
                raise ValueError(f"model {model_dir}: {wav.name} is not a pitch-class file and metadata has no pitched_map")
            if pc not in by_pc:
                raise ValueError(f"model {model_dir}: no grain in metadata.json has pitch class {wav.stem!r}")
            # Review finding #1: the raw median of grains spanning octaves can land on another pitch
            # class (median([36, 48]) = 42 → f#2 for a 'c' file) and even overwrite another key.
            # Snap to THIS file's pitch class: median octave, then pc + 12 * octave.
            octaves = [m // 12 for m in by_pc[pc]]
            midi = int(round(statistics.median(octaves))) * 12 + pc
            note = midi_to_note(midi)
            if note in notes:
                raise ValueError(f"model {model_dir}: pitch-class files {notes[note].name} and {wav.name} both resolve to {note}")
            notes[note] = wav
    return dict(sorted(notes.items(), key=lambda kv: _note_midi(kv[0])))


def _note_midi(note: str) -> int:
    """Inverse of :func:`midi_to_note` (used only for stable ordering)."""
    idx = len(note.rstrip("0123456789-"))
    return NOTE_NAMES.index(note[:idx]) + (int(note[idx:]) + 1) * 12


def resolve_kit(kit_dir: Path) -> dict[str, Path]:
    """Locate the four kit one-shots, failing loudly if any is missing."""
    found = {s: kit_dir / f"{s}.wav" for s in KIT_SOUNDS}
    missing = [f"{s}.wav" for s, p in found.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"kit dir {kit_dir} is missing one-shot(s): {', '.join(missing)}")
    return found


def build_instruments(models: list[Path], kit_dir: Path, out_dir: Path, base_url: str) -> InstrumentManifest:
    """Copy wavs into ``out_dir`` and write ``out_dir/samples.json``; returns the manifest."""
    base = normalize_base_url(base_url)
    kit = resolve_kit(kit_dir)  # validate everything before writing anything
    plans: dict[str, dict[str, Path]] = {}
    for model_dir in models:
        meta = load_metadata(model_dir)
        name = meta.name or model_dir.name
        if name in plans or name in KIT_SOUNDS:
            raise ValueError(f"duplicate or reserved instrument name {name!r} (model {model_dir})")
        plans[name] = resolve_notes(model_dir, meta)

    instruments: dict[str, dict[str, str]] = {}
    for name, notes in plans.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        instruments[name] = {}
        for note, src in notes.items():
            rel = f"{name}/{note_filename(note)}"
            shutil.copyfile(src, out_dir / rel)
            instruments[name][note] = rel
    (out_dir / "kit").mkdir(parents=True, exist_ok=True)
    kit_entries: dict[str, list[str]] = {}
    for sound, src in kit.items():
        shutil.copyfile(src, out_dir / "kit" / src.name)
        kit_entries[sound] = [f"kit/{src.name}"]

    manifest = InstrumentManifest(base=base, instruments=instruments, kit=kit_entries)
    (out_dir / "samples.json").write_text(json.dumps(manifest.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the hosted note-keyed instrument manifest.")
    ap.add_argument("--models", nargs="+", required=True, type=Path, help="granular model dirs (models/<name>)")
    ap.add_argument("--kit", required=True, type=Path, help="sample pack drums/ dir with bd/sd/hh/oh.wav")
    ap.add_argument("--out", required=True, type=Path, help="output dir (becomes <base-url>)")
    ap.add_argument("--base-url", required=True, help="absolute URL the --out dir is served at")
    args = ap.parse_args(argv)
    try:
        manifest = build_instruments(args.models, args.kit, args.out, args.base_url)
    except (FileNotFoundError, ValueError) as exc:
        print(f"build_instruments: {exc}", file=sys.stderr)
        return 1
    summary = {name: sorted(notes) for name, notes in manifest.instruments.items()}
    print(json.dumps({"samples_json": str(args.out / "samples.json"), "base": manifest.base,
                      "instruments": summary, "kit": sorted(manifest.kit)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
