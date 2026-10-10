#!/usr/bin/env python3
"""Pure helpers for ``scripts/editable-pipeline.sh`` (spec 004), plus a tiny CLI the shell calls.

* ``slug_from_title``  kebab-case ASCII slug ("VAGABUNDO NÃO NAMORA" -> "vagabundo-nao-namora")
* ``num_bars``         ``floor(duration_s * bpm / 240)`` (4 beats per bar, 60 s per minute)
* ``next_version_dir`` next ``vNNN`` directory path under a track cache dir (not created)
* ``promote_run``      write ``vNNN/{output.strudel, render.wav, comparison.json, metadata.json}``

CLI (each prints one value / path on stdout, errors on stderr with exit 1)::

    pipeline_helpers.py slug "<title>"
    pipeline_helpers.py num-bars <duration_s> <bpm>
    pipeline_helpers.py next-version <track_dir>
    pipeline_helpers.py promote --track-dir D --mode M --strudel F --wav F --comparison F [--env-correction F] \\
        --generator "<knobs>" --render "<capture>" --compare "<args>" [--vocal-mode instrument]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

GENERATION_MODES: tuple[str, ...] = ("sample-instrument", "synth")
_VERSION_RE = re.compile(r"^v(\d+)$")


class RunMeta(BaseModel):
    """Free-text provenance the driver supplies for a promoted run."""

    generator: str
    render: str
    compare: str
    vocal_mode: str = "instrument"


class RunMetadata(BaseModel):
    """The ``vNNN/metadata.json`` document (same keys as the Regime CLT v026/v027 entries)."""

    version: int
    created_at: str
    generation_mode: str
    editability: str
    vocal_mode: str
    generator: str
    render: str
    compare: str
    similarity_overall: float | None
    similarity_section_aware: float | None
    frequency_balance: float | None
    tempo_similarity: float | None
    env_correction: str | None = None


def slug_from_title(title: str) -> str:
    """Kebab-case ASCII slug; accents are folded ("NÃO" -> "nao")."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    if not slug:
        raise ValueError(f"title {title!r} has no ASCII letters or digits to build a slug from")
    return slug


def num_bars(duration_s: float, bpm: float) -> int:
    """Whole 4/4 bars in ``duration_s`` at ``bpm``: floor(duration * bpm / 240)."""
    if duration_s <= 0 or bpm <= 0:
        raise ValueError(f"duration and bpm must be positive, got duration={duration_s} bpm={bpm}")
    return math.floor(duration_s * bpm / 240)


def next_version_dir(track_dir: Path) -> Path:
    """Path of the next ``vNNN`` under ``track_dir`` (v001 when none exist). Does not create it."""
    nums = [int(m.group(1)) for p in track_dir.iterdir() if p.is_dir() and (m := _VERSION_RE.match(p.name))] \
        if track_dir.is_dir() else []
    return track_dir / f"v{(max(nums) + 1 if nums else 1):03d}"


def _score(comparison: dict[str, object]) -> dict[str, float | None]:
    """Pull the four headline numbers; all None when the editability gate rejected the run."""
    inner = comparison.get("comparison")
    if not isinstance(inner, dict):
        return {"overall": None, "section": None, "freq": None, "tempo": None}

    def num(key: str) -> float | None:
        v = inner.get(key)
        return round(float(v), 4) if isinstance(v, (int, float)) else None

    return {"overall": num("overall_similarity"), "section": num("section_aware_similarity"),
            "freq": num("frequency_balance_similarity"), "tempo": num("tempo_similarity")}


ENV_CORRECTION_FILE = "env_correction.json"


def promote_run(track_dir: Path, mode: str, strudel: Path, wav: Path, comparison: Path, meta: RunMeta,
                env_correction: Path | None = None) -> Path:
    """Create the next ``vNNN`` dir holding the run's four artifacts and return it.

    ``env_correction`` (optional): the per-bar correction JSON the generator applied; copied to
    ``vNNN/env_correction.json`` and named in ``metadata.json`` so the promoted ``output.strudel``
    stays reproducible. Absent -> no file and ``metadata.env_correction`` is null.

    Raises FileNotFoundError if any input is missing and ValueError for an unknown mode; nothing is
    written in either case.
    """
    if mode not in GENERATION_MODES:
        raise ValueError(f"mode must be one of {GENERATION_MODES}, got {mode!r}")
    inputs = [("strudel", strudel), ("wav", wav), ("comparison", comparison)]
    if env_correction is not None:
        inputs.append(("env_correction", env_correction))
    for label, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"promote_run: {label} file not found: {path}")
    cmp_doc = json.loads(comparison.read_text(encoding="utf-8"))
    if not isinstance(cmp_doc, dict):
        raise ValueError(f"promote_run: {comparison} is not a comparison.json object")
    score = _score(cmp_doc)

    vdir = next_version_dir(track_dir)
    vdir.mkdir(parents=True)
    shutil.copyfile(strudel, vdir / "output.strudel")
    shutil.copyfile(wav, vdir / "render.wav")
    shutil.copyfile(comparison, vdir / "comparison.json")
    if env_correction is not None:
        shutil.copyfile(env_correction, vdir / ENV_CORRECTION_FILE)
    doc = RunMetadata(
        version=int(vdir.name[1:]),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        generation_mode=mode,
        editability=str(cmp_doc.get("editability", "unknown")),
        vocal_mode=meta.vocal_mode,
        generator=meta.generator,
        render=meta.render,
        compare=meta.compare,
        similarity_overall=score["overall"],
        similarity_section_aware=score["section"],
        frequency_balance=score["freq"],
        tempo_similarity=score["tempo"],
        env_correction=ENV_CORRECTION_FILE if env_correction is not None else None,
    )
    (vdir / "metadata.json").write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return vdir


def stamp_track_metadata(track_dir: Path, *, genre_override: str | None = None,
                         detector_json: Path | None = None) -> dict:
    """Consolidate the per-track facts the pipeline needs into ``<track>/metadata.json``.

    The Go cache writes only title/url/video_id there; bpm/key/style live in the latest
    ``vNNN/metadata.json`` (spec 004 defect #6). This copies them up, adds the audio duration
    (``original.wav``), and records genre provenance: the heuristic ``style`` from the Go run,
    the deep detector's verdict (``detect_genre_dl.py -o`` JSON: detected_genre/confidence/rankings)
    when given, and an explicit override (logged as such). ``genre`` = override if given, else the
    detector's genre, else the heuristic style. Nothing here is a per-track constant.
    """
    track_dir = Path(track_dir)
    meta_path = track_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    versions = sorted(p for p in track_dir.glob("v[0-9][0-9][0-9]") if (p / "metadata.json").exists())
    if not versions:
        raise FileNotFoundError(f"no vNNN/metadata.json under {track_dir} — run stage 1 first")
    # Review #8: promoted vNNN dirs carry no bpm — use the latest version that has one.
    with_bpm = [v for v in versions if json.loads((v / "metadata.json").read_text()).get("bpm") not in (None, "")]
    if not with_bpm:
        raise ValueError(f"no vNNN/metadata.json under {track_dir} carries a bpm (extraction analysis missing)")
    vmeta = json.loads((with_bpm[-1] / "metadata.json").read_text())
    for src, dst in (("bpm", "bpm"), ("key", "key"), ("style", "style_heuristic")):
        if src in vmeta and vmeta[src] not in (None, ""):
            meta[dst] = vmeta[src]
    if "bpm" not in meta:
        raise ValueError(f"{with_bpm[-1]}/metadata.json has no bpm")
    meta["bpm"] = float(meta["bpm"])
    orig = track_dir / "original.wav"
    if orig.exists():
        import soundfile as sf
        info = sf.info(str(orig))
        meta["duration"] = round(info.frames / info.samplerate, 2)
    if detector_json is not None and Path(detector_json).exists():
        det = json.loads(Path(detector_json).read_text())
        meta["genre_detected"] = det.get("detected_genre")
        meta["genre_confidence"] = det.get("confidence")
        mt = det.get("model_type") or "unknown"   # review #7: do not assert CLAP when the heuristic ran
        meta["genre_detector"] = f"detect_genre_dl.py ({mt})"
    if genre_override:
        meta["genre"] = genre_override
        meta["genre_override"] = True
        meta["genre_override_note"] = (
            f"--genre {genre_override} given; detector said {meta.get('genre_detected')!r} "
            f"(confidence {meta.get('genre_confidence')}), heuristic style {meta.get('style_heuristic')!r}")
    elif meta.get("genre_detected"):
        meta["genre"] = meta["genre_detected"]; meta["genre_override"] = False
    elif meta.get("style_heuristic"):
        meta["genre"] = meta["style_heuristic"]; meta["genre_override"] = False
    meta["stamped_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def record_mode_floor(thresholds_path: Path, *, mode: str, genre: str, comparison_path: Path, run: str,
                      margin: float = 0.02) -> dict | None:
    """Spec 004 §2.3: when ``modes.<mode>.genres`` has no floor for ``genre`` yet, derive one from THIS
    run — floor = round(measured − margin, 2) for overall and section-aware — and record the measured
    block (overall, section_aware, run, margin, editability). Returns the block written, or None when
    a floor already exists (floors are never changed here) or the run is not detector-passing."""
    import yaml
    thresholds_path = Path(thresholds_path)
    th = yaml.safe_load(thresholds_path.read_text()) or {}
    mode_key = mode.replace("-", "_")
    block = th.setdefault("modes", {}).setdefault(mode_key, {})
    if genre in ((block.get("genres") or {})):
        return None
    comp = json.loads(Path(comparison_path).read_text())
    if comp.get("editability") != "pass" or not comp.get("comparison"):
        return None
    c = comp["comparison"]
    overall = float(c["overall_similarity"]); sa = c.get("section_aware_similarity")
    measured = {"overall": round(overall, 4), "run": run, "margin": margin, "editability": "pass"}
    if sa is not None:
        measured["section_aware"] = round(float(sa), 4)
    block.setdefault("genres", {})[genre] = round(overall - margin, 2)
    if sa is not None:
        block.setdefault("section_aware", {})[genre] = round(float(sa) - margin, 2)
    block.setdefault("measured", {})[genre] = measured
    thresholds_path.write_text(yaml.safe_dump(th, sort_keys=False, allow_unicode=True))
    return measured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Helpers for scripts/editable-pipeline.sh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slug").add_argument("title")
    nb = sub.add_parser("num-bars")
    nb.add_argument("duration", type=float)
    nb.add_argument("bpm", type=float)
    sub.add_parser("next-version").add_argument("track_dir", type=Path)
    rf = sub.add_parser("record-floor")
    rf.add_argument("--thresholds", required=True, type=Path)
    rf.add_argument("--mode", required=True)
    rf.add_argument("--genre", required=True)
    rf.add_argument("--comparison", required=True, type=Path)
    rf.add_argument("--run", required=True)
    rf.add_argument("--margin", type=float, default=0.02)
    st = sub.add_parser("stamp")
    st.add_argument("track_dir", type=Path)
    st.add_argument("--genre", default=None, help="explicit genre override (logged in metadata)")
    st.add_argument("--detector-json", type=Path, default=None, help="detect_genre_dl.py -o output")
    pr = sub.add_parser("promote")
    pr.add_argument("--track-dir", required=True, type=Path)
    pr.add_argument("--mode", required=True, choices=GENERATION_MODES)
    pr.add_argument("--strudel", required=True, type=Path)
    pr.add_argument("--wav", required=True, type=Path)
    pr.add_argument("--comparison", required=True, type=Path)
    pr.add_argument("--generator", required=True)
    pr.add_argument("--render", required=True)
    pr.add_argument("--compare", required=True)
    pr.add_argument("--vocal-mode", default="instrument")
    pr.add_argument("--env-correction", type=Path, default=None,
                    help="per-bar correction JSON the generator applied; persisted as vNNN/env_correction.json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "slug":
            print(slug_from_title(args.title))
        elif args.cmd == "num-bars":
            print(num_bars(args.duration, args.bpm))
        elif args.cmd == "record-floor":
            res = record_mode_floor(args.thresholds, mode=args.mode, genre=args.genre, comparison_path=args.comparison,
                                    run=args.run, margin=args.margin)
            print(json.dumps({"recorded": res is not None, "measured": res}))
        elif args.cmd == "stamp":
            m = stamp_track_metadata(args.track_dir, genre_override=args.genre, detector_json=args.detector_json)
            print(json.dumps({k: m.get(k) for k in ("bpm", "key", "genre", "genre_detected", "genre_confidence", "genre_override", "duration", "style_heuristic")}))
        elif args.cmd == "next-version":
            print(next_version_dir(args.track_dir))
        else:
            meta = RunMeta(generator=args.generator, render=args.render, compare=args.compare,
                           vocal_mode=args.vocal_mode)
            print(promote_run(args.track_dir, args.mode, args.strudel, args.wav, args.comparison, meta,
                                     env_correction=args.env_correction))
    except (FileNotFoundError, ValueError) as exc:
        print(f"pipeline_helpers: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
