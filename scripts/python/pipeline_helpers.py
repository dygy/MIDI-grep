#!/usr/bin/env python3
"""Pure helpers for ``scripts/editable-pipeline.sh`` (spec 004), plus a tiny CLI the shell calls.

* ``slug_from_title``  kebab-case ASCII slug ("VAGABUNDO NÃO NAMORA" -> "vagabundo-nao-namora")
* ``num_bars``         ``floor(duration_s * bpm / 240)`` (4 beats per bar, 60 s per minute)
* ``next_version_dir`` next ``vNNN`` directory path under a track cache dir (not created)
* ``promote_run``      write ``vNNN/{output.strudel, render.wav, comparison.json, metadata.json}``
* ``build_report``     stage 10: charts, render stems, per-stem comparison, iteration history, ``report.html``

CLI (each prints one value / path on stdout, errors on stderr with exit 1)::

    pipeline_helpers.py slug "<title>"
    pipeline_helpers.py num-bars <duration_s> <bpm>
    pipeline_helpers.py render-seconds <duration_s> <bpm>
    pipeline_helpers.py next-version <track_dir>
    pipeline_helpers.py control --genre G --exclude-key K
    pipeline_helpers.py promote --track-dir D --mode M --strudel F --wav F --comparison F [--env-correction F] \\
        --generator "<knobs>" --render "<capture>" --compare "<args>"
    pipeline_helpers.py report --track-dir D --version vNNN --seconds S [--calib-workdir W] \\
        [--no-iteration-stems] [--force] [--config F] [--gate-verdict PASS|FAIL --gate-message M]
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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


class RunMetadata(BaseModel):
    """The ``vNNN/metadata.json`` document (same keys as the Regime CLT v026/v027 entries)."""

    version: int
    created_at: str
    generation_mode: str
    editability: str
    generator: str
    render: str
    compare: str
    similarity_overall: float | None
    similarity_section_aware: float | None
    frequency_balance: float | None
    tempo_similarity: float | None
    env_correction: str | None = None
    # Measured from the piece itself (strudel header written by generate_dynamic_strudel.py); omitted when absent.
    bpm: float | None = None
    genre: str | None = None
    notes: int | None = None
    drum_hits: int | None = None
    # Stage 10 (build_report) adds report / render_stems / stem_comparison / iterations to the file later.


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


def render_seconds(duration_s: float, bpm: float) -> float:
    """Seconds of audio the generated code covers: ``num_bars`` whole bars = bars * 240 / bpm, floored to the ms.

    This is the render length AND the compare window: rendering longer makes the looped Strudel wrap back to bar 0
    (audio past the song end); the song's trailing partial bar has no generated code, so it is not rendered or scored.
    """
    return math.floor(num_bars(duration_s, bpm) * 240 / bpm * 1000) / 1000


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


def strudel_header_facts(code: str) -> dict[str, float | int | str]:
    """``bpm`` / ``genre`` / ``notes`` / ``drum_hits`` from the generator's header comments (only those present)."""
    out: dict[str, float | int | str] = {}
    if m := re.search(r"^//\s*BPM:\s*(\d+(?:\.\d+)?)", code, re.M):
        out["bpm"] = float(m.group(1))
    if m := re.search(r"^//\s*Genre:\s*(\S[^\n]*)", code, re.M):
        out["genre"] = m.group(1).strip()
    if m := re.search(r"^//\s*Notes:\s*(\d+)", code, re.M):
        out["notes"] = int(m.group(1))
    if m := re.search(r"^//\s*Drums:\s*(\d+)\s*hits", code, re.M):
        out["drum_hits"] = int(m.group(1))
    return out


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
        generator=meta.generator,
        render=meta.render,
        compare=meta.compare,
        similarity_overall=score["overall"],
        similarity_section_aware=score["section"],
        frequency_balance=score["freq"],
        tempo_similarity=score["tempo"],
        env_correction=ENV_CORRECTION_FILE if env_correction is not None else None,
        **strudel_header_facts(strudel.read_text(encoding="utf-8")),
    )
    optional = ("bpm", "genre", "notes", "drum_hits")   # keys the header did not supply are left out, not null
    (vdir / "metadata.json").write_text(
        doc.model_dump_json(indent=2, exclude={k for k in optional if getattr(doc, k) is None}) + "\n",
        encoding="utf-8")
    return vdir


def _refine_track_bpm(track_dir: Path, meta: dict, *, coarse: float) -> None:
    """Replace the tempogram-grid bpm with the phase-coherence refinement (spec 001 §2.3: play at the
    correct speed). Refines on ``drums.wav`` when present (the cleanest pulse), else ``original.wav``;
    records ``bpm``, ``bpm_coarse``, ``bpm_method``, ``bpm_confidence``. No audio -> bpm is left as is."""
    from tempo_refine import refine_bpm_file
    src = next((p for p in (track_dir / "drums.wav", track_dir / "original.wav") if p.exists()), None)
    if src is None:
        return
    r = refine_bpm_file(str(src), coarse)
    meta["bpm_coarse"] = round(coarse, 4)
    meta["bpm"] = round(float(r["bpm"]), 3)
    meta["bpm_method"] = f"{r['method']} ({src.name})"
    meta["bpm_confidence"] = round(float(r["confidence"]), 4)


UNREFERENCED = "unreferenced"   # genre key meaning "no trusted genre": the gate uses its default floor, memory finds no neighbours


def stamp_track_metadata(track_dir: Path, *, genre_override: str | None = None,
                         detector_json: Path | None = None, labels_path: Path | None = None,
                         embeddings_root: Path | None = None, stems_root: Path | None = None) -> dict:
    """Consolidate the per-track facts the pipeline needs into ``<track>/metadata.json``.

    The Go cache writes only title/url/video_id there; bpm/key/style live in the latest
    ``vNNN/metadata.json`` (spec 004 defect #6). This copies them up, adds the audio duration
    (``original.wav``), and records genre provenance: the heuristic ``style`` from the Go run,
    the deep detector's verdict (``detect_genre_dl.py -o`` JSON: detected_genre/confidence/rankings)
    when given, and an explicit override (logged as such). ``genre`` precedence (``genre_basis``): ``--genre`` override
    > the owner's label registry (``eval/datasets/genre_labels.yaml``, certain) > kNN of the track's CLAP embedding
    against the labelled tracks (``genre_knn.py``) > the CLAP zero-shot verdict > the heuristic style. When the kNN
    ran and no labelled genre supports the track, the CLAP verdict stays as the palette guess but is marked
    uncertain / unreferenced. Nothing here is a per-track constant.
    """
    import genre_knn as GK
    track_dir = Path(track_dir)
    meta_path = track_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    versions = sorted(p for p in track_dir.glob("v[0-9][0-9][0-9]") if (p / "metadata.json").exists())
    if not versions:
        raise FileNotFoundError(f"no vNNN/metadata.json under {track_dir} — run stage 1 first")
    # Review #8: promoted vNNN dirs carry no bpm — use the latest version that has one.
    with_bpm = [v for v in versions if json.loads((v / "metadata.json").read_text()).get("bpm") not in (None, "")]
    if with_bpm:
        vmeta = json.loads((with_bpm[-1] / "metadata.json").read_text())
        for src, dst in (("bpm", "bpm"), ("key", "key"), ("style", "style_heuristic")):
            if src in vmeta and vmeta[src] not in (None, ""):
                meta[dst] = vmeta[src]
        coarse = float(vmeta["bpm"])
    elif meta.get("bpm") not in (None, ""):
        # An earlier stamp (or a hand-stamped track) is the only bpm source: re-refine from ITS coarse value.
        coarse = float(meta.get("bpm_coarse") or meta["bpm"])
    else:
        raise ValueError(f"no vNNN/metadata.json under {track_dir} carries a bpm (extraction analysis missing)")
    meta["bpm"] = float(coarse)
    _refine_track_bpm(track_dir, meta, coarse=coarse)
    orig = track_dir / "original.wav"
    if orig.exists():
        import soundfile as sf
        info = sf.info(str(orig))
        meta["duration"] = round(info.frames / info.samplerate, 2)
    key = track_dir.name
    labels = GK.load_labels(labels_path)
    for stale in ("genre_knn", "genre_label_source", "genre_uncertain", "genre_candidates", "genre_margin_z",
                  "genre_uncertain_reasons"):
        meta.pop(stale, None)          # a re-stamp must not inherit another run's evidence
    embedding = model_id = None
    if detector_json is not None and Path(detector_json).exists():
        det = json.loads(Path(detector_json).read_text())
        meta["genre_detected"] = det.get("detected_genre")
        meta["genre_confidence"] = det.get("confidence")
        mt = det.get("model_type") or "unknown"   # review #7: do not assert CLAP when the heuristic ran
        meta["genre_detector"] = f"detect_genre_dl.py ({mt})"
        # Old detector JSON has no verdict on its own noise: such a result is NOT trusted as certain.
        meta["genre_uncertain"] = bool(det.get("uncertain", True))
        meta["genre_candidates"] = list(det.get("candidates") or [det.get("detected_genre")])
        meta["genre_margin_z"] = det.get("margin_z")
        meta["genre_uncertain_reasons"] = list(det.get("uncertain_reasons") or
                                               ([] if "uncertain" in det else ["detector JSON predates the margin rule"]))
        if det.get("audio_embedding"):
            embedding, model_id = det["audio_embedding"], det.get("embedding_model")
    # kNN against the owner-labelled tracks: only for a track that is neither overridden nor itself labelled.
    knn = None
    if not genre_override and key not in labels:
        if embedding is None or not model_id:
            knn = {"status": "unavailable", "genre": None, "supported": [], "per_genre": {},
                   "reasons": ["no audio embedding in the detector output"]}
        else:
            knn = GK.classify_track(embedding, model_id, labels, exclude_key=key, root=embeddings_root, stems_root=stems_root)
        meta["genre_knn"] = {"status": knn["status"], "genre": knn["genre"], "supported": knn["supported"],
                             "scores": {g: round(p["score"], 4) for g, p in knn["per_genre"].items()},
                             "loo_min": {g: round(p["loo_min"], 4) for g, p in knn["per_genre"].items() if "loo_min" in p},
                             "n_members": {g: p["n_members"] for g, p in knn["per_genre"].items()},
                             "reasons": knn["reasons"]}
    if genre_override:
        meta["genre"] = genre_override
        meta["genre_override"] = True
        meta["genre_basis"] = "override"
        meta["genre_override_note"] = (
            f"--genre {genre_override} given; detector said {meta.get('genre_detected')!r} "
            f"(confidence {meta.get('genre_confidence')}), heuristic style {meta.get('style_heuristic')!r}")
    elif key in labels:
        lab = labels[key]
        meta["genre"] = lab["genre"]; meta["genre_override"] = False
        meta["genre_basis"] = "label"; meta["genre_uncertain"] = False
        meta["genre_candidates"] = [lab["genre"]]
        meta["genre_label_source"] = f"{lab['source']}: {lab['note']}".strip(": ")
    elif knn is not None and knn["genre"]:
        meta["genre"] = knn["genre"]; meta["genre_override"] = False
        meta["genre_basis"] = "knn"; meta["genre_uncertain"] = False
        meta["genre_candidates"] = list(knn["supported"])
    elif meta.get("genre_detected"):
        meta["genre"] = meta["genre_detected"]; meta["genre_override"] = False
        if knn is not None and knn["status"] == "ok":
            # The kNN COULD decide and no labelled genre supports this track: whatever CLAP says is a guess.
            meta["genre_uncertain"] = True
            meta["genre_uncertain_reasons"] = (list(meta.get("genre_uncertain_reasons") or [])
                                               + ["kNN: no labelled genre supports this track (" + "; ".join(knn["reasons"]) + ")"])
        meta["genre_basis"] = "detector-uncertain" if meta.get("genre_uncertain") else "detector"
    elif meta.get("style_heuristic"):
        meta["genre"] = meta["style_heuristic"]; meta["genre_override"] = False
        meta["genre_basis"] = "heuristic-style"   # BPM/style guess from the Go run: not audio evidence of a genre
        meta["genre_uncertain"] = True
    # `genre_ref` is the genre the reference tables are consulted for (eval floors, calibration neighbours): the
    # label itself only when it is trusted (override, owner label, kNN, or a detection that cleared the margin
    # rule), else UNREFERENCED. `genre` stays the best guess (sound palette, generator header).
    if meta.get("genre"):
        meta["genre_ref"] = meta["genre"] if meta.get("genre_basis") in ("override", "label", "knn", "detector") else UNREFERENCED
        if meta["genre_ref"] == UNREFERENCED:
            print(f"pipeline_helpers: GENRE UNCERTAIN - using {meta['genre']!r} for the sound palette only; no genre "
                  f"floors / calibration neighbours are applied (basis {meta.get('genre_basis')}, candidates "
                  f"{meta.get('genre_candidates')}, reasons {meta.get('genre_uncertain_reasons')}). Pass --genre to decide "
                  f"or add the track to eval/datasets/genre_labels.yaml.", file=sys.stderr)
    meta["stamped_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def record_mode_floor(thresholds_path: Path, *, mode: str, genre: str, comparison_path: Path, run: str,
                      margin: float = 0.02, labels_path: Path | None = None) -> dict | None:
    """Spec 004 §2.3 (amended 2026-10-29): when ``modes.<mode>.genres`` has no floor for ``genre`` yet AND
    ``eval/datasets/genre_labels.yaml`` labels at least ``genre_knn.MIN_MEMBERS`` (2) tracks with that genre, derive
    one from THIS run — floor = round(measured − margin, 2) for overall and section-aware — and record the measured
    block (overall, section_aware, run, margin, editability). Returns the block written, or None (reason on stderr)
    when the genre is unverified, has too few labelled tracks, already has a floor (never changed here) or the run
    is not detector-passing.

    Why 2: a floor from a single track is passed by that track by construction (measured − margin), so it says
    nothing about generalisation (fix #10 lesson); at least two tracks are needed for any statement about the
    genre, and for a leave-one-out. It is the same ``MIN_MEMBERS`` notion the genre kNN uses for a within-genre spread."""
    import yaml
    import genre_knn as GK
    if not genre or genre == UNREFERENCED:   # an unverified genre must not grow a floor of its own
        return None
    thresholds_path = Path(thresholds_path)
    th = yaml.safe_load(thresholds_path.read_text()) or {}
    if genre in ((th.get("modes", {}).get(mode.replace("-", "_"), {}) or {}).get("genres") or {}):
        return None                           # an existing floor is never touched (and needs no label count)
    n = sum(1 for v in GK.load_labels(labels_path).values() if v["genre"] == genre)
    if n < GK.MIN_MEMBERS:
        print(f"pipeline_helpers: no floor recorded: {genre} has {n} labelled track{'s' if n != 1 else ''}; a floor needs "
              f">= {GK.MIN_MEMBERS} (spec 004 §2.3, amended 2026-10-29)", file=sys.stderr)
        return None
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


# ----------------------------------------------------------------------------- stage 10: report
SCRIPTS_DIR = Path(__file__).resolve().parent
RENDER_STEMS: tuple[str, ...] = ("melodic", "drums", "bass")
RENDER_STEMS_MODES: tuple[str, ...] = ("solo", "demucs")
DEFAULT_RENDER_STEMS = "solo"     # Demucs on our own render is only for work on the separation stage itself
ITERATION_PHASE = "calibration"

Runner = Callable[[Sequence[str]], None]
"""Runs one external command (separate.py, compare_audio.py, generate_report.py, ffmpeg); raises on failure."""


class ReportError(RuntimeError):
    """A report-stage step failed or did not produce its artifact. Never swallowed: reports are mandatory."""


def subprocess_runner(argv: Sequence[str]) -> None:
    """Default :data:`Runner`: stdout is discarded (the CLI's stdout carries one path), stderr is inherited."""
    subprocess.run([str(a) for a in argv], check=True, stdout=subprocess.DEVNULL)


class GateVerdict(BaseModel):
    verdict: str
    message: str = ""


class IterationEntry(BaseModel):
    """One row of ``iterations.json`` as ``generate_report.py`` reads it (paths relative to vNNN/)."""

    version: int
    label: str = ""
    similarity: float | None      # None: the editability detector rejected this render (no score exists)
    editability: str | None = None
    section_aware: float | None = None
    was_best: bool = False
    phase: str = ITERATION_PHASE
    render_path: str | None = None
    strudel_path: str | None = None
    stem_melodic_path: str | None = None
    stem_drums_path: str | None = None
    stem_bass_path: str | None = None


class IterationsManifest(BaseModel):
    iterations: list[IterationEntry]
    best_similarity: float
    matched_promoted: bool        # False: no iteration is byte-identical to the promoted render.wav
    gate: GateVerdict | None = None


class ReportResult(BaseModel):
    report: Path
    render_stems: list[str]
    stem_comparison: str
    iterations: str | None


class _Cal(BaseModel):
    n: int
    wav: Path
    strudel: Path | None
    similarity: float | None
    section_aware: float | None
    editability: str | None


def _find_audio(directory: Path, stem: str) -> Path | None:
    # "melodic" falls back to the legacy "piano" stem exactly like generate_report.py's find_stem
    for name in (stem, "piano") if stem == "melodic" else (stem,):
        for ext in (".wav", ".mp3"):
            p = directory / f"{name}{ext}"
            if p.is_file():
                return p
    return None


def _read_scores(cmp_json: Path) -> tuple[float | None, float | None, str | None]:
    """(overall, section-aware, editability). A rejected run (``comparison: null``) has NO score: overall is None,
    never 0.0."""
    doc = json.loads(cmp_json.read_text(encoding="utf-8"))
    edit = doc.get("editability")
    inner = doc.get("comparison")
    if not isinstance(inner, dict) or not isinstance(inner.get("overall_similarity"), (int, float)):
        return None, None, str(edit) if edit is not None else "fail"
    sa = inner.get("section_aware_similarity")
    return float(inner["overall_similarity"]), (float(sa) if isinstance(sa, (int, float)) else None), \
        str(edit) if edit is not None else None


def _collect_cals(workdir: Path) -> list[_Cal]:
    """``calN.{wav,strudel,cmp.json}`` of an auto-calibrate workdir; else the single ``best.*`` render
    (the synth stage renders once). Entries lacking a wav or a cmp.json are skipped."""
    found: list[_Cal] = []
    for wav in sorted(workdir.glob("cal*.wav"), key=lambda p: p.name):
        m = re.fullmatch(r"cal(\d+)\.wav", wav.name)
        cmp_json = workdir / f"cal{m.group(1)}.cmp.json" if m else None
        if m and cmp_json is not None and cmp_json.is_file():
            sim, sa, edit = _read_scores(cmp_json)
            strudel = workdir / f"cal{m.group(1)}.strudel"
            found.append(_Cal(n=int(m.group(1)), wav=wav, strudel=strudel if strudel.is_file() else None,
                              similarity=sim, section_aware=sa, editability=edit))
    if not found and (workdir / "best.wav").is_file() and (workdir / "best.cmp.json").is_file():
        sim, sa, edit = _read_scores(workdir / "best.cmp.json")
        strudel = workdir / "best.strudel"
        found.append(_Cal(n=1, wav=workdir / "best.wav", strudel=strudel if strudel.is_file() else None,
                          similarity=sim, section_aware=sa, editability=edit))
    return sorted(found, key=lambda c: c.n)


def _need(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ReportError(f"{label} was not produced: {path}")
    return path


EMBED_BITRATE = "128k"   # report audio is embedded as base64: keep it small, it is for listening not scoring


def _fresh(twin: Path, *sources: Path) -> bool:
    """A twin is reusable only when it exists, is non-empty and is not older than any of its sources."""
    return twin.is_file() and twin.stat().st_size > 0 and all(twin.stat().st_mtime >= src.stat().st_mtime
                                                               for src in sources)


def _to_mp3(run: Runner, src: Path, dst: Path, *, reencode: bool = True) -> None:
    """Write an embed-grade mp3 of ``src`` (an mp3 source is copied only when ``reencode`` is False)."""
    if src.suffix == ".mp3" and not reencode:
        shutil.copyfile(src, dst)
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-b:a", EMBED_BITRATE, dst])
    _need(dst, f"{dst.name}")


def _write_iterations(vdir: Path, cals: list[_Cal], *, workdir: Path, render: Path, with_stems: bool, force: bool,
                      python: str, run: Runner, gate: GateVerdict | None) -> IterationsManifest:
    """Copy each iteration's render (+ stems, + strudel) under ``vNNN/iterations/`` and describe them."""
    idir = vdir / "iterations"
    idir.mkdir(exist_ok=True)
    matched = [c.n for c in cals if filecmp.cmp(c.wav, render, shallow=False)]
    best_n = matched[0] if matched else None      # lineage: best == byte-identical to the promoted render, nothing else
    if best_n is None:
        print(f"pipeline_helpers: WARNING: no iteration in {workdir} is byte-identical to {render}; "
              f"history is from another run, no iteration is marked best", file=sys.stderr)
    entries: list[IterationEntry] = []
    for c in cals:
        tag = f"cal{c.n}"
        entry = IterationEntry(version=c.n, label=f"iter {c.n}", editability=c.editability,
                               similarity=None if c.similarity is None else round(c.similarity, 6),
                               section_aware=None if c.section_aware is None else round(c.section_aware, 6),
                               was_best=c.n == best_n)
        mix = idir / f"{tag}.mp3"
        if force or not _fresh(mix, c.wav):
            _to_mp3(run, c.wav, mix)
        entry.render_path = f"iterations/{mix.name}"
        if c.strudel is not None:
            code = idir / f"output_iter_{c.n:03d}.strudel"
            shutil.copyfile(c.strudel, code)
            entry.strudel_path = f"iterations/{code.name}"
        if with_stems:
            outs = {s: idir / f"{tag}_{s}.mp3" for s in RENDER_STEMS}
            if force or not all(_fresh(p, c.wav) for p in outs.values()):
                if c.n == best_n:       # same audio as the promoted render: reuse its stems, no second Demucs pass
                    for s in RENDER_STEMS:
                        _to_mp3(run, _embed_twin(vdir, f"render_{s}"), outs[s], reencode=False)
                else:
                    with tempfile.TemporaryDirectory(dir=idir, prefix=".sep_") as tmp:
                        run([python, SCRIPTS_DIR / "separate.py", c.wav, tmp, "--mode", "full", "--prefix", tag])
                        for s in RENDER_STEMS:
                            src = _find_audio(Path(tmp), f"{tag}_{s}")
                            if src is None:
                                raise ReportError(f"separate.py produced no {tag}_{s} for {c.wav}")
                            _to_mp3(run, src, outs[s])
            entry.stem_melodic_path = f"iterations/{outs['melodic'].name}"
            entry.stem_drums_path = f"iterations/{outs['drums'].name}"
            entry.stem_bass_path = f"iterations/{outs['bass'].name}"
        entries.append(entry)
    scored = [e.similarity for e in entries if e.similarity is not None]
    manifest = IterationsManifest(iterations=entries, best_similarity=max(scored, default=0.0),
                                  matched_promoted=best_n is not None, gate=gate)
    (vdir / "iterations.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def _embed_twin(vdir: Path, name: str) -> Path:
    """The embed-grade mp3 of a render stem (``embed/<name>.mp3``)."""
    return _need(vdir / "embed" / f"{name}.mp3", f"embed/{name}.mp3")


def _need_stem(vdir: Path, name: str) -> Path:
    p = _find_audio(vdir, name)
    if p is None:
        raise ReportError(f"{name}.wav/mp3 missing in {vdir}")
    return p


def solo_render_stems(vdir: Path, render: Path, *, run: Runner) -> None:
    """``render_{melodic,drums,bass}.wav`` from the renderer (one voice group sounding per render) and
    ``render_vocals.wav`` as silence (we never generate vocals). Only files that are missing are produced; the
    measured sum-vs-render check is printed to stderr and stored in ``metadata.json`` (``render_stems_check``)."""
    import soundfile as sf

    import render_jobs
    import renderer
    import solo_stems
    code = (vdir / "output.strudel").read_text(encoding="utf-8")
    info = sf.info(str(render))
    seconds = info.frames / info.samplerate
    with tempfile.TemporaryDirectory(dir=vdir, prefix=".solo_") as tmp:
        jobs: list[tuple[Path, Path]] = []                    # (solo strudel, output wav) - the renders are independent
        for g in RENDER_STEMS:
            out = vdir / f"render_{g}.wav"
            solo = solo_stems.solo_code(code, g)
            if solo is None:                                  # the generator emitted no channel for this group
                solo_stems.write_silence(out, info.frames, info.samplerate, info.channels)
                continue
            src = Path(tmp) / f"{g}.strudel"
            src.write_text(solo, encoding="utf-8")
            jobs.append((src, out))

        def render_one(job: tuple[Path, Path]) -> None:
            src, out = job
            run(["node", renderer.renderer_script(), src, "-o", out, "-d", f"{seconds:.3f}"])
            _need(out, out.name)

        if jobs:
            workers = render_jobs.derive_jobs(len(jobs)).jobs   # cores + free memory vs one render's footprint; MIDIGREP_RENDER_JOBS
            if workers <= 1:
                for job in jobs:
                    render_one(job)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(render_one, job) for job in jobs]
                    try:
                        for f in futures:
                            f.result()                        # the first failure is raised; the pool then waits for the rest
                    except BaseException:
                        for f in futures:
                            f.cancel()
                        raise
    solo_stems.write_silence(vdir / "render_vocals.wav", info.frames, info.samplerate, info.channels)
    chk = solo_stems.sum_check(render, [vdir / f"render_{g}.wav" for g in RENDER_STEMS])
    print(f"render stems (solo): sum(bass+drums+melodic) vs render.wav: correlation {chk.correlation:.5f}, "
          f"residual {chk.residual_db:.1f} dB, gain ratio {chk.gain_ratio:.4f}", file=sys.stderr)
    meta_path = vdir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta["render_stems_check"] = {"source": "solo", **chk.model_dump()}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_report(track_dir: Path, version_dir: Path, *, seconds: float, calib_workdir: Path | None = None,
                 iteration_stems: bool = True, force: bool = False, config: Path | None = None,
                 gate: GateVerdict | None = None, run: Runner = subprocess_runner,
                 python: str = sys.executable, render_stems: str = DEFAULT_RENDER_STEMS) -> ReportResult:
    """Stage 10: make a promoted ``vNNN`` complete and write ``vNNN/report.html``. Idempotent.

    1. charts: ``compare_audio.py --chart`` into a temp dir; only ``comparison.png`` + ``chart_*.png`` are copied
       (that CLI path also writes ``<chart dir>/comparison.json`` — the stamped one in vNNN must not be touched);
    2. render stems -> ``render_{melodic,drums,bass,vocals}``: ``render_stems="solo"`` (default) renders
       ``output.strudel`` once per voice group with the other groups' ``$:`` channels commented out (see
       ``solo_stems.py``) and writes silence for vocals; ``"demucs"`` runs ``separate.py render.wav --prefix render``;
    3. per-stem comparison: ``compare_audio.py --stems`` -> ``stem_comparison.json`` + ``chart_stem_*.png``;
    4. iteration history from ``calib_workdir`` (``iterations.json`` + ``iterations/``); untouched when None;
    5. ``generate_report.py`` -> ``report.html``.

    Each artifact is skipped when present unless ``force``. ``run`` executes every external command, so tests
    inject a fake. Raises :class:`ReportError` (or whatever ``run`` raises) on any failure.
    """
    track_dir, vdir = Path(track_dir), Path(version_dir)
    if not _VERSION_RE.match(vdir.name):
        raise ValueError(f"version dir must be named vNNN, got {vdir.name!r}")
    version = int(vdir.name[1:])
    if render_stems not in RENDER_STEMS_MODES:
        raise ValueError(f"render_stems must be one of {RENDER_STEMS_MODES}, got {render_stems!r}")
    render = vdir / "render.wav"
    if not render.is_file():
        raise FileNotFoundError(f"report: {render} not found")
    original = write_instrumental(track_dir)   # the scoring reference: bass+drums+melodic, no vocals
    orig_stems = {s: _find_audio(track_dir, s) for s in RENDER_STEMS}
    missing = [s for s, p in orig_stems.items() if p is None]
    if missing:
        raise FileNotFoundError(f"report: original stems missing in {track_dir}: {', '.join(missing)}")
    cfg = ["--config", config] if config is not None else []
    rebuilt = force
    external_calls: list[int] = []
    base_run = run

    def run(argv: Sequence[object]) -> None:       # noqa: F811 — counts calls: any rebuilt artifact => new report
        external_calls.append(1)
        base_run(argv)

    if force or not (vdir / "comparison.png").is_file() or not (vdir / "chart_frequency.png").is_file():
        rebuilt = True
        with tempfile.TemporaryDirectory(dir=vdir, prefix=".charts_") as tmp:
            run([python, SCRIPTS_DIR / "compare_audio.py", original, render, "--chart", Path(tmp) / "comparison.png",
                 "-d", seconds, *cfg])
            _need(Path(tmp) / "comparison.png", "comparison.png")
            for png in [Path(tmp) / "comparison.png", *sorted(Path(tmp).glob("chart_*.png"))]:
                shutil.copyfile(png, vdir / png.name)

    if force or any(_find_audio(vdir, f"render_{s}") is None for s in RENDER_STEMS):
        rebuilt = True
        if render_stems == "demucs":
            run([python, SCRIPTS_DIR / "separate.py", render, vdir, "--mode", "full", "--prefix", "render"])
        else:
            solo_render_stems(vdir, render, run=run)
    for s_ in RENDER_STEMS:
        _need_stem(vdir, f"render_{s_}")

    rendered = {s_: _need_stem(vdir, f"render_{s_}") for s_ in RENDER_STEMS}   # compare_audio reads mp3 or wav
    stem_json = vdir / "stem_comparison.json"
    if force or not stem_json.is_file() or not (vdir / "chart_stem_overview.png").is_file():
        rebuilt = True
        pairs = [a for s_ in RENDER_STEMS
                 for a in (f"--original-{s_}", orig_stems[s_], f"--rendered-{s_}", rendered[s_])]
        run([python, SCRIPTS_DIR / "compare_audio.py", "--stems", "--output-dir", vdir, *pairs, "-d", seconds, *cfg])
        _need(stem_json, "stem_comparison.json")

    # Size + freshness: the report base64-embeds audio, so it gets 128k mp3 twins (never decoded from a lossy
    # intermediate: encoded from the Demucs file / original wav). A twin is reused only if not older than its source.
    if force or not _fresh(vdir / "render.mp3", render):
        rebuilt = True
        _to_mp3(run, render, vdir / "render.mp3")                      # render.wav stays: it is the canonical artifact
    embed = vdir / "embed"
    embed.mkdir(exist_ok=True)
    twin_jobs = [(f"render_{s_}", rendered[s_]) for s_ in RENDER_STEMS]
    twin_jobs += [(n, p) for n in (*RENDER_STEMS, "vocals") if (p := _find_audio(track_dir, n)) is not None]
    for name, src in twin_jobs:        # the track-level wavs and Demucs' render_*.mp3 are only ever read
        if force or not _fresh(embed / f"{name}.mp3", src):
            rebuilt = True
            _to_mp3(run, src, embed / f"{name}.mp3")

    iterations_name: str | None = "iterations.json" if (vdir / "iterations.json").is_file() else None
    if calib_workdir is not None:
        cals = _collect_cals(Path(calib_workdir))
        if cals:
            before = (vdir / "iterations.json").read_bytes() if iterations_name else b""
            _write_iterations(vdir, cals, workdir=Path(calib_workdir), render=render, with_stems=iteration_stems, force=force, python=python,
                              run=run, gate=gate)
            iterations_name = "iterations.json"
            rebuilt = rebuilt or (vdir / "iterations.json").read_bytes() != before
        else:
            raise ReportError(f"no calN.* / best.* iteration renders with a cmp.json in {calib_workdir}")

    report = vdir / "report.html"
    meta_path = vdir / "metadata.json"

    def update_meta(**fields: object) -> bool:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        updated = {**meta, **fields}
        if updated != meta:
            meta_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return updated != meta

    # Everything generate_report.py READS (gate in metadata.json, iterations.json) is on disk before it runs.
    gate_changed = update_meta(**({"gate": gate.model_dump()} if gate is not None else {}),
                               render_stems=[rendered[s].name for s in RENDER_STEMS],
                               stem_comparison=stem_json.name, iterations=iterations_name)
    if rebuilt or external_calls or gate_changed or not report.is_file():
        run([python, SCRIPTS_DIR / "generate_report.py", track_dir, "-v", version, "-o", report])
    _need(report, "report.html")
    update_meta(report=report.name)           # a version declares its report only once it exists
    return ReportResult(report=report, render_stems=[rendered[s].name for s in RENDER_STEMS],
                        stem_comparison=stem_json.name, iterations=iterations_name)



INSTRUMENTAL_STEMS: tuple[str, ...] = ("bass", "drums", "melodic")
INSTRUMENTAL_FILE = "instrumental.wav"


def write_instrumental(track_dir: Path, *, force: bool = False) -> Path:
    """``<track>/instrumental.wav`` = bass + drums + melodic, the scoring REFERENCE (product decision: vocals are
    never generated, so the target is the mix without the singer).

    A plain sample-aligned sum (every stem from sample 0; a shorter stem is zero-padded to the longest). No
    loudness normalisation: the only gain is clipping protection - when the summed peak exceeds 1.0 the whole mix
    is divided by that peak (one gain, inter-stem balance untouched). Written as 32-bit float. Rebuilt when
    missing or older than any stem. ``vocals.wav`` stays on disk, unused. Raises if a stem is missing."""
    import numpy as np
    import soundfile as sf

    track_dir = Path(track_dir)
    out = track_dir / INSTRUMENTAL_FILE
    stems = [track_dir / f"{s}.wav" for s in INSTRUMENTAL_STEMS]
    if not stems[2].is_file() and (track_dir / "piano.wav").is_file():
        stems[2] = track_dir / "piano.wav"        # legacy name of the melodic stem (see _find_audio)
    missing = [p.name for p in stems if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"instrumental: {', '.join(missing)} missing in {track_dir}")
    if out.is_file() and not force and out.stat().st_mtime >= max(p.stat().st_mtime for p in stems):
        return out
    loaded, sr = [], None
    for p in stems:
        y, fs = sf.read(str(p), dtype="float32", always_2d=True)
        if sr is None:
            sr = fs
        elif fs != sr:
            raise ValueError(f"instrumental: {p.name} is {fs} Hz but the other stems are {sr} Hz")
        loaded.append(y)
    ch = min(y.shape[1] for y in loaded)
    n = max(len(y) for y in loaded)
    mix = np.zeros((n, ch), dtype=np.float64)
    for y in loaded:
        mix[: len(y)] += y[:, :ch] if y.shape[1] == ch else y.mean(axis=1, keepdims=True)
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix /= peak
    sf.write(str(out), mix.astype(np.float32), sr, subtype="FLOAT")
    return out


def _has_instrumental_stems(d: Path) -> bool:
    return all((d / f"{s}.wav").is_file() for s in ("bass", "drums")) and any(
        (d / f"{s}.wav").is_file() for s in ("melodic", "piano"))


def resolve_stem_dir(original_audio: str | Path, output_dir: str | Path | None = None) -> Path:
    """The directory that holds the track's separated stems (bass/drums/melodic.wav).

    That is the stem-cache dir ``.cache/stems/<key>/``: the parent of ``original_audio`` when the pipeline passes
    the cached copy, otherwise (a cache miss hands over the user's own input file, whose folder has no stems)
    the version dir ``output_dir`` or its parent. Raises FileNotFoundError, naming what was searched, when none
    of them has the stems."""
    candidates = [Path(original_audio).resolve().parent]
    if output_dir:
        out = Path(output_dir).resolve()
        candidates += [out, out.parent]
    for d in candidates:
        if _has_instrumental_stems(d):
            return d
    raise FileNotFoundError("no bass/drums/melodic stems in any of: " + ", ".join(str(c) for c in candidates))


def instrumental_path(track_dir: Path) -> Path:
    """The scoring reference of a track (built on demand from its stems)."""
    return write_instrumental(track_dir)


CONTROL_NONE_EXIT = 3   # `control`: lookup worked, there is simply no other same-genre reference track


def control_track_path(genre: str, exclude_key: str, repo_root: Path | None = None) -> Path | None:
    """``instrumental.wav`` of the first ``eval/datasets/reference_tracks.yaml`` track with the same genre and a
    different cache_key - the chance-control song for ``compare_audio.py --control``. It is the control's
    INSTRUMENTAL (built from its stems on demand), because the scored reference is an instrumental too.
    A candidate with an ``original.wav`` but no complete stems is skipped (the next one is tried); the lookup raises
    ``FileNotFoundError`` only when such candidates exist and none of them has stems."""
    import yaml  # local: only this subcommand needs it

    root = repo_root or Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "eval" / "datasets" / "reference_tracks.yaml").read_text(encoding="utf-8")) or {}
    unusable: list[str] = []
    for track in doc.get("tracks") or []:
        key = track.get("cache_key")
        if not key or key == exclude_key or track.get("genre") != genre:
            continue
        d = root / ".cache" / "stems" / key
        if not (d / "original.wav").is_file():
            continue
        try:
            return instrumental_path(d)
        except FileNotFoundError as exc:   # original.wav but no (complete) stems: try the next candidate
            unusable.append(f"{key}: {exc}")
    if unusable:
        raise FileNotFoundError("every same-genre candidate lacks its stems - " + "; ".join(unusable))
    return None


CALIBRATION_FIELDS = (6, 9)   # auto-calibrate.sh --start / BEST params: "BM SG CL LLPF HG MG [BL [LH [SU [stem|off]]]]" (SU = sub octave up 0/1;
                              # the optional 10th token is the calibrator-chosen --bass-shape, only after all 9 numbers)
BASS_SHAPES = ("stem", "off")


def params_valid(text: str) -> bool:
    """True when ``text`` is a best.params line in the CURRENT calibration format: 6 to 8 numeric fields.

    The pre-"no vocals" format carried a removed CV (cal-vocal) field and always had 9; ``auto-calibrate.sh
    --start`` rejects it (exit 64). Such a file is stale, not migratable: the best.wav / comparison next to it
    were rendered by the old recorder and scored on the old scale, so the whole stage is rerun."""
    fields = text.split()
    if len(fields) == CALIBRATION_FIELDS[1] + 1 and fields[-1] in BASS_SHAPES:
        fields = fields[:-1]
    if not CALIBRATION_FIELDS[0] <= len(fields) <= CALIBRATION_FIELDS[1]:
        return False
    try:
        vals = [float(f) for f in fields]
    except ValueError:
        return False
    if len(vals) == 9:
        # The 9th field is SU (0/1) after BL/LH. The removed pre-branch layout also had 9 fields, with the cal-vocal
        # CV (~1) in 7th place where a cutoff must be 0 or >= 20 Hz - that is how the two are told apart.
        bl, lh, su = vals[6], vals[7], vals[8]
        if su not in (0.0, 1.0) or any(v != 0 and v < 20 for v in (bl, lh)):
            return False
    return True


def newer_than(stamp: Path, *roots: Path) -> bool:
    """True when ``stamp`` is missing or any regular file under ``roots`` (other than ``stamp``) is newer than it.

    Stage 6 uses it to decide whether the hosted copy of the pack is still current: ``.uploaded`` is written
    after an upload, so anything a later stage rebuilt in the pack has a newer mtime."""
    stamp = Path(stamp)
    if not stamp.is_file():
        return True
    ref = stamp.stat().st_mtime_ns
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for f in files:
            if f.is_file() and f.resolve() != stamp.resolve() and f.stat().st_mtime_ns > ref:
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Helpers for scripts/editable-pipeline.sh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slug").add_argument("title")
    nb = sub.add_parser("num-bars")
    nb.add_argument("duration", type=float)
    nb.add_argument("bpm", type=float)
    rs = sub.add_parser("render-seconds", help="render length / compare window: whole bars of the track, in seconds")
    rs.add_argument("duration", type=float)
    rs.add_argument("bpm", type=float)
    sub.add_parser("next-version").add_argument("track_dir", type=Path)
    ct = sub.add_parser("control", help="instrumental.wav of another same-genre reference track (chance control)")
    ct.add_argument("--genre", required=True)
    ct.add_argument("--exclude-key", required=True, help="cache_key of the track being scored")
    pv = sub.add_parser("params-valid", help="print ok|stale: is this best.params in the current calibration format")
    pv.add_argument("file", type=Path)
    nw = sub.add_parser("newer", help="print stale|fresh: is any file under PATH newer than STAMP (or STAMP missing)")
    nw.add_argument("stamp", type=Path)
    nw.add_argument("paths", nargs="+", type=Path)
    ins = sub.add_parser("instrumental", help="write <track>/instrumental.wav (bass+drums+melodic) and print its path")
    ins.add_argument("track_dir", type=Path)
    ins.add_argument("--force", action="store_true")
    rf = sub.add_parser("record-floor")
    rf.add_argument("--thresholds", required=True, type=Path)
    rf.add_argument("--mode", required=True)
    rf.add_argument("--genre", required=True)
    rf.add_argument("--comparison", required=True, type=Path)
    rf.add_argument("--run", required=True)
    rf.add_argument("--margin", type=float, default=0.02)
    rf.add_argument("--labels", type=Path, default=None, help="genre label registry (default eval/datasets/genre_labels.yaml)")
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
    pr.add_argument("--env-correction", type=Path, default=None,
                    help="per-bar correction JSON the generator applied; persisted as vNNN/env_correction.json")
    rp = sub.add_parser("report", help="stage 10: charts, render stems, per-stem comparison, iterations, report.html")
    rp.add_argument("--track-dir", required=True, type=Path)
    rp.add_argument("--version", required=True, help="vNNN or N")
    rp.add_argument("--seconds", required=True, type=float, help="compare window (same -d the run was scored with)")
    rp.add_argument("--calib-workdir", type=Path, default=None, help="auto-calibrate workdir (calN.* / best.*)")
    rp.add_argument("--no-iteration-stems", action="store_true", help="skip Demucs on each iteration render")
    rp.add_argument("--force", action="store_true", help="redo every artifact")
    rp.add_argument("--config", type=Path, default=None, help="synth_config.json for compare_audio.py")
    rp.add_argument("--render-stems", choices=RENDER_STEMS_MODES, default=DEFAULT_RENDER_STEMS,
                    help="how render_{melodic,drums,bass,vocals} are made: solo = the renderer, one voice group "
                         "sounding per render (vocals = silence); demucs = separate.py on render.wav (only when "
                         "working on the separation stage)")
    rp.add_argument("--gate-verdict", choices=("PASS", "FAIL"), default=None)
    rp.add_argument("--gate-message", default="")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "report":
            vname = args.version if args.version.startswith("v") else f"v{int(args.version):03d}"
            gate = GateVerdict(verdict=args.gate_verdict, message=args.gate_message) if args.gate_verdict else None
            res = build_report(args.track_dir, args.track_dir / vname, seconds=args.seconds,
                               calib_workdir=args.calib_workdir, iteration_stems=not args.no_iteration_stems,
                               force=args.force, config=args.config, gate=gate,
                               render_stems=args.render_stems)
            print(res.report)
        elif args.cmd == "slug":
            print(slug_from_title(args.title))
        elif args.cmd == "render-seconds":
            print(f"{render_seconds(args.duration, args.bpm):g}")
        elif args.cmd == "num-bars":
            print(num_bars(args.duration, args.bpm))
        elif args.cmd == "record-floor":
            res = record_mode_floor(args.thresholds, mode=args.mode, genre=args.genre, comparison_path=args.comparison,
                                    run=args.run, margin=args.margin, labels_path=args.labels)
            print(json.dumps({"recorded": res is not None, "measured": res}))
        elif args.cmd == "stamp":
            m = stamp_track_metadata(args.track_dir, genre_override=args.genre, detector_json=args.detector_json)
            print(json.dumps({k: m.get(k) for k in ("bpm", "key", "genre", "genre_ref", "genre_basis", "genre_uncertain", "genre_candidates", "genre_detected", "genre_confidence", "genre_override", "duration", "style_heuristic")}))
        elif args.cmd == "instrumental":
            print(write_instrumental(args.track_dir, force=args.force))
        elif args.cmd == "params-valid":
            ok = args.file.is_file() and params_valid(args.file.read_text())
            print("ok" if ok else "stale")
        elif args.cmd == "newer":
            print("stale" if newer_than(args.stamp, *args.paths) else "fresh")
        elif args.cmd == "control":
            # exit 0 + path: control found; exit 3: legitimately none (no other same-genre reference track with an
            # original.wav); exit 1 (the generic handler below): the lookup itself failed (bad/missing yaml, ...)
            try:
                found = control_track_path(args.genre, args.exclude_key)
            except Exception as exc:  # noqa: BLE001 - bad yaml / bad shape: the lookup itself failed -> fail closed (exit 1)
                print(f"pipeline_helpers: control lookup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            if found is None:
                print(f"pipeline_helpers: no other {args.genre} reference track with an original.wav", file=sys.stderr)
                return CONTROL_NONE_EXIT
            print(found)
        elif args.cmd == "next-version":
            print(next_version_dir(args.track_dir))
        else:
            meta = RunMeta(generator=args.generator, render=args.render, compare=args.compare)
            print(promote_run(args.track_dir, args.mode, args.strudel, args.wav, args.comparison, meta,
                                     env_correction=args.env_correction))
    except (FileNotFoundError, ValueError, ReportError, OSError, subprocess.CalledProcessError) as exc:
        print(f"pipeline_helpers: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
