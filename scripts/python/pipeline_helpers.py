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
    pipeline_helpers.py next-version <track_dir>
    pipeline_helpers.py control --genre G --exclude-key K
    pipeline_helpers.py promote --track-dir D --mode M --strudel F --wav F --comparison F [--env-correction F] \\
        --generator "<knobs>" --render "<capture>" --compare "<args>" [--vocal-mode instrument]
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


# ----------------------------------------------------------------------------- stage 10: report
SCRIPTS_DIR = Path(__file__).resolve().parent
RENDER_STEMS: tuple[str, ...] = ("melodic", "drums", "bass")
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


def build_report(track_dir: Path, version_dir: Path, *, seconds: float, calib_workdir: Path | None = None,
                 iteration_stems: bool = True, force: bool = False, config: Path | None = None,
                 gate: GateVerdict | None = None, run: Runner = subprocess_runner,
                 python: str = sys.executable) -> ReportResult:
    """Stage 10: make a promoted ``vNNN`` complete and write ``vNNN/report.html``. Idempotent.

    1. charts: ``compare_audio.py --chart`` into a temp dir; only ``comparison.png`` + ``chart_*.png`` are copied
       (that CLI path also writes ``<chart dir>/comparison.json`` — the stamped one in vNNN must not be touched);
    2. render stems: ``separate.py render.wav --prefix render`` -> ``render_{melodic,drums,bass,vocals}``;
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
    render = vdir / "render.wav"
    if not render.is_file():
        raise FileNotFoundError(f"report: {render} not found")
    original = track_dir / "original.wav"
    if not original.is_file():
        raise FileNotFoundError(f"report: {original} not found")
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
        run([python, SCRIPTS_DIR / "separate.py", render, vdir, "--mode", "full", "--prefix", "render"])
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



CONTROL_NONE_EXIT = 3   # `control`: lookup worked, there is simply no other same-genre reference track


def control_track_path(genre: str, exclude_key: str, repo_root: Path | None = None) -> Path | None:
    """``original.wav`` of the first ``eval/datasets/reference_tracks.yaml`` track with the same genre and
    a different cache_key whose file exists - the chance-control song for ``compare_audio.py --control``."""
    import yaml  # local: only this subcommand needs it

    root = repo_root or Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "eval" / "datasets" / "reference_tracks.yaml").read_text(encoding="utf-8")) or {}
    for track in doc.get("tracks") or []:
        key = track.get("cache_key")
        if not key or key == exclude_key or track.get("genre") != genre:
            continue
        wav = root / ".cache" / "stems" / key / "original.wav"
        if wav.is_file():
            return wav
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Helpers for scripts/editable-pipeline.sh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slug").add_argument("title")
    nb = sub.add_parser("num-bars")
    nb.add_argument("duration", type=float)
    nb.add_argument("bpm", type=float)
    sub.add_parser("next-version").add_argument("track_dir", type=Path)
    ct = sub.add_parser("control", help="original.wav of another same-genre reference track (chance control)")
    ct.add_argument("--genre", required=True)
    ct.add_argument("--exclude-key", required=True, help="cache_key of the track being scored")
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
    rp = sub.add_parser("report", help="stage 10: charts, render stems, per-stem comparison, iterations, report.html")
    rp.add_argument("--track-dir", required=True, type=Path)
    rp.add_argument("--version", required=True, help="vNNN or N")
    rp.add_argument("--seconds", required=True, type=float, help="compare window (same -d the run was scored with)")
    rp.add_argument("--calib-workdir", type=Path, default=None, help="auto-calibrate workdir (calN.* / best.*)")
    rp.add_argument("--no-iteration-stems", action="store_true", help="skip Demucs on each iteration render")
    rp.add_argument("--force", action="store_true", help="redo every artifact")
    rp.add_argument("--config", type=Path, default=None, help="synth_config.json for compare_audio.py")
    rp.add_argument("--gate-verdict", choices=("PASS", "FAIL"), default=None)
    rp.add_argument("--gate-message", default="")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "report":
            vname = args.version if args.version.startswith("v") else f"v{int(args.version):03d}"
            gate = GateVerdict(verdict=args.gate_verdict, message=args.gate_message) if args.gate_verdict else None
            res = build_report(args.track_dir, args.track_dir / vname, seconds=args.seconds,
                               calib_workdir=args.calib_workdir, iteration_stems=not args.no_iteration_stems,
                               force=args.force, config=args.config, gate=gate)
            print(res.report)
        elif args.cmd == "slug":
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
            meta = RunMeta(generator=args.generator, render=args.render, compare=args.compare,
                           vocal_mode=args.vocal_mode)
            print(promote_run(args.track_dir, args.mode, args.strudel, args.wav, args.comparison, meta,
                                     env_correction=args.env_correction))
    except (FileNotFoundError, ValueError, ReportError, OSError, subprocess.CalledProcessError) as exc:
        print(f"pipeline_helpers: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
