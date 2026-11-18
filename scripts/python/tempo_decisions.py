"""The registry of DECIDED tempos (``eval/datasets/tempo_decisions.yaml``): cache_key -> the coarse bpm a track is played at.

A decided tempo (``editable-pipeline.sh --bpm B``, an adopted autopilot tempo trial) used to live only as
``bpm_decided`` in ``<track>/metadata.json``. That file is not ours alone: the Go stem cache rewrote it with
title/url/video_id whenever it re-separated or re-titled a track, and stage 1 then added a fresh ``vNNN`` carrying the
extraction's tempogram estimate - so the next run silently went back to the wrong tempo (BibiBabydoll: 150 decided,
112.5 again) and every restart had to be given ``--bpm`` by hand. The registry sits beside the genre label registries,
is committed, and survives cache wipes and re-extractions; ``pipeline_helpers.stamp_track_metadata`` reads it before
anything the extraction says.

Trial copies (``autopilot.TRIAL_TAG`` in the key) never write here: a trial is a question, not a decision.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "eval" / "datasets" / "tempo_decisions.yaml"
TRIAL_MARK = " ~trial "          # autopilot.TRIAL_TAG (not imported: autopilot pulls in the whole batch machinery)

_HEADER = ("# DECIDED tempos (coarse bpm, refined by phase coherence at every stamp), written by\n"
           "# `pipeline_helpers.py set-bpm` (editable-pipeline.sh --bpm B, an adopted autopilot tempo trial).\n"
           "# Every stamp of the track starts from this bpm instead of the extraction's estimate. To decide another tempo\n"
           "# run the track with --bpm B (or edit bpm_coarse); to hand it back to the extraction delete the row AND\n"
           "# `bpm_decided` in the track's metadata.json (the pre-registry fallback).\n")


def is_trial(key: str) -> bool:
    return TRIAL_MARK in key


def _path(path: Path | str | None) -> Path:
    """The explicit path, else ``MIDIGREP_TEMPO_DECISIONS`` (tests point it at a temp file), else the committed registry."""
    import os
    return Path(path or os.environ.get("MIDIGREP_TEMPO_DECISIONS") or REGISTRY_PATH)


def load(path: Path | str | None = None) -> dict[str, dict]:
    """{cache_key: {bpm_coarse, by, date, evidence}}. A row without a positive bpm_coarse is an error, not skipped."""
    import yaml
    p = _path(path)
    if not p.is_file():
        return {}
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for row in doc.get("decisions") or []:
        key, bpm = row.get("cache_key"), row.get("bpm_coarse")
        if not key or not isinstance(bpm, (int, float)) or bpm <= 0:
            raise ValueError(f"{p}: every decision needs cache_key and a positive bpm_coarse, got {row!r}")
        out[key] = {"bpm_coarse": float(bpm), "by": row.get("by", ""), "date": row.get("date", ""),
                    "evidence": row.get("evidence") or {}}
    return out


def get(key: str, path: Path | str | None = None) -> dict | None:
    return load(path).get(key)


def record(key: str, bpm_coarse: float, *, by: str, evidence: dict | None = None,
           path: Path | str | None = None) -> Path | None:
    """Write (or replace) ``key``'s decision atomically. A trial copy is never recorded (returns None)."""
    import yaml
    if is_trial(key):
        return None
    p = _path(path)
    doc = (yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None) or {}
    rows = [r for r in (doc.get("decisions") or []) if r.get("cache_key") != key]
    row = {"cache_key": key, "bpm_coarse": float(bpm_coarse), "by": by, "date": _dt.date.today().isoformat()}
    if evidence:
        row["evidence"] = evidence
    rows.append(row)
    rows.sort(key=lambda r: r["cache_key"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(_HEADER + yaml.safe_dump({"decisions": rows}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(p)
    return p
