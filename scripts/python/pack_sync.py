#!/usr/bin/env python3
"""Incremental sample-pack hosting: which files does Strudel actually fetch, and which of them changed.

Strudel's ``samples(url)`` fetches the manifest (``samples.json`` / ``strudel.json``) and then only the files that
manifest references. Everything else in a pack directory (``drums_bands.json``, ``*.mid``, ``pack.json``,
``register.json`` ... generator inputs) is never served, so a change there must not trigger an upload.

``.uploaded.json`` (``{base, bucket, files: {relpath: sha256}}``) records what was hosted last; the next run uploads
only files whose hash changed or that are new. A legacy ``.uploaded`` (plain base string) has no hashes, so the
state is "unknown" and one full upload is planned.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAMES = ("samples.json", "strudel.json")
STATE_NAME = ".uploaded.json"


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """What to do for one pack: ``upload`` (relpaths), ``deleted`` (hosted before, no longer served), ``full``."""
    upload: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    full: bool = False
    served: int = 0

    @property
    def is_noop(self) -> bool:
        return not self.upload


def _refs(value: object) -> Iterator[str]:
    match value:
        case str():
            yield value
        case list() | tuple():
            for v in value:
                yield from _refs(v)
        case dict():
            for k, v in value.items():
                if not str(k).startswith("_"):  # _base and friends are not files
                    yield from _refs(v)


def served_files(pack_dir: Path) -> list[str]:
    """Pack-relative paths Strudel can fetch: every manifest found in the pack plus the files it references.

    Manifests are discovered (any ``samples.json``/``strudel.json`` below ``pack_dir``); a reference is resolved
    against the directory of the manifest that names it (``_base`` ends at that directory). Absolute URLs and
    references to files that do not exist are skipped."""
    pack_dir = Path(pack_dir)
    out: set[str] = set()
    for name in MANIFEST_NAMES:
        for manifest in pack_dir.rglob(name):
            if not manifest.is_file():
                continue
            out.add(manifest.relative_to(pack_dir).as_posix())
            try:
                data = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            for ref in _refs(data):
                if "://" in ref:
                    continue
                target = manifest.parent / ref
                if target.is_file():
                    out.add(target.relative_to(pack_dir).as_posix())
    return sorted(out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_served(pack_dir: Path) -> dict[str, str]:
    return {rel: _sha256(Path(pack_dir) / rel) for rel in served_files(pack_dir)}


def load_state(pack_dir: Path) -> dict | None:
    """The recorded upload state, or None when absent/legacy/unreadable (-> unknown)."""
    try:
        state = json.loads((Path(pack_dir) / STATE_NAME).read_text())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and isinstance(state.get("files"), dict) else None


def plan_sync(pack_dir: Path, base: str, bucket: str = "", *, force: bool = False,
              hashes: dict[str, str] | None = None) -> SyncPlan:
    """Diff the served files against the recorded state. Unknown state, another base/bucket, or ``force`` -> full."""
    hashes = hash_served(pack_dir) if hashes is None else hashes
    state = load_state(pack_dir)
    if force or state is None or state.get("base") != base or state.get("bucket", bucket) != bucket:
        return SyncPlan(upload=sorted(hashes), full=True, served=len(hashes))
    old: dict[str, str] = state["files"]
    return SyncPlan(
        upload=sorted(r for r, h in hashes.items() if old.get(r) != h),
        deleted=sorted(set(old) - set(hashes)),
        served=len(hashes),
    )


def write_state(pack_dir: Path, base: str, bucket: str, hashes: dict[str, str]) -> None:
    tmp = Path(pack_dir) / (STATE_NAME + ".tmp")
    tmp.write_text(json.dumps({"base": base, "bucket": bucket, "files": hashes}, indent=1, sort_keys=True))
    tmp.replace(Path(pack_dir) / STATE_NAME)
