"""On-disk cache for the REFERENCE side of compare_audio.py (the original and the --control song).

compare_audio.py runs ~45 times per track and the original (instrumental.wav) and the control never
change inside a track, yet every call recomputed all of their features. Each entry here is keyed by
everything its value depends on:

  * the SHA-256 of the file's CONTENT (not its path or mtime),
  * the sample rate, the analysis duration and the exact sample count that was analysed,
  * every analysis parameter the caller lists (n_mfcc, hop, band table, target RMS, ...),
  * a version hash of the feature code itself: compare_audio.py + this file + the librosa / numpy /
    scipy / soxr versions. Edit the feature code and every entry misses.

Entries live under <repo>/.cache/compare-features/ (git-ignored), one pickle per key, written to a temp file
in the same directory and moved into place with os.replace(): a concurrent compare sees either no file or a
complete one, and two writers of one key write identical bytes. A corrupt / unreadable entry is a miss.

MIDIGREP_COMPARE_CACHE=0 bypasses it (nothing read, nothing written); MIDIGREP_COMPARE_CACHE_DIR relocates it.
A hit returns exactly the object the computation produced (pickle round-trips floats and numpy arrays
bit-for-bit), so scores are identical with and without the cache.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FORMAT = 1   # bump when the entry layout changes


def enabled() -> bool:
    return os.environ.get("MIDIGREP_COMPARE_CACHE", "1").strip().lower() not in ("0", "off", "false", "no")


def cache_dir() -> Path:
    override = os.environ.get("MIDIGREP_COMPARE_CACHE_DIR")
    return Path(override) if override else _HERE.parents[1] / ".cache" / "compare-features"


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Hash of the feature code and the numeric libraries it runs on."""
    h = hashlib.sha256()
    for name in ("compare_audio.py", "compare_cache.py"):
        h.update(name.encode())
        h.update((_HERE / name).read_bytes())
    for mod in ("librosa", "numpy", "scipy", "soxr", "resampy", "numba"):
        try:
            h.update(f"{mod}={__import__(mod).__version__}".encode())
        except Exception:
            h.update(f"{mod}=absent".encode())
    return h.hexdigest()[:20]


def file_digest(path) -> str:
    """SHA-256 of the file's bytes. Not memoised: a (path, size, mtime) memo can serve a stale digest for a file
    rewritten with the same size inside one mtime tick, and each compare is its own process anyway."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_path(parts: dict) -> Path:
    blob = json.dumps({"format": _FORMAT, "code": code_version(), **parts}, sort_keys=True, default=repr)
    return cache_dir() / (hashlib.sha256(blob.encode()).hexdigest() + ".pkl")


def _load(path: Path):
    try:
        with open(path, "rb") as f:
            return True, pickle.load(f)
    except FileNotFoundError:
        return False, None
    except Exception:   # truncated / corrupt / unpicklable entry: recompute
        return False, None


def _store(path: Path, value) -> None:
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        tmp = None
    except Exception:   # a cache that cannot be written must never fail the compare
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def get_or_compute(parts: dict, compute):
    """Return the cached value for `parts` or compute(), store and return it. Disabled -> compute() only."""
    if not enabled():
        return compute()
    path = _entry_path(parts)
    ok, value = _load(path)
    if ok:
        return value
    value = compute()
    if value is not None:   # a failed computation (None) is never remembered
        _store(path, value)
    return value
