# @layer: unit
# @regression
"""compare_audio.py recomputed every feature of the ORIGINAL and of the --control song on every call
(~45 compares per track). The reference side never changes inside a track, so it is cached on disk
(compare_cache.py) keyed by file content, analysis parameters and a version hash of the feature code.

  * a second compare against the same original does not recompute the original's features;
  * a cached compare returns exactly what an uncached one does (the whole results dict);
  * changing the original's bytes, or the feature-code version, invalidates the entry;
  * MIDIGREP_COMPARE_CACHE=0 bypasses it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_audio as C   # noqa: E402
import compare_cache        # noqa: E402

SR = 22050


def _song(path, seed, bpm=120.0, seconds=6.0, f0=110.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.02 * rng.standard_normal(len(t))
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        n = min(2000, len(y) - i)
        y[i:i + n] += 0.6 * np.exp(-np.arange(n) / 300.0) * rng.standard_normal(n)
    soundfile.write(str(path), y.astype(np.float32), SR)
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MIDIGREP_COMPARE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MIDIGREP_COMPARE_CACHE", raising=False)
    monkeypatch.setattr(C, "_QUIET_MODE", True)
    return {
        "orig": _song(tmp_path / "orig.wav", 1),
        "rend": _song(tmp_path / "rend.wav", 2, f0=130.0),
        "ctrl": _song(tmp_path / "ctrl.wav", 3, bpm=100.0, f0=90.0),
        "tmp": tmp_path,
    }


class Counter:
    """Counts how often the feature functions run (every call, whatever the input)."""
    NAMES = ("compute_spectral_features", "compute_mfcc_features", "compute_chroma_features",
             "compute_rhythm_features", "compute_frequency_bands")

    def __init__(self, monkeypatch):
        self.n = dict.fromkeys(self.NAMES, 0)
        for name in self.NAMES:
            real = getattr(C, name)

            def wrap(*a, _real=real, _name=name, **k):
                self.n[_name] += 1
                return _real(*a, **k)
            monkeypatch.setattr(C, name, wrap)

    def reset(self):
        for k in self.n:
            self.n[k] = 0


def test_second_compare_does_not_recompute_the_original(env, monkeypatch):
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    # whole-file features: once for the original, once for the render
    assert cnt.n["compute_spectral_features"] == 2
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 6)
    # second compare: only the render side is computed
    assert cnt.n["compute_spectral_features"] == 1
    assert cnt.n["compute_rhythm_features"] == 1
    assert cnt.n["compute_chroma_features"] == 1


def test_control_pair_is_not_rescored(env, monkeypatch):
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6, control_path=env["ctrl"])
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 6, control_path=env["ctrl"])
    # the render side only: the original AND the whole control-vs-original score come from the cache
    assert cnt.n["compute_spectral_features"] == 1


def test_cached_and_uncached_results_are_equal(env, monkeypatch):
    def run():
        return json.dumps(C.compare_audio(env["orig"], env["rend"], 6, control_path=env["ctrl"]), sort_keys=True)
    monkeypatch.setenv("MIDIGREP_COMPARE_CACHE", "0")
    uncached = run()
    monkeypatch.delenv("MIDIGREP_COMPARE_CACHE")
    cold, warm = run(), run()
    assert cold == uncached
    assert warm == uncached


def test_bypass_env_never_reads_or_writes(env, monkeypatch):
    monkeypatch.setenv("MIDIGREP_COMPARE_CACHE", "0")
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    C.compare_audio(env["orig"], env["rend"], 6)
    assert cnt.n["compute_spectral_features"] == 4
    assert not (env["tmp"] / "cache").exists()


def test_changed_original_bytes_invalidate(env, monkeypatch):
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    _song(Path(env["orig"]), 99, f0=220.0)           # same path, different content
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 6)
    assert cnt.n["compute_spectral_features"] == 2   # original recomputed


def test_feature_code_version_invalidates(env, monkeypatch):
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    monkeypatch.setattr(compare_cache, "code_version", lambda: "a-different-version")
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 6)
    assert cnt.n["compute_spectral_features"] == 2


def test_analysis_duration_is_part_of_the_key(env, monkeypatch):
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 4)
    assert cnt.n["compute_spectral_features"] == 2


def test_corrupt_entry_is_a_miss_not_a_failure(env, monkeypatch):
    C.compare_audio(env["orig"], env["rend"], 6)
    for p in (env["tmp"] / "cache").glob("*.pkl"):
        p.write_bytes(b"not a pickle")
    cnt = Counter(monkeypatch)
    res = C.compare_audio(env["orig"], env["rend"], 6)
    assert res["comparison"]["overall_similarity"] >= 0
    assert cnt.n["compute_spectral_features"] == 2


def test_tests_run_against_an_isolated_cache_not_the_repo_cache():
    # a warm repo cache made test_temporal_similarity's call-counting test fail (it passed only on a cold one)
    repo_cache = Path(__file__).resolve().parents[3] / ".cache" / "compare-features"
    assert compare_cache.cache_dir() != repo_cache
    assert "compare-cache" in str(compare_cache.cache_dir())


def test_same_size_rewrite_with_the_same_mtime_is_still_a_new_file(env, monkeypatch):
    import os
    cnt = Counter(monkeypatch)
    C.compare_audio(env["orig"], env["rend"], 6)
    st = os.stat(env["orig"])
    _song(Path(env["orig"]), 99, f0=220.0)                 # same path, same size, different samples
    os.utime(env["orig"], ns=(st.st_atime_ns, st.st_mtime_ns))   # ... and the same mtime
    cnt.reset()
    C.compare_audio(env["orig"], env["rend"], 6)
    assert cnt.n["compute_spectral_features"] == 2
