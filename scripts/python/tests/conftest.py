"""Every test gets its own compare-feature cache directory.

compare_audio.py caches the original's features on disk (compare_cache.py). Without isolation a test that
counts feature calls (test_temporal_similarity.py::test_window_halves_cover_the_same_aligned_audio assumes the
first two MFCC calls are the whole-file ones) passes on a cold cache and fails on a warm one - and every test run
would read and write the repo's real .cache/compare-features/. Subprocess CLI tests inherit the variable.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_compare_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MIDIGREP_COMPARE_CACHE_DIR", str(tmp_path_factory.mktemp("compare-cache")))
    monkeypatch.delenv("MIDIGREP_COMPARE_CACHE", raising=False)
