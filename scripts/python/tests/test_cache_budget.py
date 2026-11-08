"""cache_budget.py: regenerable material is pruned to a hard budget; source material and calibration memory never are.

Regression (owner, 2026-11-08): ``.cache`` reached 42 GB - every vNNN (~300 MB) and every calibration iteration's
renders were kept forever - and the owner deleted ~70 GB by hand, the venv included.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_budget as CB  # noqa: E402

MB = 1024 ** 2


def _blob(p: Path, mb: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * int(mb * MB))


def _version(track: Path, n: int, mode: str, overall: float) -> Path:
    v = track / f"v{n:03d}"
    _blob(v / "render.wav", 2)
    for s in CB.RENDER_STEM_WAVS:
        _blob(v / s, 1)
    _blob(v / "iterations" / "cal1.mp3", 1)
    _blob(v / "embed" / "a.mp3", 1)
    _blob(v / "report.html", 1)
    (v / "comparison.json").write_text(json.dumps({"generation_mode": mode,
                                                   "comparison": {"overall_similarity": overall}}))
    return v


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / ".cache"
    stems = root / "stems"
    t = stems / "Track A"
    for s in ("bass", "drums", "melodic", "vocals", "original", "instrumental"):
        _blob(t / f"{s}.wav", 1)
    (t / "metadata.json").write_text("{}")
    _blob(t / "sample_pack" / "drums" / "bd.wav", 0.5)
    w = t / ".pipeline" / "sample-instrument"
    for f in ("cal1.wav", "cal1.bass.wav", "cal2.wav", "mix1.wav"):
        _blob(w / f, 1)
    _blob(w / "best.wav", 1)
    (w / "best.params").write_text("x")
    (w / "cal1.cmp.json").write_text("{}")
    _blob(t / ".pipeline" / "stems" / "bass.wav", 1)
    _version(t, 1, "sample-instrument", 0.90)      # best scoring
    _version(t, 2, "sample-instrument", 0.80)
    _version(t, 3, "sample-instrument", 0.85)      # newest
    _version(t, 4, "synth", 0.70)
    _version(t, 5, "synth", 0.60)                  # newest synth
    refs = tmp_path / "reference_tracks.yaml"
    refs.write_text("tracks:\n  - cache_key: \"Track A\"\n    version: v002\n")
    monkeypatch.setattr(CB, "CACHE", root)
    monkeypatch.setattr(CB, "STEMS", stems)
    monkeypatch.setattr(CB, "REFERENCES", refs)
    monkeypatch.setattr(CB, "GROWTH", root / "runlogs" / "disk_growth.json")
    return t


def test_iteration_renders_always_go_and_best_and_memory_stay(cache):
    p = CB.plan(target=10 ** 12, current=CB.size(CB.CACHE))       # far under budget: only tier 1
    CB.apply(p)
    w = cache / ".pipeline" / "sample-instrument"
    assert {a[0] for a in p.actions} == {"always"}
    assert sorted(f.name for f in w.iterdir()) == ["best.params", "best.wav", "cal1.cmp.json"]
    assert all((cache / f"v{n:03d}").is_dir() for n in range(1, 6))


def test_over_budget_keeps_newest_best_and_pinned_versions_and_every_stem(cache):
    CB.apply(CB.plan(target=1, current=CB.size(CB.CACHE)))
    kept = sorted(d.name for d in cache.iterdir() if CB.VERSION_RE.match(d.name))
    # v001 best sample-instrument, v002 pinned by the reference dataset, v003 newest, v004 best synth, v005 newest synth
    assert kept == ["v001", "v002", "v003", "v004", "v005"]
    for s in ("bass", "drums", "melodic", "vocals", "original", "instrumental"):
        assert (cache / f"{s}.wav").is_file()
    assert (cache / "sample_pack" / "drums" / "bd.wav").is_file()
    # the non-newest protected pieces are slimmed to report + render.wav; the newest stay whole
    assert not (cache / "v001" / "iterations").exists() and not (cache / "v001" / "render_bass.wav").exists()
    assert (cache / "v001" / "report.html").is_file() and (cache / "v001" / "render.wav").is_file()
    assert (cache / "v003" / "iterations").is_dir() and (cache / "v003" / "render_bass.wav").is_file()


def test_an_unprotected_version_is_pruned(cache):
    _version(cache, 6, "synth", 0.50)              # newest synth now; v005 (0.60) is neither newest nor best
    CB.apply(CB.plan(target=1, current=CB.size(CB.CACHE)))
    assert not (cache / "v005").exists() and (cache / "v004").is_dir() and (cache / "v006").is_dir()


def test_a_track_a_run_holds_is_never_touched(cache):
    (cache / ".pipeline" / CB.RUNNING).write_text(str(os.getpid()))
    p = CB.plan(target=1, current=CB.size(CB.CACHE))
    assert not any(str(path).startswith(str(cache)) for _, path, _ in p.actions)


def test_a_dead_runs_marker_does_not_protect_the_track(cache):
    (cache / ".pipeline" / CB.RUNNING).write_text("999999999")
    assert not CB.active(cache)


def test_preflight_needs_the_largest_recorded_growth_plus_ram(cache, monkeypatch, capsys):
    CB.record_growth(100, 100 + 5 * CB.GB)
    CB.record_growth(100, 100 + 2 * CB.GB)
    assert CB.run_growth() == 5 * CB.GB
    monkeypatch.setattr(CB, "ram_bytes", lambda: 24 * CB.GB)

    class U:
        free = 20 * CB.GB
    monkeypatch.setattr(CB.shutil, "disk_usage", lambda _p: U)
    assert CB.main(["preflight"]) == 4
    assert "29.0 GB needed" in capsys.readouterr().err


def test_an_invalid_budget_is_an_error_not_a_default():
    with pytest.raises(SystemExit):
        CB.budget_bytes({"MIDIGREP_CACHE_BUDGET_GB": "lots"})
    assert CB.budget_bytes({"MIDIGREP_CACHE_BUDGET_GB": "2"}) == 2 * CB.GB
