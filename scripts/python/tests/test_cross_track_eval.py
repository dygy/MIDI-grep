"""T4 cross-track harness: tests for the calibration math and regression check (no audio needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cross_track_eval as cte  # noqa: E402


def test_recommend_bars_flags_discriminating_dim():
    # pitch clearly above chance, rhythm at chance
    matched = {"corr": [0.6, 0.7], "pitch": [0.85, 0.80], "rhythm": [0.12, 0.13], "timbre": [0.06, 0.06]}
    mismatched = {"corr": [0.15, 0.18], "pitch": [0.57, 0.58], "rhythm": [0.12, 0.14], "timbre": [0.06, 0.07]}
    rec = cte.recommend_bars(matched, mismatched)
    assert rec["pitch"]["discriminating"] is True
    assert rec["rhythm"]["discriminating"] is False    # matched ≈ chance
    # the 2σ bar sits above the chance mean
    assert rec["pitch"]["recommended_bar"] > rec["pitch"]["mismatched_mean"]


def test_recommend_bars_handles_missing_data():
    rec = cte.recommend_bars({"corr": []}, {"corr": []})
    assert "note" in rec["corr"]


def test_regression_check_detects_drop(tmp_path, capsys, monkeypatch):
    base = {"per_track": {"trk": {"bass": {"corr": 0.60, "pitch": 0.80}}}}
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(base))
    monkeypatch.setattr(cte, "BASELINE", bpath)
    # current bass corr dropped 0.20 (beyond tol 0.08) → regression
    cal = {"per_track": {"trk": {"bass": {"corr": 0.40, "pitch": 0.81}}}}
    rc = cte.regression_check(cal, tol=0.08)
    assert rc == 1
    assert "bass" in capsys.readouterr().out


def test_regression_check_passes_within_tol(tmp_path, monkeypatch):
    base = {"per_track": {"trk": {"bass": {"corr": 0.60}}}}
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(base))
    monkeypatch.setattr(cte, "BASELINE", bpath)
    cal = {"per_track": {"trk": {"bass": {"corr": 0.55}}}}  # within 0.08
    assert cte.regression_check(cal, tol=0.08) == 0


def test_discover_returns_empty_for_missing_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cte, "CACHE", tmp_path / "nope")
    assert cte.discover(5) == []
