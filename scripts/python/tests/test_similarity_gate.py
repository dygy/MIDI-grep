"""Eval gate tests for MIDI-grep similarity.

Two layers:

1. Gate-logic tests — synthetic comparison dicts exercise eval/gate.py directly. These always
   run and prove the gate enforces floors + the worst-band guardrail.

2. Dataset gate — eval/datasets/reference_tracks.yaml pins real tracks to genres. Each track's
   comparison.json must clear its genre floor in eval/thresholds.yaml. Tracks whose comparison.json
   is missing are SKIPPED (so a fresh checkout stays green); present ones are FAILED on a breach.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_similarity_gate.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

yaml = pytest.importorskip("yaml")

from eval.gate import (  # noqa: E402
    evaluate_comparison,
    floor_for_genre,
    load_thresholds,
)

THRESHOLDS = load_thresholds()


# ---------------------------------------------------------------------------
# Layer 1: gate-logic tests (synthetic, always run)
# ---------------------------------------------------------------------------

def _write_comparison(tmp_path: Path, overall: float, worst_band_pct: float | None = None) -> Path:
    comp = {"overall_similarity": overall}
    if worst_band_pct is not None:
        comp["worst_band_diff"] = worst_band_pct
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"comparison": comp}))
    return p


def test_unknown_genre_uses_default_floor():
    assert floor_for_genre("totally_made_up", THRESHOLDS) == THRESHOLDS["default"]


def test_genre_normalisation():
    # "Electro Swing" / "electro-swing" both resolve to the electro_swing floor.
    expected = THRESHOLDS["genres"]["electro_swing"]
    assert floor_for_genre("Electro Swing", THRESHOLDS) == expected
    assert floor_for_genre("electro-swing", THRESHOLDS) == expected


def test_pass_when_above_floor(tmp_path):
    path = _write_comparison(tmp_path, overall=0.90, worst_band_pct=5.0)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert res.passed, res.message


def test_fail_when_below_floor(tmp_path):
    path = _write_comparison(tmp_path, overall=0.10, worst_band_pct=5.0)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert not res.passed
    assert "similarity" in res.message


def test_worst_band_guardrail_fails_even_with_high_overall(tmp_path):
    # High overall but a single band wildly off must still fail.
    bad_band_pct = (THRESHOLDS["max_worst_band_diff"] * 100) + 10
    path = _write_comparison(tmp_path, overall=0.95, worst_band_pct=bad_band_pct)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert not res.passed
    assert "worst band" in res.message


def test_aggregate_weighted_overall_supported(tmp_path):
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"aggregate": {"weighted_overall": 0.80}}))
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed, res.message


# ---------------------------------------------------------------------------
# Layer 1b: section-aware gate tests (synthetic, always run)
# ---------------------------------------------------------------------------

def _write_comparison_with_section_aware(
    tmp_path: Path,
    overall: float,
    section_aware: float | None,
    worst_band_pct: float | None = None,
) -> Path:
    """Build a comparison.json that optionally includes section_aware_similarity."""
    comp: dict = {"overall_similarity": overall}
    if worst_band_pct is not None:
        comp["worst_band_diff"] = worst_band_pct
    if section_aware is not None:
        comp["section_aware_similarity"] = section_aware
        comp["section_aware_window_count"] = 3
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"comparison": comp}))
    return p


def test_section_aware_pass_when_above_floor(tmp_path):
    """Both overall and section_aware above their floors — should pass."""
    path = _write_comparison_with_section_aware(tmp_path, overall=0.90, section_aware=0.70)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.section_aware_similarity == pytest.approx(0.70)
    assert res.section_aware_passed is True


def test_section_aware_fail_when_below_floor(tmp_path):
    """Overall passes but section_aware is below its floor — gate should fail."""
    # electro_swing section_aware_floor = 0.48 (from thresholds.yaml)
    path = _write_comparison_with_section_aware(tmp_path, overall=0.90, section_aware=0.20)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert not res.passed, "should fail because section_aware < sa_floor"
    assert "section_aware" in res.message
    assert res.section_aware_passed is False


def test_section_aware_absent_behaves_as_before(tmp_path):
    """Legacy comparison.json without section_aware_similarity — gate skips the check."""
    path = _write_comparison(tmp_path, overall=0.90, worst_band_pct=5.0)
    res = evaluate_comparison(path, genre="electro_swing", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.section_aware_similarity is None
    assert res.section_aware_passed is None


def test_section_aware_unknown_genre_uses_global_floor(tmp_path):
    """Unknown genre falls back to section_aware.default floor (0.45)."""
    sub_a = tmp_path / "a"
    sub_a.mkdir()
    sub_b = tmp_path / "b"
    sub_b.mkdir()

    # section_aware=0.50 > default 0.45 — should pass
    path = _write_comparison_with_section_aware(sub_a, overall=0.90, section_aware=0.50)
    res = evaluate_comparison(path, genre="completely_unknown_genre", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.section_aware_similarity == pytest.approx(0.50)

    # section_aware=0.30 < default 0.45 — should fail
    path2 = _write_comparison_with_section_aware(sub_b, overall=0.90, section_aware=0.30)
    res2 = evaluate_comparison(path2, genre="completely_unknown_genre", thresholds=THRESHOLDS)
    assert not res2.passed, "0.30 section_aware should fail against 0.45 default floor"
    assert res2.section_aware_passed is False


def test_section_aware_from_aggregate_dict(tmp_path):
    """section_aware_similarity in aggregate (per-stem mode) is picked up correctly."""
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({
        "aggregate": {
            "weighted_overall": 0.80,
            "section_aware_similarity": 0.60,
        }
    }))
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.section_aware_similarity == pytest.approx(0.60)
    assert res.section_aware_passed is True


# ---------------------------------------------------------------------------
# Layer 2: dataset gate (real tracks, skip when renders absent)
# ---------------------------------------------------------------------------

def _load_dataset() -> list[dict]:
    ds = REPO_ROOT / "eval" / "datasets" / "reference_tracks.yaml"
    if not ds.exists():
        return []
    data = yaml.safe_load(ds.read_text()) or {}
    return data.get("tracks") or []


def _resolve_comparison_path(track: dict) -> Path | None:
    if track.get("comparison"):
        return REPO_ROOT / track["comparison"]
    if track.get("cache_key") and track.get("version"):
        return (
            REPO_ROOT
            / ".cache"
            / "stems"
            / track["cache_key"]
            / track["version"]
            / "comparison.json"
        )
    return None


_DATASET = _load_dataset()


@pytest.mark.skipif(not _DATASET, reason="no reference tracks configured in reference_tracks.yaml")
@pytest.mark.parametrize("track", _DATASET, ids=[t.get("name", "?") for t in _DATASET])
def test_reference_track_meets_gate(track):
    path = _resolve_comparison_path(track)
    if path is None:
        pytest.fail(f"track {track.get('name')!r} has neither 'comparison' nor cache_key+version")
    if not path.exists():
        pytest.skip(f"comparison.json not found for {track.get('name')!r}: {path} (run an extraction first)")
    res = evaluate_comparison(path, genre=track.get("genre"), thresholds=THRESHOLDS)
    assert res.passed, res.message
