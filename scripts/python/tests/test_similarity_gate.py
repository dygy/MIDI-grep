# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Eval gate tests for MIDI-grep similarity.

Two layers (plus the spec-003 per-mode / editability layer, 1c):

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
# Layer 1c: spec 003 — per-mode floors + editability short-circuit (synthetic, always run)
# ---------------------------------------------------------------------------

from eval.gate import (  # noqa: E402
    GateResult,
    resolve_floor,
    section_aware_floor_for_genre,
)
from eval.gate import main as gate_main  # noqa: E402


def _mode_thresholds() -> dict:
    """A thresholds dict with a MEASURED per-mode block for one genre only. Values are
    synthetic test data — never copied into eval/thresholds.yaml (floors there are measured)."""
    return {
        "default": 0.55,
        "genres": {"brazilian_funk": 0.62, "electro_swing": 0.60},
        "max_worst_band_diff": 0.30,
        "section_aware": {"default": 0.45, "genres": {"brazilian_funk": 0.50}},
        "modes": {
            "sample_instrument": {
                "genres": {"brazilian_funk": 0.80},
                "section_aware": {"brazilian_funk": 0.70},
                "measured": {"brazilian_funk": {"overall": 0.83, "section_aware": 0.73, "run": "x/v001"}},
            },
            # synth: block present but EMPTY (the shape thresholds.yaml ships before measurement)
            "synth": {"genres": None, "section_aware": None, "measured": None},
        },
    }


def test_floor_prefers_mode_block_when_present():
    th = _mode_thresholds()
    assert floor_for_genre("brazilian_funk", th, mode="sample-instrument") == pytest.approx(0.80)
    # mode key normalisation: dash/underscore/case are equivalent
    assert floor_for_genre("brazilian_funk", th, mode="Sample_Instrument") == pytest.approx(0.80)
    assert resolve_floor("brazilian_funk", th, mode="sample-instrument") == (0.80, "modes.sample_instrument")


def test_floor_falls_back_to_genre_when_mode_has_no_entry():
    th = _mode_thresholds()
    # genre not measured for this mode → genre-wide floor
    assert floor_for_genre("electro_swing", th, mode="sample-instrument") == pytest.approx(0.60)
    assert resolve_floor("electro_swing", th, mode="sample-instrument")[1] == "genres"
    # mode block present but empty (synth before measurement) → genre-wide floor
    assert floor_for_genre("brazilian_funk", th, mode="synth") == pytest.approx(0.62)
    # unknown mode / no mode → unchanged legacy behaviour
    assert floor_for_genre("brazilian_funk", th, mode="made_up") == pytest.approx(0.62)
    assert floor_for_genre("brazilian_funk", th) == pytest.approx(0.62)
    assert floor_for_genre("unlisted", th, mode="sample-instrument") == pytest.approx(0.55)


def test_section_aware_floor_prefers_mode_block_then_falls_back():
    th = _mode_thresholds()
    assert section_aware_floor_for_genre("brazilian_funk", th, mode="sample-instrument") == pytest.approx(0.70)
    assert section_aware_floor_for_genre("brazilian_funk", th, mode="synth") == pytest.approx(0.50)
    assert section_aware_floor_for_genre("brazilian_funk", th) == pytest.approx(0.50)
    assert section_aware_floor_for_genre("electro_swing", th, mode="sample-instrument") == pytest.approx(0.45)


def test_shipped_thresholds_have_modes_shape_with_no_guessed_values():
    """eval/thresholds.yaml ships the modes: block STRUCTURE only — a floor appears there only
    after a detector-passing render is measured (Slice 4). Guard against typed guesses."""
    modes = THRESHOLDS.get("modes")
    assert isinstance(modes, dict) and set(modes) >= {"sample_instrument", "synth"}
    for mode_name, block in modes.items():
        block = block or {}
        assert set(block) <= {"genres", "section_aware", "measured"}, mode_name
        floors = block.get("genres") or {}
        measured = block.get("measured") or {}
        for genre_key in floors:
            assert genre_key in measured, (
                f"modes.{mode_name}.genres.{genre_key} has a floor with no measured: entry — "
                "floors must come from a real run, never a guess"
            )
        # every measured floor must be reproducible: floor == measured.overall − margin, with the run named
        for genre_key, floor in floors.items():
            m = measured[genre_key]
            assert {"overall", "run", "margin"} <= set(m), f"modes.{mode_name}.measured.{genre_key} incomplete"
            assert floor == pytest.approx(round(m["overall"] - m["margin"], 2), abs=1e-9), (
                f"modes.{mode_name}.genres.{genre_key}={floor} is not measured.overall − margin"
            )
            if "section_aware" in (block.get("section_aware") or {}):
                assert block["section_aware"][genre_key] == pytest.approx(
                    round(m["section_aware"] - m["margin"], 2), abs=1e-9)
    # a measured mode floor is what the mode lookup returns; an unmeasured mode falls back to the genre floor
    si = (modes.get("sample_instrument") or {}).get("genres") or {}
    expected_si = si.get("brazilian_funk", THRESHOLDS["genres"]["brazilian_funk"])
    assert floor_for_genre("brazilian_funk", THRESHOLDS, mode="sample-instrument") == expected_si
    sy = (modes.get("synth") or {}).get("genres") or {}
    expected_sy = sy.get("brazilian_funk", THRESHOLDS["genres"]["brazilian_funk"])
    assert floor_for_genre("brazilian_funk", THRESHOLDS, mode="synth") == expected_sy


def test_evaluate_reads_generation_mode_from_json_and_uses_mode_floor(tmp_path):
    th = _mode_thresholds()
    p = tmp_path / "comparison.json"
    # 0.70 clears the genre floor (0.62) but NOT the measured sample-instrument floor (0.80)
    p.write_text(json.dumps({
        "generation_mode": "sample-instrument",
        "editability": "pass",
        "editability_violations": [],
        "comparison": {"overall_similarity": 0.70, "worst_band_diff": 5.0},
    }))
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=th)
    assert res.mode == "sample-instrument"
    assert res.editability == "pass"
    assert res.floor == pytest.approx(0.80)
    assert res.floor_source == "modes.sample_instrument"
    assert not res.passed and "floor 0.800" in res.message
    # explicit mode= overrides the JSON key: synth has no measured floor → genre floor → pass
    res2 = evaluate_comparison(p, genre="brazilian_funk", thresholds=th, mode="synth")
    assert res2.mode == "synth" and res2.floor == pytest.approx(0.62) and res2.passed, res2.message


def test_evaluate_fails_on_editability_fail_even_with_high_score(tmp_path):
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({
        "generation_mode": "sample-instrument",
        "editability": "fail",
        "editability_violations": ["R1 line 196 [vocalsfull]: reconstruction-by-playback"],
        "comparison": {"overall_similarity": 0.95, "worst_band_diff": 2.0},
    }))
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert isinstance(res, GateResult)
    assert not res.passed
    assert "editability: fail" in res.message
    assert "vocalsfull" in res.message
    assert res.editability == "fail" and res.mode == "sample-instrument"


def test_evaluate_fails_on_short_circuit_payload_with_null_comparison(tmp_path):
    """The exact shape compare_audio.py --strudel writes on a detector fail (exit 3)."""
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({
        "editability": "fail",
        "generation_mode": "loops",
        "editability_violations": ["R5: loop-only output"],
        "comparison": None,
    }))
    res = evaluate_comparison(p, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res.passed
    assert res.message.startswith("FAIL [brazilian_funk] editability: fail")
    # a bare comparison: null (no editability key) is equally unscoreable
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps({"comparison": None}))
    res2 = evaluate_comparison(p2, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert not res2.passed and "editability: fail" in res2.message


def test_legacy_comparison_without_new_keys_is_unchanged(tmp_path):
    path = _write_comparison(tmp_path, overall=0.90, worst_band_pct=5.0)
    res = evaluate_comparison(path, genre="brazilian_funk", thresholds=THRESHOLDS)
    assert res.passed, res.message
    assert res.mode is None and res.editability is None and res.floor_source == "genres"


def test_cli_accepts_mode_and_reports_editability_fail(tmp_path, capsys):
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"editability": "fail", "editability_violations": ["R1"], "comparison": None}))
    rc = gate_main([str(p), "--genre", "brazilian_funk", "--mode", "sample-instrument"])
    out = capsys.readouterr().out
    assert rc == 1 and "editability: fail" in out
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"comparison": {"overall_similarity": 0.9}}))
    assert gate_main([str(ok), "--genre", "brazilian_funk", "--mode", "synth"]) == 0


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
    expected = track.get("expected", "pass")
    if expected == "shortfall":
        # An honestly recorded shortfall (spec 004 §2.3): the track is kept as a reference, the floor
        # is NOT lowered, and the record must agree with the run — it fails the gate, the output is
        # still a detector-passing deliverable, and the recorded numbers are the run's numbers.
        assert not res.passed, f"{track.get('name')!r} is recorded as a shortfall but now clears the gate — update the record"
        assert res.editability == "pass", f"a shortfall entry must still be an editable deliverable: {res.message}"
        comp = json.loads(path.read_text())["comparison"]
        for key in ("overall", "section_aware"):
            if key in (track.get("measured") or {}):
                field = "overall_similarity" if key == "overall" else "section_aware_similarity"
                assert comp[field] == pytest.approx(track["measured"][key], abs=5e-4), f"{key} recorded != run"
    else:
        assert res.passed, res.message
