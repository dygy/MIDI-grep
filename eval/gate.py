"""Similarity eval gate for MIDI-grep.

Reads `eval/thresholds.yaml` (per-genre regression floors) and evaluates a
`comparison.json` produced by `scripts/python/compare_audio.py` against the floor
for its genre. Code-authoritative: callers (pytest, CI, the iteration loop) decide
pass/fail from `GateResult.passed`, independent of any dashboard.

Usage (library):
    from eval.gate import load_thresholds, evaluate_comparison
    th = load_thresholds()
    res = evaluate_comparison("path/to/comparison.json", genre="electro_swing", thresholds=th)
    assert res.passed, res.message

Usage (CLI):
    python eval/gate.py path/to/comparison.json --genre electro_swing
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a standard dep via pyyaml
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = REPO_ROOT / "eval" / "thresholds.yaml"


@dataclass
class GateResult:
    passed: bool
    genre: str
    similarity: float
    floor: float
    worst_band_diff: float | None
    message: str
    # Section-aware fields (optional — populated when the comparison dict carries
    # section_aware_similarity; absent for legacy comparison dicts).
    section_aware_similarity: float | None = None
    section_aware_floor: float | None = None
    section_aware_passed: bool | None = None


def load_thresholds(path: str | Path = THRESHOLDS_PATH) -> dict[str, Any]:
    """Load the thresholds YAML. Raises if pyyaml is missing or the file is absent."""
    if yaml is None:
        raise RuntimeError("pyyaml is required to load thresholds (pip install pyyaml)")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"thresholds file not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def floor_for_genre(genre: str | None, thresholds: dict[str, Any]) -> float:
    """Return the similarity floor for a genre, falling back to the default."""
    default = float(thresholds.get("default", 0.55))
    if not genre:
        return default
    genres = thresholds.get("genres", {}) or {}
    # Normalise: lower-case, spaces/dashes -> underscores.
    key = genre.strip().lower().replace("-", "_").replace(" ", "_")
    return float(genres.get(key, default))


def _read_similarity(
    comparison: dict[str, Any],
) -> tuple[float, float | None, float | None]:
    """Extract (overall_similarity, worst_band_diff_fraction, section_aware_similarity).

    Supports both the per-stem aggregate (`aggregate.weighted_overall`) and the single
    comparison block (`comparison.overall_similarity`). worst_band_diff is stored as a
    percentage in compare_audio.py, so it's converted back to a fraction.

    section_aware_similarity is read from:
      - comparison.section_aware_similarity  (single-file mode, compare_audio())
      - aggregate.section_aware_similarity   (per-stem mode, compare_stems())
    Returns None when the field is absent (old comparison dicts).
    """
    comp = comparison.get("comparison", {})
    agg = comparison.get("aggregate", {})
    if "overall_similarity" in comp:
        similarity = float(comp["overall_similarity"])
    elif "weighted_overall" in agg:
        similarity = float(agg["weighted_overall"])
    else:
        raise KeyError(
            "comparison.json has neither comparison.overall_similarity "
            "nor aggregate.weighted_overall"
        )
    worst = comp.get("worst_band_diff")
    worst_frac = float(worst) / 100.0 if worst is not None else None

    # Section-aware: prefer single-file field, then aggregate field, then absent.
    _sas = comp.get("section_aware_similarity") or agg.get("section_aware_similarity")
    section_aware = float(_sas) if _sas is not None else None

    return similarity, worst_frac, section_aware


def section_aware_floor_for_genre(genre: str | None, thresholds: dict[str, Any]) -> float | None:
    """Return the optional section_aware_floor for a genre (or the global one).

    Returns None when no section_aware floor is configured at all, so callers can
    skip the check for backward compatibility.
    """
    sa = thresholds.get("section_aware", {}) or {}
    # Check per-genre first
    if genre:
        key = genre.strip().lower().replace("-", "_").replace(" ", "_")
        genres_sa = sa.get("genres", {}) or {}
        if key in genres_sa:
            return float(genres_sa[key])
    # Fall back to global section_aware default
    default_sa = sa.get("default")
    return float(default_sa) if default_sa is not None else None


def evaluate_comparison(
    comparison_path: str | Path,
    genre: str | None,
    thresholds: dict[str, Any] | None = None,
) -> GateResult:
    """Evaluate one comparison.json against the genre floor and worst-band guardrail.

    Also performs an OPTIONAL section-aware check when:
      - the comparison dict contains section_aware_similarity (produced by compare_audio.py
        >= the Task C version), AND
      - thresholds.yaml contains a section_aware.default or per-genre floor.

    The section-aware check is informational in the message but can fail the gate
    independently (GateResult.section_aware_passed is False → overall passed=False).
    Backward compat: if either field is absent the section-aware check is silently skipped.
    """
    thresholds = thresholds or load_thresholds()
    data = json.loads(Path(comparison_path).read_text())
    similarity, worst_frac, section_aware = _read_similarity(data)
    floor = floor_for_genre(genre, thresholds)
    max_worst = thresholds.get("max_worst_band_diff")
    sa_floor = section_aware_floor_for_genre(genre, thresholds)

    reasons: list[str] = []
    passed = True
    if similarity < floor:
        passed = False
        reasons.append(f"similarity {similarity:.3f} < floor {floor:.3f}")
    if max_worst is not None and worst_frac is not None and worst_frac > float(max_worst):
        passed = False
        reasons.append(
            f"worst band diff {worst_frac:.2%} > guardrail {float(max_worst):.2%}"
        )

    # Section-aware check (optional — only when both data and floor are present)
    sa_passed: bool | None = None
    if section_aware is not None and sa_floor is not None:
        sa_passed = section_aware >= sa_floor
        if not sa_passed:
            passed = False
            reasons.append(
                f"section_aware_similarity {section_aware:.3f} < section_aware_floor {sa_floor:.3f}"
            )

    g = genre or "default"
    if passed:
        msg = f"PASS [{g}] similarity {similarity:.3f} >= floor {floor:.3f}"
        if worst_frac is not None:
            msg += f" (worst band {worst_frac:.2%})"
        if section_aware is not None and sa_floor is not None:
            msg += f"; section_aware {section_aware:.3f} >= sa_floor {sa_floor:.3f}"
    else:
        msg = f"FAIL [{g}] " + "; ".join(reasons)
    return GateResult(
        passed=passed,
        genre=g,
        similarity=similarity,
        floor=floor,
        worst_band_diff=worst_frac,
        message=msg,
        section_aware_similarity=section_aware,
        section_aware_floor=sa_floor,
        section_aware_passed=sa_passed,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MIDI-grep similarity eval gate")
    ap.add_argument("comparison_json", help="path to a comparison.json from compare_audio.py")
    ap.add_argument("--genre", default=None, help="genre key (e.g. electro_swing)")
    ap.add_argument("--thresholds", default=str(THRESHOLDS_PATH))
    args = ap.parse_args(argv)
    res = evaluate_comparison(args.comparison_json, args.genre, load_thresholds(args.thresholds))
    print(res.message)
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
