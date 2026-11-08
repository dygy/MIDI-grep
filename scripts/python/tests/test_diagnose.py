"""diagnose.py: a broken run is named by what is broken, not hidden in its overall score.

Regression (owner, 2026-11-08): BibiBabydoll v006/v007 were reported at 0.744 / 0.722 overall while
``tempo_similarity`` was 0.27 / 0.17 - the stamp said 112.52 BPM (phase-coherence confidence 0.045, every reference
>= 0.10) and the original measured 101.33. Nothing in the run summary said so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("librosa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose as D  # noqa: E402


def _comparison(tempo_orig: float, tempo_rend: float, *, chroma: float = 0.80, tempo_sim: float = 1.0) -> dict:
    temporal = {"lag_seconds": 0.0, "chroma_corr": chroma, "mel_corr": 0.80, "onset_f1": 0.75, "envelope_corr": 0.90,
                "temporal_similarity": (chroma + 0.80 + 0.75 + 0.90) / 4}
    win = {"t0": 0.0, "t1": 10.0, "win": 0.70, "mfcc": 0.97, "band": 0.90, "energy": 0.88, "chroma_corr": 0.20,
           "mel_corr": 0.78, "onset_f1": 0.74, "envelope_corr": 0.88, "band_diff": {"sub_bass": 0.30, "mid": -0.05}}
    return {"generation_mode": "sample-instrument",
            "original": {"rhythm": {"tempo": tempo_orig, "tempo_candidates": []}},
            "rendered": {"rhythm": {"tempo": tempo_rend}},
            "comparison": {"frequency_balance_similarity": 0.97, "mfcc_similarity": 0.99, "energy_similarity": 0.90,
                           "brightness_similarity": 0.93, "tempo_similarity": tempo_sim, "chroma_similarity": 0.99,
                           "temporal": temporal, "overall_similarity": 0.88, "section_aware_similarity": 0.86,
                           "margin_over_chance": 0.30, "section_windows": [win]}}


def _piece(root: Path, name: str, comp: dict, bpm: float, conf: float) -> Path:
    track = root / name
    v = track / "v001"
    v.mkdir(parents=True)
    (v / "comparison.json").write_text(json.dumps(comp))
    (track / "metadata.json").write_text(json.dumps({"bpm": bpm, "bpm_confidence": conf}))
    return v


@pytest.fixture
def refs(tmp_path: Path) -> Path:
    vs = [_piece(tmp_path, f"ref{i}", _comparison(135.0, 135.0, chroma=0.70 + 0.05 * i), 135.0, 0.10 + 0.1 * i)
          for i in range(4)]
    out = tmp_path / "reference_metrics.json"
    D.record_refs(vs, out)
    return out


def test_a_tempo_stamp_the_original_contradicts_is_red_and_named(tmp_path, refs):
    v = _piece(tmp_path, "bibi", _comparison(101.33, 113.99, chroma=0.385, tempo_sim=0.167), 112.516, 0.0454)
    d = D.diagnose(v, refs_path=refs)
    tempo_flags = [f for f in d["flags"] if f.startswith("tempo:")]
    assert any("stamped 112.52 BPM but the original measures 101.33" in f and "confidence 0.045" in f
               for f in tempo_flags)
    assert not d["tempo"]["stamp_matches_original"] and d["tempo"]["confidence_below_references"]
    # the components below every reference are RED, the costliest first
    names = [r["metric"] for r in d["red_metrics"]]
    assert "mix.tempo_similarity" in names and "mix.temporal.chroma_corr" in names
    assert d["red_metrics"][0]["cost"] >= d["red_metrics"][-1]["cost"]
    assert d["red_count"] == len(d["flags"]) > 0


def test_a_run_inside_the_reference_range_has_no_red_flag(tmp_path, refs):
    v = _piece(tmp_path, "fine", _comparison(135.0, 135.0, chroma=0.78), 135.0, 0.30)
    d = D.diagnose(v, refs_path=refs)
    assert d["red_count"] == 0 and d["flags"] == []


def test_half_time_is_the_same_tempo(tmp_path, refs):
    v = _piece(tmp_path, "half", _comparison(67.5, 135.0), 135.0, 0.30)
    t = D.diagnose(v, refs_path=refs)["tempo"]
    assert t["stamp_matches_original"] and t["render_matches_stamp"]


def test_the_worst_section_names_the_component_that_drags_it(tmp_path, refs):
    v = _piece(tmp_path, "sec", _comparison(135.0, 135.0), 135.0, 0.30)
    s = D.diagnose(v, refs_path=refs)["worst_sections"][0]
    assert s["drag"] == "chroma_corr" and s["worst_band"] == "sub_bass"


def test_cost_uses_the_scores_own_weights():
    import compare_audio as CA
    assert D.weight("mix.frequency_balance_similarity") == pytest.approx(
        CA.STATIC_SHARE * CA.STATIC_WEIGHTS["frequency_balance_similarity"])
    assert D.weight("mix.temporal.onset_f1") == pytest.approx((1 - CA.STATIC_SHARE) / len(CA.TEMPORAL_COMPONENTS))
    assert D.weight("bass.overall_similarity") == 0.0


def test_a_stem_the_original_does_not_have_is_skipped_not_flagged(tmp_path, refs):
    # Tatuagem No Pescoço (2026-11-08): bass.wav is bleed at -63 dB, the low end is a tuned 808 kick in the drums;
    # its bass metrics compared a render with bleed and showed 20 RED instead of 12.
    import numpy as np
    sf = pytest.importorskip("soundfile")
    comp = _comparison(135.0, 135.0)
    v = _piece(tmp_path, "tatuagem", comp, 135.0, 0.6)
    (v / "stem_comparison.json").write_text(json.dumps({"stems": {
        "bass": {"comparison": {"overall_similarity": 0.1, "frequency_balance_similarity": 0.0}},
        "drums": {"comparison": {"overall_similarity": 0.75}}}}))
    t = np.arange(22050) / 22050
    for name, amp in (("bass", 0.0007), ("drums", 0.5), ("melodic", 0.4)):
        sf.write(str(v.parent / f"{name}.wav"), (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 22050)
    d = D.diagnose(v, refs_path=refs)
    assert d["absent_stems"] == ["bass"]
    assert not any(r["metric"].startswith("bass.") for r in d["red_metrics"] + d["within_reference_range"])
