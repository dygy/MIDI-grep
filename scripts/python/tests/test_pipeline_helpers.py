# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""pipeline_helpers.py: slug / num_bars / next_version_dir / promote_run (spec 004 §2.1)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from pipeline_helpers import RunMeta, next_version_dir, num_bars, promote_run, slug_from_title  # noqa: E402

META = RunMeta(generator="gen knobs", render="raw capture, device ratio 0.8260", compare="compare_audio.py -d 135 --strudel")


@pytest.mark.parametrize("title,slug", [
    ("VAGABUNDO NÃO NAMORA", "vagabundo-nao-namora"),
    ("Regime CLT (Dj Brunin XM, Aurora Shukita)", "regime-clt-dj-brunin-xm-aurora-shukita"),
    ("  Ça va -- très bien!  ", "ca-va-tres-bien"),
])
def test_slug_folds_accents_and_kebabs(title: str, slug: str) -> None:
    assert slug_from_title(title) == slug


def test_slug_with_no_ascii_is_an_error() -> None:
    with pytest.raises(ValueError):
        slug_from_title("!!!")


@pytest.mark.parametrize("dur,bpm,bars", [(137.76, 136, 78), (151.0, 130, 81), (1.0, 120, 0), (240.0, 120, 120)])
def test_num_bars_floors(dur: float, bpm: float, bars: int) -> None:
    assert num_bars(dur, bpm) == bars


def test_num_bars_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        num_bars(0, 136)


def test_next_version_dir_fresh_and_continuing(tmp_path: Path) -> None:
    assert next_version_dir(tmp_path / "nope").name == "v001"
    assert next_version_dir(tmp_path).name == "v001"
    for n in ("v001", "v009", "v027", "sample_pack", "v1x"):
        (tmp_path / n).mkdir()
    (tmp_path / "v100").write_text("a file, not a version dir")
    assert next_version_dir(tmp_path).name == "v028"


def _inputs(tmp_path: Path, doc: dict[str, object]) -> tuple[Path, Path, Path]:
    s, w, c = tmp_path / "b.strudel", tmp_path / "b.wav", tmp_path / "b.cmp.json"
    s.write_text("$: note('c1')")
    w.write_bytes(b"RIFF")
    c.write_text(json.dumps(doc))
    return s, w, c


def test_promote_run_writes_four_files_and_metadata(tmp_path: Path) -> None:
    doc = {"editability": "pass", "generation_mode": "synth", "comparison": {
        "overall_similarity": 0.78791, "section_aware_similarity": 0.8319,
        "frequency_balance_similarity": 0.6424, "tempo_similarity": 0.4624}}
    s, w, c = _inputs(tmp_path, doc)
    track = tmp_path / "track"
    track.mkdir()
    (track / "v004").mkdir()
    vdir = promote_run(track, "synth", s, w, c, META)
    assert vdir.name == "v005"
    assert sorted(p.name for p in vdir.iterdir()) == ["comparison.json", "metadata.json", "output.strudel", "render.wav"]
    meta = json.loads((vdir / "metadata.json").read_text())
    assert set(meta) == {"version", "created_at", "generation_mode", "editability", "vocal_mode", "generator", "render",
                         "compare", "similarity_overall", "similarity_section_aware", "frequency_balance",
                         "tempo_similarity", "env_correction"}
    assert meta["env_correction"] is None and not (vdir / "env_correction.json").exists()
    assert (meta["version"], meta["generation_mode"], meta["editability"], meta["vocal_mode"]) == (5, "synth", "pass", "instrument")
    assert (meta["similarity_overall"], meta["tempo_similarity"]) == (0.7879, 0.4624)
    assert json.loads((vdir / "comparison.json").read_text()) == doc


def test_promote_run_persists_env_correction(tmp_path: Path) -> None:
    s, w, c = _inputs(tmp_path, {"editability": "pass", "comparison": {}})
    env = tmp_path / "best.env.json"
    env.write_text(json.dumps({"bars": 2, "voices": {"bass": [1.0, 1.2]}, "clamp": [0.8, 1.5]}))
    vdir = promote_run(tmp_path / "t", "sample-instrument", s, w, c, META, env_correction=env)
    assert json.loads((vdir / "env_correction.json").read_text()) == json.loads(env.read_text())
    assert json.loads((vdir / "metadata.json").read_text())["env_correction"] == "env_correction.json"


def test_promote_run_missing_env_correction_errors_and_creates_nothing(tmp_path: Path) -> None:
    s, w, c = _inputs(tmp_path, {"editability": "pass", "comparison": {}})
    track = tmp_path / "t"
    track.mkdir()
    with pytest.raises(FileNotFoundError):
        promote_run(track, "synth", s, w, c, META, env_correction=tmp_path / "nope.json")
    assert list(track.iterdir()) == []


def test_promote_run_editability_fail_reports_no_similarity(tmp_path: Path) -> None:
    s, w, c = _inputs(tmp_path, {"editability": "fail", "comparison": None})
    meta = json.loads((promote_run(tmp_path / "t", "sample-instrument", s, w, c, META) / "metadata.json").read_text())
    assert meta["editability"] == "fail" and meta["similarity_overall"] is None


def test_promote_run_missing_wav_errors_and_creates_nothing(tmp_path: Path) -> None:
    s, w, c = _inputs(tmp_path, {"editability": "pass", "comparison": {}})
    w.unlink()
    track = tmp_path / "t"
    with pytest.raises(FileNotFoundError, match="wav"):
        promote_run(track, "synth", s, w, c, META)
    assert not track.exists()


def test_promote_run_unknown_mode(tmp_path: Path) -> None:
    s, w, c = _inputs(tmp_path, {"comparison": None})
    with pytest.raises(ValueError, match="mode"):
        promote_run(tmp_path / "t", "loops", s, w, c, META)


def test_cli_slug_and_num_bars() -> None:
    cli = SCRIPTS / "pipeline_helpers.py"
    out = subprocess.run([sys.executable, str(cli), "slug", "VAGABUNDO NÃO NAMORA"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "vagabundo-nao-namora"
    out = subprocess.run([sys.executable, str(cli), "num-bars", "151", "130"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "81"


# --- stamp_track_metadata (spec 004 defect #6: track metadata lacks bpm/key/genre/duration) -----
def _track(tmp_path, bpm=129.19921875, key="C# minor", style="trance"):
    track = tmp_path / "VAGABUNDO"
    (track / "v001").mkdir(parents=True)
    (track / "metadata.json").write_text(json.dumps({"title": "VAGABUNDO", "url": "u", "video_id": "v"}))
    (track / "v001" / "metadata.json").write_text(json.dumps({"bpm": bpm, "key": key, "style": style, "version": 1}))
    return track


def test_stamp_copies_bpm_key_and_heuristic_style_up(tmp_path):
    from pipeline_helpers import stamp_track_metadata
    m = stamp_track_metadata(_track(tmp_path))
    assert m["bpm"] == 129.19921875 and m["key"] == "C# minor"
    assert m["style_heuristic"] == "trance" and m["genre"] == "trance" and m["genre_override"] is False
    assert json.loads((tmp_path / "VAGABUNDO" / "metadata.json").read_text())["title"] == "VAGABUNDO"  # preserved


def test_stamp_prefers_detector_then_override_and_logs_it(tmp_path):
    from pipeline_helpers import stamp_track_metadata
    track = _track(tmp_path)
    det = tmp_path / "genre.json"
    det.write_text(json.dumps({"detected_genre": "house", "confidence": 0.31, "rankings": []}))
    m = stamp_track_metadata(track, detector_json=det)
    assert m["genre"] == "house" and m["genre_detected"] == "house" and m["genre_override"] is False
    m2 = stamp_track_metadata(track, genre_override="brazilian_funk", detector_json=det)
    assert m2["genre"] == "brazilian_funk" and m2["genre_override"] is True
    assert "house" in m2["genre_override_note"] and "trance" in m2["genre_override_note"]


def test_stamp_fails_clearly_without_a_version_dir(tmp_path):
    from pipeline_helpers import stamp_track_metadata
    bare = tmp_path / "bare"; bare.mkdir()
    (bare / "metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        stamp_track_metadata(bare)


# review #8 — a promoted vNNN (no bpm) must not hide the analysis version
def test_stamp_uses_latest_version_that_has_a_bpm(tmp_path):
    from pipeline_helpers import stamp_track_metadata
    track = _track(tmp_path)
    (track / "v002").mkdir(); (track / "v002" / "metadata.json").write_text(json.dumps({"version": 2, "generation_mode": "synth"}))
    assert stamp_track_metadata(track)["bpm"] == 129.19921875


# review #7 — detector provenance comes from the detector JSON, not a hardcoded label
def test_stamp_records_the_detectors_model_type(tmp_path):
    from pipeline_helpers import stamp_track_metadata
    track = _track(tmp_path); det = tmp_path / "g.json"
    det.write_text(json.dumps({"detected_genre": "house", "confidence": 0.3, "model_type": "fallback"}))
    assert stamp_track_metadata(track, detector_json=det)["genre_detector"] == "detect_genre_dl.py (fallback)"


# spec 004 §2.3 / review #4 — floors for a NEW genre come from the run; existing floors are untouched
def test_record_mode_floor_derives_from_measurement_and_never_touches_existing(tmp_path):
    import yaml
    from pipeline_helpers import record_mode_floor
    th = tmp_path / "thresholds.yaml"
    th.write_text(yaml.safe_dump({"genres": {"brazilian_funk": 0.62}, "modes": {"sample_instrument": {"genres": {"brazilian_funk": 0.88}, "section_aware": {"brazilian_funk": 0.9}, "measured": {}}}}))
    cmp_ = tmp_path / "comparison.json"
    cmp_.write_text(json.dumps({"editability": "pass", "generation_mode": "sample-instrument",
                                "comparison": {"overall_similarity": 0.8123, "section_aware_similarity": 0.7777}}))
    rec = record_mode_floor(th, mode="sample-instrument", genre="house", comparison_path=cmp_, run="Some Track/v004")
    d = yaml.safe_load(th.read_text())["modes"]["sample_instrument"]
    assert rec["overall"] == 0.8123 and d["genres"]["house"] == 0.79 and d["section_aware"]["house"] == 0.76
    assert d["measured"]["house"]["run"] == "Some Track/v004" and d["genres"]["brazilian_funk"] == 0.88  # untouched
    # existing floor → no-op; replay run → no-op
    assert record_mode_floor(th, mode="sample-instrument", genre="brazilian_funk", comparison_path=cmp_, run="x") is None
    cmp_.write_text(json.dumps({"editability": "fail", "comparison": None}))
    assert record_mode_floor(th, mode="sample-instrument", genre="techno", comparison_path=cmp_, run="x") is None
