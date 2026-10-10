# @layer: integration
# @spec: 004-second-reference-track
# @regression
"""Spec 004 acceptance tests: the WHOLE feature against functional-spec.md section 2.1-2.5.

The per-slice suites (test_build_instruments, test_pipeline_helpers, test_similarity_gate,
internal/audio/youtube_test.go, cmd/midi-grep/main_test.go) prove each component. This file proves the
feature's contract on whole deliverables: the cached v002 / v003 runs of "VAGABUNDO NAO NAMORA", the
reference dataset, the floors, the docs and the general (not per-track) shape of the defect fixes.

E2E renders (BlackHole) are out of CI scope: cases that need the cached track SKIP when
`.cache/stems/VAGABUNDO NÃO NAMORA/` is absent. Static/negative cases run everywhere. Nothing here
touches the network, Demucs, Ollama, yt-dlp or the recorder.

Criterion -> test map
  2.1 two editable pieces       test_21_sample_instrument_piece_has_separate_editable_voice_blocks
                                test_21_vocal_is_notes_on_the_slug_vocal_instrument_not_a_stem_replay
                                test_21_vocal_replay_and_loop_pieces_are_not_editable          (negative)
                                test_21_synth_piece_uses_only_builtin_sounds_and_no_song_audio
                                test_21_a_sampled_piece_is_not_a_synth_piece                   (negative)
                                test_21_both_pieces_pass_the_editability_check
                                test_21_tempo_is_set_from_the_song_bpm
                                test_21_wrong_or_missing_setcps_is_detected                    (negative)
  2.2 honest measurement        test_22_comparisons_state_mode_editability_and_both_similarities
                                test_22_unstamped_or_unscored_comparison_is_rejected           (negative)
                                test_22_tempo_is_within_half_a_percent_of_the_song
                                test_22_a_sped_up_recording_is_detected                        (negative)
                                test_22_promoted_run_metadata_matches_its_comparison
                                test_22_compare_audio_stamps_mode_and_editability_on_an_editable_piece
                                test_22_failing_editability_never_reports_similarity           (negative)
  2.3 permanent reference       test_23_dataset_has_both_modes_pointing_at_the_recorded_runs
                                test_23_dataset_entry_missing_a_mode_or_run_is_detected        (negative)
                                test_23_shortfall_records_equal_the_run_and_floors_are_untouched
                                test_23_a_tampered_record_or_lowered_floor_is_detected         (negative)
                                test_23_floors_are_derived_from_the_measured_run_minus_margin
                                test_23_claude_md_names_the_second_track_with_its_run_ids
  2.4 hosted pieces             test_24_manifests_are_host_resolved_and_shaped_for_strudel_samples
                                test_24_malformed_manifest_shapes_are_rejected                 (negative)
                                test_24_build_instruments_emits_a_host_resolved_manifest
                                test_24_every_sound_the_piece_plays_resolves_in_a_hosted_manifest
                                test_24_a_sound_missing_from_the_manifests_is_detected         (negative)
                                test_24_piece_loads_the_public_r2_urls_under_the_tracks_own_folder
                                test_24_local_or_foreign_urls_are_detected                     (negative)
  2.5 defects fixed generally   test_25_pipeline_code_has_no_per_track_constants
                                test_25_a_per_track_constant_would_be_caught                   (negative)
                                test_25_defect_fix_tests_exist_in_go
                                test_25_auto_calibrate_does_not_kill_the_capture
                                test_25_a_pkill_before_render_would_be_caught                  (negative)
                                test_25_clap_detector_loads_safetensors
                                test_25_stamp_records_genre_provenance_for_an_override
                                test_25_stamp_without_override_records_no_override           (negative)
                                test_25_cached_track_metadata_carries_provenance
                                test_25_regime_clt_references_still_clear_their_floors
                                test_25_promote_is_track_agnostic_and_rejections_carry_no_similarity

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_spec004_acceptance.py -q
"""
from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from urllib.parse import urlparse

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))

yaml = pytest.importorskip("yaml")

from editability_check import check_editability, to_json_fields  # noqa: E402
from eval.gate import evaluate_comparison, load_thresholds  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    RunMeta,
    promote_run,
    stamp_track_metadata,
)

# --- the track under test (names only; no behaviour in scripts depends on them) --------------------
TRACK_KEY = "VAGABUNDO NÃO NAMORA"
TRACK_NAME = "VAGABUNDO NÃO NAMORA (Christopher Luz)"
SLUG = "vagabundo-nao-namora"
SOUND = "vagabundo_nao_namora"
R2_HOST = "https://pub-56831423fee34641805da07cfdaf6812.r2.dev"
R2_BASE = f"{R2_HOST}/midi-grep/{SLUG}/"
REGIME_KEY = "Regime CLT (Dj Brunin XM, Aurora Shukita)"

STEMS = REPO / ".cache" / "stems"
TRACK = STEMS / TRACK_KEY
FIX = TESTS / "fixtures" / "editability"
DATASET = REPO / "eval" / "datasets" / "reference_tracks.yaml"
THRESHOLDS = REPO / "eval" / "thresholds.yaml"

STAMP_KEYS = {"editability", "generation_mode", "editability_violations", "editable_voice_count", "texture_voice_count"}
NOTE_RE = re.compile(r"^[a-g][#s]?\d$")   # Strudel accepts both c#3 and cs3 spellings

needs_track = pytest.mark.skipif(not (TRACK / "v002").is_dir(), reason=f"cached track {TRACK_KEY!r} absent")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json(path: Path) -> dict:
    return json.loads(_read(path))


def _sample_piece() -> str:
    return _read(TRACK / "v002" / "output.strudel")


def _synth_piece() -> str:
    return _read(TRACK / "v003" / "output.strudel")


def _comparison(version: str) -> dict:
    return _json(TRACK / version / "comparison.json")


def _code_lines(code: str) -> str:
    """Strudel source without `//` comment text (header comments are not code)."""
    return "\n".join(re.sub(r"(?<!:)//.*$", "", ln) for ln in code.splitlines())


# =========================================================================================================
# 2.1  Two editable pieces from the new track
# =========================================================================================================

def _let_blocks(code: str) -> set[str]:
    return set(re.findall(r"^let (\w+) = \[", code, flags=re.M))


def _playing_sounds(code: str) -> list[str]:
    """Every sound name chained with `.s("x")` / `.sound("x")` in non-comment code."""
    return re.findall(r"\.(?:s|sound)\(\s*\"([^\"]+)\"", _code_lines(code))


@needs_track
def test_21_sample_instrument_piece_has_separate_editable_voice_blocks():
    code = _sample_piece()
    # bass / lead / vocal are separate, named bar arrays (separate editable blocks of notes) ...
    assert {"bass", "lead", "vocal"} <= _let_blocks(code)
    res = check_editability(code)
    labels = {v.label for v in res.editable_voices}
    # ... and the drums are their own editable pattern voices (bd / sd / hh)
    assert {"bass", "lead", "vocal"} <= labels
    assert {"bd", "sd", "hh"} <= labels


@needs_track
def test_21_vocal_is_notes_on_the_slug_vocal_instrument_not_a_stem_replay():
    code = _sample_piece()
    body = _code_lines(code)
    assert re.search(rf"note\(cat\(\.\.\.vocal\)\)\.s\(\"{SOUND}_vocal\"\)", body)
    assert not re.search(r"vocalsfull|originalfull|origseg|loopAt\(|\.slice\(", body)
    voice = next(v for v in check_editability(code).editable_voices if v.label == "vocal")
    assert voice.sounds == [f"{SOUND}_vocal"] and voice.arrays == ["vocal"]


def test_21_vocal_replay_and_loop_pieces_are_not_editable():
    # negative counterpart: a full-stem vocal replay and a loop-only pack output are rejected
    for name in ("v023_vocal_replay.strudel", "output_loops.strudel", "v012_loop_replay.strudel"):
        res = check_editability(_read(FIX / name))
        assert not res.passed, f"{name} must not pass the editability check"
        assert to_json_fields(res)["editability"] == "fail"
    # and a tampered copy of the real piece (vocal swapped for a stem replay) fails
    if (TRACK / "v002").is_dir():
        tampered = re.sub(rf'\.s\("{SOUND}_vocal"\)', '.s("vocalsfull")', _sample_piece())
        assert not check_editability(tampered).passed


def _is_pure_synth(code: str) -> bool:
    body = _code_lines(code)
    return not re.search(r"\bsamples\(|https?://|\.wav\b|\.mp3\b|vocalsfull|\bvox\d|loop\b", body) \
        and f"{SOUND}_" not in body


@needs_track
def test_21_synth_piece_uses_only_builtin_sounds_and_no_song_audio():
    code = _synth_piece()
    assert _is_pure_synth(code)
    assert "samples(" not in code
    from strudel_validation import VALID_SOUNDS
    sounds = set(_playing_sounds(code))
    assert sounds, "synth piece must play something"
    assert sounds <= set(VALID_SOUNDS), f"non-built-in sounds: {sorted(sounds - set(VALID_SOUNDS))}"
    assert check_editability(code).generation_mode == "synth"


@needs_track
def test_21_a_sampled_piece_is_not_a_synth_piece():
    # negative counterpart: the sampled piece (hosted URLs, song instruments) fails the synth predicate
    assert not _is_pure_synth(_sample_piece())
    assert not _is_pure_synth('await samples("https://x.test/a.json")\nnote("c3").s("sawtooth")')
    assert not _is_pure_synth('note("c3").s("sawtooth")\ns("vocalsfull")')


@needs_track
@pytest.mark.parametrize("version,mode", [("v002", "sample-instrument"), ("v003", "synth")])
def test_21_both_pieces_pass_the_editability_check(version, mode):
    code = _read(TRACK / version / "output.strudel")
    res = check_editability(code)
    assert res.passed, res.violations
    assert res.generation_mode == mode
    assert len(res.editable_voices) >= 3
    assert not res.replay_voices
    # the same verdict, stamped by the run, is what the comparison records
    assert to_json_fields(res)["editability"] == _comparison(version)["editability"] == "pass"


def _setcps_error(code: str, bpm: float) -> str | None:
    """None when `setcps(cps)` exists and cps == bpm/60/4 within rounding; else the reason."""
    body = _code_lines(code)
    m = re.search(r"const cps = ([0-9.]+)", body)
    if not m or not re.search(r"\bsetcps\(\s*cps\s*\)", body):
        return "no setcps(cps) from a declared cps"
    expected = bpm / 60 / 4
    return None if abs(float(m.group(1)) - expected) / expected < 0.005 else f"cps {m.group(1)} != {expected:.6f}"


@needs_track
@pytest.mark.parametrize("version", ["v002", "v003"])
def test_21_tempo_is_set_from_the_song_bpm(version):
    bpm = _json(TRACK / "metadata.json")["bpm"]
    assert _setcps_error(_read(TRACK / version / "output.strudel"), bpm) is None


def test_21_wrong_or_missing_setcps_is_detected():
    good = "const cps = 0.538330\nsetcps(cps)\n"
    assert _setcps_error(good, 129.2) is None
    assert _setcps_error(good, 136.0) is not None             # tempo not from this song
    assert _setcps_error("const cps = 0.538330\n", 129.2)     # declared but never applied
    assert _setcps_error("note('c3')", 129.2)                 # absent altogether


# =========================================================================================================
# 2.2  Honest measurement of both pieces
# =========================================================================================================

def _stamp_problems(doc: dict, expected_mode: str) -> list[str]:
    """What is wrong with a run's comparison.json (empty list = honest, complete record)."""
    problems = []
    if doc.get("editability") != "pass":
        problems.append("editability not pass")
    if doc.get("generation_mode") != expected_mode:
        problems.append("generation_mode missing or wrong")
    comp = doc.get("comparison")
    if not isinstance(comp, dict):
        return problems + ["no similarity block"]
    for key in ("overall_similarity", "section_aware_similarity"):
        if not isinstance(comp.get(key), (int, float)):
            problems.append(f"{key} missing")
    if not comp.get("section_aware_window_count"):
        problems.append("no section windows")
    return problems


@needs_track
@pytest.mark.parametrize("version,mode", [("v002", "sample-instrument"), ("v003", "synth")])
def test_22_comparisons_state_mode_editability_and_both_similarities(version, mode):
    doc = _comparison(version)
    assert _stamp_problems(doc, mode) == []
    assert STAMP_KEYS <= set(doc)
    assert 0 < doc["comparison"]["overall_similarity"] <= 1
    assert 0 < doc["comparison"]["section_aware_similarity"] <= 1


def test_22_unstamped_or_unscored_comparison_is_rejected():
    good = {"editability": "pass", "generation_mode": "synth",
            "comparison": {"overall_similarity": 0.7, "section_aware_similarity": 0.7, "section_aware_window_count": 9}}
    assert _stamp_problems(good, "synth") == []
    assert "generation_mode missing or wrong" in _stamp_problems({**good, "generation_mode": None}, "synth")
    assert "generation_mode missing or wrong" in _stamp_problems(good, "sample-instrument")
    assert "editability not pass" in _stamp_problems({**good, "editability": "fail"}, "synth")
    assert "no similarity block" in _stamp_problems({"editability": "pass", "generation_mode": "synth", "comparison": None}, "synth")
    no_sa = {**good, "comparison": {"overall_similarity": 0.7}}
    assert "section_aware_similarity missing" in _stamp_problems(no_sa, "synth")


def _tempo_error_fraction(comp: dict, bpm: float) -> float:
    return abs(float(comp["tempo_diff_bpm"])) / bpm


@needs_track
@pytest.mark.parametrize("version", ["v002", "v003"])
def test_22_tempo_is_within_half_a_percent_of_the_song(version):
    comp = _comparison(version)["comparison"]
    bpm = _json(TRACK / "metadata.json")["bpm"]
    assert _tempo_error_fraction(comp, bpm) <= 0.005
    assert comp["tempo_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_22_a_sped_up_recording_is_detected():
    # the 2026-06 recorder defect: a 136 BPM song read as ~103-170 BPM; 25% off must trip the 0.5% bound
    for diff in (33.0, 6.8, -6.8):
        assert _tempo_error_fraction({"tempo_diff_bpm": diff}, 136.0) > 0.005
    assert _tempo_error_fraction({"tempo_diff_bpm": 0.0}, 129.2) <= 0.005


@needs_track
@pytest.mark.parametrize("version", ["v002", "v003"])
def test_22_promoted_run_metadata_matches_its_comparison(version):
    meta, doc = _json(TRACK / version / "metadata.json"), _comparison(version)
    comp = doc["comparison"]
    assert meta["generation_mode"] == doc["generation_mode"]
    assert meta["editability"] == doc["editability"] == "pass"
    assert meta["similarity_overall"] == pytest.approx(comp["overall_similarity"], abs=5e-4)
    assert meta["similarity_section_aware"] == pytest.approx(comp["section_aware_similarity"], abs=5e-4)
    assert meta["tempo_similarity"] == pytest.approx(comp["tempo_similarity"], abs=5e-4)
    assert "RAW capture" in meta["render"] and "--strudel" in meta["compare"]


# compare_audio.py end to end on tiny synthetic audio: the real CLI, the real detector.
soundfile = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def wavs(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("spec004_cmp")
    sr = 22050
    t = np.arange(int(2.0 * sr)) / sr
    rng = np.random.default_rng(4)
    a = (0.30 * np.sin(2 * np.pi * 110 * t) + 0.10 * np.sin(2 * np.pi * 880 * t) + 0.02 * rng.standard_normal(t.size)).astype("float32")
    b = (0.30 * np.sin(2 * np.pi * 115 * t) + 0.08 * np.sin(2 * np.pi * 900 * t) + 0.02 * rng.standard_normal(t.size)).astype("float32")
    o, r = d / "orig.wav", d / "rend.wav"
    soundfile.write(str(o), a, sr)
    soundfile.write(str(r), b, sr)
    return o, r


def _compare(wavs, strudel: Path, out: Path) -> subprocess.CompletedProcess:
    o, r = wavs
    return subprocess.run([sys.executable, str(SCRIPTS / "compare_audio.py"), str(o), str(r), "-d", "2", "-j",
                           "--strudel", str(strudel), "-o", str(out)],
                          capture_output=True, text=True, timeout=240, cwd=str(SCRIPTS))


def test_22_compare_audio_stamps_mode_and_editability_on_an_editable_piece(wavs, tmp_path):
    out = tmp_path / "comparison.json"
    proc = _compare(wavs, FIX / "synth_pass.strudel", out)
    assert proc.returncode == 0, proc.stderr[-600:]
    doc = json.loads(out.read_text())
    assert doc["editability"] == "pass" and doc["generation_mode"] == "synth"
    assert isinstance(doc["comparison"]["overall_similarity"], float)


def test_22_failing_editability_never_reports_similarity(wavs, tmp_path):
    for name in ("v023_vocal_replay.strudel", "v012_loop_replay.strudel"):
        out = tmp_path / f"{name}.json"
        proc = _compare(wavs, FIX / name, out)
        assert proc.returncode == 3, proc.stderr[-600:]
        doc = json.loads(out.read_text())
        assert doc["editability"] == "fail" and doc["comparison"] is None
        assert doc["editability_violations"], "the result must say why it was rejected"
        assert "similarity" not in out.read_text() and "similarity" not in proc.stdout


# =========================================================================================================
# 2.3  The track becomes a permanent reference
# =========================================================================================================

def _dataset_entries(doc: dict) -> list[dict]:
    return [t for t in doc["tracks"] if t.get("cache_key") == TRACK_KEY]


def _dataset_problems(entries: list[dict], runs: dict[str, str]) -> list[str]:
    """Both modes present, each pointing at the recorded run it came from."""
    problems = []
    by_mode = {e.get("mode"): e for e in entries}
    for mode, version in runs.items():
        e = by_mode.get(mode)
        if e is None:
            problems.append(f"mode {mode} missing")
        elif e.get("version") != version:
            problems.append(f"mode {mode} points at {e.get('version')}, not {version}")
        elif e.get("genre") != "brazilian_funk":
            problems.append(f"mode {mode} genre {e.get('genre')}")
    return problems


RUNS = {"sample-instrument": "v002", "synth": "v003"}


def test_23_dataset_has_both_modes_pointing_at_the_recorded_runs():
    entries = _dataset_entries(yaml.safe_load(_read(DATASET)))
    assert _dataset_problems(entries, RUNS) == []
    # the pointed-at runs are exactly what the flow log recorded (resolvable when the cache exists)
    for e in entries:
        path = STEMS / e["cache_key"] / e["version"] / "comparison.json"
        if path.exists():
            assert _json(path)["generation_mode"] == e["mode"]


def test_23_dataset_entry_missing_a_mode_or_run_is_detected():
    entries = _dataset_entries(yaml.safe_load(_read(DATASET)))
    assert "mode synth missing" in _dataset_problems([e for e in entries if e["mode"] != "synth"], RUNS)
    wrong = [{**e, "version": "v009"} if e["mode"] == "synth" else e for e in entries]
    assert any("points at v009" in p for p in _dataset_problems(wrong, RUNS))
    assert _dataset_problems([], RUNS)


# Floors of the first track at the moment spec 004 started; "never lowered" == these exact values hold.
REGIME_FLOORS = {
    "sample_instrument": {"genres": 0.88, "section_aware": 0.90},
    "synth": {"genres": 0.77, "section_aware": 0.81},
}


def _floor_problems(th: dict) -> list[str]:
    problems = []
    for mode, floors in REGIME_FLOORS.items():
        block = th["modes"][mode]
        for table, want in floors.items():
            got = block[table]["brazilian_funk"]
            if got < want:
                problems.append(f"{mode}.{table}.brazilian_funk lowered to {got} (< {want})")
            elif got != want:
                problems.append(f"{mode}.{table}.brazilian_funk changed to {got} (expected {want}; raising needs its own spec)")
        run = block["measured"]["brazilian_funk"]["run"]
        if not run.startswith(REGIME_KEY):
            problems.append(f"{mode} floor source is {run!r}, not the first track")
    return problems


def _shortfall_problems(entry: dict, comparison: dict, th: dict) -> list[str]:
    """A shortfall record is honest iff the run really falls below a floor AND the record == the run."""
    comp = comparison["comparison"]
    mode = entry["mode"].replace("-", "_")
    floor = th["modes"][mode]["genres"]["brazilian_funk"]
    sa_floor = th["modes"][mode]["section_aware"]["brazilian_funk"]
    problems = []
    below = comp["overall_similarity"] < floor or comp["section_aware_similarity"] < sa_floor \
        or comp["worst_band_diff"] > th["max_worst_band_diff"]
    if entry.get("expected") == "shortfall" and not below:
        problems.append("recorded as shortfall but clears every floor")
    if entry.get("expected") != "shortfall" and below:
        problems.append("below a floor but not recorded as a shortfall")
    rec = entry.get("measured", {})
    if abs(rec.get("overall", comp["overall_similarity"]) - comp["overall_similarity"]) > 5e-4:
        problems.append("recorded overall != run")
    if abs(rec.get("section_aware", comp["section_aware_similarity"]) - comp["section_aware_similarity"]) > 5e-4:
        problems.append("recorded section_aware != run")
    return problems


@needs_track
@pytest.mark.parametrize("version", ["v002", "v003"])
def test_23_shortfall_records_equal_the_run_and_floors_are_untouched(version):
    th = yaml.safe_load(_read(THRESHOLDS))
    assert _floor_problems(th) == []
    entry = next(e for e in _dataset_entries(yaml.safe_load(_read(DATASET))) if e["version"] == version)
    assert _shortfall_problems(entry, _comparison(version), th) == []
    # the shipped gate agrees: fails the floor yet the deliverable is editable
    res = evaluate_comparison(TRACK / version / "comparison.json", genre="brazilian_funk", thresholds=load_thresholds())
    assert not res.passed and res.editability == "pass"
    # the shortfall is explained next to the floors it did not clear
    assert re.search(rf"shortfall .*{re.escape(TRACK_KEY)}.*{version}", _read(THRESHOLDS))


def test_23_a_tampered_record_or_lowered_floor_is_detected():
    th = yaml.safe_load(_read(THRESHOLDS))
    lowered = json.loads(json.dumps(th))
    lowered["modes"]["sample_instrument"]["section_aware"]["brazilian_funk"] = 0.85   # "make the shortfall pass"
    assert any("lowered" in p for p in _floor_problems(lowered))
    foreign = json.loads(json.dumps(th))
    foreign["modes"]["synth"]["measured"]["brazilian_funk"]["run"] = f"{TRACK_KEY}/v003"
    assert any("not the first track" in p for p in _floor_problems(foreign))

    comp = {"comparison": {"overall_similarity": 0.9282, "section_aware_similarity": 0.8721, "worst_band_diff": 0.05}}
    ok = {"mode": "sample-instrument", "expected": "shortfall", "measured": {"overall": 0.9282, "section_aware": 0.8721}}
    assert _shortfall_problems(ok, comp, th) == []
    assert "recorded overall != run" in _shortfall_problems({**ok, "measured": {"overall": 0.95}}, comp, th)
    assert "recorded as shortfall but clears every floor" in _shortfall_problems(
        ok, {"comparison": {**comp["comparison"], "section_aware_similarity": 0.95}}, th)
    assert "below a floor but not recorded as a shortfall" in _shortfall_problems({**ok, "expected": "pass"}, comp, th)


def test_23_floors_are_derived_from_the_measured_run_minus_margin():
    th = yaml.safe_load(_read(THRESHOLDS))
    for mode, block in th["modes"].items():
        for genre, m in block["measured"].items():
            assert block["genres"][genre] == pytest.approx(round(m["overall"] - m["margin"], 2), abs=1e-9), (mode, genre)
            assert block["section_aware"][genre] == pytest.approx(round(m["section_aware"] - m["margin"], 2), abs=1e-9), (mode, genre)
            # genre stays brazilian_funk, so no new per-genre block may have been typed in by hand
    assert set(th["modes"]["sample_instrument"]["genres"]) == {"brazilian_funk"}
    assert set(th["modes"]["synth"]["genres"]) == {"brazilian_funk"}
    # negative: the shipped relation would not hold for a hand-typed value
    assert round(0.9282 - 0.02, 2) != 0.85


def test_23_claude_md_names_the_second_track_with_its_run_ids():
    text = _read(REPO / "CLAUDE.md")
    assert TRACK_KEY in text and "Second reference track" in text
    para = text[text.index("Second reference track"):][:1400]
    assert "`v002`" in para and "`v003`" in para
    assert "92.8%" in para and "87.2%" in para and "72.1%" in para and "75.2%" in para
    assert "shortfall" in para and "NOT lowered" in para
    # the first track's numbers sit in the same section (side by side)
    assert "Regime CLT" in text[: text.index("Second reference track")]


# =========================================================================================================
# 2.4  The pieces play from the public host
# =========================================================================================================

def _manifest_problems(doc: dict, *, expect_notes: set[str], expect_arrays: set[str]) -> list[str]:
    """Strudel `samples(jsonUrl)` shape: absolute `_base` ending `/`, note dicts, single-element arrays."""
    problems = []
    base = doc.get("_base")
    if not isinstance(base, str) or not base.startswith("https://") or not base.endswith("/"):
        problems.append("_base must be an absolute https URL ending in '/'")
    for name in expect_notes:
        entry = doc.get(name)
        if not isinstance(entry, dict) or not entry:
            problems.append(f"{name}: not a note-keyed dict")
        elif not all(NOTE_RE.match(k) for k in entry):
            problems.append(f"{name}: non-note keys {[k for k in entry if not NOTE_RE.match(k)]}")
        elif not all(isinstance(v, str) and v.endswith(".wav") and not v.startswith("/") for v in entry.values()):
            problems.append(f"{name}: values must be relative .wav paths")
    for name in expect_arrays:
        entry = doc.get(name)
        if not (isinstance(entry, list) and entry and all(isinstance(v, str) for v in entry)):
            problems.append(f"{name}: not an array of paths")
    for k, v in doc.items():
        if k != "_base" and isinstance(v, str):
            problems.append(f"{k}: bare string value (samples() only accepts arrays/dicts)")
    return problems


@needs_track
def test_24_manifests_are_host_resolved_and_shaped_for_strudel_samples():
    pack = TRACK / "sample_pack"
    inst = _json(pack / "instruments" / "samples.json")
    assert _manifest_problems(inst, expect_notes={f"{SOUND}_bass", f"{SOUND}_lead"},
                              expect_arrays={"bd", "sd", "hh", "oh"}) == []
    assert inst["_base"] == f"{R2_BASE}instruments/"
    assert all(len(inst[k]) == 1 for k in ("bd", "sd", "hh", "oh")), "kit entries are single-element arrays"
    main = _json(pack / "samples.json")
    assert _manifest_problems(main, expect_notes={f"{SOUND}_vocal"}, expect_arrays={"vox0"}) == []
    assert main["_base"] == R2_BASE


def test_24_malformed_manifest_shapes_are_rejected():
    good = {"_base": "https://h.test/p/", "x_bass": {"g1": "x_bass/g1.wav"}, "bd": ["kit/bd.wav"]}
    assert _manifest_problems(good, expect_notes={"x_bass"}, expect_arrays={"bd"}) == []
    cases = {
        "no trailing slash": {**good, "_base": "https://h.test/p"},
        "relative base": {**good, "_base": "/p/"},
        "localhost http": {**good, "_base": "http://localhost:5555/p/"},
        "bare string value": {**good, "bd": "kit/bd.wav"},
        "scalar kit entry as str": {**good, "oh": "kit/oh.wav"},
        "non-note key": {**good, "x_bass": {"low": "x_bass/low.wav"}},
        "missing instrument": {k: v for k, v in good.items() if k != "x_bass"},
        "absolute value path": {**good, "x_bass": {"g1": "/x_bass/g1.wav"}},
    }
    for label, doc in cases.items():
        assert _manifest_problems(doc, expect_notes={"x_bass"}, expect_arrays={"bd"}), label


def test_24_build_instruments_emits_a_host_resolved_manifest(tmp_path):
    """The producer, not just the cached output: fixture model -> manifest that passes the same shape check."""
    from build_instruments import KIT_SOUNDS, build_instruments
    model = tmp_path / "models" / "demo_bass"
    (model / "pitched").mkdir(parents=True)
    (model / "pitched" / "g_sharp_1.wav").write_bytes(b"x")
    (model / "pitched" / "c2.wav").write_bytes(b"y")
    (model / "metadata.json").write_text(json.dumps(
        {"name": "demo_bass", "pitched_map": {"g#1": "pitched/g_sharp_1.wav", "c2": "pitched/c2.wav"}, "grains": []}))
    kit = tmp_path / "drums"
    kit.mkdir()
    for s in KIT_SOUNDS:
        (kit / f"{s}.wav").write_bytes(b"k")
    out = tmp_path / "out"
    build_instruments([model], kit, out, "https://h.test/midi-grep/demo/instruments")   # no trailing slash on input
    doc = json.loads((out / "samples.json").read_text())
    assert _manifest_problems(doc, expect_notes={"demo_bass"}, expect_arrays=set(KIT_SOUNDS)) == []
    assert doc["_base"] == "https://h.test/midi-grep/demo/instruments/"
    # every manifest path exists in the pack that would be uploaded
    for rel in [*doc["demo_bass"].values(), *(p for k in KIT_SOUNDS for p in doc[k])]:
        assert (out / rel).is_file(), rel


def _unresolved_sounds(code: str, manifests: list[dict], builtin: set[str]) -> list[str]:
    hosted = {k for m in manifests for k in m if k != "_base"}
    plays = set(_playing_sounds(code))
    # drum voices are `s("bd ~ sd ...")` with mini-notation tokens
    for m in re.finditer(r"(?<![.\w])s\(\s*\"([^\"]+)\"", _code_lines(code)):
        plays.update(t for t in re.split(r"[\s\[\]<>~*/!@?(),:.]+", m.group(1)) if t and not t.isdigit())
    return sorted(s for s in plays if s not in hosted and s not in builtin)


@needs_track
def test_24_every_sound_the_piece_plays_resolves_in_a_hosted_manifest():
    from strudel_validation import VALID_SOUNDS
    pack = TRACK / "sample_pack"
    manifests = [_json(pack / "instruments" / "samples.json"), _json(pack / "samples.json")]
    assert _unresolved_sounds(_sample_piece(), manifests, set(VALID_SOUNDS)) == []
    # the four required instrument families are all hosted
    hosted = {k for m in manifests for k in m}
    assert {f"{SOUND}_bass", f"{SOUND}_lead", f"{SOUND}_vocal", "bd", "sd", "hh"} <= hosted


@needs_track
def test_24_a_sound_missing_from_the_manifests_is_detected():
    from strudel_validation import VALID_SOUNDS
    pack = TRACK / "sample_pack"
    manifests = [_json(pack / "instruments" / "samples.json"), _json(pack / "samples.json")]
    broken = [{k: v for k, v in m.items() if k != f"{SOUND}_vocal"} for m in manifests]
    assert f"{SOUND}_vocal" in _unresolved_sounds(_sample_piece(), broken, set(VALID_SOUNDS))
    ghost = _sample_piece().replace(f'.s("{SOUND}_lead")', '.s("ghost_lead")')
    assert "ghost_lead" in _unresolved_sounds(ghost, manifests, set(VALID_SOUNDS))


def _sample_urls(code: str) -> list[str]:
    return re.findall(r"samples\(\s*\"([^\"]+)\"", _code_lines(code))


def _url_problems(urls: list[str], slug: str, host: str) -> list[str]:
    problems = []
    if len(urls) < 2:
        problems.append("expected an instruments manifest and a pack manifest")
    for u in urls:
        p = urlparse(u)
        if p.scheme != "https" or f"{p.scheme}://{p.netloc}" != host:
            problems.append(f"{u}: not on the project's public host")
        elif not p.path.startswith(f"/midi-grep/{slug}/"):
            problems.append(f"{u}: not under this track's own folder")
    return problems


@needs_track
def test_24_piece_loads_the_public_r2_urls_under_the_tracks_own_folder():
    urls = _sample_urls(_sample_piece())
    assert _url_problems(urls, SLUG, R2_HOST) == []
    assert set(urls) == {f"{R2_BASE}instruments/samples.json", f"{R2_BASE}samples.json"}
    # the first track lives next to it on the same host (siblings under midi-grep/)
    regime = STEMS / REGIME_KEY / "v026" / "output.strudel"
    if regime.exists():
        regime_urls = _sample_urls(_read(regime))
        assert regime_urls and {urlparse(u).netloc for u in regime_urls} == {urlparse(u).netloc for u in urls}
        assert {Path(urlparse(u).path).parts[1] for u in regime_urls} == {"midi-grep"}
        assert not any(f"/{SLUG}/" in u for u in regime_urls)


def test_24_local_or_foreign_urls_are_detected():
    ok = [f"{R2_BASE}instruments/samples.json", f"{R2_BASE}samples.json"]
    assert _url_problems(ok, SLUG, R2_HOST) == []
    assert _url_problems(["http://localhost:5555/vagabundo-nao-namora/samples.json", ok[1]], SLUG, R2_HOST)
    assert _url_problems([ok[0], f"{R2_HOST}/midi-grep/regime-clt/samples.json"], SLUG, R2_HOST)
    assert _url_problems([ok[0], "https://evil.test/midi-grep/vagabundo-nao-namora/samples.json"], SLUG, R2_HOST)
    assert _url_problems(ok[:1], SLUG, R2_HOST)       # only one manifest loaded


# =========================================================================================================
# 2.5  Defects are fixed in the pipeline, not around it
# =========================================================================================================

def _py_code_only(src: str) -> str:
    """Python source without comments and docstrings (examples in prose are not constants)."""
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    kept = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT or tok.start[0] in doc_lines:
            continue
        kept.append(tok.string)
    return " ".join(kept)


def _sh_code_only(src: str) -> str:
    out = []
    for ln in src.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s+#\s.*$", "", ln))
    return "\n".join(out)


# the song's own facts, as they appear in its metadata / the spec: none may live in pipeline code
def _track_literals(meta: dict) -> list[re.Pattern]:
    bpm = str(meta["bpm"])
    return [
        re.compile(rf"\b{re.escape(str(round(meta['bpm'])))}(\.\d+)?\b"),   # 129 / 129.2 / 129.19921875
        re.compile(re.escape(bpm)),
        re.compile(re.escape(meta["key"]), re.I),                           # "C# minor"
        re.compile(r"vagabundo|nao[-_ ]namora|n[ãa]o namora|SKjOR5EOR8Y|christopher", re.I),
    ]


FALLBACK_META = {"bpm": 129.19921875, "key": "C# minor"}
PIPELINE_CODE = [SCRIPTS.parent / "editable-pipeline.sh", SCRIPTS / "build_instruments.py", SCRIPTS / "pipeline_helpers.py"]


def _literal_hits(path: Path, src: str | None = None) -> list[str]:
    meta = _json(TRACK / "metadata.json") if (TRACK / "metadata.json").exists() else FALLBACK_META
    src = _read(path) if src is None else src
    code = _sh_code_only(src) if path.suffix == ".sh" else _py_code_only(src)
    return [pat.pattern for pat in _track_literals(meta) if pat.search(code)]


@pytest.mark.parametrize("path", PIPELINE_CODE, ids=lambda p: p.name)
def test_25_pipeline_code_has_no_per_track_constants(path):
    assert path.is_file(), path
    assert _literal_hits(path) == [], f"{path.name} embeds the second track's facts"


def test_25_a_per_track_constant_would_be_caught():
    sh = PIPELINE_CODE[0]
    assert _literal_hits(sh, "BPM=129.2\n")
    assert _literal_hits(sh, 'KEY="C# minor"\n')
    assert _literal_hits(sh, 'SLUG=vagabundo-nao-namora\n')
    assert _literal_hits(sh, "URL=https://youtu.be/SKjOR5EOR8Y\n")
    py = PIPELINE_CODE[1]
    assert _literal_hits(py, "BPM = 129\n")
    assert _literal_hits(py, 'NAME = "VAGABUNDO NÃO NAMORA"\n')
    # prose is not code: comments and docstrings do not count
    assert _literal_hits(py, '"""e.g. VAGABUNDO NÃO NAMORA at 129 BPM"""\nx = 1  # 129\n') == []
    assert _literal_hits(sh, "# 129.2 BPM example\necho hi  # vagabundo\n") == []


def _go_test_names(path: Path) -> set[str]:
    return set(re.findall(r"^func (Test\w+)\(", _read(path), flags=re.M))


def test_25_defect_fix_tests_exist_in_go():
    yt = _go_test_names(REPO / "internal" / "audio" / "youtube_test.go")
    assert {"TestYtDlpBinaryPrefersEnvOverride", "TestYtDlpBinaryPrefersProjectVenv", "TestYtDlpBinaryFallsBackToPath"} <= yt
    main = _go_test_names(REPO / "cmd" / "midi-grep" / "main_test.go")
    assert {"TestShouldGenerateReport", "TestAbsOutputDir"} <= main
    # the fixed functions exist in production code (a test of a deleted function would not compile)
    assert re.search(r"^func ytDlpBinary\(", _read(REPO / "internal" / "audio" / "youtube.go"), re.M)
    main_go = _read(REPO / "cmd" / "midi-grep" / "main.go")
    assert re.search(r"^func shouldGenerateReport\(", main_go, re.M) and re.search(r"^func absOutputDir\(", main_go, re.M)
    # negative: the defect fixes are asserted from both sides (accept AND reject cases)
    src = _read(REPO / "cmd" / "midi-grep" / "main_test.go")
    assert re.search(r'"none"|\bnone\b', src), "shouldGenerateReport must be tested with --render none"
    assert re.search(r'absOutputDir\("/', src), "absOutputDir must be tested with an absolute input"


def _kills_capture(script: str) -> list[str]:
    return [ln for ln in _sh_code_only(script).splitlines() if re.search(r"\bpkill\b[^\n]*-9|\bkill\s+-9\b|\bpkill\s+-KILL", ln)]


def test_25_auto_calibrate_does_not_kill_the_capture():
    sh = REPO / "scripts" / "auto-calibrate.sh"
    assert _kills_capture(_read(sh)) == []
    assert not _kills_capture(_read(REPO / "scripts" / "editable-pipeline.sh"))
    # it waits for the recorder to go idle instead
    assert re.search(r"pgrep", _sh_code_only(_read(sh)))


def test_25_a_pkill_before_render_would_be_caught():
    assert _kills_capture("pkill -9 ffmpeg\n./render.sh\n")
    assert _kills_capture("  pkill -9 -f record-strudel-blackhole || true\n")
    assert not _kills_capture("# Never pkill -9 the capture\nwhile pgrep ffmpeg; do sleep 1; done\n")


def test_25_clap_detector_loads_safetensors():
    code = _py_code_only(_read(SCRIPTS / "detect_genre_dl.py"))
    assert "use_safetensors = True" in code
    # the PRIMARY CLAP load passes it; the plain load is only the except-branch fallback after it
    src = _read(SCRIPTS / "detect_genre_dl.py")
    calls = [m.group(1) for m in re.finditer(r"ClapModel\.from_pretrained\(([^)]*)\)", src)]
    assert calls and "use_safetensors=True" in calls[0], calls


def _vdir(tmp_path: Path, bpm: float = 140.0, key: str = "A minor") -> Path:
    track = tmp_path / "Some Other Song"
    (track / "v001").mkdir(parents=True)
    (track / "v001" / "metadata.json").write_text(json.dumps({"bpm": bpm, "key": key, "style": "house"}))
    (track / "metadata.json").write_text(json.dumps({"title": "Some Other Song"}))
    return track


def test_25_stamp_records_genre_provenance_for_an_override(tmp_path):
    track = _vdir(tmp_path)
    det = tmp_path / "det.json"
    det.write_text(json.dumps({"detected_genre": "trance", "confidence": 0.31}))
    meta = stamp_track_metadata(track, genre_override="brazilian_funk", detector_json=det)
    assert meta["genre"] == "brazilian_funk" and meta["genre_override"] is True
    assert meta["genre_detected"] == "trance" and meta["genre_confidence"] == 0.31
    assert "trance" in meta["genre_override_note"] and "brazilian_funk" in meta["genre_override_note"]
    assert meta["bpm"] == 140.0 and meta["key"] == "A minor" and meta["style_heuristic"] == "house"
    assert json.loads((track / "metadata.json").read_text()) == meta        # persisted, not just returned


def test_25_stamp_without_override_records_no_override(tmp_path):
    track = _vdir(tmp_path)
    det = tmp_path / "det.json"
    det.write_text(json.dumps({"detected_genre": "house", "confidence": 0.7}))
    meta = stamp_track_metadata(track, detector_json=det)
    assert meta["genre"] == "house" and meta["genre_override"] is False
    assert "genre_override_note" not in meta
    # no detector, no override: falls back to the heuristic, still marked as not overridden
    track2 = _vdir(tmp_path / "b")
    meta2 = stamp_track_metadata(track2)
    assert meta2["genre"] == "house" and meta2["genre_override"] is False and "genre_detected" not in meta2
    # missing analysis is an error, not a silent default
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        stamp_track_metadata(empty)


@needs_track
def test_25_cached_track_metadata_carries_provenance():
    meta = _json(TRACK / "metadata.json")
    for key in ("bpm", "key", "duration", "genre", "genre_detected", "genre_override", "style_heuristic"):
        assert key in meta, key
    assert meta["genre_override"] is True and meta["genre"] != meta["style_heuristic"] or "genre_override_note" in meta
    assert "genre_override_note" in meta and str(meta["genre_detected"]) in meta["genre_override_note"]
    # the bpm in every piece's cps is this analysed bpm (stage 1b feeds the driver)
    for version in ("v002", "v003"):
        assert _setcps_error(_read(TRACK / version / "output.strudel"), meta["bpm"]) is None


@pytest.mark.parametrize("version,floor_ok", [("v026", True), ("v027", True)])
def test_25_regime_clt_references_still_clear_their_floors(version, floor_ok):
    path = STEMS / REGIME_KEY / version / "comparison.json"
    if not path.exists():
        pytest.skip(f"{REGIME_KEY}/{version} cache absent")
    res = evaluate_comparison(path, genre="brazilian_funk", thresholds=load_thresholds())
    assert res.passed is floor_ok, res.message
    # negative: the same record with its section-aware score knocked below the floor must fail
    doc = _json(path)
    doc["comparison"]["section_aware_similarity"] = 0.10
    broken = path.parent / ".spec004_probe.json"
    try:
        broken.write_text(json.dumps(doc))
        assert not evaluate_comparison(broken, genre="brazilian_funk", thresholds=load_thresholds()).passed
    finally:
        broken.unlink(missing_ok=True)


def test_promote_run_is_general_acceptance(tmp_path):
    """The promote step is track-agnostic: an editable run on any title yields the four stamped files,
    and a rejected run yields a record with no similarity."""
    s, w = tmp_path / "o.strudel", tmp_path / "r.wav"
    s.write_text("x")
    w.write_bytes(b"w")
    ok, bad = tmp_path / "ok.json", tmp_path / "bad.json"
    ok.write_text(json.dumps({"editability": "pass", "generation_mode": "synth", "comparison": {
        "overall_similarity": 0.5, "section_aware_similarity": 0.4, "frequency_balance_similarity": 0.3, "tempo_similarity": 1.0}}))
    bad.write_text(json.dumps({"editability": "fail", "generation_mode": "loops", "comparison": None}))
    meta = RunMeta(generator="g", render="r", compare="c")
    v1 = promote_run(tmp_path / "Any Title", "synth", s, w, ok, meta)
    v2 = promote_run(tmp_path / "Any Title", "synth", s, w, bad, meta)
    assert (v1.name, v2.name) == ("v001", "v002")
    m1, m2 = json.loads((v1 / "metadata.json").read_text()), json.loads((v2 / "metadata.json").read_text())
    assert m1["similarity_overall"] == 0.5 and m1["editability"] == "pass"
    assert m2["similarity_overall"] is None and m2["editability"] == "fail"
