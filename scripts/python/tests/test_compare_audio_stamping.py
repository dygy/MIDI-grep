# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""compare_audio.py --strudel stamping tests (spec 003 §2.2-D, "honest measurement").

Runs the real `scripts/python/compare_audio.py` CLI as a subprocess on 2-second synthetic WAVs
(written with soundfile — no Demucs, no render, no network) and proves:

  * a replay fixture (tests/fixtures/editability/v012_loop_replay.strudel) → exit 3, the output
    JSON is {"editability": "fail", "generation_mode": …, "editability_violations": […],
    "comparison": null} and NO `overall_similarity` is produced anywhere;
  * a passing fixture (v023_minus_vocal.strudel) → exit 0 and the five `to_json_fields()` keys
    are present at the TOP LEVEL of the results at every write site (stdout JSON, `-o`, the
    `--chart` comparison.json, and `--stems … --output-dir` stem_comparison.json);
  * without `--strudel` nothing changes (no new keys) — metric math untouched.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_compare_audio_stamping.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

SCRIPTS = Path(__file__).resolve().parent.parent
COMPARE = SCRIPTS / "compare_audio.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "editability"
FAIL_FIXTURE = FIXTURES / "v012_loop_replay.strudel"
PASS_FIXTURE = FIXTURES / "v023_minus_vocal.strudel"

STAMP_KEYS = {
    "editability",
    "generation_mode",
    "editability_violations",
    "editable_voice_count",
    "texture_voice_count",
}
SR = 22050
SECONDS = 2.0


@pytest.fixture(scope="module")
def wavs(tmp_path_factory) -> tuple[Path, Path]:
    """Two short, slightly different synthetic signals (so every metric is well-defined)."""
    d = tmp_path_factory.mktemp("stamping")
    t = np.arange(int(SECONDS * SR)) / SR
    rng = np.random.default_rng(3)
    orig = (0.30 * np.sin(2 * np.pi * 110 * t) + 0.10 * np.sin(2 * np.pi * 880 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    rend = (0.30 * np.sin(2 * np.pi * 115 * t) + 0.08 * np.sin(2 * np.pi * 900 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    o, r = d / "orig.wav", d / "rend.wav"
    soundfile.write(str(o), orig, SR)
    soundfile.write(str(r), rend, SR)
    return o, r


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(COMPARE), *extra],
        capture_output=True, text=True, timeout=240, cwd=str(SCRIPTS),
    )


def _json_from_stdout(proc: subprocess.CompletedProcess) -> dict:
    assert proc.stdout.strip(), f"no JSON on stdout; stderr tail: {proc.stderr[-800:]}"
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# detector FAIL → exit 3, no score
# ---------------------------------------------------------------------------

def test_fail_fixture_exits_3_and_writes_short_circuit_payload(wavs, tmp_path):
    orig, rend = wavs
    out = tmp_path / "comparison.json"
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "-j", "--strudel", str(FAIL_FIXTURE), "-o", str(out))
    assert proc.returncode == 3, proc.stderr[-800:]

    payload = json.loads(out.read_text())
    assert payload["editability"] == "fail"
    assert payload["comparison"] is None
    assert payload["generation_mode"] == "loops"
    assert payload["editability_violations"], "violations must be listed"
    assert any("R1" in v for v in payload["editability_violations"])
    assert set(payload) == STAMP_KEYS | {"comparison"}

    # the honesty contract: no similarity anywhere — not in the file, not on stdout
    assert "overall_similarity" not in out.read_text()
    assert "overall_similarity" not in proc.stdout
    assert _json_from_stdout(proc)["comparison"] is None  # -j echoes the same payload


def test_fail_fixture_without_output_path_prints_payload_only(wavs):
    orig, rend = wavs
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "--strudel", str(FAIL_FIXTURE))
    assert proc.returncode == 3
    payload = _json_from_stdout(proc)
    assert payload["editability"] == "fail" and payload["comparison"] is None
    assert "overall_similarity" not in proc.stdout


def test_fail_fixture_with_chart_writes_payload_as_comparison_json(wavs, tmp_path):
    """The --chart path is how the Go pipeline / ai_improver final step writes comparison.json."""
    orig, rend = wavs
    chart_dir = tmp_path / "v999"
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "--strudel", str(FAIL_FIXTURE),
                "-c", str(chart_dir / "comparison.png"))
    assert proc.returncode == 3
    payload = json.loads((chart_dir / "comparison.json").read_text())
    assert payload["editability"] == "fail" and payload["comparison"] is None
    assert not (chart_dir / "comparison.png").exists(), "no chart for an unscored render"


# ---------------------------------------------------------------------------
# detector PASS → scored + the five keys at the top level, at every write site
# ---------------------------------------------------------------------------

def test_pass_fixture_stamps_stdout_and_output_file(wavs, tmp_path):
    orig, rend = wavs
    out = tmp_path / "comparison.json"
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "-j", "--strudel", str(PASS_FIXTURE), "-o", str(out))
    assert proc.returncode == 0, proc.stderr[-800:]

    for results in (_json_from_stdout(proc), json.loads(out.read_text())):
        assert STAMP_KEYS <= set(results), sorted(results)
        assert results["editability"] == "pass"
        assert results["generation_mode"] == "sample-instrument"
        assert results["editability_violations"] == []
        assert results["editable_voice_count"] >= 2
        assert results["texture_voice_count"] == 0
        # ...and the metric still ran, untouched
        assert 0.0 <= results["comparison"]["overall_similarity"] <= 1.0
        assert "frequency_balance_similarity" in results["comparison"]
        assert "section_aware_similarity" in results["comparison"]


def test_pass_fixture_stamps_chart_write_site(wavs, tmp_path):
    """save_comparison_json() — the `--chart` write site (compare_audio.py ~:1042)."""
    orig, rend = wavs
    chart_dir = tmp_path / "v998"
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "--strudel", str(PASS_FIXTURE),
                "-c", str(chart_dir / "comparison.png"))
    assert proc.returncode == 0, proc.stderr[-800:]
    results = json.loads((chart_dir / "comparison.json").read_text())
    assert STAMP_KEYS <= set(results)
    assert results["editability"] == "pass"
    assert "overall_similarity" in results["comparison"]


def test_pass_fixture_stamps_stems_write_site(wavs, tmp_path):
    """The per-stem `stem_comparison.json` write site (compare_audio.py ~:1781)."""
    orig, rend = wavs
    out_dir = tmp_path / "stems_out"
    proc = _run("--stems", "--original-bass", str(orig), "--rendered-bass", str(rend),
                "-d", str(SECONDS), "--window-size", "1.0", "--output-dir", str(out_dir),
                "-j", "--strudel", str(PASS_FIXTURE))
    assert proc.returncode == 0, proc.stderr[-800:]
    results = json.loads((out_dir / "stem_comparison.json").read_text())
    assert STAMP_KEYS <= set(results)
    assert results["editability"] == "pass" and results["generation_mode"] == "sample-instrument"
    assert "weighted_overall" in results["aggregate"]
    assert STAMP_KEYS <= set(_json_from_stdout(proc))


def test_stems_mode_fail_fixture_writes_payload_to_output_dir(wavs, tmp_path):
    orig, rend = wavs
    out_dir = tmp_path / "stems_fail"
    proc = _run("--stems", "--original-bass", str(orig), "--rendered-bass", str(rend),
                "-d", str(SECONDS), "--output-dir", str(out_dir), "--strudel", str(FAIL_FIXTURE))
    assert proc.returncode == 3
    payload = json.loads((out_dir / "stem_comparison.json").read_text())
    assert payload["editability"] == "fail" and payload["comparison"] is None
    assert "weighted_overall" not in (out_dir / "stem_comparison.json").read_text()


# ---------------------------------------------------------------------------
# without --strudel: unchanged
# ---------------------------------------------------------------------------

def test_without_strudel_no_new_keys(wavs):
    orig, rend = wavs
    proc = _run(str(orig), str(rend), "-d", str(SECONDS), "-j")
    assert proc.returncode == 0, proc.stderr[-800:]
    results = _json_from_stdout(proc)
    assert not (STAMP_KEYS & set(results)), "no --strudel → no stamping keys"
    assert "overall_similarity" in results["comparison"]
