# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Report editability verdict tests (spec 003 §2.2-F, Slice 5).

The HTML report must show the generation mode and the editability verdict and must never
present a replay run's similarity as a headline score:

    editability: pass   → headline "<pct>% — mode: <generation_mode> · editable: pass"
    editability: fail   → red "REPLAY / UNVERIFIED — not a deliverable" badge, violations
                          listed, NO percentage headline
    keys absent (legacy)→ headline unchanged + "editability: not checked (pre-spec-003 run)"

The per-stem section carries the caption that its stems come from demucs re-separation.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_report_editability.py -q
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_report  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report"

BADGE = "REPLAY / UNVERIFIED — not a deliverable"
LEGACY_NOTE = "editability: not checked (pre-spec-003 run)"
STEM_CAPTION = (
    "stems obtained by demucs re-separation of the rendered mix "
    "(lossy, not a true stem-match view)"
)


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _headline(html: str) -> str | None:
    m = re.search(r'<div class="overall-headline"[^>]*>(.*?)</div>\s*</div>', html, re.S)
    return m.group(1) if m else None


# 44-byte header of an empty 16-bit mono 44.1 kHz WAV — generate_report only base64-embeds it.
_EMPTY_WAV = (
    b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (44100).to_bytes(4, "little")
    + (88200).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    + b"data" + (0).to_bytes(4, "little")
)


def _build_report(
    tmp_path: Path,
    comparison_name: str,
    with_stems: bool = False,
    with_similarity_png: bool = False,
) -> str:
    """Write a minimal cache/version dir and run the real generate_report()."""
    cache = tmp_path / "yt_fixture"
    version = cache / "v001"
    version.mkdir(parents=True)
    if with_similarity_png:
        (version / "chart_similarity.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    for stem in ("melodic", "drums", "bass", "vocals"):
        (cache / f"{stem}.wav").write_bytes(_EMPTY_WAV)
    for name in ("render", "render_melodic", "render_drums", "render_bass"):
        (version / f"{name}.wav").write_bytes(_EMPTY_WAV)
    shutil.copy(FIXTURES / comparison_name, version / "comparison.json")
    if with_stems:
        shutil.copy(FIXTURES / "stem_comparison_min.json", version / "stem_comparison.json")
    (version / "output.strudel").write_text("// BPM: 136\n$: note('c3').s('sawtooth')\n")
    out = tmp_path / "report.html"
    generate_report.generate_report(str(cache), str(version), str(out))
    return out.read_text()


# --------------------------------------------------------------------------- unit: charts block


def test_pass_headline_has_mode_and_editable():
    html = generate_report.generate_charts_html(_load("comparison_pass_sample_instrument.json"))
    headline = _headline(html)
    assert headline is not None, "pass run must render the overall-headline block"
    assert "mode:" in headline and "editable:" in headline
    assert "94% — mode: sample-instrument · editable: pass" in headline
    assert BADGE not in html
    assert LEGACY_NOTE not in html


def test_fail_renders_badge_and_no_percentage_headline():
    html = generate_report.generate_charts_html(_load("comparison_fail_replay.json"))
    assert BADGE in html
    assert 'class="editability-badge"' in html
    assert _headline(html) is None, "replay run must not render a percentage headline"
    # No similarity score anywhere in the block — the fixture carries comparison: null
    assert "Overall Similarity" not in html
    assert "Similarity Scores" not in html
    assert not re.search(r"\d+%", html), "no percentage may appear for a replay run"
    # violations are listed under the badge
    for violation in _load("comparison_fail_replay.json")["editability_violations"]:
        assert generate_report.html.escape(violation) in html


def test_fail_with_comparison_data_still_hides_scores():
    """A stamped failing run that still carries numbers (e.g. v023) shows no score."""
    data = _load("comparison_pass_sample_instrument.json")
    data["editability"] = "fail"
    data["editability_violations"] = ["R1: vocal stem replay"]
    html = generate_report.generate_charts_html(data)
    assert BADGE in html
    assert _headline(html) is None
    assert "Similarity Scores" not in html
    assert "94%" not in html


def test_legacy_without_keys_keeps_headline_and_notes_unchecked():
    html = generate_report.generate_charts_html(_load("comparison_legacy_no_keys.json"))
    headline = _headline(html)
    assert headline is not None
    assert "94%" in headline
    assert "mode:" not in headline and "editable:" not in headline
    assert LEGACY_NOTE in html
    assert BADGE not in html


def test_stem_section_has_demucs_caption():
    html = generate_report.generate_stem_comparison_html(_load("stem_comparison_min.json"), {})
    assert STEM_CAPTION in html


def test_stem_section_withholds_scores_for_replay_run():
    html = generate_report.generate_stem_comparison_html(
        _load("stem_comparison_min.json"), {}, editability="fail"
    )
    assert STEM_CAPTION in html
    assert "Weighted Per-Stem Similarity" not in html
    assert not re.search(r"\d+%", html)


# --------------------------------------------------------------------------- end-to-end report


@pytest.mark.parametrize(
    "fixture, expect, forbid",
    [
        (
            "comparison_pass_sample_instrument.json",
            ["mode: sample-instrument · editable: pass", STEM_CAPTION],
            [BADGE, LEGACY_NOTE],
        ),
        (
            "comparison_fail_replay.json",
            [BADGE, STEM_CAPTION, "R2: fewer than 2 editable note() voices"],
            ["Overall Similarity", LEGACY_NOTE],
        ),
        (
            "comparison_legacy_no_keys.json",
            ["Overall Similarity", LEGACY_NOTE, STEM_CAPTION],
            [BADGE, "editable:"],
        ),
    ],
)
def test_full_report_renders_verdict(tmp_path, fixture, expect, forbid):
    html = _build_report(tmp_path, fixture, with_stems=True)
    for s in expect:
        assert s in html, f"missing {s!r} for {fixture}"
    for s in forbid:
        assert s not in html, f"unexpected {s!r} for {fixture}"


def test_full_report_fail_has_no_percentage_headline(tmp_path):
    html = _build_report(tmp_path, "comparison_fail_replay.json", with_similarity_png=True)
    assert _headline(html) is None
    assert 'class="editability-badge"' in html
    # the legacy chart_similarity.png gauge is a replay score by another route
    assert "Similarity Scores" not in html


def test_full_report_pass_keeps_similarity_gauge(tmp_path):
    html = _build_report(
        tmp_path, "comparison_pass_sample_instrument.json", with_similarity_png=True
    )
    assert "Similarity Scores (with Overall Gauge)" in html
