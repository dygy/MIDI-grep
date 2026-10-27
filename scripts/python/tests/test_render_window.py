# @layer: integration
# @regression
"""The render length and compare window follow the track (whole bars = bars*240/bpm), not 170 s / 135 s constants:
a render past the song end makes the looped Strudel wrap back to bar 0, and the report drew it squeezed."""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline_helpers as ph  # noqa: E402
from test_editable_pipeline_resume import SLUG, TITLE, repo, run, calls, set_bpm  # noqa: E402,F401

SCRIPTS = Path(__file__).resolve().parent.parent
AUTO = SCRIPTS.parent / "auto-calibrate.sh"


@pytest.mark.parametrize("duration,bpm", [(137.76, 135.017), (151.38, 129.989), (100.0, 129.989), (60.0, 90.0)])
def test_render_seconds_is_the_whole_bars_of_the_track(duration, bpm):
    s = ph.render_seconds(duration, bpm)
    assert s == pytest.approx(ph.num_bars(duration, bpm) * 240 / bpm, abs=1e-3)
    assert s <= duration and duration - s < 240 / bpm          # never past the song, never more than a bar short


def test_render_seconds_cli_matches():
    out = subprocess.run([sys.executable, str(SCRIPTS / "pipeline_helpers.py"), "render-seconds", "137.76", "135.017"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert float(out) == ph.render_seconds(137.76, 135.017) != 170


def test_pipeline_renders_and_compares_the_derived_seconds(repo):
    set_bpm(repo.track, 129.989)                                  # duration 100 s -> 54 bars
    r = run(repo, "--local", "--mode", "synth")
    want = ph.render_seconds(100.0, 129.989)
    ac = [c for c in calls(repo) if c.startswith("auto-calibrate")][0]
    assert f"--dur {want:g}" in ac and "--dur 170" not in ac, ac
    assert f"compare={want:g}s" in r.stderr, r.stderr


def test_explicit_dur_still_overrides(repo):
    run(repo, "--local", "--mode", "synth", "--dur", "42")
    assert "--dur 42" in [c for c in calls(repo) if c.startswith("auto-calibrate")][0]


def test_auto_calibrate_defaults_to_the_bar_seconds_and_scores_that_window(tmp_path, monkeypatch):
    from test_calibrate_loop import drive
    log = tmp_path / "argv.log"
    # drive() stubs: the recorder logs its -d, compare logs its -d
    import test_calibrate_loop as tcl
    monkeypatch.setattr(tcl, "STUB_REC", ('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo wav > "$2"; [ "$1" = "-d" ] && echo "rec $2" >> '
                    + f'"{log}"; shift; done\n'))
    monkeypatch.setattr(tcl, "STUB_CMP", 'import sys\nopen(%r, "a").write("cmp " + sys.argv[sys.argv.index("-d") + 1] + "\\n")\n' % str(log) + tcl.STUB_CMP)
    r, _ = drive(tmp_path, [(0.8, 0.8)], "--iters", "1", "--bars", "77", "--bpm", "135.017")
    want = f"{math.floor(77 * 240 / 135.017 * 1000) / 1000:g}"
    assert log.read_text().split("\n")[:2] == [f"rec {want}", f"cmp {want}"], (log.read_text(), r.stdout, r.stderr)


def test_report_waveforms_share_one_time_axis():
    src = (SCRIPTS / "generate_report.py").read_text()
    body = src[src.index("function drawTrackWaveform"):src.index("// Draw all waveforms")]
    assert re.search(r"buffer\.duration\s*/\s*maxDuration", body), "each waveform must span duration/maxDuration of the lane"
    assert "i < width" not in body, "stretching every buffer over the full lane width squeezes the shorter one"
