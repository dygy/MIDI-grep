# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""pipeline_helpers.build_report (stage 10): every promoted vNNN gets charts, render stems, a per-stem
comparison, an iteration history and report.html. External commands are injected fakes, so no Demucs,
BlackHole or model download is needed."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from pipeline_helpers import GateVerdict, ReportError, build_report, main  # noqa: E402

SR = 8000


def _sine(path: Path, hz: float) -> None:
    t = np.arange(SR) / SR
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32"), SR)


class FakeRunner:
    """Records argv and writes what the real command would write."""

    def __init__(self, fail_on: str | None = None, ffmpeg_writes: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.solo_sources: dict[str, str] = {}
        self.fail_on = fail_on
        self.ffmpeg_writes = ffmpeg_writes

    def __call__(self, argv: Sequence[object]) -> None:
        a = [str(x) for x in argv]
        self.calls.append(a)
        tool = Path(a[0]).name if Path(a[0]).name == "ffmpeg" else Path(a[1]).name
        if tool == self.fail_on:
            raise RuntimeError(f"{tool} failed")
        if tool == "compare_audio.py" and "--chart" in a:
            out = Path(a[a.index("--chart") + 1])
            out.write_bytes(b"png")
            for n in ("frequency", "similarity", "waveform"):
                (out.parent / f"chart_{n}.png").write_bytes(b"png")
            (out.parent / "comparison.json").write_text('{"clobbered": true}')   # what the real CLI does
        elif tool == "compare_audio.py" and "--stems" in a:
            d = Path(a[a.index("--output-dir") + 1])
            (d / "stem_comparison.json").write_text("{}")
            (d / "chart_stem_overview.png").write_bytes(b"png")
        elif tool == "separate.py":
            out, prefix = Path(a[3]), a[a.index("--prefix") + 1]
            for s in ("melodic", "drums", "bass", "vocals"):
                (out / f"{prefix}_{s}.mp3").write_bytes(b"demucs-mp3")     # the real separate.py leaves mp3 only
        elif tool == "render-strudel-offline.js":                 # the renderer: one solo group per call
            src = Path(a[2]).read_text()
            self.solo_sources[Path(a[a.index("-o") + 1]).name] = src
            live = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("//"))
            for n in re.findall(r"duckorbit\((\d+)\)", live):    # the engine's own refusal
                if f"orbit({n})" not in live.replace(f"duckorbit({n})", ""):
                    raise RuntimeError(f"duck target orbit {n} does not exist; no output written")
            _sine(Path(a[a.index("-o") + 1]), 200.0)
        elif tool == "generate_report.py":
            Path(a[a.index("-o") + 1]).write_text("<html>report</html>")
        elif tool == "ffmpeg" and self.ffmpeg_writes:
            Path(a[-1]).write_bytes(b"mp3")

    def tools(self) -> list[str]:
        return [Path(c[0]).name if Path(c[0]).name == "ffmpeg" else Path(c[1]).name for c in self.calls]


STRUDEL = """let bass = ["c2"]
$: note(cat(...bass)).s("sine").orbit(2)  // BASS — low end
$: note("c4").s("sawtooth")  // LEAD — melody
$: stack(
  note("e4").s("sawtooth"),
  note("g4").s("sawtooth")
)  // texture — CHORDS — harmony
$: s("bd").bank("RolandTR808").duckorbit(2).duckdepth(0.5)  // DRUMS — groove
"""
STAMPED = '{"editability": "pass", "generation_mode": "sample-instrument", "comparison": {"overall_similarity": 0.9}}'


@pytest.fixture()
def track(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(track_dir, vdir, calibration workdir) with cal1 (worse), cal2 (== promoted render), best.*."""
    t = tmp_path / "Some Track"
    v = t / "v002"
    v.mkdir(parents=True)
    for name, hz in (("original", 110), ("bass", 55), ("drums", 90), ("melodic", 440), ("vocals", 600)):
        _sine(t / f"{name}.wav", hz)
    w = tmp_path / "cal"
    w.mkdir()
    _sine(w / "cal1.wav", 200)
    _sine(w / "cal2.wav", 330)
    (v / "render.wav").write_bytes((w / "cal2.wav").read_bytes())      # promoted render == cal2 (the best)
    (v / "comparison.json").write_text(STAMPED)
    (v / "metadata.json").write_text(json.dumps({"version": 2, "generation_mode": "sample-instrument"}))
    (v / "output.strudel").write_text(STRUDEL)
    for n, sim in ((1, 0.80), (2, 0.91)):
        (w / f"cal{n}.strudel").write_text(f"// cal{n}")
        (w / f"cal{n}.cmp.json").write_text(json.dumps({"comparison": {"overall_similarity": sim,
                                                                         "section_aware_similarity": sim - 0.01}}))
    (w / "best.wav").write_bytes((w / "cal2.wav").read_bytes())
    return t, v, w


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_creates_every_artifact_and_declares_report(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    res = build_report(t, v, seconds=1, calib_workdir=w, run=run, render_stems="demucs")
    assert res.report == v / "report.html" and (v / "report.html").is_file()
    for name in ("comparison.png", "chart_frequency.png", "chart_waveform.png", "stem_comparison.json",
                 "chart_stem_overview.png", "render_melodic.mp3", "render_drums.mp3", "render_bass.mp3",
                 "render.mp3", "embed/render_melodic.mp3", "embed/render_bass.mp3", "iterations.json", "embed/melodic.mp3", "embed/vocals.mp3"):
        assert (v / name).is_file(), name
    meta = json.loads((v / "metadata.json").read_text())
    assert meta["report"] == "report.html" and meta["stem_comparison"] == "stem_comparison.json"
    assert meta["render_stems"] == ["render_melodic.mp3", "render_drums.mp3", "render_bass.mp3"]
    assert meta["iterations"] == "iterations.json" and meta["generation_mode"] == "sample-instrument"
    gen = next(c for c in run.calls if c[1].endswith("generate_report.py"))
    assert gen[2:] == [str(t), "-v", "2", "-o", str(v / "report.html")]


def test_stamped_comparison_json_is_byte_identical(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    before = _sha(v / "comparison.json")
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    assert _sha(v / "comparison.json") == before
    assert not list(v.glob(".charts_*")), "temp chart dir must be cleaned up"


def test_rerun_is_idempotent_and_force_redoes(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    again = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=again)
    assert again.calls == []
    forced = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=forced, force=True)
    assert {"compare_audio.py", "separate.py", "generate_report.py"} <= set(forced.tools())


def test_iterations_json_marks_best_and_paths_resolve(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run, gate=GateVerdict(verdict="PASS", message="ok"), render_stems="demucs")
    doc = json.loads((v / "iterations.json").read_text())
    its = {i["version"]: i for i in doc["iterations"]}
    assert [i["was_best"] for i in doc["iterations"]] == [False, True]
    assert doc["best_similarity"] == pytest.approx(0.91) and doc["gate"]["verdict"] == "PASS"
    assert its[2]["section_aware"] == pytest.approx(0.90) and its[1]["phase"] == "calibration"
    for it in its.values():
        for key in ("render_path", "strudel_path", "stem_melodic_path", "stem_drums_path", "stem_bass_path"):
            assert (v / it[key]).is_file(), (it["version"], key)
    # the best iteration reuses the render stems; only the other one goes through Demucs
    seps = [c for c in run.calls if c[1].endswith("separate.py")]
    assert [c[c.index("--prefix") + 1] for c in seps] == ["render", "cal1"]


def test_no_iteration_stems_skips_demucs_on_iterations(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, iteration_stems=False, run=run, render_stems="demucs")
    assert [c[c.index("--prefix") + 1] for c in run.calls if c[1].endswith("separate.py")] == ["render"]
    it = json.loads((v / "iterations.json").read_text())["iterations"][0]
    assert it["stem_bass_path"] is None and it["render_path"] == "iterations/cal1.mp3"


def test_failing_report_runner_raises(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    with pytest.raises(RuntimeError, match="generate_report.py failed"):
        build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner(fail_on="generate_report.py"))


def test_report_that_is_not_written_is_an_error(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    silent = FakeRunner()
    inner = silent.__call__

    def no_report(argv: Sequence[object]) -> None:
        if str(argv[1]).endswith("generate_report.py"):
            return
        inner(argv)

    with pytest.raises(ReportError, match="report.html"):
        build_report(t, v, seconds=1, calib_workdir=w, run=no_report)


def test_cli_exits_nonzero_when_a_step_cannot_run(tmp_path: Path) -> None:
    assert main(["report", "--track-dir", str(tmp_path), "--version", "v009", "--seconds", "1"]) == 1


def test_missing_calibration_renders_is_an_error(track: tuple[Path, Path, Path], tmp_path: Path) -> None:
    t, v, _ = track
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ReportError, match="iteration"):
        build_report(t, v, seconds=1, calib_workdir=empty, run=FakeRunner())


def _ffmpeg_targets(run: FakeRunner) -> list[str]:
    return [Path(c[-1]).name for c in run.calls if Path(c[0]).name == "ffmpeg"]


def test_demucs_mp3s_are_compared_directly_and_nothing_pre_existing_is_deleted(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run, render_stems="demucs")
    cmp = next(c for c in run.calls if "--stems" in c)
    assert [Path(x).name for x in cmp if x.endswith(".mp3")] == ["render_melodic.mp3", "render_drums.mp3", "render_bass.mp3"]
    assert not any(c[-1].endswith(".wav") for c in run.calls if Path(c[0]).name == "ffmpeg"), "no mp3->wav decode"
    for f in ("render.wav", "render_vocals.mp3", "render_melodic.mp3"):
        assert (v / f).is_file()
    assert (t / "bass.wav").is_file(), "original stems are never touched"
    assert (v / "render_melodic.mp3").read_bytes() == b"demucs-mp3", "Demucs' mp3 is the source, never replaced"


def test_ffmpeg_asked_for_render_mp3_and_original_stem_mp3s(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run)
    targets = _ffmpeg_targets(run)
    assert "render.mp3" in targets
    assert {"melodic.mp3", "drums.mp3", "bass.mp3", "vocals.mp3", "render_melodic.mp3"} <= set(targets)
    assert not list((v / "embed").glob("*.wav"))


def test_generate_report_embeds_mp3_when_wav_and_mp3_both_exist(tmp_path: Path) -> None:
    import generate_report as gr
    (tmp_path / "x.wav").write_bytes(b"RIFF" + b"\0" * 4000)
    (tmp_path / "x.mp3").write_bytes(b"ID3mp3")
    assert gr.encode_audio_base64(str(tmp_path / "x.wav")).startswith("data:audio/mpeg;base64,")
    (tmp_path / "y.wav").write_bytes(b"RIFF")
    assert gr.encode_audio_base64(str(tmp_path / "y.wav")).startswith("data:audio/wav;base64,")


def test_stale_twin_is_re_encoded_fresh_one_is_reused(track: tuple[Path, Path, Path]) -> None:
    import os
    t, v, w = track
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    for f in v.rglob("*.mp3"):                      # make every twin comfortably newer than its source...
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 1000))
    again = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=again)
    assert again.calls == []
    future = (v / "render.mp3").stat().st_mtime + 5000
    os.utime(v / "render.wav", (future, future))     # ...then touch the wav newer
    os.utime(t / "bass.wav", (future, future))
    redo = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=redo)
    targets = _ffmpeg_targets(redo)
    assert "render.mp3" in targets and "bass.mp3" in targets and "melodic.mp3" not in targets
    assert any(c[1].endswith("generate_report.py") for c in redo.calls), "stale audio => report rebuilt"


def test_piano_is_the_legacy_fallback_for_melodic(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    (t / "melodic.wav").rename(t / "piano.wav")
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run)
    cmp = next(c for c in run.calls if "--stems" in c)
    assert str(t / "piano.wav") == cmp[cmp.index("--original-melodic") + 1]
    assert (v / "embed" / "melodic.mp3").is_file()


def test_lineage_best_only_when_byte_identical(track: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    t, v, w = track
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    doc = json.loads((v / "iterations.json").read_text())
    assert doc["matched_promoted"] is True and [i["was_best"] for i in doc["iterations"]] == [False, True]
    assert doc["iterations"][1]["label"] == "iter 2"


def test_lineage_mismatch_marks_no_best_and_warns(track: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    t, v, w = track
    _sine(v / "render.wav", 777)                      # promoted render is from some other run
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    doc = json.loads((v / "iterations.json").read_text())
    assert doc["matched_promoted"] is False and not any(i["was_best"] for i in doc["iterations"])
    err = capsys.readouterr().err
    assert str(w) in err and "byte-identical" in err
    # no free ride on the best-iteration shortcut: both iterations are separated by Demucs
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run, force=True, render_stems="demucs")
    assert [c[c.index("--prefix") + 1] for c in run.calls if c[1].endswith("separate.py")] == ["render", "cal1", "cal2"]


def test_editability_failed_iteration_has_null_similarity(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    (w / "cal1.cmp.json").write_text(json.dumps({"editability": "fail", "comparison": None}))
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    doc = json.loads((v / "iterations.json").read_text())
    first = doc["iterations"][0]
    assert first["similarity"] is None and first["editability"] == "fail"
    assert doc["best_similarity"] == pytest.approx(0.91)


def test_gate_is_stored_in_metadata_and_iterations(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner(), gate=GateVerdict(verdict="FAIL", message="below floor"))
    assert json.loads((v / "metadata.json").read_text())["gate"] == {"verdict": "FAIL", "message": "below floor"}
    assert json.loads((v / "iterations.json").read_text())["gate"]["verdict"] == "FAIL"


# ------------------------------------------------------------------ generate_report rendering
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report"


def _pass_cmp() -> dict:
    return json.loads((FIXTURES / "comparison_pass_sample_instrument.json").read_text())


def test_gate_fail_renders_visible_fail_without_success_wording() -> None:
    import generate_report as gr
    html = gr.generate_charts_html(_pass_cmp(), {"verdict": "FAIL", "message": "section 0.87 < floor 0.90"})
    assert "eval gate: FAIL" in html and "section 0.87 &lt; floor 0.90" in html
    assert "eval gate: PASS" not in html
    assert not any(w in html.lower() for w in ("success", "passed", "ok</"))


def test_gate_pass_renders_pass_and_absent_gate_renders_nothing() -> None:
    import generate_report as gr
    assert "eval gate: PASS" in gr.generate_charts_html(_pass_cmp(), {"verdict": "PASS", "message": "ok"})
    assert "eval gate" not in gr.generate_charts_html(_pass_cmp())
    replay = json.loads((FIXTURES / "comparison_fail_replay.json").read_text())
    assert "eval gate: FAIL" in gr.generate_charts_html(replay, {"verdict": "FAIL", "message": ""})


def test_full_report_reads_gate_from_metadata(tmp_path: Path) -> None:
    import generate_report as gr
    from test_report_editability import _EMPTY_WAV
    cache = tmp_path / "yt_fixture"
    v = cache / "v001"
    v.mkdir(parents=True)
    for stem in ("melodic", "drums", "bass", "vocals"):
        (cache / f"{stem}.wav").write_bytes(_EMPTY_WAV)
    for name in ("render", "render_melodic", "render_drums", "render_bass"):
        (v / f"{name}.wav").write_bytes(_EMPTY_WAV)
    (v / "comparison.json").write_text((FIXTURES / "comparison_pass_sample_instrument.json").read_text())
    (v / "metadata.json").write_text(json.dumps({"gate": {"verdict": "FAIL", "message": "below floor"}}))
    (v / "output.strudel").write_text("// BPM: 136\n$: note('c3').s('sawtooth')\n")
    out = tmp_path / "r.html"
    gr.generate_report(str(cache), str(v), str(out))
    assert "eval gate: FAIL" in out.read_text()


def _iters(**extra: object) -> dict:
    rows = [{"version": 1, "label": "iter 1", "similarity": None, "editability": "fail", "was_best": False,
             "phase": "calibration"},
            {"version": 2, "label": "iter 2", "similarity": 0.91, "was_best": True, "phase": "calibration"}]
    return {"iterations": rows, "best_similarity": 0.91, **extra}


def test_iteration_rows_are_labelled_iter_n_and_rejected_row_has_no_percentage(tmp_path: Path) -> None:
    import generate_report as gr
    data = gr.resolve_iteration_paths(_iters(matched_promoted=True), tmp_path)
    html = gr.generate_iteration_progression_html(data)
    assert "iter 1" in html and "iter 2" in html and ">v1<" not in html and ">v2<" not in html
    assert "rejected (replay)" in html and "REPLAY / UNVERIFIED" in html
    assert "calibration history (promoted render not among" not in html
    assert "0.0%" not in html


def test_lineage_mismatch_is_labelled_in_the_report(tmp_path: Path) -> None:
    import generate_report as gr
    data = gr.resolve_iteration_paths(_iters(matched_promoted=False), tmp_path)
    assert "calibration history (promoted render not among these iterations)" in gr.generate_iteration_progression_html(data)


def test_go_path_llm_iterations_keep_their_v_labels(tmp_path: Path) -> None:
    import generate_report as gr
    data = {"iterations": [{"version": 3, "similarity": 0.7, "phase": "deterministic"}], "best_similarity": 0.7}
    assert ">v3<" in gr.generate_iteration_progression_html(gr.resolve_iteration_paths(data, tmp_path))


def test_stale_mp3_twin_is_not_embedded(tmp_path: Path) -> None:
    import os
    import generate_report as gr
    (tmp_path / "x.wav").write_bytes(b"RIFF")
    (tmp_path / "x.mp3").write_bytes(b"ID3")
    os.utime(tmp_path / "x.wav", (2_000_000_000, 2_000_000_000))      # wav newer than its twin
    assert gr.encode_audio_base64(str(tmp_path / "x.wav")).startswith("data:audio/wav;")


def test_gate_and_iterations_are_on_disk_before_generate_report_runs(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    seen: dict[str, object] = {}

    class Checking(FakeRunner):
        def __call__(self, argv: Sequence[object]) -> None:
            if str(argv[1]).endswith("generate_report.py"):
                seen["gate"] = json.loads((v / "metadata.json").read_text()).get("gate")
                seen["iterations"] = (v / "iterations.json").is_file()
                seen["report_declared"] = "report" in json.loads((v / "metadata.json").read_text())
            super().__call__(argv)

    build_report(t, v, seconds=1, calib_workdir=w, run=Checking(), gate=GateVerdict(verdict="FAIL", message="m"))
    assert seen == {"gate": {"verdict": "FAIL", "message": "m"}, "iterations": True, "report_declared": False}
    assert json.loads((v / "metadata.json").read_text())["report"] == "report.html"


def test_new_gate_on_an_already_reported_version_rebuilds_the_report(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    build_report(t, v, seconds=1, calib_workdir=w, run=FakeRunner())
    again = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=again, gate=GateVerdict(verdict="PASS", message="ok"))
    assert again.tools() == ["generate_report.py"]


# ---- render stems from the renderer (default) vs Demucs (opt-in) ----

def _audible(src: str) -> list[str]:
    """Channel last lines that are NOT silenced with a trailing .gain(0) before their label comment."""
    return [ln for ln in src.splitlines() if re.search(r"\)  // (texture — )?[A-Z]+ — ", ln) and ".gain(0)  //" not in ln]


def test_default_makes_no_separate_call_and_renders_one_voice_group_per_stem(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, calib_workdir=w, run=run, iteration_stems=False)
    assert not [c for c in run.calls if c[1].endswith("separate.py")], "Demucs is opt-in"
    assert set(run.solo_sources) == {"render_melodic.wav", "render_drums.wav", "render_bass.wav"}
    heard = {f: [ln.split("  // ")[-1].split(" — ")[-2] for ln in _audible(src)] for f, src in run.solo_sources.items()}
    assert heard["render_bass.wav"] == ["BASS"] and heard["render_drums.wav"] == ["DRUMS"]
    assert sorted(heard["render_melodic.wav"]) == ["CHORDS", "LEAD"]
    data, _ = sf.read(str(v / "render_vocals.wav"))
    assert not np.any(data) and len(data) == SR, "vocals = silence of the render's length"
    meta = json.loads((v / "metadata.json").read_text())
    assert meta["render_stems_check"]["source"] == "solo" and "correlation" in meta["render_stems_check"]


def test_muted_group_keeps_its_orbit_so_the_duck_target_exists() -> None:
    """Regression: commenting the BASS channel out made the drums' `.duckorbit(2)` fail the render."""
    import solo_stems
    drums_only = solo_stems.solo_code(STRUDEL, "drums")
    assert drums_only is not None
    bass_line = next(ln for ln in drums_only.splitlines() if "// BASS" in ln)
    assert not bass_line.startswith("//") and ".orbit(2).gain(0)  // BASS" in bass_line
    bass_only = solo_stems.solo_code(STRUDEL, "bass")
    assert bass_only is not None and ".duckorbit(2).duckdepth(0.5).gain(0)  // DRUMS" in bass_only, \
        "the muted kick still fires, so it still ducks the bass"


def test_solo_splitter_silences_other_groups_and_rejects_unlabelled_channels() -> None:
    import solo_stems
    only_bass = solo_stems.solo_code(STRUDEL, "bass")
    assert only_bass is not None and "let bass" in only_bass
    assert only_bass.count(".gain(0)") == 3          # lead, the chords stack, drums
    assert 'e4' in only_bass, "muted channels stay in the code (orbits, ducks)"
    multi = [ln for ln in only_bass.splitlines() if ln.endswith("// texture — CHORDS — harmony")]
    assert multi == [")" + ".gain(0)" + "  // texture — CHORDS — harmony"]
    assert solo_stems.solo_code("$: s('bd')  // DRUMS — x\n", "bass") is None, "no channel for the group -> None"
    with pytest.raises(solo_stems.StemSplitError, match="no role label"):
        solo_stems.solo_code("$: note('c2')\n", "bass")


def test_demucs_opt_in_still_separates(track: tuple[Path, Path, Path]) -> None:
    t, v, w = track
    run = FakeRunner()
    build_report(t, v, seconds=1, run=run, render_stems="demucs")
    assert [c[c.index("--prefix") + 1] for c in run.calls if c[1].endswith("separate.py")] == ["render"]
    assert not run.solo_sources


def test_cli_render_stems_flag_defaults_to_solo() -> None:
    import pipeline_helpers
    assert pipeline_helpers.DEFAULT_RENDER_STEMS == "solo"
