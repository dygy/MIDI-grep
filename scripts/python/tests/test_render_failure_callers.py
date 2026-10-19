# @layer: integration
# @spec: 001-core-pipeline
# @regression
"""Every caller of a Strudel renderer treats a non-zero exit as failure: the file it left behind (the renderer
writes before some of its checks) is never analysed or scored."""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))


def _fake_run_writing_wav(rc: int):
    """subprocess.run stand-in for `node <renderer> in.strudel -o out.wav -d N`: writes the wav, exits rc."""
    def fake(cmd, *a, **k):
        out = cmd[cmd.index("-o") + 1]
        Path(out).write_bytes(b"RIFFxxxx")
        return subprocess.CompletedProcess(cmd, rc, stdout="Saved: " + out, stderr="silent render" if rc else "")
    return fake


# --- python callers ------------------------------------------------------------------------------------
def test_audition_does_not_analyse_a_failed_render(tmp_path, monkeypatch):
    import audition_drum_banks as ab
    rec = tmp_path / "rec.js"
    rec.write_text("//")
    monkeypatch.setattr(ab, "RECORDER", rec)
    monkeypatch.setattr(ab.subprocess, "run", _fake_run_writing_wav(3))
    monkeypatch.setattr(ab, "analyze_stem_timbre", lambda p: {"brightness": 1.0})
    assert ab.audition("RolandTR808", 1) is None
    monkeypatch.setattr(ab.subprocess, "run", _fake_run_writing_wav(0))
    assert ab.audition("RolandTR808", 1) == {"brightness": 1.0}


def test_solo_drums_render_with_nonzero_exit_returns_none_and_deletes_the_file(tmp_path, monkeypatch):
    import renderer
    import stem_match as sm
    rec = tmp_path / "rec.js"
    rec.write_text("//")
    monkeypatch.setattr(renderer, "renderer_script", lambda env=None: rec)
    monkeypatch.setattr(sm, "_extract_drum_voice", lambda p: "$: s('bd')")
    version = tmp_path / "v001"
    version.mkdir()
    (version / "output.strudel").write_text("$: s('bd')")
    monkeypatch.setattr(subprocess, "run", _fake_run_writing_wav(3))
    assert sm._render_solo_drums(tmp_path, version) is None
    assert not (version / "render_drums_solo.wav").exists()
    monkeypatch.setattr(subprocess, "run", _fake_run_writing_wav(0))
    assert sm._render_solo_drums(tmp_path, version) == version / "render_drums_solo.wav"


def _stub_node(tmp_path: Path, body: str) -> Path:
    s = tmp_path / "stub_renderer.sh"
    s.write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n' + body)
    s.chmod(s.stat().st_mode | stat.S_IEXEC)
    return s


def test_editability_test_render_rejects_a_file_from_a_failed_renderer(tmp_path, monkeypatch):
    import editability_test as et
    monkeypatch.setattr(et.time, "sleep", lambda s: None)
    rec = _stub_node(tmp_path, 'echo wav > "$out"; echo "Saved: $out"; exit 3\n')
    wav = tmp_path / "a.wav"
    with pytest.raises(RuntimeError, match="render failed"):
        et.render(tmp_path / "x.strudel", wav, 1.0, recorder=rec, node="/bin/bash")
    assert not wav.exists()
    rec = _stub_node(tmp_path, 'echo wav > "$out"; echo "Saved: $out"\n')
    et.render(tmp_path / "x.strudel", wav, 1.0, recorder=rec, node="/bin/bash")
    assert wav.exists()


# --- auto-calibrate.sh ---------------------------------------------------------------------------------
def test_auto_calibrate_does_not_score_a_render_that_exited_nonzero(tmp_path):
    from test_calibrate_bass_lpf import AUTO, run_driver
    marker = tmp_path / "compared"
    cmp_body = f"open({str(marker)!r}, 'w').write('x')\nimport json\nprint(json.dumps({{'comparison': {{'overall_similarity': 0.8}}}}))\n"
    import test_calibrate_bass_lpf as t
    old_cmp, t.STUB_CMP = t.STUB_CMP, cmp_body
    try:
        (tmp_path / "rec.sh").write_text("")   # run_driver rewrites it; the failing stub is injected below
        t.STUB_REC, saved = '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo wav > "$2"; shift; done\nexit 3\n', t.STUB_REC
        try:
            r = run_driver(tmp_path, AUTO, "--mode", "synth", "--start", "0.447 0.707 0.969 5199 0.0 0.672")
        finally:
            t.STUB_REC = saved
    finally:
        t.STUB_CMP = old_cmp
    assert "RENDER FAILED (rc=3" in r.stdout and r.returncode == 70, r.stdout + r.stderr
    assert not marker.exists(), "compare_audio ran on the file of a failed render"


# --- sample-pipeline.sh render block ------------------------------------------------------------------
def _sample_pipeline_render_block() -> str:
    text = (ROOT / "scripts" / "sample-pipeline.sh").read_text()
    start = text.index('RENDERER="$(')
    end = text.index('echo "==> Done."')
    return text[start:end]


@pytest.mark.parametrize("node_rc,scored", [(3, False), (0, True)])
def test_sample_pipeline_scores_only_after_a_clean_render(tmp_path, node_rc, scored):
    root = tmp_path / "root"
    (root / "scripts" / "python").mkdir(parents=True)
    (root / "scripts" / "node" / "dist").mkdir(parents=True)
    (root / "scripts" / "node" / "node_modules" / "puppeteer").mkdir(parents=True)
    (root / "scripts" / "node" / "dist" / "render-strudel-offline.js").write_text("//")
    (root / "scripts" / "python" / "renderer.py").write_text((SCRIPTS / "renderer.py").read_text())
    (root / "scripts" / "python" / "pipeline_helpers.py").write_text("print('/tmp/ref.wav')\n")
    (root / "scripts" / "python" / "compare_audio.py").write_text(
        "import json;print(json.dumps({'comparison': {'overall_similarity': 0.8}}))\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    node = bindir / "node"
    node.write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo wav > "$2"; shift; done\n'
                    f'echo "Saved: x"; exit {node_rc}\n')
    node.chmod(node.stat().st_mode | stat.S_IEXEC)
    pack = tmp_path / "pack"
    pack.mkdir()
    harness = (f'set -euo pipefail\nROOT={root}\nPY={sys.executable}\nPACK={pack}\nMODE=m\nSTEMS_DIR=/x\n'
               f'OUT=$PACK/output_m.strudel\nREND=$PACK/render_m.wav\n' + _sample_pipeline_render_block() + 'echo REACHED_END\n')
    env = {"PATH": f"{bindir}:/usr/bin:/bin"}
    r = subprocess.run(["/bin/bash", "-c", harness], capture_output=True, text=True, env=env, timeout=60)
    assert ("overall 80.0%" in r.stdout) is scored, r.stdout + r.stderr
    if not scored:
        assert r.returncode != 0 and "render failed (rc=3)" in r.stderr and not (pack / "render_m.wav").exists()
    else:
        assert r.returncode == 0 and "REACHED_END" in r.stdout
