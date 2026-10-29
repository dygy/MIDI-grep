"""The calibrate code stamp covers the offline renderer (2026-10-28: a deterministic-renderer change moved scores, yet a
track whose Python had not changed reused its old calibration: Dumb Blond Bitches kept v008/v009)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibration_memory as M  # noqa: E402


def test_a_renderer_change_changes_the_calibrate_stamp(tmp_path, monkeypatch):
    ts = tmp_path / "render.ts"
    ts.write_text("const duck = 'late';\n")
    monkeypatch.setattr(M, "_renderer_files", lambda node: {ts}, raising=False)
    before = M.code_version("calibrate")[0]
    ts.write_text("const duck = 'on the param timeline';\n")
    assert M.code_version("calibrate")[0] != before, "a renderer edit must invalidate calibrated renders"


def test_the_renderer_files_follow_local_imports_and_name_the_engine(tmp_path, monkeypatch):
    node = tmp_path / "node"
    (node / "src").mkdir(parents=True)
    (node / "src" / M.RENDER_ROOT).write_text("import { a } from './helpers.js';\nimport p from 'puppeteer';\n")
    (node / "src" / "helpers.ts").write_text("import { b } from './deep.js';\n")
    (node / "src" / "deep.ts").write_text("export const b = 1;\n")
    (node / "src" / "unrelated.ts").write_text("x\n")
    (node / "package-lock.json").write_text("{}")
    monkeypatch.delenv("MIDIGREP_STRUDEL_BUNDLE", raising=False)
    assert {p.name for p in M._renderer_files(node)} == {M.RENDER_ROOT, "helpers.ts", "deep.ts", "package-lock.json"}
    bundle = tmp_path / "index.js"
    bundle.write_text("engine")
    monkeypatch.setenv("MIDIGREP_STRUDEL_BUNDLE", str(bundle))
    assert bundle in M._renderer_files(node) and node / "package-lock.json" not in M._renderer_files(node)
