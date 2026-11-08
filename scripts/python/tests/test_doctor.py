"""doctor.py: a broken environment is named (and repaired) before a batch, never discovered by failing every track.

Regression (2026-11-08): the venv, ~/.cache/puppeteer and bin/ were deleted by hand while a batch ran; the venv loss
failed the batch track by track, and a "node ok" check alone would have let every render fail on the missing Chromium;
without bin/clickhouse the calibration memory silently returned nothing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doctor as DR  # noqa: E402


def _cp(rc: int, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, out, err)


def test_a_crashing_check_is_a_fail_with_its_reason(monkeypatch):
    def boom(_fix):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(DR, "CHECKS", {"venv": boom, "ollama": lambda f: ("warn", "down")})
    res = DR.examine()
    assert res["venv"] == ("FAIL", "RuntimeError: probe exploded")
    assert not DR.runnable(res)
    assert DR.runnable({"ollama": ("warn", "down"), "venv": ("fixed", "reinstalled")})   # warn / fixed still run


def test_missing_renderer_browser_is_installed_on_fix(monkeypatch):
    probes = iter([_cp(1, err="Chromium for the renderer is not installed"), _cp(0, out="/path/chrome")])
    ran = []

    def fake_run(cmd, cwd=None, timeout=None):
        if "renderer.py" in " ".join(map(str, cmd)):
            return next(probes)
        ran.append(cmd)
        return _cp(0)
    monkeypatch.setattr(DR, "_run", fake_run)
    status, detail = DR.check_browser(fix=True)
    assert status == "fixed" and ran == [["npx", "puppeteer", "browsers", "install", "chrome"]]


def test_missing_browser_without_fix_fails(monkeypatch):
    monkeypatch.setattr(DR, "_run", lambda *a, **k: _cp(1, err="Chromium for the renderer is not installed"))
    assert DR.check_browser(fix=False)[0] == "FAIL"


def test_missing_clickhouse_fails_loudly_instead_of_cold_starting(monkeypatch, tmp_path):
    monkeypatch.setattr(DR, "ROOT", tmp_path)
    status, detail = DR.check_clickhouse(fix=False)
    assert status == "FAIL" and "cold starts" in detail
