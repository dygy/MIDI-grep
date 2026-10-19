"""upload_wrangler retries transient per-object failures, fails fast on the rest."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upload_r2  # noqa: E402

TRANSIENT = "A fetch request failed, likely due to a connectivity issue"
AUTH = "Authentication error [code: 10000]"


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    (tmp_path / "strudel.json").write_text("{}")
    return tmp_path


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch):
    """Patch subprocess.run with scripted (returncode, stderr) results; no real sleeping."""
    calls: list[list[str]] = []
    script: list[tuple[int, str]] = []

    def fake(cmd, **kw):
        calls.append(cmd)
        rc, err = script.pop(0) if script else (0, "")
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

    monkeypatch.setattr(upload_r2.subprocess, "run", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(upload_r2, "have", lambda c: True)
    return calls, script


def test_transient_then_success_completes(pack, run):
    calls, script = run
    script += [(1, TRANSIENT), (0, "")]
    upload_r2.upload_wrangler(pack, "b", "p")
    assert len(calls) == 2


def test_auth_failure_exits_without_retry(pack, run):
    calls, script = run
    script += [(1, AUTH)]
    with pytest.raises(SystemExit, match="wrangler upload failed"):
        upload_r2.upload_wrangler(pack, "b", "p")
    assert len(calls) == 1


def test_persistent_transient_exits_after_bound(pack, run):
    calls, script = run
    script += [(1, TRANSIENT)] * 50
    with pytest.raises(SystemExit, match="wrangler upload failed"):
        upload_r2.upload_wrangler(pack, "b", "p")
    assert len(calls) > 1, "never retried"
    assert len(calls) == upload_r2.PUT_MAX_ATTEMPTS
