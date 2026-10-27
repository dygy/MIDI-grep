# @layer: integration
# @regression
"""Every checkout of the repo (a git worktree included) uses the MAIN checkout's ClickHouse binary and database:
a worktree run once read an empty per-worktree store, warm-started cold and recorded nothing."""
from __future__ import annotations

import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clickhouse_store as CS  # noqa: E402


def _git(*a: str, cwd: Path) -> None:
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def test_a_worktree_resolves_to_the_main_checkout(tmp_path: Path):
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", "--detach", str(wt), cwd=main)
    assert CS.shared_root(wt).resolve() == main.resolve()
    assert CS.shared_root(main).resolve() == main.resolve()


def test_outside_git_the_checkout_itself_is_used(tmp_path: Path):
    assert CS.shared_root(tmp_path) == tmp_path
