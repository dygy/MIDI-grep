"""autopilot.py: the orchestrator works on the failing tracks, tries remedies on copies, adopts only score gains.

Regression (owner, 2026-11-08): passing references were re-rendered again and again while failing tracks (Creagina,
BibiBabydoll...) sat untouched, and every number waited for a human to read it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autopilot as AP  # noqa: E402


def _track(root: Path, name: str) -> Path:
    t = root / name
    (t / "v001").mkdir(parents=True)
    for f in AP.LINKED:
        (t / f).write_bytes(b"x")
    (t / "metadata.json").write_text(json.dumps({"bpm": 112.5, "bpm_confidence": 0.045}))
    (t / "v001" / "metadata.json").write_text(json.dumps({"bpm": 112.5}))
    return t


@pytest.fixture
def env(tmp_path, monkeypatch):
    stems = tmp_path / "stems"
    stems.mkdir()
    monkeypatch.setattr(AP, "STEMS", stems)
    monkeypatch.setattr(AP, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(AP, "HISTORY", tmp_path / "logs" / "remedies.jsonl")
    monkeypatch.setattr(AP, "LOCK", tmp_path / "logs" / "autopilot.pid")
    monkeypatch.setattr(AP, "references", lambda: {"Ref"})
    monkeypatch.setattr(AP, "healthy", lambda fix=True: (True, {}))
    monkeypatch.setattr(AP, "noise", lambda: 0.005)
    return stems


def _status(name, deficit, tempo=False):
    return AP.Status(name, deficit, tempo, {"sample-instrument": {"overall": 0.7}})


def test_the_queue_is_failing_non_reference_tracks_worst_first(env, monkeypatch):
    for n in ("Ref", "Good", "Bad", "Worse", f"Bad{AP.TRIAL_TAG}150"):
        _track(env, n)
    table = {"Ref": _status("Ref", 0.2), "Good": _status("Good", -0.05), "Bad": _status("Bad", 0.03),
             "Worse": _status("Worse", 0.10), "Tempo": _status("Tempo", -0.2, True)}
    _track(env, "Tempo")
    monkeypatch.setattr(AP, "status", lambda t: table.get(t.name, _status(t.name, 0.5)))
    assert [s.track for s in AP.queue()] == ["Worse", "Bad", "Tempo"]   # no reference, no trial copy, no passing track


def test_a_tempo_trial_that_scores_better_is_adopted_on_the_real_track(env, monkeypatch):
    _track(env, "Bibi")
    calls = []

    def fake_run(title, *extra, log):
        calls.append((title, extra))
        if AP.TRIAL_TAG in title:
            assert (env / title / "drums.wav").is_symlink() and "--local" in extra
            return {"runs": [{"similarity_overall": 0.85}, {"similarity_overall": 0.83}]}
        return {"runs": [{"similarity_overall": 0.86 if "--bpm" in extra else 0.74}, {"similarity_overall": 0.72}]}
    monkeypatch.setattr(AP, "run_pipeline", fake_run)
    monkeypatch.setattr(AP, "status", lambda t: _status(t.name, 0.1, True))
    monkeypatch.setattr(AP, "tempo_candidates", lambda t: [
        {"bpm": 150.01, "ratio": 1.3333, "confidence": 0.368, "stamp_confidence": 0.045}])
    rep = AP.run(["Bibi"], with_refs=False, max_trials=2)
    d = rep["tracks"][0]["decisions"][0]
    assert d["remedy"] == "tempo" and d["adopted"] and d["trial_mean"] == pytest.approx(0.84)
    assert calls[-1] == ("Bibi", ("--bpm", "150.01"))                  # adopted: the real track re-run at that tempo
    assert not any(AP.TRIAL_TAG in p.name for p in env.iterdir())     # the trial copy is gone
    hist = [json.loads(l) for l in AP.HISTORY.read_text().splitlines()]
    assert hist[0]["adopted"] and hist[0]["candidate"]["key"] == 1.3333


def test_a_trial_inside_the_noise_is_rejected_and_nothing_is_re_run(env, monkeypatch):
    _track(env, "Regime")
    calls = []

    def fake_run(title, *extra, log):
        calls.append(title)
        return {"runs": [{"similarity_overall": 0.912 if AP.TRIAL_TAG in title else 0.910}]}
    monkeypatch.setattr(AP, "run_pipeline", fake_run)
    monkeypatch.setattr(AP, "status", lambda t: _status(t.name, 0.1, True))
    monkeypatch.setattr(AP, "tempo_candidates", lambda t: [
        {"bpm": 168.78, "ratio": 1.25, "confidence": 0.459, "stamp_confidence": 0.106}])
    rep = AP.run(["Regime"], with_refs=False, max_trials=2)
    assert not rep["tracks"][0]["decisions"][0]["adopted"]
    assert calls == ["Regime", f"Regime{AP.TRIAL_TAG}168.78-BPM-x1.25"]


def test_past_winners_are_tried_first(env):
    AP.remember("X", {"remedy": "tempo", "candidate": {"key": 0.75}, "adopted": True})
    cands = [{"key": 1.3333, "label": "a"}, {"key": 0.75, "label": "b"}]
    assert [c["key"] for c in AP.order_candidates("tempo", cands)] == [0.75, 1.3333]


def test_one_tracks_crash_does_not_stop_the_batch(env, monkeypatch):
    _track(env, "A")
    _track(env, "B")

    def fake_run(title, *extra, log):
        if title == "A":
            raise RuntimeError("boom")
        return {"runs": [{"similarity_overall": 0.8}]}
    monkeypatch.setattr(AP, "run_pipeline", fake_run)
    monkeypatch.setattr(AP, "status", lambda t: _status(t.name, 0.1))
    rep = AP.run(["A", "B"], with_refs=False, max_trials=1)
    assert "boom" in rep["tracks"][0]["error"] and rep["tracks"][1]["run"]["ok"]


def test_a_broken_environment_stops_the_batch_with_the_reason(env, monkeypatch):
    _track(env, "A")
    monkeypatch.setattr(AP, "run_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(AP, "status", lambda t: _status(t.name, 0.1))
    states = iter([(True, {}), (False, {"venv": ["FAIL", "missing"]})])
    monkeypatch.setattr(AP, "healthy", lambda fix=True: next(states))
    with pytest.raises(SystemExit, match="not repairable"):
        AP.run(["A"], with_refs=False, max_trials=1)
    assert not AP.LOCK.exists()                                         # the lock is released on the way out


def test_a_second_autopilot_is_refused(env):
    import os
    AP.LOGS.mkdir(parents=True, exist_ok=True)
    AP.LOCK.write_text(str(os.getppid()))                              # another live process
    with pytest.raises(SystemExit, match="another autopilot"):
        AP.acquire_lock()


def test_a_stale_lock_and_its_snapshot_are_cleaned_up(env, monkeypatch):
    AP.LOGS.mkdir(parents=True, exist_ok=True)
    AP.LOCK.write_text("999999999")                                   # a killed batch's pid
    assert AP.lock_holder() is None
    dropped = []
    monkeypatch.setattr(AP, "ROOT", env.parent / "MIDI-grep")
    stale = env.parent / "MIDI-grep-autopilot-20260101-000000"
    stale.mkdir()
    monkeypatch.setattr(AP, "drop_snapshot", lambda w: dropped.append(w))
    assert AP.drop_stale_snapshots() == [stale] and dropped == [stale]
    AP.acquire_lock()                                                  # the stale lock does not block a new batch
