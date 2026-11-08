#!/usr/bin/env python3
"""The orchestrator's autopilot: pick the failing tracks, run them, read the diagnosis, apply the remedies it knows,
keep a remedy only when the score says so, and check the references once at the end.

Why (owner, 2026-11-08): the orchestrator was "too dumb, absolutely unreliable" - it re-rendered passing references
while the failing tracks sat untouched, reported overall scores while a sub-metric screamed (BibiBabydoll's tempo), and
waited for a human to read every number. This loop makes those decisions itself and writes down why.

One batch (``autopilot.py run``) - one at a time (``autopilot.pid``); the environment doctor (``doctor.py --fix``) runs
before the batch and before every track: a broken environment is repaired and the track retried once, or the batch
stops with the reason instead of failing track after track:

1. QUEUE - every track under ``.cache/stems`` that is not a reference (``eval/datasets/reference_tracks.yaml``), whose
   newest piece in some mode fails its gate floor (``eval/gate.py``) or carries a diagnosis RED tempo flag; the largest
   deficit first. ``--tracks`` overrides the selection.
2. RUN - ``editable-pipeline.sh --title <track>`` (stage 12 diagnoses the promoted pieces).
3. REMEDY - for each red flag with a known remedy, a TRIAL on a throw-away copy of the track (stems symlinked, local
   hosting), scored like any run; the remedy is adopted on the real track only when the trial's mean overall over the
   modes beats the run's by more than ``calibration_noise``:
   * ``tempo`` (the stamp disagrees with the measured tempo, or its phase-coherence confidence is below every
     reference's): candidates are the stamp at the dotted ratios a funk groove locks a beat tracker onto (x4/3, x3/4,
     x3/2, x2/3), each refined by phase coherence; those more coherent than the stamp are tried, most coherent first.
     Coherence alone does NOT decide - on Regime CLT (scored 0.91 at 135 BPM) the x5/4 grid is more coherent (0.459 vs
     0.106) - the trial's score does.
4. REGRESSION - the references run once (``--no-refs`` skips); a reference whose newest overall falls below its
   previous newest by more than ``calibration_noise`` is reported as a regression.
5. REPORT - ``.cache/runlogs/autopilot/<time>.json`` (every decision with its numbers) and a ``.md`` table beside it.

``--snapshot`` (use it for any batch you do not babysit): the batch runs from a frozen git worktree of the CURRENT code
(uncommitted changes included) beside the repo; its data - ``.cache``, ``models``, ``bin``, the venv, the Node build,
``eval/datasets`` and ``eval/thresholds.yaml`` - are symlinks into this checkout, so results land here while edits to
the code here cannot change a batch half-way (2026-11-08: a 10-hour batch would otherwise freeze all development).
The worktree is removed when the batch ends.

Usage:
  autopilot.py queue                       what a batch would work on, worst first (no runs)
  autopilot.py health                      every track's state + the systemic problems across them (no runs)
  autopilot.py run [--tracks T ...] [--no-refs] [--max-trials N] [--snapshot]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "eval"))
STEMS = ROOT / ".cache" / "stems"
LOGS = ROOT / ".cache" / "runlogs" / "autopilot"
PIPELINE = ROOT / "scripts" / "editable-pipeline.sh"
DOTTED_RATIOS = (4 / 3, 3 / 4, 3 / 2, 2 / 3)   # the subdivisions a 3+3+2 groove makes a beat tracker lock onto
TRIAL_TAG = " ~trial "                        # trial copies: "<track> ~trial <bpm>" (never queued, always removed)
HISTORY = LOGS / "remedies.jsonl"            # every trial ever made: the next batch tries past winners first
LOCK = LOGS / "autopilot.pid"
LINKED = ("original.wav", "bass.wav", "drums.wav", "melodic.wav", "vocals.wav", "instrumental.wav")


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def noise() -> float:
    import diagnose
    return diagnose._noise()


def references() -> set[str]:
    import cache_budget
    return {k for k, _ in cache_budget.pinned()}


def newest_per_mode(track: Path) -> dict[str, Path]:
    import cache_budget
    out: dict[str, Path] = {}
    for v in cache_budget._versions(track):
        if v.mode != "unknown":
            out[v.mode] = v.path          # _versions is sorted by number: the last one wins
    return out


@dataclass
class Status:
    track: str
    deficit: float                         # max over modes of (floor - score), overall and section-aware; > 0 = failing
    tempo_red: bool
    modes: dict[str, dict] = field(default_factory=dict)


def status(track: Path) -> Status:
    import diagnose
    import gate
    thr = gate.load_thresholds()
    genre = (_load(track / "metadata.json") or {}).get("genre_ref")
    st = Status(track.name, deficit=float("-inf"), tempo_red=False)
    for mode, v in newest_per_mode(track).items():
        try:
            d = diagnose.diagnose(v, track)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            st.modes[mode] = {"version": v.name, "error": str(exc)}
            continue
        floor = gate.resolve_floor(genre, thr, mode)[0]
        sa_floor = gate.resolve_section_aware_floor(genre, thr, mode)[0]
        gaps = [floor - (d["overall"] or 0.0)] + ([sa_floor - (d["section_aware"] or 0.0)] if sa_floor else [])
        t = d["tempo"]
        red = (not t["stamp_matches_original"]) or bool(t["confidence_below_references"])
        st.deficit = max(st.deficit, max(gaps))
        st.tempo_red = st.tempo_red or red
        st.modes[mode] = {"version": v.name, "overall": d["overall"], "section_aware": d["section_aware"],
                          "floor": floor, "section_aware_floor": sa_floor, "red_count": d["red_count"],
                          "flags": d["flags"][:5],
                          "red_metrics": [{"metric": r["metric"], "cost": r["cost"], "value": r["value"],
                                           "ref_min": r["ref_min"]} for r in d["red_metrics"]]}
    return st


def queue(only: list[str] | None = None) -> list[Status]:
    refs = references()
    out = []
    for t in sorted(p for p in STEMS.iterdir() if (p / "metadata.json").is_file()) if STEMS.is_dir() else []:
        if TRIAL_TAG in t.name or (only is None and t.name in refs) or (only is not None and t.name not in only):
            continue
        s = status(t)
        if only is not None or s.deficit > 0 or s.tempo_red:
            out.append(s)
    return sorted(out, key=lambda s: -s.deficit)


def run_pipeline(title: str, *extra: str, log: Path) -> dict | None:
    """One editable-pipeline run; its JSON summary (stdout) or None when it failed (log has stderr)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as err:
        p = subprocess.run([str(PIPELINE), "--title", title, *extra], stdout=subprocess.PIPE, stderr=err, text=True)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None


def mean_overall(summary: dict | None) -> float | None:
    runs = (summary or {}).get("runs") or []
    vals = [r["similarity_overall"] for r in runs if isinstance(r.get("similarity_overall"), (int, float))]
    return sum(vals) / len(vals) if vals else None


def tempo_candidates(track: Path) -> list[dict]:
    """Dotted-ratio tempi more phase-coherent than the stamp, most coherent first."""
    import librosa
    import tempo_refine as TR
    meta = _load(track / "metadata.json") or {}
    stamp = float(meta["bpm"])
    base = float(meta.get("bpm_confidence") or 0.0)
    y, sr = librosa.load(str(track / "drums.wav"), sr=22050, mono=True)
    seen, out = set(), []
    for r in DOTTED_RATIOS:
        res = TR.refine_bpm(y, sr, stamp * r)
        key = round(res["bpm"], 1)
        if res["confidence"] > base and key not in seen:
            seen.add(key)
            out.append({"bpm": round(res["bpm"], 3), "ratio": round(r, 4), "confidence": round(res["confidence"], 4),
                        "stamp_confidence": base})
    return sorted(out, key=lambda c: -c["confidence"])


def make_trial(track: Path, label: str) -> Path:
    import re
    tag = re.sub(r"[^0-9A-Za-z.]+", "-", label).strip("-")[:24]
    trial = track.parent / f"{track.name}{TRIAL_TAG}{tag}"
    if trial.exists():
        shutil.rmtree(trial)
    trial.mkdir()
    for f in LINKED:
        if (track / f).exists():
            os.symlink(track / f, trial / f)
    shutil.copy2(track / "metadata.json", trial / "metadata.json")
    newest = sorted(track.glob("v[0-9][0-9][0-9]"))
    if newest and (newest[-1] / "metadata.json").exists():      # the stamp needs one vNNN/metadata.json
        (trial / "v000").mkdir()
        shutil.copy2(newest[-1] / "metadata.json", trial / "v000" / "metadata.json")
    return trial


def decide(baseline: float | None, trial: float | None, eps: float) -> bool:
    return baseline is not None and trial is not None and trial > baseline + eps


def past_wins(remedy: str) -> dict:
    """candidate key -> how often a trial of ``remedy`` with that key was adopted (remedies.jsonl)."""
    wins: dict = {}
    try:
        for line in HISTORY.read_text().splitlines():
            r = json.loads(line)
            if r.get("remedy") == remedy and r.get("adopted"):
                k = r["candidate"]["key"]
                wins[k] = wins.get(k, 0) + 1
    except (OSError, ValueError, KeyError):
        pass
    return wins


def remember(track: str, decision: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "track": track, **decision},
                           ensure_ascii=False) + "\n")


def order_candidates(remedy: str, cands: list[dict]) -> list[dict]:
    """Past winners first (by key), then the remedy's own order."""
    wins = past_wins(remedy)
    return sorted(cands, key=lambda c: -wins.get(c["key"], 0))


def acquire_lock() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    pid = lock_holder()
    if pid is not None and pid != os.getpid():
        raise SystemExit(f"another autopilot (pid {pid}) is running - {LOCK}")
    LOCK.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if int(LOCK.read_text().split()[0]) == os.getpid():
            LOCK.unlink()
    except (OSError, ValueError, IndexError):
        pass


def healthy(fix: bool = True) -> tuple[bool, dict]:
    import doctor
    res = doctor.examine(fix=fix)
    return doctor.runnable(res), {k: list(v) for k, v in res.items()}


def run(only: list[str] | None, with_refs: bool, max_trials: int) -> dict:
    acquire_lock()
    try:
        return _run(only, with_refs, max_trials)
    finally:
        release_lock()


def run_checked(title: str, *extra: str, log: Path, report: dict) -> dict | None:
    """run_pipeline; on failure the doctor checks (and repairs) the environment and the run is retried once when the
    environment was the cause. Raises SystemExit when the environment cannot be repaired."""
    summary = run_pipeline(title, *extra, log=log)
    if summary is not None:
        return summary
    ok, res = healthy(fix=True)
    report.setdefault("doctor", []).append({"after_failure_of": title, "runnable": ok, "checks": res})
    if not ok:
        raise SystemExit(f"environment broken and not repairable after {title} failed: {res}")
    if any(v[0] == "fixed" for v in res.values()):
        return run_pipeline(title, *extra, log=log.with_suffix(".retry.log"))
    return None


class Remedy:
    """A fix the autopilot can TRY for a diagnosis: ``applies`` reads the track's status, ``candidates`` lists trial
    variants as editable-pipeline arguments (+ a ``key`` the history counts wins by). The trial runs on a copy; only a
    score gain adopts it."""
    name = "remedy"

    def applies(self, st: Status) -> bool:
        raise NotImplementedError

    def candidates(self, track: Path) -> list[dict]:
        raise NotImplementedError


class TempoRemedy(Remedy):
    name = "tempo"

    def applies(self, st: Status) -> bool:
        return st.tempo_red

    def candidates(self, track: Path) -> list[dict]:
        return [{"key": c["ratio"], "label": f"{c['bpm']} BPM (x{c['ratio']})", "args": ["--bpm", str(c["bpm"])],
                 "detail": c} for c in tempo_candidates(track)]


REMEDIES: list[Remedy] = [TempoRemedy()]


def work_track(s: Status, entry: dict, out_dir: Path, report: dict, eps: float, max_trials: int, save) -> None:
    track = STEMS / s.track
    print(f"[autopilot] {s.track}: deficit {s.deficit:+.4f}{' tempo RED' if s.tempo_red else ''} - running",
          file=sys.stderr)
    summary = run_checked(s.track, log=out_dir / f"{s.track}.log", report=report)
    entry["run"] = {"ok": summary is not None, "mean_overall": mean_overall(summary),
                    "runs": [{k: r.get(k) for k in ("mode", "similarity_overall", "similarity_section_aware",
                                                     "gate_verdict", "diagnosis")} for r in (summary or {}).get("runs", [])]}
    save()
    if summary is None:
        entry["decisions"].append({"remedy": None, "why": f"pipeline failed - see {out_dir / (s.track + '.log')}"})
        return
    best = mean_overall(summary)
    for rem in REMEDIES:
        after = status(track)
        if not rem.applies(after):
            continue
        for c in order_candidates(rem.name, rem.candidates(track))[:max_trials]:
            trial = make_trial(track, c["label"])
            try:
                ts = run_checked(trial.name, *c["args"], "--local", log=out_dir / f"{s.track} trial {c['label']}.log",
                                 report=report)
            finally:
                shutil.rmtree(trial, ignore_errors=True)
            tm = mean_overall(ts)
            won = decide(best, tm, eps)
            decision = {"remedy": rem.name, "candidate": {k: c[k] for k in ("key", "label", "args")},
                        "detail": c.get("detail"), "baseline_mean": best, "trial_mean": tm, "adopted": won,
                        "trial_runs": [{k: r.get(k) for k in ("mode", "similarity_overall", "similarity_section_aware")}
                                       for r in (ts or {}).get("runs", [])]}
            entry["decisions"].append(decision)
            remember(s.track, decision)
            save()
            print(f"[autopilot] {s.track}: {rem.name} trial {c['label']}: mean {tm} vs {best} -> "
                  f"{'ADOPT' if won else 'reject'}", file=sys.stderr)
            if won:
                final = run_checked(s.track, *c["args"], log=out_dir / f"{s.track} adopt {rem.name}.log", report=report)
                entry.setdefault("adopted_runs", []).append({"remedy": rem.name, "ok": final is not None,
                                                             "mean_overall": mean_overall(final)})
                best = mean_overall(final) if final is not None else best
                save()
                break
    entry["after"] = asdict(status(track))


def _run(only: list[str] | None, with_refs: bool, max_trials: int) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = LOGS / stamp
    eps = noise()
    report: dict = {"started": stamp, "noise": eps, "tracks": [], "references": []}

    def save() -> None:
        (LOGS / f"{stamp}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    LOGS.mkdir(parents=True, exist_ok=True)
    ok, res = healthy(fix=True)
    report["doctor"] = [{"before_batch": True, "runnable": ok, "checks": res}]
    save()
    if not ok:
        raise SystemExit(f"environment not runnable: {res}")
    q = queue(only)
    report["queue"] = [asdict(s) for s in q]
    save()
    for s in q:
        entry: dict = {"track": s.track, "before": asdict(s), "decisions": []}
        report["tracks"].append(entry)
        try:
            work_track(s, entry, out_dir, report, eps, max_trials, save)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - one track's crash is recorded; the batch goes on
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[autopilot] {s.track}: ERROR {entry['error']} - next track", file=sys.stderr)
        save()
    if with_refs:
        for name in sorted(references()):
            track = STEMS / name
            if not (track / "metadata.json").is_file():
                continue
            before = {m: (_load(v / "comparison.json") or {}).get("comparison", {}).get("overall_similarity")
                      for m, v in newest_per_mode(track).items()}
            summary = run_checked(name, log=out_dir / f"ref {name}.log", report=report)
            now = {r["mode"]: r.get("similarity_overall") for r in (summary or {}).get("runs", [])}
            regress = {m: (before.get(m), now.get(m)) for m in now
                       if before.get(m) is not None and now.get(m) is not None and now[m] < before[m] - eps}
            report["references"].append({"track": name, "before": before, "now": now, "ok": summary is not None,
                                         "regressions": regress})
            save()
    report["finished"] = time.strftime("%Y%m%d-%H%M%S")
    save()
    (LOGS / f"{stamp}.md").write_text(markdown(report))
    return report


def systemic(rep: dict) -> list[dict]:
    """RED metrics across the batch's tracks (state after the batch, else before), by total overall-score cost then by
    how many tracks share them: what to fix next for everyone, not track by track."""
    agg: dict[tuple[str, str], dict] = {}
    for e in rep["tracks"]:
        st = e.get("after") or e["before"]
        for mode, v in st.get("modes", {}).items():
            for r in v.get("red_metrics", []):
                a = agg.setdefault((mode, r["metric"]), {"mode": mode, "metric": r["metric"], "cost": 0.0, "tracks": []})
                a["cost"] += r["cost"]
                a["tracks"].append(e["track"])
    return sorted(agg.values(), key=lambda a: (-a["cost"], -len(a["tracks"])))


def markdown(rep: dict) -> str:
    """The batch report, in Russian (the owner reads this repo's reports in Russian)."""
    yes, no = "ПРИНЯТО", "отклонено"
    lines = [f"# Пачка автопилота {rep['started']}", "",
             "| трек | до (режим: overall) | после | решения |", "|---|---|---|---|"]
    for e in rep["tracks"]:
        b = ", ".join(f"{m}: {v.get('overall', 0):.3f}" for m, v in e["before"]["modes"].items() if "overall" in v)
        a = ", ".join(f"{m}: {v.get('overall', 0):.3f}" for m, v in (e.get("after") or {}).get("modes", {}).items()
                      if "overall" in v)
        d = "; ".join(f"{x['remedy']} {x.get('candidate', {}).get('label', '')}: {yes if x.get('adopted') else no}"
                      + (f" ({x['trial_mean']:.3f} против {x['baseline_mean']:.3f})"
                         if x.get("trial_mean") is not None and x.get("baseline_mean") is not None else "")
                      if x.get("remedy") else x["why"] for x in e["decisions"])
        if e.get("error"):
            d = (d + "; " if d else "") + f"ОШИБКА: {e['error']}"
        lines.append(f"| {e['track']} | {b} | {a or '-'} | {d or '-'} |")
    sysm = systemic(rep)
    if sysm:
        lines += ["", "## Системные проблемы (красные метрики по всей пачке)", "",
                  "| режим | метрика | суммарная цена | треков | треки |", "|---|---|---|---|---|"]
        for a in sysm[:12]:
            lines.append(f"| {a['mode']} | {a['metric']} | {a['cost']:.3f} | {len(a['tracks'])} | "
                         f"{', '.join(t[:24] for t in a['tracks'])} |")
    if rep["references"]:
        lines += ["", "## Эталоны (регрессия)", "", "| эталон | было | стало | регрессии |", "|---|---|---|---|"]
        for r in rep["references"]:
            fmt = lambda d: ", ".join(f"{m}: {v:.4f}" for m, v in d.items() if isinstance(v, (int, float))) or "-"  # noqa: E731
            lines.append(f"| {r['track']} | {fmt(r['before'])} | {fmt(r['now'])} | {r['regressions'] or 'нет'} |")
    return "\n".join(lines) + "\n"


SNAPSHOT_LINKS = (".cache", "models", "bin", "scripts/python/.venv", "scripts/node/node_modules", "scripts/node/dist",
                  "eval/datasets", "eval/thresholds.yaml")


def make_snapshot(stamp: str) -> Path:
    """A detached worktree at HEAD + this checkout's uncommitted changes; data paths symlinked back here."""
    w = ROOT.parent / f"{ROOT.name}-autopilot-{stamp}"
    git = ["git", "-C", str(ROOT)]
    subprocess.run([*git, "worktree", "add", "--detach", str(w), "HEAD"], check=True, capture_output=True)
    diff = subprocess.run([*git, "diff", "HEAD", "--binary"], check=True, capture_output=True).stdout
    if diff:
        subprocess.run(["git", "-C", str(w), "apply", "--whitespace=nowarn"], input=diff, check=True,
                       capture_output=True)
    others = subprocess.run([*git, "ls-files", "--others", "--exclude-standard", "-z"], check=True,
                            capture_output=True).stdout.decode().split("\0")
    for rel in filter(None, others):
        if any(rel == l or rel.startswith(l + "/") for l in SNAPSHOT_LINKS):
            continue
        dst = w / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for rel in SNAPSHOT_LINKS:
        src, dst = ROOT / rel, w / rel
        if not src.exists():
            continue
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        elif dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)
    return w


def drop_snapshot(w: Path) -> None:
    subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(w)], capture_output=True)
    shutil.rmtree(w, ignore_errors=True)
    subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"], capture_output=True)


def lock_holder() -> int | None:
    """The live autopilot's pid, None when no autopilot runs (a dead pid in the lock is stale)."""
    try:
        pid = int(LOCK.read_text().split()[0])
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError, IndexError):
        return None


def drop_stale_snapshots() -> list[Path]:
    """Snapshots a killed batch left behind (its finally never ran) - only when no autopilot is running."""
    if lock_holder() is not None:
        return []
    gone = []
    for w in ROOT.parent.glob(f"{ROOT.name}-autopilot-*"):
        drop_snapshot(w)
        gone.append(w)
    return gone


def run_in_snapshot(argv: list[str]) -> int:
    import signal
    for w in drop_stale_snapshots():
        print(f"[autopilot] removed a stale snapshot {w}", file=sys.stderr)
    w = make_snapshot(time.strftime("%Y%m%d-%H%M%S"))
    print(f"[autopilot] frozen code in {w}", file=sys.stderr)
    child = subprocess.Popen([str(w / "scripts" / "python" / ".venv" / "bin" / "python"),
                              str(w / "scripts" / "python" / "autopilot.py"), *argv], start_new_session=True)

    def stop(signum, _frame):   # TERM / INT: end the whole batch tree (pipelines, renders), then clean up
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except OSError:
            pass
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        try:
            return child.wait()
        except BaseException:
            stop(None, None)
            child.wait()
            raise
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
        drop_snapshot(w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("queue")
    sub.add_parser("health")
    r = sub.add_parser("run")
    r.add_argument("--tracks", nargs="+")
    r.add_argument("--no-refs", action="store_true")
    r.add_argument("--max-trials", type=int, default=2)
    r.add_argument("--snapshot", action="store_true", help="run from a frozen worktree of the current code")
    raw = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(raw)
    if a.cmd == "run" and a.snapshot:
        return run_in_snapshot([x for x in raw if x != "--snapshot"])
    if a.cmd == "health":
        sts = [status(t) for t in sorted(p for p in STEMS.iterdir() if (p / "metadata.json").is_file()
                                          and TRIAL_TAG not in p.name)]
        rep = {"started": time.strftime("%Y%m%d-%H%M%S"), "references": [],
               "tracks": [{"track": st.track, "decisions": [], "before": asdict(st)} for st in sts]}
        md = markdown(rep).replace("# Пачка автопилота", "# Здоровье треков")
        print(md)
        return 0
    if a.cmd == "queue":
        for s in queue():
            print(f"{s.deficit:+.4f} {'TEMPO ' if s.tempo_red else '      '}{s.track}  "
                  + " ".join(f"{m} {v.get('overall', float('nan')):.3f}" for m, v in s.modes.items()))
        return 0
    rep = run(a.tracks, not a.no_refs, a.max_trials)
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
