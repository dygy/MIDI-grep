#!/usr/bin/env python3
"""ClickHouse as the calibration memory of ``scripts/auto-calibrate.sh``.

Every calibration step, every run BEST and every promoted version is a row of ``midi_grep.calibration_steps``
(``clickhouse_store.py`` migration 2). The next run reads it back:

  warm-start   cal1's knobs (+ per-bar env correction) from
               (a) this track + mode's best recorded row (current code version first, else the newest recorded one),
               (b) the nearest recorded track of the same genre and mode (distance below),
               (c) nothing -> the caller's cold defaults.
               Budget: ``calibrate_loop.iteration_cap`` (3 for a/b, MAX_ITERS otherwise).
  responses    per-knob damping exponent = 1 / median observed elasticity d ln(band-share ratio) / d ln(knob),
               from recorded parent -> child step pairs; knobs with < MIN_PAIRS pairs are omitted (sqrt damping stays).
  record-*     writers. A failed write prints a loud WARNING and exits 0: memory must never break a run.
  backfill     import iters.tsv workdirs and promoted vNNN versions found under ``.cache/stems``.

Neighbour distance (no tuned weights, three components each normalised by its theoretical range, averaged):
  bpm   |ln(b1/b2)| folded to the nearest octave (range ln(sqrt 2))
  key   circle-of-fifths distance of the relative-major tonics (range 6); unparsable -> component dropped
  bands L1 distance of the 6 band-share vectors of the ORIGINAL (range 2)
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import re
import time
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import calibrate_loop as CL  # noqa: E402
import clickhouse_store as CS  # noqa: E402

TABLE = "midi_grep.calibration_steps"
KNOBS = ("bass_mult", "sub_gain", "cal_lead", "lead_lpf", "hat_gain", "master_gain", "bass_lpf", "lead_hpf", "sub_up")
SHAPES = CL.SHAPES              # generate_dynamic_strudel.py --bass-shape; the first is the generator default (wins ties)
DEFAULT_SHAPE = SHAPES[0]
AUX_KINDS = ("shape", "leadfit")   # extra trial renders: they answer a question, they never seed a knob start
BANDS = ("sub_bass", "bass", "low_mid", "mid", "high_mid", "high")
MIN_PAIRS = 3                  # pairs a knob needs before its learned exponent replaces the sqrt damping
MIN_LOG_STEP = 0.05            # same threshold calibrate_dynamic.measured_exponent uses: smaller moves are noise
STRENGTH_BOUNDS = (0.25, 0.75)  # safety clip on a learned exponent (damp() clamps each step again)
EPS = 1e-4
_FIFTHS = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7,
           "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def warn(msg: str) -> None:
    print(f"[memory] WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------- identity of the code that produced a step
def _local_imports(path: Path, seen: set[Path], src: Path) -> None:
    if path in seen or not path.is_file():
        return
    seen.add(path)
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return
    for n in ast.walk(tree):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
        for nm in names:
            _local_imports(src / (nm.split(".")[0] + ".py"), seen, src)


# what decides the sample pack's audio (stage 4) - and so, through it, every render calibrated on that pack
PACK_ROOTS = ("build_sample_pack.py",)
# what cuts the stem instruments (stage 5: the granular trainer, its note grains, the manifest)
INSTRUMENT_ROOTS = ("rave/trainer.py", "pitched_grains.py", "build_instruments.py")
STAGE_ROOTS = {"pack": PACK_ROOTS, "instruments": INSTRUMENT_ROOTS}
CALIBRATE_ROOTS = ("generate_dynamic_strudel.py", "calibrate_dynamic.py", "calibrate_loop.py", "mix_dynamics.py",
                   "compare_audio.py") + PACK_ROOTS + INSTRUMENT_ROOTS
# the offline renderer every calibrated render is made by (its sources and their local imports), and the engine it runs:
# a renderer change (2026-10-28: deterministic ducks/reverb moved scores by ~0.003) must not reuse renders it did not make
RENDER_ROOT = "render-strudel-offline.ts"
_TS_IMPORT = re.compile(r"""from\s+['"]\./([\w-]+)\.js['"]""")


def _renderer_files(node: Path) -> set[Path]:
    """The offline renderer's TypeScript sources (local imports followed) and the engine it loads: the bundle named by
    MIDIGREP_STRUDEL_BUNDLE when set, else the npm lock file that pins @strudel/web."""
    seen: set[Path] = set()
    todo = [node / "src" / RENDER_ROOT]
    while todo:
        p = todo.pop()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        todo += [node / "src" / f"{m}.ts" for m in _TS_IMPORT.findall(p.read_text())]
    bundle = (os.environ.get("MIDIGREP_STRUDEL_BUNDLE") or "").strip()
    engine = Path(bundle) if bundle else node / "package-lock.json"
    if engine.is_file():
        seen.add(engine)
    return seen


def code_version(stage: str = "calibrate") -> tuple[str, str]:
    """(content hash of everything that decides a stage's output, git short commit). "calibrate": a calibrated
    render - generator, calibrator, the loop driver, the mix stage, the metric, the pack and instrument builders (the
    one-shots and note grains they cut are what the render plays) - and their local imports; "pack" / "instruments":
    that stage's builders and their local imports."""
    # the scripts as they were invoked (not symlink-resolved): a checkout hashes its own files
    src = Path(os.path.abspath(__file__)).parent
    seen: set[Path] = set()
    for f in STAGE_ROOTS.get(stage, CALIBRATE_ROOTS):
        _local_imports(src / f, seen, src)
    driver = src.parents[1] / "scripts" / "auto-calibrate.sh"
    if stage not in STAGE_ROOTS and driver.is_file():
        seen.add(driver)
    if stage not in STAGE_ROOTS:
        seen |= _renderer_files(src.parents[1] / "scripts" / "node")
    h = hashlib.sha1()
    for p in sorted(seen):
        h.update(p.name.encode()); h.update(p.read_bytes())
    try:
        g = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        commit = g.stdout.strip() if g.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return h.hexdigest()[:10], commit


# ---------------------------------------------------------------- rows
def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def parse_knobs(text: str) -> dict[str, float]:
    """'BM SG CL LLPF HG MG [BL [LH [SU]]]' -> knob dict (missing tail = 0)."""
    v = text.split()
    if len(v) < 6:
        raise ValueError(f"need >= 6 knob fields, got {text!r}")
    v += ["0"] * (9 - len(v))
    return {k: float(x) for k, x in zip(KNOBS, v[:9])}


def split_shape(text: str) -> tuple[str, str]:
    """'BM ... SU [stem|off]' -> (the numeric knob fields, the bass shape; the generator default when absent)."""
    v = text.split()
    if v and v[-1] in SHAPES:
        return " ".join(v[:-1]), v[-1]
    return text, DEFAULT_SHAPE


def format_knobs(k: dict[str, float]) -> str:
    out = []
    for name in KNOBS:
        x = float(k.get(name, 0))
        out.append(str(int(round(x))) if name in ("lead_lpf", "bass_lpf", "lead_hpf", "sub_up") else f"{x:.4g}")
    return " ".join(out)


def step_key(run_id: str, tag: str, kind: str) -> str:
    return hashlib.sha1(f"{run_id}|{tag}|{kind}".encode()).hexdigest()


def make_row(*, kind: str, run_id: str, tag: str, track_key: str, mode: str, knobs: dict[str, float], ctx: dict,
             cmp_json: dict | None, margin, passed, parent: str = "", alpha: float = 1.0, env: dict | None = None,
             is_best: bool = False, source: str = "live", created_at: str | None = None,
             code: tuple[str, str] | None = None, bass_shape: str = DEFAULT_SHAPE, lead_fit: str = "") -> dict:
    comp = (cmp_json or {}).get("comparison") or {}
    temporal = comp.get("temporal") or {}
    ob = ((cmp_json or {}).get("original") or {}).get("bands") or {}
    rb = ((cmp_json or {}).get("rendered") or {}).get("bands") or {}
    cv, commit = code or code_version()
    row = {
        "step_key": step_key(run_id, tag, kind), "kind": kind, "track_key": track_key,
        "track_hash": ctx.get("track_hash", ""), "title": ctx.get("title", ""), "genre": ctx.get("genre", ""),
        "bpm": _f(ctx.get("bpm")), "key": ctx.get("key", "") or "", "mode": mode, "bars": int(_f(ctx.get("bars"))),
        "code_version": ctx.get("code_version") or cv, "git_commit": commit, "run_id": run_id, "tag": tag,
        "parent_tag": parent if parent not in ("-", None) else "", "alpha": _f(alpha, 1.0),
        "warm_source": ctx.get("warm_source", ""),
        **{k: _f(knobs.get(k)) for k in KNOBS[:-1]}, "sub_up": int(_f(knobs.get("sub_up"))),
        "bass_shape": bass_shape if bass_shape in SHAPES else DEFAULT_SHAPE,
        "lead_fit": lead_fit,
        "env_json": json.dumps(env, separators=(",", ":")) if env else "",
        "env_bars": int(env.get("bars", 0)) if env else 0,
        "overall": _f(comp.get("overall_similarity")), "section_aware": _f(comp.get("section_aware_similarity")),
        "static_similarity": _f(comp.get("static_similarity")), "temporal_similarity": _f(temporal.get("temporal_similarity")),
        "margin": None if margin is None or not math.isfinite(float(margin)) else float(margin),
        "gate_pass": -1 if passed is None else int(bool(passed)),
        "margin_over_chance": _f(comp.get("margin_over_chance")),
        "orig_bands": json.dumps({b: _f(ob.get(b)) for b in BANDS}) if ob else "",
        "rend_bands": json.dumps({b: _f(rb.get(b)) for b in BANDS}) if rb else "",
        "centroid_orig": _f(((cmp_json or {}).get("original") or {}).get("spectral", {}).get("centroid_mean")),
        "centroid_rend": _f(((cmp_json or {}).get("rendered") or {}).get("spectral", {}).get("centroid_mean")),
        "is_run_best": int(is_best), "source": source,
    }
    if created_at:
        row["created_at"] = created_at
    return row


def existing_keys(track_key: str) -> set[str]:
    q = f"SELECT step_key FROM {TABLE} WHERE track_key = '{CS.sanitize_sql_value(track_key)}'"
    return {r["step_key"] for r in CS.clickhouse_query(q)}


def write_rows(rows: list[dict]) -> int:
    """Insert rows whose step_key is new. Returns the number written; raises on a ClickHouse failure."""
    if not rows:
        return 0
    if not CS._ensure_clickhouse():
        raise RuntimeError("ClickHouse unavailable")
    have: set[str] = set()
    for tk in {r["track_key"] for r in rows}:
        have |= existing_keys(tk)
    t0 = time.time_ns()   # write order across writes and processes; seq = position inside this write (see history)
    new = [{**r, "seq": i, "write_ns": t0 + i} for i, r in enumerate(r for r in rows if r["step_key"] not in have)]
    # one statement needs identical columns: created_at is either on every row or on none
    for grp in ([r for r in new if "created_at" in r], [r for r in new if "created_at" not in r]):
        CS.clickhouse_insert_many(TABLE, grp)
    return len(new)


def history(track_key: str | None = None, mode: str | None = None, genre: str | None = None) -> list[dict]:
    cond = []
    if track_key is not None:
        cond.append(f"track_key = '{CS.sanitize_sql_value(track_key)}'")
    if mode:
        cond.append(f"mode = '{CS.sanitize_sql_value(mode)}'")
    if genre:
        cond.append(f"genre = '{CS.sanitize_sql_value(genre)}'")
    q = f"SELECT *, toString(created_at) AS created FROM {TABLE}" + (" WHERE " + " AND ".join(cond) if cond else "") + " ORDER BY created_at, write_ns, seq"
    return CS.clickhouse_query(q)


# ---------------------------------------------------------------- ranking
def rank_best(rows: list[dict], noise: float | None = None) -> dict:
    """The row calibrate_loop would call BEST: gate pass, then margin (ties within noise -> overall, section-aware)."""
    n = CL.noise() if noise is None else noise
    lines = []
    for i, r in enumerate(rows):
        m = r.get("margin")
        gate = "" if m is None else f"{m} {'PASS' if r.get('gate_pass') == 1 else 'FAIL'}"
        lines.append("\t".join([str(i), str(r["overall"]), "n/a" if m is None else str(m), gate, "", "", str(r["section_aware"]), "", "1"]))
    best = CL.select([CL.Row(ln) for ln in lines], n)
    return rows[int(best.tag)]


def pick_version_group(rows: list[dict], current: str) -> list[dict]:
    """Rows of the current code version when any exist, else of the most recently recorded version."""
    cur = [r for r in rows if r["code_version"] == current]
    if cur:
        return cur
    newest = max(rows, key=lambda r: r["created"])["code_version"]
    return [r for r in rows if r["code_version"] == newest]


# ---------------------------------------------------------------- neighbours
def _key_pos(key: str) -> int | None:
    m = re.match(r"\s*([A-Ga-g][#b]?)\s*(major|minor|maj|min|m)?", key or "")
    if not m:
        return None
    pc = _FIFTHS.get(m.group(1).lower())
    if pc is None:
        return None
    if (m.group(2) or "major").lower() in ("minor", "min", "m"):
        pc = (pc + 3) % 12                       # relative major
    return (pc * 7) % 12                         # position on the circle of fifths


def distance(a: dict, b: dict) -> float:
    """Mean of the normalised components both sides can supply (see module doc)."""
    comps = []
    if a.get("bpm") and b.get("bpm"):
        d = min(abs(math.log(a["bpm"] / (b["bpm"] * f))) for f in (0.5, 1.0, 2.0))
        comps.append(d / math.log(math.sqrt(2)))
    ka, kb = _key_pos(a.get("key", "")), _key_pos(b.get("key", ""))
    if ka is not None and kb is not None:
        d = abs(ka - kb)
        comps.append(min(d, 12 - d) / 6.0)
    if a.get("bands") and b.get("bands"):
        comps.append(sum(abs(a["bands"].get(k, 0.0) - b["bands"].get(k, 0.0)) for k in BANDS) / 2.0)
    return sum(comps) / len(comps) if comps else 1.0


def orig_bands_of_audio(path: Path) -> dict[str, float]:
    try:
        import compare_audio as CA
        y = CA.load_audio(str(path))
        return CA.compute_frequency_bands(y) if y is not None else {}
    except Exception as e:      # noqa: BLE001 - the band profile is one of three distance components; drop it, say so
        warn(f"cannot measure the band profile of {path} ({type(e).__name__}); neighbour distance uses bpm + key only")
        return {}


def _profile(rows: list[dict]) -> dict:
    r = next((x for x in reversed(rows) if x.get("orig_bands")), rows[-1])
    return {"bpm": r["bpm"], "key": r["key"], "bands": json.loads(r["orig_bands"]) if r.get("orig_bands") else {}}


def warm_start(track_key: str, mode: str, genre: str, bpm: float, key: str, bars: int, *, orig_bands: dict | None = None,
               stems: Path | None = None, current_code: str | None = None) -> dict:
    """{'source': track|neighbour|cold, 'knobs': 'BM ...'|'', 'env': dict|None, 'detail': str}."""
    cur = current_code or code_version()[0]
    own = history(track_key, mode)
    if own:
        # kind 'shape' rows (the extra render of the OTHER bass shape) never seed a start by themselves: a winning one
        # is also written as a 'best' row. They only say that the shape question was answered for this code version.
        shape_tested = any(r["kind"] == "shape" and r["code_version"] == cur for r in own)
        best = rank_best(pick_version_group([r for r in own if r["kind"] not in AUX_KINDS] or own, cur))
        env = json.loads(best["env_json"]) if best["env_json"] else None
        if env and int(env.get("bars", 0)) != bars:
            env = None
        return {"source": "track", "knobs": format_knobs(best), "env": env,
                "bass_shape": best.get("bass_shape") or DEFAULT_SHAPE, "shape_tested": shape_tested,
                "detail": f"{track_key}/{mode} best {best['kind']} {best['tag']} code {best['code_version']} "
                          f"overall {best['overall']:.4f} margin {best['margin']} ({len(own)} recorded rows)"}
    cands = [r for r in history(None, mode, genre) if r["track_key"] != track_key and r["kind"] not in AUX_KINDS]
    if cands:
        if orig_bands is None and stems is not None and (stems / "instrumental.wav").is_file():
            orig_bands = orig_bands_of_audio(stems / "instrumental.wav")
        me = {"bpm": bpm, "key": key, "bands": orig_bands or {}}
        by_track: dict[str, list[dict]] = {}
        for r in cands:
            by_track.setdefault(r["track_key"], []).append(r)
        near = min(by_track, key=lambda t: (distance(me, _profile(by_track[t])), t))
        best = rank_best(pick_version_group(by_track[near], cur))
        knobs = parse_knobs(format_knobs(best))
        knobs["bass_lpf"] = knobs["lead_hpf"] = knobs["sub_up"] = 0   # note-bound / hypothesis knobs: re-derived, not copied
        return {"source": "neighbour", "knobs": format_knobs(knobs), "env": None,
                "bass_shape": DEFAULT_SHAPE, "shape_tested": False,   # a per-piece decision: re-measured, never copied
                "detail": f"nearest {genre}/{mode} track '{near}' (distance {distance(me, _profile(by_track[near])):.3f}) "
                          f"best {best['tag']} overall {best['overall']:.4f}"}
    return {"source": "cold", "knobs": "", "env": None, "bass_shape": DEFAULT_SHAPE, "shape_tested": False,
            "detail": f"no recorded {genre}/{mode} history"}


# ---------------------------------------------------------------- fitted synth lead (auto-calibrate.sh lead-fit stage)
def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as e:
        warn(f"cannot read {path} ({e})")
        return ""


def _lead_fit_of(row: dict) -> dict | None:
    try:
        d = json.loads(row.get("lead_fit") or "")
    except ValueError:
        return None
    return d if isinstance(d, dict) and d.get("decision") in ("keep", "palette") else None


def lead_warm(track_key: str, mode: str, genre: str, bpm: float, key: str, *, orig_bands: dict | None = None,
              stems: Path | None = None, current_code: str | None = None) -> dict:
    """What the memory knows about the fitted synth lead.
    source: 'track-keep'   this track's fitted vector for the CURRENT code version won -> one verify render of it
            'track-palette' this track already answered 'palette' for the current code version -> nothing to test
            'track'        this track has a vector from an older code version -> warm start, short search
            'neighbour'    nearest same-genre track's KEPT vector -> warm start, short search
            'none'         nothing recorded -> cold search from the measured values
    spec: the vector to start from (None for 'none' / 'track-palette')."""
    cur = current_code or code_version()[0]
    own = [r for r in history(track_key, mode) if r["kind"] == "leadfit" and _lead_fit_of(r)]
    if own:
        same = [r for r in own if r["code_version"] == cur]
        row = max(same or own, key=lambda r: r["created"])
        d = _lead_fit_of(row)
        if same:
            if d["decision"] == "keep":
                return {"source": "track-keep", "spec": d["spec"], "detail": f"{track_key}/{mode} kept vector, code {row['code_version']}"}
            return {"source": "track-palette", "spec": None, "detail": f"{track_key}/{mode} answered palette for code {cur}"}
        kept = [r for r in own if _lead_fit_of(r)["decision"] == "keep"]
        if kept:
            return {"source": "track", "spec": _lead_fit_of(max(kept, key=lambda r: r["created"]))["spec"],
                    "detail": f"{track_key}/{mode} vector from older code {row['code_version']}"}
    cands = [r for r in history(None, mode, genre) if r["track_key"] != track_key and r["kind"] == "leadfit"
             and (_lead_fit_of(r) or {}).get("decision") == "keep"]
    if cands:
        if orig_bands is None and stems is not None and (stems / "instrumental.wav").is_file():
            orig_bands = orig_bands_of_audio(stems / "instrumental.wav")
        me = {"bpm": bpm, "key": key, "bands": orig_bands or {}}
        by_track: dict[str, list[dict]] = {}
        for r in cands:
            by_track.setdefault(r["track_key"], []).append(r)
        near = min(by_track, key=lambda t: (distance(me, _profile(by_track[t])), t))
        row = max(by_track[near], key=lambda r: r["created"])
        return {"source": "neighbour", "spec": _lead_fit_of(row)["spec"],
                "detail": f"nearest {genre}/{mode} track '{near}' kept vector"}
    return {"source": "none", "spec": None, "detail": f"no recorded lead fit for {genre}/{mode}"}


# ---------------------------------------------------------------- learned step sizes
def _ratio(row: dict, bands: tuple[str, ...]) -> float | None:
    try:
        o, r = json.loads(row["orig_bands"]), json.loads(row["rend_bands"])
    except (ValueError, KeyError, TypeError):
        return None
    return (sum(o.get(b, 0.0) for b in bands) + EPS) / (sum(r.get(b, 0.0) for b in bands) + EPS)


def _driving_bands(knob: str, su: int) -> tuple[str, ...]:
    # mirrors calibrate_dynamic.py: with the octave up the two gains swap their driving bands
    return {"sub_gain": ("bass",) if su else ("sub_bass",), "bass_mult": ("sub_bass",) if su else ("bass",),
            "cal_lead": ("low_mid", "mid")}[knob]


def learn_responses(rows: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """({knob: damping exponent}, {knob: pairs used}) from parent -> child steps of the same run."""
    by_run: dict[tuple[str, str], dict] = {(r["run_id"], r["tag"]): r for r in rows if r["kind"] == "step"}
    elast: dict[str, list[float]] = {"sub_gain": [], "bass_mult": [], "cal_lead": []}
    for (run, _), c in by_run.items():
        p = by_run.get((run, c["parent_tag"])) if c["parent_tag"] else None
        if p is None or p["sub_up"] != c["sub_up"]:
            continue
        for knob in elast:
            if p[knob] <= 0 or c[knob] <= 0:
                continue
            lf = math.log(c[knob] / p[knob])
            bands = _driving_bands(knob, int(c["sub_up"]))
            rp, rc = _ratio(p, bands), _ratio(c, bands)
            if abs(lf) < MIN_LOG_STEP or rp is None or rc is None:
                continue
            e = -(math.log(rc) - math.log(rp)) / lf
            if math.isfinite(e) and e > 0.05:
                elast[knob].append(e)
    out, n = {}, {k: len(v) for k, v in elast.items()}
    for knob, es in elast.items():
        if len(es) >= MIN_PAIRS:
            out[knob] = round(min(STRENGTH_BOUNDS[1], max(STRENGTH_BOUNDS[0], 1.0 / statistics.median(es))), 3)
    return out, n


# ---------------------------------------------------------------- workdir / version readers
def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def track_ctx(track_dir: Path) -> dict:
    meta = _read_json(track_dir / "metadata.json") or {}
    h = ""
    orig = track_dir / "original.wav"
    if orig.is_file():
        try:
            h = CS.get_track_hash(str(orig))
        except OSError:
            h = ""
    return {"title": meta.get("title", track_dir.name), "genre": meta.get("genre_ref") or meta.get("genre", ""), "bpm": meta.get("bpm", 0),   # genre_ref: an untrusted guess is no neighbour key
            "key": meta.get("key", ""), "track_hash": h}


def rows_from_workdir(workdir: Path, track_key: str, mode: str, ctx: dict, run_id: str, *, only_tag: str | None = None,
                      source: str = "live", created_at: str | None = None, code_version_override: str | None = None,
                      best_tag: str | None = None, tsv_name: str = "iters.tsv", kind: str = "step",
                      lead_fit: str = "") -> list[dict]:
    tsv = workdir / tsv_name
    rows = []
    cv = code_version()
    if code_version_override:
        ctx = {**ctx, "code_version": code_version_override}
    for ln in tsv.read_text().splitlines():
        f = ln.split("\t") + [""] * 10
        tag, overall, margin, gate, params, envused, sa, parent, alpha, shape = f[:10]
        if only_tag and tag != only_tag:
            continue
        cmp_json = _read_json(workdir / f"{tag}.cmp.json")
        env = _read_json(workdir / f"{tag}.envused.json")
        if env is None and envused not in ("", "-"):
            env = _read_json(Path(envused)) or _read_json(workdir / Path(envused).name)
        try:
            knobs = parse_knobs(params)
        except ValueError:
            continue
        g = gate.split()
        m = None if margin in ("", "n/a") else float(margin)
        passed = None if m is None or len(g) < 2 else g[1] == "PASS"
        if cmp_json is None:
            cmp_json = {"comparison": {"overall_similarity": float(overall), "section_aware_similarity": _f(sa)}}
        rows.append(make_row(kind=kind, run_id=run_id, tag=tag, track_key=track_key, mode=mode, knobs=knobs, ctx=ctx,
                             cmp_json=cmp_json, margin=m, passed=passed, parent=parent, alpha=_f(alpha, 1.0), env=env,
                             is_best=(tag == best_tag), source=source, created_at=created_at, code=cv,
                             bass_shape=shape or DEFAULT_SHAPE, lead_fit=lead_fit))
        if best_tag and tag == best_tag:
            rows.append({**rows[-1], "kind": "best", "step_key": step_key(run_id, tag, "best"), "is_run_best": 1})
    return rows


_GEN_KNOBS = re.compile(r"calibrate_dynamic\.py: ([0-9. ]+?(?: (?:stem|off))?)[;)]")
# editable-pipeline.sh stage 9 writes "final --master-gain X" when the loudness stage moved the gain after the loop
_FINAL_MASTER = re.compile(r"final --master-gain ([0-9.]+)")


def promoted_row(track_dir: Path, version: str, *, source: str = "live", created_at: str | None = None) -> dict | None:
    vdir = track_dir / version
    meta, cmp_json = _read_json(vdir / "metadata.json"), _read_json(vdir / "comparison.json")
    if not meta or not cmp_json or not meta.get("generation_mode"):
        return None
    m = _GEN_KNOBS.search(meta.get("generator", ""))
    if not m:
        return None
    numeric, shape = split_shape(m.group(1))
    try:
        knobs = parse_knobs(numeric)
    except ValueError:
        return None
    fm = _FINAL_MASTER.search(meta.get("generator", ""))
    if fm:       # the row's score is of the piece at the loudness stage's gain, so the knobs must say that gain, not the loop's
        knobs["master_gain"] = float(fm.group(1))
    mode, ctx = meta["generation_mode"], track_ctx(track_dir)
    ctx["bars"] = (_read_json(vdir / "env_correction.json") or {}).get("bars", 0)
    margin = passed = None
    try:
        import gate_margin
        gm = gate_margin.margins(vdir / "comparison.json", ctx["genre"] or None, mode)   # no genre: the default floor, never funk's
        margin, passed = gm["margin"], gm["passed"]
    except Exception as e:      # noqa: BLE001 - the margin is optional metadata
        warn(f"{track_dir.name}/{version}: gate margin unavailable ({e})")
    return make_row(kind="promoted", run_id=f"promoted:{version}", tag=version, track_key=track_dir.name, mode=mode, knobs=knobs,
                    ctx=ctx, cmp_json=cmp_json, margin=margin, passed=passed, env=_read_json(vdir / "env_correction.json"),
                    is_best=True, source=source, created_at=created_at or meta.get("created_at", "")[:19].replace("T", " ") or None,
                    bass_shape=shape)


# Scores before this date are on the OLD scale (time-averaged, full-mix reference, BlackHole pitch shift; CLAUDE.md
# "score was rescaled 2026-10-18", instrumental target + offline renderer 2026-10-19): their 0.93 would outrank every
# honest 0.80 and seed knobs tuned against a metric that could not tell songs apart. Never imported.
SCALE_SINCE = "2026-10-19 00:00:00"


def backfill(cache: Path, since: str = SCALE_SINCE) -> dict[str, int]:
    """Import every iters.tsv workdir and promoted version under <cache>/<track>/ (idempotent), new-scale only."""
    counts = {"workdir_rows": 0, "promoted_rows": 0, "new": 0, "skipped_old_scale": 0}
    cutoff = datetime.fromisoformat(since)
    for track in sorted(p for p in cache.iterdir() if p.is_dir()):
        ctx = track_ctx(track)
        rows: list[dict] = []
        for tsv in sorted(track.glob("**/iters.tsv")):
            wd, first = tsv.parent, tsv.read_text().splitlines()[:1]
            if not first or wd.name not in ("sample-instrument", "synth") or not (wd / "cal1.cmp.json").is_file():
                continue
            stamp = datetime.fromtimestamp(tsv.stat().st_mtime)
            if stamp < cutoff:
                counts["skipped_old_scale"] += 1; continue
            rid = "backfill:" + hashlib.sha1((str(wd.relative_to(cache)) + first[0]).encode()).hexdigest()[:12]
            # best of that run, by the loop's own rule
            rs = [CL.Row(ln) for ln in tsv.read_text().splitlines() if ln.strip()]
            best = CL.select(rs, CL.noise()).tag if rs else None
            r = rows_from_workdir(wd, track.name, wd.name, ctx, rid, source=f"backfill:{wd.relative_to(cache)}",
                                  created_at=stamp.strftime("%Y-%m-%d %H:%M:%S"), best_tag=best,
                                  code_version_override="backfill-" + stamp.strftime("%Y%m%d"))
            counts["workdir_rows"] += len(r); rows += r
        for v in sorted(track.glob("v[0-9][0-9][0-9]")):
            pr = promoted_row(track, v.name, source=f"backfill:{track.name}/{v.name}")
            if pr and datetime.fromisoformat((pr.get("created_at") or "1970-01-01")[:19].replace("T", " ")) < cutoff:
                counts["skipped_old_scale"] += 1; continue
            if pr:
                stamp = pr.get("created_at") or ""
                pr["code_version"] = "backfill-" + (stamp[:10].replace("-", "") or "unknown")
                counts["promoted_rows"] += 1; rows.append(pr)
        counts["new"] += write_rows(rows)
    return counts


# ---------------------------------------------------------------- CLI
def _ctx_from_args(a) -> dict:
    ctx = track_ctx(Path(a.stems)) if a.stems else {}
    ctx.update({"genre": a.genre, "bpm": a.bpm, "key": a.key or ctx.get("key", ""), "bars": a.bars,
                "warm_source": a.warm_source or ""})
    return ctx


def main() -> int:
    # clickhouse_store prints its migration chatter on stdout; stdout of this CLI is machine-read, so everything but the
    # explicit result line goes to stderr
    with contextlib.redirect_stdout(sys.stderr):
        return _main()


def emit(line: str) -> None:
    print(line, file=sys.__stdout__)


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--stems", default="")
        p.add_argument("--mode", required=True)
        p.add_argument("--genre", default="")
        p.add_argument("--bpm", type=float, default=0.0)
        p.add_argument("--key", default="")
        p.add_argument("--bars", type=int, default=0)
        p.add_argument("--warm-source", default="")

    for name in ("record-step", "record-best"):
        p = sp.add_parser(name); common(p)
        p.add_argument("--workdir", required=True); p.add_argument("--run-id", required=True)
        p.add_argument("--tag", required=True, help="the step (record-step) or the run BEST (record-best)")
        p.add_argument("--tsv", default="iters.tsv", help="rows file inside --workdir (shape.tsv for the bass-shape trial)")
        p.add_argument("--kind", default="step", choices=("step", "shape", "leadfit"),
                       help="shape = the other-bass-shape trial render; leadfit = the fitted-synth-lead trial")
        p.add_argument("--lead-fit", default="", help="JSON file (decision + fitted vector) stored in the lead_fit column")
        if name == "record-best":
            p.add_argument("--best-env", default="")
    p = sp.add_parser("record-promoted"); p.add_argument("--track-dir", required=True); p.add_argument("--version", required=True)
    p = sp.add_parser("warm-start"); common(p)
    p.add_argument("--track-key", required=True); p.add_argument("--env-out", default="")
    p.add_argument("--orig-bands-json", default="")
    p = sp.add_parser("lead-warm"); common(p)
    p.add_argument("--track-key", required=True); p.add_argument("--spec-out", required=True)
    p = sp.add_parser("responses"); p.add_argument("--mode", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("backfill"); p.add_argument("--cache", default=str(ROOT / ".cache" / "stems"))
    sp.add_parser("summary")
    p = sp.add_parser("code-version", help="print the code hash a calibrated render (or the sample pack / instrument models) is stamped with")
    p.add_argument("--stage", choices=("calibrate", "pack", "instruments"), default="calibrate")
    a = ap.parse_args()

    if a.cmd == "code-version":
        print(code_version(a.stage)[0], file=sys.__stdout__)   # main() sends stdout to stderr; this is the one machine-read line
        return 0

    if a.cmd in ("record-step", "record-best"):
        try:
            wd = Path(a.workdir); key = Path(a.stems).name
            rows = rows_from_workdir(wd, key, a.mode, _ctx_from_args(a), a.run_id, only_tag=a.tag, tsv_name=a.tsv,
                                     kind=a.kind, best_tag=a.tag if a.cmd == "record-best" else None,
                                     lead_fit=_read_text(Path(a.lead_fit)) if a.lead_fit else "")
            if a.cmd == "record-best":
                rows = [r for r in rows if r["kind"] == "best"]
                env = _read_json(Path(a.best_env)) if a.best_env else None
                if env:
                    for r in rows:
                        r["env_json"], r["env_bars"] = json.dumps(env, separators=(",", ":")), int(env.get("bars", 0))
            print(f"[memory] {a.cmd} {key}/{a.mode} {a.tag}: wrote {write_rows(rows)} row(s)", file=sys.stderr)
        except Exception as e:      # noqa: BLE001 - recording must never break a calibration run
            warn(f"{a.cmd} FAILED, this step is NOT in ClickHouse: {type(e).__name__}: {e}")
        return 0
    if a.cmd == "record-promoted":
        try:
            td = Path(a.track_dir)
            row = promoted_row(td, a.version)
            print(f"[memory] promoted {td.name}/{a.version}: " + (f"wrote {write_rows([row])} row(s)" if row else "no knobs in metadata, skipped"),
                  file=sys.stderr)
        except Exception as e:      # noqa: BLE001
            warn(f"record-promoted FAILED: {type(e).__name__}: {e}")
        return 0
    if a.cmd == "warm-start":
        try:
            ob = json.loads(a.orig_bands_json) if a.orig_bands_json else None
            w = warm_start(a.track_key, a.mode, a.genre, a.bpm, a.key, a.bars, orig_bands=ob, stems=Path(a.stems) if a.stems else None)
        except Exception as e:      # noqa: BLE001 - fall back to a cold start, loudly
            warn(f"warm-start FAILED, starting cold: {type(e).__name__}: {e}")
            w = {"source": "cold", "knobs": "", "env": None, "detail": "memory unavailable"}
        env_path = "-"
        if w["env"] and a.env_out:
            Path(a.env_out).write_text(json.dumps(w["env"], indent=2) + "\n"); env_path = a.env_out
        emit("\t".join([w["source"], str(CL.iteration_cap(w["source"])), w["knobs"] or "-", env_path, w["detail"],
                         w.get("bass_shape", DEFAULT_SHAPE), "1" if w.get("shape_tested") else "0"]))
        return 0
    if a.cmd == "lead-warm":
        try:
            w = lead_warm(a.track_key, a.mode, a.genre, a.bpm, a.key or "", stems=Path(a.stems))
        except Exception as e:      # noqa: BLE001 - the memory must never break a run
            warn(f"lead-warm failed ({type(e).__name__}: {e}); cold search")
            w = {"source": "none", "spec": None, "detail": "memory unavailable"}
        if w["spec"]:
            Path(a.spec_out).write_text(json.dumps(w["spec"], indent=1) + "\n")
        emit("\t".join([w["source"], a.spec_out if w["spec"] else "-", w["detail"]]))
        return 0
    if a.cmd == "responses":
        try:
            resp, n = learn_responses(history(None, a.mode))
            Path(a.out).write_text(json.dumps(resp, indent=2) + "\n")
            print(f"[memory] learned damping exponents {resp} from pairs {n} (need >= {MIN_PAIRS}; others stay sqrt)", file=sys.stderr)
        except Exception as e:      # noqa: BLE001
            warn(f"responses FAILED, fixed sqrt damping stays: {type(e).__name__}: {e}")
            Path(a.out).write_text("{}\n")
        return 0
    if a.cmd == "backfill":
        emit(json.dumps(backfill(Path(a.cache))))
        return 0
    if a.cmd == "summary":
        for r in CS.clickhouse_query(f"SELECT track_key, mode, kind, count() AS n, max(overall) AS best_overall FROM {TABLE} "
                                     "GROUP BY track_key, mode, kind ORDER BY track_key, mode, kind"):
            print(r)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
