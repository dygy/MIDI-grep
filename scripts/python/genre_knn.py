#!/usr/bin/env python3
"""Genre of a track from the OWNER'S LABELS, and for an unlabelled track from its audio embedding's nearest labelled tracks.

Why: a zero-shot CLAP prompt cannot tell "Brazilian funk" from "electro / retro wave with vocals" (measured, see
context/spec/003-editable-strudel-generation). The owner's labels are the truth; a new track is classified by how
close it is to the labelled tracks of each genre IN COMPARISON WITH how close those tracks are to each other.

Registry:   ``eval/datasets/genre_labels.yaml``  (cache_key -> genre, owner-provided; separate from the gate dataset).
Embedding:  the CLAP audio embedding of ``original.wav``, mean over all analysis windows (``detect_genre_dl.mean_embedding``).
            Cached per embedding-space id (model + checkpoint + windowing) under ``.cache/genre-embeddings/<id-hash>/``.

kNN rule (no constant is tuned to a track):

* score of x for genre G  =  mean cosine of x to the labelled members of G.
* G's own spread          =  the same score of every member against the OTHER members (leave-one-out, k values).
* x is SUPPORTED by G iff G has >= 2 labelled members and score(x) >= min(LOO scores of G): x is no more atypical of
  G than G's least typical member. This is a conformal-style test at level 1/(k+1): x would have to be the single
  lowest of the k+1 tracks to be rejected, so the level follows from k and nothing else. (The members' scores use
  k-1 neighbours, x's uses k; with the few members of a hand-labelled set that is a small, acknowledged mismatch.)
* the best-supported genre = the supported G with the most head-room, score(x) - min(LOO of G).
* nothing supported, or a genre with < 2 members (spread not estimable)  ->  ``genre = None`` ("unknown").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "eval" / "datasets" / "genre_labels.yaml"
STEMS_ROOT = REPO_ROOT / ".cache" / "stems"
MIN_MEMBERS = 2   # a within-genre spread needs at least one pair of labelled tracks


def embeddings_root() -> Path:
    return Path(os.environ.get("MIDIGREP_GENRE_EMB_DIR") or REPO_ROOT / ".cache" / "genre-embeddings")


# --- the label registry -------------------------------------------------------------------------------------------

def load_labels(path: Path | str | None = None) -> dict[str, dict]:
    """{cache_key: {genre, source, note}} from the owner's registry (empty when the file is absent)."""
    import yaml
    p = Path(path) if path else LABELS_PATH
    if not p.is_file():
        return {}
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for row in doc.get("labels") or []:
        key, genre = row.get("cache_key"), row.get("genre")
        if not key or not genre:
            raise ValueError(f"{p}: every label needs cache_key and genre, got {row!r}")
        if key in out and out[key]["genre"] != genre:
            raise ValueError(f"{p}: {key!r} is labelled both {out[key]['genre']!r} and {genre!r}")
        out[key] = {"genre": genre, "source": row.get("source", ""), "note": row.get("note", "")}
    return out


# --- the embedding cache ------------------------------------------------------------------------------------------

def _space_dir(model_id: str, root: Path | None = None) -> Path:
    return (Path(root) if root else embeddings_root()) / hashlib.sha1(model_id.encode()).hexdigest()[:12]


def save_embedding(key: str, vec, model_id: str, audio_path: Path | None = None, root: Path | None = None) -> Path:
    d = _space_dir(model_id, root)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"cache_key": key, "model": model_id, "audio_size": audio_path.stat().st_size if audio_path and audio_path.exists() else None}
    f = d / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({**meta, "embedding": [float(v) for v in vec]}))
    tmp.replace(f)
    return f


def load_embedding(key: str, model_id: str, audio_path: Path | None = None, root: Path | None = None):
    """The cached embedding of ``key`` in this embedding space, or None (absent, or ``audio_path`` changed size)."""
    f = _space_dir(model_id, root) / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    if d.get("cache_key") != key or d.get("model") != model_id:
        return None
    if audio_path is not None and audio_path.exists() and d.get("audio_size") not in (None, audio_path.stat().st_size):
        return None
    return np.asarray(d["embedding"], dtype=np.float64)


def labelled_embeddings(labels: dict[str, dict], model_id: str, root: Path | None = None,
                        stems_root: Path | None = None, exclude: str | None = None) -> tuple[dict[str, tuple[str, np.ndarray]], list[str]]:
    """({cache_key: (genre, unit embedding)} for the labelled tracks with a cached embedding, [keys without one])."""
    stems = Path(stems_root) if stems_root else STEMS_ROOT
    have, missing = {}, []
    for key, row in labels.items():
        if key == exclude:
            continue
        v = load_embedding(key, model_id, stems / key / "original.wav", root)
        if v is None:
            missing.append(key)
        else:
            have[key] = (row["genre"], v / np.linalg.norm(v))
    return have, missing


# --- the classifier -----------------------------------------------------------------------------------------------

def classify(x, labelled: dict[str, tuple[str, np.ndarray]], exclude_key: str | None = None) -> dict:
    """kNN verdict for embedding ``x`` against ``labelled`` ({key: (genre, embedding)}). Pure numpy.

    Returns {genre (None = unknown), supported: [genres, best first], per_genre: {G: {...}}, reasons: [str]}.
    ``exclude_key`` drops that labelled track from the references (leave-one-out evaluation)."""
    x = np.asarray(x, dtype=np.float64)
    x = x / np.linalg.norm(x)
    by_genre: dict[str, list[np.ndarray]] = {}
    for key, (g, v) in labelled.items():
        if key != exclude_key:
            by_genre.setdefault(g, []).append(np.asarray(v, dtype=np.float64) / np.linalg.norm(v))
    per, reasons = {}, []
    for g, M in sorted(by_genre.items()):
        k = len(M)
        row = {"n_members": k, "score": float(np.mean([x @ m for m in M])), "supported": False}
        if k < MIN_MEMBERS:
            row["why"] = f"{k} labelled member: no within-genre spread to compare with"
            reasons.append(f"{g}: {row['why']}")
        else:
            loo = [float(np.mean([M[i] @ M[j] for j in range(k) if j != i])) for i in range(k)]
            row.update(loo_min=min(loo), loo_mean=float(np.mean(loo)), loo=loo,
                       headroom=row["score"] - min(loo), supported=row["score"] >= min(loo))
            if not row["supported"]:
                reasons.append(f"{g}: score {row['score']:.3f} is below its least typical member's {min(loo):.3f}")
        per[g] = row
    ok = sorted((g for g in per if per[g]["supported"]), key=lambda g: -per[g]["headroom"])
    if not per:
        reasons.append("no labelled tracks to compare with")
    return {"genre": ok[0] if ok else None, "supported": ok, "per_genre": per, "reasons": reasons}


def classify_track(x, model_id: str, labels: dict[str, dict], exclude_key: str | None = None,
                   root: Path | None = None, stems_root: Path | None = None) -> dict:
    """``classify`` against the labelled tracks whose embeddings are cached; ``status`` says when it could not run."""
    have, missing = labelled_embeddings(labels, model_id, root, stems_root, exclude=exclude_key)
    if not have:
        return {"status": "unavailable", "genre": None, "supported": [], "per_genre": {}, "missing": missing,
                "reasons": ["no labelled track has a cached embedding (run genre_knn.py ensure)"]}
    res = classify(x, have)
    res["status"] = "ok"
    res["missing"] = missing
    if missing:
        res["reasons"].append(f"{len(missing)} labelled track(s) have no cached embedding and were not used")
    return res


def ensure_labelled(labels: dict[str, dict], root: Path | None = None, stems_root: Path | None = None) -> list[str]:
    """Embed (CLAP, same windowing as the detector) every labelled track with an original.wav and no cached
    embedding. Returns the keys embedded. Loads the model only when something is missing."""
    stems = Path(stems_root) if stems_root else STEMS_ROOT
    todo = [k for k in labels if (stems / k / "original.wav").is_file()]
    done: list[str] = []
    model = mt = None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import detect_genre_dl as dg
    mid = dg.embedding_model_id(dg.expected_model_type())      # the space id is known without loading the model
    for key in todo:
        audio = stems / key / "original.wav"
        if load_embedding(key, mid, audio, root) is not None:
            continue
        if model is None:
            model, mt = dg.load_clap_model()
            if model is None:
                raise RuntimeError("CLAP unavailable: cannot embed the labelled tracks")
            mid = dg.embedding_model_id(mt)
            if load_embedding(key, mid, audio, root) is not None:
                continue
        vec, mid = dg.track_embedding(str(audio), model, mt)
        save_embedding(key, vec, mid, audio, root)
        done.append(key)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ensure", help="embed the labelled tracks that have no cached embedding")
    e.add_argument("--labels", type=Path)
    loo = sub.add_parser("loo", help="leave-one-out: classify each labelled track from the others (needs cached embeddings)")
    loo.add_argument("--labels", type=Path)
    loo.add_argument("--model-id", required=True)
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    if args.cmd == "ensure":
        print(json.dumps({"embedded": ensure_labelled(labels)}, ensure_ascii=False))
        return 0
    have, missing = labelled_embeddings(labels, args.model_id)
    out = {}
    for key, (g, v) in have.items():
        r = classify(v, have, exclude_key=key)
        out[key] = {"label": g, "knn": r["genre"], "supported": r["supported"],
                    "scores": {gg: round(p["score"], 4) for gg, p in r["per_genre"].items()},
                    "loo_min": {gg: round(p["loo_min"], 4) for gg, p in r["per_genre"].items() if "loo_min" in p},
                    "reasons": r["reasons"]}
    print(json.dumps({"missing": missing, "loo": out}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
