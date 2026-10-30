# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""The owner's genre labels are the truth; a new track is classified by kNN on audio embeddings against them.

Owner decision 2026-10-29 (option 1): CLAP zero-shot cannot separate Brazilian funk from "retro wave / Polish electro"
(Enter - Nigdy Wiecej Wojny was labelled funk), so the genre of a track is, in order:
``--genre`` override > eval/datasets/genre_labels.yaml > kNN against the labelled tracks > CLAP (uncertain when the
kNN could run and no genre supports the track).

Synthetic embeddings only: clusters on separate axes of a small space, so the tests are about the DECISION LOGIC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))

MODEL = "synthetic:model|win10s"
DIM = 12


def _cluster(axis, n, seed, noise=0.08):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = np.zeros(DIM); v[axis] = 1.0
        v += rng.normal(0, noise, DIM)
        out.append(v / np.linalg.norm(v))
    return out


def _world(tmp_path, funk=4, retro=0, other=0):
    """A labels registry + cached embeddings: funk on axis 0, retro_wave on axis 1, trance on axis 2."""
    import genre_knn as GK
    labels, root = [], tmp_path / "emb"
    for genre, axis, n in (("brazilian_funk", 0, funk), ("retro_wave", 1, retro), ("trance", 2, other)):
        for i, v in enumerate(_cluster(axis, n, seed=axis * 100 + 1)):
            key = f"{genre}-{i}"
            labels.append({"cache_key": key, "genre": genre, "source": "owner", "note": "synthetic"})
            GK.save_embedding(key, v, MODEL, root=root)
    reg = tmp_path / "genre_labels.yaml"
    reg.write_text(yaml.safe_dump({"labels": labels}))
    return reg, root


def _track(tmp_path, name):
    track = tmp_path / "tracks" / name
    (track / "v001").mkdir(parents=True)
    (track / "metadata.json").write_text(json.dumps({"title": name}))
    (track / "v001" / "metadata.json").write_text(json.dumps({"bpm": 130.0, "key": "C minor", "style": "trance", "version": 1}))
    return track


def _stamp(tmp_path, name, det, reg, root, **kw):
    from pipeline_helpers import stamp_track_metadata
    track = _track(tmp_path, name)
    f = tmp_path / f"{name}.genre.json"
    f.write_text(json.dumps(det))
    return stamp_track_metadata(track, detector_json=f, labels_path=reg, embeddings_root=root,
                                stems_root=tmp_path / "stems", **kw)


def _det(vec, genre="house", uncertain=False, candidates=None):
    return {"detected_genre": genre, "confidence": 0.9, "uncertain": uncertain, "candidates": candidates or [genre],
            "audio_embedding": [float(v) for v in vec], "embedding_model": MODEL}


def test_owner_label_beats_the_detector_and_the_knn(tmp_path):
    # labelled retro_wave, but CLAP is sure it is house and the embedding sits right on the funk cluster
    reg, root = _world(tmp_path, funk=4)
    doc = yaml.safe_load(reg.read_text())
    doc["labels"].append({"cache_key": "Enter", "genre": "retro_wave", "source": "owner", "note": "owner 2026-10-29"})
    reg.write_text(yaml.safe_dump(doc))
    m = _stamp(tmp_path, "Enter", _det(_cluster(0, 1, 99)[0]), reg, root)
    assert (m["genre"], m["genre_basis"], m["genre_ref"], m["genre_uncertain"]) == ("retro_wave", "label", "retro_wave", False)
    assert "owner" in m["genre_label_source"] and "genre_knn" not in m      # the label decides: no kNN is consulted


def test_override_beats_the_label(tmp_path):
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "brazilian_funk-0", _det(_cluster(0, 1, 5)[0]), reg, root, genre_override="trance")
    assert (m["genre"], m["genre_basis"], m["genre_ref"]) == ("trance", "override", "trance")


def test_knn_assigns_the_supported_genre_over_a_contrary_uncertain_clap(tmp_path):
    reg, root = _world(tmp_path, funk=4, retro=3)
    x = _cluster(0, 1, 77)[0]                                   # sounds like the funk cluster
    m = _stamp(tmp_path, "New", _det(x, genre="lofi", uncertain=True, candidates=["lofi", "house"]), reg, root)
    assert (m["genre"], m["genre_basis"], m["genre_ref"], m["genre_uncertain"]) == ("brazilian_funk", "knn", "brazilian_funk", False)
    assert m["genre_knn"]["supported"] == ["brazilian_funk"] and m["genre_knn"]["scores"]["brazilian_funk"] > m["genre_knn"]["scores"]["retro_wave"]


def test_knn_picks_the_best_supported_of_several_genres(tmp_path):
    import genre_knn as GK
    reg, root = _world(tmp_path, funk=4, retro=4)
    labelled, _ = GK.labelled_embeddings(GK.load_labels(reg), MODEL, root=root)
    centre = lambda axis: np.eye(DIM)[axis]                     # the cluster centre is never the least typical point
    assert GK.classify(centre(1), labelled)["genre"] == "retro_wave"
    assert GK.classify(centre(0), labelled)["genre"] == "brazilian_funk"


def test_knn_returns_unknown_when_no_genre_supports_the_track(tmp_path):
    import genre_knn as GK
    reg, root = _world(tmp_path, funk=4, retro=4)
    labelled, _ = GK.labelled_embeddings(GK.load_labels(reg), MODEL, root=root)
    res = GK.classify(_cluster(5, 1, 3)[0], labelled)           # a third, unlabelled region of the space
    assert res["genre"] is None and res["supported"] == []
    assert any("below its least typical member" in r for r in res["reasons"])


def test_a_single_member_genre_gives_unknown_even_for_an_identical_track(tmp_path):
    import genre_knn as GK
    reg, root = _world(tmp_path, funk=3, retro=1)
    labelled, _ = GK.labelled_embeddings(GK.load_labels(reg), MODEL, root=root)
    only_retro = labelled["retro_wave-0"][1]
    res = GK.classify(only_retro, labelled)
    assert res["genre"] is None, res
    assert res["per_genre"]["retro_wave"]["supported"] is False and "no within-genre spread" in res["per_genre"]["retro_wave"]["why"]


def test_leave_one_out_removes_the_track_from_its_own_references(tmp_path):
    import genre_knn as GK
    reg, root = _world(tmp_path, funk=4, retro=1)
    labelled, _ = GK.labelled_embeddings(GK.load_labels(reg), MODEL, root=root)
    res = GK.classify(labelled["retro_wave-0"][1], labelled, exclude_key="retro_wave-0")
    assert "retro_wave" not in res["per_genre"] and res["genre"] is None     # nothing left for retro_wave


def test_unlabelled_track_near_no_genre_falls_back_to_clap_as_uncertain(tmp_path):
    reg, root = _world(tmp_path, funk=4, retro=3)
    far = _cluster(7, 1, 4)[0]
    m = _stamp(tmp_path, "Far", _det(far, genre="house", uncertain=False), reg, root)     # CLAP is "certain"
    assert m["genre"] == "house"                                                          # kept as the palette guess
    assert m["genre_uncertain"] is True and m["genre_ref"] == "unreferenced" and m["genre_basis"] == "detector-uncertain"
    assert any(r.startswith("kNN:") for r in m["genre_uncertain_reasons"])
    assert m["genre_knn"]["status"] == "ok" and m["genre_knn"]["genre"] is None


def test_without_an_embedding_the_knn_is_unavailable_and_clap_decides_as_before(tmp_path):
    reg, root = _world(tmp_path, funk=4)
    det = {"detected_genre": "house", "confidence": 0.9, "uncertain": False, "candidates": ["house"]}   # no embedding
    m = _stamp(tmp_path, "Old", det, reg, root)
    assert m["genre_knn"]["status"] == "unavailable" and m["genre"] == "house" and m["genre_basis"] == "detector"


def test_labels_registry_rejects_a_contradicting_duplicate(tmp_path):
    import genre_knn as GK
    reg = tmp_path / "l.yaml"
    reg.write_text(yaml.safe_dump({"labels": [{"cache_key": "A", "genre": "x"}, {"cache_key": "A", "genre": "y"}]}))
    with pytest.raises(ValueError):
        GK.load_labels(reg)


def test_the_shipped_registry_labels_the_five_owner_tracks():
    import genre_knn as GK
    labels = GK.load_labels()
    funk = {k for k, v in labels.items() if v["genre"] == "brazilian_funk"}
    assert funk == {"VAGABUNDO NÃO NAMORA", "Regime CLT (Dj Brunin XM, Aurora Shukita)", "Dumb Blond Bitches",
                    "Automotivo XM, Taka Taka Taka Tá (Dj Brunin XM"}
    enter = labels["Enter - Nigdy Więcej Wojny (prod. Enter) (2022)"]
    assert enter["genre"] == "retro_wave" and "Polish electro" in enter["note"]


def test_a_labelled_genre_without_reference_floors_is_not_gated_by_borrowed_funk_floors(tmp_path):
    """End to end for the owner's Enter case: label -> genre_ref retro_wave -> gate says 'no measured floor', the
    control finds no same-genre reference, and the palette is synthwave's."""
    import pipeline_helpers as PH
    from eval.gate import evaluate_comparison_data, load_thresholds
    from sound_selector import GENRE_PALETTES, palette_sounds
    reg = tmp_path / "labels.yaml"
    reg.write_text(yaml.safe_dump({"labels": [{"cache_key": "Enter", "genre": "retro_wave", "source": "owner", "note": "n"}]}))
    m = _stamp(tmp_path, "Enter", {"detected_genre": "brazilian_funk", "confidence": 0.9, "uncertain": False,
                                   "candidates": ["brazilian_funk"]}, reg, tmp_path / "emb")
    assert m["genre"] == "retro_wave" and m["genre_ref"] == "retro_wave"
    th = load_thresholds()
    comp = {"generation_mode": "sample-instrument",
            "comparison": {"overall_similarity": 0.80, "static_similarity": 0.80, "worst_band_diff": 5.0}}
    res = evaluate_comparison_data(comp, m["genre_ref"], th, "sample-instrument")
    assert res.reference_floor is False and res.floor_source == "genres" and res.floor == 0.55   # not funk's floor
    assert "no measured floor for retro_wave" in res.message
    assert PH.control_track_path(m["genre_ref"], "Enter") is None                                 # no same-genre reference
    assert palette_sounds(m["genre"], "bass") == GENRE_PALETTES["synthwave"]["bass"]
