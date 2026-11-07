# @layer: unit
# @spec: 004-one-command-editable-pipeline
# @regression
"""Automatic genre labelling (owner request 2026-11-07: "I should not have to hint anything"): a track is labelled
without the owner when the audio verdict (kNN, or CLAP when the kNN supports no genre) and the verdict from its
published metadata (genre_text.py, local LLM, evidence quoted verbatim) agree. The automatic label then counts like an
owner label; the owner's label still wins.

Synthetic embeddings (the kNN world of test_genre_labels_knn.py) and a fake LLM: the tests are about the decision logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from test_genre_labels_knn import _cluster, _det, _stamp, _track, _world  # noqa: F401 (shared synthetic world)


def _text(tmp_path, name, genre, status="ok", evidence=("funk carioca",)):
    f = tmp_path / f"{name}.text.json"
    f.write_text(json.dumps({"status": status, "genre": genre, "evidence": list(evidence), "model": "fake"}))
    return f


def _auto_rows(reg: Path) -> dict:
    p = reg.with_name("genre_labels_auto.yaml")
    return {r["cache_key"]: r for r in (yaml.safe_load(p.read_text())["labels"] if p.is_file() else [])}


# --- the stamp ------------------------------------------------------------------------------------------------------

def test_knn_and_metadata_agree_writes_an_auto_label_and_trusts_it(tmp_path):
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0], genre="house", uncertain=True), reg, root,
               text_json=_text(tmp_path, "new", "brazilian_funk"))
    assert m["genre"] == "brazilian_funk" and m["genre_basis"] == "auto-label" and m["genre_ref"] == "brazilian_funk"
    assert m["genre_auto"] == {"audio": "brazilian_funk", "audio_source": "kNN", "text": "brazilian_funk", "labelled": True}
    row = _auto_rows(reg)["new"]
    assert row["genre"] == "brazilian_funk" and row["source"] == "auto" and "funk carioca" in row["note"]


def test_disagreement_writes_nothing_and_keeps_the_audio_decision(tmp_path):
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0]), reg, root,
               text_json=_text(tmp_path, "new", "trance", evidence=("trance",)))
    assert m["genre_basis"] == "knn" and m["genre"] == "brazilian_funk"
    assert m["genre_auto"]["labelled"] is False and _auto_rows(reg) == {}


def test_metadata_alone_never_labels(tmp_path):
    # far from every labelled cluster, CLAP unsure: only the text names a genre -> no label, still unreferenced
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "new", _det(_cluster(5, 1, seed=3)[0], genre="brazilian_funk", uncertain=True), reg, root,
               text_json=_text(tmp_path, "new", "brazilian_funk"))
    assert m["genre_auto"]["audio"] is None and _auto_rows(reg) == {}
    assert m["genre_ref"] == "unreferenced"


def test_unavailable_text_is_no_evidence(tmp_path):
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0]), reg, root,
               text_json=_text(tmp_path, "new", None, status="unavailable", evidence=()))
    assert m["genre_basis"] == "knn" and m["genre_auto"]["labelled"] is False and _auto_rows(reg) == {}


def test_a_new_genre_is_bootstrapped_by_confident_clap_plus_metadata(tmp_path):
    # no phonk member exists, so the kNN cannot support it; CLAP clears its margin rule and the metadata agrees
    reg, root = _world(tmp_path, funk=4)
    m = _stamp(tmp_path, "phonk1", _det(_cluster(6, 1, seed=4)[0], genre="brazilian_phonk", uncertain=False), reg, root,
               text_json=_text(tmp_path, "phonk1", "brazilian_phonk", evidence=("phonk",)))
    assert m["genre_basis"] == "auto-label" and m["genre_auto"]["audio_source"] == "CLAP"
    import genre_knn as GK
    assert GK.load_labels(reg)["phonk1"]["genre"] == "brazilian_phonk"


def test_auto_labels_join_the_knn_references(tmp_path):
    import genre_knn as GK
    reg, root = _world(tmp_path, funk=2)
    for i, v in enumerate(_cluster(0, 3, seed=55)):
        GK.save_embedding(f"auto-{i}", v, "synthetic:model|win10s", root=root)
        GK.add_auto_label(f"auto-{i}", "brazilian_funk", "test", owner_path=reg)
    res = GK.classify_track(_cluster(0, 1, seed=9)[0], "synthetic:model|win10s", GK.load_labels(reg), root=root)
    assert res["per_genre"]["brazilian_funk"]["n_members"] == 5


def test_owner_label_beats_an_auto_label(tmp_path):
    import genre_knn as GK
    reg, _ = _world(tmp_path, funk=2)
    GK.add_auto_label("x", "brazilian_funk", "auto", owner_path=reg)
    doc = yaml.safe_load(reg.read_text()); doc["labels"].append({"cache_key": "x", "genre": "retro_wave", "source": "owner"})
    reg.write_text(yaml.safe_dump(doc))
    assert GK.load_labels(reg)["x"]["genre"] == "retro_wave"


def test_a_re_stamp_of_an_auto_labelled_track_is_fresh(tmp_path):
    from pipeline_helpers import label_stale
    reg, root = _world(tmp_path, funk=4)
    _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0]), reg, root, text_json=_text(tmp_path, "new", "brazilian_funk"))
    assert label_stale(tmp_path / "tracks" / "new", reg) is False


def test_an_unlabelled_track_without_a_metadata_verdict_is_stale(tmp_path):
    from pipeline_helpers import label_stale
    reg, root = _world(tmp_path, funk=4)
    _stamp(tmp_path, "old", _det(_cluster(0, 1, seed=7)[0]), reg, root)      # stamped before automatic labelling
    assert label_stale(tmp_path / "tracks" / "old", reg) is True


# --- the metadata classifier ----------------------------------------------------------------------------------------

INFO = {"title": "Automotivo Holiday (Dj Brunin XM, Kinechan & Mc Lullu)", "channel": "Dj Brunin XM",
        "tags": ["funk carioca", "funk tamborzao"]}


def _classify(tmp_path, answer):
    import genre_text as GT
    track = _track(tmp_path, "t")
    (track / "metadata.json").write_text(json.dumps({"title": "x", "video_id": "abc"}))
    return GT.classify(track, tmp_path / "out.json", fetch=lambda url: INFO, ask=lambda prompt, model: answer)


def test_grounded_answer_is_kept(tmp_path):
    r = _classify(tmp_path, {"genre": "brazilian_funk", "evidence": ["Funk Carioca", "Dj Brunin XM"], "reason": "tags"})
    assert r["status"] == "ok" and r["genre"] == "brazilian_funk" and r["evidence"] == ["Funk Carioca", "Dj Brunin XM"]


def test_invented_evidence_makes_the_answer_unknown(tmp_path):
    r = _classify(tmp_path, {"genre": "brazilian_funk", "evidence": ["baile funk anthem"]})
    assert r["genre"] == "unknown" and r["evidence_rejected"] == ["baile funk anthem"]


def test_a_genre_outside_the_detector_set_is_unknown(tmp_path):
    r = _classify(tmp_path, {"genre": "reggaeton", "evidence": ["funk carioca"]})
    assert r["genre"] == "unknown"


def test_llm_failure_is_unavailable_not_a_crash(tmp_path):
    import genre_text as GT
    track = _track(tmp_path, "t")
    def boom(prompt, model):
        raise ConnectionError("ollama down")
    r = GT.classify(track, tmp_path / "out.json", fetch=lambda url: INFO, ask=boom)
    assert r["status"] == "unavailable" and r["genre"] is None


def test_the_verdict_is_cached_per_metadata_and_model(tmp_path):
    import genre_text as GT
    calls = []
    def ask(prompt, model):
        calls.append(model); return {"genre": "brazilian_funk", "evidence": ["funk carioca"]}
    track = _track(tmp_path, "t")
    (track / "metadata.json").write_text(json.dumps({"video_id": "abc"}))
    for _ in range(2):
        GT.classify(track, tmp_path / "o.json", fetch=lambda url: INFO, ask=ask)
    GT.classify(track, tmp_path / "o.json", model="other", fetch=lambda url: INFO, ask=ask)
    assert calls == ["mistral-small", "other"]


# --- evidence must NAME the genre (2026-11-07: "BibiBabydoll" / "Mc Erikah" alone labelled tracks funk) -------------

def test_artist_names_alone_are_not_evidence(tmp_path):
    r = _classify(tmp_path, {"genre": "brazilian_funk", "evidence": ["Dj Brunin XM", "Mc Lullu"], "reason": "artists"})
    assert r["genre"] == "unknown" and "names the genre" in r["why"]
    assert r["evidence"] == ["Dj Brunin XM", "Mc Lullu"]          # still recorded: grounded, just not enough


def test_a_quote_naming_the_genre_carries_the_artist_quotes(tmp_path):
    r = _classify(tmp_path, {"genre": "brazilian_funk", "evidence": ["Dj Brunin XM", "funk tamborzao"]})
    assert r["genre"] == "brazilian_funk"


def test_genre_names_come_from_the_genre_set_not_a_word_list():
    import genre_text as GT
    assert GT.names_genre("Funk Carioca", "brazilian_funk")
    assert GT.names_genre("FUNKEIRO", "brazilian_funk")
    assert not GT.names_genre("brazilian", "brazilian_funk")      # shared with brazilian_phonk: names neither
    assert GT.names_genre("Phonk Brasileiro", "brazilian_phonk")
    assert GT.names_genre("synthwave mix", "retro_wave")          # the genre's palette alias (sound_selector)
    assert GT.names_genre("Retrowave", "retro_wave")
    assert GT.names_genre("Lo-Fi beats", "lofi")                  # separators between the parts of a name
    assert not GT.names_genre("Mc Erikah", "brazilian_funk")
    assert not GT.names_genre("Shockwave", "retro_wave")          # a name inside another word is not the name


def test_an_auto_label_from_an_older_rule_is_not_a_label_and_the_track_is_re_decided(tmp_path):
    import genre_knn as GK
    from pipeline_helpers import label_stale
    reg, root = _world(tmp_path, funk=4)
    _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0]), reg, root, text_json=_text(tmp_path, "new", "brazilian_funk"))
    assert "new" in GK.load_labels(reg)
    p = GK.auto_labels_path(reg)
    doc = yaml.safe_load(p.read_text())
    for row in doc["labels"]:
        row.pop("rule", None)                                     # written before the rule had a version
    p.write_text(yaml.safe_dump(doc))
    assert "new" not in GK.load_labels(reg)
    assert label_stale(tmp_path / "tracks" / "new", reg) is True
    # re-decided: the metadata no longer agrees -> the stale row leaves the registry
    from pipeline_helpers import stamp_track_metadata
    stamp_track_metadata(tmp_path / "tracks" / "new", detector_json=tmp_path / "new.genre.json", labels_path=reg,
                         embeddings_root=root, stems_root=tmp_path / "stems",
                         text_json=_text(tmp_path, "new", "unknown", evidence=()))
    assert "new" not in _auto_rows(reg)
    assert label_stale(tmp_path / "tracks" / "new", reg) is False   # decided under the current rule: fresh


def test_a_track_stamped_from_a_label_that_is_gone_is_stale(tmp_path):
    import genre_knn as GK
    from pipeline_helpers import label_stale
    reg, root = _world(tmp_path, funk=4)
    _stamp(tmp_path, "new", _det(_cluster(0, 1, seed=7)[0]), reg, root, text_json=_text(tmp_path, "new", "brazilian_funk"))
    GK.auto_labels_path(reg).unlink()                            # the row was deleted ("delete a row to re-decide")
    assert label_stale(tmp_path / "tracks" / "new", reg) is True
