# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""Stage 6 re-uploaded the WHOLE sample pack whenever any file in it was newer than the last upload, e.g. stage 3
rewriting drums_bands.json (a generator input Strudel never fetches). upload_r2 now hashes only the files Strudel
fetches (the manifests + what they reference) and uploads just those whose hash changed. No network: the backend
is mocked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upload_r2  # noqa: E402

BASE = "https://pub.example.dev"
PREFIX = "midi-grep/trk"


def make_pack(root: Path) -> Path:
    pack = root / "sample_pack"
    (pack / "drums").mkdir(parents=True)
    (pack / "instruments" / "kit").mkdir(parents=True)
    (pack / "drums" / "bd.wav").write_bytes(b"bd")
    (pack / "drums" / "sd.wav").write_bytes(b"sd")
    (pack / "instruments" / "kit" / "bd.wav").write_bytes(b"kbd")
    (pack / "instruments" / "bass").mkdir()
    (pack / "instruments" / "bass" / "c1.wav").write_bytes(b"c1")
    (pack / "strudel.json").write_text(json.dumps({"bd": "drums/bd.wav", "sd": "drums/sd.wav"}))
    (pack / "samples.json").write_text(json.dumps(
        {"_base": f"{BASE}/{PREFIX}/", "bd": ["drums/bd.wav"], "sd": ["drums/sd.wav"]}))
    (pack / "instruments" / "samples.json").write_text(json.dumps(
        {"_base": f"{BASE}/{PREFIX}/instruments/", "bd": ["kit/bd.wav"], "trk_bass": {"c1": "bass/c1.wav"}}))
    # never fetched by Strudel
    (pack / "drums_bands.json").write_text("{}")
    (pack / "bass.mid").write_bytes(b"mid")
    (pack / "pack.json").write_text("{}")
    (pack / "instruments" / "verification.json").write_text("{}")
    return pack


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(pack_dir: Path, bucket: str, prefix: str, files: list[Path] | None = None) -> None:
        chosen = upload_r2.iter_files(pack_dir) if files is None else files
        calls.append(sorted(f.relative_to(pack_dir).as_posix() for f in chosen))

    monkeypatch.setattr(upload_r2, "upload_wrangler", fake)
    return calls


def run(pack: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, base: str = BASE) -> int:
    monkeypatch.setattr(sys, "argv", ["upload_r2", "--pack-dir", str(pack), "--prefix", PREFIX, "--bucket", "b",
                                      "--public-base", base, "--backend", "wrangler", *extra])
    return upload_r2.main()


def test_first_upload_sends_only_served_files_and_records_manifest(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    assert len(uploads) == 1
    assert "drums_bands.json" not in uploads[0] and "bass.mid" not in uploads[0]
    assert "instruments/bass/c1.wav" in uploads[0] and "instruments/samples.json" in uploads[0]
    state = json.loads((pack / ".uploaded.json").read_text())
    assert state["base"] == f"{BASE}/{PREFIX}" and set(state["files"]) == set(uploads[0])


def test_changing_only_drums_bands_uploads_nothing(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    uploads.clear()
    (pack / "drums_bands.json").write_text('{"rewritten": 1}')
    (pack / "bass.mid").write_bytes(b"other")
    assert run(pack, monkeypatch, "--check") == 0
    assert run(pack, monkeypatch) == 0
    assert uploads == []


def test_changing_one_sample_uploads_exactly_that_file(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    uploads.clear()
    (pack / "drums" / "sd.wav").write_bytes(b"sd-new")
    assert run(pack, monkeypatch, "--check") == 3
    run(pack, monkeypatch)
    assert uploads == [["drums/sd.wav"]]
    uploads.clear()
    run(pack, monkeypatch)
    assert uploads == []


def test_base_change_uploads_everything_served(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    first = uploads[0]
    uploads.clear()
    run(pack, monkeypatch, base="https://other.example.dev")
    assert uploads == [first]


def test_legacy_uploaded_base_string_means_unknown_full_upload(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    (pack / ".uploaded").write_text(f"{BASE}/{PREFIX}")
    assert run(pack, monkeypatch, "--check") == 3
    run(pack, monkeypatch)
    assert len(uploads) == 1 and len(uploads[0]) == 7
    assert (pack / ".uploaded.json").exists()


def test_full_flag_uploads_all_files_and_still_records(tmp_path, monkeypatch, uploads):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    uploads.clear()
    run(pack, monkeypatch, "--full")
    assert "drums_bands.json" in uploads[0] and ".uploaded.json" not in uploads[0]
    uploads.clear()
    assert run(pack, monkeypatch, "--check") == 0


def test_failed_upload_does_not_record_state(tmp_path, monkeypatch):
    pack = make_pack(tmp_path)

    def boom(*a, **k):
        raise SystemExit("wrangler upload failed")

    monkeypatch.setattr(upload_r2, "upload_wrangler", boom)
    with pytest.raises(SystemExit):
        run(pack, monkeypatch)
    assert not (pack / ".uploaded.json").exists()


def test_removed_served_file_is_reported_not_deleted(tmp_path, monkeypatch, uploads, capsys):
    pack = make_pack(tmp_path)
    run(pack, monkeypatch)
    uploads.clear()
    (pack / "strudel.json").write_text(json.dumps({"bd": "drums/bd.wav"}))
    (pack / "samples.json").write_text(json.dumps({"_base": f"{BASE}/{PREFIX}/", "bd": ["drums/bd.wav"]}))
    capsys.readouterr()
    run(pack, monkeypatch)
    assert uploads[0] == ["samples.json", "strudel.json"]
    assert "no longer served" in capsys.readouterr().err
