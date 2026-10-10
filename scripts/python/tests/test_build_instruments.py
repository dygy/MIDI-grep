# @layer: unit
# @spec: 004-second-reference-track
# @regression
"""build_instruments.py: granular model dirs -> hosted, note-keyed Strudel manifest (spec 004 §2.2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from build_instruments import KIT_SOUNDS, build_instruments, main  # noqa: E402

BASE = "https://example.test/x/instruments/"


def _grains(*notes: int) -> list[dict[str, object]]:
    return [{"index": i, "file": f"g{i:04d}.wav", "pitch_hz": 100.0, "midi_note": n} for i, n in enumerate(notes)]


@pytest.fixture
def legacy_model(tmp_path: Path) -> Path:
    """Legacy trainer layout: pitched/<pitchclass>.wav, no pitched_map, per-grain midi_note."""
    model = tmp_path / "models" / "demo_bass"
    (model / "pitched").mkdir(parents=True)
    for pc in ("ds", "g", "a"):
        (model / "pitched" / f"{pc}.wav").write_bytes(b"RIFF" + pc.encode())
    # ds -> 39, g -> 31, a -> median(33, 33, 69) = 33 (an octave-error grain must not win)
    meta = {"name": "demo_bass", "grains": _grains(39, 31, 31, 33, 33, 69)}
    (model / "metadata.json").write_text(json.dumps(meta))
    return model


@pytest.fixture
def mapped_model(tmp_path: Path) -> Path:
    """Current trainer layout: metadata.pitched_map names the notes directly."""
    model = tmp_path / "models" / "demo_lead"
    (model / "pitched").mkdir(parents=True)
    (model / "pitched" / "e3.wav").write_bytes(b"e3")
    (model / "pitched" / "f_sharp_3.wav").write_bytes(b"f#3")
    meta = {"name": "demo_lead", "pitched_map": {"e3": "pitched/e3.wav", "f#3": "pitched/f_sharp_3.wav"}, "grains": []}
    (model / "metadata.json").write_text(json.dumps(meta))
    return model


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    d = tmp_path / "drums"
    d.mkdir()
    for s in KIT_SOUNDS:
        (d / f"{s}.wav").write_bytes(b"kit" + s.encode())
    return d


def test_legacy_model_uses_pitch_class_medians_and_sharp_keys(legacy_model: Path, kit: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_instruments([legacy_model], kit, out, BASE)
    doc = json.loads((out / "samples.json").read_text())
    assert doc["demo_bass"] == {
        "g1": "demo_bass/g1.wav",
        "a1": "demo_bass/a1.wav",
        "d#2": "demo_bass/d_sharp_2.wav",
    }
    assert (out / "demo_bass" / "d_sharp_2.wav").read_bytes() == b"RIFFds"


def test_pitched_map_model_keys_come_from_the_map(mapped_model: Path, kit: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_instruments([mapped_model], kit, out, BASE)
    doc = json.loads((out / "samples.json").read_text())
    assert doc["demo_lead"] == {"e3": "demo_lead/e3.wav", "f#3": "demo_lead/f_sharp_3.wav"}


def test_base_is_absolute_with_trailing_slash(legacy_model: Path, kit: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_instruments([legacy_model], kit, out, "https://example.test/x/instruments")
    assert json.loads((out / "samples.json").read_text())["_base"] == BASE


def test_kit_entries_are_single_element_arrays(legacy_model: Path, kit: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_instruments([legacy_model], kit, out, BASE)
    doc = json.loads((out / "samples.json").read_text())
    for s in KIT_SOUNDS:
        assert doc[s] == [f"kit/{s}.wav"]
        assert (out / "kit" / f"{s}.wav").is_file()


def test_every_manifest_path_exists_under_out(legacy_model: Path, mapped_model: Path, kit: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_instruments([legacy_model, mapped_model], kit, out, BASE)
    doc = json.loads((out / "samples.json").read_text())
    for key, val in doc.items():
        if key == "_base":
            continue
        files = val if isinstance(val, list) else list(val.values())
        assert all((out / f).is_file() for f in files), key


def test_missing_kit_one_shot_is_a_clear_error_and_writes_nothing(legacy_model: Path, kit: Path, tmp_path: Path) -> None:
    (kit / "oh.wav").unlink()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match=r"missing one-shot.*oh\.wav"):
        build_instruments([legacy_model], kit, out, BASE)
    assert not out.exists()


def test_relative_base_url_rejected(legacy_model: Path, kit: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_instruments([legacy_model], kit, tmp_path / "out", "instruments/")


def test_cli_missing_kit_exits_1(legacy_model: Path, kit: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (kit / "bd.wav").unlink()
    rc = main(["--models", str(legacy_model), "--kit", str(kit), "--out", str(tmp_path / "o"), "--base-url", BASE])
    assert rc == 1
    assert "bd.wav" in capsys.readouterr().err


# review finding #1: the legacy median must stay on the file's pitch class
def test_legacy_median_snaps_to_the_files_pitch_class(tmp_path):
    import json, struct, wave
    from build_instruments import ModelMetadata, resolve_notes
    model = tmp_path / "legacy"; (model / "pitched").mkdir(parents=True)
    for stem in ("c", "fs"):
        with wave.open(str(model / "pitched" / f"{stem}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(struct.pack("<h", 0) * 441)
    # 'c' grains at midi 36 and 48 (median 42 = f#2 — the bug); 'f#' grains at 42
    grains = [{"index": i, "file": f"g{i}.wav", "pitch_hz": 100.0, "midi_note": m, "onset_sec": 0.0}
              for i, m in enumerate([36, 48, 42])]
    (model / "metadata.json").write_text(json.dumps({"name": "legacy", "type": "granular", "grains": grains}))
    meta = ModelMetadata.model_validate(json.loads((model / "metadata.json").read_text()))
    notes = resolve_notes(model, meta)
    assert set(notes) == {"c2", "f#2"} or set(notes) == {"c3", "f#2"}, notes   # c stays a C; f# stays an F#
    assert notes[[n for n in notes if n.startswith("c")][0]].name == "c.wav"
