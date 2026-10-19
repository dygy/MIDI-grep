# @layer: unit
# @regression
"""The generator's header carries note/drum counts, genre and the refined BPM; report parsers + promote read them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pretty_midi
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import generate_dynamic_strudel as G  # noqa: E402
import generate_report as R  # noqa: E402
import pipeline_helpers as ph  # noqa: E402
from generate_hybrid_strudel import build_detected_drums  # noqa: E402

BPM = 129.989


def _midi(path: Path, pitches: list[int], bpm: float) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    step = 60.0 / bpm * 4 / 16
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=i * 2 * step, end=i * 2 * step + step))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def _piece(tmp_path: Path) -> tuple[str, int, int]:
    bass = G.midi_to_bars(_midi(tmp_path / "b.mid", [36, 38, 40, 41, 43], BPM), BPM, 2, pick="low", lo=28, hi=48)
    lead = G.midi_to_bars(_midi(tmp_path / "l.mid", [60, 62, 64], BPM), BPM, 2, pick="high", lo=48, hi=84)
    step = 60.0 / BPM * 4 / 16
    hits = [{"time": s * step, "type": t} for t, steps in (("bd", [0, 8, 16, 24]), ("sd", [4, 20]), ("hh", [2, 6, 10]))
            for s in steps]
    dj = tmp_path / "d.json"
    dj.write_text(json.dumps({"hits": hits}))
    drums = build_detected_drums(dj, BPM, 2)
    notes, dh = G.count_note_events(bass, lead), G.count_drum_hits(drums)
    code = "\n".join(["// MIDI-grep dynamic", "// genre=brazilian_funk",
                      *G.count_header_lines("brazilian_funk", notes, dh),
                      f"// BPM: {G.fmt_bpm(BPM)}", "// Key: C# minor", G.js_array("bass", bass or [])])
    return code, notes, dh


def test_header_counts_equal_events_and_parse(tmp_path: Path) -> None:
    code, notes, dh = _piece(tmp_path)
    assert (notes, dh) == (8, 9)                      # 5 bass + 3 lead notes; 4 bd + 2 sd + 3 hh
    info = R.extract_info_from_strudel(code)
    assert info["notes"] == notes and info["drum_hits"] == dh
    assert info["genre"] == "brazilian_funk"
    assert info["bpm"] == pytest.approx(BPM)          # refined tempo, not the rounded 130
    assert "style" not in info


def test_promote_copies_header_facts_into_metadata(tmp_path: Path) -> None:
    code, notes, dh = _piece(tmp_path)
    s = tmp_path / "o.strudel"; s.write_text(code)
    w = tmp_path / "r.wav"; w.write_bytes(b"x")
    c = tmp_path / "c.json"; c.write_text(json.dumps({"editability": "pass", "comparison": {}}))
    vdir = ph.promote_run(tmp_path / "t", "synth", s, w, c, ph.RunMeta(generator="g", render="r", compare="c"))
    meta = json.loads((vdir / "metadata.json").read_text())
    assert (meta["notes"], meta["drum_hits"], meta["genre"]) == (notes, dh, "brazilian_funk")
    assert meta["bpm"] == pytest.approx(BPM)
