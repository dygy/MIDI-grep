# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Granular instruments must play the written notes (spec 003 §2.3b): note-aligned grains, measured labels.

Old trainer: every pitched sample was a hardcoded 100 ms onset grain pitched by pyin on ~1 analysis frame
(a 41 Hz note and 300 ms notes both came out 0.10 s, mislabelled or unvoiced). New trainer with the stem's
MIDI: one grain per note, as long as the note, labelled by the measured pitch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
pretty_midi = pytest.importorskip("pretty_midi")
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

SR = 44100
# (midi, duration_s): includes a 41 Hz bass note (E1) and 300 ms notes; each pitch twice
NOTES = [(28, 0.60), (35, 0.30), (38, 0.30), (43, 0.30), (47, 0.30), (28, 0.60), (35, 0.30), (38, 0.30)]
GAP = 0.08


def _tone(midi: int, dur: float) -> np.ndarray:
    f = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 5))
    return (y * np.exp(-1.2 * t) * np.minimum(1, t / 0.004)).astype(np.float32)


@pytest.fixture(scope="module")
def stem(tmp_path_factory: pytest.TempPathFactory):
    d = tmp_path_factory.mktemp("stem")
    parts, pm_notes, t = [np.zeros(int(0.2 * SR), np.float32)], [], 0.2
    for m, dur in NOTES:
        parts += [_tone(m, dur), np.zeros(int(GAP * SR), np.float32)]
        pm_notes.append((m, t, t + dur)); t += dur + GAP
    wav = d / "bass.wav"
    soundfile.write(str(wav), np.concatenate(parts), SR)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    for m, s, e in pm_notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=m, start=s, end=e))
    pm.instruments.append(inst); mid = d / "bass.mid"; pm.write(str(mid))
    return wav, mid, d


def _train(stem, name: str, **kw) -> dict:
    from rave.trainer import GranularTrainer
    wav, mid, d = stem
    return GranularTrainer(str(d / "models")).train(str(wav), name, **kw)


def test_note_aligned_grains_are_as_long_as_the_notes_and_labelled_by_measurement(stem) -> None:
    meta = _train(stem, "notes", midi_path=str(stem[1]))
    assert meta["grain_mode"] == "notes"
    want = {"e1": 28, "b1": 35, "d2": 38, "g2": 43, "b2": 47}
    assert set(meta["pitched_map"]) == set(want), meta["pitched_map"]
    model = stem[2] / "models" / "notes"
    for key, midi in want.items():
        y, sr = soundfile.read(str(model / meta["pitched_map"][key]))
        written = max(d for m, d in NOTES if m == midi)
        assert len(y) / sr >= 0.9 * written - 0.01, (key, len(y) / sr, written)   # NOT 0.10 s
        assert len(y) / sr > 0.15
    # label = measured pitch of the stable part: every grain agrees with its transcription within a semitone
    assert all(abs(g["midi_note"] - g["transcribed_note"]) == 0 for g in meta["grains"])


def test_cli_without_midi_or_grain_ms_derives_the_grain_length_from_the_stem(stem, tmp_path: Path) -> None:
    """The old CLI default (--grain-ms 100) forced a fixed 0.10 s grain; now it is the stem's own P25 IOI."""
    import subprocess
    import librosa
    wav, _mid, _ = stem
    r = subprocess.run([sys.executable, "-m", "rave.cli", "train", str(wav), "--name", "auto", "--mode", "granular",
                        "--output", str(tmp_path / "m")], cwd=SCRIPTS, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(SCRIPTS)})
    assert r.returncode == 0, r.stderr[-800:]
    meta = json.loads((tmp_path / "m" / "auto" / "metadata.json").read_text())
    y, sr = librosa.load(str(wav), sr=44100, mono=True)
    ioi = np.diff(librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr))) / sr
    assert meta["grain_mode"] == "onsets"
    assert meta["grain_duration_ms"] == pytest.approx(float(np.percentile(ioi, 25)) * 1000, abs=0.01)


def test_sample_pack_builds_no_vocal_material_and_prunes_stale_vocal_entries(tmp_path: Path) -> None:
    """Vocals are never generated: build_sample_pack drops vocal entries (``*_vocal``, ``vox<N>``, ``vocals*``)."""
    import shutil
    import subprocess
    stems = tmp_path / "stems"; out = tmp_path / "pack"
    stems.mkdir(); out.mkdir()
    for name in ("bass", "drums", "melodic", "vocals"):
        shutil.copy(stem_wav_for_pack(tmp_path), stems / f"{name}.wav")
    (out / "strudel.json").write_text(json.dumps({"x_vocal": {"c4": "vocals/vocal_60.wav"}, "vox0": "vocals/vox0.wav",
                                                  "vocalsloop": ["loops/vocals_000.wav"], "bd": "drums/bd.wav"}))
    (out / "samples.json").write_text(json.dumps({"_base": "http://h/", "x_vocal": {"c4": "v.wav"}, "vox0": ["v.wav"]}))
    r = subprocess.run([sys.executable, str(SCRIPTS / "build_sample_pack.py"), "--stems-dir", str(stems), "--out", str(out),
                        "--bpm", "120", "--prefix", "x", "--base-url", "http://h/", "--sections", "bass,melodic"],
                       capture_output=True, text=True, cwd=SCRIPTS)
    assert r.returncode == 0, r.stderr[-600:]
    for f in ("strudel.json", "samples.json", "pack.json"):
        text = (out / f).read_text().lower()
        assert "vocal" not in text.replace("no vocal material (vocals are never generated).", "") and "vox" not in text, f
    assert not (out / "vocals").exists()


def stem_wav_for_pack(tmp_path: Path) -> Path:
    p = tmp_path / "src.wav"
    if not p.exists():
        parts = [np.zeros(int(0.1 * SR), np.float32)]
        for m in (40, 45, 40, 47, 45, 40):
            parts += [_tone(m, 0.4), np.zeros(int(0.05 * SR), np.float32)]
        soundfile.write(str(p), np.concatenate(parts), SR)
    return p


# ---- polyphony: pyin is monophonic, so a transcribed label is verified by harmonic salience --------------

def _chord(midis, dur=0.3):
    return sum(_tone(m, dur) for m in midis).astype(np.float32)


def test_salience_accepts_the_transcribed_note_of_a_chord_and_rejects_octave_and_ghost_labels() -> None:
    import pitched_grains as pg
    chord = _chord([52, 56, 59])                       # E3 G#3 B3 (E major triad)
    lo, hi = 40.0, 76.0
    assert pg.check_label(chord, SR, 52, lo, hi).accepted        # root
    assert pg.check_label(chord, SR, 56, lo, hi).accepted        # a chord tone, not the root
    assert not pg.check_label(chord, SR, 40, lo, hi).accepted    # octave error (E2): below the fundamental
    assert not pg.check_label(chord, SR, 64, lo, hi).accepted    # octave above (E4): only a partial of E3
    assert not pg.check_label(chord, SR, 54, lo, hi).accepted    # ghost: F#3 is in no voice


def test_polyphonic_stem_keeps_the_written_notes(tmp_path: Path) -> None:
    from rave.trainer import GranularTrainer
    chords = [[52, 56, 59], [57, 61, 64], [52, 56, 59], [55, 59, 62]]
    parts, notes, t = [np.zeros(int(0.1 * SR), np.float32)], [], 0.1
    for c in chords:
        parts += [_chord(c), np.zeros(int(0.05 * SR), np.float32)]
        notes += [(m, t, t + 0.3) for m in c]
        t += 0.35
    wav = tmp_path / "poly.wav"; soundfile.write(str(wav), np.concatenate(parts), SR)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(velocity=90, pitch=m, start=s, end=e) for m, s, e in notes]
    pm.instruments.append(inst); mid = tmp_path / "poly.mid"; pm.write(str(mid))
    meta = GranularTrainer(str(tmp_path / "m")).train(str(wav), "poly", midi_path=str(mid))
    written = {m for c in chords for m in c}
    got = {g["midi_note"] for g in meta["grains"]}
    assert got <= written, got - written                      # no wrong label gets in
    assert len(got) >= 0.7 * len(written), (sorted(got), sorted(written))   # and most written notes are sampled
