# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""The bass octave comes from the STEM (bass_register.py), not from a typed fold window.

Regression: generate_dynamic_strudel folded every bass note into MIDI [28,48], so VAGABUNDO's held C#1 sub
(MIDI 25) was written at 37 (45% exact vs 75% for Basic Pitch's own lowest note). Synthetic stem f0 tracks
here; no audio models, no render.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_bass_register.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

import bass_register as BR  # noqa: E402
import generate_dynamic_strudel as G  # noqa: E402

BPM = 120.0
STEP = 60.0 / BPM * 4 / 16          # 0.125 s
HOP = 0.0232


def _register(spans: list[tuple[float, float, float | None]], total: float = 8.0) -> BR.Register:
    """f0 track: (t0, t1, midi|None) spans; everything else unvoiced. Constant loud RMS."""
    n = int(total / HOP)
    midi = [None] * n
    for t0, t1, m in spans:
        for i in range(int(t0 / HOP), min(n, int(t1 / HOP))):
            midi[i] = m
    return BR.Register({"hop_s": HOP, "midi": midi, "rms": [0.1] * n})


def _mid(tmp_path: Path, notes: list[tuple[float, int]], name="bass.mid") -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for t, p in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + STEP))
    pm.instruments.append(inst)
    path = tmp_path / name
    pm.write(str(path))
    return path


def _pitches(bars):
    return {i: G._bars_pitches([b]) for i, b in enumerate(bars)}


def _flat(bars):
    out = {}
    for bi, b in enumerate(bars):
        for si, tok in enumerate(b.split()):
            if tok != "~":
                out[bi * 16 + si] = G._bars_pitches([tok])[0]
    return out


def test_window_is_the_stems_own_register():
    reg = _register([(0.0, 5.0, 25.4), (5.0, 7.0, 37.1), (7.0, 7.05, 60.0)])   # 60 = 2-frame outlier
    lo, hi = reg.window()
    assert (lo, hi) == (25, 37)


def test_sub_notes_stay_at_the_stem_octave(tmp_path):
    """BP lowest notes at MIDI 25/26/27 where the stem sounds 25-27: they must not be folded up to 37-39."""
    reg = _register([(0.0, 0.5, 25.3), (0.5, 1.0, 26.2), (1.0, 1.5, 27.1), (1.5, 8.0, 25.4)])
    mid = _mid(tmp_path, [(0.0, 25), (0.5, 26), (1.0, 27)])
    bars = G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg)
    assert _flat(bars) == {0: 25, 4: 26, 8: 27}


def test_genuine_octave_jump_is_preserved(tmp_path):
    """The stem really jumps C#1 -> C#2 (riff): both register notes survive."""
    reg = _register([(0.0, 0.4, 25.5), (0.5, 0.9, 37.3), (1.0, 8.0, 25.5)])
    mid = _mid(tmp_path, [(0.0, 25), (0.5, 37), (1.0, 25)])
    got = _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg))
    assert got == {0: 25, 4: 37, 8: 25}


def test_transcription_octave_error_is_corrected(tmp_path):
    """BP wrote the 2nd/3rd partial (MIDI 37 / 49 / 61) over a stem that sounds 25.5: snap down to 25."""
    reg = _register([(0.0, 8.0, 25.5)])
    mid = _mid(tmp_path, [(0.0, 37), (0.5, 49), (1.0, 61)])
    got = _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg))
    assert got == {0: 25, 4: 25, 8: 25}


def test_nearest_candidate_wins_not_lowest(tmp_path):
    """Two transcribed notes on one step: a spurious lower pitch class loses to the one matching the stem f0."""
    reg = _register([(0.0, 8.0, 30.1)])
    mid = _mid(tmp_path, [(0.0, 28), (0.0, 42)])          # 42 = 30 + 12 (right class), 28 = wrong class
    assert _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg)) == {0: 30}


def test_unvoiced_step_falls_back_to_window_fold(tmp_path):
    reg = _register([(0.0, 0.4, 25.5), (2.0, 8.0, 25.5)])   # unvoiced 0.4-2.0 s
    mid = _mid(tmp_path, [(1.0, 61), (1.5, 20)])
    got = _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg))
    assert all(25 <= p <= 37 for p in got.values()) and got[8] == 37 and got[12] == 32


def test_no_register_keeps_the_given_window(tmp_path):
    """Lead path / no measurement: plain fold into [lo,hi], highest wins."""
    mid = _mid(tmp_path, [(0.0, 90), (0.0, 50), (0.5, 20)])
    assert _flat(G.midi_to_bars(mid, BPM, 4, pick="high", lo=48, hi=79)) == {0: 78, 4: 56}


def test_onset_driven_path_uses_the_register_too(tmp_path):
    """onset_driven_bars: a 35 Hz sub burst every 0.5 s with BP writing the octave above (37)."""
    sr = 22050
    y = np.zeros(int(4 * sr))
    t = np.arange(int(0.3 * sr)) / sr
    burst = np.sin(2 * np.pi * 35.0 * t) * np.exp(-6 * t)
    for k in range(8):
        a = int(k * 0.5 * sr)
        y[a:a + len(burst)] += burst
    stem = tmp_path / "bass.wav"
    sf.write(stem, y, sr)
    reg = _register([(0.0, 4.0, 25.5)], total=4.0)
    mid = _mid(tmp_path, [(k * 0.5, 37) for k in range(8)])
    bars = G.onset_driven_bars(stem, mid, BPM, 2, pick="low", lo=25, hi=37, register=reg)
    notes = _flat(bars)
    assert notes and set(notes.values()) == {25}


def test_smallest_interval():
    assert BR.smallest_interval({25: 10.0, 26: 1.0, 37: 5.0, 60: 0.1}, 0.9) == (25, 37)


def test_old_signature_is_gone_as_fallback_only():
    """The typed window survives only as a named, loudly-warned fallback constant."""
    assert G.LEGACY_BASS_WINDOW == (28, 48) and G.LEGACY_LEAD_WINDOW == (48, 79)
    src = (SCRIPTS / "generate_dynamic_strudel.py").read_text()
    assert not re.search(r"lo=28\s*,\s*hi=48", src)


# ---- end to end through the CLI on the cached Regime CLT pack (skipped without it) ----
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
needs_pack = pytest.mark.skipif(not (PACK / "bass.mid").exists(), reason="Regime CLT pack not cached")


def _gen(tmp_path, *extra):
    out = tmp_path / "o.strudel"
    r = subprocess.run([sys.executable, str(SCRIPTS / "generate_dynamic_strudel.py"), "--stems-dir", str(STEMS),
                        "--pack-dir", str(PACK), "--bass-midi", str(PACK / "bass.mid"),
                        "--lead-midi", str(PACK / "melodic.mid"), "--drums-json", str(PACK / "drums_bands.json"),
                        "--base-url", "http://localhost:1/", "--mode", "synth", "--bpm", "135.017", "--genre", "brazilian_funk", "--num-bars", "77",
                        "--out", str(out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    txt = out.read_text()
    bass = re.search(r"let bass = \[(.*?)\n\]", txt, re.S).group(1)
    return G._bars_pitches(re.findall(r'"([^"]*)"', bass)), r.stderr, txt


@needs_pack
def test_missing_f0_falls_back_loudly(tmp_path):
    pitches, err, _ = _gen(tmp_path, "--bass-f0", str(tmp_path / "absent.json"))
    assert "FALLING BACK" in err and min(pitches) >= 28


@needs_pack
def test_cli_with_measured_register_keeps_sub_notes(tmp_path):
    f0 = tmp_path / "f0.json"
    BR.measure(STEMS / "bass.wav", f0)
    pitches, err, _ = _gen(tmp_path, "--bass-f0", str(f0))
    assert "FALLING BACK" not in err
    assert min(pitches) < 28 and np.median(pitches) <= 30        # stem median is 28.2


@needs_pack
def test_sub_octave_never_goes_under_the_sub_floor(tmp_path):
    f0 = tmp_path / "f0.json"
    BR.measure(STEMS / "bass.wav", f0)
    _, err, txt = _gen(tmp_path, "--bass-f0", str(f0), "--sub-octave", "2")
    assert "sub layer under" in err and ".add(note(-24))" not in txt


# --- non-octave partials: the 3rd/5th harmonic of the sub is a transcription of the PARTIAL, not a note ---

def test_non_octave_partial_resolves_to_the_stem_fundamental(tmp_path):
    """Basic Pitch hears the 3rd (+19) and 5th (+28) partial of a C#1 808 as G#2 (44) / F3 (53). Neither shares a
    pitch class with the stem f0 (25.5), so the old fold wrote G#1 (32) / F1 (29) = a wrong note under a sub that
    sounds C#1. The stem's f0 is the fundamental: write it."""
    reg = _register([(0.0, 8.0, 25.4)])
    mid = _mid(tmp_path, [(0.0, 44), (0.5, 53), (1.0, 37)])
    got = _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg))
    assert got == {0: 25, 4: 25, 8: 25}


def test_unrelated_wrong_pitch_class_is_still_folded(tmp_path):
    """A candidate that is NOT a harmonic of the stem f0 keeps the old window fold (a real note change)."""
    reg = _register([(0.0, 8.0, 30.1)])
    mid = _mid(tmp_path, [(0.0, 33)])                      # +2.9 st: no integer harmonic within a semitone
    assert _flat(G.midi_to_bars(mid, BPM, 4, pick="low", lo=25, hi=37, register=reg)) == {0: 33}


def test_harmonic_interval_table_is_derived_not_typed():
    """k * f0 sits at 12*log2(k) semitones; only harmonics whose neighbours are >= 2 tolerances apart count."""
    assert BR.harmonic_of(30.0 + 19.02, 30.0) == 3        # 3rd harmonic
    assert BR.harmonic_of(30.0 + 12.0, 30.0) == 2
    assert BR.harmonic_of(30.0 + 12 * np.log2(10), 30.0) is None  # the 10th is within 2 st of the 9th: ambiguous
    assert BR.harmonic_of(30.0 + 12 * np.log2(5), 30.0) == 5
    assert BR.harmonic_of(30.0 + 2.9, 30.0) is None
