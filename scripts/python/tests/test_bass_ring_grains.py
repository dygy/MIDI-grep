# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""A stem-built bass sample must keep the 808's ring (fix bass-grains-cut-at-transcribed-end).

Measured on both reference stems: the transcriber cuts an 808 at about one 16th step (median 0.07-0.09 s)
while the Demucs bass stem keeps sounding that note for 0.2-0.3 s (seconds when held). Every bass grain was
cut at the transcribed end, so the commonest notes were 0.10-0.25 s samples (g1 0.10 s), the sub tail was
gone, and the render came out 10-12 dB short below 60 Hz (20-45 Hz -10 / -12 dB, 45-60 Hz -6 / -9.5 dB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
pretty_midi = pytest.importorskip("pretty_midi")
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import pitched_grains as pg  # noqa: E402

SR = 44100
STEP = 0.55          # one struck note every 0.55 s
RING = 0.45          # ... rings 0.45 s (decaying), far beyond the 0.08 s the transcriber writes
STUB = 0.08
PITCHES = [28, 31, 28, 31, 28, 31, 28, 31, 28, 31, 28, 31]


def _808(midi: int, dur: float) -> np.ndarray:
    f = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * SR)) / SR
    y = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-3.0 * t)
    return (y * np.minimum(1.0, t / 0.003)).astype(np.float32)


@pytest.fixture(scope="module")
def stem(tmp_path_factory: pytest.TempPathFactory):
    d = tmp_path_factory.mktemp("ringstem")
    y = np.zeros(int((len(PITCHES) * STEP + 0.5) * SR), np.float32)
    notes = []
    for i, m in enumerate(PITCHES):
        t0 = 0.3 + i * STEP
        tone = _808(m, RING)
        y[int(t0 * SR): int(t0 * SR) + len(tone)] += tone
        notes.append(pg.Note(t0, t0 + STUB, m))          # what the transcriber writes: a stub
    return y, notes


def test_trainer_samples_keep_the_ring(stem, tmp_path: Path) -> None:
    """End to end through the trainer (what stage 5 runs): the pitched samples are as long as the stem rings.
    Old trainer: every sample was cut at the transcribed 0.08 s end (or, too short to verify, dropped)."""
    from rave.trainer import GranularTrainer
    y, notes = stem
    wav, mid = tmp_path / "bass.wav", tmp_path / "bass.mid"
    soundfile.write(str(wav), y, SR)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=n.pitch, start=n.start, end=n.end))
    pm.instruments.append(inst); pm.write(str(mid))
    meta = GranularTrainer(str(tmp_path / "models")).train(str(wav), "ring", midi_path=str(mid))
    durs = meta["pitched_durations_sec"]
    assert set(durs) == {"e1", "g1"}, durs
    assert all(d >= 0.8 * RING for d in durs.values()), durs          # was <= 0.08 s


def test_grains_last_as_long_as_the_stem_rings(stem) -> None:
    y, notes = stem
    fmin, fmax = pg.register_hz([n.pitch for n in notes])
    grains = pg.extract_note_grains(y, SR, notes, fmin, fmax, ring=pg.StemRing(y, SR))
    chosen = pg.select_per_pitch(grains, pg.ring_target_length(grains, STUB), STUB)
    assert set(chosen) == {28, 31}
    for m, g in chosen.items():
        assert g.ringed and g.duration_s >= 0.8 * RING, (m, g.duration_s)    # was 0.08 s
        assert g.duration_s <= STEP                                           # never into the next attack
        # the tail is there: the last quarter of the second half still carries signal
        tail = np.abs(g.audio[int(0.4 * len(g.audio)): int(0.7 * len(g.audio))]).max()
        assert tail > 0.02 * np.abs(g.audio).max()


def test_ring_end_is_bounded_by_the_next_attack_and_keeps_echoes_short() -> None:
    attacks = np.array([1.0, 1.3, 2.0])
    # rings past the next attack -> stops at it
    assert pg.ring_end(1.0, 1.08, attacks, 0.05, sounding_end=1.9) == pytest.approx(1.3)
    # goes quiet before the next attack -> stops there
    assert pg.ring_end(1.0, 1.08, attacks, 0.05, sounding_end=1.2) == pytest.approx(1.2)
    # never shorter than the transcribed end
    assert pg.ring_end(2.0, 2.4, attacks, 0.05, sounding_end=2.1) == pytest.approx(2.4)
    # a note with no attack of its own (transcription echo) keeps its transcribed span
    assert pg.ring_end(1.6, 1.68, attacks, 0.05, sounding_end=2.5) == pytest.approx(1.68)


def test_a_pitch_change_without_a_new_attack_ends_the_grain() -> None:
    """Legato: the 808 slides to another pitch with no re-attack; the sample of the first note must stop there."""
    t1 = np.arange(int(0.3 * SR)) / SR
    f1, f2 = 440 * 2 ** ((28 - 69) / 12), 440 * 2 ** ((33 - 69) / 12)
    y = np.concatenate([np.sin(2 * np.pi * f1 * t1), np.sin(2 * np.pi * f2 * t1)]).astype(np.float32)
    fmin, fmax = pg.register_hz([28, 33])
    cut = pg.pitch_stable_end(y, SR, 28, fmin, fmax, start_s=0.05)
    assert 0.2 <= cut <= 0.36, cut


def test_a_pitch_is_represented_by_a_grain_at_the_level_the_stem_plays_it() -> None:
    """The cleanest-sounding grain of a pitch can be its quietest one (Regime e1: peak 0.04 vs the stem's 0.13-0.8)."""
    def g(q, rms):
        return pg.Grain(np.zeros(10, np.float32), 28, 28, "salience", q, 0.0, 0.5, True, rms)
    chosen = pg.select_per_pitch([g(9.0, 0.01), g(2.0, 0.20), g(3.0, 0.25), g(1.0, 0.30)], 0.4, 0.0)
    assert chosen[28].rms >= 0.20 and chosen[28].quality == 3.0      # quiet outlier skipped; best of the typical-or-louder


def test_the_pitch_level_is_the_median_of_all_its_grains_not_of_the_long_ones() -> None:
    """Regime d#1: the grains long enough to carry the note were mostly quiet passages (rms 0.01-0.18) while the
    pitch's other grains sat at 0.4-0.6; the pool's own median licensed a 0.18 grain and every d#1 note rendered
    8 dB under the stem."""
    def g(q, rms, dur):
        return pg.Grain(np.zeros(10, np.float32), 27, 27, "salience", q, 0.0, dur, True, rms)
    grains = [g(1.0, 0.01, 0.6), g(6.0, 0.06, 0.6), g(1.0, 0.18, 0.5), g(0.7, 0.52, 0.5),     # long: mostly quiet
              g(2.0, 0.45, 0.2), g(2.0, 0.50, 0.2), g(2.0, 0.55, 0.2), g(2.0, 0.60, 0.2)]   # short: the usual level
    chosen = pg.select_per_pitch(grains, 0.45, 0.0)
    assert chosen[27].rms == 0.52, chosen[27].rms


def test_attack_is_at_sample_zero() -> None:
    """VAGABUNDO c#1 started 79 ms into its file (transcribed onset ahead of the stem's hit)."""
    y = np.concatenate([np.zeros(int(0.08 * SR), np.float32), _808(25, 0.4)])
    a = pg.attack_aligned(y, SR)
    assert (len(y) - len(a)) / SR == pytest.approx(0.08, abs=0.005)
    assert np.argmax(np.abs(a) >= 0.1 * np.abs(a).max()) <= 2      # attack within 2 samples of the start

