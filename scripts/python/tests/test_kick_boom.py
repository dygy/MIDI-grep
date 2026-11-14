"""The synth kick boom (kick_boom.py) and its level lever (calibrate_dynamic.py --kick-boom-scale).

Sem Aliança synth (2026-11-14): the drums stem rang a tuned 808 at 45 Hz through the beat (0.74 of its peak power at the
beat's end) under bass notes at 58 / 73 / 59 Hz, and the RolandTR808 bd is short and at 54 Hz: the render's sub_bass share
in bars 11-29 sat at 36 % vs 64 %. A sine on the kick hits at the measured pitch, at the bass voice's measured level
fraction, scored 0.7797 -> 0.7920 overall / 0.7367 -> 0.7605 section-aware by hand (Regime 0.9082 -> 0.9077, VAGABUNDO
0.8462 -> 0.8507).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bass_register as BR  # noqa: E402
import kick_boom as KB  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / "calibrate_dynamic.py"
FIX = Path(__file__).resolve().parent / "fixtures" / "calibrate"
SR = 22050
BPM = 120.0
BAR = 60.0 / BPM * 4


def _write(path: Path, x: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), np.stack([x, x], axis=1).astype(np.float32), SR)


@pytest.fixture(scope="module")
def stems(tmp_path_factory):
    """4 bars. Drums: a kick click on every beat; bars 1-2 also ring a 45 Hz tone through each beat (the boom), bar 3
    only clicks. Bass: a 73 Hz note in every bar (a different pitch than the boom: it is not the bass)."""
    d = tmp_path_factory.mktemp("boom")
    n = int(4 * BAR * SR)
    t = np.arange(n) / SR
    beat = 60.0 / BPM
    drums = np.zeros(n)
    for b in range(16):
        i = int(b * beat * SR)
        k = np.arange(int(0.02 * SR))
        drums[i:i + k.size] += 0.6 * np.exp(-k / (0.004 * SR)) * np.sin(2 * np.pi * 120 * k / SR)   # click (percussive)
        if 4 <= b < 12:                                                                               # bars 1-2: boom
            m = np.arange(int(beat * SR))
            drums[i:i + m.size] += 0.5 * np.exp(-m / (2.0 * SR)) * np.sin(2 * np.pi * 45.0 * m / SR)
    bass = 0.3 * np.sin(2 * np.pi * 73.4 * t)
    _write(d / "drums.wav", drums)
    _write(d / "bass.wav", bass)
    BR.measure(d / "bass.wav", d / "bass_f0.json")
    return d


def test_the_boom_is_found_where_the_drums_ring_a_tone_at_its_own_pitch(stems):
    boom = KB.measure(stems / "drums.wav", stems / "bass.wav", stems / "bass_f0.json", BPM, 4, silence=0.1)
    assert [b.pitch is not None for b in boom] == [False, True, True, False]     # clicks alone are not a boom
    for b in boom[1:3]:
        assert abs(440.0 * 2 ** ((b.pitch - 69) / 12) - 45.0) < 3.0               # the drums' tone, not the bass at 73 Hz
        assert b.level > 0


def test_each_kick_starts_a_note_held_to_the_next_kick():
    boom = [KB.BoomBar(None, 0.0), KB.BoomBar(29.52, 1.2)]
    kicks = {0, 4, 16, 22, 28}                     # bar 0: no boom; bar 1 kicks on steps 0, 6, 12
    assert KB.bar_strings(boom, kicks, 16) == ["~ " * 15 + "~", "29.52@6 29.52@6 29.52@4"]
    assert KB.bar_strings([KB.BoomBar(30.0, 1.0)], {3}, 16) == ["~ ~ ~ 30@13"]   # rests before the first kick
    assert KB.bar_strings([KB.BoomBar(30.0, 1.0)], set(), 16) == ["~ " * 15 + "~"]  # a boom bar with no kick: nothing


def _cal(*extra: str, comparison: Path = FIX / "sem_alianca_synth_cal3.json") -> dict[str, str]:
    out = subprocess.run([sys.executable, str(CAL), "--comparison", str(comparison), "--mode", "synth",
                          "--lead-lpf", "6638", "--cal-lead", "0.535", "--sub-gain", "0.838", "--bass-mult", "1.3",
                          "--master-gain", "2.299", *extra], capture_output=True, text=True, check=True).stdout.split()
    return {out[i]: out[i + 1] for i in range(0, len(out), 2)}


def _solo(tmp_path: Path, orig: dict, rend: dict) -> Path:
    p = tmp_path / "bass.json"
    p.write_text(json.dumps({"master_gain": 2.299, "level_orig": orig, "level_render": rend,
                             "ratio": {b: orig[b] / rend[b] for b in orig}}))
    return p


def test_the_boom_scale_is_the_bass_voice_level_fraction_over_sub_and_bass(tmp_path):
    # Sem Aliança cal3's bass solo: render sub+bass 106.4e6 vs original 868.6e6 -> sqrt 0.35
    solo = _solo(tmp_path, {"sub_bass": 428584964.1, "bass": 440017117.4}, {"sub_bass": 59198836.0, "bass": 47160432.0})
    out = _cal("--sub-up", "0.898", "--sub-up-max", "0", "--bass-solo", str(solo))
    assert float(out["--kick-boom-scale"]) == pytest.approx(0.35, abs=0.005)
    # moving energy between the two bands (what f does) leaves it unchanged
    solo2 = _solo(tmp_path, {"sub_bass": 428584964.1, "bass": 440017117.4}, {"sub_bass": 26179634.0, "bass": 80179634.0})
    out2 = _cal("--sub-up", "0.898", "--sub-up-max", "0", "--bass-solo", str(solo2))
    assert float(out2["--kick-boom-scale"]) == pytest.approx(float(out["--kick-boom-scale"]), abs=1e-3)


def test_no_measurement_says_nothing_and_a_split_step_holds_the_scale(tmp_path):
    assert "--kick-boom-scale" not in _cal("--sub-up", "0.898", "--sub-up-max", "0")          # no bass solo, scale 0
    solo = _solo(tmp_path, {"sub_bass": 428584964.1, "bass": 440017117.4}, {"sub_bass": 59198836.0, "bass": 47160432.0})
    # sub/bass 60/28 vs the original's 45/45 at f = 0 asks for a split step: the scale is held (the split's render must
    # differ in f alone)
    d = json.loads((FIX / "sem_alianca_synth_cal3.json").read_text())
    d["rendered"]["bands"].update(sub_bass=0.60, bass=0.28)
    cmp = tmp_path / "c.json"
    cmp.write_text(json.dumps(d))
    out = _cal("--sub-up", "0", "--kick-boom-scale", "0.2", "--bass-solo", str(solo), comparison=cmp)
    assert out["--sub-up"] != "0" and float(out["--kick-boom-scale"]) == pytest.approx(0.2)


def test_the_boom_channel_is_a_drums_voice_for_the_solo_stems():
    # its label must be one token (solo_stems._LABEL_RE) - "KICK BOOM" failed stage 10 on Sem Aliança v009 - and the boom is
    # drums material: the bass solo (bass_solo.py, which sets the boom's own scale) must not hear it
    import solo_stems as SS
    lines = ['$: note(cat(...bass)).s("sine")  // BASS — low end',
             '$: note(cat(...boom)).s("sine").gain(0.5)  // KICK_BOOM — the 808 tone under the kick — edit `boom`']
    assert [(c.label, c.group) for c in SS.find_channels(lines)] == [("BASS", "bass"), ("KICK_BOOM", "drums")]


def test_the_level_is_the_fundamental_alone_absolute(stems, tmp_path):
    # the same boom with a strong 2nd harmonic (a saturated 808): the level stays the fundamental's, not the band's
    import soundfile as sf
    d, _ = sf.read(str(stems / "drums.wav"), always_2d=True)
    n, beat = d.shape[0], 60.0 / BPM
    extra = np.zeros(n)
    for b in range(4, 12):
        i, m = int(b * beat * SR), np.arange(int(beat * SR))
        extra[i:i + m.size] += 0.5 * np.exp(-m / (2.0 * SR)) * np.sin(2 * np.pi * 90.0 * m / SR)
    _write(tmp_path / "drums.wav", d[:, 0] + extra)
    pure = KB.measure(stems / "drums.wav", stems / "bass.wav", stems / "bass_f0.json", BPM, 4, silence=0.1)
    rich = KB.measure(tmp_path / "drums.wav", stems / "bass.wav", stems / "bass_f0.json", BPM, 4, silence=0.1)
    for p, r in zip(pure[1:3], rich[1:3]):
        assert r.level == pytest.approx(p.level, rel=0.15)
        assert p.level == pytest.approx(0.5 / np.sqrt(2), rel=0.25)     # the tone's own RMS (decay ~10 % over a beat)


def test_the_engine_sine_unit_is_recorded():
    u = KB.sine_unit()
    assert u is not None and 0.0 < u < 1.0, "run: kick_boom.py probe"
