# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Bass note shape measured from the bass stem (bass_shape.py).

Regression (2026-10-20): the original bass is one continuous 808 tone; the generator wrote one 1-step hit per
transcribed trigger (VAGABUNDO: 650 triggers vs 203 audible stem attacks), each re-triggering a <= 0.4 s grain.
Held notes, measured glides, sample loop points and a kick sidechain now come from the stem itself.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_bass_shape.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

import bass_shape as BS  # noqa: E402


# ---- hold ---------------------------------------------------------------------------------------------------
def _stem(tmp_path: Path, notes, sr=22050, dur=2.0) -> Path:
    """Sustained sine notes [(t0, t1, hz)] each with a short noise-burst attack."""
    rng = np.random.default_rng(1)
    y = np.zeros(int(sr * dur))
    for t0, t1, hz in notes:
        i0, i1 = int(t0 * sr), int(t1 * sr)
        t = np.arange(i1 - i0) / sr
        seg = 0.4 * np.sin(2 * np.pi * hz * t)
        seg[:200] += 0.5 * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
        seg *= np.minimum(1.0, (len(seg) - np.arange(len(seg))) / 400)      # short release
        y[i0:i1] += seg
    p = tmp_path / "bass.wav"
    sf.write(str(p), y, sr)
    return p


def test_unreattacked_same_pitch_triggers_become_one_held_note(tmp_path):
    bpm = 120.0                                    # step = 0.125 s
    stem = _stem(tmp_path, [(0.0, 0.5, 55.0), (0.5, 1.0, 82.4)])
    # the transcription wrote every 16th: A1 on steps 0-3, E2 on 4-7 (8 triggers, 2 audible attacks)
    bars = ["a1 a1 a1 a1 e2 e2 e2 e2 ~ ~ ~ ~ ~ ~ ~ ~"]
    ev = BS.StemEvidence.load(stem)
    notes, info = BS.shape_bars(bars, bpm, ev, None)
    out = BS.format_bars(notes, 1, 16, lambda p: str(p))[0].split()
    assert info["triggers_in"] == 8 and len(notes) == 2
    assert [n.length for n in notes] == [4, 4]
    assert out[0].endswith("@4") and out[1].endswith("@4")


def test_hold_never_crosses_the_bar_end(tmp_path):
    stem = _stem(tmp_path, [(0.0, 2.0, 55.0)])
    ev = BS.StemEvidence.load(stem)
    notes, _ = BS.shape_bars(["~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ a1 a1 ~ ~", "a1 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"], 120.0, ev, None)
    first = notes[0]
    assert first.step + first.length <= 16


# ---- glide --------------------------------------------------------------------------------------------------
def _f0(frames):
    midi = np.array(frames, dtype=float)
    times = np.arange(len(midi)) * 0.02
    return midi, times, ~np.isnan(midi)


def _two_notes(t_b):
    a = BS.Note(step=0, pitch=33, length=4, t=0.0, attack=True)
    b = BS.Note(step=4, pitch=36, length=4, t=t_b, attack=False)
    return [a, b]


def test_glide_time_is_the_measured_transition_duration():
    # leaves A1 (33) at frame 9, passes 34 and 35, arrives at 36 at frame 13 -> 4 hops = 0.08 s
    midi, times, act = _f0([33] * 10 + [34, 35] + [36] * 12)
    notes = _two_notes(t_b=0.22)
    assert BS.measure_glides(notes, midi, times, act, step_dur=0.125) == 1
    assert notes[1].glide == pytest.approx(0.06, abs=0.021) and notes[1].proven


def test_pitch_step_without_intermediate_frames_is_not_a_glide():
    midi, times, act = _f0([33] * 11 + [36] * 12)
    notes = _two_notes(t_b=0.22)
    assert BS.measure_glides(notes, midi, times, act, step_dur=0.125) == 0
    assert notes[1].glide == 0.0 and notes[1].proven


def test_glide_pattern_is_parallel_to_the_note_pattern():
    notes = [BS.Note(0, 33, 2, t=0.0), BS.Note(2, 36, 2, glide=0.06, t=0.1), BS.Note(8, 33, 8, t=0.5)]
    name = lambda p: f"n{p}"  # noqa: E731
    assert BS.format_bars(notes, 1, 16, name)[0] == "n33@2 n36@2 ~ ~ ~ ~ n33@8"
    assert BS.format_bars(notes, 1, 16, name, glide=True)[0] == "0@2 0.06@2 ~ ~ ~ ~ 0@8"


# ---- loop points --------------------------------------------------------------------------------------------
def test_loop_is_whole_periods_between_zero_crossings_after_the_attack():
    sr, hz = 44100, 40.0
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    y = 0.5 * np.sin(2 * np.pi * hz * t + 0.3)
    y[:800] += np.random.default_rng(0).standard_normal(800) * 0.6 * np.exp(-np.arange(800) / 150)   # attack
    y *= np.minimum(1.0, (n - np.arange(n)) / 3000)
    lp = BS.loop_points(y, sr, hz)
    assert lp is not None
    b, e = lp[0] * n, lp[1] * n
    period = sr / hz
    assert b > 800                                           # after the attack
    assert (e - b) / period == pytest.approx(round((e - b) / period), abs=0.06)   # whole periods
    assert y[int(round(b)) - 1] < 0 <= y[int(round(b))]      # upward zero crossing


def test_grain_without_a_steady_part_has_no_loop():
    sr = 44100
    y = np.sin(2 * np.pi * 40 * np.arange(int(0.04 * sr)) / sr)   # ~1.6 periods
    assert BS.loop_points(y, sr, 40.0) is None


# ---- sidechain ----------------------------------------------------------------------------------------------
def _rms(dip_db: float, hop=0.005, dur=60.0, ioi=0.5, dip_s=0.12):
    n = int(dur / hop)
    db = np.full(n, -12.0) + np.random.default_rng(3).normal(0, 0.3, n)
    for k in np.arange(1.0, dur - 1, ioi):
        i = int(k / hop)
        db[i:i + int(dip_s / hop)] += dip_db
    return 10 ** (db / 20), hop, list(np.arange(1.0, dur - 1, ioi))


def test_kick_dip_in_the_stem_gives_a_measured_duck():
    rms, hop, kicks = _rms(-6.0)
    d = BS.measure_duck(rms, hop, kicks)
    assert d is not None
    assert d.depth_db == pytest.approx(6.0, abs=1.0)
    assert 0 <= d.onset_s <= 0.12 and 0.02 < d.attack_s < 0.5
    g = 10 ** (-d.depth_db / 20)
    assert d.depth == pytest.approx((1 - g) ** 2, abs=0.01)    # Strudel: gain = 1 - sqrt(depth)


def test_no_dip_means_no_ducking():
    rms, hop, kicks = _rms(0.0)
    assert BS.measure_duck(rms, hop, kicks) is None


# ---- generator output ---------------------------------------------------------------------------------------
GEN = SCRIPTS / "generate_dynamic_strudel.py"
PIECES = {
    "regime": dict(stems="Regime CLT (Dj Brunin XM, Aurora Shukita)", bpm="135.017", bars="77", sound="regime",
                   base="https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/regime-clt"),
    "vag": dict(stems="VAGABUNDO NÃO NAMORA", bpm="129.989", bars="81", sound="vagabundo_nao_namora",
                base="https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/vagabundo-nao-namora"),
}


def _have(piece):
    st = REPO / ".cache" / "stems" / PIECES[piece]["stems"]
    return all(p.exists() for p in [st / "bass.wav", st / "melodic.wav", st / "drums.wav", st / "sample_pack" / "bass.mid",
                                    st / "sample_pack" / "melodic.mid", st / "sample_pack" / "drums_bands.json",
                                    st / ".pipeline" / "bass_f0.json", st / "sample_pack" / "instruments" / "samples.json"])


regime = pytest.mark.skipif(not _have("regime"), reason="Regime CLT cache absent")
vag = pytest.mark.skipif(not _have("vag"), reason="VAGABUNDO cache absent")


def _generate(tmp_path, *extra, mode="sample-instrument", piece="regime"):
    pc = PIECES[piece]
    st = REPO / ".cache" / "stems" / pc["stems"]
    pack = st / "sample_pack"
    out = tmp_path / f"{piece}-{mode}.strudel"
    cmd = [sys.executable, str(GEN), "--stems-dir", str(st), "--pack-dir", str(pack),
           "--bass-midi", str(pack / "bass.mid"), "--lead-midi", str(pack / "melodic.mid"),
           "--drums-json", str(pack / "drums_bands.json"), "--base-url", pc["base"],
           "--mode", mode, "--bpm", pc["bpm"], "--key", "C# minor", "--genre", "brazilian_funk",
           "--num-bars", pc["bars"], "--drum-mode", "extracted", *extra, "--out", str(out)]
    if mode == "sample-instrument":
        cmd += ["--samples-url", f"{pc['base']}/instruments/samples.json",
                "--bass-sound", f"{pc['sound']}_bass", "--lead-sound", f"{pc['sound']}_lead"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    return out.read_text(), json.loads(r.stdout[r.stdout.index("{"):])


def _bass_tokens(code):
    body = code.split("let bass = [")[1].split("]")[0]
    return [t for t in re.findall(r"[a-g]s?-?\d+(?:@\d+)?|~", body) if t != "~"]


@regime
def test_generated_bass_is_held_notes_not_one_step_hits(tmp_path):
    code, summary = _generate(tmp_path)
    toks = _bass_tokens(code)
    held = [t for t in toks if "@" in t]
    assert len(held) > 0.25 * len(toks)
    assert summary["bass_shape"]["notes"] < 0.8 * summary["bass_shape"]["triggers_in"]     # repeats merged
    assert summary["editability"] == "pass"


@regime
def test_old_behaviour_is_one_flag_away(tmp_path):
    code, _ = _generate(tmp_path, "--bass-shape", "off", "--bass-duck", "off")
    assert not any("@" in t for t in _bass_tokens(code))
    assert "withLoop" not in code and "duckorbit" not in code


@regime
def test_no_duck_where_the_stem_has_none(tmp_path):
    code, summary = _generate(tmp_path)
    assert summary["bass_duck"] is None and "duckorbit" not in code      # Regime's bass swells at the kick


@vag
def test_sample_mode_emits_loop_table_and_duck_measured_from_the_stem(tmp_path):
    code, summary = _generate(tmp_path, piece="vag")
    assert "const bassLoop = {" in code and "loopBegin" in code
    assert summary["bass_loops"]["pitches_looped"] >= 1
    d = summary["bass_duck"]
    assert d and d["depth_db"] >= BS.DUCK_MIN_DB
    assert f".duckdepth({d['depth']})" in code and f".duckattack({d['attack_s']})" in code
    assert code.count(".orbit(2)") == 3 and "note(\"c1\").s(\"sine\").orbit(2).gain(0)" in code
    assert "typeof Pattern.prototype.glide" in code or "glide(" not in code      # glide guard when glide is used
    assert summary["editability"] == "pass"


@regime
def test_synth_mode_gets_held_notes_without_sample_loops(tmp_path):
    code, summary = _generate(tmp_path, mode="synth")
    assert any("@" in t for t in _bass_tokens(code))
    assert "bassLoop" not in code and "samples(" not in code


@regime
def test_synth_bass_lpf_keeps_the_oscillator_derivation(tmp_path):
    """A raw sawtooth's harmonics are not the stem's: the sounding-energy rolloff (3284 Hz on Regime) is for the
    sample layer only; synth mode keeps the median 85 % rolloff over all frames (538 Hz)."""
    code, summary = _generate(tmp_path, mode="synth")
    assert summary["bass_lpf_analysis"] == 538
    assert ".lpf(538)" in code and ".lpf(3284)" not in code


@regime
def test_sample_layer_lpf_follows_the_sounding_energy(tmp_path):
    _, summary = _generate(tmp_path)
    assert summary["bass_lpf_analysis"] == 3284


# ---- attacks follow the stem ---------------------------------------------------------------------------------
def _register(spans, hop=0.01, dur=2.0):
    """A bass_register.Register whose f0 track holds [(t0, t1, midi)]."""
    import bass_register as BR
    n = int(dur / hop)
    midi, rms = [None] * n, [0.0] * n
    for t0, t1, m in spans:
        for i in range(int(t0 / hop), min(n, int(t1 / hop))):
            midi[i], rms[i] = float(m), 0.3
    return BR.Register({"hop_s": hop, "midi": midi, "rms": rms})


def _add_ripples(path, times, amp=0.03):
    y, sr = sf.read(str(path))
    rng = np.random.default_rng(2)
    for t in times:
        i = int(t * sr)
        y[i:i + 200] += amp * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    sf.write(str(path), y, sr)


def test_a_strong_stem_attack_the_transcription_missed_is_written_with_the_f0_pitch(tmp_path):
    bpm = 120.0                                    # step = 0.125 s
    # the stem re-attacks A1 at steps 0, 4, 8, 12 (a gap between each); the transcription has no note at step 8
    stem = _stem(tmp_path, [(0.0, 0.4, 55.0), (0.5, 0.9, 55.0), (1.0, 1.4, 55.0), (1.5, 1.9, 55.0)])
    _add_ripples(stem, [0.25, 0.75, 1.25, 1.7])      # weak bumps inside the notes: the stem's non-attack onset class
    reg = _register([(0.0, 0.4, 33), (0.5, 0.9, 33), (1.0, 1.4, 33), (1.5, 1.9, 33)])
    ev = BS.StemEvidence.load(stem)
    notes, info = BS.shape_bars(["a1 ~ ~ ~ a1 ~ ~ ~ ~ ~ ~ ~ a1 ~ ~ ~"], bpm, ev, reg)
    by_step = {n.step: n for n in notes}
    assert 8 in by_step, f"the stem's attack at step 8 (t=1.0 s) was not written; written steps {sorted(by_step)}"
    assert by_step[8].attack and by_step[8].pitch == 33
    assert info["attacks_added"] == 1


def test_attack_time_is_the_rise_not_the_backtracked_minimum(tmp_path):
    # a note that swells for 60 ms before its sharp rise: the attack is where the level jumps (steepest rise), 1.0 s
    sr = 22050
    y = np.zeros(int(sr * 2.0))
    t = np.arange(len(y)) / sr
    env = np.where(t < 0.94, 0.0, np.where(t < 1.0, 0.03, np.where(t < 1.4, 0.5, 0.0)))
    env = np.convolve(env, np.ones(int(0.004 * sr)) / int(0.004 * sr), mode="same")
    y = env * np.sin(2 * np.pi * 55.0 * t)
    sf.write(str(tmp_path / "bass.wav"), y, sr)
    ev = BS.StemEvidence.load(tmp_path / "bass.wav")
    assert float(np.min(np.abs(ev.peak_attacks - 1.0))) < 0.02, ev.peak_attacks
