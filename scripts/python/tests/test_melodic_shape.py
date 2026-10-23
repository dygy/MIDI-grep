# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""Lead + chords retrigger only where the melodic stem attacks, and hold between attacks (melodic_shape.py).

Regression (2026-10-23): the written lead was a 16th-note retrigger stream. On Regime CLT 1062 of 1126 consecutive
written notes were one step apart against 226 real stem onsets (80 % of written onsets had no stem onset; VAGABUNDO
61 %); swapping the original melodic stem into our mix lifted the mix onset F1 0.70 -> 0.93. The stem plays a note
and lets it ring: the written note must be struck once at the stem's attack and held (``c4@3``).

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_melodic_shape.py -q
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

BPM = 120.0                                    # step = 0.125 s, bar = 2 s


def _stem(tmp_path: Path, notes, sr=22050, dur=4.0) -> Path:
    """Sustained sine notes [(t0, t1, hz)], each with a noise-burst attack and a short release."""
    rng = np.random.default_rng(1)
    y = np.zeros(int(sr * dur))
    for t0, t1, hz in notes:
        i0, i1 = int(t0 * sr), int(t1 * sr)
        t = np.arange(i1 - i0) / sr
        seg = 0.4 * np.sin(2 * np.pi * hz * t)
        seg[:200] += 0.5 * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
        seg *= np.minimum(1.0, (len(seg) - np.arange(len(seg))) / 2000)
        y[i0:i1] += seg
    p = tmp_path / "melodic.wav"
    sf.write(str(p), y, sr)
    return p


def test_retrigger_stream_becomes_attack_notes_held_to_the_next_attack(tmp_path):
    import bass_shape as BS
    import melodic_shape as MS
    stem = _stem(tmp_path, [(0.25, 0.75, 440.0), (0.75, 1.25, 660.0)])  # two audible attacks, at steps 2 and 6
    lead = ["~ ~ a4 a4 a4 a4 e5 e5 e5 e5 ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]  # transcription: a trigger on EVERY step
    ev = BS.StemEvidence.load(stem)
    out, chords, info = MS.shape_melodic(lead, None, BPM, ev)
    toks = out[0].split()
    assert toks[2] == "a4@4" and toks[3] == "e5@4", out[0]    # the second token after a hold is the next slot
    assert info["lead"]["triggers_in"] == 8 and info["lead"]["notes"] == 2


def test_hold_stops_when_the_stem_goes_quiet_and_at_the_bar_end(tmp_path):
    import bass_shape as BS
    import melodic_shape as MS
    stem = _stem(tmp_path, [(0.25, 0.5, 440.0), (1.5, 3.5, 330.0)])    # a 2-step note; then a note over the bar line
    lead = ["~ ~ a4 a4 a4 a4 a4 a4 ~ ~ ~ ~ e4 e4 e4 e4", "e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4"]
    ev = BS.StemEvidence.load(stem)
    out, _, _ = MS.shape_melodic(lead, None, BPM, ev)
    toks = out[0].split()
    assert toks[2].startswith("a4") and int(toks[2].split("@")[1]) <= 4, out[0]     # quiet after ~2 steps, not 6
    assert toks[-1] == "e4@4", out[0]                                               # the bar end bounds the hold
    assert all(sum(int(t.split("@")[1]) if "@" in t else 1 for t in b.split()) == 16 for b in out)


def test_chords_follow_the_lead_attacks_and_end_at_the_next_one(tmp_path):
    import bass_shape as BS
    import melodic_shape as MS
    stem = _stem(tmp_path, [(0.25, 0.75, 440.0), (0.75, 1.25, 660.0)])
    lead = ["~ ~ a4 a4 a4 a4 e5 e5 e5 e5 ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]
    chords = ["~ ~ [c4,e4] [c4,e4] ~ ~ [d4,f4] ~ [d4,f4] ~ ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]
    ev = BS.StemEvidence.load(stem)
    _, c, _ = MS.shape_melodic(lead, chords, BPM, ev)
    toks = c[0].split()
    assert toks[2] == "[c4,e4]@4" and toks[3] == "[d4,f4]@4", c[0]     # struck on the lead's attacks, ended at the next
    assert sum(1 for t in toks if t != "~") == 2                       # the chords between attacks are gone


def test_a_stem_without_strong_onsets_leaves_the_voice_as_transcribed():
    import bass_shape as BS
    import melodic_shape as MS
    ev = BS.StemEvidence(np.zeros(22050))
    lead = ["a4 a4 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"]
    out, _, _ = MS.shape_melodic(lead, None, BPM, ev)
    assert len(out) == 1 and out[0].split()[0].startswith("a4")


def _legato_stem(tmp_path, second_hz, sr=22050):
    """One struck 440 Hz note (attack burst at 0.25 s) that glides to ``second_hz`` between 0.65 and 0.85 s and rings
    on until 1.25 s: the pitch moves, nothing strikes."""
    rng = np.random.default_rng(1)
    t = np.arange(int(sr * 4.0)) / sr
    f = np.interp(t, [0.65, 0.85], [440.0, second_hz])
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = 0.4 * np.sin(phase) * ((t >= 0.25) & (t < 1.25))
    i0 = int(0.25 * sr)
    y[i0:i0 + 200] += 0.5 * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    y[int(1.25 * sr) - 2000:int(1.25 * sr)] *= np.linspace(1, 0, 2000)
    p = tmp_path / "melodic.wav"
    sf.write(str(p), y, sr)
    return p


def test_a_pitch_change_inside_a_hold_is_written_as_a_legato_note_not_dropped(tmp_path):
    """Regression (2026-10-23): ``c4@3`` held one pitch to the next attack. Where the transcription changes pitch
    inside the hold and the stem confirms it (no new onset: legato/slide) the change was dropped; on the steps it
    concerned the stem's chroma favoured the dropped pitch 71 % of the time (Regime CLT, VAGABUNDO 68 %)."""
    import melodic_shape as MS
    stem = _legato_stem(tmp_path, 494.0)                           # B4 continues the A4 without an attack
    lead = ["~ ~ a4 a4 a4 a4 b4 b4 b4 b4 ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]
    out, _, info = MS.shape_melodic(lead, None, BPM, MS.load_evidence(stem))
    assert out[0].split()[2] == "a4@4", out[0]                      # struck once, ends where the pitch changes
    leg = info["lead"]["legato_bars"]
    assert leg is not None and leg[0].split()[6].startswith("b4@"), leg
    assert info["lead"]["legato_notes"] == 1
    assert out[0].count("a4") == 1 and out[0].count("b4") == 0      # the legato pitch is NOT a second attack


def test_a_transcribed_pitch_the_stem_does_not_play_is_not_written(tmp_path):
    import melodic_shape as MS
    stem = _legato_stem(tmp_path, 440.0)                           # the stem keeps the A4; the b4 is a flicker
    lead = ["~ ~ a4 a4 a4 a4 b4 b4 b4 b4 ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]
    out, _, info = MS.shape_melodic(lead, None, BPM, MS.load_evidence(stem))
    assert info["lead"]["legato_bars"] is None and info["lead"]["legato_notes"] == 0
    assert "b4" not in out[0]


def test_no_pitch_change_means_no_legato_voice(tmp_path):
    import melodic_shape as MS
    stem = _legato_stem(tmp_path, 440.0)
    lead = ["~ ~ a4 a4 a4 a4 a4 a4 ~ ~ ~ ~ ~ ~ ~ ~", "~ " * 15 + "~"]
    _, _, info = MS.shape_melodic(lead, None, BPM, MS.load_evidence(stem))
    assert info["lead"]["legato_bars"] is None and info["lead"]["legato_notes"] == 0


# ---- generator output (needs the Regime CLT cache) ----------------------------------------------------------
GEN = SCRIPTS / "generate_dynamic_strudel.py"
ST = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
BASE = "https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/regime-clt"
have = pytest.mark.skipif(not all((ST / p).exists() for p in ["bass.wav", "melodic.wav", "sample_pack/bass.mid",
                                                              "sample_pack/melodic.mid", "sample_pack/drums_bands.json",
                                                              ".pipeline/bass_f0.json",
                                                              "sample_pack/instruments/samples.json"]),
                          reason="Regime CLT cache absent")


def _generate(tmp_path, *extra, mode="sample-instrument"):
    pack = ST / "sample_pack"
    out = tmp_path / "o.strudel"
    cmd = [sys.executable, str(GEN), "--stems-dir", str(ST), "--pack-dir", str(pack),
           "--bass-midi", str(pack / "bass.mid"), "--lead-midi", str(pack / "melodic.mid"),
           "--drums-json", str(pack / "drums_bands.json"), "--base-url", BASE, "--mode", mode,
           "--bpm", "135.017", "--key", "C# minor", "--genre", "brazilian_funk", "--num-bars", "77",
           "--drum-mode", "extracted", *extra, "--out", str(out)]
    if mode == "sample-instrument":
        cmd += ["--samples-url", f"{BASE}/instruments/samples.json",
                "--bass-sound", "regime_clt_dj_brunin_xm_aurora_shukita_bass",
                "--lead-sound", "regime_clt_dj_brunin_xm_aurora_shukita_lead"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    return out.read_text(), json.loads(r.stdout[r.stdout.index("{"):])


def _voice(code, name):
    body = code.split(f"let {name} = [")[1].split("\n]")[0]
    return [t for t in re.findall(r"\[[^\]]*\](?:@\d+)?|[a-g]s?-?\d+(?:@\d+)?|~", body) if t != "~"]


@have
def test_generated_lead_is_struck_on_stem_attacks_and_held(tmp_path):
    code, summary = _generate(tmp_path)
    toks = _voice(code, "lead")
    held = [t for t in toks if "@" in t]
    assert len(held) > 0.4 * len(toks), (len(held), len(toks))
    assert len(toks) < 0.5 * summary["melodic_shape"]["lead"]["triggers_in"]       # the retrigger stream is gone
    assert "withLeadLoop(" in code and "note(cat(...lead))" in code
    assert summary["editability"] == "pass"


@have
def test_old_lead_behaviour_is_one_flag_away(tmp_path):
    code, _ = _generate(tmp_path, "--melodic-shape", "off")
    assert not any("@" in t for t in _voice(code, "lead")) and "leadLoop" not in code



@have
def test_generated_piece_writes_pitch_changes_as_an_editable_legato_voice(tmp_path):
    code, summary = _generate(tmp_path)
    assert "let leadLegato = [" in code and "noLoopAttack(note(cat(...leadLegato)))" in code
    assert "const legatoAttack = " in code
    assert "melodic_shape" in summary and "legato_bars" not in summary["melodic_shape"]["lead"]   # arrays live in code
    assert summary["editability"] == "pass"
