# @layer: unit
# @regression
"""Loudness of the generated Strudel is matched to the original by measurement (mix_dynamics.py).

The similarity score is level-invariant (compare_audio.rms_normalize), so before this module a render could be 6-13 dB
quieter and 6-8 dB peakier (crest) than the mastered original and score the same. The tests pin what makes the match
derived instead of typed in: the profile of a signal, the verdict against the reference (level, crest), the
next master gain, the accept rule, and the auto-calibrate stage that moves --master-gain onto the reference RMS.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_mix_dynamics.py -q
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

import mix_dynamics as M  # noqa: E402

SR = 8000


def tone(amp: float, seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * 220 * t)


def test_profile_reports_rms_peak_crest_and_the_short_term_distribution():
    p = M.profile(tone(0.5), SR)
    assert p["rms_db"] == pytest.approx(20 * math.log10(0.5 / math.sqrt(2)), abs=0.05)
    assert p["peak_db"] == pytest.approx(20 * math.log10(0.5), abs=0.05)
    assert p["crest_db"] == pytest.approx(3.01, abs=0.1)
    assert p["st_p10_db"] == pytest.approx(p["st_p90_db"], abs=0.1)           # a steady tone has no loudness spread
    assert p["clip_frac"] == 0.0


def test_profile_counts_samples_at_full_scale_and_near_the_signals_own_peak():
    p = M.profile(np.clip(tone(2.0), -1.0, 1.0), SR)                           # a tone driven 6 dB into the rails
    assert p["peak_db"] == pytest.approx(0.0, abs=0.01)
    assert p["clip_frac"] == pytest.approx(2 / 3, abs=0.01)                    # |2 sin| >= 1 for 120 of every 180 degrees
    assert p["plateau_frac"] >= p["clip_frac"]                                 # everything clipped is within PLATEAU_DB of the peak


def test_level_verdict_needs_the_reference_rms_and_crest():
    ref = {"rms_db": -11.0, "crest_db": 10.0, "peak_db": -1.0, "clip_frac": 0.0, "plateau_frac": 0.012}
    good = {"rms_db": -11.5, "crest_db": 11.0, "peak_db": 0.0, "clip_frac": 0.010}
    assert M.level_verdict(ref, good)["ok"]
    assert not M.level_verdict(ref, {**good, "rms_db": -18.6})["ok"]          # the reported 7 dB-quiet render
    assert not M.level_verdict(ref, {**good, "crest_db": 17.9})["ok"]         # the reported 17-18 dB crest


def test_next_master_closes_the_rms_error_in_dB():
    ref = {"rms_db": -11.8}
    assert M.next_master(ref, {"rms_db": -19.0}, 0.78) == pytest.approx(0.78 * 10 ** (7.2 / 20), rel=1e-3)
    # a render that is louder than the reference is turned DOWN (the calibrator's own master lever can only go up)
    assert M.next_master(ref, {"rms_db": -9.8}, 2.0) == pytest.approx(2.0 * 10 ** (-2.0 / 20), rel=1e-3)


def test_choose_round_takes_the_best_qualifying_round_and_none_when_the_score_loss_is_beyond_noise():
    ok, bad = {"ok": True, "problems": []}, {"ok": False, "problems": ["rms"]}
    best = {"overall": 0.80, "section": 0.75}
    rounds = [{"level": bad, "overall": 0.81, "section": 0.76},
              {"level": ok, "overall": 0.797, "section": 0.748},
              {"level": ok, "overall": 0.801, "section": 0.746}]
    assert M.choose_round(best, rounds, 0.005) == 2                           # highest overall among the qualifying
    assert M.choose_round(best, [{"level": ok, "overall": 0.80, "section": 0.74}], 0.005) is None   # section -0.010
    assert M.choose_round(best, [{"level": bad, "overall": 0.80, "section": 0.75}], 0.005) is None


CODE = """// header
setcps(0.5)

$: stack(
  note(cat(...bass)).s("b").gain(0.1),
  note(cat(...bass)).s("sine").gain(0.2)
)  // BASS — low end

$: note(cat(...lead)).s("l").gain(0.3)  // LEAD — melody

$: note(cat(...chords)).s("l").gain(0.1)  // CHORDS — harmony

$: s("bd*4").gain(0.4)  // DRUMS — groove
"""


def test_split_voices_keeps_one_voice_per_program_and_melodic_is_lead_plus_chords():
    voices = M.split_voices(CODE)
    active = lambda code: [ln for ln in code.splitlines() if ln.startswith("$:")]
    assert set(voices) == {"bass", "melodic", "drums"}
    assert active(voices["bass"]) == ["$: stack("]
    assert [("LEAD" in ln, "CHORDS" in ln) for ln in active(voices["melodic"])] == [(True, False), (False, True)]
    assert active(voices["drums"]) == ['$: s("bd*4").gain(0.4)  // DRUMS — groove']


def test_share_errors_are_the_renders_energy_share_over_the_originals_in_dB():
    o = M.energy_shares({"bass": 0.5, "melodic": 0.25, "drums": 0.25})
    r = M.energy_shares({"bass": 0.25, "melodic": 0.5, "drums": 0.25})
    err = M.share_errors_db(o, r)
    assert err["bass"] == pytest.approx(-3.01, abs=0.01) and err["melodic"] == pytest.approx(3.01, abs=0.01)
    assert err["drums"] == pytest.approx(0.0, abs=1e-9)


# ---- the auto-calibrate.sh stage moves --master-gain onto the reference RMS (stubbed generator / renderer / scorer) ----
def _wav(path: Path, y: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), y.astype(np.float32), 44100, subtype="PCM_16")


def run_stage(tmp_path: Path, *, score_after: float, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    """auto-calibrate.sh with one calibration iteration followed by the loudness stage. The stub generator writes its
    --master-gain into the piece, the stub renderer writes a tone whose amplitude is proportional to it (clipped at full
    scale like the 16-bit writer), the stub scorer returns 0.80 for the loop's gain and ``score_after`` for a louder one."""
    stems = tmp_path / "stems"
    (stems / "sample_pack").mkdir(parents=True)
    _wav(stems / "instrumental.wav", tone(0.70, 12.0))                          # the "mastered" reference
    py = sys.executable
    stubs = {
        "gen.py": """
import sys
a = sys.argv
open(a[a.index('--out') + 1], 'w').write('MG=' + a[a.index('--master-gain') + 1] + '\\n')
print('{"bass_lpf":0,"bass_lpf_analysis":0,"bass_note_max_hz":0,"lead_hpf":0,"lead_note_min_hz":0,"sub_up":0}')
""",
        "rec.py": """
import sys, numpy as np, soundfile as sf
src, out = sys.argv[1], sys.argv[sys.argv.index('-o') + 1]
mg = float(open(src).read().split('=')[1])
t = np.arange(8000 * 12) / 8000
sf.write(out, np.clip(0.2 * mg * np.sin(2 * np.pi * 220 * t), -1, 1).astype(np.float32), 44100, subtype='PCM_16')
""",
        "cmp.py": f"""
import json, sys
mg = float(open(sys.argv[2].replace('.wav', '.strudel')).read().split('=')[1])
score = 0.80 if mg <= 0.8 else {score_after}
print(json.dumps({{'comparison': {{'overall_similarity': score, 'section_aware_similarity': score}}}}))
""",
        "cal.py": "print('--bass-mult 0.4 --sub-gain 0.7 --cal-lead 1.0 --lead-lpf 5000 --hat-gain 0 --master-gain 0.78')\n",
        "gate.py": "print('0.1 PASS')\n",
    }
    for name, text in stubs.items():
        (tmp_path / name).write_text(text)
    env = {"PATH": "/usr/bin:/bin", "AC_PY": py, "AC_GEN": str(tmp_path / "gen.py"), "AC_CMP": str(tmp_path / "cmp.py"),
           "AC_CAL": str(tmp_path / "cal.py"), "AC_GATE": str(tmp_path / "gate.py"), "AC_REC": str(tmp_path / "rec.py"),
           "AC_NODE": py, "AC_SETTLE_S": "0", "AC_PGREP": "false", "AC_RECORDER_WAIT_S": "0",
           "AC_REF": str(stems / "instrumental.wav"), "AC_NOISE": "0.005"}
    return subprocess.run(["bash", str(REPO / "scripts" / "auto-calibrate.sh"), "--stems", str(stems), "--base", "http://x/",
                           "--bpm", "120", "--bars", "4", "--iters", "1", "--dur", "12", "--workdir", str(tmp_path / "w"),
                           "--start", "0.4 0.7 1.0 5000 0 0.78 0 0 0", *(extra or [])],
                          capture_output=True, text=True, env=env, cwd=str(REPO))


def test_stage_raises_a_quiet_render_onto_the_reference_rms_and_keeps_it(tmp_path):
    r = run_stage(tmp_path, score_after=0.80)
    assert "[mix] kept round" in r.stdout, r.stdout + r.stderr
    kept_mg = float((tmp_path / "w" / "best.master").read_text())
    assert (tmp_path / "w" / "best.strudel").read_text().strip() == f"MG={kept_mg:g}"
    assert kept_mg > 0.78 * 2                                                  # the loop's gain gave a render ~13 dB under the reference
    ref = M.profile(*M.load_mono(tmp_path / "stems" / "instrumental.wav"))
    got = M.profile(*M.load_mono(tmp_path / "w" / "best.wav"))
    assert abs(got["rms_db"] - ref["rms_db"]) <= M.RMS_TOL_DB


def test_stage_leaves_the_unmixed_best_when_the_louder_piece_scores_worse_than_the_noise(tmp_path):
    r = run_stage(tmp_path, score_after=0.70)
    assert "the unmixed BEST stays" in r.stdout, r.stdout + r.stderr
    assert (tmp_path / "w" / "best.strudel").read_text().strip() == "MG=0.78"
    assert not (tmp_path / "w" / "best.master").exists()


def test_stage_can_be_switched_off(tmp_path):
    r = run_stage(tmp_path, score_after=0.80, extra=["--no-mix-dynamics"])
    assert "[mix]" not in r.stdout and (tmp_path / "w" / "best.strudel").read_text().strip() == "MG=0.78", r.stdout + r.stderr
