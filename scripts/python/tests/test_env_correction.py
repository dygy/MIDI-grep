# @layer: integration
# @spec: 004-second-reference-track
# @regression
"""Bug #10 — per-section bass/lead balance via ``--env-correction`` (consumer side).

``generate_dynamic_strudel.py --env-correction PATH`` multiplies each voice's per-bar gain
envelope by a measured per-bar multiplier (JSON from ``calibrate_dynamic.py
--env-correction-out``) BEFORE ``gain_pattern`` folds base*env into the single ``.gain("<..>")``.

Generation only (no Demucs / render). Skipped when the Regime CLT sample pack is absent.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_env_correction.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

from editability_check import check_editability  # noqa: E402
from generate_dynamic_strudel import (  # noqa: E402
    ENV_CORRECTION_CLAMP, apply_env_correction, compose_env_multipliers, load_env_correction,
)

GEN = SCRIPTS / "generate_dynamic_strudel.py"
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
BASE = "https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/regime-clt"
NBARS = 8
SPLIT = 4          # bars [0, SPLIT) get the boost, the rest the cut
BOOST, CUT = 1.3, 0.8

_REQUIRED = [
    STEMS / "vocals.wav", STEMS / "bass.wav", STEMS / "melodic.wav", STEMS / "drums.wav",
    PACK / "bass.mid", PACK / "melodic.mid", PACK / "drums_bands.json", PACK / "vocals.mid",
    PACK / "strudel.json",
]
needs_pack = pytest.mark.skipif(not all(p.exists() for p in _REQUIRED),
                                reason="Regime CLT sample_pack not present in .cache")

ARGS = [
    "--stems-dir", str(STEMS), "--pack-dir", str(PACK),
    "--bass-midi", str(PACK / "bass.mid"), "--lead-midi", str(PACK / "melodic.mid"),
    "--drums-json", str(PACK / "drums_bands.json"), "--base-url", BASE,
    "--mode", "synth", "--bpm", "136", "--key", "C# minor", "--genre", "brazilian_funk",
    "--num-bars", str(NBARS), "--drum-mode", "bank", "--vocal-mode", "instrument",
]


def _run(out: Path, *extra: str) -> tuple[subprocess.CompletedProcess, str]:
    proc = subprocess.run([sys.executable, str(GEN), *ARGS, *extra, "--out", str(out)],
                          capture_output=True, text=True, timeout=600)
    return proc, (out.read_text(encoding="utf-8") if out.exists() else "")


def _write_env(path: Path, *, bars: int = NBARS, bass: list[float] | None = None) -> Path:
    bass = bass or [BOOST] * SPLIT + [CUT] * (bars - SPLIT)
    path.write_text(json.dumps({
        "window_s": 10.0, "bpm": 136.0, "bars": bars, "source": "hand-written test fixture",
        "voices": {"bass": bass, "lead": [1.0] * bars, "master": [1.0] * bars},
    }))
    return path


def _gain_groups(code: str, anchor: str) -> list[list[float]]:
    """Per-bar step values of the ``.gain("<..>")`` on the first line containing ``anchor``."""
    line = next(ln for ln in code.splitlines() if anchor in ln and '.gain("<' in ln)
    pat = re.search(r'\.gain\("<(.*?)>"\)', line).group(1)
    return [[float(x) for x in g.split()] for g in re.findall(r"\[([^\]]*)\]", pat)]


BASS_ANCHOR = 'note(cat(...bass)).s("'      # the bass line (the sub layer is `.s("sine")`)
LEAD_ANCHOR = "note(cat(...lead))"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not all(p.exists() for p in _REQUIRED):
        pytest.skip("Regime CLT sample_pack not present in .cache")
    d = tmp_path_factory.mktemp("env_correction")
    plain = _run(d / "plain.strudel")
    corr_json = _write_env(d / "corr.env.json")
    corrected = _run(d / "corrected.strudel", "--env-correction", str(corr_json))
    return {"plain": plain, "corrected": corrected, "json": corr_json, "dir": d}


def test_bass_bars_scale_by_exactly_the_multipliers(runs):
    assert runs["plain"][0].returncode == 0, runs["plain"][0].stderr[-1500:]
    assert runs["corrected"][0].returncode == 0, runs["corrected"][0].stderr[-1500:]
    base = _gain_groups(runs["plain"][1], BASS_ANCHOR)
    corr = _gain_groups(runs["corrected"][1], BASS_ANCHOR)
    assert len(base) == len(corr) == NBARS
    checked = 0
    for b, (bs, cs) in enumerate(zip(base, corr)):
        factor = BOOST if b < SPLIT else CUT
        for v0, v1 in zip(bs, cs):
            if v0 == 0.0:
                assert v1 == 0.0, f"bar {b}: hard silence must stay 0"
            else:
                assert v1 == pytest.approx(v0 * factor, abs=2e-3), f"bar {b}: {v0} -> {v1}"
                checked += 1
    assert checked > 0


def test_sub_layer_shares_the_bass_envelope(runs):
    base = _gain_groups(runs["plain"][1], '.s("sine")')
    corr = _gain_groups(runs["corrected"][1], '.s("sine")')
    for b, (bs, cs) in enumerate(zip(base, corr)):
        for v0, v1 in zip(bs, cs):
            assert v1 == pytest.approx(v0 * (BOOST if b < SPLIT else CUT), abs=2e-3)


def test_lead_unchanged_when_its_multipliers_are_one(runs):
    assert (_gain_groups(runs["plain"][1], LEAD_ANCHOR)
            == _gain_groups(runs["corrected"][1], LEAD_ANCHOR))


def test_header_summary_and_editability(runs):
    proc, code = runs["corrected"]
    assert "// env_correction: corr.env.json" in code.splitlines()[:12]
    assert "env_correction" not in runs["plain"][1]
    summary = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert summary["env_correction"] == str(runs["json"])
    assert summary["editability"] == "pass"
    assert check_editability(code).passed


@needs_pack
def test_malformed_json_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.env.json"
    bad.write_text("{not json")
    proc, _ = _run(tmp_path / "o.strudel", "--env-correction", str(bad))
    assert proc.returncode != 0
    assert "--env-correction" in proc.stderr and "bad.env.json" in proc.stderr
    assert "Traceback" not in proc.stderr


@needs_pack
def test_missing_voices_is_a_clear_error(tmp_path):
    bad = tmp_path / "novoices.env.json"
    bad.write_text(json.dumps({"bars": NBARS}))
    proc, _ = _run(tmp_path / "o.strudel", "--env-correction", str(bad))
    assert proc.returncode != 0 and "voices" in proc.stderr and "Traceback" not in proc.stderr


@needs_pack
def test_mismatched_bars_warns_and_holds_last_multiplier(tmp_path):
    short = _write_env(tmp_path / "short.env.json", bars=SPLIT + 1,
                       bass=[BOOST] * SPLIT + [CUT])
    proc, code = _run(tmp_path / "o.strudel", "--env-correction", str(short))
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "WARNING env-correction" in proc.stderr and "holding the last" in proc.stderr
    plain_proc, plain = _run(tmp_path / "p.strudel")
    base = _gain_groups(plain, BASS_ANCHOR)
    corr = _gain_groups(code, BASS_ANCHOR)
    for b in range(NBARS):
        factor = BOOST if b < SPLIT else CUT      # bars past the file's end hold CUT
        for v0, v1 in zip(base[b], corr[b]):
            assert v1 == pytest.approx(v0 * factor, abs=2e-3)


# ---- review findings 6 + 7 (generator side) -------------------------------------------------

def _env_doc(path: Path, *, bass: list[float], master: list[float], clamp: list[float] | None = None) -> Path:
    doc = {"window_s": 10.0, "bpm": 136.0, "bars": len(bass), "source": "unit fixture",
           "voices": {"bass": bass, "lead": [1.0] * len(bass), "master": master}}
    if clamp is not None:
        doc["clamp"] = clamp
    path.write_text(json.dumps(doc))
    return path


def test_composed_multiplier_is_clamped_to_the_files_clamp(tmp_path):
    f = _env_doc(tmp_path / "c.json", bass=[2.0, 0.5], master=[2.0, 0.5], clamp=[0.7, 1.5])
    corr = load_env_correction(f, 2)
    assert corr[ENV_CORRECTION_CLAMP] == [0.7, 1.5]
    m = compose_env_multipliers(corr["bass"], corr["master"], corr[ENV_CORRECTION_CLAMP])
    assert m == pytest.approx([1.5, 0.7])          # 2.0*2.0=4.0 -> 1.5 ; 0.5*0.5=0.25 -> 0.7


def test_composed_multiplier_falls_back_to_the_files_own_range(tmp_path):
    f = _env_doc(tmp_path / "c.json", bass=[2.0, 0.5], master=[2.0, 0.5])    # no clamp key
    corr = load_env_correction(f, 2)
    assert corr[ENV_CORRECTION_CLAMP] == [0.5, 2.0]
    m = compose_env_multipliers(corr["bass"], corr["master"], corr[ENV_CORRECTION_CLAMP])
    assert m == pytest.approx([2.0, 0.5])


def test_malformed_clamp_is_a_clear_error(tmp_path):
    f = _env_doc(tmp_path / "c.json", bass=[1.0], master=[1.0], clamp=[2.0, 1.0])
    with pytest.raises(ValueError, match="clamp"):
        load_env_correction(f, 1)


def test_apply_env_correction_has_no_cap_and_keeps_silence():
    out = apply_env_correction([[0.0, 1.0], [0.5, 0.5]], [1.3, 0.8])
    assert out[0] == pytest.approx([0.0, 1.3]) and out[1] == pytest.approx([0.4, 0.4])
    assert apply_env_correction(None, [1.0]) is None


@needs_pack
def test_none_env_voice_still_gets_the_per_bar_correction(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    for p in STEMS.glob("*.wav"):
        if p.name != "bass.wav":                  # bass stem missing -> bass_env is None
            (stems / p.name).symlink_to(p)
    args = [a if a != str(STEMS) else str(stems) for a in ARGS]
    outs = {}
    for tag, extra in (("plain", []), ("corr", ["--env-correction", str(_write_env(tmp_path / "e.json"))])):
        out = tmp_path / f"{tag}.strudel"
        proc = subprocess.run([sys.executable, str(GEN), *args, *extra, "--out", str(out)],
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-1500:]
        outs[tag] = out.read_text(encoding="utf-8")
    bass_plain = next(ln for ln in outs["plain"].splitlines() if BASS_ANCHOR in ln)
    assert '.gain("<' not in bass_plain, "precondition: no env -> flat base gain"
    groups = _gain_groups(outs["corr"], BASS_ANCHOR)
    assert len(groups) == NBARS
    flat = {round(v, 3) for g in groups[:SPLIT] for v in g}
    cut = {round(v, 3) for g in groups[SPLIT:] for v in g}
    base = float(re.search(r"\.gain\(([\d.]+)\)", bass_plain).group(1))
    assert flat == {round(base * BOOST, 3)} and cut == {round(base * CUT, 3)}
