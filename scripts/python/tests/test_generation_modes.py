# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""Slice 4 — two honest generation modes (spec 003 §2.2-B/E, tasks.md Slice 4).

Runs ``generate_dynamic_strudel.py --mode {sample-instrument,synth}`` (generation only — no
Demucs, no render) against the cached Regime CLT sample pack and checks each output with the
Slice 1 detector (``editability_check.py``):

    --mode synth              → no ``samples(`` load, no custom sample names, header
                                ``// generation_mode: synth``, detector PASS with mode ``synth``,
                                every instrument/bank pick comes from the sound_selector palette
                                for the genre (CLAUDE.md ZERO HARDCODING)
    --mode sample-instrument  → unchanged behaviour: pitched pack instruments, PASS,
                                header ``// generation_mode: sample-instrument``

Skipped when the Regime CLT ``sample_pack/`` is absent (same pattern as test_vocal_modes.py).

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_generation_modes.py -q
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
from sound_selector import WAVEFORMS, retrieve_genre_context  # noqa: E402

GEN = SCRIPTS / "generate_dynamic_strudel.py"
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
BASE = "https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/regime-clt"
GENRE = "brazilian_funk"

_REQUIRED = [
    STEMS / "vocals.wav", STEMS / "bass.wav", STEMS / "melodic.wav", STEMS / "drums.wav",
    PACK / "bass.mid", PACK / "melodic.mid", PACK / "drums_bands.json", PACK / "vocals.mid",
    PACK / "strudel.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="Regime CLT sample_pack not present in .cache",
)

# the Slice 3 default-output invocation (v023 knobs), minus --out / --mode
BASE_ARGS = [
    "--stems-dir", str(STEMS), "--pack-dir", str(PACK),
    "--bass-midi", str(PACK / "bass.mid"), "--lead-midi", str(PACK / "melodic.mid"),
    "--drums-json", str(PACK / "drums_bands.json"),
    "--base-url", BASE, "--samples-url", f"{BASE}/instruments/samples.json",
    "--bass-sound", "regime_bass", "--lead-sound", "regime_lead",
    "--bpm", "136", "--key", "C# minor", "--genre", GENRE, "--num-bars", "78",
    "--sub-octave", "1", "--lead-hpf", "95", "--drum-mode", "extracted",
    "--bass-mult", "0.422", "--sub-gain", "1.002", "--cal-lead", "2.629", "--lead-lpf", "9000",
    "--hat-gain", "0.153", "--master-gain", "0.78",
]


def _palette_sounds(genre: str) -> set[str]:
    """Every sound name the sound_selector RAG offers for the genre — parsed from the same string
    the LLM prompts receive, so the test and the generator share one source of truth."""
    ctx = retrieve_genre_context(genre)
    body = ctx.split("—", 1)[1]
    names: set[str] = set()
    for part in body.split("|"):
        _, _, items = part.partition(":")
        names.update(t.strip() for t in items.split(",") if t.strip())
    assert names, ctx
    return names


def _chained_instruments(code: str) -> list[str]:
    """Names picked by a chained ``.s("…")`` (instrument choice, not drum-hit pattern data)."""
    return re.findall(r'\.s\("([^"]+)"\)', code)


def _banks(code: str) -> list[str]:
    return re.findall(r'\.bank\("([^"]+)"\)', code)


def _run(out: Path, *extra: str) -> tuple[subprocess.CompletedProcess, str, dict]:
    cmd = [sys.executable, str(GEN), *BASE_ARGS, *extra, "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"generator failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
    code = out.read_text(encoding="utf-8")
    summary = json.loads(proc.stdout[proc.stdout.index("{"):])
    return proc, code, summary


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("generation_modes")
    return {
        "synth": _run(d / "synth.strudel", "--mode", "synth"),
        "sample": _run(d / "sample-instrument.strudel", "--mode", "sample-instrument"),
        "default": _run(d / "default.strudel"),
    }


# ── synth ─────────────────────────────────────────────────────────────────────────────────
def test_synth_has_no_samples_load_and_no_custom_sample_names(outputs):
    _, code, summary = outputs["synth"]
    assert "samples(" not in code
    assert "regime_" not in code and "trackbass" not in code and "tracklead" not in code
    assert "vox" not in code and "vocalsfull" not in code and "drumsfull" not in code
    assert summary["uses_custom_samples"] is False


def test_synth_header_and_detector_verdict(outputs):
    _, code, summary = outputs["synth"]
    assert "// generation_mode: synth" in code.splitlines()[:8]
    assert summary["mode"] == "synth" and summary["generation_mode"] == "synth"
    assert summary["editability"] == "pass", summary["editability_violations"]
    res = check_editability(code)
    assert res.passed, res.violations
    assert res.generation_mode == "synth"
    assert len(res.editable_voices) >= 3   # bass, lead, drums (+ vocal)


def test_synth_sounds_come_from_the_genre_palette(outputs):
    _, code, summary = outputs["synth"]
    palette = _palette_sounds(GENRE)
    picks = [summary["bass_sound"], summary["lead_sound"]]
    if summary.get("vocal_sound"):
        picks.append(summary["vocal_sound"])
    assert picks and all(p in palette for p in picks), (picks, sorted(palette))
    # every chained instrument in the code is a palette sound or a built-in oscillator (the
    # sub-sine reinforcement layer); no sample-pack name sneaks through
    chained = _chained_instruments(code)
    assert chained
    assert all(n in palette or n in WAVEFORMS for n in chained), chained
    banks = _banks(code)
    assert banks and all(b in palette for b in banks), banks


def test_synth_forces_the_bank_drum_path_and_keeps_editable_bar_arrays(outputs):
    proc, code, summary = outputs["synth"]
    assert summary["drums"] is True
    assert _banks(code), "synth drums must ride a library drum machine via .bank()"
    assert "extracted" in proc.stderr.lower()      # --drum-mode extracted was overridden, loudly
    assert re.search(r"^let bass = \[", code, re.M) and re.search(r"^let lead = \[", code, re.M)
    assert "setcps(cps)" in code
    assert ".gain(" in code                          # envelopes still baked in


def test_synth_vocal_is_a_palette_instrument_line(outputs):
    _, code, summary = outputs["synth"]
    assert summary["vocal_mode"] == "instrument"
    assert re.search(r"^let vocal = \[", code, re.M)
    assert summary["vocal_sound"] in _palette_sounds(GENRE)


# ── sample-instrument ─────────────────────────────────────────────────────────────────────
def test_sample_instrument_is_unchanged_and_passes(outputs):
    _, code, summary = outputs["sample"]
    assert "// generation_mode: sample-instrument" in code.splitlines()[:8]
    assert summary["mode"] == "sample-instrument"
    assert summary["generation_mode"] == "sample-instrument"
    assert 'await samples("' in code
    assert '.s("regime_bass")' in code and '.s("regime_lead")' in code
    res = check_editability(code)
    assert res.passed, res.violations
    assert res.generation_mode == "sample-instrument"


def test_default_mode_is_sample_instrument(outputs):
    _, code_default, summary_default = outputs["default"]
    _, code_sample, _ = outputs["sample"]
    assert summary_default["mode"] == "sample-instrument"
    assert code_default == code_sample
