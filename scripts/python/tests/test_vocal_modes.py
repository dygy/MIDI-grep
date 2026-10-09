# @layer: integration
# @spec: 003-editable-strudel-generation
# @regression
"""Slice 3 — the vocal voice is editable data (spec 003 §2.2-B, tasks.md Slice 3).

Runs ``generate_dynamic_strudel.py`` (generation only — no Demucs, no render) for every
``--vocal-mode`` against the cached Regime CLT sample pack and checks each output with the
Slice 1 detector (``editability_check.py``):

    instrument (default)  → PASS, ``let vocal = [...]`` non-empty, ``note(cat(...vocal)).s("<prefix>_vocal")``
    chops                 → PASS, ``let vox = [...]`` non-empty, ``s(cat(...vox))``
    texture               → PASS only with >= 2 editable voices; the loop line carries ``// texture``
    none                  → PASS, no vocal voice at all
    --vocal-loop (alias)  → deprecated alias of texture (warning on stderr)

Skipped when the Regime CLT ``sample_pack/`` (with ``vocals.mid`` + ``vocals/``) is absent.

Run: scripts/python/.venv/bin/python -m pytest scripts/python/tests/test_vocal_modes.py -q
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

GEN = SCRIPTS / "generate_dynamic_strudel.py"
STEMS = REPO / ".cache" / "stems" / "Regime CLT (Dj Brunin XM, Aurora Shukita)"
PACK = STEMS / "sample_pack"
# nothing is fetched at generation time — the R2 base is only baked into the samples() URLs
BASE = "https://pub-56831423fee34641805da07cfdaf6812.r2.dev/midi-grep/regime-clt"

_REQUIRED = [
    STEMS / "vocals.wav", STEMS / "bass.wav", STEMS / "melodic.wav", STEMS / "drums.wav",
    PACK / "bass.mid", PACK / "melodic.mid", PACK / "drums_bands.json",
    PACK / "vocals.mid", PACK / "vocals", PACK / "samples.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="Regime CLT sample_pack with vocals.mid + vocals/ not present in .cache",
)

# v023's invocation (session handoff memo / calibration_params.json), minus --out
V023_ARGS = [
    "--stems-dir", str(STEMS), "--pack-dir", str(PACK),
    "--bass-midi", str(PACK / "bass.mid"), "--lead-midi", str(PACK / "melodic.mid"),
    "--drums-json", str(PACK / "drums_bands.json"),
    "--base-url", BASE, "--samples-url", f"{BASE}/instruments/samples.json",
    "--bass-sound", "regime_bass", "--lead-sound", "regime_lead",
    "--bpm", "136", "--key", "C# minor", "--genre", "brazilian_funk", "--num-bars", "78",
    "--sub-octave", "1", "--lead-hpf", "95", "--drum-mode", "extracted",
    "--bass-mult", "0.422", "--sub-gain", "1.002", "--cal-lead", "2.629", "--lead-lpf", "9000",
    "--hat-gain", "0.153", "--master-gain", "0.78",
]


def _pack_manifest() -> dict:
    """The builder's canonical manifest (``strudel.json``). ``samples.json`` is the same map
    host-resolved, but other tools (``generate_sample_strudel.py``) rewrite it from their own
    key subset, so the editable-vocal contract is checked against the builder's file."""
    return json.loads((PACK / "strudel.json").read_text())


def _vocal_sound_name() -> str:
    """The pack's pitched vocal instrument key (``<prefix>_vocal``) — never hardcoded here."""
    keys = [k for k, v in _pack_manifest().items() if k.endswith("_vocal") and isinstance(v, dict) and v]
    assert keys, "pack strudel.json has no <prefix>_vocal pitched map — run build_sample_pack.py --sections vocals"
    return keys[0]


def _array(code: str, name: str) -> list[str]:
    m = re.search(rf"^let {name} = \[\n(.*?)\n\]", code, re.M | re.S)
    if not m:
        return []
    return re.findall(r'"([^"]*)"', m.group(1))


def _nonrest_tokens(bars: list[str]) -> list[str]:
    return [t for b in bars for t in b.split() if t != "~"]


def _run(out: Path, *extra: str, args: list[str] | None = None) -> tuple[subprocess.CompletedProcess, str, dict]:
    cmd = [sys.executable, str(GEN), *(V023_ARGS if args is None else args), *extra, "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"generator failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
    code = out.read_text(encoding="utf-8")
    summary = json.loads(proc.stdout[proc.stdout.index("{"):])
    return proc, code, summary


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """Generate every variant ONCE (each run re-analyses four stems; ~10-20 s per run)."""
    d = tmp_path_factory.mktemp("vocal_modes")
    res = {}
    res["default"] = _run(d / "default.strudel")                                  # no flag → instrument
    res["chops"] = _run(d / "chops.strudel", "--vocal-mode", "chops")
    res["texture"] = _run(d / "texture.strudel", "--vocal-mode", "texture")
    res["none"] = _run(d / "none.strudel", "--vocal-mode", "none")
    res["alias"] = _run(d / "alias.strudel", "--vocal-loop")
    # texture with only ONE editable voice: lead only (no bass MIDI, no drums pattern)
    one_voice = [a for a in V023_ARGS]
    one_voice[one_voice.index("--bass-midi") + 1] = str(d / "missing_bass.mid")
    i = one_voice.index("--drums-json")
    del one_voice[i:i + 2]
    res["texture_one_voice"] = _run(d / "texture_one_voice.strudel", "--vocal-mode", "texture", args=one_voice)
    return res


# ── instrument (default) ──────────────────────────────────────────────────────────────────
def test_default_mode_is_instrument_and_passes(outputs):
    _, code, summary = outputs["default"]
    assert summary["vocal_mode"] == "instrument"
    res = check_editability(code)
    assert res.passed, res.violations
    assert res.generation_mode == "sample-instrument"
    assert "// generation_mode: sample-instrument" in code


def test_instrument_emits_nonempty_vocal_array_on_pitched_vocal_instrument(outputs):
    _, code, summary = outputs["default"]
    bars = _array(code, "vocal")
    assert bars and len(bars) == 78, "let vocal = [...] must have one string per bar"
    assert _nonrest_tokens(bars), "let vocal must carry notes, not only rests"
    assert summary["vocal_bars"] == len(bars)
    assert f'note(cat(...vocal)).s("{_vocal_sound_name()}")' in code
    assert "vocalsfull" not in code
    res = check_editability(code)
    labels = [v.label for v in res.editable_voices]
    assert "vocal" in labels, labels
    assert len(res.editable_voices) >= 4


def test_instrument_vocal_line_is_folded_into_the_detected_range(outputs):
    """Range-folding rule: every emitted vocal note lies inside the pack-derived [lo, hi]."""
    _, code, summary = outputs["default"]
    lo, hi = summary["vocal_range"]
    assert hi - lo >= 12, "detected vocal range must span at least an octave"
    import pretty_midi  # local import: audio stack only needed here
    names = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
    for tok in _nonrest_tokens(_array(code, "vocal")):
        m = re.fullmatch(r"([a-g]s?)(-?\d+)", tok)
        assert m, tok
        midi = names.index(m.group(1)) + 12 * (int(m.group(2)) + 1)
        assert lo <= midi <= hi, f"{tok} ({midi}) outside [{lo},{hi}]"
    del pretty_midi


# ── chops ─────────────────────────────────────────────────────────────────────────────────
def test_chops_emits_nonempty_vox_pattern_and_passes(outputs):
    _, code, summary = outputs["chops"]
    assert summary["vocal_mode"] == "chops"
    bars = _array(code, "vox")
    assert bars and len(bars) == 78
    toks = _nonrest_tokens(bars)
    assert toks and all(re.fullmatch(r"vox\d+", t) for t in toks), toks[:10]
    assert all(len(b.split()) == 16 for b in bars), "vox bars must be 16-step"
    assert "s(cat(...vox))" in code
    assert "vocalsfull" not in code
    res = check_editability(code)
    assert res.passed, res.violations
    assert res.generation_mode == "sample-instrument"


def test_chops_reference_existing_one_shots(outputs):
    _, code, _ = outputs["chops"]
    manifest = _pack_manifest()
    for t in set(_nonrest_tokens(_array(code, "vox"))):
        assert t in manifest, f"{t} missing from pack strudel.json"
        path = manifest[t][0] if isinstance(manifest[t], list) else manifest[t]
        assert (PACK / path).exists(), path


def test_builder_manifest_rules_for_vocal_entries():
    """samples.json as written by build_sample_pack.py: absolute _base ending in '/', the
    pitched map note-keyed, every vox one-shot a single-element array (CLAUDE.md rules).
    Skipped (not failed) when a concurrent generate_sample_strudel.py run has rewritten the
    file without the vocal keys — that writer is outside this slice."""
    s = json.loads((PACK / "samples.json").read_text())
    vocal_keys = [k for k, v in s.items() if k.endswith("_vocal") and isinstance(v, dict)]
    if not vocal_keys:
        pytest.skip("samples.json rewritten without vocal keys by another tool; re-run "
                    "build_sample_pack.py --sections vocals")
    assert s["_base"].startswith("http") and s["_base"].endswith("/")
    assert all(re.fullmatch(r"[a-g]s?-?\d+", n) for n in s[vocal_keys[0]])
    vox = {k: v for k, v in s.items() if re.fullmatch(r"vox\d+", k)}
    assert vox and all(isinstance(v, list) and len(v) == 1 for v in vox.values())


# ── texture ───────────────────────────────────────────────────────────────────────────────
def test_texture_keeps_loop_with_marker_and_passes_with_two_editable_voices(outputs):
    _, code, summary = outputs["texture"]
    assert summary["vocal_mode"] == "texture"
    loop_lines = [ln for ln in code.splitlines() if 's("vocalsfull")' in ln]
    assert len(loop_lines) == 1 and "// texture" in loop_lines[0]
    res = check_editability(code)
    assert res.passed, res.violations
    assert len(res.texture_voices) == 1
    assert len(res.editable_voices) >= 2


def test_texture_is_dropped_when_fewer_than_two_editable_voices(outputs):
    proc, code, summary = outputs["texture_one_voice"]
    assert "vocalsfull" not in code, "generator must not emit the loop with < 2 editable voices"
    assert summary["vocal_mode"] == "none"
    assert "texture" in proc.stderr.lower()
    res = check_editability(code)
    assert res.passed, res.violations
    assert len(res.editable_voices) == 1
    # and had the loop been emitted anyway, the detector rejects it (R4)
    forced = code + '\n$: s("vocalsfull").slice(16, run(16)).slow(16).clip(1)  // texture\n'
    forced_res = check_editability(forced)
    assert not forced_res.passed
    assert any(v.startswith("R4") for v in forced_res.violations), forced_res.violations


# ── none ──────────────────────────────────────────────────────────────────────────────────
def test_none_omits_the_vocal_voice(outputs):
    _, code, summary = outputs["none"]
    assert summary["vocal_mode"] == "none"
    assert "vocalsfull" not in code and "let vocal" not in code and "let vox" not in code
    res = check_editability(code)
    assert res.passed, res.violations
    assert not res.texture_voices


# ── deprecated alias ──────────────────────────────────────────────────────────────────────
def test_vocal_loop_alias_is_texture_with_deprecation_warning(outputs):
    proc, code, summary = outputs["alias"]
    assert "deprecat" in proc.stderr.lower()
    assert summary["vocal_mode"] == "texture"
    assert any('s("vocalsfull")' in ln and "// texture" in ln for ln in code.splitlines())
    assert check_editability(code).passed
