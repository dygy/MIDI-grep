#!/usr/bin/env python3
"""Unit tests for Task F — voice-level knowledge RAG.

Tests:
  1. band_to_voice() maps every band to the correct voice prefix.
  2. retrieve_relevant_knowledge() returns "" gracefully when ClickHouse is unavailable.
  3. extract_voice_params_from_orchestrated_code() correctly parses gain/lpf from arrange() blocks.
  4. learn_from_improvement() detects orchestrated format and returns 0 for non-improvement.
  5. The gap-hint rendered by _targeted_gap_hint still works (format check).
  6. retrieve_relevant_knowledge() returns "" when bands are close (no retrieval needed).
  7. retrieve_relevant_knowledge() returns "" when band dicts are empty.
"""
import sys
from pathlib import Path

# Insert the scripts/python directory onto the path so imports work without a venv prefix.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from clickhouse_store import (
    band_to_voice,
    retrieve_relevant_knowledge,
    extract_voice_params_from_orchestrated_code,
    learn_from_improvement,
    _BAND_TO_VOICE,
    _BAND_TO_FX,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def _check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"  {FAIL}  {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _failures.append(name)


# ---------------------------------------------------------------------------
# Test 1: band_to_voice() mapping — every known band, plus unknown fallback
# ---------------------------------------------------------------------------
print("\n[T1] band_to_voice() mapping")

_expected = {
    "sub_bass":  "voice.bass",
    "bass":      "voice.bass",
    "low_mid":   "voice.bass",
    "mid":       "voice.lead",
    "high_mid":  "voice.lead",
    "high":      "voice.lead",
}
for band, expected_voice in _expected.items():
    got = band_to_voice(band)
    _check(
        f"band_to_voice('{band}') == '{expected_voice}'",
        got == expected_voice,
        f"got '{got}'"
    )

# Unknown band → fallback to voice.bass
_check(
    "band_to_voice('unknown') falls back to 'voice.bass'",
    band_to_voice("unknown") == "voice.bass",
    f"got '{band_to_voice('unknown')}'"
)


# ---------------------------------------------------------------------------
# Test 2: retrieve_relevant_knowledge() graceful degradation (no ClickHouse)
# ---------------------------------------------------------------------------
print("\n[T2] retrieve_relevant_knowledge() degrades gracefully without ClickHouse")

# Patch _ensure_clickhouse to return False for this test block
import clickhouse_store as _cs
_orig_ensure = _cs._ensure_clickhouse

def _always_false():
    return False

_cs._ensure_clickhouse = _always_false  # type: ignore[assignment]
try:
    result = retrieve_relevant_knowledge(
        comparison={"original": {"bands": {"sub_bass": 0.1, "bass": 0.2}},
                    "rendered": {"bands": {"sub_bass": 0.05, "bass": 0.1}}},
        genre="brazilian_funk",
        bpm=130.0,
    )
    _check(
        "returns '' when ClickHouse unavailable",
        result == "",
        f"got {result!r}"
    )
finally:
    _cs._ensure_clickhouse = _orig_ensure


# ---------------------------------------------------------------------------
# Test 3: retrieve_relevant_knowledge() — empty bands → ""
# ---------------------------------------------------------------------------
print("\n[T3] retrieve_relevant_knowledge() with empty band dicts")

result = retrieve_relevant_knowledge(
    comparison={"original": {}, "rendered": {}},
    genre="house",
    bpm=128.0,
)
_check(
    "returns '' when original/rendered bands are missing",
    result == "",
    f"got {result!r}"
)


# ---------------------------------------------------------------------------
# Test 4: retrieve_relevant_knowledge() — bands all close → no retrieval
# ---------------------------------------------------------------------------
print("\n[T4] retrieve_relevant_knowledge() — bands within threshold")

result = retrieve_relevant_knowledge(
    comparison={
        "original": {"bands": {"sub_bass": 0.30, "bass": 0.25, "low_mid": 0.20,
                                "mid": 0.15, "high_mid": 0.06, "high": 0.04}},
        "rendered": {"bands": {"sub_bass": 0.31, "bass": 0.25, "low_mid": 0.20,
                                "mid": 0.14, "high_mid": 0.06, "high": 0.04}},
    },
    genre="lofi",
    bpm=80.0,
)
# All diffs < 0.03, so retrieval is skipped regardless of CH availability
_check(
    "returns '' when worst band diff < 0.03",
    result == "",
    f"got {result!r}"
)


# ---------------------------------------------------------------------------
# Test 5: extract_voice_params_from_orchestrated_code()
# ---------------------------------------------------------------------------
print("\n[T5] extract_voice_params_from_orchestrated_code()")

_ORCH_CODE = """\
setcps(0.5333)

$: arrange(
  [8, note("c2 ~ ~ ~").sound("gm_synth_bass_1").gain(0.35).lpf(sine.range(440, 1080).slow(16))],
  [12, note("g2 e2 c2 g2").sound("gm_synth_bass_1").gain(0.85).lpf(sine.range(660, 1620).slow(16))]
)

$: arrange(
  [8, note("c4 ~ ~ ~").sound("gm_lead_2_sawtooth").gain(0.30).lpf(sine.range(2475, 6075).slow(8))],
  [12, note("g4 e4 c4 g4").sound("gm_lead_2_sawtooth").gain(0.80).lpf(sine.range(3740, 9180).slow(8))]
)

$: arrange(
  [8, s("bd ~ ~ ~").bank("RolandTR808").gain(0.40)],
  [12, s("bd hh sd hh").bank("RolandTR808").gain(0.85)]
)
"""

params = extract_voice_params_from_orchestrated_code(_ORCH_CODE)

# bass block: gain=0.35, lpf derived from sine.range(440, 1080) → centre = 760
_check(
    "voice.bass extracted",
    "voice.bass" in params,
    f"params keys: {list(params)}"
)
_check(
    "voice.bass.gain == 0.35",
    params.get("voice.bass", {}).get("gain") == 0.35,
    f"got {params.get('voice.bass', {}).get('gain')}"
)
# sine.range(440, 1080) → centre = (440+1080)//2 = 760
_check(
    "voice.bass.lpf == 760 (centre of sine.range(440, 1080))",
    params.get("voice.bass", {}).get("lpf") == 760,
    f"got {params.get('voice.bass', {}).get('lpf')}"
)

# lead block: gain=0.30
_check(
    "voice.lead extracted",
    "voice.lead" in params,
    f"params keys: {list(params)}"
)
_check(
    "voice.lead.gain == 0.30",
    params.get("voice.lead", {}).get("gain") == 0.30,
    f"got {params.get('voice.lead', {}).get('gain')}"
)
# sine.range(2475, 6075) → centre = 4275
_check(
    "voice.lead.lpf == 4275 (centre of sine.range(2475, 6075))",
    params.get("voice.lead", {}).get("lpf") == 4275,
    f"got {params.get('voice.lead', {}).get('lpf')}"
)

# drums block: gain=0.40, no lpf
_check(
    "voice.drums extracted",
    "voice.drums" in params,
    f"params keys: {list(params)}"
)
_check(
    "voice.drums.gain == 0.40",
    params.get("voice.drums", {}).get("gain") == 0.40,
    f"got {params.get('voice.drums', {}).get('gain')}"
)
_check(
    "voice.drums has no lpf",
    "lpf" not in params.get("voice.drums", {}),
    f"lpf unexpectedly present: {params.get('voice.drums', {}).get('lpf')}"
)


# ---------------------------------------------------------------------------
# Test 6: extract_voice_params_from_orchestrated_code() — bare .lpf(N) fallback
# ---------------------------------------------------------------------------
print("\n[T6] extract_voice_params — bare .lpf(N) fallback")

_BARE_LPF_CODE = """\
setcps(0.5)
$: arrange(
  [4, note("c2 ~").sound("sawtooth").gain(0.55).lpf(600)]
)
$: arrange(
  [4, note("c4 ~").sound("gm_lead_2_sawtooth").gain(0.45).lpf(3500)]
)
$: arrange(
  [4, s("bd ~").bank("RolandTR808").gain(0.70)]
)
"""
bare_params = extract_voice_params_from_orchestrated_code(_BARE_LPF_CODE)
_check(
    "voice.bass.lpf == 600.0 (bare .lpf(N))",
    bare_params.get("voice.bass", {}).get("lpf") == 600.0,
    f"got {bare_params.get('voice.bass', {}).get('lpf')}"
)
_check(
    "voice.lead.lpf == 3500.0 (bare .lpf(N))",
    bare_params.get("voice.lead", {}).get("lpf") == 3500.0,
    f"got {bare_params.get('voice.lead', {}).get('lpf')}"
)


# ---------------------------------------------------------------------------
# Test 7: learn_from_improvement() — non-improvement returns 0
# ---------------------------------------------------------------------------
print("\n[T7] learn_from_improvement() returns 0 for regressions")

count = learn_from_improvement(
    track_hash="aabbccdd",
    genre="house",
    bpm=128.0,
    key_type="minor",
    old_code=_ORCH_CODE,
    new_code=_ORCH_CODE,   # same code → no improvement
    old_similarity=0.60,
    new_similarity=0.55,   # regression
)
_check(
    "returns 0 when new_similarity < old_similarity",
    count == 0,
    f"got {count}"
)


# ---------------------------------------------------------------------------
# Test 8: learn_from_improvement() — orchestrated format detection
# ---------------------------------------------------------------------------
print("\n[T8] learn_from_improvement() detects orchestrated code format")

# Swap gains slightly so there IS a detectable change, but CH is unavailable so
# store_knowledge will no-op and return falsy.  We check that the detection
# path runs without errors (count may be 0 if CH is down — that's acceptable).
_ORCH_CODE_IMPROVED = _ORCH_CODE.replace("gain(0.35)", "gain(0.50)")
try:
    count2 = learn_from_improvement(
        track_hash="aabbccdd",
        genre="house",
        bpm=128.0,
        key_type="minor",
        old_code=_ORCH_CODE,
        new_code=_ORCH_CODE_IMPROVED,
        old_similarity=0.55,
        new_similarity=0.65,
    )
    _check(
        "runs without exception on orchestrated code (count >= 0)",
        count2 >= 0,
        f"got {count2}"
    )
except Exception as e:
    _check("runs without exception on orchestrated code", False, str(e))


# ---------------------------------------------------------------------------
# Test 9: _targeted_gap_hint output format (imported from ai_improver)
# ---------------------------------------------------------------------------
print("\n[T9] _targeted_gap_hint returns actionable string")

try:
    from ai_improver import _targeted_gap_hint

    comp = {
        "original": {"bands": {"sub_bass": 0.30, "bass": 0.25}},
        "rendered": {"bands": {"sub_bass": 0.10, "bass": 0.15}},  # sub_bass -20%, bass -10%
    }
    band_diffs = {
        "sub_bass": 0.10 - 0.30,  # -0.20
        "bass": 0.15 - 0.25,      # -0.10
        "low_mid": 0.0,
        "mid": 0.0,
        "high_mid": 0.0,
        "high": 0.0,
    }
    hint = _targeted_gap_hint("bass", band_diffs, comp, "brazilian_funk", 130.0, None)
    _check(
        "_targeted_gap_hint returns non-empty string",
        isinstance(hint, str) and len(hint) > 0,
        f"got empty/non-string: {hint!r}"
    )
    _check(
        "_targeted_gap_hint mentions 'too quiet' for negative sub_bass diff",
        "too quiet" in hint,
        f"got: {hint!r}"
    )
except ImportError as e:
    print(f"  SKIP  _targeted_gap_hint test (ai_improver not importable: {e})")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def test_all_voice_knowledge_checks():
    """pytest entrypoint — the checks above run at import; assert none failed."""
    assert not _failures, f"{len(_failures)} voice-knowledge checks failed: {_failures}"


if __name__ == "__main__":
    print()
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED — {_failures}")
        sys.exit(1)
    print("RESULT: All checks PASSED")
    sys.exit(0)
