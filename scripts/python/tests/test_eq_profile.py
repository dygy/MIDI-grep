# @layer: unit
# @regression
"""eq_profile: the listening EQ profile (gain recovery, loudness match, cuts, preset, the line in output.strudel)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import eq_profile as eq  # noqa: E402

SR = eq.SAMPLE_RATE
CONTRACT_KEYS = ["v", "preset", "bands", "lowCut", "highCut", "outputGain", "basis", "why"]


def _noise(seconds: float = 30.0, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * SR))


def _shape(y: np.ndarray, band_gain_db: list[float]) -> np.ndarray:
    """``y`` with a known per-band gain (the profile's own band table) applied in the frequency domain."""
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    edges = eq.band_edges()
    g = np.zeros(len(freqs))
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        g[(freqs >= lo) & (freqs < hi)] = band_gain_db[i]
    return np.fft.irfft(spec * 10 ** (g / 20), n=len(y))


def test_recovers_a_known_band_tilt():
    render = _noise()
    tilt = [0, 0, 6.0, 0, 0, 0, -6.0, 0, 0, 0]                  # original = render with +6 at 125, -6 at 2000
    profile, _ = eq.compute_profile(_shape(render, tilt), render)
    gains = [b.gain for b in profile.bands]
    # absolute gains carry the loudness offset; the differences between bands are the known tilt
    assert gains[2] - gains[0] == pytest.approx(6.0, abs=0.5)
    assert gains[6] - gains[0] == pytest.approx(-6.0, abs=0.5)
    assert gains[4] - gains[0] == pytest.approx(0.0, abs=0.5)


def test_loudness_match_keeps_render_power():
    render = _noise()
    tilt = [3, 3, 6.0, 0, 0, 0, -9.0, 0, 6, 6]
    profile, diag = eq.compute_profile(_shape(render, tilt), render)
    p_r = eq.band_powers(render / np.sqrt(np.mean(render ** 2)))
    w = p_r / p_r.sum()
    mean_change = 10 * np.log10(np.sum(w * 10 ** (np.array([b.gain for b in profile.bands]) / 10)))
    assert mean_change == pytest.approx(0.0, abs=0.2)
    assert diag["clamped_bands"] == []


def test_identical_signals_give_flat_profile():
    y = _noise()
    profile, _ = eq.compute_profile(y, y)
    assert all(b.gain == 0 for b in profile.bands)
    assert profile.lowCut is None and profile.highCut is None
    assert "already matches" in profile.why


def test_gains_are_clamped_to_the_players_range():
    render = _noise()
    tilt = [0, 0, 0, 0, 0, 0, 0, 0, 0, -60.0]
    profile, diag = eq.compute_profile(_shape(render, tilt), render)
    assert min(b.gain for b in profile.bands) >= -eq.EQ_GAIN_LIMIT
    assert max(b.gain for b in profile.bands) <= eq.EQ_GAIN_LIMIT
    assert diag["clamped_bands"] == [] or 16000 in diag["clamped_bands"]


def test_high_cliff_in_the_original_becomes_a_high_cut():
    render = _noise()
    original = _shape(render, [0] * 8 + [-60.0, -60.0])
    profile, _ = eq.compute_profile(original, render)
    assert profile.highCut is not None and profile.highCut.enabled
    assert profile.highCut.frequency == pytest.approx(eq.band_edges()[8], abs=0.1)
    assert profile.lowCut is None
    assert profile.bands[8].gain == 0 and profile.bands[9].gain == 0       # the cut does that job


def test_gentle_rolloff_is_not_a_cut():
    render = _noise()
    original = _shape(render, [0, 0, 0, 0, -1, -2, -3, -4, -5, -6])
    profile, _ = eq.compute_profile(original, render)
    assert profile.lowCut is None and profile.highCut is None


def test_low_cliff_becomes_a_low_cut():
    render = _noise()
    original = _shape(render, [-60.0, -60.0, -60.0] + [0] * 7)
    profile, _ = eq.compute_profile(original, render)
    assert profile.lowCut is not None
    assert profile.lowCut.frequency == pytest.approx(eq.band_edges()[3], abs=0.1)
    assert profile.highCut is None


@pytest.mark.parametrize("genre,kw,expected", [
    ("brazilian_funk", {}, "brazilianFunk"),
    ("house", {}, "house"),
    ("brazilian_phonk", {}, "brazilianPhonk"),
    ("retro_wave", {}, "synthwave"),
    ("brazilian_funk", {"uncertain": True}, None),
    ("brazilian_funk", {"genre_ref": "unreferenced"}, None),
    ("jazz", {}, None),                      # a profile preset, not a genre one
    ("klezmer", {}, None),
    (None, {}, None),
])
def test_genre_to_preset(genre, kw, expected):
    assert eq.preset_for_genre(genre, **kw) == expected


CODE = "// MIDI-grep dynamic\n// genre=x\n// BPM: 120\nconst cps = 0.5\n$: s(\"bd\")\n"


def test_insert_places_line_after_header_and_is_idempotent():
    line = eq.LINE_TAG + '{"v":1}'
    once = eq.insert_profile_line(CODE, line)
    assert once.split("\n")[3] == line and once.split("\n")[4].startswith("const cps")
    assert eq.insert_profile_line(once, line) == once
    other = eq.LINE_TAG + '{"v":2}'
    twice = eq.insert_profile_line(once, other)
    assert twice.count("@eq-profile") == 1 and other in twice and line not in twice


def test_line_is_compact_json_with_exactly_the_contract_keys():
    render = _noise()
    profile, _ = eq.compute_profile(_shape(render, [0, 0, 4, 0, 0, 0, 0, 0, 0, 0]), render, preset="brazilianFunk")
    line = profile.line()
    assert "\n" not in line and line.startswith("// @eq-profile {")
    doc = json.loads(line[len(eq.LINE_TAG):])
    assert list(doc) == CONTRACT_KEYS
    assert [b["frequency"] for b in doc["bands"]] == [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    assert all(set(b) == {"frequency", "gain"} for b in doc["bands"])
    assert doc["outputGain"] == 0 and doc["basis"] == "instrumental vs render" and doc["preset"] == "brazilianFunk"
    assert all(round(b["gain"], 1) == b["gain"] for b in doc["bands"])


def test_write_profile_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")
    track = tmp_path / "track"
    (track / "v001").mkdir(parents=True)
    ref = _noise(8)
    sf.write(str(track / "instrumental.wav"), _shape(ref, [0, 0, 6, 0, 0, 0, 0, 0, 0, 0]) * 0.05, SR, subtype="FLOAT")
    sf.write(str(track / "v001" / "render.wav"), ref * 0.05, SR, subtype="FLOAT")
    (track / "v001" / "output.strudel").write_text(CODE, encoding="utf-8")
    (track / "metadata.json").write_text(json.dumps({"genre": "house", "genre_uncertain": False}), encoding="utf-8")
    profile = eq.write_profile(track, "v001")
    assert profile.preset == "house"
    assert (track / "v001" / "output.strudel").read_text(encoding="utf-8").count("@eq-profile") == 1
    eq.write_profile(track, "v001")                                   # rerun replaces, never duplicates
    assert (track / "v001" / "output.strudel").read_text(encoding="utf-8").count("@eq-profile") == 1
    saved = json.loads((track / "v001" / eq.PROFILE_JSON).read_text(encoding="utf-8"))
    assert saved["diagnostics"]["sample_rate"] == SR and list(saved)[:8] == CONTRACT_KEYS
