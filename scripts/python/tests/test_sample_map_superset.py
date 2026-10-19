# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""``generate_sample_strudel.build_sample_map`` must be a SUPERSET of the pack manifest, minus vocals.

Every key of ``strudel.json`` survives the rewrite in a shape Strudel's ``samples()`` accepts, EXCEPT vocal
entries (``*_vocal`` pitched maps, ``vox<N>`` chops, ``vocals*`` loops): a singer is never generated (product
decision), so a pack built before it must not leak vocal samples into the hosted manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from generate_sample_strudel import build_sample_map  # noqa: E402

MANIFEST = {
    "drumsloop": ["loops/drums_000.wav", "loops/drums_001.wav"],
    "bassloop": ["loops/bass_000.wav"],
    "bd": "drums/bd.wav", "sd": "drums/sd.wav", "hh": "drums/hh.wav",
    "trackbass": {"cs2": "bass/cs2.wav", "e2": "bass/e2.wav", "g2": "bass/g2.wav"},
    "tracklead": {"cs4": "melodic/cs4.wav"},
    "regime_vocal": {"cs3": "vocals/vocal_49.wav", "e3": "vocals/vocal_52.wav"},
    "vox0": ["vocals/vox0.wav"],
    "vox1": "vocals/vox1.wav",
    "vocalsloop": ["loops/vocals_000.wav"],
    "custom_thing": "misc/x.wav",
}


VOCAL_KEYS = {"regime_vocal", "vox0", "vox1", "vocalsloop"}


def test_every_non_vocal_manifest_key_survives_the_rewrite():
    out = build_sample_map(MANIFEST, "instrument")
    assert set(MANIFEST) - VOCAL_KEYS <= set(out), sorted(set(MANIFEST) - VOCAL_KEYS - set(out))
    assert out["custom_thing"] == ["misc/x.wav"]                     # unknown keys still pass through (shape fixed)


def test_vocal_entries_are_never_carried():
    for mode in ("instrument", "hybrid", "loops"):
        out = build_sample_map(MANIFEST, mode)
        assert not (VOCAL_KEYS & set(out)), sorted(VOCAL_KEYS & set(out))


def test_existing_instrument_rules_are_unchanged():
    out = build_sample_map(MANIFEST, "instrument")
    assert out["bd"] == ["drums/bd.wav"]
    assert out["trackbass"] == ["bass/e2.wav"]                       # representative sample
    assert out["drumsloop"] == MANIFEST["drumsloop"]
    assert "oh" not in out
    assert all(isinstance(v, (list, dict)) for v in out.values())
