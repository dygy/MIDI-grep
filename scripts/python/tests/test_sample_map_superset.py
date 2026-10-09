# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""``generate_sample_strudel.build_sample_map`` must be a SUPERSET of the pack manifest.

Slice 3 found that rewriting ``samples.json`` through ``generate_sample_strudel.py`` dropped the
pack's pitched vocal (``<prefix>_vocal``) and the vocal chops (``vox<N>``), silencing the
editable vocal voice of ``generate_dynamic_strudel.py``. Every key of ``strudel.json`` must
survive the rewrite, in a shape Strudel's ``samples()`` accepts.
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
    "custom_thing": "misc/x.wav",
}


def test_every_manifest_key_survives_the_rewrite():
    out = build_sample_map(MANIFEST, "instrument")
    assert set(MANIFEST) <= set(out), sorted(set(MANIFEST) - set(out))


def test_vocal_entries_keep_their_shapes():
    out = build_sample_map(MANIFEST, "instrument")
    assert out["regime_vocal"] == MANIFEST["regime_vocal"]          # note-keyed pitched map
    assert out["vox0"] == ["vocals/vox0.wav"]
    assert out["vox1"] == ["vocals/vox1.wav"]                        # string → single-element array
    assert out["custom_thing"] == ["misc/x.wav"]


def test_existing_instrument_rules_are_unchanged():
    out = build_sample_map(MANIFEST, "instrument")
    assert out["bd"] == ["drums/bd.wav"]
    assert out["trackbass"] == ["bass/e2.wav"]                       # representative sample
    assert out["drumsloop"] == MANIFEST["drumsloop"]
    assert "oh" not in out
    assert all(isinstance(v, (list, dict)) for v in out.values())
