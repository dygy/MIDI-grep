# @layer: unit
# @spec: 001-core-pipeline
# @regression
"""Product decision: vocals are never generated, so every score's REFERENCE is the instrumental
(bass + drums + melodic), and the chance control is the other track's instrumental."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
import pipeline_helpers as ph  # noqa: E402


def _stems(d: Path, n: int = 4410) -> None:
    for name, amp in (("original", 0.5), ("bass", 0.1), ("drums", 0.2), ("melodic", 0.3), ("vocals", 0.4)):
        soundfile.write(str(d / f"{name}.wav"), (amp * np.ones((n, 2))).astype("float32"), 44100, subtype="FLOAT")


def test_instrumental_excludes_the_vocal_stem(tmp_path: Path) -> None:
    _stems(tmp_path)
    y, _ = soundfile.read(str(ph.instrumental_path(tmp_path)), always_2d=True)
    assert np.allclose(y, 0.6, atol=1e-6)            # 0.1 + 0.2 + 0.3 (a vocal in the sum would read 1.0)


def test_instrumental_is_rebuilt_when_a_stem_changes(tmp_path: Path) -> None:
    import os
    import time
    _stems(tmp_path)
    ph.write_instrumental(tmp_path)
    soundfile.write(str(tmp_path / "bass.wav"), (0.2 * np.ones((4410, 2))).astype("float32"), 44100, subtype="FLOAT")
    os.utime(tmp_path / "bass.wav", (time.time() + 5, time.time() + 5))
    y, _ = soundfile.read(str(ph.write_instrumental(tmp_path)), always_2d=True)
    assert np.allclose(y, 0.7, atol=1e-6)


def test_control_resolves_to_the_other_tracks_instrumental(tmp_path: Path) -> None:
    (tmp_path / "eval" / "datasets").mkdir(parents=True)
    (tmp_path / "eval" / "datasets" / "reference_tracks.yaml").write_text(
        "tracks:\n  - {genre: g, cache_key: A}\n  - {genre: g, cache_key: B}\n")
    for k in "AB":
        d = tmp_path / ".cache" / "stems" / k
        d.mkdir(parents=True)
        _stems(d)
    found = ph.control_track_path("g", "A", repo_root=tmp_path)
    assert found == tmp_path / ".cache" / "stems" / "B" / "instrumental.wav"


def test_every_scorer_uses_the_instrumental_not_original_wav() -> None:
    drv = (ROOT / "scripts" / "editable-pipeline.sh").read_text()
    assert 'compare_audio.py" "$INSTRUMENTAL" "$WAV"' in drv and 'compare_audio.py" "$TRACK/original.wav"' not in drv
    auto = (ROOT / "scripts" / "auto-calibrate.sh").read_text()
    assert 'pipeline_helpers.py" instrumental "$STEMS"' in auto and '"$CMP" "$ORIG" "$WAV"' in auto
    assert not re.search(r'ORIG="\$STEMS/original\.wav"', auto)
    src = (SCRIPTS / "pipeline_helpers.py").read_text()
    assert "original = write_instrumental(track_dir)" in src                       # stage 10 report
    imp = (SCRIPTS / "ai_improver.py").read_text()
    assert "pipeline_helpers.write_instrumental(" in imp                # ai_improver compare


def test_generator_and_pack_have_no_vocal_surface() -> None:
    gen = (SCRIPTS / "generate_dynamic_strudel.py").read_text()
    for needle in ("--vocal-mode", "--vocal-octave", "--cal-vocal", "let vocal", "vocal_g", "vocals.mid"):
        assert needle not in gen, needle
    cal = (SCRIPTS / "calibrate_dynamic.py").read_text()
    assert "cal_vocal" not in cal and "--cal-vocal" not in cal
    pack = (SCRIPTS / "build_sample_pack.py").read_text()
    assert "build_vocals" not in pack and 'default="drums,bass,melodic,loops"' in pack
    drv = (ROOT / "scripts" / "editable-pipeline.sh").read_text()
    assert "vocals.mid" not in drv and "--vocal-mode" not in drv
