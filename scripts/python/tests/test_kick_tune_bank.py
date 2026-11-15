"""Synth drums: the palette kick tuned to the stem's 808 (drum_pitch bank shots, hit_ratios fill).

Tatuagem No Pescoço synth (2026-11-15, v010 0.769 FAIL): Demucs filed the whole low end under drums (bass stem 0.003 of
the drums' low band) - a tuned 808 kick whose tail sits at 35.2 Hz, played by the RolandTR808 bd at 47.3 Hz; 80 % of the
mix's chroma energy is below 130 Hz (chroma_corr 0.43). Every hit tuned: 0.7711 -> 0.7851 (references within noise).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drum_pitch as DP  # noqa: E402

SR = 22050


def _kick(f0: float, n: int, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / 0.12) * np.sin(2 * np.pi * f0 * t)


def test_fill_writes_the_unread_hits_of_a_library_kick_at_the_common_reading():
    shot = _kick(48.0, int(0.25 * SR))
    y = np.zeros(int(12 * SR))
    hits = []
    rng = np.random.default_rng(0)
    for i in range(20):
        t = 0.5 + 0.5 * i
        k = int(t * SR)
        if i % 5 == 4:   # a hit whose tail is noise: unreadable
            y[k:k + int(0.25 * SR)] += 0.3 * rng.standard_normal(int(0.25 * SR))
        else:
            y[k:k + int(0.25 * SR)] += _kick(36.0, int(0.25 * SR))
        hits.append({"time": t, "type": "bd"})
    plain = DP.hit_ratios(y, SR, hits, "bd", shot)
    filled = DP.hit_ratios(y, SR, hits, "bd", shot, fill=True)
    assert len(plain) == 16 and len(filled) == 20
    common = np.median(list(plain.values()))
    assert common == pytest.approx(36.0 / 48.0, rel=0.03)
    assert all(filled[h["time"]] == pytest.approx(common) for h in hits if h["time"] not in plain)


def test_the_palette_drum_machine_kick_is_recorded_with_a_readable_tail():
    rec = DP.bank_shot("RolandTR808")
    assert rec is not None, "run: drum_pitch.py probe RolandTR808"
    x, sr = rec
    assert 40.0 < DP.oneshot_pitch(x, sr) < 60.0


def test_bank_mode_tunes_every_kick_against_the_recorded_bank_kick(tmp_path):
    import soundfile as sf
    import generate_dynamic_strudel as G
    x, sr = DP.bank_shot("RolandTR808")
    y = np.zeros(int(12 * sr))
    hits = []
    for i in range(20):
        t = 0.5 + 0.5 * i
        k = int(t * sr)
        y[k:k + int(0.25 * sr)] += _kick(36.0, int(0.25 * sr), sr) if i % 5 else 0.01 * np.ones(int(0.25 * sr))
        hits.append({"time": t, "type": "bd"})
    (tmp_path / "stems").mkdir()
    sf.write(str(tmp_path / "stems" / "drums.wav"), y, sr)
    (tmp_path / "hits.json").write_text(json.dumps({"hits": hits}))
    args = argparse.Namespace(stems_dir=tmp_path / "stems", drums_json=tmp_path / "hits.json", pack_dir=tmp_path,
                              drum_mode="bank", genre="brazilian_funk")
    got = G.measured_drum_pitch(args)["bd"]
    assert len(got) == 20                                                   # the unread hits too (fill)
    assert np.median(list(got.values())) == pytest.approx(36.0 / DP.oneshot_pitch(x, sr), rel=0.03)
    args.drum_mode = "extracted"                                            # no extracted kit here: nothing to tune
    assert G.measured_drum_pitch(args) is None
