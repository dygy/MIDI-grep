# @layer: integration
# @spec: 001-core-pipeline
# @regression
"""separate.py must emit lossless, sample-aligned stems that sum to the input at unity gain.

Old behaviour: demucs CLI with --mp3 -> mp3 bitstreams (copied under .wav in the cache) delayed by the
encoder delay (+25.1 ms = 1105 samples measured on both reference tracks), ~16-27 dB reconstruction SNR.
Set SEPARATE_PY to test another copy of the script (default: scripts/python/separate.py).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
SCRIPTS = Path(__file__).resolve().parent.parent
SEP = Path(os.environ.get("SEPARATE_PY", SCRIPTS / "separate.py"))
SR = 44100


def _have_weights() -> bool:
    return any(Path.home().joinpath(".cache/torch/hub/checkpoints").glob("955717e8*"))


def _mix(seconds=6.0):
    rng = np.random.default_rng(7)
    t = np.arange(int(SR * seconds)) / SR
    kick = np.zeros_like(t)
    for b in np.arange(0, seconds, 0.5):
        i = int(b * SR); n = int(0.12 * SR)
        kick[i:i + n] += np.sin(2 * np.pi * (140 * np.exp(-np.arange(n) / SR * 25)).cumsum() / SR) * np.exp(-np.arange(n) / (0.03 * SR))
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    lead = 0.15 * sum(np.sin(2 * np.pi * f * t) for f in (440, 554, 659))
    noise = 0.01 * rng.standard_normal(len(t))
    m = (0.6 * kick + bass + lead + noise).astype(np.float32)
    return np.stack([m, m * 0.9])


def _run(tmp_path, quality="fast"):
    wav = tmp_path / "in.wav"
    sf.write(wav, _mix().T, SR, subtype="FLOAT")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SEP), str(wav), str(out), "--mode", "full", "--quality", quality],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    return wav, out


@pytest.mark.skipif(not _have_weights(), reason="htdemucs weights not cached")
def test_stems_are_lossless_aligned_and_sum_to_input(tmp_path):
    wav, out = _run(tmp_path)
    mix, _ = sf.read(wav, dtype="float32"); mix = mix.T
    stems = {}
    for name in ("melodic", "bass", "drums", "vocals"):
        p = out / f"{name}.wav"
        assert p.exists(), f"{name}.wav missing (old code wrote {sorted(x.name for x in out.iterdir())})"
        head = p.read_bytes()[:4]
        assert head == b"RIFF", f"{name}.wav is not a WAV container (starts {head!r}); mp3 bitstream under a wav name"
        a, sr = sf.read(p, dtype="float32")
        assert sr == SR and a.shape[0] == mix.shape[1], f"{name}: length {a.shape[0]} != input {mix.shape[1]}"
        stems[name] = a.T
    total = sum(stems.values())
    err = mix - total
    snr = 10 * np.log10((mix ** 2).sum() / max((err ** 2).sum(), 1e-20))
    assert snr > 60, f"stems sum to input at only {snr:.1f} dB SNR (unity gain violated)"
    # delay: the stem sum must be aligned with the input to the sample
    a, b = mix.mean(0), total.mean(0)
    import scipy.signal as ss
    lag = int(np.argmax(ss.correlate(a, b, mode="full", method="fft"))) - (len(b) - 1)
    assert lag == 0, f"stems delayed by {lag} samples ({lag / SR * 1000:.1f} ms)"


def test_no_mp3_in_source():
    src = SEP.read_text()
    assert "'--mp3'" not in src and '"--mp3"' not in src, "separate.py still asks demucs for mp3 output"
