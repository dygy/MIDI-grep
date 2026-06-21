#!/usr/bin/env python3
"""A2 — closed-loop envelope fitting (EXPERIMENTAL — measured NEGATIVE per-cycle, see WARNING).

Idea: the per-cycle gain envelope is sampled open-loop (gain = the original stem's loudness), but the
synth + soft-clip render it back COMPRESSED and non-linearly distorted (measured: rendered dynamic
range ~half the original, structured residual autocorr 0.34). This corrects the gain from an actual
render: where the rendered stem came out too quiet vs the original it boosts that cycle's gain, too
loud it cuts — a damped multiplicative step.

WARNING (June 2026): measured on Caravan it HURT shape (bass 0.685→0.607, melodic 0.448→0.370, mean
0.476→0.412). Why: the feedback signal is the DEMUCS-SEPARATED rendered stem, which is noisy
(separation artifacts + render variance). A per-cycle multiplicative correction injects that noise
INTO the gain envelope, making it less smooth than the original → worse correlation. The systematic
compression is real but per-cycle feedback is too noisy to invert. NOT wired into the orchestrator.
A viable A2 would fit a LOW-PARAMETER systematic correction (e.g. one nonlinear dynamic-range
expansion term) rather than 87 independent per-cycle gains — left as future work.

Usage: python fit_envelope.py "<track_dir>" v0NN [--alpha 0.5] [--out output_a2.strudel]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("librosa required", file=sys.stderr)
    sys.exit(2)

SR = 22050
HOP = 5512
# voice → (rendered stem basename) ; the gain envelope lives on each voice's arrange
VOICE_STEM = {"bass": "bass", "lead": "melodic", "drums": "drums"}


def _env_pc(path: str, n: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    e = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    xs = np.linspace(0, len(e) - 1, n)
    return np.interp(xs, np.arange(len(e)), e)


def _best_lag(o: np.ndarray, r: np.ndarray, cap: int) -> int:
    on = (o - o.mean()) / (o.std() + 1e-9)
    rn = (r - r.mean()) / (r.std() + 1e-9)
    best, blag, n = -9.0, 0, len(o)
    for lag in range(-cap, cap + 1):
        a, b = (on[lag:], rn[:n - lag]) if lag >= 0 else (on[:n + lag], rn[-lag:])
        m = min(len(a), len(b))
        if m > n // 2:
            c = float(np.mean(a[:m] * b[:m]))
            if c > best:
                best, blag = c, lag
    return blag


def _rendered_path(version_dir: Path, stem: str) -> Path | None:
    for ext in (".wav", ".mp3"):
        p = version_dir / f"render_{stem}{ext}"
        if p.exists():
            return p
    solo = version_dir / "render_drums_solo.wav"
    return solo if (stem == "drums" and solo.exists()) else None


def correct_voice_gain(gains: list[float], orig_path: str, rend_path: str, alpha: float) -> list[float]:
    n = len(gains)
    o = _env_pc(orig_path, n)
    r = _env_pc(rend_path, n)
    on = o / (o.max() + 1e-9)
    rn = r / (r.max() + 1e-9)
    lag = _best_lag(o, r, cap=max(2, n // 12))
    rn = np.roll(rn, -lag)  # align rendered to original (shift back by the lag)
    out = []
    for c in range(n):
        ratio = (on[c] + 0.02) / (rn[c] + 0.02)        # >1 → rendered too quiet here → boost
        g = gains[c] * (ratio ** alpha)
        out.append(round(float(np.clip(g, 0.0, 1.0)), 3))
    return out


def _replace_gain_pattern(block: str, new_vals: list[float]) -> str:
    """Replace the `.gain("<…>")` envelope on a voice block with corrected values."""
    vals = " ".join(str(v) for v in new_vals)
    return re.sub(r'\.gain\("<[^"]*>"\)', f'.gain("<{vals}>")', block, count=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2 closed-loop envelope fitting (one step)")
    ap.add_argument("track_dir")
    ap.add_argument("version")
    ap.add_argument("--alpha", type=float, default=0.5, help="correction damping (0=none, 1=full)")
    ap.add_argument("--strudel", default="output.strudel")
    ap.add_argument("--out", default="output_a2.strudel")
    ap.add_argument("--voices", nargs="*", default=["bass", "lead", "drums"])
    args = ap.parse_args(argv)
    track_dir = Path(args.track_dir)
    vdir = track_dir / args.version
    s = (vdir / args.strudel).read_text()

    parts = re.split(r"(?m)^(\$:)", s)
    pre = parts[0]
    blocks = [parts[i + 1] for i in range(1, len(parts), 2)]
    # voice order in the assembled file is bass, lead, drums
    order = ["bass", "lead", "drums"][:len(blocks)]

    new_blocks = []
    for voice, block in zip(order, blocks):
        m = re.search(r'\.gain\("<([^"]*)>"\)', block)
        rend = _rendered_path(vdir, VOICE_STEM[voice])
        orig = track_dir / f"{VOICE_STEM[voice]}.wav"
        if voice in args.voices and m and rend and orig.exists():
            gains = [float(x) for x in m.group(1).split()]
            new = correct_voice_gain(gains, str(orig), str(rend), args.alpha)
            block = _replace_gain_pattern(block, new)
            rng_old = max(gains) - min(gains)
            rng_new = max(new) - min(new)
            print(f"  {voice}: corrected {len(gains)} gains  range {rng_old:.2f}→{rng_new:.2f}")
        else:
            print(f"  {voice}: skipped (no envelope or no render)")
        new_blocks.append("$:" + block.rstrip())

    out = pre + "\n\n".join(new_blocks) + "\n"
    (vdir / args.out).write_text(out)
    print(f"wrote {vdir / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
