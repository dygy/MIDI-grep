#!/usr/bin/env python3
"""Honest multi-dimensional stem self-test: does each rendered stem track the ORIGINAL stem over
TIME — on LOUDNESS (shape), PITCH (notes), RHYTHM (groove), and TIMBRE (tone colour)?

A stem can match loudness perfectly while playing the wrong notes or the wrong rhythm, so loudness
alone (the original metric) is not enough. T1 adds three more time-aligned dimensions, all using the
same resample + best-lag framework (a constant offset / global-tempo difference is forgiven; a
reshuffled shape is not). SHAPE remains the hard gate; pitch/rhythm/timbre are reported diagnostics.

This is the test that earlier proxies missed: aggregate scores read "0.93" and presence read "ON",
yet the stems looked 0% alike because our drums were a uniform block and our bass played constantly
while the original was sparse. The metric here is the TIME-ALIGNED ENVELOPE CORRELATION: resample
each stem's loudness envelope onto a common coarse grid and correlate rendered vs original. A uniform
block vs a dynamic part → near-zero correlation → FAIL. That matches what the eye/ear perceives.

PASS bar (per stem): envelope correlation >= CORR_PASS AND silence-fraction within SIL_TOL.

DRUMS use a SOLO render when available (render_drums_solo.{wav,mp3}) instead of the demucs-separated
stem: demucs files our 808 sub-kick into the BASS stem, so the separated drums stem under-reads the
real drum track. `--render-solo-drums` renders the drum voice alone (BlackHole) to produce it.

Run:  python stem_match.py "<track_dir>" v0NN [--render-solo-drums]
Exit code 0 = PASS, 1 = FAIL (so a loop can gate on it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("librosa required", file=sys.stderr)
    sys.exit(2)

SR = 22050
STEMS = ("bass", "drums", "melodic")
FRAME = 2048
HOP = 5512          # ~0.25 s frames — coarse "shape", not transients
CORR_PASS = 0.45    # envelope correlation needed to count as "tracks the original"
SIL_TOL = 0.30      # rendered silence-fraction must be within this of the original's
N_POINTS = 240      # both envelopes are resampled to this many points before correlating
MAX_LAG = 24        # residual-lag search, in resampled points (~±10% of the song)

# --- T1 multi-dimensional thresholds ---------------------------------------
# A stem can match LOUDNESS perfectly while playing the wrong NOTES or the wrong RHYTHM, so the
# verdict tracks four time-aligned dimensions. Each is correlated/cosine-matched with the SAME
# resample + best-lag framework (so a constant offset / global-tempo difference is forgiven, a
# reshuffled shape is not).
#
# The diagnostic bars below are CALIBRATED by T4 (cross_track_eval.py): each is the "beats chance by
# 2σ" point from comparing matched (render vs own original) against mismatched (render vs OTHER
# tracks' originals) over the fixture set. So a '*' means "better than random at ~95% confidence",
# not an arbitrary number. Re-run `cross_track_eval.py` and update these as the renders improve.
#   chance means measured June 2026 (8 tracks): pitch 0.576, shape 0.162, rhythm 0.123, timbre 0.062.
#   NB pitch (chroma) has a HIGH chance floor (0.576) — the old 0.50 bar was BELOW chance, meaningless.
PITCH_PASS = 0.80   # chroma cosine — same pitch classes at the same time (bass/melodic only)
RHYTHM_PASS = 0.21  # onset-envelope corr at a FINE hop — the groove the 0.25 s loudness env misses
TIMBRE_PASS = 0.15  # MFCC-contour cosine — same tone colour over time
N_RHYTHM = 480      # finer grid for rhythm (~0.3 s/pt) so hit-timing isn't averaged away
RHYTHM_HOP = 512    # ~23 ms onset frames before resampling
PITCHLESS = {"drums"}  # stems with no meaningful pitch — chroma is noise, so skip it


def _envelope(y: np.ndarray) -> np.ndarray:
    env = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    return env


def _resample_matrix(m: np.ndarray, n: int) -> np.ndarray:
    """Resample a [D x T] feature matrix to [D x n] along time (per-row linear interp)."""
    if m.shape[1] < 2:
        return np.zeros((m.shape[0], n))
    xs = np.linspace(0, m.shape[1] - 1, n)
    src = np.arange(m.shape[1])
    return np.vstack([np.interp(xs, src, m[d]) for d in range(m.shape[0])])


def _best_lag_colsim(o: np.ndarray, r: np.ndarray) -> float:
    """Best-lag mean per-column COSINE similarity between two [D x n] feature matrices. Used for
    multi-dim features (chroma, MFCC): at each lag, align in time and average the per-frame cosine."""
    n = o.shape[1]
    def colcos(a, b):
        num = (a * b).sum(axis=0)
        den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9
        return float(np.mean(num / den))
    best = colcos(o, r)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            a, b = o[:, lag:], r[:, :n - lag]
        else:
            a, b = o[:, :n + lag], r[:, -lag:]
        if a.shape[1] > n // 2:
            best = max(best, colcos(a, b))
    return best


def _pitch_match(o: np.ndarray, r: np.ndarray) -> float:
    """Chroma agreement over time: do we play the same pitch CLASSES at the same moments?"""
    co = librosa.feature.chroma_cqt(y=o, sr=SR, hop_length=HOP)
    cr = librosa.feature.chroma_cqt(y=r, sr=SR, hop_length=HOP)
    return _best_lag_colsim(_resample_matrix(co, N_POINTS), _resample_matrix(cr, N_POINTS))


def _timbre_match(o: np.ndarray, r: np.ndarray) -> float:
    """MFCC-contour agreement over time (coeffs 1-12, dropping c0 which is just loudness)."""
    mo = librosa.feature.mfcc(y=o, sr=SR, n_mfcc=13, hop_length=HOP)[1:]
    mr = librosa.feature.mfcc(y=r, sr=SR, n_mfcc=13, hop_length=HOP)[1:]
    # standardise each coeff so cosine isn't dominated by one large-variance coefficient
    def z(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    return _best_lag_colsim(z(_resample_matrix(mo, N_POINTS)), z(_resample_matrix(mr, N_POINTS)))


def _rhythm_match(o: np.ndarray, r: np.ndarray) -> float:
    """Onset-strength envelope correlation at a FINE hop — captures hit-timing / groove that the
    coarse 0.25 s loudness envelope averages away. 1-D, so reuse the scalar best-lag correlation."""
    oo = librosa.onset.onset_strength(y=o, sr=SR, hop_length=RHYTHM_HOP)
    ro = librosa.onset.onset_strength(y=r, sr=SR, hop_length=RHYTHM_HOP)
    oo, ro = _resample(oo, N_RHYTHM), _resample(ro, N_RHYTHM)
    on = (oo - oo.mean()) / (oo.std() + 1e-9)
    rn = (ro - ro.mean()) / (ro.std() + 1e-9)
    # finer grid → wider lag budget (same ~10% of song)
    n = N_RHYTHM
    best = float(np.mean(on * rn))
    cap = N_RHYTHM // 10
    for lag in range(-cap, cap + 1):
        if lag >= 0:
            a, b = on[lag:], rn[:n - lag]
        else:
            a, b = on[:n + lag], rn[-lag:]
        m = min(len(a), len(b))
        if m > n // 2:
            best = max(best, float(np.mean(a[:m] * b[:m])))
    return best


def _resample(env: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Resample an envelope to a fixed number of points. This normalises the two stems to a common
    timeline so a constant time offset or a global tempo/length difference (e.g. the recorder trims
    leading silence, making our render shorter than the untouched original) does NOT read as a shape
    mismatch — only a genuinely different loud/quiet SHAPE does. A uniform block stays uniform after
    resampling (std≈0 → correlation≈0), so it still FAILS, which is the point."""
    if len(env) < 2:
        return np.zeros(n)
    xs = np.linspace(0, len(env) - 1, n)
    return np.interp(xs, np.arange(len(env)), env)


def _best_lag_corr(oe_n: np.ndarray, re_n: np.ndarray) -> float:
    """Max correlation over a capped residual lag — tolerates a constant offset, not a reshuffle."""
    n = len(oe_n)
    best = float(np.mean(oe_n * re_n))  # zero-lag
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            a, b = oe_n[lag:], re_n[:n - lag]
        else:
            a, b = oe_n[:n + lag], re_n[-lag:]
        m = min(len(a), len(b))
        if m > n // 2:
            best = max(best, float(np.mean(a[:m] * b[:m])))
    return best


def _silence_frac(env: np.ndarray) -> float:
    peak = float(env.max()) if len(env) else 0.0
    if peak < 1e-9:
        return 1.0
    return float(np.mean(env < peak * 0.1))


def _rendered_path(version_dir: Path, stem: str) -> Path:
    for ext in (".wav", ".mp3"):
        p = version_dir / f"render_{stem}{ext}"
        if p.exists():
            return p
    return version_dir / f"render_{stem}.wav"


def _solo_drums_path(version_dir: Path) -> Path | None:
    """A directly-rendered drums-only file, if present (NO demucs). Preferred for the drums verdict
    because demucs files our 808 sub-kick into the bass stem, leaving the separated drums stem too
    thin to reflect the real drum track. Produce one with --render-solo-drums."""
    for name in ("render_drums_solo.wav", "render_drums_solo.mp3"):
        p = version_dir / name
        if p.exists():
            return p
    return None


def _load(p: Path) -> np.ndarray:
    if p is None or not p.exists():
        return np.zeros(0)
    try:
        y, _ = librosa.load(str(p), sr=SR, mono=True)
        return y
    except Exception:  # noqa: BLE001
        return np.zeros(0)


def _shape_corr(o: np.ndarray, r: np.ndarray, stem: str = "") -> dict:
    """Compare two stem signals across FOUR time-aligned dimensions (all resample + best-lag, so a
    constant offset / global-tempo difference is forgiven but a reshuffled shape is not):

      shape  — loudness envelope correlation (the primary gate: right loud/quiet over time)
      pitch  — chroma cosine: right pitch classes at the right time (skipped for drums)
      rhythm — onset-envelope correlation at a fine hop: right groove / hit-timing
      timbre — MFCC-contour cosine: right tone colour over time

    The per-stem PASS stays defined by SHAPE + silence (backward-compatible with the existing gate);
    pitch/rhythm/timbre are reported as diagnostics (each with its own indicative pass mark) so a stem
    that nails loudness but plays wrong notes/rhythm is no longer invisible."""
    if len(o) < SR or len(r) < SR:
        return {"corr": 0.0, "sil_orig": None, "sil_rend": None, "pass": False, "note": "missing/short stem",
                "pitch": None, "rhythm": None, "timbre": None}
    oe, re_ = _envelope(o), _envelope(r)
    oe_r, re_r = _resample(oe), _resample(re_)
    oe_n = (oe_r - oe_r.mean()) / (oe_r.std() + 1e-9)
    re_n = (re_r - re_r.mean()) / (re_r.std() + 1e-9)
    corr = _best_lag_corr(oe_n, re_n)
    so, sr_ = _silence_frac(oe), _silence_frac(re_)
    ok = (corr >= CORR_PASS) and (abs(so - sr_) <= SIL_TOL)
    out = {"corr": round(corr, 3), "sil_orig": round(so, 2), "sil_rend": round(sr_, 2), "pass": ok}
    # Extra dimensions (diagnostic). Guarded so a feature-extraction failure never breaks the verdict.
    try:
        out["pitch"] = None if stem in PITCHLESS else round(_pitch_match(o, r), 3)
    except Exception:  # noqa: BLE001
        out["pitch"] = None
    try:
        out["rhythm"] = round(_rhythm_match(o, r), 3)
    except Exception:  # noqa: BLE001
        out["rhythm"] = None
    try:
        out["timbre"] = round(_timbre_match(o, r), 3)
    except Exception:  # noqa: BLE001
        out["timbre"] = None
    return out


def stem_shape_match(track_dir: Path, version_dir: Path) -> dict:
    results = {}
    for stem in STEMS:
        o = _load(track_dir / f"{stem}.wav")
        # Drums: prefer a direct solo-drums render over the demucs-separated stem (demucs mis-routes
        # our 808 sub-kick to the bass stem, so the separated drums stem under-reads the real track).
        solo = _solo_drums_path(version_dir) if stem == "drums" else None
        r = _load(solo) if solo is not None else _load(_rendered_path(version_dir, stem))
        res = _shape_corr(o, r, stem)
        if stem == "drums" and solo is not None and not res.get("note"):
            res["note"] = "solo render, no demucs"
        results[stem] = res
    overall = all(v["pass"] for v in results.values())
    mean_corr = round(float(np.mean([v["corr"] for v in results.values()])), 3)
    return {"stems": results, "mean_corr": mean_corr, "pass": overall}


def _extract_drum_voice(strudel_path: Path) -> str | None:
    """Pull setcps + the drum `$:` block (the one using `.bank(`) out of a full Strudel file, so the
    drums can be rendered in ISOLATION. Returns None if no drum voice is found."""
    try:
        text = strudel_path.read_text()
    except OSError:
        return None
    pre = "\n".join(ln for ln in text.splitlines() if ln.strip().startswith("setcps"))
    # split into `$:` blocks; keep the one containing .bank( (drums)
    import re
    parts = re.split(r"(?m)^\$:", text)
    drum_block = next((b for b in parts[1:] if ".bank(" in b), None)
    if drum_block is None:
        return None
    return f"{pre}\n\n$:{drum_block.rstrip()}\n"


def _render_solo_drums(track_dir: Path, version_dir: Path) -> Path | None:
    """Render the drum voice alone via BlackHole → render_drums_solo.wav (the honest, demucs-free drum
    signal). Skips gracefully if the recorder/strudel file is missing. Run this WITHOUT any other
    BlackHole render in flight — concurrent captures from the one device produce silence."""
    import subprocess
    recorder = Path(__file__).resolve().parent.parent / "node" / "dist" / "record-strudel-blackhole.js"
    strudel = version_dir / "output.strudel"
    if not recorder.exists() or not strudel.exists():
        print(f"  [solo-drums] skipped (recorder or output.strudel missing)", file=sys.stderr)
        return None
    voice = _extract_drum_voice(strudel)
    if voice is None:
        print("  [solo-drums] no drum voice found in output.strudel", file=sys.stderr)
        return None
    solo_strudel = version_dir / "drums_solo.strudel"
    solo_strudel.write_text(voice)
    out = version_dir / "render_drums_solo.wav"
    # Match the original drums' duration so the timelines line up before resampling.
    try:
        dur = int(librosa.get_duration(path=str(track_dir / "drums.wav"))) + 2
    except Exception:  # noqa: BLE001
        dur = 170
    print(f"  [solo-drums] rendering drum voice alone ({dur}s) via BlackHole…", file=sys.stderr)
    try:
        subprocess.run(["node", str(recorder), str(solo_strudel), "-o", str(out), "-d", str(dur)],
                       check=False, capture_output=True, text=True, timeout=dur + 120)
    except Exception as e:  # noqa: BLE001
        print(f"  [solo-drums] render failed: {e}", file=sys.stderr)
        return None
    return out if out.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stem SHAPE match: rendered vs original over time")
    ap.add_argument("track_dir")
    ap.add_argument("version")
    ap.add_argument("--render-solo-drums", action="store_true",
                    help="Render the drum voice alone (BlackHole) and use it for the drums verdict "
                         "instead of the demucs-separated stem (avoids the 808→bass mis-routing). "
                         "Run with no other BlackHole render in flight.")
    args = ap.parse_args(argv)
    track_dir = Path(args.track_dir)
    version_dir = track_dir / args.version
    if args.render_solo_drums and _solo_drums_path(version_dir) is None:
        _render_solo_drums(track_dir, version_dir)
    res = stem_shape_match(track_dir, version_dir)
    print(f"=== STEM MATCH: {args.version}  (shape>={CORR_PASS} sil±{SIL_TOL} | "
          f"pitch>={PITCH_PASS} rhythm>={RHYTHM_PASS} timbre>={TIMBRE_PASS}) ===")

    def _dim(val, bar):
        if val is None:
            return "   --  "
        return f"{val:+.2f}{'*' if val >= bar else ' '}"  # '*' marks a dimension over its bar

    for stem, v in res["stems"].items():
        mark = "PASS" if v["pass"] else "FAIL"
        print(f"  {stem:8} {mark}  shape {v['corr']:+.3f}  "
              f"pitch {_dim(v.get('pitch'), PITCH_PASS)}  "
              f"rhythm {_dim(v.get('rhythm'), RHYTHM_PASS)}  "
              f"timbre {_dim(v.get('timbre'), TIMBRE_PASS)}  "
              f"sil {v['sil_orig']}/{v['sil_rend']}"
              + (f"  ({v['note']})" if v.get('note') else ""))
    print("  (shape = the gate; pitch/rhythm/timbre are diagnostics, '*' = over bar)")
    print(f"=== mean shape-corr {res['mean_corr']}  → OVERALL {'PASS' if res['pass'] else 'FAIL'} ===")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
