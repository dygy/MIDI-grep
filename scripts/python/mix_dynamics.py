#!/usr/bin/env python3
"""Loudness of the generated Strudel, matched to the original by measurement.

The similarity score is level-invariant (``compare_audio.rms_normalize`` scales both signals to one RMS before any
feature is computed), so the calibration loop never sees how loud a render is: a raw sample render sat 6-13 dB under the
mastered instrumental (crest 17-18 dB against ~10 dB) and scored the same as a loud one. This module closes that gap with
ONE measured lever, the generator's ``--master-gain``, read off the render itself:

  master   new = old * 10 ** ((reference_rms - render_rms) / 20)   (a few rounds: clipping at full scale makes the
           map from gain to RMS sub-linear, so one step lands short)
  crest    not a lever: with the RMS matched and the peak at full scale it follows (peak - rms). It is VERIFIED against the
           reference's crest (``level_verdict``)
  clipping the limiter, and REPORTED. superdough has no master bus (the orbit has reverb, delay, djf and duck only), and the
           per-voice tools do not bound the SUM: `.compressor` is built per hap and its attack cannot catch a transient
           (a bass voice limited to -0.5 dBFS still peaked at +6.6 dBFS), and a tanh saturator per layer with ceilings
           that add up to the reference peak needed master +17 dB and still ended at crest 16.6 (both measured on
           VAGABUNDO, 2026-10-20) - while a plain gain with the output clipping at full scale (what the Web Audio
           destination does in the REPL too) reached the reference rms and crest with no score loss. The cost is shown, not
           hidden: ``clip_frac`` (share of render samples at full scale) is printed next to ``plateau_frac`` (the share of
           the reference's samples within ``PLATEAU_DB`` of its own peak, i.e. the mass at the ceiling a mastered original
           has): 0.5-1.5 % clipped against 1.3-1.7 % near the peak in the four measured pieces.

A round is kept only when level (RMS) and crest match the reference AND neither score (overall, section-aware)
dropped by more than the calibration noise; otherwise the unmixed BEST stays.

``shares`` renders each voice alone and compares its energy share of the mix with the original stems' (diagnostic: the
share-of-mix balance from stems rather than from full-mix bands). Applied as gains it was measured worse or inconsistent
on the score (VAGABUNDO sample 0.7805 vs 0.7895, Regime sample 0.8164 vs 0.8112 with section-aware 0.7960 vs 0.7993),
because Demucs stems carry bleed (Regime's drum stem holds 3 % of the energy, the bass stem 85 %), so it is reported, not applied.

CLI::

    mix_dynamics.py next-master --reference REF.wav --render R.wav --seconds N --master 0.78
    mix_dynamics.py choose  --reference REF.wav --seconds N --best-cmp C.json --tolerance T --round WAV CMP ...
    mix_dynamics.py report  --reference REF.wav --render R.wav --seconds N
    mix_dynamics.py shares  --strudel S --stems-dir D --seconds N
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer  # noqa: E402

VOICES = ("bass", "melodic", "drums")
# strudel channel labels (generate_dynamic_strudel.py comments) -> voice
LABEL_VOICE = {"BASS": "bass", "LEAD": "melodic", "CHORDS": "melodic", "DRUMS": "drums"}
# What counts as "the level and dynamics of the original" (dB)
RMS_TOL_DB = 1.0
CREST_TOL_DB = 1.5
# a sample is clipped at (or above) this amplitude: the 16-bit writer saturates at 32767/32768
FULL_SCALE = 0.9995
# the mastered original's "mass at the ceiling": the share of its samples within this many dB of its own peak
PLATEAU_DB = 3.0


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-9))


def load_mono(path: Path, seconds: float | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1)
    if seconds:
        y = y[: int(seconds * sr)]
    return y, sr


def profile(y: np.ndarray, sr: int) -> dict[str, float]:
    """rms / peak / crest (dBFS), the 3 s short-term loudness distribution, and two shares of samples: those at full
    scale (``clip_frac``) and those within PLATEAU_DB of the signal's own peak (``plateau_frac``)."""
    a = np.abs(y)
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    pk = float(a.max()) if len(a) else 0.0
    win, hop = int(3 * sr), int(sr)
    st = [20 * math.log10(max(float(np.sqrt(np.mean(y[i:i + win] ** 2))), 1e-9)) for i in range(0, max(len(y) - win, 1), hop)]
    return {"rms_db": db(rms), "peak_db": db(pk), "crest_db": db(pk) - db(rms),
            "st_p10_db": float(np.percentile(st, 10)), "st_p50_db": float(np.percentile(st, 50)),
            "st_p90_db": float(np.percentile(st, 90)),
            "clip_frac": float(np.mean(a >= FULL_SCALE)) if len(a) else 0.0,
            "plateau_frac": float(np.mean(a >= pk * 10 ** (-PLATEAU_DB / 20.0))) if len(a) else 0.0}


def level_verdict(reference: dict, render_profile: dict) -> dict:
    """Does the render have the reference's level (RMS) and dynamics (crest)? (``clip_frac`` is reported, not judged.)"""
    rms_err = render_profile["rms_db"] - reference["rms_db"]
    crest_err = render_profile["crest_db"] - reference["crest_db"]
    problems = []
    if abs(rms_err) > RMS_TOL_DB:
        problems.append(f"rms {rms_err:+.1f} dB off")
    if abs(crest_err) > CREST_TOL_DB:
        problems.append(f"crest {crest_err:+.1f} dB off")
    return {"rms_err_db": rms_err, "crest_err_db": crest_err, "problems": problems, "ok": not problems}


def next_master(reference: dict, render_profile: dict, master: float) -> float:
    """The ``--master-gain`` for the next round: close the RMS error."""
    return round(master * 10 ** ((reference["rms_db"] - render_profile["rms_db"]) / 20.0), 4)


def choose_round(best_scores: dict, rounds: list[dict], tolerance: float) -> int | None:
    """Index of the round to keep, or None (the unmixed best stays).

    A round qualifies when its level and crest match the reference (``level_verdict`` ok) and neither score
    (overall, section-aware) dropped more than ``tolerance`` below the unmixed best. Of the qualifying rounds the
    highest overall wins."""
    ok = [i for i, r in enumerate(rounds)
          if r["level"]["ok"]
          and r["overall"] >= best_scores["overall"] - tolerance
          and r["section"] >= best_scores["section"] - tolerance]
    return max(ok, key=lambda i: rounds[i]["overall"]) if ok else None


# ---------------------------------------------------------------- stem shares (diagnostic)
def split_voices(code: str) -> dict[str, str]:
    """One program per voice: every ``$:`` channel of another voice is commented out (channels are separated by blank
    lines and end with a ``// LABEL - comment`` tail, see generate_dynamic_strudel.assemble)."""
    blocks = code.split("\n\n")

    def label(b: str) -> str | None:
        m = re.search(r"// (?:texture — )?([A-Z]+) — [^\n]*$", b.rstrip())
        return m.group(1) if m else None

    out = {}
    for voice in VOICES:
        keep = []
        for b in blocks:
            if b.lstrip().startswith("$:") and LABEL_VOICE.get(label(b) or "") != voice:
                b = "\n".join("// " + ln for ln in b.split("\n"))
            keep.append(b)
        out[voice] = "\n\n".join(keep)
    return out


def render(strudel: Path, wav: Path, seconds: float) -> None:
    r = subprocess.run(["node", str(renderer.renderer_script()), str(strudel), "-o", str(wav), "-d", f"{seconds:g}"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError(f"render of {strudel} failed (rc={r.returncode}): {r.stderr[-400:] or r.stdout[-400:]}")


def energy_shares(energies: dict[str, float]) -> dict[str, float]:
    total = sum(energies.values())
    return {v: (e / total if total > 0 else 0.0) for v, e in energies.items()}


def share_errors_db(original: dict[str, float], rendered: dict[str, float]) -> dict[str, float]:
    """Per voice, the render's energy share over the original's, in dB (0 = the same share of the mix)."""
    return {v: 10 * math.log10(max(rendered[v], 1e-12) / max(original[v], 1e-12)) for v in original}


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next-master")
    n.add_argument("--reference", type=Path, required=True)
    n.add_argument("--render", type=Path, required=True)
    n.add_argument("--seconds", type=float, required=True)
    n.add_argument("--master", type=float, required=True, help="the --master-gain the render was generated with")
    c = sub.add_parser("choose", help="pick the best qualifying round: --round WAV CMP_JSON (repeatable)")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--seconds", type=float, required=True)
    c.add_argument("--best-cmp", type=Path, required=True, help="comparison.json of the unmixed best render")
    c.add_argument("--round", nargs=2, action="append", metavar=("WAV", "CMP"), required=True)
    c.add_argument("--tolerance", type=float, required=True, help="score points a round may lose (calibration noise)")
    r = sub.add_parser("report")
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--render", type=Path, required=True)
    r.add_argument("--seconds", type=float, required=True)
    s = sub.add_parser("shares")
    s.add_argument("--strudel", type=Path, required=True)
    s.add_argument("--stems-dir", type=Path, required=True)
    s.add_argument("--seconds", type=float, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "shares":
        orig, rend = {}, {}
        with tempfile.TemporaryDirectory(prefix="mixshares.") as td:
            for voice, code in split_voices(a.strudel.read_text()).items():
                st, wav = Path(td) / f"{voice}.strudel", Path(td) / f"{voice}.wav"
                st.write_text(code)
                render(st, wav, a.seconds)
                y, _ = load_mono(wav, a.seconds)
                rend[voice] = float(np.mean(y ** 2))
        for voice in VOICES:
            y, _ = load_mono(a.stems_dir / f"{voice}.wav", a.seconds)
            orig[voice] = float(np.mean(y ** 2))
        so, sr_ = energy_shares(orig), energy_shares(rend)
        err = share_errors_db(so, sr_)
        for voice in VOICES:
            print(f"{voice:8s} original share {100 * so[voice]:5.1f} %  render share {100 * sr_[voice]:5.1f} %  error {err[voice]:+.1f} dB")
        return 0

    ref = profile(*load_mono(a.reference, a.seconds))
    if a.cmd == "choose":
        def scores(path):
            cmp = json.loads(Path(path).read_text())["comparison"]
            return {"overall": float(cmp["overall_similarity"]), "section": float(cmp.get("section_aware_similarity", 0.0))}
        best = scores(a.best_cmp)
        rounds = []
        for wav, cmp in a.round:
            prof = profile(*load_mono(Path(wav), a.seconds))
            rounds.append({**scores(cmp), "level": level_verdict(ref, prof), "profile": prof})
        pick = choose_round(best, rounds, a.tolerance)
        print(f"reference rms {ref['rms_db']:.1f} peak {ref['peak_db']:.1f} crest {ref['crest_db']:.1f} plateau "
              f"{100 * ref['plateau_frac']:.2f} % within {PLATEAU_DB:g} dB of peak | unmixed best overall {best['overall']:.4f} section {best['section']:.4f}", file=sys.stderr)
        for i, r_ in enumerate(rounds):
            lost = r_["overall"] < best["overall"] - a.tolerance or r_["section"] < best["section"] - a.tolerance
            why = "kept" if i == pick else "; ".join(r_["level"]["problems"] + (["score loss beyond tolerance"] if lost else [])) \
                or "qualifies, a higher overall won"
            print(f"round {i + 1}: overall {r_['overall']:.4f} section {r_['section']:.4f} rms {r_['profile']['rms_db']:.1f} "
                  f"peak {r_['profile']['peak_db']:.1f} crest {r_['profile']['crest_db']:.1f} clipped "
                  f"{100 * r_['profile']['clip_frac']:.2f} % -> {why}", file=sys.stderr)
        print("none" if pick is None else pick + 1)
        return 0
    rend = profile(*load_mono(a.render, a.seconds))
    if a.cmd == "next-master":
        print(next_master(ref, rend, a.master))
        return 0
    print(json.dumps({"reference": ref, "render": rend, "verdict": level_verdict(ref, rend)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
