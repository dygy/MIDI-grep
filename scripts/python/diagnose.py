#!/usr/bin/env python3
"""Self-diagnosis of a promoted piece: what is broken, ranked by the score it costs - not just the overall number.

Why (owner, 2026-11-08): BibiBabydoll was reported by its overall score while ``tempo_similarity`` sat at 0.17 (the
stamped tempo 112.5 BPM had confidence 0.045, the lowest of 19 tracks, and the original measured 101.3). The overall
score averages such a failure away; this module makes it impossible to miss.

What it checks, from files the pipeline already wrote (``comparison.json``, ``stem_comparison.json``, the track's
``metadata.json``) - no audio is loaded, so it costs no memory:

* every score component of the mix and of each stem (bass / drums / melodic) against the range the REFERENCE tracks
  reach in the same mode (``eval/datasets/reference_metrics.json``, written by ``record-refs`` from promoted reference
  pieces and committed, so a cache wipe cannot erase it). A component below every reference by more than
  ``calibration_noise`` is RED; a mix component's cost is its weight in the overall score (``compare_audio``'s own
  constants) times its gap to the references' median;
* the tempo: the stamped bpm against the original's measured tempo and the render's, within one tempogram bin
  (``tempo_refine.tempogram_bin_bpm``) at a musical ratio (1, 2, 1/2), and the stamp's phase-coherence confidence
  against the lowest the references have;
* the worst section windows, each with the component that drags it down.

A stem the original does not really have (Tatuagem No Pescoço: bass.wav is bleed at -63 dB, the low end is a tuned
808 kick in the drums) is ABSENT: its RMS is under the generator's gate ``silence`` fraction of the loudest stem's.
Its per-stem metrics compare a render with bleed and are skipped (listed in ``absent_stems``), not flagged.

Usage:
  diagnose.py run <vNNN dir> [--track <track dir>] [--json-out <file>]   writes <vNNN>/diagnosis.json, prints it
  diagnose.py record-refs <vNNN dir>...                                  (re)writes the reference ranges
  diagnose.py table <pipeline summary.json>...                           one line per run, red flags first
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
REFS = ROOT / "eval" / "datasets" / "reference_metrics.json"
THRESHOLDS = ROOT / "eval" / "thresholds.yaml"
STEMS = ("bass", "drums", "melodic")
STEM_STATIC = ("overall_similarity", "frequency_balance_similarity", "brightness_similarity", "energy_similarity")
TEMPO_RATIOS = (1.0, 2.0, 0.5)          # half / double time are the same pulse; anything else is a different tempo
WORST_SECTIONS = 3                      # how many section windows the report names
SECTION_COMPONENTS = ("mfcc", "band", "energy", "chroma_corr", "mel_corr", "onset_f1", "envelope_corr")


def _ca():
    import compare_audio
    return compare_audio


def _noise() -> float:
    import yaml
    return float(yaml.safe_load(THRESHOLDS.read_text())["calibration_noise"])


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _mode(comp: dict, vdir: Path) -> str:
    m = comp.get("generation_mode")
    if m:
        return str(m)
    meta = _load(vdir / "metadata.json") or {}
    return str(meta.get("generation_mode") or meta.get("mode") or "unknown")


def stem_sources(vdir: Path, track: Path) -> dict[str, Path]:
    """The original each render stem was compared with: ``vNNN/stem_sources.json`` (stage 10 writes it; a stage-3c
    view's bass/drums when one existed), else the track's own stems."""
    rec = _load(vdir / "stem_sources.json") or {}
    return {st: (track / rec[st]) if st in rec else (track / f"{st}.wav") for st in STEMS}


def absent_stems(track: Path, sources: dict[str, Path] | None = None) -> list[str]:
    """Stems whose RMS is below the generator's gate silence fraction of the loudest stem's (bleed, not a part).
    ``sources`` = the files the stem comparison used (:func:`stem_sources`); default the track's stems."""
    import inspect
    import numpy as np
    import soundfile as sf
    import generate_dynamic_strudel as G
    silence = inspect.signature(G.bar_env_pattern).parameters["silence"].default
    rms = {}
    for st in STEMS:
        p = (sources or {}).get(st) or track / f"{st}.wav"
        if p.is_file():
            y, _ = sf.read(str(p), dtype="float32", always_2d=True)
            rms[st] = float(np.sqrt(np.mean(np.square(y.mean(1)))))
    loud = max(rms.values(), default=0.0)
    return [st for st, v in rms.items() if loud > 0 and v < silence * loud]


def metrics(vdir: Path, skip_stems: tuple[str, ...] = ()) -> dict[str, float]:
    """Flat ``name -> value`` of every component the diagnosis compares (all higher-is-better)."""
    ca = _ca()
    comp = _load(vdir / "comparison.json")
    if comp is None:
        raise FileNotFoundError(f"{vdir}/comparison.json missing or unreadable")
    c = comp["comparison"]
    out = {f"mix.{k}": float(c[k]) for k in ca.STATIC_WEIGHTS if k in c}
    t = c.get("temporal") or {}
    out.update({f"mix.temporal.{k}": float(t[k]) for k in ca.TEMPORAL_COMPONENTS if k in t})
    for k in ("overall_similarity", "section_aware_similarity", "margin_over_chance"):
        if isinstance(c.get(k), (int, float)):
            out[f"mix.{k}"] = float(c[k])
    sc = _load(vdir / "stem_comparison.json") or {}
    for stem, d in (sc.get("stems") or {}).items():
        if stem in skip_stems:
            continue
        sc_c = d.get("comparison") or {}
        st = sc_c.get("temporal") or {}
        out.update({f"{stem}.{k}": float(sc_c[k]) for k in STEM_STATIC if isinstance(sc_c.get(k), (int, float))})
        out.update({f"{stem}.temporal.{k}": float(st[k]) for k in ca.TEMPORAL_COMPONENTS
                    if isinstance(st.get(k), (int, float))})
    return out


def weight(name: str) -> float:
    """The component's weight in the MIX overall score (0 for stem metrics and aggregates)."""
    ca = _ca()
    if name.startswith("mix.temporal."):
        return (1 - ca.STATIC_SHARE) / len(ca.TEMPORAL_COMPONENTS)
    k = name.removeprefix("mix.")
    return ca.STATIC_SHARE * ca.STATIC_WEIGHTS[k] if name.startswith("mix.") and k in ca.STATIC_WEIGHTS else 0.0


def tempo_check(comp: dict, meta: dict, refs: dict) -> dict:
    """Stamped bpm vs the original's and the render's measured tempo, and the stamp's confidence vs the references."""
    import tempo_refine as TR
    stamped = float(meta.get("bpm") or 0.0)
    orig = float(comp["original"]["rhythm"]["tempo"])
    rend = float(comp["rendered"]["rhythm"]["tempo"])
    tol = TR.tempogram_bin_bpm(max(stamped, orig, 1.0), 22050)

    def agrees(a: float, b: float) -> bool:
        return any(abs(a - b * r) <= tol for r in TEMPO_RATIOS)

    conf = meta.get("bpm_confidence")
    ref_conf = refs.get("bpm_confidence") or []
    out = {"stamped_bpm": stamped, "original_measured": orig, "render_measured": rend, "tolerance_bpm": tol,
           "stamp_matches_original": agrees(orig, stamped), "render_matches_stamp": agrees(rend, stamped),
           "confidence": conf, "reference_confidence_min": min(ref_conf) if ref_conf else None,
           "original_candidates": comp["original"]["rhythm"].get("tempo_candidates")}
    out["confidence_below_references"] = bool(conf is not None and ref_conf and float(conf) < min(ref_conf))
    return out


def worst_sections(comp: dict) -> list[dict]:
    c = comp["comparison"]
    wins = [w for w in c.get("section_windows") or [] if isinstance(w.get("win"), (int, float))]
    whole = {k: (c.get("temporal") or {}).get(k) for k in SECTION_COMPONENTS}
    whole.update({"mfcc": c.get("mfcc_similarity"), "band": c.get("frequency_balance_similarity"),
                  "energy": c.get("energy_similarity")})
    out = []
    for w in sorted(wins, key=lambda w: w["win"])[:WORST_SECTIONS]:
        gaps = {k: whole[k] - w[k] for k in SECTION_COMPONENTS
                if isinstance(w.get(k), (int, float)) and isinstance(whole.get(k), (int, float))}
        drag = max(gaps, key=gaps.get) if gaps else None
        bd = w.get("band_diff") or {}
        band = max(bd, key=lambda b: abs(bd[b])) if bd else None
        out.append({"t0": w["t0"], "t1": w["t1"], "score": w["win"], "drag": drag,
                    "drag_value": w.get(drag) if drag else None, "drag_track_value": whole.get(drag) if drag else None,
                    "worst_band": band, "worst_band_diff": bd.get(band) if band else None})
    return out


def diagnose(vdir: Path, track: Path | None = None, refs_path: Path = REFS) -> dict:
    vdir = vdir.resolve()
    track = (track or vdir.parent).resolve()
    comp = _load(vdir / "comparison.json")
    if comp is None:
        raise FileNotFoundError(f"{vdir}/comparison.json missing or unreadable")
    meta = _load(track / "metadata.json") or {}
    mode = _mode(comp, vdir)
    allrefs = _load(refs_path) or {}
    refs = (allrefs.get("modes") or {}).get(mode) or {}
    noise = _noise()
    absent = absent_stems(track, stem_sources(vdir, track))
    vals = metrics(vdir, tuple(absent))
    red, ok = [], []
    for name, v in sorted(vals.items()):
        series = (refs.get("metrics") or {}).get(name)
        if not series:
            continue
        lo, med = min(series), statistics.median(series)
        row = {"metric": name, "value": round(v, 4), "ref_min": round(lo, 4), "ref_median": round(med, 4),
               "cost": round(weight(name) * max(0.0, med - v), 4)}
        (red if v < lo - noise else ok).append(row)
    red.sort(key=lambda r: (-r["cost"], r["value"] - r["ref_min"]))
    tempo = tempo_check(comp, meta, refs)
    flags = []
    if not tempo["stamp_matches_original"]:
        flags.append(f"tempo: stamped {tempo['stamped_bpm']:.2f} BPM but the original measures "
                     f"{tempo['original_measured']:.2f} (no x1/x2/x0.5 within {tempo['tolerance_bpm']:.2f} BPM)"
                     + (f"; stamp confidence {tempo['confidence']:.3f} < references' lowest "
                        f"{tempo['reference_confidence_min']:.3f}" if tempo["confidence_below_references"] else "")
                     + " - the piece likely plays at the wrong tempo")
    elif tempo["confidence_below_references"]:
        flags.append(f"tempo: stamp confidence {tempo['confidence']:.3f} < references' lowest "
                     f"{tempo['reference_confidence_min']:.3f} (the stamp agrees with the original's measured tempo)")
    if not tempo["render_matches_stamp"]:
        flags.append(f"tempo: the render measures {tempo['render_measured']:.2f} BPM, the piece is written at "
                     f"{tempo['stamped_bpm']:.2f}")
    for r in red:
        flags.append(f"{r['metric']} {r['value']:.3f} < every reference ({r['ref_min']:.3f}..median "
                     f"{r['ref_median']:.3f})" + (f", costs {r['cost']:.3f} of the overall score" if r["cost"] else ""))
    return {"version_dir": str(vdir), "track": track.name, "mode": mode, "references": len(refs.get("sources") or []),
            "red_count": len(flags), "flags": flags, "red_metrics": red, "tempo": tempo, "absent_stems": absent,
            "worst_sections": worst_sections(comp), "within_reference_range": ok,
            "overall": vals.get("mix.overall_similarity"), "section_aware": vals.get("mix.section_aware_similarity")}


def render_text(d: dict) -> str:
    lines = [f"DIAGNOSIS {d['track'][:40]} {Path(d['version_dir']).name} {d['mode']}: "
             f"overall {d['overall']:.4f} / section-aware {d['section_aware']:.4f} - "
             + (f"{d['red_count']} RED" if d["red_count"] else "nothing below the references")
             + ("" if d["references"] else " (NO reference ranges for this mode - run record-refs)")
             + (f" [absent in the original: {', '.join(d['absent_stems'])}]" if d.get("absent_stems") else "")]
    lines += [f"  RED {f}" for f in d["flags"]]
    for s in d["worst_sections"]:
        lines.append(f"  worst {s['t0']:.0f}-{s['t1']:.0f}s score {s['score']:.3f}: {s['drag']} "
                     f"{s['drag_value']:.3f} (track {s['drag_track_value']:.3f})"
                     + (f", {s['worst_band']} {s['worst_band_diff']:+.3f}" if s["worst_band"] else ""))
    return "\n".join(lines)


def record_refs(vdirs: list[Path], out: Path = REFS) -> dict:
    """Reference ranges per mode from promoted reference pieces (+ their tracks' tempo-stamp confidence)."""
    modes: dict[str, dict] = {}
    for v in vdirs:
        v = v.resolve()
        comp = _load(v / "comparison.json")
        if comp is None:
            raise FileNotFoundError(f"{v}/comparison.json missing")
        m = modes.setdefault(_mode(comp, v), {"sources": [], "metrics": {}, "bpm_confidence": []})
        m["sources"].append(f"{v.parent.name}/{v.name}")
        for k, val in metrics(v).items():
            m["metrics"].setdefault(k, []).append(round(val, 4))
        conf = (_load(v.parent / "metadata.json") or {}).get("bpm_confidence")
        if conf is not None:
            m["bpm_confidence"].append(float(conf))
    doc = {"note": "written by scripts/python/diagnose.py record-refs; ranges the reference tracks reach per mode",
           "modes": modes}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def table(summaries: list[Path]) -> str:
    rows = []
    for p in summaries:
        s = _load(p)
        if not s:
            rows.append(f"{p}: unreadable summary")
            continue
        for r in s.get("runs", []):
            d = r.get("diagnosis") or {}
            head = (f"{Path(s['track_dir']).name[:28]:28} {r['mode'][:6]:6} {r.get('similarity_overall', 0):.4f} / "
                    f"{r.get('similarity_section_aware', 0):.4f} {r.get('gate_verdict', '?'):4} "
                    f"RED {d.get('red_count', '?')}")
            rows.append(head)
            rows += [f"      {f}" for f in (d.get("flags") or [])[:4]]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("vdir", type=Path)
    r.add_argument("--track", type=Path)
    r.add_argument("--json-out", type=Path)
    rr = sub.add_parser("record-refs")
    rr.add_argument("vdirs", type=Path, nargs="+")
    t = sub.add_parser("table")
    t.add_argument("summaries", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        d = diagnose(a.vdir, a.track)
        (a.json_out or a.vdir / "diagnosis.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        print(render_text(d))
    elif a.cmd == "record-refs":
        doc = record_refs(a.vdirs)
        print(json.dumps({m: v["sources"] for m, v in doc["modes"].items()}, indent=1, ensure_ascii=False))
    else:
        print(table(a.summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
