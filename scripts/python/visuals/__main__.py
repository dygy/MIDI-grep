"""Build ``vNNN/visuals.strudel``: the piece plus a Hydra block (see ``visuals/__init__.py``).

    cd scripts/python && .venv/bin/python -m visuals "<track cache dir>" [--version v063]

Writes into the version dir: ``visual.jpg`` (the painted scene), ``visuals.strudel`` (paste into the Strudel REPL),
``visuals.json`` (vision, prompts, chain, its source, validation, wiring) and, unless ``--no-check``, the headless
check's frames + verdict (``visuals_check/``, ``scripts/node/dist/check-hydra.js``).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import hydra_code as HC
from . import scene_image, vision as VI

REPO = Path(__file__).resolve().parents[3]
CHECKER = REPO / "scripts" / "node" / "dist" / "check-hydra.js"


def latest_version(track_dir: Path) -> Path:
    vs = sorted((p for p in track_dir.glob("v[0-9][0-9][0-9]") if (p / "output.strudel").exists()), key=lambda p: p.name)
    if not vs:
        sys.exit(f"no vNNN/output.strudel under {track_dir}")
    return vs[-1]


def checker_args(vis: Path, image: Path, out_dir: Path, wiring: dict) -> list[str]:
    """check-hydra.js arguments: the scene reference carries the chain's source zoom (``with_edge_margin``)."""
    return [str(vis), "--image", str(image), "-o", str(out_dir),
            "--wired", ",".join(k for k, v in wiring.items() if v), "--scene-zoom", str(HC.EDGE_ZOOM)]


def hydra_block(controls: str, image_src: str, plan: dict, wiring: dict) -> str:
    chain = HC.with_edge_margin(" ".join(plan["chain"].split()))
    pretty = re.sub(r"\)\.(?=[a-zA-Z])", ")\n  .", chain)
    wired = "\n".join(f"// │   {k:<10} ← {v or 'not in this piece (always 0)'}" for k, v in wiring.items())
    img = plan["image"]
    return f"""

// ┌── VISUALS — Hydra over a painted scene (scripts/python/visuals) ─────────────
// │ scene : {plan['image_prompt']}
// │         painted by {img['model']} (seed {img['seed']}, {img['steps']} step(s), {img['size'][0]}x{img['size'][1]}); chain by {plan['chain_source']}
// │ Every control is read LIVE from this piece's patterns at Strudel's clock: edit the bar arrays and the picture
// │ follows. Edit the chain at the bottom like any Hydra code (hydra.ojack.xyz/docs).
{wired}
// └──────────────────────────────────────────────────────────────────────────────
await initHydra({{ detectAudio: false }})
{HC.SCENE_SOURCE}
s0.initImage('{image_src}')
{controls}

{pretty}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="visuals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir", type=Path, help=".cache/stems/<track>/ (holds metadata.json, instrumental.wav, vNNN/)")
    ap.add_argument("--version", help="vNNN to decorate (default: the latest with output.strudel)")
    ap.add_argument("--ollama-model", default="mistral-small", help="self-hosted LLM for the scene prompt + chain")
    ap.add_argument("--no-llm", action="store_true", help="skip Ollama: CLAP-derived prompt + the fallback chain")
    ap.add_argument("--image-model", default="stabilityai/sd-turbo",
                    help="diffusers text-to-image model (stabilityai/sdxl-turbo paints more detail, ~7 GB)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", default="640x384", help="WxH of the scene, multiples of 64")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--image-url", help="reference this hosted image instead of embedding visual.jpg as a data URL")
    ap.add_argument("--no-clap", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the headless Hydra check")
    args = ap.parse_args(argv)

    track = args.track_dir.resolve()
    vdir = track / args.version if args.version else latest_version(track)
    code = (vdir / "output.strudel").read_text()
    print(f"[visuals] {vdir}")

    vision = VI.build_vision(track, code, use_clap=not args.no_clap)
    print(f"[visuals] vision: {json.dumps({k: v for k, v in vision.items() if k != 'sections'}, ensure_ascii=False)[:400]}")

    controls, wiring = HC.controls_js(code)
    plan = {"image_prompt": HC.fallback_image_prompt(vision), "negative_prompt": "cartoon, illustration, text, watermark",
            "chain": HC.fallback_chain(), "chain_source": "fallback (no LLM)", "llm": None}
    if not args.no_llm:
        try:
            ans = HC.ask_ollama(vision, wiring, args.ollama_model, args.seed)
            plan["llm"] = {"model": args.ollama_model, "answer": ans}
            if isinstance(ans.get("image_prompt"), str) and HC.scene_prompt(ans["image_prompt"]):
                plan["image_prompt"] = HC.scene_prompt(ans["image_prompt"])
                plan["negative_prompt"] = str(ans.get("negative_prompt") or plan["negative_prompt"])
            errs = HC.validate_chain(str(ans.get("chain") or ""))
            if errs:  # one repair round: the errors go back to the LLM with its own answer
                print(f"[visuals] LLM chain rejected, asking for a fix: {errs}")
                plan["llm"]["first_answer"], plan["llm"]["first_errors"] = ans, errs
                fixed = HC.ask_ollama(vision, wiring, args.ollama_model, args.seed, repair=(ans, errs))
                if isinstance(fixed.get("chain"), str):
                    ans = {**ans, "chain": fixed["chain"]}
                    plan["llm"]["answer"] = ans
                    errs = HC.validate_chain(ans["chain"])
            plan["llm"]["chain_errors"] = errs
            if errs:
                plan["chain_source"] = f"fallback ({args.ollama_model} chain rejected: {errs[0]})"
                print(f"[visuals] LLM chain rejected: {errs}")
            elif HC.copies_example(ans["chain"]):
                # VAGABUNDO v073, Dumb Blond Bitches v020 (2026-11-01): the answer was the prompt's example verbatim
                plan["chain_source"] = f"fallback ({args.ollama_model} copied the prompt's example chain)"
                print(f"[visuals] LLM chain is the prompt's example verbatim")
            else:
                plan["chain"], plan["chain_source"] = ans["chain"], f"{args.ollama_model} (validated)"
        except Exception as exc:
            plan["chain_source"] = f"fallback (Ollama failed: {type(exc).__name__})"
            print(f"[visuals] Ollama unavailable ({type(exc).__name__}: {str(exc)[:160]}); fallback prompt + chain")
    print(f"[visuals] scene: {plan['image_prompt']}")
    print(f"[visuals] chain: {plan['chain_source']}")

    w, h = (int(x) for x in args.size.lower().split("x"))
    img_path = vdir / "visual.jpg"
    plan["image"] = scene_image.paint(plan["image_prompt"], img_path, args.image_model, args.seed, w, h, args.steps,
                                      plan["negative_prompt"])
    image_src = args.image_url or "data:image/jpeg;base64," + base64.b64encode(img_path.read_bytes()).decode()

    out = vdir / "visuals.strudel"

    def write_and_check() -> dict | None:
        out.write_text(code.rstrip() + "\n" + hydra_block(controls, image_src, plan, wiring))
        if args.no_check:
            return None
        if not CHECKER.exists():
            sys.exit(f"{CHECKER} missing: cd scripts/node && npm run build (or pass --no-check)")
        r = subprocess.run(["node", str(CHECKER), *checker_args(out, img_path, vdir / "visuals_check", wiring)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr[-2000:])
        check_json = vdir / "visuals_check" / "check.json"
        return json.loads(check_json.read_text()) if check_json.exists() else {"verdict": "FAIL", "failures": [r.stderr[-500:]]}

    check = write_and_check()
    rejected = None
    if check and check["verdict"] != "PASS" and plan["chain"] != HC.fallback_chain():
        # the LLM chain passed the whitelist but not the run (wrong argument types, no motion, scene washed out)
        rejected = {"chain": plan["chain"], "check": check}
        print(f"[visuals] {plan['chain_source']} failed the check ({'; '.join(check.get('failures', []))[:300]}); "
              "re-checking with the fallback chain")
        plan["chain"], plan["chain_source"] = HC.fallback_chain(), f"fallback ({args.ollama_model} chain failed the check)"
        check = write_and_check()
    record = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "piece": "output.strudel",
              "vision": vision, **{k: plan[k] for k in ("image_prompt", "negative_prompt", "chain", "chain_source",
                                                        "llm", "image")},
              "wiring": wiring, "visuals": out.name, "check": check, "rejected_chain": rejected}
    (vdir / "visuals.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"[visuals] wrote {out}")
    verdict = (record.get("check") or {}).get("verdict", "UNCHECKED")
    print(f"[visuals] check: {verdict}")
    return 0 if verdict in ("PASS", "UNCHECKED") else 1


if __name__ == "__main__":
    sys.exit(main())
