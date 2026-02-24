#!/usr/bin/env python3
"""
Build training data for fine-tuning Ollama on Strudel code generation.

Crawls .cache/stems/ for strudel files, comparison.json, and metadata.json.
Produces JSONL suitable for Ollama/Unsloth fine-tuning.

Three example types:
1. Audio-to-Code: metadata + frequency bands → strudel code (similarity > 0.50)
2. Iteration Improvement: old code + comparison feedback → improved code (>2% gain)
3. Negative Examples: invalid sound names/methods → correction explanation

Usage:
    python build_training_data.py [--cache-dir .cache/stems] [--output training_data/combined.jsonl]
    python build_training_data.py --stats  # Show stats only, no output
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import sound validation sets
try:
    from ollama_agent import VALID_SOUNDS, VALID_DRUM_BANKS, INVALID_GM_PATTERNS
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from ollama_agent import VALID_SOUNDS, VALID_DRUM_BANKS, INVALID_GM_PATTERNS
    except ImportError:
        VALID_SOUNDS = set()
        VALID_DRUM_BANKS = set()
        INVALID_GM_PATTERNS = []


def load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_strudel(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text if len(text) >= 100 else None
    except Exception:
        return None


def extract_bands(comparison: Dict) -> Optional[Dict[str, float]]:
    """Extract frequency band data from comparison.json."""
    comp = comparison.get("comparison", {})
    bands = comp.get("band_differences", {})
    if not bands:
        return None
    return {
        "sub_bass": bands.get("sub_bass", 0),
        "bass": bands.get("bass", 0),
        "low_mid": bands.get("low_mid", 0),
        "mid": bands.get("mid", 0),
        "high_mid": bands.get("high_mid", 0),
        "high": bands.get("high", 0),
    }


def get_similarity(comparison: Dict) -> float:
    comp = comparison.get("comparison", {})
    return comp.get("overall_similarity", 0)


def build_audio_context(metadata: Dict, comparison: Optional[Dict] = None) -> str:
    """Build a concise audio context string from metadata."""
    parts = []
    bpm = metadata.get("bpm", 0)
    if bpm:
        parts.append(f"BPM: {bpm:.0f}")
    key = metadata.get("key", "")
    if key:
        parts.append(f"Key: {key}")
    genre = metadata.get("genre", "")
    if genre:
        parts.append(f"Genre: {genre}")
    style = metadata.get("style", "")
    if style:
        parts.append(f"Style: {style}")

    if comparison:
        bands = extract_bands(comparison)
        if bands:
            band_strs = [f"{k}: {v:.2f}" for k, v in bands.items() if abs(v) > 0.01]
            if band_strs:
                parts.append(f"Frequency bands: {', '.join(band_strs)}")

    return ". ".join(parts)


def build_comparison_feedback(comparison: Dict) -> str:
    """Build comparison feedback string for iteration improvement examples."""
    comp = comparison.get("comparison", {})
    sim = comp.get("overall_similarity", 0)
    lines = [f"Overall similarity: {sim*100:.1f}%"]

    bands = extract_bands(comparison)
    if bands:
        for name, diff in bands.items():
            if abs(diff) > 0.05:
                direction = "too loud" if diff > 0 else "too quiet"
                lines.append(f"  {name}: {diff*100:+.1f}% ({direction})")

    mfcc = comp.get("mfcc_similarity", 0)
    if mfcc:
        lines.append(f"Timbre (MFCC): {mfcc*100:.1f}%")
    brightness = comp.get("brightness_similarity", 0)
    if brightness:
        lines.append(f"Brightness: {brightness*100:.1f}%")

    insights = comparison.get("insights", [])
    if insights:
        lines.append("Insights: " + "; ".join(insights[:3]))

    return "\n".join(lines)


def make_audio_to_code_example(metadata: Dict, code: str, comparison: Optional[Dict]) -> Dict:
    """Type 1: audio analysis → strudel code."""
    context = build_audio_context(metadata, comparison)
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Generate Strudel code for this track. {context}"
            },
            {
                "role": "assistant",
                "content": f"```javascript\n{code}\n```"
            }
        ]
    }


def make_iteration_example(
    metadata: Dict,
    old_code: str,
    new_code: str,
    old_comparison: Dict,
    new_similarity: float
) -> Dict:
    """Type 2: old code + comparison feedback → improved code."""
    context = build_audio_context(metadata)
    feedback = build_comparison_feedback(old_comparison)
    old_sim = get_similarity(old_comparison)

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Improve this Strudel code. {context}\n\n"
                    f"Current code (similarity {old_sim*100:.1f}%):\n"
                    f"```javascript\n{old_code}\n```\n\n"
                    f"Comparison feedback:\n{feedback}\n\n"
                    f"Generate improved code."
                )
            },
            {
                "role": "assistant",
                "content": f"```javascript\n{new_code}\n```"
            }
        ]
    }


def find_invalid_sounds_in_code(code: str) -> List[Tuple[str, str]]:
    """Find invalid sounds in code and suggest corrections."""
    if not VALID_SOUNDS:
        return []

    corrections = []
    for m in re.finditer(r'\.sound\(["\']([^"\']+)["\']', code):
        for s in m.group(1).strip("<>").split():
            s = s.strip()
            if s and s not in VALID_SOUNDS:
                # Suggest correction
                if s.startswith("gm_pad_") and re.match(r'gm_pad_\d+_', s):
                    fix = re.sub(r'gm_pad_\d+_', 'gm_pad_', s)
                    corrections.append((s, fix))
                elif s.startswith("gm_fx_") and re.match(r'gm_fx_\d+_', s):
                    fix = re.sub(r'gm_fx_\d+_', 'gm_fx_', s)
                    corrections.append((s, fix))
                else:
                    corrections.append((s, "gm_piano"))
    return corrections


NEGATIVE_EXAMPLES = [
    {
        "invalid": '.sound("gm_pad_4_choir")',
        "valid": '.sound("gm_pad_choir")',
        "reason": "GM pad names have no numbers. Use gm_pad_choir, not gm_pad_4_choir.",
    },
    {
        "invalid": '.sound("gm_fx_1_rain")',
        "valid": '.sound("gm_fx_rain")',
        "reason": "GM FX names have no numbers. Use gm_fx_rain, not gm_fx_1_rain.",
    },
    {
        "invalid": '.sound("gm_electric_piano_1")',
        "valid": '.sound("gm_epiano1")',
        "reason": "Electric pianos use shortened names: gm_epiano1, gm_epiano2.",
    },
    {
        "invalid": '.sound("gm_acoustic_grand_piano")',
        "valid": '.sound("gm_piano")',
        "reason": "Acoustic piano is just gm_piano in Strudel.",
    },
    {
        "invalid": '.volume(0.5).peak(1000)',
        "valid": '.gain(0.5).hpf(1000)',
        "reason": ".volume() and .peak() don't exist in Strudel. Use .gain() and .hpf().",
    },
    {
        "invalid": '.eq(800).filter("lowpass")',
        "valid": '.lpf(800)',
        "reason": ".eq() and .filter() don't exist. Use .lpf() for low-pass and .hpf() for high-pass.",
    },
    {
        "invalid": '.bank("tr808")',
        "valid": '.bank("RolandTR808")',
        "reason": "Drum bank names use CamelCase with manufacturer: RolandTR808, not tr808.",
    },
]


def make_negative_example(neg: Dict) -> Dict:
    """Type 3: invalid code → correction."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Fix this Strudel code: {neg['invalid']}"
            },
            {
                "role": "assistant",
                "content": f"The correct code is: {neg['valid']}\n\n{neg['reason']}"
            }
        ]
    }


def crawl_cache(cache_dir: Path) -> Tuple[List[Dict], Dict]:
    """Crawl .cache/stems/ and collect training examples.

    Returns (examples, stats).
    """
    examples = []
    stats = {
        "tracks_scanned": 0,
        "versions_scanned": 0,
        "audio_to_code": 0,
        "iteration_improvement": 0,
        "negative": 0,
        "skipped_short": 0,
        "skipped_low_sim": 0,
        "skipped_no_metadata": 0,
    }

    seen_codes = set()

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}", file=sys.stderr)
        return examples, stats

    # Each subdirectory under cache_dir is a track
    for track_dir in sorted(cache_dir.iterdir()):
        if not track_dir.is_dir():
            continue

        stats["tracks_scanned"] += 1

        # Collect all versions for this track
        versions = []
        for vdir in sorted(track_dir.iterdir()):
            if not vdir.is_dir() or not re.match(r'^v\d+$', vdir.name):
                continue

            stats["versions_scanned"] += 1

            strudel_path = vdir / "output.strudel"
            metadata_path = vdir / "metadata.json"
            comparison_path = vdir / "comparison.json"

            code = load_strudel(strudel_path)
            metadata = load_json(metadata_path)
            comparison = load_json(comparison_path)

            if not code:
                stats["skipped_short"] += 1
                continue
            if not metadata:
                stats["skipped_no_metadata"] += 1
                continue

            version_num = int(vdir.name[1:])
            similarity = get_similarity(comparison) if comparison else 0

            versions.append({
                "version": version_num,
                "code": code,
                "metadata": metadata,
                "comparison": comparison,
                "similarity": similarity,
            })

        # Type 1: Audio-to-Code (best version per track with similarity > 0.50)
        if versions:
            best = max(versions, key=lambda v: v["similarity"])
            if best["similarity"] >= 0.50:
                code_hash = hash(best["code"])
                if code_hash not in seen_codes:
                    seen_codes.add(code_hash)
                    examples.append(make_audio_to_code_example(
                        best["metadata"], best["code"], best["comparison"]
                    ))
                    stats["audio_to_code"] += 1
            elif best["similarity"] > 0 and best["similarity"] < 0.30:
                stats["skipped_low_sim"] += 1

        # Type 2: Iteration Improvement (consecutive versions with >2% improvement)
        versions.sort(key=lambda v: v["version"])
        for i in range(len(versions) - 1):
            old = versions[i]
            new = versions[i + 1]

            if not old["comparison"]:
                continue

            improvement = new["similarity"] - old["similarity"]
            if improvement < 0.02:
                continue

            # Deduplicate
            pair_hash = hash((old["code"], new["code"]))
            if pair_hash in seen_codes:
                continue
            seen_codes.add(pair_hash)

            examples.append(make_iteration_example(
                old["metadata"],
                old["code"],
                new["code"],
                old["comparison"],
                new["similarity"],
            ))
            stats["iteration_improvement"] += 1

    # Type 3: Negative examples (always included)
    for neg in NEGATIVE_EXAMPLES:
        examples.append(make_negative_example(neg))
        stats["negative"] += 1

    return examples, stats


def main():
    parser = argparse.ArgumentParser(description="Build training data for Strudel fine-tuning")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).parent.parent.parent / ".cache" / "stems"),
        help="Path to .cache/stems/ directory"
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent.parent / "training_data" / "combined.jsonl"),
        help="Output JSONL path"
    )
    parser.add_argument("--stats", action="store_true", help="Show stats only, no output file")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"Scanning {cache_dir}...", file=sys.stderr)

    examples, stats = crawl_cache(cache_dir)

    print(f"\n--- Training Data Stats ---", file=sys.stderr)
    print(f"Tracks scanned:        {stats['tracks_scanned']}", file=sys.stderr)
    print(f"Versions scanned:      {stats['versions_scanned']}", file=sys.stderr)
    print(f"Audio-to-Code:         {stats['audio_to_code']}", file=sys.stderr)
    print(f"Iteration Improvement: {stats['iteration_improvement']}", file=sys.stderr)
    print(f"Negative Examples:     {stats['negative']}", file=sys.stderr)
    print(f"Total examples:        {len(examples)}", file=sys.stderr)
    print(f"Skipped (short code):  {stats['skipped_short']}", file=sys.stderr)
    print(f"Skipped (low sim):     {stats['skipped_low_sim']}", file=sys.stderr)
    print(f"Skipped (no metadata): {stats['skipped_no_metadata']}", file=sys.stderr)

    if args.stats:
        return

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(examples)} examples to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
