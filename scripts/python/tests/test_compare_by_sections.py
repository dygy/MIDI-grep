"""Smoke test for compare_by_sections.

Builds two synthetic 12s mono WAVs:
  - "original" : 4s of low sine (110Hz) | 4s of mid sine (440Hz) | 4s of high noise
  - "rendered" : 12s of pure 440Hz tone (a static loop, ignoring structure)

Checks:
  - by_section returns one entry per section
  - section 1 (low band) and section 3 (high band) score worse than
    section 2 (which matches)
  - issues mention the missing/extra bands
"""
import json
import os
import sys
import tempfile
import wave
import struct
import math

# Allow running directly: ensure scripts/python is on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def write_wav(path, samples, sr=22050):
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", max(-32767, min(32767, int(s * 32767))))
                          for s in samples)
        wf.writeframes(frames)


def sine(freq, duration, sr=22050, amp=0.4):
    n = int(duration * sr)
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def main():
    sr = 22050
    tmpdir = tempfile.mkdtemp(prefix="midigrep-sec-test-")
    orig = os.path.join(tmpdir, "orig.wav")
    rend = os.path.join(tmpdir, "rend.wav")

    # Original: 3 distinct 4s sections
    orig_samples = sine(80, 4, sr) + sine(440, 4, sr) + sine(4000, 4, sr)
    # Rendered: pure 440Hz throughout (matches section 2 only)
    rend_samples = sine(440, 12, sr)

    write_wav(orig, orig_samples, sr)
    write_wav(rend, rend_samples, sr)

    sections = [
        {"start": 0.0,  "end": 4.0,  "duration": 4.0, "energy": 0.3, "dominant_chord": "Am"},
        {"start": 4.0,  "end": 8.0,  "duration": 4.0, "energy": 0.6, "dominant_chord": "C"},
        {"start": 8.0,  "end": 12.0, "duration": 4.0, "energy": 0.9, "dominant_chord": "G"},
    ]

    from compare_audio import compare_by_sections
    results = compare_by_sections(orig, rend, sections, duration=12.0)

    print(json.dumps(results, indent=2))

    assert len(results) == 3, f"expected 3 sections, got {len(results)}"
    s1, s2, s3 = results

    # Section 2 (rendered matches) should score highest
    assert s2["similarity"] > s1["similarity"], \
        f"section 2 should beat section 1: {s2['similarity']:.2f} vs {s1['similarity']:.2f}"
    assert s2["similarity"] > s3["similarity"], \
        f"section 2 should beat section 3: {s2['similarity']:.2f} vs {s3['similarity']:.2f}"

    # Section 1 should report bass missing in rendered (bass too quiet)
    s1_issues = " ".join(s1.get("issues", []))
    assert ("bass" in s1_issues or "sub_bass" in s1_issues), \
        f"section 1 should flag bass issue, got: {s1_issues!r}"

    # Section 3 should report high band missing
    s3_issues = " ".join(s3.get("issues", []))
    assert "high" in s3_issues, \
        f"section 3 should flag high band issue, got: {s3_issues!r}"

    # Pass-through fields
    assert s1["dominant_chord"] == "Am"

    print("PASS: compare_by_sections discriminates structural mismatch")


if __name__ == "__main__":
    main()
