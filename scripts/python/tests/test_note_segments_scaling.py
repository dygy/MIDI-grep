# @layer: unit
# @spec: 003-editable-strudel-generation
# @regression
"""note_segments clips each note at the next LATER onset; the lookup must not be a linear scan per note."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
from pitched_grains import Note, note_segments  # noqa: E402


def _reference(notes: list[Note], total_s: float) -> list[tuple[Note, float, float]]:
    """The original linear-scan semantics, kept here as the oracle."""
    onsets = sorted({round(n.start, 4) for n in notes})
    out = []
    for n in notes:
        nxt = next((o for o in onsets if o > round(n.start, 4) + 1e-4), None)
        end = min(n.end, nxt if nxt is not None else n.end, total_s)
        if end > n.start:
            out.append((n, float(n.start), float(end)))
    return out


def test_same_result_as_the_linear_scan_incl_chords_and_ties() -> None:
    rng = np.random.default_rng(7)
    starts = np.round(rng.uniform(0, 30, 400), 3)
    starts = np.concatenate([starts, starts[:60], starts[:20] + 0.00004])   # chords + sub-tolerance neighbours
    notes = [Note(float(s), float(s + rng.uniform(0.05, 2.0)), int(rng.integers(30, 80))) for s in starts]
    assert note_segments(notes, 28.0) == _reference(notes, 28.0)


def test_chord_members_do_not_clip_each_other() -> None:
    chord = [Note(1.0, 2.0, 40), Note(1.0, 1.5, 47), Note(1.5, 3.0, 50)]
    segs = {n.pitch: (s, e) for n, s, e in note_segments(chord, 10.0)}
    assert segs[40] == (1.0, 1.5) and segs[47] == (1.0, 1.5) and segs[50] == (1.5, 3.0)


def test_scales_to_a_long_polyphonic_stem() -> None:
    n = 20000
    notes = [Note(i * 0.01, i * 0.01 + 0.5, 40 + i % 30) for i in range(n)]
    t0 = time.perf_counter()
    segs = note_segments(notes, n * 0.01 + 1)
    elapsed = time.perf_counter() - t0
    assert len(segs) == n
    assert elapsed < 2.0, f"{elapsed:.1f} s for {n} notes - the onset scan is quadratic"
