"""Edge cases of the drum one-shot cut and of the drum grid phase (review of improve/generalize).

1. `decay_end` looks for a hit's peak before the next onset of ANY class. A foreign hit detected while the hit is still
   rising (an "hh" detected on a kick's own attack) put that limit on the rising edge, so the "peak" was a point of the
   rise and the decay threshold, hence the one-shot's length, depended on where the foreign hit happened to fall.
2. The drum stack is shifted by `.late(phase)` over a cyclic `cat(...)`. A phase < 0 sent bar 0's step 0 to negative
   time (lost) and replayed it just before the end of the piece (the next repetition's step 0).
3. The phase `build_detected_drums` slotted with and the phase `.late()` undoes were computed separately.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import drum_oneshots as O  # noqa: E402
import generate_hybrid_strudel as H  # noqa: E402

SR = 44_100


# ---- 1. peak search vs a foreign hit detected during the rise --------------------------------------------------

def _swelling_kick(n=0.8):
    """An 808 whose body swells for ~60 ms (a slow attack) before it decays."""
    t = np.arange(int(n * SR)) / SR
    return np.sin(2 * np.pi * (45 + 60 * np.exp(-t * 20)) * t) * (1 - np.exp(-t / 0.03)) * np.exp(-t * 7)


def _kick_length_ms(hh_delay: float | None) -> float:
    y = np.zeros(int(23 * SR))
    hits = []
    for i in range(14):
        t0 = i * 1.5
        a = int(t0 * SR)
        k = _swelling_kick()
        y[a: a + len(k)] += k
        hits.append({"time": t0 + 0.012, "type": "bd"})
        if hh_delay is not None:
            hits.append({"time": t0 + 0.012 + hh_delay, "type": "hh"})
    x = O.cut_class(y, SR, sorted(hits, key=lambda h: h["time"]), "bd", O.stem_noise_floor(y, SR))
    return len(x) / SR * 1000


def test_kick_length_does_not_depend_on_where_a_foreign_hit_falls_in_its_rise():
    alone = _kick_length_ms(None)
    # an "hh" detected 10 / 20 / 40 ms after the kick, while its body is still rising: the kick is the same kick
    for delay in (0.010, 0.020, 0.040):
        ms = _kick_length_ms(delay)
        assert abs(ms - alone) < 0.05 * alone, f"hh at +{delay * 1000:.0f} ms: {ms:.0f} ms vs {alone:.0f} ms alone"


# ---- 2/3. the .late() shift and where the phase comes from -----------------------------------------------------

BPM = 120.0
BAR = 2.0
STEP = BAR / 16
NBARS = 4


def _heard(voices, late_cycles: float, nbars: int = NBARS, quantize: int = 16) -> list[float]:
    """Seconds at which the generated stack sounds: `stack(cat(bar strings...)).late(x)`, played from 0 to *nbars*.

    Models Strudel's semantics: the `cat` repeats every *nbars* cycles and `.late(x)` moves every event x cycles on, so
    the repetitions on either side of the piece are heard wherever they land inside [0, nbars)."""
    out: list[float] = []
    for v in voices:
        for b, bar in enumerate(re.findall(r's\("([^"]*)"\)', v)):
            for s, tok in enumerate(bar.split()):
                if tok == "~":
                    continue
                for rep in (-1, 0, 1):
                    t = rep * nbars + b + s / quantize + late_cycles
                    if 0 <= t < nbars:
                        out.append(t * BAR)
    return sorted(out)


def _write(tmp_path: Path, times: list[float]) -> Path:
    j = tmp_path / "d.json"
    j.write_text(json.dumps({"hits": [{"time": t, "type": "bd"} for t in times]}))
    return j


def _unmatched(heard: list[float], hits: list[float]) -> tuple[list[float], list[float]]:
    """(heard sounds with no stem hit within a step, stem hits with no heard sound within a step)."""
    h = np.array(heard)
    t = np.array(hits)
    phantom = [float(x) for x in h if np.min(np.abs(t - x)) > STEP]
    lost = [float(x) for x in t if np.min(np.abs(h - x)) > STEP]
    return phantom, lost


def test_early_grid_neither_loses_the_first_hit_nor_replays_it_at_the_end(tmp_path):
    # the grid sits 40 ms BEFORE the step lines; a hit at 10 ms (nearest grid point: -40 ms, i.e. step 0), then hits
    # from the third step on, none within the last 0.6 s
    hits = [0.010] + [k * STEP - 0.040 for k in range(3, 60)]
    j = _write(tmp_path, hits)
    voices = H.build_detected_drums(j, BPM, NBARS)
    phantom, lost = _unmatched(_heard(voices, H.drums_late(j, BPM)), hits)
    assert phantom == [], f"sounds with no hit behind them (a replay of step 0 at the end): {phantom}"
    assert lost == [], f"hits that never sound: {lost}"


def test_late_grid_has_no_hit_at_the_start_that_the_stem_does_not_have(tmp_path):
    # grid 40 ms AFTER the step lines, a hit on the last step of the last bar and none on step 0
    hits = [k * STEP + 0.040 for k in range(2, 64)]
    j = _write(tmp_path, hits)
    voices = H.build_detected_drums(j, BPM, NBARS)
    phantom, lost = _unmatched(_heard(voices, H.drums_late(j, BPM)), hits)
    assert phantom == [] and lost == [], (phantom, lost)


def test_the_voices_carry_the_phase_they_were_slotted_with(tmp_path):
    # hits on an 8-step bar grid, 100 ms late: the phase measured for 8 steps is not the one measured for 16
    step8 = BAR / 8
    hits = [(k * step8 + 0.100) for k in range(1, 30)]
    j = _write(tmp_path, hits)
    voices = H.build_detected_drums(j, BPM, NBARS, quantize=8)
    heard = _heard(voices, voices.late, quantize=8)
    phantom, lost = _unmatched(heard, hits)
    assert phantom == [] and lost == []
    assert np.allclose(sorted(heard), sorted(hits), atol=1e-3)
