"""Render stems straight from the renderer: one voice group sounding per render (no Demucs).

We generate every voice ourselves, so ``render_{bass,drums,melodic}`` do not need a separation model: the same
``.strudel`` is rendered with every top-level ``$:`` channel of the other groups commented out - exactly the
"comment a line to MUTE it" the DJ flow header documents. A channel's group comes from the generator's own role
marker (``generate_dynamic_strudel.py`` ``channels.append((LABEL, comment, code[, marker]))`` -> the trailing
``// [marker — ]LABEL — comment``): ``BASS`` -> bass, ``DRUMS`` -> drums, every other pitched label (LEAD, CHORDS,
...) -> melodic. A channel with no such label is an error, never a guess from sound names.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel

GROUPS: tuple[str, ...] = ("melodic", "drums", "bass")
LABEL_GROUP: dict[str, str] = {"BASS": "bass", "DRUMS": "drums"}   # every other pitched label is melodic
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CHANNEL_RE = re.compile(r"^\$:")


class StemSplitError(ValueError):
    """The strudel file has a channel whose voice group cannot be read from the generator's marker."""


class Channel(BaseModel):
    first: int          # 0-based first line of the `$:` block
    last: int           # 0-based last line (inclusive)
    label: str
    group: str


def _scan(line: str, depth: int, in_str: str | None) -> tuple[int, str | None, str]:
    """Paren depth after ``line`` (strings/``//`` comments ignored) and the line's trailing ``//`` comment."""
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        elif ch in "\"'`":
            in_str = ch
        elif line.startswith("//", i):
            return depth, in_str, line[i + 2:]
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    return depth, in_str, ""


def _label(comment: str) -> str | None:
    """``texture — LEAD — melody...`` / ``BASS — low end`` -> the first all-caps token between the em-dashes."""
    for part in (p.strip() for p in comment.split("—")):
        if _LABEL_RE.match(part):
            return part
    return None


def find_channels(lines: list[str]) -> list[Channel]:
    """Every top-level ``$:`` block with its role label and voice group."""
    chans: list[Channel] = []
    i = 0
    while i < len(lines):
        if not _CHANNEL_RE.match(lines[i]):
            i += 1
            continue
        first, depth, in_str, comment, j = i, 0, None, "", i
        while True:
            depth, in_str, c = _scan(lines[j], depth, in_str)
            comment = c or comment
            nxt = next((k for k in range(j + 1, len(lines)) if lines[k].strip()), None)
            if depth <= 0 and in_str is None and not (nxt is not None and lines[nxt].lstrip().startswith(".")):
                break
            j += 1
            if j >= len(lines):
                raise StemSplitError(f"unterminated `$:` block starting at line {first + 1}")
        label = _label(comment)
        if label is None:
            raise StemSplitError(f"channel at line {first + 1} has no role label comment "
                                 f"(`// LABEL — ...`), cannot assign a voice group: {lines[first][:60]!r}")
        chans.append(Channel(first=first, last=j, label=label, group=LABEL_GROUP.get(label, "melodic")))
        i = j + 1
    return chans


def _silence(last_line: str) -> str:
    """Append ``.gain(0)`` to a channel's expression, ahead of its trailing ``// LABEL`` comment.

    Gain is last-wins in Strudel, so the channel keeps its orbits (a duck target such as the bass' orbit is
    defined by the BASS channel and ducked by the DRUMS one; commenting a channel out would make the engine fail
    with "duck target orbit N does not exist") and its haps still fire (ducking), but it is silent."""
    depth, in_str, cut = 0, None, len(last_line)
    i = 0
    while i < len(last_line):          # position of the trailing `//` comment, outside strings
        ch = last_line[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        elif ch in "\"'`":
            in_str = ch
        elif last_line.startswith("//", i):
            cut = i
            break
        i += 1
    code, comment = last_line[:cut].rstrip(), last_line[cut:]
    return code + ".gain(0)" + ("  " + comment if comment else "")


def solo_code(code: str, group: str) -> str | None:
    """``code`` with every channel outside ``group`` commented out; None when ``group`` has no channel."""
    if group not in GROUPS:
        raise ValueError(f"unknown voice group {group!r}")
    lines = code.split("\n")
    chans = find_channels(lines)
    if not any(c.group == group for c in chans):
        return None
    for c in chans:
        if c.group != group:
            lines[c.last] = _silence(lines[c.last])
    return "\n".join(lines)


def write_silence(path: Path, frames: int, samplerate: int, channels: int) -> None:
    sf.write(str(path), np.zeros((frames, channels), dtype="float32"), samplerate, subtype="PCM_16")


class SumCheck(BaseModel):
    correlation: float
    residual_db: float      # 10*log10(energy(render - sum) / energy(render)); more negative = closer
    gain_ratio: float       # rms(sum) / rms(render)


def sum_check(render: Path, stems: list[Path]) -> SumCheck:
    """How closely the solo stems add back to ``render.wav`` (sample-aligned, trimmed to the shortest)."""
    r, _ = sf.read(str(render), dtype="float64", always_2d=True)
    parts = [sf.read(str(p), dtype="float64", always_2d=True)[0] for p in stems]
    n = min([len(r), *(len(p) for p in parts)])
    ch = r.shape[1]
    tot = sum(p[:n, :ch] if p.shape[1] >= ch else np.repeat(p[:n, :1], ch, axis=1) for p in parts)
    r = r[:n]
    er = float(np.sum(r ** 2)) or 1e-20
    res = float(np.sum((r - tot) ** 2)) or 1e-20
    a, b = r.ravel() - r.mean(), tot.ravel() - tot.mean()
    denom = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))) or 1e-20
    return SumCheck(correlation=float(np.sum(a * b) / denom), residual_db=10 * float(np.log10(res / er)),
                    gain_ratio=float(np.sqrt(np.sum(tot ** 2) / er)))
