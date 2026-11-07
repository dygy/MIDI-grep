"""Genre of a track from its PUBLISHED METADATA (YouTube title, channel, tags, description), read by the local LLM.

Why: the audio evidence alone (CLAP zero-shot, kNN against the labelled tracks) separates genres only weakly, and the
label registry used to grow only when the owner added a track by hand. A track's metadata is a second, INDEPENDENT
source: uploaders tag their genre ("funk carioca", "automotivo", "phonk") and name the artists. Stage 1b
(``pipeline_helpers.py stamp``) labels a track automatically when this verdict and the audio verdict agree
(``eval/datasets/genre_labels_auto.yaml``); neither source decides alone.

The answer is restricted to the genres the pipeline knows (``detect_genre_dl.GENRE_DEFINITIONS``) or ``unknown``, and
it must QUOTE its evidence: every quoted piece has to occur verbatim in the metadata, and at least one of them has to
NAME the genre (``names_genre``: "funk carioca" names brazilian_funk; "Mc Erikah" or "BibiBabydoll" names an artist),
else the verdict is ``unknown``. A model that names a genre without anything in the text to point at, or only from what
it believes about an artist, is answering from its own prior, not from the metadata.

    genre_text.py <track_dir> -o <track>/.pipeline/genre_text.json [--model mistral-small]

Writes ``<track>/.pipeline/youtube_info.json`` (the metadata fields used) and the verdict JSON; a verdict for the same
metadata, model and prompt version is reused. Ollama down / yt-dlp failing gives ``status: unavailable`` (exit 0): the
stamp then simply has no text evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import unicodedata
import sys
from pathlib import Path

PROMPT_VERSION = 2          # also the automatic-label rule version (genre_knn: an auto label of another version is void)
DEFAULT_MODEL = "mistral-small"
INFO_FIELDS = ("title", "channel", "uploader", "artist", "track", "album", "tags", "categories", "description")
DESCRIPTION_CHARS = 1500     # the head of the description carries the credits/tags; the rest is links and boilerplate
MIN_EVIDENCE_CHARS = 3       # a quoted piece shorter than this ("a", "mc") points at nothing in particular
UNKNOWN = "unknown"


# --- metadata -----------------------------------------------------------------------------------------------------

def _ytdlp() -> str:
    venv = Path(__file__).resolve().parent / ".venv" / "bin" / "yt-dlp"
    return str(venv) if venv.exists() else "yt-dlp"


def _cookie_args() -> list[str]:
    """Same cookie env as the Go downloader (internal/audio/youtube.go ytDlpAuthArgs)."""
    out: list[str] = []
    if os.environ.get("MIDIGREP_YTDLP_COOKIES_FROM_BROWSER"):
        out += ["--cookies-from-browser", os.environ["MIDIGREP_YTDLP_COOKIES_FROM_BROWSER"]]
    if os.environ.get("MIDIGREP_YTDLP_COOKIES"):
        out += ["--cookies", os.environ["MIDIGREP_YTDLP_COOKIES"]]
    return out


def trim_info(raw: dict) -> dict:
    """The fields the classifier reads, description cut to its head; empty fields dropped."""
    out = {}
    for k in INFO_FIELDS:
        v = raw.get(k)
        if k == "description" and v:
            v = v[:DESCRIPTION_CHARS]
        if v not in (None, "", []):
            out[k] = v
    return out


def track_info(track_dir: Path, fetch=None) -> dict:
    """``youtube_info.json`` for the track: cached, else fetched with yt-dlp from metadata.json's url / video_id, else
    (a local file, or yt-dlp failing) just the title the cache recorded. ``fetch(url) -> raw dict`` is injectable."""
    track_dir = Path(track_dir)
    cached = track_dir / ".pipeline" / "youtube_info.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    mp = track_dir / "metadata.json"
    meta = json.loads(mp.read_text()) if mp.is_file() else {}
    url = meta.get("url") or (f"https://youtu.be/{meta['video_id']}" if meta.get("video_id") else None)
    info: dict = {}
    if url:
        try:
            info = trim_info((fetch or _fetch_ytdlp)(url))
        except Exception as exc:   # network / bot wall: the title alone is still evidence
            print(f"genre_text: metadata fetch failed for {url}: {exc}", file=sys.stderr)
    if not info:
        info = {"title": meta.get("title") or track_dir.name}
        return info                 # not cached: a later run may reach YouTube
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return info


def _fetch_ytdlp(url: str) -> dict:
    r = subprocess.run([_ytdlp(), "-J", "--skip-download", "--no-playlist", *_cookie_args(), url],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"yt-dlp exit {r.returncode}")
    return json.loads(r.stdout)


def info_text(info: dict) -> str:
    lines = []
    for k in INFO_FIELDS:
        v = info.get(k)
        if v in (None, "", []):
            continue
        lines.append(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return "\n".join(lines)


# --- the classifier -----------------------------------------------------------------------------------------------

def genre_choices() -> dict[str, str]:
    """{genre key: one-line description} - the detector's genre set, so text and audio answer in the same keys."""
    from detect_genre_dl import GENRE_DEFINITIONS
    return {g: d["descriptions"][0] for g, d in GENRE_DEFINITIONS.items()}


def _name_parts(text: str) -> list[str]:
    return [w for w in re.split(r"[^0-9a-z]+", text.casefold()) if w]


def genre_names(genre: str, genres=None) -> set[str]:
    """The words that NAME ``genre``: the parts of its key and of its palette alias (``sound_selector.GENRE_ALIASES``),
    minus every word another genre's name also has ("brazilian" names both funk and phonk, so it names neither)."""
    from detect_genre_dl import GENRE_DEFINITIONS
    from sound_selector import GENRE_ALIASES
    genres = list(GENRE_DEFINITIONS) if genres is None else list(genres)

    def own(g: str) -> set[str]:
        return set(_name_parts(g)) | set(_name_parts(GENRE_ALIASES.get(g, "")))
    shared = set().union(*(own(g) for g in genres if g != genre)) if len(genres) > 1 else set()
    return own(genre) - shared


def names_genre(quote: str, genre: str) -> bool:
    """True when ``quote`` names ``genre``: one of its words (accents folded; two adjacent words also joined, so "Lo-Fi"
    reads as "lofi") STARTS with one of ``genre_names`` ("funkeiro" names funk; "shockwave" does not name retro wave)."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", quote) if not unicodedata.combining(c))
    words = _name_parts(folded)
    cands = words + [a + b for a, b in zip(words, words[1:])]
    names = genre_names(genre)
    return any(c.startswith(n) for c in cands for n in names)


def build_prompt(info: dict, choices: dict[str, str]) -> str:
    opts = "\n".join(f'- "{g}": {desc} (names: {", ".join(sorted(genre_names(g, choices)))})'
                     for g, desc in choices.items())
    return f"""You classify the music GENRE of one track from its YouTube metadata.

Allowed answers (pick exactly one key):
{opts}
- "{UNKNOWN}": the metadata does not let you tell which of these genres it is, or it is none of them.

Rules:
- Decide only from the metadata below. An artist, label or channel name alone is NOT enough, whatever you know
  about them: the metadata itself has to name the genre (a tag, a title word, a description word containing one of
  the genre's names above).
- "evidence" must list the words from the metadata that support your answer, each copied EXACTLY as it appears.
  At least one of them must contain a name of the genre. No paraphrase, no translation.
- If the metadata names none of the allowed genres, answer "{UNKNOWN}" with empty evidence.

Metadata:
{info_text(info)}

Return only JSON: {{"genre": "<one allowed key>", "evidence": ["<exact quote>", ...], "reason": "<one sentence>"}}"""


def ground(answer: dict, info: dict, choices: dict[str, str]) -> dict:
    """Validate the model's answer: a known key, and evidence that occurs verbatim (case-insensitive) in the metadata.
    Anything else becomes ``unknown`` with the reason recorded."""
    genre = str(answer.get("genre") or "").strip()
    quotes = [str(q).strip() for q in (answer.get("evidence") or []) if isinstance(q, (str, int, float))]
    hay = info_text(info).casefold()
    grounded = [q for q in quotes if len(q) >= MIN_EVIDENCE_CHARS and q.casefold() in hay]
    out = {"genre": genre, "evidence": grounded, "reason": str(answer.get("reason") or "")[:300]}
    dropped = [q for q in quotes if q not in grounded]
    if dropped:
        out["evidence_rejected"] = dropped
    if genre == UNKNOWN:
        out["evidence"] = []
    elif genre not in choices:
        out.update(genre=UNKNOWN, why=f"answer {genre!r} is not an allowed genre")
    elif not grounded:
        out.update(genre=UNKNOWN, why="no quoted evidence occurs in the metadata")
    elif not any(names_genre(q, genre) for q in grounded):
        out.update(genre=UNKNOWN, why=f"no quoted evidence names the genre ({', '.join(sorted(genre_names(genre)))}); "
                                      "artist / label names alone are not evidence")
    return out


def ask_ollama(prompt: str, model: str, url: str | None = None) -> dict:
    import requests
    url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    r = requests.post(f"{url}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "keep_alive": 0,        # unload at once: the renderer / Demucs need the memory next (24 GB machine)
        "options": {"temperature": 0, "seed": 0, "num_predict": 512, "num_ctx": 4096},
    }, timeout=600)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def classify(track_dir: Path, out_path: Path | None = None, model: str = DEFAULT_MODEL,
             fetch=None, ask=None) -> dict:
    """The text verdict for the track (cached in ``out_path`` per metadata + model + prompt version)."""
    info = track_info(track_dir, fetch=fetch)
    choices = genre_choices()
    prompt = build_prompt(info, choices)
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, model, prompt], ensure_ascii=False).encode()).hexdigest()[:16]
    if out_path and Path(out_path).is_file():
        prev = json.loads(Path(out_path).read_text())
        if prev.get("input_sha") == key and prev.get("status") == "ok":
            return prev
    res = {"status": "ok", "model": model, "prompt_version": PROMPT_VERSION, "input_sha": key,
           "sources": sorted(info)}
    try:
        res.update(ground((ask or ask_ollama)(prompt, model), info, choices))
    except Exception as exc:   # Ollama down, model missing, unparsable answer: no text evidence this run
        res.update(status="unavailable", genre=None, evidence=[], why=f"{type(exc).__name__}: {exc}"[:300])
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--model", default=os.environ.get("MIDIGREP_GENRE_TEXT_MODEL", DEFAULT_MODEL))
    args = ap.parse_args(argv)
    import genre_knn as GK
    lab = GK.load_labels().get(Path(args.track_dir).name)
    if lab:   # a labelled track is decided by its label; no model load
        print(json.dumps({"status": "skipped", "why": f"labelled {lab['genre']} ({lab['source']})"}, ensure_ascii=False))
        return 0
    res = classify(args.track_dir, args.out, model=args.model)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
