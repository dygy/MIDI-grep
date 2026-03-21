#!/usr/bin/env python3
"""
ClickHouse storage layer for MIDI-grep.

Contains all ClickHouse-related functions extracted from ai_improver.py:
schema migrations, query helpers, run/knowledge storage, and artist/genre presets.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ClickHouse connection
CLICKHOUSE_BIN = Path(__file__).parent.parent.parent / "bin" / "clickhouse"
CLICKHOUSE_DB = Path(__file__).parent.parent.parent / ".clickhouse" / "db"

# Import extract_parameters_from_code from strudel_params when available.
# Fall back to the local definition below if strudel_params.py does not exist yet.
try:
    from strudel_params import extract_parameters_from_code
    _STRUDEL_PARAMS_IMPORTED = True
except ImportError:
    _STRUDEL_PARAMS_IMPORTED = False


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

# Migrations: list of (version, description, queries)
# Each migration is atomic - all queries must succeed
MIGRATIONS = [
    (1, "Initial schema", [
        "CREATE DATABASE IF NOT EXISTS midi_grep",
        """CREATE TABLE midi_grep.schema_version (
            version UInt32,
            applied_at DateTime DEFAULT now(),
            description String
        ) ENGINE = MergeTree()
        ORDER BY version""",
        """CREATE TABLE midi_grep.runs (
            track_hash String,
            track_name String,
            version UInt32,
            created_at DateTime DEFAULT now(),
            bpm Float64,
            key String,
            key_type String,
            style String,
            genre String,
            strudel_code String,
            similarity_overall Float64,
            similarity_mfcc Float64,
            similarity_chroma Float64,
            similarity_frequency Float64,
            similarity_rhythm Float64,
            band_sub_bass Float64,
            band_bass Float64,
            band_low_mid Float64,
            band_mid Float64,
            band_high_mid Float64,
            band_high Float64,
            mix_params String,
            improved_from_version Nullable(UInt32),
            ai_suggestions String
        ) ENGINE = MergeTree()
        ORDER BY (track_hash, version)""",
        """CREATE TABLE midi_grep.knowledge (
            id UUID DEFAULT generateUUIDv4(),
            created_at DateTime DEFAULT now(),
            parameter_name String,
            parameter_old_value String,
            parameter_new_value String,
            similarity_improvement Float64,
            confidence Float64,
            genre String,
            bpm_range_low Float64,
            bpm_range_high Float64,
            key_type String,
            track_hash String
        ) ENGINE = MergeTree()
        ORDER BY (parameter_name, created_at)""",
    ]),
    # Add future migrations here:
    # (2, "Add new column", ["ALTER TABLE midi_grep.runs ADD COLUMN foo String"]),
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _run_clickhouse_query(query: str) -> subprocess.CompletedProcess:
    """Run a ClickHouse query and return result."""
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _get_schema_version() -> int:
    """Get current schema version from DB, or 0 if not initialized."""
    result = _run_clickhouse_query(
        "SELECT version FROM midi_grep.schema_version ORDER BY version DESC LIMIT 1"
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_clickhouse():
    """Initialize ClickHouse with migrations. Fails fast if any migration fails."""
    CLICKHOUSE_DB.mkdir(parents=True, exist_ok=True)

    # Ensure database exists first
    result = _run_clickhouse_query("CREATE DATABASE IF NOT EXISTS midi_grep")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create database: {result.stderr}")

    current_version = _get_schema_version()

    # Apply pending migrations
    for version, description, queries in MIGRATIONS:
        if version <= current_version:
            continue  # Already applied

        print(f"  [ClickHouse] Applying migration {version}: {description}")

        # Run all queries for this migration
        for query in queries:
            result = _run_clickhouse_query(query)
            if result.returncode != 0:
                # Check if it's just "already exists" for CREATE statements
                if "already exists" in result.stderr.lower():
                    continue
                raise RuntimeError(
                    f"Migration {version} failed on query:\n{query}\n\nError: {result.stderr}"
                )

        # Record successful migration
        record_query = f"INSERT INTO midi_grep.schema_version (version, description) VALUES ({version}, '{description}')"
        result = _run_clickhouse_query(record_query)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to record migration {version}: {result.stderr}")

        print(f"  [ClickHouse] Migration {version} applied successfully", file=sys.stderr)

    final_version = _get_schema_version()
    if final_version > 0:
        print(f"  [ClickHouse] Schema at version {final_version}", file=sys.stderr)


_clickhouse_initialized = False
HAS_CLICKHOUSE = True


def _ensure_clickhouse() -> bool:
    """Lazy ClickHouse initialization on first use."""
    global _clickhouse_initialized, HAS_CLICKHOUSE
    if _clickhouse_initialized:
        return HAS_CLICKHOUSE
    _clickhouse_initialized = True
    try:
        init_clickhouse()
        HAS_CLICKHOUSE = True
    except Exception as e:
        print(f"ClickHouse initialization failed (non-fatal): {e}", file=sys.stderr)
        HAS_CLICKHOUSE = False
    return HAS_CLICKHOUSE


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def sanitize_sql_value(value: str) -> str:
    """Escape a string value for safe use in ClickHouse SQL."""
    return str(value).replace("\\", "\\\\").replace("'", "''")


def clickhouse_query(query: str, format: str = "JSONEachRow") -> List[Dict]:
    """Execute a ClickHouse query and return results."""
    if not _ensure_clickhouse():
        return []
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query,
        "--format", format
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ClickHouse error: {result.stderr}", file=sys.stderr)
        return []

    if not result.stdout.strip():
        return []

    rows = []
    for line in result.stdout.strip().split('\n'):
        if line:
            rows.append(json.loads(line))
    return rows


def clickhouse_insert(table: str, data: Dict[str, Any]):
    """Insert a row into ClickHouse. Raises RuntimeError on failure."""
    if not _ensure_clickhouse():
        return
    cols = list(data.keys())
    vals = []
    for k in cols:
        v = data[k]
        if v is None:
            vals.append("NULL")
        elif isinstance(v, str):
            vals.append(f"'{sanitize_sql_value(v)}'")
        elif isinstance(v, bool):
            vals.append(str(1 if v else 0))
        elif isinstance(v, (int, float)):
            vals.append(str(v))
        else:
            vals.append(f"'{sanitize_sql_value(json.dumps(v))}'")

    query = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals)})"

    result = _run_clickhouse_query(query)
    if result.returncode != 0:
        raise RuntimeError(f"ClickHouse insert failed: {result.stderr}\nQuery: {query[:500]}")


# ---------------------------------------------------------------------------
# Track identification
# ---------------------------------------------------------------------------

def get_track_hash(audio_path: str) -> str:
    """Generate a hash for the audio file."""
    with open(audio_path, 'rb') as f:
        # Read first 1MB for hash (faster than full file)
        content = f.read(1024 * 1024)
    return hashlib.sha256(content).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Run / knowledge queries
# ---------------------------------------------------------------------------

def get_previous_run(track_hash: str) -> Optional[Dict]:
    """Get the most recent run for a track."""
    query = f"""
        SELECT *
        FROM midi_grep.runs
        WHERE track_hash = '{sanitize_sql_value(track_hash)}'
        ORDER BY version DESC
        LIMIT 1
    """
    rows = clickhouse_query(query)
    return rows[0] if rows else None


def get_best_run(track_hash: str) -> Optional[Dict]:
    """Get the best-scoring run for a track."""
    query = f"""
        SELECT *
        FROM midi_grep.runs
        WHERE track_hash = '{sanitize_sql_value(track_hash)}'
        ORDER BY similarity_overall DESC
        LIMIT 1
    """
    rows = clickhouse_query(query)
    return rows[0] if rows else None


def get_learned_knowledge(genre: str, bpm: float, key_type: str) -> List[Dict]:
    """Get relevant learned knowledge for this track context (legacy — genre-only).

    Prefer retrieve_relevant_knowledge() for query-aware retrieval.
    """
    # First try exact genre match
    query = f"""
        SELECT parameter_name, parameter_new_value, similarity_improvement, confidence, genre
        FROM midi_grep.knowledge
        WHERE (
            (genre = '{sanitize_sql_value(genre)}' AND bpm_range_low <= {bpm} AND bpm_range_high >= {bpm})
            OR genre = ''
        )
          AND confidence > 0.5
          AND similarity_improvement > 0.05
        ORDER BY similarity_improvement DESC
        LIMIT 20
    """
    results = clickhouse_query(query)

    if results:
        genres_found = set(r.get('genre', '') for r in results)
        print(f"       Found {len(results)} knowledge items from genres: {genres_found}")

    return results


def retrieve_relevant_knowledge(
    comparison: Dict,
    genre: str,
    bpm: float,
    tried_params: Optional[Dict] = None,
) -> str:
    """Query-aware retrieval: find knowledge relevant to the CURRENT problem.

    Instead of retrieving by genre alone, this looks at what's actually wrong
    (which bands are off, which direction) and finds proven fixes for those
    specific problems. Results are formatted as compact context for LLM injection.

    This is the real RAG function — retrieval is driven by the current query,
    not just a static category.

    Args:
        comparison: Current comparison.json dict with band differences
        genre: Current track genre
        bpm: Current track BPM
        tried_params: Dict of param_name -> [values already tried] to exclude

    Returns:
        Compact string for LLM prompt injection, or "" if nothing relevant found.
    """
    if not _ensure_clickhouse():
        return ""

    # Identify what's wrong from comparison data
    orig_bands = comparison.get("original", {}).get("bands", {})
    rend_bands = comparison.get("rendered", {}).get("bands", {})

    if not orig_bands or not rend_bands:
        return ""

    # Find the worst bands (sorted by absolute difference)
    band_diffs = {}
    for band in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
        diff = rend_bands.get(band, 0) - orig_bands.get(band, 0)
        band_diffs[band] = diff

    worst_bands = sorted(band_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    if not worst_bands or abs(worst_bands[0][1]) < 0.03:
        return ""  # Bands are close enough, no retrieval needed

    # Map worst band to likely parameter prefix
    band_to_fx = {
        "sub_bass": "bassFx",
        "bass": "bassFx",
        "low_mid": "bassFx",
        "mid": "midFx",
        "high_mid": "highFx",
        "high": "highFx",
    }

    # Build query: prioritize fixes for the worst bands, prefer same genre + similar BPM
    worst_fx = band_to_fx.get(worst_bands[0][0], "bassFx")
    direction = "too quiet" if worst_bands[0][1] < 0 else "too loud"

    safe_genre = sanitize_sql_value(genre)
    query = f"""
        SELECT
            parameter_name, parameter_new_value, similarity_improvement,
            genre, bpm_range_low, bpm_range_high
        FROM midi_grep.knowledge
        WHERE parameter_name LIKE '{worst_fx}%'
          AND similarity_improvement > 0.03
        ORDER BY
            -- Prefer same genre (10x boost)
            multiIf(genre = '{safe_genre}', 10, genre = '', 2, 1) *
            -- Prefer similar BPM
            (1.0 / (1 + abs(bpm_range_low + bpm_range_high - {bpm * 2}) / 2)) *
            -- Prefer higher improvement
            similarity_improvement
        DESC
        LIMIT 5
    """

    results = clickhouse_query(query)

    # Also query for the second-worst band if it's significantly off
    if len(worst_bands) > 1 and abs(worst_bands[1][1]) > 0.08:
        second_fx = band_to_fx.get(worst_bands[1][0], "midFx")
        if second_fx != worst_fx:
            query2 = f"""
                SELECT
                    parameter_name, parameter_new_value, similarity_improvement,
                    genre, bpm_range_low, bpm_range_high
                FROM midi_grep.knowledge
                WHERE parameter_name LIKE '{second_fx}%'
                  AND similarity_improvement > 0.03
                ORDER BY similarity_improvement DESC
                LIMIT 3
            """
            results.extend(clickhouse_query(query2))

    if not results:
        return ""

    # Filter out already-tried values
    if tried_params:
        filtered = []
        for r in results:
            param = r.get("parameter_name", "")
            val = r.get("parameter_new_value", "")
            if param not in tried_params or val not in tried_params[param]:
                filtered.append(r)
        results = filtered

    if not results:
        return ""

    # Format as compact, actionable context for LLM
    lines = [f"## Proven fixes for {worst_bands[0][0]} ({direction}):"]
    seen = set()
    for r in results:
        param = r.get("parameter_name", "")
        val = r.get("parameter_new_value", "")
        key = f"{param}={val}"
        if key in seen:
            continue
        seen.add(key)
        genre_tag = f" [{r.get('genre', 'any')}]" if r.get("genre") else ""
        lines.append(
            f"- Set {param}={val} (+{r['similarity_improvement']*100:.0f}% improvement{genre_tag})"
        )

    return "\n".join(lines)


def find_similar_tracks(
    orig_bands: Dict[str, float],
    genre: str,
    bpm: float,
    exclude_hash: str = "",
    min_similarity: float = 0.70,
) -> List[Dict]:
    """Find tracks with similar frequency profiles for code reuse.

    Cross-track retrieval: instead of matching by genre label,
    find runs whose frequency band distribution is closest to the
    current track. This enables learning across genres when the
    spectral profile is similar.

    Args:
        orig_bands: Original track's frequency band distribution
        genre: Current genre (used for tiebreaking, not filtering)
        bpm: Current BPM (used for tiebreaking)
        exclude_hash: Track hash to exclude (current track)
        min_similarity: Minimum similarity_overall for candidate runs

    Returns:
        List of dicts with track_hash, strudel_code, similarity_overall, genre
    """
    if not _ensure_clickhouse():
        return []

    sub_bass = orig_bands.get("sub_bass", 0)
    bass = orig_bands.get("bass", 0)
    low_mid = orig_bands.get("low_mid", 0)
    mid = orig_bands.get("mid", 0)
    high_mid = orig_bands.get("high_mid", 0)
    high = orig_bands.get("high", 0)

    exclude_clause = ""
    if exclude_hash:
        exclude_clause = f"AND track_hash != '{sanitize_sql_value(exclude_hash)}'"

    # L1 distance across 6 frequency bands — lower = more similar spectral profile
    query = f"""
        SELECT
            track_hash, strudel_code, similarity_overall, genre, bpm,
            (
                abs(band_sub_bass - {sub_bass}) +
                abs(band_bass - {bass}) +
                abs(band_low_mid - {low_mid}) +
                abs(band_mid - {mid}) +
                abs(band_high_mid - {high_mid}) +
                abs(band_high - {high})
            ) AS spectral_distance
        FROM midi_grep.runs
        WHERE similarity_overall > {min_similarity}
          {exclude_clause}
          AND strudel_code != ''
        ORDER BY spectral_distance ASC
        LIMIT 3
    """

    results = clickhouse_query(query)
    if results:
        for r in results:
            print(f"       Similar track: {r.get('genre', '?')} "
                  f"({r.get('similarity_overall', 0)*100:.0f}% sim, "
                  f"dist={r.get('spectral_distance', 99):.3f})")
    return results


# ---------------------------------------------------------------------------
# Parameter extraction (local fallback when strudel_params.py is absent)
# ---------------------------------------------------------------------------

if not _STRUDEL_PARAMS_IMPORTED:
    def extract_parameters_from_code(code: str) -> Dict[str, Dict[str, Any]]:
        """Extract effect parameters from Strudel code for learning."""
        params = {}

        # Pattern to match effect functions and their parameters
        fx_pattern = r'let\s+(\w+Fx)\s*=\s*p\s*=>\s*p([^\n]*(?:\n\s+\.[^\n]*)*)'

        for match in re.finditer(fx_pattern, code, re.MULTILINE):
            fx_name = match.group(1)  # e.g., "bassFx"
            fx_body = match.group(2)  # e.g., ".sound("supersaw").gain(0.15)..."

            fx_params = {}

            # Extract .gain(value)
            gain_m = re.search(r'\.gain\(([0-9.]+)\)', fx_body)
            if gain_m:
                fx_params['gain'] = float(gain_m.group(1))

            # Extract .hpf(value)
            hpf_m = re.search(r'\.hpf\(([0-9.]+)\)', fx_body)
            if hpf_m:
                fx_params['hpf'] = float(hpf_m.group(1))

            # Extract .lpf(value)
            lpf_m = re.search(r'\.lpf\(([0-9.]+)\)', fx_body)
            if lpf_m:
                fx_params['lpf'] = float(lpf_m.group(1))

            # Extract .attack(value)
            attack_m = re.search(r'\.attack\(([0-9.]+)\)', fx_body)
            if attack_m:
                fx_params['attack'] = float(attack_m.group(1))

            # Extract .decay(value)
            decay_m = re.search(r'\.decay\(([0-9.]+)\)', fx_body)
            if decay_m:
                fx_params['decay'] = float(decay_m.group(1))

            # Extract .sustain(value)
            sustain_m = re.search(r'\.sustain\(([0-9.]+)\)', fx_body)
            if sustain_m:
                fx_params['sustain'] = float(sustain_m.group(1))

            # Extract .release(value)
            release_m = re.search(r'\.release\(([0-9.]+)\)', fx_body)
            if release_m:
                fx_params['release'] = float(release_m.group(1))

            # Extract .room(value)
            room_m = re.search(r'\.room\(([0-9.]+)\)', fx_body)
            if room_m:
                fx_params['room'] = float(room_m.group(1))

            # Extract .sound("value")
            sound_m = re.search(r'\.sound\("([^"]+)"\)', fx_body)
            if sound_m:
                fx_params['sound'] = sound_m.group(1)

            if fx_params:
                params[fx_name] = fx_params

        return params


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def learn_from_improvement(
    track_hash: str,
    genre: str,
    bpm: float,
    key_type: str,
    old_code: str,
    new_code: str,
    old_similarity: float,
    new_similarity: float
) -> int:
    """
    Compare before/after code and store successful parameter changes as knowledge.
    Returns number of knowledge entries stored.
    """
    if new_similarity <= old_similarity:
        return 0  # No improvement, nothing to learn

    improvement = new_similarity - old_similarity
    if improvement < 0.01:  # Less than 1% improvement, skip
        return 0

    old_params = extract_parameters_from_code(old_code)
    new_params = extract_parameters_from_code(new_code)

    entries_stored = 0

    for fx_name in new_params:
        if fx_name not in old_params:
            continue

        for param_name in new_params[fx_name]:
            if param_name not in old_params[fx_name]:
                continue

            old_val = old_params[fx_name][param_name]
            new_val = new_params[fx_name][param_name]

            # Skip if values are the same
            if old_val == new_val:
                continue

            # Store the learned improvement
            full_param_name = f"{fx_name}.{param_name}"
            success = store_knowledge(
                track_hash=track_hash,
                genre=genre,
                bpm=bpm,
                key_type=key_type,
                parameter_name=full_param_name,
                old_value=str(old_val),
                new_value=str(new_val),
                improvement=improvement
            )
            if success:
                entries_stored += 1
                print(f"       Learned: {full_param_name}: {old_val} -> {new_val} (+{improvement*100:.1f}%)")

    return entries_stored


# ---------------------------------------------------------------------------
# Artist helpers
# ---------------------------------------------------------------------------

def normalize_artist(name: str) -> str:
    """Normalize artist name for consistent matching."""
    normalized = name.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    return normalized


def get_artist_presets(artist: str) -> Dict[str, Dict]:
    """
    Get artist-specific parameter presets from learned knowledge.
    Uses the artist_knowledge table populated by learn_artist.py.
    """
    if not artist:
        return {}

    artist_normalized = normalize_artist(artist)

    query = f"""
        SELECT parameter_name, parameter_value, avg_similarity, track_count
        FROM midi_grep.artist_knowledge
        WHERE artist_normalized = '{sanitize_sql_value(artist_normalized)}'
          AND confidence > 0.3
        ORDER BY avg_similarity DESC
    """

    try:
        rows = clickhouse_query(query)
    except:
        return {}  # Table might not exist yet

    presets = {}
    for row in rows:
        param_name = row.get('parameter_name', '')
        value = row.get('parameter_value', '')

        if '.' in param_name:
            fx_name, param = param_name.split('.', 1)
            if fx_name not in presets:
                presets[fx_name] = {}
            try:
                presets[fx_name][param] = float(value)
            except ValueError:
                presets[fx_name][param] = value

    return presets


def detect_artist_from_path(path: str) -> Optional[str]:
    """Try to detect artist name from file path or cache directory."""
    # Common patterns in YouTube titles: "Artist - Song Title"
    # or "Song Title - Artist"
    path_str = str(path)

    # Look for patterns like "Artist Name - "
    match = re.search(r'/([^/]+)\s*-\s*[^/]+/(?:melodic|v\d+)', path_str)
    if match:
        potential_artist = match.group(1).strip()
        # Filter out common non-artist words
        skip_words = ['official', 'video', 'audio', 'lyric', 'hd', '4k', 'remix']
        if not any(w in potential_artist.lower() for w in skip_words):
            return potential_artist

    return None


# ---------------------------------------------------------------------------
# Genre presets
# ---------------------------------------------------------------------------

def get_genre_presets(genre: str) -> Dict[str, Any]:
    """
    Get genre-specific parameter presets from successful runs.
    Analyzes high-performing runs and extracts average parameters.
    """
    # Query successful runs for this genre (>80% similarity)
    query = f"""
        SELECT
            strudel_code,
            similarity_overall,
            bpm
        FROM midi_grep.runs
        WHERE genre = '{sanitize_sql_value(genre)}'
          AND similarity_overall > 0.80
        ORDER BY similarity_overall DESC
        LIMIT 10
    """
    rows = clickhouse_query(query)

    if not rows:
        # Fall back to all high-performing runs
        query = """
            SELECT
                strudel_code,
                similarity_overall,
                bpm
            FROM midi_grep.runs
            WHERE similarity_overall > 0.85
            ORDER BY similarity_overall DESC
            LIMIT 10
        """
        rows = clickhouse_query(query)

    if not rows:
        return {}

    # Aggregate parameters from successful runs
    param_sums = {}
    param_counts = {}

    for row in rows:
        code = row.get('strudel_code', '')
        params = extract_parameters_from_code(code)

        for fx_name, fx_params in params.items():
            if fx_name not in param_sums:
                param_sums[fx_name] = {}
                param_counts[fx_name] = {}

            for param, value in fx_params.items():
                if isinstance(value, (int, float)):
                    if param not in param_sums[fx_name]:
                        param_sums[fx_name][param] = 0
                        param_counts[fx_name][param] = 0
                    param_sums[fx_name][param] += value
                    param_counts[fx_name][param] += 1

    # Calculate averages
    presets = {}
    for fx_name in param_sums:
        presets[fx_name] = {}
        for param in param_sums[fx_name]:
            if param_counts[fx_name][param] > 0:
                avg = param_sums[fx_name][param] / param_counts[fx_name][param]
                presets[fx_name][param] = round(avg, 3)

    return presets


# ---------------------------------------------------------------------------
# Run and knowledge storage
# ---------------------------------------------------------------------------

def store_run(
    track_hash: str,
    track_name: str,
    version: int,
    strudel_code: str,
    bpm: float,
    key: str,
    style: str,
    genre: str,
    comparison: Dict,
    mix_params: Dict,
    improved_from: Optional[int] = None,
    ai_suggestions: Optional[List[str]] = None
):
    """Store a run in ClickHouse."""
    # Extract data from comparison structure
    comp_scores = comparison.get("comparison", {})
    orig_bands = comparison.get("original", {}).get("bands", {})
    rend_bands = comparison.get("rendered", {}).get("bands", {})

    # Compute band differences (rendered - original)
    band_diffs = {}
    for band in ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"]:
        band_diffs[band] = rend_bands.get(band, 0) - orig_bands.get(band, 0)

    data = {
        "track_hash": track_hash,
        "track_name": track_name,
        "version": version,
        "bpm": bpm,
        "key": key,
        "style": style,
        "genre": genre,
        "strudel_code": strudel_code,
        "similarity_overall": comp_scores.get("overall_similarity", 0),
        "similarity_mfcc": comp_scores.get("mfcc_similarity", 0),
        "similarity_chroma": comp_scores.get("chroma_similarity", 0),
        "similarity_frequency": comp_scores.get("frequency_balance_similarity", 0),
        "similarity_rhythm": comp_scores.get("tempo_similarity", 0),
        "band_sub_bass": band_diffs.get("sub_bass", 0),
        "band_bass": band_diffs.get("bass", 0),
        "band_low_mid": band_diffs.get("low_mid", 0),
        "band_mid": band_diffs.get("mid", 0),
        "band_high_mid": band_diffs.get("high_mid", 0),
        "band_high": band_diffs.get("high", 0),
        "mix_params": json.dumps(mix_params),
        "improved_from_version": improved_from,
        "ai_suggestions": json.dumps(ai_suggestions) if ai_suggestions else ""
    }

    return clickhouse_insert("midi_grep.runs", data)


def store_knowledge(
    track_hash: str,
    genre: str,
    bpm: float,
    key_type: str,
    parameter_name: str,
    old_value: str,
    new_value: str,
    improvement: float
):
    """Store learned knowledge for this specific track."""
    data = {
        "track_hash": track_hash,
        "genre": genre,
        "bpm_range_low": max(0, bpm - 20),
        "bpm_range_high": bpm + 20,
        "key_type": key_type,
        "parameter_name": parameter_name,
        "parameter_old_value": old_value,
        "parameter_new_value": new_value,
        "similarity_improvement": improvement,
        "confidence": 1.0
    }
    return clickhouse_insert("midi_grep.knowledge", data)
