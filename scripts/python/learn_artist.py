#!/usr/bin/env python3
"""
Learn an artist's style by processing multiple tracks.

Uses YouTube search to find tracks, downloads them, and processes through
the AI pipeline to build artist-specific knowledge.

Usage:
    python learn_artist.py "Dj Brunin XM" --limit 5
    python learn_artist.py "Parov Stelar" --genre electro_swing --limit 10
"""

import argparse
import subprocess
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ai_improver import (
    clickhouse_query,
    clickhouse_insert,
    extract_parameters_from_code,
    get_track_hash,
    CLICKHOUSE_BIN,
    CLICKHOUSE_DB
)


def ensure_artist_table():
    """Create artist_tracks table if it doesn't exist."""
    query = """
    CREATE TABLE IF NOT EXISTS midi_grep.artist_tracks (
        id UUID DEFAULT generateUUIDv4(),
        created_at DateTime DEFAULT now(),
        artist String,
        artist_normalized String,
        track_name String,
        youtube_url String,
        track_hash String,
        genre String,
        bpm Float32,
        key String,
        similarity Float32,
        processed Bool DEFAULT false
    ) ENGINE = MergeTree()
    ORDER BY (artist_normalized, created_at)
    """
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query
    ]
    subprocess.run(cmd, capture_output=True)


def ensure_artist_knowledge_table():
    """Create artist_knowledge table for artist-specific learnings."""
    query = """
    CREATE TABLE IF NOT EXISTS midi_grep.artist_knowledge (
        id UUID DEFAULT generateUUIDv4(),
        created_at DateTime DEFAULT now(),
        artist String,
        artist_normalized String,
        parameter_name String,
        parameter_value String,
        avg_similarity Float32,
        track_count UInt32,
        confidence Float32
    ) ENGINE = MergeTree()
    ORDER BY (artist_normalized, parameter_name)
    """
    cmd = [
        str(CLICKHOUSE_BIN), "local",
        "--path", str(CLICKHOUSE_DB),
        "--query", query
    ]
    subprocess.run(cmd, capture_output=True)


def normalize_artist(name: str) -> str:
    """Normalize artist name for consistent matching."""
    # Lowercase, remove special chars, collapse spaces
    normalized = name.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    return normalized


def search_youtube(artist: str, limit: int = 10) -> List[Dict]:
    """Search YouTube for artist's tracks using yt-dlp."""
    search_query = f"ytsearch{limit}:{artist} official"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s|%(title)s|%(duration)s",
        search_query
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"YouTube search failed: {result.stderr}")
        return []

    lines = result.stdout.strip().split('\n')
    tracks = []

    for line in lines:
        if not line or '|' not in line:
            continue

        parts = line.split('|')
        if len(parts) < 3:
            continue

        video_id = parts[0].strip()
        title = parts[1].strip()
        duration_str = parts[2].strip()

        if not video_id or not title:
            continue

        # Parse duration (can be float like "233.0")
        try:
            dur = float(duration_str) if duration_str else 0
            # Skip very short (<1min) or very long (>10min) videos
            if dur < 60 or dur > 600:
                continue
        except:
            continue

        tracks.append({
            "id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": int(dur)
        })

    return tracks


def get_processed_tracks(artist_normalized: str) -> set:
    """Get already processed track URLs for this artist."""
    query = f"""
        SELECT youtube_url
        FROM midi_grep.artist_tracks
        WHERE artist_normalized = '{artist_normalized}'
          AND processed = 1
    """
    rows = clickhouse_query(query)
    return {row.get('youtube_url', '') for row in rows}


def process_track(url: str, artist: str, genre: str = "") -> Optional[Dict]:
    """Process a single track through the MIDI-grep pipeline."""
    print(f"\n  Processing: {url}")

    # Build command
    cmd = [
        "./bin/midi-grep", "extract",
        "--url", url,
        "--render", "auto",
        "--iterate", "1"  # Just one iteration for learning
    ]

    if genre:
        cmd.extend(["--genre", genre])

    # Run extraction (handle non-UTF-8 output)
    result = subprocess.run(cmd, capture_output=True, timeout=600)

    # Decode with error handling for non-ASCII characters
    try:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
    except:
        stdout = str(result.stdout)
        stderr = str(result.stderr)

    if result.returncode != 0:
        print(f"    Failed: {stderr[:200]}")
        return None

    # Parse output for metadata
    output = stdout + stderr

    # Extract BPM
    bpm_match = re.search(r'BPM[:\s]+(\d+\.?\d*)', output)
    bpm = float(bpm_match.group(1)) if bpm_match else 0

    # Extract key
    key_match = re.search(r'Key[:\s]+([A-G][#b]?\s*(?:major|minor))', output, re.I)
    key = key_match.group(1) if key_match else ""

    # Extract similarity
    sim_match = re.search(r'similarity[:\s]+(\d+\.?\d*)%', output, re.I)
    similarity = float(sim_match.group(1)) / 100 if sim_match else 0

    # Find the cache directory
    cache_match = re.search(r'\.cache/stems/([^/\n]+)', output)
    cache_dir = None
    if cache_match:
        cache_dir = Path(".cache/stems") / cache_match.group(1)

    return {
        "bpm": bpm,
        "key": key,
        "similarity": similarity,
        "cache_dir": str(cache_dir) if cache_dir else None
    }


def store_artist_track(
    artist: str,
    track_name: str,
    url: str,
    track_hash: str,
    genre: str,
    bpm: float,
    key: str,
    similarity: float
):
    """Store a processed track in the artist_tracks table."""
    data = {
        "artist": artist,
        "artist_normalized": normalize_artist(artist),
        "track_name": track_name,
        "youtube_url": url,
        "track_hash": track_hash,
        "genre": genre,
        "bpm": bpm,
        "key": key,
        "similarity": similarity,
        "processed": 1
    }
    return clickhouse_insert("midi_grep.artist_tracks", data)


def aggregate_artist_knowledge(artist: str):
    """Aggregate learnings from all processed tracks for an artist."""
    artist_normalized = normalize_artist(artist)

    # Get all successful runs for this artist's tracks
    query = f"""
        SELECT r.strudel_code, r.similarity_overall, r.genre, r.bpm
        FROM midi_grep.runs r
        INNER JOIN midi_grep.artist_tracks t
            ON r.track_hash = t.track_hash
        WHERE t.artist_normalized = '{artist_normalized}'
          AND r.similarity_overall > 0.70
        ORDER BY r.similarity_overall DESC
    """
    runs = clickhouse_query(query)

    if not runs:
        print(f"No successful runs found for {artist}")
        return

    print(f"\nAggregating knowledge from {len(runs)} successful tracks...")

    # Aggregate parameters
    param_values = {}  # param_name -> [(value, similarity), ...]

    for run in runs:
        code = run.get('strudel_code', '')
        similarity = run.get('similarity_overall', 0)

        params = extract_parameters_from_code(code)

        for fx_name, fx_params in params.items():
            for param, value in fx_params.items():
                if isinstance(value, (int, float)):
                    full_name = f"{fx_name}.{param}"
                    if full_name not in param_values:
                        param_values[full_name] = []
                    param_values[full_name].append((value, similarity))

    # Store aggregated knowledge
    entries = 0
    for param_name, values in param_values.items():
        if len(values) < 2:
            continue

        # Weighted average by similarity
        total_weight = sum(sim for _, sim in values)
        weighted_avg = sum(val * sim for val, sim in values) / total_weight
        avg_sim = sum(sim for _, sim in values) / len(values)

        # Best value (highest similarity)
        best_val = max(values, key=lambda x: x[1])[0]

        # Store in artist_knowledge
        data = {
            "artist": artist,
            "artist_normalized": artist_normalized,
            "parameter_name": param_name,
            "parameter_value": str(round(best_val, 4)),
            "avg_similarity": avg_sim,
            "track_count": len(values),
            "confidence": min(1.0, len(values) / 5.0)  # Confidence based on sample size
        }

        success = clickhouse_insert("midi_grep.artist_knowledge", data)
        if success:
            entries += 1
            print(f"  {param_name}: {best_val:.4f} (from {len(values)} tracks, {avg_sim*100:.1f}% avg)")

    print(f"\n✓ Stored {entries} artist-specific parameters for {artist}")


def get_artist_presets(artist: str) -> Dict[str, Dict]:
    """Get artist-specific presets from learned knowledge."""
    artist_normalized = normalize_artist(artist)

    query = f"""
        SELECT parameter_name, parameter_value, avg_similarity, track_count
        FROM midi_grep.artist_knowledge
        WHERE artist_normalized = '{artist_normalized}'
          AND confidence > 0.3
        ORDER BY avg_similarity DESC
    """
    rows = clickhouse_query(query)

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


def show_artist_stats(artist: str):
    """Show statistics for an artist."""
    artist_normalized = normalize_artist(artist)

    # Track count
    query = f"""
        SELECT
            count() as track_count,
            avg(similarity) as avg_similarity,
            avg(bpm) as avg_bpm
        FROM midi_grep.artist_tracks
        WHERE artist_normalized = '{artist_normalized}'
          AND processed = 1
    """
    stats = clickhouse_query(query)

    if not stats or stats[0].get('track_count', 0) == 0:
        print(f"No tracks processed for {artist}")
        return

    s = stats[0]
    print(f"\n{'='*60}")
    print(f"Artist: {artist}")
    print(f"{'='*60}")
    print(f"Tracks processed: {s['track_count']}")
    print(f"Average similarity: {s['avg_similarity']*100:.1f}%")
    print(f"Average BPM: {s['avg_bpm']:.0f}")

    # Show learned parameters
    presets = get_artist_presets(artist)
    if presets:
        print(f"\nLearned parameters:")
        for fx_name, params in presets.items():
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {fx_name}: {param_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Learn an artist\'s style from their tracks'
    )
    parser.add_argument('artist', nargs='?', help='Artist name to learn')
    parser.add_argument('--limit', type=int, default=5,
                        help='Number of tracks to process (default: 5)')
    parser.add_argument('--genre', default='',
                        help='Genre hint (e.g., electro_swing, brazilian_funk)')
    parser.add_argument('--stats', action='store_true',
                        help='Show stats for artist instead of processing')
    parser.add_argument('--list', action='store_true',
                        help='List all learned artists')
    parser.add_argument('--aggregate', action='store_true',
                        help='Re-aggregate knowledge for artist')

    args = parser.parse_args()

    # Ensure tables exist
    ensure_artist_table()
    ensure_artist_knowledge_table()

    if args.list:
        query = """
            SELECT
                artist,
                count() as tracks,
                avg(similarity) as avg_sim
            FROM midi_grep.artist_tracks
            WHERE processed = 1
            GROUP BY artist
            ORDER BY tracks DESC
        """
        artists = clickhouse_query(query)

        if not artists:
            print("No artists learned yet")
            return

        print("\nLearned Artists:")
        print("-" * 50)
        for a in artists:
            print(f"  {a['artist']}: {a['tracks']} tracks ({a['avg_sim']*100:.1f}% avg)")
        return

    if not args.artist:
        parser.print_help()
        return

    if args.stats:
        show_artist_stats(args.artist)
        return

    if args.aggregate:
        aggregate_artist_knowledge(args.artist)
        return

    # Search for tracks
    print(f"\n🔍 Searching YouTube for: {args.artist}")
    tracks = search_youtube(args.artist, args.limit * 2)  # Search more, filter later

    if not tracks:
        print("No tracks found")
        return

    print(f"Found {len(tracks)} potential tracks")

    # Filter already processed
    processed = get_processed_tracks(normalize_artist(args.artist))
    tracks = [t for t in tracks if t['url'] not in processed][:args.limit]

    if not tracks:
        print("All tracks already processed")
        show_artist_stats(args.artist)
        return

    print(f"Processing {len(tracks)} new tracks...")

    # Process each track
    successful = 0
    for i, track in enumerate(tracks, 1):
        print(f"\n[{i}/{len(tracks)}] {track['title']}")

        result = process_track(track['url'], args.artist, args.genre)

        if result:
            # Generate track hash from URL
            track_hash = hashlib.sha256(track['url'].encode()).hexdigest()[:16]

            # Store track info
            store_artist_track(
                artist=args.artist,
                track_name=track['title'],
                url=track['url'],
                track_hash=track_hash,
                genre=args.genre or "auto",
                bpm=result['bpm'],
                key=result['key'],
                similarity=result['similarity']
            )

            successful += 1
            print(f"    ✓ BPM: {result['bpm']:.0f}, Key: {result['key']}, Similarity: {result['similarity']*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"Processed {successful}/{len(tracks)} tracks successfully")

    # Aggregate knowledge
    if successful > 0:
        aggregate_artist_knowledge(args.artist)

    # Show final stats
    show_artist_stats(args.artist)


if __name__ == '__main__':
    main()
