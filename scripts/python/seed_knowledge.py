#!/usr/bin/env python3
"""
Seed the knowledge table from existing successful runs.
Analyzes parameter patterns in high-performing runs and stores them as knowledge.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai_improver import (
    clickhouse_query,
    clickhouse_insert,
    extract_parameters_from_code,
    store_knowledge
)


def seed_knowledge_from_runs():
    """Analyze successful runs and seed the knowledge table."""

    # Query high-performing runs (>80% similarity)
    query = """
        SELECT
            genre,
            bpm,
            key,
            similarity_overall,
            strudel_code
        FROM midi_grep.runs
        WHERE similarity_overall > 0.80
        ORDER BY similarity_overall DESC
        LIMIT 50
    """
    runs = clickhouse_query(query)

    if not runs:
        print("No high-performing runs found")
        return

    print(f"Analyzing {len(runs)} high-performing runs...")

    # Aggregate parameters by genre
    genre_params = {}  # genre -> {fx_name -> {param -> [values]}}

    for run in runs:
        genre = run.get('genre', 'unknown')
        code = run.get('strudel_code', '')
        similarity = run.get('similarity_overall', 0)

        params = extract_parameters_from_code(code)

        if genre not in genre_params:
            genre_params[genre] = {}

        for fx_name, fx_params in params.items():
            if fx_name not in genre_params[genre]:
                genre_params[genre][fx_name] = {}

            for param, value in fx_params.items():
                if isinstance(value, (int, float)):
                    if param not in genre_params[genre][fx_name]:
                        genre_params[genre][fx_name][param] = []
                    genre_params[genre][fx_name][param].append((value, similarity))

    # Calculate weighted averages and store as knowledge
    entries_stored = 0

    for genre, fx_dict in genre_params.items():
        print(f"\n--- {genre if genre else 'generic'} ---")

        for fx_name, params in fx_dict.items():
            for param, values in params.items():
                if len(values) < 2:
                    continue

                # Weighted average by similarity
                total_weight = sum(sim for _, sim in values)
                weighted_avg = sum(val * sim for val, sim in values) / total_weight

                # Use the highest-similarity value as "new"
                best_val = max(values, key=lambda x: x[1])[0]
                avg_sim = sum(sim for _, sim in values) / len(values)

                full_param_name = f"{fx_name}.{param}"

                # Store as knowledge
                success = store_knowledge(
                    genre=genre,
                    bpm=120,  # Generic BPM range
                    key_type="",  # Any key
                    parameter_name=full_param_name,
                    old_value="default",
                    new_value=str(round(best_val, 4)),
                    improvement=avg_sim - 0.70  # Improvement over baseline
                )

                if success:
                    entries_stored += 1
                    print(f"  {full_param_name}: {best_val:.4f} (avg sim: {avg_sim*100:.1f}%)")

    print(f"\n✓ Stored {entries_stored} knowledge entries")


def show_knowledge():
    """Display current knowledge table contents."""
    query = """
        SELECT
            genre,
            parameter_name,
            parameter_new_value,
            similarity_improvement,
            confidence
        FROM midi_grep.knowledge
        ORDER BY genre, parameter_name
        LIMIT 100
    """
    rows = clickhouse_query(query)

    if not rows:
        print("Knowledge table is empty")
        return

    print(f"\nKnowledge table ({len(rows)} entries):")
    print("-" * 80)

    current_genre = None
    for row in rows:
        genre = row.get('genre', '')
        if genre != current_genre:
            current_genre = genre
            print(f"\n[{genre if genre else 'generic'}]")

        param = row.get('parameter_name', '')
        value = row.get('parameter_new_value', '')
        improvement = row.get('similarity_improvement', 0)

        print(f"  {param}: {value} (+{improvement*100:.1f}%)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Seed knowledge from successful runs')
    parser.add_argument('--show', action='store_true', help='Show current knowledge')
    parser.add_argument('--seed', action='store_true', help='Seed knowledge from runs')

    args = parser.parse_args()

    if args.show:
        show_knowledge()
    elif args.seed:
        seed_knowledge_from_runs()
    else:
        parser.print_help()
