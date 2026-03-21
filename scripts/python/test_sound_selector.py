#!/usr/bin/env python3
"""Tests for sound_selector.py — genre-aware sound RAG."""

import pytest
from sound_selector import retrieve_genre_context, GENRE_PALETTES

# Import valid sounds for cross-validation
try:
    from ollama_agent import VALID_SOUNDS, VALID_DRUM_BANKS, INVALID_GM_PATTERNS
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    VALID_SOUNDS = set()
    VALID_DRUM_BANKS = set()
    INVALID_GM_PATTERNS = []


class TestRetrieveGenreContext:
    """Test retrieve_genre_context() — the core RAG function."""

    def test_returns_string(self):
        ctx = retrieve_genre_context("brazilian_funk")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_genre_name(self):
        ctx = retrieve_genre_context("brazilian_funk")
        assert "brazilian_funk" in ctx

    def test_contains_categories(self):
        ctx = retrieve_genre_context("house")
        assert "Bass:" in ctx
        assert "Lead:" in ctx
        assert "Drums:" in ctx

    def test_unknown_genre_falls_back(self):
        ctx = retrieve_genre_context("nonexistent_genre_12345")
        assert "default" in ctx.lower() or len(ctx) > 0

    def test_empty_genre_falls_back(self):
        ctx = retrieve_genre_context("")
        assert len(ctx) > 0

    def test_none_genre_falls_back(self):
        ctx = retrieve_genre_context(None)
        assert len(ctx) > 0

    @pytest.mark.parametrize("genre", list(GENRE_PALETTES.keys()))
    def test_all_genres_return_context(self, genre):
        ctx = retrieve_genre_context(genre)
        assert isinstance(ctx, str)
        assert len(ctx) > 20

    def test_genre_normalization(self):
        """Different casings should resolve to the same palette."""
        ctx1 = retrieve_genre_context("Brazilian Funk")
        ctx2 = retrieve_genre_context("brazilian-funk")
        # Both should contain the genre character (resolving to same palette)
        assert "aggressive" in ctx1.lower() or "punchy" in ctx1.lower()
        assert "aggressive" in ctx2.lower() or "punchy" in ctx2.lower()

    def test_token_efficiency(self):
        """Context should be compact — under 100 tokens (~400 chars)."""
        for genre in GENRE_PALETTES:
            ctx = retrieve_genre_context(genre)
            assert len(ctx) < 500, f"Genre {genre} context too long: {len(ctx)} chars"


class TestGenrePalettes:
    """Test GENRE_PALETTES structure and data quality."""

    def test_all_palettes_have_required_keys(self):
        for genre, palette in GENRE_PALETTES.items():
            assert "drums" in palette, f"{genre} missing drums"
            assert "bass" in palette, f"{genre} missing bass"
            assert "lead" in palette, f"{genre} missing lead"
            assert "character" in palette, f"{genre} missing character"

    def test_all_palettes_have_sounds(self):
        for genre, palette in GENRE_PALETTES.items():
            for role in ["drums", "bass", "lead"]:
                sounds = palette[role]
                assert len(sounds) > 0, f"{genre} {role} is empty"

    def test_default_palette_exists(self):
        assert "default" in GENRE_PALETTES

    @pytest.mark.skipif(not HAS_VALIDATION, reason="ollama_agent not available")
    def test_all_sounds_are_valid(self):
        """Cross-validate: every sound in palettes should be in VALID_SOUNDS or VALID_DRUM_BANKS."""
        invalid_entries = []
        for genre, palette in GENRE_PALETTES.items():
            for role in ["bass", "lead", "pad", "high"]:
                for sound in palette.get(role, []):
                    if sound not in VALID_SOUNDS:
                        invalid_entries.append(f"{genre}.{role}: {sound}")
            for bank in palette.get("drums", []):
                if bank not in VALID_DRUM_BANKS and bank not in VALID_SOUNDS:
                    invalid_entries.append(f"{genre}.drums: {bank}")
        assert invalid_entries == [], f"Invalid sounds in palettes:\n" + "\n".join(invalid_entries)

    @pytest.mark.skipif(not HAS_VALIDATION, reason="ollama_agent not available")
    def test_no_invalid_gm_patterns(self):
        """No sound should match INVALID_GM_PATTERNS (hallucinated numbered names)."""
        import re
        bad = []
        for genre, palette in GENRE_PALETTES.items():
            for role in ["bass", "lead", "pad", "high"]:
                for sound in palette.get(role, []):
                    for pattern in INVALID_GM_PATTERNS:
                        if re.search(pattern, sound):
                            bad.append(f"{genre}.{role}: {sound} matches {pattern}")
        assert bad == [], f"Sounds matching invalid patterns:\n" + "\n".join(bad)
