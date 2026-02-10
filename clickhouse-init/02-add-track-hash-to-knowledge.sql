-- Migration: Add track_hash to knowledge table for per-track learning
-- Instead of genre-based learning, we now learn per-track

ALTER TABLE midi_grep.knowledge ADD COLUMN IF NOT EXISTS track_hash String DEFAULT '';

-- Create index on track_hash for fast lookups
ALTER TABLE midi_grep.knowledge ADD INDEX IF NOT EXISTS idx_knowledge_track_hash track_hash TYPE bloom_filter GRANULARITY 1;
