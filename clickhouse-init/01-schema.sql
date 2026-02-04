-- MIDI-grep ClickHouse Schema
-- This file is executed on container startup

CREATE DATABASE IF NOT EXISTS midi_grep;

-- Runs table: stores each extraction/improvement run
CREATE TABLE IF NOT EXISTS midi_grep.runs (
    id UUID DEFAULT generateUUIDv4(),
    track_hash String,
    track_name String,
    version UInt32,
    created_at DateTime DEFAULT now(),

    -- Input parameters
    bpm Float32,
    key String,
    style String,
    genre String,

    -- Generated code
    strudel_code String,

    -- Comparison scores (0-1)
    similarity_overall Float32,
    similarity_mfcc Float32,
    similarity_chroma Float32,
    similarity_frequency Float32,
    similarity_rhythm Float32,

    -- Band differences (rendered - original)
    band_sub_bass Float32,
    band_bass Float32,
    band_low_mid Float32,
    band_mid Float32,
    band_high_mid Float32,
    band_high Float32,

    -- Mix parameters used
    mix_params String, -- JSON

    -- AI improvement metadata
    improved_from_version Nullable(UInt32),
    ai_suggestions String, -- JSON: what AI suggested to change

    INDEX idx_track_hash track_hash TYPE bloom_filter GRANULARITY 1
) ENGINE = MergeTree()
ORDER BY (track_hash, version);

-- Knowledge table: stores learned patterns for cross-track learning
CREATE TABLE IF NOT EXISTS midi_grep.knowledge (
    id UUID DEFAULT generateUUIDv4(),
    created_at DateTime DEFAULT now(),

    -- Context (when does this knowledge apply)
    genre String,
    bpm_range_low Float32,
    bpm_range_high Float32,
    key_type String, -- 'major', 'minor'

    -- What was changed
    parameter_name String,
    parameter_old_value String,
    parameter_new_value String,

    -- Impact
    similarity_improvement Float32, -- positive = better
    confidence Float32, -- how many runs confirmed this

    -- Source runs
    run_ids Array(UUID)

) ENGINE = MergeTree()
ORDER BY (genre, parameter_name);

-- Best runs table: tracks the best result per track
CREATE TABLE IF NOT EXISTS midi_grep.best_runs (
    track_hash String,
    best_version UInt32,
    best_similarity Float32,
    updated_at DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY track_hash;

-- Analytics views

-- View: Average improvement by genre
CREATE VIEW IF NOT EXISTS midi_grep.v_genre_stats AS
SELECT
    genre,
    count() as total_runs,
    avg(similarity_overall) as avg_similarity,
    max(similarity_overall) as best_similarity,
    avgIf(similarity_overall - lagInFrame(similarity_overall) OVER (PARTITION BY track_hash ORDER BY version), version > 1) as avg_improvement
FROM midi_grep.runs
GROUP BY genre;

-- View: Most effective parameter changes
CREATE VIEW IF NOT EXISTS midi_grep.v_effective_changes AS
SELECT
    parameter_name,
    genre,
    avg(similarity_improvement) as avg_improvement,
    count() as times_used,
    sum(confidence) as total_confidence
FROM midi_grep.knowledge
GROUP BY parameter_name, genre
ORDER BY avg_improvement DESC;
