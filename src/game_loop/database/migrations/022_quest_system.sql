-- Migration 022: Quest System Tables
-- Create tables for quest system functionality

-- Quests table
CREATE TABLE IF NOT EXISTS quest_definitions (
    quest_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    difficulty VARCHAR(50) NOT NULL,
    steps JSONB NOT NULL,
    prerequisites JSONB DEFAULT '[]',
    rewards JSONB DEFAULT '{}',
    time_limit FLOAT DEFAULT NULL,
    repeatable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quest progress table
CREATE TABLE IF NOT EXISTS quest_progress (
    progress_id SERIAL PRIMARY KEY,
    quest_id VARCHAR(255) REFERENCES quest_definitions(quest_id) ON DELETE CASCADE,
    player_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    current_step INTEGER DEFAULT 0,
    completed_steps JSONB DEFAULT '[]',
    step_progress JSONB DEFAULT '{}',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(quest_id, player_id)
);

-- Quest interactions log
CREATE TABLE IF NOT EXISTS quest_interactions (
    interaction_id SERIAL PRIMARY KEY,
    quest_id VARCHAR(255) REFERENCES quest_definitions(quest_id) ON DELETE CASCADE,
    player_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    interaction_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_quest_progress_player ON quest_progress(player_id);
CREATE INDEX IF NOT EXISTS idx_quest_progress_status ON quest_progress(status);
CREATE INDEX IF NOT EXISTS idx_quest_progress_quest_player ON quest_progress(quest_id, player_id);
CREATE INDEX IF NOT EXISTS idx_quest_interactions_quest ON quest_interactions(quest_id);
CREATE INDEX IF NOT EXISTS idx_quest_interactions_player ON quest_interactions(player_id);
CREATE INDEX IF NOT EXISTS idx_quest_interactions_type ON quest_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_quest_definitions_category ON quest_definitions(category);
CREATE INDEX IF NOT EXISTS idx_quest_definitions_difficulty ON quest_definitions(difficulty);

-- Add constraints for valid enum values (only if they don't exist)
DO $$ BEGIN
    ALTER TABLE quest_definitions ADD CONSTRAINT chk_quest_category
        CHECK (category IN ('delivery', 'exploration', 'combat', 'puzzle', 'social', 'crafting', 'collection'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE quest_definitions ADD CONSTRAINT chk_quest_difficulty
        CHECK (difficulty IN ('trivial', 'easy', 'medium', 'hard', 'legendary'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE quest_progress ADD CONSTRAINT chk_quest_status
        CHECK (status IN ('available', 'active', 'completed', 'failed', 'expired', 'abandoned'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE quest_interactions ADD CONSTRAINT chk_interaction_type
        CHECK (interaction_type IN ('discover', 'accept', 'progress', 'complete', 'abandon', 'query'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Add check constraints for data integrity
DO $$ BEGIN
    ALTER TABLE quest_progress ADD CONSTRAINT chk_current_step_non_negative
        CHECK (current_step >= 0);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE quest_definitions ADD CONSTRAINT chk_time_limit_positive
        CHECK (time_limit IS NULL OR time_limit > 0);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Comments for documentation
COMMENT ON TABLE quest_definitions IS 'Stores quest definitions and metadata';
COMMENT ON TABLE quest_progress IS 'Tracks individual player progress on quests';
COMMENT ON TABLE quest_interactions IS 'Logs all quest-related interactions for analytics';

-- Skip comments since existing table uses different column names
-- COMMENT ON COLUMN quest_definitions.steps IS 'JSONB array containing quest step definitions';
-- COMMENT ON COLUMN quest_definitions.prerequisites IS 'JSONB array of quest IDs that must be completed first';
-- COMMENT ON COLUMN quest_definitions.rewards IS 'JSONB object containing reward definitions';
-- COMMENT ON COLUMN quest_definitions.time_limit IS 'Time limit in seconds, NULL for no limit';

COMMENT ON COLUMN quest_progress.completed_steps IS 'JSONB array of completed step IDs';
COMMENT ON COLUMN quest_progress.step_progress IS 'JSONB object tracking progress within individual steps';

COMMENT ON COLUMN quest_interactions.interaction_data IS 'JSONB object containing interaction-specific data';
