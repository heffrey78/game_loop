-- Migration: Add tables for game state management
--
-- Adds tables to store serialized player state and world state.
-- Updates the game_sessions table to reference these state tables.

-- Enable UUID generation if not already enabled
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing game_sessions table if needed (commented out for safety)
-- This is commented out to avoid accidental data loss!
-- You should manually handle data migration if needed
-- DROP TABLE IF EXISTS game_sessions;

-- Check if game_sessions table has the expected columns
DO $$
BEGIN
    -- If session_id doesn't exist, then we need to alter the table
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'game_sessions' AND column_name = 'session_id'
    ) THEN
        -- Rename the id column to session_id if it exists
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'id'
        ) THEN
            ALTER TABLE game_sessions RENAME COLUMN id TO session_id;
        END IF;

        -- Add missing columns if needed
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'player_state_id'
        ) THEN
            ALTER TABLE game_sessions ADD COLUMN player_state_id UUID;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'world_state_id'
        ) THEN
            ALTER TABLE game_sessions ADD COLUMN world_state_id UUID;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'save_name'
        ) THEN
            ALTER TABLE game_sessions ADD COLUMN save_name VARCHAR(255) NOT NULL DEFAULT 'New Save';
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'updated_at'
        ) THEN
            ALTER TABLE game_sessions ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW();
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'game_sessions' AND column_name = 'game_version'
        ) THEN
            ALTER TABLE game_sessions ADD COLUMN game_version VARCHAR(50);
        END IF;
    END IF;
END $$;

-- Create index on updated_at if needed
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'game_sessions' AND indexname = 'idx_game_sessions_updated_at'
    ) THEN
        CREATE INDEX idx_game_sessions_updated_at ON game_sessions(updated_at DESC);
    END IF;
END $$;

-- Table to store snapshots of player state (serialized as JSON)
CREATE TABLE IF NOT EXISTS player_states (
    player_id UUID NOT NULL,
    session_id UUID NOT NULL,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, session_id)
);

-- Create indexes for player_states if they don't exist
CREATE INDEX IF NOT EXISTS idx_player_states_session_id ON player_states(session_id);
CREATE INDEX IF NOT EXISTS idx_player_states_updated_at ON player_states(updated_at DESC);

-- Table to store snapshots of world state (serialized as JSON)
CREATE TABLE IF NOT EXISTS world_states (
    world_id UUID NOT NULL,
    session_id UUID NOT NULL,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (world_id, session_id)
);

-- Create indexes for world_states if they don't exist
CREATE INDEX IF NOT EXISTS idx_world_states_session_id ON world_states(session_id);
CREATE INDEX IF NOT EXISTS idx_world_states_updated_at ON world_states(updated_at DESC);

-- Optional: Add foreign key constraints from game_sessions to the state tables
-- This requires player_state_id and world_state_id in game_sessions to reference
-- the specific state snapshot IDs used in player_states/world_states.
-- ALTER TABLE game_sessions ADD CONSTRAINT fk_game_sessions_player_state
--    FOREIGN KEY (player_state_id, session_id) REFERENCES player_states(player_id, session_id) ON DELETE RESTRICT;
-- ALTER TABLE game_sessions ADD CONSTRAINT fk_game_sessions_world_state
--    FOREIGN KEY (world_state_id, session_id) REFERENCES world_states(world_id, session_id) ON DELETE RESTRICT;
-- Note: The above FKs assume player_id/world_id in the state tables match player_state_id/world_state_id in game_sessions.
-- Adjust according to the chosen ID strategy.
