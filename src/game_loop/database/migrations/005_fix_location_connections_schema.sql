-- Migration: Fix location_connections table schema
--
-- This migration corrects the column names in the location_connections table
-- to match what the application models expect.
-- Changes source_id -> from_location_id and target_id -> to_location_id

-- First check if the old column names exist and rename them
DO $$
BEGIN
    -- Check if 'source_id' exists and rename to 'from_location_id'
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'location_connections'
        AND column_name = 'source_id'
    ) THEN
        ALTER TABLE location_connections RENAME COLUMN source_id TO from_location_id;
    END IF;

    -- Check if 'target_id' exists and rename to 'to_location_id'
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'location_connections'
        AND column_name = 'target_id'
    ) THEN
        ALTER TABLE location_connections RENAME COLUMN target_id TO to_location_id;
    END IF;
END $$;

-- Add missing columns that the application models expect
ALTER TABLE location_connections
ADD COLUMN IF NOT EXISTS connection_type VARCHAR(50) DEFAULT 'path';

ALTER TABLE location_connections
ADD COLUMN IF NOT EXISTS is_visible BOOLEAN DEFAULT true;

ALTER TABLE location_connections
ADD COLUMN IF NOT EXISTS requirements_json JSONB DEFAULT '{}'::jsonb;

-- Remove columns that don't match the model
DO $$
BEGIN
    -- Remove is_hidden column if it exists (model uses is_visible instead)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'location_connections'
        AND column_name = 'is_hidden'
    ) THEN
        -- Migrate data from is_hidden to is_visible (inverse logic)
        UPDATE location_connections SET is_visible = NOT is_hidden WHERE is_hidden IS NOT NULL;
        ALTER TABLE location_connections DROP COLUMN is_hidden;
    END IF;

    -- Remove state_json column if it exists (not in the model)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'location_connections'
        AND column_name = 'state_json'
    ) THEN
        ALTER TABLE location_connections DROP COLUMN state_json;
    END IF;
END $$;

-- Drop old indexes if they exist
DROP INDEX IF EXISTS location_connections_source_idx;
DROP INDEX IF EXISTS location_connections_target_idx;

-- Create new indexes with correct column names for performance
CREATE INDEX IF NOT EXISTS idx_location_connections_from_location
ON location_connections(from_location_id);

CREATE INDEX IF NOT EXISTS idx_location_connections_to_location
ON location_connections(to_location_id);

-- Create composite index for efficient bidirectional lookups
CREATE INDEX IF NOT EXISTS idx_location_connections_bidirectional
ON location_connections(from_location_id, to_location_id);
