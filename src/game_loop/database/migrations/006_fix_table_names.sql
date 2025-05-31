-- Migration: Fix table name mismatches between migrations and SQLAlchemy models
--
-- This migration renames tables to match the names expected by SQLAlchemy models.
-- The migrations created singular table names but the models expect plural names.

-- Fix player_inventory -> player_inventories
ALTER TABLE IF EXISTS player_inventory RENAME TO player_inventories;

-- Fix player_history -> player_histories
ALTER TABLE IF EXISTS player_history RENAME TO player_histories;

-- Update any references to the old table names in triggers
-- Drop old triggers that reference the old table names (they may not exist)
DROP TRIGGER IF EXISTS update_player_inventory_updated_at ON player_inventory;
DROP TRIGGER IF EXISTS update_player_history_updated_at ON player_history;

-- Recreate triggers with correct table names (only if function exists)
DO $$
BEGIN
    -- Check if the function exists before creating triggers
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column') THEN
        CREATE TRIGGER update_player_inventories_updated_at
        BEFORE UPDATE ON player_inventories
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

        CREATE TRIGGER update_player_histories_updated_at
        BEFORE UPDATE ON player_histories
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Note: This migration is safe to run multiple times due to IF EXISTS clauses
