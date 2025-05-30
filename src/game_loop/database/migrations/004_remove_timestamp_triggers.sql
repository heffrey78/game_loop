-- Migration: Remove timestamp triggers to prevent conflicts with SQLAlchemy
--
-- This migration removes the timestamp triggers that were created in migration 003.
-- These triggers conflict with SQLAlchemy's native timestamp handling and cause
-- timing issues where both the trigger and SQLAlchemy try to set the same timestamp.
-- SQLAlchemy will handle timestamp updates natively through the TimestampMixin.

-- Drop all timestamp triggers
DROP TRIGGER IF EXISTS update_regions_updated_at ON regions;
DROP TRIGGER IF EXISTS update_locations_updated_at ON locations;
DROP TRIGGER IF EXISTS update_objects_updated_at ON objects;
DROP TRIGGER IF EXISTS update_npcs_updated_at ON npcs;
DROP TRIGGER IF EXISTS update_location_connections_updated_at ON location_connections;
DROP TRIGGER IF EXISTS update_players_updated_at ON players;
DROP TRIGGER IF EXISTS update_player_inventory_updated_at ON player_inventory;
DROP TRIGGER IF EXISTS update_player_knowledge_updated_at ON player_knowledge;
DROP TRIGGER IF EXISTS update_player_skills_updated_at ON player_skills;
DROP TRIGGER IF EXISTS update_player_history_updated_at ON player_history;
DROP TRIGGER IF EXISTS update_quests_updated_at ON quests;
DROP TRIGGER IF EXISTS update_world_rules_updated_at ON world_rules;
DROP TRIGGER IF EXISTS update_evolution_events_updated_at ON evolution_events;
DROP TRIGGER IF EXISTS update_game_sessions_updated_at ON game_sessions;

-- Drop the trigger function as it's no longer needed
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Note: SQLAlchemy will now handle all timestamp updates through the TimestampMixin
-- with onupdate=func.clock_timestamp() which provides more precise timing control.
