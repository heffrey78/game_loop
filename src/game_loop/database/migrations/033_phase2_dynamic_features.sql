-- Migration: Phase 2 Dynamic World Features
-- Adds support for dynamic content generation, player behavior tracking, and adaptive systems

-- Add content metadata to locations table for tracking generated content
ALTER TABLE locations 
ADD COLUMN IF NOT EXISTS dynamic_content JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS content_generation_seed INTEGER,
ADD COLUMN IF NOT EXISTS last_content_update TIMESTAMP DEFAULT NOW();

-- Add player behavior tracking table
CREATE TABLE IF NOT EXISTS player_behavior_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    session_id UUID,
    action_type VARCHAR(50) NOT NULL,
    location_id UUID REFERENCES locations(id),
    direction VARCHAR(20),
    location_type VARCHAR(50),
    expansion_depth INTEGER,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for behavior tracking queries
CREATE INDEX IF NOT EXISTS idx_player_behavior_player_id ON player_behavior_tracking(player_id);
CREATE INDEX IF NOT EXISTS idx_player_behavior_action_type ON player_behavior_tracking(action_type);
CREATE INDEX IF NOT EXISTS idx_player_behavior_timestamp ON player_behavior_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_player_behavior_location_type ON player_behavior_tracking(location_type);

-- Add dynamic content tracking table
CREATE TABLE IF NOT EXISTS dynamic_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location_id UUID NOT NULL REFERENCES locations(id),
    content_type VARCHAR(50) NOT NULL, -- 'object', 'npc', 'quest_hook'
    content_name VARCHAR(200) NOT NULL,
    content_description TEXT,
    generation_context JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    last_interaction TIMESTAMP
);

-- Create indexes for dynamic content queries
CREATE INDEX IF NOT EXISTS idx_dynamic_content_location_id ON dynamic_content(location_id);
CREATE INDEX IF NOT EXISTS idx_dynamic_content_type ON dynamic_content(content_type);
CREATE INDEX IF NOT EXISTS idx_dynamic_content_active ON dynamic_content(is_active);

-- Add enhanced terrain and generation metadata
ALTER TABLE locations 
ADD COLUMN IF NOT EXISTS terrain_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS generation_quality_score FLOAT DEFAULT 0.5,
ADD COLUMN IF NOT EXISTS player_interaction_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_visited TIMESTAMP;

-- Update existing terrain types based on location types
UPDATE locations 
SET terrain_type = CASE 
    WHEN location_type = 'urban_street' OR name LIKE '%Street%' THEN 'urban'
    WHEN location_type = 'industrial_zone' OR name LIKE '%Industrial%' OR name LIKE '%Factory%' THEN 'industrial'
    WHEN location_type = 'basement_corridor' OR name LIKE '%Basement%' OR name LIKE '%Underground%' THEN 'underground'
    WHEN location_type = 'office_space' OR name LIKE '%Office%' OR name LIKE '%Building%' THEN 'building'
    ELSE 'unknown'
END
WHERE terrain_type IS NULL;

-- Create view for generation analytics
CREATE OR REPLACE VIEW generation_analytics AS
SELECT 
    l.terrain_type,
    COUNT(*) as location_count,
    AVG(l.generation_quality_score) as avg_quality,
    AVG(l.player_interaction_count) as avg_interactions,
    COUNT(dc.id) as total_content_items,
    COUNT(CASE WHEN dc.content_type = 'object' THEN 1 END) as object_count,
    COUNT(CASE WHEN dc.content_type = 'npc' THEN 1 END) as npc_count,
    COUNT(CASE WHEN dc.content_type = 'quest_hook' THEN 1 END) as quest_hook_count
FROM locations l
LEFT JOIN dynamic_content dc ON l.id = dc.location_id AND dc.is_active = true
WHERE l.is_dynamic = true
GROUP BY l.terrain_type;

-- Create view for player behavior analysis
CREATE OR REPLACE VIEW player_behavior_analysis AS
SELECT 
    player_id,
    COUNT(*) as total_actions,
    COUNT(DISTINCT location_type) as explored_types,
    MAX(expansion_depth) as max_depth,
    AVG(expansion_depth) as avg_depth,
    mode() WITHIN GROUP (ORDER BY direction) as preferred_direction,
    mode() WITHIN GROUP (ORDER BY location_type) as preferred_location_type,
    MIN(timestamp) as first_action,
    MAX(timestamp) as last_action
FROM player_behavior_tracking
WHERE action_type = 'exploration'
GROUP BY player_id;

-- Create function to clean up old behavior tracking data
CREATE OR REPLACE FUNCTION cleanup_old_behavior_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete behavior tracking data older than 30 days
    DELETE FROM player_behavior_tracking 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Also clean up old dynamic content that hasn't been interacted with
    DELETE FROM dynamic_content 
    WHERE is_active = false 
    AND created_at < NOW() - INTERVAL '7 days'
    AND last_interaction IS NULL;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to update location quality scores
CREATE OR REPLACE FUNCTION update_location_quality_scores()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Update quality scores based on player interactions and content richness
    UPDATE locations 
    SET generation_quality_score = LEAST(1.0, GREATEST(0.1, 
        0.5 + -- Base score
        (player_interaction_count * 0.1) + -- Interaction bonus
        (COALESCE((
            SELECT COUNT(*) * 0.05 
            FROM dynamic_content dc 
            WHERE dc.location_id = locations.id AND dc.is_active = true
        ), 0)) -- Content richness bonus
    ))
    WHERE is_dynamic = true;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Add function to track player exploration
CREATE OR REPLACE FUNCTION track_player_exploration(
    p_player_id UUID,
    p_session_id UUID,
    p_location_id UUID,
    p_direction VARCHAR(20),
    p_location_type VARCHAR(50),
    p_expansion_depth INTEGER,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    tracking_id UUID;
BEGIN
    INSERT INTO player_behavior_tracking (
        player_id, session_id, action_type, location_id, 
        direction, location_type, expansion_depth, metadata
    ) VALUES (
        p_player_id, p_session_id, 'exploration', p_location_id,
        p_direction, p_location_type, p_expansion_depth, p_metadata
    ) RETURNING id INTO tracking_id;
    
    -- Update location interaction count
    UPDATE locations 
    SET player_interaction_count = player_interaction_count + 1,
        last_visited = NOW()
    WHERE id = p_location_id;
    
    RETURN tracking_id;
END;
$$ LANGUAGE plpgsql;

-- Add comments explaining the migration
COMMENT ON TABLE player_behavior_tracking IS 'Tracks player exploration patterns for adaptive content generation';
COMMENT ON TABLE dynamic_content IS 'Stores dynamically generated content (NPCs, objects, quest hooks) for locations';
COMMENT ON COLUMN locations.terrain_type IS 'Terrain classification for smart generation logic';
COMMENT ON COLUMN locations.generation_quality_score IS 'Quality score for generated locations based on player engagement';
COMMENT ON COLUMN locations.dynamic_content IS 'Metadata about generated content in this location';
COMMENT ON COLUMN locations.content_generation_seed IS 'Random seed used for content generation to ensure consistency';