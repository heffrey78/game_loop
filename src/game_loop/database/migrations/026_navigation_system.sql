-- Navigation System Migration
-- Adds navigation-related tables and columns for world boundaries and pathfinding

-- Add navigation-related columns to locations table
ALTER TABLE locations
ADD COLUMN IF NOT EXISTS boundary_type VARCHAR(20) DEFAULT 'internal',
ADD COLUMN IF NOT EXISTS expansion_priority FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS navigation_metadata JSONB DEFAULT '{}';

-- Create navigation paths table for caching
CREATE TABLE IF NOT EXISTS navigation_paths (
    path_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_location_id UUID NOT NULL REFERENCES locations(id),
    end_location_id UUID NOT NULL REFERENCES locations(id),
    path_data JSONB NOT NULL,
    total_cost FLOAT NOT NULL,
    criteria VARCHAR(20) NOT NULL,
    is_valid BOOLEAN DEFAULT true,
    last_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_path UNIQUE(start_location_id, end_location_id, criteria)
);

-- Create connection requirements table
CREATE TABLE IF NOT EXISTS connection_requirements (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_location_id UUID NOT NULL REFERENCES locations(id),
    to_location_id UUID NOT NULL REFERENCES locations(id),
    direction VARCHAR(20) NOT NULL,
    requirements JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_connection_requirement 
        UNIQUE(from_location_id, to_location_id)
);

-- Create expansion points table for tracking world expansion opportunities
CREATE TABLE IF NOT EXISTS expansion_points (
    expansion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location_id UUID NOT NULL REFERENCES locations(id),
    direction VARCHAR(20) NOT NULL,
    priority FLOAT NOT NULL DEFAULT 0.0,
    context JSONB NOT NULL DEFAULT '{}',
    suggested_theme VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_expansion_point UNIQUE(location_id, direction)
);

-- Create world boundaries table for tracking boundary classifications
CREATE TABLE IF NOT EXISTS world_boundaries (
    boundary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location_id UUID NOT NULL REFERENCES locations(id) UNIQUE,
    boundary_type VARCHAR(20) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_navigation_paths_locations 
    ON navigation_paths(start_location_id, end_location_id);
    
CREATE INDEX IF NOT EXISTS idx_navigation_paths_criteria 
    ON navigation_paths(criteria);
    
CREATE INDEX IF NOT EXISTS idx_navigation_paths_valid 
    ON navigation_paths(is_valid);

CREATE INDEX IF NOT EXISTS idx_connection_requirements_from 
    ON connection_requirements(from_location_id);
    
CREATE INDEX IF NOT EXISTS idx_connection_requirements_to 
    ON connection_requirements(to_location_id);

CREATE INDEX IF NOT EXISTS idx_locations_boundary_type 
    ON locations(boundary_type);
    
CREATE INDEX IF NOT EXISTS idx_locations_expansion_priority 
    ON locations(expansion_priority);

CREATE INDEX IF NOT EXISTS idx_expansion_points_priority 
    ON expansion_points(priority DESC);
    
CREATE INDEX IF NOT EXISTS idx_expansion_points_active 
    ON expansion_points(is_active);

CREATE INDEX IF NOT EXISTS idx_world_boundaries_type 
    ON world_boundaries(boundary_type);

-- Add constraints to ensure valid boundary types
ALTER TABLE locations ADD CONSTRAINT check_boundary_type 
    CHECK (boundary_type IN ('edge', 'frontier', 'internal', 'isolated'));
    
ALTER TABLE world_boundaries ADD CONSTRAINT check_world_boundary_type 
    CHECK (boundary_type IN ('edge', 'frontier', 'internal', 'isolated'));

-- Add constraints to ensure valid pathfinding criteria
ALTER TABLE navigation_paths ADD CONSTRAINT check_criteria 
    CHECK (criteria IN ('shortest', 'safest', 'scenic', 'fastest'));

-- Add check for valid directions
ALTER TABLE connection_requirements ADD CONSTRAINT check_direction 
    CHECK (direction IN ('north', 'south', 'east', 'west', 'up', 'down', 'in', 'out'));
    
ALTER TABLE expansion_points ADD CONSTRAINT check_expansion_direction 
    CHECK (direction IN ('north', 'south', 'east', 'west', 'up', 'down', 'in', 'out'));

-- Create function to automatically update last_updated timestamps
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_expansion_points_last_updated
    BEFORE UPDATE ON expansion_points
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_world_boundaries_last_updated
    BEFORE UPDATE ON world_boundaries
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_column();

-- Create function to clean up old invalid navigation paths
CREATE OR REPLACE FUNCTION cleanup_invalid_navigation_paths()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM navigation_paths 
    WHERE is_valid = false 
    AND last_validated < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add comment documentation
COMMENT ON TABLE navigation_paths IS 'Cached pathfinding results for performance optimization';
COMMENT ON TABLE connection_requirements IS 'Requirements that must be met to traverse connections';
COMMENT ON TABLE expansion_points IS 'Locations where the world can be expanded';
COMMENT ON TABLE world_boundaries IS 'Classification of location boundary types';

COMMENT ON COLUMN locations.boundary_type IS 'Type of world boundary: edge, frontier, internal, or isolated';
COMMENT ON COLUMN locations.expansion_priority IS 'Priority score for world expansion from this location';
COMMENT ON COLUMN locations.navigation_metadata IS 'Additional metadata for navigation algorithms';

COMMENT ON COLUMN navigation_paths.path_data IS 'JSON representation of the complete path including nodes and directions';
COMMENT ON COLUMN navigation_paths.criteria IS 'Pathfinding criteria used: shortest, safest, scenic, or fastest';

COMMENT ON COLUMN connection_requirements.requirements IS 'JSON object defining items, skills, or state requirements';

COMMENT ON COLUMN expansion_points.context IS 'Context information for generating new locations';
COMMENT ON COLUMN expansion_points.suggested_theme IS 'Suggested theme for the new location';