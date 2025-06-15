-- Migration: Enhanced Dynamic World Generation
-- Adds support for expansion depth tracking, placeholder connections, and improved metadata

-- Add expansion depth and generation metadata to locations table
ALTER TABLE locations 
ADD COLUMN IF NOT EXISTS expansion_depth INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS can_expand BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS generated_from_location_id UUID REFERENCES locations(id),
ADD COLUMN IF NOT EXISTS generation_metadata JSONB DEFAULT '{}';

-- Create index for expansion depth queries
CREATE INDEX IF NOT EXISTS idx_locations_expansion_depth ON locations(expansion_depth);
CREATE INDEX IF NOT EXISTS idx_locations_can_expand ON locations(can_expand);
CREATE INDEX IF NOT EXISTS idx_locations_generated_from ON locations(generated_from_location_id);

-- Add connection type to location_connections for placeholder support
ALTER TABLE location_connections 
ADD COLUMN IF NOT EXISTS connection_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS is_placeholder BOOLEAN DEFAULT false;

-- Create index for placeholder connections
CREATE INDEX IF NOT EXISTS idx_location_connections_placeholder ON location_connections(is_placeholder);

-- Update existing dynamic locations to have expansion capabilities
UPDATE locations 
SET can_expand = true, 
    expansion_depth = 1,
    generation_metadata = jsonb_build_object(
        'location_type', COALESCE(location_type, 'unknown'),
        'is_dynamic', is_dynamic,
        'can_expand', true
    )
WHERE is_dynamic = true;

-- Update existing static boundary locations to allow expansion
UPDATE locations 
SET can_expand = true,
    expansion_depth = 0,
    generation_metadata = jsonb_build_object(
        'location_type', 'boundary',
        'is_dynamic', false,
        'can_expand', true
    )
WHERE name IN ('Building Lobby', 'Emergency Stairwell', 'Underground Parking Garage', 
               'Office', 'Conference Room', 'Reception Area', 'Break Room', 'Storage Room');

-- Create a view for expansion analysis
CREATE OR REPLACE VIEW expansion_analysis AS
SELECT 
    l.id,
    l.name,
    l.expansion_depth,
    l.can_expand,
    l.is_dynamic,
    l.generated_from_location_id,
    source.name as generated_from_name,
    COUNT(lc.id) as connection_count,
    COUNT(CASE WHEN lc.is_placeholder THEN 1 END) as placeholder_count
FROM locations l
LEFT JOIN locations source ON l.generated_from_location_id = source.id
LEFT JOIN location_connections lc ON l.id = lc.from_location_id
GROUP BY l.id, l.name, l.expansion_depth, l.can_expand, l.is_dynamic, 
         l.generated_from_location_id, source.name;

-- Create function to clean up old placeholder connections
CREATE OR REPLACE FUNCTION cleanup_old_placeholders()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete placeholder connections older than 24 hours that haven't been used
    DELETE FROM location_connections 
    WHERE is_placeholder = true 
    AND to_location_id = '00000000-0000-0000-0000-000000000000'
    AND created_at < NOW() - INTERVAL '24 hours';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add comment explaining the migration
COMMENT ON COLUMN locations.expansion_depth IS 'Tracks how many levels deep this location is from the original world';
COMMENT ON COLUMN locations.can_expand IS 'Whether this location can generate new connected areas';
COMMENT ON COLUMN locations.generated_from_location_id IS 'The location this was generated from (if dynamic)';
COMMENT ON COLUMN locations.generation_metadata IS 'Metadata about the generation process and location characteristics';
COMMENT ON COLUMN location_connections.is_placeholder IS 'Whether this connection is a placeholder that triggers generation';
COMMENT ON COLUMN location_connections.connection_metadata IS 'Metadata about the connection type and generation context';