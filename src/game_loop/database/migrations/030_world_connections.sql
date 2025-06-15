-- Migration 030: World Connection Management System
-- Creates tables and indexes for world connections, connection metadata, and connectivity graph

-- Connection management tables
CREATE TABLE IF NOT EXISTS world_connections (
    connection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_location_id UUID NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    target_location_id UUID NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    connection_type VARCHAR(50) NOT NULL,
    difficulty INTEGER NOT NULL CHECK (difficulty >= 1 AND difficulty <= 10),
    travel_time INTEGER NOT NULL CHECK (travel_time > 0),
    description TEXT NOT NULL,
    visibility VARCHAR(20) NOT NULL DEFAULT 'visible' CHECK (visibility IN ('visible', 'hidden', 'secret', 'partially_hidden')),
    reversible BOOLEAN NOT NULL DEFAULT TRUE,
    requirements JSONB DEFAULT '[]',
    condition_flags JSONB DEFAULT '{}',
    special_features JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    embedding_vector vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Prevent duplicate connections between same locations
    CONSTRAINT unique_connection_pair UNIQUE (source_location_id, target_location_id),
    -- Prevent self-connections
    CONSTRAINT no_self_connection CHECK (source_location_id != target_location_id)
);

-- Connection generation metadata
CREATE TABLE IF NOT EXISTS connection_generation_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES world_connections(connection_id) ON DELETE CASCADE,
    generation_purpose VARCHAR(50) NOT NULL CHECK (generation_purpose IN ('expand_world', 'quest_path', 'exploration', 'narrative_enhancement', 'player_request')),
    generation_context JSONB NOT NULL,
    validation_results JSONB,
    quality_scores JSONB,
    generation_time_ms INTEGER,
    llm_model_used VARCHAR(100),
    template_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- World connectivity graph for pathfinding and analysis
CREATE TABLE IF NOT EXISTS world_connectivity_graph (
    graph_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location_id UUID NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    connected_location_id UUID NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    connection_id UUID NOT NULL REFERENCES world_connections(connection_id) ON DELETE CASCADE,
    path_distance INTEGER DEFAULT 1,
    traversal_cost INTEGER DEFAULT 1,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(location_id, connected_location_id, connection_id),
    -- Prevent self-connections in graph
    CONSTRAINT graph_no_self_connection CHECK (location_id != connected_location_id)
);

-- Connection archetypes for theme management
CREATE TABLE IF NOT EXISTS connection_archetypes (
    archetype_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    connection_type VARCHAR(50) NOT NULL,
    typical_difficulty INTEGER NOT NULL CHECK (typical_difficulty >= 1 AND typical_difficulty <= 10),
    typical_travel_time INTEGER NOT NULL CHECK (typical_travel_time > 0),
    terrain_affinities JSONB DEFAULT '{}',
    theme_compatibility JSONB DEFAULT '{}',
    generation_templates JSONB DEFAULT '{}',
    rarity VARCHAR(20) DEFAULT 'common' CHECK (rarity IN ('common', 'uncommon', 'rare', 'legendary')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Connection validation history for quality tracking
CREATE TABLE IF NOT EXISTS connection_validation_history (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES world_connections(connection_id) ON DELETE CASCADE,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_valid BOOLEAN NOT NULL,
    validation_errors JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    consistency_score FLOAT CHECK (consistency_score >= 0.0 AND consistency_score <= 1.0),
    logical_soundness FLOAT CHECK (logical_soundness >= 0.0 AND logical_soundness <= 1.0),
    terrain_compatibility FLOAT CHECK (terrain_compatibility >= 0.0 AND terrain_compatibility <= 1.0),
    validator_version VARCHAR(20)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_world_connections_source_location ON world_connections(source_location_id);
CREATE INDEX IF NOT EXISTS idx_world_connections_target_location ON world_connections(target_location_id);
CREATE INDEX IF NOT EXISTS idx_world_connections_type ON world_connections(connection_type);
CREATE INDEX IF NOT EXISTS idx_world_connections_difficulty ON world_connections(difficulty);
CREATE INDEX IF NOT EXISTS idx_world_connections_visibility ON world_connections(visibility);
CREATE INDEX IF NOT EXISTS idx_world_connections_reversible ON world_connections(reversible);
CREATE INDEX IF NOT EXISTS idx_world_connections_created_at ON world_connections(created_at);

-- Vector similarity index for embeddings
CREATE INDEX IF NOT EXISTS idx_world_connections_embedding ON world_connections USING ivfflat (embedding_vector vector_cosine_ops);

-- Generation metadata indexes
CREATE INDEX IF NOT EXISTS idx_connection_generation_metadata_connection ON connection_generation_metadata(connection_id);
CREATE INDEX IF NOT EXISTS idx_connection_generation_metadata_purpose ON connection_generation_metadata(generation_purpose);
CREATE INDEX IF NOT EXISTS idx_connection_generation_metadata_created_at ON connection_generation_metadata(created_at);

-- Connectivity graph indexes
CREATE INDEX IF NOT EXISTS idx_connectivity_graph_location ON world_connectivity_graph(location_id);
CREATE INDEX IF NOT EXISTS idx_connectivity_graph_connected_location ON world_connectivity_graph(connected_location_id);
CREATE INDEX IF NOT EXISTS idx_connectivity_graph_connection ON world_connectivity_graph(connection_id);
CREATE INDEX IF NOT EXISTS idx_connectivity_graph_path_distance ON world_connectivity_graph(path_distance);
CREATE INDEX IF NOT EXISTS idx_connectivity_graph_traversal_cost ON world_connectivity_graph(traversal_cost);

-- Archetype indexes
CREATE INDEX IF NOT EXISTS idx_connection_archetypes_type ON connection_archetypes(connection_type);
CREATE INDEX IF NOT EXISTS idx_connection_archetypes_rarity ON connection_archetypes(rarity);

-- Validation history indexes
CREATE INDEX IF NOT EXISTS idx_connection_validation_history_connection ON connection_validation_history(connection_id);
CREATE INDEX IF NOT EXISTS idx_connection_validation_history_timestamp ON connection_validation_history(validation_timestamp);
CREATE INDEX IF NOT EXISTS idx_connection_validation_history_is_valid ON connection_validation_history(is_valid);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_world_connections_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_world_connections_updated_at
    BEFORE UPDATE ON world_connections
    FOR EACH ROW
    EXECUTE FUNCTION update_world_connections_updated_at();

CREATE TRIGGER trigger_connection_archetypes_updated_at
    BEFORE UPDATE ON connection_archetypes
    FOR EACH ROW
    EXECUTE FUNCTION update_world_connections_updated_at();

-- Function to automatically maintain connectivity graph
CREATE OR REPLACE FUNCTION maintain_connectivity_graph()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Add forward connection to graph
        INSERT INTO world_connectivity_graph (location_id, connected_location_id, connection_id, path_distance, traversal_cost)
        VALUES (NEW.source_location_id, NEW.target_location_id, NEW.connection_id, 1, NEW.travel_time);

        -- Add reverse connection if reversible
        IF NEW.reversible THEN
            INSERT INTO world_connectivity_graph (location_id, connected_location_id, connection_id, path_distance, traversal_cost)
            VALUES (NEW.target_location_id, NEW.source_location_id, NEW.connection_id, 1, NEW.travel_time);
        END IF;

        RETURN NEW;
    END IF;

    IF TG_OP = 'UPDATE' THEN
        -- Update existing graph entries
        UPDATE world_connectivity_graph
        SET traversal_cost = NEW.travel_time, last_updated = NOW()
        WHERE connection_id = NEW.connection_id;

        -- Handle reversibility changes
        IF OLD.reversible AND NOT NEW.reversible THEN
            -- Remove reverse connection
            DELETE FROM world_connectivity_graph
            WHERE connection_id = NEW.connection_id
            AND location_id = NEW.target_location_id
            AND connected_location_id = NEW.source_location_id;
        ELSIF NOT OLD.reversible AND NEW.reversible THEN
            -- Add reverse connection
            INSERT INTO world_connectivity_graph (location_id, connected_location_id, connection_id, path_distance, traversal_cost)
            VALUES (NEW.target_location_id, NEW.source_location_id, NEW.connection_id, 1, NEW.travel_time)
            ON CONFLICT DO NOTHING;
        END IF;

        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        -- Remove all graph entries for this connection
        DELETE FROM world_connectivity_graph WHERE connection_id = OLD.connection_id;
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_maintain_connectivity_graph
    AFTER INSERT OR UPDATE OR DELETE ON world_connections
    FOR EACH ROW
    EXECUTE FUNCTION maintain_connectivity_graph();

-- View for connection analytics
CREATE OR REPLACE VIEW connection_analytics AS
SELECT
    c.connection_type,
    COUNT(*) as total_connections,
    AVG(c.difficulty) as avg_difficulty,
    AVG(c.travel_time) as avg_travel_time,
    COUNT(CASE WHEN c.reversible THEN 1 END) as reversible_count,
    COUNT(CASE WHEN c.visibility = 'hidden' THEN 1 END) as hidden_count,
    COUNT(CASE WHEN jsonb_array_length(c.requirements) > 0 THEN 1 END) as with_requirements
FROM world_connections c
GROUP BY c.connection_type
ORDER BY total_connections DESC;

-- View for location connectivity summary
CREATE OR REPLACE VIEW location_connectivity AS
SELECT
    l.id as location_id,
    l.name as location_name,
    COUNT(DISTINCT wc1.connection_id) as outgoing_connections,
    COUNT(DISTINCT wc2.connection_id) as incoming_connections,
    COUNT(DISTINCT
        CASE WHEN wc1.connection_id IS NOT NULL OR wc2.connection_id IS NOT NULL
        THEN COALESCE(wc1.connection_id, wc2.connection_id) END
    ) as total_unique_connections
FROM locations l
LEFT JOIN world_connections wc1 ON l.id = wc1.source_location_id
LEFT JOIN world_connections wc2 ON l.id = wc2.target_location_id
GROUP BY l.id, l.name
ORDER BY total_unique_connections DESC;

-- Insert default connection archetypes
INSERT INTO connection_archetypes (name, description, connection_type, typical_difficulty, typical_travel_time, terrain_affinities, theme_compatibility, rarity) VALUES
('Basic Passage', 'A simple passage connecting two areas', 'passage', 2, 30,
 '{"underground": 0.9, "indoor": 0.8, "mountain": 0.7}',
 '{"Dungeon": 0.9, "Cave": 0.9, "Castle": 0.7}', 'common'),

('Stone Bridge', 'A sturdy bridge spanning between elevated areas', 'bridge', 3, 45,
 '{"river": 0.9, "canyon": 0.9, "mountain": 0.8}',
 '{"Mountain": 0.9, "Forest": 0.7, "Village": 0.8}', 'common'),

('Magical Portal', 'A magical portal providing instant travel', 'portal', 1, 5,
 '{"magical": 0.9, "tower": 0.8, "shrine": 0.8}',
 '{"Magical": 0.9, "Tower": 0.9, "Shrine": 0.8}', 'rare'),

('Forest Path', 'A natural path worn by travelers', 'path', 2, 60,
 '{"forest": 0.9, "grassland": 0.9, "wilderness": 0.8}',
 '{"Forest": 0.9, "Mountain": 0.8, "Wilderness": 0.9}', 'common'),

('Underground Tunnel', 'A tunnel burrowed through earth and stone', 'tunnel', 4, 90,
 '{"underground": 0.9, "mountain": 0.8, "cave": 0.9}',
 '{"Underground": 0.9, "Cave": 0.9, "Mine": 0.9}', 'uncommon'),

('Paved Road', 'A constructed road for easy travel', 'road', 1, 45,
 '{"urban": 0.9, "grassland": 0.8, "plains": 0.9}',
 '{"City": 0.9, "Town": 0.9, "Village": 0.9}', 'common')
ON CONFLICT (name) DO NOTHING;

-- Create indexes on JSONB fields for faster queries
CREATE INDEX IF NOT EXISTS idx_world_connections_requirements_gin ON world_connections USING GIN (requirements);
CREATE INDEX IF NOT EXISTS idx_world_connections_condition_flags_gin ON world_connections USING GIN (condition_flags);
CREATE INDEX IF NOT EXISTS idx_world_connections_special_features_gin ON world_connections USING GIN (special_features);
CREATE INDEX IF NOT EXISTS idx_world_connections_metadata_gin ON world_connections USING GIN (metadata);

-- Function to find shortest path between two locations
CREATE OR REPLACE FUNCTION find_shortest_path(start_location UUID, end_location UUID)
RETURNS TABLE(location_id UUID, connection_id UUID, step_number INTEGER, total_cost INTEGER) AS $$
DECLARE
    current_location UUID := start_location;
    step_count INTEGER := 0;
    total_traversal_cost INTEGER := 0;
BEGIN
    -- Simple pathfinding implementation (in practice, would use more sophisticated algorithm)
    -- This is a basic example - real implementation would use Dijkstra's or A* algorithm

    -- Direct connection check
    FOR location_id, connection_id IN
        SELECT wcg.connected_location_id, wcg.connection_id
        FROM world_connectivity_graph wcg
        WHERE wcg.location_id = start_location AND wcg.connected_location_id = end_location
        LIMIT 1
    LOOP
        step_count := step_count + 1;
        SELECT wcg.traversal_cost INTO total_traversal_cost
        FROM world_connectivity_graph wcg
        WHERE wcg.connection_id = find_shortest_path.connection_id;

        RETURN QUERY SELECT location_id, connection_id, step_count, total_traversal_cost;
        RETURN;
    END LOOP;

    -- If no direct connection found, return empty result
    -- (Real implementation would find multi-hop paths)
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Function to get connection statistics
CREATE OR REPLACE FUNCTION get_connection_statistics()
RETURNS TABLE(
    total_connections BIGINT,
    connection_types JSONB,
    avg_difficulty NUMERIC,
    avg_travel_time NUMERIC,
    reversible_percentage NUMERIC,
    hidden_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_connections,
        jsonb_object_agg(connection_type, type_count) as connection_types,
        AVG(difficulty) as avg_difficulty,
        AVG(travel_time) as avg_travel_time,
        (COUNT(CASE WHEN reversible THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC * 100) as reversible_percentage,
        (COUNT(CASE WHEN visibility = 'hidden' THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC * 100) as hidden_percentage
    FROM (
        SELECT
            connection_type,
            COUNT(*) as type_count,
            difficulty,
            travel_time,
            reversible,
            visibility
        FROM world_connections
        GROUP BY connection_type, difficulty, travel_time, reversible, visibility
    ) stats
    GROUP BY ();
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE world_connections IS 'Primary table storing all connections between locations in the world';
COMMENT ON TABLE connection_generation_metadata IS 'Metadata about how connections were generated for analysis and debugging';
COMMENT ON TABLE world_connectivity_graph IS 'Graph representation of world connectivity for pathfinding algorithms';
COMMENT ON TABLE connection_archetypes IS 'Template definitions for different types of connections';
COMMENT ON TABLE connection_validation_history IS 'History of connection validation results for quality tracking';

COMMENT ON COLUMN world_connections.embedding_vector IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN world_connections.requirements IS 'JSON array of conditions required to use this connection';
COMMENT ON COLUMN world_connections.condition_flags IS 'JSON object of dynamic conditions affecting the connection';
COMMENT ON COLUMN world_connections.special_features IS 'JSON array of special features or properties of the connection';

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON world_connections TO game_loop_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON connection_generation_metadata TO game_loop_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON world_connectivity_graph TO game_loop_app;
-- GRANT SELECT ON connection_analytics TO game_loop_app;
-- GRANT SELECT ON location_connectivity TO game_loop_app;
