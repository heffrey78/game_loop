-- Object Generation System Database Schema
-- This migration creates tables for dynamic object generation, storage, and management

-- Object archetypes and templates
CREATE TABLE IF NOT EXISTS object_archetypes (
    archetype_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    typical_properties JSONB DEFAULT '{}',
    location_affinities JSONB DEFAULT '{}',
    interaction_templates JSONB DEFAULT '{}',
    cultural_variations JSONB DEFAULT '{}',
    rarity VARCHAR(20) DEFAULT 'common',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Object properties storage
CREATE TABLE IF NOT EXISTS object_properties (
    property_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    interactions JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_object_properties_object_id 
        FOREIGN KEY (object_id) REFERENCES world_objects(object_id) 
        ON DELETE CASCADE
);

-- Object generation history and metrics
CREATE TABLE IF NOT EXISTS object_generation_history (
    generation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID NOT NULL,
    generation_context JSONB DEFAULT '{}',
    generation_metadata JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    generation_purpose VARCHAR(50),
    location_theme VARCHAR(50),
    player_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_object_generation_object_id 
        FOREIGN KEY (object_id) REFERENCES world_objects(object_id) 
        ON DELETE CASCADE
);

-- Object placement information
CREATE TABLE IF NOT EXISTS object_placements (
    placement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID NOT NULL,
    location_id UUID NOT NULL,
    placement_data JSONB DEFAULT '{}',
    visibility_rules JSONB DEFAULT '{}',
    placement_type VARCHAR(20) DEFAULT 'floor',
    visibility VARCHAR(20) DEFAULT 'visible',
    accessibility VARCHAR(20) DEFAULT 'accessible',
    discovery_difficulty INTEGER DEFAULT 1,
    spatial_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_object_placements_object_id 
        FOREIGN KEY (object_id) REFERENCES world_objects(object_id) 
        ON DELETE CASCADE,
    CONSTRAINT fk_object_placements_location_id 
        FOREIGN KEY (location_id) REFERENCES locations(location_id) 
        ON DELETE CASCADE,
    CONSTRAINT check_discovery_difficulty 
        CHECK (discovery_difficulty >= 1 AND discovery_difficulty <= 10)
);

-- Object themes for consistency management
CREATE TABLE IF NOT EXISTS object_themes (
    theme_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    typical_materials JSONB DEFAULT '[]',
    common_object_types JSONB DEFAULT '[]',
    cultural_elements JSONB DEFAULT '{}',
    style_descriptors JSONB DEFAULT '[]',
    forbidden_elements JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Object search and discovery tracking
CREATE TABLE IF NOT EXISTS object_discoveries (
    discovery_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID NOT NULL,
    player_id UUID,
    location_id UUID NOT NULL,
    discovery_method VARCHAR(50),
    discovery_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    search_query TEXT,
    context_data JSONB DEFAULT '{}',
    CONSTRAINT fk_object_discoveries_object_id 
        FOREIGN KEY (object_id) REFERENCES world_objects(object_id) 
        ON DELETE CASCADE,
    CONSTRAINT fk_object_discoveries_location_id 
        FOREIGN KEY (location_id) REFERENCES locations(location_id) 
        ON DELETE CASCADE
);

-- Add embedding column to world_objects if not exists
-- This supports semantic search capabilities
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'world_objects' AND column_name = 'embedding'
    ) THEN
        ALTER TABLE world_objects ADD COLUMN embedding vector(1536);
    END IF;
END $$;

-- Add object_type column to world_objects for easier querying
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'world_objects' AND column_name = 'object_type'
    ) THEN
        ALTER TABLE world_objects ADD COLUMN object_type VARCHAR(50);
    END IF;
END $$;

-- Add value column to world_objects for economic modeling
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'world_objects' AND column_name = 'value'
    ) THEN
        ALTER TABLE world_objects ADD COLUMN value INTEGER DEFAULT 0;
    END IF;
END $$;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_object_properties_object_id ON object_properties(object_id);
CREATE INDEX IF NOT EXISTS idx_object_generation_history_object_id ON object_generation_history(object_id);
CREATE INDEX IF NOT EXISTS idx_object_generation_history_purpose ON object_generation_history(generation_purpose);
CREATE INDEX IF NOT EXISTS idx_object_generation_history_theme ON object_generation_history(location_theme);
CREATE INDEX IF NOT EXISTS idx_object_placements_object_id ON object_placements(object_id);
CREATE INDEX IF NOT EXISTS idx_object_placements_location_id ON object_placements(location_id);
CREATE INDEX IF NOT EXISTS idx_object_placements_visibility ON object_placements(visibility);
CREATE INDEX IF NOT EXISTS idx_object_discoveries_object_id ON object_discoveries(object_id);
CREATE INDEX IF NOT EXISTS idx_object_discoveries_player_id ON object_discoveries(player_id);
CREATE INDEX IF NOT EXISTS idx_world_objects_object_type ON world_objects(object_type);
CREATE INDEX IF NOT EXISTS idx_world_objects_value ON world_objects(value);

-- Vector similarity search index (requires pgvector extension)
-- This enables fast semantic search of objects
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS idx_world_objects_embedding 
        ON world_objects USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- pgvector extension not available, skip vector index
        NULL;
END $$;

-- JSONB indexes for faster property searches
CREATE INDEX IF NOT EXISTS idx_object_properties_properties_gin ON object_properties USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_object_properties_interactions_gin ON object_properties USING gin(interactions);
CREATE INDEX IF NOT EXISTS idx_object_generation_metadata_gin ON object_generation_history USING gin(generation_metadata);
CREATE INDEX IF NOT EXISTS idx_object_placements_placement_data_gin ON object_placements USING gin(placement_data);

-- Functional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_object_properties_object_type 
ON object_properties((properties->>'object_type'));

CREATE INDEX IF NOT EXISTS idx_object_properties_material 
ON object_properties((properties->>'material'));

CREATE INDEX IF NOT EXISTS idx_object_properties_size 
ON object_properties((properties->>'size'));

CREATE INDEX IF NOT EXISTS idx_object_properties_portable 
ON object_properties((interactions->>'portable'));

-- Insert default object archetypes
INSERT INTO object_archetypes (name, description, typical_properties, location_affinities, interaction_templates, rarity)
VALUES 
    ('sword', 'A bladed weapon for combat and ceremony', 
     '{"object_type": "weapon", "material": "iron", "size": "medium", "weight": "heavy", "special_properties": ["sharp", "balanced", "weapon"]}',
     '{"Village": 0.6, "City": 0.8, "Castle": 0.9, "Battlefield": 0.95}',
     '{"combat": ["attack", "parry", "flourish"], "utility": ["cut", "pry"]}',
     'uncommon'),
     
    ('hammer', 'A tool for building and crafting',
     '{"object_type": "tool", "material": "iron_and_wood", "size": "medium", "weight": "heavy", "special_properties": ["blunt", "tool", "crafting"]}',
     '{"Village": 0.8, "City": 0.7, "Forge": 0.95, "Workshop": 0.9}',
     '{"crafting": ["hammer", "forge", "shape"], "combat": ["bludgeon"]}',
     'common'),
     
    ('chest', 'A storage container for valuables',
     '{"object_type": "container", "material": "wood", "size": "large", "weight": "heavy", "special_properties": ["storage", "lockable", "container"]}',
     '{"Village": 0.7, "City": 0.6, "Dungeon": 0.8, "Treasure_Room": 0.95}',
     '{"storage": ["open", "close", "lock", "unlock"], "utility": ["examine", "search"]}',
     'common'),
     
    ('book', 'A collection of written knowledge',
     '{"object_type": "knowledge", "material": "parchment_and_leather", "size": "small", "weight": "light", "special_properties": ["readable", "knowledge", "fragile"]}',
     '{"Library": 0.95, "Study": 0.9, "City": 0.7, "School": 0.85}',
     '{"knowledge": ["read", "study", "research"], "utility": ["examine", "carry"]}',
     'uncommon'),
     
    ('herb', 'A natural plant with medicinal or culinary properties',
     '{"object_type": "natural", "material": "plant", "size": "tiny", "weight": "light", "special_properties": ["medicinal", "consumable", "natural"]}',
     '{"Forest": 0.9, "Garden": 0.8, "Meadow": 0.85, "Wilderness": 0.7}',
     '{"healing": ["consume", "brew", "apply"], "utility": ["gather", "examine"]}',
     'common'),
     
    ('gem', 'A precious stone of value and beauty',
     '{"object_type": "treasure", "material": "crystal", "size": "tiny", "weight": "light", "special_properties": ["precious", "beautiful", "magical_conduit"], "value": 200}',
     '{"Mine": 0.7, "Cave": 0.6, "Treasure_Room": 0.9, "Jewelry_Shop": 0.8}',
     '{"treasure": ["appraise", "trade", "admire"], "magic": ["channel", "enhance"]}',
     'rare')
ON CONFLICT (name) DO NOTHING;

-- Insert default object themes
INSERT INTO object_themes (name, description, typical_materials, common_object_types, cultural_elements, style_descriptors, forbidden_elements)
VALUES 
    ('Village', 'Simple, practical objects suited for rural life',
     '["wood", "iron", "cloth", "leather", "ceramic"]',
     '["tool", "furniture", "container", "weapon", "clothing"]',
     '{"style": "rustic", "craftsmanship": "local", "decoration": "simple"}',
     '["weathered", "handmade", "practical", "sturdy", "well-used"]',
     '["luxury", "ornate", "magical", "exotic"]'),
     
    ('Forest', 'Natural objects and items suited for wilderness',
     '["wood", "stone", "bone", "plant", "hide"]',
     '["natural", "tool", "weapon", "container", "survival_gear"]',
     '{"style": "natural", "craftsmanship": "primitive", "decoration": "carved"}',
     '["organic", "rough", "primitive", "weathered", "natural"]',
     '["metal", "refined", "delicate", "manufactured"]'),
     
    ('City', 'Refined objects reflecting urban sophistication',
     '["steel", "brass", "silk", "marble", "glass"]',
     '["tool", "furniture", "art", "weapon", "luxury"]',
     '{"style": "refined", "craftsmanship": "professional", "decoration": "detailed"}',
     '["polished", "elegant", "sophisticated", "ornate", "quality"]',
     '["crude", "primitive", "temporary", "makeshift"]'),
     
    ('Dungeon', 'Ancient, mystical, or abandoned objects',
     '["stone", "ancient_metal", "crystal", "bone", "unknown"]',
     '["treasure", "trap", "relic", "weapon", "mystery"]',
     '{"style": "ancient", "craftsmanship": "mysterious", "decoration": "runic"}',
     '["ancient", "mysterious", "magical", "ominous", "forgotten"]',
     '["modern", "bright", "cheerful", "mundane"]')
ON CONFLICT (name) DO NOTHING;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers
DROP TRIGGER IF EXISTS update_object_archetypes_updated_at ON object_archetypes;
CREATE TRIGGER update_object_archetypes_updated_at
    BEFORE UPDATE ON object_archetypes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_object_properties_updated_at ON object_properties;
CREATE TRIGGER update_object_properties_updated_at
    BEFORE UPDATE ON object_properties
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_object_placements_updated_at ON object_placements;
CREATE TRIGGER update_object_placements_updated_at
    BEFORE UPDATE ON object_placements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO game_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO game_user;

-- Migration completion verification
DO $$
BEGIN
    RAISE NOTICE 'Object Generation System migration completed successfully';
    RAISE NOTICE 'Created tables: object_archetypes, object_properties, object_generation_history, object_placements, object_themes, object_discoveries';
    RAISE NOTICE 'Added columns to world_objects: embedding, object_type, value';
    RAISE NOTICE 'Created indexes for performance optimization';
    RAISE NOTICE 'Inserted default archetypes and themes';
END $$;