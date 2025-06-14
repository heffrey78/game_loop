-- 027_location_generation.sql
-- Location Generation System Migration
-- Creates tables for location themes, generation history, theme transitions, and caching

-- Location themes table
CREATE TABLE location_themes (
    theme_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    visual_elements JSONB DEFAULT '[]',
    atmosphere TEXT,
    typical_objects JSONB DEFAULT '[]',
    typical_npcs JSONB DEFAULT '[]',
    generation_parameters JSONB DEFAULT '{}',
    parent_theme_id UUID REFERENCES location_themes(theme_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Location generation history
CREATE TABLE location_generation_history (
    generation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location_id UUID NOT NULL REFERENCES locations(location_id),
    generation_context JSONB NOT NULL,
    generated_content JSONB NOT NULL,
    validation_result JSONB,
    generation_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Theme transitions for consistency
CREATE TABLE theme_transitions (
    transition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_theme_id UUID NOT NULL REFERENCES location_themes(theme_id),
    to_theme_id UUID NOT NULL REFERENCES location_themes(theme_id),
    transition_rules JSONB NOT NULL,
    compatibility_score FLOAT CHECK (compatibility_score >= 0 AND compatibility_score <= 1),
    is_valid BOOLEAN DEFAULT true,
    
    CONSTRAINT unique_theme_transition UNIQUE(from_theme_id, to_theme_id)
);

-- Location generation cache
CREATE TABLE location_generation_cache (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_hash VARCHAR(64) NOT NULL UNIQUE,
    generated_location JSONB NOT NULL,
    cache_expires_at TIMESTAMP NOT NULL,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add location generation metadata to locations table
ALTER TABLE locations
ADD COLUMN IF NOT EXISTS theme_id UUID REFERENCES location_themes(theme_id),
ADD COLUMN IF NOT EXISTS generation_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS last_generated_at TIMESTAMP;

-- Indexes for performance
CREATE INDEX idx_location_themes_parent ON location_themes(parent_theme_id);
CREATE INDEX idx_generation_history_location ON location_generation_history(location_id);
CREATE INDEX idx_theme_transitions_themes ON theme_transitions(from_theme_id, to_theme_id);
CREATE INDEX idx_generation_cache_hash ON location_generation_cache(context_hash);
CREATE INDEX idx_generation_cache_expires ON location_generation_cache(cache_expires_at);
CREATE INDEX idx_locations_theme ON locations(theme_id);

-- Insert default location themes
INSERT INTO location_themes (name, description, visual_elements, atmosphere, typical_objects, typical_npcs, generation_parameters) VALUES
(
    'Forest',
    'Dense woodland areas with towering trees, dappled sunlight, and natural paths',
    '["tall trees", "filtered sunlight", "leaf litter", "moss-covered rocks", "winding paths"]',
    'peaceful yet mysterious',
    '["fallen logs", "mushrooms", "berries", "bird nests", "stone markers"]',
    '["forest animals", "hermits", "rangers", "druids"]',
    '{"complexity": "medium", "danger_level": "low", "exploration_reward": "high"}'
),
(
    'Village',
    'Small settlements with rustic buildings, friendly inhabitants, and community spaces',
    '["thatched roofs", "cobblestone paths", "market squares", "gardens", "smoke from chimneys"]',
    'warm and welcoming',
    '["market stalls", "wells", "benches", "flower boxes", "signposts"]',
    '["merchants", "villagers", "children", "elders", "craftspeople"]',
    '{"complexity": "high", "danger_level": "very_low", "social_interaction": "high"}'
),
(
    'Mountain',
    'Rocky peaks and alpine environments with challenging terrain and scenic vistas',
    '["jagged peaks", "rocky outcrops", "sparse vegetation", "snow caps", "steep paths"]',
    'majestic and challenging',
    '["boulders", "ice formations", "mountain flowers", "caves", "observation points"]',
    '["mountain goats", "climbers", "hermits", "eagles"]',
    '{"complexity": "high", "danger_level": "medium", "physical_challenge": "high"}'
),
(
    'Ruins',
    'Ancient structures reclaimed by nature, holding secrets and mysteries',
    '["crumbling walls", "overgrown vegetation", "weathered stone", "broken columns", "hidden passages"]',
    'mysterious and haunting',
    '["ancient artifacts", "rubble", "inscriptions", "hidden chambers", "treasure chests"]',
    '["archaeologists", "treasure hunters", "spirits", "guardians"]',
    '{"complexity": "very_high", "danger_level": "medium", "mystery_level": "high"}'
),
(
    'Riverside',
    'Areas along flowing water with lush vegetation and peaceful sounds',
    '["flowing water", "riverside vegetation", "smooth stones", "gentle currents", "wildlife tracks"]',
    'tranquil and refreshing',
    '["fishing spots", "water plants", "smooth pebbles", "driftwood", "wildlife"]',
    '["fishermen", "water spirits", "travelers", "animals coming to drink"]',
    '{"complexity": "low", "danger_level": "very_low", "relaxation": "high"}'
);