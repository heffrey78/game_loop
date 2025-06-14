-- Migration 028: NPC Generation System
-- Implements database schema for comprehensive NPC generation system

-- NPC Archetypes and Templates
CREATE TABLE IF NOT EXISTS npc_archetypes (
    archetype_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    typical_traits JSONB DEFAULT '[]',
    typical_motivations JSONB DEFAULT '[]',
    speech_patterns JSONB DEFAULT '{}',
    location_affinities JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NPC Personalities
CREATE TABLE IF NOT EXISTS npc_personalities (
    personality_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    npc_id UUID NOT NULL REFERENCES npcs(npc_id) ON DELETE CASCADE,
    archetype_id UUID REFERENCES npc_archetypes(archetype_id),
    traits JSONB NOT NULL DEFAULT '[]',
    motivations JSONB NOT NULL DEFAULT '[]',
    fears JSONB NOT NULL DEFAULT '[]',
    speech_patterns JSONB NOT NULL DEFAULT '{}',
    relationship_tendencies JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NPC Knowledge Base
CREATE TABLE IF NOT EXISTS npc_knowledge (
    knowledge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    npc_id UUID NOT NULL REFERENCES npcs(npc_id) ON DELETE CASCADE,
    world_knowledge JSONB DEFAULT '{}',
    local_knowledge JSONB DEFAULT '{}',
    personal_history JSONB DEFAULT '[]',
    relationships JSONB DEFAULT '{}',
    secrets JSONB DEFAULT '[]',
    expertise_areas JSONB DEFAULT '[]',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NPC Dialogue States
CREATE TABLE IF NOT EXISTS npc_dialogue_states (
    dialogue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    npc_id UUID NOT NULL REFERENCES npcs(npc_id) ON DELETE CASCADE,
    current_mood VARCHAR(50) DEFAULT 'neutral',
    relationship_level FLOAT DEFAULT 0.0,
    conversation_history JSONB DEFAULT '[]',
    active_topics JSONB DEFAULT '[]',
    available_quests JSONB DEFAULT '[]',
    interaction_count INTEGER DEFAULT 0,
    last_interaction TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NPC Generation History
CREATE TABLE IF NOT EXISTS npc_generation_history (
    generation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    npc_id UUID NOT NULL REFERENCES npcs(npc_id) ON DELETE CASCADE,
    generation_context JSONB NOT NULL,
    generated_content JSONB NOT NULL,
    validation_result JSONB,
    generation_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add embedding support to existing NPCs table
ALTER TABLE npcs ADD COLUMN IF NOT EXISTS embedding_vector vector(768);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS npcs_embedding_idx ON npcs USING ivfflat (embedding_vector vector_cosine_ops);

-- Update existing NPCs table with generation metadata
ALTER TABLE npcs ADD COLUMN IF NOT EXISTS generation_metadata JSONB DEFAULT '{}';
ALTER TABLE npcs ADD COLUMN IF NOT EXISTS archetype VARCHAR(100);
ALTER TABLE npcs ADD COLUMN IF NOT EXISTS last_generated_at TIMESTAMP;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS npc_personalities_npc_id_idx ON npc_personalities(npc_id);
CREATE INDEX IF NOT EXISTS npc_personalities_archetype_id_idx ON npc_personalities(archetype_id);
CREATE INDEX IF NOT EXISTS npc_knowledge_npc_id_idx ON npc_knowledge(npc_id);
CREATE INDEX IF NOT EXISTS npc_dialogue_states_npc_id_idx ON npc_dialogue_states(npc_id);
CREATE INDEX IF NOT EXISTS npc_dialogue_states_last_interaction_idx ON npc_dialogue_states(last_interaction);
CREATE INDEX IF NOT EXISTS npc_generation_history_npc_id_idx ON npc_generation_history(npc_id);

-- Create indexes for archetype queries
CREATE INDEX IF NOT EXISTS npcs_archetype_idx ON npcs(archetype);

-- Create GIN indexes for JSONB queries
CREATE INDEX IF NOT EXISTS npc_personalities_traits_gin_idx ON npc_personalities USING GIN (traits);
CREATE INDEX IF NOT EXISTS npc_knowledge_expertise_gin_idx ON npc_knowledge USING GIN (expertise_areas);
CREATE INDEX IF NOT EXISTS npc_dialogue_active_topics_gin_idx ON npc_dialogue_states USING GIN (active_topics);

-- Insert default archetypes
INSERT INTO npc_archetypes (name, description, typical_traits, typical_motivations, speech_patterns, location_affinities)
VALUES 
    ('merchant', 'A trader who buys and sells goods', 
     '["persuasive", "business-minded", "social"]', 
     '["profit", "reputation", "trade_routes"]',
     '{"formality": "polite", "verbosity": "moderate"}',
     '{"Village": 0.9, "City": 0.8, "Town": 0.7, "Crossroads": 0.6, "Forest": 0.2}'),
    
    ('guard', 'A protector of people and places',
     '["vigilant", "dutiful", "protective"]',
     '["duty", "safety", "order"]',
     '{"formality": "formal", "verbosity": "concise"}',
     '{"City": 0.9, "Town": 0.8, "Village": 0.7, "Castle": 0.9, "Forest": 0.3}'),
    
    ('scholar', 'A learned person devoted to study and research',
     '["knowledgeable", "curious", "analytical"]',
     '["knowledge", "discovery", "teaching"]',
     '{"formality": "formal", "verbosity": "verbose"}',
     '{"Library": 0.9, "Academy": 0.9, "City": 0.6, "Tower": 0.8, "Forest": 0.4}'),
    
    ('hermit', 'A solitary person who lives apart from society',
     '["wise", "reclusive", "self-sufficient"]',
     '["solitude", "wisdom", "nature"]',
     '{"formality": "casual", "verbosity": "cryptic"}',
     '{"Forest": 0.9, "Mountain": 0.8, "Cave": 0.7, "Wilderness": 0.9, "City": 0.1}'),
    
    ('innkeeper', 'A host who provides food, drink, and lodging',
     '["hospitable", "social", "practical"]',
     '["hospitality", "community", "stories"]',
     '{"formality": "casual", "verbosity": "moderate"}',
     '{"Inn": 0.9, "Tavern": 0.9, "Village": 0.7, "Town": 0.8, "Forest": 0.2}'),
    
    ('artisan', 'A skilled craftsperson who creates goods',
     '["skilled", "creative", "dedicated"]',
     '["craftsmanship", "beauty", "utility"]',
     '{"formality": "casual", "verbosity": "moderate"}',
     '{"Workshop": 0.9, "Village": 0.7, "Town": 0.8, "City": 0.6, "Forest": 0.3}'),
    
    ('wanderer', 'A traveler who roams from place to place',
     '["adventurous", "experienced", "independent"]',
     '["exploration", "freedom", "stories"]',
     '{"formality": "casual", "verbosity": "storytelling"}',
     '{"Crossroads": 0.8, "Forest": 0.7, "Mountain": 0.6, "Path": 0.9, "City": 0.4}')
ON CONFLICT (name) DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for npc_dialogue_states updated_at
DROP TRIGGER IF EXISTS update_npc_dialogue_states_updated_at ON npc_dialogue_states;
CREATE TRIGGER update_npc_dialogue_states_updated_at
    BEFORE UPDATE ON npc_dialogue_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to automatically update knowledge last_updated
CREATE OR REPLACE FUNCTION update_knowledge_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for npc_knowledge last_updated
DROP TRIGGER IF EXISTS update_npc_knowledge_last_updated ON npc_knowledge;
CREATE TRIGGER update_npc_knowledge_last_updated
    BEFORE UPDATE ON npc_knowledge
    FOR EACH ROW
    EXECUTE FUNCTION update_knowledge_last_updated();

-- Create view for complete NPC data
CREATE OR REPLACE VIEW npc_complete_data AS
SELECT 
    n.npc_id,
    n.name,
    n.description,
    n.dialogue,
    n.archetype,
    n.generation_metadata,
    n.last_generated_at,
    n.embedding_vector,
    p.personality_id,
    p.traits,
    p.motivations,
    p.fears,
    p.speech_patterns,
    p.relationship_tendencies,
    k.knowledge_id,
    k.world_knowledge,
    k.local_knowledge,
    k.personal_history,
    k.relationships,
    k.secrets,
    k.expertise_areas,
    k.last_updated as knowledge_last_updated,
    d.dialogue_id,
    d.current_mood,
    d.relationship_level,
    d.conversation_history,
    d.active_topics,
    d.available_quests,
    d.interaction_count,
    d.last_interaction,
    a.name as archetype_name,
    a.description as archetype_description
FROM npcs n
LEFT JOIN npc_personalities p ON n.npc_id = p.npc_id
LEFT JOIN npc_knowledge k ON n.npc_id = k.npc_id
LEFT JOIN npc_dialogue_states d ON n.npc_id = d.npc_id
LEFT JOIN npc_archetypes a ON p.archetype_id = a.archetype_id OR n.archetype = a.name;

-- Grant appropriate permissions (adjust as needed for your user setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON npc_archetypes TO game_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON npc_personalities TO game_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON npc_knowledge TO game_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON npc_dialogue_states TO game_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON npc_generation_history TO game_user;
-- GRANT SELECT ON npc_complete_data TO game_user;