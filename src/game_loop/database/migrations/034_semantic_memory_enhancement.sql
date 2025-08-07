-- Migration 034: Semantic Memory Enhancement
-- Extends conversation system with semantic memory capabilities including vector embeddings,
-- memory confidence tracking, emotional weighting, and NPC memory personality configuration

-- Enable pgvector extension for vector similarity operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Add semantic memory fields to conversation_exchanges table
ALTER TABLE conversation_exchanges 
ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(3,2) DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS emotional_weight DECIMAL(3,2) DEFAULT 0.5,
ADD COLUMN IF NOT EXISTS trust_level_required DECIMAL(3,2) DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS memory_embedding VECTOR(384); -- Standard embedding dimension

-- Add constraints for semantic memory fields
ALTER TABLE conversation_exchanges 
ADD CONSTRAINT IF NOT EXISTS chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
ADD CONSTRAINT IF NOT EXISTS chk_emotional_weight CHECK (emotional_weight >= 0.0 AND emotional_weight <= 1.0),
ADD CONSTRAINT IF NOT EXISTS chk_trust_level_required CHECK (trust_level_required >= 0.0 AND trust_level_required <= 1.0);

-- Extend exchange_metadata to support emotional context
COMMENT ON COLUMN conversation_exchanges.confidence_score IS 'Memory confidence based on age, access patterns, and emotional significance (0.0-1.0)';
COMMENT ON COLUMN conversation_exchanges.emotional_weight IS 'Emotional significance of this exchange (0.0-1.0)';
COMMENT ON COLUMN conversation_exchanges.trust_level_required IS 'Minimum relationship trust required to access this memory (0.0-1.0)';
COMMENT ON COLUMN conversation_exchanges.memory_embedding IS 'Vector embedding for semantic similarity search (384 dimensions)';

-- Create memory_embeddings table for optimized vector operations
CREATE TABLE memory_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id UUID NOT NULL REFERENCES conversation_exchanges(exchange_id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create memory_access_log table for confidence degradation tracking
CREATE TABLE memory_access_log (
    access_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id UUID NOT NULL REFERENCES conversation_exchanges(exchange_id) ON DELETE CASCADE,
    accessed_by UUID NOT NULL, -- NPC ID who accessed the memory
    access_context JSONB NOT NULL DEFAULT '{}', -- Context of memory access
    confidence_at_access DECIMAL(3,2) NOT NULL,
    access_type VARCHAR(50) NOT NULL DEFAULT 'retrieval',
    accessed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_access_confidence CHECK (confidence_at_access >= 0.0 AND confidence_at_access <= 1.0),
    CONSTRAINT chk_access_type CHECK (access_type IN ('retrieval', 'reference', 'update', 'decay_calculation'))
);

-- Create memory_personality_config table for individual NPC memory characteristics
CREATE TABLE memory_personality_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id) ON DELETE CASCADE,
    decay_rate_modifier DECIMAL(3,2) NOT NULL DEFAULT 1.0, -- Multiplier for memory decay rate
    emotional_sensitivity DECIMAL(3,2) NOT NULL DEFAULT 1.0, -- Sensitivity to emotional content
    detail_retention_strength DECIMAL(3,2) NOT NULL DEFAULT 0.5, -- Better at remembering details vs names
    name_retention_strength DECIMAL(3,2) NOT NULL DEFAULT 0.5, -- Better at remembering names vs details  
    uncertainty_threshold DECIMAL(3,2) NOT NULL DEFAULT 0.3, -- Below this confidence, express uncertainty
    max_memory_capacity INTEGER NOT NULL DEFAULT 10000, -- Maximum memories to retain
    memory_clustering_preference DECIMAL(3,2) NOT NULL DEFAULT 0.5, -- Tendency to cluster related memories
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints for personality modifiers
    CONSTRAINT chk_decay_rate_modifier CHECK (decay_rate_modifier > 0.0 AND decay_rate_modifier <= 5.0),
    CONSTRAINT chk_emotional_sensitivity CHECK (emotional_sensitivity >= 0.0 AND emotional_sensitivity <= 2.0),
    CONSTRAINT chk_detail_retention CHECK (detail_retention_strength >= 0.0 AND detail_retention_strength <= 1.0),
    CONSTRAINT chk_name_retention CHECK (name_retention_strength >= 0.0 AND name_retention_strength <= 1.0),
    CONSTRAINT chk_uncertainty_threshold CHECK (uncertainty_threshold >= 0.0 AND uncertainty_threshold <= 1.0),
    CONSTRAINT chk_memory_clustering CHECK (memory_clustering_preference >= 0.0 AND memory_clustering_preference <= 1.0),
    
    -- Unique constraint - one config per NPC
    CONSTRAINT unique_npc_memory_config UNIQUE (npc_id)
);

-- Create emotional_context table for mood-based memory filtering
CREATE TABLE emotional_context (
    context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id UUID NOT NULL REFERENCES conversation_exchanges(exchange_id) ON DELETE CASCADE,
    sentiment_score DECIMAL(3,2) NOT NULL DEFAULT 0.0, -- -1.0 (negative) to 1.0 (positive)
    emotional_keywords TEXT[] DEFAULT '{}', -- Keywords that triggered emotional analysis
    participant_emotions JSONB NOT NULL DEFAULT '{}', -- Emotions of conversation participants
    emotional_intensity DECIMAL(3,2) NOT NULL DEFAULT 0.0, -- Overall emotional intensity (0.0-1.0)
    relationship_impact_score DECIMAL(3,2) NOT NULL DEFAULT 0.0, -- Impact on relationship (0.0-1.0)
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_sentiment_score CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    CONSTRAINT chk_emotional_intensity CHECK (emotional_intensity >= 0.0 AND emotional_intensity <= 1.0),
    CONSTRAINT chk_relationship_impact CHECK (relationship_impact_score >= 0.0 AND relationship_impact_score <= 1.0)
);

-- Create performance indexes for semantic memory operations

-- Indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_embedding ON memory_embeddings 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_conversation_exchanges_embedding ON conversation_exchanges 
USING ivfflat (memory_embedding vector_cosine_ops) WITH (lists = 100);

-- Indexes for memory confidence and access patterns
CREATE INDEX idx_conversation_exchanges_confidence ON conversation_exchanges(confidence_score);
CREATE INDEX idx_conversation_exchanges_emotional_weight ON conversation_exchanges(emotional_weight);
CREATE INDEX idx_conversation_exchanges_last_accessed ON conversation_exchanges(last_accessed);
CREATE INDEX idx_conversation_exchanges_access_count ON conversation_exchanges(access_count);

-- Composite indexes for memory retrieval queries
CREATE INDEX idx_exchanges_confidence_emotional ON conversation_exchanges(confidence_score, emotional_weight);
CREATE INDEX idx_exchanges_speaker_confidence ON conversation_exchanges(speaker_id, confidence_score);
CREATE INDEX idx_exchanges_conversation_confidence ON conversation_exchanges(conversation_id, confidence_score DESC);

-- Indexes for memory access tracking
CREATE INDEX idx_memory_access_log_exchange ON memory_access_log(exchange_id);
CREATE INDEX idx_memory_access_log_accessed_by ON memory_access_log(accessed_by);
CREATE INDEX idx_memory_access_log_accessed_at ON memory_access_log(accessed_at);

-- Indexes for emotional context
CREATE INDEX idx_emotional_context_exchange ON emotional_context(exchange_id);
CREATE INDEX idx_emotional_context_sentiment ON emotional_context(sentiment_score);
CREATE INDEX idx_emotional_context_intensity ON emotional_context(emotional_intensity);

-- Indexes for memory personality config
CREATE INDEX idx_memory_personality_npc ON memory_personality_config(npc_id);

-- Create functions for memory confidence calculation
CREATE OR REPLACE FUNCTION calculate_memory_confidence(
    base_confidence DECIMAL,
    age_days DECIMAL,
    emotional_weight DECIMAL,
    access_count INTEGER,
    decay_rate_modifier DECIMAL DEFAULT 1.0
) RETURNS DECIMAL AS $$
DECLARE
    emotional_amplifier DECIMAL;
    access_bonus DECIMAL;
    age_decay DECIMAL;
    final_confidence DECIMAL;
BEGIN
    -- Emotional amplifier: emotional memories decay slower
    emotional_amplifier := 1.0 + (emotional_weight * 2.0); -- Max 3x retention for highly emotional
    
    -- Access bonus: frequently accessed memories stay stronger
    access_bonus := LEAST(0.2, access_count * 0.01); -- Max 20% bonus, diminishing returns
    
    -- Age-based exponential decay
    age_decay := EXP(-0.1 * decay_rate_modifier * age_days / emotional_amplifier);
    
    -- Calculate final confidence
    final_confidence := (base_confidence + access_bonus) * age_decay;
    
    -- Ensure bounds
    RETURN GREATEST(0.0, LEAST(1.0, final_confidence));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create trigger function to update confidence scores on access
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    -- Update access tracking
    NEW.last_accessed = CURRENT_TIMESTAMP;
    NEW.access_count = OLD.access_count + 1;
    
    -- Recalculate confidence based on access pattern
    -- This would be called by the application, but we set a reasonable default
    IF NEW.confidence_score = OLD.confidence_score THEN
        NEW.confidence_score = calculate_memory_confidence(
            OLD.confidence_score,
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - NEW.timestamp)) / 86400.0, -- age in days
            NEW.emotional_weight,
            NEW.access_count,
            1.0 -- default decay rate modifier
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to track memory access (disabled by default for performance)
-- CREATE TRIGGER trigger_update_memory_access
--     BEFORE UPDATE OF last_accessed ON conversation_exchanges
--     FOR EACH ROW
--     EXECUTE FUNCTION update_memory_access();

-- Create view for semantic memory retrieval with confidence scoring
CREATE VIEW semantic_memory_view AS
SELECT 
    ce.exchange_id,
    ce.conversation_id,
    ce.speaker_id,
    ce.message_text,
    ce.message_type,
    ce.emotion,
    ce.timestamp,
    ce.confidence_score,
    ce.emotional_weight,
    ce.trust_level_required,
    ce.last_accessed,
    ce.access_count,
    ce.memory_embedding,
    
    -- Calculate age-based metrics
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 as age_days,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.last_accessed)) / 86400.0 as days_since_access,
    
    -- Emotional context data
    ec.sentiment_score,
    ec.emotional_intensity,
    ec.relationship_impact_score,
    ec.emotional_keywords,
    
    -- Conversation context
    cc.player_id,
    cc.npc_id,
    cc.mood as conversation_mood,
    cc.relationship_level,
    
    -- NPC personality factors
    np.traits as npc_traits,
    np.background_story,
    
    -- Memory personality config
    mpc.decay_rate_modifier,
    mpc.emotional_sensitivity,
    mpc.uncertainty_threshold,
    mpc.detail_retention_strength,
    mpc.name_retention_strength
    
FROM conversation_exchanges ce
LEFT JOIN emotional_context ec ON ce.exchange_id = ec.exchange_id
LEFT JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
LEFT JOIN npc_personalities np ON cc.npc_id = np.npc_id
LEFT JOIN memory_personality_config mpc ON cc.npc_id = mpc.npc_id
WHERE ce.confidence_score > 0.0; -- Only include memories with some confidence

-- Create view for memory statistics and analytics
CREATE VIEW memory_statistics AS
SELECT 
    cc.npc_id,
    COUNT(ce.exchange_id) as total_memories,
    AVG(ce.confidence_score) as avg_confidence,
    AVG(ce.emotional_weight) as avg_emotional_weight,
    COUNT(CASE WHEN ce.emotional_weight > 0.7 THEN 1 END) as high_emotion_memories,
    COUNT(CASE WHEN ce.confidence_score < 0.3 THEN 1 END) as low_confidence_memories,
    MAX(ce.last_accessed) as most_recent_access,
    AVG(ce.access_count) as avg_access_count,
    
    -- Age distribution
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 1 THEN 1 END) as memories_last_day,
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 7 THEN 1 END) as memories_last_week,
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 30 THEN 1 END) as memories_last_month
    
FROM conversation_exchanges ce
LEFT JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
GROUP BY cc.npc_id;

-- Insert default memory personality configurations for existing NPCs
INSERT INTO memory_personality_config (npc_id, decay_rate_modifier, emotional_sensitivity, detail_retention_strength, name_retention_strength, uncertainty_threshold)
SELECT 
    npc_id,
    1.0, -- Default decay rate
    1.0, -- Default emotional sensitivity
    0.5, -- Balanced detail retention
    0.5, -- Balanced name retention
    0.3  -- Default uncertainty threshold
FROM npc_personalities
WHERE npc_id NOT IN (SELECT npc_id FROM memory_personality_config)
ON CONFLICT (npc_id) DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE memory_embeddings IS 'Vector embeddings for conversation exchanges to enable semantic similarity search';
COMMENT ON TABLE memory_access_log IS 'Tracks memory access patterns for confidence degradation calculations';  
COMMENT ON TABLE memory_personality_config IS 'Individual NPC memory characteristics and behavior modifiers';
COMMENT ON TABLE emotional_context IS 'Emotional analysis data for conversation exchanges';

COMMENT ON FUNCTION calculate_memory_confidence IS 'Calculates memory confidence based on age, emotional weight, and access patterns';
COMMENT ON VIEW semantic_memory_view IS 'Comprehensive view combining memory data with confidence scoring and emotional context';
COMMENT ON VIEW memory_statistics IS 'Analytics view for monitoring memory system performance and patterns';

-- Update migration tracking (if exists)
-- INSERT INTO schema_migrations (version, applied_at) VALUES ('034', CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING;