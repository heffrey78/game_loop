-- Migration: Add conversation threading tables for topic continuity and persistent NPC memory
-- Description: Creates tables for conversation threads, player memory profiles, and topic evolution tracking
-- Version: 006
-- Date: 2024-01-15

BEGIN;

-- =====================================================
-- Conversation Threads Table
-- =====================================================
-- Persistent conversation threads that span multiple game sessions
CREATE TABLE conversation_threads (
    thread_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id) ON DELETE CASCADE,
    
    -- Thread metadata
    primary_topic VARCHAR(255) NOT NULL,
    thread_title VARCHAR(500),
    thread_status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Topic progression
    topic_evolution JSONB NOT NULL DEFAULT '[]'::jsonb,
    subtopics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    resolved_questions TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    pending_questions TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    
    -- Relationship tracking
    trust_progression JSONB NOT NULL DEFAULT '[]'::jsonb,
    emotional_arc JSONB NOT NULL DEFAULT '[]'::jsonb,
    relationship_milestones JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Session tracking
    session_count INTEGER NOT NULL DEFAULT 0,
    last_session_id UUID,
    next_conversation_hooks TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    
    -- Importance and priority
    importance_score DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    priority_level VARCHAR(10) NOT NULL DEFAULT 'normal',
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_referenced TIMESTAMP WITH TIME ZONE,
    dormant_since TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_thread_importance_score CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    CONSTRAINT chk_thread_status CHECK (thread_status IN ('active', 'dormant', 'concluded', 'abandoned')),
    CONSTRAINT chk_thread_priority CHECK (priority_level IN ('urgent', 'high', 'normal', 'low')),
    CONSTRAINT chk_thread_session_count CHECK (session_count >= 0)
);

-- Indexes for conversation threads
CREATE INDEX idx_conversation_threads_player_npc ON conversation_threads(player_id, npc_id);
CREATE INDEX idx_conversation_threads_primary_topic ON conversation_threads(primary_topic);
CREATE INDEX idx_conversation_threads_status ON conversation_threads(thread_status);
CREATE INDEX idx_conversation_threads_last_referenced ON conversation_threads(last_referenced DESC);
CREATE INDEX idx_conversation_threads_importance ON conversation_threads(importance_score DESC);

-- =====================================================
-- Player Memory Profiles Table
-- =====================================================
-- NPC-specific memory profiles for individual players
CREATE TABLE player_memory_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id) ON DELETE CASCADE,
    
    -- Player characteristics as remembered by NPC
    remembered_name VARCHAR(255),
    player_traits JSONB NOT NULL DEFAULT '{}'::jsonb,
    player_interests TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    player_dislikes TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    player_goals TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    player_secrets TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    
    -- Relationship state
    relationship_level DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    trust_level DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    familiarity_score DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    
    -- Interaction patterns
    conversation_style VARCHAR(50) NOT NULL DEFAULT 'formal',
    preferred_topics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    avoided_topics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    shared_experiences JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Memory quality and importance
    memory_accuracy DECIMAL(3,2) NOT NULL DEFAULT 1.0,
    memory_importance DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    last_memory_quality VARCHAR(20) NOT NULL DEFAULT 'clear',
    
    -- Emotional context
    last_interaction_mood VARCHAR(50) NOT NULL DEFAULT 'neutral',
    emotional_associations JSONB NOT NULL DEFAULT '{}'::jsonb,
    sentiment_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Interaction tracking
    total_interactions INTEGER NOT NULL DEFAULT 0,
    successful_interactions INTEGER NOT NULL DEFAULT 0,
    memorable_moments JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Timing and updates
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_interaction TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_profile_relationship_level CHECK (relationship_level >= -1.0 AND relationship_level <= 1.0),
    CONSTRAINT chk_profile_trust_level CHECK (trust_level >= 0.0 AND trust_level <= 1.0),
    CONSTRAINT chk_profile_familiarity_score CHECK (familiarity_score >= 0.0 AND familiarity_score <= 1.0),
    CONSTRAINT chk_profile_memory_accuracy CHECK (memory_accuracy >= 0.0 AND memory_accuracy <= 1.0),
    CONSTRAINT chk_profile_memory_importance CHECK (memory_importance >= 0.0 AND memory_importance <= 1.0),
    CONSTRAINT chk_profile_conversation_style CHECK (conversation_style IN ('formal', 'casual', 'intimate', 'professional', 'friendly')),
    CONSTRAINT chk_profile_memory_quality CHECK (last_memory_quality IN ('clear', 'hazy', 'confused', 'fragmented', 'vivid')),
    
    -- Unique constraint: one profile per player-NPC pair
    UNIQUE(player_id, npc_id)
);

-- Indexes for player memory profiles
CREATE INDEX idx_player_memory_profiles_player ON player_memory_profiles(player_id);
CREATE INDEX idx_player_memory_profiles_npc ON player_memory_profiles(npc_id);
CREATE INDEX idx_player_memory_profiles_relationship ON player_memory_profiles(relationship_level DESC);
CREATE INDEX idx_player_memory_profiles_last_interaction ON player_memory_profiles(last_interaction DESC);

-- =====================================================
-- Topic Evolution Table
-- =====================================================
-- Track how conversation topics evolve and transform over time
CREATE TABLE topic_evolutions (
    evolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES conversation_threads(thread_id) ON DELETE CASCADE,
    session_id UUID REFERENCES conversation_contexts(conversation_id) ON DELETE SET NULL,
    
    -- Topic transition
    source_topic VARCHAR(255) NOT NULL,
    target_topic VARCHAR(255) NOT NULL,
    transition_type VARCHAR(20) NOT NULL,
    
    -- Context and reasoning
    transition_reason TEXT,
    player_initiated BOOLEAN NOT NULL DEFAULT FALSE,
    confidence_score DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    
    -- Semantic information
    topic_keywords TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    emotional_context VARCHAR(50) NOT NULL DEFAULT 'neutral',
    conversation_depth VARCHAR(20) NOT NULL DEFAULT 'surface',
    
    -- Relationship to other topics
    parent_topics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    child_topics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    related_topics TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    
    -- Quality and importance
    evolution_quality VARCHAR(20) NOT NULL DEFAULT 'smooth',
    importance_to_relationship DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    memory_formation_likelihood DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    
    -- Timing
    evolved_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_evolution_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_evolution_importance CHECK (importance_to_relationship >= 0.0 AND importance_to_relationship <= 1.0),
    CONSTRAINT chk_evolution_memory_likelihood CHECK (memory_formation_likelihood >= 0.0 AND memory_formation_likelihood <= 1.0),
    CONSTRAINT chk_evolution_transition_type CHECK (transition_type IN ('natural', 'forced', 'interrupted', 'concluded', 'branched')),
    CONSTRAINT chk_evolution_depth CHECK (conversation_depth IN ('surface', 'moderate', 'deep', 'intimate')),
    CONSTRAINT chk_evolution_quality CHECK (evolution_quality IN ('smooth', 'awkward', 'natural', 'forced', 'seamless'))
);

-- Indexes for topic evolutions
CREATE INDEX idx_topic_evolutions_thread ON topic_evolutions(thread_id);
CREATE INDEX idx_topic_evolutions_session ON topic_evolutions(session_id);
CREATE INDEX idx_topic_evolutions_evolved_at ON topic_evolutions(evolved_at DESC);
CREATE INDEX idx_topic_evolutions_source_topic ON topic_evolutions(source_topic);
CREATE INDEX idx_topic_evolutions_target_topic ON topic_evolutions(target_topic);

-- =====================================================
-- Update existing tables with threading relationships
-- =====================================================
-- Add thread_id foreign key to conversation_contexts table
ALTER TABLE conversation_contexts 
ADD COLUMN thread_id UUID REFERENCES conversation_threads(thread_id) ON DELETE SET NULL;

-- Add index for the new foreign key
CREATE INDEX idx_conversation_contexts_thread ON conversation_contexts(thread_id);

-- =====================================================
-- Comments for documentation
-- =====================================================
COMMENT ON TABLE conversation_threads IS 'Persistent conversation threads that span multiple game sessions';
COMMENT ON COLUMN conversation_threads.topic_evolution IS 'JSON array tracking topic progressions with timestamps and reasons';
COMMENT ON COLUMN conversation_threads.trust_progression IS 'JSON array tracking trust level changes over time';
COMMENT ON COLUMN conversation_threads.next_conversation_hooks IS 'Array of topics NPC should bring up in next conversation';

COMMENT ON TABLE player_memory_profiles IS 'NPC-specific memory profiles for individual players';
COMMENT ON COLUMN player_memory_profiles.player_traits IS 'JSON object with observed player traits and confidence scores';
COMMENT ON COLUMN player_memory_profiles.sentiment_history IS 'JSON array tracking sentiment changes over interactions';
COMMENT ON COLUMN player_memory_profiles.memorable_moments IS 'JSON array of highly memorable interactions with importance scores';

COMMENT ON TABLE topic_evolutions IS 'Track how conversation topics evolve and transform over time';
COMMENT ON COLUMN topic_evolutions.transition_type IS 'Type of topic transition: natural, forced, interrupted, concluded, branched';
COMMENT ON COLUMN topic_evolutions.evolution_quality IS 'Quality assessment of topic transition: smooth, awkward, natural, forced, seamless';

-- =====================================================
-- Trigger to update last_updated timestamps
-- =====================================================
CREATE OR REPLACE FUNCTION update_threading_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_conversation_threads_updated
    BEFORE UPDATE ON conversation_threads
    FOR EACH ROW EXECUTE FUNCTION update_threading_timestamp();

CREATE TRIGGER trg_player_memory_profiles_updated
    BEFORE UPDATE ON player_memory_profiles
    FOR EACH ROW EXECUTE FUNCTION update_threading_timestamp();

COMMIT;