-- Migration 024: Fix conversation schema with proper UUID types and constraints
-- This migration addresses critical schema inconsistencies identified in commit 22 code review

-- Drop any dependent views first
DROP VIEW IF EXISTS active_conversations CASCADE;

-- Drop existing conversation tables in correct order (to handle foreign key dependencies)
DROP TABLE IF EXISTS conversation_knowledge CASCADE;
DROP TABLE IF EXISTS conversation_exchanges CASCADE;
DROP TABLE IF EXISTS conversation_contexts CASCADE;
DROP TABLE IF EXISTS npc_personalities CASCADE;

-- Create npc_personalities table with proper UUID types
CREATE TABLE npc_personalities (
    npc_id UUID PRIMARY KEY,
    traits JSONB NOT NULL DEFAULT '{}',
    knowledge_areas TEXT[] NOT NULL DEFAULT '{}',
    speech_patterns JSONB NOT NULL DEFAULT '{}',
    relationships JSONB NOT NULL DEFAULT '{}',
    background_story TEXT NOT NULL DEFAULT '',
    default_mood VARCHAR(50) NOT NULL DEFAULT 'neutral',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create conversation_contexts table with proper foreign key relationships
CREATE TABLE conversation_contexts (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id),
    topic VARCHAR(255),
    mood VARCHAR(50) NOT NULL DEFAULT 'neutral',
    relationship_level DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    context_data JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT chk_relationship_level CHECK (relationship_level >= -1.0 AND relationship_level <= 1.0),
    CONSTRAINT chk_status CHECK (status IN ('active', 'ended', 'paused', 'abandoned'))
);

-- Create conversation_exchanges table with cascade delete
CREATE TABLE conversation_exchanges (
    exchange_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversation_contexts(conversation_id) ON DELETE CASCADE,
    speaker_id UUID NOT NULL,
    message_text TEXT NOT NULL,
    message_type VARCHAR(20) NOT NULL,
    emotion VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exchange_metadata JSONB NOT NULL DEFAULT '{}',
    
    CONSTRAINT chk_message_type CHECK (message_type IN ('greeting', 'question', 'statement', 'farewell', 'system'))
);

-- Create conversation_knowledge table for extracted information
CREATE TABLE conversation_knowledge (
    knowledge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversation_contexts(conversation_id) ON DELETE CASCADE,
    information_type VARCHAR(50) NOT NULL,
    extracted_info JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    source_exchange_id UUID REFERENCES conversation_exchanges(exchange_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- Create performance indexes
CREATE INDEX idx_conversation_contexts_player_npc ON conversation_contexts(player_id, npc_id);
CREATE INDEX idx_conversation_contexts_status ON conversation_contexts(status);
CREATE INDEX idx_conversation_contexts_last_updated ON conversation_contexts(last_updated);
CREATE INDEX idx_conversation_contexts_npc_id ON conversation_contexts(npc_id);

CREATE INDEX idx_conversation_exchanges_conversation ON conversation_exchanges(conversation_id);
CREATE INDEX idx_conversation_exchanges_timestamp ON conversation_exchanges(timestamp);
CREATE INDEX idx_conversation_exchanges_speaker ON conversation_exchanges(speaker_id);
CREATE INDEX idx_conversation_exchanges_type ON conversation_exchanges(message_type);

CREATE INDEX idx_conversation_knowledge_conversation ON conversation_knowledge(conversation_id);
CREATE INDEX idx_conversation_knowledge_type ON conversation_knowledge(information_type);
CREATE INDEX idx_conversation_knowledge_created ON conversation_knowledge(created_at);

CREATE INDEX idx_npc_personalities_updated ON npc_personalities(updated_at);

-- Add comments for documentation
COMMENT ON TABLE npc_personalities IS 'Stores NPC personality data with traits, knowledge areas, and relationships';
COMMENT ON TABLE conversation_contexts IS 'Tracks conversation state between players and NPCs';
COMMENT ON TABLE conversation_exchanges IS 'Individual messages within conversations';
COMMENT ON TABLE conversation_knowledge IS 'Information extracted from conversations for game knowledge base';

COMMENT ON COLUMN conversation_contexts.relationship_level IS 'Player-NPC relationship level from -1.0 (hostile) to 1.0 (friendly)';
COMMENT ON COLUMN conversation_knowledge.confidence_score IS 'Confidence level of extracted information from 0.0 to 1.0';