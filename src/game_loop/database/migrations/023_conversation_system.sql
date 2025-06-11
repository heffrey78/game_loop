-- Migration 023: Add Conversation and Query System Tables
-- Creates tables for conversation contexts, exchanges, NPC personalities, and query logs

-- Conversation contexts table
CREATE TABLE conversation_contexts (
    conversation_id VARCHAR(255) PRIMARY KEY,
    player_id VARCHAR(255) NOT NULL,
    npc_id VARCHAR(255) NOT NULL,
    topic VARCHAR(500),
    mood VARCHAR(100) NOT NULL DEFAULT 'neutral',
    relationship_level FLOAT NOT NULL DEFAULT 0.0,
    context_data JSONB DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

-- Conversation exchanges table
CREATE TABLE conversation_exchanges (
    exchange_id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversation_contexts(conversation_id) ON DELETE CASCADE,
    speaker_id VARCHAR(255) NOT NULL,
    message_text TEXT NOT NULL,
    message_type VARCHAR(100) NOT NULL,
    emotion VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- NPC personalities table
CREATE TABLE npc_personalities (
    npc_id VARCHAR(255) PRIMARY KEY,
    traits JSONB NOT NULL DEFAULT '{}',
    knowledge_areas JSONB NOT NULL DEFAULT '[]',
    speech_patterns JSONB NOT NULL DEFAULT '{}',
    relationships JSONB NOT NULL DEFAULT '{}',
    background_story TEXT,
    default_mood VARCHAR(100) NOT NULL DEFAULT 'neutral',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Query logs table
CREATE TABLE query_logs (
    query_id VARCHAR(255) PRIMARY KEY,
    player_id VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    query_type VARCHAR(100) NOT NULL,
    response_text TEXT,
    information_type VARCHAR(100),
    confidence FLOAT,
    sources JSONB DEFAULT '[]',
    related_queries JSONB DEFAULT '[]',
    context_data JSONB DEFAULT '{}',
    processing_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    errors JSONB DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for conversation_contexts
CREATE INDEX idx_conversation_contexts_player_id ON conversation_contexts(player_id);
CREATE INDEX idx_conversation_contexts_npc_id ON conversation_contexts(npc_id);
CREATE INDEX idx_conversation_contexts_status ON conversation_contexts(status);
CREATE INDEX idx_conversation_contexts_started_at ON conversation_contexts(started_at);
CREATE INDEX idx_conversation_contexts_player_npc ON conversation_contexts(player_id, npc_id);

-- Create indexes for conversation_exchanges
CREATE INDEX idx_conversation_exchanges_conversation_id ON conversation_exchanges(conversation_id);
CREATE INDEX idx_conversation_exchanges_speaker_id ON conversation_exchanges(speaker_id);
CREATE INDEX idx_conversation_exchanges_created_at ON conversation_exchanges(created_at);
CREATE INDEX idx_conversation_exchanges_message_type ON conversation_exchanges(message_type);

-- Create indexes for npc_personalities
CREATE INDEX idx_npc_personalities_updated_at ON npc_personalities(updated_at);

-- Create indexes for query_logs
CREATE INDEX idx_query_logs_player_id ON query_logs(player_id);
CREATE INDEX idx_query_logs_query_type ON query_logs(query_type);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX idx_query_logs_success ON query_logs(success);
CREATE INDEX idx_query_logs_player_type ON query_logs(player_id, query_type);

-- Add constraints
ALTER TABLE conversation_contexts 
    ADD CONSTRAINT check_relationship_level 
    CHECK (relationship_level >= -1.0 AND relationship_level <= 1.0);

ALTER TABLE conversation_contexts 
    ADD CONSTRAINT check_status 
    CHECK (status IN ('active', 'ended', 'paused', 'abandoned'));

ALTER TABLE conversation_exchanges 
    ADD CONSTRAINT check_message_type 
    CHECK (message_type IN ('greeting', 'question', 'statement', 'farewell', 'system'));

ALTER TABLE query_logs 
    ADD CONSTRAINT check_query_type 
    CHECK (query_type IN ('world_info', 'object_info', 'npc_info', 'location_info', 'help', 'status', 'inventory', 'quest_info'));

ALTER TABLE query_logs 
    ADD CONSTRAINT check_confidence 
    CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0));

-- Create trigger to update last_updated timestamp on conversation_contexts
CREATE OR REPLACE FUNCTION update_conversation_context_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_conversation_context_timestamp
    BEFORE UPDATE ON conversation_contexts
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_context_timestamp();

-- Create trigger to update updated_at timestamp on npc_personalities
CREATE OR REPLACE FUNCTION update_npc_personality_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_npc_personality_timestamp
    BEFORE UPDATE ON npc_personalities
    FOR EACH ROW
    EXECUTE FUNCTION update_npc_personality_timestamp();

-- Create view for active conversations
CREATE VIEW active_conversations AS
SELECT 
    cc.*,
    np.traits as npc_traits,
    np.knowledge_areas as npc_knowledge_areas,
    np.background_story as npc_background,
    (
        SELECT COUNT(*) 
        FROM conversation_exchanges ce 
        WHERE ce.conversation_id = cc.conversation_id
    ) as exchange_count,
    (
        SELECT MAX(ce.created_at) 
        FROM conversation_exchanges ce 
        WHERE ce.conversation_id = cc.conversation_id
    ) as last_exchange_at
FROM conversation_contexts cc
LEFT JOIN npc_personalities np ON cc.npc_id = np.npc_id
WHERE cc.status = 'active';

-- Create view for conversation statistics
CREATE VIEW conversation_statistics AS
SELECT 
    player_id,
    npc_id,
    COUNT(*) as total_conversations,
    AVG(relationship_level) as avg_relationship,
    MAX(relationship_level) as max_relationship,
    MIN(relationship_level) as min_relationship,
    COUNT(CASE WHEN status = 'ended' THEN 1 END) as completed_conversations,
    COUNT(CASE WHEN status = 'abandoned' THEN 1 END) as abandoned_conversations,
    AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60) as avg_duration_minutes
FROM conversation_contexts
GROUP BY player_id, npc_id;