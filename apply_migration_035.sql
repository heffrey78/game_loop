-- Migration 035: Memory Clustering Support (Modified for PostgreSQL compatibility)

-- Add clustering fields to conversation_exchanges table
ALTER TABLE conversation_exchanges 
ADD COLUMN IF NOT EXISTS memory_cluster_id INTEGER,
ADD COLUMN IF NOT EXISTS cluster_assignment_timestamp TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS cluster_confidence_score DECIMAL(3,2);

-- Add constraints for clustering fields (without IF NOT EXISTS for compatibility)
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_cluster_confidence_score') THEN
        ALTER TABLE conversation_exchanges 
        ADD CONSTRAINT chk_cluster_confidence_score 
        CHECK (cluster_confidence_score IS NULL OR (cluster_confidence_score >= 0.0 AND cluster_confidence_score <= 1.0));
    END IF;
END $$;

-- Add comments for clustering fields
COMMENT ON COLUMN conversation_exchanges.memory_cluster_id IS 'Cluster assignment ID for grouping semantically related memories (NULL if not clustered)';
COMMENT ON COLUMN conversation_exchanges.cluster_assignment_timestamp IS 'Timestamp when cluster assignment was last updated (NULL if not clustered)';
COMMENT ON COLUMN conversation_exchanges.cluster_confidence_score IS 'Confidence in cluster membership strength (0.0-1.0, NULL if not clustered)';

-- Create performance indexes for clustering operations (optimized for sub-50ms queries)
CREATE INDEX IF NOT EXISTS idx_exchanges_npc_cluster_confidence 
ON conversation_exchanges(speaker_id, memory_cluster_id, cluster_confidence_score, timestamp)
WHERE memory_cluster_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_exchanges_cluster_performance
ON conversation_exchanges(memory_cluster_id, cluster_confidence_score, last_accessed)
WHERE memory_cluster_id IS NOT NULL AND cluster_confidence_score > 0.3;

CREATE INDEX IF NOT EXISTS idx_exchanges_cluster_timestamp 
ON conversation_exchanges(cluster_assignment_timestamp)
WHERE cluster_assignment_timestamp IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_exchanges_conversation_cluster
ON conversation_exchanges(conversation_id, memory_cluster_id, cluster_confidence_score)
WHERE memory_cluster_id IS NOT NULL;

-- Create foreign key constraint
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_memory_cluster_id') THEN
        ALTER TABLE conversation_exchanges 
        ADD CONSTRAINT fk_memory_cluster_id 
        FOREIGN KEY (memory_cluster_id) REFERENCES memory_clusters(cluster_id) ON DELETE SET NULL;
    END IF;
END $$;