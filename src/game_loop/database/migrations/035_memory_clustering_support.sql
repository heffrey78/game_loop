-- Migration 035: Memory Clustering Support
-- Extends conversation_exchanges table with clustering fields for memory organization
-- and semantic grouping of related memories based on emotional and thematic similarity

-- Add clustering fields to conversation_exchanges table
ALTER TABLE conversation_exchanges 
ADD COLUMN IF NOT EXISTS memory_cluster_id INTEGER,
ADD COLUMN IF NOT EXISTS cluster_assignment_timestamp TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS cluster_confidence_score DECIMAL(3,2);

-- Add constraints for clustering fields
ALTER TABLE conversation_exchanges 
ADD CONSTRAINT IF NOT EXISTS chk_cluster_confidence_score CHECK (cluster_confidence_score IS NULL OR (cluster_confidence_score >= 0.0 AND cluster_confidence_score <= 1.0));

-- Add comments for clustering fields
COMMENT ON COLUMN conversation_exchanges.memory_cluster_id IS 'Cluster assignment ID for grouping semantically related memories (NULL if not clustered)';
COMMENT ON COLUMN conversation_exchanges.cluster_assignment_timestamp IS 'Timestamp when cluster assignment was last updated (NULL if not clustered)';
COMMENT ON COLUMN conversation_exchanges.cluster_confidence_score IS 'Confidence in cluster membership strength (0.0-1.0, NULL if not clustered)';

-- Create performance indexes for clustering operations (optimized for sub-50ms queries)

-- Primary composite index for performance-critical NPC clustering queries
CREATE INDEX IF NOT EXISTS idx_exchanges_npc_cluster_confidence 
ON conversation_exchanges(speaker_id, memory_cluster_id, cluster_confidence_score, timestamp)
WHERE memory_cluster_id IS NOT NULL;

-- Optimized index for high-confidence cluster membership queries
CREATE INDEX IF NOT EXISTS idx_exchanges_cluster_performance
ON conversation_exchanges(memory_cluster_id, cluster_confidence_score, last_accessed)
WHERE memory_cluster_id IS NOT NULL AND cluster_confidence_score > 0.3;

-- Time-based index for cluster assignment tracking and batch processing
CREATE INDEX IF NOT EXISTS idx_exchanges_cluster_timestamp 
ON conversation_exchanges(cluster_assignment_timestamp)
WHERE cluster_assignment_timestamp IS NOT NULL;

-- Composite index for conversation-based cluster queries (specialized use case)
CREATE INDEX IF NOT EXISTS idx_exchanges_conversation_cluster
ON conversation_exchanges(conversation_id, memory_cluster_id, cluster_confidence_score)
WHERE memory_cluster_id IS NOT NULL;

-- Create memory_clusters table for cluster metadata and management
CREATE TABLE IF NOT EXISTS memory_clusters (
    cluster_id SERIAL PRIMARY KEY,
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id) ON DELETE CASCADE,
    cluster_name VARCHAR(255),
    cluster_theme TEXT,
    emotional_profile JSONB NOT NULL DEFAULT '{}', -- Average emotional characteristics
    semantic_centroid VECTOR(384), -- Cluster center in embedding space
    member_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_avg_confidence CHECK (avg_confidence >= 0.0 AND avg_confidence <= 1.0),
    CONSTRAINT chk_member_count CHECK (member_count >= 0)
);

-- Indexes for memory_clusters table
CREATE INDEX IF NOT EXISTS idx_memory_clusters_npc ON memory_clusters(npc_id);
CREATE INDEX IF NOT EXISTS idx_memory_clusters_updated ON memory_clusters(updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_clusters_member_count ON memory_clusters(member_count);

-- Create vector index optimization functions
CREATE OR REPLACE FUNCTION calculate_optimal_vector_lists(table_name TEXT, min_lists INTEGER DEFAULT 10) 
RETURNS INTEGER AS $$
DECLARE
    row_count INTEGER;
    optimal_lists INTEGER;
BEGIN
    -- Get current row count for the table
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO row_count;
    
    -- Calculate optimal lists: max(min_lists, rows/1000)
    optimal_lists := GREATEST(min_lists, row_count / 1000);
    
    -- For very small datasets, use smaller lists to avoid over-partitioning
    IF row_count < 1000 THEN
        optimal_lists := LEAST(optimal_lists, GREATEST(1, row_count / 100));
    END IF;
    
    RETURN optimal_lists;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION rebuild_vector_index(
    index_name TEXT,
    table_name TEXT,
    column_name TEXT,
    distance_op TEXT DEFAULT 'vector_cosine_ops'
) RETURNS INTEGER AS $$
DECLARE
    optimal_lists INTEGER;
    rebuild_sql TEXT;
BEGIN
    -- Calculate optimal lists parameter
    optimal_lists := calculate_optimal_vector_lists(table_name);
    
    -- Build the index recreation SQL
    rebuild_sql := format(
        'CREATE INDEX %I ON %I USING ivfflat (%I %s) WITH (lists = %s)',
        index_name,
        table_name, 
        column_name,
        distance_op,
        optimal_lists
    );
    
    -- Drop existing index if it exists
    EXECUTE format('DROP INDEX IF EXISTS %I', index_name);
    
    -- Create new index with optimal parameters
    EXECUTE rebuild_sql;
    
    RETURN optimal_lists;
END;
$$ LANGUAGE plpgsql;

-- Create index for semantic similarity search on cluster centroids with dynamic optimization
SELECT rebuild_vector_index(
    'idx_memory_clusters_centroid',
    'memory_clusters', 
    'semantic_centroid',
    'vector_cosine_ops'
);

-- Create enhanced K-means clustering assignment function with iterative optimization
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(
    npc_id_param UUID,
    max_clusters INTEGER DEFAULT 10,
    similarity_threshold DECIMAL DEFAULT 0.3,
    max_iterations INTEGER DEFAULT 5,
    enable_reassignment BOOLEAN DEFAULT TRUE
) RETURNS INTEGER AS $$
DECLARE
    processed_memories INTEGER := 0;
    memory_record RECORD;
    best_cluster_id INTEGER;
    best_similarity DECIMAL;
    cluster_similarity DECIMAL;
    cluster_record RECORD;
    new_cluster_id INTEGER;
    iteration INTEGER := 0;
    assignments_changed INTEGER;
    total_assignments_changed INTEGER := 0;
    convergence_threshold INTEGER := 2; -- Stop if fewer than N assignments change
BEGIN
    -- First pass: assign unassigned memories to clusters
    FOR memory_record IN 
        SELECT ce.exchange_id, ce.memory_embedding, ce.emotional_weight, ce.confidence_score
        FROM conversation_exchanges ce
        JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
        WHERE cc.npc_id = npc_id_param 
            AND ce.memory_embedding IS NOT NULL
            AND ce.memory_cluster_id IS NULL
            AND ce.confidence_score > 0.1  -- Only cluster memories with some confidence
        ORDER BY ce.emotional_weight DESC, ce.confidence_score DESC  -- Prioritize emotional/confident memories
    LOOP
        best_cluster_id := NULL;
        best_similarity := 0.0;
        
        -- Find the best matching cluster for this memory
        FOR cluster_record IN
            SELECT cluster_id, semantic_centroid
            FROM memory_clusters 
            WHERE npc_id = npc_id_param 
                AND semantic_centroid IS NOT NULL
            ORDER BY member_count DESC  -- Check larger clusters first
        LOOP
            -- Calculate cosine similarity between memory and cluster centroid
            SELECT 1 - (memory_record.memory_embedding <=> cluster_record.semantic_centroid)
            INTO cluster_similarity;
            
            -- Track the best match
            IF cluster_similarity > best_similarity AND cluster_similarity >= similarity_threshold THEN
                best_similarity := cluster_similarity;
                best_cluster_id := cluster_record.cluster_id;
            END IF;
        END LOOP;
        
        -- If no suitable cluster found and we haven't reached max clusters, create new one
        IF best_cluster_id IS NULL THEN
            -- Check if we can create a new cluster
            IF (SELECT COUNT(*) FROM memory_clusters WHERE npc_id = npc_id_param) < max_clusters THEN
                INSERT INTO memory_clusters (
                    npc_id,
                    cluster_name,
                    semantic_centroid,
                    member_count,
                    avg_confidence,
                    emotional_profile
                ) VALUES (
                    npc_id_param,
                    'Cluster-' || (SELECT COUNT(*) + 1 FROM memory_clusters WHERE npc_id = npc_id_param),
                    memory_record.memory_embedding,
                    1,
                    memory_record.confidence_score,
                    jsonb_build_object('emotional_weight', memory_record.emotional_weight)
                ) RETURNING cluster_id INTO new_cluster_id;
                
                best_cluster_id := new_cluster_id;
                best_similarity := 1.0;  -- Perfect match for new cluster
            ELSE
                -- Force assignment to the closest cluster even if below threshold
                SELECT cluster_id INTO best_cluster_id
                FROM memory_clusters 
                WHERE npc_id = npc_id_param 
                    AND semantic_centroid IS NOT NULL
                ORDER BY (memory_record.memory_embedding <=> semantic_centroid) ASC
                LIMIT 1;
                
                -- Calculate actual similarity for forced assignment
                IF best_cluster_id IS NOT NULL THEN
                    SELECT 1 - (memory_record.memory_embedding <=> semantic_centroid)
                    INTO best_similarity
                    FROM memory_clusters
                    WHERE cluster_id = best_cluster_id;
                END IF;
            END IF;
        END IF;
        
        -- Assign the memory to the best cluster
        IF best_cluster_id IS NOT NULL THEN
            UPDATE conversation_exchanges 
            SET 
                memory_cluster_id = best_cluster_id,
                cluster_assignment_timestamp = CURRENT_TIMESTAMP,
                cluster_confidence_score = LEAST(1.0, best_similarity)
            WHERE exchange_id = memory_record.exchange_id;
            
            processed_memories := processed_memories + 1;
        END IF;
    END LOOP;
    
    -- If reassignment is enabled, perform iterative K-means optimization
    IF enable_reassignment AND processed_memories > 0 THEN
        WHILE iteration < max_iterations LOOP
            iteration := iteration + 1;
            assignments_changed := 0;
            
            -- Update all cluster centroids first
            PERFORM update_cluster_centroids(npc_id_param);
            
            -- Check each assigned memory for potential reassignment
            FOR memory_record IN 
                SELECT ce.exchange_id, ce.memory_embedding, ce.memory_cluster_id, ce.cluster_confidence_score
                FROM conversation_exchanges ce
                JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
                WHERE cc.npc_id = npc_id_param 
                    AND ce.memory_embedding IS NOT NULL
                    AND ce.memory_cluster_id IS NOT NULL
                    AND ce.confidence_score > 0.1
            LOOP
                best_cluster_id := memory_record.memory_cluster_id;
                best_similarity := memory_record.cluster_confidence_score;
                
                -- Find potentially better cluster assignment
                FOR cluster_record IN
                    SELECT cluster_id, semantic_centroid
                    FROM memory_clusters 
                    WHERE npc_id = npc_id_param 
                        AND semantic_centroid IS NOT NULL
                        AND cluster_id != memory_record.memory_cluster_id  -- Don't check current cluster
                LOOP
                    -- Calculate similarity to this alternative cluster
                    SELECT 1 - (memory_record.memory_embedding <=> cluster_record.semantic_centroid)
                    INTO cluster_similarity;
                    
                    -- If this cluster is significantly better, consider reassignment
                    IF cluster_similarity > best_similarity + 0.05 THEN  -- 5% improvement threshold
                        best_similarity := cluster_similarity;
                        best_cluster_id := cluster_record.cluster_id;
                    END IF;
                END LOOP;
                
                -- Reassign if we found a better cluster
                IF best_cluster_id != memory_record.memory_cluster_id THEN
                    UPDATE conversation_exchanges 
                    SET 
                        memory_cluster_id = best_cluster_id,
                        cluster_assignment_timestamp = CURRENT_TIMESTAMP,
                        cluster_confidence_score = LEAST(1.0, best_similarity)
                    WHERE exchange_id = memory_record.exchange_id;
                    
                    assignments_changed := assignments_changed + 1;
                    total_assignments_changed := total_assignments_changed + 1;
                END IF;
            END LOOP;
            
            -- Log iteration progress
            RAISE DEBUG 'K-means iteration %: % assignments changed', iteration, assignments_changed;
            
            -- Check for convergence (few assignments changed)
            IF assignments_changed <= convergence_threshold THEN
                RAISE DEBUG 'K-means converged after % iterations', iteration;
                EXIT;
            END IF;
        END LOOP;
        
        -- Final centroid update and statistics refresh
        PERFORM update_cluster_centroids(npc_id_param);
        
        -- Update all cluster statistics
        FOR cluster_record IN
            SELECT cluster_id FROM memory_clusters WHERE npc_id = npc_id_param
        LOOP
            PERFORM update_cluster_statistics(cluster_record.cluster_id);
        END LOOP;
        
        RAISE NOTICE 'K-means clustering completed: % initial assignments, % reassignments over % iterations', 
            processed_memories, total_assignments_changed, iteration;
    ELSE
        -- Even without reassignment, update statistics for all affected clusters
        FOR cluster_record IN
            SELECT DISTINCT memory_cluster_id as cluster_id
            FROM conversation_exchanges ce
            JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
            WHERE cc.npc_id = npc_id_param AND ce.memory_cluster_id IS NOT NULL
        LOOP
            PERFORM update_cluster_statistics(cluster_record.cluster_id);
        END LOOP;
    END IF;
    
    RETURN processed_memories;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Memory clustering assignment failed for NPC %: %', npc_id_param, SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create function to update cluster centroids based on member embeddings
CREATE OR REPLACE FUNCTION update_cluster_centroids(npc_id_param UUID DEFAULT NULL)
RETURNS INTEGER AS $$
DECLARE
    cluster_record RECORD;
    updated_count INTEGER := 0;
    centroid_vector VECTOR(384);
BEGIN
    -- Update centroids for specified NPC or all NPCs
    FOR cluster_record IN
        SELECT DISTINCT mc.cluster_id, mc.npc_id
        FROM memory_clusters mc
        WHERE (npc_id_param IS NULL OR mc.npc_id = npc_id_param)
            AND EXISTS(
                SELECT 1 FROM conversation_exchanges ce 
                WHERE ce.memory_cluster_id = mc.cluster_id 
                    AND ce.memory_embedding IS NOT NULL
            )
    LOOP
        -- Calculate average embedding (centroid) for this cluster
        SELECT AVG(ce.memory_embedding) 
        INTO centroid_vector
        FROM conversation_exchanges ce
        WHERE ce.memory_cluster_id = cluster_record.cluster_id
            AND ce.memory_embedding IS NOT NULL;
        
        -- Update the cluster centroid
        UPDATE memory_clusters
        SET 
            semantic_centroid = centroid_vector,
            updated_at = CURRENT_TIMESTAMP
        WHERE cluster_id = cluster_record.cluster_id;
        
        updated_count := updated_count + 1;
    END LOOP;
    
    RAISE NOTICE 'Updated centroids for % clusters', updated_count;
    RETURN updated_count;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to update cluster centroids: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create function for batch processing all NPC memory clustering
CREATE OR REPLACE FUNCTION process_all_npc_clustering()
RETURNS TABLE (
    npc_id UUID,
    memories_processed INTEGER,
    clusters_created INTEGER,
    processing_duration INTERVAL
) AS $$
DECLARE
    npc_record RECORD;
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    processed_count INTEGER;
    initial_cluster_count INTEGER;
    final_cluster_count INTEGER;
BEGIN
    FOR npc_record IN
        SELECT DISTINCT cc.npc_id
        FROM conversation_contexts cc
        JOIN conversation_exchanges ce ON cc.conversation_id = ce.conversation_id
        WHERE ce.memory_embedding IS NOT NULL
            AND ce.memory_cluster_id IS NULL
            AND ce.confidence_score > 0.1
    LOOP
        start_time := CURRENT_TIMESTAMP;
        
        -- Count initial clusters
        SELECT COUNT(*) INTO initial_cluster_count
        FROM memory_clusters
        WHERE memory_clusters.npc_id = npc_record.npc_id;
        
        -- Process clustering for this NPC
        processed_count := assign_memory_to_clusters(npc_record.npc_id);
        
        -- Update centroids after assignment
        PERFORM update_cluster_centroids(npc_record.npc_id);
        
        -- Count final clusters
        SELECT COUNT(*) INTO final_cluster_count
        FROM memory_clusters
        WHERE memory_clusters.npc_id = npc_record.npc_id;
        
        end_time := CURRENT_TIMESTAMP;
        
        -- Return results for this NPC
        RETURN QUERY
        SELECT 
            npc_record.npc_id,
            processed_count,
            final_cluster_count - initial_cluster_count,
            end_time - start_time;
    END LOOP;
    
    RETURN;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Batch clustering processing failed: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Add foreign key constraint for referential integrity (after memory_clusters table exists)
ALTER TABLE conversation_exchanges 
ADD CONSTRAINT fk_memory_cluster_id 
FOREIGN KEY (memory_cluster_id) REFERENCES memory_clusters(cluster_id) ON DELETE SET NULL;

-- Create optimized cluster analysis views (performance-optimized structure)

-- Lightweight basic metrics view for frequent access
CREATE VIEW cluster_basic_metrics AS
SELECT 
    mc.cluster_id,
    mc.npc_id,
    mc.cluster_name,
    mc.cluster_theme,
    mc.member_count as stored_member_count,
    mc.avg_confidence as stored_avg_confidence,
    mc.created_at,
    mc.updated_at,
    mc.last_accessed,
    
    -- Basic real-time metrics (efficient single pass)
    COUNT(ce.exchange_id) as actual_member_count,
    COALESCE(AVG(ce.cluster_confidence_score), 0) as current_cluster_confidence,
    
    -- Simple health check
    CASE 
        WHEN COUNT(ce.exchange_id) = 0 THEN 'empty'
        WHEN AVG(ce.cluster_confidence_score) < 0.3 THEN 'low_confidence'
        WHEN AVG(ce.cluster_confidence_score) > 0.7 THEN 'high_confidence'
        ELSE 'moderate_confidence'
    END as cluster_health
    
FROM memory_clusters mc
LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
GROUP BY mc.cluster_id, mc.npc_id, mc.cluster_name, mc.cluster_theme, 
         mc.member_count, mc.avg_confidence, mc.created_at, mc.updated_at, mc.last_accessed;

-- Detailed analytics view for expensive calculations (accessed less frequently)
-- Can be materialized for better performance
CREATE VIEW cluster_detailed_analytics AS
SELECT 
    mc.cluster_id,
    
    -- Memory quality metrics (moderate cost)
    AVG(ce.confidence_score) as avg_memory_confidence,
    AVG(ce.emotional_weight) as avg_emotional_weight,
    AVG(ce.access_count) as avg_member_access_count,
    MAX(ce.last_accessed) as most_recent_member_access,
    
    -- Age calculations (expensive - separated into detailed view)
    AVG(EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - ce.timestamp) / 86400.0) as avg_age_days,
    
    -- Emotional context (requires additional JOIN)
    AVG(ec.sentiment_score) as avg_sentiment,
    AVG(ec.emotional_intensity) as avg_emotional_intensity
    
FROM memory_clusters mc
LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
LEFT JOIN emotional_context ec ON ce.exchange_id = ec.exchange_id
WHERE ce.exchange_id IS NOT NULL  -- Only calculate for clusters with actual data
GROUP BY mc.cluster_id;

-- Create materialized view for expensive analytics (refreshed periodically)
CREATE MATERIALIZED VIEW cluster_detailed_analytics_cache AS
SELECT 
    mc.cluster_id,
    AVG(ce.confidence_score) as avg_memory_confidence,
    AVG(ce.emotional_weight) as avg_emotional_weight,
    AVG(ce.access_count) as avg_member_access_count,
    MAX(ce.last_accessed) as most_recent_member_access,
    AVG(EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - ce.timestamp) / 86400.0) as avg_age_days,
    AVG(ec.sentiment_score) as avg_sentiment,
    AVG(ec.emotional_intensity) as avg_emotional_intensity,
    CURRENT_TIMESTAMP as cache_updated_at
FROM memory_clusters mc
LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
LEFT JOIN emotional_context ec ON ce.exchange_id = ec.exchange_id
WHERE ce.exchange_id IS NOT NULL
GROUP BY mc.cluster_id;

-- Create index on materialized view for fast lookups
CREATE INDEX idx_cluster_detailed_cache_cluster_id ON cluster_detailed_analytics_cache(cluster_id);
CREATE INDEX idx_cluster_detailed_cache_updated ON cluster_detailed_analytics_cache(cache_updated_at);

-- Performance monitoring function for cluster analysis queries
CREATE OR REPLACE FUNCTION test_cluster_analysis_performance(
    sample_cluster_count INTEGER DEFAULT 100,
    target_ms DECIMAL DEFAULT 50.0
) RETURNS TABLE (
    test_name TEXT,
    execution_time_ms DECIMAL,
    meets_target BOOLEAN,
    sample_rows INTEGER,
    performance_status TEXT
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    sample_clusters INTEGER[];
    basic_metrics_time DECIMAL;
    detailed_analytics_time DECIMAL;
    combined_view_time DECIMAL;
BEGIN
    -- Get sample of cluster IDs for consistent testing
    SELECT array_agg(cluster_id) INTO sample_clusters
    FROM (
        SELECT cluster_id FROM memory_clusters 
        ORDER BY updated_at DESC 
        LIMIT sample_cluster_count
    ) s;
    
    -- Test 1: Basic metrics view performance
    start_time := clock_timestamp();
    PERFORM COUNT(*) FROM cluster_basic_metrics 
    WHERE cluster_id = ANY(sample_clusters);
    end_time := clock_timestamp();
    basic_metrics_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_basic_metrics'::TEXT,
        basic_metrics_time,
        basic_metrics_time <= target_ms,
        array_length(sample_clusters, 1),
        CASE 
            WHEN basic_metrics_time <= target_ms * 0.5 THEN 'excellent'
            WHEN basic_metrics_time <= target_ms THEN 'good'
            WHEN basic_metrics_time <= target_ms * 2 THEN 'needs_attention'
            ELSE 'critical'
        END;
    
    -- Test 2: Detailed analytics view performance
    start_time := clock_timestamp();
    PERFORM COUNT(*) FROM cluster_detailed_analytics 
    WHERE cluster_id = ANY(sample_clusters);
    end_time := clock_timestamp();
    detailed_analytics_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_detailed_analytics'::TEXT,
        detailed_analytics_time,
        detailed_analytics_time <= target_ms * 2, -- Allow 2x target for detailed analytics
        array_length(sample_clusters, 1),
        CASE 
            WHEN detailed_analytics_time <= target_ms THEN 'excellent'
            WHEN detailed_analytics_time <= target_ms * 2 THEN 'good'
            WHEN detailed_analytics_time <= target_ms * 4 THEN 'needs_attention'
            ELSE 'critical'
        END;
    
    -- Test 3: Combined view performance
    start_time := clock_timestamp();
    PERFORM COUNT(*) FROM memory_cluster_analysis 
    WHERE cluster_id = ANY(sample_clusters);
    end_time := clock_timestamp();
    combined_view_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'memory_cluster_analysis'::TEXT,
        combined_view_time,
        combined_view_time <= target_ms,
        array_length(sample_clusters, 1),
        CASE 
            WHEN combined_view_time <= target_ms * 0.5 THEN 'excellent'
            WHEN combined_view_time <= target_ms THEN 'good'
            WHEN combined_view_time <= target_ms * 2 THEN 'needs_attention'
            ELSE 'critical'
        END;
    
    RETURN;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Performance testing failed: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for memory_cluster_analysis for sub-50ms performance
CREATE MATERIALIZED VIEW IF NOT EXISTS memory_cluster_analysis_cache AS
SELECT 
    cbm.cluster_id,
    cbm.npc_id,
    cbm.cluster_name,
    cbm.cluster_theme,
    cbm.stored_member_count,
    cbm.stored_avg_confidence,
    cbm.created_at,
    cbm.updated_at,
    cbm.last_accessed,
    cbm.actual_member_count,
    cbm.current_cluster_confidence,
    cbm.cluster_health,
    
    -- Detailed metrics (pre-calculated for performance)
    cda.avg_memory_confidence,
    cda.avg_emotional_weight,
    cda.avg_member_access_count,
    cda.most_recent_member_access,
    cda.avg_age_days,
    cda.avg_sentiment,
    cda.avg_emotional_intensity,
    
    -- Cache metadata
    CURRENT_TIMESTAMP as cache_updated_at

FROM cluster_basic_metrics cbm
LEFT JOIN cluster_detailed_analytics cda ON cbm.cluster_id = cda.cluster_id;

-- Create indexes on the materialized view for fast access
CREATE INDEX IF NOT EXISTS idx_memory_cluster_analysis_cache_cluster_id ON memory_cluster_analysis_cache(cluster_id);
CREATE INDEX IF NOT EXISTS idx_memory_cluster_analysis_cache_npc_id ON memory_cluster_analysis_cache(npc_id);
CREATE INDEX IF NOT EXISTS idx_memory_cluster_analysis_cache_health ON memory_cluster_analysis_cache(cluster_health);
CREATE INDEX IF NOT EXISTS idx_memory_cluster_analysis_cache_updated ON memory_cluster_analysis_cache(cache_updated_at);

-- Create optimized view that uses cached data by default, falls back to live data
CREATE OR REPLACE VIEW memory_cluster_analysis AS
SELECT 
    cluster_id,
    npc_id,
    cluster_name,
    cluster_theme,
    stored_member_count,
    stored_avg_confidence,
    created_at,
    updated_at,
    last_accessed,
    actual_member_count,
    current_cluster_confidence,
    cluster_health,
    avg_memory_confidence,
    avg_emotional_weight,
    avg_member_access_count,
    most_recent_member_access,
    avg_age_days,
    avg_sentiment,
    avg_emotional_intensity
FROM memory_cluster_analysis_cache
WHERE cache_updated_at > CURRENT_TIMESTAMP - INTERVAL '1 hour' -- Use cache if fresh

UNION ALL

-- Fall back to live calculation if cache is stale
SELECT 
    cbm.cluster_id,
    cbm.npc_id,
    cbm.cluster_name,
    cbm.cluster_theme,
    cbm.stored_member_count,
    cbm.stored_avg_confidence,
    cbm.created_at,
    cbm.updated_at,
    cbm.last_accessed,
    cbm.actual_member_count,
    cbm.current_cluster_confidence,
    cbm.cluster_health,
    cda.avg_memory_confidence,
    cda.avg_emotional_weight,
    cda.avg_member_access_count,
    cda.most_recent_member_access,
    cda.avg_age_days,
    cda.avg_sentiment,
    cda.avg_emotional_intensity
FROM cluster_basic_metrics cbm
LEFT JOIN cluster_detailed_analytics cda ON cbm.cluster_id = cda.cluster_id
WHERE NOT EXISTS (
    SELECT 1 FROM memory_cluster_analysis_cache 
    WHERE cache_updated_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
    LIMIT 1
);

-- Create function to update cluster statistics (thread-safe with optimistic concurrency control)
CREATE OR REPLACE FUNCTION update_cluster_statistics(cluster_id_param INTEGER)
RETURNS VOID AS $$
DECLARE
    new_member_count INTEGER;
    new_avg_confidence DECIMAL;
    current_updated_at TIMESTAMP WITH TIME ZONE;
    rows_updated INTEGER;
    retry_count INTEGER := 0;
    max_retries INTEGER := 3;
BEGIN
    -- Retry loop for optimistic concurrency control
    WHILE retry_count < max_retries LOOP
        -- Get current updated_at timestamp to detect concurrent changes
        SELECT updated_at INTO current_updated_at
        FROM memory_clusters 
        WHERE cluster_id = cluster_id_param;
        
        -- If cluster doesn't exist, exit gracefully
        IF current_updated_at IS NULL THEN
            RAISE NOTICE 'Cluster % does not exist, skipping statistics update', cluster_id_param;
            RETURN;
        END IF;
        
        -- Calculate current statistics atomically within a transaction
        BEGIN
            -- Use READ COMMITTED isolation to get consistent snapshot
            SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
            
            SELECT 
                COUNT(exchange_id),
                COALESCE(AVG(cluster_confidence_score), 0.0)
            INTO new_member_count, new_avg_confidence
            FROM conversation_exchanges
            WHERE memory_cluster_id = cluster_id_param;
            
            -- Update cluster metadata with optimistic locking check
            UPDATE memory_clusters 
            SET 
                member_count = new_member_count,
                avg_confidence = new_avg_confidence,
                updated_at = CURRENT_TIMESTAMP
            WHERE cluster_id = cluster_id_param 
                AND updated_at = current_updated_at; -- Optimistic lock check
            
            GET DIAGNOSTICS rows_updated = ROW_COUNT;
            
            -- If update succeeded, we're done
            IF rows_updated = 1 THEN
                RAISE DEBUG 'Updated cluster % statistics: members=%, confidence=%', 
                    cluster_id_param, new_member_count, new_avg_confidence;
                RETURN;
            END IF;
            
        EXCEPTION
            WHEN serialization_failure OR deadlock_detected THEN
                -- Transaction conflict detected, retry
                RAISE DEBUG 'Transaction conflict updating cluster % statistics, retrying (attempt %)', 
                    cluster_id_param, retry_count + 1;
        END;
        
        -- If we reach here, either optimistic lock failed or transaction conflict occurred
        retry_count := retry_count + 1;
        
        -- Small random delay to reduce contention
        PERFORM pg_sleep(random() * 0.01); -- 0-10ms random delay
        
    END LOOP;
    
    -- If we exhausted retries, fall back to row-level locking
    RAISE WARNING 'Optimistic updates failed for cluster %, using row-level lock', cluster_id_param;
    
    -- Final attempt with explicit row lock
    PERFORM update_cluster_statistics_locked(cluster_id_param);
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to update cluster % statistics: %', cluster_id_param, SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Fallback function with explicit row-level locking for high contention scenarios
CREATE OR REPLACE FUNCTION update_cluster_statistics_locked(cluster_id_param INTEGER)
RETURNS VOID AS $$
DECLARE
    new_member_count INTEGER;
    new_avg_confidence DECIMAL;
    cluster_exists BOOLEAN;
BEGIN
    -- Acquire row-level lock on the cluster to prevent concurrent updates
    SELECT EXISTS(
        SELECT 1 FROM memory_clusters 
        WHERE cluster_id = cluster_id_param 
        FOR UPDATE NOWAIT -- Fail fast if lock unavailable
    ) INTO cluster_exists;
    
    -- If cluster doesn't exist, exit gracefully
    IF NOT cluster_exists THEN
        RAISE NOTICE 'Cluster % does not exist, skipping statistics update', cluster_id_param;
        RETURN;
    END IF;
    
    -- Calculate current statistics atomically
    SELECT 
        COUNT(exchange_id),
        COALESCE(AVG(cluster_confidence_score), 0.0)
    INTO new_member_count, new_avg_confidence
    FROM conversation_exchanges
    WHERE memory_cluster_id = cluster_id_param;
    
    -- Update cluster metadata (row is already locked)
    UPDATE memory_clusters 
    SET 
        member_count = new_member_count,
        avg_confidence = new_avg_confidence,
        updated_at = CURRENT_TIMESTAMP
    WHERE cluster_id = cluster_id_param;
    
    -- Log the update for debugging
    RAISE DEBUG 'Updated cluster % statistics with lock: members=%, confidence=%', 
        cluster_id_param, new_member_count, new_avg_confidence;
    
EXCEPTION
    WHEN lock_not_available THEN
        RAISE WARNING 'Could not acquire lock for cluster %, statistics update skipped', cluster_id_param;
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to update cluster % statistics with lock: %', cluster_id_param, SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create function for automatic cluster cleanup (removes empty clusters)
CREATE OR REPLACE FUNCTION cleanup_empty_clusters()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete clusters with no members
    DELETE FROM memory_clusters 
    WHERE cluster_id IN (
        SELECT mc.cluster_id 
        FROM memory_clusters mc
        LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
        WHERE ce.exchange_id IS NULL
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for cluster health monitoring and summary statistics
CREATE OR REPLACE FUNCTION get_cluster_health_summary()
RETURNS TABLE (
    total_clusters INTEGER,
    healthy_clusters INTEGER,
    low_confidence_clusters INTEGER,
    empty_clusters INTEGER,
    avg_cluster_size DECIMAL,
    avg_cluster_confidence DECIMAL,
    clusters_needing_attention INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_clusters,
        COUNT(CASE WHEN cluster_health = 'high_confidence' THEN 1 END)::INTEGER as healthy_clusters,
        COUNT(CASE WHEN cluster_health = 'low_confidence' THEN 1 END)::INTEGER as low_confidence_clusters,
        COUNT(CASE WHEN cluster_health = 'empty' THEN 1 END)::INTEGER as empty_clusters,
        COALESCE(AVG(actual_member_count), 0)::DECIMAL as avg_cluster_size,
        COALESCE(AVG(current_cluster_confidence), 0)::DECIMAL as avg_cluster_confidence,
        COUNT(CASE WHEN cluster_health IN ('empty', 'low_confidence') THEN 1 END)::INTEGER as clusters_needing_attention
    FROM cluster_basic_metrics;
END;
$$ LANGUAGE plpgsql;

-- Create function to refresh all materialized views with performance monitoring
CREATE OR REPLACE FUNCTION refresh_cluster_analytics_cache()
RETURNS TABLE (
    cache_name TEXT,
    refresh_duration INTERVAL,
    clusters_updated INTEGER,
    refresh_timestamp TIMESTAMP WITH TIME ZONE,
    performance_ms DECIMAL
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    updated_count INTEGER;
    performance_ms_val DECIMAL;
BEGIN
    -- Refresh cluster_detailed_analytics_cache
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW cluster_detailed_analytics_cache;
    end_time := clock_timestamp();
    
    SELECT COUNT(*) INTO updated_count FROM cluster_detailed_analytics_cache;
    performance_ms_val := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_detailed_analytics_cache'::TEXT,
        end_time - start_time,
        updated_count,
        end_time,
        performance_ms_val;
    
    -- Refresh memory_cluster_analysis_cache
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW memory_cluster_analysis_cache;
    end_time := clock_timestamp();
    
    SELECT COUNT(*) INTO updated_count FROM memory_cluster_analysis_cache;
    performance_ms_val := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'memory_cluster_analysis_cache'::TEXT,
        end_time - start_time,
        updated_count,
        end_time,
        performance_ms_val;
    
    -- Log the refresh for monitoring
    RAISE NOTICE 'All cluster analytics caches refreshed successfully';
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to refresh cluster analytics caches: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create function for intelligent cache refresh based on performance thresholds
CREATE OR REPLACE FUNCTION refresh_cache_if_needed(
    performance_threshold_ms DECIMAL DEFAULT 50.0,
    cache_max_age_minutes INTEGER DEFAULT 60
) RETURNS TABLE (
    action_taken TEXT,
    reason TEXT,
    performance_result TEXT
) AS $$
DECLARE
    cache_age_minutes INTEGER;
    current_performance RECORD;
    needs_refresh BOOLEAN := FALSE;
    refresh_reason TEXT;
BEGIN
    -- Check cache age
    SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(cache_updated_at))) / 60.0
    INTO cache_age_minutes
    FROM memory_cluster_analysis_cache;
    
    -- Check current performance
    SELECT * INTO current_performance
    FROM test_cluster_analysis_performance(50, performance_threshold_ms)
    WHERE test_name = 'memory_cluster_analysis'
    LIMIT 1;
    
    -- Determine if refresh is needed
    IF cache_age_minutes IS NULL THEN
        needs_refresh := TRUE;
        refresh_reason := 'cache_empty';
    ELSIF cache_age_minutes > cache_max_age_minutes THEN
        needs_refresh := TRUE;
        refresh_reason := 'cache_stale';
    ELSIF current_performance.performance_status IN ('needs_attention', 'critical') THEN
        needs_refresh := TRUE;
        refresh_reason := 'performance_degraded';
    END IF;
    
    IF needs_refresh THEN
        -- Refresh the cache
        PERFORM refresh_cluster_analytics_cache();
        
        -- Test performance after refresh
        SELECT * INTO current_performance
        FROM test_cluster_analysis_performance(50, performance_threshold_ms)
        WHERE test_name = 'memory_cluster_analysis'
        LIMIT 1;
        
        RETURN QUERY SELECT 
            'cache_refreshed'::TEXT,
            refresh_reason,
            current_performance.performance_status;
    ELSE
        RETURN QUERY SELECT 
            'no_action'::TEXT,
            'cache_fresh_and_performant'::TEXT,
            current_performance.performance_status;
    END IF;
    
EXCEPTION
    WHEN OTHERS THEN
        RETURN QUERY SELECT 
            'error'::TEXT,
            SQLERRM,
            'unknown'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Create function to check cache freshness and auto-refresh if needed
CREATE OR REPLACE FUNCTION ensure_fresh_cluster_analytics(max_age_minutes INTEGER DEFAULT 60)
RETURNS BOOLEAN AS $$
DECLARE
    cache_age_minutes INTEGER;
    needs_refresh BOOLEAN;
BEGIN
    -- Check the age of the cache
    SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(cache_updated_at))) / 60.0
    INTO cache_age_minutes
    FROM cluster_detailed_analytics_cache;
    
    -- If no cache exists or it's too old, refresh it
    needs_refresh := cache_age_minutes IS NULL OR cache_age_minutes > max_age_minutes;
    
    IF needs_refresh THEN
        PERFORM refresh_cluster_analytics_cache();
        RETURN TRUE;
    END IF;
    
    RETURN FALSE;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Failed to check cluster analytics cache freshness: %', SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Update the semantic_memory_view to include clustering information
DROP VIEW IF EXISTS semantic_memory_view;
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
    
    -- Clustering information
    ce.memory_cluster_id,
    ce.cluster_assignment_timestamp,
    ce.cluster_confidence_score,
    mc.cluster_name,
    mc.cluster_theme,
    
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
    mpc.name_retention_strength,
    mpc.memory_clustering_preference
    
FROM conversation_exchanges ce
LEFT JOIN emotional_context ec ON ce.exchange_id = ec.exchange_id
LEFT JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
LEFT JOIN npc_personalities np ON cc.npc_id = np.npc_id
LEFT JOIN memory_personality_config mpc ON cc.npc_id = mpc.npc_id
LEFT JOIN memory_clusters mc ON ce.memory_cluster_id = mc.cluster_id
WHERE ce.confidence_score > 0.0; -- Only include memories with some confidence

-- Add clustering support to memory statistics view
DROP VIEW IF EXISTS memory_statistics;
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
    
    -- Clustering statistics
    COUNT(CASE WHEN ce.memory_cluster_id IS NOT NULL THEN 1 END) as clustered_memories,
    COUNT(DISTINCT ce.memory_cluster_id) as unique_clusters,
    AVG(ce.cluster_confidence_score) as avg_cluster_confidence,
    
    -- Age distribution
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 1 THEN 1 END) as memories_last_day,
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 7 THEN 1 END) as memories_last_week,
    COUNT(CASE WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0 < 30 THEN 1 END) as memories_last_month
    
FROM conversation_exchanges ce
LEFT JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
GROUP BY cc.npc_id;

-- Create trigger function to automatically update cluster statistics when exchanges are modified (thread-safe)
CREATE OR REPLACE FUNCTION trigger_update_cluster_statistics()
RETURNS TRIGGER AS $$
DECLARE
    old_cluster_id INTEGER;
    new_cluster_id INTEGER;
BEGIN
    -- Handle INSERT operations (OLD doesn't exist)
    IF TG_OP = 'INSERT' THEN
        IF NEW.memory_cluster_id IS NOT NULL THEN
            BEGIN
                PERFORM update_cluster_statistics(NEW.memory_cluster_id);
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'Failed to update statistics for new cluster %: %', NEW.memory_cluster_id, SQLERRM;
            END;
        END IF;
        RETURN NEW;
    END IF;
    
    -- Handle UPDATE operations
    IF TG_OP = 'UPDATE' THEN
        old_cluster_id := OLD.memory_cluster_id;
        new_cluster_id := NEW.memory_cluster_id;
        
        -- Update statistics for old cluster if it exists and is changing
        IF old_cluster_id IS NOT NULL AND (new_cluster_id IS NULL OR old_cluster_id != new_cluster_id) THEN
            BEGIN
                PERFORM update_cluster_statistics(old_cluster_id);
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'Failed to update statistics for old cluster %: %', old_cluster_id, SQLERRM;
            END;
        END IF;
        
        -- Update statistics for new cluster if it exists and is different from old
        IF new_cluster_id IS NOT NULL AND (old_cluster_id IS NULL OR new_cluster_id != old_cluster_id) THEN
            BEGIN
                PERFORM update_cluster_statistics(new_cluster_id);
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'Failed to update statistics for new cluster %: %', new_cluster_id, SQLERRM;
            END;
        END IF;
        
        RETURN NEW;
    END IF;
    
    -- Handle DELETE operations
    IF TG_OP = 'DELETE' THEN
        IF OLD.memory_cluster_id IS NOT NULL THEN
            BEGIN
                PERFORM update_cluster_statistics(OLD.memory_cluster_id);
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'Failed to update statistics for deleted cluster %: %', OLD.memory_cluster_id, SQLERRM;
            END;
        END IF;
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create clustering trigger function to automatically assign new memories
CREATE OR REPLACE FUNCTION trigger_auto_cluster_assignment()
RETURNS TRIGGER AS $$
DECLARE
    npc_id_val UUID;
    unassigned_count INTEGER;
    clustering_threshold INTEGER := 5; -- Trigger clustering after N new memories
BEGIN
    -- Only process if this is a new memory with embedding
    IF TG_OP = 'INSERT' AND NEW.memory_embedding IS NOT NULL AND NEW.confidence_score > 0.1 THEN
        -- Get NPC ID for this memory
        SELECT cc.npc_id INTO npc_id_val
        FROM conversation_contexts cc
        WHERE cc.conversation_id = NEW.conversation_id;
        
        -- Check if we have enough unassigned memories to trigger clustering
        SELECT COUNT(*) INTO unassigned_count
        FROM conversation_exchanges ce
        JOIN conversation_contexts cc ON ce.conversation_id = cc.conversation_id
        WHERE cc.npc_id = npc_id_val
            AND ce.memory_embedding IS NOT NULL
            AND ce.memory_cluster_id IS NULL
            AND ce.confidence_score > 0.1;
        
        -- If we've accumulated enough unassigned memories, trigger clustering
        IF unassigned_count >= clustering_threshold THEN
            BEGIN
                -- Asynchronously trigger clustering (non-blocking)
                PERFORM assign_memory_to_clusters(npc_id_val, 10, 0.3, 3, TRUE);
                RAISE NOTICE 'Auto-clustering triggered for NPC % after % unassigned memories', npc_id_val, unassigned_count;
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'Auto-clustering failed for NPC %: %', npc_id_val, SQLERRM;
            END;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic cluster assignment (can be enabled/disabled)
-- Uncomment to enable automatic clustering:
-- CREATE TRIGGER trigger_auto_clustering
--     AFTER INSERT ON conversation_exchanges
--     FOR EACH ROW
--     EXECUTE FUNCTION trigger_auto_cluster_assignment();

-- Create trigger for cluster statistics (can be enabled/disabled for performance)
-- CREATE TRIGGER trigger_cluster_statistics_update
--     AFTER UPDATE OF memory_cluster_id, cluster_confidence_score ON conversation_exchanges
--     FOR EACH ROW
--     EXECUTE FUNCTION trigger_update_cluster_statistics();

-- Create function for batch cluster management with advanced features
CREATE OR REPLACE FUNCTION manage_memory_clusters(
    npc_id_param UUID DEFAULT NULL,
    operation TEXT DEFAULT 'full_clustering', -- 'full_clustering', 'reassignment_only', 'cleanup_only'
    max_clusters INTEGER DEFAULT 10,
    similarity_threshold DECIMAL DEFAULT 0.3,
    enable_cleanup BOOLEAN DEFAULT TRUE
) RETURNS TABLE (
    operation_type TEXT,
    npc_id UUID,
    memories_processed INTEGER,
    clusters_created INTEGER,
    clusters_removed INTEGER,
    processing_duration INTERVAL,
    performance_notes TEXT
) AS $$
DECLARE
    npc_record RECORD;
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    processed_count INTEGER;
    initial_cluster_count INTEGER;
    final_cluster_count INTEGER;
    removed_clusters INTEGER := 0;
    performance_notes_val TEXT;
BEGIN
    -- Process specific NPC or all NPCs
    FOR npc_record IN
        SELECT DISTINCT cc.npc_id
        FROM conversation_contexts cc
        JOIN conversation_exchanges ce ON cc.conversation_id = ce.conversation_id
        WHERE (npc_id_param IS NULL OR cc.npc_id = npc_id_param)
            AND ce.memory_embedding IS NOT NULL
            AND (
                (operation = 'full_clustering' AND ce.memory_cluster_id IS NULL) OR
                (operation = 'reassignment_only' AND ce.memory_cluster_id IS NOT NULL) OR
                (operation = 'cleanup_only')
            )
            AND ce.confidence_score > 0.1
    LOOP
        start_time := clock_timestamp();
        processed_count := 0;
        performance_notes_val := '';
        
        -- Count initial clusters
        SELECT COUNT(*) INTO initial_cluster_count
        FROM memory_clusters
        WHERE memory_clusters.npc_id = npc_record.npc_id;
        
        -- Execute requested operation
        IF operation IN ('full_clustering', 'reassignment_only') THEN
            processed_count := assign_memory_to_clusters(
                npc_record.npc_id, 
                max_clusters, 
                similarity_threshold,
                5, -- max iterations
                operation = 'reassignment_only' -- enable reassignment for reassignment_only mode
            );
            
            performance_notes_val := format('Processed %s memories', processed_count);
        END IF;
        
        -- Cleanup empty clusters if enabled
        IF enable_cleanup THEN
            removed_clusters := cleanup_empty_clusters();
            performance_notes_val := performance_notes_val || format(', removed %s empty clusters', removed_clusters);
        END IF;
        
        -- Update centroids and refresh cache
        PERFORM update_cluster_centroids(npc_record.npc_id);
        PERFORM refresh_cache_if_needed(50.0, 30); -- Refresh if performance degrades
        
        -- Count final clusters
        SELECT COUNT(*) INTO final_cluster_count
        FROM memory_clusters
        WHERE memory_clusters.npc_id = npc_record.npc_id;
        
        end_time := clock_timestamp();
        
        -- Return results for this NPC
        RETURN QUERY
        SELECT 
            operation,
            npc_record.npc_id,
            processed_count,
            final_cluster_count - initial_cluster_count,
            removed_clusters,
            end_time - start_time,
            performance_notes_val;
    END LOOP;
    
    RETURN;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Memory cluster management failed: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Add table comments
COMMENT ON TABLE memory_clusters IS 'Metadata and centroids for memory clusters, organizing semantically related memories';
COMMENT ON FUNCTION update_cluster_statistics IS 'Updates cluster metadata statistics based on current member exchanges (thread-safe with row-level locking)';
COMMENT ON FUNCTION cleanup_empty_clusters IS 'Removes clusters that no longer have any member exchanges';
COMMENT ON FUNCTION calculate_optimal_vector_lists IS 'Calculates optimal lists parameter for pgvector ivfflat indexes based on table size';
COMMENT ON FUNCTION rebuild_vector_index IS 'Rebuilds vector index with optimal lists parameter for current data size';
COMMENT ON FUNCTION assign_memory_to_clusters IS 'Enhanced K-means clustering with iterative optimization, confidence scoring, and automatic cluster reassignment';
COMMENT ON FUNCTION update_cluster_centroids IS 'Recalculates cluster centroids based on current member embeddings';
COMMENT ON FUNCTION process_all_npc_clustering IS 'Batch processes memory clustering for all NPCs with unassigned memories';
COMMENT ON FUNCTION refresh_cluster_analytics_cache IS 'Refreshes materialized view cache for cluster analytics with performance monitoring';
COMMENT ON FUNCTION ensure_fresh_cluster_analytics IS 'Ensures cluster analytics cache is fresh, auto-refreshing if needed';
COMMENT ON VIEW memory_cluster_analysis IS 'Comprehensive analysis view for memory cluster health and statistics';
COMMENT ON MATERIALIZED VIEW cluster_detailed_analytics_cache IS 'Cached expensive cluster analytics calculations for performance optimization';
COMMENT ON FUNCTION test_cluster_analysis_performance IS 'Performance testing function for cluster analysis views with configurable thresholds';
COMMENT ON FUNCTION refresh_cache_if_needed IS 'Intelligent cache refresh based on performance degradation and age thresholds';
COMMENT ON FUNCTION trigger_auto_cluster_assignment IS 'Automatic clustering trigger for new memories with configurable batch thresholds';
COMMENT ON FUNCTION manage_memory_clusters IS 'Comprehensive cluster management with full clustering, reassignment, and cleanup operations';

-- Rollback instructions (commented for reference)
-- To rollback this migration:
-- DROP VIEW IF EXISTS memory_cluster_analysis;
-- DROP VIEW IF EXISTS cluster_basic_metrics;
-- DROP VIEW IF EXISTS cluster_detailed_analytics;
-- DROP MATERIALIZED VIEW IF EXISTS cluster_detailed_analytics_cache;
-- DROP VIEW IF EXISTS memory_statistics;  
-- DROP VIEW IF EXISTS semantic_memory_view;
-- DROP FUNCTION IF EXISTS get_cluster_health_summary();
-- DROP FUNCTION IF EXISTS refresh_cluster_analytics_cache();
-- DROP FUNCTION IF EXISTS ensure_fresh_cluster_analytics(INTEGER);
-- DROP FUNCTION IF EXISTS cleanup_empty_clusters();
-- DROP FUNCTION IF EXISTS update_cluster_statistics(INTEGER);
-- DROP FUNCTION IF EXISTS trigger_update_cluster_statistics();
-- DROP FUNCTION IF EXISTS assign_memory_to_clusters(UUID, INTEGER, DECIMAL);
-- DROP FUNCTION IF EXISTS update_cluster_centroids(UUID);
-- DROP FUNCTION IF EXISTS process_all_npc_clustering();
-- DROP FUNCTION IF EXISTS calculate_optimal_vector_lists(TEXT, INTEGER);
-- DROP FUNCTION IF EXISTS rebuild_vector_index(TEXT, TEXT, TEXT, TEXT);
-- ALTER TABLE conversation_exchanges DROP CONSTRAINT IF EXISTS fk_memory_cluster_id;
-- DROP TABLE IF EXISTS memory_clusters;
-- DROP INDEX IF EXISTS idx_exchanges_conversation_cluster;
-- DROP INDEX IF EXISTS idx_exchanges_cluster_timestamp;
-- DROP INDEX IF EXISTS idx_exchanges_cluster_performance;
-- DROP INDEX IF EXISTS idx_exchanges_npc_cluster_confidence;
-- ALTER TABLE conversation_exchanges DROP COLUMN IF EXISTS cluster_confidence_score;
-- ALTER TABLE conversation_exchanges DROP COLUMN IF EXISTS cluster_assignment_timestamp;
-- ALTER TABLE conversation_exchanges DROP COLUMN IF EXISTS memory_cluster_id;

-- Update migration tracking (if exists)
-- INSERT INTO schema_migrations (version, applied_at) VALUES ('035', CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING;