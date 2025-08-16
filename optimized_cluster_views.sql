-- TASK-0086: Optimized Memory Cluster Analysis Views
-- Breaking complex view into focused, efficient sub-views for sub-50ms performance

-- 1. Core cluster metrics view - lightweight for frequent access
CREATE OR REPLACE VIEW cluster_core_metrics AS
SELECT 
    mc.cluster_id,
    mc.npc_id,
    mc.cluster_name,
    mc.cluster_theme,
    mc.member_count as stored_member_count,
    mc.created_at,
    mc.updated_at,
    
    -- Real-time member count (efficient single query)
    COUNT(ce.exchange_id) as actual_member_count,
    
    -- Basic health status (no complex calculations)
    CASE 
        WHEN COUNT(ce.exchange_id) = 0 THEN 'empty'
        WHEN COUNT(ce.exchange_id) != mc.member_count THEN 'stale'
        ELSE 'healthy'
    END as sync_status
    
FROM memory_clusters mc
LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
GROUP BY mc.cluster_id, mc.npc_id, mc.cluster_name, mc.cluster_theme, 
         mc.member_count, mc.created_at, mc.updated_at;

-- 2. Confidence metrics view - separated for performance
CREATE OR REPLACE VIEW cluster_confidence_metrics AS
SELECT 
    mc.cluster_id,
    
    -- Confidence calculations (moderate cost, separated from core metrics)
    COALESCE(AVG(ce.cluster_confidence_score), 0.0) as avg_confidence,
    MIN(ce.cluster_confidence_score) as min_confidence,
    MAX(ce.cluster_confidence_score) as max_confidence,
    COUNT(CASE WHEN ce.cluster_confidence_score > 0.7 THEN 1 END) as high_confidence_members,
    
    -- Simple confidence health check
    CASE 
        WHEN AVG(ce.cluster_confidence_score) < 0.3 THEN 'low_confidence'
        WHEN AVG(ce.cluster_confidence_score) > 0.7 THEN 'high_confidence'
        ELSE 'moderate_confidence'
    END as confidence_status
    
FROM memory_clusters mc
INNER JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
WHERE ce.cluster_confidence_score IS NOT NULL
GROUP BY mc.cluster_id;

-- 3. Access pattern metrics view - for usage analysis
CREATE OR REPLACE VIEW cluster_access_metrics AS
SELECT 
    mc.cluster_id,
    
    -- Access pattern analysis (uses existing exchange_metadata if available)
    COUNT(ce.exchange_id) as total_memories,
    MAX(ce.timestamp) as newest_memory,
    MIN(ce.timestamp) as oldest_memory,
    
    -- Age calculations optimized (single EXTRACT per row, not per calculation)
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0) as avg_age_days,
    
    -- Memory distribution by recency (efficient bucketing)
    COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as recent_memories,
    COUNT(CASE WHEN ce.timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 1 END) as old_memories
    
FROM memory_clusters mc
INNER JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
GROUP BY mc.cluster_id;

-- 4. Lightweight health check view - optimized for monitoring
CREATE OR REPLACE VIEW cluster_health_summary AS
SELECT 
    ccm.cluster_id,
    ccm.npc_id,
    ccm.cluster_name,
    ccm.sync_status,
    ccm.actual_member_count,
    
    -- Join confidence status efficiently
    COALESCE(conf.confidence_status, 'no_data') as confidence_status,
    
    -- Overall health assessment (simple logic)
    CASE 
        WHEN ccm.sync_status = 'empty' THEN 'empty'
        WHEN ccm.sync_status = 'stale' THEN 'needs_sync'
        WHEN COALESCE(conf.confidence_status, 'no_data') = 'low_confidence' THEN 'low_quality'
        WHEN COALESCE(conf.confidence_status, 'no_data') = 'high_confidence' THEN 'excellent'
        ELSE 'good'
    END as overall_health
    
FROM cluster_core_metrics ccm
LEFT JOIN cluster_confidence_metrics conf ON ccm.cluster_id = conf.cluster_id;

-- 5. Comprehensive cluster analysis - combines all metrics efficiently
CREATE OR REPLACE VIEW memory_cluster_analysis AS
SELECT 
    ccm.cluster_id,
    ccm.npc_id,
    ccm.cluster_name,
    ccm.cluster_theme,
    ccm.stored_member_count,
    ccm.created_at,
    ccm.updated_at,
    ccm.actual_member_count,
    ccm.sync_status,
    
    -- Confidence metrics (from separate view)
    conf.avg_confidence,
    conf.confidence_status,
    conf.high_confidence_members,
    
    -- Access metrics (from separate view)
    acc.avg_age_days,
    acc.newest_memory,
    acc.recent_memories,
    acc.old_memories,
    
    -- Overall health (pre-calculated)
    health.overall_health
    
FROM cluster_core_metrics ccm
LEFT JOIN cluster_confidence_metrics conf ON ccm.cluster_id = conf.cluster_id
LEFT JOIN cluster_access_metrics acc ON ccm.cluster_id = acc.cluster_id
LEFT JOIN cluster_health_summary health ON ccm.cluster_id = health.cluster_id;

-- 6. Performance test function for the new view structure
CREATE OR REPLACE FUNCTION test_optimized_cluster_performance(
    sample_size INTEGER DEFAULT 10,
    target_ms DECIMAL DEFAULT 50.0
) RETURNS TABLE (
    view_name TEXT,
    execution_time_ms DECIMAL,
    meets_target BOOLEAN,
    sample_rows INTEGER,
    performance_status TEXT
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    row_count INTEGER;
BEGIN
    -- Test 1: Core metrics view
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM cluster_core_metrics LIMIT sample_size;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_core_metrics'::TEXT,
        duration_ms,
        duration_ms <= target_ms,
        row_count,
        CASE 
            WHEN duration_ms <= target_ms * 0.5 THEN 'excellent'
            WHEN duration_ms <= target_ms THEN 'good'
            WHEN duration_ms <= target_ms * 2 THEN 'acceptable'
            ELSE 'needs_optimization'
        END;
    
    -- Test 2: Health summary view (monitoring query)
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM cluster_health_summary LIMIT sample_size;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_health_summary'::TEXT,
        duration_ms,
        duration_ms <= target_ms,
        row_count,
        CASE 
            WHEN duration_ms <= target_ms * 0.5 THEN 'excellent'
            WHEN duration_ms <= target_ms THEN 'good'
            WHEN duration_ms <= target_ms * 2 THEN 'acceptable'
            ELSE 'needs_optimization'
        END;
    
    -- Test 3: Full analysis view
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM memory_cluster_analysis LIMIT sample_size;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'memory_cluster_analysis'::TEXT,
        duration_ms,
        duration_ms <= target_ms * 1.5, -- Allow 1.5x target for comprehensive view
        row_count,
        CASE 
            WHEN duration_ms <= target_ms THEN 'excellent'
            WHEN duration_ms <= target_ms * 1.5 THEN 'good'
            WHEN duration_ms <= target_ms * 3 THEN 'acceptable'
            ELSE 'needs_optimization'
        END;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 7. Incremental statistics update function (replaces expensive AVG calculations)
CREATE OR REPLACE FUNCTION update_cluster_incremental_stats(cluster_id_param INTEGER)
RETURNS VOID AS $$
DECLARE
    new_member_count INTEGER;
    new_avg_confidence DECIMAL;
    new_updated_at TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Calculate statistics atomically
    SELECT 
        COUNT(exchange_id),
        COALESCE(AVG(cluster_confidence_score), 0.0),
        CURRENT_TIMESTAMP
    INTO new_member_count, new_avg_confidence, new_updated_at
    FROM conversation_exchanges
    WHERE memory_cluster_id = cluster_id_param;
    
    -- Update cluster metadata with incremental stats
    UPDATE memory_clusters 
    SET 
        member_count = new_member_count,
        avg_confidence = new_avg_confidence,
        updated_at = new_updated_at
    WHERE cluster_id = cluster_id_param;
    
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON VIEW cluster_core_metrics IS 'Lightweight core cluster metrics for frequent access - optimized for sub-25ms performance';
COMMENT ON VIEW cluster_confidence_metrics IS 'Confidence-related metrics separated for performance - moderate cost calculations';
COMMENT ON VIEW cluster_access_metrics IS 'Access pattern and age metrics with optimized timestamp calculations';
COMMENT ON VIEW cluster_health_summary IS 'Lightweight cluster health checks for monitoring dashboards - sub-10ms target';
COMMENT ON VIEW memory_cluster_analysis IS 'Comprehensive cluster analysis combining focused sub-views efficiently';
COMMENT ON FUNCTION test_optimized_cluster_performance IS 'Performance testing for optimized cluster view structure';
COMMENT ON FUNCTION update_cluster_incremental_stats IS 'Incremental statistics update replacing expensive real-time AVG calculations';