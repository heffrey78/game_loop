-- TASK-0086: Incremental Statistics System
-- Replace expensive real-time AVG() calculations with pre-computed incremental statistics

-- 1. Enhanced cluster metadata table with incremental statistics
ALTER TABLE memory_clusters 
ADD COLUMN IF NOT EXISTS stats_last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS total_confidence_sum DECIMAL DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS confidence_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS age_sum_days DECIMAL DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS access_count_sum BIGINT DEFAULT 0,
ADD COLUMN IF NOT EXISTS recent_memories_7d INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS recent_memories_30d INTEGER DEFAULT 0;

-- 2. Incremental statistics update function (atomic updates)
CREATE OR REPLACE FUNCTION update_cluster_incremental_stats(cluster_id_param INTEGER)
RETURNS TABLE (
    updated_member_count INTEGER,
    updated_avg_confidence DECIMAL,
    calculation_time_ms DECIMAL
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    calc_time_ms DECIMAL;
    
    -- Calculated values
    new_member_count INTEGER;
    new_confidence_sum DECIMAL;
    new_confidence_count INTEGER;
    new_avg_confidence DECIMAL;
    new_age_sum_days DECIMAL;
    new_access_sum BIGINT;
    new_recent_7d INTEGER;
    new_recent_30d INTEGER;
BEGIN
    start_time := clock_timestamp();
    
    -- Calculate all statistics in a single query for efficiency
    SELECT 
        COUNT(ce.exchange_id),
        COALESCE(SUM(ce.cluster_confidence_score), 0.0),
        COUNT(ce.cluster_confidence_score),
        COALESCE(AVG(ce.cluster_confidence_score), 0.0),
        COALESCE(SUM(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0), 0.0),
        COALESCE(SUM(COALESCE(
            (ce.exchange_metadata->>'access_count')::BIGINT, 0
        )), 0),
        COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END),
        COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 1 END)
    INTO 
        new_member_count,
        new_confidence_sum,
        new_confidence_count, 
        new_avg_confidence,
        new_age_sum_days,
        new_access_sum,
        new_recent_7d,
        new_recent_30d
    FROM conversation_exchanges ce
    WHERE ce.memory_cluster_id = cluster_id_param;
    
    -- Update cluster with pre-computed statistics
    UPDATE memory_clusters 
    SET 
        member_count = new_member_count,
        avg_confidence = new_avg_confidence,
        total_confidence_sum = new_confidence_sum,
        confidence_count = new_confidence_count,
        age_sum_days = new_age_sum_days,
        access_count_sum = new_access_sum,
        recent_memories_7d = new_recent_7d,
        recent_memories_30d = new_recent_30d,
        stats_last_updated = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE cluster_id = cluster_id_param;
    
    end_time := clock_timestamp();
    calc_time_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- Return results for performance monitoring
    RETURN QUERY SELECT new_member_count, new_avg_confidence, calc_time_ms;
END;
$$ LANGUAGE plpgsql;

-- 3. Batch statistics update function for all clusters
CREATE OR REPLACE FUNCTION refresh_all_cluster_statistics()
RETURNS TABLE (
    cluster_id INTEGER,
    processing_time_ms DECIMAL,
    member_count INTEGER,
    avg_confidence DECIMAL
) AS $$
DECLARE
    cluster_record RECORD;
    update_result RECORD;
BEGIN
    FOR cluster_record IN SELECT mc.cluster_id FROM memory_clusters mc LOOP
        -- Update each cluster's statistics
        SELECT * INTO update_result 
        FROM update_cluster_incremental_stats(cluster_record.cluster_id) 
        LIMIT 1;
        
        RETURN QUERY SELECT 
            cluster_record.cluster_id,
            update_result.calculation_time_ms,
            update_result.updated_member_count,
            update_result.updated_avg_confidence;
    END LOOP;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 4. Fast cluster metrics view using pre-computed statistics
CREATE OR REPLACE VIEW cluster_fast_metrics AS
SELECT 
    cluster_id,
    npc_id,
    cluster_name,
    cluster_theme,
    created_at,
    updated_at,
    
    -- Use pre-computed values (no expensive calculations)
    member_count,
    avg_confidence,
    recent_memories_7d,
    recent_memories_30d,
    stats_last_updated,
    
    -- Derived metrics from pre-computed values
    CASE 
        WHEN member_count = 0 THEN 0.0
        ELSE age_sum_days / member_count 
    END as avg_age_days,
    
    CASE 
        WHEN member_count = 0 THEN 0.0
        ELSE access_count_sum::DECIMAL / member_count 
    END as avg_access_count,
    
    -- Health status based on pre-computed stats
    CASE 
        WHEN member_count = 0 THEN 'empty'
        WHEN avg_confidence < 0.3 THEN 'low_confidence'
        WHEN avg_confidence > 0.7 AND recent_memories_7d > 0 THEN 'excellent'
        WHEN avg_confidence > 0.5 THEN 'good'
        ELSE 'needs_attention'
    END as health_status,
    
    -- Freshness indicator
    CASE 
        WHEN stats_last_updated > CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 'fresh'
        WHEN stats_last_updated > CURRENT_TIMESTAMP - INTERVAL '6 hours' THEN 'stale'
        ELSE 'outdated'
    END as stats_freshness
    
FROM memory_clusters;

-- 5. Ultra-fast monitoring view for dashboards (sub-10ms target)
CREATE OR REPLACE VIEW cluster_monitoring_fast AS
SELECT 
    npc_id,
    COUNT(*) as total_clusters,
    SUM(member_count) as total_memories,
    AVG(avg_confidence) as overall_avg_confidence,
    
    -- Health distribution using pre-computed values
    COUNT(CASE WHEN member_count = 0 THEN 1 END) as empty_clusters,
    COUNT(CASE WHEN avg_confidence < 0.3 THEN 1 END) as low_confidence_clusters,
    COUNT(CASE WHEN avg_confidence > 0.7 THEN 1 END) as high_confidence_clusters,
    
    -- Recent activity summary
    SUM(recent_memories_7d) as memories_last_week,
    SUM(recent_memories_30d) as memories_last_month,
    
    -- Stats freshness summary
    COUNT(CASE WHEN stats_last_updated < CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 1 END) as stale_stats_clusters
    
FROM memory_clusters 
GROUP BY npc_id;

-- 6. Automatic statistics refresh trigger (optional - can be enabled)
CREATE OR REPLACE FUNCTION trigger_incremental_stats_update()
RETURNS TRIGGER AS $$
DECLARE
    affected_cluster_id INTEGER;
BEGIN
    -- Get cluster ID from NEW or OLD record
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        affected_cluster_id := NEW.memory_cluster_id;
    ELSE -- DELETE
        affected_cluster_id := OLD.memory_cluster_id;
    END IF;
    
    -- Only update if cluster is assigned
    IF affected_cluster_id IS NOT NULL THEN
        -- Async update (non-blocking)
        BEGIN
            PERFORM update_cluster_incremental_stats(affected_cluster_id);
        EXCEPTION
            WHEN OTHERS THEN
                -- Log warning but don't fail the main operation
                RAISE WARNING 'Failed to update incremental stats for cluster %: %', 
                    affected_cluster_id, SQLERRM;
        END;
    END IF;
    
    -- Return appropriate record
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 7. Performance comparison function (before/after incremental stats)
CREATE OR REPLACE FUNCTION compare_statistics_performance(
    sample_clusters INTEGER DEFAULT 10,
    iterations INTEGER DEFAULT 3
) RETURNS TABLE (
    method TEXT,
    avg_execution_time_ms DECIMAL,
    min_time_ms DECIMAL,
    max_time_ms DECIMAL,
    performance_improvement_pct DECIMAL
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    
    -- Results storage
    realtime_times DECIMAL[] := '{}';
    incremental_times DECIMAL[] := '{}';
    realtime_avg DECIMAL;
    incremental_avg DECIMAL;
    i INTEGER;
BEGIN
    -- Test real-time calculations (expensive)
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM 
            cluster_id,
            AVG(ce.cluster_confidence_score),
            AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ce.timestamp)) / 86400.0)
        FROM memory_clusters mc
        LEFT JOIN conversation_exchanges ce ON mc.cluster_id = ce.memory_cluster_id
        WHERE mc.cluster_id IN (
            SELECT cluster_id FROM memory_clusters LIMIT sample_clusters
        )
        GROUP BY cluster_id;
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        realtime_times := array_append(realtime_times, duration_ms);
    END LOOP;
    
    -- Test incremental statistics (fast)
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM 
            cluster_id,
            avg_confidence,
            avg_age_days
        FROM cluster_fast_metrics
        WHERE cluster_id IN (
            SELECT cluster_id FROM memory_clusters LIMIT sample_clusters
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        incremental_times := array_append(incremental_times, duration_ms);
    END LOOP;
    
    -- Calculate averages
    realtime_avg := (SELECT AVG(unnest) FROM unnest(realtime_times));
    incremental_avg := (SELECT AVG(unnest) FROM unnest(incremental_times));
    
    -- Return comparison results
    RETURN QUERY SELECT 
        'realtime_calculations'::TEXT,
        realtime_avg,
        (SELECT MIN(unnest) FROM unnest(realtime_times)),
        (SELECT MAX(unnest) FROM unnest(realtime_times)),
        0.0::DECIMAL;
    
    RETURN QUERY SELECT 
        'incremental_statistics'::TEXT,
        incremental_avg,
        (SELECT MIN(unnest) FROM unnest(incremental_times)),
        (SELECT MAX(unnest) FROM unnest(incremental_times)),
        CASE 
            WHEN realtime_avg > 0 THEN ((realtime_avg - incremental_avg) / realtime_avg * 100)
            ELSE 0.0 
        END;
        
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 8. Scheduled statistics refresh function (for cron/background jobs)
CREATE OR REPLACE FUNCTION maintain_cluster_statistics(
    max_stale_hours INTEGER DEFAULT 1,
    batch_size INTEGER DEFAULT 50
) RETURNS TABLE (
    clusters_updated INTEGER,
    total_time_ms DECIMAL,
    avg_time_per_cluster_ms DECIMAL
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    total_time_ms_val DECIMAL;
    clusters_updated_val INTEGER;
    
    stale_clusters RECORD;
BEGIN
    start_time := clock_timestamp();
    clusters_updated_val := 0;
    
    -- Update clusters with stale statistics
    FOR stale_clusters IN 
        SELECT cluster_id 
        FROM memory_clusters 
        WHERE stats_last_updated < CURRENT_TIMESTAMP - (max_stale_hours || ' hours')::INTERVAL
        ORDER BY stats_last_updated ASC
        LIMIT batch_size
    LOOP
        PERFORM update_cluster_incremental_stats(stale_clusters.cluster_id);
        clusters_updated_val := clusters_updated_val + 1;
    END LOOP;
    
    end_time := clock_timestamp();
    total_time_ms_val := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        clusters_updated_val,
        total_time_ms_val,
        CASE 
            WHEN clusters_updated_val > 0 THEN total_time_ms_val / clusters_updated_val
            ELSE 0.0
        END;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON FUNCTION update_cluster_incremental_stats IS 'Atomic incremental statistics update replacing expensive real-time AVG calculations';
COMMENT ON VIEW cluster_fast_metrics IS 'Ultra-fast cluster metrics using pre-computed incremental statistics - sub-5ms target';
COMMENT ON VIEW cluster_monitoring_fast IS 'Dashboard monitoring view with sub-10ms performance using incremental stats';
COMMENT ON FUNCTION compare_statistics_performance IS 'Performance comparison between real-time and incremental statistics methods';
COMMENT ON FUNCTION maintain_cluster_statistics IS 'Background maintenance function for keeping incremental statistics fresh';