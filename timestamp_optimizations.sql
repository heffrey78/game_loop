-- TASK-0086: Timestamp and Date Arithmetic Optimizations
-- Optimize expensive date calculations for sub-50ms cluster analysis performance

-- 1. Pre-computed timestamp buckets for efficient age calculations
CREATE OR REPLACE FUNCTION create_timestamp_buckets()
RETURNS VOID AS $$
BEGIN
    -- Create a lookup table for common time intervals
    CREATE TEMP TABLE IF NOT EXISTS timestamp_buckets AS
    SELECT 
        interval_name,
        interval_value,
        CURRENT_TIMESTAMP - interval_value as cutoff_time
    FROM (VALUES
        ('1_hour', '1 hour'::INTERVAL),
        ('6_hours', '6 hours'::INTERVAL), 
        ('1_day', '1 day'::INTERVAL),
        ('7_days', '7 days'::INTERVAL),
        ('30_days', '30 days'::INTERVAL),
        ('90_days', '90 days'::INTERVAL),
        ('1_year', '1 year'::INTERVAL)
    ) as intervals(interval_name, interval_value);
END;
$$ LANGUAGE plpgsql;

-- 2. Optimized age calculation function using epoch differences
CREATE OR REPLACE FUNCTION calculate_age_bucket(timestamp_value TIMESTAMP WITH TIME ZONE)
RETURNS TEXT AS $$
DECLARE
    seconds_diff BIGINT;
    days_diff DECIMAL;
BEGIN
    -- Use epoch subtraction (faster than interval arithmetic)
    seconds_diff := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - timestamp_value));
    days_diff := seconds_diff / 86400.0;
    
    -- Return age bucket for efficient grouping
    CASE 
        WHEN days_diff < 1 THEN RETURN 'today'
        WHEN days_diff < 7 THEN RETURN 'this_week'
        WHEN days_diff < 30 THEN RETURN 'this_month'
        WHEN days_diff < 90 THEN RETURN 'this_quarter'
        WHEN days_diff < 365 THEN RETURN 'this_year'
        ELSE RETURN 'older'
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 3. Fast timestamp range queries using indexes
CREATE INDEX IF NOT EXISTS idx_conversation_exchanges_timestamp_buckets
ON conversation_exchanges (
    CASE 
        WHEN timestamp > CURRENT_TIMESTAMP - INTERVAL '1 day' THEN 'today'
        WHEN timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 'week'  
        WHEN timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 'month'
        ELSE 'older'
    END,
    timestamp
);

-- 4. Materialized view for pre-computed age metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS cluster_age_metrics AS
SELECT 
    ce.memory_cluster_id as cluster_id,
    
    -- Pre-computed age statistics using efficient epoch calculations
    COUNT(*) as total_memories,
    MIN(EXTRACT(EPOCH FROM ce.timestamp)) as oldest_epoch,
    MAX(EXTRACT(EPOCH FROM ce.timestamp)) as newest_epoch,
    AVG(EXTRACT(EPOCH FROM ce.timestamp)) as avg_epoch,
    
    -- Current epoch for age calculations
    EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) as current_epoch,
    
    -- Pre-computed age values in days (avoid repeated calculations)
    (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) - MAX(EXTRACT(EPOCH FROM ce.timestamp))) / 86400.0 as newest_age_days,
    (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) - MIN(EXTRACT(EPOCH FROM ce.timestamp))) / 86400.0 as oldest_age_days,
    (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) - AVG(EXTRACT(EPOCH FROM ce.timestamp))) / 86400.0 as avg_age_days,
    
    -- Bucketed counts for fast filtering
    COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 day' THEN 1 END) as count_today,
    COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as count_week,
    COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 1 END) as count_month,
    COUNT(CASE WHEN ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '90 days' THEN 1 END) as count_quarter,
    
    -- Cache timestamp for freshness tracking
    CURRENT_TIMESTAMP as computed_at
    
FROM conversation_exchanges ce
WHERE ce.memory_cluster_id IS NOT NULL
GROUP BY ce.memory_cluster_id;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_cluster_age_metrics_cluster_id 
ON cluster_age_metrics(cluster_id);

CREATE INDEX IF NOT EXISTS idx_cluster_age_metrics_computed_at
ON cluster_age_metrics(computed_at);

-- 5. Ultra-fast cluster age summary view
CREATE OR REPLACE VIEW cluster_age_summary_fast AS
SELECT 
    cluster_id,
    total_memories,
    
    -- Use pre-computed age values (no real-time calculations)
    newest_age_days,
    oldest_age_days, 
    avg_age_days,
    
    -- Activity categories based on pre-computed counts
    count_today,
    count_week,
    count_month,
    count_quarter,
    (total_memories - count_quarter) as count_older,
    
    -- Activity status using pre-computed values
    CASE 
        WHEN count_today > 0 THEN 'very_active'
        WHEN count_week > 0 THEN 'active'
        WHEN count_month > 0 THEN 'moderate'
        WHEN count_quarter > 0 THEN 'low'
        ELSE 'inactive'
    END as activity_level,
    
    -- Freshness of these calculations
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - computed_at)) / 3600.0 as cache_age_hours,
    
    CASE 
        WHEN computed_at > CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 'fresh'
        WHEN computed_at > CURRENT_TIMESTAMP - INTERVAL '6 hours' THEN 'stale'  
        ELSE 'outdated'
    END as cache_status
    
FROM cluster_age_metrics;

-- 6. Efficient timestamp range query function
CREATE OR REPLACE FUNCTION get_clusters_by_activity(
    activity_level TEXT DEFAULT 'active',
    limit_count INTEGER DEFAULT 50
) RETURNS TABLE (
    cluster_id INTEGER,
    activity_score DECIMAL,
    recent_activity_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cam.cluster_id,
        -- Activity score based on recency weighting
        (cam.count_today * 4.0 + cam.count_week * 2.0 + cam.count_month * 1.0)::DECIMAL as activity_score,
        CASE activity_level
            WHEN 'very_active' THEN cam.count_today
            WHEN 'active' THEN cam.count_week
            WHEN 'moderate' THEN cam.count_month
            WHEN 'low' THEN cam.count_quarter
            ELSE cam.total_memories
        END as recent_activity_count
    FROM cluster_age_metrics cam
    WHERE 
        CASE activity_level
            WHEN 'very_active' THEN cam.count_today > 0
            WHEN 'active' THEN cam.count_week > 0  
            WHEN 'moderate' THEN cam.count_month > 0
            WHEN 'low' THEN cam.count_quarter > 0
            ELSE TRUE
        END
    ORDER BY activity_score DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 7. Optimized cluster analysis with fast timestamps
CREATE OR REPLACE VIEW memory_cluster_analysis_optimized AS
SELECT 
    mc.cluster_id,
    mc.npc_id,
    mc.cluster_name,
    mc.cluster_theme,
    mc.created_at,
    mc.updated_at,
    
    -- Use incremental statistics (no calculations)
    mc.member_count,
    mc.avg_confidence, 
    mc.recent_memories_7d,
    mc.recent_memories_30d,
    mc.stats_last_updated,
    
    -- Use pre-computed age metrics (no timestamp arithmetic)
    cam.avg_age_days,
    cam.newest_age_days,
    cam.oldest_age_days,
    
    -- Activity level from pre-computed buckets
    CASE 
        WHEN cam.count_today > 0 THEN 'very_active'
        WHEN cam.count_week > 0 THEN 'active'
        WHEN cam.count_month > 0 THEN 'moderate'
        WHEN cam.count_quarter > 0 THEN 'low'
        ELSE 'inactive'
    END as activity_level,
    
    -- Health assessment using all pre-computed values
    CASE 
        WHEN mc.member_count = 0 THEN 'empty'
        WHEN mc.avg_confidence < 0.3 THEN 'low_confidence'
        WHEN cam.count_week = 0 AND cam.avg_age_days > 30 THEN 'dormant'
        WHEN mc.avg_confidence > 0.7 AND cam.count_week > 0 THEN 'excellent'
        WHEN mc.avg_confidence > 0.5 THEN 'good'
        ELSE 'needs_attention'
    END as overall_health,
    
    -- Cache status indicators
    CASE 
        WHEN mc.stats_last_updated < CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN TRUE
        ELSE FALSE
    END as stats_need_refresh,
    
    CASE 
        WHEN cam.computed_at < CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN TRUE
        ELSE FALSE  
    END as age_metrics_need_refresh
    
FROM memory_clusters mc
LEFT JOIN cluster_age_metrics cam ON mc.cluster_id = cam.cluster_id;

-- 8. Refresh age metrics function with performance monitoring
CREATE OR REPLACE FUNCTION refresh_cluster_age_metrics()
RETURNS TABLE (
    refresh_duration_ms DECIMAL,
    clusters_processed INTEGER,
    cache_status TEXT
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    cluster_count INTEGER;
BEGIN
    start_time := clock_timestamp();
    
    -- Refresh the materialized view
    REFRESH MATERIALIZED VIEW cluster_age_metrics;
    
    -- Get count of processed clusters
    SELECT COUNT(*) INTO cluster_count FROM cluster_age_metrics;
    
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        duration_ms,
        cluster_count,
        CASE 
            WHEN duration_ms < 100 THEN 'excellent'
            WHEN duration_ms < 500 THEN 'good'
            WHEN duration_ms < 1000 THEN 'acceptable'
            ELSE 'needs_optimization'
        END;
END;
$$ LANGUAGE plpgsql;

-- 9. Performance comparison for timestamp optimizations
CREATE OR REPLACE FUNCTION compare_timestamp_performance(
    sample_size INTEGER DEFAULT 10,
    iterations INTEGER DEFAULT 3
) RETURNS TABLE (
    method TEXT,
    avg_time_ms DECIMAL,
    performance_category TEXT
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    total_time DECIMAL;
    i INTEGER;
BEGIN
    -- Test 1: Real-time timestamp calculations (slow)
    total_time := 0;
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM 
            memory_cluster_id,
            AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - timestamp)) / 86400.0),
            COUNT(CASE WHEN timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END)
        FROM conversation_exchanges
        WHERE memory_cluster_id IS NOT NULL
        GROUP BY memory_cluster_id
        LIMIT sample_size;
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
    END LOOP;
    
    RETURN QUERY SELECT 
        'realtime_timestamp_calc'::TEXT,
        total_time / iterations,
        CASE 
            WHEN (total_time / iterations) > 100 THEN 'slow'
            WHEN (total_time / iterations) > 50 THEN 'moderate'
            ELSE 'fast'
        END;
    
    -- Test 2: Pre-computed age metrics (fast)
    total_time := 0;
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM 
            cluster_id,
            avg_age_days,
            count_week
        FROM cluster_age_metrics
        LIMIT sample_size;
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
    END LOOP;
    
    RETURN QUERY SELECT 
        'precomputed_age_metrics'::TEXT,
        total_time / iterations,
        CASE 
            WHEN (total_time / iterations) > 10 THEN 'moderate'
            WHEN (total_time / iterations) > 5 THEN 'fast'
            ELSE 'excellent'
        END;
    
    -- Test 3: Optimized analysis view
    total_time := 0;
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM *
        FROM memory_cluster_analysis_optimized
        LIMIT sample_size;
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
    END LOOP;
    
    RETURN QUERY SELECT 
        'optimized_analysis_view'::TEXT,
        total_time / iterations,
        CASE 
            WHEN (total_time / iterations) > 50 THEN 'moderate'
            WHEN (total_time / iterations) > 25 THEN 'fast'
            ELSE 'excellent'
        END;
        
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON MATERIALIZED VIEW cluster_age_metrics IS 'Pre-computed age and timestamp metrics for sub-50ms cluster analysis performance';
COMMENT ON VIEW cluster_age_summary_fast IS 'Ultra-fast cluster age summary using pre-computed epoch differences';
COMMENT ON VIEW memory_cluster_analysis_optimized IS 'Fully optimized cluster analysis with pre-computed statistics and age metrics';
COMMENT ON FUNCTION refresh_cluster_age_metrics IS 'Refresh materialized age metrics with performance monitoring';
COMMENT ON FUNCTION compare_timestamp_performance IS 'Performance comparison for timestamp calculation optimizations';