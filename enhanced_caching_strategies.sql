-- TASK-0086: Enhanced Materialized View Refresh Strategies and Caching
-- Intelligent caching system for optimal memory cluster analysis performance

-- 1. Cache management metadata table
CREATE TABLE IF NOT EXISTS cluster_cache_metadata (
    cache_name VARCHAR(100) PRIMARY KEY,
    last_refresh TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    refresh_duration_ms DECIMAL NOT NULL DEFAULT 0,
    row_count INTEGER NOT NULL DEFAULT 0,
    cache_size_bytes BIGINT,
    refresh_strategy VARCHAR(50) NOT NULL DEFAULT 'manual',
    auto_refresh_interval INTERVAL,
    performance_threshold_ms DECIMAL DEFAULT 50.0,
    staleness_tolerance_minutes INTEGER DEFAULT 60,
    last_performance_test TIMESTAMP WITH TIME ZONE,
    performance_status VARCHAR(20) DEFAULT 'unknown'
);

-- Initialize cache metadata
INSERT INTO cluster_cache_metadata (cache_name, refresh_strategy, auto_refresh_interval, staleness_tolerance_minutes) VALUES
('cluster_age_metrics', 'scheduled', '30 minutes'::INTERVAL, 30),
('cluster_detailed_analytics_cache', 'on_demand', '1 hour'::INTERVAL, 60),
('memory_cluster_analysis_cache', 'performance_based', '15 minutes'::INTERVAL, 15)
ON CONFLICT (cache_name) DO NOTHING;

-- 2. Intelligent cache refresh function with performance-based decisions
CREATE OR REPLACE FUNCTION smart_cache_refresh(
    cache_name_param VARCHAR(100),
    force_refresh BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    action_taken TEXT,
    refresh_time_ms DECIMAL,
    rows_processed INTEGER,
    cache_status TEXT,
    next_refresh_recommended TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    cache_info RECORD;
    current_performance DECIMAL;
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    refresh_duration DECIMAL;
    new_row_count INTEGER;
    cache_age_minutes INTEGER;
    needs_refresh BOOLEAN := FALSE;
    refresh_reason TEXT;
BEGIN
    -- Get current cache metadata
    SELECT * INTO cache_info FROM cluster_cache_metadata WHERE cache_name = cache_name_param;
    
    IF cache_info IS NULL THEN
        RETURN QUERY SELECT 'error'::TEXT, 0::DECIMAL, 0, 'cache_not_found'::TEXT, NULL::TIMESTAMP WITH TIME ZONE;
        RETURN;
    END IF;
    
    -- Calculate cache age
    cache_age_minutes := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - cache_info.last_refresh)) / 60.0;
    
    -- Determine if refresh is needed
    IF force_refresh THEN
        needs_refresh := TRUE;
        refresh_reason := 'forced';
    ELSIF cache_age_minutes > cache_info.staleness_tolerance_minutes THEN
        needs_refresh := TRUE;
        refresh_reason := 'stale';
    ELSIF cache_info.refresh_strategy = 'performance_based' THEN
        -- Test current performance
        start_time := clock_timestamp();
        
        CASE cache_name_param
            WHEN 'cluster_age_metrics' THEN
                PERFORM COUNT(*) FROM cluster_age_metrics;
            WHEN 'memory_cluster_analysis_cache' THEN
                PERFORM COUNT(*) FROM memory_cluster_analysis_cache;
            ELSE
                PERFORM 1; -- Default test
        END CASE;
        
        end_time := clock_timestamp();
        current_performance := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        
        IF current_performance > cache_info.performance_threshold_ms THEN
            needs_refresh := TRUE;
            refresh_reason := 'performance_degraded';
        END IF;
    END IF;
    
    -- Perform refresh if needed
    IF needs_refresh THEN
        start_time := clock_timestamp();
        
        -- Refresh the appropriate cache
        CASE cache_name_param
            WHEN 'cluster_age_metrics' THEN
                REFRESH MATERIALIZED VIEW cluster_age_metrics;
                SELECT COUNT(*) INTO new_row_count FROM cluster_age_metrics;
                
            WHEN 'cluster_detailed_analytics_cache' THEN
                REFRESH MATERIALIZED VIEW cluster_detailed_analytics_cache;
                SELECT COUNT(*) INTO new_row_count FROM cluster_detailed_analytics_cache;
                
            WHEN 'memory_cluster_analysis_cache' THEN
                -- Refresh if it exists, otherwise skip
                IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'memory_cluster_analysis_cache') THEN
                    REFRESH MATERIALIZED VIEW memory_cluster_analysis_cache;
                    SELECT COUNT(*) INTO new_row_count FROM memory_cluster_analysis_cache;
                ELSE
                    new_row_count := 0;
                END IF;
                
            ELSE
                new_row_count := 0;
        END CASE;
        
        end_time := clock_timestamp();
        refresh_duration := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        
        -- Update cache metadata
        UPDATE cluster_cache_metadata SET
            last_refresh = CURRENT_TIMESTAMP,
            refresh_duration_ms = refresh_duration,
            row_count = new_row_count,
            last_performance_test = CURRENT_TIMESTAMP,
            performance_status = CASE 
                WHEN refresh_duration < performance_threshold_ms THEN 'excellent'
                WHEN refresh_duration < performance_threshold_ms * 2 THEN 'good'
                ELSE 'needs_optimization'
            END
        WHERE cache_name = cache_name_param;
        
        -- Calculate next refresh time
        RETURN QUERY SELECT 
            ('refreshed_' || refresh_reason)::TEXT,
            refresh_duration,
            new_row_count,
            CASE 
                WHEN refresh_duration < cache_info.performance_threshold_ms THEN 'optimal'
                WHEN refresh_duration < cache_info.performance_threshold_ms * 2 THEN 'acceptable'
                ELSE 'slow'
            END,
            CURRENT_TIMESTAMP + cache_info.auto_refresh_interval;
    ELSE
        RETURN QUERY SELECT 
            'no_refresh_needed'::TEXT,
            0::DECIMAL,
            cache_info.row_count,
            'fresh'::TEXT,
            cache_info.last_refresh + cache_info.auto_refresh_interval;
    END IF;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 3. Batch cache refresh with prioritization
CREATE OR REPLACE FUNCTION refresh_all_caches(
    max_refresh_time_seconds INTEGER DEFAULT 30
) RETURNS TABLE (
    cache_name VARCHAR(100),
    refresh_result TEXT,
    duration_ms DECIMAL,
    priority_order INTEGER
) AS $$
DECLARE
    cache_record RECORD;
    refresh_result RECORD;
    total_time_elapsed INTEGER := 0;
    start_time TIMESTAMP WITH TIME ZONE;
    priority_order INTEGER := 1;
BEGIN
    start_time := clock_timestamp();
    
    -- Refresh caches in priority order (most critical first)
    FOR cache_record IN
        SELECT 
            cm.cache_name,
            -- Priority calculation based on staleness and performance impact
            CASE cm.refresh_strategy
                WHEN 'performance_based' THEN 1  -- Highest priority
                WHEN 'scheduled' THEN 2
                ELSE 3
            END + 
            CASE 
                WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - cm.last_refresh)) > cm.staleness_tolerance_minutes * 60 THEN 0
                ELSE 1
            END as priority_score
        FROM cluster_cache_metadata cm
        WHERE cm.auto_refresh_interval IS NOT NULL
        ORDER BY priority_score ASC, cm.last_refresh ASC
    LOOP
        -- Check if we have time remaining
        total_time_elapsed := EXTRACT(EPOCH FROM (clock_timestamp() - start_time));
        IF total_time_elapsed >= max_refresh_time_seconds THEN
            RETURN QUERY SELECT cache_record.cache_name, 'skipped_time_limit'::TEXT, 0::DECIMAL, priority_order;
            priority_order := priority_order + 1;
            CONTINUE;
        END IF;
        
        -- Perform refresh
        SELECT * INTO refresh_result 
        FROM smart_cache_refresh(cache_record.cache_name) 
        LIMIT 1;
        
        RETURN QUERY SELECT 
            cache_record.cache_name,
            refresh_result.action_taken,
            refresh_result.refresh_time_ms,
            priority_order;
            
        priority_order := priority_order + 1;
    END LOOP;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 4. Cache performance monitoring view
CREATE OR REPLACE VIEW cache_performance_summary AS
SELECT 
    cache_name,
    last_refresh,
    refresh_duration_ms,
    row_count,
    refresh_strategy,
    performance_status,
    
    -- Cache age and freshness
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_refresh)) / 60.0 as age_minutes,
    
    CASE 
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_refresh)) / 60.0 <= staleness_tolerance_minutes THEN 'fresh'
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_refresh)) / 60.0 <= staleness_tolerance_minutes * 2 THEN 'stale'
        ELSE 'outdated'
    END as freshness_status,
    
    -- Performance indicators
    CASE 
        WHEN refresh_duration_ms <= performance_threshold_ms THEN 'meets_target'
        WHEN refresh_duration_ms <= performance_threshold_ms * 2 THEN 'acceptable'
        ELSE 'exceeds_target'
    END as performance_vs_target,
    
    -- Next refresh estimation
    last_refresh + auto_refresh_interval as next_scheduled_refresh,
    
    -- Efficiency metrics
    CASE 
        WHEN row_count > 0 THEN refresh_duration_ms / row_count
        ELSE 0
    END as ms_per_row
    
FROM cluster_cache_metadata;

-- 5. Adaptive caching strategy function
CREATE OR REPLACE FUNCTION optimize_cache_strategy(
    cache_name_param VARCHAR(100),
    performance_history_days INTEGER DEFAULT 7
) RETURNS TABLE (
    recommendation TEXT,
    current_strategy VARCHAR(50),
    suggested_strategy VARCHAR(50),
    suggested_interval INTERVAL,
    reasoning TEXT
) AS $$
DECLARE
    cache_info RECORD;
    avg_performance DECIMAL;
    refresh_frequency INTEGER;
BEGIN
    -- Get cache information
    SELECT * INTO cache_info FROM cluster_cache_metadata WHERE cache_name = cache_name_param;
    
    IF cache_info IS NULL THEN
        RETURN QUERY SELECT 'error'::TEXT, ''::VARCHAR(50), ''::VARCHAR(50), NULL::INTERVAL, 'Cache not found'::TEXT;
        RETURN;
    END IF;
    
    -- Analyze performance patterns (simplified for demo)
    avg_performance := cache_info.refresh_duration_ms;
    
    -- Make recommendations based on performance
    IF avg_performance <= cache_info.performance_threshold_ms * 0.5 THEN
        -- Very fast refresh - can be more frequent
        RETURN QUERY SELECT 
            'increase_frequency'::TEXT,
            cache_info.refresh_strategy,
            'scheduled'::VARCHAR(50),
            '15 minutes'::INTERVAL,
            'Cache refreshes very quickly, can update more frequently'::TEXT;
            
    ELSIF avg_performance > cache_info.performance_threshold_ms * 2 THEN
        -- Slow refresh - should be less frequent or on-demand
        RETURN QUERY SELECT 
            'reduce_frequency'::TEXT,
            cache_info.refresh_strategy,
            'on_demand'::VARCHAR(50),
            '2 hours'::INTERVAL,
            'Cache refresh is slow, consider on-demand or longer intervals'::TEXT;
            
    ELSE
        -- Performance is acceptable
        RETURN QUERY SELECT 
            'maintain_current'::TEXT,
            cache_info.refresh_strategy,
            cache_info.refresh_strategy,
            cache_info.auto_refresh_interval,
            'Current caching strategy is performing well'::TEXT;
    END IF;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 6. Cache warmup function for critical paths
CREATE OR REPLACE FUNCTION warmup_cluster_caches()
RETURNS TABLE (
    cache_warmed TEXT,
    warmup_time_ms DECIMAL,
    status TEXT
) AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
    end_time TIMESTAMP WITH TIME ZONE;
    duration_ms DECIMAL;
    cache_count INTEGER;
BEGIN
    -- Warm up cluster_age_metrics
    start_time := clock_timestamp();
    PERFORM smart_cache_refresh('cluster_age_metrics', TRUE);
    SELECT COUNT(*) INTO cache_count FROM cluster_age_metrics;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'cluster_age_metrics'::TEXT,
        duration_ms,
        CASE 
            WHEN duration_ms < 100 THEN 'excellent'
            WHEN duration_ms < 500 THEN 'good'
            ELSE 'slow'
        END;
    
    -- Warm up main analysis views by running sample queries
    start_time := clock_timestamp();
    PERFORM COUNT(*) FROM memory_cluster_analysis_optimized;
    PERFORM COUNT(*) FROM cluster_fast_metrics WHERE health_status = 'excellent';
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    RETURN QUERY SELECT 
        'analysis_views'::TEXT,
        duration_ms,
        CASE 
            WHEN duration_ms < 50 THEN 'excellent'
            WHEN duration_ms < 100 THEN 'good' 
            ELSE 'needs_optimization'
        END;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 7. Cache invalidation trigger (for real-time scenarios)
CREATE OR REPLACE FUNCTION trigger_cache_invalidation()
RETURNS TRIGGER AS $$
BEGIN
    -- Mark relevant caches as needing refresh
    UPDATE cluster_cache_metadata 
    SET 
        performance_status = 'needs_refresh',
        last_performance_test = CURRENT_TIMESTAMP
    WHERE cache_name IN ('cluster_age_metrics', 'memory_cluster_analysis_cache')
        AND refresh_strategy = 'performance_based';
        
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- 8. Cache health check function
CREATE OR REPLACE FUNCTION check_cache_health()
RETURNS TABLE (
    overall_health TEXT,
    caches_healthy INTEGER,
    caches_degraded INTEGER,
    caches_failed INTEGER,
    recommendations TEXT[]
) AS $$
DECLARE
    healthy_count INTEGER := 0;
    degraded_count INTEGER := 0;
    failed_count INTEGER := 0;
    recommendations_array TEXT[] := '{}';
    cache_record RECORD;
BEGIN
    -- Check each cache
    FOR cache_record IN SELECT * FROM cache_performance_summary LOOP
        CASE cache_record.performance_vs_target
            WHEN 'meets_target' THEN healthy_count := healthy_count + 1;
            WHEN 'acceptable' THEN degraded_count := degraded_count + 1;
            ELSE failed_count := failed_count + 1;
        END CASE;
        
        -- Add specific recommendations
        IF cache_record.freshness_status = 'outdated' THEN
            recommendations_array := array_append(recommendations_array, 
                'Refresh ' || cache_record.cache_name || ' (outdated)');
        END IF;
        
        IF cache_record.performance_vs_target = 'exceeds_target' THEN
            recommendations_array := array_append(recommendations_array,
                'Optimize ' || cache_record.cache_name || ' performance');
        END IF;
    END LOOP;
    
    RETURN QUERY SELECT 
        CASE 
            WHEN failed_count = 0 AND degraded_count <= 1 THEN 'excellent'
            WHEN failed_count <= 1 THEN 'good'
            WHEN failed_count <= 2 THEN 'degraded'
            ELSE 'critical'
        END,
        healthy_count,
        degraded_count, 
        failed_count,
        recommendations_array;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE cluster_cache_metadata IS 'Metadata for intelligent cache management and refresh strategies';
COMMENT ON FUNCTION smart_cache_refresh IS 'Intelligent cache refresh with performance-based decision making';
COMMENT ON FUNCTION refresh_all_caches IS 'Batch cache refresh with priority-based ordering and time limits';
COMMENT ON VIEW cache_performance_summary IS 'Real-time cache performance and health monitoring';
COMMENT ON FUNCTION warmup_cluster_caches IS 'Cache warmup for critical performance paths';
COMMENT ON FUNCTION check_cache_health IS 'Comprehensive cache health assessment and recommendations';