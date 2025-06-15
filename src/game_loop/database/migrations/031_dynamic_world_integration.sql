-- Migration 031: Dynamic World Integration
-- 
-- This migration adds comprehensive database schema for tracking dynamic world generation,
-- player behavior analysis, content discovery, and quality monitoring.

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Generation trigger tracking
CREATE TABLE generation_triggers (
    trigger_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    session_id UUID NOT NULL,
    trigger_type VARCHAR(50) NOT NULL CHECK (trigger_type IN (
        'location_boundary', 'exploration', 'quest_need', 'content_gap',
        'player_preference', 'narrative_requirement', 'world_expansion'
    )),
    trigger_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    location_id UUID,
    action_that_triggered TEXT,
    priority_score FLOAT NOT NULL DEFAULT 0.5 CHECK (priority_score BETWEEN 0 AND 1),
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    generation_result JSONB,
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT NULL
);

-- Indexes for generation triggers
CREATE INDEX idx_generation_triggers_player_session 
ON generation_triggers(player_id, session_id);
CREATE INDEX idx_generation_triggers_type_priority 
ON generation_triggers(trigger_type, priority_score DESC);
CREATE INDEX idx_generation_triggers_location 
ON generation_triggers(location_id) WHERE location_id IS NOT NULL;
CREATE INDEX idx_generation_triggers_triggered_at 
ON generation_triggers(triggered_at);
CREATE INDEX idx_generation_triggers_processing_status 
ON generation_triggers(processed_at) WHERE processed_at IS NULL;

-- Player behavior patterns
CREATE TABLE player_behavior_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    pattern_type VARCHAR(50) NOT NULL CHECK (pattern_type IN (
        'exploration_style', 'content_preference', 'interaction_style',
        'difficulty_preference', 'engagement_pattern', 'discovery_pattern'
    )),
    pattern_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence_score FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence_score BETWEEN 0 AND 1),
    observed_frequency INTEGER DEFAULT 1,
    first_observed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    pattern_strength FLOAT DEFAULT 0.5 CHECK (pattern_strength BETWEEN 0 AND 1),
    validation_count INTEGER DEFAULT 0
);

-- Indexes for player behavior patterns
CREATE INDEX idx_player_behavior_patterns_player 
ON player_behavior_patterns(player_id);
CREATE INDEX idx_player_behavior_patterns_type 
ON player_behavior_patterns(pattern_type);
CREATE INDEX idx_player_behavior_patterns_confidence 
ON player_behavior_patterns(confidence_score DESC);
CREATE INDEX idx_player_behavior_patterns_updated 
ON player_behavior_patterns(last_updated);

-- Content discovery events
CREATE TABLE content_discovery_events (
    discovery_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    session_id UUID NOT NULL,
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL CHECK (content_type IN (
        'location', 'npc', 'object', 'connection', 'quest', 'secret'
    )),
    discovery_method VARCHAR(50) NOT NULL CHECK (discovery_method IN (
        'exploration', 'quest', 'hint', 'accident', 'guidance', 'search'
    )),
    location_id UUID,
    discovery_context JSONB DEFAULT '{}'::jsonb,
    time_to_discovery_seconds INTEGER,
    discovery_difficulty FLOAT CHECK (discovery_difficulty BETWEEN 0 AND 1),
    player_satisfaction INTEGER CHECK (player_satisfaction BETWEEN 1 AND 5),
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for content discovery events
CREATE INDEX idx_content_discovery_events_player 
ON content_discovery_events(player_id);
CREATE INDEX idx_content_discovery_events_content 
ON content_discovery_events(content_id, content_type);
CREATE INDEX idx_content_discovery_events_method 
ON content_discovery_events(discovery_method);
CREATE INDEX idx_content_discovery_events_location 
ON content_discovery_events(location_id) WHERE location_id IS NOT NULL;
CREATE INDEX idx_content_discovery_events_discovered_at 
ON content_discovery_events(discovered_at);

-- Content interaction tracking
CREATE TABLE content_interactions (
    interaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL CHECK (interaction_type IN (
        'examine', 'use', 'take', 'drop', 'talk', 'move', 'explore', 'ignore'
    )),
    interaction_duration_seconds INTEGER,
    interaction_outcome VARCHAR(50) CHECK (interaction_outcome IN (
        'success', 'failure', 'partial', 'abandoned', 'repeated'
    )),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    interaction_data JSONB DEFAULT '{}'::jsonb,
    session_id UUID NOT NULL,
    location_id UUID,
    interacted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for content interactions
CREATE INDEX idx_content_interactions_player 
ON content_interactions(player_id);
CREATE INDEX idx_content_interactions_content 
ON content_interactions(content_id, content_type);
CREATE INDEX idx_content_interactions_type_outcome 
ON content_interactions(interaction_type, interaction_outcome);
CREATE INDEX idx_content_interactions_satisfaction 
ON content_interactions(satisfaction_score) WHERE satisfaction_score IS NOT NULL;
CREATE INDEX idx_content_interactions_interacted_at 
ON content_interactions(interacted_at);

-- Generation quality metrics
CREATE TABLE generation_quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    quality_dimension VARCHAR(50) NOT NULL CHECK (quality_dimension IN (
        'overall', 'consistency', 'creativity', 'relevance', 'engagement',
        'technical', 'narrative', 'mechanical', 'thematic'
    )),
    quality_score FLOAT NOT NULL CHECK (quality_score BETWEEN 0 AND 1),
    measurement_method VARCHAR(50) NOT NULL CHECK (measurement_method IN (
        'automated', 'player_feedback', 'behavioral_analysis', 'peer_review', 'hybrid'
    )),
    measurement_context JSONB DEFAULT '{}'::jsonb,
    confidence_level FLOAT DEFAULT 0.5 CHECK (confidence_level BETWEEN 0 AND 1),
    sample_size INTEGER DEFAULT 1,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for generation quality metrics
CREATE INDEX idx_generation_quality_metrics_content 
ON generation_quality_metrics(content_id, content_type);
CREATE INDEX idx_generation_quality_metrics_dimension 
ON generation_quality_metrics(quality_dimension);
CREATE INDEX idx_generation_quality_metrics_score 
ON generation_quality_metrics(quality_score DESC);
CREATE INDEX idx_generation_quality_metrics_method 
ON generation_quality_metrics(measurement_method);
CREATE INDEX idx_generation_quality_metrics_measured_at 
ON generation_quality_metrics(measured_at);

-- World generation status
CREATE TABLE world_generation_status (
    status_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_region VARCHAR(100),
    generation_system VARCHAR(50) NOT NULL CHECK (generation_system IN (
        'location_generator', 'npc_generator', 'object_generator', 
        'connection_manager', 'pipeline_coordinator', 'quality_monitor'
    )),
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'active', 'idle', 'busy', 'error', 'maintenance', 'disabled'
    )),
    last_generation_at TIMESTAMP WITH TIME ZONE,
    next_scheduled_generation TIMESTAMP WITH TIME ZONE,
    generation_count INTEGER DEFAULT 0,
    average_quality_score FLOAT CHECK (average_quality_score BETWEEN 0 AND 1),
    average_generation_time_ms FLOAT,
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    status_data JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for world generation status
CREATE INDEX idx_world_generation_status_system 
ON world_generation_status(generation_system);
CREATE INDEX idx_world_generation_status_status 
ON world_generation_status(status);
CREATE INDEX idx_world_generation_status_region 
ON world_generation_status(world_region) WHERE world_region IS NOT NULL;
CREATE INDEX idx_world_generation_status_updated 
ON world_generation_status(updated_at);

-- Player preference learning
CREATE TABLE player_preferences (
    preference_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    preference_category VARCHAR(50) NOT NULL CHECK (preference_category IN (
        'content_type', 'theme', 'difficulty', 'interaction_style', 
        'exploration_style', 'narrative_style'
    )),
    preference_key VARCHAR(100) NOT NULL,
    preference_value FLOAT NOT NULL CHECK (preference_value BETWEEN 0 AND 1),
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    evidence_count INTEGER DEFAULT 1,
    learned_from_actions INTEGER DEFAULT 0,
    learned_from_feedback INTEGER DEFAULT 0,
    first_observed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for player preferences
CREATE INDEX idx_player_preferences_player 
ON player_preferences(player_id);
CREATE INDEX idx_player_preferences_category_key 
ON player_preferences(preference_category, preference_key);
CREATE INDEX idx_player_preferences_confidence 
ON player_preferences(confidence DESC);
CREATE UNIQUE INDEX idx_player_preferences_unique 
ON player_preferences(player_id, preference_category, preference_key);

-- Content generation history
CREATE TABLE content_generation_history (
    generation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    generation_trigger_id UUID REFERENCES generation_triggers(trigger_id),
    generator_system VARCHAR(50) NOT NULL,
    generation_context JSONB DEFAULT '{}'::jsonb,
    generation_parameters JSONB DEFAULT '{}'::jsonb,
    generation_time_ms INTEGER,
    quality_scores JSONB DEFAULT '{}'::jsonb,
    validation_results JSONB DEFAULT '{}'::jsonb,
    player_id UUID,
    session_id UUID,
    location_context UUID,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for content generation history
CREATE INDEX idx_content_generation_history_content 
ON content_generation_history(content_id, content_type);
CREATE INDEX idx_content_generation_history_trigger 
ON content_generation_history(generation_trigger_id) WHERE generation_trigger_id IS NOT NULL;
CREATE INDEX idx_content_generation_history_generator 
ON content_generation_history(generator_system);
CREATE INDEX idx_content_generation_history_player 
ON content_generation_history(player_id) WHERE player_id IS NOT NULL;
CREATE INDEX idx_content_generation_history_generated_at 
ON content_generation_history(generated_at);
CREATE INDEX idx_content_generation_history_success 
ON content_generation_history(success);

-- Quality improvement tracking
CREATE TABLE quality_improvements (
    improvement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    improvement_type VARCHAR(50) NOT NULL CHECK (improvement_type IN (
        'parameter_adjustment', 'template_update', 'algorithm_change', 
        'validation_enhancement', 'feedback_integration'
    )),
    target_system VARCHAR(50) NOT NULL,
    target_quality_dimension VARCHAR(50),
    description TEXT NOT NULL,
    implementation_data JSONB DEFAULT '{}'::jsonb,
    expected_impact FLOAT CHECK (expected_impact BETWEEN 0 AND 1),
    actual_impact FLOAT CHECK (actual_impact BETWEEN 0 AND 1),
    implementation_difficulty VARCHAR(20) CHECK (implementation_difficulty IN (
        'easy', 'medium', 'hard'
    )),
    implemented_at TIMESTAMP WITH TIME ZONE,
    measured_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for quality improvements
CREATE INDEX idx_quality_improvements_type 
ON quality_improvements(improvement_type);
CREATE INDEX idx_quality_improvements_system 
ON quality_improvements(target_system);
CREATE INDEX idx_quality_improvements_impact 
ON quality_improvements(actual_impact DESC) WHERE actual_impact IS NOT NULL;
CREATE INDEX idx_quality_improvements_implemented 
ON quality_improvements(implemented_at) WHERE implemented_at IS NOT NULL;

-- Content clusters tracking
CREATE TABLE content_clusters (
    cluster_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_theme VARCHAR(100) NOT NULL,
    anchor_location_id UUID NOT NULL,
    coherence_score FLOAT CHECK (coherence_score BETWEEN 0 AND 1),
    cluster_size INTEGER DEFAULT 0,
    creation_trigger_id UUID REFERENCES generation_triggers(trigger_id),
    cluster_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content cluster membership
CREATE TABLE content_cluster_members (
    membership_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID NOT NULL REFERENCES content_clusters(cluster_id) ON DELETE CASCADE,
    content_id UUID NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    cluster_role VARCHAR(50) DEFAULT 'member', -- 'anchor', 'core', 'member', 'peripheral'
    contribution_score FLOAT CHECK (contribution_score BETWEEN 0 AND 1),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for content clusters
CREATE INDEX idx_content_clusters_theme 
ON content_clusters(cluster_theme);
CREATE INDEX idx_content_clusters_anchor 
ON content_clusters(anchor_location_id);
CREATE INDEX idx_content_clusters_coherence 
ON content_clusters(coherence_score DESC);
CREATE INDEX idx_content_cluster_members_cluster 
ON content_cluster_members(cluster_id);
CREATE INDEX idx_content_cluster_members_content 
ON content_cluster_members(content_id, content_type);
CREATE UNIQUE INDEX idx_content_cluster_members_unique 
ON content_cluster_members(cluster_id, content_id);

-- Generation performance analytics
CREATE TABLE generation_performance_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type VARCHAR(50) NOT NULL,
    operation_parameters JSONB DEFAULT '{}'::jsonb,
    execution_time_ms INTEGER NOT NULL,
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    database_queries INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT true,
    error_details TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance logs
CREATE INDEX idx_generation_performance_logs_operation 
ON generation_performance_logs(operation_type);
CREATE INDEX idx_generation_performance_logs_timestamp 
ON generation_performance_logs(timestamp);
CREATE INDEX idx_generation_performance_logs_execution_time 
ON generation_performance_logs(execution_time_ms);
CREATE INDEX idx_generation_performance_logs_success 
ON generation_performance_logs(success);

-- Views for common queries

-- Player behavior summary view
CREATE VIEW player_behavior_summary AS
SELECT 
    player_id,
    COUNT(*) as total_patterns,
    AVG(confidence_score) as avg_confidence,
    MAX(last_updated) as last_pattern_update,
    ARRAY_AGG(DISTINCT pattern_type) as pattern_types
FROM player_behavior_patterns
GROUP BY player_id;

-- Content discovery analytics view
CREATE VIEW content_discovery_analytics AS
SELECT 
    content_type,
    discovery_method,
    COUNT(*) as discovery_count,
    AVG(time_to_discovery_seconds) as avg_discovery_time,
    AVG(player_satisfaction) as avg_satisfaction,
    COUNT(*) FILTER (WHERE player_satisfaction >= 4) * 100.0 / COUNT(*) as satisfaction_rate
FROM content_discovery_events
WHERE discovered_at >= NOW() - INTERVAL '30 days'
GROUP BY content_type, discovery_method;

-- Quality trends view
CREATE VIEW quality_trends AS
SELECT 
    content_type,
    quality_dimension,
    DATE_TRUNC('day', measured_at) as measurement_date,
    AVG(quality_score) as avg_quality,
    COUNT(*) as measurement_count,
    STDDEV(quality_score) as quality_variance
FROM generation_quality_metrics
WHERE measured_at >= NOW() - INTERVAL '30 days'
GROUP BY content_type, quality_dimension, DATE_TRUNC('day', measured_at)
ORDER BY measurement_date;

-- Generation efficiency view
CREATE VIEW generation_efficiency AS
SELECT 
    generation_system,
    COUNT(*) as total_generations,
    AVG(generation_time_ms) as avg_generation_time,
    AVG(average_quality_score) as avg_quality,
    SUM(error_count) as total_errors,
    MAX(updated_at) as last_activity
FROM world_generation_status
GROUP BY generation_system;

-- Trigger to update world_generation_status timestamps
CREATE OR REPLACE FUNCTION update_generation_status_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_generation_status_timestamp
    BEFORE UPDATE ON world_generation_status
    FOR EACH ROW
    EXECUTE FUNCTION update_generation_status_timestamp();

-- Trigger to update player behavior pattern timestamps
CREATE OR REPLACE FUNCTION update_pattern_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_pattern_timestamp
    BEFORE UPDATE ON player_behavior_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_pattern_timestamp();

-- Function to clean old logs (for maintenance)
CREATE OR REPLACE FUNCTION clean_old_performance_logs(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM generation_performance_logs 
    WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE generation_triggers IS 'Tracks events that trigger dynamic world generation';
COMMENT ON TABLE player_behavior_patterns IS 'Stores learned patterns about player behavior for content personalization';
COMMENT ON TABLE content_discovery_events IS 'Logs when players discover generated content';
COMMENT ON TABLE content_interactions IS 'Tracks how players interact with generated content';
COMMENT ON TABLE generation_quality_metrics IS 'Stores quality assessments for generated content';
COMMENT ON TABLE world_generation_status IS 'Monitors the status and health of generation systems';
COMMENT ON TABLE player_preferences IS 'Learned player preferences for content generation';
COMMENT ON TABLE content_generation_history IS 'Complete history of all content generation events';
COMMENT ON TABLE quality_improvements IS 'Tracks quality improvement implementations and their impact';
COMMENT ON TABLE content_clusters IS 'Groups related generated content for thematic coherence';
COMMENT ON TABLE generation_performance_logs IS 'Performance monitoring for generation systems';

-- Grant permissions (adjust as needed for your user setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO game_loop_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO game_loop_user;