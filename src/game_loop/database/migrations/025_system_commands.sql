-- Migration 025: System command infrastructure
-- Add tables for save system, settings, and tutorial progress

-- Add tables for save system and settings
CREATE TABLE game_saves (
    save_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    save_name VARCHAR(100) NOT NULL,
    player_id UUID NOT NULL,
    description TEXT,
    save_data JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_path VARCHAR(255),
    file_size INTEGER,
    player_level INTEGER DEFAULT 1,
    location VARCHAR(255),
    play_time INTERVAL DEFAULT INTERVAL '0 seconds',
    
    UNIQUE(player_id, save_name)
);

CREATE TABLE user_settings (
    setting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID,
    setting_name VARCHAR(50) NOT NULL,
    setting_value JSONB NOT NULL,
    setting_category VARCHAR(50) NOT NULL DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(player_id, setting_name)
);

CREATE TABLE tutorial_progress (
    progress_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    tutorial_type VARCHAR(50) NOT NULL,
    current_step INTEGER NOT NULL DEFAULT 0,
    completed_steps INTEGER[] DEFAULT '{}',
    skill_level VARCHAR(20) NOT NULL DEFAULT 'beginner',
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_completed BOOLEAN NOT NULL DEFAULT FALSE,
    
    UNIQUE(player_id, tutorial_type)
);

CREATE TABLE help_topics (
    topic_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_key VARCHAR(100) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    keywords TEXT[] DEFAULT '{}',
    examples TEXT[] DEFAULT '{}',
    related_topics TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE setting_definitions (
    definition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    default_value JSONB NOT NULL,
    allowed_values JSONB,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    validation_rules JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_game_saves_player_created ON game_saves(player_id, created_at DESC);
CREATE INDEX idx_game_saves_player_name ON game_saves(player_id, save_name);
CREATE INDEX idx_user_settings_player ON user_settings(player_id);
CREATE INDEX idx_user_settings_category ON user_settings(setting_category);
CREATE INDEX idx_tutorial_progress_player ON tutorial_progress(player_id);
CREATE INDEX idx_tutorial_progress_type ON tutorial_progress(tutorial_type);
CREATE INDEX idx_help_topics_category ON help_topics(category);
CREATE INDEX idx_help_topics_keywords ON help_topics USING GIN(keywords);
CREATE INDEX idx_setting_definitions_category ON setting_definitions(category);

-- Add comments for documentation
COMMENT ON TABLE game_saves IS 'Stores saved game states with metadata for loading';
COMMENT ON TABLE user_settings IS 'Stores user preferences and configuration settings';
COMMENT ON TABLE tutorial_progress IS 'Tracks player progress through tutorial systems';
COMMENT ON TABLE help_topics IS 'Stores help content and documentation';
COMMENT ON TABLE setting_definitions IS 'Defines available settings and their constraints';

COMMENT ON COLUMN game_saves.save_data IS 'Complete game state serialized as JSON';
COMMENT ON COLUMN game_saves.metadata IS 'Additional save metadata like location, level, etc.';
COMMENT ON COLUMN user_settings.setting_value IS 'Setting value stored as JSON for flexibility';
COMMENT ON COLUMN tutorial_progress.completed_steps IS 'Array of completed tutorial step numbers';
COMMENT ON COLUMN help_topics.keywords IS 'Keywords for searching help content';
COMMENT ON COLUMN setting_definitions.validation_rules IS 'JSON rules for validating setting values';

-- Insert default setting definitions
INSERT INTO setting_definitions (setting_name, description, default_value, value_type, category) VALUES
('auto_save_interval', 'Automatic save interval in minutes', '5', 'integer', 'gameplay'),
('max_auto_saves', 'Maximum number of auto-save files to keep', '10', 'integer', 'gameplay'),
('tutorial_enabled', 'Enable tutorial hints and guidance', 'true', 'boolean', 'interface'),
('help_verbosity', 'Level of detail in help responses', '"detailed"', 'string', 'interface'),
('command_suggestions', 'Show command suggestions', 'true', 'boolean', 'interface'),
('color_output', 'Enable colored text output', 'true', 'boolean', 'display'),
('text_speed', 'Text display speed (slow, normal, fast)', '"normal"', 'string', 'display'),
('save_compression', 'Compress save files', 'true', 'boolean', 'storage'),
('max_save_slots', 'Maximum number of save slots per player', '20', 'integer', 'storage');

-- Insert default help topics
INSERT INTO help_topics (topic_key, title, content, category, keywords, examples) VALUES
('basic_commands', 'Basic Commands', 'Learn the fundamental commands for interacting with the game world.', 'getting_started', 
 ARRAY['commands', 'basic', 'help', 'start'], 
 ARRAY['look around', 'go north', 'take item', 'talk to guard']),

('movement', 'Movement and Navigation', 'How to move around the game world and navigate between locations.', 'gameplay',
 ARRAY['movement', 'go', 'walk', 'travel', 'navigation'],
 ARRAY['go north', 'walk to town', 'enter building', 'climb stairs']),

('inventory', 'Inventory Management', 'Managing your items, equipment, and belongings.', 'gameplay',
 ARRAY['inventory', 'items', 'take', 'drop', 'use'],
 ARRAY['take sword', 'drop torch', 'use key', 'examine backpack']),

('conversation', 'Talking to Characters', 'How to interact and have conversations with NPCs.', 'gameplay',
 ARRAY['talk', 'conversation', 'npc', 'dialogue', 'ask'],
 ARRAY['talk to merchant', 'ask about quest', 'greet guard', 'say goodbye']),

('save_load', 'Saving and Loading', 'How to save your progress and load saved games.', 'system',
 ARRAY['save', 'load', 'game', 'progress', 'continue'],
 ARRAY['save game', 'save as my_adventure', 'load game', 'list saves']),

('settings', 'Game Settings', 'Customizing your game experience through settings.', 'system',
 ARRAY['settings', 'preferences', 'config', 'options'],
 ARRAY['show settings', 'set auto_save_interval 10', 'reset settings']);