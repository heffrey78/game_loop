# Commit 23: System Command Processing Implementation Plan

## Overview

This commit implements the final action processing component - system commands that handle meta-game functionality like save/load, help, settings, and game control. This completes the action processing system started in commits 18-22.

## Objectives

- Implement SystemCommandProcessor for meta-game commands
- Create comprehensive save/load functionality
- Add intelligent help system with context awareness
- Implement tutorial and guidance systems
- Add game control commands (settings, exit, etc.)
- Ensure all system commands integrate seamlessly with existing game loop

## Current State Analysis

Based on the codebase review, the following components are already implemented:
- Action classification system with command routing
- Game state management with persistence
- Database infrastructure with migrations
- Query and conversation systems (Commit 22)
- Physical action and object interaction processors

## Implementation Tasks

### 1. System Command Processor Core

**File**: `src/game_loop/core/command_handlers/system_command_processor.py`

```python
class SystemCommandProcessor:
    """Handles meta-game system commands like save, load, help, settings."""
    
    def __init__(self, game_state_manager, session_manager, config_manager):
        self.game_state = game_state_manager
        self.session_manager = session_manager
        self.config_manager = config_manager
        self.help_system = HelpSystem()
        self.tutorial_manager = TutorialManager()
    
    async def process_command(self, command_type: SystemCommandType, 
                            args: dict, context: dict) -> ActionResult:
        """Route system commands to appropriate handlers."""
        
    async def handle_save_game(self, save_name: str = None) -> ActionResult:
        """Save current game state with optional custom name."""
        
    async def handle_load_game(self, save_name: str = None) -> ActionResult:
        """Load saved game state."""
        
    async def handle_help_request(self, topic: str = None) -> ActionResult:
        """Provide contextual help information."""
        
    async def handle_tutorial_request(self, tutorial_type: str = None) -> ActionResult:
        """Start or continue tutorial guidance."""
        
    async def handle_settings_command(self, setting: str = None, value: str = None) -> ActionResult:
        """View or modify game settings."""
        
    async def handle_game_exit(self, force: bool = False) -> ActionResult:
        """Handle game exit with optional auto-save."""
```

### 2. Enhanced Save/Load System

**File**: `src/game_loop/core/save_system/save_manager.py`

```python
class SaveManager:
    """Enhanced save/load system with multiple save slots and metadata."""
    
    def __init__(self, session_manager, game_state_manager):
        self.session_manager = session_manager
        self.game_state = game_state_manager
        self.save_directory = Path("saves")
        self.auto_save_interval = 300  # 5 minutes
        
    async def create_save(self, save_name: str = None, 
                         description: str = None) -> SaveResult:
        """Create a complete game save with metadata."""
        
    async def load_save(self, save_name: str) -> LoadResult:
        """Load a complete game save and restore state."""
        
    async def list_saves(self) -> list[SaveMetadata]:
        """List all available saves with metadata."""
        
    async def delete_save(self, save_name: str) -> bool:
        """Delete a save file."""
        
    async def auto_save(self) -> SaveResult:
        """Perform automatic save with rotation."""
        
    def generate_save_summary(self, game_state: dict) -> str:
        """Generate human-readable save summary."""
```

### 3. Intelligent Help System

**File**: `src/game_loop/core/help/help_system.py`

```python
class HelpSystem:
    """Context-aware help system that provides relevant assistance."""
    
    def __init__(self, llm_client, semantic_search):
        self.llm_client = llm_client
        self.semantic_search = semantic_search
        self.help_topics = self._load_help_topics()
        
    async def get_help(self, topic: str = None, 
                      context: dict = None) -> HelpResponse:
        """Get contextual help based on current game state."""
        
    async def get_contextual_suggestions(self, context: dict) -> list[str]:
        """Suggest relevant commands based on current situation."""
        
    async def search_help(self, query: str) -> list[HelpTopic]:
        """Search help content using semantic similarity."""
        
    def _analyze_context_for_help(self, context: dict) -> HelpContext:
        """Analyze game context to provide relevant help."""
        
    def _load_help_topics(self) -> dict:
        """Load help content from files and database."""
```

### 4. Tutorial and Guidance System

**File**: `src/game_loop/core/tutorial/tutorial_manager.py`

```python
class TutorialManager:
    """Adaptive tutorial system that guides new players."""
    
    def __init__(self, game_state_manager, llm_client):
        self.game_state = game_state_manager
        self.llm_client = llm_client
        self.tutorial_steps = self._load_tutorial_content()
        
    async def check_tutorial_triggers(self, context: dict) -> list[TutorialPrompt]:
        """Check if current situation should trigger tutorial hints."""
        
    async def start_tutorial(self, tutorial_type: str) -> TutorialSession:
        """Start a specific tutorial sequence."""
        
    async def get_next_hint(self, player_id: str) -> TutorialHint:
        """Get the next tutorial hint for the player."""
        
    def track_player_progress(self, player_id: str, action: str):
        """Track player actions to assess tutorial needs."""
        
    def _assess_player_skill(self, player_id: str) -> PlayerSkillLevel:
        """Assess player skill to customize tutorial content."""
```

### 5. Settings Management

**File**: `src/game_loop/core/settings/settings_manager.py`

```python
class SettingsManager:
    """Manage user preferences and game settings."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.user_settings = {}
        self.setting_definitions = self._load_setting_definitions()
        
    async def get_setting(self, setting_name: str, player_id: str = None) -> Any:
        """Get current value of a setting."""
        
    async def set_setting(self, setting_name: str, value: Any, 
                         player_id: str = None) -> SettingResult:
        """Set a setting value with validation."""
        
    async def list_settings(self, category: str = None) -> list[SettingInfo]:
        """List available settings with descriptions."""
        
    async def reset_settings(self, category: str = None, 
                           player_id: str = None) -> bool:
        """Reset settings to defaults."""
        
    def validate_setting_value(self, setting_name: str, value: Any) -> bool:
        """Validate a setting value against constraints."""
```

### 6. Integration with Action Classification

**File**: `src/game_loop/core/actions/action_classifier.py` (update)

Add system command recognition:

```python
class SystemCommandType(Enum):
    SAVE_GAME = "save_game"
    LOAD_GAME = "load_game"
    HELP = "help"
    TUTORIAL = "tutorial"
    SETTINGS = "settings"
    QUIT_GAME = "quit_game"
    AUTO_SAVE = "auto_save"
    LIST_SAVES = "list_saves"

async def classify_system_command(self, text: str) -> SystemCommandClassification:
    """Classify system/meta-game commands."""
    
    system_patterns = {
        SystemCommandType.SAVE_GAME: [
            r"save(\s+game)?(\s+as\s+[\w\s]+)?",
            r"create\s+save",
            r"save\s+state"
        ],
        SystemCommandType.LOAD_GAME: [
            r"load(\s+game)?(\s+[\w\s]+)?",
            r"restore(\s+save)?",
            r"continue(\s+game)?"
        ],
        SystemCommandType.HELP: [
            r"help(\s+\w+)?",
            r"how\s+do\s+i",
            r"what\s+can\s+i\s+do",
            r"commands?"
        ],
        # ... more patterns
    }
```

### 7. Database Schema Updates

**File**: `src/game_loop/database/migrations/025_system_commands.sql`

```sql
-- Add tables for save system and settings
CREATE TABLE game_saves (
    save_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    save_name VARCHAR(100) NOT NULL,
    player_id UUID NOT NULL,
    description TEXT,
    save_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_path VARCHAR(255),
    file_size INTEGER,
    
    UNIQUE(player_id, save_name)
);

CREATE TABLE user_settings (
    setting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID,
    setting_name VARCHAR(50) NOT NULL,
    setting_value JSONB NOT NULL,
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
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(player_id, tutorial_type)
);

-- Indexes for performance
CREATE INDEX idx_game_saves_player_created ON game_saves(player_id, created_at DESC);
CREATE INDEX idx_user_settings_player ON user_settings(player_id);
CREATE INDEX idx_tutorial_progress_player ON tutorial_progress(player_id);
```

### 8. Command Integration

**File**: `src/game_loop/core/command_handlers/command_handler_factory.py` (update)

```python
async def route_system_command(self, classification: SystemCommandClassification, 
                              context: dict) -> ActionResult:
    """Route system commands to SystemCommandProcessor."""
    
    return await self.system_processor.process_command(
        classification.command_type,
        classification.args,
        context
    )
```

### 9. Data Models

**File**: `src/game_loop/core/models/system_models.py`

```python
@dataclass
class SaveMetadata:
    save_name: str
    description: str
    created_at: datetime
    file_size: int
    player_level: int
    location: str
    play_time: timedelta

@dataclass
class HelpResponse:
    topic: str
    content: str
    related_topics: list[str]
    contextual_suggestions: list[str]
    examples: list[str]

@dataclass
class TutorialHint:
    hint_type: str
    message: str
    suggested_action: str
    priority: int

@dataclass
class SettingInfo:
    name: str
    description: str
    current_value: Any
    default_value: Any
    allowed_values: list[Any]
    category: str
```

### 10. Template Updates

**File**: `templates/system_commands/` (new directory)

Create templates for:
- `save_confirmation.txt`
- `load_confirmation.txt`
- `help_topic.txt`
- `tutorial_hint.txt`
- `settings_list.txt`
- `error_messages.txt`

## Testing Strategy

### Unit Tests

**File**: `tests/unit/core/command_handlers/test_system_command_processor.py`

```python
@pytest.mark.asyncio
class TestSystemCommandProcessor:
    async def test_save_game_success(self, processor, mock_game_state):
        """Test successful game save operation."""
        
    async def test_load_game_with_valid_save(self, processor, mock_save_data):
        """Test loading a valid save file."""
        
    async def test_help_request_with_context(self, processor, mock_context):
        """Test contextual help generation."""
        
    async def test_settings_modification(self, processor, mock_settings):
        """Test setting modification and validation."""
        
    async def test_tutorial_trigger_detection(self, processor, mock_player_state):
        """Test tutorial system triggers."""
```

### Integration Tests

**File**: `tests/integration/system_commands/test_save_load_integration.py`

```python
@pytest.mark.asyncio
class TestSaveLoadIntegration:
    async def test_full_save_load_cycle(self, game_loop, test_session):
        """Test complete save and load cycle with real game state."""
        
    async def test_auto_save_functionality(self, game_loop):
        """Test automatic save triggers and rotation."""
        
    async def test_save_file_corruption_handling(self, save_manager):
        """Test handling of corrupted save files."""
```

## Implementation Checklist

- [ ] Implement SystemCommandProcessor core class
- [ ] Create enhanced SaveManager with metadata
- [ ] Implement intelligent HelpSystem with context awareness
- [ ] Create TutorialManager with adaptive guidance
- [ ] Implement SettingsManager with validation
- [ ] Update action classification for system commands
- [ ] Create database migration for new tables
- [ ] Update command routing in factory
- [ ] Create data models for system commands
- [ ] Create templates for system command responses
- [ ] Write comprehensive unit tests
- [ ] Write integration tests for save/load
- [ ] Add help content files
- [ ] Test tutorial system triggers
- [ ] Verify settings persistence
- [ ] Test error handling and recovery
- [ ] Update documentation

## Verification Criteria

1. **Save/Load Functionality**:
   - Can save game state with custom names
   - Can load saved games completely
   - Auto-save works at specified intervals
   - Save file management (list, delete) works
   - Corrupted save files are handled gracefully

2. **Help System**:
   - Provides relevant help based on context
   - Suggests appropriate commands for current situation
   - Search functionality finds relevant topics
   - Help content is comprehensive and useful

3. **Tutorial System**:
   - Detects when players need guidance
   - Provides contextual hints and suggestions
   - Tracks player progress appropriately
   - Adapts to player skill level

4. **Settings Management**:
   - Can view and modify game settings
   - Setting validation works correctly
   - Settings persist across sessions
   - Default reset functionality works

5. **Integration**:
   - System commands integrate seamlessly with game loop
   - Action classification correctly identifies system commands
   - Command routing works for all system command types
   - Error handling provides useful feedback

## Dependencies

- Existing game state management system
- Database infrastructure with migration support
- Action classification and routing system
- Template system for responses
- LLM integration for contextual help

## Risk Assessment

**Low Risk**:
- Save/load system (building on existing state management)
- Settings management (straightforward CRUD operations)
- Basic help system implementation

**Medium Risk**:
- Tutorial system complexity and trigger detection
- Contextual help generation with LLM integration
- Save file format evolution and backward compatibility

**Mitigation Strategies**:
- Implement comprehensive error handling for save/load operations
- Create fallback mechanisms for LLM-dependent features
- Design extensible save file format with version management
- Thorough testing of edge cases and error conditions

## Success Metrics

- All system commands work reliably
- Save/load preserves complete game state
- Help system provides useful assistance
- Tutorial system improves new player experience
- Settings changes persist correctly
- No data loss during system operations
- Response times under 2 seconds for all system commands

This implementation plan completes the action processing system by adding essential meta-game functionality that enhances the overall user experience and provides robust game state management.