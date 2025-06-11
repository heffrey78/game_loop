# Commit 21: Quest Interaction System Implementation Plan

## Overview

This commit focuses on implementing a comprehensive quest interaction system that allows players to discover, engage with, and complete quests through natural language interactions. The system will integrate with the existing object systems and provide dynamic quest progression tracking.

## Reference

Based on the main implementation plan section:
> ### Commit 21: Quest Interaction System
> - Create QuestInteractionProcessor
> - Implement quest data and progress retrieval
> - Add quest step validation
> - Create quest progress updating
> - Implement quest completion and rewards
> - **Verification**: Test quest progression with sample quests, verify step completion updates progress, confirm rewards are granted upon completion

## Components to Implement

### 1. Quest Data Models (`src/game_loop/quests/models.py`)

```python
@dataclass
class QuestStep:
    step_id: str
    description: str
    requirements: Dict[str, Any]
    completion_conditions: List[str]
    rewards: Dict[str, Any]
    optional: bool = False

@dataclass
class Quest:
    quest_id: str
    title: str
    description: str
    category: QuestCategory
    difficulty: QuestDifficulty
    steps: List[QuestStep]
    prerequisites: List[str]
    rewards: Dict[str, Any]
    time_limit: Optional[float]
    repeatable: bool = False

@dataclass
class QuestProgress:
    quest_id: str
    player_id: str
    status: QuestStatus
    current_step: int
    completed_steps: List[str]
    step_progress: Dict[str, Any]
    started_at: float
    updated_at: float
```

### 2. Quest Interaction Processor (`src/game_loop/core/quest/quest_processor.py`)

```python
class QuestInteractionProcessor:
    """
    Process quest-related interactions and manage quest progression.
    
    Handles quest discovery, acceptance, progress tracking, and completion.
    Integrates with the existing object and action systems.
    """

    async def process_quest_interaction(
        self,
        interaction_type: QuestInteractionType,
        player_id: str,
        quest_context: Dict[str, Any],
        game_state: GameState
    ) -> QuestInteractionResult

    async def discover_available_quests(
        self,
        player_id: str,
        location_id: str,
        context: Dict[str, Any]
    ) -> List[Quest]

    async def accept_quest(
        self,
        player_id: str,
        quest_id: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]

    async def update_quest_progress(
        self,
        player_id: str,
        quest_id: str,
        action_result: ActionResult,
        context: Dict[str, Any]
    ) -> List[QuestUpdate]

    async def complete_quest_step(
        self,
        player_id: str,
        quest_id: str,
        step_id: str,
        completion_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]

    async def complete_quest(
        self,
        player_id: str,
        quest_id: str,
        context: Dict[str, Any]
    ) -> QuestCompletionResult
```

### 3. Quest Manager (`src/game_loop/quests/quest_manager.py`)

```python
class QuestManager:
    """
    Central manager for quest system operations.
    
    Coordinates quest data, progress tracking, and integration
    with other game systems.
    """

    async def get_player_active_quests(
        self,
        player_id: str
    ) -> List[QuestProgress]

    async def get_quest_by_id(
        self,
        quest_id: str
    ) -> Optional[Quest]

    async def validate_quest_prerequisites(
        self,
        player_id: str,
        quest_id: str,
        game_state: GameState
    ) -> Tuple[bool, List[str]]

    async def check_step_completion_conditions(
        self,
        player_id: str,
        quest_id: str,
        step_id: str,
        action_result: ActionResult
    ) -> bool

    async def grant_quest_rewards(
        self,
        player_id: str,
        rewards: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]

    async def generate_dynamic_quest(
        self,
        player_id: str,
        context: Dict[str, Any],
        quest_type: str
    ) -> Optional[Quest]
```

### 4. Quest Repository (`src/game_loop/database/repositories/quest.py`)

```python
class QuestRepository(BaseRepository):
    """Repository for quest data persistence."""

    async def get_quest(self, quest_id: str) -> Optional[Quest]
    async def create_quest(self, quest: Quest) -> bool
    async def update_quest(self, quest: Quest) -> bool
    async def delete_quest(self, quest_id: str) -> bool

    async def get_player_progress(
        self,
        player_id: str,
        quest_id: str
    ) -> Optional[QuestProgress]

    async def update_progress(
        self,
        progress: QuestProgress
    ) -> bool

    async def get_available_quests(
        self,
        player_id: str,
        location_id: Optional[str] = None
    ) -> List[Quest]

    async def get_completed_quests(
        self,
        player_id: str
    ) -> List[QuestProgress]
```

### 5. Database Schema Updates (`src/game_loop/database/migrations/022_quest_system.sql`)

```sql
-- Quests table
CREATE TABLE quests (
    quest_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    difficulty VARCHAR(50) NOT NULL,
    steps JSONB NOT NULL,
    prerequisites JSONB DEFAULT '[]',
    rewards JSONB DEFAULT '{}',
    time_limit FLOAT DEFAULT NULL,
    repeatable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quest progress table
CREATE TABLE quest_progress (
    progress_id SERIAL PRIMARY KEY,
    quest_id VARCHAR(255) REFERENCES quests(quest_id),
    player_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    current_step INTEGER DEFAULT 0,
    completed_steps JSONB DEFAULT '[]',
    step_progress JSONB DEFAULT '{}',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(quest_id, player_id)
);

-- Quest interactions log
CREATE TABLE quest_interactions (
    interaction_id SERIAL PRIMARY KEY,
    quest_id VARCHAR(255) REFERENCES quests(quest_id),
    player_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    interaction_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_quest_progress_player ON quest_progress(player_id);
CREATE INDEX idx_quest_progress_status ON quest_progress(status);
CREATE INDEX idx_quest_interactions_quest ON quest_interactions(quest_id);
CREATE INDEX idx_quest_interactions_player ON quest_interactions(player_id);
```

### 6. Integration with Object Systems (`src/game_loop/core/quest/quest_integration.py`)

```python
class QuestObjectIntegration:
    """
    Integration layer between quest system and object systems.
    
    Handles quest triggers from object interactions, inventory changes,
    location visits, and other game events.
    """

    async def process_action_for_quests(
        self,
        player_id: str,
        action_result: ActionResult,
        context: Dict[str, Any]
    ) -> List[QuestUpdate]

    async def check_quest_triggers(
        self,
        player_id: str,
        trigger_type: str,
        trigger_data: Dict[str, Any]
    ) -> List[Quest]

    async def update_quest_objectives(
        self,
        player_id: str,
        objective_type: str,
        objective_data: Dict[str, Any]
    ) -> List[QuestProgress]
```

## Implementation Steps

### Phase 1: Core Quest Models and Database
1. **Create quest data models** with proper type hints and validation
2. **Implement database schema** with migrations for quest tables
3. **Create quest repository** with CRUD operations
4. **Add unit tests** for models and repository

### Phase 2: Quest Manager and Business Logic
1. **Implement QuestManager** with core quest operations
2. **Add quest validation logic** for prerequisites and conditions
3. **Create reward system integration** with inventory and player state
4. **Add quest progress tracking** with step completion logic

### Phase 3: Quest Interaction Processor
1. **Create QuestInteractionProcessor** for handling quest actions
2. **Implement quest discovery** based on location and player state
3. **Add quest acceptance flow** with validation and initialization
4. **Create quest completion handling** with rewards and cleanup

### Phase 4: Integration with Game Systems
1. **Integrate with ObjectSystemIntegration** for quest triggers
2. **Connect to action processing** for automatic quest progress updates
3. **Add quest-related commands** to input processing
4. **Create quest UI elements** for output generation

### Phase 5: Dynamic Quest Generation
1. **Implement basic quest templates** for common quest types
2. **Add context-aware quest generation** based on player location and state
3. **Create quest difficulty scaling** based on player progress
4. **Add quest variety and randomization** to prevent repetition

## Testing Strategy

### Unit Tests (`tests/unit/quests/`)
```python
# Test quest models and validation
class TestQuestModels

# Test quest manager operations  
class TestQuestManager

# Test quest repository CRUD operations
class TestQuestRepository

# Test quest interaction processor
class TestQuestInteractionProcessor

# Test quest integration with other systems
class TestQuestObjectIntegration
```

### Integration Tests (`tests/integration/quests/`)
```python
# Test complete quest workflows
class TestQuestWorkflow

# Test quest database operations
class TestQuestDatabaseIntegration

# Test quest system with game loop
class TestQuestGameIntegration
```

### Sample Test Scenarios
1. **Quest Discovery**: Player enters location, available quests are discovered
2. **Quest Acceptance**: Player accepts quest, progress tracking begins
3. **Step Progression**: Player performs actions, quest steps are completed
4. **Quest Completion**: All steps complete, rewards are granted
5. **Quest Failure**: Time limit exceeded or failure conditions met
6. **Nested Quests**: Completing one quest unlocks others
7. **Repeatable Quests**: Daily/weekly quests that reset

## Configuration and Templates

### Quest Templates (`data/quest_templates/`)
```yaml
# delivery_quest.yaml
delivery_quest:
  category: "delivery"
  difficulty: "easy"
  steps:
    - description: "Pick up {item} from {source_location}"
      requirements:
        location: "{source_location}"
        action: "take"
        target: "{item}"
    - description: "Deliver {item} to {target_location}"
      requirements:
        location: "{target_location}"
        action: "give"
        target: "{item}"
  rewards:
    experience: 100
    gold: 50

# exploration_quest.yaml
exploration_quest:
  category: "exploration"
  difficulty: "medium"
  steps:
    - description: "Visit {location_1}"
      requirements:
        location: "{location_1}"
        action: "look"
    - description: "Find the {secret_item} in {location_2}"
      requirements:
        location: "{location_2}"
        action: "search"
        target: "{secret_item}"
```

### Quest Configuration (`config/quest_config.yaml`)
```yaml
quest_system:
  max_active_quests: 10
  quest_timeout_hours: 24
  reward_multipliers:
    easy: 1.0
    medium: 1.5
    hard: 2.0
    legendary: 3.0
  
  generation:
    enabled: true
    frequency: 0.3  # 30% chance per location visit
    max_generated_per_day: 5
    
  categories:
    - delivery
    - exploration
    - combat
    - puzzle
    - social
```

## Performance Considerations

1. **Quest Progress Caching**: Cache active quest progress in memory
2. **Lazy Loading**: Load quest steps and details only when needed
3. **Batch Updates**: Group quest progress updates for efficiency
4. **Index Optimization**: Proper database indexes for quest queries
5. **Background Processing**: Move non-critical quest processing to background

## Error Handling

1. **Quest Not Found**: Graceful handling when quest IDs are invalid
2. **Progress Corruption**: Recovery mechanisms for corrupted quest state
3. **Concurrent Updates**: Handle multiple simultaneous quest updates
4. **Reward Failures**: Rollback quest completion if reward granting fails
5. **Database Errors**: Proper error propagation and user feedback

## Integration Points

### With Existing Systems:
1. **ObjectSystemIntegration**: Quest triggers from object interactions
2. **InventoryManager**: Reward granting and item requirements
3. **GameStateManager**: Quest progress persistence
4. **OutputGenerator**: Quest-related message formatting
5. **InputProcessor**: Quest command recognition and routing

### Future Extensions:
1. **NPCManager**: Quest givers and quest-related dialogues
2. **LocationManager**: Location-specific quest availability
3. **EventSystem**: Time-based and triggered quest events
4. **AchievementSystem**: Quest completion achievements

## Verification Criteria

1. ✅ **Quest Discovery**: Players can discover available quests in locations
2. ✅ **Quest Acceptance**: Players can accept quests with proper validation
3. ✅ **Progress Tracking**: Quest progress updates correctly with player actions
4. ✅ **Step Completion**: Individual quest steps complete when conditions are met
5. ✅ **Quest Completion**: Completed quests grant rewards and update player state
6. ✅ **Error Handling**: Invalid quest operations are handled gracefully
7. ✅ **Performance**: Quest system responds within acceptable time limits
8. ✅ **Integration**: Quest system works seamlessly with existing game systems

## Success Metrics

- All quest workflow tests pass
- Quest system handles 100+ concurrent active quests
- Quest discovery and completion response time < 100ms
- No memory leaks in quest progress tracking
- Proper integration with all existing game systems
- Complete test coverage (>90%) for quest components

---

*This implementation plan provides a comprehensive roadmap for implementing the quest interaction system as outlined in the main game loop implementation plan.*