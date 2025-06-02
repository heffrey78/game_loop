# Commit 17: Search Integration with Game Loop

## Overview

This commit builds upon the semantic search system implemented in Commit 16 to deeply integrate search capabilities with the game loop and core gameplay mechanics. By enhancing the relationship between search functionality and gameplay, we aim to create more intuitive, context-aware interactions that leverage the sophisticated search capabilities we've built. This integration will make search a fundamental aspect of gameplay rather than just a utility feature.

## Goals

1. Integrate search capabilities directly into the main game loop
2. Create context-aware search triggers based on player actions and game state
3. Implement search-driven gameplay mechanics and puzzles
4. Develop a system for dynamic content discovery through search
5. Enhance NPC interactions with search-powered knowledge and memory
6. Create a search-based hint and guidance system for players
7. Implement performance optimizations for search during gameplay

## Implementation Tasks

### 1. Game Loop Search Integration (`src/game_loop/core/search_integration.py`)

**Purpose**: Connect the main game loop with search functionality to enable seamless integration.

**Key Components**:
- Search middleware for the game loop
- Search events and triggers
- Search result handling and application
- Performance monitoring and optimizations

**Methods to Implement**:
```python
class GameLoopSearchIntegration:
    def __init__(self, game_loop, search_service, game_state_manager):
        self.game_loop = game_loop
        self.search_service = search_service
        self.state_manager = game_state_manager
        self._enabled_search_features = {}
        self._search_metrics = {}

    def initialize(self) -> None:
        """Initialize search integration with the game loop"""

    async def pre_process_input(self, player_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process player input before main game loop handling"""

    async def post_process_output(self, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance game output with search-derived information"""

    async def process_implicit_search(self, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform implicit searches based on game context without explicit player query"""

    async def handle_search_triggered_events(self, search_results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Handle game events triggered by search results"""

    def register_search_feature(self, feature_name: str, handler: Callable, priority: int = 5) -> None:
        """Register a search-based feature to be integrated in the game loop"""

    def unregister_search_feature(self, feature_name: str) -> bool:
        """Unregister a previously registered search feature"""

    async def search_with_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a search with current game context integrated"""

    async def update_search_metrics(self, metric_name: str, value: Any) -> None:
        """Update performance and usage metrics for search operations"""

    def get_search_metrics(self) -> Dict[str, Any]:
        """Get metrics about search usage and performance"""

    def _build_search_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure relevant search context from game state"""

    def _select_search_strategy(self, query: str, context: Dict[str, Any]) -> str:
        """Select the most appropriate search strategy based on input and context"""

    def _evaluate_result_relevance(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate and possibly rerank results based on current game context"""
```

### 2. Search-Based Gameplay Mechanics (`src/game_loop/gameplay/search_mechanics.py`)

**Purpose**: Implement gameplay mechanics that leverage search capabilities.

**Key Components**:
- Search-based puzzles and challenges
- Knowledge discovery mechanics
- Search triggers and rewards
- Dynamic difficulty adjustment based on search behavior

**Methods to Implement**:
```python
class SearchGameplayMechanics:
    def __init__(self, search_service, game_state_manager, event_system):
        self.search = search_service
        self.state = game_state_manager
        self.events = event_system
        self._registered_mechanics = {}

    async def register_search_mechanics(self) -> None:
        """Register all search-based gameplay mechanics"""

    async def check_knowledge_discovery(self, player_id: str, search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if a search result triggers knowledge discovery"""

    async def evaluate_puzzle_solution(self, puzzle_id: str, search_query: str, results: List[Dict[str, Any]]) -> bool:
        """Evaluate if a search query and results solve a specific puzzle"""

    async def generate_search_challenge(self, difficulty: float, player_id: str, topic: str = None) -> Dict[str, Any]:
        """Generate a search-based challenge for the player"""

    async def process_location_search_triggers(self, location_id: str, player_id: str) -> List[Dict[str, Any]]:
        """Process search triggers associated with a location when player enters"""

    async def update_npc_knowledge_from_search(self, npc_id: str, search_result: Dict[str, Any]) -> None:
        """Update an NPC's knowledge based on search results"""

    async def calculate_search_rewards(self, player_id: str, search_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate rewards for player based on search behavior and discoveries"""

    def _evaluate_search_difficulty(self, query: str, result_count: int, time_spent: float) -> float:
        """Evaluate the difficulty of a search based on various factors"""

    def _check_search_achievement_criteria(self, player_id: str, search_stats: Dict[str, Any]) -> List[str]:
        """Check if player's search behavior meets any achievement criteria"""

    async def _generate_follow_up_challenges(self, completed_challenge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate follow-up challenges based on a completed challenge"""
```

### 3. Context-Aware Search Triggers (`src/game_loop/search/context_triggers.py`)

**Purpose**: Create a system for triggering searches based on game context.

**Key Components**:
- Context evaluation rules
- Trigger conditions and priorities
- Implicit search generation
- Context mapping to search parameters

**Methods to Implement**:
```python
class ContextualSearchTrigger:
    def __init__(self, search_service, context_evaluator):
        self.search = search_service
        self.evaluator = context_evaluator
        self._registered_triggers = []
        self._trigger_history = {}

    def register_trigger(self, trigger_config: Dict[str, Any]) -> str:
        """Register a new contextual search trigger"""

    def unregister_trigger(self, trigger_id: str) -> bool:
        """Unregister an existing trigger"""

    async def evaluate_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate current context against all triggers"""

    async def execute_triggered_searches(self, context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute searches for all triggers that match the current context"""

    def update_trigger_history(self, trigger_id: str, execution_result: Dict[str, Any]) -> None:
        """Update history of trigger executions for throttling and analytics"""

    async def generate_implicit_query(self, trigger_config: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate an implicit search query based on trigger and context"""

    def _should_execute_trigger(self, trigger_id: str, context: Dict[str, Any]) -> bool:
        """Determine if a trigger should execute based on history and conditions"""

    def _extract_search_params(self, trigger_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search parameters from trigger configuration and context"""

    async def _combine_trigger_results(self, results_by_trigger: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine and prioritize results from multiple triggers"""
```

### 4. Dynamic Content Discovery System (`src/game_loop/content/discovery_system.py`)

**Purpose**: Create a system for dynamically discovering and presenting game content based on player searches and context.

**Key Components**:
- Content relevance scoring
- Progressive content revelation
- Player interest tracking
- Adaptive content recommendations

**Methods to Implement**:
```python
class ContentDiscoverySystem:
    def __init__(self, search_service, content_manager, player_profile_manager):
        self.search = search_service
        self.content = content_manager
        self.profiles = player_profile_manager
        self._discovery_settings = {}
        self._interest_models = {}

    async def discover_relevant_content(self, player_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover content relevant to the current player and context"""

    async def process_search_for_interests(self, player_id: str, search_query: str,
                                        search_results: List[Dict[str, Any]]) -> None:
        """Process search query and results to update player interest model"""

    async def recommend_content(self, player_id: str, category: str = None,
                             count: int = 3) -> List[Dict[str, Any]]:
        """Generate content recommendations for a player"""

    async def update_discovery_progress(self, player_id: str, content_id: str,
                                     interaction_type: str) -> Dict[str, Any]:
        """Update a player's content discovery progress"""

    async def generate_discovery_path(self, player_id: str, target_content_id: str) -> List[Dict[str, Any]]:
        """Generate a discovery path to guide player toward specific content"""

    def calculate_content_relevance(self, content: Dict[str, Any],
                                  player_interests: Dict[str, float]) -> float:
        """Calculate relevance score of content for a player"""

    async def get_player_discovery_stats(self, player_id: str) -> Dict[str, Any]:
        """Get statistics about a player's content discovery"""

    def _update_interest_model(self, player_id: str, query: str,
                             selected_content: List[str]) -> Dict[str, float]:
        """Update player's interest model based on search and selection"""

    async def _fetch_undiscovered_content(self, player_id: str,
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch content the player hasn't discovered yet"""

    def _apply_discovery_rules(self, content: List[Dict[str, Any]],
                              player_progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply rules to determine which content can be discovered"""
```

### 5. NPC Search Knowledge System (`src/game_loop/npcs/knowledge_system.py`)

**Purpose**: Enhance NPC interactions by integrating search capabilities into their knowledge and memory systems.

**Key Components**:
- NPC memory and knowledge representation
- Search-based response generation
- Knowledge acquisition from search
- Memory persistence and recall

**Methods to Implement**:
```python
class NPCSearchKnowledge:
    def __init__(self, search_service, npc_manager):
        self.search = search_service
        self.npc_manager = npc_manager
        self._knowledge_bases = {}

    async def initialize_npc_knowledge(self, npc_id: str) -> Dict[str, Any]:
        """Initialize or load an NPC's knowledge base"""

    async def query_npc_knowledge(self, npc_id: str, query: str,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Query an NPC's knowledge about a topic"""

    async def update_npc_knowledge(self, npc_id: str, new_knowledge: Dict[str, Any]) -> None:
        """Add new knowledge to an NPC's knowledge base"""

    async def generate_npc_response(self, npc_id: str, player_query: str,
                                 dialogue_context: Dict[str, Any]) -> str:
        """Generate an NPC response using their knowledge and search"""

    async def check_knowledge_conflicts(self, npc_id: str,
                                     new_information: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if new information conflicts with existing NPC knowledge"""

    async def forget_knowledge(self, npc_id: str, topic: str) -> None:
        """Make an NPC forget knowledge about a specific topic"""

    async def share_knowledge_between_npcs(self, source_npc_id: str,
                                        target_npc_id: str, topics: List[str]) -> Dict[str, Any]:
        """Share knowledge between NPCs"""

    def calculate_knowledge_confidence(self, npc_id: str, topic: str) -> float:
        """Calculate an NPC's confidence in their knowledge about a topic"""

    def _format_knowledge_for_query(self, knowledge_items: List[Dict[str, Any]]) -> str:
        """Format knowledge items for inclusion in a search query"""

    async def _search_with_npc_perspective(self, npc_id: str, query: str) -> Dict[str, Any]:
        """Perform search using an NPC's unique perspective and knowledge access"""
```

### 6. Player Guidance System (`src/game_loop/player/guidance_system.py`)

**Purpose**: Create a dynamic hint and guidance system powered by search.

**Key Components**:
- Contextual hint generation
- Progressive guidance based on player actions
- Difficulty-aware hint sophistication
- Search-based tutorial content

**Methods to Implement**:
```python
class SearchGuidanceSystem:
    def __init__(self, search_service, player_state_manager, difficulty_manager):
        self.search = search_service
        self.player_state = player_state_manager
        self.difficulty = difficulty_manager
        self._guidance_settings = {}
        self._hint_history = {}

    async def generate_contextual_hint(self, player_id: str, context: Dict[str, Any],
                                     hint_level: int = 1) -> Dict[str, Any]:
        """Generate a contextual hint for the player's current situation"""

    async def check_for_stuck_player(self, player_id: str, state_history: List[Dict[str, Any]]) -> bool:
        """Check if a player appears to be stuck and might need guidance"""

    async def generate_tutorial_content(self, player_id: str, topic: str) -> Dict[str, Any]:
        """Generate tutorial content for a specific gameplay topic"""

    async def suggest_next_action(self, player_id: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest possible next actions to the player"""

    async def explain_game_mechanic(self, player_id: str, mechanic_name: str) -> Dict[str, Any]:
        """Provide an explanation of a game mechanic"""

    async def track_hint_effectiveness(self, hint_id: str, was_helpful: bool) -> None:
        """Track whether hints are actually helping players"""

    def determine_hint_level(self, player_id: str, context: Dict[str, Any]) -> int:
        """Determine appropriate hint level based on player experience and settings"""

    async def get_player_guidance_history(self, player_id: str) -> Dict[str, Any]:
        """Get history of guidance provided to a player"""

    def _format_hint_for_difficulty(self, hint_content: str, difficulty_level: float) -> str:
        """Adjust hint clarity based on difficulty settings"""

    async def _search_for_hint_content(self, topic: str, context: Dict[str, Any],
                                     specificity: float) -> List[Dict[str, Any]]:
        """Search for appropriate hint content"""
```

### 7. Search Performance Optimization for Gameplay (`src/game_loop/search/gameplay_optimizations.py`)

**Purpose**: Optimize search performance specifically for gameplay contexts.

**Key Components**:
- Predictive pre-caching
- Performance profiling
- Search prioritization
- Resource management

**Methods to Implement**:
```python
class GameplaySearchOptimizer:
    def __init__(self, search_service, game_profiler):
        self.search = search_service
        self.profiler = game_profiler
        self._optimization_settings = {}
        self._performance_logs = {}

    async def initialize_optimization(self) -> None:
        """Initialize search optimization systems"""

    async def pre_cache_context_searches(self, upcoming_context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-cache likely searches based on upcoming game context"""

    def prioritize_search_tasks(self, pending_searches: List[Dict[str, Any]],
                              game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize pending searches based on game state and player needs"""

    async def optimize_search_parameters(self, search_params: Dict[str, Any],
                                      performance_target: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust search parameters to meet performance targets"""

    async def monitor_search_impact(self, search_id: str, game_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor how search operations impact game performance"""

    async def reduce_search_frequency(self, current_frequency: Dict[str, float],
                                   performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Adaptively reduce search frequency if performance issues are detected"""

    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for search optimization"""

    def predict_search_volume(self, upcoming_events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Predict upcoming search volume based on scheduled game events"""

    async def allocate_search_resources(self, available_resources: Dict[str, Any],
                                     expected_demand: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate available resources to different search operations"""

    def _calculate_search_priority_score(self, search_context: Dict[str, Any]) -> float:
        """Calculate priority score for a search operation"""

    async def _log_performance_data(self, search_id: str, metrics: Dict[str, Any]) -> None:
        """Log performance data for analysis and optimization"""
```

## File Structure

```
src/game_loop/
├── core/
│   ├── search_integration.py      # Main game loop integration
├── gameplay/
│   ├── search_mechanics.py        # Search-based gameplay mechanics
├── search/
│   ├── context_triggers.py        # Contextual search triggers
│   ├── gameplay_optimizations.py  # Performance optimizations
├── content/
│   ├── discovery_system.py        # Content discovery system
├── npcs/
│   ├── knowledge_system.py        # NPC knowledge integration
├── player/
│   ├── guidance_system.py         # Player guidance system
```

## Testing Strategy

### Unit Tests

1. **Game Loop Integration Tests** (`tests/unit/core/test_search_integration.py`):
   - Test pre-process input handling
   - Test post-process output enhancement
   - Test implicit search processing
   - Test search context building
   - Test feature registration/unregistration

2. **Gameplay Mechanics Tests** (`tests/unit/gameplay/test_search_mechanics.py`):
   - Test knowledge discovery
   - Test puzzle solution evaluation
   - Test challenge generation
   - Test reward calculation
   - Test location search triggers

3. **Context Trigger Tests** (`tests/unit/search/test_context_triggers.py`):
   - Test trigger registration
   - Test context evaluation
   - Test implicit query generation
   - Test trigger execution
   - Test result combination

4. **Content Discovery Tests** (`tests/unit/content/test_discovery_system.py`):
   - Test content relevance calculation
   - Test interest model updates
   - Test recommendation generation
   - Test discovery path creation
   - Test progress tracking

5. **NPC Knowledge Tests** (`tests/unit/npcs/test_knowledge_system.py`):
   - Test knowledge initialization
   - Test knowledge querying
   - Test response generation
   - Test knowledge sharing
   - Test knowledge conflicts

6. **Guidance System Tests** (`tests/unit/player/test_guidance_system.py`):
   - Test hint generation
   - Test stuck player detection
   - Test tutorial content
   - Test next action suggestions
   - Test hint effectiveness tracking

7. **Optimization Tests** (`tests/unit/search/test_gameplay_optimizations.py`):
   - Test pre-caching
   - Test search prioritization
   - Test parameter optimization
   - Test impact monitoring
   - Test resource allocation

### Integration Tests

1. **Game Loop Search Integration** (`tests/integration/core/test_search_game_loop.py`):
   - Test complete game loop with search integration
   - Test search in different game states
   - Test performance with search enabled
   - Test feature interactions

2. **Search & Gameplay Integration** (`tests/integration/gameplay/test_search_gameplay.py`):
   - Test search-based puzzles
   - Test discovery mechanics
   - Test challenge progression
   - Test search-triggered events
   - Test reward systems

3. **NPC & Search Integration** (`tests/integration/npcs/test_npc_search.py`):
   - Test NPC dialogues using search
   - Test knowledge acquisition
   - Test memory persistence
   - Test multi-NPC knowledge sharing
   - Test player-NPC knowledge transfer

4. **Player Guidance Integration** (`tests/integration/player/test_search_guidance.py`):
   - Test hint systems in gameplay
   - Test tutorial delivery
   - Test adaptive difficulty
   - Test guidance effectiveness
   - Test contextual help

### Performance Tests

1. **In-Game Search Performance** (`tests/performance/test_gameplay_search.py`):
   - Test search latency during gameplay
   - Test impact on game loop timing
   - Test concurrent search operations
   - Test optimization effectiveness
   - Test resource usage patterns

2. **Complex Gameplay Scenario Tests** (`tests/performance/test_search_scenarios.py`):
   - Test realistic gameplay scenarios with heavy search usage
   - Test search performance in different game settings
   - Test scaling with player count
   - Test event-heavy sequences
   - Test recovery after performance spikes

## Verification Criteria

### Functional Verification
- [ ] Search integrates seamlessly with main game loop
- [ ] Context-aware search correctly triggers based on player actions and game state
- [ ] Search-based gameplay mechanics function correctly
- [ ] Content discovery appropriately reveals game content based on context
- [ ] NPC knowledge system provides contextually relevant responses
- [ ] Player guidance system offers helpful, appropriate hints
- [ ] Pre-caching effectively improves performance for anticipated searches

### Performance Verification
- [ ] Search operations add < 50ms to game loop cycle time
- [ ] Gameplay remains smooth (60+ FPS) during search operations
- [ ] Concurrent search requests are properly prioritized
- [ ] Memory usage remains stable during extended gameplay
- [ ] Pre-caching reduces average search latency by > 40%
- [ ] Search system scales appropriately with increasing game world complexity
- [ ] Optimization systems effectively respond to performance constraints

### Player Experience Verification
- [ ] Search-based mechanics are intuitive and enhance gameplay
- [ ] Hints and guidance are perceived as helpful rather than intrusive
- [ ] Content discovery feels natural and rewarding
- [ ] NPC interactions leveraging search feel more intelligent
- [ ] Search-based puzzles can be solved through logical deduction
- [ ] Players can effectively find information they need
- [ ] Adaptation to player search behavior improves over time

## Dependencies

### Required Components
- Semantic Search System (from Commit 16)
- Game Loop System
- Event System
- Player State Manager
- NPC Manager
- Content Management System
- Game Profiler

### Configuration Updates
- Add search integration settings
- Add gameplay mechanics configuration
- Add NPC knowledge access settings
- Add performance threshold configuration
- Add player guidance preferences
- Add content discovery rules

## Integration Points

1. **With Game Loop**: Pre and post processing hooks
2. **With Event System**: Search-triggered events
3. **With Player System**: Player state and profile integration
4. **With NPC System**: Knowledge and dialogue integration
5. **With Content System**: Dynamic content discovery
6. **With Performance Monitoring**: Search impact analysis

## Migration Considerations

- Progressive feature enabling through configuration
- Fallback mechanisms for search failures
- Graceful degradation for performance-constrained environments
- Backward compatibility with existing gameplay systems
- Player preference settings for search features

## Code Quality Requirements

- [ ] All code passes black, ruff, and mypy linting
- [ ] Comprehensive docstrings for all public methods
- [ ] Type hints for all function parameters and return values
- [ ] Thorough error handling for all search operations
- [ ] Performance annotations for resource-intensive operations
- [ ] Detailed logging for search operations and impacts
- [ ] Runtime performance monitoring and alerting

## Documentation Updates

- [ ] Update game loop architecture documentation with search integration
- [ ] Create player guide for search-based gameplay features
- [ ] Document search-based puzzle design patterns for content creators
- [ ] Add NPC knowledge configuration guide
- [ ] Create performance optimization guide for different deployment environments
- [ ] Document new gameplay mechanics enabled by search
- [ ] Update API documentation with new search integration endpoints

## Future Considerations

This search integration with the game loop will serve as the foundation for:
- **Future**: Enhanced procedural content generation guided by search patterns
- **Future**: Multiplayer search-based cooperative challenges
- **Future**: Player behavior analysis based on search patterns
- **Future**: Adaptive storyline development based on search interests
- **Future**: Community knowledge base built from aggregate search patterns
- **Future**: Search-based competitive gameplay modes

The design should maintain flexibility for these future enhancements while ensuring current implementation is performant and enhances player experience.
