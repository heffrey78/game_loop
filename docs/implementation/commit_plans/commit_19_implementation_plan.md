# Commit 19: Physical Action Processing

## Overview

This commit implements the physical action processing system that handles movement, environment interaction, and spatial actions within the game world. Building upon the action type determination system from Commit 18, this implementation creates specialized processors for physical actions including navigation, environmental manipulation, and spatial puzzles. The system integrates with the existing semantic search, game state management, and world generation systems to provide realistic and engaging physical interactions.

## Goals

1. Implement comprehensive movement validation and processing system
2. Create environment interaction handling for physical manipulation
3. Add action feasibility checking based on game physics and constraints  
4. Develop spatial navigation system with pathfinding capabilities
5. Implement environmental state changes from physical actions
6. Create realistic physics-based interaction limitations
7. Integrate with existing world state and location management systems
8. Add support for complex multi-step physical actions

## Implementation Tasks

### 1. Physical Action Processor (`src/game_loop/core/command_handlers/physical_action_processor.py`)

**Purpose**: Core processor for handling all physical actions in the game world.

**Key Components**:
- Movement command processing
- Environment interaction validation
- Physics constraint checking
- Action feasibility analysis
- State update coordination

**Methods to Implement**:
```python
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PhysicalActionType(Enum):
    MOVEMENT = "movement"
    MANIPULATION = "manipulation"
    CLIMBING = "climbing"
    JUMPING = "jumping"
    PUSHING = "pushing"
    PULLING = "pulling"
    OPENING = "opening"
    CLOSING = "closing"
    BREAKING = "breaking"
    BUILDING = "building"

@dataclass
class PhysicalActionResult:
    success: bool
    action_type: PhysicalActionType
    affected_entities: List[str]
    state_changes: Dict[str, Any]
    energy_cost: float
    time_elapsed: float
    side_effects: List[str]
    description: str
    error_message: Optional[str] = None

class PhysicalActionProcessor:
    def __init__(self, game_state_manager, search_service, physics_engine):
        self.state_manager = game_state_manager
        self.search_service = search_service
        self.physics = physics_engine
        self._action_handlers = {}
        self._constraint_validators = []
        self._initialize_handlers()

    async def process_physical_action(self, action_classification, context: Dict[str, Any]) -> PhysicalActionResult:
        """Main entry point for processing physical actions"""

    async def validate_action_feasibility(self, action_type: PhysicalActionType, 
                                        target_entities: List[str],
                                        context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if the physical action is feasible in current context"""

    async def calculate_action_requirements(self, action_type: PhysicalActionType,
                                          target_entities: List[str]) -> Dict[str, Any]:
        """Calculate energy, time, and resource requirements for action"""

    async def execute_movement_action(self, direction: str, distance: Optional[float],
                                    context: Dict[str, Any]) -> PhysicalActionResult:
        """Execute movement to a new location or direction"""

    async def execute_manipulation_action(self, target_entity: str, action_verb: str,
                                        context: Dict[str, Any]) -> PhysicalActionResult:
        """Execute object manipulation (push, pull, lift, etc.)"""

    async def execute_environmental_action(self, target_entity: str, action_type: PhysicalActionType,
                                         context: Dict[str, Any]) -> PhysicalActionResult:
        """Execute environmental interactions (climbing, jumping, etc.)"""

    async def apply_physics_constraints(self, action_type: PhysicalActionType,
                                      entities: List[str],
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply physics constraints and limitations to actions"""

    async def update_world_state(self, action_result: PhysicalActionResult,
                               context: Dict[str, Any]) -> None:
        """Update world state based on successful physical action"""

    async def calculate_side_effects(self, action_type: PhysicalActionType,
                                   entities: List[str],
                                   context: Dict[str, Any]) -> List[str]:
        """Calculate secondary effects of physical actions"""

    def register_action_handler(self, action_type: PhysicalActionType, 
                              handler: Callable) -> None:
        """Register a handler for a specific physical action type"""

    def register_constraint_validator(self, validator: Callable) -> None:
        """Register a physics constraint validator"""

    async def _validate_entity_accessibility(self, entity_id: str, 
                                           context: Dict[str, Any]) -> bool:
        """Check if entity is accessible for physical interaction"""

    async def _calculate_energy_cost(self, action_type: PhysicalActionType,
                                   difficulty: float) -> float:
        """Calculate energy cost for physical action"""

    def _initialize_handlers(self) -> None:
        """Initialize default action handlers"""
```

### 2. Movement System (`src/game_loop/core/movement/movement_manager.py`)

**Purpose**: Handle player and entity movement through the game world.

**Key Components**:
- Direction parsing and validation
- Location transition management
- Path finding and navigation
- Movement constraints and obstacles
- Travel time and energy calculations

**Methods to Implement**:
```python
class MovementManager:
    def __init__(self, world_state_manager, location_service, physics_engine):
        self.world_state = world_state_manager
        self.locations = location_service
        self.physics = physics_engine
        self._movement_cache = {}
        self._pathfinding_cache = {}

    async def process_movement_command(self, player_id: str, direction: str,
                                     context: Dict[str, Any]) -> PhysicalActionResult:
        """Process a movement command from the player"""

    async def validate_movement(self, from_location: str, to_location: str,
                              player_state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate that movement between locations is possible"""

    async def find_path(self, start_location: str, target_location: str,
                       constraints: Dict[str, Any]) -> Optional[List[str]]:
        """Find a path between two locations"""

    async def calculate_travel_time(self, from_location: str, to_location: str,
                                  movement_type: str) -> float:
        """Calculate time required for movement"""

    async def get_available_exits(self, location_id: str,
                                player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available exits from current location"""

    async def handle_location_transition(self, player_id: str, from_location: str,
                                       to_location: str) -> Dict[str, Any]:
        """Handle the transition between locations"""

    async def check_movement_obstacles(self, location_id: str, direction: str,
                                     player_state: Dict[str, Any]) -> List[str]:
        """Check for obstacles preventing movement"""

    async def apply_movement_effects(self, player_id: str, movement_result: PhysicalActionResult) -> None:
        """Apply effects of movement on player state"""

    def parse_direction_input(self, direction_input: str) -> Optional[str]:
        """Parse player direction input into standardized direction"""

    async def _update_location_state(self, location_id: str, changes: Dict[str, Any]) -> None:
        """Update location state after movement events"""

    async def _validate_location_capacity(self, location_id: str) -> bool:
        """Check if location can accommodate another entity"""

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction string to standard format"""
```

### 3. Environment Interaction System (`src/game_loop/core/environment/interaction_manager.py`)

**Purpose**: Manage interactions between entities and environmental elements.

**Key Components**:
- Object manipulation mechanics
- Environmental state tracking
- Interaction constraint validation
- Dynamic environment responses
- Tool and skill requirements

**Methods to Implement**:
```python
class EnvironmentInteractionManager:
    def __init__(self, world_state_manager, object_manager, physics_engine):
        self.world_state = world_state_manager
        self.objects = object_manager
        self.physics = physics_engine
        self._interaction_handlers = {}
        self._environmental_states = {}

    async def process_environment_interaction(self, player_id: str, target_entity: str,
                                            interaction_type: str,
                                            context: Dict[str, Any]) -> PhysicalActionResult:
        """Process interaction with environmental element"""

    async def validate_interaction_requirements(self, interaction_type: str,
                                              target_entity: str,
                                              player_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that player meets requirements for interaction"""

    async def execute_object_manipulation(self, object_id: str, manipulation_type: str,
                                        player_state: Dict[str, Any]) -> PhysicalActionResult:
        """Execute manipulation of environmental objects"""

    async def handle_container_interactions(self, container_id: str, action: str,
                                          item_id: Optional[str],
                                          context: Dict[str, Any]) -> PhysicalActionResult:
        """Handle interactions with containers (chests, doors, etc.)"""

    async def process_tool_usage(self, tool_id: str, target_id: str, action: str,
                               context: Dict[str, Any]) -> PhysicalActionResult:
        """Process usage of tools on environmental targets"""

    async def handle_environmental_puzzles(self, puzzle_element: str, action: str,
                                         context: Dict[str, Any]) -> PhysicalActionResult:
        """Handle interactions with puzzle elements"""

    async def update_environmental_state(self, entity_id: str, state_changes: Dict[str, Any]) -> None:
        """Update environmental entity state"""

    async def check_interaction_side_effects(self, interaction_type: str, entities: List[str],
                                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for side effects of environmental interactions"""

    async def get_interaction_options(self, entity_id: str, player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available interaction options for an entity"""

    def register_interaction_handler(self, interaction_type: str, handler: Callable) -> None:
        """Register handler for specific interaction type"""

    async def _validate_object_accessibility(self, object_id: str, player_position: Dict[str, Any]) -> bool:
        """Check if object is within reach for interaction"""

    async def _calculate_interaction_difficulty(self, interaction_type: str, target_entity: str) -> float:
        """Calculate difficulty level for interaction"""

    async def _apply_wear_and_tear(self, entity_id: str, usage_intensity: float) -> None:
        """Apply wear and tear effects to frequently used objects"""
```

### 4. Physics Constraint System (`src/game_loop/core/physics/constraint_engine.py`)

**Purpose**: Apply realistic physics constraints to physical actions.

**Key Components**:
- Weight and size limitations
- Structural integrity checking
- Energy and stamina requirements
- Environmental physics simulation
- Collision detection and response

**Methods to Implement**:
```python
class PhysicsConstraintEngine:
    def __init__(self, configuration_manager):
        self.config = configuration_manager
        self._constraint_rules = {}
        self._physics_constants = {}
        self._load_physics_configuration()

    async def validate_physical_constraints(self, action_type: PhysicalActionType,
                                          entities: List[str],
                                          player_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate physical constraints for an action"""

    async def calculate_strength_requirements(self, action_type: PhysicalActionType,
                                            target_mass: float,
                                            difficulty_modifiers: Dict[str, float]) -> float:
        """Calculate strength requirements for physical action"""

    async def check_spatial_constraints(self, action_location: Dict[str, Any],
                                      required_space: Dict[str, float],
                                      entities: List[str]) -> bool:
        """Check if sufficient space exists for action"""

    async def validate_structural_integrity(self, entity_id: str, applied_force: float,
                                          force_type: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if entity can withstand applied force"""

    async def calculate_energy_expenditure(self, action_type: PhysicalActionType,
                                         duration: float,
                                         intensity: float,
                                         player_stats: Dict[str, Any]) -> float:
        """Calculate energy cost for physical action"""

    async def simulate_collision_effects(self, entity1: str, entity2: str,
                                       impact_force: float) -> Dict[str, Any]:
        """Simulate effects of collision between entities"""

    async def check_balance_and_stability(self, entity_id: str, action_type: PhysicalActionType,
                                        environmental_factors: Dict[str, Any]) -> bool:
        """Check if entity maintains balance during action"""

    async def apply_gravity_effects(self, entity_id: str, height: float,
                                  support_structure: Optional[str]) -> Dict[str, Any]:
        """Apply gravity effects to elevated entities"""

    def register_constraint_rule(self, rule_name: str, validator: Callable,
                                priority: int = 5) -> None:
        """Register a new physics constraint rule"""

    async def get_constraint_violations(self, action_type: PhysicalActionType,
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of constraint violations for action"""

    def _load_physics_configuration(self) -> None:
        """Load physics constants and rules from configuration"""

    async def _calculate_leverage_factor(self, tool_id: Optional[str], action_type: PhysicalActionType) -> float:
        """Calculate leverage factor when using tools"""

    def _get_environmental_resistance(self, location_id: str, action_type: PhysicalActionType) -> float:
        """Get environmental resistance factors"""
```

### 5. Spatial Navigation System (`src/game_loop/core/navigation/spatial_navigator.py`)

**Purpose**: Advanced navigation and pathfinding capabilities.

**Key Components**:
- 3D spatial mapping
- Intelligent pathfinding algorithms
- Dynamic obstacle avoidance
- Navigation assistance and hints
- Landmark-based navigation

**Methods to Implement**:
```python
class SpatialNavigator:
    def __init__(self, world_graph_manager, location_service, search_service):
        self.world_graph = world_graph_manager
        self.locations = location_service
        self.search = search_service
        self._navigation_cache = {}
        self._landmark_map = {}

    async def find_optimal_path(self, start_location: str, target_location: str,
                              preferences: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Find optimal path between locations considering preferences"""

    async def get_navigation_directions(self, current_location: str, target_location: str,
                                      player_knowledge: Dict[str, Any]) -> List[str]:
        """Get step-by-step navigation directions"""

    async def identify_landmarks(self, location_id: str, visibility_range: float) -> List[Dict[str, Any]]:
        """Identify notable landmarks visible from location"""

    async def calculate_travel_estimates(self, path: List[str], movement_speed: float,
                                       obstacles: List[str]) -> Dict[str, Any]:
        """Calculate time and energy estimates for travel"""

    async def find_alternative_routes(self, blocked_path: List[str], target: str,
                                    constraints: Dict[str, Any]) -> List[List[str]]:
        """Find alternative routes when primary path is blocked"""

    async def update_navigation_knowledge(self, player_id: str, discovered_path: List[str],
                                        path_quality: float) -> None:
        """Update player's navigation knowledge with discovered paths"""

    async def get_exploration_suggestions(self, current_location: str,
                                        exploration_history: List[str]) -> List[Dict[str, Any]]:
        """Suggest unexplored areas for discovery"""

    async def validate_path_accessibility(self, path: List[str], player_capabilities: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that player can traverse entire path"""

    async def create_mental_map(self, player_id: str, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create player's mental map based on exploration"""

    def register_navigation_algorithm(self, algorithm_name: str, algorithm: Callable) -> None:
        """Register custom pathfinding algorithm"""

    async def _calculate_path_difficulty(self, path: List[str], player_stats: Dict[str, Any]) -> float:
        """Calculate overall difficulty of traversing path"""

    async def _identify_chokepoints(self, path: List[str]) -> List[Dict[str, Any]]:
        """Identify potential problem areas along path"""

    def _estimate_visibility_range(self, location_id: str, weather_conditions: Dict[str, Any]) -> float:
        """Estimate visibility range considering environmental factors"""
```

### 6. Physical Action Integration (`src/game_loop/core/actions/physical_action_integration.py`)

**Purpose**: Integrate physical action processing with the main game systems.

**Key Components**:
- Action classification integration
- State management coordination
- Event system integration
- Performance optimization
- Error handling and recovery

**Methods to Implement**:
```python
class PhysicalActionIntegration:
    def __init__(self, physical_processor, movement_manager, environment_manager,
                 physics_engine, game_state_manager):
        self.physical_processor = physical_processor
        self.movement = movement_manager
        self.environment = environment_manager
        self.physics = physics_engine
        self.state = game_state_manager
        self._action_metrics = {}

    async def process_classified_physical_action(self, action_classification,
                                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for processing classified physical actions"""

    async def coordinate_multi_step_actions(self, action_sequence: List[Dict[str, Any]],
                                          context: Dict[str, Any]) -> List[PhysicalActionResult]:
        """Coordinate execution of multi-step physical actions"""

    async def handle_action_interruptions(self, active_action: Dict[str, Any],
                                        interruption_event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interruptions during physical action execution"""

    async def optimize_action_performance(self, action_type: PhysicalActionType,
                                        frequency_data: Dict[str, Any]) -> None:
        """Optimize performance for frequently used actions"""

    async def validate_action_chain_feasibility(self, action_chain: List[Dict[str, Any]],
                                              context: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        """Validate that a chain of actions can be completed"""

    async def apply_learning_effects(self, player_id: str, action_type: PhysicalActionType,
                                   success_rate: float) -> None:
        """Apply skill learning effects from repeated actions"""

    async def handle_concurrent_actions(self, actions: List[Dict[str, Any]],
                                      context: Dict[str, Any]) -> List[PhysicalActionResult]:
        """Handle multiple physical actions happening simultaneously"""

    async def generate_action_feedback(self, action_result: PhysicalActionResult,
                                     context: Dict[str, Any]) -> str:
        """Generate descriptive feedback for physical action results"""

    async def update_world_physics_state(self, action_results: List[PhysicalActionResult]) -> None:
        """Update world physics state based on action results"""

    def register_action_integration_hook(self, hook_type: str, handler: Callable) -> None:
        """Register integration hooks for action processing"""

    async def _validate_resource_availability(self, required_resources: Dict[str, Any],
                                            player_state: Dict[str, Any]) -> bool:
        """Validate that player has required resources for action"""

    async def _calculate_composite_action_cost(self, actions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate total cost for multiple related actions"""

    def _should_cache_action_result(self, action_type: PhysicalActionType,
                                  context: Dict[str, Any]) -> bool:
        """Determine if action result should be cached"""
```

## File Structure

```
src/game_loop/
├── core/
│   ├── command_handlers/
│   │   ├── physical_action_processor.py     # Main physical action processor
│   │   └── __init__.py
│   ├── movement/
│   │   ├── __init__.py
│   │   └── movement_manager.py              # Movement and navigation
│   ├── environment/
│   │   ├── __init__.py
│   │   └── interaction_manager.py           # Environment interactions
│   ├── physics/
│   │   ├── __init__.py
│   │   └── constraint_engine.py             # Physics constraints
│   ├── navigation/
│   │   ├── __init__.py
│   │   └── spatial_navigator.py             # Advanced navigation
│   ├── actions/
│   │   ├── physical_action_integration.py   # Integration layer
│   │   └── __init__.py
```

## Testing Strategy

### Unit Tests

1. **Physical Action Processor Tests** (`tests/unit/core/command_handlers/test_physical_action_processor.py`):
   - Test action classification handling
   - Test feasibility validation
   - Test physics constraint application
   - Test state update coordination
   - Test error handling and recovery

2. **Movement Manager Tests** (`tests/unit/core/movement/test_movement_manager.py`):
   - Test direction parsing
   - Test location transition validation
   - Test pathfinding algorithms
   - Test movement constraint checking
   - Test travel time calculations

3. **Environment Interaction Tests** (`tests/unit/core/environment/test_interaction_manager.py`):
   - Test object manipulation mechanics
   - Test container interactions
   - Test tool usage validation
   - Test puzzle element handling
   - Test side effect calculations

4. **Physics Constraint Tests** (`tests/unit/core/physics/test_constraint_engine.py`):
   - Test strength requirement calculations
   - Test spatial constraint validation
   - Test structural integrity checks
   - Test energy expenditure calculations
   - Test collision effect simulation

5. **Spatial Navigator Tests** (`tests/unit/core/navigation/test_spatial_navigator.py`):
   - Test pathfinding accuracy
   - Test navigation direction generation
   - Test landmark identification
   - Test alternative route finding
   - Test mental map creation

### Integration Tests

1. **Physical Action Flow Tests** (`tests/integration/core/test_physical_action_flow.py`):
   - Test complete action processing pipeline
   - Test integration with action classification
   - Test state management updates
   - Test search system integration
   - Test event emission and handling

2. **Movement Integration Tests** (`tests/integration/movement/test_movement_integration.py`):
   - Test movement with world state updates
   - Test location transition effects
   - Test movement energy and time costs
   - Test movement obstacle handling
   - Test pathfinding with dynamic obstacles

3. **Environment System Tests** (`tests/integration/environment/test_environment_system.py`):
   - Test environment interactions with physics
   - Test tool and object interactions
   - Test environmental state persistence
   - Test puzzle solution validation
   - Test complex interaction chains

4. **Physics Simulation Tests** (`tests/integration/physics/test_physics_simulation.py`):
   - Test realistic physics behavior
   - Test constraint enforcement
   - Test energy and stamina systems
   - Test collision and damage effects
   - Test environmental physics factors

### Performance Tests

1. **Action Processing Performance** (`tests/performance/test_physical_action_performance.py`):
   - Test action processing speed
   - Test pathfinding performance
   - Test physics calculation optimization
   - Test concurrent action handling
   - Test memory usage during complex actions

2. **Navigation Performance** (`tests/performance/test_navigation_performance.py`):
   - Test pathfinding algorithm efficiency
   - Test spatial query performance
   - Test navigation cache effectiveness
   - Test large world navigation
   - Test real-time navigation updates

## Verification Criteria

### Functional Verification
- [ ] Movement commands work correctly for all directions and distances
- [ ] Environment interactions update world state appropriately
- [ ] Physics constraints prevent impossible actions realistically
- [ ] Pathfinding provides efficient routes between locations
- [ ] Action feasibility checking accurately prevents invalid actions
- [ ] Multi-step actions execute in correct sequence
- [ ] Energy and time costs are calculated realistically
- [ ] Environmental puzzles can be solved through physical actions

### Performance Verification
- [ ] Simple movement actions complete in < 50ms
- [ ] Complex pathfinding completes in < 200ms
- [ ] Physics calculations complete in < 100ms
- [ ] Environment interactions complete in < 150ms
- [ ] Concurrent action processing scales linearly
- [ ] Memory usage remains stable during extended gameplay
- [ ] Action caching improves repeated action performance

### Integration Verification
- [ ] Integrates seamlessly with action classification system
- [ ] Updates game state consistently across all systems
- [ ] Works with semantic search for target validation
- [ ] Maintains world physics consistency
- [ ] Handles errors gracefully without breaking game state
- [ ] Emits appropriate events for other systems
- [ ] Provides clear feedback for all action results

## Dependencies

### Existing Components
- Action Type Classifier (from Commit 18)
- Game State Manager (from Commit 9)
- Semantic Search Service (from Commit 16)
- World State Manager (from Commit 9)
- Location Management System
- Object Management System

### New Configuration Requirements
- Physics constants and constraints configuration
- Movement speed and energy cost settings
- Environment interaction rules configuration
- Pathfinding algorithm preferences
- Action feasibility thresholds

## Integration Points

1. **With Action Classification**: Process classified physical actions
2. **With Game State**: Update world and player state after actions
3. **With Semantic Search**: Validate action targets and find alternatives
4. **With World Generation**: Handle dynamic world changes from actions
5. **With Event System**: Emit events for action results and state changes
6. **With NPC System**: Handle NPC reactions to physical actions

## Migration Considerations

- Design for extensibility to add new physical action types
- Create clear interfaces for physics constraint validators
- Plan for backward compatibility as physics systems evolve
- Consider versioning for constraint rules and physics constants
- Allow for A/B testing of different physics behaviors

## Code Quality Requirements

- [ ] All code passes black, ruff, and mypy linting
- [ ] Comprehensive docstrings for all public methods
- [ ] Type hints for all function parameters and return values
- [ ] Error handling for all external dependencies
- [ ] Logging for physics calculations and constraint violations
- [ ] Performance monitoring for critical action paths
- [ ] Unit test coverage > 90%

## Documentation Updates

- [ ] Create physical action processing guide
- [ ] Document physics constraint system
- [ ] Add examples of complex physical interactions
- [ ] Create movement and navigation guide
- [ ] Document environment interaction mechanics
- [ ] Add troubleshooting guide for physics issues
- [ ] Update architecture diagram with physical systems

## Future Considerations

This physical action processing system will serve as the foundation for:
- **Commit 20**: Object Interaction System (detailed inventory management)
- **Commit 21**: Quest Interaction System (quest-driven physical tasks)
- **Future**: Advanced physics simulation (fluids, thermodynamics)
- **Future**: Complex mechanical systems (machines, vehicles)
- **Future**: Collaborative physical actions (team lifting, construction)
- **Future**: Skill-based physical action improvements
- **Future**: Dynamic environmental physics changes
- **Future**: Virtual reality integration for physical actions

The design should be flexible enough to support these future enhancements while maintaining realistic physics behavior and high performance for basic physical actions.