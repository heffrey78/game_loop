# Commit 12: Initial Game Flow Integration - Implementation Plan

## Overview

This commit integrates the foundational components built in previous commits into a cohesive game flow system. The goal is to connect input processing to game state management, link NLP processing to database queries, implement basic action processing flow, and add simple response generation. This creates the first end-to-end game experience where players can interact with the world through natural language and see their actions reflected in persistent game state.

## Goals

- Connect InputProcessor and NLPProcessor to GameStateManager for context-aware processing
- Integrate database queries with semantic search capabilities for NLP processing
- Implement a basic action processing pipeline that handles common game actions
- Connect OutputGenerator to action results for consistent response formatting
- Create a complete input → processing → action → state update → output flow
- Ensure all components work together seamlessly in the main game loop
- Maintain backward compatibility with existing functionality

## Technical Requirements

### Dependencies
- All previous commit implementations (Commits 1-11)
- Working InputProcessor, NLPProcessor, GameStateManager, and OutputGenerator
- Functional database connection with ORM models
- Rich text formatting and template system

### Core Integration Points

1. **Input-to-State Integration**: InputProcessor receives game context from GameStateManager
2. **NLP-to-Database Integration**: NLP queries use database for object/location resolution
3. **Action Processing Pipeline**: Standardized flow for executing and responding to actions
4. **State-to-Output Integration**: Action results generate formatted responses via OutputGenerator
5. **Main Game Loop Coordination**: All components work together in GameLoop.run()

## Implementation Details

### 1. Enhanced Input Processing Integration (`src/game_loop/core/input_processor.py`)

```python
from typing import Any, Dict, Optional
from game_loop.state.manager import GameStateManager
from game_loop.core.nlp_processor import NLPProcessor

class InputProcessor:
    """Enhanced input processor with game state integration."""

    def __init__(self, nlp_processor: NLPProcessor, game_state_manager: GameStateManager):
        self.nlp_processor = nlp_processor
        self.game_state_manager = game_state_manager
        # ...existing initialization...

    async def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input with game state context."""
        # Get current game context if not provided
        if context is None:
            context = await self._get_current_context()

        # Process with enhanced context
        processed_input = await self._process_with_context(user_input, context)
        return processed_input

    async def _get_current_context(self) -> Dict[str, Any]:
        """Get current game context for input processing."""
        player_state = await self.game_state_manager.get_current_player_state()
        world_state = await self.game_state_manager.get_current_world_state()

        return {
            "location_id": player_state.location_id,
            "inventory": player_state.inventory,
            "nearby_objects": world_state.get_location_objects(player_state.location_id),
            "nearby_npcs": world_state.get_location_npcs(player_state.location_id),
            "available_exits": world_state.get_location_exits(player_state.location_id)
        }
```

### 2. NLP-Database Integration (`src/game_loop/core/nlp_processor.py`)

```python
from game_loop.database.repositories.world import WorldRepository
from game_loop.database.repositories.player import PlayerRepository

class NLPProcessor:
    """Enhanced NLP processor with database integration."""

    def __init__(self, ollama_client, world_repository: WorldRepository, player_repository: PlayerRepository):
        # ...existing initialization...
        self.world_repository = world_repository
        self.player_repository = player_repository

    async def resolve_objects(self, object_references: List[str], location_id: str) -> List[Dict[str, Any]]:
        """Resolve object references using database lookup."""
        resolved_objects = []

        # Get objects at current location
        location_objects = await self.world_repository.get_location_objects(location_id)

        for ref in object_references:
            # Try exact name match first
            exact_match = next((obj for obj in location_objects if obj.name.lower() == ref.lower()), None)
            if exact_match:
                resolved_objects.append({
                    "id": exact_match.id,
                    "name": exact_match.name,
                    "type": exact_match.object_type,
                    "match_confidence": 1.0
                })
                continue

            # Use semantic search for partial matches
            semantic_matches = await self._semantic_object_search(ref, location_objects)
            if semantic_matches:
                resolved_objects.extend(semantic_matches)

        return resolved_objects

    async def resolve_locations(self, location_references: List[str], current_location_id: str) -> List[Dict[str, Any]]:
        """Resolve location references using database and exit information."""
        # Get available exits from current location
        available_exits = await self.world_repository.get_location_exits(current_location_id)

        resolved_locations = []
        for ref in location_references:
            # Match against available exits
            exit_match = next((exit for exit in available_exits if ref.lower() in exit.direction.lower()), None)
            if exit_match:
                resolved_locations.append({
                    "direction": exit_match.direction,
                    "destination_id": exit_match.destination_id,
                    "destination_name": exit_match.destination_name,
                    "match_confidence": 0.9
                })

        return resolved_locations
```

### 3. Action Processing Pipeline (`src/game_loop/core/action_processor.py`)

```python
from typing import Any, Dict, Optional
from game_loop.state.models import ActionResult
from game_loop.state.manager import GameStateManager
from game_loop.core.output_generator import OutputGenerator

class ActionProcessor:
    """Central action processing pipeline."""

    def __init__(self, game_state_manager: GameStateManager, output_generator: OutputGenerator):
        self.game_state_manager = game_state_manager
        self.output_generator = output_generator
        self._action_handlers = self._register_action_handlers()

    async def process_action(self, action_data: Dict[str, Any]) -> ActionResult:
        """Process an action and return the result."""
        action_type = action_data.get("action_type", "unknown")

        # Get appropriate handler
        handler = self._action_handlers.get(action_type, self._handle_unknown_action)

        # Execute action
        try:
            result = await handler(action_data)

            # Update game state if action was successful
            if result.success:
                await self.game_state_manager.update_after_action(result)

            return result

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"An error occurred while processing your action: {str(e)}",
                error_details={"exception": str(e), "action_type": action_type}
            )

    async def _handle_movement(self, action_data: Dict[str, Any]) -> ActionResult:
        """Handle movement actions."""
        direction = action_data.get("direction")
        player_state = await self.game_state_manager.get_current_player_state()

        # Validate movement
        available_exits = await self.game_state_manager.get_available_exits(player_state.location_id)
        valid_exit = next((exit for exit in available_exits if exit.direction.lower() == direction.lower()), None)

        if not valid_exit:
            return ActionResult(
                success=False,
                feedback_message=f"You cannot go {direction} from here.",
                action_type="movement"
            )

        # Execute movement
        new_location = await self.game_state_manager.get_location(valid_exit.destination_id)

        return ActionResult(
            success=True,
            feedback_message=f"You go {direction}.",
            action_type="movement",
            location_change=True,
            new_location_id=valid_exit.destination_id,
            location_data=new_location.model_dump()
        )

    async def _handle_interaction(self, action_data: Dict[str, Any]) -> ActionResult:
        """Handle object interaction actions."""
        object_id = action_data.get("object_id")
        interaction_type = action_data.get("interaction_type", "examine")

        # Get object data
        game_object = await self.game_state_manager.get_object(object_id)
        if not game_object:
            return ActionResult(
                success=False,
                feedback_message="You don't see that here.",
                action_type="interaction"
            )

        # Handle different interaction types
        if interaction_type == "examine":
            return ActionResult(
                success=True,
                feedback_message=f"You examine the {game_object.name}. {game_object.description}",
                action_type="interaction",
                object_data=game_object.model_dump()
            )
        elif interaction_type == "take":
            return await self._handle_take_object(game_object, action_data)
        # Add more interaction types as needed

        return ActionResult(
            success=False,
            feedback_message=f"You cannot {interaction_type} the {game_object.name}.",
            action_type="interaction"
        )

    def _register_action_handlers(self) -> Dict[str, Any]:
        """Register action type handlers."""
        return {
            "movement": self._handle_movement,
            "interaction": self._handle_interaction,
            "dialogue": self._handle_dialogue,
            "system": self._handle_system_command,
            "query": self._handle_query
        }
```

### 4. Enhanced Game Loop Integration (`src/game_loop/core/game_loop.py`)

```python
from game_loop.core.action_processor import ActionProcessor
from game_loop.core.output_generator import OutputGenerator
from game_loop.state.manager import GameStateManager

class GameLoop:
    """Enhanced game loop with integrated action processing."""

    def __init__(
        self,
        input_processor,
        config_manager,
        game_state_manager: GameStateManager,
        output_generator: OutputGenerator
    ):
        self.input_processor = input_processor
        self.config_manager = config_manager
        self.game_state_manager = game_state_manager
        self.output_generator = output_generator
        self.action_processor = ActionProcessor(game_state_manager, output_generator)

    async def run(self):
        """Run the main game loop with integrated processing."""
        # Display initial game state
        await self._display_initial_state()

        while True:
            try:
                # Get user input
                user_input = input("> ").strip()

                if not user_input:
                    continue

                # Check for meta commands first
                if await self._handle_meta_commands(user_input):
                    continue

                # Process the turn
                await self._process_turn(user_input)

            except KeyboardInterrupt:
                await self._handle_exit()
                break
            except Exception as e:
                await self.output_generator.display_error(f"An unexpected error occurred: {str(e)}")

    async def _process_turn(self, user_input: str) -> None:
        """Process a complete game turn."""
        # Get current context
        context = await self._get_turn_context()

        # Process input with context
        processed_input = await self.input_processor.process(user_input, context)

        # Convert processed input to action data
        action_data = await self._convert_to_action_data(processed_input)

        # Process the action
        action_result = await self.action_processor.process_action(action_data)

        # Generate and display response
        await self.output_generator.generate_response(action_result)

        # Handle any location changes
        if action_result.location_change:
            await self._display_new_location(action_result.new_location_id)

    async def _get_turn_context(self) -> Dict[str, Any]:
        """Get context for the current turn."""
        player_state = await self.game_state_manager.get_current_player_state()
        return {
            "player_state": player_state.model_dump(),
            "location_id": player_state.location_id,
            "session_id": await self.game_state_manager.get_current_session_id()
        }

    async def _handle_meta_commands(self, user_input: str) -> bool:
        """Handle meta-game commands like save, load, quit."""
        command = user_input.lower().strip()

        if command in ["quit", "exit", "q"]:
            await self._handle_exit()
            return True
        elif command == "save":
            await self._handle_save()
            return True
        elif command == "load":
            await self._handle_load()
            return True
        elif command in ["look", "l"]:
            await self._handle_look()
            return True
        elif command in ["inventory", "i"]:
            await self._handle_inventory()
            return True

        return False

    async def _display_initial_state(self) -> None:
        """Display the initial game state."""
        player_state = await self.game_state_manager.get_current_player_state()
        location_description = await self.game_state_manager.get_location_description(player_state.location_id)

        await self.output_generator.display_system_message("Welcome to the Game Loop!")
        await self.output_generator.display_location(location_description)
```

### 5. Database Query Integration (`src/game_loop/database/repositories/search.py`)

```python
from typing import List, Dict, Any, Optional
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from game_loop.database.models.world import Location, WorldObject, NPC

class SearchRepository:
    """Repository for semantic search and query operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def search_objects_by_name(self, name_query: str, location_id: str) -> List[WorldObject]:
        """Search for objects by name similarity."""
        stmt = select(WorldObject).where(
            WorldObject.location_id == location_id,
            WorldObject.name.ilike(f"%{name_query}%")
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def search_objects_semantic(self, query_vector: List[float], location_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for objects using semantic similarity (preparation for vector search)."""
        # Placeholder for future vector search implementation
        # For now, fall back to text search
        return await self.search_objects_by_name(query_vector[0] if query_vector else "", location_id)

    async def get_location_context(self, location_id: str) -> Dict[str, Any]:
        """Get full context for a location including objects, NPCs, and exits."""
        # Get location
        location_stmt = select(Location).where(Location.id == location_id)
        location_result = await self.session.execute(location_stmt)
        location = location_result.scalar_one_or_none()

        if not location:
            return {}

        # Get objects
        objects_stmt = select(WorldObject).where(WorldObject.location_id == location_id)
        objects_result = await self.session.execute(objects_stmt)
        objects = objects_result.scalars().all()

        # Get NPCs
        npcs_stmt = select(NPC).where(NPC.location_id == location_id)
        npcs_result = await self.session.execute(npcs_stmt)
        npcs = npcs_result.scalars().all()

        return {
            "location": location,
            "objects": objects,
            "npcs": npcs,
            "exits": location.connections or []
        }
```

## File Structure Changes

```
src/game_loop/
├── core/
│   ├── game_loop.py           # Enhanced with action processing integration
│   ├── input_processor.py     # Enhanced with state context integration
│   ├── nlp_processor.py       # Enhanced with database integration
│   ├── action_processor.py    # New - Central action processing pipeline
│   └── command_mapper.py      # Enhanced with database object resolution
├── database/repositories/
│   └── search.py              # New - Semantic search and query operations
└── integration/
    └── game_flow.py           # New - End-to-end integration utilities
```

## Integration Points

### 1. Input Processing Flow
```
User Input → InputProcessor → NLPProcessor → Database Queries → Resolved Actions
```

### 2. Action Processing Flow
```
Resolved Actions → ActionProcessor → State Updates → ActionResult → OutputGenerator
```

### 3. Database Integration Flow
```
NLP Processing → SearchRepository → World/Player Repositories → Context Data
```

### 4. Response Generation Flow
```
ActionResult → OutputGenerator → TemplateManager → Rich Text Output
```

## Implementation Steps

### Step 1: Core Integration Setup
1. Enhance InputProcessor with GameStateManager integration
2. Add database query capabilities to NLPProcessor
3. Create ActionProcessor class
4. Update GameLoop to use integrated components

### Step 2: Database Query Integration
1. Create SearchRepository for semantic operations
2. Enhance existing repositories with context methods
3. Add query optimization for location-based searches
4. Implement object/NPC resolution logic

### Step 3: Action Processing Pipeline
1. Implement basic action handlers (movement, interaction, dialogue)
2. Add action validation and error handling
3. Create ActionResult generation logic
4. Integrate with state update mechanisms

### Step 4: Response Integration
1. Connect ActionProcessor output to OutputGenerator
2. Ensure proper template selection for different action types
3. Add location change handling
4. Implement error response formatting

### Step 5: End-to-End Testing
1. Test complete input-to-output flow
2. Verify state persistence across actions
3. Test error handling and edge cases
4. Validate response formatting consistency

## Testing Strategy

### Unit Tests (`tests/unit/integration/`)

1. **test_action_processor.py**
   - Test individual action handlers
   - Test action validation and error handling
   - Test ActionResult generation

2. **test_input_integration.py**
   - Test InputProcessor with game state context
   - Test NLP-database integration
   - Test object/location resolution

3. **test_game_flow.py**
   - Test complete turn processing
   - Test meta-command handling
   - Test state persistence

### Integration Tests (`tests/integration/core/`)

1. **test_end_to_end_flow.py**
   - Test complete input-to-output cycles
   - Test multi-turn game sessions
   - Test save/load functionality

2. **test_database_integration.py**
   - Test NLP-database query integration
   - Test semantic search functionality
   - Test context retrieval performance

## Success Criteria

- [ ] InputProcessor successfully integrates with GameStateManager for context-aware processing
- [ ] NLPProcessor can resolve objects and locations using database queries
- [ ] ActionProcessor handles basic game actions (movement, interaction, examination)
- [ ] Complete input-to-output flow works for common game scenarios
- [ ] State changes persist correctly across game turns
- [ ] OutputGenerator produces properly formatted responses for all action types
- [ ] Meta-commands (save, load, quit, look, inventory) work correctly
- [ ] Error handling provides useful feedback for invalid actions
- [ ] Database queries perform adequately for real-time gameplay
- [ ] All tests pass and provide adequate coverage
- [ ] Code follows project linting standards (black, ruff, mypy)

## Verification Procedures

### Manual Testing
1. **Basic Game Flow**
   - Start a new game and verify initial state display
   - Test movement between locations
   - Test object examination and interaction
   - Test inventory commands

2. **State Persistence**
   - Perform actions and verify state changes persist
   - Test save/load functionality
   - Verify game state consistency after reload

3. **Error Handling**
   - Test invalid commands and verify helpful error messages
   - Test actions on non-existent objects
   - Test movement to blocked/invalid directions

### Automated Testing
1. Run unit test suite: `poetry run pytest tests/unit/integration/`
2. Run integration tests: `poetry run pytest tests/integration/core/`
3. Run end-to-end tests: `poetry run pytest tests/integration/test_end_to_end_flow.py`

### Performance Testing
1. Test response times for common actions
2. Verify database query performance with realistic data
3. Test memory usage during extended gameplay sessions

## Dependencies to Add

No new external dependencies required - this commit integrates existing components.

## Notes

- This commit creates the first playable version of the game with persistent state
- Focus on basic functionality - advanced features will be added in later commits
- Ensure backward compatibility with existing components
- Performance optimization may be needed as the system grows
- Consider caching frequently accessed data (locations, objects) for better performance
- Error handling should be comprehensive but not overwhelming to users

## Future Enhancements (Post-Commit)

- Advanced action types (combat, crafting, dialogue trees)
- Sophisticated semantic search with vector embeddings
- Performance optimization and caching
- Real-time world events and evolution
- Advanced NLP understanding and context
- Plugin system for custom actions and content

This implementation plan establishes the foundation for a fully functional text adventure game with natural language processing, persistent state, and rich output formatting. Subsequent commits will build upon this foundation to add more sophisticated gameplay features.
