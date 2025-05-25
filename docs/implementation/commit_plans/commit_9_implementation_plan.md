Commit 9: Implementation and Integration Instructions

Goal: Implement and integrate the Game State Management system for tracking, updating, persisting, and loading game state.

Prerequisites:

    Existing EnhancedInputProcessor, NlpProcessor, ConfigManager.
    Database infrastructure set up (DbConnection providing an async connection pool).
    Project structure as per previous commits.

I. Setup and Model Definition

    Create State Directory:
        Create a new directory: src/game_loop/state/
        Add an empty __init__.py file: src/game_loop/state/__init__.py

    Define State Models (src/game_loop/state/models.py):
        Create the file src/game_loop/state/models.py.
        Implement the Pydantic models exactly as defined in the plan:
            InventoryItem
            PlayerKnowledge
            PlayerStats
            PlayerProgress
            PlayerState
            Location (Consider if the full objects and npcs dicts are needed here or just IDs/summaries for the top-level WorldState)
            WorldState
            GameSession
        Define ActionResult: Add a Pydantic model to represent the outcome of a player's action. This will be returned by action execution logic and consumed by GameStateManager.update_after_action.
    Python

    # src/game_loop/state/models.py
    from pydantic import BaseModel, Field
    from typing import Dict, List, Optional, Any, Union
    from datetime import datetime
    from uuid import UUID, uuid4

    # ... (Paste InventoryItem, PlayerKnowledge, PlayerStats, PlayerProgress, PlayerState here) ...
    # ... (Paste Location, WorldState, GameSession here) ...

    # Define ActionResult Model
    class ActionResult(BaseModel):
        """Result of executing a player action."""
        success: bool = True
        feedback_message: str = "" # Message to show the player (e.g., "You take the key.")

        # State Change Flags/Data
        location_change: bool = False
        new_location_id: Optional[UUID] = None

        inventory_changes: Optional[List[Dict[str, Any]]] = None # e.g., [{"action": "add", "item": InventoryItem}, {"action": "remove", "item_id": UUID}]

        knowledge_updates: Optional[List[PlayerKnowledge]] = None # New knowledge acquired

        stat_changes: Optional[Dict[str, Union[int, float]]] = None # e.g., {"health": -10}

        object_changes: Optional[List[Dict[str, Any]]] = None # Changes to world objects state
        npc_changes: Optional[List[Dict[str, Any]]] = None # Changes to NPC state
        location_state_changes: Optional[Dict[str, Any]] = None # Changes to current location's state flags

        triggers_evolution: bool = False
        evolution_trigger: Optional[str] = None
        evolution_data: Optional[Dict[str, Any]] = None
        priority: int = 5 # Default priority for evolution events
        timestamp: datetime = Field(default_factory=datetime.now)

        # Optionally add the originating command/intent
        command: Optional[str] = None
        processed_input: Optional[Any] = None # Reference to ProcessedInput if needed

II. Implement State Management Components

(Implement the classes based on the plan provided earlier, placing them in the src/game_loop/state/ directory. Ensure they handle database interactions correctly using the provided db_pool)

    Implement PlayerStateTracker (src/game_loop/state/player_state.py):
        Create the file.
        Implement the PlayerStateTracker class as per the plan.
        Use the db_pool passed during initialization to acquire connections (async with self.db_pool.acquire() as conn:) for database operations (create, load, update).
        Implement methods like create_player, load_state, get_state, update_from_action (parsing ActionResult), update_location, update_inventory, etc.
        Use PlayerState model for internal representation and database serialization (using Pydantic's .model_dump()/.model_validate()).

    Implement WorldStateTracker (src/game_loop/state/world_state.py):
        Create the file.
        Implement the WorldStateTracker class as per the plan.
        Use db_pool for database operations.
        Implement initialize_world_state, load_state, get_state, update_from_action (parsing ActionResult), get_location_description, get_location_objects/npcs, queue_evolution_event, process_evolution_queue.
        Manage world data (locations, objects, NPCs). Consider strategies for handling large worlds (e.g., loading only necessary location data into cache).
        Use WorldState and Location models.

    Implement SessionManager (src/game_loop/state/session_manager.py):
        Create the file.
        Implement the SessionManager class as per the plan.
        Use db_pool for saving/loading session metadata and potentially pointers to state data.
        Implement create_session, load_session (retrieving state data IDs/references), save_session (storing state data, returning save ID), list_saved_games, delete_saved_game.
        Handle serialization/deserialization of PlayerState and WorldState data (using Pydantic methods) for storage (e.g., as JSONB in PostgreSQL).
        Use the GameSession model.

    Implement GameStateManager (src/game_loop/state/manager.py):
        Create the file.
        Implement the GameStateManager facade class as per the plan.
        __init__: Takes config_manager, db_pool.
        initialize: Instantiates PlayerStateTracker, WorldStateTracker, SessionManager, passing them the db_pool. Calls their respective initialize methods.
        Implement methods coordinating the underlying trackers: create_new_game, load_game, save_game, update_after_action, get_current_state, get_location_description.

III. Database Schema

    Update Schema (00X_add_state_tables.sql):
        Create a new migration file in src/game_loop/database/migrations/.
        Define SQL CREATE TABLE statements for:
            game_sessions: Store GameSession metadata (session ID, player ID, world ID, save name, timestamps, play time, version).
            player_states: Store serialized PlayerState data (e.g., player ID (PK), session ID (FK), state_data (JSONB), created_at, updated_at).
            world_states: Store serialized WorldState data (e.g., world ID (PK), session ID (FK), state_data (JSONB), created_at, updated_at).
            Consider indexing relevant columns (e.g., session_id, player_id, save_name).
        Ensure the init_db.py script or your migration tool applies this new migration.

IV. Integration with Existing Code

    Main Application (src/game_loop/main.py):
        Import: from game_loop.state.manager import GameStateManager
        Instantiate & Initialize: Follow the instantiation example provided in the previous discussion (inside async def main(), after db_connection and config_manager are ready). Get the db_pool from db_connection.
        Handle New/Load: Implement logic (prompt user, check args) to decide whether to call game_state_manager.create_new_game(...) or game_state_manager.load_game(...).
        Inject Dependency: Pass the initialized game_state_manager instance to the GameLoop constructor.
        Shutdown: Call await game_state_manager.shutdown() before closing the DB connection.

    Game Loop (src/game_loop/core/game_loop.py):
        Modify __init__: Add game_state_manager: GameStateManager parameter and store it as self.game_state_manager. Remove any old, simple state variables.
        Modify run():
            Before the main while loop, get and display the initial location description: desc = await self.game_state_manager.get_location_description() followed by print(desc).
        Modify process_turn():
            Get Context: Fetch necessary context (e.g., location ID) from self.game_state_manager before calling the input processor.
            Python

            player_state = await self.game_state_manager.player_state.get_state() # Or a more specific query
            context = {"location_id": player_state.get("location_id")}

            Pass Context: Pass context to self.input_processor.process(user_input, context=context).
            Handle Meta-Commands: Check processed_input or user_input for commands like save, load (at start), quit, look, inventory. Call corresponding self.game_state_manager methods and return appropriate messages.
            Execute Action: Call your command execution logic (e.g., await self.execute_command(processed_input)). This logic must now return an ActionResult object or None.
            Update State: If action_result is not None, update the state: await self.game_state_manager.update_after_action(action_result).
            Generate Output: Get feedback from action_result.feedback_message and the current location description await self.game_state_manager.get_location_description() to form the output string.

    Input/NLP Processors (enhanced_input_processor.py, nlp_processor.py):
        Modify the primary processing methods (e.g., EnhancedInputProcessor.process) to accept context: Optional[dict] = None.
        Use the context dictionary within the processing logic or LLM prompts to provide situational awareness (e.g., passing current location ID).

    Action Execution Logic (Command Handlers):
        Refactor: Locate the code that currently handles specific game commands (e.g., move, take, use, talk). This might be in GameLoop, separate functions, or classes called by CommandMapper.
        Query State: Modify these handlers to query the GameStateManager (passed in or accessed via GameLoop) for necessary state information before acting (e.g., await game_state_manager.world_state.get_location_objects(...)).
        Return ActionResult: Change the return type of these handlers. Instead of printing output or directly modifying state, they must construct and return an instance of the ActionResult model, populating it with all the consequences of the action (feedback message, state changes, etc.). If an action is invalid or has no effect, return None or an ActionResult with success=False and a relevant feedback message.

V. Testing

    Implement Unit Tests: Create tests in tests/unit/state/ for:
        test_state_models.py: Test model validation, defaults, serialization.
        test_player_state_tracker.py: Mock DB interactions, test state updates.
        test_world_state_tracker.py: Mock DB, test world updates, evolution queue.
        test_session_manager.py: Mock DB, test save/load session logic.
        test_game_state_manager.py: Test coordination between managers.
    Implement Integration Tests: Create tests in tests/integration/state/:
        test_state_persistence.py: Test full save -> load cycle with database interaction. Verify data integrity.
        test_state_updates.py: Simulate actions, verify state is updated correctly in memory and DB.
        Update test_game_loop_integration.py: Test the GameLoop with the integrated state manager, including save/load commands.

VI. Documentation

    Update README.md: Add sections explaining the state management system, how saving/loading works for the user.
    Code Documentation: Add comprehensive docstrings to all new classes and public methods in the src/game_loop/state/ directory.
    Database Schema: Document the new state-related tables in docs/database/schema.md.

This detailed guide should provide a clear path for implementing and integrating the game state management system. Remember to implement incrementally and test frequently.
