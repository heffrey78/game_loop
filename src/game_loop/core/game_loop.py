"""
Core game loop implementation for Game Loop.
Handles the main game loop, input processing, and output generation.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg
from rich.console import Console

from game_loop.config.manager import ConfigManager
from game_loop.config.models import GameConfig
from game_loop.core.actions.action_classifier import ActionTypeClassifier
from game_loop.core.command_handlers.factory import CommandHandlerFactory
from game_loop.core.enhanced_input_processor import EnhancedInputProcessor
from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand
from game_loop.core.location import LocationDisplay
from game_loop.core.navigation.pathfinder import PathfindingService
from game_loop.core.navigation.validator import NavigationValidator
from game_loop.core.rules.game_rules_manager import GameRulesManager
from game_loop.core.rules.rule_triggers import RuleTriggerManager
from game_loop.core.rules.rules_engine import RulesEngine
from game_loop.core.world.boundary_manager import WorldBoundaryManager
from game_loop.core.world.connection_graph import LocationConnectionGraph
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient
from game_loop.state.manager import GameStateManager
from game_loop.state.models import ActionResult, Location, PlayerState, WorldState

logger = logging.getLogger(__name__)


class GameLoop:
    """Main game loop implementation for the Game Loop text adventure."""

    def __init__(
        self, config: GameConfig, db_pool: asyncpg.Pool, console: Console | None = None
    ):
        """
        Initialize the game loop with configuration and database pool.

        Args:
            config: Configuration for the game
            db_pool: Database connection pool
            console: Rich console for output, creates new one if not provided
        """
        self.config = config
        self.db_pool = db_pool
        self.console = console if console else Console()
        self.location_display = LocationDisplay(self.console)
        self.running = False

        # Initialize config manager for LLM with correct prompt template directory
        from pathlib import Path

        from game_loop.config.manager import ConfigManager as GameConfigManager

        self.config_manager = ConfigManager()
        self.game_config_manager = GameConfigManager(config_file=None)
        self.game_config_manager.config = config

        # Update the prompt template directory to point to our implementation
        project_root = Path(__file__).parent.parent.parent
        prompt_dir = project_root / "game_loop" / "llm" / "prompts"
        self.config_manager.config.prompts.template_dir = str(prompt_dir)

        # Initialize GameStateManager with the correct config manager type
        self.state_manager = GameStateManager(self.game_config_manager, self.db_pool)

        # Initialize database session factory
        self.session_factory = DatabaseSessionFactory(config.database)

        # Initialize LLM client with longer timeout for dialogue generation
        self.ollama_client = OllamaClient(timeout=120.0)

        # Initialize action classifier
        self.action_classifier = ActionTypeClassifier(
            config_manager=self.config_manager,
            ollama_client=self.ollama_client,
        )

        # For now, initialize semantic search as None - will be set up properly later
        self.semantic_search = None

        # Create enhanced input processor with NLP capabilities
        self.input_processor = EnhancedInputProcessor(
            config_manager=self.config_manager,
            console=self.console,
            use_nlp=config.features.use_nlp,
            game_state_manager=self.state_manager,
        )

        # Fallback to basic input processor if needed
        self.basic_input_processor = InputProcessor(
            self.console,
            game_state_manager=self.state_manager,
        )

        # Initialize the command handler factory with all dependencies
        self.command_handler_factory = CommandHandlerFactory(
            console=self.console,
            state_manager=self.state_manager,
            session_factory=self.session_factory,
            config_manager=self.config_manager,
            llm_client=self.ollama_client,
            semantic_search=self.semantic_search,
            action_classifier=self.action_classifier,
        )

        # Initialize navigation components
        self.connection_graph = LocationConnectionGraph()
        self.boundary_manager = (
            None  # Will be initialized when world state is available
        )
        self.navigation_validator = NavigationValidator(self.connection_graph)
        self.pathfinding_service = (
            None  # Will be initialized when world state is available
        )

        # Initialize rules engine and management
        self.rules_engine = RulesEngine()
        self.rules_manager = GameRulesManager(self.rules_engine)
        self.rule_trigger_manager = RuleTriggerManager(self.rules_engine)

    async def initialize(self, session_id: UUID | None = None) -> None:
        """Initialize the game environment, loading or creating game state."""
        self.console.print("[bold green]Initializing Game Loop...[/bold green]")

        try:
            # Initialize database session factory
            await self.session_factory.initialize()

            # Load default game rules
            await self._load_default_rules()

            # Initialize the state manager, attempting to load if session_id provided
            await self.state_manager.initialize(session_id)

            player_state: PlayerState | None = None
            world_state: WorldState | None = None

            if session_id:
                # Attempt to load the game
                player_state, world_state = await self.state_manager.load_game(
                    session_id
                )  # noqa: E501
                if player_state and world_state:
                    self.console.print(f"Loaded game session {session_id}.")
                else:
                    self.console.print(
                        f"[bold red]Failed to load game session "
                        f"{session_id}. Starting new game.[/bold red]"
                    )
                    session_id = None

            if not session_id:
                # Create a new game if no session ID or loading failed
                player_name = self._get_player_name()
                # TODO: Get save name, maybe from player input or default
                save_name = f"{player_name}'s Game"
                player_state, world_state = await self.state_manager.create_new_game(
                    player_name, save_name
                )

                if not player_state or not world_state:
                    raise RuntimeError("Failed to create a new game.")

                self.console.print(
                    f"\n[bold green]Welcome to the adventure, "
                    f"{player_state.name}![/bold green]\n"
                )
            else:
                # Welcome back message for loaded game
                if player_state:
                    self.console.print(
                        f"\n[bold green]Welcome back, "
                        f"{player_state.name}![/bold green]\n"
                    )

        except Exception as e:
            logger.exception(f"Error during game initialization: {e}")
            self.console.print(
                f"[bold red]Error initializing game: {e}. " f"Exiting.[/bold red]"
            )
            self.running = False
            return

        # Initial display depends on successful load/create
        if self.state_manager.get_current_session_id():
            # Initialize navigation components with world state
            await self._initialize_navigation_system()
            await self._display_current_location()  # Use await
        else:
            self.console.print(
                "[bold red]Failed to initialize or load game state.[/bold red]"
            )  # noqa: E501
            self.running = False

    def stop(self) -> None:
        """Stop the game loop."""
        self.console.print("[bold]Farewell, adventurer! Your journey ends here.[/bold]")
        self.running = False

    async def _initialize_navigation_system(self) -> None:
        """Initialize the navigation system with current world state."""
        try:
            world_state = self.state_manager.world_tracker.get_state()
            if not world_state:
                logger.warning("No world state available for navigation initialization")
                return

            # Initialize boundary manager
            self.boundary_manager = WorldBoundaryManager(world_state)

            # Initialize pathfinding service
            self.pathfinding_service = PathfindingService(
                world_state, self.connection_graph
            )

            # Build the connection graph from world state
            await self._build_connection_graph(world_state)

            logger.info("Navigation system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing navigation system: {e}")

    async def _build_connection_graph(self, world_state: WorldState) -> None:
        """Build the connection graph from world state."""
        try:
            # Clear existing graph
            self.connection_graph.clear()

            # Add all locations to graph
            for location_id, location in world_state.locations.items():
                self.connection_graph.add_location(
                    location_id,
                    {
                        "name": location.name,
                        "type": location.state_flags.get("type", "generic"),
                    },
                )

            # Add all connections
            for location_id, location in world_state.locations.items():
                for direction, destination_id in location.connections.items():
                    # Check if destination exists in world state
                    if destination_id in world_state.locations:
                        self.connection_graph.add_connection(
                            location_id,
                            destination_id,
                            direction,
                            bidirectional=False,  # Add each direction explicitly
                        )

            logger.info(
                f"Built connection graph with {self.connection_graph.get_location_count()} locations and {self.connection_graph.get_connection_count()} connections"
            )

        except Exception as e:
            logger.error(f"Error building connection graph: {e}")

    def _get_player_name(self) -> str:
        """Get a name for the player character."""
        self.console.print("\n[bold]Enter your character name:[/bold]")
        player_name = input("> ").strip()
        return player_name if player_name else "Adventurer"

    async def run(self) -> None:
        """Run the main game loop."""
        self.running = True

        # Display the initial location
        await self._display_current_location()

        while self.running:
            try:
                await self._process_input_async()
            except Exception as e:
                logger.exception(f"Unexpected error in game loop: {e}")
                self.console.print(f"[bold red]An error occurred: {e}[/bold red]")

        # When the loop ends, consider saving the game state automatically
        # await self._auto_save_game()

    async def _display_current_location(self) -> None:
        """Display the current location to the player using rich output."""
        location = await self.state_manager.get_current_location_details()
        if location:
            from game_loop.core.location import Location as OldLocation

            old_location = OldLocation(
                id=str(location.location_id),
                name=location.name,
                description=location.description,
            )

            # Add connections
            for direction, loc_id in location.connections.items():
                old_location.add_connection(direction, str(loc_id))

            # Add objects from the location
            for _obj_id, obj in location.objects.items():
                if not obj.is_hidden:
                    old_location.add_item(obj.name.lower().replace(" ", "_"))

            # Add NPCs as placeholder objects in the old format
            for _npc_id, npc in location.npcs.items():
                old_location.add_npc(npc.name.lower().replace(" ", "_"))

            # Now display the converted location
            self.location_display.display_location(old_location)
        else:
            self.console.print(
                "[bold red]Error: Current location not found.[/bold red]"
            )

    def _extract_game_context(self) -> dict[str, Any]:
        """
        Extract relevant game state as context for NLP processing.

        Returns:
            Dictionary containing game state context
        """
        context = {}
        player_state = self.state_manager.player_tracker.get_state()
        current_location_id = None
        current_location = None

        if player_state and player_state.current_location_id:
            current_location_id = player_state.current_location_id
            world_state = self.state_manager.world_tracker.get_state()
            if world_state:
                current_location = world_state.locations.get(current_location_id)

        if current_location:
            context["current_location"] = {
                "id": str(current_location.location_id),
                "name": current_location.name,
                "description": current_location.description,
            }

            # Add connections
            connections = {}
            for direction, dest_id in current_location.connections.items():
                if world_state:
                    dest_location = world_state.locations.get(dest_id)
                    if dest_location:
                        connections[f"connection_{direction}"] = (
                            f"{direction} to {dest_location.name}"
                        )
            context["connections"] = connections

            # Add visible objects from the current location
            visible_objects = {}
            for i, (_obj_id, obj) in enumerate(current_location.objects.items()):
                if not obj.is_hidden:
                    visible_objects[f"object_{i}"] = obj.name
            context["visible_objects"] = visible_objects

            # Add NPCs from the current location
            npcs = {}
            for i, (_npc_id, npc) in enumerate(current_location.npcs.items()):
                npcs[f"npc_{i}"] = npc.name
            context["npcs"] = npcs

        # Add player information
        if player_state:
            # Create player info with name and ID
            player_info = {
                "name": player_state.name,
                "player_id": (
                    str(player_state.player_id) if player_state.player_id else None
                ),
            }

            # Add inventory
            if player_state.inventory:
                player_info["inventory"] = ", ".join(
                    item.name for item in player_state.inventory
                )
            else:
                player_info["inventory"] = "empty"

            # Add knowledge and stats if available
            if player_state.knowledge:
                player_info["knowledge"] = ", ".join(
                    [k.content for k in player_state.knowledge]
                )

            if player_state.stats:
                # Convert PlayerStats object to dictionary by extracting its attributes
                stats_dict = {
                    "health": player_state.stats.health,
                    "max_health": player_state.stats.max_health,
                    "mana": player_state.stats.mana,
                    "max_mana": player_state.stats.max_mana,
                    "strength": player_state.stats.strength,
                    "dexterity": player_state.stats.dexterity,
                    "intelligence": player_state.stats.intelligence,
                }
                player_info["stats"] = str(stats_dict)

            context["player"] = player_info
            # Also add player_id at the top level for system commands
            context["player_id"] = (
                str(player_state.player_id) if player_state.player_id else None
            )

        return context

    def _process_input(self) -> None:
        """Process player input and execute appropriate actions.

        This is a synchronous wrapper around _process_input_async to maintain
        backward compatibility with existing tests that don't use await.
        """
        try:
            import asyncio

            # Use get_event_loop().run_until_complete() to handle the coroutine
            # in a synchronous context
            asyncio.get_event_loop().run_until_complete(self._process_input_async())
        except Exception as e:
            self.console.print(f"[bold red]Error processing input: {e}[/bold red]")

    async def _process_input_async(self) -> None:
        """Process player input and execute appropriate actions asynchronously."""
        self.console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
        user_input = input("> ").strip()

        if not user_input:
            self.console.print("[yellow]Please enter a command.[/yellow]")
            return

        # Extract game context for NLP processing
        game_context = self._extract_game_context()

        # First check if this is a system command
        system_result = await self.command_handler_factory.handle_command(
            user_input, game_context
        )

        if system_result:
            # This was a system command, display the result
            if system_result.feedback_message:
                self.console.print(system_result.feedback_message)

            # Check if this was a quit command and stop the game loop
            if user_input.lower().strip() in ["quit", "exit", "/quit", "/exit"]:
                self.stop()

            return

        # Not a system command, process normally
        command = None

        try:
            # Process the input through the enhanced input processor
            command = await self.input_processor.process_input_async(
                user_input, game_context
            )

            # Update conversation context with this exchange
            response = "Command processed successfully."
            await self.input_processor.update_conversation_context(user_input, response)

        except Exception:
            self.console.print("[yellow]Using simplified input processing...[/yellow]")
            try:
                basic_processor = InputProcessor(self.console)
                command = await basic_processor.process_input_async(user_input)
            except Exception as inner_e:
                self.console.print(
                    f"[bold red]Input processing error: " f"{inner_e}[/bold red]"
                )
                return

        if command:
            await self._execute_command(command)

    async def _execute_command(self, command: ParsedCommand) -> ActionResult | None:
        """Execute a processed command and return ActionResult."""
        player_state = self.state_manager.player_tracker.get_state()
        await self.state_manager.player_tracker.get_current_location_id()
        current_location = await self.state_manager.get_current_location_details()
        world_state = self.state_manager.world_tracker.get_state()

        # Verify required state exists before proceeding
        if not player_state:
            return ActionResult(
                success=False, feedback_message="Error: Player state not initialized."
            )

        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        if not world_state:
            return ActionResult(
                success=False, feedback_message="Error: World state not initialized."
            )

        # Evaluate rules before command execution
        pre_execution_result = await self._evaluate_rules_pre_command(
            command, player_state, current_location, world_state
        )
        if pre_execution_result and not pre_execution_result.success:
            return pre_execution_result

        action_result = None
        try:
            if command.command_type == CommandType.MOVEMENT:
                if command.subject:
                    action_result = await self._handle_movement(
                        command.subject, player_state, current_location
                    )
                else:
                    action_result = ActionResult(
                        success=False, feedback_message="No direction specified."
                    )

            elif command.command_type == CommandType.LOOK:
                await self._display_current_location()
                # No state change, just visual output
                action_result = ActionResult(success=True, feedback_message="")

            elif command.command_type == CommandType.INVENTORY:
                await self._display_inventory(player_state)
                # No state change, just visual output
                action_result = ActionResult(success=True, feedback_message="")

            elif command.command_type == CommandType.TAKE:
                if command.subject:
                    action_result = await self._handle_take(
                        command.subject, player_state, current_location
                    )
                else:
                    action_result = ActionResult(
                        success=False, feedback_message="What do you want to take?"
                    )

            elif command.command_type == CommandType.DROP:
                if command.subject:
                    action_result = await self._handle_drop(
                        command.subject, player_state, current_location
                    )
                else:
                    action_result = ActionResult(
                        success=False, feedback_message="What do you want to drop?"
                    )

            elif command.command_type == CommandType.USE:
                # Use the command handler strategy pattern
                handler = self.command_handler_factory.get_handler(CommandType.USE)
                action_result = await handler.handle(command)

            elif command.command_type == CommandType.EXAMINE:
                if command.subject:
                    action_result = await self._handle_examine(
                        command.subject, player_state, current_location, world_state
                    )
                else:
                    action_result = ActionResult(
                        success=False, feedback_message="What do you want to examine?"
                    )

            elif command.command_type == CommandType.TALK:
                if command.subject:
                    action_result = await self._handle_talk(
                        command.subject, player_state, current_location
                    )
                else:
                    action_result = ActionResult(
                        success=False, feedback_message="Who do you want to talk to?"
                    )

            elif command.command_type == CommandType.HELP:
                self._display_help()
                # No state change, just visual output
                action_result = ActionResult(success=True, feedback_message="")

            elif command.command_type == CommandType.QUIT:
                # Possibly update state before shutting down
                action_result = ActionResult(
                    success=True,
                    feedback_message="Farewell, adventurer! Your journey ends here.",
                )
                self.stop()

            else:
                # Unknown command
                error_message = self.input_processor.format_error_message(command)
                action_result = ActionResult(
                    success=False, feedback_message=error_message
                )

        except Exception as e:
            logger.exception(f"Error executing command: {e}")
            action_result = ActionResult(
                success=False, feedback_message=f"Error: {str(e)}"
            )

        # If we have a result with a feedback message, display it
        if action_result and action_result.feedback_message:
            if action_result.success:
                self.console.print(f"[green]{action_result.feedback_message}[/green]")
            else:
                self.console.print(f"[yellow]{action_result.feedback_message}[/yellow]")

        # If the action was successful, update the game state
        if (
            action_result
            and action_result.success
            and (
                action_result.location_change
                or action_result.inventory_changes
                or action_result.object_changes
                or action_result.npc_changes
                or action_result.location_state_changes
                or action_result.knowledge_updates
                or action_result.stat_changes
                or action_result.progress_updates
                or action_result.triggers_evolution
            )
        ):
            await self.state_manager.update_after_action(action_result)

            # Evaluate rules after command execution and state changes
            await self._evaluate_rules_post_command(
                command, action_result, player_state, current_location, world_state
            )

            # If the location changed, display the new location
            if action_result.location_change:
                self.console.print("")  # Add a line break
                await self._display_current_location()

        return action_result

    async def _handle_movement(
        self,
        direction: str,
        player_state: PlayerState,
        current_location: Location | None,
    ) -> ActionResult:
        """Handle movement in a given direction, returning ActionResult."""
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Normalize direction (e.g., "n" -> "north")
        direction_map = {"n": "north", "s": "south", "e": "east", "w": "west"}
        normalized_direction = direction_map.get(direction.lower(), direction.lower())

        # Check if movement is possible in the given direction
        destination_id = current_location.connections.get(normalized_direction)
        if not destination_id:
            # Try dynamic world generation for the missing direction
            new_destination = await self._attempt_dynamic_expansion(
                current_location, normalized_direction, player_state
            )

            if new_destination:
                destination_id = new_destination
                self.console.print(
                    f"[dim cyan]As you move {normalized_direction}, the path ahead materializes...[/dim cyan]"
                )
            else:
                return ActionResult(
                    success=False,
                    feedback_message=f"You cannot go {normalized_direction} from here.",
                )
        elif str(destination_id) == "00000000-0000-0000-0000-000000000000":
            # This is a placeholder connection - generate the actual location
            new_destination = await self._attempt_dynamic_expansion(
                current_location, normalized_direction, player_state
            )

            if new_destination:
                destination_id = new_destination
                self.console.print(
                    f"[dim cyan]As you explore {normalized_direction}, a new area unfolds before you...[/dim cyan]"
                )
            else:
                return ActionResult(
                    success=False,
                    feedback_message=f"The path {normalized_direction} seems to lead nowhere.",
                )

        # Use navigation validator if available
        if self.navigation_validator:
            try:
                validation_result = await self.navigation_validator.validate_movement(
                    player_state, current_location, destination_id, normalized_direction
                )

                if not validation_result.success:
                    return ActionResult(
                        success=False, feedback_message=validation_result.message
                    )
            except Exception as e:
                logger.warning(f"Navigation validation failed: {e}")
                # Continue with basic movement if validation fails

        # Create ActionResult with location change
        return ActionResult(
            success=True,
            feedback_message=f"You go {normalized_direction}.",
            location_change=True,
            new_location_id=destination_id,
        )

    async def _display_inventory(self, player_state: PlayerState) -> None:
        """Display the player's inventory."""
        if not player_state:
            self.console.print(
                "[bold red]Error: Player state not initialized.[/bold red]"
            )
            return

        inventory = player_state.inventory

        self.console.print("[bold]Inventory:[/bold]")
        if not inventory:
            self.console.print("Your inventory is empty.")
            return

        for item in inventory:
            self.console.print(f"- {item.name} ({item.quantity})")
            if item.description:
                self.console.print(f"  {item.description}")

    async def _handle_take(
        self,
        item_name: str,
        player_state: PlayerState,
        current_location: Location | None,
    ) -> ActionResult:
        """Handle taking an item from the current location, returning ActionResult."""
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Normalize item name
        normalized_item_name = item_name.replace("the ", "").strip().lower()

        # Find the item in the current location
        item_to_take = None
        item_id = None
        for obj_id, obj in current_location.objects.items():
            if obj.name.lower() == normalized_item_name and obj.is_takeable:
                item_to_take = obj
                item_id = obj_id
                break

        if not item_to_take:
            return ActionResult(
                success=False,
                feedback_message=f"You don't see any takeable {item_name} here.",
            )

        # Create an inventory item from the world object
        from game_loop.state.models import InventoryItem

        inventory_item = InventoryItem(
            name=item_to_take.name,
            description=item_to_take.description,
            attributes=item_to_take.state.get("attributes", {}),
        )

        # Define changes to be applied
        inventory_add = {"action": "add", "item": inventory_item.model_dump()}
        object_remove = {
            "action": "remove",
            "object_id": item_id,
            "location_id": current_location.location_id,
        }

        return ActionResult(
            success=True,
            feedback_message=f"You take the {item_to_take.name}.",
            inventory_changes=[inventory_add],
            object_changes=[object_remove],
        )

    async def _handle_drop(
        self,
        item_name: str,
        player_state: PlayerState,
        current_location: Location | None,
    ) -> ActionResult:
        """Handle dropping an item, returning ActionResult."""
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        normalized_item_name = item_name.replace("the ", "").strip().lower()

        # Find item in player's inventory
        item_to_drop = None
        for item in player_state.inventory:
            if item.name.lower() == normalized_item_name:
                item_to_drop = item
                break

        if not item_to_drop:
            return ActionResult(
                success=False, feedback_message=f"You don't have a {item_name}."
            )

        # Create a WorldObject representation of the dropped item
        from game_loop.state.models import WorldObject

        dropped_object = WorldObject(
            name=item_to_drop.name,
            description=item_to_drop.description,
            is_takeable=True,
            state={"attributes": item_to_drop.attributes},
        )

        # Define changes for ActionResult
        inventory_remove = {"action": "remove", "item_id": item_to_drop.item_id}
        object_add = {
            "action": "add",
            "location_id": current_location.location_id,
            "object": dropped_object.model_dump(),
        }

        return ActionResult(
            success=True,
            feedback_message=f"You drop the {item_to_drop.name}.",
            inventory_changes=[inventory_remove],
            object_changes=[object_add],
        )

    async def _handle_examine(
        self,
        object_name: str,
        player_state: PlayerState,
        current_location: Location | None,
        world_state: WorldState,
    ) -> ActionResult:
        """Handle examining an object, returning ActionResult."""
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        normalized_object_name = object_name.replace("the ", "").strip().lower()

        # Handle "examine here" or "look"
        if normalized_object_name == "here" or normalized_object_name == "around":
            await self._display_current_location()
            return ActionResult(success=True, feedback_message="")

        # Handle "look in/inside X"
        looking_inside = False
        container_name = normalized_object_name
        if normalized_object_name.startswith(
            "in "
        ) or normalized_object_name.startswith("inside "):
            prefix = "in " if normalized_object_name.startswith("in ") else "inside "
            container_name = normalized_object_name[len(prefix) :].strip()
            looking_inside = True

        target_object = None
        target_location_type = None

        # 1. Check player inventory
        for item in player_state.inventory:
            if item.name.lower() == container_name:
                target_object = item
                target_location_type = "inventory"
                break

        # 2. Check location objects if not found
        if not target_object:
            for _obj_id, obj in current_location.objects.items():
                if obj.name.lower() == container_name:
                    target_object = obj
                    target_location_type = "location_object"
                    break

        # 3. Check location NPCs if not found
        if not target_object:
            for _npc_id, npc in current_location.npcs.items():
                if npc.name.lower() == container_name:
                    target_object = npc
                    target_location_type = "location_npc"
                    break

        if not target_object:
            return ActionResult(
                success=False,
                feedback_message=f"You don't see any '{object_name}' here.",
            )

        # --- Handle Looking Inside ---
        if looking_inside:
            is_container = False
            contents = []
            if target_location_type == "inventory":
                is_container = target_object.attributes.get("is_container", False)
                contents = target_object.attributes.get("contained_items", [])
            elif target_location_type == "location_object":
                is_container = target_object.is_container
                contents = target_object.contained_items

            if not is_container:
                return ActionResult(
                    success=False,
                    feedback_message=f"The {target_object.name} is not a container.",
                )

            if not contents:
                return ActionResult(
                    success=True, feedback_message=f"The {target_object.name} is empty."
                )
            else:
                # For simplicity, just use UUIDs
                contents_str = ", ".join(
                    f"item {str(uuid)[:8]}..." for uuid in contents
                )
                return ActionResult(
                    success=True,
                    feedback_message=f"Inside the {target_object.name}"
                    f", you find: {contents_str}",
                )

        # --- Handle Regular Examination ---
        else:
            description = target_object.description

            # Add more details for objects or NPCs with state
            if hasattr(target_object, "state") and target_object.state:
                state_desc = ", ".join(
                    f"{k}: {v}"
                    for k, v in target_object.state.items()
                    if k != "attributes"
                )
                if state_desc:
                    description += f" ({state_desc})"

            return ActionResult(success=True, feedback_message=description)

    async def _handle_talk(
        self,
        character_name: str,
        player_state: PlayerState,
        current_location: Location | None,
    ) -> ActionResult:
        """Handle talking to an NPC, with LLM-powered dialogue generation."""
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        normalized_name = (
            character_name.replace("to ", "").replace("with ", "").strip().lower()
        )

        # Check if the NPC exists in the current location
        target_npc = None
        npc_id = None
        for n_id, npc in current_location.npcs.items():
            if npc.name.lower() == normalized_name:
                target_npc = npc
                npc_id = n_id
                break

        if not target_npc:
            return ActionResult(
                success=False,
                feedback_message=f"You don't see anyone called "
                f"'{character_name}' here.",
            )

        # Try to generate LLM-powered dialogue if use_nlp is enabled
        if self.config.features.use_nlp:
            try:
                dialogue_response = await self._generate_npc_dialogue_response(
                    target_npc, current_location, player_state
                )
                if dialogue_response:
                    return ActionResult(
                        success=True, feedback_message=dialogue_response
                    )
            except Exception as e:
                logger.warning(
                    f"LLM dialogue generation failed: {str(e)}, falling back to basic responses",
                    exc_info=True,
                )

        # Fallback to simple response based on NPC state
        dialogue_state = target_npc.dialogue_state
        if dialogue_state == "friendly":
            response = (
                f"{target_npc.name} smiles at you warmly. "
                f"'Hello there! How can I help you?'"
            )
        elif dialogue_state == "hostile":
            response = (
                f"{target_npc.name} glares at you suspiciously. 'What do you want?'"
            )
        else:  # neutral or default
            response = f"{target_npc.name} nods politely. 'Greetings, traveler.'"

        # In a full implementation, we'd use NLP to generate dialogue based on:
        # - NPC's knowledge
        # - Player's previous interactions
        # - Current world state
        # - NPC's personality

        # We might also add knowledge gained from the conversation:
        knowledge_gained = None
        from game_loop.state.models import PlayerKnowledge

        if hasattr(target_npc, "knowledge") and target_npc.knowledge:
            if not target_npc.state.get("hostile", False):
                # Give the player a random piece of knowledge the NPC has (simplified)
                import random

                if random.random() < 0.3:  # 30% chance
                    knowledge_item = random.choice(target_npc.knowledge)
                    # Create a new PlayerKnowledge object
                    new_knowledge = PlayerKnowledge(
                        content=knowledge_item.content,
                        source=f"conversation with {target_npc.name}",
                        topic=(
                            knowledge_item.topic
                            if hasattr(knowledge_item, "topic")
                            else "general"
                        ),
                    )
                    knowledge_gained = [new_knowledge]
                    response += f" '{knowledge_item.content}'"

        # Return ActionResult with conversation outcome
        return ActionResult(
            success=True,
            feedback_message=response,
            knowledge_updates=knowledge_gained,
            npc_changes=(
                [{"npc_id": npc_id, "update": {"state.talked_to": True}}]
                if npc_id
                else None
            ),
        )

    def _display_help(self) -> None:
        """Display available commands."""
        self.console.print("\n[bold]Game Commands:[/bold]")
        self.console.print(
            "- [bold]go/move[/bold] <direction>: "
            "Move in a direction (north, south, east, west, etc.)"
        )
        self.console.print("- [bold]look[/bold]: Look around the current location")
        self.console.print("- [bold]examine[/bold] <object>: Look closely at an object")
        self.console.print("- [bold]inventory/i[/bold]: Check your inventory")
        self.console.print("- [bold]take/get[/bold] <item>: Pick up an item")
        self.console.print(
            "- [bold]drop[/bold] <item>: Drop an item from your inventory"
        )
        self.console.print(
            "- [bold]use[/bold] <item> [on <target>]: "
            "Use an item, optionally on a target"
        )
        self.console.print("- [bold]talk[/bold] to <character>: Talk to a character")

        self.console.print("\n[bold]System Commands:[/bold]")
        self.console.print(
            "- [bold]save[/bold] [name]: Save your game (optionally with a name)"
        )
        self.console.print("- [bold]load[/bold] [name]: Load a saved game")
        self.console.print("- [bold]list saves[/bold]: Show all your saved games")
        self.console.print("- [bold]settings[/bold]: View or modify game settings")
        self.console.print(
            "- [bold]help[/bold] [topic]: Show help (optionally for a specific topic)"
        )
        self.console.print("- [bold]tutorial[/bold]: Get tutorial guidance")
        self.console.print("- [bold]quit/exit[/bold]: Quit the game")

    async def _auto_save_game(self) -> None:
        """Auto-save the current game state."""
        if self.state_manager.get_current_session_id():
            try:
                await self.state_manager.save_game()
                self.console.print("[dim]Game auto-saved.[/dim]")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
                # Don't show error to user as this is automatic

    async def _attempt_dynamic_expansion(
        self, current_location: Location, direction: str, player_state: PlayerState
    ) -> UUID | None:
        """
        Attempt to dynamically generate a new location in the specified direction.

        Args:
            current_location: The current location
            direction: The direction the player wants to go
            player_state: Current player state

        Returns:
            UUID of the new location if generation succeeds, None otherwise
        """
        try:
            # Check if this location can expand (either predefined or dynamically generated)
            can_expand = await self._can_location_expand(current_location, direction)
            if not can_expand:
                return None

            # Determine the type of location to generate
            location_type = await self._determine_location_type(
                current_location, direction
            )
            if not location_type:
                return None

            # Generate a new location with multiple exits
            new_location_id = await self._generate_enhanced_location(
                current_location, direction, location_type
            )

            if new_location_id:
                # Add bidirectional connection
                await self._add_connection(
                    current_location.location_id, new_location_id, direction
                )

                # Generate additional exits for the new location
                await self._generate_additional_exits(
                    new_location_id, current_location, direction
                )

                # Rebuild connection graph for navigation
                world_state = self.state_manager.world_tracker.get_state()
                if world_state and self.connection_graph:
                    await self._build_connection_graph(world_state)

                return new_location_id

            return None

        except Exception as e:
            logger.error(f"Error attempting dynamic expansion: {e}")
            return None

    async def _can_location_expand(self, location: Location, direction: str) -> bool:
        """
        Check if a location can expand in the given direction.

        Args:
            location: The location to check
            direction: The direction of expansion

        Returns:
            True if expansion is possible, False otherwise
        """
        try:
            # Check for predefined expansion opportunities
            expansion_opportunities = {
                "Building Lobby": ["outside", "north", "south", "loading"],
                "Emergency Stairwell": ["up", "north", "south"],
                "Underground Parking Garage": ["down", "east", "maintenance"],
                "Office": ["east", "west", "north", "south"],
                "Conference Room": ["east", "west", "north", "south"],
                "Reception Area": ["east", "west", "north", "south"],
                "Break Room": ["east", "west", "north", "south"],
                "Storage Room": ["east", "west", "north", "south"],
            }

            # Check if this is a predefined boundary location
            location_name = location.name
            if location_name in expansion_opportunities:
                return direction in expansion_opportunities[location_name]

            # Check if this is a dynamically generated location that can expand
            # Look for expansion metadata in location state
            expansion_depth = location.state_flags.get("expansion_depth", 0)
            max_expansion_depth = 5  # Prevent infinite expansion

            if expansion_depth < max_expansion_depth:
                # Check if location has expandable characteristics
                location_type = location.state_flags.get("location_type", "unknown")

                # Dynamic locations can generally expand in logical directions
                if location.state_flags.get("is_dynamic", False):
                    return self._is_logical_direction(location_type, direction)

            return False

        except Exception as e:
            logger.error(f"Error checking if location can expand: {e}")
            return False

    def _is_logical_direction(self, location_type: str, direction: str) -> bool:
        """
        Check if a direction makes logical sense for a location type.

        Args:
            location_type: The type of location
            direction: The direction to check

        Returns:
            True if the direction is logical, False otherwise
        """
        logical_directions = {
            "urban_street": ["north", "south", "east", "west", "inside"],
            "industrial_district": ["north", "south", "east", "west", "inside", "down"],
            "basement_access": ["north", "south", "east", "west", "down"],
            "upper_floor": ["north", "south", "east", "west", "up", "down"],
            "mechanical_room": ["north", "south", "east", "west", "maintenance"],
            "sublevel": ["north", "south", "east", "west", "down", "maintenance"],
            "industrial_zone": ["north", "south", "east", "west", "loading"],
            "loading_dock": ["north", "south", "east", "west", "outside"],
            "utility_tunnels": ["north", "south", "east", "west", "maintenance", "up"],
            "roof_access": ["north", "south", "east", "west", "down"],
        }

        return direction in logical_directions.get(
            location_type, ["north", "south", "east", "west"]
        )

    async def _determine_location_type(
        self, source_location: Location, direction: str
    ) -> str | None:
        """
        Enhanced location type determination with smart expansion logic.
        Considers terrain consistency, directional awareness, and location hierarchies.

        Args:
            source_location: The location expanding from
            direction: The direction of expansion

        Returns:
            The type of location to generate, or None if no suitable type
        """
        try:
            # Predefined expansion mapping for original world locations
            expansion_mapping = {
                "Building Lobby": {
                    "outside": "urban_street",
                    "north": "industrial_district",
                    "south": "basement_access",
                    "loading": "loading_dock",
                },
                "Emergency Stairwell": {
                    "up": "upper_floor",
                    "north": "mechanical_room",
                    "south": "roof_access",
                },
                "Underground Parking Garage": {
                    "down": "sublevel",
                    "east": "industrial_zone",
                    "maintenance": "utility_tunnels",
                },
            }

            # Check for predefined mapping first
            location_name = source_location.name
            if location_name in expansion_mapping:
                return expansion_mapping[location_name].get(direction)

            # Get source location metadata
            source_type = source_location.state_flags.get("location_type", "unknown")
            source_depth = source_location.state_flags.get("expansion_depth", 0)
            terrain_type = await self._get_terrain_type(source_location)
            elevation = await self._get_elevation_level(source_location, direction)

            # Smart directional logic with terrain consistency
            location_type = await self._apply_smart_expansion_rules(
                source_type, direction, terrain_type, elevation, source_depth
            )

            # Validate location type hierarchy
            if not await self._validate_location_hierarchy(
                source_type, location_type, direction
            ):
                logger.warning(
                    f"Invalid location hierarchy: {source_type} -> {location_type} via {direction}"
                )
                return self._get_fallback_location_type(source_type, direction)

            return location_type

        except Exception as e:
            logger.error(f"Error determining location type: {e}")
            return None

    async def _get_terrain_type(self, location: Location) -> str:
        """
        Determine the terrain type of a location for consistency checking.

        Args:
            location: The location to analyze

        Returns:
            Terrain type (urban, industrial, underground, building, natural)
        """
        location_type = location.state_flags.get("location_type", "unknown")

        terrain_mapping = {
            "urban_street": "urban",
            "industrial_district": "industrial",
            "industrial_zone": "industrial",
            "factory_interior": "industrial",
            "basement_access": "underground",
            "basement_corridor": "underground",
            "sublevel": "underground",
            "utility_tunnels": "underground",
            "building_interior": "building",
            "office_space": "building",
            "upper_floor": "building",
            "mechanical_room": "building",
            "loading_dock": "industrial",
            "roof_access": "building",
        }

        return terrain_mapping.get(location_type, "unknown")

    async def _get_elevation_level(
        self, source_location: Location, direction: str
    ) -> int:
        """
        Calculate elevation level based on direction and source location.

        Args:
            source_location: The source location
            direction: Direction of movement

        Returns:
            Elevation level (negative for underground, 0 for ground, positive for above)
        """
        current_elevation = source_location.state_flags.get("elevation", 0)

        elevation_changes = {"up": 1, "down": -1, "stairs": 1, "elevator": 1}

        return current_elevation + elevation_changes.get(direction, 0)

    async def _apply_smart_expansion_rules(
        self,
        source_type: str,
        direction: str,
        terrain_type: str,
        elevation: int,
        depth: int,
    ) -> str:
        """
        Apply smart expansion rules considering all context factors.

        Args:
            source_type: Type of the source location
            direction: Direction of expansion
            terrain_type: Terrain type for consistency
            elevation: Elevation level
            depth: Expansion depth from origin

        Returns:
            Appropriate location type for generation
        """
        # Terrain-consistent transitions
        terrain_transitions = {
            "urban": {
                "horizontal": [
                    "urban_street",
                    "commercial_district",
                    "residential_area",
                ],
                "inside": ["building_interior", "office_building", "retail_space"],
                "down": ["basement_access", "subway_station", "underground_passage"],
            },
            "industrial": {
                "horizontal": [
                    "industrial_zone",
                    "factory_complex",
                    "warehouse_district",
                ],
                "inside": [
                    "factory_interior",
                    "manufacturing_floor",
                    "storage_facility",
                ],
                "down": ["industrial_sublevel", "utility_tunnels", "maintenance_areas"],
            },
            "underground": {
                "horizontal": [
                    "basement_corridor",
                    "tunnel_system",
                    "underground_passage",
                ],
                "down": ["sublevel", "deep_tunnels", "underground_complex"],
                "up": ["ground_access", "stairwell_exit", "elevator_shaft"],
            },
            "building": {
                "horizontal": ["office_space", "building_corridor", "conference_area"],
                "up": ["upper_floor", "executive_level", "roof_access"],
                "down": ["lower_floor", "basement_access", "service_level"],
            },
        }

        # Determine movement type
        movement_type = self._classify_direction(direction, elevation)

        # Get possible transitions for this terrain
        transitions = terrain_transitions.get(terrain_type, {})
        candidates = transitions.get(movement_type, [])

        if not candidates:
            # Fallback to basic type transitions
            return self._get_basic_transition(source_type, direction)

        # Apply depth-based variation
        if depth >= 3:
            # Deeper locations get more specialized/unique types
            specialized_types = {
                "urban": ["abandoned_district", "ruined_quarter", "ghost_town"],
                "industrial": [
                    "derelict_factory",
                    "toxic_wasteland",
                    "abandoned_complex",
                ],
                "underground": [
                    "forgotten_tunnels",
                    "ancient_catacombs",
                    "deep_chambers",
                ],
                "building": ["hidden_floors", "secret_chambers", "abandoned_wings"],
            }
            depth_candidates = specialized_types.get(terrain_type, candidates)
            candidates.extend(depth_candidates)

        # Select based on direction preference and randomization
        import random

        return random.choice(candidates)

    def _classify_direction(self, direction: str, elevation: int) -> str:
        """
        Classify direction into movement type for terrain transitions.

        Args:
            direction: Direction string
            elevation: Elevation change

        Returns:
            Movement type (horizontal, up, down, inside)
        """
        if direction in ["up", "stairs up", "climb"]:
            return "up"
        elif direction in ["down", "stairs down", "descend"]:
            return "down"
        elif direction in ["inside", "enter", "through"]:
            return "inside"
        else:
            return "horizontal"

    def _get_basic_transition(self, source_type: str, direction: str) -> str:
        """
        Fallback basic transitions when smart rules don't apply.

        Args:
            source_type: Source location type
            direction: Direction of movement

        Returns:
            Basic location type
        """
        basic_transitions = {
            "urban_street": {
                "north": "urban_street",
                "south": "urban_street",
                "east": "urban_street",
                "west": "urban_street",
                "inside": "building_interior",
            },
            "industrial_district": {
                "north": "industrial_zone",
                "south": "industrial_zone",
                "east": "industrial_zone",
                "west": "industrial_zone",
                "inside": "factory_interior",
                "down": "industrial_sublevel",
            },
            "basement_access": {
                "north": "basement_corridor",
                "south": "basement_corridor",
                "east": "basement_corridor",
                "west": "basement_corridor",
                "down": "sublevel",
            },
            "upper_floor": {
                "north": "office_space",
                "south": "office_space",
                "east": "office_space",
                "west": "office_space",
                "up": "executive_floor",
                "down": "lower_floor",
            },
        }

        transitions = basic_transitions.get(source_type, {})
        return transitions.get(direction, "unknown_area")

    async def _validate_location_hierarchy(
        self, source_type: str, target_type: str, direction: str
    ) -> bool:
        """
        Validate that the location type transition makes logical sense.

        Args:
            source_type: Source location type
            target_type: Proposed target location type
            direction: Direction of movement

        Returns:
            True if the transition is valid
        """
        # Define invalid transitions
        invalid_transitions = {
            # Can't go from underground directly to sky/roof areas
            ("basement_corridor", "roof_access"),
            ("sublevel", "upper_floor"),
            ("utility_tunnels", "office_space"),
            # Can't go from outdoor to indoor without "inside" direction
            ("urban_street", "office_space"),
            ("industrial_zone", "basement_corridor"),
        }

        # Check for explicitly invalid transitions
        if (source_type, target_type) in invalid_transitions:
            return False

        # Direction-specific validations
        if (
            direction == "up"
            and "basement" in target_type
            and "sublevel" not in source_type
        ):
            return False

        if direction == "down" and "upper" in target_type:
            return False

        return True

    def _get_fallback_location_type(self, source_type: str, direction: str) -> str:
        """
        Get a safe fallback location type when validation fails.

        Args:
            source_type: Source location type
            direction: Direction of movement

        Returns:
            Safe fallback location type
        """
        # Simple, safe fallbacks
        safe_fallbacks = {
            "urban_street": "urban_street",
            "industrial_zone": "industrial_zone",
            "basement_corridor": "basement_corridor",
            "office_space": "office_space",
        }

        return safe_fallbacks.get(source_type, "unknown_area")

    async def _generate_enhanced_location(
        self, source_location: Location, direction: str, location_type: str
    ) -> UUID | None:
        """
        Generate an enhanced location with expansion metadata.

        Args:
            source_location: The source location
            direction: Direction of expansion
            location_type: Type of location to generate

        Returns:
            UUID of the new location if successful, None otherwise
        """
        try:
            from uuid import uuid4

            from game_loop.state.models import Location as LocationModel

            # Enhanced location templates with multiple exit hints and new terrain-aware types
            location_templates = {
                # Original types
                "urban_street": {
                    "name": "Abandoned Street",
                    "description": "A desolate urban street stretches before you, lined with empty storefronts and broken streetlights. Debris litters the asphalt, and the silence is unsettling. Streets continue in multiple directions, and you can see various building entrances along the sidewalks.",
                },
                "industrial_district": {
                    "name": "Industrial District",
                    "description": "You emerge into a sprawling industrial complex filled with massive machinery and conveyor systems. Steam hisses from pipes overhead, and the air thrums with the sound of distant generators. Multiple pathways lead to different sections of the industrial complex.",
                },
                "basement_access": {
                    "name": "Basement Corridor",
                    "description": "A narrow concrete corridor extends in multiple directions, lit by flickering fluorescent strips. You can hear the hum of electrical systems and the drip of water from unseen leaks. Several passages branch off, leading deeper into the building's underground network.",
                },
                "upper_floor": {
                    "name": "Upper Office Floor",
                    "description": "This higher floor contains executive offices and conference rooms, all abandoned but better preserved than the lower levels. Large windows show the city beyond, while multiple corridors lead to different wings of the floor.",
                },
                "mechanical_room": {
                    "name": "Mechanical Room",
                    "description": "A room filled with building systems: massive HVAC units, electrical panels, and cooling systems. The mechanical sounds are louder here, and you can see maintenance walkways leading to other parts of the building's infrastructure.",
                },
                "sublevel": {
                    "name": "Sub-Basement",
                    "description": "This lower level contains building utilities and storage areas. The ceiling is lower here, supported by concrete pillars. Multiple tunnels branch off in different directions, possibly connecting to other buildings or the city's utility network.",
                },
                "industrial_zone": {
                    "name": "Industrial Processing Area",
                    "description": "A large space filled with industrial equipment and processing machinery. Conveyor belts and assembly stations stretch across the floor. Various exits lead to other processing areas and what appears to be different industrial sectors.",
                },
                "loading_dock": {
                    "name": "Loading Dock",
                    "description": "A concrete platform where trucks once loaded and unloaded cargo. Large rolling doors face outward toward what was once a busy industrial area. Multiple truck bays and pathways lead to other logistics areas.",
                },
                "utility_tunnels": {
                    "name": "Utility Tunnels",
                    "description": "Underground maintenance tunnels that run beneath the building complex. Pipes and electrical conduits line the walls, and you can hear the echo of your footsteps. The tunnels branch in multiple directions, likely connecting to other buildings.",
                },
                "roof_access": {
                    "name": "Rooftop Access",
                    "description": "You emerge onto a rooftop area with HVAC equipment and communication arrays. The view shows the surrounding industrial district and urban landscape. Multiple maintenance walkways connect to other building sections.",
                },
                "building_interior": {
                    "name": "Building Interior",
                    "description": "The interior of an abandoned building with multiple rooms and corridors. Dust motes dance in shafts of light from broken windows. You can see doorways leading to different sections of the building.",
                },
                "factory_interior": {
                    "name": "Factory Floor",
                    "description": "A vast factory floor with abandoned assembly lines and industrial equipment. The space echoes with your footsteps, and you can see multiple work areas and passages leading to different parts of the facility.",
                },
                "basement_corridor": {
                    "name": "Basement Corridor",
                    "description": "A long underground corridor with concrete walls and exposed pipes overhead. Emergency lighting casts eerie shadows, and you can see multiple branching passages leading to different areas of the basement complex.",
                },
                "office_space": {
                    "name": "Office Complex",
                    "description": "An abandoned office space with cubicles and workstations covered in dust. Papers are scattered on desks, and you can see hallways leading to different departments and office areas.",
                },
                # New terrain-aware types - Urban
                "commercial_district": {
                    "name": "Commercial District",
                    "description": "A once-bustling commercial area now stands empty. Shop windows are boarded up or shattered, and faded signs hang askew. Multiple streets branch off from this central area, leading to different commercial sectors and residential zones.",
                },
                "residential_area": {
                    "name": "Residential Area",
                    "description": "Rows of abandoned houses line quiet streets. Overgrown lawns and broken fences speak of a neighborhood long evacuated. Side streets and alleyways provide multiple paths through the residential district.",
                },
                "office_building": {
                    "name": "Office Building",
                    "description": "A multi-story office building with empty lobbies and darkened windows. The building directory shows multiple floors and departments. Stairwells and elevators provide access to different levels, while exits lead to surrounding areas.",
                },
                "retail_space": {
                    "name": "Retail Space",
                    "description": "A large retail space that once housed multiple shops and services. Empty storefronts line the corridors, and shopping carts sit abandoned. Multiple entrances and corridors branch off to different retail sections.",
                },
                "subway_station": {
                    "name": "Subway Station",
                    "description": "An underground transit station with empty platforms and silent tracks. Fluorescent lights flicker overhead, and you can see tunnel entrances leading in multiple directions. Stairways lead up to street level.",
                },
                "underground_passage": {
                    "name": "Underground Passage",
                    "description": "A utilitarian underground passage connecting different areas of the city. Concrete walls are marked with maintenance symbols, and the passage branches into multiple tunnels leading to unknown destinations.",
                },
                # New terrain-aware types - Industrial
                "factory_complex": {
                    "name": "Factory Complex",
                    "description": "A sprawling industrial facility with multiple manufacturing buildings. Smokestacks rise into the sky, and you can see conveyor systems connecting different structures. Paths lead to various production areas and loading zones.",
                },
                "warehouse_district": {
                    "name": "Warehouse District",
                    "description": "Massive storage warehouses stretch in all directions. Loading bays and truck docks create a maze of industrial infrastructure. Multiple access roads and service corridors provide routes through the district.",
                },
                "manufacturing_floor": {
                    "name": "Manufacturing Floor",
                    "description": "A large production floor filled with specialized machinery and assembly stations. Overhead cranes and conveyor systems create a complex network. Multiple sections serve different aspects of the manufacturing process.",
                },
                "storage_facility": {
                    "name": "Storage Facility",
                    "description": "A vast storage area with towering shelves and industrial equipment. Forklifts sit abandoned between aisles of stored goods. Multiple warehouse sections and loading areas connect to this central storage hub.",
                },
                "industrial_sublevel": {
                    "name": "Industrial Sublevel",
                    "description": "Underground levels of the industrial complex house heavy machinery and power systems. The air is thick with the smell of oil and metal. Multiple mechanical passages lead to different subsystem areas.",
                },
                "maintenance_areas": {
                    "name": "Maintenance Areas",
                    "description": "Service areas dedicated to facility maintenance. Tool stations and equipment storage areas are scattered throughout. Maintenance tunnels and service corridors provide access to different building systems.",
                },
                # New terrain-aware types - Underground
                "tunnel_system": {
                    "name": "Tunnel System",
                    "description": "An extensive network of underground tunnels connecting various facilities. Emergency lighting provides minimal illumination, and you can hear the echo of dripping water. Multiple passages branch off into the darkness.",
                },
                "deep_tunnels": {
                    "name": "Deep Tunnels",
                    "description": "These tunnels extend deep underground, far below the surface structures. The walls are carved from rock, and the air is cool and still. Multiple passages lead even deeper into the underground network.",
                },
                "underground_complex": {
                    "name": "Underground Complex",
                    "description": "A sophisticated underground facility with multiple chambers and corridors. Emergency systems still function, providing basic lighting and ventilation. Various passages lead to different sections of the complex.",
                },
                "ground_access": {
                    "name": "Ground Access",
                    "description": "A transition area connecting underground facilities to surface levels. Heavy doors and security checkpoints mark the boundary between levels. Multiple exits lead both up to the surface and back down to underground areas.",
                },
                "stairwell_exit": {
                    "name": "Stairwell Exit",
                    "description": "A concrete stairwell providing access between underground and surface levels. Emergency lighting illuminates the steps, and you can see multiple landing areas leading to different floors and sections.",
                },
                "elevator_shaft": {
                    "name": "Elevator Shaft",
                    "description": "A service area around an elevator shaft with access to multiple building levels. Emergency ladders and maintenance platforms provide alternate routes. Doors lead to different floors and service areas.",
                },
                # New terrain-aware types - Building
                "building_corridor": {
                    "name": "Building Corridor",
                    "description": "A long hallway connecting various rooms and offices. Fluorescent lighting flickers overhead, and you can see doorways leading to different departments and service areas throughout the building.",
                },
                "conference_area": {
                    "name": "Conference Area",
                    "description": "A section of the building dedicated to meetings and presentations. Multiple conference rooms line the corridors, and you can see passages leading to different wings and administrative areas.",
                },
                "executive_level": {
                    "name": "Executive Level",
                    "description": "The upper floors reserved for executive offices and boardrooms. Floor-to-ceiling windows provide views of the surrounding area. Multiple suites and administrative areas branch off from the main corridor.",
                },
                "lower_floor": {
                    "name": "Lower Floor",
                    "description": "A ground-level floor with general offices and public areas. The ceilings are lower here, and you can see passages leading to different departments and building services.",
                },
                "service_level": {
                    "name": "Service Level",
                    "description": "Building service areas housing maintenance equipment and utilities. Access panels and service corridors provide routes to different building systems and mechanical areas.",
                },
                # Specialized deep-exploration types
                "abandoned_district": {
                    "name": "Abandoned District",
                    "description": "A completely deserted urban area where nature has begun to reclaim the streets. Vines grow through broken windows, and trees push through cracked pavement. Hidden paths wind through the urban decay.",
                },
                "ruined_quarter": {
                    "name": "Ruined Quarter",
                    "description": "Once-prosperous buildings now stand as empty shells, their upper floors collapsed and walls crumbling. Despite the destruction, multiple routes wind through the rubble-strewn passages.",
                },
                "ghost_town": {
                    "name": "Ghost Town",
                    "description": "An eerily preserved area where everything remains as it was suddenly abandoned. Cars sit in the middle of streets, and doors hang open. Multiple pathways lead through this time-frozen landscape.",
                },
                "derelict_factory": {
                    "name": "Derelict Factory",
                    "description": "A massive industrial facility in advanced decay. Rust has consumed most metal surfaces, and machinery sits frozen in time. Dangerous but navigable passages lead through the industrial ruins.",
                },
                "toxic_wasteland": {
                    "name": "Toxic Wasteland",
                    "description": "An area heavily contaminated by industrial processes. Warning signs in multiple languages dot the landscape, and strange growths emerge from contaminated soil. Hazardous but passable routes cross the wasteland.",
                },
                "abandoned_complex": {
                    "name": "Abandoned Complex",
                    "description": "A vast industrial complex that was suddenly evacuated. Equipment sits ready for operation, but dust and silence fill the air. Multiple facility sections remain accessible through various passages.",
                },
                "forgotten_tunnels": {
                    "name": "Forgotten Tunnels",
                    "description": "Ancient tunnels that predate the modern structures above. Strange markings cover the walls, and you sense these passages have been here far longer than anything else. Multiple branches lead to unknown destinations.",
                },
                "ancient_catacombs": {
                    "name": "Ancient Catacombs",
                    "description": "Stone passages carved long ago for purposes now forgotten. The architecture is different from modern construction, suggesting great age. Multiple chambers and passages form a maze-like network.",
                },
                "deep_chambers": {
                    "name": "Deep Chambers",
                    "description": "Vast underground spaces that seem too large and perfect to be natural. The purpose of these chambers is unclear, but they connect to multiple tunnel systems leading in all directions.",
                },
                "hidden_floors": {
                    "name": "Hidden Floors",
                    "description": "Secret levels of the building not shown on any directory. The architecture suggests these areas were designed for concealment. Multiple hidden passages connect to other covert areas.",
                },
                "secret_chambers": {
                    "name": "Secret Chambers",
                    "description": "Concealed rooms accessible only through hidden entrances. The chambers appear to have been used for classified activities. Multiple concealed passages lead to other secure areas.",
                },
                "abandoned_wings": {
                    "name": "Abandoned Wings",
                    "description": "Entire sections of the building that were sealed off and forgotten. Dust sheets cover furniture, and the air is stale. Multiple sealed doors lead to other abandoned sections.",
                },
                # Fallback
                "unknown_area": {
                    "name": "Mysterious Area",
                    "description": "A strange area that defies easy description. The space seems to shift and change as you look at it, with passages leading in directions that shouldn't be possible.",
                },
            }

            template = location_templates.get(
                location_type, location_templates["unknown_area"]
            )

            new_location_id = uuid4()

            # Calculate expansion depth
            source_depth = source_location.state_flags.get("expansion_depth", 0)
            new_depth = source_depth + 1

            # Add to database with expansion metadata
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO locations (id, name, short_desc, full_desc, location_type, is_dynamic, created_by, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    new_location_id,
                    template["name"],
                    f"A dynamically generated {location_type.replace('_', ' ')}",
                    template["description"],
                    "dynamic",
                    True,  # is_dynamic
                    "dynamic_generation",
                    f'{{"expansion_depth": {new_depth}, "location_type": "{location_type}", "is_dynamic": true, "can_expand": true}}',
                )

            # Add to world state
            world_state = self.state_manager.world_tracker.get_state()
            if world_state:
                new_location = LocationModel(
                    location_id=new_location_id,
                    name=template["name"],
                    description=template["description"],
                    state_flags={
                        "expansion_depth": new_depth,
                        "location_type": location_type,
                        "is_dynamic": True,
                        "can_expand": True,
                    },
                )
                world_state.locations[new_location_id] = new_location

            # Generate dynamic content for the new location
            await self._generate_dynamic_content(
                new_location_id, location_type, new_depth
            )

            # Enhance with LLM-powered descriptions if available
            if self.config.features.use_nlp:
                await self._enhance_location_with_llm(
                    new_location_id,
                    source_location,
                    direction,
                    location_type,
                    new_depth,
                )

            # Track player behavior for adaptive generation
            await self._track_player_exploration(
                source_location, direction, location_type, new_depth
            )

            logger.info(
                f"Generated enhanced location: {template['name']} ({new_location_id}) at depth {new_depth}"
            )
            return new_location_id

        except Exception as e:
            logger.error(f"Error generating enhanced location: {e}")
            return None

    async def _generate_additional_exits(
        self, location_id: UUID, source_location: Location, entry_direction: str
    ) -> None:
        """
        Generate additional exits for a newly created location.

        Args:
            location_id: The new location's ID
            source_location: The source location it was generated from
            entry_direction: The direction used to enter this location
        """
        try:
            # Get the new location from world state
            world_state = self.state_manager.world_tracker.get_state()
            if not world_state or location_id not in world_state.locations:
                return

            new_location = world_state.locations[location_id]
            location_type = new_location.state_flags.get("location_type", "unknown")

            # Direction mapping for reverse connections
            reverse_directions = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
                "up": "down",
                "down": "up",
                "outside": "inside",
                "inside": "outside",
                "loading": "return",
                "maintenance": "exit",
            }

            # Determine available directions (excluding the entry direction)
            reverse_entry = reverse_directions.get(entry_direction, "back")
            possible_directions = self._get_possible_directions(location_type)

            # Remove the reverse entry direction since that's already connected
            available_directions = [
                d for d in possible_directions if d != reverse_entry
            ]

            # Generate 1-3 additional exits randomly
            import random

            num_additional_exits = random.randint(1, min(3, len(available_directions)))
            selected_directions = random.sample(
                available_directions, num_additional_exits
            )

            for direction in selected_directions:
                # Create a "placeholder" connection that will be generated when explored
                await self._create_placeholder_connection(
                    location_id, direction, location_type
                )

            logger.info(
                f"Generated {len(selected_directions)} additional exits for {new_location.name}"
            )

        except Exception as e:
            logger.error(f"Error generating additional exits: {e}")

    def _get_possible_directions(self, location_type: str) -> list[str]:
        """
        Get possible directions for a location type.

        Args:
            location_type: The type of location

        Returns:
            List of possible directions
        """
        direction_sets = {
            "urban_street": ["north", "south", "east", "west", "inside"],
            "industrial_district": ["north", "south", "east", "west", "inside", "down"],
            "basement_access": ["north", "south", "east", "west", "down"],
            "upper_floor": ["north", "south", "east", "west", "up", "down"],
            "mechanical_room": ["north", "south", "east", "west", "maintenance"],
            "sublevel": ["north", "south", "east", "west", "down", "maintenance"],
            "industrial_zone": ["north", "south", "east", "west", "loading"],
            "loading_dock": ["north", "south", "east", "west", "outside"],
            "utility_tunnels": ["north", "south", "east", "west", "maintenance", "up"],
            "roof_access": ["north", "south", "east", "west", "down"],
            "building_interior": ["north", "south", "east", "west", "up", "down"],
            "factory_interior": ["north", "south", "east", "west", "up", "down"],
            "basement_corridor": ["north", "south", "east", "west", "down"],
            "office_space": ["north", "south", "east", "west", "up", "down"],
        }

        return direction_sets.get(location_type, ["north", "south", "east", "west"])

    async def _enhance_location_with_llm(
        self,
        location_id: UUID,
        source_location: Location,
        direction: str,
        location_type: str,
        depth: int,
    ) -> None:
        """
        Enhance location with LLM-powered descriptions and content.

        Args:
            location_id: The location to enhance
            source_location: The source location for context
            direction: Direction of expansion
            location_type: Type of location
            depth: Expansion depth
        """
        try:
            # Get player preferences for context
            preferences = self._get_player_preferences()

            # Generate enhanced location description
            enhanced_description = await self._generate_llm_location_description(
                location_id,
                source_location,
                direction,
                location_type,
                depth,
                preferences,
            )

            if enhanced_description:
                await self._update_location_description(
                    location_id, enhanced_description
                )

            # Enhance any generated NPCs with dialogue
            await self._enhance_npcs_with_llm_dialogue(
                location_id, location_type, enhanced_description or {}
            )

            # Enhance any generated objects with detailed descriptions
            await self._enhance_objects_with_llm_descriptions(
                location_id, location_type, enhanced_description or {}
            )

            logger.info(f"Enhanced location {location_id} with LLM-powered content")

        except Exception as e:
            logger.warning(f"LLM enhancement failed for location {location_id}: {e}")
            # Continue without LLM enhancement - fallback to basic generation

    async def _generate_llm_location_description(
        self,
        location_id: UUID,
        source_location: Location,
        direction: str,
        location_type: str,
        depth: int,
        preferences: dict,
    ) -> dict | None:
        """
        Generate enhanced location description using LLM.

        Args:
            location_id: Target location ID
            source_location: Source location for context
            direction: Direction of expansion
            location_type: Type of location
            depth: Expansion depth
            preferences: Player preferences

        Returns:
            Enhanced description data or None if generation fails
        """
        try:
            # Prepare context for LLM prompt
            context = {
                "source_location_name": source_location.name,
                "source_location_type": source_location.state_flags.get(
                    "location_type", "unknown"
                ),
                "source_location_description": source_location.description,
                "direction": direction,
                "location_type": location_type,
                "terrain_type": await self._get_terrain_type(source_location),
                "expansion_depth": depth,
                "exploration_style": preferences.get(
                    "exploration_style", "casual_explorer"
                ),
                "experience_level": preferences.get("experience_level", "intermediate"),
                "player_preferences": preferences,
            }

            # Add adjacent locations for context
            world_state = self.state_manager.world_tracker.get_state()
            adjacent_locations = []
            if world_state:
                for connected_id in source_location.connections.values():
                    if connected_id in world_state.locations:
                        adj_loc = world_state.locations[connected_id]
                        adjacent_locations.append(
                            {
                                "name": adj_loc.name,
                                "location_type": adj_loc.state_flags.get(
                                    "location_type", "unknown"
                                ),
                                "description": (
                                    adj_loc.description[:200] + "..."
                                    if len(adj_loc.description) > 200
                                    else adj_loc.description
                                ),
                            }
                        )

            context["adjacent_locations"] = adjacent_locations

            # Load and render prompt template
            prompt = await self._render_llm_prompt("location_description", context)

            # Generate with LLM
            response = await self._call_llm(prompt, temperature=0.8, max_tokens=1000)

            if response and "response" in response:
                import json

                try:
                    # Try to parse JSON response
                    enhanced_data = json.loads(response["response"])
                    return enhanced_data
                except json.JSONDecodeError:
                    # If not valid JSON, extract key information from text
                    return self._extract_description_from_text(response["response"])

            return None

        except Exception as e:
            logger.error(f"Error generating LLM location description: {e}")
            return None

    async def _enhance_npcs_with_llm_dialogue(
        self, location_id: UUID, location_type: str, location_description: dict
    ) -> None:
        """
        Enhance NPCs with LLM-generated dialogue.

        Args:
            location_id: Target location ID
            location_type: Type of location
            location_description: Enhanced location description data
        """
        try:
            world_state = self.state_manager.world_tracker.get_state()
            if not world_state or location_id not in world_state.locations:
                return

            location = world_state.locations[location_id]
            npcs = location.state_flags.get("npcs", [])

            for npc_data in npcs:
                enhanced_dialogue = await self._generate_llm_npc_dialogue(
                    npc_data, location_type, location_description
                )

                if enhanced_dialogue:
                    # Add enhanced dialogue to NPC data
                    npc_data["dialogue"] = enhanced_dialogue
                    npc_data["has_llm_dialogue"] = True

        except Exception as e:
            logger.error(f"Error enhancing NPCs with LLM dialogue: {e}")

    async def _enhance_objects_with_llm_descriptions(
        self, location_id: UUID, location_type: str, location_description: dict
    ) -> None:
        """
        Enhance objects with LLM-generated detailed descriptions.

        Args:
            location_id: Target location ID
            location_type: Type of location
            location_description: Enhanced location description data
        """
        try:
            world_state = self.state_manager.world_tracker.get_state()
            if not world_state or location_id not in world_state.locations:
                return

            location = world_state.locations[location_id]
            objects = location.state_flags.get("objects", [])

            for obj_data in objects:
                enhanced_description = await self._generate_llm_object_description(
                    obj_data, location_type, location_description
                )

                if enhanced_description:
                    # Update object with enhanced description
                    obj_data["enhanced_description"] = enhanced_description[
                        "detailed_description"
                    ]
                    obj_data["interactive_hints"] = enhanced_description.get(
                        "interactive_hints", []
                    )
                    obj_data["has_llm_enhancement"] = True

        except Exception as e:
            logger.error(f"Error enhancing objects with LLM descriptions: {e}")

    async def _generate_llm_npc_dialogue(
        self, npc_data: dict, location_type: str, location_description: dict
    ) -> dict | None:
        """
        Generate dialogue for an NPC using LLM.

        Args:
            npc_data: NPC information
            location_type: Type of location
            location_description: Location description data

        Returns:
            Generated dialogue data or None
        """
        try:
            context = {
                "npc_type": npc_data["name"],
                "location_name": location_description.get("name", "Unknown Location"),
                "location_type": location_type,
                "location_description": location_description.get("description", ""),
                "player_context": self._get_player_preferences(),
            }

            prompt = await self._render_llm_prompt("npc_dialogue", context)

            response = await self._call_llm(prompt, temperature=0.9, max_tokens=800)

            if response and "response" in response:
                import json

                try:
                    return json.loads(response["response"])
                except json.JSONDecodeError:
                    return {"greeting": response["response"][:200]}

            return None

        except Exception as e:
            logger.error(f"Error generating NPC dialogue: {e}")
            return None

    async def _generate_llm_object_description(
        self, obj_data: dict, location_type: str, location_description: dict
    ) -> dict | None:
        """
        Generate enhanced object description using LLM.

        Args:
            obj_data: Object information
            location_type: Type of location
            location_description: Location description data

        Returns:
            Enhanced object description data or None
        """
        try:
            context = {
                "object_name": obj_data["name"],
                "location_name": location_description.get("name", "Unknown Location"),
                "location_type": location_type,
                "location_description": location_description.get("description", ""),
                "object_context": obj_data.get("description", ""),
            }

            prompt = await self._render_llm_prompt("object_enhancement", context)

            response = await self._call_llm(prompt, temperature=0.7, max_tokens=600)

            if response and "response" in response:
                import json

                try:
                    return json.loads(response["response"])
                except json.JSONDecodeError:
                    return {"detailed_description": response["response"][:300]}

            return None

        except Exception as e:
            logger.error(f"Error generating object description: {e}")
            return None

    async def _generate_npc_dialogue_response(
        self, npc: Any, location: Location, player_state: PlayerState
    ) -> str | None:
        """
        Generate a dynamic dialogue response for an NPC using LLM.
        
        This follows best practices by requesting plain text responses instead of JSON,
        making the system more reliable and user-friendly.

        Args:
            npc: The NPC being talked to
            location: Current location
            player_state: Current player state

        Returns:
            Generated dialogue response or None
        """
        try:
            # Create a simple, clear prompt for natural dialogue
            prompt = self._create_dialogue_prompt(npc, location, player_state)

            # Request plain text response (no JSON parsing needed)
            response = await self._call_llm(
                prompt, 
                temperature=0.8, 
                max_tokens=150,  # Shorter for focused dialogue
                format=None
            )

            if response and "response" in response:
                dialogue_text = response["response"].strip()
                
                # Clean up any unwanted formatting
                dialogue_text = self._clean_dialogue_text(dialogue_text)
                
                if dialogue_text:
                    # Simple, reliable formatting
                    return f"{npc.name} looks at you. \"{dialogue_text}\""

            return None

        except Exception as e:
            logger.error(f"Error generating NPC dialogue response: {e}")
            return None

    def _create_dialogue_prompt(self, npc: Any, location: Location, player_state: PlayerState) -> str:
        """Create a focused prompt for NPC dialogue generation."""
        
        # Determine NPC personality based on name/type
        npc_personality = self._get_npc_personality_hint(npc.name)
        
        prompt = f"""You are roleplaying as {npc.name}, a {npc_personality} character in {location.name}.

Setting: {location.description[:200]}...

Guidelines:
- Respond as {npc.name} would speak
- Keep response to 1-2 sentences
- Stay in character
- Be helpful but realistic for the setting
- No special formatting or markup

Player approaches you to talk. What do you say?

Response:"""
        
        return prompt

    def _get_npc_personality_hint(self, npc_name: str) -> str:
        """Get a personality hint based on NPC name."""
        name_lower = npc_name.lower()
        
        if "guard" in name_lower or "security" in name_lower:
            return "professional security guard"
        elif "merchant" in name_lower or "shop" in name_lower:
            return "friendly merchant"
        elif "scholar" in name_lower or "librarian" in name_lower:
            return "knowledgeable scholar"
        elif "innkeeper" in name_lower or "bartender" in name_lower:
            return "welcoming innkeeper"
        else:
            return "local resident"

    def _clean_dialogue_text(self, text: str) -> str:
        """Clean up dialogue text from any unwanted formatting."""
        # Remove any JSON artifacts
        text = text.replace("```json", "").replace("```", "")
        text = text.replace("{", "").replace("}", "")
        text = text.replace('"greeting":', "").replace('"', "")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Ensure it doesn't start with common JSON keys
        if text.startswith(("greeting:", "dialogue:", "response:")):
            text = text.split(":", 1)[1].strip()
            
        return text.strip()

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 800,
        format: str | None = "json",
    ) -> dict[str, Any] | None:
        """
        Helper method to call LLM with consistent parameters.

        Args:
            prompt: The prompt to send
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            format: Format for response ("json" or None)

        Returns:
            Response dict or None if failed
        """
        try:
            from game_loop.llm.ollama.client import OllamaModelParameters

            params = OllamaModelParameters(
                model="qwen2.5:3b",
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                format=format,
            )

            response = await self.ollama_client.generate_completion(
                prompt=prompt, params=params, raw_response=True
            )

            return response

        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}", exc_info=True)
            return None

    async def _render_llm_prompt(self, template_name: str, context: dict) -> str:
        """
        Render an LLM prompt template with context.

        Args:
            template_name: Name of the template file
            context: Context data for template rendering

        Returns:
            Rendered prompt string
        """
        try:
            import re
            from pathlib import Path

            # Load template file
            template_path = (
                Path(__file__).parent.parent
                / "llm"
                / "prompts"
                / "world_generation"
                / f"{template_name}.txt"
            )

            if not template_path.exists():
                logger.warning(f"LLM template not found: {template_path}")
                return f"Generate creative content for {template_name} with context: {context}"

            with open(template_path) as f:
                template = f.read()

            # Simple template rendering (replace {{variable}} with values)
            def replace_var(match):
                var_name = match.group(1)
                return str(context.get(var_name, f"[{var_name}]"))

            # Replace simple variables
            rendered = re.sub(r"\{\{(\w+)\}\}", replace_var, template)

            # Handle array iterations (simplified)
            if (
                "{{#adjacent_locations}}" in rendered
                and "adjacent_locations" in context
            ):
                locations_text = ""
                for loc in context["adjacent_locations"]:
                    locations_text += f"- **{loc['name']}** ({loc['location_type']}): {loc['description']}\n"

                # Replace the loop section
                pattern = r"\{\{#adjacent_locations\}\}.*?\{\{/adjacent_locations\}\}"
                rendered = re.sub(pattern, locations_text, rendered, flags=re.DOTALL)

            return rendered

        except Exception as e:
            logger.error(f"Error rendering LLM prompt template: {e}")
            return f"Generate creative content for {template_name}"

    async def _update_location_description(
        self, location_id: UUID, enhanced_description: dict
    ) -> None:
        """
        Update location with enhanced LLM-generated description.

        Args:
            location_id: Location ID to update
            enhanced_description: Enhanced description data from LLM
        """
        try:
            # Update in world state
            world_state = self.state_manager.world_tracker.get_state()
            if world_state and location_id in world_state.locations:
                location = world_state.locations[location_id]

                # Update with enhanced content
                if "name" in enhanced_description:
                    location.name = enhanced_description["name"]

                if "description" in enhanced_description:
                    location.description = enhanced_description["description"]

                # Add LLM enhancement metadata
                location.state_flags["llm_enhanced"] = True
                location.state_flags["enhancement_data"] = enhanced_description

            # Update in database
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE locations 
                    SET name = $2, full_desc = $3, state_json = state_json || $4
                    WHERE id = $1
                    """,
                    location_id,
                    enhanced_description.get("name", "Enhanced Location"),
                    enhanced_description.get("description", "An enhanced location."),
                    '{"llm_enhanced": true}',
                )

        except Exception as e:
            logger.error(f"Error updating location description: {e}")

    def _extract_description_from_text(self, text_response: str) -> dict:
        """
        Extract structured description data from unstructured LLM text response.

        Args:
            text_response: Raw text response from LLM

        Returns:
            Structured description data
        """
        # Simple extraction logic for fallback
        lines = text_response.strip().split("\n")

        # Try to extract name (usually first line or after "Name:")
        name = "Generated Location"
        description = text_response[:500]  # Truncate for safety

        for line in lines:
            if line.startswith("Name:") or line.startswith("**"):
                name = line.replace("Name:", "").replace("**", "").strip()
                break

        return {"name": name, "description": description, "source": "llm_fallback"}

    async def _track_player_exploration(
        self, source_location: Location, direction: str, location_type: str, depth: int
    ) -> None:
        """
        Track player exploration patterns for adaptive content generation.

        Args:
            source_location: The location the player expanded from
            direction: Direction of exploration
            location_type: Type of location generated
            depth: Exploration depth
        """
        try:
            # Get current player state
            player_state, _ = self.state_manager.get_current_state()
            if not player_state:
                return

            # Initialize behavior tracking if not present
            if (
                not hasattr(player_state, "behavior_stats")
                or player_state.behavior_stats is None
            ):
                player_state.behavior_stats = {
                    "preferred_directions": {},
                    "preferred_location_types": {},
                    "exploration_depth": 0,
                    "total_expansions": 0,
                    "session_start": None,
                }

            stats = player_state.behavior_stats

            # Track direction preferences
            if direction not in stats["preferred_directions"]:
                stats["preferred_directions"][direction] = 0
            stats["preferred_directions"][direction] += 1

            # Track location type preferences
            if location_type not in stats["preferred_location_types"]:
                stats["preferred_location_types"][location_type] = 0
            stats["preferred_location_types"][location_type] += 1

            # Update exploration metrics
            stats["exploration_depth"] = max(stats["exploration_depth"], depth)
            stats["total_expansions"] += 1

            # Set session start if first expansion
            if stats["session_start"] is None:
                import time

                stats["session_start"] = time.time()

            logger.info(
                f"Tracked exploration: {direction} -> {location_type} (depth: {depth})"
            )

        except Exception as e:
            logger.error(f"Error tracking player exploration: {e}")

    def _get_player_preferences(self) -> dict[str, Any]:
        """
        Get player preferences based on tracked behavior.

        Returns:
            Dictionary containing player preference data
        """
        try:
            player_state, _ = self.state_manager.get_current_state()
            if (
                not player_state
                or not hasattr(player_state, "behavior_stats")
                or player_state.behavior_stats is None
            ):
                return {
                    "preferred_directions": [],
                    "preferred_location_types": [],
                    "exploration_style": "unknown",
                    "experience_level": "beginner",
                }

            stats = player_state.behavior_stats

            # Analyze direction preferences
            direction_counts = stats.get("preferred_directions", {})
            preferred_directions = sorted(
                direction_counts.keys(), key=lambda k: direction_counts[k], reverse=True
            )[:3]

            # Analyze location type preferences
            type_counts = stats.get("preferred_location_types", {})
            preferred_types = sorted(
                type_counts.keys(), key=lambda k: type_counts[k], reverse=True
            )[:3]

            # Determine exploration style
            total_expansions = stats.get("total_expansions", 0)
            max_depth = stats.get("exploration_depth", 0)

            if max_depth >= 4:
                exploration_style = "deep_explorer"
            elif total_expansions >= 10:
                exploration_style = "broad_explorer"
            elif len(preferred_directions) <= 2:
                exploration_style = "focused_explorer"
            else:
                exploration_style = "casual_explorer"

            # Determine experience level
            if total_expansions >= 20:
                experience_level = "expert"
            elif total_expansions >= 10:
                experience_level = "experienced"
            elif total_expansions >= 5:
                experience_level = "intermediate"
            else:
                experience_level = "beginner"

            return {
                "preferred_directions": preferred_directions,
                "preferred_location_types": preferred_types,
                "exploration_style": exploration_style,
                "experience_level": experience_level,
                "total_expansions": total_expansions,
                "max_depth": max_depth,
            }

        except Exception as e:
            logger.error(f"Error getting player preferences: {e}")
            return {
                "preferred_directions": [],
                "preferred_location_types": [],
                "exploration_style": "unknown",
                "experience_level": "beginner",
            }

    async def _generate_dynamic_content(
        self, location_id: UUID, location_type: str, depth: int
    ) -> None:
        """
        Generate dynamic content (NPCs, objects, interactions) for a new location.

        Args:
            location_id: The location to populate with content
            location_type: The type of location for context-appropriate content
            depth: Expansion depth for complexity scaling
        """
        try:
            # Determine content generation probability based on location type and depth
            content_probability = self._get_content_probability(location_type, depth)

            import random

            # Generate objects based on location type
            if random.random() < content_probability.get("objects", 0.3):
                await self._generate_contextual_objects(
                    location_id, location_type, depth
                )

            # Generate NPCs (less common, more interesting at deeper levels)
            if random.random() < content_probability.get("npcs", 0.1 + (depth * 0.05)):
                await self._generate_contextual_npcs(location_id, location_type, depth)

            # Generate simple quest hooks or interactions
            if random.random() < content_probability.get("interactions", 0.2):
                await self._generate_quest_hooks(location_id, location_type, depth)

            logger.info(
                f"Generated dynamic content for location {location_id} (type: {location_type}, depth: {depth})"
            )

        except Exception as e:
            logger.error(f"Error generating dynamic content: {e}")

    def _get_content_probability(
        self, location_type: str, depth: int
    ) -> dict[str, float]:
        """
        Get adaptive content generation probabilities based on location characteristics and player preferences.

        Args:
            location_type: Type of location
            depth: Expansion depth

        Returns:
            Dictionary of content type probabilities
        """
        base_probabilities = {
            # Urban locations have more objects, fewer NPCs
            "urban_street": {"objects": 0.4, "npcs": 0.1, "interactions": 0.3},
            "commercial_district": {"objects": 0.6, "npcs": 0.2, "interactions": 0.4},
            "residential_area": {"objects": 0.5, "npcs": 0.15, "interactions": 0.25},
            # Industrial locations have equipment and hazards
            "industrial_zone": {"objects": 0.7, "npcs": 0.05, "interactions": 0.3},
            "factory_interior": {"objects": 0.8, "npcs": 0.1, "interactions": 0.4},
            "warehouse_district": {"objects": 0.6, "npcs": 0.08, "interactions": 0.2},
            # Office/building locations have documents and equipment
            "office_space": {"objects": 0.5, "npcs": 0.12, "interactions": 0.35},
            "building_interior": {"objects": 0.4, "npcs": 0.1, "interactions": 0.3},
            # Underground locations are more mysterious
            "basement_corridor": {"objects": 0.3, "npcs": 0.08, "interactions": 0.4},
            "utility_tunnels": {"objects": 0.4, "npcs": 0.05, "interactions": 0.3},
            "deep_chambers": {"objects": 0.2, "npcs": 0.15, "interactions": 0.6},
        }

        probabilities = base_probabilities.get(
            location_type, {"objects": 0.3, "npcs": 0.1, "interactions": 0.2}
        )

        # Apply player preferences for adaptive generation
        preferences = self._get_player_preferences()

        # Boost content generation for preferred location types
        if location_type in preferences["preferred_location_types"]:
            type_boost = 1.2
            probabilities = {
                key: min(value * type_boost, 0.9)
                for key, value in probabilities.items()
            }

        # Adjust based on exploration style
        exploration_style = preferences["exploration_style"]
        if exploration_style == "deep_explorer":
            # Deep explorers get more interactions and mysterious content
            probabilities["interactions"] = min(
                probabilities["interactions"] * 1.3, 0.8
            )
            probabilities["npcs"] = min(probabilities["npcs"] * 1.2, 0.6)
        elif exploration_style == "broad_explorer":
            # Broad explorers get more objects to discover
            probabilities["objects"] = min(probabilities["objects"] * 1.3, 0.9)
        elif exploration_style == "focused_explorer":
            # Focused explorers get higher quality content
            depth_boost = 1.4
            probabilities = {
                key: min(value * depth_boost, 0.9)
                for key, value in probabilities.items()
            }

        # Adjust based on experience level
        experience_level = preferences["experience_level"]
        if experience_level == "expert":
            # Experts get more complex interactions
            probabilities["interactions"] = min(
                probabilities["interactions"] * 1.4, 0.8
            )
            probabilities["npcs"] = min(probabilities["npcs"] * 1.3, 0.6)
        elif experience_level == "beginner":
            # Beginners get more objects to help them learn
            probabilities["objects"] = min(probabilities["objects"] * 1.2, 0.8)

        # Increase complexity with depth
        depth_multiplier = 1.0 + (depth * 0.1)
        return {
            key: min(value * depth_multiplier, 0.9)
            for key, value in probabilities.items()
        }

    async def _generate_contextual_objects(
        self, location_id: UUID, location_type: str, depth: int
    ) -> None:
        """
        Generate objects appropriate for the location type.

        Args:
            location_id: Target location
            location_type: Type for context
            depth: Depth for item quality/rarity
        """
        try:
            object_templates = {
                "urban_street": [
                    "abandoned car",
                    "street sign",
                    "trash bin",
                    "broken streetlight",
                    "newspaper stand",
                ],
                "commercial_district": [
                    "cash register",
                    "shopping cart",
                    "store display",
                    "advertising sign",
                    "security camera",
                ],
                "residential_area": [
                    "mailbox",
                    "garden tool",
                    "bicycle",
                    "house key",
                    "family photo",
                ],
                "industrial_zone": [
                    "machinery part",
                    "tool box",
                    "safety equipment",
                    "control panel",
                    "warning sign",
                ],
                "factory_interior": [
                    "conveyor belt",
                    "assembly tool",
                    "industrial crane",
                    "quality control station",
                    "shipping container",
                ],
                "office_space": [
                    "desk computer",
                    "office chair",
                    "filing cabinet",
                    "coffee maker",
                    "employee handbook",
                ],
                "basement_corridor": [
                    "electrical panel",
                    "pipe fitting",
                    "maintenance log",
                    "emergency flashlight",
                    "access card",
                ],
                "utility_tunnels": [
                    "valve wheel",
                    "pressure gauge",
                    "maintenance tool",
                    "pipe section",
                    "junction box",
                ],
                "deep_chambers": [
                    "ancient artifact",
                    "mysterious device",
                    "stone tablet",
                    "crystal formation",
                    "ceremonial object",
                ],
            }

            templates = object_templates.get(
                location_type, ["mysterious object", "unusual item", "unknown device"]
            )

            import random

            num_objects = random.randint(1, min(3, depth + 1))

            for i in range(num_objects):
                object_name = random.choice(templates)
                object_desc = (
                    f"A {object_name} found in this {location_type.replace('_', ' ')}."
                )

                # Add more detailed descriptions for deeper locations
                if depth >= 2:
                    quality_descriptors = [
                        "weathered",
                        "ancient",
                        "mysterious",
                        "sophisticated",
                        "unusual",
                    ]
                    descriptor = random.choice(quality_descriptors)
                    object_desc = (
                        f"A {descriptor} {object_name} that seems significant."
                    )

                # Store object in location's state
                await self._add_object_to_location(
                    location_id, object_name, object_desc
                )

        except Exception as e:
            logger.error(f"Error generating contextual objects: {e}")

    async def _generate_contextual_npcs(
        self, location_id: UUID, location_type: str, depth: int
    ) -> None:
        """
        Generate NPCs appropriate for the location type.

        Args:
            location_id: Target location
            location_type: Type for context
            depth: Depth for NPC complexity
        """
        try:
            npc_templates = {
                "urban_street": [
                    "homeless person",
                    "street musician",
                    "scavenger",
                    "wanderer",
                ],
                "commercial_district": [
                    "shop keeper",
                    "security guard",
                    "maintenance worker",
                    "lost customer",
                ],
                "office_space": [
                    "office worker",
                    "security guard",
                    "janitor",
                    "IT technician",
                ],
                "industrial_zone": [
                    "factory worker",
                    "supervisor",
                    "maintenance tech",
                    "safety inspector",
                ],
                "basement_corridor": [
                    "maintenance worker",
                    "security guard",
                    "facilities manager",
                ],
                "deep_chambers": [
                    "mysterious figure",
                    "ancient guardian",
                    "underground dweller",
                    "archaeologist",
                ],
            }

            templates = npc_templates.get(
                location_type, ["mysterious person", "unknown individual"]
            )

            import random

            npc_type = random.choice(templates)

            # Generate basic NPC attributes
            npc_name = f"A {npc_type}"
            npc_desc = f"A {npc_type} who seems to belong in this {location_type.replace('_', ' ')}."

            # Add more interesting NPCs at deeper levels
            if depth >= 3:
                advanced_descriptors = [
                    "enigmatic",
                    "knowledgeable",
                    "suspicious",
                    "helpful",
                    "dangerous",
                ]
                descriptor = random.choice(advanced_descriptors)
                npc_desc = f"An {descriptor} {npc_type} with stories to tell."

            # Store NPC in location's state
            await self._add_npc_to_location(location_id, npc_name, npc_desc)

        except Exception as e:
            logger.error(f"Error generating contextual NPCs: {e}")

    async def _generate_quest_hooks(
        self, location_id: UUID, location_type: str, depth: int
    ) -> None:
        """
        Generate simple quest hooks or interactive elements.

        Args:
            location_id: Target location
            location_type: Type for context
            depth: Depth for interaction complexity
        """
        try:
            quest_templates = {
                "urban_street": [
                    "graffiti message",
                    "posted notice",
                    "emergency alert",
                    "missing person poster",
                ],
                "office_space": [
                    "urgent memo",
                    "computer terminal",
                    "voice message",
                    "security alert",
                ],
                "industrial_zone": [
                    "warning notice",
                    "maintenance request",
                    "emergency protocol",
                    "system alert",
                ],
                "basement_corridor": [
                    "maintenance log",
                    "security breach",
                    "system malfunction",
                    "access denied",
                ],
                "deep_chambers": [
                    "ancient inscription",
                    "mysterious mechanism",
                    "glowing symbol",
                    "hidden passage",
                ],
            }

            templates = quest_templates.get(
                location_type, ["mysterious sign", "unusual marking"]
            )

            import random

            hook_type = random.choice(templates)
            hook_desc = f"You notice {hook_type} that seems important."

            # Add depth-based complexity
            if depth >= 2:
                hook_desc += " It might be worth investigating further."

            # Store quest hook as a special object
            await self._add_quest_hook_to_location(location_id, hook_type, hook_desc)

        except Exception as e:
            logger.error(f"Error generating quest hooks: {e}")

    async def _add_object_to_location(
        self, location_id: UUID, name: str, description: str
    ) -> None:
        """Add an object to a location's inventory."""
        try:
            # For now, store as state metadata - could be expanded to use proper object system
            world_state = self.state_manager.world_tracker.get_state()
            if world_state and location_id in world_state.locations:
                location = world_state.locations[location_id]
                if "objects" not in location.state_flags:
                    location.state_flags["objects"] = []
                location.state_flags["objects"].append(
                    {"name": name, "description": description}
                )

        except Exception as e:
            logger.error(f"Error adding object to location: {e}")

    async def _add_npc_to_location(
        self, location_id: UUID, name: str, description: str
    ) -> None:
        """Add an NPC to a location."""
        try:
            world_state = self.state_manager.world_tracker.get_state()
            if world_state and location_id in world_state.locations:
                location = world_state.locations[location_id]
                if "npcs" not in location.state_flags:
                    location.state_flags["npcs"] = []
                location.state_flags["npcs"].append(
                    {"name": name, "description": description}
                )

        except Exception as e:
            logger.error(f"Error adding NPC to location: {e}")

    async def _add_quest_hook_to_location(
        self, location_id: UUID, name: str, description: str
    ) -> None:
        """Add a quest hook to a location."""
        try:
            world_state = self.state_manager.world_tracker.get_state()
            if world_state and location_id in world_state.locations:
                location = world_state.locations[location_id]
                if "quest_hooks" not in location.state_flags:
                    location.state_flags["quest_hooks"] = []
                location.state_flags["quest_hooks"].append(
                    {"name": name, "description": description}
                )

        except Exception as e:
            logger.error(f"Error adding quest hook to location: {e}")

    async def _create_placeholder_connection(
        self, location_id: UUID, direction: str, location_type: str
    ) -> None:
        """
        Create a placeholder connection that will trigger generation when explored.

        Args:
            location_id: The location to add the connection to
            direction: The direction of the connection
            location_type: The type of the source location
        """
        try:
            # Add the direction to the location's connections with a special placeholder UUID
            # This will be detected in _handle_movement and trigger generation
            world_state = self.state_manager.world_tracker.get_state()
            if world_state and location_id in world_state.locations:
                # Use a special UUID that indicates this needs generation
                from uuid import UUID

                placeholder_uuid = UUID("00000000-0000-0000-0000-000000000000")

                world_state.locations[location_id].connections[
                    direction
                ] = placeholder_uuid

                # Don't add placeholder connections to database - they're memory-only until generated
                # The placeholder UUID will be replaced when the connection is actually explored

                logger.info(
                    f"Created placeholder connection: {direction} from {location_id}"
                )

        except Exception as e:
            logger.error(f"Error creating placeholder connection: {e}")

    async def _add_connection(
        self, from_location_id: UUID, to_location_id: UUID, direction: str
    ) -> None:
        """Add a bidirectional connection between two locations."""
        try:
            # Direction mapping for reverse connections
            reverse_directions = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
                "up": "down",
                "down": "up",
                "outside": "inside",
                "inside": "outside",
                "loading": "return",
                "maintenance": "exit",
            }

            reverse_direction = reverse_directions.get(direction, "back")

            from uuid import uuid4

            # Add to database
            async with self.db_pool.acquire() as conn:
                # Forward connection
                await conn.execute(
                    """
                    INSERT INTO location_connections (id, from_location_id, to_location_id, direction, connection_type, requirements_json, is_visible)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    uuid4(),
                    from_location_id,
                    to_location_id,
                    direction,
                    "path",
                    "{}",
                    True,
                )

                # Reverse connection
                await conn.execute(
                    """
                    INSERT INTO location_connections (id, from_location_id, to_location_id, direction, connection_type, requirements_json, is_visible)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    uuid4(),
                    to_location_id,
                    from_location_id,
                    reverse_direction,
                    "path",
                    "{}",
                    True,
                )

            # Add to world state
            world_state = self.state_manager.world_tracker.get_state()
            if world_state:
                if from_location_id in world_state.locations:
                    world_state.locations[from_location_id].connections[
                        direction
                    ] = to_location_id
                if to_location_id in world_state.locations:
                    world_state.locations[to_location_id].connections[
                        reverse_direction
                    ] = from_location_id

            logger.info(
                f"Added connection: {direction} ({from_location_id} -> {to_location_id})"
            )

        except Exception as e:
            logger.error(f"Error adding connection: {e}")

    async def _load_default_rules(self) -> None:
        """Load default game rules for new games."""
        try:
            from pathlib import Path

            # Try to load default rules file
            rules_file = Path("rules/default_game_rules.yaml")
            if rules_file.exists():
                loaded_count = self.rules_engine.load_rules_from_file(str(rules_file))
                logger.info(f"Loaded {loaded_count} default rules from {rules_file}")
                self.console.print(f"[green]Loaded {loaded_count} game rules[/green]")
            else:
                # Try alternative path
                project_root = Path(__file__).parent.parent.parent
                rules_file = project_root / "rules" / "default_game_rules.yaml"
                if rules_file.exists():
                    loaded_count = self.rules_engine.load_rules_from_file(
                        str(rules_file)
                    )
                    logger.info(
                        f"Loaded {loaded_count} default rules from {rules_file}"
                    )
                    self.console.print(
                        f"[green]Loaded {loaded_count} game rules[/green]"
                    )
                else:
                    logger.warning(
                        "No default rules file found, rules engine will start empty"
                    )
                    self.console.print("[yellow]No default rules file found[/yellow]")
        except Exception as e:
            logger.error(f"Error loading default rules: {e}")
            self.console.print(f"[red]Failed to load default rules: {e}[/red]")

    async def _evaluate_rules_pre_command(
        self,
        command: ParsedCommand,
        player_state: PlayerState,
        current_location: Location,
        world_state: WorldState,
    ) -> ActionResult | None:
        """Evaluate rules before command execution to check for blocking conditions."""
        try:
            from game_loop.core.rules.rule_models import (
                ActionType,
                RuleEvaluationContext,
            )

            # Create evaluation context
            context = RuleEvaluationContext(
                player_state=self._convert_player_state_for_rules(player_state),
                world_state=self._convert_world_state_for_rules(world_state),
                current_action=(
                    command.command_type.name if command.command_type else "unknown"
                ),
                action_parameters={
                    "subject": command.subject,
                    "target": command.target,
                    "parameters": command.parameters,
                },
                current_location=(
                    str(current_location.location_id) if current_location else None
                ),
                location_data=(
                    self._convert_location_for_rules(current_location)
                    if current_location
                    else {}
                ),
            )

            # Evaluate rules
            results = self.rules_engine.evaluate_rules(context)

            # Check for blocking actions
            for result in results:
                if result.triggered:
                    rule = self.rules_engine.get_rule(result.rule_id)
                    if rule:
                        for action in rule.actions:
                            if action.action_type == ActionType.BLOCK_ACTION:
                                reason = action.parameters.get(
                                    "reason", "Action blocked by game rules"
                                )
                                suggestions = action.parameters.get("suggestions", [])

                                feedback = reason
                                if suggestions:
                                    feedback += "\n\nSuggestions:"
                                    for suggestion in suggestions:
                                        feedback += f"\n  • {suggestion}"

                                return ActionResult(
                                    success=False, feedback_message=feedback
                                )

            return None  # No blocking rules triggered

        except Exception as e:
            logger.error(f"Error evaluating pre-command rules: {e}")
            return None

    async def _evaluate_rules_post_command(
        self,
        command: ParsedCommand,
        action_result: ActionResult,
        player_state: PlayerState,
        current_location: Location,
        world_state: WorldState,
    ) -> None:
        """Evaluate rules after command execution to trigger notifications and effects."""
        try:
            from game_loop.core.rules.rule_models import (
                RuleEvaluationContext,
            )

            # Get updated state after command
            updated_player_state = self.state_manager.player_tracker.get_state()
            updated_world_state = self.state_manager.world_tracker.get_state()

            # Create evaluation context with updated state
            context = RuleEvaluationContext(
                player_state=self._convert_player_state_for_rules(
                    updated_player_state or player_state
                ),
                world_state=self._convert_world_state_for_rules(
                    updated_world_state or world_state
                ),
                current_action=(
                    command.command_type.name if command.command_type else "unknown"
                ),
                action_parameters={
                    "subject": command.subject,
                    "target": command.target,
                    "parameters": command.parameters,
                    "result": "success" if action_result.success else "failure",
                },
                current_location=(
                    str(current_location.location_id) if current_location else None
                ),
                location_data=(
                    self._convert_location_for_rules(current_location)
                    if current_location
                    else {}
                ),
            )

            # Evaluate rules
            results = self.rules_engine.evaluate_rules(context)

            # Process triggered rules
            for result in results:
                if result.triggered:
                    rule = self.rules_engine.get_rule(result.rule_id)
                    if rule:
                        await self._apply_rule_actions(
                            rule.actions, updated_player_state or player_state
                        )

            # Trigger event-based rules through trigger manager
            event_data = {
                "player_state": self._convert_player_state_for_rules(
                    updated_player_state or player_state
                ),
                "world_state": self._convert_world_state_for_rules(
                    updated_world_state or world_state
                ),
                "action": (
                    command.command_type.name if command.command_type else "unknown"
                ),
                "action_parameters": {
                    "subject": command.subject,
                    "result": "success" if action_result.success else "failure",
                },
                "timestamp": str(datetime.now()),
            }

            # Process action event
            self.rule_trigger_manager.process_event("action_performed", event_data)

        except Exception as e:
            logger.error(f"Error evaluating post-command rules: {e}")

    def _convert_player_state_for_rules(self, player_state: PlayerState) -> dict:
        """Convert PlayerState to dict format for rule evaluation."""
        try:
            state_dict = {
                "health": (
                    player_state.health if hasattr(player_state, "health") else 100
                ),
                "max_health": (
                    player_state.max_health
                    if hasattr(player_state, "max_health")
                    else 100
                ),
                "level": player_state.level if hasattr(player_state, "level") else 1,
                "experience": (
                    player_state.experience
                    if hasattr(player_state, "experience")
                    else 0
                ),
                "inventory_count": (
                    len(player_state.inventory)
                    if hasattr(player_state, "inventory")
                    else 0
                ),
                "max_inventory": 10,  # Default max inventory
                "location_id": (
                    player_state.current_location_id
                    if hasattr(player_state, "current_location_id")
                    else None
                ),
            }

            # Add stats if available
            if hasattr(player_state, "stats") and player_state.stats:
                state_dict["stats"] = {
                    "strength": player_state.stats.strength,
                    "dexterity": player_state.stats.dexterity,
                    "intelligence": player_state.stats.intelligence,
                }

            return state_dict
        except Exception as e:
            logger.error(f"Error converting player state for rules: {e}")
            return {
                "health": 100,
                "max_health": 100,
                "level": 1,
                "experience": 0,
                "inventory_count": 0,
                "max_inventory": 10,
            }

    def _convert_world_state_for_rules(self, world_state: WorldState) -> dict:
        """Convert WorldState to dict format for rule evaluation."""
        try:
            return {
                "time_of_day": "day",  # Default
                "weather": "clear",  # Default
                "danger_level": 1,  # Default
            }
        except Exception as e:
            logger.error(f"Error converting world state for rules: {e}")
            return {"time_of_day": "day", "weather": "clear", "danger_level": 1}

    def _convert_location_for_rules(self, location: Location) -> dict:
        """Convert Location to dict format for rule evaluation."""
        try:
            return {
                "id": location.location_id,
                "name": location.name,
                "description": location.description,
                "has_interesting_objects": (
                    len(location.objects) > 0 if hasattr(location, "objects") else False
                ),
                "light_level": 5,  # Default bright
                "has_shelter": True,  # Default assumption
            }
        except Exception as e:
            logger.error(f"Error converting location for rules: {e}")
            return {
                "id": "unknown",
                "name": "Unknown",
                "light_level": 5,
                "has_shelter": True,
            }

    async def trigger_rules_for_state_change(
        self, change_type: str, old_state: dict = None, new_state: dict = None
    ) -> None:
        """Trigger rule evaluation for specific state changes."""
        try:
            # Get current game state
            player_state = self.state_manager.player_tracker.get_state()
            world_state = self.state_manager.world_tracker.get_state()
            current_location = await self.state_manager.get_current_location_details()

            if not player_state:
                return

            # Create event data for trigger
            event_data = {
                "player_state": self._convert_player_state_for_rules(player_state),
                "world_state": (
                    self._convert_world_state_for_rules(world_state)
                    if world_state
                    else {}
                ),
                "change_type": change_type,
                "old_state": old_state or {},
                "new_state": new_state or {},
                "timestamp": str(datetime.now()),
            }

            # Map change types to trigger events
            event_mapping = {
                "health_change": "health_changed",
                "inventory_change": "inventory_changed",
                "location_change": "location_changed",
                "level_change": "state_changed",
                "experience_change": "state_changed",
            }

            event_type = event_mapping.get(change_type, "state_changed")

            # Process through trigger manager
            self.rule_trigger_manager.process_event(event_type, event_data)

        except Exception as e:
            logger.error(f"Error triggering rules for state change {change_type}: {e}")

    async def _apply_rule_actions(
        self, actions: list, player_state: PlayerState
    ) -> None:
        """Apply rule actions to the game state."""
        try:
            from game_loop.core.rules.rule_models import ActionType

            for action in actions:
                if action.action_type == ActionType.SEND_MESSAGE:
                    message = action.parameters.get("message", "")
                    style = action.parameters.get("style", "info")

                    # Apply style formatting
                    if style == "critical":
                        self.console.print(f"[bold red]{message}[/bold red]")
                    elif style == "warning":
                        self.console.print(f"[yellow]{message}[/yellow]")
                    elif style == "success":
                        self.console.print(f"[green]{message}[/green]")
                    elif style == "hint":
                        self.console.print(f"[cyan]{message}[/cyan]")
                    else:
                        self.console.print(f"[blue]{message}[/blue]")

                elif action.action_type == ActionType.MODIFY_STATE:
                    # Apply state modifications (placeholder implementation)
                    target_path = action.target_path
                    parameters = action.parameters

                    if target_path and "value" in parameters:
                        # For now, just log state modifications
                        # In a full implementation, this would update the actual game state
                        logger.info(
                            f"Rule triggered state modification: {target_path} = {parameters.get('value')}"
                        )
                        self.console.print(f"[dim]State modified: {target_path}[/dim]")

                elif action.action_type == ActionType.GRANT_REWARD:
                    # Apply rewards (placeholder implementation)
                    parameters = action.parameters
                    experience = parameters.get("experience", 0)
                    gold = parameters.get("gold", 0)
                    message = parameters.get("message", "")

                    if experience > 0:
                        logger.info(f"Rule granted {experience} experience")
                        self.console.print(f"[green]+{experience} experience![/green]")

                    if gold > 0:
                        logger.info(f"Rule granted {gold} gold")
                        self.console.print(f"[yellow]+{gold} gold![/yellow]")

                    if message:
                        self.console.print(f"[green]{message}[/green]")

                elif action.action_type == ActionType.TRIGGER_EVENT:
                    # Trigger other game events (placeholder implementation)
                    parameters = action.parameters
                    event_type = parameters.get("event_type", "")
                    event_data = parameters.get("data", {})

                    logger.info(f"Rule triggered event: {event_type}")
                    self.console.print(f"[cyan]Event triggered: {event_type}[/cyan]")

                    # Could trigger additional rule evaluations here
                    if event_type and hasattr(self, "rule_trigger_manager"):
                        self.rule_trigger_manager.process_event(event_type, event_data)

        except Exception as e:
            logger.error(f"Error applying rule actions: {e}")
