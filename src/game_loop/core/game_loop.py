"""
Core game loop implementation for Game Loop.
Handles the main game loop, input processing, and output generation.
"""

import logging
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

        # Initialize LLM client
        self.ollama_client = OllamaClient()

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

    async def initialize(self, session_id: UUID | None = None) -> None:
        """Initialize the game environment, loading or creating game state."""
        self.console.print("[bold green]Initializing Game Loop...[/bold green]")

        try:
            # Initialize database session factory
            await self.session_factory.initialize()

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
            return ActionResult(
                success=False,
                feedback_message=f"You cannot go {normalized_direction} from here.",
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
        """Handle talking to an NPC, returning ActionResult."""
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

        # For now, just provide a simple response based on NPC state
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
