"""
Core game loop implementation for Game Loop.
Handles the main game loop, input processing, and output generation.
"""

from typing import Any

from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.enhanced_input_processor import EnhancedInputProcessor
from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand
from game_loop.core.location import LocationDisplay, create_demo_location
from game_loop.core.state import GameState, Location, WorldState
from game_loop.llm.config import ConfigManager


class GameLoop:
    """Main game loop implementation for the Game Loop text adventure."""

    def __init__(self, config: GameConfig, console: Console | None = None):
        """
        Initialize the game loop with configuration.

        Args:
            config: Configuration for the game
            console: Rich console for output, creates new one if not provided
        """
        self.config = config
        self.console = console if console else Console()
        self.location_display = LocationDisplay(self.console)
        self.game_state = GameState()
        self.running = False

        # Initialize config manager for LLM with correct prompt template directory
        from pathlib import Path

        self.config_manager = ConfigManager()

        # Update the prompt template directory to point to our implementation
        project_root = Path(__file__).parent.parent.parent
        prompt_dir = project_root / "game_loop" / "llm" / "prompts"
        self.config_manager.prompt_config.template_dir = str(prompt_dir)

        # Create enhanced input processor with NLP capabilities
        self.input_processor = EnhancedInputProcessor(
            config_manager=self.config_manager,
            console=self.console,
            use_nlp=config.features.use_nlp,
        )

        # Fallback to basic input processor if needed
        self.basic_input_processor = InputProcessor(self.console)

    def initialize(self) -> None:
        """Initialize the game environment and load initial state."""
        self.console.print("[bold green]Initializing Game Loop...[/bold green]")

        # In a full implementation, we would:
        # 1. Load world data from database
        # 2. Initialize any services needed (LLM, embedding, etc.)
        # 3. Load saved games if applicable

        # For this basic implementation, we'll create a demo world
        self._create_demo_world()

        # Ask for player name
        player_name = self._get_player_name()

        # Initialize the game state with the player name and starting location
        starting_location_id = (
            "forest_clearing"  # In a full game, this might be configurable
        )
        self.game_state.initialize_new_game(player_name, starting_location_id)

        self.console.print(
            f"\n[bold green]Welcome to the adventure, {player_name}![/bold green]\n"
        )

    def _create_demo_world(self) -> None:
        """Create a demo world for testing purposes."""
        # Create the world state
        world = WorldState()

        # Create a demo location
        forest_clearing = create_demo_location()

        # Add some additional connected locations
        dark_forest = Location(
            id="dark_forest",
            name="Dark Forest",
            description=(
                "Tall trees block out most of the sunlight, creating an eerie "
                "atmosphere. The forest floor is covered in moss and fallen leaves."
            ),
        )
        dark_forest.add_connection("south", "forest_clearing")

        river_bank = Location(
            id="river_bank",
            name="River Bank",
            description=(
                "A clear river flows gently past. The water looks refreshing, and "
                "you can see fish swimming beneath the surface."
            ),
        )
        river_bank.add_connection("west", "forest_clearing")

        forest_path = Location(
            id="forest_path",
            name="Forest Path",
            description=(
                "A narrow path winds through the forest. Bird songs fill the air, "
                "and occasional sunbeams break through the canopy."
            ),
        )
        forest_path.add_connection("north", "forest_clearing")

        ancient_ruins = Location(
            id="ancient_ruins",
            name="Ancient Ruins",
            description="Stone pillars and crumbling walls rise from the forest floor. "
            "This place must have been magnificent centuries ago.",
        )
        ancient_ruins.add_connection("east", "forest_clearing")

        # Create items
        from game_loop.core.state import Item

        # Create a rusty sword item
        rusty_sword = Item(
            id="rusty_sword",
            name="Rusty Sword",
            description="An old sword with a worn handle and a rusty blade. "
            "Despite its condition, it still looks functional.",
            is_container=False,
        )
        world.add_item(rusty_sword)

        # Create a leather pouch (container)
        leather_pouch = Item(
            id="leather_pouch",
            name="Leather Pouch",
            description="A small leather pouch with a drawstring closure. "
            "It could hold small items.",
            is_container=True,
        )
        world.add_item(leather_pouch)

        # Add items to the forest clearing location
        forest_clearing.items = ["rusty_sword", "leather_pouch"]

        # Add the locations to the world
        world.add_location(forest_clearing)
        world.add_location(dark_forest)
        world.add_location(river_bank)
        world.add_location(forest_path)
        world.add_location(ancient_ruins)

        # Set the world state
        self.game_state.world = world

    def _get_player_name(self) -> str:
        """Get the player's name via console input."""
        self.console.print("\n[bold]What is your name, adventurer?[/bold]")
        return input("> ").strip() or "Adventurer"

    async def start(self) -> None:
        """Start the main game loop."""
        self.running = True

        # Display initial location
        self._display_current_location()

        # Main game loop
        while self.running:
            await self._process_input_async()

    def stop(self) -> None:
        """Stop the game loop."""
        self.console.print("[bold]Farewell, adventurer! Your journey ends here.[/bold]")
        self.running = False

    def _display_current_location(self) -> None:
        """Display the current location to the player."""
        location = self.game_state.get_current_location()
        if location:
            self.location_display.display_location(location)
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

        # Add current location information
        location = self.game_state.get_current_location()
        if location:
            context["current_location"] = {
                "id": location.id,
                "name": location.name,
                "description": location.description,
            }

            # Add connections
            connections = []
            for direction, dest_id in location.connections.items():
                if dest_location := self.game_state.world.get_location(dest_id):
                    connections.append(
                        {"direction": direction, "destination": dest_location.name}
                    )
            # Fix: Store the list of connections in the context dictionary
            context["connections"] = {
                f"connection_{i}": f"{conn['direction']} to {conn['destination']}"
                for i, conn in enumerate(connections)
            }

            # Add actual visible objects from the current location
            visible_objects = []
            if hasattr(location, "items") and location.items:
                for item_id in location.items:
                    visible_objects.append(
                        {
                            "name": item_id,
                            "description": f"A {item_id.replace('_', ' ')}.",
                        }
                    )
            # Store visible objects with just the name property
            context["visible_objects"] = {
                f"object_{i}": obj["name"] for i, obj in enumerate(visible_objects)
            }

            # Add actual NPCs from the current location
            npcs = []
            if hasattr(location, "npcs") and location.npcs:
                for npc_id in location.npcs:
                    npcs.append(
                        {
                            "name": npc_id,
                            "description": f"A {npc_id.replace('_', ' ')}.",
                        }
                    )
            # Fix: Store NPCs using just the name property
            context["npcs"] = {f"npc_{i}": npc["name"] for i, npc in enumerate(npcs)}

        # Add player information
        if self.game_state.player:
            # Create player info with name
            player_info = {"name": self.game_state.player.name}
            if self.game_state.player.inventory:
                player_info["inventory"] = ", ".join(self.game_state.player.inventory)
            else:
                player_info["inventory"] = "empty"

            context["player"] = player_info

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
            # If NLP processing fails, fall back to basic pattern matching
            self.console.print("[yellow]Using simplified input processing...[/yellow]")
            try:
                # Create a new basic input processor to avoid any state issues
                # This ensures we don't have any lingering coroutines
                basic_processor = InputProcessor(self.console)
                command = await basic_processor.process_input_async(user_input)
            except Exception as inner_e:
                self.console.print(
                    f"[bold red]Input processing error: " f"{inner_e}[/bold red]"
                )
                return

        # Execute the command if we got one
        if command:
            self._execute_command(command)

    def _execute_command(self, command: ParsedCommand) -> None:
        """
        Execute a processed command.

        Args:
            command: The parsed command to execute
        """
        if command.command_type == CommandType.MOVEMENT:
            if command.subject:
                self._handle_movement(command.subject)
            else:
                self.console.print("[yellow]No direction specified.[/yellow]")

        elif command.command_type == CommandType.LOOK:
            self._display_current_location()

        elif command.command_type == CommandType.INVENTORY:
            self._display_inventory()

        elif command.command_type == CommandType.TAKE:
            if command.subject:
                self._handle_take(command.subject)
            else:
                self.console.print("[yellow]What do you want to take?[/yellow]")

        elif command.command_type == CommandType.DROP:
            if command.subject:
                self._handle_drop(command.subject)
            else:
                self.console.print("[yellow]What do you want to drop?[/yellow]")

        elif command.command_type == CommandType.USE:
            if command.subject:
                self._handle_use(command.subject, command.target)
            else:
                self.console.print("[yellow]What do you want to use?[/yellow]")

        elif command.command_type == CommandType.EXAMINE:
            if command.subject:
                self._handle_examine(command.subject)
            else:
                self.console.print("[yellow]What do you want to examine?[/yellow]")

        elif command.command_type == CommandType.TALK:
            if command.subject:
                self._handle_talk(command.subject)
            else:
                self.console.print("[yellow]Who do you want to talk to?[/yellow]")

        elif command.command_type == CommandType.HELP:
            self._display_help()

        elif command.command_type == CommandType.QUIT:
            self.stop()

        else:
            # Unknown command
            error_message = self.input_processor.format_error_message(command)
            self.console.print(f"[yellow]{error_message}[/yellow]")

    def _handle_movement(self, direction: str) -> None:
        """Handle movement in a given direction."""
        # Normalize direction (e.g., "n" -> "north")
        direction_map = {"n": "north", "s": "south", "e": "east", "w": "west"}
        direction = direction_map.get(direction, direction)

        # Get current location
        current_location = self.game_state.get_current_location()
        if not current_location:
            self.console.print(
                "[bold red]Error: Current location not found.[/bold red]"
            )
            return

        # Check if movement is possible in the given direction
        destination_id = current_location.get_connection(direction)
        if not destination_id:
            self.console.print(f"[yellow]You cannot go {direction} from here.[/yellow]")
            return

        # Update player's location if player exists
        if self.game_state.player is None:
            self.console.print(
                "[bold red]Error: Player state not initialized.[/bold red]"
            )
            return

        self.game_state.player.update_location(destination_id)

        # Output movement result
        self.console.print(f"[green]You go {direction}.[/green]\n")

        # Display new location
        self._display_current_location()

    def _display_inventory(self) -> None:
        """Display the player's inventory."""
        if not self.game_state.player:
            self.console.print(
                "[bold red]Error: Player state not initialized.[/bold red]"
            )
            return

        inventory = self.game_state.player.inventory

        self.console.print("[bold]Inventory:[/bold]")
        if not inventory:
            self.console.print("Your inventory is empty.")
            return

        for item in inventory:
            self.console.print(f"- {item}")

    def _handle_take(self, item_name: str) -> None:
        """
        Handle taking an item from the current location.

        Args:
            item_name: The name of the item to take
        """
        # Normalize item name
        item_name = item_name.replace("the ", "").strip()
        item_id = item_name.lower().replace(" ", "_")

        # Get current location
        current_location = self.game_state.get_current_location()
        if not current_location:
            self.console.print(
                "[bold red]Error: Current location not found.[/bold red]"
            )
            return

        # Check if item exists in current location
        if item_id not in current_location.items:
            # Check if the item is in a container in the current location
            found = False
            for loc_item_id in current_location.items:
                container = self.game_state.world.get_item(loc_item_id)
                if container and container.is_container and container.has_item(item_id):
                    # Remove item from container
                    container.remove_from_container(item_id)
                    found = True
                    break

            if not found:
                self.console.print(f"[yellow]There is no {item_name} here.[/yellow]")
                return
        else:
            # Remove the item from the location
            current_location.remove_item(item_id)

        # Add the item to player's inventory
        if self.game_state.player:
            self.game_state.player.add_to_inventory(item_id)
            self.console.print(f"[green]You take the {item_name}.[/green]")
        else:
            self.console.print(
                "[bold red]Error: Player state not initialized.[/bold red]"
            )

    def _handle_drop(self, item_name: str) -> None:
        """
        Handle dropping an item from inventory to the current location.

        Args:
            item_name: The name of the item to drop
        """
        # Normalize item name
        item_name = item_name.replace("the ", "").strip()
        item_id = item_name.lower().replace(" ", "_")

        # Check if item exists in player's inventory
        if (
            not self.game_state.player
            or item_id not in self.game_state.player.inventory
        ):
            self.console.print(
                f"[yellow]You don't have a {item_name} in your inventory.[/yellow]"
            )
            return

        # Get current location
        current_location = self.game_state.get_current_location()
        if not current_location:
            self.console.print(
                "[bold red]Error: Current location not found.[/bold red]"
            )
            return

        # Remove item from inventory and add to location
        self.game_state.player.remove_from_inventory(item_id)
        current_location.add_item(item_id)

        self.console.print(f"[green]You drop the {item_name}.[/green]")

    def _handle_use(self, item_name: str, target_name: str | None) -> None:
        """
        Handle using an item, possibly on a target.

        Args:
            item_name: The name of the item to use
            target_name: The name of the target to use the item on (if any)
        """
        # Remove potential "the" prefixes from item and target names
        item_name = item_name.replace("the ", "").strip()
        item_id = item_name.lower().replace(" ", "_")

        if target_name:
            target_name = target_name.replace("the ", "").strip()
            target_id = target_name.lower().replace(" ", "_")

            # Special handling for "put X in Y" and similar patterns
            if "in" in target_name:
                # Extract the container name from "in container"
                container_name = target_name.replace("in ", "").strip()
                container_id = target_id

                # Check if item exists in player inventory
                if (
                    not self.game_state.player
                    or item_id not in self.game_state.player.inventory
                ):
                    self.console.print(
                        f"[yellow]You don't have a {item_name} "
                        f"in your inventory.[/yellow]"
                    )
                    return

                # Find where the container is located
                current_location = self.game_state.get_current_location()
                container_location = self.game_state.find_item_location(container_id)

                # Check if container exists and is accessible
                if not container_location or (
                    container_location != "player"
                    and container_location
                    != (current_location.id if current_location is not None else None)
                ):
                    self.console.print(
                        f"[yellow]You don't see any {container_name} here.[/yellow]"
                    )
                    return

                # Check if the container is actually a container
                container = self.game_state.world.get_item(container_id)
                if not container or not container.is_container:
                    self.console.print(
                        f"[yellow]You can't put anything in the "
                        f"{container_name}.[/yellow]"
                    )
                    return

                # Remove item from inventory and add to container
                self.game_state.player.remove_from_inventory(item_id)
                if self.game_state.world.put_item_in_container(item_id, container_id):
                    self.console.print(
                        f"[green]You put the {item_name} "
                        f"in the {container_name}.[/green]"
                    )
                else:
                    # If failed, return item to inventory
                    self.game_state.player.add_to_inventory(item_id)
                    self.console.print(
                        f"[yellow]You couldn't put the {item_name} "
                        f"in the {container_name}.[/yellow]"
                    )
                return
            else:
                self.console.print(
                    f"[yellow]Using the {item_name} on the {target_name} "
                    f"is not implemented yet.[/yellow]"
                )
        else:
            self.console.print(
                f"[yellow]Using the {item_name} is not implemented yet.[/yellow]"
            )

    def _handle_examine(self, object_name: str) -> None:
        """
        Handle examining an object in the current location or inventory.

        Args:
            object_name: The name of the object to examine
        """
        # Normalize object name
        object_name = object_name.replace("the ", "").strip().lower()
        object_id = object_name.replace(" ", "_")

        # Handle "look in" command variation
        looking_inside = False
        if object_name.startswith("in "):
            looking_inside = True
            object_name = object_name[3:].strip()
            object_id = object_name.replace(" ", "_")

        # Find where the object is located
        current_location = self.game_state.get_current_location()
        object_location = self.game_state.find_item_location(object_id)

        # Check if object exists and is accessible
        # Fix: Handle the potential None value by using empty string as default
        player_location_id = ""
        if current_location is not None:  # Fix for line 522 error
            player_location_id = current_location.id

        if not object_location or (
            object_location != "player" and object_location != player_location_id
        ):
            self.console.print(
                f"[yellow]You don't see any {object_name} here.[/yellow]"
            )
            return

        # Get the object
        object_item = self.game_state.world.get_item(object_id)
        if not object_item:
            self.console.print(
                f"[yellow]You don't see any {object_name} here.[/yellow]"
            )
            return

        # If explicitly looking inside or the object is a container
        if looking_inside or object_item.is_container:
            if not object_item.is_container:
                self.console.print(
                    f"[yellow]The {object_name} is not a container.[/yellow]"
                )
                return

            # Display container contents
            contents = self.game_state.world.get_container_contents(object_id)
            if not contents:
                self.console.print(f"[green]The {object_name} is empty.[/green]")
                return

            self.console.print(f"[green]Inside the {object_name}, you find:[/green]")
            for item_id in contents:
                item = self.game_state.world.get_item(item_id)
                if item:
                    item_name = item.name
                else:
                    item_name = item_id.replace("_", " ")
                self.console.print(f"- {item_name}")
            return

        # Regular examination of object
        self.console.print(f"[green]{object_item.description}[/green]")

    def _handle_talk(self, character_name: str) -> None:
        """
        Handle talking to a character in the current location.

        Args:
            character_name: The name of the character to talk to
        """
        # In a full implementation, this would:
        # 1. Check if the character exists in the current location
        # 2. Generate dialogue based on the character's knowledge and state
        # 3. Possibly update the game state based on the conversation

        # For now, just show a placeholder message
        self.console.print(
            f"[yellow]Talking to {character_name} is not implemented yet.[/yellow]"
        )

    def _display_help(self) -> None:
        """Display help information to the player."""
        self.console.print("[bold]Available Commands:[/bold]")
        self.console.print("- [bold]north, n, go north[/bold]: Move north")
        self.console.print("- [bold]south, s, go south[/bold]: Move south")
        self.console.print("- [bold]east, e, go east[/bold]: Move east")
        self.console.print("- [bold]west, w, go west[/bold]: Move west")
        self.console.print("- [bold]look, l[/bold]: Look around")
        self.console.print("- [bold]inventory, i[/bold]: Check your inventory")
        self.console.print("- [bold]take [object][/bold]: Pick up an object")
        self.console.print(
            "- [bold]drop [object][/bold]: Drop an object from your inventory"
        )
        self.console.print("- [bold]use [object][/bold]: Use an object")
        self.console.print(
            "- [bold]use [object] on [target][/bold]: Use an object on a target"
        )
        self.console.print(
            "- [bold]put [object] in [container][/bold]: "
            "Put an object inside a container"
        )
        self.console.print(
            "- [bold]place [object] on [surface][/bold]: Place an object on a surface"
        )
        self.console.print("- [bold]examine [object][/bold]: Examine an object closely")
        self.console.print("- [bold]talk to [character][/bold]: Talk to a character")
        self.console.print("- [bold]help, h, ?[/bold]: Display this help message")
        self.console.print("- [bold]quit, exit, q[/bold]: Quit the game")
