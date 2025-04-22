"""
Core game loop implementation for Game Loop.
Handles the main game loop, input processing, and output generation.
"""

from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand
from game_loop.core.location import LocationDisplay, create_demo_location
from game_loop.core.state import GameState, Location, WorldState


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
        self.input_processor = InputProcessor(self.console)

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

    def start(self) -> None:
        """Start the main game loop."""
        self.running = True

        # Display initial location
        self._display_current_location()

        # Main game loop
        while self.running:
            self._process_input()

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

    def _process_input(self) -> None:
        """Process player input and execute appropriate actions."""
        self.console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
        user_input = input("> ").strip()

        if not user_input:
            self.console.print("[yellow]Please enter a command.[/yellow]")
            return

        # Process the input through the input processor
        command = self.input_processor.process_input(user_input)
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
        # In a full implementation, this would:
        # 1. Check if the item exists in the current location
        # 2. Check if the item can be taken
        # 3. Add the item to the player's inventory
        # 4. Remove the item from the location

        # For now, just show a placeholder message
        self.console.print(
            f"[yellow]Taking the {item_name} is not implemented yet.[/yellow]"
        )

    def _handle_drop(self, item_name: str) -> None:
        """
        Handle dropping an item from inventory to the current location.

        Args:
            item_name: The name of the item to drop
        """
        # In a full implementation, this would:
        # 1. Check if the item exists in the player's inventory
        # 2. Remove the item from the player's inventory
        # 3. Add the item to the current location

        # For now, just show a placeholder message
        self.console.print(
            f"[yellow]Dropping the {item_name} is not implemented yet.[/yellow]"
        )

    def _handle_use(self, item_name: str, target_name: str | None) -> None:
        """
        Handle using an item, possibly on a target.

        Args:
            item_name: The name of the item to use
            target_name: The name of the target to use the item on (if any)
        """
        # In a full implementation, this would:
        # 1. Check if the item exists in the player's inventory
        # 2. Check if the target exists (if provided)
        # 3. Determine the effect of using the item (possibly on the target)
        # 4. Apply the effect

        # For now, just show a placeholder message
        if target_name:
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
        # In a full implementation, this would:
        # 1. Check if the object exists in the current location or inventory
        # 2. Provide a detailed description of the object

        # For now, just show a placeholder message
        self.console.print(
            f"[yellow]Examining the {object_name} is not implemented yet.[/yellow]"
        )

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
        self.console.print("- [bold]examine [object][/bold]: Examine an object closely")
        self.console.print("- [bold]talk to [character][/bold]: Talk to a character")
        self.console.print("- [bold]help, h, ?[/bold]: Display this help message")
        self.console.print("- [bold]quit, exit, q[/bold]: Quit the game")
