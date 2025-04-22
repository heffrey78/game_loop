"""
Core game loop implementation for Game Loop.
Handles the main game loop, input processing, and output generation.
"""

from rich.console import Console

from game_loop.config.models import GameConfig
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
        user_input = input("> ").strip().lower()

        # Process quit command
        if user_input in ["quit", "exit", "q"]:
            self.stop()
            return

        # Process movement commands
        if user_input in ["north", "south", "east", "west", "n", "s", "e", "w"]:
            self._handle_movement(user_input)
            return

        # Process look command
        if user_input in ["look", "l"]:
            self._display_current_location()
            return

        # Process help command
        if user_input in ["help", "h", "?"]:
            self._display_help()
            return

        # Default response for unrecognized commands
        # In a full implementation, this would use NLP to understand the intent
        self.console.print(
            "[yellow]I'm not sure what you mean. Type 'help' for a list of "
            "commands.[/yellow]"
        )

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

    def _display_help(self) -> None:
        """Display help information to the player."""
        self.console.print("[bold]Available Commands:[/bold]")
        self.console.print("- [bold]north, n[/bold]: Move north")
        self.console.print("- [bold]south, s[/bold]: Move south")
        self.console.print("- [bold]east, e[/bold]: Move east")
        self.console.print("- [bold]west, w[/bold]: Move west")
        self.console.print("- [bold]look, l[/bold]: Look around")
        self.console.print("- [bold]help, h, ?[/bold]: Display this help message")
        self.console.print("- [bold]quit, exit, q[/bold]: Quit the game")
