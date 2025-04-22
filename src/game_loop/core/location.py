"""
Location handling for Game Loop.
Manages location display, description generation, and navigation between locations.
"""

from rich.console import Console
from rich.panel import Panel

from game_loop.core.state import Location


class LocationDisplay:
    """Handles displaying location information to the player."""

    def __init__(self, console: Console):
        """Initialize the LocationDisplay with a Rich console."""
        self.console = console

    def display_location(self, location: Location) -> None:
        """Display the current location to the player."""
        # Create a panel with the location name and description
        location_panel = Panel(
            f"{location.description}\n\n"
            f"[bold cyan]Exits:[/bold cyan] {self._format_exits(location)}",
            title=f"[bold yellow]{location.name}[/bold yellow]",
            border_style="bright_blue",
            expand=False,
        )

        self.console.print(location_panel)

        # Display items in the location, if any
        if location.items:
            self.console.print("[bold green]You see:[/bold green]")
            for item_id in location.items:
                # In a full implementation, we'd look up item details from a database
                # or state
                self.console.print(f"- {item_id}")
            self.console.print()

        # Display NPCs in the location, if any
        if location.npcs:
            self.console.print("[bold magenta]Characters present:[/bold magenta]")
            for npc_id in location.npcs:
                # In a full implementation, we'd look up NPC details from a database
                # or state
                self.console.print(f"- {npc_id}")
            self.console.print()

    def _format_exits(self, location: Location) -> str:
        """Format the list of exits from a location."""
        if not location.connections:
            return "None"

        return ", ".join(
            f"[bold white]{direction}[/bold white]"
            for direction in location.connections.keys()
        )


class NavigationManager:
    """Manages movement between locations."""

    @staticmethod
    def validate_movement(location: Location, direction: str) -> str | None:
        """
        Validate if movement in the specified direction is possible.

        Returns:
            The destination location ID if movement is valid, None otherwise.
        """
        return location.get_connection(direction.lower())

    @staticmethod
    def get_available_directions(location: Location) -> list[str]:
        """Get all available directions from the current location."""
        return list(location.connections.keys())

    @staticmethod
    def add_connection(
        source: Location,
        direction: str,
        target_id: str,
        bidirectional: bool = True,
        return_direction: str | None = None,
    ) -> None:
        """
        Add a connection between locations.

        Args:
            source: Source location
            direction: Direction from source to target
            target_id: Target location ID
            bidirectional: If True, add a connection back from target to source
            return_direction: Direction from target to source (if bidirectional)
                             Defaults to opposite direction if None
        """
        source.add_connection(direction, target_id)

        # If bidirectional, we'll need to get the target location and add a
        # connection back
        # In a complete implementation, this would be done using a WorldState instance


def create_demo_location() -> Location:
    """
    Create a demo location for testing purposes.

    Returns:
        A demo location object with sample data.
    """
    location = Location(
        id="forest_clearing",
        name="Forest Clearing",
        description=(
            "You stand in a peaceful clearing in the heart of the forest. "
            "Sunlight filters through the canopy, illuminating a carpet of "
            "wildflowers. A gentle breeze carries the scent of pine and fresh earth."
        ),
    )

    # Add sample connections
    location.add_connection("north", "dark_forest")
    location.add_connection("east", "river_bank")
    location.add_connection("south", "forest_path")
    location.add_connection("west", "ancient_ruins")

    # Add sample items and NPCs
    location.items = ["rusty_sword", "leather_pouch"]
    location.npcs = ["forest_ranger"]

    return location
