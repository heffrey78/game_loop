"""
Game state management for Game Loop.
Handles tracking player state, world state, and game session information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class PlayerState:
    """Represents the state of the player in the game."""

    name: str
    current_location_id: str
    inventory: list[str] = field(default_factory=list)
    visited_locations: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    quests: dict[str, dict] = field(default_factory=dict)

    def add_to_inventory(self, item_id: str) -> None:
        """Add an item to the player's inventory."""
        if item_id not in self.inventory:
            self.inventory.append(item_id)

    def remove_from_inventory(self, item_id: str) -> bool:
        """Remove an item from the player's inventory."""
        if item_id in self.inventory:
            self.inventory.remove(item_id)
            return True
        return False

    def update_location(self, location_id: str) -> None:
        """Update the player's current location."""
        self.current_location_id = location_id
        if location_id not in self.visited_locations:
            self.visited_locations.append(location_id)


@dataclass
class Location:
    """Represents a location in the game world."""

    id: str
    name: str
    description: str
    connections: dict[str, str] = field(
        default_factory=dict
    )  # direction -> location_id
    items: list[str] = field(default_factory=list)
    npcs: list[str] = field(default_factory=list)
    visited: bool = False

    def add_connection(self, direction: str, location_id: str) -> None:
        """Add a connection to another location."""
        self.connections[direction.lower()] = location_id

    def get_connection(self, direction: str) -> str | None:
        """Get the location ID connected in the specified direction."""
        return self.connections.get(direction.lower())


@dataclass
class WorldState:
    """Represents the state of the game world."""

    locations: dict[str, Location] = field(default_factory=dict)
    current_time: datetime = field(default_factory=datetime.now)

    def add_location(self, location: Location) -> None:
        """Add a location to the world state."""
        self.locations[location.id] = location

    def get_location(self, location_id: str) -> Location | None:
        """Get a location by its ID."""
        return self.locations.get(location_id)

    def mark_location_visited(self, location_id: str) -> None:
        """Mark a location as visited."""
        if location_id in self.locations:
            self.locations[location_id].visited = True


@dataclass
class GameState:
    """Main container for game state."""

    session_id: UUID = field(default_factory=uuid4)
    player: PlayerState | None = None
    world: WorldState = field(default_factory=WorldState)
    start_time: datetime = field(default_factory=datetime.now)

    def initialize_new_game(self, player_name: str, starting_location_id: str) -> None:
        """Initialize a new game with the specified player name and starting
        location."""
        self.player = PlayerState(
            name=player_name, current_location_id=starting_location_id
        )
        self.start_time = datetime.now()

        # Mark the starting location as visited
        if starting_location_id in self.world.locations:
            self.world.mark_location_visited(starting_location_id)
            self.player.visited_locations.append(starting_location_id)

    def get_current_location(self) -> Location | None:
        """Get the player's current location."""
        if not self.player:
            return None
        return self.world.get_location(self.player.current_location_id)
