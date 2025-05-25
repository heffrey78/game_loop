"""
Game state management for Game Loop.
Handles tracking player state, world state, and game session information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Item:
    """Represents an item in the game world."""

    id: str
    name: str
    description: str
    is_container: bool = False
    contents: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def add_to_container(self, item_id: str) -> bool:
        """Add an item to this container."""
        if not self.is_container:
            return False
        if item_id not in self.contents:
            self.contents.append(item_id)
        return True

    def remove_from_container(self, item_id: str) -> bool:
        """Remove an item from this container."""
        if not self.is_container or item_id not in self.contents:
            return False
        self.contents.remove(item_id)
        return True

    def has_item(self, item_id: str) -> bool:
        """Check if the container has a specific item."""
        return item_id in self.contents if self.is_container else False


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

    def add_item(self, item_id: str) -> None:
        """Add an item to the location."""
        if item_id not in self.items:
            self.items.append(item_id)

    def add_npc(self, npc_id: str) -> None:
        """Add an NPC to the location."""
        if npc_id not in self.npcs:
            self.npcs.append(npc_id)

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the location."""
        if item_id in self.items:
            self.items.remove(item_id)
            return True
        return False


@dataclass
class WorldState:
    """Represents the state of the game world."""

    locations: dict[str, Location] = field(default_factory=dict)
    items: dict[str, Item] = field(default_factory=dict)
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

    def add_item(self, item: Item) -> None:
        """Add an item to the world state."""
        self.items[item.id] = item

    def get_item(self, item_id: str) -> Item | None:
        """Get an item by its ID."""
        return self.items.get(item_id)

    def put_item_in_container(self, item_id: str, container_id: str) -> bool:
        """Put an item inside a container."""
        container = self.get_item(container_id)
        if not container or not container.is_container:
            return False

        return container.add_to_container(item_id)

    def get_container_contents(self, container_id: str) -> list[str]:
        """Get the contents of a container."""
        container = self.get_item(container_id)
        if not container or not container.is_container:
            return []

        return container.contents


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

    def find_item_location(self, item_id: str) -> str | None:
        """
        Find where an item is currently located.
        Returns:
        - "player" if in player inventory
        - container_id if in a container
        - location_id if in a location
        - None if not found
        """
        # Check player inventory
        if self.player and item_id in self.player.inventory:
            return "player"

        # Check all containers
        for container_id, item in self.world.items.items():
            if item.is_container and item_id in item.contents:
                return container_id

        # Check all locations
        for location_id, location in self.world.locations.items():
            if item_id in location.items:
                return location_id

        return None

    def get_items_in_view(self, exclude_contained: bool = True) -> list[str]:
        """
        Get all items visible in the current location.

        Args:
            exclude_contained: If True, don't include items inside containers

        Returns:
            List of item IDs visible in the current location
        """
        location = self.get_current_location()
        if not location:
            return []

        items = location.items.copy()

        if not exclude_contained:
            # Add items in visible containers
            for item_id in location.items:
                item = self.world.get_item(item_id)
                if item and item.is_container:
                    items.extend(item.contents)

        return items
