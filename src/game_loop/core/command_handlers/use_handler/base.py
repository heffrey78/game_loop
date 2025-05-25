"""
Base interface for usage handlers.
"""

from abc import ABC, abstractmethod

from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState


class UsageHandler(ABC):
    """Base class for all item usage handlers."""

    @abstractmethod
    async def validate(
        self,
        item_to_use: InventoryItem,
        player_state: PlayerState,
        current_location: Location,
    ) -> bool:
        """
        Validate if this usage is possible.

        Args:
            item_to_use: The inventory item to be used
            player_state: The current player state
            current_location: The current location

        Returns:
            True if the usage is valid, False otherwise
        """
        pass

    @abstractmethod
    async def handle(
        self,
        item_to_use: InventoryItem,
        target_name: str | None,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """
        Handle the item usage and return the result.

        Args:
            item_to_use: The inventory item to be used
            target_name: The name of the target object (if any)
            player_state: The current player state
            current_location: The current location

        Returns:
            ActionResult describing the outcome of the action
        """
        pass
