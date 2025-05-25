"""
Self usage handler implementation.
"""

from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState

from .base import UsageHandler


class SelfUsageHandler(UsageHandler):
    """
    Handler for self-use scenarios.
    Handles using an item by itself without a target.
    """

    async def validate(
        self,
        item_to_use: InventoryItem,
        player_state: PlayerState,
        current_location: Location,
    ) -> bool:
        """Validate if the self usage is possible."""
        # Validation would be moved here
        # For now, just return True as validation is done in handle
        return True

    async def handle(
        self,
        item_to_use: InventoryItem,
        target_name: str | None,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle using an item on its own (e.g., consuming a potion)."""
        # This is a placeholder for future implementation
        # In a real implementation, we would check for self-use effects
        return ActionResult(
            success=False,
            feedback_message=f"Using the {item_to_use.name} is not implemented yet.",
        )
