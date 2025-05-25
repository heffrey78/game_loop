"""
Target usage handler implementation.
"""

from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState

from .base import UsageHandler


class TargetUsageHandler(UsageHandler):
    """
    Handler for "use X on Y" usage scenarios.
    Handles using an item on another object or NPC.
    """

    async def validate(
        self,
        item_to_use: InventoryItem,
        player_state: PlayerState,
        current_location: Location,
    ) -> bool:
        """Validate if the target usage is possible."""
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
        """Handle using an item on another object or NPC."""
        if not target_name:
            return ActionResult(
                success=False,
                feedback_message="What do you want to use this on?",
            )

        normalized_target_name = target_name.lower()

        # This is a placeholder for future implementation
        # In a real implementation, we would check for specific item-target interactions
        return ActionResult(
            success=False,
            feedback_message=f"Using the {item_to_use.name} on the "
            f"{normalized_target_name} is not implemented yet.",
        )
