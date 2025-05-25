"""
UseHandler implementation for handling item usage commands.
"""

from typing import TYPE_CHECKING

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.command_handlers.use_handler.factory import UsageHandlerFactory
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    # WorldObject is used as Item in the game
    from game_loop.state.models import WorldObject as Item
    from game_loop.state.player_state import PlayerState


class UseHandler(CommandHandler):
    """
    Handler for USE commands in the Game Loop.

    Handles various item usage scenarios including:
    - Using an item on its own
    - Using an item on another object
    - Putting an item into a container

    This implementation uses the Strategy pattern with a nested factory
    to delegate to specialized handlers for different usage scenarios.
    """

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle a USE command and return the result.

        Args:
            command: The parsed USE command to handle

        Returns:
            ActionResult describing the outcome of the command
        """
        # Get required game state
        player_state, current_location, _ = await self.get_required_state()

        # Basic validation
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        if not command.subject:
            return ActionResult(
                success=False, feedback_message="What do you want to use?"
            )

        # Normalize names
        subject = command.subject if command.subject else ""
        normalized_item_name = self.normalize_name(subject)

        # Find the item in the player's inventory
        item_to_use = self._find_item_in_inventory(
            player_state, normalized_item_name or ""
        )
        if not item_to_use:
            return ActionResult(
                success=False, feedback_message=f"You don't have a {command.subject}."
            )

        # Get the appropriate handler using factory
        usage_factory = UsageHandlerFactory()
        usage_handler = usage_factory.get_handler(command.target)

        # Delegate the handling to the appropriate handler
        if player_state is None:
            raise ValueError("Player state is required for use commands")
        return await usage_handler.handle(
            item_to_use, command.target, player_state, current_location
        )

    def _find_item_in_inventory(
        self, player_state: "PlayerState | None", normalized_item_name: str
    ) -> "Item | None":
        """Find an item in the player's inventory by its normalized name."""
        if not player_state or not player_state.inventory:
            return None

        for item in player_state.inventory:
            if item.name.lower() == normalized_item_name:
                return item

        return None
