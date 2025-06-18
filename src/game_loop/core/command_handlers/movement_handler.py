"""
Movement command handler for the Game Loop.
Handles player movement between locations.
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class MovementCommandHandler(CommandHandler):
    """
    Handler for movement commands in the Game Loop.

    Handles player movement between locations with validation,
    stamina checking, and rich feedback.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        """
        Initialize the movement handler.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
        """
        super().__init__(console, state_manager)

        # Direction mappings for movement
        self.direction_mappings = {
            "north": ["north", "n", "go north"],
            "south": ["south", "s", "go south"],
            "east": ["east", "e", "go east"],
            "west": ["west", "w", "go west"],
            "up": ["up", "u", "go up"],
            "down": ["down", "d", "go down"],
        }

        # Create reverse mapping for quick lookup
        self.command_to_direction = {}
        for direction, commands in self.direction_mappings.items():
            for command in commands:
                self.command_to_direction[command.lower()] = direction

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle a movement command and return the result.

        Args:
            command: The parsed movement command to handle

        Returns:
            ActionResult describing the outcome of the movement
        """
        # Get required game state
        player_state, current_location, world_state = await self.get_required_state()

        # Basic validation
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        if not player_state:
            return ActionResult(
                success=False, feedback_message="Error: Cannot access player state."
            )

        # Determine movement direction
        direction = self._extract_direction(command)
        if not direction:
            return ActionResult(
                success=False,
                feedback_message="I don't understand that direction. Try: north, south, east, west, up, or down.",
            )

        # Check if movement is possible
        movement_check = await self._validate_movement(
            player_state, current_location, direction
        )
        if not movement_check.success:
            return movement_check

        # Attempt the movement
        try:
            movement_result = await self._perform_movement(
                player_state, current_location, direction
            )
            return movement_result

        except Exception as e:
            logger.error(f"Error during movement: {e}")
            return ActionResult(
                success=False,
                feedback_message="Something went wrong during movement. Please try again.",
            )

    def _extract_direction(self, command: ParsedCommand) -> str | None:
        """
        Extract the movement direction from the parsed command.

        Args:
            command: The parsed command

        Returns:
            The normalized direction string, or None if not found
        """
        # Check the main action first
        if command.action.lower() in self.command_to_direction:
            return self.command_to_direction[command.action.lower()]

        # Check subject as fallback
        if command.subject and command.subject.lower() in self.command_to_direction:
            return self.command_to_direction[command.subject.lower()]

        return None

    async def _validate_movement(
        self, player_state: "PlayerState", current_location: "Location", direction: str
    ) -> ActionResult:
        """
        Validate if the movement is allowed.

        Args:
            player_state: Current player state
            current_location: Current location
            direction: Direction to move

        Returns:
            ActionResult indicating if movement is valid
        """
        # Check if the direction exists from current location
        if (
            not hasattr(current_location, "connections")
            or not current_location.connections
        ):
            # If no connections data, check if we can create/explore in this direction
            return ActionResult(
                success=True, feedback_message=f"Exploring {direction}..."
            )

        # Check for blocked paths or requirements
        if direction in current_location.connections:
            connection = current_location.connections[direction]

            # Check if connection has requirements
            if hasattr(connection, "requirements") and connection.requirements:
                # For now, just check basic requirements
                # In future, this could check for keys, stats, etc.
                pass

        # Check player stamina/health if applicable
        if hasattr(player_state, "health") and player_state.health <= 0:
            return ActionResult(
                success=False,
                feedback_message="You are too weak to move. Rest or find healing first.",
            )

        return ActionResult(success=True)

    async def _perform_movement(
        self, player_state: "PlayerState", current_location: "Location", direction: str
    ) -> ActionResult:
        """
        Perform the actual movement operation.

        Args:
            player_state: Current player state
            current_location: Current location
            direction: Direction to move

        Returns:
            ActionResult describing the movement outcome
        """
        try:
            # Get the destination location
            destination = await self._get_destination(current_location, direction)

            if not destination:
                return ActionResult(
                    success=False,
                    feedback_message=f"You can't go {direction} from here.",
                )

            # Update player location
            old_location_id = player_state.current_location_id
            player_state.current_location_id = destination.location_id

            # Update state manager
            await self.state_manager.player_tracker.update_state(player_state)

            # Create success result with location change
            success_message = f"You head {direction}."

            # Add atmospheric description if available
            if (
                hasattr(destination, "short_description")
                and destination.short_description
            ):
                success_message += f"\n\n{destination.short_description}"

            return ActionResult(
                success=True,
                feedback_message=success_message,
                location_change=True,
                location_data={
                    "old_location_id": old_location_id,
                    "new_location_id": destination.location_id,
                    "direction": direction,
                },
            )

        except Exception as e:
            logger.error(f"Error performing movement: {e}")
            return ActionResult(
                success=False, feedback_message="Movement failed. Please try again."
            )

    async def _get_destination(
        self, current_location: "Location", direction: str
    ) -> "Location | None":
        """
        Get the destination location for the movement.

        Args:
            current_location: Current location
            direction: Direction to move

        Returns:
            The destination Location object, or None if not accessible
        """
        try:
            # Check if connection exists
            if (
                hasattr(current_location, "connections")
                and current_location.connections
            ):
                if direction in current_location.connections:
                    connection = current_location.connections[direction]

                    # Get the destination location ID
                    destination_id = None
                    if hasattr(connection, "to_location_id"):
                        destination_id = connection.to_location_id
                    elif hasattr(connection, "destination_id"):
                        destination_id = connection.destination_id

                    if destination_id:
                        # Get the destination location details
                        return await self.state_manager.get_location_details(
                            destination_id
                        )

            # If no existing connection, try to generate new location
            # This integrates with the dynamic world generation system

            # Check if we have a boundary manager available
            if hasattr(self.state_manager, "world_boundary_manager"):
                boundary_manager = self.state_manager.world_boundary_manager

                # Try to get or create a location in the specified direction
                new_location = await boundary_manager.get_or_create_location(
                    current_location.location_id, direction
                )

                if new_location:
                    return new_location

            return None

        except Exception as e:
            logger.error(f"Error getting destination: {e}")
            return None

    def _generate_movement_feedback(
        self, direction: str, destination: "Location"
    ) -> str:
        """
        Generate rich feedback for successful movement.

        Args:
            direction: Direction moved
            destination: Destination location

        Returns:
            Formatted feedback message
        """
        feedback = f"You head {direction}."

        # Add destination context if available
        if hasattr(destination, "name") and destination.name:
            feedback += f"\n\n[bold]{destination.name}[/bold]"

        if hasattr(destination, "short_description") and destination.short_description:
            feedback += f"\n{destination.short_description}"

        return feedback
