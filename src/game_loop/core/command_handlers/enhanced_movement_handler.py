"""
Enhanced movement command handler with navigation safety and tracking.

This handler extends the basic movement system with connectivity validation,
navigation tracking, and landmark-based movement.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console

from game_loop.core.command_handlers.movement_handler import MovementCommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.core.world.connection_manager import WorldConnectionManager
from game_loop.core.world.navigation_tracker import PlayerNavigationTracker
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class EnhancedMovementCommandHandler(MovementCommandHandler):
    """
    Enhanced movement handler with navigation safety and tracking.

    Extends basic movement with:
    - Connection validation to prevent dead ends
    - Navigation tracking for breadcrumbs and landmarks
    - Landmark-based navigation commands
    - Automatic connectivity repair
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        super().__init__(console, state_manager)

        # Initialize navigation systems
        self.connection_manager = WorldConnectionManager(state_manager)
        self.navigation_tracker = PlayerNavigationTracker(state_manager)

        # Add navigation command patterns
        self.navigation_commands = {
            "retrace": ["retrace", "backtrack", "back", "return"],
            "landmark": ["go to", "head to", "navigate to", "find way to"],
        }

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle movement command with enhanced navigation features.

        Args:
            command: The parsed movement command to handle

        Returns:
            ActionResult describing the outcome of the movement
        """
        # Check if this is a special navigation command
        navigation_result = await self._handle_navigation_command(command)
        if navigation_result:
            return navigation_result

        # Proceed with standard movement but with enhanced validation
        return await self._handle_standard_movement(command)

    async def _handle_navigation_command(
        self, command: ParsedCommand
    ) -> ActionResult | None:
        """Handle special navigation commands like 'go to reception' or 'retrace steps'."""
        try:
            command_text = f"{command.action} {command.subject or ''}".strip().lower()

            # Get player state for navigation
            player_state, current_location, world_state = (
                await self.get_required_state()
            )
            if not player_state:
                return None

            player_id = getattr(player_state, "player_id", "default_player")

            # Check for navigation commands
            nav_result = await self.navigation_tracker.handle_navigation_command(
                command_text, player_id
            )

            if nav_result and nav_result.get("success"):
                return self._format_navigation_result(nav_result)

            return None

        except Exception as e:
            logger.error(f"Error handling navigation command: {e}")
            return None

    def _format_navigation_result(self, nav_result: dict[str, Any]) -> ActionResult:
        """Format navigation result into ActionResult."""
        nav_type = nav_result.get("type", "navigation")
        message = nav_result.get("message", "Navigation guidance provided.")
        directions = nav_result.get("directions", [])

        if nav_type == "retrace":
            feedback = f"{message}\n\nDirections: {' → '.join(directions)}"
        elif nav_type == "landmark_navigation":
            target = nav_result.get("target", "destination")
            feedback = f"Path to {target}:\n{' → '.join(directions)}"
        else:
            feedback = message

        return ActionResult(
            success=True,
            feedback_message=feedback,
            metadata={"navigation_type": nav_type, "directions": directions},
        )

    async def _handle_standard_movement(self, command: ParsedCommand) -> ActionResult:
        """Handle standard movement with enhanced validation."""
        # Get required game state
        player_state, current_location, world_state = await self.get_required_state()

        # Basic validation
        if not current_location or not player_state:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location or player state.",
            )

        # Determine movement direction
        direction = self._extract_direction(command)
        if not direction:
            return ActionResult(
                success=False,
                feedback_message="I don't understand that direction. Try: north, south, east, west, up, or down.",
            )

        # Enhanced movement validation
        movement_check = await self._enhanced_movement_validation(
            player_state, current_location, direction
        )
        if not movement_check.success:
            return movement_check

        # Attempt the movement with tracking
        try:
            movement_result = await self._perform_enhanced_movement(
                player_state, current_location, direction
            )
            return movement_result

        except Exception as e:
            logger.error(f"Error during enhanced movement: {e}")
            return ActionResult(
                success=False,
                feedback_message="Something went wrong during movement. Please try again.",
            )

    async def _enhanced_movement_validation(
        self, player_state: "PlayerState", current_location: "Location", direction: str
    ) -> ActionResult:
        """Enhanced movement validation with connectivity checking."""
        try:
            # Perform basic validation first
            basic_validation = await self._validate_movement(
                player_state, current_location, direction
            )
            if not basic_validation.success:
                return basic_validation

            # Check if exit actually leads somewhere valid
            exit_valid = await self._validate_exit_connectivity(
                current_location, direction
            )
            if not exit_valid:
                # Try to provide helpful guidance
                suggestion = await self._suggest_alternative_directions(
                    current_location
                )
                return ActionResult(
                    success=False,
                    feedback_message=f"The path {direction} seems to lead nowhere. {suggestion}",
                )

            return ActionResult(success=True)

        except Exception as e:
            logger.error(f"Error in enhanced movement validation: {e}")
            return ActionResult(success=True)  # Fall back to allowing movement

    async def _validate_exit_connectivity(
        self, location: "Location", direction: str
    ) -> bool:
        """Validate that an exit leads to a reachable location."""
        try:
            # Check if we have connection data
            if not hasattr(location, "connections") or not location.connections:
                # No connection data - assume dynamic generation will handle it
                return True

            if direction not in location.connections:
                # No exit in this direction
                return False

            connection = location.connections[direction]

            # Get destination ID
            destination_id = None
            if hasattr(connection, "to_location_id"):
                destination_id = connection.to_location_id
            elif hasattr(connection, "destination_id"):
                destination_id = connection.destination_id

            if not destination_id:
                return False

            # Check if destination exists and is accessible
            destination = await self.state_manager.get_location_details(destination_id)
            return destination is not None

        except Exception as e:
            logger.error(f"Error validating exit connectivity: {e}")
            return True  # Default to allowing movement

    async def _suggest_alternative_directions(self, location: "Location") -> str:
        """Suggest alternative directions from current location."""
        try:
            if not hasattr(location, "connections") or not location.connections:
                return "Try exploring in different directions to discover new areas."

            available_directions = list(location.connections.keys())
            if available_directions:
                return f"Available directions: {', '.join(available_directions)}"
            else:
                return "No obvious exits are visible. Try looking around for hidden passages."

        except Exception as e:
            logger.error(f"Error suggesting alternatives: {e}")
            return "Try looking around for other paths."

    async def _perform_enhanced_movement(
        self, player_state: "PlayerState", current_location: "Location", direction: str
    ) -> ActionResult:
        """Perform movement with navigation tracking."""
        try:
            # Get destination using parent method
            destination = await self._get_destination(current_location, direction)

            if not destination:
                # Try to trigger dynamic generation
                destination = await self._attempt_dynamic_generation(
                    current_location, direction
                )

                if not destination:
                    return ActionResult(
                        success=False,
                        feedback_message=f"You can't go {direction} from here.",
                    )

            # Ensure bidirectional connection exists
            await self._ensure_bidirectional_connection(
                current_location.location_id, destination.location_id, direction
            )

            # Update player location
            old_location_id = player_state.current_location_id
            player_state.current_location_id = destination.location_id

            # Update state manager
            await self.state_manager.player_tracker.update_state(player_state)

            # Track navigation
            player_id = getattr(player_state, "player_id", "default_player")
            await self.navigation_tracker.track_movement(
                player_id, old_location_id, destination.location_id, direction
            )

            # Create enhanced success result
            success_message = await self._generate_enhanced_movement_feedback(
                direction, destination, old_location_id
            )

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
            logger.error(f"Error performing enhanced movement: {e}")
            return ActionResult(
                success=False, feedback_message="Movement failed. Please try again."
            )

    async def _attempt_dynamic_generation(
        self, current_location: "Location", direction: str
    ) -> Optional["Location"]:
        """Attempt to generate new location if none exists."""
        try:
            # Check if we have boundary manager for dynamic generation
            if hasattr(self.state_manager, "world_boundary_manager"):
                boundary_manager = self.state_manager.world_boundary_manager
                return await boundary_manager.get_or_create_location(
                    current_location.location_id, direction
                )
            return None

        except Exception as e:
            logger.error(f"Error in dynamic generation: {e}")
            return None

    async def _ensure_bidirectional_connection(
        self, from_location_id, to_location_id, direction: str
    ) -> None:
        """Ensure bidirectional connection exists between locations."""
        try:
            # Create bidirectional connection if it doesn't exist
            await self.connection_manager.create_bidirectional_connection(
                from_location_id, to_location_id, direction, f"Path {direction}"
            )

        except Exception as e:
            logger.error(f"Error ensuring bidirectional connection: {e}")

    async def _generate_enhanced_movement_feedback(
        self, direction: str, destination: "Location", old_location_id
    ) -> str:
        """Generate enhanced feedback with navigation context."""
        try:
            feedback = f"You head {direction}."

            # Add destination context
            if hasattr(destination, "name") and destination.name:
                feedback += f"\n\n[bold]{destination.name}[/bold]"

            if (
                hasattr(destination, "short_description")
                and destination.short_description
            ):
                feedback += f"\n{destination.short_description}"

            # Add navigation hints if this is a landmark
            dest_name = getattr(destination, "name", "").lower()
            if any(
                keyword in dest_name for keyword in ["reception", "lobby", "entrance"]
            ):
                feedback += "\n\n[dim]This location has been marked as a landmark for navigation.[/dim]"

            return feedback

        except Exception as e:
            logger.error(f"Error generating enhanced feedback: {e}")
            return f"You head {direction}."

    async def get_navigation_status(self, player_id) -> dict[str, Any]:
        """Get current navigation status for debugging/admin purposes."""
        try:
            # Get connectivity status
            connectivity = await self.connection_manager.validate_world_connectivity()

            # Get navigation summary
            nav_summary = self.navigation_tracker.get_navigation_summary()

            # Get connection statistics
            conn_stats = await self.connection_manager.get_connection_statistics()

            return {
                "connectivity": connectivity,
                "navigation_summary": nav_summary,
                "connection_statistics": conn_stats,
            }

        except Exception as e:
            logger.error(f"Error getting navigation status: {e}")
            return {"error": str(e)}
