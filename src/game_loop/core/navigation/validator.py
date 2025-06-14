"""
Navigation validator for movement validation and rules enforcement.

This module validates player movements and enforces navigation rules.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from ...state.models import Location, PlayerState
from ..models.navigation_models import NavigationError, NavigationResult
from ..world.connection_graph import LocationConnectionGraph

logger = logging.getLogger(__name__)


class NavigationValidator:
    """Validates navigation actions and movements."""

    def __init__(self, connection_graph: LocationConnectionGraph):
        self.connection_graph = connection_graph

    async def validate_movement(
        self,
        player_state: PlayerState,
        from_location: Location,
        to_location_id: UUID,
        direction: str,
    ) -> NavigationResult:
        """Validate if a movement is allowed."""
        # Check if connection exists
        if not self.connection_graph.has_connection(
            from_location.location_id, to_location_id
        ):
            return NavigationResult(
                success=False,
                error=NavigationError.NO_CONNECTION,
                message=f"No connection exists to the {direction}.",
            )

        # Get connection info
        connection_info = self.connection_graph.get_connection_info(
            from_location.location_id, to_location_id
        )

        # Check requirements
        if connection_info and connection_info.requirements:
            validation_result = await self._check_requirements(
                player_state, connection_info.requirements
            )
            if not validation_result.success:
                return validation_result

        # Check if the connection is blocked
        if from_location.state_flags.get(f"blocked_{direction}", False):
            return NavigationResult(
                success=False,
                error=NavigationError.BLOCKED,
                message=f"The way {direction} is blocked.",
            )

        return NavigationResult(success=True, message=f"You can go {direction}.")

    async def _check_requirements(
        self, player_state: PlayerState, requirements: dict[str, Any]
    ) -> NavigationResult:
        """Check if player meets movement requirements."""
        # Check item requirements
        if "required_items" in requirements:
            for item_name in requirements["required_items"]:
                if not any(item.name == item_name for item in player_state.inventory):
                    return NavigationResult(
                        success=False,
                        error=NavigationError.MISSING_REQUIREMENT,
                        message=f"You need a {item_name} to go this way.",
                    )

        # Check skill requirements
        if "required_skills" in requirements:
            # For now, we'll use a simple skill system based on player stats
            # In a real implementation, we might add a skills dict to PlayerStats
            player_skills = {}
            if player_state.stats:
                # Map stats to skills for this example
                player_skills = {
                    "climbing": player_state.stats.strength,
                    "swimming": player_state.stats.dexterity,
                    "magic": player_state.stats.intelligence,
                }
            for skill, min_level in requirements["required_skills"].items():
                if player_skills.get(skill, 0) < min_level:
                    return NavigationResult(
                        success=False,
                        error=NavigationError.INSUFFICIENT_SKILL,
                        message=f"Your {skill} skill is too low.",
                    )

        # Check state requirements
        if "required_state" in requirements:
            # For now, we'll use player progress flags for state requirements
            # In a real implementation, we might add a state dict to PlayerState
            player_flags = player_state.progress.flags if player_state.progress else {}
            for key, value in requirements["required_state"].items():
                if player_flags.get(key) != value:
                    return NavigationResult(
                        success=False,
                        error=NavigationError.INVALID_STATE,
                        message="You're not in the right state for this action.",
                    )

        return NavigationResult(success=True, message="Requirements met.")

    def get_valid_directions(
        self, location: Location, player_state: PlayerState | None = None
    ) -> dict[str, bool]:
        """Get all directions and their validity from a location."""
        valid_directions = {}

        for direction, destination_id in location.connections.items():
            if player_state:
                # Full validation with player state
                try:
                    result = asyncio.run(
                        self.validate_movement(
                            player_state, location, destination_id, direction
                        )
                    )
                    valid_directions[direction] = result.success
                except Exception as e:
                    logger.error(f"Error validating direction {direction}: {e}")
                    valid_directions[direction] = False
            else:
                # Simple check for connection existence
                valid_directions[direction] = not location.state_flags.get(
                    f"blocked_{direction}", False
                )

        return valid_directions

    def can_player_access_location(
        self, player_state: PlayerState, location: Location
    ) -> bool:
        """Check if player can access a location based on any requirements."""
        # Check if location has access requirements
        location_requirements = location.state_flags.get("access_requirements", {})
        if not location_requirements:
            return True

        try:
            result = asyncio.run(
                self._check_requirements(player_state, location_requirements)
            )
            return result.success
        except Exception as e:
            logger.error(f"Error checking location access: {e}")
            return False

    def validate_connection_exists(
        self, from_location_id: UUID, to_location_id: UUID
    ) -> NavigationResult:
        """Validate that a connection exists between two locations."""
        if self.connection_graph.has_connection(from_location_id, to_location_id):
            return NavigationResult(success=True, message="Connection exists.")
        else:
            return NavigationResult(
                success=False,
                error=NavigationError.NO_CONNECTION,
                message="No connection exists between these locations.",
            )

    def get_blocked_directions(self, location: Location) -> list[str]:
        """Get all blocked directions from a location."""
        blocked = []
        for direction in location.connections.keys():
            if location.state_flags.get(f"blocked_{direction}", False):
                blocked.append(direction)
        return blocked

    def set_direction_blocked(
        self, location: Location, direction: str, blocked: bool = True
    ) -> None:
        """Set a direction as blocked or unblocked."""
        location.state_flags[f"blocked_{direction}"] = blocked
