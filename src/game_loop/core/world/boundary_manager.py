"""
World boundary manager for detecting world edges and expansion points.

This module manages world boundaries and identifies suitable locations for expansion.
"""

import logging
from typing import Any
from uuid import UUID

from ...state.models import Location, WorldState
from ..models.navigation_models import BoundaryType, ExpansionPoint

logger = logging.getLogger(__name__)


class WorldBoundaryManager:
    """Manages world boundaries and identifies expansion points."""

    def __init__(self, world_state: WorldState):
        self.world_state = world_state
        self._boundary_cache: dict[UUID, BoundaryType] = {}

    async def detect_boundaries(self) -> dict[UUID, BoundaryType]:
        """Detect and classify all world boundaries."""
        boundaries = {}

        for location_id, location in self.world_state.locations.items():
            boundary_type = await self._classify_location_boundary(location)
            boundaries[location_id] = boundary_type
            self._boundary_cache[location_id] = boundary_type

        return boundaries

    async def _classify_location_boundary(self, location: Location) -> BoundaryType:
        """Classify a single location's boundary type."""
        if not location.connections:
            return BoundaryType.ISOLATED

        # Check connection counts in cardinal directions
        connected_directions = set(location.connections.keys())
        cardinal_directions = {"north", "south", "east", "west"}
        missing_cardinal = cardinal_directions - connected_directions

        if len(missing_cardinal) >= 3:
            return BoundaryType.EDGE
        elif len(missing_cardinal) >= 2:
            return BoundaryType.FRONTIER
        else:
            return BoundaryType.INTERNAL

    async def find_expansion_points(self) -> list[ExpansionPoint]:
        """Find suitable points for world expansion."""
        expansion_points = []
        boundaries = await self.detect_boundaries()

        for location_id, boundary_type in boundaries.items():
            if boundary_type in [BoundaryType.EDGE, BoundaryType.FRONTIER]:
                location = self.world_state.locations[location_id]
                missing_connections = self._get_missing_connections(location)

                for direction in missing_connections:
                    expansion_point = ExpansionPoint(
                        location_id=location_id,
                        direction=direction,
                        priority=self._calculate_expansion_priority(
                            location, direction
                        ),
                        context=self._gather_expansion_context(location),
                    )
                    expansion_points.append(expansion_point)

        return sorted(expansion_points, key=lambda x: x.priority, reverse=True)

    def _get_missing_connections(self, location: Location) -> list[str]:
        """Get directions without connections."""
        all_directions = {"north", "south", "east", "west"}
        return list(all_directions - set(location.connections.keys()))

    def _calculate_expansion_priority(
        self, location: Location, direction: str
    ) -> float:
        """Calculate priority for expanding in a given direction."""
        # Higher priority for locations with more player visits
        visit_score = location.state_flags.get("visit_count", 0) * 0.3

        # Higher priority for locations with fewer existing connections
        connection_score = (4 - len(location.connections)) * 0.2

        # Higher priority for cardinal directions
        direction_score = (
            0.5 if direction in ["north", "south", "east", "west"] else 0.3
        )

        return visit_score + connection_score + direction_score

    def _gather_expansion_context(self, location: Location) -> dict[str, Any]:
        """Gather context for expansion generation."""
        return {
            "location_name": location.name,
            "location_type": location.state_flags.get("type", "generic"),
            "themes": location.state_flags.get("themes", []),
            "description": location.description,
            "existing_connections": list(location.connections.keys()),
        }

    def get_boundary_type(self, location_id: UUID) -> BoundaryType | None:
        """Get cached boundary type for a location."""
        return self._boundary_cache.get(location_id)

    def clear_cache(self) -> None:
        """Clear the boundary cache."""
        self._boundary_cache.clear()

    async def update_boundary_for_location(self, location_id: UUID) -> BoundaryType:
        """Update boundary classification for a specific location."""
        if location_id in self.world_state.locations:
            location = self.world_state.locations[location_id]
            boundary_type = await self._classify_location_boundary(location)
            self._boundary_cache[location_id] = boundary_type
            return boundary_type
        else:
            logger.warning(f"Location {location_id} not found in world state")
            return BoundaryType.ISOLATED

    def get_expansion_candidates(
        self, max_candidates: int = 10
    ) -> list[ExpansionPoint]:
        """Get top expansion candidates synchronously."""
        try:
            # Use the synchronous implementation to avoid event loop issues
            expansion_points = []
            for location_id, location in self.world_state.locations.items():
                missing_connections = self._get_missing_connections(location)
                for direction in missing_connections:
                    expansion_point = ExpansionPoint(
                        location_id=location_id,
                        direction=direction,
                        priority=self._calculate_expansion_priority(location, direction),
                        context=self._gather_expansion_context(location),
                    )
                    expansion_points.append(expansion_point)

            return sorted(expansion_points, key=lambda x: x.priority, reverse=True)[
                :max_candidates
            ]
        except Exception as e:
            logger.error(f"Error getting expansion candidates: {e}")
            return []
