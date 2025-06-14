"""
Unit tests for WorldBoundaryManager.
"""

from uuid import uuid4

import pytest

from game_loop.core.models.navigation_models import BoundaryType
from game_loop.core.world.boundary_manager import WorldBoundaryManager
from game_loop.state.models import Location, WorldState


class TestWorldBoundaryManager:
    """Test cases for WorldBoundaryManager."""

    @pytest.mark.asyncio
    async def test_boundary_detection(self):
        """Test boundary type detection."""
        # Create test world
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()
        loc4_id = uuid4()

        locations = {
            loc1_id: Location(
                location_id=loc1_id,
                name="Edge Location",
                description="A location at the edge",
                connections={"north": loc2_id},  # Only one connection
            ),
            loc2_id: Location(
                location_id=loc2_id,
                name="Frontier Location",
                description="A frontier location",
                connections={"south": loc1_id, "north": loc3_id},  # Two connections
            ),
            loc3_id: Location(
                location_id=loc3_id,
                name="Internal Location",
                description="An internal location",
                connections={
                    "south": loc2_id,
                    "north": loc4_id,
                    "east": uuid4(),
                    "west": uuid4(),
                },  # Four connections
            ),
            loc4_id: Location(
                location_id=loc4_id,
                name="Isolated Location",
                description="An isolated location",
                connections={},  # No connections
            ),
        }

        world_state = WorldState(locations=locations)
        manager = WorldBoundaryManager(world_state)

        boundaries = await manager.detect_boundaries()

        assert boundaries[loc1_id] == BoundaryType.EDGE
        assert boundaries[loc2_id] == BoundaryType.FRONTIER
        assert boundaries[loc3_id] == BoundaryType.INTERNAL
        assert boundaries[loc4_id] == BoundaryType.ISOLATED

    @pytest.mark.asyncio
    async def test_expansion_points(self):
        """Test finding expansion points."""
        # Create test world with edge and frontier locations
        loc1_id = uuid4()
        loc2_id = uuid4()

        locations = {
            loc1_id: Location(
                location_id=loc1_id,
                name="Edge Location",
                description="A location at the edge",
                connections={"north": loc2_id},
                state_flags={"visit_count": 5},  # Visited location
            ),
            loc2_id: Location(
                location_id=loc2_id,
                name="Frontier Location",
                description="A frontier location",
                connections={"south": loc1_id},
                state_flags={"visit_count": 2},
            ),
        }

        world_state = WorldState(locations=locations)
        manager = WorldBoundaryManager(world_state)

        expansion_points = await manager.find_expansion_points()

        # Both locations should have expansion points
        assert len(expansion_points) > 0

        # Check that expansion points have required attributes
        for point in expansion_points:
            assert point.location_id in [loc1_id, loc2_id]
            assert point.direction in ["north", "south", "east", "west"]
            assert point.priority >= 0
            assert "location_name" in point.context

    def test_missing_connections(self):
        """Test getting missing connections."""
        location = Location(
            location_id=uuid4(),
            name="Test Location",
            description="A test location",
            connections={"north": uuid4(), "east": uuid4()},
        )

        world_state = WorldState(locations={location.location_id: location})
        manager = WorldBoundaryManager(world_state)

        missing = manager._get_missing_connections(location)

        assert "south" in missing
        assert "west" in missing
        assert "north" not in missing
        assert "east" not in missing

    def test_expansion_priority_calculation(self):
        """Test expansion priority calculation."""
        location = Location(
            location_id=uuid4(),
            name="Test Location",
            description="A test location",
            connections={"north": uuid4()},
            state_flags={"visit_count": 10},
        )

        world_state = WorldState(locations={location.location_id: location})
        manager = WorldBoundaryManager(world_state)

        # Test with cardinal direction (should have higher priority)
        priority_cardinal = manager._calculate_expansion_priority(location, "south")

        # Test with non-cardinal direction
        priority_non_cardinal = manager._calculate_expansion_priority(location, "up")

        assert priority_cardinal > priority_non_cardinal

    def test_expansion_context_gathering(self):
        """Test gathering expansion context."""
        location = Location(
            location_id=uuid4(),
            name="Forest Clearing",
            description="A peaceful clearing",
            connections={"north": uuid4()},
            state_flags={"type": "forest", "themes": ["nature", "peaceful"]},
        )

        world_state = WorldState(locations={location.location_id: location})
        manager = WorldBoundaryManager(world_state)

        context = manager._gather_expansion_context(location)

        assert context["location_name"] == "Forest Clearing"
        assert context["location_type"] == "forest"
        assert context["themes"] == ["nature", "peaceful"]
        assert context["description"] == "A peaceful clearing"
        assert "north" in context["existing_connections"]

    @pytest.mark.asyncio
    async def test_boundary_cache(self):
        """Test boundary caching functionality."""
        loc_id = uuid4()
        location = Location(
            location_id=loc_id,
            name="Test Location",
            description="A test location",
            connections={"north": uuid4()},
        )

        world_state = WorldState(locations={loc_id: location})
        manager = WorldBoundaryManager(world_state)

        # Detect boundaries to populate cache
        await manager.detect_boundaries()

        # Check cache
        cached_type = manager.get_boundary_type(loc_id)
        assert cached_type == BoundaryType.EDGE

        # Clear cache and verify
        manager.clear_cache()
        cached_type_after_clear = manager.get_boundary_type(loc_id)
        assert cached_type_after_clear is None

    @pytest.mark.asyncio
    async def test_update_boundary_for_location(self):
        """Test updating boundary for a specific location."""
        loc_id = uuid4()
        location = Location(
            location_id=loc_id,
            name="Test Location",
            description="A test location",
            connections={"north": uuid4()},
        )

        world_state = WorldState(locations={loc_id: location})
        manager = WorldBoundaryManager(world_state)

        # Update boundary for location
        boundary_type = await manager.update_boundary_for_location(loc_id)
        assert boundary_type == BoundaryType.EDGE

        # Check that it's cached
        cached_type = manager.get_boundary_type(loc_id)
        assert cached_type == BoundaryType.EDGE

    def test_get_expansion_candidates_sync(self):
        """Test getting expansion candidates synchronously."""
        loc_id = uuid4()
        location = Location(
            location_id=loc_id,
            name="Test Location",
            description="A test location",
            connections={"north": uuid4()},
            state_flags={"visit_count": 5},
        )

        world_state = WorldState(locations={loc_id: location})
        manager = WorldBoundaryManager(world_state)

        candidates = manager.get_expansion_candidates(max_candidates=5)

        assert len(candidates) > 0
        assert all(candidate.location_id == loc_id for candidate in candidates)
        assert all(
            candidate.direction in ["south", "east", "west"] for candidate in candidates
        )
