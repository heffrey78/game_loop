"""
Unit tests for MovementManager.

This module tests the movement processing functionality including
movement validation, execution, and pathfinding integration.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionResult,
    PhysicalActionType,
)
from game_loop.core.movement.movement_manager import MovementManager


class TestMovementManager:
    """Test cases for MovementManager."""

    @pytest.fixture
    def mock_world_state_manager(self):
        """Fixture for mock world state manager."""
        return Mock()

    @pytest.fixture
    def mock_location_service(self):
        """Fixture for mock location service."""
        return Mock()

    @pytest.fixture
    def mock_physics_engine(self):
        """Fixture for mock physics engine."""
        mock_physics = Mock()
        mock_physics.validate_physical_constraints = AsyncMock(
            return_value=(True, None)
        )
        return mock_physics

    @pytest.fixture
    def movement_manager(
        self, mock_world_state_manager, mock_location_service, mock_physics_engine
    ):
        """Fixture for MovementManager instance."""
        return MovementManager(
            world_state_manager=mock_world_state_manager,
            location_service=mock_location_service,
            physics_engine=mock_physics_engine,
        )

    @pytest.fixture
    def sample_context(self):
        """Fixture for sample movement context."""
        return {
            "player_id": "player_1",
            "player_state": {
                "current_location": "forest_clearing",
                "energy": 100,
                "strength": 50,
                "movement_speed": 1.0,
            },
            "current_location": "forest_clearing",
        }

    @pytest.mark.asyncio
    async def test_process_movement_command_success(
        self, movement_manager, sample_context
    ):
        """Test successful movement command processing."""
        result = await movement_manager.process_movement_command(
            "player_1", "north", sample_context
        )

        assert isinstance(result, PhysicalActionResult)
        assert result.action_type == PhysicalActionType.MOVEMENT

    @pytest.mark.asyncio
    async def test_process_movement_command_invalid_direction(
        self, movement_manager, sample_context
    ):
        """Test movement command with invalid direction."""
        result = await movement_manager.process_movement_command(
            "player_1", "invalid_direction", sample_context
        )

        assert isinstance(result, PhysicalActionResult)

    @pytest.mark.asyncio
    async def test_validate_movement_success(self, movement_manager, sample_context):
        """Test successful movement validation."""
        is_valid, error = await movement_manager.validate_movement(
            "forest_clearing",
            "forest_clearing_north",
            sample_context.get("player_state", {}),
        )

        assert isinstance(is_valid, bool)
        if not is_valid:
            assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_find_path(self, movement_manager, sample_context):
        """Test pathfinding functionality."""
        path = await movement_manager.find_path(
            "forest_clearing", "mountain_peak", sample_context
        )

        assert isinstance(path, list) or path is None

    @pytest.mark.asyncio
    async def test_calculate_travel_time(self, movement_manager):
        """Test travel time calculation."""
        time_cost = await movement_manager.calculate_travel_time(
            "forest_clearing", "nearby_village", "walking"
        )

        assert isinstance(time_cost, (int, float))
        assert time_cost >= 0

    @pytest.mark.asyncio
    async def test_get_available_exits(self, movement_manager, sample_context):
        """Test getting available exits from location."""
        exits = await movement_manager.get_available_exits(
            "forest_clearing", sample_context.get("player_state", {})
        )

        assert isinstance(exits, list)

    @pytest.mark.asyncio
    async def test_handle_location_transition(self, movement_manager, sample_context):
        """Test location transition handling."""
        result = await movement_manager.handle_location_transition(
            "player_1", "forest_clearing", "mountain_base"
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_check_movement_obstacles(self, movement_manager, sample_context):
        """Test movement obstacle checking."""
        obstacles = await movement_manager.check_movement_obstacles(
            "forest_clearing", "north", sample_context
        )

        assert isinstance(obstacles, list)

    @pytest.mark.asyncio
    async def test_apply_movement_effects(self, movement_manager, sample_context):
        """Test movement effects application."""
        from game_loop.core.command_handlers.physical_action_processor import (
            PhysicalActionResult,
            PhysicalActionType,
        )

        movement_result = PhysicalActionResult(
            success=True,
            action_type=PhysicalActionType.MOVEMENT,
            affected_entities=["player_1"],
            state_changes={"player_location": "mountain_base"},
            energy_cost=5.0,
            time_elapsed=3.0,
            side_effects=[],
            description="Movement successful",
        )

        # This method returns None but updates state
        result = await movement_manager.apply_movement_effects(
            "player_1", movement_result
        )

        assert result is None  # Method returns None

    @pytest.mark.asyncio
    async def test_location_validation(self, movement_manager):
        """Test location capacity validation."""
        is_valid = await movement_manager._validate_location_capacity("test_location")

        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_get_location_exits(self, movement_manager):
        """Test getting location exits."""
        exits = await movement_manager._get_location_exits("test_location")

        assert isinstance(exits, list)

    @pytest.mark.asyncio
    async def test_get_location_arrival_effects(self, movement_manager):
        """Test getting location arrival effects."""
        effects = await movement_manager._get_location_arrival_effects("test_location")

        assert isinstance(effects, list)

    @pytest.mark.asyncio
    async def test_check_location_events(self, movement_manager, sample_context):
        """Test location event checking."""
        events = await movement_manager._check_location_events(
            "test_location", "player_1"
        )

        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_get_location_obstacles(self, movement_manager, sample_context):
        """Test getting location obstacles."""
        obstacles = await movement_manager._get_location_obstacles(
            "test_location", "north"
        )

        assert isinstance(obstacles, list)

    @pytest.mark.asyncio
    async def test_get_movement_required_items(self, movement_manager, sample_context):
        """Test getting movement required items."""
        items = await movement_manager._get_movement_required_items(
            "test_location", "north"
        )

        assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, movement_manager):
        """Test error handling in movement processing."""
        # Test with invalid context
        result = await movement_manager.process_movement_command(
            "invalid_player", "north", {}
        )

        assert isinstance(result, PhysicalActionResult)

    def test_direction_aliases_initialization(self, movement_manager):
        """Test direction aliases initialization."""
        assert hasattr(movement_manager, "_direction_aliases")
        assert isinstance(movement_manager._direction_aliases, dict)

    def test_movement_cache_initialization(self, movement_manager):
        """Test movement cache initialization."""
        assert hasattr(movement_manager, "_movement_cache")
        assert isinstance(movement_manager._movement_cache, dict)

    def test_pathfinding_cache_initialization(self, movement_manager):
        """Test pathfinding cache initialization."""
        assert hasattr(movement_manager, "_pathfinding_cache")
        assert isinstance(movement_manager._pathfinding_cache, dict)


@pytest.mark.asyncio
async def test_integration_with_world_state():
    """Test MovementManager integration with world state manager."""
    # Create mocks
    mock_world_state = Mock()
    mock_location_service = Mock()
    mock_physics = Mock()
    mock_physics.validate_physical_constraints = AsyncMock(return_value=(True, None))

    # Create movement manager
    movement_manager = MovementManager(
        world_state_manager=mock_world_state,
        location_service=mock_location_service,
        physics_engine=mock_physics,
    )

    # Test movement command
    context = {
        "player_id": "test_player",
        "player_state": {
            "current_location": "start",
            "energy": 100,
            "strength": 50,
            "movement_speed": 1.5,
        },
    }

    result = await movement_manager.process_movement_command(
        "test_player", "north", context
    )

    # Verify result
    assert isinstance(result, PhysicalActionResult)
    assert result.action_type == PhysicalActionType.MOVEMENT


if __name__ == "__main__":
    pytest.main([__file__])
