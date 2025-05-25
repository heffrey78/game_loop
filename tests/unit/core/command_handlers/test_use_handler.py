"""
Test for the refactored UseHandler implementation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from game_loop.core.command_handlers.use_handler.container_handler import (
    ContainerUsageHandler,
)
from game_loop.core.command_handlers.use_handler.factory import UsageHandlerFactory
from game_loop.core.command_handlers.use_handler.self_handler import SelfUsageHandler
from game_loop.core.command_handlers.use_handler.target_handler import (
    TargetUsageHandler,
)
from game_loop.core.command_handlers.use_handler.use_handler import UseHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState


class TestUseHandler:
    """Tests for the refactored UseHandler class with strategy pattern."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_state_manager(self) -> MagicMock:
        manager = MagicMock()
        manager.player_tracker = MagicMock()
        manager.player_tracker.get_state = MagicMock(
            return_value=self._create_mock_player_state()
        )
        manager.player_tracker.get_current_location_id = AsyncMock(
            return_value="room-1"
        )
        manager.get_current_location_details = AsyncMock(
            return_value=self._create_mock_location()
        )
        return manager

    @pytest.fixture
    def use_handler(self, mock_console, mock_state_manager) -> UseHandler:
        return UseHandler(mock_console, mock_state_manager)

    def _create_mock_player_state(self) -> PlayerState:
        player = MagicMock(spec=PlayerState)
        player.inventory = [
            InventoryItem(
                name="brass key",
                description="A small brass key",
                attributes={"is_key": True},
            ),
            InventoryItem(
                name="backpack",
                description="A leather backpack",
                attributes={"is_container": True, "contained_items": []},
            ),
        ]
        return player

    def _create_mock_location(self) -> Location:
        location = MagicMock(spec=Location)
        location.location_id = "room-1"
        location.objects = {
            "chest-1": MagicMock(
                name="wooden chest", is_container=True, contained_items=[]
            )
        }
        return location

    @pytest.mark.asyncio
    async def test_factory_returns_correct_handler(self) -> None:
        """Test that factory returns the correct handler for each scenario."""
        factory = UsageHandlerFactory()

        # Test container usage
        handler = factory.get_handler("put in backpack")
        assert isinstance(handler, ContainerUsageHandler)

        # Test target usage
        handler = factory.get_handler("door")
        assert isinstance(handler, TargetUsageHandler)

        # Test self-use
        handler = factory.get_handler(None)
        assert isinstance(handler, SelfUsageHandler)

    @pytest.mark.asyncio
    async def test_use_handler_delegates_to_container_handler(
        self,
        use_handler: UseHandler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that use handler delegates to container handler correctly."""

        # Mock get_required_state to bypass its internals
        async def mock_get_required_state() -> tuple[PlayerState, Location, None]:
            return (
                self._create_mock_player_state(),
                self._create_mock_location(),
                None,  # No need for world_state in this test
            )

        use_handler.get_required_state = mock_get_required_state

        command = ParsedCommand(
            command_type="USE",
            action="put",
            subject="brass key",
            target="put in backpack",
        )

        # Create a mock for the container handler
        mock_container_handler = AsyncMock()
        mock_container_handler.handle = AsyncMock(
            return_value=ActionResult(
                success=True, feedback_message="You put the brass key in the backpack."
            )
        )

        # Create a mock factory class and instance
        mock_factory = MagicMock()
        mock_factory.get_handler.return_value = mock_container_handler

        # Create a mock factory class that returns our mock instance
        mock_factory_class = MagicMock(return_value=mock_factory)

        # Patch the UsageHandlerFactory class
        monkeypatch.setattr(
            "game_loop.core.command_handlers.use_handler.use_handler.UsageHandlerFactory",
            mock_factory_class,
        )

        # Execute the use handler
        result = await use_handler.handle(command)

        # Verify that the factory was created
        assert mock_factory_class.called

        # Verify that the factory's get_handler was called with the correct target
        mock_factory.get_handler.assert_called_once_with("put in backpack")

        # Verify that the container handler's handle method was called
        mock_container_handler.handle.assert_called_once()

        # Verify that the result came from the container handler
        assert result.success is True
        assert result.feedback_message == "You put the brass key in the backpack."
