"""
Unit tests for the core game loop implementation.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest
from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop
from game_loop.state.models import ActionResult


class TestGameLoop:
    """Test cases for the GameLoop class."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console for testing."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def mock_db_pool(self) -> MagicMock:
        """Create a mock database pool for testing."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        # For async context manager support
        mock_pool.__aenter__ = AsyncMock(return_value=mock_pool)
        mock_pool.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.fixture
    def game_config(self) -> GameConfig:
        """Create a test game configuration."""
        return GameConfig()

    @pytest.fixture
    def game_loop(
        self, game_config: GameConfig, mock_db_pool: MagicMock, mock_console: MagicMock
    ) -> GameLoop:
        """Create a test game loop instance."""
        with patch(
            "game_loop.state.manager.GameStateManager"
        ) as mock_state_manager_class:
            # Create a mock state manager
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager

            # Create the game loop with mocks
            loop = GameLoop(game_config, mock_db_pool, mock_console)

            # Mock some common methods
            loop.state_manager = mock_state_manager

            # Important: Use AsyncMock for async methods
            get_location_mock = AsyncMock()
            mock_location = MagicMock()
            mock_location.location_id = uuid4()
            mock_location.name = "Test Location"
            mock_location.description = "A test location"
            mock_location.connections = {}
            mock_location.objects = {}
            mock_location.npcs = {}
            get_location_mock.return_value = mock_location
            loop.state_manager.get_current_location_details = get_location_mock

            # Mock the player tracker
            mock_player_tracker = MagicMock()
            player_state = MagicMock()
            player_state.current_location_id = mock_location.location_id
            mock_player_tracker.get_state = MagicMock(return_value=player_state)

            # Important: Use AsyncMock for async methods
            get_location_id_mock = AsyncMock()
            get_location_id_mock.return_value = player_state.current_location_id
            mock_player_tracker.get_current_location_id = get_location_id_mock

            loop.state_manager.player_tracker = mock_player_tracker
            loop.state_manager.world_tracker = MagicMock()

            return loop

    def test_initialization(self, game_loop: GameLoop) -> None:
        """Test that the game loop initializes correctly."""
        # Check initial state
        assert not game_loop.running
        assert game_loop.state_manager is not None

    @pytest.mark.asyncio
    @patch("builtins.input", return_value="TestPlayer")
    async def test_get_player_name(
        self, mock_input: MagicMock, game_loop: GameLoop
    ) -> None:
        """Test getting the player name."""
        name = game_loop._get_player_name()
        assert name == "TestPlayer"

        # Check that console output was generated
        game_loop.console.print.assert_called_once()

    @pytest.mark.asyncio
    @patch("builtins.input", return_value="TestPlayer")
    async def test_initialize_game(
        self, mock_input: MagicMock, game_loop: GameLoop
    ) -> None:
        """Test game initialization."""
        # Mock the state manager's create_new_game method
        player_state = MagicMock()
        player_state.name = "TestPlayer"
        world_state = MagicMock()
        game_loop.state_manager.create_new_game = AsyncMock(
            return_value=(player_state, world_state)
        )

        await game_loop.initialize()

        # Verify initialize was called
        game_loop.state_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_display_current_location(self, game_loop: GameLoop) -> None:
        """Test displaying the current location."""
        # Test displaying the location
        await game_loop._display_current_location()

        # Verify the location display was called
        assert game_loop.console.print.called

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["look", "quit"])
    async def test_process_input_look(
        self, mock_input: MagicMock, game_loop: GameLoop
    ) -> None:
        """Test processing the 'look' command."""
        # Setup the command handler
        from game_loop.core.input_processor import CommandType, ParsedCommand

        # Mock the input processor to return a LOOK command
        game_loop.input_processor.process_input_async = AsyncMock(
            return_value=ParsedCommand(
                command_type=CommandType.LOOK, action="look", subject=None
            )
        )

        # Mock the execute command method
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.LOOK:
                await game_loop._display_current_location()
                return ActionResult(
                    success=True, feedback_message="You see the location."
                )
            return None

        game_loop._execute_command = mock_execute_command

        game_loop.running = True

        # Process the look command
        await game_loop._process_input_async()

        # Verify console output
        assert game_loop.console.print.called

        # Restore original method
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["north", "quit"])
    async def test_process_input_movement(
        self, mock_input: MagicMock, game_loop: GameLoop
    ) -> None:
        """Test processing movement commands."""
        # Setup the command handler
        from game_loop.core.input_processor import CommandType, ParsedCommand

        # Get UUIDs for testing
        target_location_id = uuid4()

        # Mock the current location
        current_location = await game_loop.state_manager.get_current_location_details()
        current_location.connections = {"north": target_location_id}

        # Mock successful movement with ActionResult using UUID
        movement_result = ActionResult(
            success=True,
            feedback_message="You go north.",
            location_change=True,
            new_location_id=target_location_id,
        )

        # Mock the _handle_movement method
        game_loop._handle_movement = AsyncMock(return_value=movement_result)

        # Mock the input processor to return a MOVEMENT command
        game_loop.input_processor.process_input_async = AsyncMock(
            return_value=ParsedCommand(
                command_type=CommandType.MOVEMENT, action="go", subject="north"
            )
        )

        # Mock the execute command method
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.MOVEMENT:
                player_state = game_loop.state_manager.player_tracker.get_state()
                location = await game_loop.state_manager.get_current_location_details()
                return await game_loop._handle_movement(
                    cmd.subject, player_state, location
                )
            return None

        game_loop._execute_command = mock_execute_command

        game_loop.running = True

        # Process the movement command
        await game_loop._process_input_async()

        # Verify movement was handled
        game_loop._handle_movement.assert_called_once()

        # Restore original method
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    async def test_handle_movement_invalid_direction(self, game_loop: GameLoop) -> None:
        """Test handling movement in an invalid direction."""
        # Mock the necessary objects
        player_state = game_loop.state_manager.player_tracker.get_state()

        # Mock the current location with no "up" connection
        current_location = await game_loop.state_manager.get_current_location_details()
        current_location.connections = {"north": uuid4()}

        # Call the method with invalid direction
        result = await game_loop._handle_movement("up", player_state, current_location)

        # Verify result
        assert not result.success
        assert "You cannot go up from here." in result.feedback_message
