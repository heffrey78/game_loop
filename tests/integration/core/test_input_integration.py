"""
Integration tests for the InputProcessor with GameLoop.
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import asyncpg
import pytest

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop
from game_loop.core.input_processor import CommandType, ParsedCommand
from game_loop.state.models import ActionResult


class TestInputProcessorIntegration:
    """Integration tests for InputProcessor and GameLoop."""

    @pytest.fixture
    def mock_db_pool(self) -> Mock:
        """Create a mock database pool for testing."""
        mock_pool = Mock(spec=asyncpg.Pool)
        # For async context manager support
        mock_pool.__aenter__ = AsyncMock(return_value=mock_pool)
        mock_pool.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.fixture
    def game_loop(self, mock_db_pool: Mock) -> GameLoop:
        """Create a GameLoop instance with mocked components for testing."""
        console_mock = Mock()
        config = GameConfig()

        with patch(
            "game_loop.state.manager.GameStateManager"
        ) as mock_state_manager_class:
            # Create a mock state manager
            mock_state_manager = Mock()
            mock_state_manager.initialize = AsyncMock()
            mock_state_manager.create_new_game = AsyncMock()
            mock_state_manager.update_after_action = AsyncMock()
            mock_state_manager.get_current_session_id = Mock(
                return_value="test-session-id"
            )

            # Set up player tracker
            mock_player_tracker = Mock()
            player_state = Mock()
            player_state.name = "TestPlayer"
            player_state.current_location_id = uuid4()
            player_state.inventory = []
            mock_player_tracker.get_state = Mock(return_value=player_state)

            # IMPORTANT: Use AsyncMock for async methods
            get_location_id_mock = AsyncMock()
            get_location_id_mock.return_value = player_state.current_location_id
            mock_player_tracker.get_current_location_id = get_location_id_mock

            mock_state_manager.player_tracker = mock_player_tracker

            # Set up world tracker
            mock_world_tracker = Mock()
            world_state = Mock()
            mock_world_tracker.get_state = Mock(return_value=world_state)
            mock_state_manager.world_tracker = mock_world_tracker

            # Create the location mock
            mock_location = Mock()
            mock_location.location_id = player_state.current_location_id
            mock_location.name = "Forest Clearing"
            mock_location.description = "A peaceful clearing in the forest."
            destination_id = uuid4()
            mock_location.connections = {"north": destination_id}
            mock_location.objects = {}
            mock_location.npcs = {}

            # IMPORTANT: Use AsyncMock for async methods
            get_location_mock = AsyncMock()
            get_location_mock.return_value = mock_location
            mock_state_manager.get_current_location_details = get_location_mock

            # Assign the mock state manager
            mock_state_manager_class.return_value = mock_state_manager

            # Create the game loop
            loop = GameLoop(config, mock_db_pool, console_mock)

            # Configure game loop's input processor with mock behaviors
            loop.input_processor.process_input_async = AsyncMock()
            loop.input_processor.update_conversation_context = AsyncMock()

            return loop

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["north", "quit"])
    async def test_movement_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that movement commands work correctly in the game loop."""
        # Create UUIDs for testing
        current_location_id = uuid4()
        destination_id = uuid4()

        # Create a mock player state directly
        player_state = Mock()
        player_state.name = "TestPlayer"
        player_state.current_location_id = current_location_id
        player_state.inventory = []

        # IMPORTANT: Mock knowledge as an empty list to make it iterable
        player_state.knowledge = []

        # Replace the get_state method with a Mock that returns our player state
        original_player_get_state = game_loop.state_manager.player_tracker.get_state
        game_loop.state_manager.player_tracker.get_state = Mock(
            return_value=player_state
        )

        # Create a mock location directly
        mock_location = Mock()
        mock_location.location_id = current_location_id
        mock_location.name = "Forest Clearing"
        mock_location.description = "A peaceful clearing in the forest."
        mock_location.connections = {"north": destination_id}
        mock_location.objects = {}
        mock_location.npcs = {}

        # Create a mock world state for location lookups
        mock_world_state = Mock()
        mock_world_state.locations = {str(current_location_id): mock_location}

        # Replace the world_tracker's get_state method with a mock
        original_world_get_state = game_loop.state_manager.world_tracker.get_state
        game_loop.state_manager.world_tracker.get_state = Mock(
            return_value=mock_world_state
        )

        # Replace the get_current_location_details method with our own AsyncMock
        original_get_location = game_loop.state_manager.get_current_location_details
        get_location_mock = AsyncMock(return_value=mock_location)
        game_loop.state_manager.get_current_location_details = get_location_mock

        # Override the extract_game_context method with a simplified version for testing
        original_extract_context = game_loop._extract_game_context
        game_loop._extract_game_context = lambda: {"player": {"name": "TestPlayer"}}

        # Configure the input processor mock to return a MOVEMENT command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(
                command_type=CommandType.MOVEMENT, action="go", subject="north"
            ),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock the movement handler to return a successful result using UUID
        movement_result = ActionResult(
            success=True,
            feedback_message="You go north.",
            location_change=True,
            new_location_id=destination_id,
        )
        # Replace the method with an AsyncMock that returns our result
        game_loop._handle_movement = AsyncMock(return_value=movement_result)

        # Mock the quit command result
        quit_result = ActionResult(
            success=True,
            feedback_message="Farewell, adventurer! Your journey ends here.",
        )

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.MOVEMENT:
                return await game_loop._handle_movement(
                    cmd.subject, player_state, mock_location
                )
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return quit_result
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify movement was handled
        game_loop._handle_movement.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original methods
        game_loop._execute_command = original_execute
        game_loop.state_manager.player_tracker.get_state = original_player_get_state
        game_loop.state_manager.world_tracker.get_state = original_world_get_state
        game_loop.state_manager.get_current_location_details = original_get_location
        game_loop._extract_game_context = original_extract_context

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["look", "quit"])
    async def test_look_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that look commands work correctly in the game loop."""
        # Configure the input processor mock to return a LOOK command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(command_type=CommandType.LOOK, action="look", subject=None),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock the display_current_location method
        original_display = game_loop._display_current_location
        game_loop._display_current_location = AsyncMock()

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.LOOK:
                await game_loop._display_current_location()
                return ActionResult(success=True, feedback_message="")
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify display_location was called
        game_loop._display_current_location.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original methods
        game_loop._display_current_location = original_display
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["inventory", "quit"])
    async def test_inventory_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that inventory commands work correctly in the game loop."""
        # Configure the input processor mock to return an INVENTORY command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(
                command_type=CommandType.INVENTORY, action="inventory", subject=None
            ),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock the display_inventory method
        original_display_inv = game_loop._display_inventory
        game_loop._display_inventory = AsyncMock()

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.INVENTORY:
                await game_loop._display_inventory(
                    game_loop.state_manager.player_tracker.get_state()
                )
                return ActionResult(success=True, feedback_message="")
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify inventory display was called
        game_loop._display_inventory.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original methods
        game_loop._display_inventory = original_display_inv
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["take sword", "quit"])
    async def test_take_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that take commands work correctly in the game loop."""
        # Configure the input processor mock to return a TAKE command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(
                command_type=CommandType.TAKE, action="take", subject="sword"
            ),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Get the location mock that we need for the test
        mock_location = await game_loop.state_manager.get_current_location_details()

        # Mock the take handler to return an appropriate result
        take_result = ActionResult(
            success=False, feedback_message="You don't see any takeable sword here."
        )

        # Replace the method with an AsyncMock that returns our result
        game_loop._handle_take = AsyncMock(return_value=take_result)

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.TAKE:
                return await game_loop._handle_take(
                    cmd.subject,
                    game_loop.state_manager.player_tracker.get_state(),
                    mock_location,
                )
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify take was handled
        game_loop._handle_take.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original method
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["go to nowhere", "quit"])
    async def test_unknown_command_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that unknown commands are handled correctly in the game loop."""
        unknown_command = ParsedCommand(
            command_type=None, action="go to nowhere", subject=None
        )
        game_loop.input_processor.process_input_async.side_effect = [
            unknown_command,
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock format_error_message
        error_msg = (
            "I don't understand 'go to nowhere'.\nType 'help' for a list of commands."
        )
        game_loop.input_processor.format_error_message = Mock(return_value=error_msg)

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type is None:
                error_message = game_loop.input_processor.format_error_message(cmd)
                return ActionResult(success=False, feedback_message=error_message)
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify error handling
        game_loop.input_processor.format_error_message.assert_called_once_with(
            unknown_command
        )

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original method
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["help", "quit"])
    async def test_help_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that help commands work correctly in the game loop."""
        # Configure the input processor mock to return a HELP command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(command_type=CommandType.HELP, action="help", subject=None),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock the _display_help method
        original_display_help = game_loop._display_help
        game_loop._display_help = Mock()

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.HELP:
                game_loop._display_help()
                return ActionResult(success=True, feedback_message="")
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify help was displayed
        game_loop._display_help.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original methods
        game_loop._display_help = original_display_help
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["examine ancient statue", "quit"])
    async def test_examine_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that examine commands work correctly in the game loop."""
        # Configure the input processor mock to return an EXAMINE command
        game_loop.input_processor.process_input_async.side_effect = [
            ParsedCommand(
                command_type=CommandType.EXAMINE,
                action="examine",
                subject="ancient statue",
            ),
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Get the location mock that we need for the test
        mock_location = await game_loop.state_manager.get_current_location_details()

        # Mock the examine handler to return an appropriate result
        examine_result = ActionResult(
            success=False, feedback_message="You don't see any ancient statue here."
        )
        game_loop._handle_examine = AsyncMock(return_value=examine_result)

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.EXAMINE:
                return await game_loop._handle_examine(
                    cmd.subject,
                    game_loop.state_manager.player_tracker.get_state(),
                    mock_location,
                    game_loop.state_manager.world_tracker.get_state(),
                )
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify examine was handled
        game_loop._handle_examine.assert_called_once()

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original method
        game_loop._execute_command = original_execute

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["use key on door", "quit"])
    async def test_use_with_target_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that use commands with targets work correctly in the game loop."""
        # Configure the input processor mock to return a USE command with target
        use_command = ParsedCommand(
            command_type=CommandType.USE, action="use", subject="key", target="door"
        )
        game_loop.input_processor.process_input_async.side_effect = [
            use_command,
            ParsedCommand(command_type=CommandType.QUIT, action="quit", subject=None),
        ]

        # Mock the command handler factory and use handler
        use_handler = AsyncMock()
        use_result = ActionResult(
            success=False,
            feedback_message="Using the key on the door is not implemented yet.",
        )
        use_handler.handle = AsyncMock(return_value=use_result)

        # Set up the mock for the command handler factory
        original_get_handler = game_loop.command_handler_factory.get_handler
        game_loop.command_handler_factory.get_handler = Mock(return_value=use_handler)

        # Override execution behavior for the test
        original_execute = game_loop._execute_command

        async def mock_execute_command(cmd):
            if cmd.command_type == CommandType.USE:
                handler = game_loop.command_handler_factory.get_handler(CommandType.USE)
                return await handler.handle(cmd)
            elif cmd.command_type == CommandType.QUIT:
                game_loop.stop()
                return ActionResult(success=True, feedback_message="")
            return None

        game_loop._execute_command = mock_execute_command

        # Start the game loop
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify use handler was called
        game_loop.command_handler_factory.get_handler.assert_called_once_with(
            CommandType.USE
        )
        use_handler.handle.assert_called_once_with(use_command)

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

        # Restore original methods
        game_loop.command_handler_factory.get_handler = original_get_handler
        game_loop._execute_command = original_execute
