"""
Integration tests for the InputProcessor with GameLoop.
"""

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop


class TestInputProcessorIntegration:
    """Integration tests for InputProcessor and GameLoop."""

    @pytest.fixture
    def game_loop(self) -> GameLoop:
        """Create a GameLoop instance with mocked console for testing."""
        console_mock = Mock()
        config = GameConfig()
        loop = GameLoop(config, console=console_mock)
        loop._create_demo_world()
        loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")
        return loop

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["north", "quit"])
    async def test_movement_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that movement commands work correctly in the game loop."""
        # Initial location should be forest_clearing
        assert game_loop.game_state.player is not None
        assert game_loop.game_state.player.current_location_id == "forest_clearing"

        # Start game and run first input (north)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify location changed to dark_forest
        assert game_loop.game_state.player is not None
        assert game_loop.game_state.player.current_location_id == "dark_forest"

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["look", "quit"])
    async def test_look_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that look commands work correctly in the game loop."""
        # Mock the location_display method to track if it was called
        original_display_location = game_loop.location_display.display_location
        display_location_called = [False]  # Use a list to track whether it was called

        def mock_display_location(*args: Any, **kwargs: Any) -> Any:
            display_location_called[0] = True
            return original_display_location(*args, **kwargs)

        # Type-safe patching using monkey patching
        game_loop.location_display = cast(
            Any, Mock(display_location=mock_display_location)
        )

        # Start game and run first input (look)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify the display_location method was called
        assert display_location_called[0], "display_location was not called"

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["inventory", "quit"])
    async def test_inventory_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that inventory commands work correctly in the game loop."""
        # Start game and run first input (inventory)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that inventory was displayed - use cast to handle type checking
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call("[bold]Inventory:[/bold]")

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["take sword", "quit"])
    async def test_take_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that take commands work correctly in the game loop."""
        # Start game and run first input (take sword)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that take message was displayed
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call("[yellow]There is no sword here.[/yellow]")

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["go to nowhere", "quit"])
    async def test_unknown_command_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that unknown commands are handled correctly in the game loop."""
        # Start game and run first input (unknown command)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that error message was displayed
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call(
            """[yellow]I don't understand 'go to nowhere'.
            Type 'help' for a list of commands.[/yellow]"""
        )

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["help", "quit"])
    async def test_help_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that help commands work correctly in the game loop."""
        # Start game and run first input (help)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that help was displayed
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call("[bold]Available Commands:[/bold]")

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["examine ancient statue", "quit"])
    async def test_examine_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that examine commands work correctly in the game loop."""
        # Start game and run first input (examine)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that examine message was displayed
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call(
            "[yellow]You don't see any ancient statue here.[/yellow]"
        )

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running

    @pytest.mark.asyncio
    @patch("builtins.input", side_effect=["use key on door", "quit"])
    async def test_use_with_target_integration(
        self, mock_input: Mock, game_loop: GameLoop
    ) -> None:
        """Test that use commands with targets work correctly in the game loop."""
        # Start game and run first input (use with target)
        game_loop.running = True
        await game_loop._process_input_async()

        # Verify that use message was displayed
        mock_console = cast(Mock, game_loop.console)
        mock_console.print.assert_any_call(
            "[yellow]Using the key on the door is not implemented yet.[/yellow]"
        )

        # Process quit command to stop the loop
        await game_loop._process_input_async()
        assert not game_loop.running
