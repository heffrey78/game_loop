"""
Integration tests for the InputProcessor with GameLoop.
"""

from unittest.mock import Mock, patch

import pytest

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop


class TestInputProcessorIntegration:
    """Integration tests for InputProcessor and GameLoop."""

    @pytest.fixture
    def game_loop(self):
        """Create a GameLoop instance with mocked console for testing."""
        console_mock = Mock()
        config = GameConfig()
        loop = GameLoop(config, console=console_mock)
        loop._create_demo_world()
        loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")
        return loop

    @patch("builtins.input", side_effect=["north", "quit"])
    def test_movement_integration(self, mock_input, game_loop):
        """Test that movement commands work correctly in the game loop."""
        # Initial location should be forest_clearing
        assert game_loop.game_state.player.current_location_id == "forest_clearing"

        # Start game and run first input (north)
        game_loop.running = True
        game_loop._process_input()

        # Verify location changed to dark_forest
        assert game_loop.game_state.player.current_location_id == "dark_forest"

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["look", "quit"])
    def test_look_integration(self, mock_input, game_loop):
        """Test that look commands work correctly in the game loop."""
        # Mock the location_display method to track if it was called
        original_display_location = game_loop.location_display.display_location
        display_location_called = [False]  # Use a list to track whether it was called

        def mock_display_location(*args, **kwargs):
            display_location_called[0] = True
            return original_display_location(*args, **kwargs)

        game_loop.location_display.display_location = mock_display_location

        # Start game and run first input (look)
        game_loop.running = True
        game_loop._process_input()

        # Verify the display_location method was called
        assert display_location_called[0], "display_location was not called"

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["inventory", "quit"])
    def test_inventory_integration(self, mock_input, game_loop):
        """Test that inventory commands work correctly in the game loop."""
        # Start game and run first input (inventory)
        game_loop.running = True
        game_loop._process_input()

        # Verify that inventory was displayed
        game_loop.console.print.assert_any_call("[bold]Inventory:[/bold]")

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["take sword", "quit"])
    def test_take_integration(self, mock_input, game_loop):
        """Test that take commands work correctly in the game loop."""
        # Start game and run first input (take sword)
        game_loop.running = True
        game_loop._process_input()

        # Verify that take message was displayed
        game_loop.console.print.assert_any_call(
            "[yellow]Taking the sword is not implemented yet.[/yellow]"
        )

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["go to nowhere", "quit"])
    def test_unknown_command_integration(self, mock_input, game_loop):
        """Test that unknown commands are handled correctly in the game loop."""
        # Start game and run first input (unknown command)
        game_loop.running = True
        game_loop._process_input()

        # Verify that error message was displayed
        game_loop.console.print.assert_any_call(
            "[yellow]I don't understand 'go to nowhere'. "
            "Type 'help' for a list of commands.[/yellow]"
        )

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["help", "quit"])
    def test_help_integration(self, mock_input, game_loop):
        """Test that help commands work correctly in the game loop."""
        # Start game and run first input (help)
        game_loop.running = True
        game_loop._process_input()

        # Verify that help was displayed
        game_loop.console.print.assert_any_call("[bold]Available Commands:[/bold]")

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["examine ancient statue", "quit"])
    def test_examine_integration(self, mock_input, game_loop):
        """Test that examine commands work correctly in the game loop."""
        # Start game and run first input (examine)
        game_loop.running = True
        game_loop._process_input()

        # Verify that examine message was displayed
        game_loop.console.print.assert_any_call(
            "[yellow]Examining the ancient statue is not implemented yet.[/yellow]"
        )

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["use key on door", "quit"])
    def test_use_with_target_integration(self, mock_input, game_loop):
        """Test that use commands with targets work correctly in the game loop."""
        # Start game and run first input (use with target)
        game_loop.running = True
        game_loop._process_input()

        # Verify that use message was displayed
        game_loop.console.print.assert_any_call(
            "[yellow]Using the key on the door is not implemented yet.[/yellow]"
        )

        # Process quit command to stop the loop
        game_loop._process_input()
        assert not game_loop.running
