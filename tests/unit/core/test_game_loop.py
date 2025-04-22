"""
Unit tests for the core game loop implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop


class TestGameLoop:
    """Test cases for the GameLoop class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def game_config(self):
        """Create a test game configuration."""
        return GameConfig()

    @pytest.fixture
    def game_loop(self, game_config, mock_console):
        """Create a test game loop instance."""
        return GameLoop(game_config, mock_console)

    def test_initialization(self, game_loop):
        """Test that the game loop initializes correctly."""
        # Check initial state
        assert not game_loop.running
        assert game_loop.game_state is not None

    def test_create_demo_world(self, game_loop):
        """Test demo world creation."""
        game_loop._create_demo_world()

        # Check that the demo world has locations
        assert len(game_loop.game_state.world.locations) > 0

        # Check for the forest clearing location
        forest_clearing = game_loop.game_state.world.get_location("forest_clearing")
        assert forest_clearing is not None
        assert forest_clearing.name == "Forest Clearing"

        # Check connections
        assert "north" in forest_clearing.connections
        assert forest_clearing.connections["north"] == "dark_forest"

    @patch("builtins.input", return_value="TestPlayer")
    def test_get_player_name(self, mock_input, game_loop):
        """Test getting the player name."""
        name = game_loop._get_player_name()
        assert name == "TestPlayer"

        # Check that console output was generated
        game_loop.console.print.assert_called_once()

    @patch("builtins.input", return_value="TestPlayer")
    def test_initialize_game(self, mock_input, game_loop):
        """Test game initialization."""
        game_loop.initialize()

        # Check that player state was created
        assert game_loop.game_state.player is not None
        assert game_loop.game_state.player.name == "TestPlayer"
        assert game_loop.game_state.player.current_location_id == "forest_clearing"

    def test_display_current_location(self, game_loop):
        """Test displaying the current location."""
        # Set up a test location and game state
        game_loop._create_demo_world()
        game_loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")

        # Test displaying the location
        game_loop._display_current_location()

        # Verify the location display was called
        assert game_loop.console.print.called

    @patch("builtins.input", side_effect=["look", "quit"])
    def test_process_input_look(self, mock_input, game_loop):
        """Test processing the 'look' command."""
        # Set up game state
        game_loop._create_demo_world()
        game_loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")
        game_loop.running = True

        # Process the look command
        game_loop._process_input()

        # Verify console output
        assert game_loop.console.print.called

        # Process the quit command to avoid infinite loop
        game_loop._process_input()
        assert not game_loop.running

    @patch("builtins.input", side_effect=["north", "quit"])
    def test_process_input_movement(self, mock_input, game_loop):
        """Test processing movement commands."""
        # Set up game state
        game_loop._create_demo_world()
        game_loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")
        game_loop.running = True

        # Initial location should be forest_clearing
        assert game_loop.game_state.player.current_location_id == "forest_clearing"

        # Process the north command
        game_loop._process_input()

        # Verify location changed to dark_forest
        assert game_loop.game_state.player.current_location_id == "dark_forest"

        # Process the quit command to avoid infinite loop
        game_loop._process_input()
        assert not game_loop.running

    def test_handle_movement_invalid_direction(self, game_loop):
        """Test handling movement in an invalid direction."""
        # Set up game state
        game_loop._create_demo_world()
        game_loop.game_state.initialize_new_game("TestPlayer", "forest_clearing")

        # Try to move in an invalid direction (up)
        game_loop._handle_movement("up")

        # Verify location did not change
        assert game_loop.game_state.player.current_location_id == "forest_clearing"

        # Verify error message was displayed
        game_loop.console.print.assert_called_with(
            "[yellow]You cannot go up from here.[/yellow]"
        )
