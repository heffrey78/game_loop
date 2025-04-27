"""
Integration tests for the NLP processing pipeline with the game loop.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop
from game_loop.core.input_processor import CommandType, ParsedCommand


@pytest.fixture
def mock_config() -> GameConfig:
    """Create a mock config for testing."""
    config = GameConfig()
    config.features.use_nlp = True
    return config


@pytest.fixture
def mock_console() -> MagicMock:
    """Create a mock console for testing."""
    console = MagicMock()
    console.print = MagicMock()
    return console


@pytest.fixture
def mock_nlp_processor() -> AsyncMock:
    """Create a mock NLP processor that returns predefined results."""
    processor = AsyncMock()
    processor.process_input = AsyncMock()
    processor.process_input.return_value = ParsedCommand(
        command_type=CommandType.LOOK,
        action="look",
        subject="surroundings",
        parameters={"confidence": 0.9},
    )
    return processor


@pytest.fixture
def mock_command_mapper() -> MagicMock:
    """Create a mock command mapper."""
    mapper = MagicMock()
    return mapper


class TestNLPIntegration:
    """Integration tests for NLP processing with the game loop."""

    @pytest.fixture
    def patched_enhanced_input_processor(
        self, mock_nlp_processor: AsyncMock, mock_command_mapper: MagicMock
    ) -> Generator[None, None, None]:
        """Create a patched EnhancedInputProcessor with mock components."""
        with patch(
            "game_loop.core.enhanced_input_processor.NLPProcessor",
            return_value=mock_nlp_processor,
        ):
            with patch(
                "game_loop.core.enhanced_input_processor.CommandMapper",
                return_value=mock_command_mapper,
            ):
                yield

    @pytest.mark.asyncio
    async def test_process_input_with_nlp(
        self,
        mock_config: GameConfig,
        mock_console: MagicMock,
        patched_enhanced_input_processor: None,
    ) -> None:
        """Test processing input with the NLP pipeline."""
        with patch(
            "game_loop.core.game_loop.EnhancedInputProcessor"
        ) as mock_processor_class:
            enhanced_processor = AsyncMock()
            # Update to use process_input_async instead of process_input
            enhanced_processor.process_input_async.return_value = ParsedCommand(
                command_type=CommandType.LOOK, action="look", subject=None
            )
            enhanced_processor.format_error_message.return_value = "I don't understand."
            # Add the update_conversation_context mock
            enhanced_processor.update_conversation_context = AsyncMock()
            mock_processor_class.return_value = enhanced_processor

            # Create the game loop with the mocked processor
            game_loop = GameLoop(config=mock_config, console=mock_console)

            # Test with various natural language inputs
            test_inputs = [
                "look around",
                "examine the surroundings",
                "check out this place",
                "what do I see here",
            ]

            for input_text in test_inputs:
                # Mock the user input
                with patch("builtins.input", return_value=input_text):
                    # Process the input
                    await game_loop._process_input_async()

                    # Verify that EnhancedInputProcessor.process_input_async was called
                    enhanced_processor.process_input_async.assert_called()

                    # Reset for next test
                    enhanced_processor.process_input_async.reset_mock()

    @pytest.mark.asyncio
    async def test_nlp_fallback_to_pattern_matching(
        self,
        mock_config: GameConfig,
        mock_console: MagicMock,
        patched_enhanced_input_processor: None,
    ) -> None:
        """Test fallback to pattern matching when NLP processing fails."""
        with patch(
            "game_loop.core.game_loop.EnhancedInputProcessor"
        ) as mock_enhanced_processor_class:
            # Configure the enhanced processor to raise an exception
            enhanced_processor = AsyncMock()
            # Update to use process_input_async instead of process_input
            enhanced_processor.process_input_async.side_effect = Exception("NLP failed")
            # Add the update_conversation_context mock
            enhanced_processor.update_conversation_context = AsyncMock()
            mock_enhanced_processor_class.return_value = enhanced_processor

            # Configure the basic processor
            with patch(
                "game_loop.core.game_loop.InputProcessor"
            ) as mock_basic_processor_class:
                basic_processor = AsyncMock()
                # Set up the process_input_async method since that's
                # the method called in the code
                basic_processor.process_input_async.return_value = ParsedCommand(
                    command_type=CommandType.LOOK, action="look", subject=None
                )
                mock_basic_processor_class.return_value = basic_processor

                # Create the game loop
                game_loop = GameLoop(config=mock_config, console=mock_console)

                # Mock the user input
                with patch("builtins.input", return_value="look around"):
                    # Process the input
                    await game_loop._process_input_async()

                    # Verify that enhanced processor was called with the correct method
                    enhanced_processor.process_input_async.assert_called_once()

                    # Verify that basic processor was used as fallback
                    # with the correct method
                    basic_processor.process_input_async.assert_called_once_with(
                        "look around"
                    )

                    # Verify that the console printed the fallback message
                    mock_console.print.assert_any_call(
                        "[yellow]Using simplified input processing...[/yellow]"
                    )

    def test_game_context_extraction(
        self, mock_config: GameConfig, mock_console: MagicMock
    ) -> None:
        """Test game context extraction for NLP processing."""
        # Create the game loop
        with patch.object(GameLoop, "_get_player_name", return_value="TestPlayer"):
            game_loop = GameLoop(config=mock_config, console=mock_console)

            # Initialize with a test world
            game_loop.initialize()

            # Extract game context
            context = game_loop._extract_game_context()

            # Verify context structure
            assert "current_location" in context
            assert "player" in context
            assert "visible_objects" in context

            # Verify location data
            assert "name" in context["current_location"]
            assert "description" in context["current_location"]

            # If player is initialized
            if game_loop.game_state.player:
                assert "name" in context["player"]
                assert "inventory" in context["player"]
            # Additional assertions for completeness
            assert "connections" in context
            assert "npcs" in context
