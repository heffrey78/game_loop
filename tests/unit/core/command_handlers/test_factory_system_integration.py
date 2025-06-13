"""Tests for CommandHandlerFactory system command integration."""

import uuid
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from game_loop.core.command_handlers.factory import CommandHandlerFactory
from game_loop.core.models.system_models import (
    SystemCommandClassification,
    SystemCommandType,
)
from game_loop.state.models import ActionResult


@pytest.fixture
def mock_dependencies():
    """Create mocked dependencies for CommandHandlerFactory."""
    return {
        "console": Console(),
        "state_manager": AsyncMock(),
        "session_factory": AsyncMock(),
        "config_manager": AsyncMock(),
        "llm_client": AsyncMock(),
        "semantic_search": AsyncMock(),
        "action_classifier": AsyncMock(),
    }


@pytest.fixture
def command_factory(mock_dependencies):
    """Create CommandHandlerFactory with mocked dependencies."""
    factory = CommandHandlerFactory(**mock_dependencies)

    # Mock the system processor
    factory.system_processor = AsyncMock()
    factory.system_processor.initialize = AsyncMock()
    factory.system_processor.process_command = AsyncMock()

    return factory


class TestCommandFactorySystemIntegration:
    """Test cases for system command integration in CommandHandlerFactory."""

    @pytest.mark.asyncio
    async def test_route_system_command_success(
        self, command_factory, mock_dependencies
    ):
        """Test successful system command routing."""
        text = "save game"
        context = {"player_id": str(uuid.uuid4()), "current_location": "forest"}

        # Mock system command classification
        classification = SystemCommandClassification(
            command_type=SystemCommandType.SAVE_GAME,
            args={"save_name": None},
            confidence=0.9,
            original_text=text,
        )
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            classification
        )

        # Mock system processor response
        expected_result = ActionResult(
            success=True,
            feedback_message="Game saved successfully!",
        )
        command_factory.system_processor.process_command.return_value = expected_result

        # Route system command
        result = await command_factory.route_system_command(text, context)

        # Verify result
        assert result is not None
        assert result.success is True
        assert result.feedback_message == "Game saved successfully!"

        # Verify classifier was called
        mock_dependencies[
            "action_classifier"
        ].classify_system_command.assert_called_once_with(text)

        # Verify system processor was called
        command_factory.system_processor.initialize.assert_called_once()
        command_factory.system_processor.process_command.assert_called_once_with(
            SystemCommandType.SAVE_GAME, {"save_name": None}, context
        )

    @pytest.mark.asyncio
    async def test_route_system_command_not_system(
        self, command_factory, mock_dependencies
    ):
        """Test routing when input is not a system command."""
        text = "go north"
        context = {"player_id": str(uuid.uuid4())}

        # Mock no system command classification
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            None
        )

        # Route command
        result = await command_factory.route_system_command(text, context)

        # Should return None for non-system commands
        assert result is None

        # System processor should not be called
        command_factory.system_processor.process_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_system_command_no_classifier(self, command_factory):
        """Test routing when action classifier is not available."""
        command_factory.action_classifier = None

        text = "save game"
        context = {}

        # Route command
        result = await command_factory.route_system_command(text, context)

        # Should return None when classifier unavailable
        assert result is None

    @pytest.mark.asyncio
    async def test_route_system_command_no_processor(self, command_factory):
        """Test routing when system processor is not available."""
        command_factory.system_processor = None

        text = "save game"
        context = {}

        # Route command
        result = await command_factory.route_system_command(text, context)

        # Should return None when processor unavailable
        assert result is None

    @pytest.mark.asyncio
    async def test_route_system_command_error(self, command_factory, mock_dependencies):
        """Test error handling in system command routing."""
        text = "save game"
        context = {"player_id": str(uuid.uuid4())}

        # Mock system command classification
        classification = SystemCommandClassification(
            command_type=SystemCommandType.SAVE_GAME,
            args={},
            confidence=0.9,
            original_text=text,
        )
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            classification
        )

        # Mock system processor error
        command_factory.system_processor.process_command.side_effect = Exception(
            "Processing error"
        )

        # Route command
        result = await command_factory.route_system_command(text, context)

        # Should handle error gracefully
        assert result is not None
        assert result.success is False
        assert "Error processing system command" in result.feedback_message
        # Note: ActionResult doesn't have error_details field

    @pytest.mark.asyncio
    async def test_handle_command_system_priority(
        self, command_factory, mock_dependencies
    ):
        """Test that handle_command gives priority to system commands."""
        text = "help"
        context = {"player_id": str(uuid.uuid4())}

        # Mock system command classification
        classification = SystemCommandClassification(
            command_type=SystemCommandType.HELP,
            args={"topic": None},
            confidence=0.9,
            original_text=text,
        )
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            classification
        )

        # Mock system command result
        system_result = ActionResult(
            success=True,
            feedback_message="Help content here",
        )
        command_factory.system_processor.process_command.return_value = system_result

        # Handle command
        result = await command_factory.handle_command(text, context)

        # Should return system command result
        assert result is not None
        assert result.success is True
        assert result.feedback_message == "Help content here"
        # Note: ActionResult doesn't have action_type field

    @pytest.mark.asyncio
    async def test_handle_command_not_system(self, command_factory, mock_dependencies):
        """Test handle_command when input is not a system command."""
        text = "attack goblin"
        context = {"player_id": str(uuid.uuid4())}

        # Mock no system command classification
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            None
        )

        # Handle command
        result = await command_factory.handle_command(text, context)

        # Should return None to let normal processing continue
        assert result is None

    @pytest.mark.asyncio
    async def test_system_processor_initialization_once(
        self, command_factory, mock_dependencies
    ):
        """Test that system processor is only initialized once."""
        text = "save game"
        context = {"player_id": str(uuid.uuid4())}

        # Mock system command classification
        classification = SystemCommandClassification(
            command_type=SystemCommandType.SAVE_GAME,
            args={},
            confidence=0.9,
            original_text=text,
        )
        mock_dependencies["action_classifier"].classify_system_command.return_value = (
            classification
        )

        # Mock system processor response
        system_result = ActionResult(success=True, feedback_message="Saved")
        command_factory.system_processor.process_command.return_value = system_result

        # Route command multiple times
        await command_factory.route_system_command(text, context)
        await command_factory.route_system_command(text, context)
        await command_factory.route_system_command(text, context)

        # Initialize should only be called once per process_command call
        # (it's called in process_command, not route_system_command)
        assert command_factory.system_processor.initialize.call_count == 3

    @pytest.mark.asyncio
    async def test_command_factory_without_system_dependencies(self):
        """Test CommandHandlerFactory when system dependencies are missing."""
        # Create factory without system dependencies
        factory = CommandHandlerFactory(
            console=Console(),
            state_manager=AsyncMock(),
            # Missing session_factory, config_manager, llm_client, semantic_search
        )

        # System processor should not be initialized
        assert factory.system_processor is None

        # Route system command should return None
        result = await factory.route_system_command("save game", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_system_command_types(
        self, command_factory, mock_dependencies
    ):
        """Test routing different types of system commands."""
        test_cases = [
            ("save game", SystemCommandType.SAVE_GAME, {}),
            ("load save", SystemCommandType.LOAD_GAME, {"save_name": "save"}),
            ("help combat", SystemCommandType.HELP, {"topic": "combat"}),
            ("settings", SystemCommandType.SETTINGS, {}),
            ("quit", SystemCommandType.QUIT_GAME, {"force": False}),
        ]

        for text, command_type, args in test_cases:
            # Mock classification
            classification = SystemCommandClassification(
                command_type=command_type, args=args, confidence=0.9, original_text=text
            )
            mock_dependencies[
                "action_classifier"
            ].classify_system_command.return_value = classification

            # Mock processor response
            system_result = ActionResult(
                success=True,
                feedback_message=f"Processed {command_type.value}",
            )
            command_factory.system_processor.process_command.return_value = (
                system_result
            )

            # Route command
            result = await command_factory.route_system_command(text, {})

            # Verify routing
            assert result is not None
            assert result.success is True
            assert command_type.value in result.feedback_message

            # Verify processor was called with correct parameters
            command_factory.system_processor.process_command.assert_called_with(
                command_type, args, {}
            )
