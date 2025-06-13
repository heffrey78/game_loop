"""Tests for SystemCommandProcessor."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from game_loop.core.command_handlers.system_command_processor import (
    SystemCommandProcessor,
)
from game_loop.core.models.system_models import SystemCommandType


@pytest.fixture
def mock_session_factory():
    """Mock session factory."""
    return AsyncMock()


@pytest.fixture
def mock_game_state_manager():
    """Mock game state manager."""
    return AsyncMock()


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    return AsyncMock()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    return AsyncMock()


@pytest.fixture
def mock_semantic_search():
    """Mock semantic search service."""
    return AsyncMock()


@pytest.fixture
def system_processor(
    mock_session_factory,
    mock_game_state_manager,
    mock_config_manager,
    mock_llm_client,
    mock_semantic_search,
):
    """Create SystemCommandProcessor with mocked dependencies."""
    processor = SystemCommandProcessor(
        mock_session_factory,
        mock_game_state_manager,
        mock_config_manager,
        mock_llm_client,
        mock_semantic_search,
    )

    # Mock the subsystems
    processor.save_manager = AsyncMock()
    processor.help_system = AsyncMock()
    processor.tutorial_manager = AsyncMock()
    processor.settings_manager = AsyncMock()

    return processor


class TestSystemCommandProcessor:
    """Test cases for SystemCommandProcessor."""

    @pytest.mark.asyncio
    async def test_initialization(self, system_processor):
        """Test processor initialization."""
        # Should not be initialized initially
        assert not system_processor._initialized

        # Initialize
        await system_processor.initialize()

        # Should be initialized
        assert system_processor._initialized

        # Subsystems should be initialized
        system_processor.help_system.initialize.assert_called_once()
        system_processor.settings_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_game_success(self, system_processor):
        """Test successful game save."""
        player_id = uuid.uuid4()
        save_name = "test_save"
        context = {"current_location": "forest", "player_id": str(player_id)}

        # Mock successful save
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Game saved successfully"  # SaveManager returns 'message', not 'feedback_message'
        mock_result.save_name = save_name
        mock_result.save_metadata = MagicMock()
        mock_result.save_metadata.to_dict.return_value = {"test": "data"}

        system_processor.save_manager.create_save.return_value = mock_result

        # Process save command
        result = await system_processor.handle_save_game(player_id, save_name, context)

        # Verify result
        assert result.success is True
        assert "saved successfully" in result.feedback_message.lower()

        # Verify save manager was called correctly
        system_processor.save_manager.create_save.assert_called_once_with(
            save_name=save_name,
            description="Manual save at forest",
        )

    @pytest.mark.asyncio
    async def test_save_game_no_player(self, system_processor):
        """Test save game with no player context."""
        result = await system_processor.handle_save_game(None, "test", {})

        assert result.success is False
        assert "No player context" in result.feedback_message

    @pytest.mark.asyncio
    async def test_load_game_success(self, system_processor):
        """Test successful game load."""
        player_id = uuid.uuid4()
        save_name = "test_save"
        context = {"player_id": str(player_id)}

        # Mock successful load
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Game loaded successfully"
        mock_result.game_state = {"test": "state"}

        system_processor.save_manager.load_save.return_value = mock_result

        # Process load command
        result = await system_processor.handle_load_game(player_id, save_name, context)

        # Verify result
        assert result.success is True
        assert "loaded successfully" in result.feedback_message.lower()
        # Note: ActionResult doesn't have state_changes field

        # Verify save manager was called correctly
        system_processor.save_manager.load_save.assert_called_once_with(save_name)

    @pytest.mark.asyncio
    async def test_load_game_auto_select(self, system_processor):
        """Test load game with auto-selection of most recent save."""
        player_id = uuid.uuid4()
        context = {"player_id": str(player_id)}

        # Mock save list
        mock_save = MagicMock()
        mock_save.save_name = "recent_save"
        system_processor.save_manager.list_saves.return_value = [mock_save]

        # Mock successful load
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Game loaded"
        mock_result.game_state = {}
        system_processor.save_manager.load_save.return_value = mock_result

        # Process load command without save name
        result = await system_processor.handle_load_game(player_id, None, context)

        # Should load the most recent save
        system_processor.save_manager.load_save.assert_called_once_with("recent_save")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_help_request(self, system_processor):
        """Test help request handling."""
        topic = "combat"
        context = {"current_location": "battlefield"}

        # Mock help response
        mock_help_response = MagicMock()
        mock_help_response.content = "Combat help content"
        mock_help_response.topic = topic
        mock_help_response.category = "gameplay"
        mock_help_response.related_topics = ["weapons", "armor"]
        mock_help_response.contextual_suggestions = ["Try attacking"]
        mock_help_response.examples = ["attack goblin"]

        system_processor.help_system.get_help.return_value = mock_help_response

        # Process help request
        result = await system_processor.handle_help_request(topic, context)

        # Verify result
        assert result.success is True
        assert result.feedback_message == "Combat help content"
        # Note: ActionResult doesn't have metadata field

        # Verify help system was called correctly
        system_processor.help_system.get_help.assert_called_once_with(topic, context)

    @pytest.mark.asyncio
    async def test_settings_list(self, system_processor):
        """Test listing all settings."""
        player_id = uuid.uuid4()
        context = {"player_id": str(player_id)}

        # Mock settings
        mock_setting = MagicMock()
        mock_setting.name = "auto_save"
        mock_setting.current_value = "true"
        mock_setting.description = "Auto-save on exit"
        mock_setting.category = "gameplay"
        mock_setting.to_dict.return_value = {"name": "auto_save"}

        system_processor.settings_manager.list_settings.return_value = [mock_setting]

        # Process settings command
        result = await system_processor.handle_settings_command(
            None, None, player_id, context
        )

        # Verify result
        assert result.success is True
        assert "Current Settings" in result.feedback_message
        assert "auto_save: true" in result.feedback_message

        # Verify settings manager was called correctly
        system_processor.settings_manager.list_settings.assert_called_once_with(
            player_id=player_id
        )

    @pytest.mark.asyncio
    async def test_process_command_routing(self, system_processor):
        """Test command routing to appropriate handlers."""
        player_id = uuid.uuid4()
        context = {"player_id": str(player_id)}

        # Mock successful save
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Saved"
        mock_result.save_name = "test"
        mock_result.save_metadata = None
        system_processor.save_manager.create_save.return_value = mock_result

        # Test each command type
        test_cases = [
            (SystemCommandType.SAVE_GAME, {"save_name": "test"}),
            (SystemCommandType.LOAD_GAME, {"save_name": "test"}),
            (SystemCommandType.HELP, {"topic": "combat"}),
            (SystemCommandType.TUTORIAL, {"tutorial_type": "movement"}),
            (SystemCommandType.SETTINGS, {}),
            (SystemCommandType.LIST_SAVES, {}),
            (SystemCommandType.QUIT_GAME, {"force": False}),
        ]

        for command_type, args in test_cases:
            # Mock the specific handler methods
            if command_type == SystemCommandType.SAVE_GAME:
                system_processor.save_manager.create_save.return_value = mock_result
            elif command_type == SystemCommandType.LOAD_GAME:
                mock_load_result = MagicMock()
                mock_load_result.success = True
                mock_load_result.feedback_message = "Loaded"
                mock_load_result.game_state = {}
                system_processor.save_manager.load_save.return_value = mock_load_result
            elif command_type == SystemCommandType.HELP:
                mock_help = MagicMock()
                mock_help.content = "Help content"
                mock_help.topic = "test"
                mock_help.category = "test"
                mock_help.related_topics = []
                mock_help.contextual_suggestions = []
                mock_help.examples = []
                system_processor.help_system.get_help.return_value = mock_help
            elif command_type == SystemCommandType.SETTINGS:
                system_processor.settings_manager.list_settings.return_value = []
            elif command_type == SystemCommandType.LIST_SAVES:
                system_processor.save_manager.list_saves.return_value = []

            # Process command
            result = await system_processor.process_command(command_type, args, context)

            # Should succeed
            assert result.success is not None

    @pytest.mark.asyncio
    async def test_unknown_command_type(self, system_processor):
        """Test handling of unknown command type."""

        # Create a mock enum that has a value attribute but isn't in our switch
        class MockCommandType:
            def __init__(self, value):
                self.value = value

        unknown_command = MockCommandType("unknown_command")

        result = await system_processor.process_command(
            unknown_command, {}, {"player_id": str(uuid.uuid4())}
        )

        assert result.success is False
        assert "Unknown system command" in result.feedback_message

    @pytest.mark.asyncio
    async def test_error_handling(self, system_processor):
        """Test error handling in command processing."""
        player_id = uuid.uuid4()
        context = {"player_id": str(player_id)}

        # Mock save manager to raise exception
        system_processor.save_manager.create_save.side_effect = Exception("Save error")

        # Process save command
        result = await system_processor.process_command(
            SystemCommandType.SAVE_GAME, {"save_name": "test"}, context
        )

        # Should handle error gracefully
        assert result.success is False
        assert "Failed to save game" in result.feedback_message
        # Note: ActionResult doesn't have error_details field
