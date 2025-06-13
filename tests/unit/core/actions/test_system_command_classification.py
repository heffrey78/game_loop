"""Tests for system command classification in ActionTypeClassifier."""

import pytest

from game_loop.config.manager import ConfigManager
from game_loop.core.actions.action_classifier import ActionTypeClassifier
from game_loop.core.models.system_models import SystemCommandType


@pytest.fixture
def action_classifier():
    """Create ActionTypeClassifier for testing."""
    config_manager = ConfigManager()
    return ActionTypeClassifier(config_manager=config_manager, enable_cache=False)


class TestSystemCommandClassification:
    """Test cases for system command classification."""

    @pytest.mark.asyncio
    async def test_save_game_patterns(self, action_classifier):
        """Test save game command pattern recognition."""
        test_inputs = [
            "save game",
            "save",
            "save game as my_save",
            "create save",
            "save state",
            "quicksave",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.SAVE_GAME
            assert result.confidence == 0.9
            assert result.original_text == input_text

    @pytest.mark.asyncio
    async def test_load_game_patterns(self, action_classifier):
        """Test load game command pattern recognition."""
        test_inputs = [
            "load game",
            "load",
            "load my_save",
            "restore save",
            "continue game",
            "quickload",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.LOAD_GAME
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_help_patterns(self, action_classifier):
        """Test help command pattern recognition."""
        test_inputs = [
            "help",
            "help combat",
            "how do i attack",
            "what can i do",
            "commands",
            "?",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.HELP
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_tutorial_patterns(self, action_classifier):
        """Test tutorial command pattern recognition."""
        test_inputs = [
            "tutorial",
            "guide",
            "show me how",
            "teach me",
            "learn",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.TUTORIAL
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_settings_patterns(self, action_classifier):
        """Test settings command pattern recognition."""
        test_inputs = [
            "settings",
            "options",
            "preferences",
            "config",
            "set auto_save true",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.SETTINGS
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_quit_patterns(self, action_classifier):
        """Test quit game command pattern recognition."""
        test_inputs = [
            "quit",
            "quit game",
            "exit",
            "leave",
            "stop",
            "end game",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.QUIT_GAME
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_list_saves_patterns(self, action_classifier):
        """Test list saves command pattern recognition."""
        test_inputs = [
            "list saves",
            "show saves",
            "save files",
            "my saves",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None, f"Failed to classify: {input_text}"
            assert result.command_type == SystemCommandType.LIST_SAVES
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_argument_extraction(self, action_classifier):
        """Test argument extraction from system commands."""
        # Test save with name
        result = await action_classifier.classify_system_command(
            "save game as my_epic_save"
        )
        assert result is not None
        assert result.command_type == SystemCommandType.SAVE_GAME
        assert result.args.get("save_name") == "my_epic_save"

        # Test load with name
        result = await action_classifier.classify_system_command("load my_epic_save")
        assert result is not None
        assert result.command_type == SystemCommandType.LOAD_GAME
        assert result.args.get("save_name") == "my_epic_save"

        # Test help with topic
        result = await action_classifier.classify_system_command("help combat system")
        assert result is not None
        assert result.command_type == SystemCommandType.HELP
        assert result.args.get("topic") == "combat system"

        # Test settings with value
        result = await action_classifier.classify_system_command("set auto_save true")
        assert result is not None
        assert result.command_type == SystemCommandType.SETTINGS
        assert result.args.get("setting") == "auto_save"
        assert result.args.get("value") == "true"

    @pytest.mark.asyncio
    async def test_non_system_commands(self, action_classifier):
        """Test that non-system commands return None."""
        test_inputs = [
            "go north",
            "attack goblin",
            "take sword",
            "look around",
            "talk to merchant",
            "examine door",
        ]

        for input_text in test_inputs:
            result = await action_classifier.classify_system_command(input_text)
            assert (
                result is None
            ), f"Incorrectly classified as system command: {input_text}"

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, action_classifier):
        """Test that command matching is case insensitive."""
        test_cases = [
            ("SAVE GAME", SystemCommandType.SAVE_GAME),
            ("Help", SystemCommandType.HELP),
            ("QUIT", SystemCommandType.QUIT_GAME),
            ("Settings", SystemCommandType.SETTINGS),
        ]

        for input_text, expected_type in test_cases:
            result = await action_classifier.classify_system_command(input_text)
            assert result is not None
            assert result.command_type == expected_type

    @pytest.mark.asyncio
    async def test_error_handling(self, action_classifier):
        """Test error handling in system command classification."""
        # Test with None input
        result = await action_classifier.classify_system_command(None)
        assert result is None

        # Test with empty string
        result = await action_classifier.classify_system_command("")
        assert result is None

        # Test with whitespace only
        result = await action_classifier.classify_system_command("   ")
        assert result is None
