"""
Tests for the InputProcessor class.
"""

from unittest.mock import Mock

import pytest

from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand


class TestInputProcessor:
    """Test cases for the InputProcessor class."""

    @pytest.fixture
    def input_processor(self) -> InputProcessor:
        """Create an InputProcessor instance for testing."""
        console_mock = Mock()
        return InputProcessor(console=console_mock)

    def test_normalize_input(self, input_processor: InputProcessor) -> None:
        """Test that input normalization works correctly."""
        # Test basic normalization
        assert input_processor._normalize_input("LOOK") == "look"

        # Test with extra whitespace
        assert input_processor._normalize_input("  examine   book  ") == "examine book"

        # Test with empty input
        assert input_processor._normalize_input("") == ""
        assert input_processor._normalize_input("   ") == ""

    @pytest.mark.asyncio
    async def test_movement_commands(self, input_processor: InputProcessor) -> None:
        """Test that movement commands are parsed correctly."""
        # Test cardinal directions
        north_cmd = await input_processor.process_input_async("north")
        assert north_cmd.command_type == CommandType.MOVEMENT
        assert north_cmd.action == "go"
        assert north_cmd.subject == "north"

        # Test abbreviated directions
        south_cmd = await input_processor.process_input_async("s")
        assert south_cmd.command_type == CommandType.MOVEMENT
        assert south_cmd.action == "go"
        assert south_cmd.subject == "south"

        # Test with "go" prefix
        east_cmd = await input_processor.process_input_async("go east")
        assert east_cmd.command_type == CommandType.MOVEMENT
        assert east_cmd.action == "go"
        assert east_cmd.subject == "east"

        # Test case insensitivity
        west_cmd = await input_processor.process_input_async("WEST")
        assert west_cmd.command_type == CommandType.MOVEMENT
        assert west_cmd.action == "go"
        assert west_cmd.subject == "west"

    @pytest.mark.asyncio
    async def test_look_commands(self, input_processor: InputProcessor) -> None:
        """Test that look commands are parsed correctly."""
        # Test basic look command
        look_cmd = await input_processor.process_input_async("look")
        assert look_cmd.command_type == CommandType.LOOK
        assert look_cmd.action == "look"

        # Test abbreviated look command
        l_cmd = await input_processor.process_input_async("l")
        assert l_cmd.command_type == CommandType.LOOK
        assert l_cmd.action == "look"

        # Test with variation
        look_cmd = await input_processor.process_input_async("look around")
        assert look_cmd.command_type == CommandType.LOOK
        assert look_cmd.action == "look"

    @pytest.mark.asyncio
    async def test_inventory_commands(self, input_processor: InputProcessor) -> None:
        """Test that inventory commands are parsed correctly."""
        # Test basic inventory command
        inv_cmd = await input_processor.process_input_async("inventory")
        assert inv_cmd.command_type == CommandType.INVENTORY
        assert inv_cmd.action == "inventory"

        # Test abbreviated inventory command
        i_cmd = await input_processor.process_input_async("i")
        assert i_cmd.command_type == CommandType.INVENTORY
        assert i_cmd.action == "inventory"

    @pytest.mark.asyncio
    async def test_help_commands(self, input_processor: InputProcessor) -> None:
        """Test that help commands are parsed correctly."""
        # Test basic help command
        help_cmd = await input_processor.process_input_async("help")
        assert help_cmd.command_type == CommandType.HELP
        assert help_cmd.action == "help"

        # Test abbreviated help command
        h_cmd = await input_processor.process_input_async("h")
        assert h_cmd.command_type == CommandType.HELP
        assert h_cmd.action == "help"

        # Test question mark
        q_cmd = await input_processor.process_input_async("?")
        assert q_cmd.command_type == CommandType.HELP
        assert q_cmd.action == "help"

    @pytest.mark.asyncio
    async def test_quit_commands(self, input_processor: InputProcessor) -> None:
        """Test that quit commands are parsed correctly."""
        # Test basic quit command
        quit_cmd = await input_processor.process_input_async("quit")
        assert quit_cmd.command_type == CommandType.QUIT
        assert quit_cmd.action == "quit"

        # Test exit command
        exit_cmd = await input_processor.process_input_async("exit")
        assert exit_cmd.command_type == CommandType.QUIT
        assert exit_cmd.action == "quit"

        # Test abbreviated quit command
        q_cmd = await input_processor.process_input_async("q")
        assert q_cmd.command_type == CommandType.QUIT
        assert q_cmd.action == "quit"

    @pytest.mark.asyncio
    async def test_take_commands(self, input_processor: InputProcessor) -> None:
        """Test that take commands are parsed correctly."""
        # Test basic take command
        take_cmd = await input_processor.process_input_async("take key")
        assert take_cmd.command_type == CommandType.TAKE
        assert take_cmd.action == "take"
        assert take_cmd.subject == "key"

        # Test with synonym
        get_cmd = await input_processor.process_input_async("get book")
        assert get_cmd.command_type == CommandType.TAKE
        assert get_cmd.action == "take"
        assert get_cmd.subject == "book"

        # Test with multi-word object
        take_multi_cmd = await input_processor.process_input_async("take rusty sword")
        assert take_multi_cmd.command_type == CommandType.TAKE
        assert take_multi_cmd.action == "take"
        assert take_multi_cmd.subject == "rusty sword"

    @pytest.mark.asyncio
    async def test_drop_commands(self, input_processor: InputProcessor) -> None:
        """Test that drop commands are parsed correctly."""
        # Test basic drop command
        drop_cmd = await input_processor.process_input_async("drop key")
        assert drop_cmd.command_type == CommandType.DROP
        assert drop_cmd.action == "drop"
        assert drop_cmd.subject == "key"

        # Test with synonym
        discard_cmd = await input_processor.process_input_async("discard map")
        assert discard_cmd.command_type == CommandType.DROP
        assert discard_cmd.action == "drop"
        assert discard_cmd.subject == "map"

    @pytest.mark.asyncio
    async def test_use_commands(self, input_processor: InputProcessor) -> None:
        """Test that use commands are parsed correctly."""
        # Test basic use command
        use_cmd = await input_processor.process_input_async("use key")
        assert use_cmd.command_type == CommandType.USE
        assert use_cmd.action == "use"
        assert use_cmd.subject == "key"
        assert use_cmd.target is None

        # Test use with target (on)
        use_on_cmd = await input_processor.process_input_async("use key on door")
        assert use_on_cmd.command_type == CommandType.USE
        assert use_on_cmd.action == "use"
        assert use_on_cmd.subject == "key"
        assert use_on_cmd.target == "door"

        # Test use with target (with)
        use_with_cmd = await input_processor.process_input_async("use hammer with nail")
        assert use_with_cmd.command_type == CommandType.USE
        assert use_with_cmd.action == "use"
        assert use_with_cmd.subject == "hammer"
        assert use_with_cmd.target == "nail"

    @pytest.mark.asyncio
    async def test_examine_commands(self, input_processor: InputProcessor) -> None:
        """Test that examine commands are parsed correctly."""
        # Test basic examine command
        examine_cmd = await input_processor.process_input_async("examine painting")
        assert examine_cmd.command_type == CommandType.EXAMINE
        assert examine_cmd.action == "examine"
        assert examine_cmd.subject == "painting"

        # Test with synonym
        inspect_cmd = await input_processor.process_input_async("inspect statue")
        assert inspect_cmd.command_type == CommandType.EXAMINE
        assert inspect_cmd.action == "examine"
        assert inspect_cmd.subject == "statue"

        # Test with "look at" variation
        look_at_cmd = await input_processor.process_input_async("look at inscription")
        assert look_at_cmd.command_type == CommandType.EXAMINE
        assert look_at_cmd.action == "examine"
        assert look_at_cmd.subject == "inscription"

    @pytest.mark.asyncio
    async def test_talk_commands(self, input_processor: InputProcessor) -> None:
        """Test that talk commands are parsed correctly."""
        # Test basic talk command
        talk_cmd = await input_processor.process_input_async("talk to innkeeper")
        assert talk_cmd.command_type == CommandType.TALK
        assert talk_cmd.action == "talk"
        assert talk_cmd.subject == "to innkeeper"

        # Test with synonym
        speak_cmd = await input_processor.process_input_async("speak with wizard")
        assert speak_cmd.command_type == CommandType.TALK
        assert speak_cmd.action == "talk"
        assert speak_cmd.subject == "with wizard"

    @pytest.mark.asyncio
    async def test_unknown_commands(self, input_processor: InputProcessor) -> None:
        """Test that unknown commands are handled correctly."""
        # Test completely unknown command
        unknown_cmd = await input_processor.process_input_async("dance")
        assert unknown_cmd.command_type == CommandType.UNKNOWN
        assert unknown_cmd.action == "unknown"
        assert unknown_cmd.subject == "dance"

        # Test command that looks like known pattern but isn't
        not_take_cmd = await input_processor.process_input_async("taken aback")
        assert not_take_cmd.command_type == CommandType.UNKNOWN
        assert not_take_cmd.action == "unknown"
        assert not_take_cmd.subject == "taken aback"

    @pytest.mark.asyncio
    async def test_empty_input(self, input_processor: InputProcessor) -> None:
        """Test handling of empty input."""
        empty_cmd = await input_processor.process_input_async("")
        assert empty_cmd.command_type == CommandType.UNKNOWN
        assert empty_cmd.action == "unknown"
        assert empty_cmd.subject is None

    def test_command_suggestions(self, input_processor: InputProcessor) -> None:
        """Test that command suggestions work correctly."""
        # Test partial direction
        north_suggestions = input_processor.generate_command_suggestions("no")
        assert "north" in north_suggestions
        assert len(north_suggestions) >= 1

        # Test partial action
        take_suggestions = input_processor.generate_command_suggestions("ta")
        assert "take [object]" in take_suggestions

        # Test empty input suggestions
        empty_suggestions = input_processor.generate_command_suggestions("")
        assert len(empty_suggestions) == 0

    def test_error_message_formatting(self, input_processor: InputProcessor) -> None:
        """Test that error messages are formatted correctly."""
        # Test unknown command
        unknown_cmd = ParsedCommand(
            command_type=CommandType.UNKNOWN, action="unknown", subject="somethingweird"
        )
        error_message = input_processor.format_error_message(unknown_cmd)
        assert "somethingweird" in error_message
        assert "help" in error_message.lower()

        # Test general error message
        help_cmd = ParsedCommand(command_type=CommandType.HELP, action="help")
        general_message = input_processor.format_error_message(help_cmd)
        assert "help" in general_message.lower()
