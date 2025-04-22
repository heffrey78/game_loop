"""
Tests for the InputProcessor class.
"""

from unittest.mock import Mock

import pytest

from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand


class TestInputProcessor:
    """Test cases for the InputProcessor class."""

    @pytest.fixture
    def input_processor(self):
        """Create an InputProcessor instance for testing."""
        console_mock = Mock()
        return InputProcessor(console=console_mock)

    def test_normalize_input(self, input_processor):
        """Test that input normalization works correctly."""
        # Test basic normalization
        assert input_processor._normalize_input("LOOK") == "look"

        # Test with extra whitespace
        assert input_processor._normalize_input("  examine   book  ") == "examine book"

        # Test with empty input
        assert input_processor._normalize_input("") == ""
        assert input_processor._normalize_input("   ") == ""

    def test_movement_commands(self, input_processor):
        """Test that movement commands are parsed correctly."""
        # Test cardinal directions
        north_cmd = input_processor.process_input("north")
        assert north_cmd.command_type == CommandType.MOVEMENT
        assert north_cmd.action == "go"
        assert north_cmd.subject == "north"

        # Test abbreviated directions
        south_cmd = input_processor.process_input("s")
        assert south_cmd.command_type == CommandType.MOVEMENT
        assert south_cmd.action == "go"
        assert south_cmd.subject == "south"

        # Test with "go" prefix
        east_cmd = input_processor.process_input("go east")
        assert east_cmd.command_type == CommandType.MOVEMENT
        assert east_cmd.action == "go"
        assert east_cmd.subject == "east"

        # Test case insensitivity
        west_cmd = input_processor.process_input("WEST")
        assert west_cmd.command_type == CommandType.MOVEMENT
        assert west_cmd.action == "go"
        assert west_cmd.subject == "west"

    def test_look_commands(self, input_processor):
        """Test that look commands are parsed correctly."""
        # Test basic look command
        look_cmd = input_processor.process_input("look")
        assert look_cmd.command_type == CommandType.LOOK
        assert look_cmd.action == "look"

        # Test abbreviated look command
        l_cmd = input_processor.process_input("l")
        assert l_cmd.command_type == CommandType.LOOK
        assert l_cmd.action == "look"

        # Test with variation
        look_cmd = input_processor.process_input("look around")
        assert look_cmd.command_type == CommandType.LOOK
        assert look_cmd.action == "look"

    def test_inventory_commands(self, input_processor):
        """Test that inventory commands are parsed correctly."""
        # Test basic inventory command
        inv_cmd = input_processor.process_input("inventory")
        assert inv_cmd.command_type == CommandType.INVENTORY
        assert inv_cmd.action == "inventory"

        # Test abbreviated inventory command
        i_cmd = input_processor.process_input("i")
        assert i_cmd.command_type == CommandType.INVENTORY
        assert i_cmd.action == "inventory"

    def test_help_commands(self, input_processor):
        """Test that help commands are parsed correctly."""
        # Test basic help command
        help_cmd = input_processor.process_input("help")
        assert help_cmd.command_type == CommandType.HELP
        assert help_cmd.action == "help"

        # Test abbreviated help command
        h_cmd = input_processor.process_input("h")
        assert h_cmd.command_type == CommandType.HELP
        assert h_cmd.action == "help"

        # Test question mark
        q_cmd = input_processor.process_input("?")
        assert q_cmd.command_type == CommandType.HELP
        assert q_cmd.action == "help"

    def test_quit_commands(self, input_processor):
        """Test that quit commands are parsed correctly."""
        # Test basic quit command
        quit_cmd = input_processor.process_input("quit")
        assert quit_cmd.command_type == CommandType.QUIT
        assert quit_cmd.action == "quit"

        # Test exit command
        exit_cmd = input_processor.process_input("exit")
        assert exit_cmd.command_type == CommandType.QUIT
        assert exit_cmd.action == "quit"

        # Test abbreviated quit command
        q_cmd = input_processor.process_input("q")
        assert q_cmd.command_type == CommandType.QUIT
        assert q_cmd.action == "quit"

    def test_take_commands(self, input_processor):
        """Test that take commands are parsed correctly."""
        # Test basic take command
        take_cmd = input_processor.process_input("take key")
        assert take_cmd.command_type == CommandType.TAKE
        assert take_cmd.action == "take"
        assert take_cmd.subject == "key"

        # Test with synonym
        get_cmd = input_processor.process_input("get book")
        assert get_cmd.command_type == CommandType.TAKE
        assert get_cmd.action == "take"
        assert get_cmd.subject == "book"

        # Test with multi-word object
        take_multi_cmd = input_processor.process_input("take rusty sword")
        assert take_multi_cmd.command_type == CommandType.TAKE
        assert take_multi_cmd.action == "take"
        assert take_multi_cmd.subject == "rusty sword"

    def test_drop_commands(self, input_processor):
        """Test that drop commands are parsed correctly."""
        # Test basic drop command
        drop_cmd = input_processor.process_input("drop key")
        assert drop_cmd.command_type == CommandType.DROP
        assert drop_cmd.action == "drop"
        assert drop_cmd.subject == "key"

        # Test with synonym
        discard_cmd = input_processor.process_input("discard map")
        assert discard_cmd.command_type == CommandType.DROP
        assert discard_cmd.action == "drop"
        assert discard_cmd.subject == "map"

    def test_use_commands(self, input_processor):
        """Test that use commands are parsed correctly."""
        # Test basic use command
        use_cmd = input_processor.process_input("use key")
        assert use_cmd.command_type == CommandType.USE
        assert use_cmd.action == "use"
        assert use_cmd.subject == "key"
        assert use_cmd.target is None

        # Test use with target (on)
        use_on_cmd = input_processor.process_input("use key on door")
        assert use_on_cmd.command_type == CommandType.USE
        assert use_on_cmd.action == "use"
        assert use_on_cmd.subject == "key"
        assert use_on_cmd.target == "door"

        # Test use with target (with)
        use_with_cmd = input_processor.process_input("use hammer with nail")
        assert use_with_cmd.command_type == CommandType.USE
        assert use_with_cmd.action == "use"
        assert use_with_cmd.subject == "hammer"
        assert use_with_cmd.target == "nail"

    def test_examine_commands(self, input_processor):
        """Test that examine commands are parsed correctly."""
        # Test basic examine command
        examine_cmd = input_processor.process_input("examine painting")
        assert examine_cmd.command_type == CommandType.EXAMINE
        assert examine_cmd.action == "examine"
        assert examine_cmd.subject == "painting"

        # Test with synonym
        inspect_cmd = input_processor.process_input("inspect statue")
        assert inspect_cmd.command_type == CommandType.EXAMINE
        assert inspect_cmd.action == "examine"
        assert inspect_cmd.subject == "statue"

        # Test with "look at" variation
        look_at_cmd = input_processor.process_input("look at inscription")
        assert look_at_cmd.command_type == CommandType.EXAMINE
        assert look_at_cmd.action == "examine"
        assert look_at_cmd.subject == "inscription"

    def test_talk_commands(self, input_processor):
        """Test that talk commands are parsed correctly."""
        # Test basic talk command
        talk_cmd = input_processor.process_input("talk to innkeeper")
        assert talk_cmd.command_type == CommandType.TALK
        assert talk_cmd.action == "talk"
        assert talk_cmd.subject == "to innkeeper"

        # Test with synonym
        speak_cmd = input_processor.process_input("speak with wizard")
        assert speak_cmd.command_type == CommandType.TALK
        assert speak_cmd.action == "talk"
        assert speak_cmd.subject == "with wizard"

    def test_unknown_commands(self, input_processor):
        """Test that unknown commands are handled correctly."""
        # Test completely unknown command
        unknown_cmd = input_processor.process_input("dance")
        assert unknown_cmd.command_type == CommandType.UNKNOWN
        assert unknown_cmd.action == "unknown"
        assert unknown_cmd.subject == "dance"

        # Test command that looks like known pattern but isn't
        not_take_cmd = input_processor.process_input("taken aback")
        assert not_take_cmd.command_type == CommandType.UNKNOWN
        assert not_take_cmd.action == "unknown"
        assert not_take_cmd.subject == "taken aback"

    def test_empty_input(self, input_processor):
        """Test handling of empty input."""
        empty_cmd = input_processor.process_input("")
        assert empty_cmd.command_type == CommandType.UNKNOWN
        assert empty_cmd.action == "unknown"
        assert empty_cmd.subject is None

    def test_command_suggestions(self, input_processor):
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

    def test_error_message_formatting(self, input_processor):
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
