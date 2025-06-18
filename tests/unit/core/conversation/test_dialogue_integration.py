"""Tests for conversation-dialogue template integration."""

from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from game_loop.core.conversation.conversation_models import (
    ConversationExchange,
    MessageType,
)
from game_loop.core.output_generator import OutputGenerator
from game_loop.core.template_manager import TemplateManager


class TestDialogueTemplateIntegration:
    """Test integration between conversation models and dialogue templates."""

    @pytest.fixture
    def console(self):
        """Create a mock console for testing."""
        return Mock(spec=Console)

    @pytest.fixture
    def template_manager(self):
        """Create a template manager for testing."""
        return TemplateManager("templates")

    @pytest.fixture
    def output_generator(self, console):
        """Create an output generator for testing."""
        return OutputGenerator(console, "templates")

    def test_basic_dialogue_rendering(self, output_generator, console):
        """Test basic dialogue rendering with template."""
        # Test simple dialogue without NPC data
        output_generator.format_dialogue("guard", "Halt! Who goes there?")

        # Verify console.print was called
        console.print.assert_called_once()

        # Get the printed content
        printed_content = console.print.call_args[0][0]

        # Should contain the speaker name and message
        assert "guard" in printed_content
        assert "Halt! Who goes there?" in printed_content
        assert "[bold magenta]" in printed_content

    def test_dialogue_with_npc_data(self, output_generator, console):
        """Test dialogue rendering with NPC data."""
        npc_data = {"name": "Captain Marcus", "title": "Guard Captain"}

        output_generator.format_dialogue(
            "guard_captain", "State your business here!", npc_data
        )

        console.print.assert_called_once()
        printed_content = console.print.call_args[0][0]

        # Should use NPC name from npc_data instead of speaker ID
        assert "Captain Marcus" in printed_content
        assert "State your business here!" in printed_content

    def test_dialogue_from_exchange_basic(self, output_generator, console):
        """Test dialogue rendering from ConversationExchange."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="guard_captain",
            message_text="Welcome to the castle.",
            message_type=MessageType.GREETING,
        )

        output_generator.format_dialogue_from_exchange(exchange)

        console.print.assert_called_once()
        printed_content = console.print.call_args[0][0]

        assert "guard_captain" in printed_content
        assert "Welcome to the castle." in printed_content

    def test_dialogue_from_exchange_with_emotion(self, output_generator, console):
        """Test dialogue rendering with emotion from ConversationExchange."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="guard",
            message_text="You shall not pass!",
            message_type=MessageType.STATEMENT,
            emotion="stern",
        )

        npc_data = {"name": "Guard Thompson"}
        output_generator.format_dialogue_from_exchange(exchange, npc_data)

        console.print.assert_called_once()
        printed_content = console.print.call_args[0][0]

        assert "Guard Thompson" in printed_content
        assert "You shall not pass!" in printed_content
        assert "(stern)" in printed_content

    def test_dialogue_with_internal_thought(self, output_generator, console):
        """Test dialogue rendering with internal thought metadata."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="merchant",
            message_text="That's a fair price.",
            metadata={"internal_thought": "I could probably get more for this..."},
        )

        output_generator.format_dialogue_from_exchange(exchange)

        console.print.assert_called_once()
        printed_content = console.print.call_args[0][0]

        assert "That's a fair price." in printed_content
        assert "I could probably get more for this..." in printed_content
        assert "thinks:" in printed_content

    def test_dialogue_with_player_exchange(self, output_generator, console):
        """Test dialogue rendering for player messages."""
        exchange = ConversationExchange.create_player_message(
            player_id="player_001",
            message_text="I'm looking for information about the temple.",
            message_type=MessageType.QUESTION,
        )

        output_generator.format_dialogue_from_exchange(exchange)

        console.print.assert_called_once()
        printed_content = console.print.call_args[0][0]

        assert "player_001" in printed_content
        assert "I'm looking for information about the temple." in printed_content

    def test_dialogue_template_fallback(self, output_generator, console):
        """Test fallback when template fails."""
        # Mock template manager to return None (template failure)
        with patch.object(
            output_generator.template_manager, "render_template", return_value=None
        ):
            output_generator.format_dialogue("guard", "Hello there!")

            # Should still call console.print (fallback to ResponseFormatter)
            console.print.assert_called_once()

            # Verify fallback was used - the call should have been made
            # Note: ResponseFormatter returns Rich Panel objects, so we just verify the call happened
            assert console.print.call_count == 1

    def test_rich_markup_filter(self, template_manager):
        """Test that rich_markup filter is available and works."""
        # Test that the filter exists
        assert "rich_markup" in template_manager.env.filters

        # Test the filter function
        filter_func = template_manager.env.filters["rich_markup"]

        # Test with normal text
        result = filter_func("Hello world")
        assert result == "Hello world"

        # Test with non-string input
        result = filter_func(123)
        assert result == "123"

    def test_template_context_variables(self, template_manager):
        """Test that all expected context variables work in template."""
        context = {
            "text": "Test message",
            "speaker": "test_speaker",
            "npc_data": {"name": "Test NPC"},
            "emotion": "happy",
            "metadata": {"internal_thought": "Test thought"},
        }

        rendered = template_manager.render_template("dialogue/speech.j2", context)

        assert rendered is not None
        assert "Test NPC" in rendered
        assert "Test message" in rendered
        assert "(happy)" in rendered
        assert "Test thought" in rendered

    def test_error_handling_in_dialogue_formatting(self, output_generator, console):
        """Test error handling when dialogue formatting fails."""
        # Create an exchange that might cause issues
        exchange = ConversationExchange.create_npc_message(
            npc_id="test_npc", message_text="Test message"
        )

        # Mock template manager to raise an exception
        with patch.object(
            output_generator.template_manager,
            "render_template",
            side_effect=Exception("Template error"),
        ):
            # Should handle the error gracefully
            output_generator.format_dialogue_from_exchange(exchange)

            # Should have printed an error message
            console.print.assert_called_once()
            printed_content = console.print.call_args[0][0]
            assert "Error displaying dialogue" in printed_content
            assert "[red]" in printed_content
