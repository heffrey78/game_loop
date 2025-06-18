"""Tests for conversation system output generation."""

from unittest.mock import Mock

import pytest

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    ConversationResult,
    MessageType,
    NPCPersonality,
)
from game_loop.core.output_generator import OutputGenerator


class TestConversationOutput:
    """Test conversation system output generation."""

    @pytest.fixture
    def output_generator(self):
        """Create an output generator for testing."""
        console = Mock()
        return OutputGenerator(console, "templates")

    def test_conversation_result_to_output(self):
        """Test converting ConversationResult to formatted output."""
        npc_response = ConversationExchange.create_npc_message(
            npc_id="elder",
            message_text="Welcome, young one. What brings you to our village?",
            emotion="welcoming",
        )

        result = ConversationResult.success_result(
            npc_response=npc_response, relationship_change=0.1, mood_change="friendly"
        )

        # Test that result contains proper dialogue exchange
        assert (
            result.npc_response.message_text
            == "Welcome, young one. What brings you to our village?"
        )
        assert result.npc_response.emotion == "welcoming"
        assert result.relationship_change == 0.1
        assert result.mood_change == "friendly"

    def test_npc_personality_to_dialogue_context(self):
        """Test that NPC personality data can be used in dialogue context."""
        personality = NPCPersonality(
            npc_id="wise_elder",
            traits={"wise": 0.9, "patient": 0.8, "cryptic": 0.6},
            knowledge_areas=["history", "lore", "ancient_secrets"],
            speech_patterns={"formality": "high", "verbosity": "medium"},
            relationships={"player_001": 0.3},
            background_story="An ancient keeper of village lore.",
            default_mood="contemplative",
        )

        # Convert personality to NPC data for dialogue
        npc_data = {
            "name": "Elder Thornwick",
            "personality": personality.traits,
            "speech_style": personality.speech_patterns,
            "mood": personality.default_mood,
        }

        # Verify the data structure is suitable for templates
        assert npc_data["name"] == "Elder Thornwick"
        assert npc_data["personality"]["wise"] == 0.9
        assert npc_data["speech_style"]["formality"] == "high"

    def test_conversation_context_to_output_data(self):
        """Test converting ConversationContext to output-ready data."""
        context = ConversationContext.create(
            player_id="player_001",
            npc_id="guard_captain",
            topic="castle_security",
            initial_mood="professional",
            relationship_level=0.3,
        )

        # Add some exchanges
        player_msg = ConversationExchange.create_player_message(
            player_id="player_001",
            message_text="Can you tell me about the security here?",
        )
        context.add_exchange(player_msg)

        npc_response = ConversationExchange.create_npc_message(
            npc_id="guard_captain",
            message_text="Security is tight. We keep watch day and night.",
            emotion="serious",
        )
        context.add_exchange(npc_response)

        # Test that context contains dialogue data
        assert len(context.conversation_history) == 2
        assert context.topic == "castle_security"
        assert context.mood == "professional"

        recent_exchanges = context.get_recent_exchanges(2)
        assert len(recent_exchanges) == 2
        assert recent_exchanges[1].emotion == "serious"

    def test_dialogue_output_with_conversation_context(self, output_generator):
        """Test dialogue output using conversation context."""
        # Create a conversation exchange with rich context
        exchange = ConversationExchange.create_npc_message(
            npc_id="blacksmith",
            message_text="Aye, I can forge that for you, but it'll cost extra.",
            message_type=MessageType.STATEMENT,
            emotion="gruff",
            metadata={
                "internal_thought": "This customer seems wealthy enough...",
                "gesture": "wipes hands on apron",
            },
        )

        npc_data = {
            "name": "Marcus the Blacksmith",
            "title": "Village Blacksmith",
            "mood": "busy",
        }

        # Test the output generation
        output_generator.format_dialogue_from_exchange(exchange, npc_data)

        # Verify console was called
        output_generator.console.print.assert_called_once()

    def test_multi_exchange_conversation_output(self, output_generator):
        """Test output for a multi-exchange conversation."""
        # Create a conversation context with multiple exchanges
        context = ConversationContext.create(player_id="player_001", npc_id="merchant")

        exchanges = [
            ConversationExchange.create_player_message(
                player_id="player_001",
                message_text="What do you have for sale?",
                message_type=MessageType.QUESTION,
            ),
            ConversationExchange.create_npc_message(
                npc_id="merchant",
                message_text="I have fine weapons and armor, traveler.",
                message_type=MessageType.STATEMENT,
                emotion="friendly",
            ),
            ConversationExchange.create_player_message(
                player_id="player_001",
                message_text="Show me your best sword.",
                message_type=MessageType.STATEMENT,
            ),
            ConversationExchange.create_npc_message(
                npc_id="merchant",
                message_text="Ah, this blade here is enchanted steel!",
                message_type=MessageType.STATEMENT,
                emotion="excited",
                metadata={"gesture": "holds up gleaming sword"},
            ),
        ]

        # Add exchanges to context
        for exchange in exchanges:
            context.add_exchange(exchange)

        # Test outputting the most recent exchange
        latest_exchange = context.get_recent_exchanges(1)[0]
        npc_data = {"name": "Gareth the Merchant"}

        output_generator.format_dialogue_from_exchange(latest_exchange, npc_data)

        # Verify output was generated
        output_generator.console.print.assert_called_once()
        printed_content = output_generator.console.print.call_args[0][0]

        assert "Gareth the Merchant" in printed_content
        assert "enchanted steel" in printed_content
        assert "(excited)" in printed_content

    def test_error_result_handling(self):
        """Test handling of conversation error results."""
        error_result = ConversationResult.error_result(
            error_message="NPC not found",
            errors=["Invalid NPC ID", "Database connection failed"],
        )

        assert error_result.success is False
        assert error_result.npc_response is None
        assert "Invalid NPC ID" in error_result.errors
        assert "Database connection failed" in error_result.errors

    def test_conversation_state_tracking_for_output(self):
        """Test that conversation state is properly tracked for output generation."""
        context = ConversationContext.create(
            player_id="player_001", npc_id="innkeeper", initial_mood="welcoming"
        )

        # Simulate a conversation that changes mood
        context.update_mood("suspicious")
        context.update_relationship(-0.1)

        # Add an exchange that reflects the mood change
        exchange = ConversationExchange.create_npc_message(
            npc_id="innkeeper",
            message_text="I'm not sure I trust you...",
            emotion="wary",
        )
        context.add_exchange(exchange)

        # Verify state changes are tracked
        assert context.mood == "suspicious"
        assert context.relationship_level == -0.1
        assert context.get_exchange_count() == 1

        # The exchange should reflect the current mood
        latest_exchange = context.get_recent_exchanges(1)[0]
        assert latest_exchange.emotion == "wary"

    def test_template_variable_mapping(self, output_generator):
        """Test that all conversation data maps correctly to template variables."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="sage",
            message_text="The ancient texts speak of great power...",
            message_type=MessageType.STATEMENT,
            emotion="mysterious",
            metadata={
                "internal_thought": "Should I reveal more?",
                "knowledge_area": "ancient_lore",
                "confidence": 0.8,
            },
        )

        npc_data = {
            "name": "Sage Aldric",
            "title": "Keeper of Ancient Knowledge",
            "personality": {"wise": 0.9, "secretive": 0.7},
        }

        # Test that format_dialogue_from_exchange handles all the data
        output_generator.format_dialogue_from_exchange(exchange, npc_data)

        # Verify the call was made (detailed template testing is in template tests)
        output_generator.console.print.assert_called_once()
