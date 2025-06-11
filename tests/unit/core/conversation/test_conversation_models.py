"""Tests for conversation system models."""

import pytest
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    ConversationResult,
    ConversationStatus,
    MessageType,
    NPCPersonality,
)


class TestConversationExchange:
    """Test ConversationExchange model."""

    def test_create_player_message(self):
        """Test creating a player message exchange."""
        exchange = ConversationExchange.create_player_message(
            player_id="test_player",
            message_text="Hello, guard!",
            message_type=MessageType.GREETING,
        )

        assert exchange.speaker_id == "test_player"
        assert exchange.message_text == "Hello, guard!"
        assert exchange.message_type == MessageType.GREETING
        assert exchange.emotion is None
        assert exchange.exchange_id is not None
        assert exchange.timestamp is not None

    def test_create_npc_message(self):
        """Test creating an NPC message exchange."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="guard_captain",
            message_text="Greetings, traveler. How may I assist you?",
            message_type=MessageType.GREETING,
            emotion="professional",
        )

        assert exchange.speaker_id == "guard_captain"
        assert exchange.message_text == "Greetings, traveler. How may I assist you?"
        assert exchange.message_type == MessageType.GREETING
        assert exchange.emotion == "professional"
        assert exchange.exchange_id is not None

    def test_exchange_to_dict(self):
        """Test converting exchange to dictionary."""
        exchange = ConversationExchange.create_player_message(
            player_id="test_player",
            message_text="What do you know about the temple?",
            message_type=MessageType.QUESTION,
        )

        exchange_dict = exchange.to_dict()

        assert exchange_dict["speaker_id"] == "test_player"
        assert exchange_dict["message_text"] == "What do you know about the temple?"
        assert exchange_dict["message_type"] == "question"
        assert exchange_dict["emotion"] is None
        assert "exchange_id" in exchange_dict
        assert "timestamp" in exchange_dict


class TestNPCPersonality:
    """Test NPCPersonality model."""

    def test_npc_personality_creation(self):
        """Test creating an NPC personality."""
        personality = NPCPersonality(
            npc_id="guard_captain",
            traits={"authoritative": 0.9, "helpful": 0.7},
            knowledge_areas=["security", "castle_layout"],
            speech_patterns={"formality": "high"},
            relationships={"player_001": 0.3},
            background_story="A veteran guard...",
            default_mood="professional",
        )

        assert personality.npc_id == "guard_captain"
        assert personality.traits == {"authoritative": 0.9, "helpful": 0.7}
        assert personality.knowledge_areas == ["security", "castle_layout"]
        assert personality.default_mood == "professional"

    def test_get_trait_strength(self):
        """Test getting trait strength."""
        personality = NPCPersonality(
            npc_id="test_npc",
            traits={"friendly": 0.8, "talkative": 0.6},
            knowledge_areas=[],
            speech_patterns={},
            relationships={},
            background_story="",
        )

        assert personality.get_trait_strength("friendly") == 0.8
        assert personality.get_trait_strength("talkative") == 0.6
        assert personality.get_trait_strength("nonexistent") == 0.0

    def test_relationship_management(self):
        """Test relationship level management."""
        personality = NPCPersonality(
            npc_id="test_npc",
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            relationships={"player_001": 0.2},
            background_story="",
        )

        # Test getting existing relationship
        assert personality.get_relationship_level("player_001") == 0.2
        assert personality.get_relationship_level("player_002") == 0.0

        # Test updating relationship
        personality.update_relationship("player_001", 0.3)
        assert personality.get_relationship_level("player_001") == 0.5

        # Test relationship bounds
        personality.update_relationship("player_001", 1.0)  # Should cap at 1.0
        assert personality.get_relationship_level("player_001") == 1.0

        personality.update_relationship("player_001", -2.5)  # Should cap at -1.0
        assert personality.get_relationship_level("player_001") == -1.0


class TestConversationContext:
    """Test ConversationContext model."""

    def test_conversation_context_creation(self):
        """Test creating a conversation context."""
        context = ConversationContext.create(
            player_id="test_player",
            npc_id="guard_captain",
            topic="castle_security",
            initial_mood="professional",
            relationship_level=0.3,
        )

        assert context.player_id == "test_player"
        assert context.npc_id == "guard_captain"
        assert context.topic == "castle_security"
        assert context.mood == "professional"
        assert context.relationship_level == 0.3
        assert context.status == ConversationStatus.ACTIVE
        assert context.conversation_id is not None

    def test_add_exchange(self):
        """Test adding exchanges to conversation."""
        context = ConversationContext.create(
            player_id="test_player",
            npc_id="guard_captain",
        )

        exchange = ConversationExchange.create_player_message(
            player_id="test_player",
            message_text="Hello!",
        )

        assert context.get_exchange_count() == 0
        context.add_exchange(exchange)
        assert context.get_exchange_count() == 1
        assert context.conversation_history[0] == exchange

    def test_get_recent_exchanges(self):
        """Test getting recent exchanges."""
        context = ConversationContext.create(
            player_id="test_player",
            npc_id="guard_captain",
        )

        # Add multiple exchanges
        for i in range(5):
            exchange = ConversationExchange.create_player_message(
                player_id="test_player",
                message_text=f"Message {i}",
            )
            context.add_exchange(exchange)

        recent = context.get_recent_exchanges(3)
        assert len(recent) == 3
        assert recent[0].message_text == "Message 2"
        assert recent[2].message_text == "Message 4"

    def test_end_conversation(self):
        """Test ending a conversation."""
        context = ConversationContext.create(
            player_id="test_player",
            npc_id="guard_captain",
        )

        assert context.status == ConversationStatus.ACTIVE
        assert context.ended_at is None

        context.end_conversation("player_left")

        assert context.status == ConversationStatus.ENDED
        assert context.ended_at is not None
        assert context.context_data["end_reason"] == "player_left"

    def test_update_mood_and_relationship(self):
        """Test updating mood and relationship."""
        context = ConversationContext.create(
            player_id="test_player",
            npc_id="guard_captain",
            initial_mood="neutral",
            relationship_level=0.0,
        )

        context.update_mood("pleased")
        assert context.mood == "pleased"

        context.update_relationship(0.2)
        assert context.relationship_level == 0.2

        # Test bounds
        context.update_relationship(1.0)  # Should cap at 1.0
        assert context.relationship_level == 1.0


class TestConversationResult:
    """Test ConversationResult model."""

    def test_success_result(self):
        """Test creating a success result."""
        npc_response = ConversationExchange.create_npc_message(
            npc_id="guard_captain",
            message_text="Certainly, I can help with that.",
        )

        result = ConversationResult.success_result(
            npc_response=npc_response,
            relationship_change=0.1,
            mood_change="helpful",
            knowledge_extracted=[{"info": "guard knows about temple"}],
        )

        assert result.success is True
        assert result.npc_response == npc_response
        assert result.relationship_change == 0.1
        assert result.mood_change == "helpful"
        assert result.knowledge_extracted == [{"info": "guard knows about temple"}]
        assert result.errors == []

    def test_error_result(self):
        """Test creating an error result."""
        result = ConversationResult.error_result(
            error_message="NPC not found",
            errors=["Invalid NPC ID", "Database error"],
        )

        assert result.success is False
        assert result.npc_response is None
        assert result.relationship_change == 0.0
        assert result.errors == ["Invalid NPC ID", "Database error"]


class TestEnums:
    """Test conversation enums."""

    def test_conversation_status_values(self):
        """Test ConversationStatus enum values."""
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.ENDED.value == "ended"
        assert ConversationStatus.PAUSED.value == "paused"
        assert ConversationStatus.ABANDONED.value == "abandoned"

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.GREETING.value == "greeting"
        assert MessageType.QUESTION.value == "question"
        assert MessageType.STATEMENT.value == "statement"
        assert MessageType.FAREWELL.value == "farewell"
        assert MessageType.SYSTEM.value == "system"