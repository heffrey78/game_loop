"""Tests for ConversationMemoryManager."""

from datetime import datetime, timedelta

import pytest

from game_loop.core.dialogue.memory_manager import ConversationMemoryManager


class TestConversationMemoryManager:
    """Test cases for ConversationMemoryManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConversationMemoryManager()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test manager initialization."""
        assert self.manager is not None
        assert hasattr(self.manager, "conversation_history")
        assert hasattr(self.manager, "relationship_scores")
        assert hasattr(self.manager, "topic_knowledge")

    @pytest.mark.asyncio
    async def test_record_conversation(self):
        """Test recording conversations."""
        await self.manager.record_conversation(
            npc_id="guard_1",
            player_id="player_1",
            topic="greeting",
            response="Hello there!",
            context={"location_id": "lobby", "player_name": "TestPlayer"},
        )

        # Check conversation was recorded
        conversation_key = "guard_1_player_1"
        assert conversation_key in self.manager.conversation_history
        assert len(self.manager.conversation_history[conversation_key]) == 1

        conversation = self.manager.conversation_history[conversation_key][0]
        assert conversation["topic"] == "greeting"
        assert conversation["response"] == "Hello there!"
        assert conversation["context"]["player_name"] == "TestPlayer"

    @pytest.mark.asyncio
    async def test_first_meeting_context(self):
        """Test context for first meeting."""
        context = self.manager.get_conversation_context("guard_1", "player_1")

        assert context["is_first_meeting"] is True
        assert context["previous_topics"] == []
        assert context["total_conversations"] == 0
        assert context["relationship_level"] == "stranger"

    @pytest.mark.asyncio
    async def test_returning_visitor_context(self):
        """Test context for returning visitor."""
        # Record a conversation first
        await self.manager.record_conversation(
            npc_id="guard_1",
            player_id="player_1",
            topic="security",
            response="Security is important.",
            context={"location_id": "lobby", "player_name": "TestPlayer"},
        )

        context = self.manager.get_conversation_context("guard_1", "player_1")

        assert context["is_first_meeting"] is False
        assert context["previous_topics"] == ["security"]
        assert context["total_conversations"] == 1
        assert context["relationship_level"] != "stranger"
        assert context["player_name_known"] is True
        assert context["preferred_name"] == "TestPlayer"

    @pytest.mark.asyncio
    async def test_relationship_progression(self):
        """Test relationship score progression."""
        # Record multiple positive interactions
        for i in range(5):
            await self.manager.record_conversation(
                npc_id="guard_1",
                player_id="player_1",
                topic="help_request",
                response=f"I can help you with that. (conversation {i+1})",
                context={"location_id": "lobby", "interaction_type": "helpful"},
            )

        context = self.manager.get_conversation_context("guard_1", "player_1")

        # Should have progressed beyond stranger
        assert context["relationship_level"] in [
            "acquaintance",
            "friend",
            "friendly_acquaintance",
        ]
        assert context["total_conversations"] == 5

    @pytest.mark.asyncio
    async def test_negative_interaction_impact(self):
        """Test negative interactions affect relationship."""
        # Start with positive interaction
        await self.manager.record_conversation(
            npc_id="guard_1",
            player_id="player_1",
            topic="greeting",
            response="Hello!",
            context={"interaction_type": "polite"},
        )

        # Add negative interaction
        await self.manager.record_conversation(
            npc_id="guard_1",
            player_id="player_1",
            topic="complaint",
            response="That is unacceptable behavior.",
            context={"interaction_type": "rude"},
        )

        context = self.manager.get_conversation_context("guard_1", "player_1")

        # Relationship should be affected by negative interaction
        assert (
            context["relationship_score"] < 10
        )  # Should be lower due to negative interaction

    @pytest.mark.asyncio
    async def test_topic_knowledge_tracking(self):
        """Test topic knowledge is tracked."""
        topics = ["security", "security", "location", "security"]

        for topic in topics:
            await self.manager.record_conversation(
                npc_id="guard_1",
                player_id="player_1",
                topic=topic,
                response="Response about " + topic,
                context={"location_id": "lobby"},
            )

        # Check topic knowledge was tracked
        assert "guard_1" in self.manager.topic_knowledge
        assert "security" in self.manager.topic_knowledge["guard_1"]
        assert self.manager.topic_knowledge["guard_1"]["security"] == 3
        assert self.manager.topic_knowledge["guard_1"]["location"] == 1

    @pytest.mark.asyncio
    async def test_conversation_frequency_calculation(self):
        """Test conversation frequency calculation."""
        # Record conversations with time gaps
        base_time = datetime.now()

        for i in range(3):
            # Simulate conversations with different timestamps
            self.manager.conversation_history["guard_1_player_1"] = [
                {
                    "timestamp": base_time - timedelta(hours=i * 2),
                    "topic": "test",
                    "response": "test",
                    "context": {},
                }
                for i in range(3)
            ]

        context = self.manager.get_conversation_context("guard_1", "player_1")

        assert "conversation_frequency" in context
        assert isinstance(context["conversation_frequency"], float)

    @pytest.mark.asyncio
    async def test_player_name_memory(self):
        """Test player name is remembered."""
        await self.manager.record_conversation(
            npc_id="guard_1",
            player_id="player_1",
            topic="greeting",
            response="Hello!",
            context={"player_name": "Alice"},
        )

        context = self.manager.get_conversation_context("guard_1", "player_1")

        assert context["player_name_known"] is True
        assert context["preferred_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_conversation_history_limit(self):
        """Test conversation history is limited."""
        # Record more than the limit (50 conversations)
        for i in range(55):
            await self.manager.record_conversation(
                npc_id="guard_1",
                player_id="player_1",
                topic=f"topic_{i}",
                response=f"Response {i}",
                context={"location_id": "lobby"},
            )

        conversation_key = "guard_1_player_1"
        history = self.manager.conversation_history[conversation_key]

        # Should be limited to 50
        assert len(history) == 50
        # Should keep the most recent ones
        assert history[-1]["topic"] == "topic_54"

    @pytest.mark.asyncio
    async def test_should_npc_remember_player(self):
        """Test NPC memory decision logic."""
        # New player - should not remember
        assert not self.manager.should_npc_remember_player("guard_1", "new_player")

        # Record some conversations
        for i in range(3):
            await self.manager.record_conversation(
                npc_id="guard_1",
                player_id="player_1",
                topic="test",
                response="test",
                context={},
            )

        # Should remember after multiple conversations
        assert self.manager.should_npc_remember_player("guard_1", "player_1")

    def test_relationship_summary(self):
        """Test relationship summary generation."""
        # Test first meeting
        summary = self.manager.get_relationship_summary("guard_1", "new_player")
        assert "first meeting" in summary.lower()

        # Add some conversation history manually for testing
        self.manager.conversation_history["guard_1_player_1"] = [
            {
                "timestamp": datetime.now(),
                "topic": "test",
                "response": "test",
                "context": {},
            }
        ] * 5
        self.manager.relationship_scores["guard_1_player_1"] = {
            "score": 15,
            "level": "friendly_acquaintance",
        }
        self.manager.name_memory["guard_1_player_1"] = {
            "known": True,
            "preferred_name": "TestPlayer",
        }

        summary = self.manager.get_relationship_summary("guard_1", "player_1")
        assert "friendly_acquaintance" in summary
        assert "5 conversations" in summary
        assert "TestPlayer" in summary

    def test_clear_old_conversations(self):
        """Test clearing old conversation data."""
        # Add some old and new conversations
        old_time = datetime.now() - timedelta(days=35)
        new_time = datetime.now() - timedelta(days=5)

        self.manager.conversation_history["guard_1_player_1"] = [
            {"timestamp": old_time, "topic": "old", "response": "old", "context": {}},
            {"timestamp": new_time, "topic": "new", "response": "new", "context": {}},
        ]

        cleared_count = self.manager.clear_old_conversations(days_old=30)

        assert cleared_count == 1
        remaining = self.manager.conversation_history["guard_1_player_1"]
        assert len(remaining) == 1
        assert remaining[0]["topic"] == "new"
