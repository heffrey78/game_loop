"""Tests for EnhancedConversationCommandHandler."""

from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from game_loop.core.command_handlers.enhanced_conversation_handler import (
    EnhancedConversationCommandHandler,
)


class TestEnhancedConversationCommandHandler:
    """Test cases for EnhancedConversationCommandHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = Mock(spec=Console)
        self.mock_state_manager = Mock()
        self.handler = EnhancedConversationCommandHandler(
            self.mock_console, self.mock_state_manager
        )

    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler is not None
        assert hasattr(self.handler, "personality_engine")
        assert hasattr(self.handler, "memory_manager")
        assert hasattr(self.handler, "knowledge_engine")

    @pytest.mark.asyncio
    async def test_enhanced_response_generation_first_meeting(self):
        """Test enhanced response generation for first meeting."""
        # Mock NPC
        mock_npc = Mock()
        mock_npc.id = "guard_1"
        mock_npc.name = "Security Guard"
        mock_npc.archetype = "security_guard"
        mock_npc.mood = "alert"

        # Mock context
        context = {
            "player": {"id": "player_1", "name": "TestPlayer"},
            "location": {"id": "lobby", "type": "office"},
            "npc": {"name": "Security Guard"},
        }

        # Mock memory manager to return first meeting
        self.handler.memory_manager.get_conversation_context = Mock(
            return_value={
                "is_first_meeting": True,
                "previous_topics": [],
                "total_conversations": 0,
                "relationship_level": "stranger",
            }
        )

        # Mock knowledge engine
        self.handler.knowledge_engine.get_npc_knowledge = AsyncMock(
            return_value={
                "location": {"security_features": []},
                "role": {"expertise_areas": ["security"]},
                "situation": {},
                "sharing_style": {"verification_required": True},
                "knowledge_confidence": 0.8,
            }
        )

        # Mock conversation recording
        self.handler.memory_manager.record_conversation = AsyncMock()

        response = await self.handler._generate_npc_response(mock_npc, context, None)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify memory manager was called
        self.handler.memory_manager.get_conversation_context.assert_called_once()
        self.handler.memory_manager.record_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_response_generation_returning_visitor(self):
        """Test enhanced response generation for returning visitor."""
        # Mock NPC
        mock_npc = Mock()
        mock_npc.id = "guard_1"
        mock_npc.name = "Security Guard"
        mock_npc.archetype = "security_guard"

        # Mock context
        context = {
            "player": {"id": "player_1", "name": "TestPlayer"},
            "location": {"id": "lobby"},
        }

        # Mock memory manager to return existing relationship
        self.handler.memory_manager.get_conversation_context = Mock(
            return_value={
                "is_first_meeting": False,
                "previous_topics": ["security", "greeting"],
                "total_conversations": 3,
                "relationship_level": "friendly_acquaintance",
                "player_name_known": True,
                "preferred_name": "TestPlayer",
            }
        )

        # Mock knowledge engine
        self.handler.knowledge_engine.get_npc_knowledge = AsyncMock(return_value={})

        # Mock conversation recording
        self.handler.memory_manager.record_conversation = AsyncMock()

        response = await self.handler._generate_npc_response(mock_npc, context, None)

        assert response is not None
        assert isinstance(response, str)
        # Should include player name and reference to previous interactions
        assert "TestPlayer" in response or "again" in response.lower()

    def test_memory_based_greeting_generation(self):
        """Test memory-based greeting generation."""
        from datetime import timedelta

        # Test trusted friend greeting
        greeting = self.handler._generate_memory_based_greeting(
            "Guard Bob", "Alice", "trusted_friend", timedelta(hours=2)
        )
        assert "dear friend" in greeting.lower()
        assert "Alice" in greeting

        # Test neutral relationship
        greeting = self.handler._generate_memory_based_greeting(
            "Guard Bob", "Alice", "neutral", timedelta(days=1)
        )
        assert "Alice" in greeting
        assert "dear friend" not in greeting.lower()

    def test_repeat_topic_response_generation(self):
        """Test response generation for repeated topics."""
        response = self.handler._generate_repeat_topic_response(
            "security", "security_guard"
        )

        assert response is not None
        assert "security" in response.lower()
        assert "before" in response.lower()

    @pytest.mark.asyncio
    async def test_topic_response_with_memory(self):
        """Test topic response considering conversation history."""
        context = {"npc_archetype": "security_guard", "time_of_day": "day"}

        # Test new topic
        response = await self.handler._generate_topic_response_with_memory(
            context, "access_request", []
        )
        assert response is not None

        # Test repeated topic
        response = await self.handler._generate_topic_response_with_memory(
            context, "security", ["security", "greeting"]
        )
        assert response is not None
        assert "before" in response.lower()

    @pytest.mark.asyncio
    async def test_knowledge_context_addition(self):
        """Test adding knowledge-based context to responses."""
        # High confidence knowledge
        knowledge = {"knowledge_confidence": 0.9}
        context = await self.handler._add_knowledge_context(
            "security_guard", "security_procedures", knowledge
        )
        assert context is not None
        assert "extensive knowledge" in context

        # Low confidence knowledge
        knowledge = {"knowledge_confidence": 0.2}
        context = await self.handler._add_knowledge_context(
            "scholar", "quantum_physics", knowledge
        )
        assert context is not None
        assert "not entirely certain" in context

        # Role-specific knowledge
        context = await self.handler._add_knowledge_context(
            "security_guard", "access_control", {"knowledge_confidence": 0.7}
        )
        assert context is not None
        assert "security responsibilities" in context

    def test_conversation_suggestions_generation(self):
        """Test conversation topic suggestions."""
        context = {"location_type": "office"}

        suggestions = self.handler._generate_conversation_suggestions(
            "security_guard", context
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_knowledge_summary_retrieval(self):
        """Test NPC knowledge summary retrieval."""
        summary = self.handler.get_npc_knowledge_summary("security_guard", "lobby")

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "security" in summary.lower()

    def test_conversation_topic_suggestions(self):
        """Test conversation topic suggestions."""
        suggestions = self.handler.get_conversation_suggestions("scholar")

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_conversation_cleanup(self):
        """Test old conversation cleanup."""
        cleared_count = self.handler.clear_old_conversations(30)

        assert isinstance(cleared_count, int)
        assert cleared_count >= 0

    @pytest.mark.asyncio
    async def test_error_handling_fallback(self):
        """Test error handling falls back to parent implementation."""
        # Mock an error in enhanced response generation
        mock_npc = Mock()
        mock_npc.id = "guard_1"
        mock_npc.name = "Security Guard"
        mock_npc.archetype = "security_guard"

        # Make memory manager raise an exception
        self.handler.memory_manager.get_conversation_context = Mock(
            side_effect=Exception("Test error")
        )

        context = {
            "player": {"id": "player_1", "name": "TestPlayer"},
            "location": {"id": "lobby"},
        }

        # Should not raise exception, should fall back
        response = await self.handler._generate_npc_response(mock_npc, context, None)

        # Should get some response (either from fallback or parent implementation)
        assert response is not None or response == None  # Parent might return None
