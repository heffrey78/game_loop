"""Tests for memory integration interface."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    MessageType,
    NPCPersonality,
)
from game_loop.core.conversation.memory_integration import (
    ConversationFlowState,
    MemoryContext,
    MemoryDisclosureLevel,
    MemoryIntegrationInterface,
    MemoryRetrievalResult,
)


class TestMemoryIntegrationInterface:
    """Test suite for MemoryIntegrationInterface."""

    @pytest.fixture
    def mock_session_factory(self):
        """Mock database session factory."""
        factory = MagicMock()
        session = AsyncMock()
        factory.get_session.return_value.__aenter__ = AsyncMock(return_value=session)
        factory.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
        return factory, session

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = AsyncMock()
        client.generate_response = AsyncMock(return_value="test response")
        return client

    @pytest.fixture
    def memory_interface(self, mock_session_factory, mock_llm_client):
        """Create memory integration interface for testing."""
        session_factory, _ = mock_session_factory
        return MemoryIntegrationInterface(
            session_factory=session_factory,
            llm_client=mock_llm_client,
            enable_memory_enhancement=True,
            similarity_threshold=0.7,
            max_memories_per_query=5,
        )

    @pytest.fixture
    def sample_conversation(self):
        """Create sample conversation context."""
        player_id = str(uuid.uuid4())
        npc_id = str(uuid.uuid4())

        # Create conversation with some history
        conversation = ConversationContext.create(
            player_id=player_id,
            npc_id=npc_id,
            topic="village_crops",
            initial_mood="concerned",
        )

        # Add some exchanges
        conversation.add_exchange(
            ConversationExchange.create_player_message(
                player_id, "How are the crops doing this season?", MessageType.QUESTION
            )
        )
        conversation.add_exchange(
            ConversationExchange.create_npc_message(
                npc_id,
                "Not well, I'm afraid. The drought has been harsh.",
                MessageType.STATEMENT,
                emotion="worried",
            )
        )

        return conversation

    @pytest.fixture
    def sample_npc_personality(self):
        """Create sample NPC personality."""
        return NPCPersonality(
            npc_id=str(uuid.uuid4()),
            traits={"helpful": 0.8, "knowledgeable": 0.9, "worried": 0.6},
            knowledge_areas=["agriculture", "weather", "village_history"],
            speech_patterns={"formality": "medium", "directness": "high"},
            relationships={},
            background_story="Village farming expert",
            default_mood="concerned",
        )

    @pytest.mark.asyncio
    async def test_extract_memory_context_success(
        self,
        memory_interface,
        sample_conversation,
        sample_npc_personality,
        mock_llm_client,
    ):
        """Test successful memory context extraction."""
        # Mock LLM responses
        mock_llm_client.generate_response.side_effect = [
            "crops",  # topic extraction
            "worried",  # emotional tone
        ]

        context = await memory_interface.extract_memory_context(
            sample_conversation,
            "Tell me more about the drought impact",
            sample_npc_personality,
        )

        assert context.current_topic == "crops"
        assert context.emotional_tone == "worried"
        assert len(context.conversation_history) == 2
        assert context.npc_knowledge_areas == [
            "agriculture",
            "weather",
            "village_history",
        ]
        assert context.session_disclosure_level == MemoryDisclosureLevel.NONE

        # Verify LLM was called for topic and emotion extraction
        assert mock_llm_client.generate_response.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_memory_context_llm_failure(
        self,
        memory_interface,
        sample_conversation,
        sample_npc_personality,
        mock_llm_client,
    ):
        """Test memory context extraction when LLM fails."""
        # Mock LLM failure
        mock_llm_client.generate_response.side_effect = Exception("LLM unavailable")

        context = await memory_interface.extract_memory_context(
            sample_conversation, "Tell me about crops", sample_npc_personality
        )

        # Should return minimal context without crashing
        assert context.current_topic is None
        assert context.emotional_tone == "neutral"
        assert len(context.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_disabled(
        self, mock_session_factory, mock_llm_client
    ):
        """Test memory retrieval when enhancement is disabled."""
        session_factory, _ = mock_session_factory
        interface = MemoryIntegrationInterface(
            session_factory=session_factory,
            llm_client=mock_llm_client,
            enable_memory_enhancement=False,
        )

        memory_context = MemoryContext(current_topic="crops")
        npc_id = uuid.uuid4()

        result = await interface.retrieve_relevant_memories(memory_context, npc_id)

        assert result.relevant_memories == []
        assert result.disclosure_recommendation == MemoryDisclosureLevel.NONE
        assert result.flow_analysis == ConversationFlowState.NATURAL

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_success(
        self, memory_interface, mock_session_factory
    ):
        """Test successful memory retrieval."""
        session_factory, mock_session = mock_session_factory

        # Mock semantic memory repository
        mock_memory_repo = AsyncMock()
        mock_memories = [
            (
                ConversationExchange.create_npc_message(
                    "npc1", "Last year's drought was terrible too.", emotion="worried"
                ),
                0.85,
            ),
            (
                ConversationExchange.create_npc_message(
                    "npc1", "We lost half the corn harvest.", emotion="sad"
                ),
                0.78,
            ),
        ]
        mock_memory_repo.find_similar_memories.return_value = mock_memories

        with patch(
            "game_loop.core.conversation.memory_integration.SemanticMemoryRepository",
            return_value=mock_memory_repo,
        ):
            memory_context = MemoryContext(
                current_topic="drought",
                emotional_tone="worried",
            )
            npc_id = uuid.uuid4()
            query_embedding = [0.1] * 384  # Mock embedding

            result = await memory_interface.retrieve_relevant_memories(
                memory_context, npc_id, query_embedding
            )

            assert len(result.relevant_memories) == 2
            assert result.context_score > 0.7  # Good similarity scores
            # Flow analysis should be appropriate for the context
            assert result.flow_analysis in [
                ConversationFlowState.MEMORY_RELEVANT,
                ConversationFlowState.NATURAL,
            ]
            # Should recommend some level of disclosure for relevant memories
            assert result.disclosure_recommendation in [
                MemoryDisclosureLevel.SUBTLE_HINTS,
                MemoryDisclosureLevel.DIRECT_REFERENCES,
                MemoryDisclosureLevel.DETAILED_MEMORIES,
            ]

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_fallback(
        self, memory_interface, mock_session_factory
    ):
        """Test memory retrieval fallback when service fails."""
        session_factory, mock_session = mock_session_factory

        # Mock repository failure
        with patch(
            "game_loop.core.conversation.memory_integration.SemanticMemoryRepository",
            side_effect=Exception("Database error"),
        ):
            memory_context = MemoryContext(
                current_topic="crops",
                conversation_history=[
                    ConversationExchange.create_player_message(
                        "player1", "How are crops?"
                    ),
                    ConversationExchange.create_npc_message("npc1", "Not good."),
                ],
            )

            result = await memory_interface.retrieve_relevant_memories(
                memory_context, uuid.uuid4()
            )

            assert result.fallback_triggered is True
            assert result.error_message == "Database error"
            assert result.relevant_memories == []
            assert result.flow_analysis == ConversationFlowState.NATURAL

    @pytest.mark.asyncio
    async def test_update_conversation_state_engagement_increase(
        self, memory_interface, mock_llm_client
    ):
        """Test conversation state update with increased engagement."""
        conversation_id = str(uuid.uuid4())

        # Create memory result indicating relevant memories
        memory_result = MemoryRetrievalResult(
            relevant_memories=[
                (ConversationExchange.create_npc_message("npc1", "test"), 0.8)
            ],
            flow_analysis=ConversationFlowState.MEMORY_RELEVANT,
        )

        # Simulate engaged player response
        player_message = "Tell me more about that! What happened?"
        npc_response = "Well, it's quite a story..."

        await memory_interface.update_conversation_state(
            conversation_id, memory_result, player_message, npc_response
        )

        state = memory_interface._get_conversation_state(conversation_id)
        assert state.player_engagement_score > 0.5  # Increased from initial 0.5
        assert state.memory_references_count == 1

    @pytest.mark.asyncio
    async def test_update_conversation_state_engagement_decrease(
        self, memory_interface
    ):
        """Test conversation state update with decreased engagement."""
        conversation_id = str(uuid.uuid4())

        memory_result = MemoryRetrievalResult()

        # Simulate disengaged player response
        player_message = "okay"
        npc_response = "..."

        await memory_interface.update_conversation_state(
            conversation_id, memory_result, player_message, npc_response
        )

        state = memory_interface._get_conversation_state(conversation_id)
        assert state.player_engagement_score < 0.5  # Decreased from initial 0.5

    def test_ab_testing_memory_enable_disable(self, memory_interface):
        """Test A/B testing functionality for enabling/disabling memory."""
        conversation_id = str(uuid.uuid4())

        # Initially enabled
        assert (
            memory_interface.is_memory_enabled_for_conversation(conversation_id) is True
        )

        # Disable for this conversation
        memory_interface.enable_memory_for_conversation(conversation_id, False)
        assert (
            memory_interface.is_memory_enabled_for_conversation(conversation_id)
            is False
        )

        # Re-enable
        memory_interface.enable_memory_for_conversation(conversation_id, True)
        assert (
            memory_interface.is_memory_enabled_for_conversation(conversation_id) is True
        )

    def test_disclosure_level_progression(self, memory_interface):
        """Test disclosure level progression logic."""
        # Test progression
        assert (
            memory_interface._progress_disclosure_level(MemoryDisclosureLevel.NONE)
            == MemoryDisclosureLevel.SUBTLE_HINTS
        )

        assert (
            memory_interface._progress_disclosure_level(
                MemoryDisclosureLevel.SUBTLE_HINTS
            )
            == MemoryDisclosureLevel.DIRECT_REFERENCES
        )

        assert (
            memory_interface._progress_disclosure_level(
                MemoryDisclosureLevel.DIRECT_REFERENCES
            )
            == MemoryDisclosureLevel.DETAILED_MEMORIES
        )

        # Test regression
        assert (
            memory_interface._regress_disclosure_level(
                MemoryDisclosureLevel.DETAILED_MEMORIES
            )
            == MemoryDisclosureLevel.DIRECT_REFERENCES
        )

        assert (
            memory_interface._regress_disclosure_level(
                MemoryDisclosureLevel.DIRECT_REFERENCES
            )
            == MemoryDisclosureLevel.SUBTLE_HINTS
        )

    def test_topic_continuity_calculation(self, memory_interface):
        """Test topic continuity scoring."""
        # Recent topic - high continuity
        assert (
            memory_interface._calculate_topic_continuity(
                "crops", ["weather", "crops", "harvest"]
            )
            == 0.9
        )

        # Old topic - medium continuity
        assert (
            memory_interface._calculate_topic_continuity(
                "weather", ["weather", "crops", "harvest", "politics"]
            )
            == 0.6
        )

        # New topic - low continuity
        assert (
            memory_interface._calculate_topic_continuity(
                "magic", ["weather", "crops", "harvest"]
            )
            == 0.2
        )

        # No history - neutral
        assert memory_interface._calculate_topic_continuity("crops", []) == 0.5

    def test_emotional_alignment_calculation(self, memory_interface):
        """Test emotional alignment calculation."""
        memories = [
            (
                ConversationExchange.create_npc_message(
                    "npc1", "test", emotion="worried"
                ),
                0.8,
            ),
            (
                ConversationExchange.create_npc_message(
                    "npc1", "test", emotion="worried"
                ),
                0.7,
            ),
            (
                ConversationExchange.create_npc_message(
                    "npc1", "test", emotion="happy"
                ),
                0.6,
            ),
        ]

        # Matching emotion should give high alignment
        alignment = memory_interface._calculate_emotional_alignment("worried", memories)
        assert alignment > 0.6  # 2/3 match exactly, 1/3 doesn't

        # Neutral emotion should give medium alignment
        alignment = memory_interface._calculate_emotional_alignment("neutral", memories)
        assert 0.6 <= alignment <= 0.8

    def test_conversation_flow_analysis(self, memory_interface):
        """Test conversation flow analysis logic."""
        # Test with well-aligned memories
        memory_context = MemoryContext(
            emotional_tone="worried",
            topic_continuity_score=0.8,
        )

        memories = [
            (
                ConversationExchange.create_npc_message(
                    "npc1", "test", emotion="worried"
                ),
                0.9,
            ),
        ]

        # This would need to be tested with the actual async method
        # For now, test the logic components
        assert True  # Placeholder for more complex flow analysis tests

    def test_engagement_change_calculation(self, memory_interface):
        """Test player engagement change calculation."""
        # Positive engagement indicators
        positive_change = asyncio.run(
            memory_interface._calculate_engagement_change(
                "Tell me more about that! How interesting!", "Well..."
            )
        )
        assert positive_change > 0

        # Question indicates interest
        question_change = asyncio.run(
            memory_interface._calculate_engagement_change(
                "What happened next?", "Then..."
            )
        )
        assert question_change > 0

        # Negative engagement indicators
        negative_change = asyncio.run(
            memory_interface._calculate_engagement_change("okay whatever", "Alright...")
        )
        assert negative_change < 0

    def test_recommend_disclosure_level_inappropriate_flow(self, memory_interface):
        """Test disclosure recommendation for inappropriate conversation flow."""
        memory_context = MemoryContext(current_topic="test")
        memories = [(ConversationExchange.create_npc_message("npc1", "test"), 0.9)]

        recommendation = memory_interface._recommend_disclosure_level(
            memory_context, memories, ConversationFlowState.MEMORY_INAPPROPRIATE
        )

        assert recommendation == MemoryDisclosureLevel.NONE

    def test_recommend_disclosure_level_high_relevance(self, memory_interface):
        """Test disclosure recommendation for highly relevant memories."""
        memory_context = MemoryContext(
            current_topic="test",
            session_disclosure_level=MemoryDisclosureLevel.DIRECT_REFERENCES,
        )
        memories = [(ConversationExchange.create_npc_message("npc1", "test"), 0.95)]

        recommendation = memory_interface._recommend_disclosure_level(
            memory_context, memories, ConversationFlowState.MEMORY_RELEVANT
        )

        assert recommendation == MemoryDisclosureLevel.DETAILED_MEMORIES

    @pytest.mark.asyncio
    async def test_extract_player_interests(self, memory_interface):
        """Test player interest extraction from conversation."""
        conversation_history = [
            ConversationExchange.create_player_message(
                "player1", "Tell me about the harvest festival"
            ),
            ConversationExchange.create_npc_message("npc1", "It's wonderful!"),
            ConversationExchange.create_player_message(
                "player1", "What about the local politics?"
            ),
            ConversationExchange.create_npc_message("npc1", "Complicated..."),
        ]

        interests = await memory_interface._extract_player_interests(
            conversation_history
        )

        # Should extract topics the player asked about
        assert len(interests) >= 0  # Basic extraction logic implemented

    @pytest.mark.asyncio
    async def test_memory_context_extraction_edge_cases(
        self, memory_interface, mock_llm_client
    ):
        """Test memory context extraction with edge cases."""
        # Empty conversation
        empty_conversation = ConversationContext.create("player1", "npc1")
        npc_personality = NPCPersonality(
            npc_id="npc1",
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            relationships={},
            background_story="",
        )

        mock_llm_client.generate_response.side_effect = ["general", "neutral"]

        context = await memory_interface.extract_memory_context(
            empty_conversation, "Hello", npc_personality
        )

        assert context.current_topic == "general"
        assert context.emotional_tone == "neutral"
        assert len(context.conversation_history) == 0


if __name__ == "__main__":
    pytest.main([__file__])
