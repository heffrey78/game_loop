"""
Basic integration tests for ConversationFlowManager.

These tests validate core functionality without requiring full database schema.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from game_loop.core.conversation.flow_manager import ConversationFlowManager
from game_loop.core.conversation.flow_templates import ConversationStage
from game_loop.core.conversation.memory_integration import (
    MemoryContext,
    MemoryIntegrationInterface,
    MemoryRetrievalResult,
    MemoryDisclosureLevel,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
)
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest_asyncio.fixture
async def mock_memory_integration() -> MemoryIntegrationInterface:
    """Create mock memory integration for basic testing."""
    mock = AsyncMock(spec=MemoryIntegrationInterface)

    # Configure basic responses
    mock.extract_memory_context.return_value = MemoryContext(
        current_topic="test_topic",
        emotional_tone="neutral",
        session_disclosure_level=MemoryDisclosureLevel.NONE,
    )

    mock.retrieve_relevant_memories.return_value = MemoryRetrievalResult(
        relevant_memories=[],
        context_score=0.0,
        disclosure_recommendation=MemoryDisclosureLevel.NONE,
    )

    return mock


@pytest_asyncio.fixture
async def test_conversation_context() -> ConversationContext:
    """Create a simple conversation context for testing."""
    return ConversationContext(
        conversation_id=str(uuid.uuid4()),  # Convert to string
        player_id=str(uuid.uuid4()),  # Convert to string
        npc_id=str(uuid.uuid4()),  # Convert to string
        topic="test",
        mood="neutral",
        relationship_level=0.3,
        status="active",
    )


@pytest_asyncio.fixture
async def test_npc_personality() -> NPCPersonality:
    """Create a test NPC personality."""
    return NPCPersonality(
        npc_id=uuid.uuid4(),
        traits={"friendly": 0.7, "helpful": 0.8},
        knowledge_areas=["general"],
        speech_patterns={"formality": "medium"},
        background_story="A helpful NPC.",
        default_mood="neutral",
        relationships={},
    )


@pytest.mark.asyncio
class TestBasicConversationFlowIntegration:
    """Basic integration tests that don't require full database setup."""

    async def test_flow_manager_initialization(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
    ):
        """Test that ConversationFlowManager initializes correctly."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        assert flow_manager.memory_integration is mock_memory_integration
        assert flow_manager.session_factory is session_factory
        assert flow_manager.flow_library is not None

    async def test_conversation_stage_caching(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_conversation_context: ConversationContext,
        test_npc_personality: NPCPersonality,
    ):
        """Test conversation stage determination and caching."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        conversation_id = test_conversation_context.conversation_id

        # First call should determine stage
        stage1 = await flow_manager._determine_conversation_stage(
            test_conversation_context, test_npc_personality
        )

        # Second call should use cached result
        stage2 = await flow_manager._determine_conversation_stage(
            test_conversation_context, test_npc_personality
        )

        assert stage1 == stage2
        assert conversation_id in flow_manager._conversation_stages
        assert flow_manager._conversation_stages[conversation_id] == stage1

    async def test_personality_stage_modifiers(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_conversation_context: ConversationContext,
    ):
        """Test personality-based stage modifications without database."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Test with cautious personality
        cautious_personality = NPCPersonality(
            npc_id=uuid.uuid4(),
            traits={"cautious": 0.8},
            knowledge_areas=["secrets"],
            speech_patterns={},
            background_story="A cautious character.",
            default_mood="neutral",
            relationships={},
        )

        # Apply personality modifiers directly
        initial_stage = ConversationStage.DEEP_CONNECTION
        modified_stage = flow_manager._apply_personality_stage_modifiers(
            initial_stage, cautious_personality
        )

        # Cautious personality should reduce progression
        assert modified_stage == ConversationStage.TRUST_DEVELOPMENT

    async def test_memory_usage_tracking(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_conversation_context: ConversationContext,
        test_npc_personality: NPCPersonality,
    ):
        """Test memory usage tracking functionality."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        conversation_id = test_conversation_context.conversation_id

        # Simulate memory enhancement
        await flow_manager.enhance_response_with_memory_patterns(
            conversation=test_conversation_context,
            personality=test_npc_personality,
            base_response="Hello there.",
            player_message="Hi!",
            npc_id=test_npc_personality.npc_id,
        )

        # Check usage statistics
        stats = flow_manager.get_memory_usage_stats(conversation_id)

        assert "total_interactions" in stats
        assert "memory_enhanced_count" in stats
        assert stats["total_interactions"] >= 1

    async def test_conversation_flow_quality_analysis(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_conversation_context: ConversationContext,
        test_npc_personality: NPCPersonality,
    ):
        """Test conversation flow quality analysis."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        conversation_id = test_conversation_context.conversation_id

        # Build some interaction history
        for i in range(3):
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=test_conversation_context,
                personality=test_npc_personality,
                base_response=f"Response {i}",
                player_message=f"Message {i}",
                npc_id=test_npc_personality.npc_id,
            )

        # Analyze quality
        analysis = await flow_manager.analyze_conversation_flow_quality(conversation_id)

        assert "quality_score" in analysis
        assert "recommendations" in analysis
        assert analysis["quality_score"] >= 0.0
        assert analysis["quality_score"] <= 1.0

    async def test_error_handling_graceful_degradation(
        self,
        session_factory: DatabaseSessionFactory,
        test_conversation_context: ConversationContext,
        test_npc_personality: NPCPersonality,
    ):
        """Test graceful error handling when memory integration fails."""
        # Create failing memory integration
        failing_memory = AsyncMock(spec=MemoryIntegrationInterface)
        failing_memory.extract_memory_context.side_effect = Exception("Service down")
        failing_memory.retrieve_relevant_memories.side_effect = Exception(
            "Service down"
        )

        flow_manager = ConversationFlowManager(
            memory_integration=failing_memory,
            session_factory=session_factory,
        )

        # Should handle errors gracefully
        enhanced_response, integration_data = (
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=test_conversation_context,
                personality=test_npc_personality,
                base_response="Hello",
                player_message="Hi",
                npc_id=test_npc_personality.npc_id,
            )
        )

        # Should return base response with error information
        assert enhanced_response == "Hello"
        assert integration_data["memory_enhanced"] is False
        assert "error" in integration_data


@pytest.mark.asyncio
class TestConversationFlowIntegrationPerformance:
    """Performance tests for conversation flow integration."""

    async def test_stage_determination_performance(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_npc_personality: NPCPersonality,
    ):
        """Test performance of stage determination operations."""
        import time

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Create multiple conversation contexts
        conversations = []
        for i in range(20):
            conv = ConversationContext(
                conversation_id=uuid.uuid4(),
                player_id=uuid.uuid4(),
                npc_id=uuid.uuid4(),
                topic=f"topic_{i}",
                relationship_level=0.1 + (i * 0.04),  # Varying relationships
                status="active",
            )
            conversations.append(conv)

        # Benchmark stage determination
        start_time = time.perf_counter()

        stages = []
        for conv in conversations:
            stage = await flow_manager._determine_conversation_stage(
                conv, test_npc_personality
            )
            stages.append(stage)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Performance assertions
        assert duration < 2.0  # Should complete quickly without real database queries
        assert len(stages) == 20
        assert all(isinstance(stage, ConversationStage) for stage in stages)

    async def test_memory_enhancement_scalability(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        test_conversation_context: ConversationContext,
        test_npc_personality: NPCPersonality,
    ):
        """Test scalability of memory enhancement operations."""
        import time

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Benchmark multiple enhancements
        start_time = time.perf_counter()

        for i in range(15):
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=test_conversation_context,
                personality=test_npc_personality,
                base_response=f"Response {i}",
                player_message=f"Message {i}",
                npc_id=test_npc_personality.npc_id,
            )

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete efficiently
        assert duration < 3.0  # 15 operations in under 3 seconds

        # Check that memory tracking is working
        stats = flow_manager.get_memory_usage_stats(
            test_conversation_context.conversation_id
        )
        assert stats["total_interactions"] == 15
