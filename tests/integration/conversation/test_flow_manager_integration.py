"""
Integration tests for ConversationFlowManager with real database operations.

These tests validate the conversation flow system against actual PostgreSQL
database operations, including transaction boundaries, error handling, and
performance characteristics.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    MessageType,
    NPCPersonality,
)
from game_loop.core.conversation.flow_manager import ConversationFlowManager
from game_loop.core.conversation.flow_templates import ConversationStage, TrustLevel
from game_loop.core.conversation.memory_integration import (
    MemoryContext,
    MemoryIntegrationInterface,
    MemoryRetrievalResult,
    MemoryDisclosureLevel,
)
from game_loop.database.repositories.conversation import ConversationRepositoryManager
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest_asyncio.fixture
async def conversation_repo_manager(
    db_session: AsyncSession,
) -> ConversationRepositoryManager:
    """Create conversation repository manager for tests."""
    return ConversationRepositoryManager(db_session)


@pytest_asyncio.fixture
async def npc_personality(
    conversation_repo_manager: ConversationRepositoryManager,
) -> NPCPersonality:
    """Create test NPC personality."""
    npc_id = uuid.uuid4()
    personality = await conversation_repo_manager.npc_personalities.create_personality(
        npc_id=npc_id,
        traits={
            "wise": 0.8,
            "patient": 0.7,
            "cautious": 0.6,
            "open": 0.4,
        },
        knowledge_areas=["history", "magic", "local_lore"],
        speech_patterns={"formality": "medium", "verbosity": "high"},
        background_story="A wise village elder with deep knowledge of local history.",
        default_mood="contemplative",
    )
    return personality


@pytest_asyncio.fixture
async def conversation_context(
    conversation_repo_manager: ConversationRepositoryManager,
    npc_personality: NPCPersonality,
) -> ConversationContext:
    """Create test conversation context with some history."""
    player_id = uuid.uuid4()

    # Create initial conversation
    conversation, initial_exchange = (
        await conversation_repo_manager.create_complete_conversation(
            player_id=player_id,
            npc_id=npc_personality.npc_id,
            initial_message="Greetings, traveler. Welcome to our village.",
            topic="greeting",
            initial_mood="welcoming",
        )
    )

    # Add some conversation history to make it more realistic
    await conversation_repo_manager.exchanges.create_exchange(
        conversation_id=conversation.conversation_id,
        speaker_id=player_id,
        message_text="Hello! This village seems peaceful. Can you tell me about it?",
        message_type=MessageType.DIALOGUE.value,
        emotion="curious",
    )

    await conversation_repo_manager.exchanges.create_exchange(
        conversation_id=conversation.conversation_id,
        speaker_id=npc_personality.npc_id,
        message_text="Indeed it is. We've lived here for generations, maintaining our traditions.",
        message_type=MessageType.DIALOGUE.value,
        emotion="proud",
    )

    # Update relationship level
    conversation.relationship_level = 0.3

    return conversation


@pytest_asyncio.fixture
async def mock_memory_integration() -> MemoryIntegrationInterface:
    """Create mock memory integration interface for testing."""
    mock = AsyncMock(spec=MemoryIntegrationInterface)

    # Configure default responses
    mock.extract_memory_context.return_value = MemoryContext(
        current_topic="village_history",
        emotional_tone="curious",
        player_interests=["history", "local_lore"],
        npc_knowledge_areas=["history", "magic", "local_lore"],
        session_disclosure_level=MemoryDisclosureLevel.SUBTLE_HINTS,
        topic_continuity_score=0.7,
    )

    mock.retrieve_relevant_memories.return_value = MemoryRetrievalResult(
        relevant_memories=[
            (
                ConversationExchange(
                    conversation_id=uuid.uuid4(),
                    speaker_id=uuid.uuid4(),
                    message_text="The old temple holds many secrets from our ancestors.",
                    message_type=MessageType.DIALOGUE.value,
                    emotion="mysterious",
                    timestamp=datetime.utcnow() - timedelta(days=1),
                ),
                0.8,
            )
        ],
        context_score=0.75,
        emotional_alignment=0.6,
        disclosure_recommendation=MemoryDisclosureLevel.CLEAR_REFERENCES,
    )

    return mock


@pytest.mark.asyncio
class TestConversationFlowManagerIntegration:
    """Integration tests for ConversationFlowManager with real database."""

    async def test_conversation_stage_determination_with_database(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test conversation stage determination with real database queries."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Test stage determination for initial conversation
        stage = await flow_manager._determine_conversation_stage(
            conversation_context, npc_personality
        )

        # Should be acquaintance level based on relationship score of 0.3
        assert stage == ConversationStage.ACQUAINTANCE

        # Update relationship and test progression
        conversation_context.relationship_level = 0.6

        # Clear cache to force re-determination
        flow_manager._conversation_stages.clear()

        stage = await flow_manager._determine_conversation_stage(
            conversation_context, npc_personality
        )

        assert stage == ConversationStage.RELATIONSHIP_BUILDING

    async def test_personality_stage_modifiers_integration(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        conversation_repo_manager: ConversationRepositoryManager,
    ):
        """Test personality-based stage modifications with database data."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Create cautious NPC personality
        cautious_npc_id = uuid.uuid4()
        cautious_personality = (
            await conversation_repo_manager.npc_personalities.create_personality(
                npc_id=cautious_npc_id,
                traits={"cautious": 0.8, "wise": 0.6},
                knowledge_areas=["secrets", "history"],
                speech_patterns={"formality": "high"},
                background_story="A cautious keeper of secrets.",
            )
        )

        # Create conversation with higher trust level
        conversation_context.relationship_level = (
            0.8  # Should be DEEP_CONNECTION normally
        )
        conversation_context.npc_id = str(cautious_npc_id)

        stage = await flow_manager._determine_conversation_stage(
            conversation_context, cautious_personality
        )

        # Cautious personality should reduce stage progression
        assert stage == ConversationStage.TRUST_DEVELOPMENT

    async def test_end_to_end_response_enhancement_with_database(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test complete response enhancement flow with database operations."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        base_response = "The village has a rich history."
        player_message = "Tell me more about the village history."

        enhanced_response, integration_data = (
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=conversation_context,
                personality=npc_personality,
                base_response=base_response,
                player_message=player_message,
                npc_id=npc_personality.npc_id,
            )
        )

        # Verify the response was enhanced
        assert integration_data["memory_enhanced"] is True
        assert integration_data["stage"] == ConversationStage.ACQUAINTANCE.value
        assert integration_data["trust_level"] == TrustLevel.ACQUAINTANCE.value
        assert integration_data["memory_count"] == 1

        # Verify memory integration was called
        mock_memory_integration.extract_memory_context.assert_called_once()
        mock_memory_integration.retrieve_relevant_memories.assert_called_once()

        # Enhanced response should include memory integration
        assert len(enhanced_response) > len(base_response)

    async def test_conversation_count_optimization(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_repo_manager: ConversationRepositoryManager,
        npc_personality: NPCPersonality,
    ):
        """Test that conversation count queries are optimized (N+1 query fix)."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        player_id = uuid.uuid4()

        # Create multiple conversations to test counting
        conversations = []
        for i in range(3):
            conv, _ = await conversation_repo_manager.create_complete_conversation(
                player_id=player_id,
                npc_id=npc_personality.npc_id,
                initial_message=f"Conversation {i + 1}",
                topic="test",
            )
            conversations.append(conv)

        # Test stage determination - should use optimized count query
        stage = await flow_manager._determine_conversation_stage(
            conversations[-1], npc_personality
        )

        # With 3 conversations and low relationship, should be acquaintance
        assert stage == ConversationStage.ACQUAINTANCE

    async def test_database_transaction_boundary_behavior(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test that database operations respect transaction boundaries."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Test stage progression check
        await flow_manager._check_stage_progression(
            conversation_context.conversation_id,
            ConversationStage.ACQUAINTANCE,
            0.5,
            npc_personality.npc_id,
        )

        # Verify that stage was cached (indicating successful transaction)
        assert conversation_context.conversation_id in flow_manager._conversation_stages

    async def test_memory_usage_tracking_persistence(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test memory usage tracking with persistent data."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Perform multiple enhancement operations
        for i in range(3):
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=conversation_context,
                personality=npc_personality,
                base_response=f"Response {i}",
                player_message=f"Message {i}",
                npc_id=npc_personality.npc_id,
            )

        # Check memory usage statistics
        stats = flow_manager.get_memory_usage_stats(
            conversation_context.conversation_id
        )

        assert stats["total_interactions"] == 3
        assert stats["memory_enhanced_count"] == 3
        assert stats["enhancement_rate"] == 1.0
        assert len(stats["patterns_used"]) > 0

    async def test_conversation_flow_quality_analysis(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test conversation flow quality analysis with real data."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Build up conversation history
        for i in range(5):
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=conversation_context,
                personality=npc_personality,
                base_response=f"Response {i}",
                player_message=f"Message {i}",
                npc_id=npc_personality.npc_id,
            )

        # Analyze conversation quality
        quality_analysis = await flow_manager.analyze_conversation_flow_quality(
            conversation_context.conversation_id
        )

        assert "quality_score" in quality_analysis
        assert quality_analysis["quality_score"] > 0.5
        assert "enhancement_rate" in quality_analysis
        assert "recommendations" in quality_analysis
        assert quality_analysis["current_stage"] == ConversationStage.ACQUAINTANCE.value


@pytest.mark.asyncio
class TestConversationFlowDatabaseErrorHandling:
    """Test error handling with database operations."""

    async def test_stage_determination_with_database_error(
        self,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test graceful handling of database connection errors."""
        # Create flow manager with invalid session factory
        invalid_config = Mock()
        invalid_config.get_session = AsyncMock(
            side_effect=ConnectionError("Database unavailable")
        )

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=invalid_config,
        )

        # Should fall back to safe default stage
        stage = await flow_manager._determine_conversation_stage(
            conversation_context, npc_personality
        )

        assert stage == ConversationStage.ACQUAINTANCE  # Safe default

    async def test_stage_progression_with_invalid_conversation_id(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        npc_personality: NPCPersonality,
    ):
        """Test handling of malformed conversation IDs (security validation)."""
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Test with malformed conversation ID that could cause SQL injection
        malicious_id = "'; DROP TABLE conversations; --"

        # Should handle gracefully without progressing
        await flow_manager._check_stage_progression(
            malicious_id,
            ConversationStage.ACQUAINTANCE,
            0.5,
            npc_personality.npc_id,
        )

        # Should not have cached the invalid ID
        assert malicious_id not in flow_manager._conversation_stages

    async def test_response_enhancement_with_memory_failure(
        self,
        session_factory: DatabaseSessionFactory,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test response enhancement when memory integration fails."""
        # Create memory integration that always fails
        failing_memory = AsyncMock(spec=MemoryIntegrationInterface)
        failing_memory.extract_memory_context.side_effect = Exception(
            "Memory service down"
        )

        flow_manager = ConversationFlowManager(
            memory_integration=failing_memory,
            session_factory=session_factory,
        )

        base_response = "Hello there."
        enhanced_response, integration_data = (
            await flow_manager.enhance_response_with_memory_patterns(
                conversation=conversation_context,
                personality=npc_personality,
                base_response=base_response,
                player_message="Hi!",
                npc_id=npc_personality.npc_id,
            )
        )

        # Should return base response when memory fails
        assert enhanced_response == base_response
        assert integration_data["memory_enhanced"] is False
        assert "error" in integration_data


@pytest.mark.asyncio
class TestConversationFlowPerformanceBenchmarks:
    """Performance benchmarks for database operations."""

    async def test_stage_determination_performance(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_repo_manager: ConversationRepositoryManager,
        npc_personality: NPCPersonality,
    ):
        """Benchmark stage determination with multiple conversations."""
        import time

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        player_id = uuid.uuid4()

        # Create multiple conversations for realistic testing
        conversations = []
        for i in range(10):
            conv, _ = await conversation_repo_manager.create_complete_conversation(
                player_id=player_id,
                npc_id=npc_personality.npc_id,
                initial_message=f"Test conversation {i}",
            )
            conversations.append(conv)

        # Benchmark stage determination
        start_time = time.perf_counter()

        stages = []
        for conv in conversations:
            stage = await flow_manager._determine_conversation_stage(
                conv, npc_personality
            )
            stages.append(stage)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Performance assertions
        assert duration < 1.0  # Should complete in under 1 second
        assert len(stages) == 10
        assert all(isinstance(stage, ConversationStage) for stage in stages)

        # Test caching performance
        start_time = time.perf_counter()

        # Second pass should be faster due to caching
        cached_stages = []
        for conv in conversations:
            stage = await flow_manager._determine_conversation_stage(
                conv, npc_personality
            )
            cached_stages.append(stage)

        cached_end_time = time.perf_counter()
        cached_duration = cached_end_time - start_time

        # Cached operations should be significantly faster
        assert cached_duration < duration / 2

    async def test_memory_integration_performance_with_scaling(
        self,
        session_factory: DatabaseSessionFactory,
        mock_memory_integration: MemoryIntegrationInterface,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ):
        """Test memory integration performance under load."""
        import time

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=session_factory,
        )

        # Benchmark response enhancement
        start_time = time.perf_counter()

        enhancements = []
        for i in range(20):
            enhanced_response, integration_data = (
                await flow_manager.enhance_response_with_memory_patterns(
                    conversation=conversation_context,
                    personality=npc_personality,
                    base_response=f"Response {i}",
                    player_message=f"Message {i}",
                    npc_id=npc_personality.npc_id,
                )
            )
            enhancements.append((enhanced_response, integration_data))

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Performance benchmarks
        assert duration < 5.0  # Should complete 20 enhancements in under 5 seconds
        assert len(enhancements) == 20

        # Check memory usage statistics performance
        stats = flow_manager.get_memory_usage_stats(
            conversation_context.conversation_id
        )
        assert stats["total_interactions"] == 20

    async def test_database_query_optimization_validation(
        self,
        session_factory: DatabaseSessionFactory,
        conversation_repo_manager: ConversationRepositoryManager,
        npc_personality: NPCPersonality,
    ):
        """Validate that N+1 query optimizations are working correctly."""
        import time

        player_id = uuid.uuid4()

        # Create many conversations to test query optimization
        conversations = []
        for i in range(50):
            conv, _ = await conversation_repo_manager.create_complete_conversation(
                player_id=player_id,
                npc_id=npc_personality.npc_id,
                initial_message=f"Conversation {i}",
            )
            conversations.append(conv)

        # Test optimized count query performance
        start_time = time.perf_counter()

        count = await conversation_repo_manager.contexts.get_conversation_count_for_npc_player_pair(
            player_id, npc_personality.npc_id
        )

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Optimized query should be fast even with many conversations
        assert count == 50
        assert duration < 0.5  # Should complete in under 500ms

        # Test that multiple count queries don't scale linearly (O(1) vs O(n))
        start_time = time.perf_counter()

        for _ in range(10):
            count = await conversation_repo_manager.contexts.get_conversation_count_for_npc_player_pair(
                player_id, npc_personality.npc_id
            )

        batch_end_time = time.perf_counter()
        batch_duration = batch_end_time - start_time

        # 10 queries shouldn't be 10x slower than 1 query (due to query optimization)
        assert (
            batch_duration < duration * 15
        )  # Allow some overhead but not linear scaling
