"""Tests for enhanced conversation flow manager with dialogue integration."""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from game_loop.core.conversation.flow_manager import EnhancedConversationFlowManager
from game_loop.core.conversation.context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueContext,
    DialogueState,
    DialogueEnhancementResult,
    IntegrationStyle,
    MemoryIntegrationPlan,
    MemoryIntegrationTiming,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
    ConversationExchange,
)
from game_loop.core.conversation.conversation_threading import (
    ConversationThreadingService,
)
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest.fixture
def mock_session_factory():
    """Create mock database session factory."""
    factory = MagicMock(spec=DatabaseSessionFactory)
    factory.get_session = AsyncMock()
    return factory


@pytest.fixture
def mock_memory_integration():
    """Create mock memory integration."""
    integration = AsyncMock()
    integration.extract_memory_context = AsyncMock()
    integration.retrieve_relevant_memories = AsyncMock()
    return integration


@pytest.fixture
def mock_threading_service():
    """Create mock conversation threading service."""
    service = MagicMock(spec=ConversationThreadingService)
    service.initiate_conversation_session = AsyncMock()
    service.analyze_threading_opportunity = AsyncMock()
    service.enhance_response_with_threading = AsyncMock()
    return service


@pytest.fixture
def mock_dialogue_engine():
    """Create mock dialogue memory integration engine."""
    engine = MagicMock(spec=DialogueMemoryIntegrationEngine)
    engine.analyze_dialogue_context = AsyncMock()
    engine.create_memory_integration_plan = AsyncMock()
    engine.enhance_dialogue_response = AsyncMock()
    return engine


@pytest.fixture
def sample_conversation():
    """Create sample conversation context."""
    return ConversationContext(
        conversation_id="conv_123",
        player_id="player_456",
        npc_id="npc_789",
        relationship_level=0.6,
        trust_level=0.5,
        conversation_history=[
            ConversationExchange(
                player_message="Hello there!",
                npc_response="Greetings, friend.",
                timestamp="2023-01-01T10:00:00Z",
                emotion="friendly",
                topic="greeting",
            )
        ],
    )


@pytest.fixture
def sample_personality():
    """Create sample NPC personality."""
    personality = MagicMock(spec=NPCPersonality)
    personality.get_trait_strength = MagicMock(return_value=0.5)
    personality.personality_traits = {"friendly": 0.7, "cautious": 0.3}
    return personality


@pytest.fixture
def enhanced_flow_manager(
    mock_memory_integration,
    mock_session_factory,
    mock_threading_service,
    mock_dialogue_engine,
):
    """Create enhanced conversation flow manager."""
    return EnhancedConversationFlowManager(
        memory_integration=mock_memory_integration,
        session_factory=mock_session_factory,
        enable_conversation_threading=True,
        enable_dialogue_integration=True,
        dialogue_integration_engine=mock_dialogue_engine,
        threading_service=mock_threading_service,
    )


class TestEnhancedConversationFlowManager:
    """Tests for EnhancedConversationFlowManager."""

    @pytest.mark.asyncio
    async def test_enhance_response_with_advanced_memory_integration_enabled(
        self,
        enhanced_flow_manager,
        sample_conversation,
        sample_personality,
        mock_dialogue_engine,
        mock_threading_service,
        mock_memory_integration,
    ):
        """Test advanced memory integration when dialogue integration is enabled."""
        # Arrange
        base_response = "That's a wonderful idea!"
        player_message = "I was thinking about our last conversation."
        npc_id = uuid.uuid4()

        # Mock threading context
        threading_context = MagicMock()
        mock_threading_service.initiate_conversation_session.return_value = (
            threading_context
        )

        # Mock dialogue context
        dialogue_context = DialogueContext(
            current_state=DialogueState.BUILDING,
            conversation_depth_level=3,
            engagement_momentum=0.8,
            memory_reference_density=0.3,
            topic_transitions=["greeting", "memories"],
            emotional_arc=["neutral", "positive"],
            last_memory_integration=None,
            player_curiosity_indicators=["thinking", "conversation"],
        )
        mock_dialogue_engine.analyze_dialogue_context.return_value = dialogue_context

        # Mock memory context and retrieval
        memory_context = MagicMock()
        memory_context.emotional_tone = "reflective"
        mock_memory_integration.extract_memory_context.return_value = memory_context

        memory_retrieval = MagicMock()
        memory_retrieval.relevant_memories = [
            (
                ConversationExchange(
                    player_message="I enjoy learning new things",
                    npc_response="That's great!",
                    timestamp="2023-01-01T09:00:00Z",
                ),
                0.9,
            )
        ]
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            memory_retrieval
        )

        # Mock integration plan
        integration_plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.8,
            timing_strategy=MemoryIntegrationTiming.NATURAL_PAUSE,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            selected_memories=memory_retrieval.relevant_memories,
            flow_disruption_risk=0.2,
            naturalness_validation=0.9,
            fallback_plan="use_traditional_patterns",
        )
        mock_dialogue_engine.create_memory_integration_plan.return_value = (
            integration_plan
        )

        # Mock enhancement result
        enhancement_result = DialogueEnhancementResult(
            enhanced_response="That's a wonderful idea! It reminds me of your curiosity about learning.",
            integration_applied=True,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            memories_referenced=["memory_1"],
            confidence_score=0.8,
            naturalness_score=0.9,
            engagement_impact=0.7,
            metadata={"timing": "natural_pause", "emotional_connection": True},
        )
        mock_dialogue_engine.enhance_dialogue_response.return_value = enhancement_result

        # Act
        result_response, metadata = (
            await enhanced_flow_manager.enhance_response_with_advanced_memory_integration(
                conversation=sample_conversation,
                personality=sample_personality,
                base_response=base_response,
                player_message=player_message,
                npc_id=npc_id,
            )
        )

        # Assert
        assert result_response == enhancement_result.enhanced_response
        assert metadata["dialogue_integration_enabled"] is True
        assert metadata["dialogue_integration_applied"] is True
        assert (
            metadata["integration_style"] == IntegrationStyle.EMOTIONAL_CONNECTION.value
        )
        assert metadata["confidence_score"] == 0.8
        assert metadata["naturalness_score"] == 0.9
        assert metadata["dialogue_state"] == DialogueState.BUILDING.value

        # Verify all components were called
        mock_threading_service.initiate_conversation_session.assert_called_once()
        mock_dialogue_engine.analyze_dialogue_context.assert_called_once()
        mock_memory_integration.extract_memory_context.assert_called_once()
        mock_memory_integration.retrieve_relevant_memories.assert_called_once()
        mock_dialogue_engine.create_memory_integration_plan.assert_called_once()
        mock_dialogue_engine.enhance_dialogue_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_response_with_fallback_to_traditional(
        self,
        enhanced_flow_manager,
        sample_conversation,
        sample_personality,
        mock_dialogue_engine,
        mock_memory_integration,
    ):
        """Test fallback to traditional memory patterns when dialogue integration fails."""
        # Arrange
        base_response = "I see your point."
        player_message = "What do you think?"
        npc_id = uuid.uuid4()

        # Mock dialogue context
        dialogue_context = DialogueContext(
            current_state=DialogueState.OPENING,
            conversation_depth_level=1,
            engagement_momentum=0.4,
            memory_reference_density=0.1,
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=None,
            player_curiosity_indicators=[],
        )
        mock_dialogue_engine.analyze_dialogue_context.return_value = dialogue_context

        # Mock memory context and retrieval
        memory_context = MagicMock()
        memory_context.emotional_tone = "neutral"
        mock_memory_integration.extract_memory_context.return_value = memory_context

        memory_retrieval = MagicMock()
        memory_retrieval.relevant_memories = [
            (
                ConversationExchange(
                    player_message="I like discussing ideas",
                    npc_response="Me too!",
                    timestamp="2023-01-01T08:00:00Z",
                ),
                0.7,
            )
        ]
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            memory_retrieval
        )

        # Mock integration plan that doesn't recommend integration
        integration_plan = MagicMock()
        integration_plan.should_integrate = True
        mock_dialogue_engine.create_memory_integration_plan.return_value = (
            integration_plan
        )

        # Mock enhancement result with no integration applied
        enhancement_result = DialogueEnhancementResult(
            enhanced_response=base_response,  # No change
            integration_applied=False,
            integration_style=IntegrationStyle.SUBTLE_HINT,
            memories_referenced=[],
            confidence_score=0.3,
            naturalness_score=0.4,
            engagement_impact=0.2,
            metadata={"fallback_reason": "low_confidence"},
        )
        mock_dialogue_engine.enhance_dialogue_response.return_value = enhancement_result

        # Mock flow library for traditional patterns
        enhanced_flow_manager.flow_library = MagicMock()
        enhanced_flow_manager.flow_library.get_trust_level_from_relationship = (
            MagicMock()
        )
        enhanced_flow_manager.flow_library.get_template_for_stage = MagicMock(
            return_value=None
        )

        # Act
        result_response, metadata = (
            await enhanced_flow_manager.enhance_response_with_advanced_memory_integration(
                conversation=sample_conversation,
                personality=sample_personality,
                base_response=base_response,
                player_message=player_message,
                npc_id=npc_id,
            )
        )

        # Assert
        assert result_response == base_response
        assert metadata["dialogue_integration_applied"] is False
        assert metadata["fallback_to_traditional"] is True

    @pytest.mark.asyncio
    async def test_enhance_response_disabled_dialogue_integration(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_conversation,
        sample_personality,
    ):
        """Test behavior when dialogue integration is disabled."""
        # Arrange
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_dialogue_integration=False,
        )

        base_response = "That makes sense."
        player_message = "I think so too."
        npc_id = uuid.uuid4()

        # Mock parent method
        with patch.object(
            EnhancedConversationFlowManager,
            "enhance_response_with_memory_patterns",
            new_callable=AsyncMock,
        ) as mock_parent:
            mock_parent.return_value = (base_response, {"memory_enhanced": False})

            # Act
            result_response, metadata = (
                await flow_manager.enhance_response_with_advanced_memory_integration(
                    conversation=sample_conversation,
                    personality=sample_personality,
                    base_response=base_response,
                    player_message=player_message,
                    npc_id=npc_id,
                )
            )

            # Assert
            assert result_response == base_response
            mock_parent.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_dialogue_readiness_enabled(
        self,
        enhanced_flow_manager,
        sample_conversation,
        sample_personality,
        mock_dialogue_engine,
        mock_memory_integration,
    ):
        """Test dialogue readiness analysis when integration is enabled."""
        # Arrange
        player_message = "Can you help me understand this better?"

        # Mock dialogue context
        dialogue_context = DialogueContext(
            current_state=DialogueState.BUILDING,
            conversation_depth_level=2,
            engagement_momentum=0.7,
            memory_reference_density=0.2,
            topic_transitions=["help", "understanding"],
            emotional_arc=["curious"],
            last_memory_integration=None,
            player_curiosity_indicators=["help", "understand", "better"],
        )
        mock_dialogue_engine.analyze_dialogue_context.return_value = dialogue_context

        # Mock memory context and retrieval
        memory_context = MagicMock()
        mock_memory_integration.extract_memory_context.return_value = memory_context

        memory_retrieval = MagicMock()
        memory_retrieval.relevant_memories = [("memory", 0.8)]
        memory_retrieval.emotional_alignment = 0.7
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            memory_retrieval
        )

        # Mock integration plan
        integration_plan = MagicMock()
        integration_plan.should_integrate = True
        integration_plan.confidence_score = 0.8
        integration_plan.flow_disruption_risk = 0.3
        integration_plan.timing_strategy = MemoryIntegrationTiming.NATURAL_PAUSE
        integration_plan.integration_style = IntegrationStyle.DIRECT_REFERENCE
        integration_plan.fallback_plan = "use_traditional_patterns"
        mock_dialogue_engine.create_memory_integration_plan.return_value = (
            integration_plan
        )

        # Act
        result = await enhanced_flow_manager.analyze_dialogue_readiness(
            conversation=sample_conversation,
            personality=sample_personality,
            player_message=player_message,
        )

        # Assert
        assert result["dialogue_integration_available"] is True
        assert result["should_integrate"] is True
        assert result["confidence_score"] == 0.8
        assert result["flow_disruption_risk"] == 0.3
        assert result["timing_strategy"] == MemoryIntegrationTiming.NATURAL_PAUSE.value
        assert result["dialogue_state"] == DialogueState.BUILDING.value
        assert result["engagement_momentum"] == 0.7
        assert result["available_memories"] == 1
        assert result["emotional_alignment"] == 0.7

    @pytest.mark.asyncio
    async def test_analyze_dialogue_readiness_disabled(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_conversation,
        sample_personality,
    ):
        """Test dialogue readiness analysis when integration is disabled."""
        # Arrange
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_dialogue_integration=False,
        )

        player_message = "Test message"

        # Act
        result = await flow_manager.analyze_dialogue_readiness(
            conversation=sample_conversation,
            personality=sample_personality,
            player_message=player_message,
        )

        # Assert
        assert result["dialogue_integration_available"] is False

    @pytest.mark.asyncio
    async def test_get_dialogue_context_summary_with_context(
        self,
        enhanced_flow_manager,
    ):
        """Test getting dialogue context summary when context exists."""
        # Arrange
        conversation_id = "conv_123"
        dialogue_context = DialogueContext(
            current_state=DialogueState.DEEPENING,
            conversation_depth_level=4,
            engagement_momentum=0.9,
            memory_reference_density=0.5,
            topic_transitions=["greeting", "weather", "personal", "deep"],
            emotional_arc=["neutral", "friendly", "interested", "engaged"],
            last_memory_integration="recent_memory",
            player_curiosity_indicators=["question", "follow_up", "deep_interest"],
        )

        # Store context in manager's cache
        enhanced_flow_manager._dialogue_contexts[conversation_id] = dialogue_context

        # Act
        result = await enhanced_flow_manager.get_dialogue_context_summary(
            conversation_id
        )

        # Assert
        assert result["context_available"] is True
        assert result["dialogue_state"] == DialogueState.DEEPENING.value
        assert result["conversation_depth"] == 4
        assert result["engagement_momentum"] == 0.9
        assert result["memory_density"] == 0.5
        assert result["topic_transitions"] == 4
        assert result["emotional_arc_length"] == 4
        assert result["last_memory_integration"] == "recent_memory"
        assert result["curiosity_indicators"] == 3

    @pytest.mark.asyncio
    async def test_get_dialogue_context_summary_no_context(
        self,
        enhanced_flow_manager,
    ):
        """Test getting dialogue context summary when no context exists."""
        # Arrange
        conversation_id = "nonexistent_conv"

        # Act
        result = await enhanced_flow_manager.get_dialogue_context_summary(
            conversation_id
        )

        # Assert
        assert result["context_available"] is False

    def test_is_dialogue_integration_enabled(
        self,
        enhanced_flow_manager,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test checking if dialogue integration is enabled."""
        # Test enabled case
        assert enhanced_flow_manager.is_dialogue_integration_enabled() is True

        # Test disabled case
        disabled_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_dialogue_integration=False,
        )
        assert disabled_manager.is_dialogue_integration_enabled() is False

    @pytest.mark.asyncio
    async def test_error_handling_in_advanced_integration(
        self,
        enhanced_flow_manager,
        sample_conversation,
        sample_personality,
        mock_dialogue_engine,
    ):
        """Test error handling in advanced memory integration."""
        # Arrange
        base_response = "I understand."
        player_message = "Test message"
        npc_id = uuid.uuid4()

        # Mock dialogue engine to raise an exception
        mock_dialogue_engine.analyze_dialogue_context.side_effect = Exception(
            "Analysis failed"
        )

        # Mock parent method for fallback
        with patch.object(
            EnhancedConversationFlowManager,
            "enhance_response_with_memory_patterns",
            new_callable=AsyncMock,
        ) as mock_parent:
            mock_parent.return_value = (
                base_response,
                {"memory_enhanced": False, "error": "fallback"},
            )

            # Act
            result_response, metadata = (
                await enhanced_flow_manager.enhance_response_with_advanced_memory_integration(
                    conversation=sample_conversation,
                    personality=sample_personality,
                    base_response=base_response,
                    player_message=player_message,
                    npc_id=npc_id,
                )
            )

            # Assert
            assert result_response == base_response
            assert "error" in metadata or "fallback" in str(metadata)
            mock_parent.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_advanced_memory_usage(
        self,
        enhanced_flow_manager,
    ):
        """Test tracking advanced memory usage."""
        # Arrange
        conversation_id = "conv_123"
        dialogue_context = DialogueContext(
            current_state=DialogueState.BUILDING,
            conversation_depth_level=2,
            engagement_momentum=0.7,
            memory_reference_density=0.3,
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=None,
            player_curiosity_indicators=[],
        )
        enhancement_result = DialogueEnhancementResult(
            enhanced_response="Enhanced response",
            integration_applied=True,
            integration_style=IntegrationStyle.SUBTLE_HINT,
            memories_referenced=["memory_1"],
            confidence_score=0.8,
            naturalness_score=0.9,
            engagement_impact=0.7,
            metadata={"test": "metadata"},
        )
        integration_plan = MagicMock()

        # Act
        await enhanced_flow_manager._track_advanced_memory_usage(
            conversation_id=conversation_id,
            dialogue_context=dialogue_context,
            enhancement_result=enhancement_result,
            integration_plan=integration_plan,
        )

        # Assert
        assert conversation_id in enhanced_flow_manager._dialogue_contexts
        assert (
            enhanced_flow_manager._dialogue_contexts[conversation_id]
            == dialogue_context
        )

        # Check that memory usage history was updated
        assert conversation_id in enhanced_flow_manager._memory_usage_history
        history = enhanced_flow_manager._memory_usage_history[conversation_id]
        assert len(history) > 0
        latest_record = history[-1]
        assert latest_record["memory_enhanced"] is True
        assert latest_record["confidence"] == 0.8
        assert latest_record["dialogue_enhanced"] is True
        assert latest_record["naturalness_score"] == 0.9
