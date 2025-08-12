"""Basic tests for dialogue memory integration components."""

import pytest
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from game_loop.core.conversation.context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueContext,
    DialogueState,
    MemoryIntegrationTiming,
    IntegrationStyle,
    MemoryIntegrationPlan,
    DialogueEnhancementResult,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
    ConversationExchange,
    MessageType,
)
from game_loop.core.conversation.flow_manager import EnhancedConversationFlowManager
from game_loop.core.conversation.dialogue_factory import DialogueAwareFlowManagerFactory
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest.fixture
def mock_session_factory():
    """Create mock database session factory."""
    factory = MagicMock(spec=DatabaseSessionFactory)
    factory.get_session = AsyncMock()
    return factory


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_memory_integration():
    """Create mock memory integration."""
    integration = AsyncMock()
    # Create a realistic memory context
    memory_context = MagicMock()
    memory_context.current_topic = "hiking"
    memory_context.emotional_tone = "enthusiastic"
    integration.extract_memory_context = AsyncMock(return_value=memory_context)

    # Create realistic memory retrieval
    memory_retrieval = MagicMock()
    memory_retrieval.relevant_memories = []
    memory_retrieval.emotional_alignment = 0.8
    integration.retrieve_relevant_memories = AsyncMock(return_value=memory_retrieval)

    return integration


@pytest.fixture
def sample_conversation():
    """Create sample conversation context."""
    exchange = ConversationExchange.create_player_message(
        player_id="123e4567-e89b-12d3-a456-426614174000",
        message_text="Hello there!",
        message_type=MessageType.GREETING,
    )

    return ConversationContext(
        conversation_id="conv_123",
        player_id="123e4567-e89b-12d3-a456-426614174000",
        npc_id="123e4567-e89b-12d3-a456-426614174001",
        relationship_level=0.6,
        conversation_history=[exchange],
    )


@pytest.fixture
def sample_personality():
    """Create sample NPC personality."""
    return NPCPersonality(
        npc_id="123e4567-e89b-12d3-a456-426614174001",
        traits={"friendly": 0.7, "helpful": 0.8},
        knowledge_areas=["outdoor_activities"],
        speech_patterns={"tone": "friendly"},
        relationships={"123e4567-e89b-12d3-a456-426614174000": 0.6},
        background_story="A helpful park ranger.",
    )


class TestDialogueContext:
    """Test DialogueContext dataclass."""

    def test_dialogue_context_creation(self):
        """Test creating DialogueContext."""
        context = DialogueContext(
            conversation_id="test_conv_123",
            current_state=DialogueState.BUILDING,
            conversation_depth_level=3,
            engagement_momentum=0.8,
            memory_reference_density=0.4,
            topic_transitions=[("greeting", "weather", 0.8)],
            emotional_arc=[("neutral", 0.5), ("positive", 0.7)],
            last_memory_integration=time.time() - 300,
            player_curiosity_indicators=["question", "follow_up"],
        )

        assert context.conversation_id == "test_conv_123"
        assert context.current_state == DialogueState.BUILDING
        assert context.conversation_depth_level == 3
        assert context.engagement_momentum == 0.8
        assert len(context.topic_transitions) == 1
        assert len(context.emotional_arc) == 2


class TestMemoryIntegrationPlan:
    """Test MemoryIntegrationPlan dataclass."""

    def test_memory_integration_plan_creation(self):
        """Test creating MemoryIntegrationPlan."""
        plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.8,
            timing_strategy=MemoryIntegrationTiming.IMMEDIATE,
            integration_style=IntegrationStyle.DIRECT_REFERENCE,
            memory_references=[{"memory_id": "mem_123", "confidence": 0.9}],
            flow_disruption_risk=0.2,
            emotional_alignment=0.8,
            fallback_plan="use_alternative",
            enhancement_text="Enhanced with memory",
        )

        assert plan.should_integrate is True
        assert plan.confidence_score == 0.8
        assert len(plan.memory_references) == 1


class TestDialogueEnhancementResult:
    """Test DialogueEnhancementResult dataclass."""

    def test_dialogue_enhancement_result_creation(self):
        """Test creating DialogueEnhancementResult."""
        result = DialogueEnhancementResult(
            enhanced_response="Enhanced response with memory",
            integration_applied=True,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            memories_referenced=2,
            confidence_score=0.85,
            naturalness_score=0.9,
            engagement_impact=0.7,
            metadata={"timing": "perfect", "style": "natural"},
        )

        assert result.integration_applied is True
        assert result.memories_referenced == 2
        assert result.confidence_score == 0.85
        assert "timing" in result.metadata


class TestEnhancedConversationFlowManager:
    """Test enhanced conversation flow manager."""

    @pytest.mark.asyncio
    async def test_enhanced_flow_manager_creation(
        self,
        mock_session_factory,
        mock_memory_integration,
    ):
        """Test creating enhanced flow manager."""
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_dialogue_integration=True,
        )

        assert flow_manager.enable_dialogue_integration is True
        assert flow_manager.memory_integration is mock_memory_integration

    @pytest.mark.asyncio
    async def test_enhanced_flow_manager_with_standard_enhancement(
        self,
        mock_session_factory,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test enhanced flow manager with standard memory patterns."""
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_dialogue_integration=False,  # Disable to test fallback
        )

        # Mock the parent method
        with patch.object(
            EnhancedConversationFlowManager,
            "enhance_response_with_memory_patterns",
            new_callable=AsyncMock,
        ) as mock_parent:
            mock_parent.return_value = ("Enhanced response", {"memory_enhanced": True})

            result_response, metadata = (
                await flow_manager.enhance_response_with_advanced_memory_integration(
                    conversation=sample_conversation,
                    personality=sample_personality,
                    base_response="Base response",
                    player_message="Test message",
                    npc_id=uuid.UUID(sample_conversation.npc_id),
                )
            )

            assert result_response == "Enhanced response"
            assert isinstance(metadata, dict)
            mock_parent.assert_called_once()


class TestDialogueAwareFlowManagerFactory:
    """Test dialogue-aware flow manager factory."""

    def test_create_enhanced_flow_manager(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
    ):
        """Test creating enhanced flow manager."""
        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
        )

        assert isinstance(flow_manager, EnhancedConversationFlowManager)
        assert flow_manager.is_dialogue_integration_enabled() is True

    def test_create_standard_flow_manager(
        self,
        mock_session_factory,
        mock_memory_integration,
    ):
        """Test creating standard flow manager."""
        flow_manager = DialogueAwareFlowManagerFactory.create_standard_flow_manager(
            session_factory=mock_session_factory,
            memory_integration=mock_memory_integration,
        )

        assert isinstance(flow_manager, EnhancedConversationFlowManager)
        assert flow_manager.is_dialogue_integration_enabled() is False

    def test_create_threading_only_flow_manager(
        self,
        mock_session_factory,
        mock_memory_integration,
    ):
        """Test creating threading-only flow manager."""
        flow_manager = (
            DialogueAwareFlowManagerFactory.create_threading_only_flow_manager(
                session_factory=mock_session_factory,
                memory_integration=mock_memory_integration,
            )
        )

        assert isinstance(flow_manager, EnhancedConversationFlowManager)
        assert flow_manager.is_dialogue_integration_enabled() is False
        assert flow_manager.enable_conversation_threading is True

    def test_create_dialogue_only_flow_manager(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
    ):
        """Test creating dialogue-only flow manager."""
        flow_manager = (
            DialogueAwareFlowManagerFactory.create_dialogue_only_flow_manager(
                session_factory=mock_session_factory,
                llm_client=mock_llm_client,
                memory_integration=mock_memory_integration,
            )
        )

        assert isinstance(flow_manager, EnhancedConversationFlowManager)
        assert flow_manager.is_dialogue_integration_enabled() is True
        assert flow_manager.enable_conversation_threading is False


class TestIntegrationEnums:
    """Test dialogue integration enums."""

    def test_dialogue_state_enum(self):
        """Test DialogueState enum values."""
        assert DialogueState.OPENING.value == "opening"
        assert DialogueState.BUILDING.value == "building"
        assert DialogueState.DEEPENING.value == "deepening"
        assert DialogueState.CLIMAX.value == "climax"
        assert DialogueState.WINDING_DOWN.value == "winding_down"
        assert DialogueState.CLOSING.value == "closing"

    def test_memory_integration_timing_enum(self):
        """Test MemoryIntegrationTiming enum values."""
        assert MemoryIntegrationTiming.IMMEDIATE.value == "immediate"
        assert MemoryIntegrationTiming.NATURAL_PAUSE.value == "natural_pause"
        assert MemoryIntegrationTiming.TOPIC_BRIDGE.value == "topic_bridge"
        assert (
            MemoryIntegrationTiming.RESPONSE_ENHANCEMENT.value == "response_enhancement"
        )
        assert MemoryIntegrationTiming.DELAYED.value == "delayed"

    def test_integration_style_enum(self):
        """Test IntegrationStyle enum values."""
        assert IntegrationStyle.SUBTLE_HINT.value == "subtle_hint"
        assert IntegrationStyle.DIRECT_REFERENCE.value == "direct_reference"
        assert IntegrationStyle.EMOTIONAL_CONNECTION.value == "emotional_connection"
        assert IntegrationStyle.COMPARATIVE.value == "comparative"
        assert IntegrationStyle.BUILDUP.value == "buildup"
        assert IntegrationStyle.NARRATIVE_WEAVING.value == "narrative_weaving"
