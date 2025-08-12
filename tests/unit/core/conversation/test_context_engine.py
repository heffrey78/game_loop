"""Tests for dialogue memory integration context engine."""

import pytest
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from game_loop.core.conversation.context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueContext,
    DialogueState,
    MemoryIntegrationTiming,
    IntegrationStyle,
    DialogueEnhancementResult,
    MemoryIntegrationPlan,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
    ConversationExchange,
    MessageType,
)
from game_loop.core.conversation.conversation_threading import ThreadingAnalysis
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
    client.extract_topic = AsyncMock(return_value="test_topic")
    client.detect_emotional_tone = AsyncMock(return_value="neutral")
    client.analyze_curiosity = AsyncMock(return_value=["question", "exploration"])
    client.assess_naturalness = AsyncMock(return_value=0.8)
    client.calculate_engagement_impact = AsyncMock(return_value=0.7)
    return client


@pytest.fixture
def mock_threading_service():
    """Create mock threading service."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_memory_integration():
    """Create mock memory integration."""
    integration = AsyncMock()
    return integration


@pytest.fixture
def sample_conversation():
    """Create sample conversation context."""
    exchange = ConversationExchange.create_player_message(
        player_id="player_456",
        message_text="Hello there!",
        message_type=MessageType.GREETING,
    )
    exchange.emotion = "friendly"

    return ConversationContext(
        conversation_id="conv_123",
        player_id="player_456",
        npc_id="npc_789",
        relationship_level=0.6,
        conversation_history=[exchange],
    )


@pytest.fixture
def sample_personality():
    """Create sample NPC personality."""
    personality = NPCPersonality(
        npc_id="npc_789",
        traits={"friendly": 0.7, "cautious": 0.3, "helpful": 0.8},
        knowledge_areas=["outdoor_activities", "nature"],
        speech_patterns={"tone": "friendly", "formality": "casual"},
        relationships={"player_456": 0.6},
        background_story="A helpful park ranger who loves the outdoors.",
    )
    # Add conflict resolution strategy as metadata
    personality.speech_patterns["conflict_resolution_strategy"] = "diplomatic"
    return personality


@pytest.fixture
def dialogue_engine(
    mock_session_factory,
    mock_llm_client,
    mock_threading_service,
    mock_memory_integration,
):
    """Create dialogue memory integration engine."""
    return DialogueMemoryIntegrationEngine(
        session_factory=mock_session_factory,
        llm_client=mock_llm_client,
        threading_service=mock_threading_service,
        memory_integration=mock_memory_integration,
        max_references_per_response=2,
        naturalness_threshold=0.7,
        engagement_threshold=0.6,
    )


class TestDialogueMemoryIntegrationEngine:
    """Tests for DialogueMemoryIntegrationEngine."""

    @pytest.mark.asyncio
    async def test_analyze_dialogue_context_basic(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test basic dialogue context analysis."""
        # Arrange
        player_message = "What do you think about the weather?"

        # Act
        result = await dialogue_engine.analyze_dialogue_context(
            conversation=sample_conversation,
            player_message=player_message,
            npc_personality=sample_personality,
            threading_context=None,
        )

        # Assert
        assert isinstance(result, DialogueContext)
        assert result.current_state in DialogueState
        assert result.conversation_depth_level >= 1
        assert 0.0 <= result.engagement_momentum <= 1.0
        assert result.memory_reference_density >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_dialogue_context_with_threading(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test dialogue context analysis with threading context."""
        # Arrange
        player_message = "Remember what we talked about last time?"
        threading_context = MagicMock()
        threading_context.topic_continuity_score = 0.8
        threading_context.active_thread = MagicMock()
        threading_context.active_thread.conversation_count = 3

        # Act
        result = await dialogue_engine.analyze_dialogue_context(
            conversation=sample_conversation,
            player_message=player_message,
            npc_personality=sample_personality,
            threading_context=threading_context,
        )

        # Assert
        assert result.conversation_depth_level >= 2  # Should be higher with threading
        assert result.engagement_momentum > 0.5  # Should show engagement

    @pytest.mark.asyncio
    async def test_create_memory_integration_plan_should_integrate(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test creating memory integration plan when integration is recommended."""
        # Arrange
        dialogue_context = DialogueContext(
            conversation_id="conv_123",
            current_state=DialogueState.DEEPENING,  # Better state for integration
            conversation_depth_level=3,
            engagement_momentum=0.9,  # Higher engagement
            memory_reference_density=0.2,  # Low density allows more integration
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=None,
            player_curiosity_indicators=["question"],
        )

        memory_retrieval = MagicMock()
        hiking_exchange = ConversationExchange.create_player_message(
            player_id="player_456",
            message_text="I love hiking",
            message_type=MessageType.STATEMENT,
        )
        memory_retrieval.relevant_memories = [(hiking_exchange, 0.9)]
        memory_retrieval.emotional_alignment = 0.8
        memory_retrieval.context_score = 0.9  # High context score for integration
        memory_retrieval.disclosure_recommendation = MagicMock()
        memory_retrieval.flow_analysis = MagicMock()

        # Act
        result = await dialogue_engine.create_memory_integration_plan(
            dialogue_context=dialogue_context,
            memory_retrieval=memory_retrieval,
            conversation=sample_conversation,
            npc_personality=sample_personality,
        )

        # Assert
        assert isinstance(result, MemoryIntegrationPlan)
        assert result.should_integrate is True
        assert result.confidence_score > 0.5
        assert result.timing_strategy in MemoryIntegrationTiming
        assert result.integration_style in IntegrationStyle

    @pytest.mark.asyncio
    async def test_create_memory_integration_plan_should_not_integrate(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test creating memory integration plan when integration is not recommended."""
        # Arrange
        dialogue_context = DialogueContext(
            conversation_id="conv_123",
            current_state=DialogueState.OPENING,
            conversation_depth_level=1,
            engagement_momentum=0.3,  # Low engagement
            memory_reference_density=0.8,  # High density prevents more integration
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=time.time() - 10,  # Recent timestamp
            player_curiosity_indicators=[],
        )

        memory_retrieval = MagicMock()
        memory_retrieval.relevant_memories = []  # No memories
        memory_retrieval.emotional_alignment = 0.2
        memory_retrieval.context_score = 0.3  # Low context score
        memory_retrieval.disclosure_recommendation = MagicMock()
        memory_retrieval.flow_analysis = MagicMock()

        # Act
        result = await dialogue_engine.create_memory_integration_plan(
            dialogue_context=dialogue_context,
            memory_retrieval=memory_retrieval,
            conversation=sample_conversation,
            npc_personality=sample_personality,
        )

        # Assert
        assert result.should_integrate is False
        assert result.confidence_score <= 0.5
        assert result.fallback_plan is not None

    @pytest.mark.asyncio
    async def test_enhance_dialogue_response_with_integration(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test enhancing dialogue response with memory integration."""
        # Arrange
        base_response = "I think that's a great idea!"

        integration_plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.8,
            timing_strategy=MemoryIntegrationTiming.NATURAL_PAUSE,
            integration_style=IntegrationStyle.SUBTLE_HINT,
            memory_references=[
                {
                    "memory": ConversationExchange.create_player_message(
                        player_id="player_456",
                        message_text="I love outdoor activities",
                        message_type=MessageType.STATEMENT,
                    ),
                    "confidence": 0.9,
                }
            ],
            flow_disruption_risk=0.2,
            emotional_alignment=0.8,
            enhancement_text="Memory reference about outdoor activities",
            fallback_plan="use_generic_response",
        )

        dialogue_context = DialogueContext(
            conversation_id="conv_123",
            current_state=DialogueState.BUILDING,
            conversation_depth_level=2,
            engagement_momentum=0.7,
            memory_reference_density=0.3,
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=None,
            player_curiosity_indicators=[],
        )

        # Act
        result = await dialogue_engine.enhance_dialogue_response(
            base_response=base_response,
            integration_plan=integration_plan,
            dialogue_context=dialogue_context,
            conversation=sample_conversation,
            npc_personality=sample_personality,
        )

        # Assert
        assert isinstance(result, DialogueEnhancementResult)
        assert result.integration_applied is True
        assert result.enhanced_response != base_response
        assert result.confidence_score > 0.0
        assert result.naturalness_score > 0.0
        assert result.memories_referenced > 0

    @pytest.mark.asyncio
    async def test_enhance_dialogue_response_fallback(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test dialogue response enhancement with fallback to base response."""
        # Arrange
        base_response = "I understand your concern."

        integration_plan = MemoryIntegrationPlan(
            should_integrate=False,
            confidence_score=0.3,
            timing_strategy=MemoryIntegrationTiming.DELAYED,
            integration_style=IntegrationStyle.SUBTLE_HINT,
            memory_references=[],
            flow_disruption_risk=0.8,  # High risk
            emotional_alignment=0.4,  # Low alignment
            enhancement_text="",
            fallback_plan="use_base_response",
        )

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

        # Act
        result = await dialogue_engine.enhance_dialogue_response(
            base_response=base_response,
            integration_plan=integration_plan,
            dialogue_context=dialogue_context,
            conversation=sample_conversation,
            npc_personality=sample_personality,
        )

        # Assert
        assert result.integration_applied is False
        assert result.enhanced_response == base_response
        assert result.memories_referenced == 0

    @pytest.mark.asyncio
    async def test_resolve_memory_conflicts_diplomatic(
        self,
        dialogue_engine,
        sample_conversation,
        sample_personality,
    ):
        """Test memory conflict resolution with diplomatic strategy."""
        # Arrange
        conflicting_memories = [
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="I hate rainy days",
                message_type=MessageType.STATEMENT,
            ),
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="I love rainy days",
                message_type=MessageType.STATEMENT,
            ),
        ]

        sample_personality.speech_patterns["conflict_resolution_strategy"] = (
            "diplomatic"
        )

        # Mock conflict assessment
        dialogue_engine._assess_memory_conflict = AsyncMock(return_value=0.8)

        # Act
        result = await dialogue_engine.resolve_memory_conflicts(
            conflicting_memories=conflicting_memories,
            conversation=sample_conversation,
            npc_personality=sample_personality,
        )

        # Assert
        assert "resolution_strategy" in result
        assert result["resolution_strategy"] == "diplomatic"
        assert "resolved_memories" in result

    @pytest.mark.asyncio
    async def test_calculate_engagement_momentum(
        self,
        dialogue_engine,
        sample_conversation,
    ):
        """Test engagement momentum calculation."""
        # Arrange
        player_message = "This is really interesting! Tell me more."
        curiosity_indicators = ["interesting", "tell me more", "question"]

        # Act
        result = await dialogue_engine._calculate_engagement_momentum(
            conversation=sample_conversation,
            player_message=player_message,
            curiosity_indicators=curiosity_indicators,
        )

        # Assert
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should be high due to curiosity indicators

    @pytest.mark.asyncio
    async def test_determine_dialogue_state_opening(
        self,
        dialogue_engine,
        sample_conversation,
    ):
        """Test dialogue state determination for opening state."""
        # Arrange
        sample_conversation.conversation_history = [
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="Hello",
                message_type=MessageType.GREETING,
            )
        ]

        # Act
        result = dialogue_engine._determine_dialogue_state(
            conversation=sample_conversation,
            player_message="How are you today?",
        )

        # Assert
        assert result == DialogueState.OPENING

    @pytest.mark.asyncio
    async def test_determine_dialogue_state_building(
        self,
        dialogue_engine,
        sample_conversation,
    ):
        """Test dialogue state determination for building state."""
        # Arrange
        sample_conversation.conversation_history = [
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="Hello",
                message_type=MessageType.GREETING,
            ),
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="How are you?",
                message_type=MessageType.QUESTION,
            ),
            ConversationExchange.create_player_message(
                player_id="player_456",
                message_text="What do you think about this place?",
                message_type=MessageType.QUESTION,
            ),
        ]

        # Act
        result = dialogue_engine._determine_dialogue_state(
            conversation=sample_conversation,
            player_message="I've been thinking about what you said earlier.",
        )

        # Assert
        assert result == DialogueState.BUILDING

    @pytest.mark.asyncio
    async def test_assess_memory_conflict(
        self,
        dialogue_engine,
    ):
        """Test memory conflict assessment."""
        # Arrange
        memory1 = ConversationExchange.create_player_message(
            player_id="player_456",
            message_text="I love cats",
            message_type=MessageType.STATEMENT,
        )

        memory2 = ConversationExchange.create_player_message(
            player_id="player_456",
            message_text="I'm allergic to cats",
            message_type=MessageType.STATEMENT,
        )

        # Act
        result = await dialogue_engine._assess_memory_conflict(memory1, memory2)

        # Assert
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should detect conflict between loving and being allergic

    @pytest.mark.asyncio
    async def test_create_natural_memory_reference_subtle(
        self,
        dialogue_engine,
        sample_personality,
    ):
        """Test creating natural memory reference with subtle style."""
        # Arrange
        memory = ConversationExchange.create_player_message(
            player_id="player_456",
            message_text="I went hiking last weekend",
            message_type=MessageType.STATEMENT,
        )

        style = IntegrationStyle.SUBTLE_HINT
        dialogue_context = DialogueContext(
            conversation_id="conv_123",
            current_state=DialogueState.BUILDING,
            conversation_depth_level=2,
            engagement_momentum=0.7,
            memory_reference_density=0.3,
            topic_transitions=[],
            emotional_arc=[],
            last_memory_integration=None,
            player_curiosity_indicators=[],
        )

        # Act
        result = await dialogue_engine._create_natural_memory_reference(
            memory=memory,
            integration_style=style,
            npc_personality=sample_personality,
            dialogue_context=dialogue_context,
        )

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be subtle, not directly quoting the memory

    @pytest.mark.asyncio
    async def test_weave_memory_into_response(
        self,
        dialogue_engine,
    ):
        """Test weaving memory reference into response."""
        # Arrange
        base_response = "That's an interesting point."
        memory_reference = "speaking of outdoor activities"
        timing = MemoryIntegrationTiming.NATURAL_PAUSE
        style = IntegrationStyle.SUBTLE_HINT

        # Act
        result = dialogue_engine._weave_memory_into_response(
            base_response=base_response,
            memory_reference=memory_reference,
            timing=timing,
            style=style,
        )

        # Assert
        assert isinstance(result, str)
        assert base_response in result
        assert memory_reference in result
        assert len(result) > len(base_response)


class TestDialogueContext:
    """Tests for DialogueContext dataclass."""

    def test_dialogue_context_creation(self):
        """Test DialogueContext creation."""
        context = DialogueContext(
            conversation_id="test_conv_123",
            current_state=DialogueState.BUILDING,
            conversation_depth_level=3,
            engagement_momentum=0.8,
            memory_reference_density=0.4,
            topic_transitions=[
                ("greeting", "weather", 0.8),
                ("weather", "hobbies", 0.9),
            ],
            emotional_arc=[("neutral", 0.5), ("positive", 0.7), ("excited", 0.9)],
            last_memory_integration=time.time() - 300,  # 5 minutes ago
            player_curiosity_indicators=["question", "follow_up"],
        )

        assert context.current_state == DialogueState.BUILDING
        assert context.conversation_depth_level == 3
        assert context.engagement_momentum == 0.8
        assert len(context.topic_transitions) == 2
        assert len(context.emotional_arc) == 3


class TestMemoryIntegrationPlan:
    """Tests for MemoryIntegrationPlan dataclass."""

    def test_memory_integration_plan_creation(self):
        """Test MemoryIntegrationPlan creation."""
        memory = ConversationExchange.create_player_message(
            player_id="test_player",
            message_text="Test message",
            message_type=MessageType.STATEMENT,
        )

        plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.8,
            timing_strategy=MemoryIntegrationTiming.IMMEDIATE,
            integration_style=IntegrationStyle.DIRECT_REFERENCE,
            memory_references=[{"memory": memory, "confidence": 0.9}],
            flow_disruption_risk=0.2,
            emotional_alignment=0.8,
            fallback_plan="use_alternative",
            enhancement_text="Enhanced with memory",
        )

        assert plan.should_integrate is True
        assert plan.confidence_score == 0.8
        assert len(plan.memory_references) == 1


class TestDialogueEnhancementResult:
    """Tests for DialogueEnhancementResult dataclass."""

    def test_dialogue_enhancement_result_creation(self):
        """Test DialogueEnhancementResult creation."""
        result = DialogueEnhancementResult(
            enhanced_response="Enhanced response with memory",
            integration_applied=True,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            memories_referenced=2,  # Should be integer, not list
            confidence_score=0.85,
            naturalness_score=0.9,
            engagement_impact=0.7,
            metadata={"timing": "perfect", "style": "natural"},
        )

        assert result.integration_applied is True
        assert result.memories_referenced == 2
        assert result.confidence_score == 0.85
        assert "timing" in result.metadata


@pytest.mark.asyncio
async def test_integration_error_handling(
    dialogue_engine,
    sample_conversation,
    sample_personality,
):
    """Test error handling in dialogue integration."""
    # Arrange
    dialogue_engine.llm_client.extract_topic.side_effect = Exception("LLM Error")

    # Act
    result = await dialogue_engine.analyze_dialogue_context(
        conversation=sample_conversation,
        player_message="Test message",
        npc_personality=sample_personality,
        threading_context=None,
    )

    # Assert - Should handle gracefully and return default context
    assert isinstance(result, DialogueContext)
    assert result.current_state == DialogueState.OPENING  # Default fallback
