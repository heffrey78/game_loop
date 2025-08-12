"""
Comprehensive integration tests for memory-driven dialogue system.

This test suite verifies the complete memory integration functionality
including dialogue context analysis, memory reference generation,
and natural language weaving of memories into NPC responses.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from game_loop.core.conversation.context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueContext,
    DialogueState,
    MemoryIntegrationPlan,
    MemoryIntegrationTiming,
    IntegrationStyle,
    DialogueEnhancementResult,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
    ConversationExchange,
    MessageType,
    ConversationStatus,
)
from game_loop.core.conversation.memory_integration import (
    MemoryContext,
    MemoryRetrievalResult,
)
from game_loop.core.conversation.flow_manager import (
    EnhancedConversationFlowManager,
)
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest.fixture
def mock_session_factory():
    """Mock database session factory."""
    factory = MagicMock(spec=DatabaseSessionFactory)
    session_mock = AsyncMock()
    factory.get_session.return_value.__aenter__.return_value = session_mock
    return factory


@pytest.fixture
def mock_memory_integration():
    """Mock memory integration interface."""
    memory_interface = AsyncMock()
    
    # Mock memory context extraction
    memory_interface.extract_memory_context.return_value = MemoryContext(
        current_topic="village history",
        emotional_tone="curious",
        conversation_history=[],
        player_interests=["village history", "ancient lore"],
        npc_knowledge_areas=["history", "lore"],
        topic_continuity_score=0.8,
    )
    
    # Mock memory retrieval
    mock_exchange = ConversationExchange(
        exchange_id="mock-exchange-id",
        speaker_id="mock-npc-id",
        message_text="I remember you mentioned wanting to learn about the ancient ruins",
        message_type=MessageType.STATEMENT,
        emotion="thoughtful",
        timestamp=datetime.now(timezone.utc).timestamp(),
    )
    
    from game_loop.core.conversation.memory_integration import MemoryDisclosureLevel, ConversationFlowState
    
    memory_interface.retrieve_relevant_memories.return_value = MemoryRetrievalResult(
        relevant_memories=[(mock_exchange, 0.85)],
        context_score=0.8,
        emotional_alignment=0.7,
        disclosure_recommendation=MemoryDisclosureLevel.DIRECT_REFERENCES,
        flow_analysis=ConversationFlowState.NATURAL,
    )
    
    return memory_interface


@pytest.fixture
def sample_conversation_context():
    """Sample conversation context for testing."""
    return ConversationContext(
        conversation_id="test-conversation-id",
        player_id="test-player-id",
        npc_id="test-npc-id",
        topic="village history",
        mood="curious",
        relationship_level=0.7,
        conversation_history=[
            ConversationExchange(
                exchange_id="exchange-1",
                speaker_id="test-player-id",
                message_text="Tell me about the village history",
                message_type=MessageType.QUESTION,
                emotion="curious",
                timestamp=datetime.now(timezone.utc).timestamp(),
            ),
        ],
        context_data={},
        status=ConversationStatus.ACTIVE,
        started_at=datetime.now(timezone.utc).timestamp(),
        last_updated=datetime.now(timezone.utc).timestamp(),
    )


@pytest.fixture
def sample_npc_personality():
    """Sample NPC personality for testing."""
    return NPCPersonality(
        npc_id="test-npc-id",
        traits={"wise": 0.8, "helpful": 0.9, "talkative": 0.7},
        knowledge_areas=["history", "ancient_secrets", "village_lore"],
        speech_patterns={"formality": "medium", "directness": "high"},
        relationships={},
        background_story="A knowledgeable village elder with deep understanding of local history",
        default_mood="contemplative",
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    return AsyncMock()

@pytest.fixture
def mock_threading_service():
    """Mock conversation threading service."""
    return AsyncMock()

@pytest.fixture
def dialogue_integration_engine(mock_memory_integration, mock_session_factory, mock_llm_client, mock_threading_service):
    """Create dialogue memory integration engine."""
    return DialogueMemoryIntegrationEngine(
        session_factory=mock_session_factory,
        llm_client=mock_llm_client,
        threading_service=mock_threading_service,
        memory_integration=mock_memory_integration,
    )


@pytest.fixture
def enhanced_flow_manager(mock_memory_integration, mock_session_factory, dialogue_integration_engine):
    """Create enhanced conversation flow manager with dialogue integration."""
    return EnhancedConversationFlowManager(
        memory_integration=mock_memory_integration,
        session_factory=mock_session_factory,
        enable_conversation_threading=True,
        enable_dialogue_integration=True,
        dialogue_integration_engine=dialogue_integration_engine,
    )


class TestDialogueContextAnalysis:
    """Test dialogue context analysis functionality."""

    @pytest.mark.asyncio
    async def test_dialogue_context_analysis_basic(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test basic dialogue context analysis."""
        player_message = "What happened to the ancient ruins?"
        
        # Mock threading context
        threading_context = None
        
        dialogue_context = await dialogue_integration_engine.analyze_dialogue_context(
            conversation=sample_conversation_context,
            player_message=player_message,
            npc_personality=sample_npc_personality,
            threading_context=threading_context,
        )
        
        assert isinstance(dialogue_context, DialogueContext)
        assert dialogue_context.current_topic == "ancient ruins"
        assert dialogue_context.current_state in DialogueState
        assert 0.0 <= dialogue_context.engagement_momentum <= 1.0
        assert 0.0 <= dialogue_context.conversation_depth_level <= 1.0
        assert isinstance(dialogue_context.player_curiosity_indicators, list)
        assert isinstance(dialogue_context.topic_transitions, list)

    @pytest.mark.asyncio
    async def test_dialogue_context_with_high_engagement(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test dialogue context analysis with high engagement scenario."""
        player_message = "That's fascinating! Tell me more about the ancient magic!"
        
        # Mock high engagement conversation
        sample_conversation_context.relationship_level = 0.9
        sample_conversation_context.mood = "excited"
        
        dialogue_context = await dialogue_integration_engine.analyze_dialogue_context(
            conversation=sample_conversation_context,
            player_message=player_message,
            npc_personality=sample_npc_personality,
            threading_context=None,
        )
        
        assert dialogue_context.engagement_momentum > 0.6
        assert dialogue_context.current_state in [DialogueState.BUILDING, DialogueState.DEEPENING]
        assert len(dialogue_context.player_curiosity_indicators) > 0


class TestMemoryIntegrationPlan:
    """Test memory integration plan creation and execution."""

    @pytest.mark.asyncio
    async def test_memory_integration_plan_creation(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test creation of memory integration plan."""
        # Mock dialogue context
        dialogue_context = DialogueContext(
            current_topic="ancient ruins",
            current_state=DialogueState.DEEPENING,
            engagement_momentum=0.8,
            conversation_depth_level=0.7,
            memory_reference_density=0.3,
            player_curiosity_indicators=["What happened", "Tell me more"],
            topic_transitions=[],
            emotional_arc=["curious", "excited"],
            last_memory_integration=None,
        )
        
        # Mock memory retrieval with high confidence
        mock_exchange = ConversationExchange(
            exchange_id="memory-exchange",
            speaker_id="test-player-id",
            message_text="I've always wondered about those old stones",
            message_type=MessageType.STATEMENT,
            emotion="curious",
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        
        from game_loop.core.conversation.memory_integration import MemoryDisclosureLevel, ConversationFlowState
        
        memory_retrieval = MemoryRetrievalResult(
            relevant_memories=[(mock_exchange, 0.9)],
            context_score=0.85,
            emotional_alignment=0.8,
            disclosure_recommendation=MemoryDisclosureLevel.DETAILED_MEMORIES,
            flow_analysis=ConversationFlowState.NATURAL,
        )
        
        integration_plan = await dialogue_integration_engine.create_memory_integration_plan(
            dialogue_context=dialogue_context,
            memory_retrieval=memory_retrieval,
            conversation=sample_conversation_context,
            npc_personality=sample_npc_personality,
        )
        
        assert isinstance(integration_plan, MemoryIntegrationPlan)
        assert integration_plan.should_integrate is True
        assert integration_plan.confidence_score >= 0.7
        assert integration_plan.timing_strategy in MemoryIntegrationTiming
        assert integration_plan.integration_style in IntegrationStyle
        assert isinstance(integration_plan.flow_disruption_risk, str)
        assert len(integration_plan.memory_references) > 0

    @pytest.mark.asyncio
    async def test_memory_integration_plan_low_confidence_rejection(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test rejection of memory integration for low confidence memories."""
        # Mock dialogue context
        dialogue_context = DialogueContext(
            current_topic="weather",
            current_state=DialogueState.OPENING,
            engagement_momentum=0.3,
            conversation_depth_level=0.2,
            memory_reference_density=0.1,
            player_curiosity_indicators=[],
            topic_transitions=[],
            emotional_arc=["neutral"],
            last_memory_integration=None,
        )
        
        # Mock low confidence memory retrieval
        memory_retrieval = MemoryRetrievalResult(
            relevant_memories=[],
            context_score=0.2,
            emotional_alignment=0.1,
            disclosure_recommendation=MemoryDisclosureLevel.NONE,
            flow_analysis=ConversationFlowState.FORCED,
        )
        
        integration_plan = await dialogue_integration_engine.create_memory_integration_plan(
            dialogue_context=dialogue_context,
            memory_retrieval=memory_retrieval,
            conversation=sample_conversation_context,
            npc_personality=sample_npc_personality,
        )
        
        assert integration_plan.should_integrate is False
        assert integration_plan.confidence_score < 0.7


class TestDialogueResponseEnhancement:
    """Test dialogue response enhancement with memory integration."""

    @pytest.mark.asyncio
    async def test_dialogue_response_enhancement_successful(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test successful dialogue response enhancement."""
        base_response = "The ancient ruins hold many secrets."
        
        # Mock high-quality integration plan
        integration_plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.85,
            timing_strategy=MemoryIntegrationTiming.NATURAL_PAUSE,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            flow_disruption_risk="LOW",
            memory_references=[
                {
                    "memory": ConversationExchange(
                        exchange_id="memory-ref",
                        speaker_id="test-player-id",
                        message_text="I've always wondered about those ruins",
                        message_type=MessageType.STATEMENT,
                        emotion="curious",
                        timestamp=datetime.now(timezone.utc).timestamp(),
                    ),
                    "relevance_score": 0.9,
                    "emotional_weight": 0.8,
                }
            ],
            fallback_plan="subtle_reference",
        )
        
        dialogue_context = DialogueContext(
            current_topic="ancient ruins",
            current_state=DialogueState.DEEPENING,
            engagement_momentum=0.8,
            conversation_depth_level=0.7,
            memory_reference_density=0.3,
            player_curiosity_indicators=["wondered"],
            topic_transitions=[],
            emotional_arc=["curious"],
            last_memory_integration=None,
        )
        
        with patch.object(
            dialogue_integration_engine,
            'create_memory_integration_plan',
            return_value=integration_plan
        ):
            enhancement_result = await dialogue_integration_engine.enhance_dialogue_response(
                base_response=base_response,
                integration_plan=integration_plan,
                dialogue_context=dialogue_context,
                conversation=sample_conversation_context,
                npc_personality=sample_npc_personality,
            )
        
        assert enhancement_result.integration_applied is True
        assert enhancement_result.enhanced_response != base_response
        assert enhancement_result.confidence_score >= 0.7
        assert enhancement_result.naturalness_score > 0.5
        assert len(enhancement_result.enhanced_response) > len(base_response)

    @pytest.mark.asyncio
    async def test_dialogue_response_enhancement_fallback(
        self,
        dialogue_integration_engine,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test dialogue response enhancement with fallback to original response."""
        base_response = "I don't know much about that."
        
        # Mock low-quality integration plan
        integration_plan = MemoryIntegrationPlan(
            should_integrate=False,
            confidence_score=0.3,
            timing_strategy=MemoryIntegrationTiming.IMMEDIATE,
            integration_style=IntegrationStyle.SUBTLE_HINT,
            flow_disruption_risk="HIGH",
            memory_references=[],
            fallback_plan="no_integration",
        )
        
        dialogue_context = DialogueContext(
            current_topic="unknown_topic",
            current_state=DialogueState.OPENING,
            engagement_momentum=0.2,
            conversation_depth_level=0.1,
            memory_reference_density=0.0,
            player_curiosity_indicators=[],
            topic_transitions=[],
            emotional_arc=["neutral"],
            last_memory_integration=None,
        )
        
        enhancement_result = await dialogue_integration_engine.enhance_dialogue_response(
            base_response=base_response,
            integration_plan=integration_plan,
            dialogue_context=dialogue_context,
            conversation=sample_conversation_context,
            npc_personality=sample_npc_personality,
        )
        
        assert enhancement_result.integration_applied is False
        assert enhancement_result.enhanced_response == base_response
        assert enhancement_result.confidence_score < 0.7


class TestEnhancedFlowManagerIntegration:
    """Test the enhanced flow manager with advanced memory integration."""

    @pytest.mark.asyncio
    async def test_enhanced_flow_manager_memory_integration(
        self,
        enhanced_flow_manager,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test enhanced flow manager with advanced memory integration."""
        base_response = "That's an interesting question about the ruins."
        player_message = "What secrets do the ancient ruins hold?"
        npc_id = uuid.UUID(sample_conversation_context.npc_id)
        
        enhanced_response, metadata = await enhanced_flow_manager.enhance_response_with_advanced_memory_integration(
            conversation=sample_conversation_context,
            personality=sample_npc_personality,
            base_response=base_response,
            player_message=player_message,
            npc_id=npc_id,
        )
        
        assert isinstance(enhanced_response, str)
        assert isinstance(metadata, dict)
        assert "dialogue_integration_enabled" in metadata
        assert metadata["dialogue_integration_enabled"] is True
        assert "confidence_score" in metadata
        assert "naturalness_score" in metadata
        assert "engagement_impact" in metadata

    @pytest.mark.asyncio
    async def test_enhanced_flow_manager_fallback_to_traditional(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test fallback to traditional memory patterns when dialogue integration fails."""
        # Create enhanced flow manager without dialogue integration
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
            enable_conversation_threading=True,
            enable_dialogue_integration=False,  # Disabled
            dialogue_integration_engine=None,
        )
        
        base_response = "Let me think about that."
        player_message = "Tell me about the village."
        npc_id = uuid.UUID(sample_conversation_context.npc_id)
        
        enhanced_response, metadata = await flow_manager.enhance_response_with_advanced_memory_integration(
            conversation=sample_conversation_context,
            personality=sample_npc_personality,
            base_response=base_response,
            player_message=player_message,
            npc_id=npc_id,
        )
        
        assert isinstance(enhanced_response, str)
        assert isinstance(metadata, dict)
        assert "dialogue_integration_enabled" not in metadata or metadata["dialogue_integration_enabled"] is False

    @pytest.mark.asyncio
    async def test_dialogue_readiness_analysis(
        self,
        enhanced_flow_manager,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test dialogue readiness analysis."""
        player_message = "I'm curious about the village's past"
        
        readiness = await enhanced_flow_manager.analyze_dialogue_readiness(
            conversation=sample_conversation_context,
            personality=sample_npc_personality,
            player_message=player_message,
        )
        
        assert isinstance(readiness, dict)
        assert "dialogue_integration_available" in readiness
        assert readiness["dialogue_integration_available"] is True
        assert "should_integrate" in readiness
        assert "confidence_score" in readiness
        assert "flow_disruption_risk" in readiness
        assert "timing_strategy" in readiness
        assert "integration_style" in readiness


class TestMemoryConflictResolution:
    """Test memory conflict resolution functionality."""

    @pytest.mark.asyncio
    async def test_memory_conflict_resolution(
        self,
        dialogue_integration_engine,
    ):
        """Test resolution of conflicting memories."""
        # Create conflicting memories
        memory1 = ConversationExchange(
            exchange_id="conflict-1",
            speaker_id="test-npc-id",
            message_text="The ruins were built by ancient elves",
            message_type=MessageType.STATEMENT,
            emotion="confident",
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        
        memory2 = ConversationExchange(
            exchange_id="conflict-2",
            speaker_id="test-npc-id",
            message_text="The ruins were built by ancient humans",
            message_type=MessageType.STATEMENT,
            emotion="uncertain",
            timestamp=datetime.now(timezone.utc).timestamp() + 100,
        )
        
        conflicting_memories = [
            {"memory": memory1, "confidence": 0.8, "trustworthiness": 0.9},
            {"memory": memory2, "confidence": 0.6, "trustworthiness": 0.7},
        ]
        
        resolved_memories = await dialogue_integration_engine.resolve_memory_conflicts(
            conflicting_memories=conflicting_memories,
            current_context="discussing ancient architecture",
            resolution_strategy="confidence_weighted",
        )
        
        assert len(resolved_memories) > 0
        # Should prefer the first memory due to higher confidence and trustworthiness
        assert resolved_memories[0]["memory"].message_text == "The ruins were built by ancient elves"


class TestNaturalLanguageMemoryWeaving:
    """Test natural language patterns for memory integration."""

    @pytest.mark.asyncio
    async def test_natural_memory_reference_generation(
        self,
        dialogue_integration_engine,
        sample_npc_personality,
    ):
        """Test generation of natural memory references."""
        memory = ConversationExchange(
            exchange_id="reference-memory",
            speaker_id="test-player-id",
            message_text="I love exploring ancient places",
            message_type=MessageType.STATEMENT,
            emotion="enthusiastic",
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        
        dialogue_context = DialogueContext(
            current_topic="ancient exploration",
            current_state=DialogueState.BUILDING,
            engagement_momentum=0.7,
            conversation_depth_level=0.6,
            memory_reference_density=0.2,
            player_curiosity_indicators=["love exploring"],
            topic_transitions=[],
            emotional_arc=["enthusiastic"],
            last_memory_integration=None,
        )
        
        memory_reference = await dialogue_integration_engine._create_natural_memory_reference(
            memory=memory,
            integration_style=IntegrationStyle.EMOTIONAL_CONNECTION,
            npc_personality=sample_npc_personality,
            dialogue_context=dialogue_context,
        )
        
        assert isinstance(memory_reference, str)
        assert len(memory_reference) > 0
        assert "exploring" in memory_reference.lower() or "ancient" in memory_reference.lower()

    @pytest.mark.asyncio
    async def test_memory_weaving_into_response(
        self,
        dialogue_integration_engine,
        sample_npc_personality,
    ):
        """Test weaving memory references into NPC responses."""
        base_response = "The ruins are quite mysterious."
        memory_text = "I remember you mentioned loving ancient places"
        
        dialogue_context = DialogueContext(
            current_topic="ancient ruins",
            current_state=DialogueState.BUILDING,
            engagement_momentum=0.7,
            conversation_depth_level=0.6,
            memory_reference_density=0.2,
            player_curiosity_indicators=[],
            topic_transitions=[],
            emotional_arc=["curious"],
            last_memory_integration=None,
        )
        
        integration_plan = MemoryIntegrationPlan(
            should_integrate=True,
            confidence_score=0.8,
            timing_strategy=MemoryIntegrationTiming.NATURAL_PAUSE,
            integration_style=IntegrationStyle.DIRECT_REFERENCE,
            flow_disruption_risk="LOW",
            memory_references=[],
            fallback_plan="subtle_hint",
        )
        
        woven_response = await dialogue_integration_engine._weave_memory_into_response(
            base_response=base_response,
            memory_text=memory_text,
            integration_plan=integration_plan,
            dialogue_context=dialogue_context,
            npc_personality=sample_npc_personality,
        )
        
        assert isinstance(woven_response, str)
        assert len(woven_response) > len(base_response)
        assert memory_text.lower() in woven_response.lower() or "ancient" in woven_response.lower()



class TestEndToEndMemoryIntegration:
    """Test complete end-to-end memory integration workflow."""

    @pytest.mark.asyncio
    async def test_complete_memory_integration_workflow(
        self,
        enhanced_flow_manager,
        sample_conversation_context,
        sample_npc_personality,
    ):
        """Test complete memory integration workflow from start to finish."""
        # Setup high-engagement conversation scenario
        sample_conversation_context.relationship_level = 0.8
        sample_conversation_context.mood = "curious"
        
        base_response = "That's a fascinating question about our village's past."
        player_message = "I'd love to hear more about the ancient traditions you mentioned before."
        npc_id = uuid.UUID(sample_conversation_context.npc_id)
        
        # Test the complete workflow
        enhanced_response, metadata = await enhanced_flow_manager.enhance_response_with_advanced_memory_integration(
            conversation=sample_conversation_context,
            personality=sample_npc_personality,
            base_response=base_response,
            player_message=player_message,
            npc_id=npc_id,
        )
        
        # Verify complete integration
        assert enhanced_response != base_response
        assert len(enhanced_response) >= len(base_response)
        assert isinstance(metadata, dict)
        
        # Verify metadata completeness
        expected_keys = [
            "dialogue_integration_enabled",
            "confidence_score",
            "naturalness_score",
            "engagement_impact",
            "dialogue_state",
            "conversation_depth",
            "engagement_momentum",
        ]
        
        for key in expected_keys:
            assert key in metadata, f"Missing metadata key: {key}"
        
        # Verify dialogue readiness analysis
        readiness = await enhanced_flow_manager.analyze_dialogue_readiness(
            conversation=sample_conversation_context,
            personality=sample_npc_personality,
            player_message=player_message,
        )
        
        assert readiness["dialogue_integration_available"] is True
        
        # Verify dialogue context summary
        context_summary = await enhanced_flow_manager.get_dialogue_context_summary(
            conversation_id=sample_conversation_context.conversation_id
        )
        
        assert isinstance(context_summary, dict)


if __name__ == "__main__":
    pytest.main([__file__])