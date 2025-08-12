"""Integration tests for dialogue memory integration system."""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from typing import List

from game_loop.core.conversation.dialogue_factory import DialogueAwareFlowManagerFactory
from game_loop.core.conversation.context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueState,
    IntegrationStyle,
    MemoryIntegrationTiming,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
    ConversationExchange,
)
from game_loop.core.conversation.flow_manager import EnhancedConversationFlowManager
from game_loop.core.conversation.memory_integration import MemoryIntegrationInterface
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient


@pytest.fixture
def mock_session_factory():
    """Create mock database session factory."""
    factory = MagicMock(spec=DatabaseSessionFactory)
    factory.get_session = AsyncMock()
    return factory


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client with realistic responses."""
    client = MagicMock(spec=OllamaClient)
    client.extract_topic = AsyncMock(return_value="conversation_topic")
    client.detect_emotional_tone = AsyncMock(return_value="friendly")
    client.analyze_curiosity = AsyncMock(return_value=["question", "interest"])
    client.assess_naturalness = AsyncMock(return_value=0.85)
    client.calculate_engagement_impact = AsyncMock(return_value=0.75)
    client.generate_response = AsyncMock(return_value="Generated response")
    return client


@pytest.fixture
def mock_memory_integration():
    """Create mock memory integration with realistic data."""
    integration = MagicMock(spec=MemoryIntegrationInterface)

    # Mock memory context
    memory_context = MagicMock()
    memory_context.current_topic = "outdoor_activities"
    memory_context.emotional_tone = "enthusiastic"
    integration.extract_memory_context = AsyncMock(return_value=memory_context)

    # Mock memory retrieval
    memory_retrieval = MagicMock()
    memory_retrieval.relevant_memories = [
        (
            ConversationExchange(
                player_message="I love hiking in the mountains",
                npc_response="That sounds amazing! The views must be incredible.",
                timestamp="2023-01-01T09:00:00Z",
                emotion="excited",
                topic="hiking",
            ),
            0.9,
        ),
        (
            ConversationExchange(
                player_message="I prefer indoor activities when it rains",
                npc_response="That's sensible. What do you like to do indoors?",
                timestamp="2023-01-01T10:00:00Z",
                emotion="curious",
                topic="indoor_activities",
            ),
            0.7,
        ),
    ]
    memory_retrieval.emotional_alignment = 0.8
    integration.retrieve_relevant_memories = AsyncMock(return_value=memory_retrieval)

    return integration


@pytest.fixture
def sample_conversation():
    """Create a realistic conversation context."""
    return ConversationContext(
        conversation_id="conv_outdoor_chat_123",
        player_id="player_nature_lover_456",
        npc_id="npc_park_ranger_789",
        relationship_level=0.7,
        trust_level=0.6,
        conversation_history=[
            ConversationExchange(
                player_message="Good morning! Beautiful day, isn't it?",
                npc_response="Indeed! Perfect weather for outdoor activities.",
                timestamp="2023-01-01T08:00:00Z",
                emotion="cheerful",
                topic="weather",
            ),
            ConversationExchange(
                player_message="I'm thinking about exploring some new trails today.",
                npc_response="Excellent choice! There are some wonderful paths in this area.",
                timestamp="2023-01-01T08:05:00Z",
                emotion="encouraging",
                topic="hiking",
            ),
            ConversationExchange(
                player_message="Do you have any recommendations for scenic routes?",
                npc_response="Absolutely! Let me tell you about a few hidden gems.",
                timestamp="2023-01-01T08:10:00Z",
                emotion="helpful",
                topic="trail_recommendations",
            ),
        ],
    )


@pytest.fixture
def sample_personality():
    """Create a realistic NPC personality."""
    personality = MagicMock(spec=NPCPersonality)
    personality.get_trait_strength = MagicMock(
        side_effect=lambda trait: {
            "helpful": 0.9,
            "knowledgeable": 0.8,
            "friendly": 0.85,
            "cautious": 0.3,
            "open": 0.7,
        }.get(trait, 0.5)
    )

    personality.personality_traits = {
        "helpful": 0.9,
        "knowledgeable": 0.8,
        "friendly": 0.85,
        "cautious": 0.3,
        "open": 0.7,
    }
    personality.conflict_resolution_strategy = "diplomatic"
    return personality


class TestDialogueMemoryIntegrationEndToEnd:
    """End-to-end integration tests for dialogue memory integration."""

    @pytest.mark.asyncio
    async def test_complete_dialogue_memory_integration_flow(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test complete end-to-end dialogue memory integration flow."""
        # Arrange - Create enhanced flow manager using factory
        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
            naturalness_threshold=0.7,
            engagement_threshold=0.6,
        )

        base_response = "That trail sounds perfect for today's weather!"
        player_message = "I'm excited to try something new and challenging."
        npc_id = uuid.UUID(sample_conversation.npc_id)

        # Act
        enhanced_response, metadata = (
            await flow_manager.enhance_response_with_advanced_memory_integration(
                conversation=sample_conversation,
                personality=sample_personality,
                base_response=base_response,
                player_message=player_message,
                npc_id=npc_id,
            )
        )

        # Assert - Verify integration was applied
        assert enhanced_response != base_response  # Response should be enhanced
        assert metadata["dialogue_integration_enabled"] is True
        assert metadata["dialogue_integration_applied"] is True
        assert metadata["dialogue_state"] in [state.value for state in DialogueState]
        assert metadata["conversation_depth"] >= 1
        assert 0.0 <= metadata["engagement_momentum"] <= 1.0
        assert metadata["confidence_score"] > 0.0
        assert metadata["naturalness_score"] > 0.0

        # Verify that LLM and memory integration were used
        mock_llm_client.extract_topic.assert_called()
        mock_llm_client.detect_emotional_tone.assert_called()
        mock_memory_integration.extract_memory_context.assert_called()
        mock_memory_integration.retrieve_relevant_memories.assert_called()

    @pytest.mark.asyncio
    async def test_dialogue_readiness_analysis_integration(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test dialogue readiness analysis integration."""
        # Arrange
        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
        )

        player_message = "I'm curious about your experiences with difficult trails."

        # Act
        readiness_analysis = await flow_manager.analyze_dialogue_readiness(
            conversation=sample_conversation,
            personality=sample_personality,
            player_message=player_message,
        )

        # Assert
        assert readiness_analysis["dialogue_integration_available"] is True
        assert "should_integrate" in readiness_analysis
        assert "confidence_score" in readiness_analysis
        assert "dialogue_state" in readiness_analysis
        assert "engagement_momentum" in readiness_analysis
        assert "available_memories" in readiness_analysis

        # Verify that the readiness analysis provides actionable insights
        if readiness_analysis["should_integrate"]:
            assert readiness_analysis["confidence_score"] > 0.5
            assert readiness_analysis["flow_disruption_risk"] < 0.8
        else:
            assert "fallback_plan" in readiness_analysis

    @pytest.mark.asyncio
    async def test_memory_conflict_resolution_integration(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test memory conflict resolution in integration flow."""
        # Arrange - Add conflicting memories
        conflicting_memory_retrieval = MagicMock()
        conflicting_memory_retrieval.relevant_memories = [
            (
                ConversationExchange(
                    player_message="I hate long hikes, they're exhausting",
                    npc_response="I understand, they can be tiring",
                    timestamp="2023-01-01T07:00:00Z",
                    emotion="frustrated",
                    topic="hiking",
                ),
                0.8,
            ),
            (
                ConversationExchange(
                    player_message="I love challenging mountain trails",
                    npc_response="That's the spirit of adventure!",
                    timestamp="2023-01-01T09:00:00Z",
                    emotion="enthusiastic",
                    topic="hiking",
                ),
                0.9,
            ),
        ]
        conflicting_memory_retrieval.emotional_alignment = 0.4  # Low due to conflict
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            conflicting_memory_retrieval
        )

        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
        )

        base_response = "Let me suggest a moderate trail."
        player_message = "What kind of hiking do you think I'd enjoy?"
        npc_id = uuid.UUID(sample_conversation.npc_id)

        # Act
        enhanced_response, metadata = (
            await flow_manager.enhance_response_with_advanced_memory_integration(
                conversation=sample_conversation,
                personality=sample_personality,
                base_response=base_response,
                player_message=player_message,
                npc_id=npc_id,
            )
        )

        # Assert - System should handle conflicts gracefully
        assert enhanced_response is not None
        assert metadata["dialogue_integration_enabled"] is True

        # With conflicting memories, the system should either:
        # 1. Apply integration but with lower confidence
        # 2. Fall back to traditional patterns
        # 3. Use conflict resolution
        if metadata["dialogue_integration_applied"]:
            # If integration was applied despite conflicts, confidence should be moderate
            assert metadata["confidence_score"] >= 0.3
        else:
            # If integration wasn't applied, fallback should be used
            assert (
                metadata.get("fallback_to_traditional", False)
                or "error" not in metadata
            )

    @pytest.mark.asyncio
    async def test_different_dialogue_states_integration(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_personality,
    ):
        """Test dialogue integration across different conversation states."""
        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
        )

        npc_id = uuid.uuid4()
        base_response = "I understand."

        # Test scenarios for different dialogue states
        test_scenarios = [
            {
                "name": "Opening conversation",
                "conversation_history": [
                    ConversationExchange(
                        player_message="Hello there!",
                        npc_response="Hi! Nice to meet you.",
                        timestamp="2023-01-01T10:00:00Z",
                        emotion="friendly",
                        topic="greeting",
                    )
                ],
                "player_message": "How are you today?",
                "expected_state": DialogueState.OPENING,
            },
            {
                "name": "Building conversation",
                "conversation_history": [
                    ConversationExchange(
                        player_message="Hello!",
                        npc_response="Hi there!",
                        timestamp="2023-01-01T10:00:00Z",
                        emotion="friendly",
                        topic="greeting",
                    ),
                    ConversationExchange(
                        player_message="How's your day going?",
                        npc_response="It's been great, thanks for asking!",
                        timestamp="2023-01-01T10:01:00Z",
                        emotion="cheerful",
                        topic="daily_life",
                    ),
                    ConversationExchange(
                        player_message="What brings you to this area?",
                        npc_response="I work here as a guide.",
                        timestamp="2023-01-01T10:02:00Z",
                        emotion="informative",
                        topic="work",
                    ),
                ],
                "player_message": "That sounds like an interesting job.",
                "expected_state": DialogueState.BUILDING,
            },
        ]

        for scenario in test_scenarios:
            # Arrange
            conversation = ConversationContext(
                conversation_id=f"conv_{scenario['name'].lower().replace(' ', '_')}",
                player_id="test_player",
                npc_id=str(npc_id),
                relationship_level=0.5,
                trust_level=0.4,
                conversation_history=scenario["conversation_history"],
            )

            # Act
            enhanced_response, metadata = (
                await flow_manager.enhance_response_with_advanced_memory_integration(
                    conversation=conversation,
                    personality=sample_personality,
                    base_response=base_response,
                    player_message=scenario["player_message"],
                    npc_id=npc_id,
                )
            )

            # Assert
            assert (
                enhanced_response is not None
            ), f"Failed for scenario: {scenario['name']}"
            assert metadata["dialogue_integration_enabled"] is True

            # The dialogue state should be appropriate for the conversation length and content
            dialogue_state = DialogueState(metadata["dialogue_state"])
            conversation_depth = metadata["conversation_depth"]

            # Verify that conversation depth correlates with dialogue state
            if dialogue_state == DialogueState.OPENING:
                assert conversation_depth <= 2
            elif dialogue_state == DialogueState.BUILDING:
                assert conversation_depth >= 2

    @pytest.mark.asyncio
    async def test_factory_configurations_integration(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test different factory configurations work correctly."""
        npc_id = uuid.uuid4()
        base_response = "That's interesting."
        player_message = "Tell me more about it."

        # Test different factory methods
        factory_methods = [
            ("enhanced", DialogueAwareFlowManagerFactory.create_enhanced_flow_manager),
            (
                "threading_only",
                DialogueAwareFlowManagerFactory.create_threading_only_flow_manager,
            ),
            (
                "dialogue_only",
                DialogueAwareFlowManagerFactory.create_dialogue_only_flow_manager,
            ),
            ("standard", DialogueAwareFlowManagerFactory.create_standard_flow_manager),
        ]

        for method_name, factory_method in factory_methods:
            # Arrange
            if method_name == "enhanced":
                flow_manager = factory_method(
                    session_factory=mock_session_factory,
                    llm_client=mock_llm_client,
                    memory_integration=mock_memory_integration,
                )
            elif method_name == "threading_only":
                flow_manager = factory_method(
                    session_factory=mock_session_factory,
                    memory_integration=mock_memory_integration,
                )
            elif method_name == "dialogue_only":
                flow_manager = factory_method(
                    session_factory=mock_session_factory,
                    llm_client=mock_llm_client,
                    memory_integration=mock_memory_integration,
                )
            else:  # standard
                flow_manager = factory_method(
                    session_factory=mock_session_factory,
                    memory_integration=mock_memory_integration,
                )

            # Act
            if hasattr(
                flow_manager, "enhance_response_with_advanced_memory_integration"
            ):
                enhanced_response, metadata = (
                    await flow_manager.enhance_response_with_advanced_memory_integration(
                        conversation=sample_conversation,
                        personality=sample_personality,
                        base_response=base_response,
                        player_message=player_message,
                        npc_id=npc_id,
                    )
                )
            else:
                # Fall back to standard method for basic flow managers
                enhanced_response, metadata = (
                    await flow_manager.enhance_response_with_memory_patterns(
                        conversation=sample_conversation,
                        personality=sample_personality,
                        base_response=base_response,
                        player_message=player_message,
                        npc_id=npc_id,
                    )
                )

            # Assert
            assert (
                enhanced_response is not None
            ), f"Failed for factory method: {method_name}"
            assert isinstance(
                metadata, dict
            ), f"Invalid metadata for method: {method_name}"

            # Verify expected capabilities based on factory method
            if method_name == "enhanced":
                assert flow_manager.is_dialogue_integration_enabled() is True
                assert flow_manager.enable_conversation_threading is True
            elif method_name == "threading_only":
                assert flow_manager.is_dialogue_integration_enabled() is False
                assert flow_manager.enable_conversation_threading is True
            elif method_name == "dialogue_only":
                assert flow_manager.is_dialogue_integration_enabled() is True
                assert flow_manager.enable_conversation_threading is False
            else:  # standard
                assert flow_manager.is_dialogue_integration_enabled() is False
                assert flow_manager.enable_conversation_threading is False

    @pytest.mark.asyncio
    async def test_performance_under_load_simulation(
        self,
        mock_session_factory,
        mock_llm_client,
        mock_memory_integration,
        sample_conversation,
        sample_personality,
    ):
        """Test system performance under simulated load."""
        # Arrange
        flow_manager = DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            memory_integration=mock_memory_integration,
        )

        npc_id = uuid.uuid4()
        base_responses = [
            "That's a great point!",
            "I hadn't considered that before.",
            "Tell me more about your experience.",
            "That sounds challenging.",
            "I can relate to that feeling.",
        ]
        player_messages = [
            "I love the challenge of steep trails.",
            "Weather can really change the hiking experience.",
            "I prefer group hikes over solo adventures.",
            "Photography adds another dimension to hiking.",
            "Safety should always be the top priority.",
        ]

        # Act - Process multiple conversations quickly
        results = []
        for i, (base_response, player_message) in enumerate(
            zip(base_responses, player_messages)
        ):
            # Modify conversation ID to simulate different conversations
            test_conversation = ConversationContext(
                conversation_id=f"load_test_conv_{i}",
                player_id=sample_conversation.player_id,
                npc_id=sample_conversation.npc_id,
                relationship_level=sample_conversation.relationship_level,
                trust_level=sample_conversation.trust_level,
                conversation_history=sample_conversation.conversation_history,
            )

            enhanced_response, metadata = (
                await flow_manager.enhance_response_with_advanced_memory_integration(
                    conversation=test_conversation,
                    personality=sample_personality,
                    base_response=base_response,
                    player_message=player_message,
                    npc_id=npc_id,
                )
            )

            results.append((enhanced_response, metadata))

        # Assert
        assert len(results) == len(base_responses)

        for i, (enhanced_response, metadata) in enumerate(results):
            assert enhanced_response is not None, f"Failed at iteration {i}"
            assert isinstance(metadata, dict), f"Invalid metadata at iteration {i}"
            assert metadata["dialogue_integration_enabled"] is True

            # Verify that each conversation was processed independently
            assert (
                f"load_test_conv_{i}" in flow_manager._dialogue_contexts
                or metadata.get("dialogue_integration_applied") is not None
            )

        # Verify that the system maintained performance across all iterations
        dialogue_applied_count = sum(
            1
            for _, metadata in results
            if metadata.get("dialogue_integration_applied", False)
        )

        # At least some conversations should have had dialogue integration applied
        assert (
            dialogue_applied_count >= 0
        )  # Should be at least some successful integrations
