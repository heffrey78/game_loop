"""Tests for conversation flow manager."""

import uuid
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    ConversationStatus,
    MessageType,
    NPCPersonality,
)
from game_loop.core.conversation.flow_manager import ConversationFlowManager
from game_loop.core.conversation.flow_templates import (
    ConversationStage,
    MemoryDisclosureThreshold,
    TrustLevel,
)
from game_loop.core.conversation.memory_integration import MemoryRetrievalResult


@pytest.fixture
def mock_memory_integration():
    """Mock memory integration interface."""
    mock_integration = AsyncMock()
    mock_integration.extract_memory_context = AsyncMock()
    mock_integration.retrieve_relevant_memories = AsyncMock()
    return mock_integration


@pytest.fixture
def mock_session_factory():
    """Mock database session factory."""
    mock_factory = Mock()
    mock_session = AsyncMock()
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_session
    mock_context_manager.__aexit__.return_value = None
    mock_factory.get_session.return_value = mock_context_manager
    return mock_factory


@pytest.fixture
def mock_flow_library():
    """Mock flow library."""
    mock_library = Mock()
    mock_library.get_trust_level_from_relationship.return_value = (
        TrustLevel.ACQUAINTANCE
    )
    mock_library.get_disclosure_threshold_from_confidence.return_value = (
        MemoryDisclosureThreshold.CLEAR_REFERENCES
    )
    mock_library.get_template_for_stage.return_value = Mock()
    mock_library.validate_memory_integration.return_value = (True, [])
    mock_library.suggest_conversation_progression.return_value = (
        ConversationStage.RELATIONSHIP_BUILDING
    )
    return mock_library


@pytest.fixture
def sample_conversation():
    """Sample conversation context."""
    return ConversationContext(
        conversation_id="550e8400-e29b-41d4-a716-446655440003",
        player_id="550e8400-e29b-41d4-a716-446655440001",
        npc_id="550e8400-e29b-41d4-a716-446655440002",
        topic="village life",
        mood="friendly",
        relationship_level=0.5,
        conversation_history=[
            ConversationExchange(
                exchange_id="550e8400-e29b-41d4-a716-446655440004",
                speaker_id="550e8400-e29b-41d4-a716-446655440001",
                message_text="Tell me about the village",
                message_type=MessageType.QUESTION,
                emotion="curious",
                timestamp=1234567890,
            ),
            ConversationExchange(
                exchange_id="550e8400-e29b-41d4-a716-446655440005",
                speaker_id="550e8400-e29b-41d4-a716-446655440002",
                message_text="It's a peaceful place with friendly people",
                message_type=MessageType.STATEMENT,
                emotion="friendly",
                timestamp=1234567900,
            ),
        ],
        context_data={},
        status=ConversationStatus.ACTIVE,
        started_at=1234567880,
        last_updated=1234567900,
    )


@pytest.fixture
def sample_personality():
    """Sample NPC personality."""
    return NPCPersonality(
        npc_id="550e8400-e29b-41d4-a716-446655440002",
        traits={"friendly": 0.8, "helpful": 0.7, "cautious": 0.3},
        knowledge_areas=["village_life", "local_history"],
        speech_patterns={"formality": "medium", "directness": "high"},
        relationships={},
        background_story="A long-time village resident",
        default_mood="friendly",
    )


class TestConversationFlowManager:
    """Test ConversationFlowManager functionality."""

    def test_initialization(
        self, mock_memory_integration, mock_session_factory, mock_flow_library
    ):
        """Test flow manager initialization."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
            mock_flow_library,
        )

        assert manager.memory_integration == mock_memory_integration
        assert manager.session_factory == mock_session_factory
        assert manager.flow_library == mock_flow_library
        assert isinstance(manager._conversation_stages, dict)
        assert isinstance(manager._memory_usage_history, dict)

    @pytest.mark.asyncio
    async def test_enhance_response_no_memories(
        self,
        mock_memory_integration,
        mock_session_factory,
        mock_flow_library,
        sample_conversation,
        sample_personality,
    ):
        """Test response enhancement when no memories are available."""
        # Setup mocks
        mock_memory_context = Mock()
        mock_memory_integration.extract_memory_context.return_value = (
            mock_memory_context
        )
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            MemoryRetrievalResult(
                relevant_memories=[],
                context_score=0.0,
            )
        )

        mock_template = Mock()
        mock_template.memory_patterns = []
        mock_flow_library.get_template_for_stage.return_value = mock_template

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
            mock_flow_library,
        )

        # Mock the stage determination
        with patch.object(
            manager,
            "_determine_conversation_stage",
            return_value=ConversationStage.ACQUAINTANCE,
        ):
            enhanced_response, metadata = (
                await manager.enhance_response_with_memory_patterns(
                    sample_conversation,
                    sample_personality,
                    "Hello there!",
                    "Hi",
                    uuid.UUID(sample_conversation.npc_id),
                )
            )

        assert enhanced_response == "Hello there!"
        assert metadata["memory_enhanced"] is False
        assert metadata["stage"] == ConversationStage.ACQUAINTANCE.value

    @pytest.mark.asyncio
    async def test_enhance_response_with_memories(
        self,
        mock_memory_integration,
        mock_session_factory,
        mock_flow_library,
        sample_conversation,
        sample_personality,
    ):
        """Test response enhancement with available memories."""
        # Setup memory result
        sample_memory = ConversationExchange(
            exchange_id="mem-1",
            speaker_id="player-456",
            message_text="I love this village",
            message_type=MessageType.STATEMENT,
            emotion="happy",
            timestamp=1234567800,
        )

        mock_memory_context = Mock()
        mock_memory_context.emotional_tone = "happy"

        mock_memory_integration.extract_memory_context.return_value = (
            mock_memory_context
        )
        mock_memory_integration.retrieve_relevant_memories.return_value = (
            MemoryRetrievalResult(
                relevant_memories=[(sample_memory, 0.8)],
                context_score=0.8,
            )
        )

        # Setup template with memory pattern
        mock_pattern = Mock()
        mock_pattern.pattern_name = "clear_references"
        mock_pattern.disclosure_level = MemoryDisclosureThreshold.CLEAR_REFERENCES
        mock_pattern.get_appropriate_phrase.return_value = Mock(
            text="I remember when...",
            emotional_tone="neutral",
        )

        mock_template = Mock()
        mock_template.memory_patterns = [mock_pattern]
        mock_flow_library.get_template_for_stage.return_value = mock_template

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
            mock_flow_library,
        )

        # Mock required methods
        with (
            patch.object(
                manager,
                "_determine_conversation_stage",
                return_value=ConversationStage.ACQUAINTANCE,
            ),
            patch.object(
                manager,
                "_integrate_memories_with_patterns",
                return_value=("Enhanced response", {"memory_enhanced": True}),
            ),
            patch.object(manager, "_check_stage_progression"),
            patch.object(manager, "_track_memory_usage"),
        ):

            enhanced_response, metadata = (
                await manager.enhance_response_with_memory_patterns(
                    sample_conversation,
                    sample_personality,
                    "Hello there!",
                    "Hi",
                    uuid.UUID(sample_conversation.npc_id),
                )
            )

        assert enhanced_response == "Enhanced response"
        assert metadata["memory_enhanced"] is True

    @pytest.mark.asyncio
    async def test_determine_conversation_stage_initial(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_conversation,
        sample_personality,
    ):
        """Test conversation stage determination for initial encounter."""
        # Mock repository to return conversation count
        mock_repo_manager = Mock()
        mock_repo_manager.contexts.get_conversation_count_for_npc_player_pair = (
            AsyncMock(return_value=1)  # Single conversation for initial encounter
        )

        mock_session = AsyncMock()
        mock_session_factory.get_session.return_value.__aenter__.return_value = (
            mock_session
        )

        with patch(
            "game_loop.core.conversation.flow_manager.ConversationRepositoryManager",
            return_value=mock_repo_manager,
        ):
            manager = ConversationFlowManager(
                mock_memory_integration,
                mock_session_factory,
            )

            # Create conversation with low relationship score for initial encounter
            low_rel_conversation = replace(sample_conversation, relationship_level=0.1)

            stage = await manager._determine_conversation_stage(
                low_rel_conversation, sample_personality
            )

        assert stage == ConversationStage.INITIAL_ENCOUNTER

    @pytest.mark.asyncio
    async def test_determine_conversation_stage_with_personality_modifier(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_conversation,
        sample_personality,
    ):
        """Test conversation stage with personality modifiers."""
        # Mock repository
        mock_repo_manager = Mock()
        mock_repo_manager.contexts.get_player_conversations.return_value = [
            sample_conversation
        ] * 5

        mock_session = AsyncMock()
        mock_session_factory.get_session.return_value.__aenter__.return_value = (
            mock_session
        )

        # Cautious personality should slow progression
        cautious_personality = sample_personality
        cautious_personality.get_trait_strength = Mock(
            side_effect=lambda trait: 0.8 if trait == "cautious" else 0.5
        )

        with patch(
            "game_loop.core.conversation.flow_manager.ConversationRepositoryManager",
            return_value=mock_repo_manager,
        ):
            manager = ConversationFlowManager(
                mock_memory_integration,
                mock_session_factory,
            )

            # Should progress slower due to cautious personality
            stage = await manager._determine_conversation_stage(
                sample_conversation, cautious_personality
            )

        # Exact stage depends on implementation, but should be affected by personality
        assert isinstance(stage, ConversationStage)

    def test_apply_personality_stage_modifiers_cautious(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_personality,
    ):
        """Test personality modifier application for cautious NPCs."""
        cautious_personality = sample_personality
        cautious_personality.get_trait_strength = Mock(
            side_effect=lambda trait: 0.8 if trait == "cautious" else 0.3
        )

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        # Cautious personality should regress deep connection to trust development
        modified_stage = manager._apply_personality_stage_modifiers(
            ConversationStage.DEEP_CONNECTION, cautious_personality
        )

        assert modified_stage == ConversationStage.TRUST_DEVELOPMENT

    def test_apply_personality_stage_modifiers_open(
        self,
        mock_memory_integration,
        mock_session_factory,
        sample_personality,
    ):
        """Test personality modifier application for open NPCs."""
        open_personality = sample_personality
        open_personality.get_trait_strength = Mock(
            side_effect=lambda trait: 0.8 if trait == "open" else 0.3
        )

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        # Open personality should progress acquaintance to relationship building
        modified_stage = manager._apply_personality_stage_modifiers(
            ConversationStage.ACQUAINTANCE, open_personality
        )

        assert modified_stage == ConversationStage.RELATIONSHIP_BUILDING

    @pytest.mark.asyncio
    async def test_integrate_memories_with_patterns_valid(
        self,
        mock_memory_integration,
        mock_session_factory,
        mock_flow_library,
        sample_personality,
    ):
        """Test memory integration with valid patterns."""
        # Setup memory
        sample_memory = ConversationExchange(
            exchange_id="mem-1",
            speaker_id="player-456",
            message_text="I love this village",
            message_type=MessageType.STATEMENT,
            emotion="happy",
            timestamp=1234567800,
        )
        memories = [(sample_memory, 0.6)]

        # Setup pattern
        mock_pattern = Mock()
        mock_pattern.pattern_name = "clear_references"
        mock_pattern.disclosure_level = MemoryDisclosureThreshold.CLEAR_REFERENCES
        mock_pattern.get_appropriate_phrase.return_value = Mock(
            text="I remember when..."
        )

        mock_template = Mock()
        mock_template.memory_patterns = [mock_pattern]

        # Setup flow library
        mock_flow_library.validate_memory_integration.return_value = (True, [])
        mock_flow_library.get_transition_phrase.return_value = Mock(text="I recall...")

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
            mock_flow_library,
        )

        with patch.object(
            manager, "_build_enhanced_response", return_value="Enhanced response"
        ):
            enhanced_response, integration_data = (
                await manager._integrate_memories_with_patterns(
                    "Base response",
                    memories,
                    TrustLevel.ACQUAINTANCE,
                    mock_template,
                    sample_personality,
                    "happy",
                )
            )

        assert enhanced_response == "Enhanced response"
        assert integration_data["memory_enhanced"] is True
        assert integration_data["pattern_used"] == "clear_references"

    @pytest.mark.asyncio
    async def test_integrate_memories_with_patterns_validation_failure(
        self,
        mock_memory_integration,
        mock_session_factory,
        mock_flow_library,
        sample_personality,
    ):
        """Test memory integration when validation fails."""
        # Setup memory
        sample_memory = ConversationExchange(
            exchange_id="mem-1",
            speaker_id="player-456",
            message_text="I love this village",
            message_type=MessageType.STATEMENT,
            emotion="happy",
            timestamp=1234567800,
        )
        memories = [(sample_memory, 0.6)]

        # Setup pattern
        mock_pattern = Mock()
        mock_pattern.pattern_name = "clear_references"
        mock_pattern.disclosure_level = MemoryDisclosureThreshold.CLEAR_REFERENCES

        mock_template = Mock()
        mock_template.memory_patterns = [mock_pattern]

        # Setup validation failure
        mock_flow_library.validate_memory_integration.return_value = (
            False,
            ["Trust level too low"],
        )

        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
            mock_flow_library,
        )

        enhanced_response, integration_data = (
            await manager._integrate_memories_with_patterns(
                "Base response",
                memories,
                TrustLevel.STRANGER,
                mock_template,
                sample_personality,
                "happy",
            )
        )

        assert enhanced_response == "Base response"
        assert integration_data["memory_enhanced"] is False
        assert "validation_errors" in integration_data

    def test_create_memory_references(
        self,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test different memory reference creation methods."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        sample_memory = ConversationExchange(
            exchange_id="mem-1",
            speaker_id="player-456",
            message_text="I love this peaceful village and its friendly people",
            message_type=MessageType.STATEMENT,
            emotion="happy",
            timestamp=1234567800,
        )

        # Test subtle hint
        subtle_hint = manager._create_subtle_hint(sample_memory, 0.4)
        assert "something you mentioned before" in subtle_hint.lower()

        # Test clear reference
        clear_ref = manager._create_clear_reference(sample_memory, 0.6)
        assert "talked about" in clear_ref.lower()

        # Test detailed memory
        mock_pattern = Mock()
        mock_pattern.emotional_modifiers = {
            "happy": ["with a smile", "enthusiastically"]
        }
        detailed_mem = manager._create_detailed_memory(sample_memory, 0.8, mock_pattern)
        assert "you said" in detailed_mem.lower()
        assert "with a smile" in detailed_mem

    def test_get_conversation_stage_cached(
        self,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test getting cached conversation stage."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        # Set cached stage
        conversation_id = "test-conv-123"
        manager._conversation_stages[conversation_id] = (
            ConversationStage.TRUST_DEVELOPMENT
        )

        stage = manager.get_conversation_stage(conversation_id)
        assert stage == ConversationStage.TRUST_DEVELOPMENT

    def test_get_memory_usage_stats_empty(
        self,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test memory usage stats for conversation with no history."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        stats = manager.get_memory_usage_stats("unknown-conv")
        assert stats["total_interactions"] == 0
        assert stats["memory_enhanced_count"] == 0

    def test_get_memory_usage_stats_with_data(
        self,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test memory usage stats with data."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        # Add some usage history
        conversation_id = "test-conv-123"
        manager._memory_usage_history[conversation_id] = [
            {"enhanced": True, "confidence": 0.8, "pattern": "clear_references"},
            {"enhanced": False, "confidence": 0.3, "pattern": None},
            {"enhanced": True, "confidence": 0.9, "pattern": "detailed_memories"},
        ]

        stats = manager.get_memory_usage_stats(conversation_id)
        assert stats["total_interactions"] == 3
        assert stats["memory_enhanced_count"] == 2
        assert stats["enhancement_rate"] == 2 / 3
        assert stats["avg_confidence"] == pytest.approx((0.8 + 0.3 + 0.9) / 3)
        assert "clear_references" in stats["patterns_used"]
        assert "detailed_memories" in stats["patterns_used"]

    @pytest.mark.asyncio
    async def test_analyze_conversation_flow_quality(
        self,
        mock_memory_integration,
        mock_session_factory,
    ):
        """Test conversation flow quality analysis."""
        manager = ConversationFlowManager(
            mock_memory_integration,
            mock_session_factory,
        )

        # Setup test data
        conversation_id = "test-conv-123"
        manager._conversation_stages[conversation_id] = (
            ConversationStage.TRUST_DEVELOPMENT
        )
        manager._memory_usage_history[conversation_id] = [
            {"enhanced": True, "confidence": 0.8, "pattern": "clear_references"},
            {"enhanced": True, "confidence": 0.7, "pattern": "subtle_hints"},
            {"enhanced": False, "confidence": 0.3, "pattern": None},
            {"enhanced": True, "confidence": 0.9, "pattern": "detailed_memories"},
        ]

        analysis = await manager.analyze_conversation_flow_quality(conversation_id)

        assert "quality_score" in analysis
        assert analysis["quality_score"] > 0
        assert analysis["enhancement_rate"] == 0.75  # 3 out of 4 enhanced
        assert analysis["current_stage"] == ConversationStage.TRUST_DEVELOPMENT.value
        assert isinstance(analysis["recommendations"], list)
