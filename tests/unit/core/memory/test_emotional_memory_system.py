"""Comprehensive tests for the emotional memory system."""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from game_loop.core.memory.affective_weighting import (
    AffectiveMemoryWeightingEngine,
    AffectiveWeightingStrategy,
)
from game_loop.core.memory.config import MemoryAlgorithmConfig
from game_loop.core.memory.dialogue_integration import (
    EmotionalDialogueContext,
    EmotionalDialogueIntegrationEngine,
    EmotionalResponseMode,
)
from game_loop.core.memory.emotional_clustering import (
    EmotionalMemoryClusteringEngine,
    ClusteringMethod,
)
from game_loop.core.memory.emotional_context import (
    EmotionalMemoryContextEngine,
    EmotionalMemoryType,
    EmotionalSignificance,
    MoodState,
    MemoryProtectionLevel,
)
from game_loop.core.memory.emotional_preservation import (
    EmotionalMemoryRecord,
    EmotionalPreservationEngine,
    EmotionalRetrievalQuery,
)
from game_loop.core.memory.mood_memory_engine import (
    MoodDependentMemoryEngine,
    MoodStateRecord,
    MoodTransition,
)
from game_loop.core.memory.trauma_protection import (
    TraumaAccessRequest,
    TraumaProtectionEngine,
    TraumaResponseType,
)


@pytest.fixture
def mock_session_factory():
    """Mock database session factory."""
    factory = MagicMock()
    factory.get_session = AsyncMock()
    return factory


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.generate_response = AsyncMock()
    return client


@pytest.fixture
def memory_config():
    """Memory algorithm configuration."""
    return MemoryAlgorithmConfig(
        emotional_weight_threshold=0.3,
        decay_rate=0.05,
        max_memory_age_days=365,
        clustering_enabled=True,
        personality_influence_strength=0.7,
    )


@pytest.fixture
def test_personality():
    """Test NPC personality."""
    return NPCPersonality(
        npc_id=uuid.uuid4(),
        traits={
            "empathetic": 0.8,
            "supportive": 0.7,
            "emotional_sensitivity": 0.6,
            "analytical": 0.4,
            "trauma_sensitive": 0.3,
            "resilient": 0.6,
        },
        knowledge_areas=["psychology", "emotional_support"],
        default_mood="content",
    )


@pytest.fixture
def test_conversation_context():
    """Test conversation context."""
    return ConversationContext(
        conversation_id=uuid.uuid4(),
        player_id=uuid.uuid4(),
        npc_id=uuid.uuid4(),
        relationship_level=0.6,
        mood="content",
        conversation_history=[],
    )


@pytest.fixture
def test_exchange():
    """Test conversation exchange."""
    return ConversationExchange(
        exchange_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        speaker_id=uuid.uuid4(),
        message_text="I'm feeling really anxious about tomorrow's presentation.",
        message_type="statement",
        emotion="anxious",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def emotional_significance():
    """Test emotional significance."""
    return EmotionalSignificance(
        overall_significance=0.8,
        emotional_type=EmotionalMemoryType.TRAUMATIC,
        intensity_score=0.9,
        personal_relevance=0.7,
        relationship_impact=0.6,
        formative_influence=0.5,
        protection_level=MemoryProtectionLevel.PROTECTED,
        mood_accessibility={
            MoodState.ANXIOUS: 0.9,
            MoodState.FEARFUL: 0.8,
            MoodState.NEUTRAL: 0.3,
        },
        decay_resistance=0.9,
        triggering_potential=0.8,
        confidence_score=0.85,
        contributing_factors=["anxiety", "performance", "fear"],
    )


class TestEmotionalMemoryContextEngine:
    """Tests for emotional memory context engine."""

    @pytest.fixture
    def engine(self, mock_session_factory, mock_llm_client, memory_config):
        """Create emotional memory context engine."""
        return EmotionalMemoryContextEngine(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            config=memory_config,
        )

    @pytest.mark.asyncio
    async def test_analyze_emotional_significance(
        self, engine, test_exchange, test_conversation_context, test_personality
    ):
        """Test emotional significance analysis."""
        # Mock the basic emotional analysis
        with patch.object(
            engine.emotional_analyzer, "analyze_emotional_weight"
        ) as mock_analyzer:
            mock_analyzer.return_value = MagicMock(
                emotional_weight=0.8,
                emotional_intensity=0.7,
                sentiment_score=0.4,
                relationship_impact=0.6,
                analysis_confidence=0.9,
                emotional_keywords=["anxious", "worried"],
            )

            significance = await engine.analyze_emotional_significance(
                test_exchange, test_conversation_context, test_personality
            )

            assert isinstance(significance, EmotionalSignificance)
            assert significance.overall_significance > 0.0
            assert significance.emotional_type in EmotionalMemoryType
            assert 0.0 <= significance.intensity_score <= 1.0
            assert significance.confidence_score > 0.0

    def test_performance_stats(self, engine):
        """Test performance statistics tracking."""
        stats = engine.get_performance_stats()

        assert "total_significance_analyses" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate_percent" in stats
        assert "avg_processing_time_ms" in stats

    def test_cache_clearing(self, engine):
        """Test cache clearing functionality."""
        # Add some data to cache
        engine._significance_cache["test"] = MagicMock()
        engine._mood_access_cache["test"] = MagicMock()

        engine.clear_caches()

        assert len(engine._significance_cache) == 0
        assert len(engine._mood_access_cache) == 0


class TestAffectiveMemoryWeightingEngine:
    """Tests for affective memory weighting engine."""

    @pytest.fixture
    def engine(self, mock_session_factory, memory_config):
        """Create affective memory weighting engine."""
        emotional_context_engine = MagicMock()
        return AffectiveMemoryWeightingEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            emotional_context_engine=emotional_context_engine,
        )

    @pytest.mark.asyncio
    async def test_calculate_affective_weight(
        self, engine, emotional_significance, test_personality
    ):
        """Test affective weight calculation."""
        affective_weight = await engine.calculate_affective_weight(
            emotional_significance=emotional_significance,
            npc_personality=test_personality,
            current_mood=MoodState.ANXIOUS,
            relationship_level=0.6,
            memory_age_hours=2.0,
            trust_level=0.7,
        )

        assert 0.0 <= affective_weight.base_affective_weight <= 1.0
        assert affective_weight.intensity_multiplier >= 1.0
        assert 0.0 <= affective_weight.final_weight <= 1.0
        assert affective_weight.weighting_strategy in AffectiveWeightingStrategy
        assert 0.0 <= affective_weight.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_mood_based_accessibility(
        self, engine, emotional_significance, test_personality
    ):
        """Test mood-based accessibility calculation."""
        accessibility = await engine.calculate_mood_based_accessibility(
            emotional_significance=emotional_significance,
            current_mood=MoodState.ANXIOUS,
            npc_personality=test_personality,
        )

        assert accessibility.current_mood == MoodState.ANXIOUS
        assert 0.0 <= accessibility.base_accessibility <= 1.0
        assert 0.0 <= accessibility.adjusted_accessibility <= 1.0
        assert hasattr(accessibility, "access_threshold")


class TestEmotionalPreservationEngine:
    """Tests for emotional preservation engine."""

    @pytest.fixture
    def engine(self, mock_session_factory):
        """Create emotional preservation engine."""
        return EmotionalPreservationEngine(
            session_factory=mock_session_factory,
            cache_size=100,
        )

    @pytest.mark.asyncio
    async def test_preserve_emotional_context(
        self, engine, test_exchange, emotional_significance
    ):
        """Test emotional context preservation."""
        # Mock the affective weight
        affective_weight = MagicMock()
        affective_weight.base_affective_weight = 0.7
        affective_weight.final_weight = 0.8

        # Mock database operations
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        engine.session_factory.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        engine.session_factory.get_session.return_value.__aexit__ = AsyncMock()

        emotional_record = await engine.preserve_emotional_context(
            exchange=test_exchange,
            emotional_significance=emotional_significance,
            affective_weight=affective_weight,
        )

        assert isinstance(emotional_record, EmotionalMemoryRecord)
        assert emotional_record.exchange_id == str(test_exchange.exchange_id)
        assert emotional_record.emotional_significance == emotional_significance
        assert emotional_record.preservation_confidence > 0.0

    @pytest.mark.asyncio
    async def test_retrieve_emotional_memories(self, engine, test_personality):
        """Test emotional memory retrieval."""
        npc_id = uuid.uuid4()
        query = EmotionalRetrievalQuery(
            target_mood=MoodState.ANXIOUS,
            significance_threshold=0.5,
            max_results=5,
        )

        # Mock database operations
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        # Mock empty result for simplicity
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        engine.session_factory.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        engine.session_factory.get_session.return_value.__aexit__ = AsyncMock()

        retrieval_result = await engine.retrieve_emotional_memories(npc_id, query)

        assert retrieval_result.query_processed == query
        assert isinstance(retrieval_result.emotional_records, list)
        assert retrieval_result.retrieval_time_ms >= 0.0


class TestMoodDependentMemoryEngine:
    """Tests for mood-dependent memory engine."""

    @pytest.fixture
    def engine(self, mock_session_factory, memory_config):
        """Create mood-dependent memory engine."""
        affective_engine = MagicMock()
        preservation_engine = MagicMock()
        return MoodDependentMemoryEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            affective_engine=affective_engine,
            preservation_engine=preservation_engine,
        )

    @pytest.mark.asyncio
    async def test_update_npc_mood(self, engine, test_conversation_context):
        """Test NPC mood updating."""
        npc_id = uuid.uuid4()

        mood_record = await engine.update_npc_mood(
            npc_id=npc_id,
            new_mood=MoodState.ANXIOUS,
            intensity=0.8,
            trigger_source="conversation",
            transition_type=MoodTransition.EXTERNAL_INFLUENCE,
            conversation_context=test_conversation_context,
        )

        assert isinstance(mood_record, MoodStateRecord)
        assert mood_record.npc_id == str(npc_id)
        assert mood_record.mood_state == MoodState.ANXIOUS
        assert mood_record.intensity == 0.8
        assert mood_record.stability > 0.0

    def test_get_current_mood(self, engine):
        """Test getting current mood."""
        npc_id = uuid.uuid4()

        # Initially no mood
        mood = engine.get_current_mood(npc_id)
        assert mood is None

        # After setting mood
        engine._current_moods[str(npc_id)] = MoodStateRecord(
            npc_id=str(npc_id),
            mood_state=MoodState.CONTENT,
            intensity=0.6,
            stability=0.8,
            trigger_source="test",
            transition_type=MoodTransition.NATURAL_PROGRESSION,
        )

        mood = engine.get_current_mood(npc_id)
        assert mood is not None
        assert mood.mood_state == MoodState.CONTENT

    def test_mood_history(self, engine):
        """Test mood history tracking."""
        npc_id = uuid.uuid4()

        # Initially no history
        history = engine.get_mood_history(npc_id)
        assert len(history) == 0

        # Add mood history
        mood_record = MoodStateRecord(
            npc_id=str(npc_id),
            mood_state=MoodState.JOYFUL,
            intensity=0.7,
            stability=0.8,
            trigger_source="test",
            transition_type=MoodTransition.NATURAL_PROGRESSION,
            timestamp=time.time(),
        )

        engine._mood_history[str(npc_id)] = [mood_record]

        history = engine.get_mood_history(npc_id, hours_back=24.0)
        assert len(history) == 1
        assert history[0].mood_state == MoodState.JOYFUL


class TestTraumaProtectionEngine:
    """Tests for trauma protection engine."""

    @pytest.fixture
    def engine(self, mock_session_factory, memory_config):
        """Create trauma protection engine."""
        mood_engine = MagicMock()
        return TraumaProtectionEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            mood_engine=mood_engine,
            enable_strict_protection=True,
        )

    @pytest.mark.asyncio
    async def test_evaluate_trauma_access_request(self, engine, test_personality):
        """Test trauma access request evaluation."""
        npc_id = uuid.uuid4()

        access_request = TraumaAccessRequest(
            npc_id=str(npc_id),
            requesting_context="conversation",
            trauma_memory_ids=["memory_1", "memory_2"],
            trust_level=0.5,  # Moderate trust
            therapeutic_intent=False,
        )

        decision = await engine.evaluate_trauma_access_request(
            npc_id=npc_id,
            access_request=access_request,
            personality=test_personality,
            current_mood=MoodState.NEUTRAL,
        )

        assert hasattr(decision, "access_granted")
        assert hasattr(decision, "decision_rationale")
        assert hasattr(decision, "risk_assessment")
        assert isinstance(decision.risk_assessment, dict)

    @pytest.mark.asyncio
    async def test_build_trauma_memory_profile(self, engine, test_personality):
        """Test building trauma memory profile."""
        npc_id = uuid.uuid4()

        # Create mock traumatic memory
        traumatic_memory = MagicMock()
        traumatic_memory.exchange_id = "trauma_1"
        traumatic_memory.emotional_significance.emotional_type = (
            EmotionalMemoryType.TRAUMATIC
        )
        traumatic_memory.emotional_significance.overall_significance = 0.9
        traumatic_memory.preserved_at = time.time() - 3600  # 1 hour ago

        memories = [traumatic_memory]

        profile = await engine.build_trauma_memory_profile(
            npc_id=npc_id,
            memories=memories,
            personality=test_personality,
        )

        assert profile.npc_id == str(npc_id)
        assert len(profile.trauma_memories) == 1
        assert profile.trauma_severity > 0.0
        assert profile.primary_response_type in TraumaResponseType

    @pytest.mark.asyncio
    async def test_detect_trauma_triggers(
        self, engine, test_conversation_context, test_personality
    ):
        """Test trauma trigger detection."""
        npc_id = uuid.uuid4()

        # Content with potential triggers
        trigger_content = "I was attacked and felt completely helpless."

        trigger_result = await engine.detect_trauma_triggers(
            npc_id=npc_id,
            conversation_content=trigger_content,
            conversation_context=test_conversation_context,
            personality=test_personality,
        )

        assert "triggers_detected" in trigger_result
        # Since we don't have trauma memories set up, should be False
        assert trigger_result["triggers_detected"] is False


class TestEmotionalMemoryClusteringEngine:
    """Tests for emotional memory clustering engine."""

    @pytest.fixture
    def engine(self, mock_session_factory):
        """Create emotional memory clustering engine."""
        return EmotionalMemoryClusteringEngine(
            session_factory=mock_session_factory,
            clustering_method=ClusteringMethod.KMEANS,
            min_cluster_size=2,
            max_clusters=5,
        )

    @pytest.mark.asyncio
    async def test_cluster_emotional_memories(self, engine, test_personality):
        """Test emotional memory clustering."""
        npc_id = uuid.uuid4()

        # Create multiple emotional memories
        memories = []
        for i in range(5):
            memory = MagicMock()
            memory.exchange_id = f"memory_{i}"
            memory.emotional_significance = MagicMock()
            memory.emotional_significance.overall_significance = 0.6 + (i * 0.1)
            memory.emotional_significance.emotional_type = (
                EmotionalMemoryType.EVERYDAY_POSITIVE
            )
            memory.emotional_significance.intensity_score = 0.5
            memory.affective_weight = MagicMock()
            memory.preserved_at = time.time() - (i * 3600)
            memories.append(memory)

        clusters = await engine.cluster_emotional_memories(
            npc_id=npc_id,
            memories=memories,
            personality=test_personality,
        )

        # With 5 memories and min_cluster_size=2, should have clusters
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert hasattr(cluster, "cluster_id")
            assert hasattr(cluster, "member_memories")
            assert len(cluster.member_memories) >= engine.min_cluster_size

    def test_get_network_statistics(self, engine):
        """Test network statistics generation."""
        npc_id = uuid.uuid4()

        # No network initially
        stats = engine.get_network_statistics(npc_id)
        assert "error" in stats

        # Create mock network
        from game_loop.core.memory.emotional_clustering import EmotionalNetwork

        network = EmotionalNetwork(npc_id=str(npc_id))
        engine._emotional_networks[str(npc_id)] = network

        stats = engine.get_network_statistics(npc_id)
        assert "total_clusters" in stats
        assert "total_memories" in stats
        assert "total_associations" in stats


class TestEmotionalDialogueIntegrationEngine:
    """Tests for emotional dialogue integration engine."""

    @pytest.fixture
    def engine(self, mock_session_factory, mock_llm_client, memory_config):
        """Create emotional dialogue integration engine."""
        # Create mock engines
        emotional_context_engine = MagicMock()
        affective_engine = MagicMock()
        preservation_engine = MagicMock()
        clustering_engine = MagicMock()
        mood_engine = MagicMock()
        trauma_engine = MagicMock()
        memory_integration = MagicMock()

        return EmotionalDialogueIntegrationEngine(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            config=memory_config,
            emotional_context_engine=emotional_context_engine,
            affective_engine=affective_engine,
            preservation_engine=preservation_engine,
            clustering_engine=clustering_engine,
            mood_engine=mood_engine,
            trauma_engine=trauma_engine,
            memory_integration=memory_integration,
        )

    @pytest.mark.asyncio
    async def test_create_emotional_dialogue_context(
        self, engine, test_conversation_context, test_personality
    ):
        """Test creating emotional dialogue context."""
        npc_id = uuid.uuid4()
        player_message = "I'm feeling really stressed about work."

        # Mock the mood engine to return a mood
        engine.mood_engine.get_current_mood.return_value = MoodStateRecord(
            npc_id=str(npc_id),
            mood_state=MoodState.CONTENT,
            intensity=0.6,
            stability=0.8,
            trigger_source="test",
            transition_type=MoodTransition.NATURAL_PROGRESSION,
        )

        # Mock the emotional context engine
        mock_significance = MagicMock()
        mock_significance.emotional_type = EmotionalMemoryType.EVERYDAY_POSITIVE
        mock_significance.overall_significance = 0.5
        engine.emotional_context_engine.analyze_emotional_significance = AsyncMock(
            return_value=mock_significance
        )

        # Mock the mood engine memory retrieval
        engine.mood_engine.get_mood_adjusted_memories = AsyncMock(return_value=([], {}))

        # Mock trauma engine
        engine.trauma_engine.evaluate_trauma_access_request = AsyncMock()
        mock_decision = MagicMock()
        mock_decision.access_granted = True
        mock_decision.denied_memories = []
        engine.trauma_engine.evaluate_trauma_access_request.return_value = mock_decision

        # Mock clustering engine
        engine.clustering_engine.get_associated_memories = AsyncMock(return_value=[])

        dialogue_context = await engine.create_emotional_dialogue_context(
            npc_id=npc_id,
            conversation_context=test_conversation_context,
            player_message=player_message,
            personality=test_personality,
        )

        assert isinstance(dialogue_context, EmotionalDialogueContext)
        assert dialogue_context.npc_id == str(npc_id)
        assert dialogue_context.current_mood == MoodState.CONTENT
        assert dialogue_context.emotional_response_mode in EmotionalResponseMode

    @pytest.mark.asyncio
    async def test_generate_emotional_response_guidance(
        self, engine, test_conversation_context, test_personality
    ):
        """Test generating emotional response guidance."""
        dialogue_context = EmotionalDialogueContext(
            npc_id=str(uuid.uuid4()),
            conversation_id=str(test_conversation_context.conversation_id),
            current_mood=MoodState.CONTENT,
            emotional_response_mode=EmotionalResponseMode.SUPPORTIVE,
        )

        guidance = await engine.generate_emotional_response_guidance(
            dialogue_context=dialogue_context,
            conversation_context=test_conversation_context,
            personality=test_personality,
            response_intent="conversational",
        )

        assert hasattr(guidance, "emotional_tone")
        assert hasattr(guidance, "empathy_level")
        assert hasattr(guidance, "vulnerability_sharing")
        assert 0.0 <= guidance.empathy_level <= 1.0
        assert 0.0 <= guidance.vulnerability_sharing <= 1.0

    def test_performance_tracking(self, engine):
        """Test performance statistics tracking."""
        stats = engine.get_performance_stats()

        expected_keys = [
            "dialogue_integrations",
            "emotional_responses_generated",
            "trauma_protections_activated",
            "therapeutic_interventions",
            "memory_references_made",
            "avg_integration_time_ms",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestIntegrationScenarios:
    """Integration tests for complex emotional memory scenarios."""

    @pytest.fixture
    def full_system(
        self, mock_session_factory, mock_llm_client, memory_config, test_personality
    ):
        """Create full emotional memory system for integration testing."""
        # Create all engines
        emotional_context_engine = EmotionalMemoryContextEngine(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            config=memory_config,
        )

        affective_engine = AffectiveMemoryWeightingEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            emotional_context_engine=emotional_context_engine,
        )

        preservation_engine = EmotionalPreservationEngine(
            session_factory=mock_session_factory,
        )

        clustering_engine = EmotionalMemoryClusteringEngine(
            session_factory=mock_session_factory,
        )

        mood_engine = MoodDependentMemoryEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            affective_engine=affective_engine,
            preservation_engine=preservation_engine,
        )

        trauma_engine = TraumaProtectionEngine(
            session_factory=mock_session_factory,
            config=memory_config,
            mood_engine=mood_engine,
        )

        memory_integration = MagicMock()  # Would be actual MemoryIntegrationInterface

        dialogue_engine = EmotionalDialogueIntegrationEngine(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            config=memory_config,
            emotional_context_engine=emotional_context_engine,
            affective_engine=affective_engine,
            preservation_engine=preservation_engine,
            clustering_engine=clustering_engine,
            mood_engine=mood_engine,
            trauma_engine=trauma_engine,
            memory_integration=memory_integration,
        )

        return {
            "emotional_context": emotional_context_engine,
            "affective": affective_engine,
            "preservation": preservation_engine,
            "clustering": clustering_engine,
            "mood": mood_engine,
            "trauma": trauma_engine,
            "dialogue": dialogue_engine,
        }

    @pytest.mark.asyncio
    async def test_traumatic_memory_protection_flow(
        self, full_system, test_personality, test_conversation_context
    ):
        """Test complete flow of traumatic memory protection."""
        npc_id = uuid.uuid4()

        # Create traumatic conversation exchange
        traumatic_exchange = ConversationExchange(
            exchange_id=uuid.uuid4(),
            conversation_id=test_conversation_context.conversation_id,
            speaker_id=test_conversation_context.player_id,
            message_text="I was attacked and couldn't escape. It was terrifying.",
            message_type="statement",
            emotion="fearful",
            timestamp=datetime.now(timezone.utc),
        )

        # Mock the basic emotional analysis to return traumatic content
        with patch.object(
            full_system["emotional_context"].emotional_analyzer,
            "analyze_emotional_weight",
        ) as mock_analyzer:
            mock_analyzer.return_value = MagicMock(
                emotional_weight=0.9,
                emotional_intensity=0.95,
                sentiment_score=-0.8,
                relationship_impact=0.3,
                analysis_confidence=0.9,
                emotional_keywords=["attacked", "terrifying", "couldn't escape"],
            )

            # Analyze emotional significance
            significance = await full_system[
                "emotional_context"
            ].analyze_emotional_significance(
                traumatic_exchange, test_conversation_context, test_personality
            )

            # Should be classified as traumatic
            assert significance.emotional_type == EmotionalMemoryType.TRAUMATIC
            assert significance.protection_level in [
                MemoryProtectionLevel.PROTECTED,
                MemoryProtectionLevel.TRAUMATIC,
            ]

            # Build trauma profile
            emotional_record = EmotionalMemoryRecord(
                exchange_id=str(traumatic_exchange.exchange_id),
                emotional_significance=significance,
                affective_weight=MagicMock(),  # Would be calculated
            )

            trauma_profile = await full_system["trauma"].build_trauma_memory_profile(
                npc_id=npc_id,
                memories=[emotional_record],
                personality=test_personality,
            )

            assert len(trauma_profile.trauma_memories) == 1
            assert trauma_profile.trauma_severity > 0.8
            assert trauma_profile.primary_response_type in TraumaResponseType

    @pytest.mark.asyncio
    async def test_mood_dependent_dialogue_generation(
        self, full_system, test_personality, test_conversation_context
    ):
        """Test mood-dependent dialogue context generation."""
        npc_id = uuid.uuid4()

        # Set NPC to anxious mood
        await full_system["mood"].update_npc_mood(
            npc_id=npc_id,
            new_mood=MoodState.ANXIOUS,
            intensity=0.8,
            trigger_source="test_scenario",
        )

        # Mock necessary components for dialogue context creation
        full_system["emotional_context"].analyze_emotional_significance = AsyncMock(
            return_value=MagicMock(
                emotional_type=EmotionalMemoryType.EVERYDAY_POSITIVE,
                overall_significance=0.4,
            )
        )

        full_system["mood"].get_mood_adjusted_memories = AsyncMock(
            return_value=([], {})
        )
        full_system["trauma"].evaluate_trauma_access_request = AsyncMock(
            return_value=MagicMock(
                access_granted=True,
                denied_memories=[],
            )
        )
        full_system["clustering"].get_associated_memories = AsyncMock(return_value=[])

        # Create dialogue context
        dialogue_context = await full_system[
            "dialogue"
        ].create_emotional_dialogue_context(
            npc_id=npc_id,
            conversation_context=test_conversation_context,
            player_message="How are you feeling today?",
            personality=test_personality,
        )

        # Anxious mood should influence dialogue context
        assert dialogue_context.current_mood == MoodState.ANXIOUS
        assert dialogue_context.mood_intensity == 0.8

        # Generate response guidance
        guidance = await full_system["dialogue"].generate_emotional_response_guidance(
            dialogue_context=dialogue_context,
            conversation_context=test_conversation_context,
            personality=test_personality,
        )

        # Should reflect anxious state in guidance
        assert guidance.emotional_tone in ["calming", "gentle", "reassuring"]

    def test_system_performance_monitoring(self, full_system):
        """Test that all engines track performance correctly."""
        engines_with_stats = [
            "emotional_context",
            "affective",
            "preservation",
            "mood",
            "trauma",
            "dialogue",
        ]

        for engine_name in engines_with_stats:
            engine = full_system[engine_name]
            stats = engine.get_performance_stats()

            # All engines should have basic performance stats
            assert isinstance(stats, dict)
            assert len(stats) > 0

            # Should have timing statistics
            timing_keys = [
                key for key in stats.keys() if "time_ms" in key or "avg" in key
            ]
            assert len(timing_keys) > 0

            # All values should be numeric
            for value in stats.values():
                assert isinstance(value, (int, float))

    def test_system_cache_management(self, full_system):
        """Test that all engines handle cache management correctly."""
        engines_with_caches = [
            "emotional_context",
            "affective",
            "preservation",
            "mood",
            "trauma",
            "dialogue",
        ]

        for engine_name in engines_with_caches:
            engine = full_system[engine_name]

            # Should have clear_caches method
            assert hasattr(engine, "clear_caches")

            # Should be callable without errors
            engine.clear_caches()

            # Performance stats should reset to defaults
            stats = engine.get_performance_stats()

            # Most counters should be 0 after cache clear
            counter_keys = [
                key
                for key in stats.keys()
                if any(
                    word in key for word in ["total_", "count", "hits", "operations"]
                )
            ]

            for key in counter_keys:
                assert stats[key] == 0


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for emotional memory system."""

    @pytest.mark.asyncio
    async def test_emotional_significance_analysis_performance(
        self,
        mock_session_factory,
        mock_llm_client,
        memory_config,
        test_exchange,
        test_conversation_context,
        test_personality,
    ):
        """Benchmark emotional significance analysis performance."""
        engine = EmotionalMemoryContextEngine(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
            config=memory_config,
        )

        # Mock the basic emotional analysis
        with patch.object(
            engine.emotional_analyzer, "analyze_emotional_weight"
        ) as mock_analyzer:
            mock_analyzer.return_value = MagicMock(
                emotional_weight=0.6,
                emotional_intensity=0.5,
                sentiment_score=0.2,
                relationship_impact=0.4,
                analysis_confidence=0.8,
                emotional_keywords=["test"],
            )

            start_time = time.time()

            # Perform multiple analyses
            for _ in range(10):
                await engine.analyze_emotional_significance(
                    test_exchange, test_conversation_context, test_personality
                )

            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_analysis = total_time / 10

            # Should complete analysis in reasonable time (< 100ms per analysis)
            assert avg_time_per_analysis < 0.1

            # Check performance stats
            stats = engine.get_performance_stats()
            assert stats["total_significance_analyses"] == 10

    @pytest.mark.asyncio
    async def test_memory_clustering_performance(
        self, mock_session_factory, test_personality
    ):
        """Benchmark memory clustering performance."""
        engine = EmotionalMemoryClusteringEngine(
            session_factory=mock_session_factory,
            min_cluster_size=2,
        )

        # Create many memories for clustering
        memories = []
        for i in range(50):  # Larger dataset
            memory = MagicMock()
            memory.exchange_id = f"memory_{i}"
            memory.emotional_significance = MagicMock()
            memory.emotional_significance.overall_significance = 0.3 + (i % 7) * 0.1
            memory.emotional_significance.emotional_type = list(EmotionalMemoryType)[
                i % len(EmotionalMemoryType)
            ]
            memory.emotional_significance.intensity_score = 0.4 + (i % 6) * 0.1
            memory.emotional_significance.protection_level = list(
                MemoryProtectionLevel
            )[i % len(MemoryProtectionLevel)]
            memory.affective_weight = MagicMock()
            memory.affective_weight.final_weight = 0.5 + (i % 5) * 0.1
            memory.preserved_at = time.time() - (i * 100)
            memories.append(memory)

        npc_id = uuid.uuid4()

        start_time = time.time()

        clusters = await engine.cluster_emotional_memories(
            npc_id=npc_id,
            memories=memories,
            personality=test_personality,
        )

        end_time = time.time()
        clustering_time = end_time - start_time

        # Should complete clustering in reasonable time (< 2 seconds for 50 memories)
        assert clustering_time < 2.0

        # Should produce reasonable clusters
        assert len(clusters) > 0
        assert len(clusters) <= engine.max_clusters

        # Check performance stats
        stats = engine.get_performance_stats()
        assert stats["clustering_operations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
