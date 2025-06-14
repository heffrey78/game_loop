"""
Unit tests for NPC data models.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from game_loop.core.models.npc_models import (
    NPCPersonality,
    NPCKnowledge,
    NPCDialogueState,
    NPCGenerationContext,
    GeneratedNPC,
    NPCStorageResult,
    NPCValidationResult,
    NPCGenerationMetrics,
    NPCArchetype,
    DialogueContext,
    DialogueResponse,
    NPCSearchCriteria,
)
from game_loop.core.models.location_models import LocationTheme
from game_loop.state.models import Location, NonPlayerCharacter, WorldObject


class TestNPCPersonality:
    """Test NPCPersonality data model."""

    def test_basic_creation(self):
        """Test basic personality creation."""
        personality = NPCPersonality(
            name="Test NPC",
            archetype="merchant",
            traits=["friendly", "greedy"],
            motivations=["profit", "reputation"],
            fears=["theft"],
            speech_patterns={"formality": "casual"},
            relationship_tendencies={"trusting": 0.7}
        )
        
        assert personality.name == "Test NPC"
        assert personality.archetype == "merchant"
        assert "friendly" in personality.traits
        assert "profit" in personality.motivations
        assert personality.speech_patterns["formality"] == "casual"

    def test_default_values(self):
        """Test default values are properly set."""
        personality = NPCPersonality(name="Test", archetype="guard")
        
        assert personality.traits == []
        assert personality.motivations == []
        assert personality.fears == []
        assert personality.speech_patterns == {}
        assert personality.relationship_tendencies == {}


class TestNPCKnowledge:
    """Test NPCKnowledge data model."""

    def test_basic_creation(self):
        """Test basic knowledge creation."""
        knowledge = NPCKnowledge(
            world_knowledge={"general": "some info"},
            local_knowledge={"location": "local info"},
            personal_history=["born here", "became merchant"],
            relationships={"john": {"type": "friend", "trust": 0.8}},
            secrets=["knows hidden passage"],
            expertise_areas=["trading", "appraisal"]
        )
        
        assert knowledge.world_knowledge["general"] == "some info"
        assert "born here" in knowledge.personal_history
        assert "trading" in knowledge.expertise_areas
        assert knowledge.relationships["john"]["trust"] == 0.8

    def test_default_values(self):
        """Test default values are properly set."""
        knowledge = NPCKnowledge()
        
        assert knowledge.world_knowledge == {}
        assert knowledge.local_knowledge == {}
        assert knowledge.personal_history == []
        assert knowledge.relationships == {}
        assert knowledge.secrets == []
        assert knowledge.expertise_areas == []


class TestNPCDialogueState:
    """Test NPCDialogueState data model."""

    def test_basic_creation(self):
        """Test basic dialogue state creation."""
        state = NPCDialogueState(
            current_mood="happy",
            relationship_level=0.5,
            conversation_history=[{"input": "hello", "response": "hi"}],
            active_topics=["greeting", "weather"],
            available_quests=["find_item"],
            interaction_count=5,
            last_interaction=datetime.now()
        )
        
        assert state.current_mood == "happy"
        assert state.relationship_level == 0.5
        assert len(state.conversation_history) == 1
        assert "greeting" in state.active_topics
        assert state.interaction_count == 5

    def test_default_values(self):
        """Test default values are properly set."""
        state = NPCDialogueState()
        
        assert state.current_mood == "neutral"
        assert state.relationship_level == 0.0
        assert state.conversation_history == []
        assert state.active_topics == []
        assert state.available_quests == []
        assert state.interaction_count == 0
        assert state.last_interaction is None


class TestNPCGenerationContext:
    """Test NPCGenerationContext data model."""

    def test_basic_creation(self):
        """Test basic context creation."""
        location = Location(
            location_id=uuid4(),
            name="Test Location",
            description="A test area",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Forest"}
        )
        
        theme = LocationTheme(
            name="Forest",
            description="Woodland area",
            visual_elements=["trees"],
            atmosphere="peaceful",
            typical_objects=["log"],
            typical_npcs=["hermit"],
            generation_parameters={}
        )
        
        context = NPCGenerationContext(
            location=location,
            location_theme=theme,
            nearby_npcs=[],
            world_state_snapshot={"total_locations": 5},
            player_level=3,
            generation_purpose="populate_location",
            constraints={"max_npcs": 2}
        )
        
        assert context.location.name == "Test Location"
        assert context.location_theme.name == "Forest"
        assert context.player_level == 3
        assert context.generation_purpose == "populate_location"
        assert context.constraints["max_npcs"] == 2


class TestGeneratedNPC:
    """Test GeneratedNPC data model."""

    def test_complete_npc_creation(self):
        """Test complete NPC with all components."""
        base_npc = NonPlayerCharacter(
            npc_id=uuid4(),
            name="Test Merchant",
            description="A friendly trader"
        )
        
        personality = NPCPersonality(
            name="Test Merchant",
            archetype="merchant",
            traits=["friendly", "business-minded"]
        )
        
        knowledge = NPCKnowledge(
            expertise_areas=["trading", "appraisal"]
        )
        
        dialogue_state = NPCDialogueState(
            current_mood="neutral"
        )
        
        generated_npc = GeneratedNPC(
            base_npc=base_npc,
            personality=personality,
            knowledge=knowledge,
            dialogue_state=dialogue_state,
            generation_metadata={"created_by": "test"},
            embedding_vector=[0.1, 0.2, 0.3]
        )
        
        assert generated_npc.base_npc.name == "Test Merchant"
        assert generated_npc.personality.archetype == "merchant"
        assert "trading" in generated_npc.knowledge.expertise_areas
        assert generated_npc.dialogue_state.current_mood == "neutral"
        assert generated_npc.generation_metadata["created_by"] == "test"
        assert len(generated_npc.embedding_vector) == 3


class TestNPCGenerationMetrics:
    """Test NPCGenerationMetrics data model."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = NPCGenerationMetrics(
            generation_time_ms=1000,
            context_collection_time_ms=200,
            llm_response_time_ms=500,
            validation_time_ms=100,
            storage_time_ms=150,
            cache_hit=False
        )
        
        assert metrics.generation_time_ms == 1000
        assert metrics.context_collection_time_ms == 200
        assert metrics.llm_response_time_ms == 500
        assert metrics.validation_time_ms == 100
        assert metrics.storage_time_ms == 150
        assert not metrics.cache_hit

    def test_total_time_calculation(self):
        """Test total time is calculated correctly."""
        metrics = NPCGenerationMetrics(
            generation_time_ms=0,  # Will be calculated
            context_collection_time_ms=200,
            llm_response_time_ms=500,
            validation_time_ms=100,
            storage_time_ms=150
        )
        
        # total_time_ms should be sum of all components
        expected_total = 200 + 500 + 100 + 150
        assert metrics.total_time_ms == expected_total

    def test_default_values(self):
        """Test default values are properly set."""
        metrics = NPCGenerationMetrics()
        
        assert metrics.generation_time_ms == 0
        assert metrics.context_collection_time_ms == 0
        assert metrics.llm_response_time_ms == 0
        assert metrics.validation_time_ms == 0
        assert metrics.storage_time_ms == 0
        assert metrics.total_time_ms == 0
        assert not metrics.cache_hit


class TestNPCArchetype:
    """Test NPCArchetype data model."""

    def test_basic_creation(self):
        """Test basic archetype creation."""
        archetype = NPCArchetype(
            name="test_merchant",
            description="A trader of goods",
            typical_traits=["persuasive", "social"],
            typical_motivations=["profit", "reputation"],
            speech_patterns={"formality": "polite"},
            location_affinities={"Village": 0.9, "Forest": 0.2},
            archetype_id=uuid4()
        )
        
        assert archetype.name == "test_merchant"
        assert archetype.description == "A trader of goods"
        assert "persuasive" in archetype.typical_traits
        assert "profit" in archetype.typical_motivations
        assert archetype.location_affinities["Village"] == 0.9


class TestDialogueContext:
    """Test DialogueContext data model."""

    def test_basic_creation(self):
        """Test basic dialogue context creation."""
        base_npc = NonPlayerCharacter(
            npc_id=uuid4(),
            name="Test NPC",
            description="A test character"
        )
        
        personality = NPCPersonality(name="Test", archetype="guard")
        knowledge = NPCKnowledge()
        dialogue_state = NPCDialogueState()
        
        generated_npc = GeneratedNPC(
            base_npc=base_npc,
            personality=personality,
            knowledge=knowledge,
            dialogue_state=dialogue_state
        )
        
        context = DialogueContext(
            npc=generated_npc,
            player_input="Hello there",
            conversation_history=[],
            current_location=None,
            world_context={"time": "day"},
            interaction_type="casual"
        )
        
        assert context.npc.base_npc.name == "Test NPC"
        assert context.player_input == "Hello there"
        assert context.interaction_type == "casual"
        assert context.world_context["time"] == "day"


class TestDialogueResponse:
    """Test DialogueResponse data model."""

    def test_basic_creation(self):
        """Test basic dialogue response creation."""
        response = DialogueResponse(
            response_text="Hello, traveler!",
            mood_change="friendly",
            relationship_change=0.1,
            new_topics=["travel", "weather"],
            quest_offered="find_herbs",
            knowledge_shared={"local_area": "info"},
            response_metadata={"generated": True}
        )
        
        assert response.response_text == "Hello, traveler!"
        assert response.mood_change == "friendly"
        assert response.relationship_change == 0.1
        assert "travel" in response.new_topics
        assert response.quest_offered == "find_herbs"
        assert response.knowledge_shared["local_area"] == "info"

    def test_default_values(self):
        """Test default values are properly set."""
        response = DialogueResponse(response_text="Hello")
        
        assert response.response_text == "Hello"
        assert response.mood_change is None
        assert response.relationship_change == 0.0
        assert response.new_topics == []
        assert response.quest_offered is None
        assert response.knowledge_shared == {}
        assert response.response_metadata == {}


class TestNPCSearchCriteria:
    """Test NPCSearchCriteria data model."""

    def test_basic_creation(self):
        """Test basic search criteria creation."""
        criteria = NPCSearchCriteria(
            query_text="friendly merchant",
            archetype="merchant",
            location_id=uuid4(),
            personality_traits=["friendly", "helpful"],
            knowledge_areas=["trading"],
            relationship_level_min=0.0,
            relationship_level_max=1.0,
            max_results=5
        )
        
        assert criteria.query_text == "friendly merchant"
        assert criteria.archetype == "merchant"
        assert criteria.location_id is not None
        assert "friendly" in criteria.personality_traits
        assert "trading" in criteria.knowledge_areas
        assert criteria.max_results == 5

    def test_default_values(self):
        """Test default values are properly set."""
        criteria = NPCSearchCriteria()
        
        assert criteria.query_text is None
        assert criteria.archetype is None
        assert criteria.location_id is None
        assert criteria.personality_traits == []
        assert criteria.knowledge_areas == []
        assert criteria.relationship_level_min is None
        assert criteria.relationship_level_max is None
        assert criteria.max_results == 10


class TestNPCStorageResult:
    """Test NPCStorageResult data model."""

    def test_successful_storage(self):
        """Test successful storage result."""
        result = NPCStorageResult(
            success=True,
            npc_id=uuid4(),
            storage_time_ms=500,
            embedding_generated=True
        )
        
        assert result.success
        assert result.npc_id is not None
        assert result.storage_time_ms == 500
        assert result.embedding_generated
        assert result.error_message is None

    def test_failed_storage(self):
        """Test failed storage result."""
        result = NPCStorageResult(
            success=False,
            storage_time_ms=100,
            error_message="Database connection failed"
        )
        
        assert not result.success
        assert result.npc_id is None
        assert result.storage_time_ms == 100
        assert not result.embedding_generated
        assert result.error_message == "Database connection failed"


class TestNPCValidationResult:
    """Test NPCValidationResult data model."""

    def test_valid_result(self):
        """Test valid NPC result."""
        result = NPCValidationResult(
            is_valid=True,
            issues=[],
            suggestions=["Could add more personality"],
            confidence_score=0.8,
            personality_score=7.0,
            knowledge_score=6.5,
            consistency_score=8.0,
            approval=True
        )
        
        assert result.is_valid
        assert result.issues == []
        assert len(result.suggestions) == 1
        assert result.confidence_score == 0.8
        assert result.approval

    def test_invalid_result(self):
        """Test invalid NPC result."""
        result = NPCValidationResult(
            is_valid=False,
            issues=["Missing description", "No personality traits"],
            suggestions=["Add description", "Define traits"],
            confidence_score=0.3
        )
        
        assert not result.is_valid
        assert len(result.issues) == 2
        assert "Missing description" in result.issues
        assert len(result.suggestions) == 2
        assert result.confidence_score == 0.3
        assert not result.approval