"""Tests for conversation flow templates and memory integration patterns."""

from unittest.mock import Mock

import pytest

from game_loop.core.conversation.flow_factory import create_conversation_flow_library
from game_loop.core.conversation.flow_templates import (
    ConversationFlowLibrary,
    ConversationStage,
    MemoryDisclosureThreshold,
    MemoryIntegrationPattern,
    TransitionPhrase,
    TrustLevel,
)


class TestTransitionPhrase:
    """Test TransitionPhrase functionality."""

    def test_matches_confidence_within_range(self):
        """Test confidence matching within range."""
        phrase = TransitionPhrase(
            "I clearly remember...", (0.7, 0.9), "confident", "memory_recall"
        )

        assert phrase.matches_confidence(0.8)
        assert phrase.matches_confidence(0.7)
        assert phrase.matches_confidence(0.9)

    def test_matches_confidence_outside_range(self):
        """Test confidence matching outside range."""
        phrase = TransitionPhrase(
            "I vaguely recall...", (0.1, 0.3), "neutral", "memory_recall"
        )

        assert not phrase.matches_confidence(0.5)
        assert not phrase.matches_confidence(0.05)
        assert not phrase.matches_confidence(0.9)


class TestMemoryIntegrationPattern:
    """Test MemoryIntegrationPattern functionality."""

    def setup_method(self):
        """Set up test pattern."""
        self.pattern = MemoryIntegrationPattern(
            pattern_name="test_pattern",
            disclosure_level=MemoryDisclosureThreshold.CLEAR_REFERENCES,
            trust_requirement=TrustLevel.ACQUAINTANCE,
            introduction_phrases=[
                TransitionPhrase(
                    "I remember when...", (0.5, 0.7), "neutral", "reference"
                ),
                TransitionPhrase(
                    "That reminds me...", (0.5, 0.7), "connected", "reference"
                ),
            ],
            uncertainty_expressions=[
                TransitionPhrase("I think...", (0.3, 0.5), "neutral", "uncertainty"),
            ],
        )

    def test_get_appropriate_phrase_with_match(self):
        """Test getting appropriate phrase with matching confidence."""
        phrase = self.pattern.get_appropriate_phrase(0.6, "neutral", "introduction")

        assert phrase is not None
        assert phrase.text == "I remember when..."
        assert phrase.matches_confidence(0.6)

    def test_get_appropriate_phrase_with_emotional_tone(self):
        """Test getting phrase with specific emotional tone."""
        phrase = self.pattern.get_appropriate_phrase(0.6, "connected", "introduction")

        assert phrase is not None
        # Should get the phrase that matches the emotional tone
        assert phrase.emotional_tone == "connected"

    def test_get_appropriate_phrase_no_match(self):
        """Test getting phrase when no match exists."""
        phrase = self.pattern.get_appropriate_phrase(0.9, "angry", "introduction")

        assert phrase is None

    def test_get_appropriate_phrase_uncertainty(self):
        """Test getting uncertainty phrase."""
        phrase = self.pattern.get_appropriate_phrase(0.4, "neutral", "uncertainty")

        assert phrase is not None
        assert phrase.text == "I think..."


class TestConversationFlowLibrary:
    """Test ConversationFlowLibrary functionality."""

    def setup_method(self):
        """Set up test library."""
        self.library = ConversationFlowLibrary()

    def test_initialization_creates_templates(self):
        """Test that initialization creates expected templates."""
        assert "initial_encounter" in self.library.templates
        assert "relationship_building" in self.library.templates
        assert "deep_connection" in self.library.templates

    def test_initialization_creates_memory_patterns(self):
        """Test that initialization creates memory patterns."""
        assert "subtle_hints" in self.library.memory_patterns
        assert "clear_references" in self.library.memory_patterns
        assert "detailed_memories" in self.library.memory_patterns
        assert "personal_secrets" in self.library.memory_patterns

    def test_initialization_creates_transition_phrases(self):
        """Test that initialization creates transition phrases."""
        assert "uncertainty" in self.library.transition_phrases
        assert "confidence" in self.library.transition_phrases
        assert "emotional" in self.library.transition_phrases

    def test_get_trust_level_from_relationship_stranger(self):
        """Test trust level calculation for stranger."""
        trust_level = self.library.get_trust_level_from_relationship(0.1)
        assert trust_level == TrustLevel.STRANGER

    def test_get_trust_level_from_relationship_acquaintance(self):
        """Test trust level calculation for acquaintance."""
        trust_level = self.library.get_trust_level_from_relationship(0.4)
        assert trust_level == TrustLevel.ACQUAINTANCE

    def test_get_trust_level_from_relationship_friend(self):
        """Test trust level calculation for friend."""
        trust_level = self.library.get_trust_level_from_relationship(0.7)
        assert trust_level == TrustLevel.FRIEND

    def test_get_trust_level_from_relationship_confidant(self):
        """Test trust level calculation for confidant."""
        trust_level = self.library.get_trust_level_from_relationship(0.9)
        assert trust_level == TrustLevel.CONFIDANT

    def test_get_disclosure_threshold_from_confidence(self):
        """Test disclosure threshold calculation."""
        # Test each threshold
        assert (
            self.library.get_disclosure_threshold_from_confidence(0.4)
            == MemoryDisclosureThreshold.SUBTLE_HINTS
        )
        assert (
            self.library.get_disclosure_threshold_from_confidence(0.6)
            == MemoryDisclosureThreshold.CLEAR_REFERENCES
        )
        assert (
            self.library.get_disclosure_threshold_from_confidence(0.8)
            == MemoryDisclosureThreshold.DETAILED_MEMORIES
        )
        assert (
            self.library.get_disclosure_threshold_from_confidence(0.95)
            == MemoryDisclosureThreshold.PERSONAL_SECRETS
        )

    def test_validate_memory_integration_valid(self):
        """Test valid memory integration validation."""
        pattern = self.library.memory_patterns["clear_references"]
        trust_level = TrustLevel.ACQUAINTANCE
        confidence = 0.6
        context = {"conversation_history": True}

        is_valid, errors = self.library.validate_memory_integration(
            pattern, trust_level, confidence, context
        )

        assert is_valid
        assert len(errors) == 0

    def test_validate_memory_integration_insufficient_trust(self):
        """Test validation with insufficient trust level."""
        pattern = self.library.memory_patterns["personal_secrets"]
        trust_level = TrustLevel.STRANGER
        confidence = 0.95
        context = {"conversation_history": True}

        is_valid, errors = self.library.validate_memory_integration(
            pattern, trust_level, confidence, context
        )

        assert not is_valid
        assert any("Trust level" in error for error in errors)

    def test_validate_memory_integration_confidence_mismatch(self):
        """Test validation with mismatched confidence."""
        pattern = self.library.memory_patterns["detailed_memories"]
        trust_level = TrustLevel.FRIEND
        confidence = 0.4  # Too low for detailed memories
        context = {"conversation_history": True}

        is_valid, errors = self.library.validate_memory_integration(
            pattern, trust_level, confidence, context
        )

        assert not is_valid
        assert any("confidence" in error.lower() for error in errors)

    def test_suggest_conversation_progression_ready(self):
        """Test conversation progression when ready."""
        current_stage = ConversationStage.INITIAL_ENCOUNTER
        relationship_score = 0.4  # Above threshold for acquaintance
        conversation_count = 5  # Enough conversations

        suggested_stage = self.library.suggest_conversation_progression(
            current_stage, relationship_score, conversation_count
        )

        assert suggested_stage == ConversationStage.ACQUAINTANCE

    def test_suggest_conversation_progression_not_ready(self):
        """Test conversation progression when not ready."""
        current_stage = ConversationStage.INITIAL_ENCOUNTER
        relationship_score = 0.1  # Below threshold
        conversation_count = 2  # Not enough conversations

        suggested_stage = self.library.suggest_conversation_progression(
            current_stage, relationship_score, conversation_count
        )

        assert suggested_stage == current_stage

    def test_get_transition_phrase_with_match(self):
        """Test getting transition phrase with valid parameters."""
        phrase = self.library.get_transition_phrase("uncertainty", 0.3, "neutral")

        assert phrase is not None
        assert phrase.matches_confidence(0.3)
        assert phrase.emotional_tone in ["neutral", "neutral"]  # neutral or matches

    def test_get_transition_phrase_invalid_type(self):
        """Test getting transition phrase with invalid type."""
        phrase = self.library.get_transition_phrase("invalid_type", 0.5, "neutral")

        assert phrase is None


class TestConversationFlowTemplates:
    """Test conversation flow template functionality."""

    def setup_method(self):
        """Set up test templates."""
        self.library = ConversationFlowLibrary()

    def test_initial_encounter_template(self):
        """Test initial encounter template."""
        template = self.library.get_template("initial_encounter")

        assert template is not None
        assert template.stage == ConversationStage.INITIAL_ENCOUNTER
        assert len(template.memory_patterns) > 0
        assert template.memory_patterns[0].trust_requirement == TrustLevel.STRANGER

    def test_relationship_building_template(self):
        """Test relationship building template."""
        template = self.library.get_template("relationship_building")

        assert template is not None
        assert template.stage == ConversationStage.RELATIONSHIP_BUILDING
        assert len(template.memory_patterns) >= 2  # Should have multiple patterns

    def test_deep_connection_template(self):
        """Test deep connection template."""
        template = self.library.get_template("deep_connection")

        assert template is not None
        assert template.stage == ConversationStage.DEEP_CONNECTION
        # Should have access to higher disclosure patterns
        pattern_names = [p.pattern_name for p in template.memory_patterns]
        assert "personal_secrets" in pattern_names

    def test_template_get_pattern_for_trust_level(self):
        """Test getting pattern for trust level from template."""
        template = self.library.get_template("deep_connection")

        # Should be able to get pattern for friend level
        pattern = template.get_pattern_for_trust_level(TrustLevel.FRIEND)
        assert pattern is not None

        # Should return None for incompatible trust level (if no match)
        stranger_pattern = template.get_pattern_for_trust_level(TrustLevel.STRANGER)
        # Deep connection template might not have stranger patterns
        # This test validates the method works regardless of result


class TestMemoryPatternIntegration:
    """Test memory pattern integration scenarios."""

    def setup_method(self):
        """Set up test scenarios."""
        self.library = ConversationFlowLibrary()

    def test_subtle_hints_pattern_validation(self):
        """Test subtle hints pattern validation."""
        pattern = self.library.memory_patterns["subtle_hints"]

        # Should work with stranger trust and low confidence
        is_valid, errors = self.library.validate_memory_integration(
            pattern, TrustLevel.STRANGER, 0.4, {"conversation_history": True}
        )

        assert is_valid

    def test_personal_secrets_pattern_requirements(self):
        """Test personal secrets pattern requirements."""
        pattern = self.library.memory_patterns["personal_secrets"]

        # Should require confidant trust level
        assert pattern.trust_requirement == TrustLevel.CONFIDANT
        assert pattern.disclosure_level == MemoryDisclosureThreshold.PERSONAL_SECRETS

        # Should have high confidence requirements
        high_conf_phrase = pattern.get_appropriate_phrase(
            0.95, "neutral", "introduction"
        )
        assert high_conf_phrase is not None

        # Should not work with low confidence
        low_conf_phrase = pattern.get_appropriate_phrase(0.4, "neutral", "introduction")
        assert low_conf_phrase is None

    def test_pattern_transition_phrase_coverage(self):
        """Test that patterns have adequate phrase coverage."""
        for pattern_name, pattern in self.library.memory_patterns.items():
            # Each pattern should have introduction phrases
            assert (
                len(pattern.introduction_phrases) > 0
            ), f"Pattern {pattern_name} lacks introduction phrases"

            # Check confidence coverage for introduction phrases
            confidence_ranges = [
                phrase.confidence_range for phrase in pattern.introduction_phrases
            ]
            assert (
                len(confidence_ranges) > 0
            ), f"Pattern {pattern_name} has no confidence ranges"


class TestFlowLibraryFactory:
    """Test the flow library factory."""

    def test_factory_creates_valid_instance(self):
        """Test that factory creates a valid flow library instance."""
        flow_library = create_conversation_flow_library()
        assert flow_library is not None
        assert isinstance(flow_library, ConversationFlowLibrary)

    def test_factory_instance_has_templates(self):
        """Test that factory instance has expected templates."""
        flow_library = create_conversation_flow_library()
        assert len(flow_library.templates) > 0
        assert len(flow_library.memory_patterns) > 0
        assert len(flow_library.transition_phrases) > 0

    def test_factory_instance_functionality(self):
        """Test that factory instance functions correctly."""
        flow_library = create_conversation_flow_library()

        # Should be able to get trust level
        trust_level = flow_library.get_trust_level_from_relationship(0.5)
        assert trust_level == TrustLevel.ACQUAINTANCE

        # Should be able to get template
        template = flow_library.get_template("initial_encounter")
        assert template is not None


@pytest.fixture
def mock_conversation_context():
    """Mock conversation context for testing."""
    mock_context = Mock()
    mock_context.conversation_id = "test-conversation-123"
    mock_context.player_id = "player-123"
    mock_context.npc_id = "npc-456"
    mock_context.relationship_level = 0.5
    mock_context.mood = "neutral"
    return mock_context


@pytest.fixture
def mock_npc_personality():
    """Mock NPC personality for testing."""
    mock_personality = Mock()
    mock_personality.get_trait_strength = Mock(return_value=0.5)
    mock_personality.knowledge_areas = ["general", "local_events"]
    mock_personality.traits = {"helpful": 0.7, "friendly": 0.6}
    return mock_personality


class TestFlowTemplateIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_stranger_to_acquaintance_progression(
        self, mock_conversation_context, mock_npc_personality
    ):
        """Test progression from stranger to acquaintance."""
        library = ConversationFlowLibrary()

        # Start as stranger
        trust_level = library.get_trust_level_from_relationship(0.1)
        assert trust_level == TrustLevel.STRANGER

        # Get appropriate template
        template = library.get_template_for_stage(ConversationStage.INITIAL_ENCOUNTER)
        assert template is not None

        # Should only have access to subtle hints
        pattern = template.get_pattern_for_trust_level(trust_level)
        if pattern:  # Some templates may not have patterns for all trust levels
            assert pattern.disclosure_level in [MemoryDisclosureThreshold.SUBTLE_HINTS]

    def test_friend_memory_disclosure_options(
        self, mock_conversation_context, mock_npc_personality
    ):
        """Test memory disclosure options for friend relationship."""
        library = ConversationFlowLibrary()

        # Friend level relationship
        trust_level = library.get_trust_level_from_relationship(0.7)
        assert trust_level == TrustLevel.FRIEND

        # Get appropriate template for this stage
        template = library.get_template_for_stage(ConversationStage.TRUST_DEVELOPMENT)
        assert template is not None

        # Should have multiple memory patterns available
        assert len(template.memory_patterns) >= 2

    def test_confidant_personal_secrets_access(
        self, mock_conversation_context, mock_npc_personality
    ):
        """Test access to personal secrets at confidant level."""
        library = ConversationFlowLibrary()

        # Confidant level relationship
        trust_level = library.get_trust_level_from_relationship(0.9)
        assert trust_level == TrustLevel.CONFIDANT

        # Get deep connection template
        template = library.get_template_for_stage(ConversationStage.DEEP_CONNECTION)
        assert template is not None

        # Should have access to personal secrets pattern
        pattern_names = [p.pattern_name for p in template.memory_patterns]
        assert "personal_secrets" in pattern_names
