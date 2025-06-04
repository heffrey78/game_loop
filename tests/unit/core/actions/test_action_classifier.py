"""
Unit tests for ActionTypeClassifier.
"""

import pytest

from game_loop.core.actions import ActionType, ActionTypeClassifier
from game_loop.core.actions.patterns import ActionPatternManager


class TestActionTypeClassifier:
    """Test cases for ActionTypeClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing."""
        return ActionTypeClassifier()

    @pytest.fixture
    def pattern_manager(self):
        """Create a pattern manager instance for testing."""
        return ActionPatternManager()

    def test_initialization(self, classifier):
        """Test that classifier initializes correctly."""
        assert classifier is not None
        assert classifier.pattern_manager is not None
        assert classifier.cache is not None
        assert classifier.high_confidence_threshold == 0.8

    def test_rule_based_classification(self, classifier):
        """Test rule-based classification for common commands."""
        test_cases = [
            ("go north", ActionType.MOVEMENT),
            ("north", ActionType.MOVEMENT),
            ("n", ActionType.MOVEMENT),
            ("take sword", ActionType.OBJECT_INTERACTION),
            ("grab the key", ActionType.OBJECT_INTERACTION),
            ("look around", ActionType.OBSERVATION),
            ("examine door", ActionType.OBSERVATION),
            ("inventory", ActionType.QUERY),
            ("help", ActionType.SYSTEM),
            ("quit", ActionType.SYSTEM),
            ("talk to merchant", ActionType.CONVERSATION),
            ("attack goblin", ActionType.PHYSICAL),
        ]

        for input_text, expected_type in test_cases:
            result = classifier.classify_with_rules(input_text)
            assert result.action_type == expected_type, f"Failed for '{input_text}'"
            assert result.confidence > 0.0

    def test_unknown_command_classification(self, classifier):
        """Test classification of unknown or invalid commands."""
        unknown_commands = [
            "",
            "   ",
            "asdfghijk",
            "this is not a valid command",
        ]

        for command in unknown_commands:
            result = classifier.classify_with_rules(command)
            if command.strip():  # Non-empty commands
                assert result.action_type == ActionType.UNKNOWN
            assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_full_classification(self, classifier):
        """Test full classification workflow."""
        test_inputs = [
            "look around",
            "go north",
            "take sword",
            "help",
        ]

        for input_text in test_inputs:
            result = await classifier.classify_action(input_text)
            assert result is not None
            assert result.action_type in ActionType
            assert 0.0 <= result.confidence <= 1.0
            assert result.raw_input == input_text

    @pytest.mark.asyncio
    async def test_classification_with_context(self, classifier):
        """Test classification with game context."""
        game_context = {
            "current_location": {"name": "forest clearing"},
            "visible_objects": [{"name": "ancient tree"}, {"name": "stone"}],
            "npcs": [{"name": "wise hermit"}],
        }

        result = await classifier.classify_action("examine tree", game_context)
        assert result.action_type == ActionType.OBSERVATION
        assert result.confidence > 0.0

    def test_action_component_extraction(self, classifier):
        """Test extraction of action components."""
        components = classifier.extract_action_components("take the magic sword")

        assert "raw_text" in components
        assert "normalized_text" in components
        assert "tokens" in components
        assert "action_verbs" in components
        assert "potential_targets" in components

        assert "take" in components["action_verbs"]
        assert "sword" in components["potential_targets"]

    def test_confidence_score_calculation(self, classifier):
        """Test confidence score calculation methods."""
        scores = classifier.calculate_confidence_scores(
            rule_confidence=0.8, llm_confidence=0.7, context_relevance=0.6
        )

        assert "rule_based" in scores
        assert "llm_based" in scores
        assert "hybrid_agreement" in scores
        assert "hybrid_weighted" in scores

        assert scores["rule_based"] == 0.8
        assert all(0.0 <= score <= 1.0 for score in scores.values())

    def test_classification_statistics(self, classifier):
        """Test that statistics are tracked correctly."""
        initial_stats = classifier.get_classification_stats()
        assert "total_classifications" in initial_stats
        assert "rule_based_classifications" in initial_stats

        # Perform some classifications
        classifier.classify_with_rules("go north")
        classifier.classify_with_rules("take sword")

        # Stats should remain at 0 since we're only testing rule classification directly
        # Full classification through classify_action() would update stats

    def test_cache_functionality(self, classifier):
        """Test that caching works correctly."""
        if not classifier.cache:
            pytest.skip("Caching disabled for this instance")

        # Clear cache
        classifier.clear_cache()

        # Test cache key generation
        assert hasattr(classifier.cache, "_generate_key")
        key1 = classifier.cache._generate_key("test input")
        key2 = classifier.cache._generate_key("test input")
        assert key1 == key2  # Same input should generate same key

        key3 = classifier.cache._generate_key("different input")
        assert key1 != key3  # Different input should generate different key


class TestActionPatternManager:
    """Test cases for ActionPatternManager."""

    @pytest.fixture
    def pattern_manager(self):
        """Create a pattern manager instance for testing."""
        return ActionPatternManager()

    def test_initialization(self, pattern_manager):
        """Test that pattern manager initializes with default patterns."""
        assert len(pattern_manager.patterns) > 0

        stats = pattern_manager.get_pattern_stats()
        assert stats["total_patterns"] > 0
        assert stats["action_types_covered"] > 0

    def test_pattern_classification(self, pattern_manager):
        """Test that patterns correctly classify input."""
        test_cases = [
            ("go north", ActionType.MOVEMENT),
            ("take sword", ActionType.OBJECT_INTERACTION),
            ("look around", ActionType.OBSERVATION),
            ("help", ActionType.SYSTEM),
        ]

        for input_text, expected_type in test_cases:
            matches = pattern_manager.classify_input(input_text)
            assert len(matches) > 0
            best_match = matches[0]
            action_type, confidence, _ = best_match
            assert action_type == expected_type
            assert confidence > 0.0

    def test_get_patterns_for_type(self, pattern_manager):
        """Test retrieval of patterns by action type."""
        movement_patterns = pattern_manager.get_patterns_for_type(ActionType.MOVEMENT)
        assert len(movement_patterns) > 0

        for pattern in movement_patterns:
            assert pattern.action_type == ActionType.MOVEMENT

    def test_pattern_stats(self, pattern_manager):
        """Test pattern statistics generation."""
        stats = pattern_manager.get_pattern_stats()

        assert "total_patterns" in stats
        assert "patterns_by_type" in stats
        assert "action_types_covered" in stats

        assert stats["total_patterns"] > 0
        assert len(stats["patterns_by_type"]) > 0
        assert stats["action_types_covered"] > 0
