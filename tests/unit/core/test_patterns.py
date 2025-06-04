"""Tests for the pattern management system."""

import re

from game_loop.core.patterns import ActionPattern, ActionPatternManager, ActionType


class TestActionPattern:
    """Test the ActionPattern class."""

    def test_pattern_creation(self):
        """Test creating an action pattern."""
        pattern = ActionPattern(
            name="test_pattern",
            action_type=ActionType.MOVEMENT,
            verbs={"go", "walk"},
            synonyms={"travel", "move"},
            regex_patterns=[re.compile(r"\bgo\s+to\b")],
            priority=1,
        )

        assert pattern.name == "test_pattern"
        assert pattern.action_type == ActionType.MOVEMENT
        assert "go" in pattern.verbs
        assert "travel" in pattern.synonyms
        assert len(pattern.regex_patterns) == 1
        assert pattern.priority == 1

    def test_pattern_matches_verb(self):
        """Test pattern matching with verbs."""
        pattern = ActionPattern(
            name="test_pattern", action_type=ActionType.MOVEMENT, verbs={"go", "walk"}
        )

        assert pattern.matches("go north")
        assert pattern.matches("Go NORTH")  # Case insensitive
        assert pattern.matches("I want to walk")
        assert not pattern.matches("run fast")

    def test_pattern_matches_synonym(self):
        """Test pattern matching with synonyms."""
        pattern = ActionPattern(
            name="test_pattern",
            action_type=ActionType.MOVEMENT,
            synonyms={"travel", "journey"},
        )

        assert pattern.matches("travel to town")
        assert pattern.matches("start a journey")
        assert not pattern.matches("take the item")

    def test_pattern_matches_regex(self):
        """Test pattern matching with regex patterns."""
        pattern = ActionPattern(
            name="test_pattern",
            action_type=ActionType.MOVEMENT,
            regex_patterns=[
                re.compile(r"\bgo\s+to\s+\w+\b"),
                re.compile(r"\b(north|south|east|west)\b"),
            ],
        )

        assert pattern.matches("go to town")
        assert pattern.matches("head north")
        assert pattern.matches("south")
        assert not pattern.matches("take the sword")

    def test_pattern_confidence_calculation(self):
        """Test confidence score calculation."""
        pattern = ActionPattern(
            name="test_pattern",
            action_type=ActionType.MOVEMENT,
            verbs={"go"},
            context_modifiers={"in_location": 1.5},
        )

        # Basic match
        confidence = pattern.get_confidence("go north")
        assert confidence > 0

        # Exact verb match at start boosts confidence
        confidence_start = pattern.get_confidence("go to town")
        confidence_middle = pattern.get_confidence("I want to go")
        assert confidence_start > confidence_middle

        # Context modifiers
        context = {"in_location": True}
        confidence_with_context = pattern.get_confidence("go north", context)
        confidence_without_context = pattern.get_confidence("go north")
        assert confidence_with_context > confidence_without_context

    def test_pattern_no_match_confidence(self):
        """Test confidence is 0 when pattern doesn't match."""
        pattern = ActionPattern(
            name="test_pattern", action_type=ActionType.MOVEMENT, verbs={"go"}
        )

        confidence = pattern.get_confidence("take the sword")
        assert confidence == 0.0


class TestActionPatternManager:
    """Test the ActionPatternManager class."""

    def test_manager_initialization(self):
        """Test manager initialization with default patterns."""
        manager = ActionPatternManager()

        # Should have patterns for all action types
        assert len(manager.patterns) > 0

        # Check that all action types have patterns
        action_types_with_patterns = {p.action_type for p in manager.patterns.values()}
        expected_types = {
            ActionType.MOVEMENT,
            ActionType.OBJECT_INTERACTION,
            ActionType.QUEST,
            ActionType.CONVERSATION,
            ActionType.QUERY,
            ActionType.SYSTEM,
            ActionType.PHYSICAL,
            ActionType.OBSERVATION,
        }
        assert expected_types.issubset(action_types_with_patterns)

    def test_pattern_registration(self):
        """Test registering new patterns."""
        manager = ActionPatternManager()
        initial_count = len(manager.patterns)

        pattern = ActionPattern(
            name="custom_pattern", action_type=ActionType.MOVEMENT, verbs={"teleport"}
        )

        manager.register_pattern(pattern)
        assert len(manager.patterns) == initial_count + 1
        assert "custom_pattern" in manager.patterns

    def test_pattern_unregistration(self):
        """Test unregistering patterns."""
        manager = ActionPatternManager()

        pattern = ActionPattern(
            name="temp_pattern", action_type=ActionType.MOVEMENT, verbs={"temp"}
        )

        manager.register_pattern(pattern)
        assert "temp_pattern" in manager.patterns

        result = manager.unregister_pattern("temp_pattern")
        assert result is True
        assert "temp_pattern" not in manager.patterns

        # Try to unregister non-existent pattern
        result = manager.unregister_pattern("non_existent")
        assert result is False

    def test_action_classification(self):
        """Test classifying user actions."""
        manager = ActionPatternManager()

        # Test movement classification
        action_type, confidence = manager.classify_action("go north")
        assert action_type == ActionType.MOVEMENT
        assert confidence > 0

        # Test object interaction
        action_type, confidence = manager.classify_action("take the sword")
        assert action_type == ActionType.OBJECT_INTERACTION
        assert confidence > 0

        # Test conversation
        action_type, confidence = manager.classify_action("talk to the guard")
        assert action_type == ActionType.CONVERSATION
        assert confidence > 0

        # Test query
        action_type, confidence = manager.classify_action("what is this?")
        assert action_type == ActionType.QUERY
        assert confidence > 0

        # Test unknown action
        action_type, confidence = manager.classify_action("qwerty zxcvbn asdfgh")
        assert action_type == ActionType.UNKNOWN
        assert confidence == 0

    def test_get_patterns_by_type(self):
        """Test getting patterns by action type."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        assert len(movement_patterns) > 0
        assert all(p.action_type == ActionType.MOVEMENT for p in movement_patterns)

        conversation_patterns = manager.get_patterns_by_type(ActionType.CONVERSATION)
        assert len(conversation_patterns) > 0
        assert all(
            p.action_type == ActionType.CONVERSATION for p in conversation_patterns
        )

    def test_add_verb_to_pattern(self):
        """Test adding verbs to existing patterns."""
        manager = ActionPatternManager()

        # Find a movement pattern
        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name

        result = manager.add_verb_to_pattern(pattern_name, "teleport")
        assert result is True
        assert "teleport" in manager.patterns[pattern_name].verbs

        # Test with non-existent pattern
        result = manager.add_verb_to_pattern("non_existent", "verb")
        assert result is False

    def test_add_synonym_to_pattern(self):
        """Test adding synonyms to existing patterns."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name

        result = manager.add_synonym_to_pattern(pattern_name, "transport")
        assert result is True
        assert "transport" in manager.patterns[pattern_name].synonyms

    def test_add_regex_to_pattern(self):
        """Test adding regex patterns to existing patterns."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name
        initial_regex_count = len(manager.patterns[pattern_name].regex_patterns)

        result = manager.add_regex_to_pattern(pattern_name, r"\bteleport\s+to\s+\w+\b")
        assert result is True
        assert (
            len(manager.patterns[pattern_name].regex_patterns)
            == initial_regex_count + 1
        )

        # Test with invalid regex
        result = manager.add_regex_to_pattern(pattern_name, r"[invalid regex")
        assert result is False

    def test_update_pattern_priority(self):
        """Test updating pattern priorities."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name

        result = manager.update_pattern_priority(pattern_name, 99)
        assert result is True
        assert manager.patterns[pattern_name].priority == 99

    def test_set_context_modifier(self):
        """Test setting context modifiers."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name

        result = manager.set_context_modifier(pattern_name, "test_context", 1.8)
        assert result is True
        assert manager.patterns[pattern_name].context_modifiers["test_context"] == 1.8

    def test_validate_patterns(self):
        """Test pattern validation."""
        manager = ActionPatternManager()

        # Add a pattern with no matching criteria
        empty_pattern = ActionPattern(
            name="empty_pattern", action_type=ActionType.MOVEMENT
        )
        manager.register_pattern(empty_pattern)

        issues = manager.validate_patterns()
        assert len(issues) > 0
        assert any("empty_pattern" in issue for issue in issues)

    def test_get_all_verbs(self):
        """Test getting all registered verbs."""
        manager = ActionPatternManager()

        all_verbs = manager.get_all_verbs()
        assert len(all_verbs) > 0
        assert "go" in all_verbs
        assert "take" in all_verbs
        assert "talk" in all_verbs

    def test_get_pattern_info(self):
        """Test getting pattern information."""
        manager = ActionPatternManager()

        movement_patterns = manager.get_patterns_by_type(ActionType.MOVEMENT)
        pattern_name = movement_patterns[0].name

        info = manager.get_pattern_info(pattern_name)
        assert info is not None
        assert info["name"] == pattern_name
        assert info["action_type"] == ActionType.MOVEMENT.value
        assert "verbs" in info
        assert "synonyms" in info
        assert "priority" in info

        # Test with non-existent pattern
        info = manager.get_pattern_info("non_existent")
        assert info is None

    def test_context_aware_classification(self):
        """Test classification with context awareness."""
        manager = ActionPatternManager()

        # Test how context affects classification
        text = "go north"

        # Without context
        action_type1, confidence1 = manager.classify_action(text)

        # With relevant context
        context = {"in_location": True, "has_exits": True}
        action_type2, confidence2 = manager.classify_action(text, context)

        assert action_type1 == action_type2 == ActionType.MOVEMENT
        # Context should boost confidence
        assert confidence2 >= confidence1

    def test_priority_handling(self):
        """Test that patterns with higher priority are preferred."""
        manager = ActionPatternManager()

        # Create two patterns that could match the same text
        high_priority = ActionPattern(
            name="high_priority",
            action_type=ActionType.MOVEMENT,
            verbs={"test"},
            priority=1,
        )

        low_priority = ActionPattern(
            name="low_priority",
            action_type=ActionType.SYSTEM,
            verbs={"test"},
            priority=10,
        )

        manager.register_pattern(high_priority)
        manager.register_pattern(low_priority)

        # The pattern with higher priority (lower number) should win
        action_type, confidence = manager.classify_action("test command")
        assert action_type == ActionType.MOVEMENT
