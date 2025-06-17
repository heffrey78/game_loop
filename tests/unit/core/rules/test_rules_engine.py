"""Tests for rules engine."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from game_loop.core.rules.rule_models import (
    ActionType,
    ConditionOperator,
    Rule,
    RuleAction,
    RuleCondition,
    RuleEngineConfig,
    RuleEvaluationContext,
    RulePriority,
)
from game_loop.core.rules.rules_engine import RulesEngine


class TestRulesEngine:
    """Test RulesEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a rules engine for testing."""
        return RulesEngine()

    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule for testing."""
        return Rule(
            name="test_rule",
            description="A test rule",
            priority=RulePriority.MEDIUM,
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=50,
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Health is low!"},
                )
            ],
            tags=["health", "warning"],
        )

    def test_engine_initialization(self, engine):
        """Test rules engine initialization."""
        assert engine.config is not None
        assert engine.evaluator is not None
        assert engine.loader is not None
        assert len(engine._rules) == 0
        assert len(engine._rules_by_name) == 0

    def test_add_single_rule(self, engine, sample_rule):
        """Test adding a single rule."""
        result = engine.add_rule(sample_rule)

        assert result is True
        assert len(engine._rules) == 1
        assert sample_rule.id in engine._rules
        assert sample_rule.name in engine._rules_by_name
        assert sample_rule in engine._rules_by_priority[RulePriority.MEDIUM]
        assert sample_rule in engine._rules_by_tag["health"]
        assert sample_rule in engine._rules_by_tag["warning"]

    def test_add_duplicate_rule_name(self, engine, sample_rule):
        """Test adding rule with duplicate name fails."""
        # Add rule first time
        result1 = engine.add_rule(sample_rule)
        assert result1 is True

        # Create different rule with same name
        duplicate_rule = Rule(name="test_rule", conditions=[], actions=[])  # Same name

        result2 = engine.add_rule(duplicate_rule)
        assert result2 is False
        assert len(engine._rules) == 1  # Still only one rule

    def test_add_multiple_rules(self, engine):
        """Test adding multiple rules."""
        rules = []
        for i in range(3):
            rule = Rule(name=f"rule_{i}", conditions=[], actions=[], tags=[f"tag_{i}"])
            rules.append(rule)

        added_count = engine.add_rules(rules)

        assert added_count == 3
        assert len(engine._rules) == 3
        assert len(engine._rules_by_name) == 3

    def test_remove_rule(self, engine, sample_rule):
        """Test removing a rule."""
        # Add rule first
        engine.add_rule(sample_rule)
        assert len(engine._rules) == 1

        # Remove rule
        result = engine.remove_rule(sample_rule.id)

        assert result is True
        assert len(engine._rules) == 0
        assert sample_rule.name not in engine._rules_by_name
        assert sample_rule not in engine._rules_by_priority[RulePriority.MEDIUM]
        assert "health" not in engine._rules_by_tag
        assert "warning" not in engine._rules_by_tag

    def test_remove_nonexistent_rule(self, engine):
        """Test removing nonexistent rule returns False."""
        nonexistent_id = uuid4()
        result = engine.remove_rule(nonexistent_id)
        assert result is False

    def test_get_rule_by_id(self, engine, sample_rule):
        """Test getting rule by ID."""
        engine.add_rule(sample_rule)

        retrieved_rule = engine.get_rule(sample_rule.id)
        assert retrieved_rule == sample_rule

        nonexistent_rule = engine.get_rule(uuid4())
        assert nonexistent_rule is None

    def test_get_rule_by_name(self, engine, sample_rule):
        """Test getting rule by name."""
        engine.add_rule(sample_rule)

        retrieved_rule = engine.get_rule_by_name("test_rule")
        assert retrieved_rule == sample_rule

        nonexistent_rule = engine.get_rule_by_name("nonexistent_rule")
        assert nonexistent_rule is None

    def test_get_rules_by_tag(self, engine, sample_rule):
        """Test getting rules by tag."""
        engine.add_rule(sample_rule)

        health_rules = engine.get_rules_by_tag("health")
        assert len(health_rules) == 1
        assert sample_rule in health_rules

        warning_rules = engine.get_rules_by_tag("warning")
        assert len(warning_rules) == 1
        assert sample_rule in warning_rules

        nonexistent_rules = engine.get_rules_by_tag("nonexistent")
        assert len(nonexistent_rules) == 0

    def test_get_rules_by_priority(self, engine):
        """Test getting rules by priority."""
        high_rule = Rule(name="high_rule", priority=RulePriority.HIGH)
        medium_rule = Rule(name="medium_rule", priority=RulePriority.MEDIUM)
        low_rule = Rule(name="low_rule", priority=RulePriority.LOW)

        engine.add_rules([high_rule, medium_rule, low_rule])

        high_rules = engine.get_rules_by_priority(RulePriority.HIGH)
        assert len(high_rules) == 1
        assert high_rule in high_rules

        medium_rules = engine.get_rules_by_priority(RulePriority.MEDIUM)
        assert len(medium_rules) == 1
        assert medium_rule in medium_rules

    def test_evaluate_rules_no_rules(self, engine):
        """Test evaluating with no rules returns empty list."""
        context = RuleEvaluationContext()
        results = engine.evaluate_rules(context)
        assert len(results) == 0

    def test_evaluate_rules_with_matching_rule(self, engine, sample_rule):
        """Test evaluating rules with matching conditions."""
        engine.add_rule(sample_rule)

        context = RuleEvaluationContext(
            player_state={"health": 30}  # Less than 50, should trigger rule
        )

        results = engine.evaluate_rules(context)

        assert len(results) == 1
        result = results[0]
        assert result.rule_id == sample_rule.id
        assert result.triggered is True
        assert result.conditions_met is True

    def test_evaluate_rules_with_non_matching_rule(self, engine, sample_rule):
        """Test evaluating rules with non-matching conditions."""
        engine.add_rule(sample_rule)

        context = RuleEvaluationContext(
            player_state={"health": 80}  # Greater than 50, should not trigger
        )

        results = engine.evaluate_rules(context)

        assert len(results) == 1
        result = results[0]
        assert result.rule_id == sample_rule.id
        assert result.triggered is False
        assert result.conditions_met is False

    def test_evaluate_rules_with_disabled_rule(self, engine, sample_rule):
        """Test evaluating disabled rules."""
        sample_rule.enabled = False
        engine.add_rule(sample_rule)

        context = RuleEvaluationContext(player_state={"health": 30})

        results = engine.evaluate_rules(context)

        assert len(results) == 1
        result = results[0]
        assert result.triggered is False

    def test_evaluate_rules_with_tag_filter(self, engine):
        """Test evaluating rules with tag filter."""
        health_rule = Rule(
            name="health_rule",
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=50,
                )
            ],
            actions=[RuleAction(action_type=ActionType.SEND_MESSAGE, parameters={})],
            tags=["health"],
        )

        combat_rule = Rule(
            name="combat_rule",
            conditions=[
                RuleCondition(
                    field_path="current_action",
                    operator=ConditionOperator.EQUALS,
                    expected_value="attack",
                )
            ],
            actions=[RuleAction(action_type=ActionType.SEND_MESSAGE, parameters={})],
            tags=["combat"],
        )

        engine.add_rules([health_rule, combat_rule])

        context = RuleEvaluationContext(
            player_state={"health": 30}, current_action="attack"
        )

        # Evaluate only health rules
        health_results = engine.evaluate_rules(context, tags=["health"])
        assert len(health_results) == 1
        assert health_results[0].rule_name == "health_rule"

        # Evaluate only combat rules
        combat_results = engine.evaluate_rules(context, tags=["combat"])
        assert len(combat_results) == 1
        assert combat_results[0].rule_name == "combat_rule"

    def test_evaluate_rules_priority_ordering(self, engine):
        """Test that rules are evaluated in priority order."""
        critical_rule = Rule(name="critical", priority=RulePriority.CRITICAL)
        high_rule = Rule(name="high", priority=RulePriority.HIGH)
        medium_rule = Rule(name="medium", priority=RulePriority.MEDIUM)
        low_rule = Rule(name="low", priority=RulePriority.LOW)

        # Add in random order
        engine.add_rules([medium_rule, critical_rule, low_rule, high_rule])

        context = RuleEvaluationContext()
        results = engine.evaluate_rules(context)

        # Should be evaluated in priority order (lower value = higher priority)
        assert len(results) == 4
        assert results[0].rule_name == "critical"
        assert results[1].rule_name == "high"
        assert results[2].rule_name == "medium"
        assert results[3].rule_name == "low"

    def test_load_rules_from_dict(self, engine):
        """Test loading rules from dictionary."""
        rules_data = {
            "rules": [
                {
                    "name": "dict_rule_1",
                    "conditions": [
                        {"field": "test", "operator": "equals", "value": "test"}
                    ],
                    "actions": [{"type": "send_message", "parameters": {}}],
                },
                {
                    "name": "dict_rule_2",
                    "conditions": [
                        {"field": "test2", "operator": "equals", "value": "test2"}
                    ],
                    "actions": [{"type": "send_message", "parameters": {}}],
                },
            ]
        }

        loaded_count = engine.load_rules_from_dict(rules_data)

        assert loaded_count == 2
        assert len(engine._rules) == 2
        assert "dict_rule_1" in engine._rules_by_name
        assert "dict_rule_2" in engine._rules_by_name

    def test_clear_rules(self, engine, sample_rule):
        """Test clearing all rules."""
        engine.add_rule(sample_rule)
        assert len(engine._rules) == 1

        engine.clear_rules()

        assert len(engine._rules) == 0
        assert len(engine._rules_by_name) == 0
        assert len(engine._rules_by_tag) == 0
        for priority_list in engine._rules_by_priority.values():
            assert len(priority_list) == 0

    def test_get_statistics(self, engine):
        """Test getting engine statistics."""
        # Add some test rules
        rules = []
        for i in range(3):
            rule = Rule(
                name=f"rule_{i}",
                priority=RulePriority.HIGH if i == 0 else RulePriority.MEDIUM,
                enabled=i < 2,  # First two enabled, last one disabled
                tags=[f"tag_{i}"],
            )
            rules.append(rule)

        engine.add_rules(rules)

        # Perform some evaluations to get timing stats
        context = RuleEvaluationContext()
        engine.evaluate_rules(context)
        engine.evaluate_rules(context)

        stats = engine.get_statistics()

        assert stats["total_rules"] == 3
        assert stats["enabled_rules"] == 2
        assert stats["disabled_rules"] == 1
        assert stats["priority_distribution"]["HIGH"] == 1
        assert stats["priority_distribution"]["MEDIUM"] == 2
        assert stats["evaluation_count"] == 2
        assert stats["avg_evaluation_time_ms"] >= 0
        assert stats["total_evaluation_time_ms"] >= 0
        assert "tag_0" in stats["tags"]


class TestRulesEngineConfig:
    """Test rules engine with custom configuration."""

    def test_engine_with_custom_config(self):
        """Test creating engine with custom configuration."""
        config = RuleEngineConfig(
            max_rules_per_evaluation=10,
            enable_conflict_detection=False,
            rule_timeout_ms=1000.0,
        )

        engine = RulesEngine(config)

        assert engine.config.max_rules_per_evaluation == 10
        assert engine.config.enable_conflict_detection is False
        assert engine.config.rule_timeout_ms == 1000.0

    def test_max_rules_per_evaluation_limit(self):
        """Test that max rules per evaluation is respected."""
        config = RuleEngineConfig(max_rules_per_evaluation=2)
        engine = RulesEngine(config)

        # Add 5 rules
        for i in range(5):
            rule = Rule(name=f"rule_{i}")
            engine.add_rule(rule)

        context = RuleEvaluationContext()
        results = engine.evaluate_rules(context)

        # Should only evaluate first 2 rules (due to limit)
        assert len(results) == 2


class TestRulesEngineFileOperations:
    """Test rules engine file operations."""

    @pytest.fixture
    def engine(self):
        """Create a rules engine for testing."""
        return RulesEngine()

    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule for testing."""
        return Rule(
            name="test_rule",
            description="A test rule",
            priority=RulePriority.MEDIUM,
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=50,
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Health is low!"},
                )
            ],
            tags=["health", "warning"],
        )

    def test_load_rules_from_yaml_file(self, engine):
        """Test loading rules from YAML file."""
        yaml_content = """
        rules:
          - name: yaml_rule
            conditions:
              - field: test
                operator: equals
                value: test
            actions:
              - type: send_message
                parameters: {}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                loaded_count = engine.load_rules_from_file(f.name)
                assert loaded_count == 1
                assert "yaml_rule" in engine._rules_by_name
            finally:
                Path(f.name).unlink()

    def test_export_rules_yaml(self, engine, sample_rule):
        """Test exporting rules to YAML file."""
        engine.add_rule(sample_rule)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            try:
                result = engine.export_rules(f.name, "yaml")
                assert result is True

                # Verify file was created and has content
                exported_path = Path(f.name)
                assert exported_path.exists()
                content = exported_path.read_text()
                assert "test_rule" in content
            finally:
                Path(f.name).unlink()

    def test_export_rules_json(self, engine, sample_rule):
        """Test exporting rules to JSON file."""
        engine.add_rule(sample_rule)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            try:
                result = engine.export_rules(f.name, "json")
                assert result is True

                # Verify file was created and has content
                exported_path = Path(f.name)
                assert exported_path.exists()
                content = exported_path.read_text()
                assert "test_rule" in content
            finally:
                Path(f.name).unlink()


class TestConflictDetection:
    """Test rule conflict detection and resolution."""

    def test_conflict_detection_disabled(self):
        """Test that conflict detection can be disabled."""
        config = RuleEngineConfig(enable_conflict_detection=False)
        engine = RulesEngine(config)

        # Add two potentially conflicting rules
        rule1 = Rule(
            name="rule1",
            conditions=[
                RuleCondition(
                    field_path="custom_data.test",
                    operator=ConditionOperator.EQUALS,
                    expected_value="trigger",
                )
            ],
            actions=[RuleAction(action_type=ActionType.MODIFY_STATE, parameters={})],
        )

        rule2 = Rule(
            name="rule2",
            conditions=[
                RuleCondition(
                    field_path="custom_data.test",
                    operator=ConditionOperator.EQUALS,
                    expected_value="trigger",
                )
            ],
            actions=[RuleAction(action_type=ActionType.MODIFY_STATE, parameters={})],
        )

        engine.add_rules([rule1, rule2])

        context = RuleEvaluationContext(custom_data={"test": "trigger"})
        results = engine.evaluate_rules(context)

        # Both rules should execute since conflict detection is disabled
        triggered_results = [r for r in results if r.triggered]
        assert len(triggered_results) == 2
