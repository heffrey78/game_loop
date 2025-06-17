"""Tests for rule loader."""

import json
import tempfile
from pathlib import Path

import pytest

from game_loop.core.rules.rule_loader import RuleLoader
from game_loop.core.rules.rule_models import (
    ActionType,
    ConditionOperator,
    RulePriority,
)


class TestRuleLoader:
    """Test RuleLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a rule loader for testing."""
        return RuleLoader()

    def test_load_from_dict_single_rule(self, loader):
        """Test loading a single rule from dictionary."""
        rule_data = {
            "name": "test_rule",
            "description": "A test rule",
            "priority": "high",
            "conditions": [
                {"field": "player_state.health", "operator": "less_than", "value": 20}
            ],
            "actions": [
                {"type": "send_message", "parameters": {"message": "Low health!"}}
            ],
            "tags": ["health", "warning"],
        }

        rules = loader.load_from_dict(rule_data)

        assert len(rules) == 1
        rule = rules[0]
        assert rule.name == "test_rule"
        assert rule.description == "A test rule"
        assert rule.priority == RulePriority.HIGH
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
        assert "health" in rule.tags

    def test_load_from_dict_multiple_rules(self, loader):
        """Test loading multiple rules from dictionary."""
        rules_data = {
            "rules": [
                {
                    "name": "rule1",
                    "conditions": [
                        {"field": "test1", "operator": "equals", "value": "test"}
                    ],
                    "actions": [{"type": "send_message", "parameters": {}}],
                },
                {
                    "name": "rule2",
                    "conditions": [
                        {"field": "test2", "operator": "equals", "value": "test"}
                    ],
                    "actions": [{"type": "send_message", "parameters": {}}],
                },
            ]
        }

        rules = loader.load_from_dict(rules_data)

        assert len(rules) == 2
        assert rules[0].name == "rule1"
        assert rules[1].name == "rule2"

    def test_load_from_yaml_string(self, loader):
        """Test loading rules from YAML string."""
        yaml_content = """
        rules:
          - name: yaml_rule
            description: Rule from YAML
            priority: medium
            conditions:
              - field: player_state.level
                operator: greater_than
                value: 5
            actions:
              - type: grant_reward
                parameters:
                  experience: 100
            tags: [level, reward]
        """

        rules = loader.load_from_string(yaml_content, "yaml")

        assert len(rules) == 1
        rule = rules[0]
        assert rule.name == "yaml_rule"
        assert rule.description == "Rule from YAML"
        assert rule.priority == RulePriority.MEDIUM
        assert "level" in rule.tags

    def test_load_from_json_string(self, loader):
        """Test loading rules from JSON string."""
        json_content = """
        {
            "rules": [
                {
                    "name": "json_rule",
                    "description": "Rule from JSON",
                    "priority": "low",
                    "conditions": [
                        {
                            "field": "inventory.count",
                            "operator": "greater_equal",
                            "value": 10
                        }
                    ],
                    "actions": [
                        {
                            "type": "block_action",
                            "parameters": {"reason": "Inventory full"}
                        }
                    ]
                }
            ]
        }
        """

        rules = loader.load_from_string(json_content, "json")

        assert len(rules) == 1
        rule = rules[0]
        assert rule.name == "json_rule"
        assert rule.priority == RulePriority.LOW
        assert len(rule.conditions) == 1
        assert rule.conditions[0].operator == ConditionOperator.GREATER_EQUAL

    def test_load_from_file_yaml(self, loader):
        """Test loading rules from YAML file."""
        yaml_content = """
        rules:
          - name: file_rule
            conditions:
              - field: test
                operator: equals
                value: test
            actions:
              - type: send_message
                parameters:
                  message: Test message
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                rules = loader.load_from_file(f.name)
                assert len(rules) == 1
                assert rules[0].name == "file_rule"
            finally:
                Path(f.name).unlink()

    def test_load_from_file_json(self, loader):
        """Test loading rules from JSON file."""
        json_content = {
            "rules": [
                {
                    "name": "json_file_rule",
                    "conditions": [
                        {"field": "test", "operator": "equals", "value": "test"}
                    ],
                    "actions": [{"type": "send_message", "parameters": {}}],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            f.flush()

            try:
                rules = loader.load_from_file(f.name)
                assert len(rules) == 1
                assert rules[0].name == "json_file_rule"
            finally:
                Path(f.name).unlink()

    def test_load_from_nonexistent_file(self, loader):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("nonexistent_file.yaml")

    def test_load_from_unsupported_format(self, loader):
        """Test loading from unsupported format raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            f.flush()

            try:
                with pytest.raises(ValueError, match="Unsupported file format"):
                    loader.load_from_file(f.name)
            finally:
                Path(f.name).unlink()


class TestRuleConditionParsing:
    """Test parsing of rule conditions."""

    @pytest.fixture
    def loader(self):
        """Create a rule loader for testing."""
        return RuleLoader()

    def test_parse_condition_operators(self, loader):
        """Test parsing different condition operators."""
        operators_test_cases = [
            ("equals", ConditionOperator.EQUALS),
            ("eq", ConditionOperator.EQUALS),
            ("==", ConditionOperator.EQUALS),
            ("not_equals", ConditionOperator.NOT_EQUALS),
            ("ne", ConditionOperator.NOT_EQUALS),
            ("!=", ConditionOperator.NOT_EQUALS),
            ("greater_than", ConditionOperator.GREATER_THAN),
            ("gt", ConditionOperator.GREATER_THAN),
            (">", ConditionOperator.GREATER_THAN),
            ("less_than", ConditionOperator.LESS_THAN),
            ("lt", ConditionOperator.LESS_THAN),
            ("<", ConditionOperator.LESS_THAN),
            ("contains", ConditionOperator.CONTAINS),
            ("regex", ConditionOperator.REGEX_MATCH),
        ]

        for operator_str, expected_operator in operators_test_cases:
            condition_data = {
                "field": "test.field",
                "operator": operator_str,
                "value": "test_value",
            }

            condition = loader._parse_condition(condition_data)
            assert condition.operator == expected_operator

    def test_parse_condition_with_description(self, loader):
        """Test parsing condition with description."""
        condition_data = {
            "field": "player_state.health",
            "operator": "less_than",
            "value": 20,
            "description": "Health is low",
        }

        condition = loader._parse_condition(condition_data)
        assert condition.field_path == "player_state.health"
        assert condition.operator == ConditionOperator.LESS_THAN
        assert condition.expected_value == 20
        assert condition.description == "Health is low"

    def test_parse_condition_alternative_field_names(self, loader):
        """Test parsing condition with alternative field names."""
        # Test 'field_path' instead of 'field'
        condition_data = {
            "field_path": "player.stats.strength",
            "operator": "greater_than",
            "expected_value": 15,
        }

        condition = loader._parse_condition(condition_data)
        assert condition.field_path == "player.stats.strength"
        assert condition.expected_value == 15


class TestRuleActionParsing:
    """Test parsing of rule actions."""

    @pytest.fixture
    def loader(self):
        """Create a rule loader for testing."""
        return RuleLoader()

    def test_parse_action_types(self, loader):
        """Test parsing different action types."""
        action_types_test_cases = [
            ("modify_state", ActionType.MODIFY_STATE),
            ("modify", ActionType.MODIFY_STATE),
            ("set", ActionType.MODIFY_STATE),
            ("send_message", ActionType.SEND_MESSAGE),
            ("message", ActionType.SEND_MESSAGE),
            ("trigger_event", ActionType.TRIGGER_EVENT),
            ("trigger", ActionType.TRIGGER_EVENT),
            ("block_action", ActionType.BLOCK_ACTION),
            ("block", ActionType.BLOCK_ACTION),
            ("grant_reward", ActionType.GRANT_REWARD),
            ("reward", ActionType.GRANT_REWARD),
            ("spawn_entity", ActionType.SPAWN_ENTITY),
            ("spawn", ActionType.SPAWN_ENTITY),
            ("custom", ActionType.CUSTOM),
        ]

        for action_type_str, expected_type in action_types_test_cases:
            action_data = {"type": action_type_str, "parameters": {}}

            action = loader._parse_action(action_data)
            assert action.action_type == expected_type

    def test_parse_action_with_target(self, loader):
        """Test parsing action with target path."""
        action_data = {
            "type": "modify_state",
            "target": "player_state.health",
            "parameters": {"value": 100},
        }

        action = loader._parse_action(action_data)
        assert action.action_type == ActionType.MODIFY_STATE
        assert action.target_path == "player_state.health"
        assert action.parameters["value"] == 100

    def test_parse_action_alternative_parameter_names(self, loader):
        """Test parsing action with alternative parameter names."""
        # Test 'params' instead of 'parameters'
        action_data = {
            "action_type": "send_message",
            "params": {"message": "Test message", "style": "warning"},
        }

        action = loader._parse_action(action_data)
        assert action.action_type == ActionType.SEND_MESSAGE
        assert action.parameters["message"] == "Test message"
        assert action.parameters["style"] == "warning"


class TestRuleParsing:
    """Test parsing of complete rules."""

    @pytest.fixture
    def loader(self):
        """Create a rule loader for testing."""
        return RuleLoader()

    def test_parse_rule_with_all_fields(self, loader):
        """Test parsing rule with all possible fields."""
        rule_data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "comprehensive_rule",
            "description": "A comprehensive test rule",
            "priority": "critical",
            "enabled": False,
            "conditions": [
                {
                    "field": "player_state.health",
                    "operator": "less_than",
                    "value": 20,
                    "description": "Low health condition",
                }
            ],
            "actions": [
                {
                    "type": "send_message",
                    "parameters": {"message": "Critical health!"},
                    "description": "Send warning message",
                }
            ],
            "tags": ["health", "critical", "warning"],
            "created_at": "2023-01-01T00:00:00Z",
            "modified_at": "2023-01-02T00:00:00Z",
        }

        rule = loader._parse_single_rule(rule_data)

        assert str(rule.id) == "550e8400-e29b-41d4-a716-446655440000"
        assert rule.name == "comprehensive_rule"
        assert rule.description == "A comprehensive test rule"
        assert rule.priority == RulePriority.CRITICAL
        assert rule.enabled is False
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
        assert set(rule.tags) == {"health", "critical", "warning"}
        assert rule.created_at == "2023-01-01T00:00:00Z"
        assert rule.modified_at == "2023-01-02T00:00:00Z"

    def test_parse_rule_minimal_fields(self, loader):
        """Test parsing rule with minimal required fields."""
        rule_data = {"name": "minimal_rule"}

        rule = loader._parse_single_rule(rule_data)

        assert rule.name == "minimal_rule"
        assert rule.priority == RulePriority.MEDIUM  # Default
        assert rule.enabled is True  # Default
        assert len(rule.conditions) == 0
        assert len(rule.actions) == 0
        assert len(rule.tags) == 0

    def test_parse_rule_invalid_priority(self, loader):
        """Test parsing rule with invalid priority defaults to medium."""
        rule_data = {"name": "invalid_priority_rule", "priority": "invalid_priority"}

        rule = loader._parse_single_rule(rule_data)
        assert rule.priority == RulePriority.MEDIUM

    def test_parse_rule_single_condition_not_list(self, loader):
        """Test parsing rule where conditions is not a list."""
        rule_data = {
            "name": "single_condition_rule",
            "conditions": {"field": "test", "operator": "equals", "value": "test"},
        }

        rule = loader._parse_single_rule(rule_data)
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field_path == "test"

    def test_parse_rule_single_action_not_list(self, loader):
        """Test parsing rule where actions is not a list."""
        rule_data = {
            "name": "single_action_rule",
            "actions": {"type": "send_message", "parameters": {"message": "test"}},
        }

        rule = loader._parse_single_rule(rule_data)
        assert len(rule.actions) == 1
        assert rule.actions[0].action_type == ActionType.SEND_MESSAGE


class TestSampleRulesCreation:
    """Test sample rules file creation."""

    @pytest.fixture
    def loader(self):
        """Create a rule loader for testing."""
        return RuleLoader()

    def test_create_sample_yaml_file(self, loader):
        """Test creating sample YAML rules file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            try:
                loader.create_sample_rules_file(f.name)

                # Verify file was created and can be loaded
                rules = loader.load_from_file(f.name)
                assert len(rules) > 0

                # Check that sample rules have expected structure
                health_rule = next(
                    (r for r in rules if r.name == "low_health_warning"), None
                )
                assert health_rule is not None
                assert health_rule.priority == RulePriority.HIGH
                assert len(health_rule.conditions) > 0
                assert len(health_rule.actions) > 0

            finally:
                Path(f.name).unlink()

    def test_create_sample_json_file(self, loader):
        """Test creating sample JSON rules file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            try:
                loader.create_sample_rules_file(f.name)

                # Verify file was created and can be loaded
                rules = loader.load_from_file(f.name)
                assert len(rules) > 0

            finally:
                Path(f.name).unlink()
