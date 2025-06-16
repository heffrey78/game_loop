"""Tests for rule evaluator."""

import pytest

from game_loop.core.rules.rule_evaluator import RuleEvaluator
from game_loop.core.rules.rule_models import (
    Rule,
    RuleAction,
    RuleCondition,
    RuleEvaluationContext,
    ActionType,
    ConditionOperator,
    RulePriority,
)


class TestRuleEvaluator:
    """Test RuleEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a rule evaluator for testing."""
        return RuleEvaluator()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample evaluation context."""
        return RuleEvaluationContext(
            player_state={
                "health": 50,
                "level": 5,
                "inventory_count": 8,
                "stats": {"strength": 15, "wisdom": 12}
            },
            current_action="take",
            action_parameters={"item": "sword", "target": "chest"},
            location_data={"danger_level": 3}
        )
    
    def test_evaluate_rule_disabled(self, evaluator, sample_context):
        """Test evaluating a disabled rule."""
        rule = Rule(
            name="disabled_rule",
            enabled=False,
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=100
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Test"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, sample_context)
        
        assert result.rule_id == rule.id
        assert result.rule_name == "disabled_rule"
        assert result.triggered is False
        assert result.conditions_met is False
        assert len(result.actions_executed) == 0
        assert result.execution_time_ms is not None
    
    def test_evaluate_rule_conditions_not_met(self, evaluator, sample_context):
        """Test evaluating a rule where conditions are not met."""
        rule = Rule(
            name="conditions_not_met",
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.GREATER_THAN,
                    expected_value=100  # Health is 50, so this won't match
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Test"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, sample_context)
        
        assert result.triggered is False
        assert result.conditions_met is False
        assert len(result.actions_executed) == 0
    
    def test_evaluate_rule_conditions_met(self, evaluator, sample_context):
        """Test evaluating a rule where conditions are met."""
        rule = Rule(
            name="conditions_met",
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=60  # Health is 50, so this will match
                ),
                RuleCondition(
                    field_path="current_action",
                    operator=ConditionOperator.EQUALS,
                    expected_value="take"  # Current action matches
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Test"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, sample_context)
        
        assert result.triggered is True
        assert result.conditions_met is True
        assert len(result.actions_executed) == 1
    
    def test_evaluate_multiple_conditions_any_fails(self, evaluator, sample_context):
        """Test that if any condition fails, the rule doesn't trigger."""
        rule = Rule(
            name="multiple_conditions",
            conditions=[
                RuleCondition(
                    field_path="player_state.health",
                    operator=ConditionOperator.LESS_THAN,
                    expected_value=60  # This will match
                ),
                RuleCondition(
                    field_path="player_state.level",
                    operator=ConditionOperator.GREATER_THAN,
                    expected_value=10  # This will NOT match (level is 5)
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "Test"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, sample_context)
        
        assert result.triggered is False
        assert result.conditions_met is False


class TestConditionEvaluators:
    """Test individual condition evaluators."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a rule evaluator for testing."""
        return RuleEvaluator()
    
    def test_equals_condition(self, evaluator):
        """Test equals condition evaluation."""
        assert evaluator._evaluate_equals("test", "test") is True
        assert evaluator._evaluate_equals("test", "other") is False
        assert evaluator._evaluate_equals(5, 5) is True
        assert evaluator._evaluate_equals(5, 6) is False
    
    def test_not_equals_condition(self, evaluator):
        """Test not equals condition evaluation."""
        assert evaluator._evaluate_not_equals("test", "other") is True
        assert evaluator._evaluate_not_equals("test", "test") is False
        assert evaluator._evaluate_not_equals(5, 6) is True
        assert evaluator._evaluate_not_equals(5, 5) is False
    
    def test_greater_than_condition(self, evaluator):
        """Test greater than condition evaluation."""
        assert evaluator._evaluate_greater_than(10, 5) is True
        assert evaluator._evaluate_greater_than(5, 10) is False
        assert evaluator._evaluate_greater_than(5, 5) is False
        assert evaluator._evaluate_greater_than("10", "5") is True
        assert evaluator._evaluate_greater_than("invalid", "5") is False
    
    def test_less_than_condition(self, evaluator):
        """Test less than condition evaluation."""
        assert evaluator._evaluate_less_than(5, 10) is True
        assert evaluator._evaluate_less_than(10, 5) is False
        assert evaluator._evaluate_less_than(5, 5) is False
        assert evaluator._evaluate_less_than("5", "10") is True
    
    def test_greater_equal_condition(self, evaluator):
        """Test greater than or equal condition evaluation."""
        assert evaluator._evaluate_greater_equal(10, 5) is True
        assert evaluator._evaluate_greater_equal(5, 5) is True
        assert evaluator._evaluate_greater_equal(5, 10) is False
    
    def test_less_equal_condition(self, evaluator):
        """Test less than or equal condition evaluation."""
        assert evaluator._evaluate_less_equal(5, 10) is True
        assert evaluator._evaluate_less_equal(5, 5) is True
        assert evaluator._evaluate_less_equal(10, 5) is False
    
    def test_contains_condition(self, evaluator):
        """Test contains condition evaluation."""
        assert evaluator._evaluate_contains("hello world", "world") is True
        assert evaluator._evaluate_contains("hello world", "planet") is False
        assert evaluator._evaluate_contains([1, 2, 3], "2") is True
    
    def test_not_contains_condition(self, evaluator):
        """Test not contains condition evaluation."""
        assert evaluator._evaluate_not_contains("hello world", "planet") is True
        assert evaluator._evaluate_not_contains("hello world", "world") is False
    
    def test_in_condition(self, evaluator):
        """Test in condition evaluation."""
        assert evaluator._evaluate_in("apple", ["apple", "banana", "cherry"]) is True
        assert evaluator._evaluate_in("grape", ["apple", "banana", "cherry"]) is False
        assert evaluator._evaluate_in("a", "apple") is True
        assert evaluator._evaluate_in("z", "apple") is False
    
    def test_not_in_condition(self, evaluator):
        """Test not in condition evaluation."""
        assert evaluator._evaluate_not_in("grape", ["apple", "banana", "cherry"]) is True
        assert evaluator._evaluate_not_in("apple", ["apple", "banana", "cherry"]) is False
    
    def test_exists_condition(self, evaluator):
        """Test exists condition evaluation."""
        assert evaluator._evaluate_exists("value", None) is True
        assert evaluator._evaluate_exists(None, None) is False
        assert evaluator._evaluate_exists(0, None) is True
        assert evaluator._evaluate_exists("", None) is True
    
    def test_not_exists_condition(self, evaluator):
        """Test not exists condition evaluation."""
        assert evaluator._evaluate_not_exists(None, None) is True
        assert evaluator._evaluate_not_exists("value", None) is False
        assert evaluator._evaluate_not_exists(0, None) is False
    
    def test_regex_match_condition(self, evaluator):
        """Test regex match condition evaluation."""
        assert evaluator._evaluate_regex_match("hello123", r"hello\d+") is True
        assert evaluator._evaluate_regex_match("hello", r"hello\d+") is False
        assert evaluator._evaluate_regex_match("test@example.com", r".*@.*\.com") is True
        assert evaluator._evaluate_regex_match("invalid_email", r".*@.*\.com") is False
        assert evaluator._evaluate_regex_match("test", r"[invalid") is False  # Invalid regex


class TestActionExecutors:
    """Test action execution (placeholder implementations)."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a rule evaluator for testing."""
        return RuleEvaluator()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample evaluation context."""
        return RuleEvaluationContext()
    
    def test_execute_modify_state_action(self, evaluator, sample_context):
        """Test modify state action execution."""
        action = RuleAction(
            action_type=ActionType.MODIFY_STATE,
            target_path="player_state.health",
            parameters={"value": 100}
        )
        
        # Should return True (placeholder implementation)
        result = evaluator._execute_modify_state(action, sample_context)
        assert result is True
    
    def test_execute_send_message_action(self, evaluator, sample_context):
        """Test send message action execution."""
        action = RuleAction(
            action_type=ActionType.SEND_MESSAGE,
            parameters={"message": "Test message"}
        )
        
        # Should return True (placeholder implementation)
        result = evaluator._execute_send_message(action, sample_context)
        assert result is True
    
    def test_execute_trigger_event_action(self, evaluator, sample_context):
        """Test trigger event action execution."""
        action = RuleAction(
            action_type=ActionType.TRIGGER_EVENT,
            parameters={"event_type": "test_event"}
        )
        
        # Should return True (placeholder implementation)
        result = evaluator._execute_trigger_event(action, sample_context)
        assert result is True
    
    def test_execute_block_action(self, evaluator, sample_context):
        """Test block action execution."""
        action = RuleAction(
            action_type=ActionType.BLOCK_ACTION,
            parameters={"reason": "Action blocked"}
        )
        
        # Should return True (placeholder implementation)
        result = evaluator._execute_block_action(action, sample_context)
        assert result is True
    
    def test_execute_custom_action(self, evaluator, sample_context):
        """Test custom action execution."""
        action = RuleAction(
            action_type=ActionType.CUSTOM,
            parameters={"handler": "custom_handler"}
        )
        
        # Should return True (placeholder implementation)
        result = evaluator._execute_custom(action, sample_context)
        assert result is True


class TestRuleEvaluationWithComplexContext:
    """Test rule evaluation with complex contexts."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a rule evaluator for testing."""
        return RuleEvaluator()
    
    def test_nested_field_access(self, evaluator):
        """Test accessing nested fields in context."""
        context = RuleEvaluationContext(
            player_state={
                "character": {
                    "stats": {"health": 75, "mana": 50},
                    "inventory": {"items": ["sword", "potion"], "gold": 100}
                }
            }
        )
        
        rule = Rule(
            name="nested_access",
            conditions=[
                RuleCondition(
                    field_path="player_state.character.stats.health",
                    operator=ConditionOperator.GREATER_THAN,
                    expected_value=50
                ),
                RuleCondition(
                    field_path="player_state.character.inventory.gold",
                    operator=ConditionOperator.GREATER_EQUAL,
                    expected_value=100
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "All conditions met"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, context)
        
        assert result.triggered is True
        assert result.conditions_met is True
    
    def test_rule_evaluation_error_handling(self, evaluator):
        """Test rule evaluation handles errors gracefully."""
        context = RuleEvaluationContext(
            player_state={"health": 50}
        )
        
        # Create a rule that might cause an error (accessing non-existent path)
        rule = Rule(
            name="error_prone_rule",
            conditions=[
                RuleCondition(
                    field_path="nonexistent.path.field",
                    operator=ConditionOperator.EQUALS,
                    expected_value="test"
                )
            ],
            actions=[
                RuleAction(
                    action_type=ActionType.SEND_MESSAGE,
                    parameters={"message": "This shouldn't execute"}
                )
            ]
        )
        
        result = evaluator.evaluate_rule(rule, context)
        
        # Rule should not trigger because condition evaluation returns False for missing paths
        assert result.triggered is False
        assert result.conditions_met is False
        assert result.error_message is None  # No error because missing path just returns None