"""Tests for rule models."""

import pytest
from uuid import uuid4

from game_loop.core.rules.rule_models import (
    Rule,
    RuleAction,
    RuleCondition,
    RuleEvaluationContext,
    RulePriority,
    RuleResult,
    ActionType,
    ConditionOperator,
    RuleEngineConfig,
)


class TestRuleCondition:
    """Test RuleCondition model."""
    
    def test_create_basic_condition(self):
        """Test creating a basic condition."""
        condition = RuleCondition(
            field_path="player_state.health",
            operator=ConditionOperator.LESS_THAN,
            expected_value=20
        )
        
        assert condition.field_path == "player_state.health"
        assert condition.operator == ConditionOperator.LESS_THAN
        assert condition.expected_value == 20
        assert condition.id is not None
        assert condition.description is None
    
    def test_condition_with_description(self):
        """Test condition with description."""
        condition = RuleCondition(
            field_path="inventory.count",
            operator=ConditionOperator.GREATER_EQUAL,
            expected_value=10,
            description="Inventory is full"
        )
        
        assert condition.description == "Inventory is full"
    
    def test_condition_serialization(self):
        """Test condition can be serialized to dict."""
        condition = RuleCondition(
            field_path="test.field",
            operator=ConditionOperator.EQUALS,
            expected_value="test_value"
        )
        
        data = condition.model_dump()
        assert data["field_path"] == "test.field"
        assert data["operator"] == ConditionOperator.EQUALS
        assert data["expected_value"] == "test_value"


class TestRuleAction:
    """Test RuleAction model."""
    
    def test_create_basic_action(self):
        """Test creating a basic action."""
        action = RuleAction(
            action_type=ActionType.SEND_MESSAGE,
            parameters={"message": "Test message"}
        )
        
        assert action.action_type == ActionType.SEND_MESSAGE
        assert action.parameters["message"] == "Test message"
        assert action.id is not None
        assert action.target_path is None
    
    def test_action_with_target(self):
        """Test action with target path."""
        action = RuleAction(
            action_type=ActionType.MODIFY_STATE,
            target_path="player_state.health",
            parameters={"value": 100}
        )
        
        assert action.target_path == "player_state.health"
        assert action.parameters["value"] == 100
    
    def test_action_serialization(self):
        """Test action can be serialized to dict."""
        action = RuleAction(
            action_type=ActionType.GRANT_REWARD,
            parameters={"experience": 100}
        )
        
        data = action.model_dump()
        assert data["action_type"] == ActionType.GRANT_REWARD
        assert data["parameters"]["experience"] == 100


class TestRule:
    """Test Rule model."""
    
    def test_create_basic_rule(self):
        """Test creating a basic rule."""
        rule = Rule(name="test_rule")
        
        assert rule.name == "test_rule"
        assert rule.priority == RulePriority.MEDIUM
        assert rule.enabled is True
        assert len(rule.conditions) == 0
        assert len(rule.actions) == 0
        assert rule.id is not None
    
    def test_rule_with_conditions_and_actions(self):
        """Test rule with conditions and actions."""
        condition = RuleCondition(
            field_path="player.health",
            operator=ConditionOperator.LESS_THAN,
            expected_value=20
        )
        
        action = RuleAction(
            action_type=ActionType.SEND_MESSAGE,
            parameters={"message": "Low health!"}
        )
        
        rule = Rule(
            name="low_health_warning",
            description="Warn when health is low",
            priority=RulePriority.HIGH,
            conditions=[condition],
            actions=[action],
            tags=["health", "warning"]
        )
        
        assert rule.name == "low_health_warning"
        assert rule.priority == RulePriority.HIGH
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
        assert "health" in rule.tags
        assert "warning" in rule.tags
    
    def test_rule_string_representation(self):
        """Test rule string representation."""
        rule = Rule(
            name="test_rule",
            conditions=[RuleCondition(field_path="test", operator=ConditionOperator.EQUALS, expected_value="test")],
            actions=[RuleAction(action_type=ActionType.SEND_MESSAGE, parameters={})]
        )
        
        rule_str = str(rule)
        assert "test_rule" in rule_str
        assert "conditions=1" in rule_str
        assert "actions=1" in rule_str


class TestRuleEvaluationContext:
    """Test RuleEvaluationContext model."""
    
    def test_create_basic_context(self):
        """Test creating a basic context."""
        context = RuleEvaluationContext()
        
        assert context.player_state is None
        assert context.world_state is None
        assert context.current_action is None
        assert len(context.action_parameters) == 0
        assert len(context.custom_data) == 0
    
    def test_context_with_data(self):
        """Test context with data."""
        context = RuleEvaluationContext(
            player_state={"health": 50, "level": 5},
            current_action="attack",
            action_parameters={"target": "orc"},
            custom_data={"test": "value"}
        )
        
        assert context.player_state["health"] == 50
        assert context.current_action == "attack"
        assert context.action_parameters["target"] == "orc"
        assert context.custom_data["test"] == "value"
    
    def test_get_value_simple_path(self):
        """Test getting value with simple path."""
        context = RuleEvaluationContext(
            player_state={"health": 75}
        )
        
        value = context.get_value("player_state.health")
        assert value == 75
    
    def test_get_value_nested_path(self):
        """Test getting value with nested path."""
        context = RuleEvaluationContext(
            player_state={
                "stats": {"strength": 15, "wisdom": 12},
                "inventory": {"count": 5}
            }
        )
        
        strength = context.get_value("player_state.stats.strength")
        assert strength == 15
        
        count = context.get_value("player_state.inventory.count")
        assert count == 5
    
    def test_get_value_nonexistent_path(self):
        """Test getting value for nonexistent path."""
        context = RuleEvaluationContext(
            player_state={"health": 100}
        )
        
        value = context.get_value("player_state.mana")
        assert value is None
        
        value = context.get_value("nonexistent.path")
        assert value is None
    
    def test_get_value_invalid_path(self):
        """Test getting value with invalid path."""
        context = RuleEvaluationContext(
            player_state={"health": 100}
        )
        
        # Path tries to access attribute of non-dict
        value = context.get_value("player_state.health.invalid")
        assert value is None


class TestRuleResult:
    """Test RuleResult model."""
    
    def test_create_successful_result(self):
        """Test creating a successful rule result."""
        rule_id = uuid4()
        result = RuleResult(
            rule_id=rule_id,
            rule_name="test_rule",
            triggered=True,
            conditions_met=True,
            actions_executed=[uuid4(), uuid4()]
        )
        
        assert result.rule_id == rule_id
        assert result.rule_name == "test_rule"
        assert result.triggered is True
        assert result.conditions_met is True
        assert len(result.actions_executed) == 2
        assert result.error_message is None
    
    def test_create_failed_result(self):
        """Test creating a failed rule result."""
        rule_id = uuid4()
        result = RuleResult(
            rule_id=rule_id,
            rule_name="test_rule",
            triggered=False,
            conditions_met=False,
            error_message="Rule evaluation failed"
        )
        
        assert result.triggered is False
        assert result.conditions_met is False
        assert result.error_message == "Rule evaluation failed"
        assert len(result.actions_executed) == 0
    
    def test_result_string_representation(self):
        """Test result string representation."""
        rule_id = uuid4()
        result = RuleResult(
            rule_id=rule_id,
            rule_name="test_rule",
            triggered=True,
            conditions_met=True
        )
        
        result_str = str(result)
        assert "test_rule" in result_str
        assert "TRIGGERED" in result_str


class TestRuleEngineConfig:
    """Test RuleEngineConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RuleEngineConfig()
        
        assert config.max_rules_per_evaluation == 1000
        assert config.enable_conflict_detection is True
        assert config.enable_performance_monitoring is True
        assert config.default_rule_priority == RulePriority.MEDIUM
        assert config.rule_timeout_ms == 5000.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RuleEngineConfig(
            max_rules_per_evaluation=500,
            enable_conflict_detection=False,
            default_rule_priority=RulePriority.HIGH,
            rule_timeout_ms=10000.0
        )
        
        assert config.max_rules_per_evaluation == 500
        assert config.enable_conflict_detection is False
        assert config.default_rule_priority == RulePriority.HIGH
        assert config.rule_timeout_ms == 10000.0