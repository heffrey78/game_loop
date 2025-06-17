"""
Rule Evaluator for Game Loop Rules Engine.
Handles evaluation of rule conditions and execution of rule actions.
"""

import re
from typing import Any
from uuid import UUID

from .rule_models import (
    ActionType,
    ConditionOperator,
    Rule,
    RuleAction,
    RuleCondition,
    RuleEvaluationContext,
    RuleResult,
)


class RuleEvaluator:
    """Evaluates rule conditions and executes rule actions."""

    def __init__(self):
        """Initialize the rule evaluator."""
        self._condition_evaluators = {
            ConditionOperator.EQUALS: self._evaluate_equals,
            ConditionOperator.NOT_EQUALS: self._evaluate_not_equals,
            ConditionOperator.GREATER_THAN: self._evaluate_greater_than,
            ConditionOperator.LESS_THAN: self._evaluate_less_than,
            ConditionOperator.GREATER_EQUAL: self._evaluate_greater_equal,
            ConditionOperator.LESS_EQUAL: self._evaluate_less_equal,
            ConditionOperator.CONTAINS: self._evaluate_contains,
            ConditionOperator.NOT_CONTAINS: self._evaluate_not_contains,
            ConditionOperator.IN: self._evaluate_in,
            ConditionOperator.NOT_IN: self._evaluate_not_in,
            ConditionOperator.EXISTS: self._evaluate_exists,
            ConditionOperator.NOT_EXISTS: self._evaluate_not_exists,
            ConditionOperator.REGEX_MATCH: self._evaluate_regex_match,
        }

        self._action_executors = {
            ActionType.MODIFY_STATE: self._execute_modify_state,
            ActionType.SEND_MESSAGE: self._execute_send_message,
            ActionType.TRIGGER_EVENT: self._execute_trigger_event,
            ActionType.BLOCK_ACTION: self._execute_block_action,
            ActionType.GRANT_REWARD: self._execute_grant_reward,
            ActionType.IMPOSE_PENALTY: self._execute_impose_penalty,
            ActionType.CHANGE_LOCATION: self._execute_change_location,
            ActionType.SPAWN_ENTITY: self._execute_spawn_entity,
            ActionType.DESPAWN_ENTITY: self._execute_despawn_entity,
            ActionType.CUSTOM: self._execute_custom,
        }

    def evaluate_rule(self, rule: Rule, context: RuleEvaluationContext) -> RuleResult:
        """
        Evaluate a rule against the provided context.

        Args:
            rule: The rule to evaluate
            context: The evaluation context

        Returns:
            RuleResult containing the evaluation outcome
        """
        import time

        start_time = time.time()

        try:
            # Check if rule is enabled
            if not rule.enabled:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    triggered=False,
                    conditions_met=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Evaluate all conditions
            conditions_met = self._evaluate_conditions(rule.conditions, context)

            # If conditions not met, return early
            if not conditions_met:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    triggered=False,
                    conditions_met=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute all actions
            executed_actions = self._execute_actions(rule.actions, context)

            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                triggered=True,
                conditions_met=True,
                actions_executed=executed_actions,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                triggered=False,
                conditions_met=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _evaluate_conditions(
        self, conditions: list[RuleCondition], context: RuleEvaluationContext
    ) -> bool:
        """
        Evaluate all conditions for a rule (ALL must be true).

        Args:
            conditions: List of conditions to evaluate
            context: The evaluation context

        Returns:
            True if all conditions are met, False otherwise
        """
        for condition in conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True

    def _evaluate_condition(
        self, condition: RuleCondition, context: RuleEvaluationContext
    ) -> bool:
        """
        Evaluate a single condition.

        Args:
            condition: The condition to evaluate
            context: The evaluation context

        Returns:
            True if condition is met, False otherwise
        """
        try:
            actual_value = context.get_value(condition.field_path)

            # Handle both enum and string operator values
            operator = condition.operator
            if isinstance(operator, str):
                # Convert string to enum
                for enum_op in ConditionOperator:
                    if enum_op.value == operator:
                        operator = enum_op
                        break

            evaluator = self._condition_evaluators.get(operator)

            if not evaluator:
                return False

            return evaluator(actual_value, condition.expected_value)

        except Exception:
            return False

    def _execute_actions(
        self, actions: list[RuleAction], context: RuleEvaluationContext
    ) -> list[UUID]:
        """
        Execute all actions for a rule.

        Args:
            actions: List of actions to execute
            context: The evaluation context

        Returns:
            List of successfully executed action IDs
        """
        executed = []

        for action in actions:
            try:
                if self._execute_action(action, context):
                    executed.append(action.id)
            except Exception:
                continue  # Log error but continue with other actions

        return executed

    def _execute_action(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """
        Execute a single action.

        Args:
            action: The action to execute
            context: The evaluation context

        Returns:
            True if action executed successfully, False otherwise
        """
        # Handle both enum and string action type values
        action_type = action.action_type
        if isinstance(action_type, str):
            # Convert string to enum
            for enum_type in ActionType:
                if enum_type.value == action_type:
                    action_type = enum_type
                    break

        executor = self._action_executors.get(action_type)
        if not executor:
            return False

        return executor(action, context)

    # Condition evaluators
    def _evaluate_equals(self, actual: Any, expected: Any) -> bool:
        return actual == expected

    def _evaluate_not_equals(self, actual: Any, expected: Any) -> bool:
        return actual != expected

    def _evaluate_greater_than(self, actual: Any, expected: Any) -> bool:
        try:
            return float(actual) > float(expected)
        except (ValueError, TypeError):
            return False

    def _evaluate_less_than(self, actual: Any, expected: Any) -> bool:
        try:
            return float(actual) < float(expected)
        except (ValueError, TypeError):
            return False

    def _evaluate_greater_equal(self, actual: Any, expected: Any) -> bool:
        try:
            return float(actual) >= float(expected)
        except (ValueError, TypeError):
            return False

    def _evaluate_less_equal(self, actual: Any, expected: Any) -> bool:
        try:
            return float(actual) <= float(expected)
        except (ValueError, TypeError):
            return False

    def _evaluate_contains(self, actual: Any, expected: Any) -> bool:
        try:
            return str(expected) in str(actual)
        except (TypeError, AttributeError):
            return False

    def _evaluate_not_contains(self, actual: Any, expected: Any) -> bool:
        try:
            return str(expected) not in str(actual)
        except (TypeError, AttributeError):
            return False

    def _evaluate_in(self, actual: Any, expected: Any) -> bool:
        try:
            if isinstance(expected, (list, tuple)):
                return actual in expected
            return str(actual) in str(expected)
        except (TypeError, AttributeError):
            return False

    def _evaluate_not_in(self, actual: Any, expected: Any) -> bool:
        try:
            if isinstance(expected, (list, tuple)):
                return actual not in expected
            return str(actual) not in str(expected)
        except (TypeError, AttributeError):
            return False

    def _evaluate_exists(self, actual: Any, expected: Any) -> bool:
        return actual is not None

    def _evaluate_not_exists(self, actual: Any, expected: Any) -> bool:
        return actual is None

    def _evaluate_regex_match(self, actual: Any, expected: Any) -> bool:
        try:
            pattern = str(expected)
            text = str(actual)
            return bool(re.match(pattern, text))
        except (TypeError, re.error):
            return False

    # Action executors (placeholder implementations)
    def _execute_modify_state(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute state modification action."""
        # This is a placeholder - actual implementation would modify game state
        # Parameters should include: target_path, new_value, operation_type
        return True

    def _execute_send_message(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute message sending action."""
        # This is a placeholder - actual implementation would send message to player
        # Parameters should include: message_text, message_type, target
        return True

    def _execute_trigger_event(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute event triggering action."""
        # This is a placeholder - actual implementation would trigger game event
        # Parameters should include: event_type, event_data
        return True

    def _execute_block_action(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute action blocking."""
        # This is a placeholder - actual implementation would prevent action execution
        # Parameters should include: reason, alternative_message
        return True

    def _execute_grant_reward(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute reward granting action."""
        # This is a placeholder - actual implementation would grant rewards
        # Parameters should include: reward_type, amount, reason
        return True

    def _execute_impose_penalty(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute penalty imposing action."""
        # This is a placeholder - actual implementation would impose penalties
        # Parameters should include: penalty_type, amount, reason
        return True

    def _execute_change_location(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute location change action."""
        # This is a placeholder - actual implementation would change player location
        # Parameters should include: target_location, transition_type
        return True

    def _execute_spawn_entity(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute entity spawning action."""
        # This is a placeholder - actual implementation would spawn game entities
        # Parameters should include: entity_type, location, properties
        return True

    def _execute_despawn_entity(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute entity despawning action."""
        # This is a placeholder - actual implementation would remove entities
        # Parameters should include: entity_id, reason
        return True

    def _execute_custom(
        self, action: RuleAction, context: RuleEvaluationContext
    ) -> bool:
        """Execute custom action."""
        # This is a placeholder - actual implementation would handle custom actions
        # Parameters should include: handler_name, custom_parameters
        return True
