"""
Rule Models for Game Loop Rules Engine.
Defines data structures for rules, conditions, actions, and results.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RulePriority(Enum):
    """Priority levels for rules."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5


class ConditionOperator(Enum):
    """Operators for rule conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    REGEX_MATCH = "regex_match"


class ActionType(Enum):
    """Types of actions that rules can perform."""

    MODIFY_STATE = "modify_state"
    SEND_MESSAGE = "send_message"
    TRIGGER_EVENT = "trigger_event"
    BLOCK_ACTION = "block_action"
    GRANT_REWARD = "grant_reward"
    IMPOSE_PENALTY = "impose_penalty"
    CHANGE_LOCATION = "change_location"
    SPAWN_ENTITY = "spawn_entity"
    DESPAWN_ENTITY = "despawn_entity"
    CUSTOM = "custom"


class RuleCondition(BaseModel):
    """Represents a condition that must be met for a rule to apply."""

    id: UUID = Field(default_factory=uuid4)
    field_path: str = Field(..., description="Dot-notation path to the field to check")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    expected_value: Any = Field(..., description="Expected value for comparison")
    description: str | None = Field(None, description="Human-readable description")

    pass


class RuleAction(BaseModel):
    """Represents an action to be performed when a rule is triggered."""

    id: UUID = Field(default_factory=uuid4)
    action_type: ActionType = Field(..., description="Type of action to perform")
    target_path: str | None = Field(
        None, description="Dot-notation path to target field"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )
    description: str | None = Field(None, description="Human-readable description")

    pass


class Rule(BaseModel):
    """Represents a complete game rule with conditions and actions."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Unique name for the rule")
    description: str | None = Field(None, description="Human-readable description")
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM, description="Rule priority"
    )
    enabled: bool = Field(default=True, description="Whether the rule is active")

    # Conditions (ALL must be true for rule to trigger)
    conditions: list[RuleCondition] = Field(
        default_factory=list, description="Rule conditions"
    )

    # Actions (ALL will be executed if rule triggers)
    actions: list[RuleAction] = Field(default_factory=list, description="Rule actions")

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last modification timestamp")

    class Config:
        pass

    def __str__(self) -> str:
        return f"Rule({self.name}, priority={self.priority.name}, conditions={len(self.conditions)}, actions={len(self.actions)})"


class RuleResult(BaseModel):
    """Result of rule evaluation and execution."""

    rule_id: UUID = Field(..., description="ID of the rule that was processed")
    rule_name: str = Field(..., description="Name of the rule")
    triggered: bool = Field(..., description="Whether the rule was triggered")
    conditions_met: bool = Field(
        ..., description="Whether all conditions were satisfied"
    )
    actions_executed: list[UUID] = Field(
        default_factory=list, description="IDs of actions that were executed"
    )
    error_message: str | None = Field(
        None, description="Error message if execution failed"
    )
    execution_time_ms: float | None = Field(
        None, description="Execution time in milliseconds"
    )

    def __str__(self) -> str:
        status = "TRIGGERED" if self.triggered else "NOT_TRIGGERED"
        return f"RuleResult({self.rule_name}: {status})"


class RuleEvaluationContext(BaseModel):
    """Context data provided for rule evaluation."""

    # Game state data
    player_state: dict[str, Any] | None = Field(
        None, description="Current player state"
    )
    world_state: dict[str, Any] | None = Field(None, description="Current world state")
    session_data: dict[str, Any] | None = Field(
        None, description="Current session data"
    )

    # Action context
    current_action: str | None = Field(
        None, description="Current action being performed"
    )
    action_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters of current action"
    )

    # Location context
    current_location: str | None = Field(None, description="Current location ID")
    location_data: dict[str, Any] | None = Field(
        None, description="Current location data"
    )

    # Timing context
    game_time: dict[str, Any] | None = Field(None, description="Game world time")
    real_time: str | None = Field(None, description="Real world timestamp")

    # Custom context
    custom_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )

    def get_value(self, field_path: str) -> Any:
        """
        Get a value from the context using dot notation.

        Args:
            field_path: Dot-notation path to the field (e.g., "player_state.health")

        Returns:
            The value at the specified path, or None if not found
        """
        parts = field_path.split(".")
        current = self.model_dump()

        try:
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None

                if current is None:
                    return None

            return current
        except (KeyError, TypeError, AttributeError):
            return None

    pass


class RuleConflict(BaseModel):
    """Represents a conflict between two or more rules."""

    conflicting_rules: list[UUID] = Field(..., description="IDs of conflicting rules")
    conflict_type: str = Field(..., description="Type of conflict")
    description: str = Field(..., description="Description of the conflict")
    resolution_strategy: str | None = Field(
        None, description="How the conflict was resolved"
    )
    resolved_rule_id: UUID | None = Field(
        None, description="ID of the rule that was chosen"
    )


class RuleEngineConfig(BaseModel):
    """Configuration for the Rules Engine."""

    max_rules_per_evaluation: int = Field(
        default=1000, description="Maximum rules to evaluate per cycle"
    )
    enable_conflict_detection: bool = Field(
        default=True, description="Whether to detect rule conflicts"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Whether to monitor rule performance"
    )
    default_rule_priority: RulePriority = Field(
        default=RulePriority.MEDIUM, description="Default priority for new rules"
    )
    rule_timeout_ms: float = Field(
        default=5000.0, description="Timeout for rule execution in milliseconds"
    )

    class Config:
        pass
